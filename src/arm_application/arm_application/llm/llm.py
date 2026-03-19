import json
import os
import re
import rospy
import textwrap
from arm_application.msg import LLMCommands, AgentFeedback
from std_msgs.msg import String
from typing import Optional, List, Dict, Any
import dashscope


class TongyiQianwenLLM:
    """通义千问LLM集成类（支持单轮复合任务）"""

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化通义千问LLM

        Args:
            api_key: 通义千问API密钥, 如果为None则从环境变量DASHSCOPE_API_KEY获取
        """
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key is required. Please set DASHSCOPE_API_KEY environment variable or provide it as parameter."
            )

        dashscope.api_key = self.api_key
        self.model = "qwen-max"

        # 仍然沿用现有单条命令消息类型
        self.pub = rospy.Publisher('/llm_commands', LLMCommands, queue_size=10)
        self.sub = rospy.Subscriber('/llm_user_input', String, self._user_input_callback)

        self.agent_feedback_sub = rospy.Subscriber(
            '/agent_feedback',
            AgentFeedback,
            self._agent_feedback_callback
        )

        self.last_feedback = None
        self.waiting_feedback = False
        self.current_step_id = 0
        self.current_session_id = "default"
        self.pending_tasks = []
        self.current_user_goal = ""
        # 最大回合数保护
        self.max_rounds = 5
        rospy.loginfo("[LLM] LLM 节点已启动, 等待用户输入...")

    def _user_input_callback(self, msg):
        """处理用户输入话题的回调函数"""
        user_input = msg.data
        rospy.loginfo(f"[LLM] 收到用户输入: {user_input}")

        if self.waiting_feedback:
            rospy.logwarn("[LLM] 当前仍在等待上一条 action 的反馈，忽略新的用户输入")
            return

        self.current_user_goal = user_input
        self.process_user_input(user_input)

    def _agent_feedback_callback(self, msg):
        self.last_feedback = msg
        self.waiting_feedback = False

        rospy.loginfo(
            f"[LLM] 收到 Agent feedback: "
            f"action={msg.action_type}, success={msg.success}, "
            f"error={msg.error_code}, message={msg.message}"
        )

        # 最小任务结束条件：
        # 如果 reset 已成功执行，则认为当前任务结束
        if msg.success and msg.action_type == "reset":
            rospy.loginfo("[LLM] 检测到 reset 已成功执行，当前任务结束，不再继续重规划")
            self.current_user_goal = ""
            self.waiting_feedback = False
            self.pending_tasks = []
            return

        # 没有目标时，不继续
        if not self.current_user_goal:
            rospy.logwarn("[LLM] 当前没有活动中的用户目标，不继续重规划")
            return

        # 防止无限循环
        if self.current_step_id >= self.max_rounds:
            rospy.logwarn("[LLM] 已达到最大回合数，停止继续规划")
            return

        # 基于反馈重新生成下一步动作
        replan_prompt = self._build_replan_prompt(self.current_user_goal, msg)
        response = self.generate(replan_prompt)

        rospy.loginfo(f"[LLM] 重规划原始模型输出: {response}")

        try:
            json_str = self._extract_json_str(response)
            if not json_str:
                rospy.logerr("[LLM] 重规划时无法提取 JSON，停止")
                return

            data = json.loads(json_str)
            tasks = self._normalize_tasks(data)

            if not tasks:
                rospy.logerr("[LLM] 重规划时未解析到有效动作，停止")
                return

            next_task = tasks[0]
            next_msg = self._task_to_msg(next_task)

            self.current_step_id += 1
            self.pub.publish(next_msg)
            self.waiting_feedback = True

            rospy.loginfo(f"[LLM] 重规划后发布第 {self.current_step_id} 条LLM指令: {next_msg}")
            rospy.loginfo("[LLM] 再次进入等待 feedback 状态")

        except Exception as e:
            rospy.logerr(f"[LLM] 重规划解析失败: {e}")
            rospy.logerr(f"[LLM] 重规划原始响应: {response}")

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.2) -> str:
        """
        调用通义千问API生成文本

        Args:
            prompt: 提示词
            max_tokens: 最大生成token数
            temperature: 生成温度，结构化输出建议设低一点

        Returns:
            生成的文本
        """
        try:
            response = dashscope.Generation.call(
                model=self.model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
            )

            if response.status_code == 200:
                return response.output.text
            else:
                raise Exception(f"API call failed: {response.message}")

        except Exception as e:
            rospy.logerr(f"[LLM] Error calling Tongyi Qianwen API: {e}")
            return ""

    # 多轮prompt共用的部分
    def _build_prompt_rules(self) -> str:
        """
        返回所有轮次共享的固定规则部分
        """
        return textwrap.dedent("""
            你是一个机械臂控制指令解析器，同时也是一个高层决策模块。
            你的任务是把输入信息转换为机械臂“下一步可执行的一个 JSON 动作”。

            你必须严格按照下面的 JSON 格式输出，且只能输出 JSON，不要输出任何解释、注释、Markdown、前缀或后缀文本。

            输出格式必须为：
            {
                "action_type": "pick" | "place" | "pick_place" | "reset" | "open_gripper" | "close_gripper" | "create" | "delete",
                "object_class_id": int,
                "object_name": "string",
                "object_x": float,
                "object_y": float,
                "object_z": float,
                "target_class_id": int,
                "target_name": "string",
                "target_x": float,
                "target_y": float,
                "target_z": float
            }

            总体规则：
            1. 顶层必须是一个 JSON 对象。
            2. 顶层不能包含 "tasks" 字段。
            3. 你每次只能输出一个动作，不能输出动作列表。
            4. 所有 JSON 的字段名和 action_type 的取值必须使用英文。
            5. 除 JSON 以外，不允许输出任何其他内容。
            6. 即使输入中包含多个顺序动作，你这一轮也只能输出“下一步最应该执行的一个动作”。

            支持的 action_type：
            - "pick"
            - "place"
            - "pick_place"
            - "reset"
            - "open_gripper"
            - "close_gripper"
            - "create"
            - "delete"

            各动作含义与填写规则：

            一、pick
            表示抓取动作。
            - 可以使用 object_class_id 指定抓取目标类别；
            - object_class_id = 0 表示 blue box;
            - object_class_id = 1 表示 green cylinder;
            - object_class_id = 2 表示 red box;
            - object_class_id = 3 表示 yellow cylinder;
            - 也可以使用 (object_x, object_y, object_z) 指定显式抓取坐标；
            - 如果明确给出了抓取坐标，则填写 object_x/object_y/object_z，并将 object_class_id 设为 -1；
            - target 相关字段全部设为默认值。

            二、place
            表示放置动作。
            - 可以使用 target_class_id 指定放置目标类别；
            - target_class_id = 0 表示 blue box;
            - target_class_id = 1 表示 green cylinder;
            - target_class_id = 2 表示 red box;
            - target_class_id = 3 表示 yellow cylinder;
            - 也可以使用 (target_x, target_y, target_z) 指定显式放置坐标；
            - 如果明确给出了放置坐标，则填写 target_x/target_y/target_z，并将 target_class_id 设为 -1；
            - object 相关字段全部设为默认值。

            三、pick_place
            表示先抓取再放置。
            - 必须同时包含 object 信息和 target 信息；
            - object 侧遵循 pick 的填写规则；
            - target 侧遵循 place 的填写规则。
            - 只有在“下一步动作本身就是一个完整 pick_place 技能”时才使用该动作。
            - 如果任务需要更稳妥地逐步执行，也可以优先输出 pick，后续再输出 place。

            四、reset
            表示机械臂复位。
            - object 和 target 相关字段全部设为默认值。

            五、open_gripper
            表示打开夹爪。
            - object 和 target 相关字段全部设为默认值。

            六、close_gripper
            表示关闭夹爪。
            - object 和 target 相关字段全部设为默认值。

            七、create
            表示创建物体。
            - 使用 object_class_id 表示创建物体的类别；
            - 使用 object_x, object_y, object_z 表示创建位置；
            - 使用 object_name 表示创建出来的物体名称；
            - target 相关字段全部设为默认值；
            - object_class_id = 0 表示 blue box;
            - object_class_id = 1 表示 green cylinder;
            - object_class_id = 2 表示 red box;
            - object_class_id = 3 表示 yellow cylinder;

            八、delete
            表示删除物体。
            - 使用 object_name 表示要删除的物体名称；
            - 其他字段全部设为默认值。

            默认值规则：
            1. 未使用的 class_id 字段统一填写 -1
            2. 未使用的坐标字段统一填写 0.0
            3. 未使用的名称字段统一填写 ""

            语义理解规则：
            1. “抓起方块”“拿起方块”“夹起方块”都应理解为 pick。
            2. “放下”“放到”“放在”如果同时包含抓取对象和目标对象，可以理解为 pick_place；
            但如果系统采用逐步执行，也可以先输出 pick 作为下一步动作。
            3. “把方块放到圆柱上”可以理解为最终目标是 pick_place，但当前这一轮仍然只能输出一个下一步动作。
            4. “生成一个蓝色方块”应理解为 create。
            5. “删除名为 box1 的物体”应理解为 delete。

            顺序任务规则（重要）：
            1. 如果输入中包含“然后”“再”“接着”“最后”“随后”等表示顺序执行的词语，不要一次性输出多个动作。
            2. 中文中的“然后、再、接着、最后”，以及英文中的“then、and then、after that、finally”，都表示整体目标中包含多个阶段。
            3. 你当前只需要根据整体目标，选择“下一步最应该执行的一个动作”。

            单步决策要求（最重要）：
            1. 你只负责当前这一轮的“下一步动作”，不要规划整个任务列表。
            2. 如果输入表达的是复合任务，优先输出第一步最合理的动作。
            3. 不要输出 "tasks"。
            4. 不要把多个动作合并成列表。
            5. 你的输出必须能被系统直接当作“当前一步命令”执行。

            稳健性要求：
            1. 必须输出合法 JSON。
            2. 不要遗漏必要字段。
            3. 不要输出数组。
            4. 不要输出多余文本。
        """).strip()

    def _build_prompt(self, user_input: str) -> str:
        """
        构造首轮 prompt（单步决策），使用共享规则底座
        """
        rules = self._build_prompt_rules()

        prompt_template = textwrap.dedent("""
            {rules}

            用户输入：
            {user_input}

            请输出下一步动作。
        """).strip()

        return prompt_template.replace("{rules}", rules).replace("{user_input}", user_input)

    def _build_replan_prompt(self, user_goal: str, feedback: AgentFeedback) -> str:
        """
        基于用户目标 + 上一轮执行反馈，生成下一步动作 prompt
        """
        rules = self._build_prompt_rules()

        prompt_template = textwrap.dedent("""
            {rules}

            用户整体目标：
            {user_goal}

            上一轮执行反馈：
            - action_type: {action_type}
            - success: {success}
            - error_code: {error_code}
            - message: {message}

            请输出下一步动作。
        """).strip()

        return (
            prompt_template
            .replace("{rules}", rules)
            .replace("{user_goal}", user_goal)
            .replace("{action_type}", str(feedback.action_type))
            .replace("{success}", str(feedback.success))
            .replace("{error_code}", str(feedback.error_code))
            .replace("{message}", str(feedback.message))
        )

    def _extract_json_str(self, response: str) -> Optional[str]:
        """
        从模型返回文本中尽量稳健地提取 JSON 字符串
        """
        if not response:
            return None

        response = response.strip()

        # 优先尝试整体解析
        try:
            json.loads(response)
            return response
        except Exception:
            pass

        # 提取最外层 JSON 对象
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            return match.group(0)

        return None

    def _normalize_tasks(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        兼容两种情况：
        1. 新格式：{"tasks": [...]}
        2. 旧格式：{单个任务字段...}
        """
        if isinstance(data, dict) and "tasks" in data and isinstance(data["tasks"], list):
            return data["tasks"]

        # 兼容旧的单任务输出
        if isinstance(data, dict) and "action_type" in data:
            return [data]

        return []

    def _task_to_msg(self, command_dict: Dict[str, Any]) -> LLMCommands:
        """
        把单个 task dict 转成现有 LLMCommands 消息
        """
        msg = LLMCommands()
        msg.action_type = command_dict.get("action_type", "reset")

        msg.object_class_id = int(command_dict.get("object_class_id", -1))
        msg.object_name = str(command_dict.get("object_name", ""))
        msg.object_x = float(command_dict.get("object_x", 0.0))
        msg.object_y = float(command_dict.get("object_y", 0.0))
        msg.object_z = float(command_dict.get("object_z", 0.0))

        msg.target_class_id = int(command_dict.get("target_class_id", -1))
        msg.target_name = str(command_dict.get("target_name", ""))
        msg.target_x = float(command_dict.get("target_x", 0.0))
        msg.target_y = float(command_dict.get("target_y", 0.0))
        msg.target_z = float(command_dict.get("target_z", 0.0))

        return msg

    def process_user_input(self, user_input: str):
        """
        处理用户输入，当前阶段只发布第一条 action，
        不再一次性连续发布全部 tasks。
        """
        prompt = self._build_prompt(user_input)
        response = self.generate(prompt)

        try:
            json_str = self._extract_json_str(response)
            if not json_str:
                rospy.logerr("[LLM] 无法从LLM响应中提取JSON")
                rospy.logerr(f"[LLM] 原始响应: {response}")
                return None

            data = json.loads(json_str)
            tasks = self._normalize_tasks(data)

            if not tasks:
                rospy.logerr("[LLM] LLM输出中未找到有效 tasks")
                rospy.logerr(f"[LLM] 解析后的数据: {data}")
                return None

            total_tasks = len(tasks)
            self.pending_tasks = list(tasks)
            self.current_step_id = 1

            first_task = self.pending_tasks.pop(0)
            msg = self._task_to_msg(first_task)

            self.pub.publish(msg)
            self.waiting_feedback = True

            rospy.loginfo(f"[LLM] 发布第 1/{total_tasks} 条LLM指令: {msg}")
            rospy.loginfo(f"[LLM] 进入等待 feedback 状态，剩余任务数: {len(self.pending_tasks)}")

            return [msg]

        except Exception as e:
            rospy.logerr(f"[LLM] 解析LLM响应时出错: {e}")
            rospy.logerr(f"[LLM] 原始响应: {response}")
            return None