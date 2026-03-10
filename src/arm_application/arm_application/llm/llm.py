import json
import os
import re
import rospy
import textwrap
from arm_application.msg import LLMCommands
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
        rospy.loginfo("LLM 节点已启动, 等待用户输入...")

    def _user_input_callback(self, msg):
        """处理用户输入话题的回调函数"""
        user_input = msg.data
        rospy.loginfo(f"收到用户输入: {user_input}")
        self.process_user_input(user_input)

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
            rospy.logerr(f"Error calling Tongyi Qianwen API: {e}")
            return ""

    def _build_prompt(self, user_input: str) -> str:
        """
        构造支持复合任务的提示词
        """
        prompt_template = textwrap.dedent("""
            你是一个机械臂控制指令解析器。
            你的任务是把用户输入的中文自然语言指令，转换为机械臂可执行的 JSON 任务列表。

            你必须严格按照下面的 JSON 格式输出，且只能输出 JSON，不要输出任何解释、注释、Markdown、前缀或后缀文本。

            输出格式必须为：
            {
            "tasks": [
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
            ]
            }

            总体规则：
            1. 顶层必须是一个 JSON 对象。
            2. 顶层必须包含 "tasks" 字段。
            3. "tasks" 必须是一个列表。
            4. 如果用户只表达了一个动作，则 "tasks" 中只包含 1 个任务。
            5. 如果用户在一句话中表达了多个顺序动作，则必须拆分为多个任务，并严格按照执行顺序输出到 "tasks" 列表中。
            6. 所有 JSON 的字段名和 action_type 的取值必须使用英文。
            7. 除 JSON 以外，不允许输出任何其他内容。

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
            - object_class_id = 0 表示 blue box；
            - object_class_id = 1 表示 green cylinder。
            - 也可以使用 (object_x, object_y, object_z) 指定显式抓取坐标；
            - 如果用户明确给出了抓取坐标，则填写 object_x/object_y/object_z，并将 object_class_id 设为 -1；
            - target 相关字段全部设为默认值。

            二、place
            表示放置动作。
            - 可以使用 target_class_id 指定放置目标类别；
            - object_class_id = 0 表示 blue box；
            - object_class_id = 1 表示 green cylinder。
            - 也可以使用 (target_x, target_y, target_z) 指定显式放置坐标；
            - 如果用户明确给出了放置坐标，则填写 target_x/target_y/target_z，并将 target_class_id 设为 -1；
            - object 相关字段全部设为默认值。

            三、pick_place
            表示先抓取再放置。
            - 必须同时包含 object 信息和 target 信息；
            - object 侧遵循 pick 的填写规则；
            - target 侧遵循 place 的填写规则。

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
            - object_class_id = 0 表示 blue box；
            - object_class_id = 1 表示 green cylinder。

            八、delete
            表示删除物体。
            - 使用 object_name 表示要删除的物体名称；
            - 其他字段全部设为默认值。

            默认值规则：
            1. 未使用的 class_id 字段统一填写 -1
            2. 未使用的坐标字段统一填写 0.0
            3. 未使用的名称字段统一填写 ""

            多任务解析规则：
            1. 如果用户输入中包含“然后”“再”“接着”“最后”“随后”等表示顺序执行的词语，通常应拆分为多个任务。
            2. 中文中的“然后、再、接着、最后”，以及英文中的“then、and then、after that、finally”，都表示顺序任务。
            3. 例如：
            用户输入：将方块放到圆柱上，然后再复位
            应输出两个任务：
            - 第一个任务为 "pick_place"
            - 第二个任务为 "reset"

            语义理解规则：
            1. “抓起方块”“拿起方块”“夹起方块”都应理解为 pick。
            2. “放下”“放到”“放在”如果同时包含抓取对象和目标对象，优先理解为 pick_place。
            3. “把方块放到圆柱上”通常应理解为 pick_place，而不是单独的 place。
            4. “生成一个蓝色方块”应理解为 create。
            5. “删除名为 box1 的物体”应理解为 delete。

            稳健性要求：
            1. 必须输出合法 JSON。
            2. 不要遗漏 "tasks"。
            3. 不要把多个动作错误合并成一个任务。
            4. 如果用户只说一个动作，不要强行拆成多个任务。
            5. 如果用户表达的是顺序动作，必须按顺序输出多个任务。

            用户输入：
            {user_input}

            请只返回 JSON 对象本身。
            """).strip()

        return prompt_template.replace("{user_input}", user_input)

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
        处理用户输入并发布一个或多个 LLMCommands 消息
        """
        prompt = self._build_prompt(user_input)
        response = self.generate(prompt)

        try:
            json_str = self._extract_json_str(response)
            if not json_str:
                rospy.logerr("无法从LLM响应中提取JSON")
                rospy.logerr(f"原始响应: {response}")
                return None

            data = json.loads(json_str)
            tasks = self._normalize_tasks(data)

            if not tasks:
                rospy.logerr("LLM输出中未找到有效 tasks")
                rospy.logerr(f"解析后的数据: {data}")
                return None

            published_msgs = []

            for idx, task_dict in enumerate(tasks):
                msg = self._task_to_msg(task_dict)
                self.pub.publish(msg)
                published_msgs.append(msg)
                rospy.loginfo(f"发布第 {idx + 1}/{len(tasks)} 条LLM指令: {msg}")

                # 给订阅端一点处理时间，避免连续发布太快
                rospy.sleep(0.1)

            return published_msgs

        except Exception as e:
            rospy.logerr(f"解析LLM响应时出错: {e}")
            rospy.logerr(f"原始响应: {response}")
            return None