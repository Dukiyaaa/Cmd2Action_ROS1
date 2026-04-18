import json
import os
import re
import rospy
import textwrap
from arm_application.msg import LLMCommands, AgentFeedback
from arm_vision.msg import DetectedObjectPool
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

        # 视觉信息
        self.detected_objects_sub = rospy.Subscriber(
            '/detected_objects',
            DetectedObjectPool,
            self._detected_objects_callback
        )

        self.latest_detected_objects = []

        self._init_task_state()
        rospy.loginfo("[LLM] LLM 节点已启动, 等待用户输入...")
    
    # 状态机管理
    def _init_task_state(self):
        self.task_state = {
            "session_id": "",
            "user_goal": "",
            "status": "idle",   # idle / running / waiting_feedback / finished / failed
            "step_id": 0,
            "max_rounds": 10,
            "history": [],
            "last_feedback": None,

            # 首轮生成的参考动作序列，后续 replan 默认优先参考它
            "reference_plan": [],

            # 最近一次 LLM 决策结果，便于调试
            # 例如：{"decision": "continue"} 或 {"decision": "finish"}
            "last_decision": None,
        }

    def _reset_task_state(self):
        self._init_task_state()
    
    def _record_action(self, command_dict: Dict[str, Any]):
        self.task_state["history"].append({
            "step_id": self.task_state["step_id"],
            "type": "action",
            "action_type": command_dict.get("action_type", "unknown"),
            "command": command_dict
        })

    def _record_feedback(self, feedback: AgentFeedback):
        feedback_record = {
            "step_id": self.task_state["step_id"],
            "type": "feedback",
            "action_type": feedback.action_type,
            "success": feedback.success,
            "error_code": feedback.error_code,
            "message": feedback.message,
            "retry_exhausted": feedback.retry_exhausted,
            "object_name": feedback.object_name,
            "object_class_id": feedback.object_class_id,
            "object_pose": (
                feedback.object_x,
                feedback.object_y,
                feedback.object_z
            ),
            "done": feedback.done
        }

        self.task_state["last_feedback"] = feedback_record
        self.task_state["history"].append(feedback_record)
        
    def _user_input_callback(self, msg):
        """处理用户输入话题的回调函数"""
        user_input = msg.data
        rospy.loginfo(f"[LLM] 收到用户输入: {user_input}")

        if self.task_state["status"] == "waiting_feedback":
            rospy.logwarn("[LLM] 当前仍在等待上一条 action 的反馈，忽略新的用户输入")
            return

        # 开启一个新任务前，先重置任务状态
        self._reset_task_state()
        self.task_state["user_goal"] = user_input
        self.task_state["status"] = "running"

        self.process_user_input(user_input)
    
    def _detected_objects_callback(self, msg):
        objects = []

        for obj in msg.objects:
            objects.append({
                "class_id": obj.class_id,
                "confidence": obj.confidence,
                "x": obj.pose.pose.position.x,
                "y": obj.pose.pose.position.y,
                "z": obj.pose.pose.position.z
            })

        self.latest_detected_objects = objects
        # rospy.loginfo(f"[LLM] 已更新视觉缓存，objects_count={len(self.latest_detected_objects)}")

    def _format_visual_context(self) -> str:
        if not self.latest_detected_objects:
            return "当前视觉观测：未检测到物体。"

        class_name_map = {
            0: "blue box",
            1: "green cylinder",
            2: "red box",
            3: "yellow cylinder"
        }

        grouped = {}
        for obj in self.latest_detected_objects:
            class_id = obj["class_id"]
            if class_id not in grouped:
                grouped[class_id] = []
            grouped[class_id].append(obj)

        lines = ["当前视觉观测："]
        for class_id, objs in grouped.items():
            class_name = class_name_map.get(class_id, f"class_{class_id}")
            lines.append(f"- {class_name}: count={len(objs)}")

            for i, obj in enumerate(objs[:3], start=1):
                lines.append(
                    f"  - #{i}: "
                    f"pos=({obj['x']:.3f}, {obj['y']:.3f}, {obj['z']:.3f}), "
                    f"confidence={obj['confidence']:.3f}"
                )

            if len(objs) > 3:
                lines.append(f"  - ... {len(objs) - 3} more")

        return "\n".join(lines)

    # 当前系统认为，所有任务必须以reset结尾
    def _is_task_finished(self, feedback: AgentFeedback) -> bool:
        # if not feedback.success:
        #     return False

        # if feedback.action_type == "reset":
        #     return True

        return False

    def _format_reference_plan(self) -> str:
        """
        将首轮生成的 reference_plan 格式化为可读文本，供 replan prompt 使用
        """
        reference_plan = self.task_state.get("reference_plan", [])
        if not reference_plan:
            return "无"

        lines = []
        for idx, task in enumerate(reference_plan, start=1):
            lines.append(
                f"- plan step {idx}: "
                f"action_type={task.get('action_type', 'unknown')}, "
                f"object_class_id={task.get('object_class_id', -1)}, "
                f"object_name={task.get('object_name', '')}, "
                f"object_xyz=({task.get('object_x', 0.0)}, {task.get('object_y', 0.0)}, {task.get('object_z', 0.0)}), "
                f"target_class_id={task.get('target_class_id', -1)}, "
                f"target_name={task.get('target_name', '')}, "
                f"target_xyz=({task.get('target_x', 0.0)}, {task.get('target_y', 0.0)}, {task.get('target_z', 0.0)})"
            )

        return "\n".join(lines)

    def _should_stop_after_failure(self, feedback: AgentFeedback) -> bool:
        # agent 已经判断没法再重试了，先停止
        if feedback.retry_exhausted:
            return True

        # 一些明显不适合继续让 LLM 硬规划的错误，先停掉
        fatal_errors = {
            "RESOLVE_POSE_FAILED",
            "RESOLVE_TARGET_POSE_FAILED",
        }

        if feedback.error_code in fatal_errors:
            return True

        return False

    def _format_recent_history(self, max_items: int = 4) -> str:
        history = self.task_state.get("history", [])
        if not history:
            return "无"

        recent = history[-max_items:]
        lines = []

        for item in recent:
            if item["type"] == "action":
                cmd = item["command"]
                lines.append(
                    f"- step {item['step_id']} action: "
                    f"action_type={item['action_type']}, "
                    f"object_class_id={cmd.get('object_class_id', -1)}, "
                    f"object_name={cmd.get('object_name', '')}, "
                    f"object_xyz=({cmd.get('object_x', 0.0)}, {cmd.get('object_y', 0.0)}, {cmd.get('object_z', 0.0)}), "
                    f"target_class_id={cmd.get('target_class_id', -1)}, "
                    f"target_name={cmd.get('target_name', '')}, "
                    f"target_xyz=({cmd.get('target_x', 0.0)}, {cmd.get('target_y', 0.0)}, {cmd.get('target_z', 0.0)})"
                )
            elif item["type"] == "feedback":
                pose = item.get("object_pose", (0.0, 0.0, 0.0))
                lines.append(
                    f"- step {item['step_id']} feedback: "
                    f"action_type={item['action_type']}, "
                    f"success={item['success']}, "
                    f"error_code={item['error_code']}, "
                    f"message={item['message']}, "
                    f"retry_exhausted={item['retry_exhausted']}, "
                    f"object_class_id={item['object_class_id']}, "
                    f"object_name={item['object_name']}, "
                    f"object_pose=({pose[0]}, {pose[1]}, {pose[2]}), "
                    f"done={item['done']}"
                )

        return "\n".join(lines)

    def _agent_feedback_callback(self, msg):
        self._record_feedback(msg)

        if self.task_state["status"] == "waiting_feedback":
            self.task_state["status"] = "running"

        rospy.loginfo(
            f"[LLM] 收到 Agent feedback: "
            f"action={msg.action_type}, success={msg.success}, "
            f"error={msg.error_code}, message={msg.message}"
        )

        # 最小任务结束条件：
        if self._is_task_finished(msg):
            rospy.loginfo("[LLM] 当前任务已完成，结束任务")
            self.task_state["status"] = "finished"
            return

        # 没有目标时，不继续
        if not self.task_state["user_goal"]:
            rospy.logwarn("[LLM] 当前没有活动中的用户目标，不继续重规划")
            return

        # 防止无限循环
        if self.task_state["step_id"] >= self.task_state["max_rounds"]:
            rospy.logwarn("[LLM] 已达到最大回合数，停止继续规划")
            self.task_state["status"] = "failed"
            return
        
        if not msg.success and self._should_stop_after_failure(msg):
            rospy.logwarn(
                f"[LLM] 当前动作失败且不再继续重规划: error_code={msg.error_code}, message={msg.message}"
            )
            self.task_state["status"] = "failed"
            return

        # 基于反馈重新生成下一步动作
        replan_prompt = self._build_replan_prompt(self.task_state["user_goal"], msg)
        response = self.generate(replan_prompt)

        rospy.loginfo(f"[LLM] 重规划原始模型输出: {response}")

        try:
            json_str = self._extract_json_str(response)
            if not json_str:
                rospy.logerr("[LLM] 重规划时无法提取 JSON，停止")
                return

            data = json.loads(json_str)

            decision = data.get("decision", "").strip()
            next_task = data.get("next_action", None)

            if decision not in ("continue", "finish"):
                rospy.logerr(f"[LLM] 重规划时 decision 非法: {decision}")
                return

            self.task_state["last_decision"] = {"decision": decision}
            rospy.loginfo(f"[LLM] 本轮 decision: {decision}")

            if decision == "finish":
                rospy.loginfo("[LLM] decision=finish，当前任务结束")
                self.task_state["status"] = "finished"
                return

            if not isinstance(next_task, dict) or "action_type" not in next_task:
                rospy.logerr("[LLM] 重规划时未解析到有效 next_action，停止")
                return

            next_msg = self._task_to_msg(next_task)

            self.task_state["step_id"] += 1
            self._record_action(next_task)
            self.pub.publish(next_msg)
            self.task_state["status"] = "waiting_feedback"

            rospy.loginfo(f"[LLM] 重规划后发布第 {self.task_state['step_id']} 条LLM指令: {next_msg}")
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

    # 首轮专用，生成动作序列，用于replan参考
    def _build_initial_plan_prompt_rules(self) -> str:
        """
        首轮规划 prompt 规则：生成 reference_plan
        """
        return textwrap.dedent("""
            你是一个机械臂高层任务规划器。
            你的任务是把用户输入转换为一个用于后续逐步执行与重规划参考的动作序列 reference_plan。

            你必须严格按照下面的 JSON 格式输出，且只能输出 JSON，不要输出任何解释、注释、Markdown、前缀或后缀文本。

            输出格式必须为：
            {
                "reference_plan": [
                    {
                        "action_type": "pick" | "place" | "reset" | "open_gripper" | "close_gripper" | "create" | "delete",
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

            顶层规则：
            1. 顶层必须是一个 JSON 对象。
            2. 顶层只能包含 "reference_plan"。
            3. "reference_plan" 必须是数组，且至少包含一个动作。
            4. 所有字段名和 action_type 的取值必须使用英文。
            5. 除 JSON 以外，不允许输出任何其他内容。

            支持的 action_type：
            - "pick"
            - "place"
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
            - object_class_id = 1 表示 green cylinder；
            - object_class_id = 2 表示 red box；
            - object_class_id = 3 表示 yellow cylinder；
            - 也可以使用 (object_x, object_y, object_z) 指定显式抓取坐标；
            - 如果明确给出了抓取坐标，则填写 object_x/object_y/object_z，并将 object_class_id 设为 -1；
            - target 相关字段全部设为默认值。

            二、place
            表示放置动作。
            - 可以使用 target_class_id 指定放置目标类别；
            - target_class_id = 0 表示 blue box；
            - target_class_id = 1 表示 green cylinder；
            - target_class_id = 2 表示 red box；
            - target_class_id = 3 表示 yellow cylinder；
            - 也可以使用 (target_x, target_y, target_z) 指定显式放置坐标；
            - 如果明确给出了放置坐标，则填写 target_x/target_y/target_z，并将 target_class_id 设为 -1；
            - object 相关字段全部设为默认值。

            三、reset
            表示机械臂复位。
            - object 和 target 相关字段全部设为默认值。
            - 不要为了形式完整而默认在计划末尾追加 reset。
            - 只有当用户明确要求复位、归位、回到初始位置，或任务确有必要时，才加入 reset。

            四、open_gripper
            表示打开夹爪。
            - object 和 target 相关字段全部设为默认值。

            五、close_gripper
            表示关闭夹爪。
            - object 和 target 相关字段全部设为默认值。

            六、create
            表示创建物体。
            - 使用 object_class_id 表示创建物体的类别；
            - 使用 object_x, object_y, object_z 表示创建位置；
            - 使用 object_name 表示创建出来的物体名称；
            - target 相关字段全部设为默认值；
            - object_class_id = 0 表示 blue box；
            - object_class_id = 1 表示 green cylinder；
            - object_class_id = 2 表示 red box；
            - object_class_id = 3 表示 yellow cylinder。

            七、delete
            表示删除物体。
            - 使用 object_name 表示要删除的物体名称；
            - 其他字段全部设为默认值。

            默认值规则：
            1. 未使用的 class_id 字段统一填写 -1。
            2. 未使用的坐标字段统一填写 0.0。
            3. 未使用的名称字段统一填写 ""。

            规划规则（最重要）：
            1. 你的输出是 reference_plan，即参考动作序列，用于后续逐步执行与 replan 参考。
            2. reference_plan 必须按执行顺序排列。
            3. reference_plan 应覆盖完成用户目标所需的关键步骤，但不要过度细化为底层控制细节。
            4. 后续系统通常会优先沿 reference_plan 执行，因此计划应尽量稳定、合理、便于跟踪。
            5. 如果用户目标是单步任务，可以只输出一个动作。
            6. 如果用户目标包含明显顺序关系（如“然后、再、接着、最后、随后、then、and then、after that、finally”），应按顺序生成多个动作。
            7. 对于“把A放到B上/旁边/那里”这类任务，通常应规划为先 pick，再 place。
            8. 不要输出与用户目标无关的动作。
            9. 不要默认追加 reset 作为结束。
            10. 当关键步骤完成且目标达成时，任务可以自然结束。

            语义理解规则：
            1. “抓起方块”“拿起方块”“夹起方块”都应理解为 pick。
            2. “放下”“放到”“放在”如果同时包含抓取对象和目标对象，通常表示整体目标包含抓取和放置两个阶段。
            3. “把方块放到圆柱上”应优先规划为 pick + place 的顺序计划。
            4. “生成一个蓝色方块”应理解为 create。
            5. “删除名为 box1 的物体”应理解为 delete。

            稳健性要求：
            1. 必须输出合法 JSON。
            2. 不要遗漏必要字段。
            3. 不要输出除 reference_plan 之外的顶层字段。
            4. 不要输出多余文本。
        """).strip()

    def _build_initial_plan_prompt(self, user_input: str) -> str:
        """
        构造首轮 prompt：生成 reference_plan
        """
        rules = self._build_initial_plan_prompt_rules()

        prompt_template = textwrap.dedent("""
            {rules}

            用户输入：
            {user_input}

            请输出 reference_plan。
        """).strip()

        return prompt_template.replace("{rules}", rules).replace("{user_input}", user_input)

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
            表示先抓取再放置的复合技能。
            - 必须同时包含 object 信息和 target 信息；
            - object 侧遵循 pick 的填写规则；
            - target 侧遵循 place 的填写规则。
            - 但在当前系统中，默认不要优先输出 pick_place。
            - 对于“先抓再放”的任务，通常应先输出 pick，等待执行反馈后，再在后续轮次输出 place。
            - 只有当系统明确要求使用单条复合技能时，才输出 pick_place。

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
            2. “放下”“放到”“放在”如果同时包含抓取对象和目标对象，表示整体目标包含抓取和放置两个阶段。
                在当前系统中，应采用逐步执行策略，优先输出 pick 作为下一步动作，后续再根据执行反馈输出 place。
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
            6. 对于包含“先抓再放”语义的任务，默认采用逐步执行，不要在第一轮直接输出 pick_place，应先输出 pick。

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

    def _build_replan_prompt_rules(self) -> str:
        return textwrap.dedent("""
            你是一个机械臂高层决策模块。
            你的任务是根据用户整体目标、首轮 reference_plan、最近执行历史和上一轮执行反馈，
            判断当前任务应继续还是结束；如果继续，则给出下一步动作。

            你必须严格按照下面的 JSON 格式输出，且只能输出 JSON，不要输出任何解释、注释、Markdown、前缀或后缀文本。

            输出格式必须为：
            {
                "decision": "continue" | "finish",
                "next_action": {
                    "action_type": "pick" | "place" | "reset" | "open_gripper" | "close_gripper" | "create" | "delete" | "none",
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
            }

            顶层规则：
            1. 顶层必须是一个 JSON 对象。
            2. 顶层必须包含 "decision"。
            3. "decision" 只能取 "continue" 或 "finish"。
            4. 当 decision = "continue" 时，必须提供合法的 "next_action"，且 next_action.action_type 不能为 "none"。
            5. 当 decision = "finish" 时，next_action 必须存在，但所有字段使用默认空动作形式：
            - action_type = "none"
            - object_class_id = -1
            - object_name = ""
            - object_x = 0.0
            - object_y = 0.0
            - object_z = 0.0
            - target_class_id = -1
            - target_name = ""
            - target_x = 0.0
            - target_y = 0.0
            - target_z = 0.0
            这样做只是为了保持固定 JSON 结构，系统在 finish 时不会执行该动作。
            6. 所有字段名和 action_type 的取值必须使用英文。
            7. 除 JSON 以外，不允许输出任何其他内容。

            支持的 action_type：
            - "pick"
            - "place"
            - "reset"
            - "open_gripper"
            - "close_gripper"
            - "create"
            - "delete"
            - "none"

            动作填写规则：
            1. pick：
            - 使用 object_class_id 指定类别，或使用 object_x/object_y/object_z 指定显式抓取坐标；
            - 若使用显式坐标，则 object_class_id = -1；
            - target 相关字段使用默认值。

            2. place：
            - 使用 target_class_id 指定目标类别，或使用 target_x/target_y/target_z 指定显式放置坐标；
            - 若使用显式坐标，则 target_class_id = -1；
            - object 相关字段使用默认值。

            3. reset / open_gripper / close_gripper：
            - object 和 target 相关字段全部使用默认值。

            4. create：
            - 使用 object_class_id、object_name、object_x/object_y/object_z；
            - target 相关字段使用默认值。

            5. delete：
            - 使用 object_name；
            - 其他字段全部使用默认值。

            6. none：
            - 仅当 decision = "finish" 时允许使用；
            - object 和 target 相关字段全部使用默认值；
            - 它只是保持固定 JSON 结构的占位动作，系统不会执行该动作。

            默认值规则：
            1. 未使用的 class_id = -1
            2. 未使用的坐标 = 0.0
            3. 未使用的名称 = ""

            决策规则（最重要）：
            1. 默认优先参考首轮 reference_plan。
            2. 如果用户目标尚未完成，decision 应为 "continue"。
            3. 如果用户目标已经完成，且没有必要再执行后续动作，decision 应为 "finish"。
            4. 不要为了形式完整而强行输出 reset 作为结束动作；结束应通过 decision="finish" 表达。
            5. 如果上一轮动作成功，通常应沿 reference_plan 继续推进。
            6. 如果上一轮动作失败，应先结合失败原因判断是否还能继续沿 reference_plan 执行。
            7. 只有在原计划明显不再适用时，才允许偏离 reference_plan。
            8. 视觉信息只是低优先级辅助参考，不能因为看到了别的物体就随意改计划。
            9. 当参考计划中的关键步骤已经完成，且用户目标已经达成时，应输出 decision="finish"。
            10. "none" 不是等待动作、跳过动作或继续观察动作，不能用于 decision="continue"。

            稳健性要求：
            1. 必须输出合法 JSON。
            2. 不要遗漏必要字段。
            3. 不要输出多余文本。
        """).strip()
    
    def _build_replan_prompt(self, user_goal: str, feedback: AgentFeedback) -> str:
        """
        基于用户目标 + 首轮 reference_plan + 最近执行历史 + 上一轮反馈，生成下一步动作 prompt
        """
        rules = self._build_replan_prompt_rules()
        reference_plan_text = self._format_reference_plan()
        recent_history = self._format_recent_history()
        visual_context = self._format_visual_context()

        prompt_template = textwrap.dedent("""
            {rules}

            下面是当前任务的首轮参考计划 reference_plan。
            后续执行通常应优先沿着这个 reference_plan 推进，而不是重新规划整个任务。
            只有当当前执行结果已经明显偏离原计划、或继续沿原计划执行明显不合理时，才允许偏离 reference_plan。

            用户整体目标：
            {user_goal}

            首轮 reference_plan：
            {reference_plan}

            最近执行历史：
            {recent_history}

            上一轮执行反馈：
            - action_type: {action_type}
            - success: {success}
            - error_code: {error_code}
            - message: {message}

            当前视觉观测（低优先级辅助参考，仅在判断目标状态明显变化、原计划目标不可继续执行、或需要补充环境信息时再参考）：
            {visual_context}

            决策要求：
            1. 你这一轮只输出“下一步动作”的单个 JSON 对象，不要输出动作列表。
            2. 默认应优先参考 reference_plan，尽量输出与原计划一致的下一步动作。
            3. 如果上一轮动作成功，通常应沿 reference_plan 继续推进。
            4. 如果上一轮动作失败，应先结合失败原因判断是否还能继续沿 reference_plan 执行。
            5. 只有在原计划明显不再适用时，才允许输出偏离原计划的动作。
            6. 不要因为视觉信息中出现其他物体就随意改计划；视觉信息只是低优先级辅助参考。
            7. 如果用户目标已经完成，应输出 decision="finish"。
            8. 如果用户目标尚未完成，应输出 decision="continue"，并提供 next_action。

            请输出 decision 和 next_action。
        """).strip()

        return (
            prompt_template
            .replace("{rules}", rules)
            .replace("{user_goal}", user_goal)
            .replace("{reference_plan}", reference_plan_text)
            .replace("{recent_history}", recent_history)
            .replace("{action_type}", str(feedback.action_type))
            .replace("{success}", str(feedback.success))
            .replace("{error_code}", str(feedback.error_code))
            .replace("{message}", str(feedback.message))
            .replace("{visual_context}", visual_context)
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

    # 用户输入goal后运行一次，后续流程在feedback的回调里继续
    def process_user_input(self, user_input: str):
        """
        处理用户输入：首轮只生成并发布一个 action。
        """
        prompt = self._build_initial_plan_prompt(user_input)
        response = self.generate(prompt)

        try:
            json_str = self._extract_json_str(response)
            if not json_str:
                rospy.logerr("[LLM] 无法从LLM响应中提取JSON")
                rospy.logerr(f"[LLM] 原始响应: {response}")
                return None

            data = json.loads(json_str)
            reference_plan = data.get("reference_plan", [])

            if not isinstance(reference_plan, list) or not reference_plan:
                rospy.logerr("[LLM] 首轮输出中未找到有效 reference_plan")
                rospy.logerr(f"[LLM] 解析后的数据: {data}")
                return None

            first_task = reference_plan[0]
            msg = self._task_to_msg(first_task)

            self.task_state["reference_plan"] = reference_plan
            self.task_state["step_id"] = 1
            self._record_action(first_task)
            self.pub.publish(msg)
            self.task_state["status"] = "waiting_feedback"

            rospy.loginfo(f"[LLM] 已保存首轮 reference_plan，steps={len(reference_plan)}")
            rospy.loginfo(f"[LLM] 发布第 {self.task_state['step_id']} 条LLM指令: {msg}")
            rospy.loginfo("[LLM] 进入等待 feedback 状态")

            return [msg]

        except Exception as e:
            rospy.logerr(f"[LLM] 解析LLM响应时出错: {e}")
            rospy.logerr(f"[LLM] 原始响应: {response}")
            return None