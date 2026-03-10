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
        You are a robotic arm command parser.
        Your job is to convert the user's natural language instruction into a valid JSON object.

        Output format must be exactly:
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

        Rules:
        1. Always output a JSON object with a top-level key "tasks".
        2. "tasks" must be a list.
        3. If the user gives only one action, "tasks" contains exactly one item.
        4. If the user gives multiple actions in one sentence, split them into multiple task items in execution order.
        5. Supported action_type values are:
           - "pick"
           - "place"
           - "pick_place"
           - "reset"
           - "open_gripper"
           - "close_gripper"
           - "create"
           - "delete"

        Action rules:
        6. For "pick":
           - use object_class_id OR (object_x, object_y, object_z)
           - if explicit coordinates are given, use coordinates and set object_class_id = -1
        7. For "place":
           - use target_class_id OR (target_x, target_y, target_z)
           - if explicit coordinates are given, use coordinates and set target_class_id = -1
        8. For "pick_place":
           - include both object and target information
        9. For "create":
           - use object_class_id to indicate object type
           - use object_x, object_y, object_z for spawn position
           - use object_name for the generated object's name
           - object_class_id = 0 means blue box
           - object_class_id = 1 means green cylinder
        10. For "delete":
           - use object_name to indicate which object to delete

        Default values:
        11. For unused class_id fields, use -1
        12. For unused coordinate fields, use 0.0
        13. For unused name fields, use ""

        Important interpretation rules:
        14. Words like "then", "and then", "after that", "finally", "再", "然后", "最后", "接着" indicate multiple sequential tasks.
        15. For commands like "put the box on the cylinder, then reset", output:
            - first task: "pick_place"
            - second task: "reset"
        16. Do not output explanations, markdown, comments, or extra text.
        17. Output all keys and string values in English only.
        18. The response must be valid JSON.

        User input:
        {user_input}

        Return only the JSON object.
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