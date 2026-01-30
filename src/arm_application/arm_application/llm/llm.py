import json
import os
import rospy
import textwrap
from arm_application.msg import LLMCommands
from std_msgs.msg import String
from typing import Dict, Any, Optional
import dashscope


class TongyiQianwenLLM:
    """通义千问LLM集成类"""
    
    def __init__(self, api_key: Optional[str] = None):
        """初始化通义千问LLM
        
        Args:
            api_key: 通义千问API密钥,如果为None则从环境变量DASHSCOPE_API_KEY获取
        """
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Please set DASHSCOPE_API_KEY environment variable or provide it as parameter.")
        
        dashscope.api_key = self.api_key
        self.model = "qwen-max"  # 可以根据需要选择其他模型
        
        # 初始化ROS发布器
        self.pub = rospy.Publisher('/llm_commands', LLMCommands, queue_size=10)
        self.sub = rospy.Subscriber('/llm_user_input', String, self._user_input_callback)
        rospy.loginfo("LLM 节点已启动，等待用户输入...")
    
    def _user_input_callback(self, msg):
        """处理用户输入话题的回调函数
        
        Args:
            msg: 包含用户自然语言指令的String消息
        """
        user_input = msg.data
        rospy.loginfo(f"收到用户输入: {user_input}")
        self.process_user_input(user_input)

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """调用通义千问API生成文本
        
        Args:
            prompt: 提示词
            max_tokens: 最大生成token数
            temperature: 生成温度,值越大越随机
        
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
            rospy.loginfo(f"Error calling Tongyi Qianwen API: {e}")
            return ""
    
    def process_user_input(self, user_input: str):
        """处理用户输入并发布LLMCommands消息
        
        Args:
            user_input: 用户输入的自然语言指令
        """
        # 预设的prompt模板,引导模型输出指定格式
        prompt_template = textwrap.dedent("""
        你是一个机械臂控制助手,需要将用户的自然语言指令转换为机械臂可执行的命令格式。

        请根据用户输入,生成以下格式的JSON响应:
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

        说明:
        1. action_type 必须是以下之一:pick(抓取)、place(放置)、pick_place(抓取并放置)、reset(复位)、open_gripper(打开夹爪)、close_gripper(关闭夹爪)、create(创建物体)、delete(删除物体)
        2. 对于 pick 动作,可以使用object_class_id或(object_x, object_y, object_z),当显示指定抓取位置时,只用(object_x, object_y, object_z);未显示指定时,使用object_class_id
        3. 对于 place 动作,可以使用target_class_id或(target_x, target_y, target_z),当显示指定放置位置时,只用(target_x, target_y, target_z);未显示指定时,使用target_class_id
        4. 对于 pick_place 动作,结合2、3点即可,先抓取object,再放置到target
        5. 当用户要求创建或者放置物体时,使用 create 动作, 用(object_x, object_y, object_z)来代表生成的位置,用object_class_id 0 表示创建蓝色方块,1 表示创建绿色圆柱体,在创建物体时要起个名字,用object_name来指定, 
        6. 当用户要求删除物体时,使用 delete 动作, 使用object_name指定要删除的物体的名字
        7. 对于不需要的字段,请设置为默认值:class_id 为 -1,坐标为 0.0,名称为 ""
        8. 请确保生成的是有效的JSON格式,不要包含任何额外的文本
        9. 请直接输出JSON,不要包含任何前缀或后缀文本,所有的内容用英文
        用户输入:
        {user_input}

        请生成符合上述格式的JSON响应:
        """).strip()
        # 填充用户输入
        prompt = prompt_template.replace("{user_input}", user_input)
        # 调用LLM生成响应
        response = self.generate(prompt)
        
        # 解析响应并发布消息
        try:
            # 提取JSON部分
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                command_dict = json.loads(json_str)
                
                # 创建并填充LLMCommands消息
                msg = LLMCommands()
                msg.action_type = command_dict.get("action_type", "reset")
                msg.object_class_id = command_dict.get("object_class_id", -1)
                msg.object_name = command_dict.get("object_name", "")
                msg.object_x = command_dict.get("object_x", 0.0)
                msg.object_y = command_dict.get("object_y", 0.0)
                msg.object_z = command_dict.get("object_z", 0.0)
                msg.target_class_id = command_dict.get("target_class_id", -1)
                msg.target_name = command_dict.get("target_name", "")
                msg.target_x = command_dict.get("target_x", 0.0)
                msg.target_y = command_dict.get("target_y", 0.0)
                msg.target_z = command_dict.get("target_z", 0.0)
                
                # 发布消息
                self.pub.publish(msg)
                rospy.loginfo(f"发布LLM指令: {msg}")
                return msg
            else:
                rospy.logerr("无法从LLM响应中提取JSON")
                return None
                
        except Exception as e:
            rospy.logerr(f"解析LLM响应时出错: {e}")
            return None