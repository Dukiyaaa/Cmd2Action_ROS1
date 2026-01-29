import sys
import os
# 添加包路径（确保能 import arm_application）
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import rospy
from planners.task_planner import TaskPlanner
from controllers.scara_controller import ScaraController
from arm_vision.msg import DetectedObjectPool
from geometry_msgs.msg import PoseStamped
from arm_application.msg import LLMCommands
from agents.object_detector import ObjectDetector
from typing import List, Tuple, Any

class Agent:
    def __init__(self):
        # 初始化依赖模块
        self.task_planner = TaskPlanner()
        self.controller = ScaraController()
        self.object_detector = ObjectDetector()
        self.controller.reset()

        # 订阅 LLM 指令 用于llm向agent发解析后的需求
        self.sub = rospy.Subscriber('/llm_commands', LLMCommands, self._llm_callback)
        rospy.loginfo("Agent 已启动，等待 LLM 指令...")
        
    def _llm_callback(self, msg):
        rospy.loginfo("收到 LLM 指令: action='%s', class_id=%d" % (msg.action_type, msg.target_class_id))
            
        if msg.action_type == "pick":
            # 查询目标物体位置
            obj_pose = self.object_detector.get_position(msg.target_class_id)
            if obj_pose is None:
                rospy.logerr("未检测到 class_id=%d 的物体！" % msg.target_class_id)
                return
            target_pose = (0,0,0)

            # 调用 Planner 获取动作序列 
            task_spec = {
                "action": msg.action_type,
                "object": obj_pose,
                "target": target_pose
            }
            action_sequence = self.task_planner.plan(task_spec)
            rospy.loginfo(f'{action_sequence}')

            # 执行动作序列
            self._execute_action_sequence(action_sequence)
        elif msg.action_type == "place":
            # 查询目标物体位置
            obj_pose = (0,0,0)
            target_pose = (0,-1.8,0.05)

            # 调用 Planner 获取动作序列 
            task_spec = {
                "action": msg.action_type,
                "object": obj_pose,
                "target": target_pose
            }
            action_sequence = self.task_planner.plan(task_spec)
            rospy.loginfo(f'{action_sequence}')

            # 执行动作序列
            self._execute_action_sequence(action_sequence)

    def _execute_action_sequence(self, seq: List[Tuple[str, ...]]):
        for action in seq:
            method_name = action[0]
            args = action[1:]
            rospy.loginfo(f'method_name:{method_name} args:{args}')
            if method_name == "move_to":
                self.controller.move_to(*args)
            elif method_name == "open_gripper":
                self.controller.open_gripper()
            elif method_name == "close_gripper":
                self.controller.close_gripper()
            elif method_name == "reset":
                self.controller.reset()
            elif method_name == "align_gripper_roll":
                self.controller.align_gripper_roll()
            else:
                rospy.logwarn(f"Unknown action: {method_name}")