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
from utils.gazebo_box_display import BoxSpawner
from utils.gazebo_cylinder_display import CylinderSpawner
from typing import List, Tuple, Any

class Agent:
    def __init__(self):
        # 初始化依赖模块
        self.task_planner = TaskPlanner()
        self.controller = ScaraController()
        self.object_detector = ObjectDetector()
        self.box_spawner = BoxSpawner()
        self.cyl_spawner = CylinderSpawner()
        self.controller.reset()

        # 订阅 LLM 指令 用于llm向agent发解析后的需求
        self.sub = rospy.Subscriber('/llm_commands', LLMCommands, self._llm_callback)
        rospy.loginfo("Agent 已启动，等待 LLM 指令...")
        
    def _llm_callback(self, msg):            
        if msg.action_type == "pick":
            if msg.object_x != 0.0 or msg.object_y != 0.0 or msg.object_z != 0.0:
                obj_pose = (msg.object_x, msg.object_y, msg.object_z)
                rospy.loginfo(f"使用显式抓取坐标: {obj_pose}")
            elif msg.object_class_id != -1:
                obj_pose = self.object_detector.get_position(msg.object_class_id)
                if obj_pose is None:
                    rospy.logerr(f"视觉未检测到 object_class_id={msg.object_class_id} 的物体！")
                    return
                rospy.loginfo(f"从视觉获取抓取位置: {obj_pose}")
            else:
                rospy.logerr("pick/pick_place 动作未提供 object_class_id 或 object 坐标！")
                return

            # 调用 Planner 获取动作序列 
            task_spec = {
                "action": msg.action_type,
                "object": obj_pose,
                "target": (-1,-1,-1)
            }
            action_sequence = self.task_planner.plan(task_spec)
            rospy.loginfo(f'{action_sequence}')

            # 执行动作序列
            self._execute_action_sequence(action_sequence)
        elif msg.action_type == "place":
            # 1. 优先使用显式坐标
            if msg.target_x != 0.0 or msg.target_y != 0.0 or msg.target_z != 0.0:
                target_pose = (msg.target_x, msg.target_y, msg.target_z)
                rospy.loginfo(f"使用显式放置坐标: {target_pose}")
            # 2. 否则用 class_id 查询视觉（如“放到蓝色托盘上”）
            elif msg.target_class_id != -1:
                target_pose = self.object_detector.get_position(msg.target_class_id)
                if target_pose is None:
                    rospy.logerr(f"视觉未检测到 target_class_id={msg.target_class_id} 的放置目标！")
                    return
                rospy.loginfo(f"从视觉获取放置位置: {target_pose}")
            else:
                rospy.logerr("place/pick_place 动作未提供 target_class_id 或 target 坐标！")
                return

            # 调用 Planner 获取动作序列 
            task_spec = {
                "action": msg.action_type,
                "object": (-1,-1,-1),
                "target": target_pose
            }
            action_sequence = self.task_planner.plan(task_spec)
            rospy.loginfo(f'{action_sequence}')

            # 执行动作序列
            self._execute_action_sequence(action_sequence)
        elif msg.action_type == "pick_place":
            # 1. 优先使用显式坐标
            if msg.object_x != 0.0 or msg.object_y != 0.0 or msg.object_z != 0.0:
                obj_pose = (msg.object_x, msg.object_y, msg.object_z)
                rospy.loginfo(f"使用显式抓取坐标: {obj_pose}")
            # 2. 否则用 class_id 查询视觉（如“抓取红色方块”）
            elif msg.object_class_id != -1:
                obj_pose = self.object_detector.get_position(msg.object_class_id)
                if obj_pose is None:
                    rospy.logerr(f"视觉未检测到 object_class_id={msg.object_class_id} 的物体！")
                    return
                rospy.loginfo(f"从视觉获取抓取位置: {obj_pose}")
            else:
                rospy.logerr("pick/pick_place 动作未提供 object_class_id 或 object 坐标！")
                return

            # 1. 优先使用显式坐标
            if msg.target_x != 0.0 or msg.target_y != 0.0 or msg.target_z != 0.0:
                target_pose = (msg.target_x, msg.target_y, msg.target_z)
                rospy.loginfo(f"使用显式放置坐标: {target_pose}")
            # 2. 否则用 class_id 查询视觉（如“放到蓝色托盘上”）
            elif msg.target_class_id != -1:
                target_pose = self.object_detector.get_position(msg.target_class_id)
                if target_pose is None:
                    rospy.logerr(f"视觉未检测到 target_class_id={msg.target_class_id} 的放置目标！")
                    return
                rospy.loginfo(f"从视觉获取放置位置: {target_pose}")
            else:
                rospy.logerr("place/pick_place 动作未提供 target_class_id 或 target 坐标！")
                return

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
        elif msg.action_type == "reset" or msg.action_type == "open_gripper" or msg.action_type == "close_gripper":
            # 调用 Planner 获取动作序列 
            task_spec = {
                "action": msg.action_type,
                "object": (-1,-1,-1),
                "target": (-1,-1,-1)
            }
            action_sequence = self.task_planner.plan(task_spec)
            rospy.loginfo(f'{action_sequence}')

            # 执行动作序列
            self._execute_action_sequence(action_sequence)
        elif msg.action_type == "create":
            if msg.object_class_id == 0:
                box_x, box_y, box_z = msg.object_x,msg.object_y,msg.object_z
                box_name = msg.object_name
                self.box_spawner.display_test_box(
                    box_pos=(box_x, box_y, box_z),
                    box_name=box_name
                )
            elif msg.object_class_id == 1:
                cyl_x, cyl_y, cyl_z = msg.object_x,msg.object_y,msg.object_z
                cyl_name = msg.object_name
                self.cyl_spawner.display_test_cylinder(
                    cyl_pos=(cyl_x, cyl_y, cyl_z),
                    cyl_name=cyl_name
                )
        elif msg.action_type == "delete":
            obj_name = msg.object_name
            self.box_spawner.delete_entity(obj_name)

    def _execute_action_sequence(self, seq: List[Tuple[str, ...]]):
        for action in seq:
            method_name = action[0]
            args = action[1:]
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