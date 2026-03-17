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
from utils.action_result import ActionResult

from config import (
    ACTION_PICK,
    ACTION_PLACE,
    ACTION_PICK_PLACE,
    ACTION_RESET,
    ACTION_OPEN_GRIPPER,
    ACTION_CLOSE_GRIPPER,
    ACTION_CREATE,
    ACTION_DELETE,
    ACTION_MOVE_TO,
    ACTION_ALIGN_GRIPPER_ROLL,
    ACTION_GRIPPER_DOWN,
    EMPTY_POSE,
    INVALID_CLASS_ID,
    OBJECT_CLASS_BLUE_BOX,
    OBJECT_CLASS_GREEN_CYLINDER,
    OBJECT_CLASS_RED_BOX,
    OBJECT_CLASS_YELLOW_CYLINDER,
)

class Agent:
    def __init__(self):
        # 初始化依赖模块
        self.task_planner = TaskPlanner()
        self.controller = ScaraController()
        self.object_detector = ObjectDetector()
        self.box_spawner = BoxSpawner()
        self.cyl_spawner = CylinderSpawner()
        self.controller.reset()

        # 订阅 LLM 指令
        self.sub = rospy.Subscriber('/llm_commands', LLMCommands, self._llm_callback)

        # 订阅 GUI 手动控制指令
        self.gui_move_to_sub = rospy.Subscriber(
            '/gui/move_to_pose',
            PoseStamped,
            self._gui_move_to_callback
        )
        self.gui_reset_sub = rospy.Subscriber(
            '/gui/reset',
            PoseStamped,
            self._gui_reset_callback
        )
        self.gui_open_gripper_sub = rospy.Subscriber(
            '/gui/open_gripper',
            PoseStamped,
            self._gui_open_gripper_callback
        )
        self.gui_close_gripper_sub = rospy.Subscriber(
            '/gui/close_gripper',
            PoseStamped,
            self._gui_close_gripper_callback
        )
        self.gui_align_gripper_roll_sub = rospy.Subscriber(
            '/gui/align_gripper_roll',
            PoseStamped,
            self._gui_align_gripper_roll_callback
        )
        self.gui_gripper_down_sub = rospy.Subscriber(
            '/gui/gripper_down',
            PoseStamped,
            self._gui_gripper_down_callback
        )
        self.gui_pick_sub = rospy.Subscriber(
            '/gui/pick',
            PoseStamped,
            self._gui_pick_callback
        )
        self.gui_place_sub = rospy.Subscriber(
            '/gui/place',
            PoseStamped,
            self._gui_place_callback
        )
        rospy.loginfo("Agent 已启动,等待 LLM / GUI 指令...")
        
    def _llm_callback(self, msg):
        if msg.action_type == ACTION_PICK:
            if msg.object_x != 0.0 or msg.object_y != 0.0 or msg.object_z != 0.0:
                obj_pose = (msg.object_x, msg.object_y, msg.object_z)
                rospy.loginfo(f"使用显式抓取坐标: {obj_pose}")
            elif msg.object_class_id != INVALID_CLASS_ID:
                obj_pose = self.object_detector.get_best_position(msg.object_class_id)
                if obj_pose is None:
                    rospy.logerr(f"视觉未检测到 object_class_id={msg.object_class_id} 的物体！")
                    return
                rospy.loginfo(f"从视觉获取抓取位置: {obj_pose}")
            else:
                rospy.logerr("pick/pick_place 动作未提供 object_class_id 或 object 坐标！")
                return

            task_spec = {
                "action": msg.action_type,
                "object": obj_pose,
                "target": EMPTY_POSE
            }
            action_sequence = self.task_planner.plan(task_spec)
            rospy.loginfo(f"{action_sequence}")

            result = self._execute_action_sequence(action_sequence)
            if not result.success:
                rospy.logerr(
                    f"LLM pick failed: code={result.error_code}, msg={result.message}"
                )
                return

            rospy.loginfo("LLM pick executed")

        elif msg.action_type == ACTION_PLACE:
            if msg.target_x != 0.0 or msg.target_y != 0.0 or msg.target_z != 0.0:
                target_pose = (msg.target_x, msg.target_y, msg.target_z)
                rospy.loginfo(f"使用显式放置坐标: {target_pose}")
            elif msg.target_class_id != INVALID_CLASS_ID:
                target_pose = self.object_detector.get_best_position(msg.target_class_id)
                if target_pose is None:
                    rospy.logerr(f"视觉未检测到 target_class_id={msg.target_class_id} 的放置目标！")
                    return
                rospy.loginfo(f"从视觉获取放置位置: {target_pose}")
            else:
                rospy.logerr("place/pick_place 动作未提供 target_class_id 或 target 坐标！")
                return

            task_spec = {
                "action": msg.action_type,
                "object": EMPTY_POSE,
                "target": target_pose
            }
            action_sequence = self.task_planner.plan(task_spec)
            rospy.loginfo(f"{action_sequence}")

            result = self._execute_action_sequence(action_sequence)
            if not result.success:
                rospy.logerr(
                    f"LLM place failed: code={result.error_code}, msg={result.message}"
                )
                return

            rospy.loginfo("LLM place executed")

        elif msg.action_type == ACTION_PICK_PLACE:
            if msg.object_x != 0.0 or msg.object_y != 0.0 or msg.object_z != 0.0:
                obj_pose = (msg.object_x, msg.object_y, msg.object_z)
                rospy.loginfo(f"使用显式抓取坐标: {obj_pose}")
            elif msg.object_class_id != INVALID_CLASS_ID:
                obj_pose = self.object_detector.get_best_position(msg.object_class_id)
                if obj_pose is None:
                    rospy.logerr(f"视觉未检测到 object_class_id={msg.object_class_id} 的物体！")
                    return
                rospy.loginfo(f"从视觉获取抓取位置: {obj_pose}")
            else:
                rospy.logerr("pick/pick_place 动作未提供 object_class_id 或 object 坐标！")
                return

            if msg.target_x != 0.0 or msg.target_y != 0.0 or msg.target_z != 0.0:
                target_pose = (msg.target_x, msg.target_y, msg.target_z)
                rospy.loginfo(f"使用显式放置坐标: {target_pose}")
            elif msg.target_class_id != INVALID_CLASS_ID:
                target_pose = self.object_detector.get_best_position(msg.target_class_id)
                if target_pose is None:
                    rospy.logerr(f"视觉未检测到 target_class_id={msg.target_class_id} 的放置目标！")
                    return
                rospy.loginfo(f"从视觉获取放置位置: {target_pose}")
            else:
                rospy.logerr("place/pick_place 动作未提供 target_class_id 或 target 坐标！")
                return

            task_spec = {
                "action": msg.action_type,
                "object": obj_pose,
                "target": target_pose
            }
            action_sequence = self.task_planner.plan(task_spec)
            rospy.loginfo(f"{action_sequence}")

            result = self._execute_action_sequence(action_sequence)
            if not result.success:
                rospy.logerr(
                    f"LLM pick_place failed: code={result.error_code}, msg={result.message}"
                )
                return

            rospy.loginfo("LLM pick_place executed")

        elif msg.action_type in (ACTION_RESET, ACTION_OPEN_GRIPPER, ACTION_CLOSE_GRIPPER):
            task_spec = {
                "action": msg.action_type,
                "object": EMPTY_POSE,
                "target": EMPTY_POSE
            }
            action_sequence = self.task_planner.plan(task_spec)
            rospy.loginfo(f"{action_sequence}")

            result = self._execute_action_sequence(action_sequence)
            if not result.success:
                rospy.logerr(
                    f"LLM {msg.action_type} failed: code={result.error_code}, msg={result.message}"
                )
                return

            rospy.loginfo(f"LLM {msg.action_type} executed")

        elif msg.action_type == ACTION_CREATE:
            if msg.object_class_id == OBJECT_CLASS_BLUE_BOX:
                box_x, box_y, box_z = msg.object_x, msg.object_y, msg.object_z
                box_name = msg.object_name
                self.box_spawner.display_test_box(
                    box_pos=(box_x, box_y, box_z),
                    box_color=(0.2, 0.6, 0.9, 1.0),
                    box_name=box_name
                )
            elif msg.object_class_id == OBJECT_CLASS_GREEN_CYLINDER:
                cyl_x, cyl_y, cyl_z = msg.object_x, msg.object_y, msg.object_z
                cyl_name = msg.object_name
                self.cyl_spawner.display_test_cylinder(
                    cyl_pos=(cyl_x, cyl_y, cyl_z),
                    cyl_color=(0.2, 0.8, 0.2, 1.0),
                    cyl_name=cyl_name
                )
            elif msg.object_class_id == OBJECT_CLASS_RED_BOX:
                box_x, box_y, box_z = msg.object_x, msg.object_y, msg.object_z
                box_name = msg.object_name
                self.box_spawner.display_test_box(
                    box_pos=(box_x, box_y, box_z),
                    box_color=(0.9, 0.2, 0.2, 1.0),
                    box_name=box_name
                )
            elif msg.object_class_id == OBJECT_CLASS_YELLOW_CYLINDER:
                cyl_x, cyl_y, cyl_z = msg.object_x, msg.object_y, msg.object_z
                cyl_name = msg.object_name
                self.cyl_spawner.display_test_cylinder(
                    cyl_pos=(cyl_x, cyl_y, cyl_z),
                    cyl_color=(0.95, 0.85, 0.2, 1.0),
                    cyl_name=cyl_name
                )

        elif msg.action_type == ACTION_DELETE:
            obj_name = msg.object_name
            self.box_spawner.delete_entity(obj_name)

    def _execute_action_sequence(self, seq: List[Tuple[str, ...]]) -> ActionResult:
        max_retry = 1

        def _attempt_recovery(failed_step: int, failed_action: str):
            rospy.logwarn(
                f"attempting recovery after failure at step={failed_step}, action={failed_action}"
            )
            recovery_result = self.controller.reset()
            if recovery_result.success:
                rospy.loginfo("recovery succeeded: reset completed")
            else:
                rospy.logerr(
                    f"recovery failed: code={recovery_result.error_code}, msg={recovery_result.message}"
                )

        for idx, action in enumerate(seq):
            method_name = action[0]
            args = action[1:]

            attempt = 0
            while attempt <= max_retry:
                if method_name == ACTION_MOVE_TO:
                    result = self.controller.move_to(*args)

                elif method_name == ACTION_OPEN_GRIPPER:
                    result = self.controller.open_gripper()

                elif method_name == ACTION_CLOSE_GRIPPER:
                    result = self.controller.close_gripper()

                elif method_name == ACTION_RESET:
                    result = self.controller.reset()

                elif method_name == ACTION_ALIGN_GRIPPER_ROLL:
                    result = self.controller.align_gripper_roll()

                elif method_name == ACTION_GRIPPER_DOWN:
                    result = self.controller.gripper_down(*args)

                else:
                    rospy.logwarn(f"Unknown action: {method_name}")
                    unknown_result = ActionResult.fail(
                        "UNKNOWN_ACTION",
                        f"unknown action: {method_name}",
                        retryable=False
                    )
                    _attempt_recovery(idx, method_name)
                    return unknown_result

                if result.success:
                    rospy.loginfo(
                        f"action succeeded at step={idx}, action={method_name}, attempt={attempt + 1}"
                    )
                    break

                rospy.logwarn(
                    f"action failed at step={idx}, action={method_name}, attempt={attempt + 1}, "
                    f"code={result.error_code}, msg={result.message}, retryable={result.retryable}"
                )

                if not result.retryable or attempt >= max_retry:
                    _attempt_recovery(idx, method_name)
                    return result

                attempt += 1
                rospy.loginfo(
                    f"retrying action at step={idx}, action={method_name}, next_attempt={attempt + 1}"
                )

        return ActionResult.ok("action sequence finished")
        
    def _gui_move_to_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        rospy.loginfo(
            f"[GUI] move_to received: x={x:.3f}, y={y:.3f}, z={z:.3f}, frame={msg.header.frame_id}"
        )

        try:
            result = self.controller.move_to(x, y, z)
            if not result.success:
                rospy.logerr(
                    f"[GUI] move_to failed: code={result.error_code}, msg={result.message}"
                )
                return

            rospy.loginfo(f"[GUI] move_to executed: ({x:.3f}, {y:.3f}, {z:.3f})")
        except Exception as e:
            rospy.logerr(f"[GUI] move_to exception: {e}")

    def _gui_reset_callback(self, msg):
        rospy.loginfo("[GUI] reset received")

        try:
            result = self.controller.reset()
            if not result.success:
                rospy.logerr(
                    f"[GUI] reset failed: code={result.error_code}, msg={result.message}"
                )
                return

            rospy.loginfo("[GUI] reset executed")
        except Exception as e:
            rospy.logerr(f"[GUI] reset exception: {e}")

    def _gui_open_gripper_callback(self, msg):
        rospy.loginfo("[GUI] open_gripper received")

        try:
            result = self.controller.open_gripper()
            if not result.success:
                rospy.logerr(
                    f"[GUI] open_gripper failed: code={result.error_code}, msg={result.message}"
                )
                return

            rospy.loginfo("[GUI] open_gripper executed")
        except Exception as e:
            rospy.logerr(f"[GUI] open_gripper exception: {e}")

    def _gui_close_gripper_callback(self, msg):
        rospy.loginfo("[GUI] close_gripper received")

        try:
            result = self.controller.close_gripper()
            if not result.success:
                rospy.logerr(
                    f"[GUI] close_gripper failed: code={result.error_code}, msg={result.message}"
                )
                return

            rospy.loginfo("[GUI] close_gripper executed")
        except Exception as e:
            rospy.logerr(f"[GUI] close_gripper exception: {e}")
        
    def _gui_align_gripper_roll_callback(self, msg):
        rospy.loginfo("[GUI] align_gripper_roll received")

        try:
            result = self.controller.align_gripper_roll()
            if not result.success:
                rospy.logerr(
                    f"[GUI] align_gripper_roll failed: code={result.error_code}, msg={result.message}"
                )
                return

            rospy.loginfo("[GUI] align_gripper_roll executed")
        except Exception as e:
            rospy.logerr(f"[GUI] align_gripper_roll exception: {e}")

    def _gui_gripper_down_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y

        rospy.loginfo(f"[GUI] gripper_down received: x={x:.3f}, y={y:.3f}")

        try:
            result = self.controller.gripper_down(x, y)
            if not result.success:
                rospy.logerr(
                    f"[GUI] gripper_down failed: code={result.error_code}, msg={result.message}"
                )
                return

            rospy.loginfo(f"[GUI] gripper_down executed: ({x:.3f}, {y:.3f})")
        except Exception as e:
            rospy.logerr(f"[GUI] gripper_down exception: {e}")

    def _gui_pick_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        obj_pose = (x, y, z)

        rospy.loginfo(
            f"[GUI] pick received: x={x:.3f}, y={y:.3f}, z={z:.3f}, frame={msg.header.frame_id}"
        )

        try:
            task_spec = {
                "action": ACTION_PICK,
                "object": obj_pose,
                "target": EMPTY_POSE
            }

            action_sequence = self.task_planner.plan(task_spec)
            rospy.loginfo(f"[GUI] pick planned: {action_sequence}")

            result = self._execute_action_sequence(action_sequence)
            if not result.success:
                rospy.logerr(
                    f"[GUI] pick failed: code={result.error_code}, msg={result.message}"
                )
                return

            rospy.loginfo("[GUI] pick executed")
        except Exception as e:
            rospy.logerr(f"[GUI] pick failed: {e}")

    def _gui_place_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        target_pose = (x, y, z)

        rospy.loginfo(
            f"[GUI] place received: x={x:.3f}, y={y:.3f}, z={z:.3f}, frame={msg.header.frame_id}"
        )

        try:
            task_spec = {
                "action": ACTION_PLACE,
                "object": EMPTY_POSE,
                "target": target_pose
            }

            action_sequence = self.task_planner.plan(task_spec)
            rospy.loginfo(f"[GUI] place planned: {action_sequence}")

            result = self._execute_action_sequence(action_sequence)
            if not result.success:
                rospy.logerr(
                    f"[GUI] place failed: code={result.error_code}, msg={result.message}"
                )
                return

            rospy.loginfo("[GUI] place executed")
        except Exception as e:
            rospy.logerr(f"[GUI] place exception: {e}")