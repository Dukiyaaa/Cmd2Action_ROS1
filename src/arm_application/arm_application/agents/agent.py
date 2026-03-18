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
            obj_pose, resolved_class_id = self._resolve_pose(
                role="pick object",
                x=msg.object_x,
                y=msg.object_y,
                z=msg.object_z,
                class_id=msg.object_class_id,
                source="LLM",
                infer_class_from_pose=True
            )
            if obj_pose is None:
                return

            result = self._execute_pick(
                obj_pose=obj_pose,
                class_id=resolved_class_id,
                source="LLM"
            )

            if not result.success:
                return
        elif msg.action_type == ACTION_PLACE:
            target_pose, _ = self._resolve_pose(
                role="place target",
                x=msg.target_x,
                y=msg.target_y,
                z=msg.target_z,
                class_id=msg.target_class_id,
                source="LLM",
                infer_class_from_pose=False
            )
            if target_pose is None:
                return

            result = self._execute_place(
                target_pose=target_pose,
                source="LLM"
            )

            if not result.success:
                return
        elif msg.action_type == ACTION_PICK_PLACE:
            obj_pose, resolved_class_id = self._resolve_pose(
                role="pick_place object",
                x=msg.object_x,
                y=msg.object_y,
                z=msg.object_z,
                class_id=msg.object_class_id,
                source="LLM",
                infer_class_from_pose=True
            )
            if obj_pose is None:
                return

            target_pose, _ = self._resolve_pose(
                role="pick_place target",
                x=msg.target_x,
                y=msg.target_y,
                z=msg.target_z,
                class_id=msg.target_class_id,
                source="LLM",
                infer_class_from_pose=False
            )
            if target_pose is None:
                return

            pick_result = self._execute_pick(
                obj_pose=obj_pose,
                class_id=resolved_class_id,
                source="LLM"
            )
            if not pick_result.success:
                rospy.logerr(
                    f"[LLM] pick_place failed in pick stage: "
                    f"code={pick_result.error_code}, msg={pick_result.message}"
                )
                return

            place_result = self._execute_place(
                target_pose=target_pose,
                source="LLM"
            )
            if not place_result.success:
                rospy.logerr(
                    f"[LLM] pick_place failed in place stage: "
                    f"code={place_result.error_code}, msg={place_result.message}"
                )
                return

            rospy.loginfo("[LLM] pick_place executed")

        elif msg.action_type in (ACTION_RESET, ACTION_OPEN_GRIPPER, ACTION_CLOSE_GRIPPER):
            rospy.loginfo(f"[LLM] {msg.action_type} received")

            task_spec = {
                "action": msg.action_type,
                "object": EMPTY_POSE,
                "target": EMPTY_POSE
            }

            result = self._execute_action_sequence(self.task_planner.plan(task_spec))
            if not result.success:
                rospy.logerr(
                    f"[LLM] {msg.action_type} failed: code={result.error_code}, msg={result.message}"
                )

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
                    rospy.logwarn(f"unknown action: {method_name}")
                    unknown_result = ActionResult.fail(
                        "UNKNOWN_ACTION",
                        f"unknown action: {method_name}",
                        retryable=False
                    )
                    _attempt_recovery(idx, method_name)
                    return unknown_result

                if result.success:
                    break

                rospy.logwarn(
                    f"action failed at step={idx}, action={method_name}, attempt={attempt + 1}, "
                    f"code={result.error_code}, msg={result.message}, retryable={result.retryable}"
                )

                if not result.retryable or attempt >= max_retry:
                    _attempt_recovery(idx, method_name)
                    return result

                attempt += 1
                rospy.logwarn(
                    f"retrying action at step={idx}, action={method_name}, next_attempt={attempt + 1}"
                )

        return ActionResult.ok("action sequence finished")
    
    def _execute_pick(self, obj_pose, class_id=None, source="Agent"):
        task_spec = {
            "action": ACTION_PICK,
            "object": obj_pose,
            "target": EMPTY_POSE
        }

        result = self._execute_action_sequence(self.task_planner.plan(task_spec))
        if not result.success:
            rospy.logerr(
                f"[{source}] pick failed: code={result.error_code}, msg={result.message}"
            )
            return result

        verified = self._verify_pick_result(
            original_pose=obj_pose,
            class_id=class_id
        )
        if not verified:
            verify_result = ActionResult.fail(
                "PICK_VERIFY_FAILED",
                "pick verification failed",
                retryable=False
            )
            rospy.logerr(f"[{source}] pick failed: verification failed")
            return verify_result

        rospy.loginfo(f"[{source}] pick executed")
        return ActionResult.ok("pick executed successfully")
    
    def _execute_place(self, target_pose: Tuple[float, float, float], source: str = "Agent") -> ActionResult:
        task_spec = {
            "action": ACTION_PLACE,
            "object": EMPTY_POSE,
            "target": target_pose
        }

        result = self._execute_action_sequence(self.task_planner.plan(task_spec))
        if not result.success:
            rospy.logerr(
                f"[{source}] place failed: code={result.error_code}, msg={result.message}"
            )
            return result

        rospy.loginfo(f"[{source}] place executed")
        return ActionResult.ok("place executed successfully")
    
    def _resolve_pose(self, role, x, y, z, class_id, source="Agent", infer_class_from_pose=False):
        """
        统一解析 object / target 的位姿来源

        Args:
            role (str): 例如 "pick object" / "place target"
            x, y, z (float): 显式坐标
            class_id (int): 类别ID
            source (str): 调用来源，用于日志，例如 "GUI" / "LLM"
            infer_class_from_pose (bool): 是否在显式坐标下尝试从视觉结果推断 class_id

        Returns:
            tuple:
                resolved_pose: (x, y, z) or None
                resolved_class_id: int or None
        """
        if x != 0.0 or y != 0.0 or z != 0.0:
            pose = (x, y, z)

            if infer_class_from_pose:
                inferred_class = self.object_detector.infer_class_id_from_pose(pose)
                if inferred_class is not None:
                    rospy.loginfo(
                        f"[{source}] {role} inferred class_id={inferred_class} from pose {pose}"
                    )
                    return pose, inferred_class

                rospy.loginfo(
                    f"[{source}] {role} received explicit pose but no nearby object detected"
                )
                return pose, None

            rospy.loginfo(
                f"[{source}] {role} received explicit pose: {pose}"
            )
            return pose, None

        if class_id != INVALID_CLASS_ID:
            pose = self.object_detector.get_best_position(class_id)
            if pose is None:
                rospy.logerr(f"[{source}] {role} failed: class_id={class_id} not detected")
                return None, None

            rospy.loginfo(f"[{source}] {role} resolved by class_id={class_id}, pose={pose}")
            return pose, class_id

        rospy.logerr(f"[{source}] {role} failed: missing class_id and explicit pose")
        return None, None
    
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
        except Exception as e:
            rospy.logerr(f"[GUI] gripper_down exception: {e}")

    def _gui_pick_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        rospy.loginfo(
            f"[GUI] pick received: x={x:.3f}, y={y:.3f}, z={z:.3f}, frame={msg.header.frame_id}"
        )

        try:
            obj_pose, resolved_class_id = self._resolve_pose(
                role="pick object",
                x=x,
                y=y,
                z=z,
                class_id=INVALID_CLASS_ID,
                source="GUI",
                infer_class_from_pose=True
            )
            if obj_pose is None:
                return

            result = self._execute_pick(
                obj_pose=obj_pose,
                class_id=resolved_class_id,
                source="GUI"
            )

            if not result.success:
                return

        except Exception as e:
            rospy.logerr(f"[GUI] pick failed: {e}")

    def _gui_place_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        rospy.loginfo(
            f"[GUI] place received: x={x:.3f}, y={y:.3f}, z={z:.3f}, frame={msg.header.frame_id}"
        )

        try:
            target_pose, _ = self._resolve_pose(
                role="place target",
                x=x,
                y=y,
                z=z,
                class_id=INVALID_CLASS_ID,
                source="GUI",
                infer_class_from_pose=False
            )
            if target_pose is None:
                return

            result = self._execute_place(
                target_pose=target_pose,
                source="GUI"
            )

            if not result.success:
                return

        except Exception as e:
            rospy.logerr(f"[GUI] place exception: {e}")

    def _verify_pick_result(self, original_pose, class_id=None, tolerance=0.05):
        if class_id is not None:
            still_exists = self.object_detector.exists_near_position(
                class_id,
                original_pose,
                tolerance
            )
        else:
            still_exists = self.object_detector.exists_any_near_position(
                original_pose,
                tolerance
            )

        if still_exists:
            rospy.logwarn("[Agent] pick check: object still at original position")
            return False

        rospy.logdebug("[Agent] pick check: object not observed at original position")
        return True