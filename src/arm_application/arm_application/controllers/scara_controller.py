# src/arm_application/controllers/scara_controller.py
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from controllers.abstract_controller import AbstractController
from utils.my_kinematics import inverse_kinematics 
import numpy as np
from std_msgs.msg import Float32
from arm_vision.msg import GripperObjectInfo
from utils.action_result import ActionResult

from config import (
    CONTROLLER_INIT_WAIT,
    JOINT_MOVE_DURATION,
    MOVE_TO_DURATION,
    OPEN_GRIPPER_DURATION,
    CLOSE_GRIPPER_DURATION,
    ALIGN_GRIPPER_DURATION,
    RESET_DURATION,
    GRIPPER_DOWN_DURATION,
    GRIPPER_OPEN_POS,
    GRIPPER_CLOSE_POS,
    RESET_THETA1,
    RESET_THETA2,
    RESET_D3,
    RESET_GRIPPER_ROLL,
    GRIPPER_DOWN_SAFE_OFFSET,
)

class ScaraController(AbstractController):
    def __init__(self):
        # 创建各关节位置控制发布器
        self.rotation1_pub = rospy.Publisher(
            '/rotation1_position_controller/command', 
            Float64, 
            queue_size=10
        )
        self.rotation2_pub = rospy.Publisher(
            '/rotation2_position_controller/command', 
            Float64, 
            queue_size=10
        )
        self.gripper_pub = rospy.Publisher(
            '/gripper_position_controller/command', 
            Float64, 
            queue_size=10
        )
        self.gripper_roll_pub = rospy.Publisher(
            '/gripper_roll_position_controller/command', 
            Float64, 
            queue_size=10
        )
        # 创建夹爪四指控制发布器
        self.finger1_pub = rospy.Publisher(
            '/finger1_position_controller/command', 
            Float64, 
            queue_size=10
        )
        self.finger2_pub = rospy.Publisher(
            '/finger2_position_controller/command', 
            Float64, 
            queue_size=10
        )
        self.finger3_pub = rospy.Publisher(
            '/finger3_position_controller/command', 
            Float64, 
            queue_size=10
        )
        self.finger4_pub = rospy.Publisher(
            '/finger4_position_controller/command', 
            Float64, 
            queue_size=10
        )

        # 当前关节状态
        self.current_joint_state = None
        rospy.Subscriber('/joint_states', JointState, self._joint_state_callback)
        self.object_height = 0.0
        self.object_yaw = 0.0
        self.object_has_yaw = False

        rospy.Subscriber('/gripper_object_info', GripperObjectInfo, self._object_info_callback)
        # 等待话题建立连接
        rospy.sleep(CONTROLLER_INIT_WAIT)

    def _joint_state_callback(self, msg):
        """关节状态回调函数"""
        self.current_joint_state = msg

    def _move_joints(self, theta1, theta2, d3, duration=JOINT_MOVE_DURATION):
        """内部方法：直接控制关节"""
        self.rotation1_pub.publish(Float64(theta1))
        self.rotation2_pub.publish(Float64(theta2))
        self.gripper_pub.publish(Float64(d3))
        rospy.sleep(duration)

    def move_to(self, x: float, y: float, z: float, duration: float = MOVE_TO_DURATION) -> ActionResult:
        """
        实现原子动作:移动到世界坐标
        """
        try:
            theta1, theta2, d3, reachable = inverse_kinematics(x, y, z, elbow="down")
            if not reachable:
                rospy.logwarn(f"目标点({x:.3f},{y:.3f},{z:.3f})不可达")
                return ActionResult.fail(
                    "IK_FAILED",
                    f"target unreachable: x={x:.3f}, y={y:.3f}, z={z:.3f}",
                    retryable=False
                )

            rospy.loginfo(f"move to {(x, y, z)}")
            self._move_joints(theta1, theta2, d3, duration)

            return ActionResult.ok(
                message=f"move_to success: x={x:.3f}, y={y:.3f}, z={z:.3f}",
                data={
                    "x": x,
                    "y": y,
                    "z": z,
                    "theta1": theta1,
                    "theta2": theta2,
                    "d3": d3,
                    "duration": duration
                }
            )

        except Exception as e:
            rospy.logerr(f"move_to exception: {e}")
            return ActionResult.fail(
                "MOVE_TO_EXCEPTION",
                f"move_to exception: {e}",
                retryable=True
            )

    def open_gripper(self, duration: float = OPEN_GRIPPER_DURATION) -> ActionResult:
        try:
            rospy.loginfo("open gripper")
            f1, f2, f3, f4 = GRIPPER_OPEN_POS
            self.finger1_pub.publish(Float64(f1))
            self.finger2_pub.publish(Float64(f2))
            self.finger3_pub.publish(Float64(f3))
            self.finger4_pub.publish(Float64(f4))
            rospy.sleep(duration)

            return ActionResult.ok(
                message="open_gripper success",
                data={
                    "duration": duration,
                    "finger_positions": [f1, f2, f3, f4]
                }
            )

        except Exception as e:
            rospy.logerr(f"open_gripper exception: {e}")
            return ActionResult.fail(
                "OPEN_GRIPPER_EXCEPTION",
                f"open_gripper exception: {e}",
                retryable=True
            )

    def close_gripper(self, duration: float = CLOSE_GRIPPER_DURATION) -> ActionResult:
        try:
            rospy.loginfo("close gripper")
            f1, f2, f3, f4 = GRIPPER_CLOSE_POS
            self.finger1_pub.publish(Float64(f1))
            self.finger2_pub.publish(Float64(f2))
            self.finger3_pub.publish(Float64(f3))
            self.finger4_pub.publish(Float64(f4))
            rospy.sleep(duration)

            return ActionResult.ok(
                message="close_gripper success",
                data={
                    "duration": duration,
                    "finger_positions": [f1, f2, f3, f4]
                }
            )

        except Exception as e:
            rospy.logerr(f"close_gripper exception: {e}")
            return ActionResult.fail(
                "CLOSE_GRIPPER_EXCEPTION",
                f"close_gripper exception: {e}",
                retryable=True
            )

    def reset(self, duration: float = RESET_DURATION) -> ActionResult:
        try:
            self._move_joints(RESET_THETA1, RESET_THETA2, RESET_D3, duration)
            self.gripper_roll_pub.publish(Float64(RESET_GRIPPER_ROLL))

            open_result = self.open_gripper()
            if not open_result.success:
                return ActionResult.fail(
                    "RESET_OPEN_GRIPPER_FAILED",
                    f"reset failed when opening gripper: {open_result.message}",
                    retryable=open_result.retryable
                )

            return ActionResult.ok(
                message="reset success",
                data={
                    "theta1": RESET_THETA1,
                    "theta2": RESET_THETA2,
                    "d3": RESET_D3,
                    "gripper_roll": RESET_GRIPPER_ROLL,
                    "duration": duration
                }
            )

        except Exception as e:
            rospy.logerr(f"reset exception: {e}")
            return ActionResult.fail(
                "RESET_EXCEPTION",
                f"reset exception: {e}",
                retryable=True
            )

    # 这个预计会改掉，后续基于传统视觉计算每个物体的yaw
    def _get_gripper_roll_yaw(self):
        """
        获取 gripper_roll_link 在世界坐标系中的 yaw 角（弧度）
        通过正向运动学计算:yaw = rotation1 + rotation2 + gripper_roll
        注意,这里的关节角、夹爪角都是相对于自身joint的转角,不是世界坐标系的转角
        返回:
            float: yaw 角度值（弧度）,如果未获取到则返回 None
        """
        if self.current_joint_state is None:
            rospy.logwarn("尚未接收到关节状态信息")
            return None
        
        try:
            # 获取各关节角度
            rotation1_idx = self.current_joint_state.name.index('rotation1')
            rotation2_idx = self.current_joint_state.name.index('rotation2')
            gripper_roll_idx = self.current_joint_state.name.index('gripper_roll')
            
            rotation1 = self.current_joint_state.position[rotation1_idx]
            rotation2 = self.current_joint_state.position[rotation2_idx]
            gripper_roll = self.current_joint_state.position[gripper_roll_idx]
            
            # 计算 gripper_roll_link 的世界 yaw 角
            # world_yaw = rotation1 + rotation2 + gripper_roll
            world_yaw = rotation1 + rotation2
            
            return world_yaw
        except ValueError:
            rospy.logwarn("未找到所需关节")
            return None
        except IndexError:
            rospy.logwarn("关节状态数据不完整")
            return None

    def _get_current_gripper_roll_joint(self):
        """
        获取当前 gripper_roll 关节角（弧度）
        """
        if self.current_joint_state is None:
            rospy.logwarn("尚未接收到关节状态信息")
            return None

        try:
            gripper_roll_idx = self.current_joint_state.name.index('gripper_roll')
            return self.current_joint_state.position[gripper_roll_idx]
        except ValueError:
            rospy.logwarn("未找到 gripper_roll 关节")
            return None
        except IndexError:
            rospy.logwarn("gripper_roll 关节状态数据不完整")
            return None
    
    def _normalize_align_yaw(self, yaw: float) -> float:
        """
        将视觉对齐角归一化到 [-pi/4, pi/4]，消除正方形/矩形的 90° 等价歧义
        """
        half_pi = np.pi / 2.0
        quarter_pi = np.pi / 4.0

        # 先拉回到 [-pi, pi]
        yaw = np.arctan2(np.sin(yaw), np.cos(yaw))

        # 再按 90° 周期折叠到 [-45°, 45°]
        while yaw > quarter_pi:
            yaw -= half_pi
        while yaw < -quarter_pi:
            yaw += half_pi

        return yaw

    def align_gripper_roll(self, duration: float = ALIGN_GRIPPER_DURATION) -> ActionResult:
        """
        根据夹爪相机估计得到的物体对齐角，旋转夹爪进行对齐
        """
        try:
            current_yaw = self._get_gripper_roll_yaw()
            current_roll_joint = self._get_current_gripper_roll_joint()

            if current_yaw is None:
                rospy.loginfo("无法获取当前夹爪世界 yaw")
                return ActionResult.fail(
                    "CURRENT_YAW_UNAVAILABLE",
                    "current gripper world yaw unavailable",
                    retryable=True
                )

            if current_roll_joint is None:
                rospy.loginfo("无法获取当前 gripper_roll 关节角")
                return ActionResult.fail(
                    "CURRENT_ROLL_JOINT_UNAVAILABLE",
                    "current gripper_roll joint unavailable",
                    retryable=True
                )

            if not self.object_has_yaw:
                rospy.loginfo("当前目标没有有效 yaw，跳过夹爪对齐")
                return ActionResult.fail(
                    "OBJECT_YAW_UNAVAILABLE",
                    "object yaw unavailable",
                    retryable=False
                )

            raw_target_yaw = self.object_yaw
            target_yaw = self._normalize_align_yaw(raw_target_yaw)

            # 这是“需要补偿的量”，不是绝对关节目标
            delta_roll = -target_yaw

            # position_controller 需要的是绝对目标角
            new_roll_joint = current_roll_joint + delta_roll

            rospy.loginfo(
                f"current_yaw={current_yaw:.3f} rad ({np.degrees(current_yaw):.1f} deg), "
                f"current_roll_joint={current_roll_joint:.3f} rad ({np.degrees(current_roll_joint):.1f} deg), "
                f"raw_target_yaw={raw_target_yaw:.3f} rad ({np.degrees(raw_target_yaw):.1f} deg), "
                f"normalized_target_yaw={target_yaw:.3f} rad ({np.degrees(target_yaw):.1f} deg), "
                f"delta_roll={delta_roll:.3f} rad ({np.degrees(delta_roll):.1f} deg), "
                f"new_roll_joint={new_roll_joint:.3f} rad ({np.degrees(new_roll_joint):.1f} deg)"
            )

            self.gripper_roll_pub.publish(Float64(new_roll_joint))
            rospy.loginfo("旋转夹爪以对齐物体方向")
            rospy.sleep(duration)

            return ActionResult.ok(
                message="align_gripper_roll success",
                data={
                    "current_yaw": current_yaw,
                    "current_roll_joint": current_roll_joint,
                    "raw_target_yaw": raw_target_yaw,
                    "normalized_target_yaw": target_yaw,
                    "delta_roll": delta_roll,
                    "new_roll_joint": new_roll_joint,
                    "duration": duration
                }
            )

        except Exception as e:
            rospy.logerr(f"align_gripper_roll exception: {e}")
            return ActionResult.fail(
                "ALIGN_GRIPPER_ROLL_EXCEPTION",
                f"align_gripper_roll exception: {e}",
                retryable=True
            )

    # 基于深度相机数据，自适应下降
    def _object_info_callback(self, msg):
        self.object_height = msg.height
        self.object_yaw = msg.yaw
        self.object_has_yaw = msg.has_yaw

    def gripper_down(self, x: float, y: float, duration: float = GRIPPER_DOWN_DURATION) -> ActionResult:
        """
        夹爪自适应下降
        """
        try:
            rospy.loginfo(f"height: {self.object_height}")
            above = self.object_height + GRIPPER_DOWN_SAFE_OFFSET
            rospy.loginfo(f"above: {above}")

            move_result = self.move_to(x, y, above)
            if not move_result.success:
                return ActionResult.fail(
                    "GRIPPER_DOWN_MOVE_FAILED",
                    f"gripper_down failed when moving down: {move_result.message}",
                    retryable=move_result.retryable
                )

            return ActionResult.ok(
                message="gripper_down success",
                data={
                    "x": x,
                    "y": y,
                    "object_height": self.object_height,
                    "target_z": above,
                    "duration": duration
                }
            )

        except Exception as e:
            rospy.logerr(f"gripper_down exception: {e}")
            return ActionResult.fail(
                "GRIPPER_DOWN_EXCEPTION",
                f"gripper_down exception: {e}",
                retryable=True
            )

