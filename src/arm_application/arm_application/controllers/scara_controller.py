# src/arm_application/controllers/scara_controller.py
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from controllers.abstract_controller import AbstractController
from utils.my_kinematics import inverse_kinematics 
import numpy as np

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
        # 等待话题建立连接
        rospy.sleep(1.0)

    def _joint_state_callback(self, msg):
        """关节状态回调函数"""
        self.current_joint_state = msg

    def _move_joints(self, theta1, theta2, d3, duration=1.5):
        """内部方法：直接控制关节"""
        self.rotation1_pub.publish(Float64(theta1))
        self.rotation2_pub.publish(Float64(theta2))
        self.gripper_pub.publish(Float64(d3))
        rospy.sleep(duration)

    def move_to(self, x: float, y: float, z: float, duration: float = 3.0) -> bool:
        """
        实现原子动作:移动到世界坐标,但目前并未考虑从任意坐标的移动
        """  
        
        theta1, theta2, d3, reachable = inverse_kinematics(x, y, z, elbow="down")
        if not reachable:
            return False
        rospy.loginfo(f'move to {x,y,z}')
        self._move_joints(theta1, theta2, d3, duration)
    
        return True

    def open_gripper(self, duration: float = 1.0) -> None:
        rospy.loginfo("open gripper")
        self.finger1_pub.publish(Float64(-0.02))
        self.finger2_pub.publish(Float64(0.02))
        self.finger3_pub.publish(Float64(0.02))
        self.finger4_pub.publish(Float64(-0.02))
        rospy.sleep(duration)

    def close_gripper(self, duration: float = 1.0) -> None:
        rospy.loginfo("close gripper")
        self.finger1_pub.publish(Float64(0.02))
        self.finger2_pub.publish(Float64(-0.02))
        self.finger3_pub.publish(Float64(-0.02))
        self.finger4_pub.publish(Float64(0.02))
        rospy.sleep(duration)

    def reset(self, duration: float = 3.0) -> None:
        self._move_joints(0.0, 0.0, 0.0, duration)
        self.gripper_roll_pub.publish(Float64(0.0))
        self.open_gripper()

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

    def align_gripper_roll(self) -> None:
        """
        对齐夹爪朝向：获取当前 yaw 角,然后旋转夹爪使其回到初始朝向(相对于世界坐标系为 0)

        """
        yaw = self._get_gripper_roll_yaw()
        rospy.loginfo(f"当前 gripper_roll yaw 角: {yaw:.3f} rad ({np.degrees(yaw):.1f} 度)")
        if yaw is not None:
            self.gripper_roll_pub.publish(Float64(-yaw))
            rospy.loginfo("旋转夹爪以对齐初始朝向")
        else:
            rospy.loginfo("无法获取 gripper_roll yaw 角")