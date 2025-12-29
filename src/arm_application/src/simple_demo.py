#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最简单的 SCARA 机器人控制 Demo (ROS1 Version)
目的: 学习如何通过 ROS1 话题控制关节位置
"""

import rospy
import numpy as np
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from gazebo_box_display import BoxSpawner
import time


class SimpleController:
    def __init__(self):
        """初始化控制器"""
        rospy.init_node('simple_controller', anonymous=True)
        
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
        
        # Gazebo 方块生成工具
        self.box = BoxSpawner()
        
        # 当前关节状态
        self.current_joint_state = None
        rospy.Subscriber('/joint_states', JointState, self._joint_state_callback)
        
        rospy.loginfo('控制器已初始化')
        
        # 等待话题建立连接
        rospy.sleep(1.0)

    def _joint_state_callback(self, msg):
        """关节状态回调函数"""
        self.current_joint_state = msg

    def move_arm_simple(self, pos1, pos2, pos3, duration=3.0):
        """
        简单的手臂运动函数
        
        参数说明:
            pos1: rotation1 关节位置 (单位: 弧度 rad)
            pos2: rotation2 关节位置 (单位: 弧度 rad)
            pos3: gripper_joint 关节位置 (单位: 米 m)
            duration: 运动时间 (秒)
        """
        rospy.loginfo("发送命令: pos1=%.3f, pos2=%.3f, pos3=%.3f" % (pos1, pos2, pos3))
        
        # 发布目标位置
        self.rotation1_pub.publish(Float64(pos1))
        self.rotation2_pub.publish(Float64(pos2))
        self.gripper_pub.publish(Float64(pos3))
        
        # 等待运动完成
        rospy.sleep(duration)

    def control_gripper(self, finger1, finger2, finger3, finger4, duration=1.0):
        """
        控制夹爪
        
        参数说明:
            finger1: finger1_joint 位置 (0.0 ~ 0.02)
            finger2: finger2_joint 位置 (-0.02 ~ 0.0)
            finger3: finger3_joint 位置 (-0.02 ~ 0.0)
            finger4: finger4_joint 位置 (0.0 ~ 0.02)
            duration: 运动时间
        """
        rospy.loginfo("发送夹爪命令: [%.3f, %.3f, %.3f, %.3f]" 
                     % (finger1, finger2, finger3, finger4))
        
        self.finger1_pub.publish(Float64(finger1))
        self.finger2_pub.publish(Float64(finger2))
        self.finger3_pub.publish(Float64(finger3))
        self.finger4_pub.publish(Float64(finger4))
        
        rospy.sleep(duration)

    def open_gripper(self, duration=1.0):
        """打开夹爪"""
        rospy.loginfo("打开夹爪...")
        self.control_gripper(-0.02, 0.02, 0.02, -0.02, duration)

    def close_gripper(self, duration=1.0):
        """关闭夹爪"""
        rospy.loginfo("关闭夹爪...")
        self.control_gripper(0.02, -0.02, -0.02, 0.02, duration)

    def world_init(self):
        """初始化世界: 生成方块等"""
        rospy.loginfo("=== 初始化世界 ===")
        
        # 移动到初始位置
        rospy.loginfo("移动到初始位置...")
        self.move_arm_simple(np.pi/2, 0.0, 0.0, duration=3.0)
        rospy.sleep(2.0)
        rospy.loginfo("完成!\n")
        
        # 生成方块
        rospy.loginfo("生成测试方块...")
        success = self.box.spawn_box(
            name='test_box',
            x=1.8, y=0.0, z=0.05,
            yaw=0.0,
            sx=0.032, sy=0.032, sz=0.032,
            color_rgba=(0.2, 0.6, 0.9, 1.0),
            mass=0.01,
            reference_frame='world'
        )
        
        if success:
            rospy.loginfo("方块生成成功!")
        else:
            rospy.logwarn("方块生成可能失败")
        
        rospy.sleep(1.0)
        
        # 打开夹爪
        self.open_gripper(duration=2.0)
        rospy.sleep(1.0)
        rospy.loginfo("初始化完成!\n")


def main():
    try:
        controller = SimpleController()
        
        # 初始化世界
        controller.world_init()
        
        # Demo 1: 移动手臂到抓取位置
        rospy.loginfo("=== Demo 1: 移动手臂到抓取位置 (下降到方块高度) ===")
        controller.move_arm_simple(np.pi/2, 0.0, -0.255, duration=3.0)
        
        rospy.loginfo("到达抓取位置，等待稳定...")
        rospy.sleep(2.0)
        rospy.loginfo("位置稳定!\n")
        
        # Demo 2: 关闭夹爪夹取方块
        rospy.loginfo("=== Demo 2: 准备夹取方块 ===")
        controller.close_gripper(duration=3.0)
        
        rospy.loginfo("等待物理引擎稳定接触和grasp_fix插件附着...")
        rospy.sleep(3.0)
        rospy.loginfo("夹取完成!\n")
        
        # Demo 3: 抬起方块
        rospy.loginfo("=== Demo 3: 抬起方块 ===")
        controller.move_arm_simple(np.pi/2, 0.0, -0.15, duration=4.0)
        rospy.loginfo("抬起完成！\n")
        
        rospy.loginfo("=== 演示完成，按 Ctrl+C 退出 ===")
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("程序被用户中断")


if __name__ == '__main__':
    main()
