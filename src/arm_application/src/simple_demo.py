#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最简单的 SCARA 机器人控制 Demo (ROS1 Version)
目的: 学习如何通过 ROS1 话题控制关节位置
"""

import sys
import os

# 添加源代码目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import rospy
import numpy as np
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from gazebo_box_display import BoxSpawner
from gazebo_cylinder_display import CylinderSpawner
import time
from my_kinematics import inverse_kinematics
from my_scara_action import ArmController

box_x = 1.0
box_y = 1.0
box_z = 0.05

class ObjectDetector:
    """
    物体检测订阅器
    只负责订阅检测话题并保存检测结果，不发布任何控制命令
    实际的抓取动作由 ArmController 执行
    """
    def __init__(self):
        """初始化检测器"""
        # 订阅检测到的物体话题
        rospy.Subscriber('/detected_objects', PoseStamped, self._detect_callback)
        
        # 检测到的物体坐标（世界坐标系）
        self.detected_object_pos = None
        self.is_processing = False  # 标记是否正在处理检测结果
        
        rospy.loginfo('[ObjectDetector] 检测器已初始化，等待检测物体...')
        
        # 等待话题建立连接
        rospy.sleep(1.0)

    def _detect_callback(self, msg):
        """检测到的物体回调函数 - 直接提取坐标"""
        # 如果正在处理，跳过新的检测
        if self.is_processing:
            return
        
        # 从 PoseStamped 消息中直接提取世界坐标
        pos_x = msg.pose.position.x
        pos_y = msg.pose.position.y
        pos_z = msg.pose.position.z
        
        rospy.loginfo(f"[检测回调] 检测到物体位置: x={pos_x:.3f}, y={pos_y:.3f}, z={pos_z:.3f}")
        
        # 保存检测到的物体位置
        self.detected_object_pos = (pos_x, pos_y, pos_z)

    def process_detected_object(self, arm_controller, place_pos=None):
        """
        处理检测到的物体：执行抓取和放置
        
        参数:
            arm_controller: ArmController 实例，用于执行抓取动作
            place_pos: 放置位置 (x, y, z)，如果为None则使用默认位置
        """
        if self.detected_object_pos is None:
            rospy.logwarn("[处理检测] 没有检测到物体")
            return False
        
        pick_pos = self.detected_object_pos
        
        # 如果没有指定放置位置，使用默认位置
        if place_pos is None:
            place_pos = (0.8, -0.65, pick_pos[2] + 0.032)  # 默认放置位置
        
        rospy.loginfo(f"[处理检测] 开始执行抓取任务")
        rospy.loginfo(f"[处理检测] 抓取位置: ({pick_pos[0]:.3f}, {pick_pos[1]:.3f}, {pick_pos[2]:.3f})")
        rospy.loginfo(f"[处理检测] 放置位置: ({place_pos[0]:.3f}, {place_pos[1]:.3f}, {place_pos[2]:.3f})")
        
        try:
            # 调用 pick_and_place 执行抓取和放置
            arm_controller.pick_and_place(pick_pos, place_pos)
            rospy.loginfo("[处理检测] 抓取任务完成")
            # 清空检测结果，避免重复执行
            self.detected_object_pos = None
            return True
        except Exception as e:
            rospy.logerr(f"[处理检测] 抓取任务失败: {e}")
            return False

    
def main():
    try:
        # 创建 ArmController 用于执行抓取动作（它会初始化 ROS 节点）
        arm_controller = ArmController()
        
        # 创建 ObjectDetector 用于订阅检测话题（只订阅，不发布控制命令）
        object_detector = ObjectDetector()
        
        # 初始化世界（生成测试方块等）
        arm_controller.world_init()

        # 生成一个测试圆柱体，方便调试视觉与抓取
        cyl_name = 'test_cylinder'
        cyl_x = 0.8
        cyl_y = -0.65
        cyl_z = 0.05

        success = arm_controller.display_test_cylinder(
            cyl_pos=(cyl_x, cyl_y, cyl_z),
            cyl_name=cyl_name
        )
        
        if not success:
            rospy.logerr("圆柱体生成失败，结束程序")
            return

        # 这个坐标方块会与夹爪不平行，可用于测试
        box_x = 0.8
        box_y = 0.65
        box_z = 0.05
        
        # 随机生成放置位置 (x: 0.3-1.2, y: -0.8-0.8, z: 0.05)
        place_x = 0.0
        place_y = -0.65
        place_z = box_z + 0.032
        
        # 生成方块名称
        box_name = 'test_box'
        
        # 显示方块
        success = arm_controller.display_test_box(
            box_pos=(box_x, box_y, box_z),
            box_name=box_name
        )
        
        if not success:
            rospy.logerr("方块生成失败，结束程序")
            return
        
        rospy.loginfo("=== 等待检测物体 ===")
        rospy.loginfo("视觉节点会发布检测到的物体坐标到 /detected_objects 话题")
        rospy.loginfo("检测到物体后会自动执行抓取和放置任务")
        
        # 主循环：定期检查是否有检测到的物体需要处理
        rate = rospy.Rate(1)  # 1 Hz
        while not rospy.is_shutdown():
            # 如果检测到物体且不在处理中，则处理它
            if object_detector.detected_object_pos is not None and not object_detector.is_processing:
                object_detector.is_processing = True
                try:
                    # 处理检测到的物体
                    success = object_detector.process_detected_object(
                        arm_controller=arm_controller,
                        place_pos=(place_x, place_y, place_z)
                    )
                    if success:
                        rospy.loginfo("=== 抓取任务完成，继续等待下一个物体 ===")
                except Exception as e:
                    rospy.logerr(f"处理检测物体时出错: {e}")
                finally:
                    object_detector.is_processing = False
            
            rate.sleep()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("程序被用户中断")


if __name__ == '__main__':
    main()
