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
            arm_controller.arm_reset()
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

        # 固定放置位置
        place_pos = (0.0, -1.8, 0.05)

        # 随机生成 5 轮方块位置：只生成方块，不生成圆柱
        # 说明：这里生成的方块位置仅用于 Gazebo 展示；实际抓取坐标以视觉话题为准
        box_z = 0.05
        min_x, max_x = 0.3, 1.0
        min_y, max_y = -0.8, 0.8

        rospy.loginfo("=== 将随机生成 5 轮方块，并等待视觉输出坐标后抓取 ===")
        rospy.loginfo(f"=== 固定放置位置: ({place_pos[0]:.3f}, {place_pos[1]:.3f}, {place_pos[2]:.3f}) ===")

        for i in range(5):
            if rospy.is_shutdown():
                break

            # 每轮生成一个新方块（不同名称）
            box_name = f"test_box_{i}"

            # 尝试随机生成可达位置（display_test_box 内部也会再检查可达性）
            max_attempts = 20
            success = False
            for attempt in range(max_attempts):
                box_x = float(np.random.uniform(min_x, max_x))
                box_y = float(np.random.uniform(min_y, max_y))
                success = arm_controller.display_test_box(
                    box_pos=(box_x, box_y, box_z),
                    box_name=box_name
                )
                if success:
                    break

            if not success:
                rospy.logerr(f"[Round {i+1}/5] 生成方块失败（尝试 {max_attempts} 次仍不可达），跳过该轮")
                continue

            rospy.loginfo(f"[Round {i+1}/5] 方块已生成: name={box_name}, pos=({box_x:.3f}, {box_y:.3f}, {box_z:.3f})")

            # 等待视觉模型推理出的坐标（/detected_objects）
            # 用时间戳防止拿到上一轮的旧坐标：先清空，再等待新值
            object_detector.detected_object_pos = None
            wait_timeout_s = 15.0
            start_t = time.time()
            rate = rospy.Rate(10)  # 10 Hz 更快响应

            rospy.loginfo(f"[Round {i+1}/5] 等待视觉坐标（超时 {wait_timeout_s:.1f}s）...")
            while not rospy.is_shutdown() and object_detector.detected_object_pos is None:
                if time.time() - start_t > wait_timeout_s:
                    rospy.logwarn(f"[Round {i+1}/5] 等待视觉坐标超时，跳过该轮")
                    arm_controller.box.delete_entity(box_name)
                    break
                rate.sleep()

            if rospy.is_shutdown():
                break

            if object_detector.detected_object_pos is None:
                continue

            # 执行抓取放置
            object_detector.is_processing = True
            try:
                ok = object_detector.process_detected_object(
                    arm_controller=arm_controller,
                    place_pos=place_pos
                )
                if ok:
                    rospy.loginfo(f"[Round {i+1}/5] 抓取放置完成，继续下一轮")
                else:
                    rospy.logwarn(f"[Round {i+1}/5] 抓取放置失败，继续下一轮")
            finally:
                object_detector.is_processing = False
                arm_controller.box.delete_entity(box_name)
        rospy.loginfo("=== 5 轮完成（或提前结束），进入 spin 保持节点运行 ===")
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("程序被用户中断")


if __name__ == '__main__':
    main()
