#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from gazebo_box_display import BoxSpawner
from gazebo_cylinder_display import CylinderSpawner
import time
import random
from my_kinematics import inverse_kinematics

time_sleep = 1.0
class ArmController:
    def __init__(self):
        """初始化控制器"""
        # rospy.init_node('arm_controller', anonymous=True)
        
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
        
        # Gazebo 方块/圆柱体生成工具
        self.box = BoxSpawner()
        self.cylinder = CylinderSpawner()
        
        # 当前关节状态
        self.current_joint_state = None
        rospy.Subscriber('/joint_states', JointState, self._joint_state_callback)
        
        rospy.loginfo('控制器已初始化')
        
        # 等待话题建立连接
        rospy.sleep(1.0)

    def _joint_state_callback(self, msg):
        """关节状态回调函数"""
        self.current_joint_state = msg

    def get_gripper_roll_yaw(self):
        """
        获取 gripper_roll_link 在世界坐标系中的 yaw 角（弧度）
        通过正向运动学计算：yaw = rotation1 + rotation2 + gripper_roll
        
        返回:
            float: yaw 角度值（弧度），如果未获取到则返回 None
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
            world_yaw = rotation1 + rotation2 + gripper_roll
            
            return world_yaw
        except ValueError:
            rospy.logwarn("未找到所需关节")
            return None
        except IndexError:
            rospy.logwarn("关节状态数据不完整")
            return None

    def align_gripper_roll(self):
        """
        对齐夹爪朝向：获取当前 yaw 角，然后旋转夹爪使其回到初始朝向（相对于世界坐标系为 0）
        """
        yaw = self.get_gripper_roll_yaw()
        if yaw is not None:
            rospy.loginfo("当前 gripper_roll yaw 角: %.3f rad (%.1f 度)" % (yaw, np.degrees(yaw)))
            self.gripper_roll_pub.publish(Float64(-yaw))
            rospy.loginfo("旋转夹爪以对齐初始朝向")
        else:
            rospy.loginfo("无法获取 gripper_roll yaw 角")

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

    def control_gripper(self, finger1, finger2, finger3, finger4, is_close, duration=1.0):
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
        
        if is_close:
            self.finger1_pub.publish(Float64(finger1))
            self.finger2_pub.publish(Float64(finger2))
            self.finger3_pub.publish(Float64(finger3))
            self.finger4_pub.publish(Float64(finger4))
        else:
            self.finger1_pub.publish(Float64(finger1))
            self.finger2_pub.publish(Float64(finger2))
            self.finger3_pub.publish(Float64(finger3))
            self.finger4_pub.publish(Float64(finger4))
        rospy.sleep(duration)

    def open_gripper(self, duration=1.0):
        """打开夹爪"""
        rospy.loginfo("打开夹爪...")
        self.control_gripper(-0.02, 0.02, 0.02, -0.02, False, duration)

    def close_gripper(self, duration=1.0):
        """关闭夹爪"""
        rospy.loginfo("关闭夹爪...")
        self.control_gripper(0.02, -0.02, -0.02, 0.02, True, duration)

    def world_init(self):
        """初始化世界: 生成方块等"""
        rospy.loginfo("=== 初始化世界 ===")

        # 打开夹爪
        self.open_gripper(duration=1.5)
        rospy.loginfo("初始化完成!\n")

    def pick_and_place(self, pick_pos, place_pos):
        pick_x, pick_y, pick_z = pick_pos
        place_x, place_y, place_z = place_pos

        origin_height = 0.5
        div = 0.187
        theta1_c, theta2_c, d3_c, reachable = inverse_kinematics(pick_x, pick_y, origin_height, elbow="down")
        if not reachable:
            rospy.logwarn("抓取位置不可达: (%.3f, %.3f, %.3f)" % (pick_x, pick_y, pick_z))
            return
        rospy.loginfo("前往抓取目标上方")
        self.move_arm_simple(theta1_c, theta2_c, d3_c, duration=3.0)

        # 在夹爪移动到物体上方后，对齐夹爪朝向
        self.align_gripper_roll()
        
        rospy.loginfo("下降夹爪")
        self.move_arm_simple(theta1_c, theta2_c, pick_z+div-origin_height, duration=3.0)
        rospy.loginfo("闭合夹爪")
        self.close_gripper(duration=1.0)
        rospy.loginfo("抬起")
        self.move_arm_simple(theta1_c, theta2_c, origin_height-pick_z+div, duration=3.0)

        rospy.loginfo("前往放置位置上方")
        theta1_c, theta2_c, d3_c, reachable = inverse_kinematics(place_x, place_y, origin_height, elbow="down")
        if not reachable:
            rospy.logwarn("放置位置不可达: (%.3f, %.3f, %.3f)" % (place_x, place_y, place_z))
            return
        self.move_arm_simple(theta1_c, theta2_c, d3_c, duration=3.0)
         
        rospy.loginfo("下降夹爪")
        self.move_arm_simple(theta1_c, theta2_c, place_z+div-origin_height, duration=3.0)
        rospy.loginfo("打开夹爪")
        self.open_gripper(duration=1.5)
         
        rospy.loginfo("抬起夹爪")
        self.move_arm_simple(theta1_c, theta2_c, origin_height-place_z+div, duration=3.0)
        rospy.loginfo("抓取放置任务完成")
    
    def arm_reset(self):
        """手臂复位到初始位置"""
        rospy.loginfo("手臂复位到初始位置")
        self.move_arm_simple(0.0, 0.0, 0.0, duration=3.0)
        
        # 直接将 gripper_roll 复位到 0
        rospy.loginfo("复位 gripper_roll 到初始角度")
        self.gripper_roll_pub.publish(Float64(0.0))
        rospy.sleep(1.0)
        
        self.open_gripper(duration=1.5)
        rospy.loginfo("手臂复位完成\n")

    def display_test_box(self, box_pos, box_size=(0.032, 0.032, 0.032), box_color=(0.2, 0.6, 0.9, 1.0), box_mass=0.01, box_name='test_box'):

        box_x, box_y, box_z = box_pos
        sx, sy, sz = box_size
        r, g, b, a = box_color
        
        # 检查方块位置是否在机械臂夹取范围内
        origin_height = 0.5
        theta1_c, theta2_c, d3_c, reachable = inverse_kinematics(box_x, box_y, origin_height, elbow="down")
        if not reachable:
            rospy.logwarn("方块位置不在机械臂夹取范围内: (%.3f, %.3f, %.3f)" % (box_x, box_y, box_z))
            rospy.logwarn("无法生成方块 '%s'" % box_name)
            return False
        
        rospy.loginfo("生成测试方块...")
        success = self.box.spawn_box(
            name=box_name,
            x=box_x, y=box_y, z=box_z,
            yaw=0.0,
            sx=sx, sy=sy, sz=sz,
            color_rgba=(r, g, b, a),
            mass=box_mass,
            reference_frame='world'
        )
        
        if success:
            rospy.loginfo("方块 '%s' 生成成功，位置在夹取范围内" % box_name)
        return success

    def display_test_cylinder(self, cyl_pos, radius=0.015, length=0.032,
                              cyl_color=(0.2, 0.8, 0.2, 1.0),
                              cyl_mass=0.01,
                              cyl_name='test_cylinder'):
        """
        在机械臂可达范围内生成一个测试圆柱体

        参数:
            cyl_pos: (x, y, z) 圆柱体中心位置
            radius: 半径 (m)
            length: 高度/长度 (m)
            cyl_color: 颜色 (r, g, b, a)
            cyl_mass: 质量 (kg)
            cyl_name: 模型名称
        """
        cyl_x, cyl_y, cyl_z = cyl_pos
        r, g, b, a = cyl_color

        # 检查圆柱体位置是否在机械臂夹取范围内（与方块相同策略）
        origin_height = 0.5
        theta1_c, theta2_c, d3_c, reachable = inverse_kinematics(cyl_x, cyl_y, origin_height, elbow="down")
        if not reachable:
            rospy.logwarn("圆柱体位置不在机械臂夹取范围内: (%.3f, %.3f, %.3f)" % (cyl_x, cyl_y, cyl_z))
            rospy.logwarn("无法生成圆柱体 '%s'" % cyl_name)
            return False

        rospy.loginfo("生成测试圆柱体...")
        success = self.cylinder.spawn_cylinder(
            name=cyl_name,
            x=cyl_x, y=cyl_y, z=cyl_z,
            yaw=0.0,
            radius=radius,
            length=length,
            color_rgba=(r, g, b, a),
            mass=cyl_mass,
            reference_frame='world'
        )

        if success:
            rospy.loginfo("圆柱体 '%s' 生成成功，位置在夹取范围内" % cyl_name)
        return success

def main():
    try:
        controller = ArmController()
        
        # 初始化世界
        controller.world_init()

        # 进行5轮随机测试
        num_tests = 5
        rospy.loginfo("=== 开始 %d 轮随机测试 ===" % num_tests)
        
        for i in range(num_tests):
            rospy.loginfo("\n=== 第 %d/%d 轮测试 ===" % (i+1, num_tests))
            
            # 随机生成方块位置 (x: 0.3-1.0, y: -0.8-0.8, z: 0.05)
            box_x = random.uniform(0.3, 1.0)
            box_y = random.uniform(-0.8, 0.8)
            box_z = 0.05
            
            # 随机生成放置位置 (x: 0.3-1.0, y: -0.8-0.8, z: 0.05)
            place_x = random.uniform(0.3, 1.0)
            place_y = random.uniform(-0.8, 0.8)
            place_z = box_z
            
            rospy.loginfo("方块位置: (%.3f, %.3f, %.3f)" % (box_x, box_y, box_z))
            rospy.loginfo("放置位置: (%.3f, %.3f, %.3f)" % (place_x, place_y, place_z))
            
            # 生成方块名称
            box_name = 'test_box_%d' % (i+1)
            
            # 显示方块
            success = controller.display_test_box(
                box_pos=(box_x, box_y, box_z),
                box_name=box_name
            )
            
            if not success:
                rospy.logwarn("第 %d 轮测试失败: 方块生成失败，跳过本轮" % (i+1))
                continue
            
            # 执行抓取和放置
            controller.pick_and_place(
                pick_pos=(box_x, box_y, box_z),
                place_pos=(place_x, place_y, place_z)
            )
            
            # 手臂复位
            controller.arm_reset()
            
            # 删除方块
            controller.box.delete_entity(box_name)
            rospy.loginfo("第 %d 轮测试完成\n" % (i+1))
            
            # 等待一下再进行下一轮
            rospy.sleep(1.0)
        
        rospy.loginfo("=== 所有测试完成，按 Ctrl+C 退出 ===")
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("程序被用户中断")


if __name__ == '__main__':
    main()
