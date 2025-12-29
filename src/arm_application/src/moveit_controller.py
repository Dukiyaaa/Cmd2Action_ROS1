#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于MoveIt的机械臂控制 (改进版)
目的: 使用MoveIt自动规划轨迹，不需要手动计算关节角度
"""

import sys
import rospy
import moveit_commander
from moveit_commander import MoveGroupCommander, PlanningSceneInterface
from moveit_msgs.msg import CollisionObject
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Float64
from tf.transformations import quaternion_from_euler
import numpy as np
import time


class MoveItController:
    """基于MoveIt的机械臂控制器"""
    
    def __init__(self, group_name="arm"):
        """
        初始化MoveIt控制器
        
        参数:
            group_name: MoveIt中定义的规划组名称 (默认: "arm")
        """
        # 初始化ROS节点
        rospy.init_node('moveit_controller', anonymous=True)
        
        # 初始化MoveIt接口
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = PlanningSceneInterface()
        self.move_group = MoveGroupCommander(group_name)
        
        # 配置规划参数
        # 规划与执行参数
        self.move_group.set_planning_time(10)  # 提高规划时间，提高成功率
        self.move_group.set_num_planning_attempts(10)
        self.move_group.set_max_velocity_scaling_factor(0.5)  # 最大速度50%
        self.move_group.set_max_acceleration_scaling_factor(0.5)
        # 规划参考帧
        try:
            self.move_group.set_pose_reference_frame('world')
        except Exception:
            pass
        # 目标容差（更宽松，避免IK过于严格导致失败）
        try:
            self.move_group.set_goal_position_tolerance(0.005)
            self.move_group.set_goal_orientation_tolerance(0.05)
            self.move_group.set_goal_joint_tolerance(0.001)
        except Exception:
            pass
        
        # 获取末端执行器链接（通常是最后一个链接）
        self.eef_link = self.move_group.get_end_effector_link()
        rospy.loginfo(f"末端执行器链接: {self.eef_link}")
        
        # 夹爪发布器（如果有的话）
        self.gripper_pub = rospy.Publisher(
            '/gripper_position_controller/command', 
            Float64, 
            queue_size=10
        )
        self.finger1_pub = rospy.Publisher(
            '/finger1_position_controller/command', Float64, queue_size=10
        )
        self.finger2_pub = rospy.Publisher(
            '/finger2_position_controller/command', Float64, queue_size=10
        )
        self.finger3_pub = rospy.Publisher(
            '/finger3_position_controller/command', Float64, queue_size=10
        )
        self.finger4_pub = rospy.Publisher(
            '/finger4_position_controller/command', Float64, queue_size=10
        )
        
        rospy.sleep(0.5)
        rospy.loginfo("MoveIt控制器初始化成功!")
    
    def _extract_trajectory(self, plan):
        """
        兼容不同MoveIt版本的plan返回类型，提取RobotTrajectory。
        可能返回：
        - RobotTrajectory 对象（具有 joint_trajectory）
        - tuple: (success, RobotTrajectory, planning_time, error_code) 或其他变体
        """
        # 直接是轨迹对象
        if hasattr(plan, 'joint_trajectory'):
            return plan
        # 元组返回，尝试从元素中找轨迹
        if isinstance(plan, tuple):
            for item in plan:
                if hasattr(item, 'joint_trajectory'):
                    return item
        return None
    
    def move_to_pose(self, x, y, z, roll=0, pitch=0.0, yaw=0.0):
        """
        移动机械臂末端执行器到目标位置和姿态
        
        参数:
            x, y, z: 目标位置 (米)
            roll, pitch, yaw: 目标姿态 (弧度，欧拉角)
        
        返回:
            success: 是否成功
        """
        # 创建目标位姿
        target_pose = Pose()
        target_pose.position.x = x
        target_pose.position.y = y
        target_pose.position.z = z
        
        # 将欧拉角转换为四元数
        quat = quaternion_from_euler(roll, pitch, yaw)
        target_pose.orientation.x = quat[0]
        target_pose.orientation.y = quat[1]
        target_pose.orientation.z = quat[2]
        target_pose.orientation.w = quat[3]
        
        rospy.loginfo(f"目标位置: x={x:.3f}, y={y:.3f}, z={z:.3f}")
        
        # 首选：位姿目标
        self.move_group.set_pose_target(target_pose, end_effector_link=self.eef_link)
        rospy.loginfo("正在规划轨迹...")
        plan = self.move_group.plan()
        traj = self._extract_trajectory(plan)

        # 失败则退化为仅位置目标（放宽姿态约束）
        if not traj or len(traj.joint_trajectory.points) == 0:
            rospy.logwarn("位姿规划失败，改为仅位置目标重试...")
            try:
                self.move_group.clear_pose_targets()
                self.move_group.set_position_target([x, y, z], self.eef_link)
                plan = self.move_group.plan()
                traj = self._extract_trajectory(plan)
            except Exception:
                traj = None
            if not traj or len(traj.joint_trajectory.points) == 0:
                rospy.logwarn("轨迹规划失败!")
                self.move_group.clear_pose_targets()
                return False
        
        rospy.loginfo(f"规划成功! 轨迹点数: {len(traj.joint_trajectory.points)}")
        
        # 执行计划
        rospy.loginfo("正在执行轨迹...")
        success = self.move_group.execute(traj, wait=True)
        
        # 清除目标
        self.move_group.clear_pose_targets()
        
        # 等待运动完成
        rospy.sleep(1.0)
        
        if success:
            rospy.loginfo("✅ 运动完成!")
        else:
            rospy.logwarn("⚠️ 运动执行可能失败")
        
        return success
    
    def move_to_joint_state(self, joint_values):
        """
        移动机械臂到指定的关节角度
        
        参数:
            joint_values: 关节角度列表 (弧度)
        
        返回:
            success: 是否成功
        """
        rospy.loginfo(f"目标关节角度: {joint_values}")
        
        # 设置目标关节角度
        self.move_group.set_joint_value_target(joint_values)
        
        # 规划
        rospy.loginfo("正在规划轨迹...")
        plan = self.move_group.plan()
        traj = self._extract_trajectory(plan)
        
        if not traj or len(traj.joint_trajectory.points) == 0:
            rospy.logwarn("轨迹规划失败!")
            self.move_group.clear_pose_targets()
            return False
        
        rospy.loginfo(f"规划成功!")
        
        # 执行
        rospy.loginfo("正在执行轨迹...")
        success = self.move_group.execute(traj, wait=True)
        
        self.move_group.clear_pose_targets()
        rospy.sleep(1.0)
        
        if success:
            rospy.loginfo("✅ 运动完成!")
        else:
            rospy.logwarn("⚠️ 运动执行可能失败")
        
        return success
    
    def open_gripper(self, duration=1.0):
        """打开夹爪"""
        rospy.loginfo("打开夹爪...")
        self.finger1_pub.publish(Float64(-0.02))
        self.finger2_pub.publish(Float64(0.02))
        self.finger3_pub.publish(Float64(0.02))
        self.finger4_pub.publish(Float64(-0.02))
        rospy.sleep(duration)
        rospy.loginfo("夹爪已打开")
    
    def close_gripper(self, duration=1.0):
        """关闭夹爪"""
        rospy.loginfo("关闭夹爪...")
        self.finger1_pub.publish(Float64(0.02))
        self.finger2_pub.publish(Float64(-0.02))
        self.finger3_pub.publish(Float64(-0.02))
        self.finger4_pub.publish(Float64(0.02))
        rospy.sleep(duration)
        rospy.loginfo("夹爪已关闭")
    
    def pick_and_place(self, pick_x, pick_y, pick_z, place_x, place_y, place_z):
        """
        完整的拾取放置流程 (不需要手动实现每一步!)
        
        参数:
            pick_x, pick_y, pick_z: 拾取位置
            place_x, place_y, place_z: 放置位置
        """
        rospy.loginfo("=" * 50)
        rospy.loginfo("开始拾取放置流程")
        rospy.loginfo("=" * 50)
        
        # Step 1: 打开夹爪
        rospy.loginfo("\n[Step 1] 打开夹爪...")
        self.open_gripper(duration=2.0)
        
        # Step 2: 移动到拾取位置上方 (z高10cm)
        rospy.loginfo("\n[Step 2] 移动到拾取点上方...")
        success = self.move_to_pose(pick_x, pick_y, pick_z + 0.1)
        if not success:
            rospy.logwarn("无法到达拾取点上方!")
            return False
        
        # Step 3: 下降到拾取位置
        rospy.loginfo("\n[Step 3] 下降到拾取点...")
        success = self.move_to_pose(pick_x, pick_y, pick_z)
        if not success:
            rospy.logwarn("无法到达拾取点!")
            return False
        
        # Step 4: 关闭夹爪夹取
        rospy.loginfo("\n[Step 4] 关闭夹爪夹取物体...")
        self.close_gripper(duration=2.0)
        
        # Step 5: 上升到安全高度
        rospy.loginfo("\n[Step 5] 上升到安全高度...")
        success = self.move_to_pose(pick_x, pick_y, pick_z + 0.1)
        if not success:
            rospy.logwarn("无法上升!")
            return False
        
        # Step 6: 移动到放置位置上方
        rospy.loginfo("\n[Step 6] 移动到放置点上方...")
        success = self.move_to_pose(place_x, place_y, place_z + 0.1)
        if not success:
            rospy.logwarn("无法到达放置点!")
            return False
        
        # Step 7: 下降到放置位置
        rospy.loginfo("\n[Step 7] 下降到放置点...")
        success = self.move_to_pose(place_x, place_y, place_z)
        if not success:
            rospy.logwarn("无法到达放置点!")
            return False
        
        # Step 8: 打开夹爪释放物体
        rospy.loginfo("\n[Step 8] 打开夹爪释放物体...")
        self.open_gripper(duration=2.0)
        
        # Step 9: 上升并回到初始位置
        rospy.loginfo("\n[Step 9] 上升并回到初始位置...")
        self.move_to_joint_state([0, 0, 0])  # 回到初始关节位置
        
        rospy.loginfo("\n" + "=" * 50)
        rospy.loginfo("✅ 拾取放置完成!")
        rospy.loginfo("=" * 50)
        
        return True
    
    def shutdown(self):
        """关闭MoveIt"""
        moveit_commander.roscpp_shutdown()
        rospy.loginfo("MoveIt已关闭")


def main():
    """主函数：演示MoveIt的使用"""
    try:
        # 初始化控制器
        controller = MoveItController(group_name="arm")
        
        # 演示1: 移动到目标位置
        rospy.loginfo("\n" + "=" * 50)
        rospy.loginfo("演示1: 移动到目标位置")
        rospy.loginfo("=" * 50)
        
        # 移动到初始位置（张开手臂）
        rospy.loginfo("移动到初始位置...")
        controller.move_to_joint_state([0, 0, 0])
        rospy.sleep(1.0)
        
        # 演示2: 完整的拾取放置流程
        rospy.loginfo("\n" + "=" * 50)
        rospy.loginfo("演示2: 完整的拾取放置流程")
        rospy.loginfo("=" * 50)
        
        # 提示: 若规划超时，可将目标改到更靠近可达区
        # 拾取点: 适当减小半径，避免超出IK可达域
        pick_pos = [1.2, 0.0, 0.08]
        # 放置点: 在工作区内选择一个安全位置
        place_pos = [0.8, 0.3, 0.05]
        
        success = controller.pick_and_place(
            pick_x=pick_pos[0], pick_y=pick_pos[1], pick_z=pick_pos[2],
            place_x=place_pos[0], place_y=place_pos[1], place_z=place_pos[2]
        )
        
        if success:
            rospy.loginfo("\n✅ 演示完成!")
        else:
            rospy.logwarn("\n⚠️ 演示过程中出现问题")
        
        rospy.loginfo("\n按 Ctrl+C 退出...")
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("程序被用户中断")
    except Exception as e:
        rospy.logerr(f"发生错误: {e}")
    finally:
        if 'controller' in locals():
            controller.shutdown()


if __name__ == '__main__':
    main()
