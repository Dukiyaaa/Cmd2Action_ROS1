#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gazebo Box Display Utilities (ROS1 Version)
在 Gazebo 中生成/删除指定尺寸与位置的方块。
用法: 在你的 ROS1 节点中实例化 BoxSpawner()，然后调用 spawn_box/delete_entity。
"""

import rospy
from gazebo_msgs.srv import SpawnModel, DeleteModel
import math


class BoxSpawner:
    def __init__(self):
        """初始化 BoxSpawner，创建服务客户端"""
        # 等待 Gazebo 服务就绪
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        rospy.wait_for_service('/gazebo/delete_model')
        
        # 创建服务代理
        self.spawn_client = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.delete_client = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        
        rospy.loginfo("BoxSpawner 初始化完成")

    def _yaw_to_quat(self, yaw):
        """将 yaw 角转换为四元数"""
        half = yaw * 0.5
        return (0.0, 0.0, math.sin(half), math.cos(half))

    def _make_box_sdf(self, name, size_xyz, color_rgba=(0.8, 0.2, 0.2, 1.0), mass=0.2):
        """
        生成方块的 SDF 描述字符串
        
        参数:
            name: 方块名称
            size_xyz: 尺寸元组 (sx, sy, sz)
            color_rgba: 颜色元组 (r, g, b, a)
            mass: 质量 (kg)
        """
        sx, sy, sz = size_xyz
        r, g, b, a = color_rgba
        
        # 计算惯性矩阵
        ixx = (mass / 12.0) * (sy * sy + sz * sz)
        iyy = (mass / 12.0) * (sx * sx + sz * sz)
        izz = (mass / 12.0) * (sx * sx + sy * sy)
        
        return """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>false</static>
    <link name="link">
      <inertial>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{ixx}</ixx>
          <iyy>{iyy}</iyy>
          <izz>{izz}</izz>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <box>
            <size>{sx} {sy} {sz}</size>
          </box>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>1000.0</mu>
              <mu2>1000.0</mu2>
            </ode>
          </friction>
          <contact>
            <ode>
              <kp>1000000.0</kp>
              <kd>100.0</kd>
              <min_depth>0.001</min_depth>
              <max_vel>0.0</max_vel>
            </ode>
          </contact>
        </surface>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>{sx} {sy} {sz}</size>
          </box>
        </geometry>
        <material>
          <ambient>{r} {g} {b} {a}</ambient>
          <diffuse>{r} {g} {b} {a}</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>""".format(
            name=name,
            mass=mass, ixx=ixx, iyy=iyy, izz=izz,
            sx=sx, sy=sy, sz=sz,
            r=r, g=g, b=b, a=a
        )

    def spawn_box(self, name, x, y, z, yaw, sx, sy, sz, 
                  color_rgba=(0.2, 0.6, 0.9, 1.0), mass=0.2, reference_frame='world'):
        """
        在 Gazebo 中生成方块
        
        参数:
            name: 方块名称
            x, y, z: 位置 (m)
            yaw: 偏航角 (rad)
            sx, sy, sz: 尺寸 (m)
            color_rgba: 颜色 (r, g, b, a)
            mass: 质量 (kg)
            reference_frame: 参考坐标系
        
        返回:
            bool: 是否生成成功
        """
        try:
            xml = self._make_box_sdf(name, (sx, sy, sz), color_rgba, mass)
            qx, qy, qz, qw = self._yaw_to_quat(yaw)
            
            # 创建初始位姿
            from geometry_msgs.msg import Pose, Point, Quaternion
            initial_pose = Pose()
            initial_pose.position = Point(x, y, z)
            initial_pose.orientation = Quaternion(qx, qy, qz, qw)
            
            # 调用服务生成模型
            success = self.spawn_client(
                model_name=name,
                model_xml=xml,
                robot_namespace='',
                initial_pose=initial_pose,
                reference_frame=reference_frame
            )
            
            # rospy.loginfo('生成方块: name=%s, pos=(%.3f,%.3f,%.3f), size=(%.3f,%.3f,%.3f)' 
            #              % (name, x, y, z, sx, sy, sz))
            return success.success
            
        except rospy.ServiceException as e:
            rospy.logerr('生成方块失败: %s' % e)
            return False

    def display_test_box(self, box_pos, box_size=(0.032, 0.032, 0.032), box_color=(0.2, 0.6, 0.9, 1.0), box_mass=0.01, box_name='test_box'):
        box_x, box_y, box_z = box_pos
        sx, sy, sz = box_size
        r, g, b, a = box_color
        
        # 检查方块位置是否在机械臂夹取范围内
        # origin_height = 0.5
        # theta1_c, theta2_c, d3_c, reachable = inverse_kinematics(box_x, box_y, origin_height, elbow="down")
        # if not reachable:
        #     rospy.logwarn("方块位置不在机械臂夹取范围内: (%.3f, %.3f, %.3f)" % (box_x, box_y, box_z))
        #     rospy.logwarn("无法生成方块 '%s'" % box_name)
        #     return False
        
        # rospy.loginfo("生成测试方块...")
        success = self.spawn_box(
            name=box_name,
            x=box_x, y=box_y, z=box_z,
            yaw=0.0,
            sx=sx, sy=sy, sz=sz,
            color_rgba=(r, g, b, a),
            mass=box_mass,
            reference_frame='world'
        )
        
        # if success:
            # rospy.loginfo("方块 '%s' 生成成功，位置在夹取范围内" % box_name)
        # return success

    def delete_entity(self, name):
        """
        删除 Gazebo 中的实体
        
        参数:
            name: 实体名称
        
        返回:
            bool: 是否删除成功
        """
        try:
            success = self.delete_client(model_name=name)
            # rospy.loginfo('删除实体: %s' % name)
            return success.success
        except rospy.ServiceException as e:
            rospy.logerr('删除实体失败: %s' % e)
            return False

__all__ = ['BoxSpawner']