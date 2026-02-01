#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gazebo Cylinder Display Utilities (ROS1 Version)
在 Gazebo 中生成/删除指定尺寸与位置的圆柱体。
用法: 在你的 ROS1 节点中实例化 CylinderSpawner(),然后调用 spawn_cylinder/delete_entity。
"""

import rospy
from gazebo_msgs.srv import SpawnModel, DeleteModel
import math


class CylinderSpawner:
    def __init__(self):
        """初始化 CylinderSpawner,创建服务客户端"""
        # 等待 Gazebo 服务就绪
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        rospy.wait_for_service('/gazebo/delete_model')

        # 创建服务代理
        self.spawn_client = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.delete_client = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

        rospy.loginfo("CylinderSpawner 初始化完成")

    def _yaw_to_quat(self, yaw):
        """将 yaw 角转换为四元数"""
        half = yaw * 0.5
        return (0.0, 0.0, math.sin(half), math.cos(half))

    def _make_cylinder_sdf(self, name, radius, length,
                           color_rgba=(0.2, 0.8, 0.2, 1.0), mass=0.2):
        """
        生成圆柱体的 SDF 描述字符串

        参数:
            name: 圆柱体名称
            radius: 半径 (m)
            length: 高度/长度 (m)
            color_rgba: 颜色元组 (r, g, b, a)
            mass: 质量 (kg)
        """
        r_color, g_color, b_color, a_color = color_rgba

        # 对于实心圆柱体的惯性矩公式:
        # 关于 x、y 轴: I = (1/12) * m * (3r^2 + h^2)
        # 关于 z 轴(对称轴): I = 0.5 * m * r^2
        ixx = (mass / 12.0) * (3.0 * radius * radius + length * length)
        iyy = ixx
        izz = 0.5 * mass * radius * radius

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
          <cylinder>
            <radius>{radius}</radius>
            <length>{length}</length>
          </cylinder>
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
          <cylinder>
            <radius>{radius}</radius>
            <length>{length}</length>
          </cylinder>
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
            radius=radius, length=length,
            r=r_color, g=g_color, b=b_color, a=a_color
        )

    def spawn_cylinder(self, name, x, y, z, yaw,
                       radius, length,
                       color_rgba=(0.2, 0.8, 0.2, 1.0),
                       mass=0.2,
                       reference_frame='world'):
        """
        在 Gazebo 中生成圆柱体

        参数:
            name: 模型名称
            x, y, z: 位置 (m)
            yaw: 偏航角 (rad)
            radius: 半径 (m)
            length: 高度/长度 (m)
            color_rgba: 颜色 (r, g, b, a)
            mass: 质量 (kg)
            reference_frame: 参考坐标系

        返回:
            bool: 是否生成成功
        """
        try:
            xml = self._make_cylinder_sdf(name, radius, length, color_rgba, mass)
            qx, qy, qz, qw = self._yaw_to_quat(yaw)

            from geometry_msgs.msg import Pose, Point, Quaternion
            initial_pose = Pose()
            initial_pose.position = Point(x, y, z)
            initial_pose.orientation = Quaternion(qx, qy, qz, qw)

            resp = self.spawn_client(
                model_name=name,
                model_xml=xml,
                robot_namespace='',
                initial_pose=initial_pose,
                reference_frame=reference_frame
            )

            rospy.loginfo(
                '生成圆柱体: name=%s, pos=(%.3f,%.3f,%.3f), radius=%.3f, length=%.3f'
                % (name, x, y, z, radius, length)
            )
            return resp.success

        except rospy.ServiceException as e:
            rospy.logerr('生成圆柱体失败: %s' % e)
            return False

    def display_test_cylinder(self, cyl_pos, radius=0.016, length=0.032,
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

        rospy.loginfo("生成测试圆柱体...")
        success = self.spawn_cylinder(
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
            rospy.loginfo("圆柱体 '%s' 生成成功,位置在夹取范围内" % cyl_name)
        return success

    def delete_entity(self, name):
        """
        删除 Gazebo 中的实体

        参数:
            name: 实体名称

        返回:
            bool: 是否删除成功
        """
        try:
            resp = self.delete_client(model_name=name)
            rospy.loginfo('删除实体: %s' % name)
            return resp.success
        except rospy.ServiceException as e:
            rospy.logerr('删除实体失败: %s' % e)
            return False


__all__ = ['CylinderSpawner']

