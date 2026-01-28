#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对象数据采集节点 (ROS1 + Gazebo),用于模型训练阶段数据集采集

功能:
- 在与 my_scara_action.py 中 pick&place 相同的可达范围内, 随机生成若干方块或圆柱体
  - 方块尺寸/颜色/质量参考 ArmController.display_test_box 默认参数
  - 圆柱体尺寸/颜色/质量参考 ArmController.display_test_cylinder 默认参数
- 订阅深度相机 RGB 图像 (/camera/color/image_raw), 将当前场景图像保存到磁盘
- 每一帧可以包含 1~N 个物体 (N 可通过参数调节)

使用说明(示例):
  rosrun arm_vision object_data_collector.py _output_dir:=/tmp/arm_dataset _num_objects_min:=1 _num_objects_max:=3
"""

import os
import sys
import random
import time

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def _add_arm_application_to_path():
    """
    将 arm_application/src 加入 sys.path, 以便导入 gazebo_box_display / gazebo_cylinder_display
    结构假定为:
      <ws_root>/src/arm_vision/src/this_file
      <ws_root>/src/arm_application/src/...
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))              # .../src/arm_vision/src
    arm_vision_dir = os.path.dirname(script_dir)                         # .../src/arm_vision
    src_dir = os.path.dirname(arm_vision_dir)                            # .../src
    arm_app_src = os.path.join(src_dir, 'arm_application', 'src')       # .../src/arm_application/src
    if os.path.isdir(arm_app_src) and arm_app_src not in sys.path:
        sys.path.insert(0, arm_app_src)


_add_arm_application_to_path()

from gazebo_box_display import BoxSpawner  # noqa: E402
from gazebo_cylinder_display import CylinderSpawner  # noqa: E402


class ObjectDataCollector:
    def __init__(self):
        rospy.init_node('object_data_collector', anonymous=True)

        # 参数配置
        self.output_dir = rospy.get_param('~output_dir', '/tmp/arm_vision_dataset')
        self.num_objects_min = rospy.get_param('~num_objects_min', 1)
        self.num_objects_max = rospy.get_param('~num_objects_max', 3)
        self.sleep_after_spawn = rospy.get_param('~sleep_after_spawn', 1.0)
        self.interval_between_scenes = rospy.get_param('~interval_between_scenes', 1.0)
        self.max_scenes = rospy.get_param('~max_scenes', 1000)

        # 与 my_scara_action.py 中保持一致的可达范围
        self.x_min, self.x_max = 0.3, 1.0
        self.y_min, self.y_max = -0.8, 0.8
        self.z_obj = 0.05

        # 方块 / 圆柱体默认参数 (来自 display_test_box / display_test_cylinder)
        self.box_size = (0.032, 0.032, 0.032)
        self.box_color = (0.2, 0.6, 0.9, 1.0)
        self.box_mass = 0.01

        self.cyl_radius = 0.015
        self.cyl_length = 0.032
        self.cyl_color = (0.2, 0.8, 0.2, 1.0)
        self.cyl_mass = 0.01

        # Gazebo 生成工具
        rospy.loginfo("等待 Gazebo 生成服务...")
        self.box_spawner = BoxSpawner()
        self.cyl_spawner = CylinderSpawner()

        # 图像订阅
        self.bridge = CvBridge()
        self.latest_rgb = None
        rospy.Subscriber('/camera/color/image_raw', Image, self._rgb_callback, queue_size=1)

        # 输出目录
        self.images_dir = os.path.join(self.output_dir, 'images')
        if not os.path.isdir(self.images_dir):
            os.makedirs(self.images_dir)

        self.scene_index = 0 + 500
        self.spawned_models = []  # 记录当前场景生成的模型名称, 便于删除

        rospy.loginfo("ObjectDataCollector 初始化完成, 输出目录: %s", self.images_dir)

    def _rgb_callback(self, msg):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            rospy.logwarn("RGB callback error: %s", e)

    def _clear_scene(self):
        """删除上一个场景中生成的所有 Gazebo 模型"""
        for name in self.spawned_models:
            # 尝试用两个 spawner 删除 (模型名不会冲突, 只会成功一个)
            try:
                self.box_spawner.delete_entity(name)
            except Exception:
                pass
            try:
                self.cyl_spawner.delete_entity(name)
            except Exception:
                pass
        self.spawned_models = []

    def _spawn_random_objects(self):
        """在可达范围内随机生成若干方块或圆柱体"""
        num_objects = random.randint(self.num_objects_min, self.num_objects_max)
        rospy.loginfo("生成 %d 个随机物体", num_objects)

        for i in range(num_objects):
            x = random.uniform(self.x_min, self.x_max)
            y = random.uniform(self.y_min, self.y_max)
            z = self.z_obj

            obj_type = random.choice(['box', 'cylinder'])
            model_name = f"{obj_type}_{self.scene_index}_{i}"

            if obj_type == 'box':
                sx, sy, sz = self.box_size
                success = self.box_spawner.spawn_box(
                    name=model_name,
                    x=x, y=y, z=z,
                    yaw=0.0,
                    sx=sx, sy=sy, sz=sz,
                    color_rgba=self.box_color,
                    mass=self.box_mass,
                    reference_frame='world'
                )
            else:
                success = self.cyl_spawner.spawn_cylinder(
                    name=model_name,
                    x=x, y=y, z=z,
                    yaw=0.0,
                    radius=self.cyl_radius,
                    length=self.cyl_length,
                    color_rgba=self.cyl_color,
                    mass=self.cyl_mass,
                    reference_frame='world'
                )

            if success:
                self.spawned_models.append(model_name)
            else:
                rospy.logwarn("生成物体失败: %s (%s, %.3f, %.3f, %.3f)", model_name, obj_type, x, y, z)

    def _save_current_image(self):
        """保存当前最新 RGB 图像到磁盘"""
        if self.latest_rgb is None:
            rospy.logwarn("当前没有可用的 RGB 图像, 跳过保存")
            return

        filename = f"scene_{self.scene_index:06d}.png"
        filepath = os.path.join(self.images_dir, filename)

        import cv2

        # urdf 中 camera_fixed_joint 的旋转导致图像需要旋转 180 度, 与 my_vision.py 保持一致
        rotated = cv2.rotate(self.latest_rgb, cv2.ROTATE_180)

        cv2.imwrite(filepath, rotated)
        rospy.loginfo("保存图像: %s", filepath)

    def run(self):
        rate = rospy.Rate(1.0 / max(self.interval_between_scenes, 0.01))

        while not rospy.is_shutdown() and self.scene_index < self.max_scenes + 500:
            rospy.loginfo("=== 场景 %d ===", self.scene_index)

            # 清理旧场景
            self._clear_scene()

            # 生成新物体
            self._spawn_random_objects()

            # 等待 Gazebo & 传感器稳定
            rospy.sleep(self.sleep_after_spawn)

            # 保存当前图像
            self._save_current_image()

            self.scene_index += 1
            rate.sleep()

        rospy.loginfo("采集完成: 总场景数 = %d", self.scene_index)


def main():
    try:
        node = ObjectDataCollector()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()

