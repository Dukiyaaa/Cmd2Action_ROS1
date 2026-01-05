#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from ultralytics import YOLO
import torch
import threading
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped


class VisionNode:
    def __init__(self):
        rospy.init_node('vision_node', anonymous=True)

        # cv图片转换器 + 线程锁
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        
        # 相机内参（从 camera_info 获取）
        #内参矩阵，fx, fy, cx, cy在这个矩阵中获得
        self.camera_matrix = None 
        # 畸变系数
        self.dist_coeffs = None 
        # 焦距
        self.fx = None 
        self.fy = None
        # 主点坐标
        self.cx = None
        self.cy = None

        # 用于存储图像数据
        self.rgb_image = None
        self.depth_image = None
        self.depth_header = None

        # TF 变换工具：用于相机系 → 世界系坐标转换
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 订阅三个话题，用于获得rgb图像、深度图、相机内参 注意这里的话题名字是在urdf中自己定义的
        rospy.Subscriber('/camera/color/image_raw', Image, self._rgb_callback)
        rospy.Subscriber('/camera/depth/image_rect_raw', Image, self._depth_callback)
        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self._camera_info_callback)

    def _rgb_callback(self, msg):
        try:
            with self.lock:
                # 将ROS图像消息转换为OpenCV格式的BGR图像
                self.rgb_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            rospy.logwarn(f"RGB callback error: {e}")

    def _depth_callback(self, msg):
        try:
            with self.lock:
                # 将ROS图像消息转换为OpenCV深度图像
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                self.depth_header = msg.header
        except Exception as e:
            rospy.logwarn(f"Depth callback error: {e}")

    def _camera_info_callback(self,msg):
        if self.fx is None:
            self.fx = msg.K[0]      # 焦距 x
            self.fy = msg.K[4]      # 焦距 y
            self.cx = msg.K[2]      # 光心 x
            self.cy = msg.K[5]      # 光心 y
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            
            rospy.loginfo(f'Camera calibrated: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}')

    # 完整坐标转换：世界-相机-图像-像素
    # 世界坐标转相机坐标
    def world_to_cam_coordinate(self, x_world, y_world, z_world):
        ps = PointStamped()
        ps.header.stamp = rospy.Time(0)
        ps.header.frame_id = 'world' 
        ps.point.x, ps.point.y, ps.point.z = x_world, y_world, z_world

        try:
            # 从 world 变换到 camera_link
            transform = self.tf_buffer.lookup_transform(
                'camera_link', 'world', rospy.Time(0), rospy.Duration(1.0)
            )
            cam_ps = tf2_geometry_msgs.do_transform_point(ps, transform)
            return (cam_ps.point.x, cam_ps.point.y, cam_ps.point.z)
        except Exception as e:
            rospy.logwarn(f"TF transform failed: {e}")
            return None
        
    # 世界转图像上的坐标，但这里的转换并不完全，只是转了一下坐标轴，由于焦距参数单位问题，没办法单独得到图像坐标(mm)
    def world_to_image_coordinate(self, x_world, y_world, z_world):
        ps = PointStamped()
        ps.header.stamp = rospy.Time(0)
        ps.header.frame_id = 'world' 
        ps.point.x, ps.point.y, ps.point.z = x_world, y_world, z_world

        try:
            # 从 world 变换到 camera_link
            transform = self.tf_buffer.lookup_transform(
                'camera_color_optical_frame', 'world', rospy.Time(0), rospy.Duration(1.0)
            )
            cam_ps = tf2_geometry_msgs.do_transform_point(ps, transform)
            return (cam_ps.point.x, cam_ps.point.y, cam_ps.point.z)
        except Exception as e:
            rospy.logwarn(f"TF transform failed: {e}")
            return None
    # 直接从世界坐标到像素坐标，中间包含世界坐标转图像坐标的部分过程
    def world_to_pixel_coordinate(self, x_world, y_world, z_world):
        ps = PointStamped()
        ps.header.stamp = rospy.Time(0)
        ps.header.frame_id = 'world' 
        ps.point.x, ps.point.y, ps.point.z = x_world, y_world, z_world

        try:
            # 从 world 变换到 camera_link
            transform = self.tf_buffer.lookup_transform(
                'camera_color_optical_frame', 'world', rospy.Time(0), rospy.Duration(1.0)
            )
            cam_ps = tf2_geometry_msgs.do_transform_point(ps, transform)
            x_pixel = (cam_ps.point.x * self.fx) / cam_ps.point.z + self.cx
            y_pixel = (cam_ps.point.y * self.fy) / cam_ps.point.z + self.cy
            return (x_pixel, y_pixel)
        except Exception as e:
            rospy.logwarn(f"TF transform failed: {e}")
            return None
    def image_to_cam_coordinate(self, u, v, depth_value):
        if self.fx is None:
            rospy.logwarn("Camera not calibrated yet")
            return None       
        
        z = float(depth_value)

        # 像素坐标 → 相机坐标
        x = (u - self.cx) * (z / self.fx)
        y = (v - self.cy) * (z / self.fy)
        
        # 返回相机坐标系下的 3D 点
        # 注意：需要根据相机在机械臂的安装位置进行坐标变换（后续可用 TF）
        return (x, y, z)

    def image_to_world_coordinate(self, u, v, depth_value):
        """
        将像素坐标 + 深度转换为世界坐标world frame。
        先用相机内参反投影到相机光学坐标系，再通过 TF 变换到 world。
        """
        cam_point = self.image_to_cam_coordinate(u, v, depth_value)
        if cam_point is None:
            return None

        ps = PointStamped()
        ps.header.stamp = self.depth_header.stamp if self.depth_header else rospy.Time(0)
        ps.header.frame_id = 'camera_color_optical_frame'
        ps.point.x, ps.point.y, ps.point.z = cam_point

        try:
            transform = self.tf_buffer.lookup_transform(
                'world', 'camera_color_optical_frame', rospy.Time(0), rospy.Duration(1.0)
            )
            world_ps = tf2_geometry_msgs.do_transform_point(ps, transform)
            return (world_ps.point.x, world_ps.point.y, world_ps.point.z)
        except Exception as e:
            rospy.logwarn(f"TF transform failed: {e}")
            return None
    
    def process_frame(self):
        # 每次进入该函数时，会获取当前最新图像并处理
        with self.lock:
            if self.rgb_image is None or self.depth_image is None:
                return
            # 采用拷贝，避免在处理过程中图像被修改
            rgb = self.rgb_image.copy()
            depth = self.depth_image.copy()
        
        #urdf，camera_fixed_joint的旋转导致图像需要旋转180度
        rgb = cv2.rotate(rgb, cv2.ROTATE_180)
        depth = cv2.rotate(depth, cv2.ROTATE_180)
        
        # cam_set = self.image_to_cam_coordinate(319, 269, 1.5)
        # world_set = self.image_to_world_coordinate(319, 269, 1.5)
        # if cam_set is not None:
        #     rospy.loginfo(f"Cam coords at (320,240) depth 1.5m: {cam_set}")
        # if world_set is not None:
        #     rospy.loginfo(f"World coords at (320,240) depth 1.5m: {world_set}")

        cam_set_back = self.world_to_cam_coordinate(1.2, 0.0, 0.05)
        if cam_set_back is not None:
            rospy.loginfo(f"World to Cam coords: {cam_set_back}")

        image_set_back = self.world_to_pixel_coordinate(1.2, 0.0, 0.05)
        if image_set_back is not None:
            rospy.loginfo(f"World to Pixel coords: {image_set_back}")
        # 显示图像
        cv2.imshow('Depth Image', depth)
        cv2.imshow('RGB Image', rgb)
        cv2.waitKey(1)

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        
        while not rospy.is_shutdown():
            self.process_frame()
            rate.sleep()
def main():
    try:
        vision_node = VisionNode()
        vision_node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Vision node interrupted")

if __name__ == '__main__':
    main()