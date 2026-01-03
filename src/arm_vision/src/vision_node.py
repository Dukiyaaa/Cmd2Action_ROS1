#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vision Node for Arm Manipulation
订阅深度相机 RGB + Depth 图像，处理物体检测
"""

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
import threading


class VisionNode:
    def __init__(self):
        rospy.init_node('vision_node', anonymous=True)
        
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        
        # 相机内参（从 camera_info 获取）
        self.camera_matrix = None
        self.dist_coeffs = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        
        # 最新的图像数据
        self.rgb_image = None
        self.depth_image = None
        
        # 订阅摄像头话题（模拟 D435i 的话题名）
        rospy.Subscriber('/camera/color/image_raw', Image, self._rgb_callback)
        rospy.Subscriber('/camera/depth/image_rect_raw', Image, self._depth_callback)
        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self._camera_info_callback)
        
        # 发布检测到的物体位置
        self.detected_objects_pub = rospy.Publisher('/detected_objects', PoseStamped, queue_size=10)
        
        rospy.loginfo('Vision Node initialized, waiting for camera feeds...')
        rospy.loginfo('Subscribed to:')
        rospy.loginfo('  - /camera/color/image_raw (RGB)')
        rospy.loginfo('  - /camera/depth/image_rect_raw (Depth)')
        rospy.loginfo('  - /camera/color/camera_info')
    
    def _rgb_callback(self, msg):
        """RGB 图像回调"""
        try:
            with self.lock:
                self.rgb_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            rospy.logwarn(f"RGB callback error: {e}")
    
    def _depth_callback(self, msg):
        """深度图像回调"""
        try:
            with self.lock:
                # 深度图通常是 uint16，单位毫米
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except Exception as e:
            rospy.logwarn(f"Depth callback error: {e}")
    
    def _camera_info_callback(self, msg):
        """相机信息回调（只处理一次）"""
        if self.fx is None:
            self.fx = msg.K[0]      # 焦距 x
            self.fy = msg.K[4]      # 焦距 y
            self.cx = msg.K[2]      # 光心 x
            self.cy = msg.K[5]      # 光心 y
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            
            rospy.loginfo(f'Camera calibrated: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}')
    
    def pixel_to_world(self, u, v, depth_value):
        """
        将像素坐标 + 深度转换为世界坐标
        
        参数:
            u, v: 像素坐标
            depth_value: 深度值（毫米）
        
        返回:
            (x, y, z): 世界坐标（米）
        """
        if self.fx is None:
            rospy.logwarn("Camera not calibrated yet")
            return None
        
        # 深度单位转换：毫米 → 米
        z = depth_value / 1000.0
        
        if z <= 0 or z > 3:  # 滤除无效深度
            return None
        
        # 反向投影：像素坐标 → 相机坐标
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        # 返回相机坐标系下的 3D 点
        # 注意：需要根据相机在机械臂的安装位置进行坐标变换（后续可用 TF）
        return (x, y, z)
    
    def process_frame(self):
        """处理一帧图像（框架，后续集成 YOLOv8）"""
        with self.lock:
            if self.rgb_image is None or self.depth_image is None:
                return
            
            rgb = self.rgb_image.copy()
            depth = self.depth_image.copy()
        
        # TODO: 这里将集成 YOLOv8 推理
        # 1. RGB 通过 YOLOv8 → 得到 bounding box
        # 2. 从 bbox 中心获取像素坐标 (u, v)
        # 3. 从 depth map 获取深度值
        # 4. 转换为世界坐标
        # 5. 发布 PoseStamped 消息
        
        # 临时：显示深度图（调试用）
        if depth is not None:
            depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imshow('Depth Image', depth_normalized)
            cv2.imshow('RGB Image', rgb)
            cv2.waitKey(1)
    
    def run(self):
        """主循环"""
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
