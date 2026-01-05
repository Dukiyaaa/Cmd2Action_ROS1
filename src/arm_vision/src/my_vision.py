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