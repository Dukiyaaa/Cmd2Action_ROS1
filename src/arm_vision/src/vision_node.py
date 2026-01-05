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
from ultralytics import YOLO
import torch
import threading


class VisionNode:
    def __init__(self):
        rospy.init_node('vision_node', anonymous=True)
        print("[DEBUG] Vision Node init started", flush=True)
        
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
        self.depth_header = None

        # YOLOv8 配置
        print("[DEBUG] Loading params...", flush=True)
        self.model_path = rospy.get_param('~model_path', 'yolov8n.pt')
        self.conf_thres = rospy.get_param('~conf', 0.45)
        device_param = rospy.get_param('~device', None)
        if device_param in (None, '', 'auto'):
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device_param
        self.class_filter = set(rospy.get_param('~class_filter', []))  # e.g. [0] for person, [] for all
        
        print(f"[DEBUG] Loading YOLO model: {self.model_path} on device: {self.device}", flush=True)
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print(f"[DEBUG] Model loaded successfully", flush=True)
        except Exception as e:
            print(f"[ERROR] Failed to load YOLO model: {e}", flush=True)
            raise
        rospy.loginfo(f'YOLOv8 loaded: {self.model_path} on {self.device}')
        
        # 订阅摄像头话题（模拟 D435i 的话题名） 这些话题是系统自动发布的
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
                self.depth_header = msg.header
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
            depth_value: 深度值（米，或毫米会自动转换）
        
        返回:
            (x, y, z): 世界坐标（米）
        """
        if self.fx is None:
            rospy.logwarn("Camera not calibrated yet")
            return None
        
        # 深度单位转换：毫米 → 米（Gazebo 深度通常直接是米，这里兼容两种情况）
        z = float(depth_value)
        if z > 20.0:  # assume millimeters
            z = z / 1000.0
        
        if z <= 0 or z > 3:  # 滤除无效深度
            return None
        
        # 反向投影：像素坐标 → 相机坐标
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        # 返回相机坐标系下的 3D 点
        # 注意：需要根据相机在机械臂的安装位置进行坐标变换（后续可用 TF）
        return (x, y, z)
    
    def process_frame(self):
        """处理一帧图像"""
        with self.lock:
            if self.rgb_image is None or self.depth_image is None:
                return
            
            rgb = self.rgb_image.copy()
            depth = self.depth_image.copy()
        
        # 调试：打印一次图像信息
        if not hasattr(self, '_debug_printed'):
            print(f"[DEBUG] RGB shape: {rgb.shape}, dtype: {rgb.dtype}", flush=True)
            print(f"[DEBUG] Depth shape: {depth.shape}, dtype: {depth.dtype}", flush=True)
            self._debug_printed = True
        
        # 确保 RGB 是 uint8 且为 3 通道
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)
        
        if len(rgb.shape) == 2:  # 如果是灰度图，转为 BGR
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        elif rgb.shape[2] == 4:  # RGBA 转 BGR
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
        
        # YOLOv8 推理（直接用 BGR，model 会自动处理）
        try:
            results = self.model(rgb, conf=self.conf_thres, verbose=False, device=self.device)
            result = results[0]
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"YOLO inference error: {e}")
            import traceback
            traceback.print_exc()
            return

        if result is None or result.boxes is None or len(result.boxes) == 0:
            return

        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        detections = []
        for box, score, cls_id in zip(xyxy, scores, classes):
            if self.class_filter and cls_id not in self.class_filter:
                continue
            if score < self.conf_thres:
                continue

            u = int((box[0] + box[2]) / 2)
            v = int((box[1] + box[3]) / 2)

            if u < 0 or v < 0 or u >= depth.shape[1] or v >= depth.shape[0]:
                continue

            depth_value = depth[v, u]
            point = self.pixel_to_world(u, v, depth_value)
            if point is None:
                continue

            pose = PoseStamped()
            if self.depth_header:
                pose.header = self.depth_header
            else:
                pose.header.stamp = rospy.Time.now()
                pose.header.frame_id = 'camera_link'
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = point
            pose.pose.orientation.w = 1.0
            self.detected_objects_pub.publish(pose)
            detections.append((u, v, cls_id, score))

        # Debug overlay
        if detections:
            for (box, score, cls_id) in zip(xyxy, scores, classes):
                if score < self.conf_thres or (self.class_filter and cls_id not in self.class_filter):
                    continue
                cv2.rectangle(rgb, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                label = f"{cls_id}:{score:.2f}"
                cv2.putText(rgb, label, (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 临时：显示深度图（调试用）
        if depth is not None:
            depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imshow('Depth Image', depth_normalized)
            cv2.imshow('RGB + YOLO', rgb)
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
