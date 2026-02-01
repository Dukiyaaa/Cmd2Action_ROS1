#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
import os
import rospkg
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from arm_vision.msg import DetectedObject,DetectedObjectPool
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
        #内参矩阵,fx, fy, cx, cy在这个矩阵中获得
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

        # 加入YOLO
        print("[DEBUG] Loading params...", flush=True)
        # 通过launch文件获取参数 逗号后面是默认参数
        # 获取包路径,构建相对路径
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('arm_vision')
        default_model_path = os.path.join(package_path, 'model', 'best.pt')
        self.model_path = rospy.get_param('~model_path', default_model_path)
        self.conf_thres = rospy.get_param('~conf', 0.45)
        device_param = rospy.get_param('~device', None)
        if device_param in (None, '', 'auto'):
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device_param
        self.class_filter = set(rospy.get_param('~class_filter', []))  # e.g. [0] for person, [] for all

        # 测试用：从文件读取一帧 RGB/Depth,而不使用订阅的实时话题
        self.test_rgb_path = rospy.get_param('~test_rgb_path', None)
        self.test_depth_path = rospy.get_param('~test_depth_path', None)
        
        print(f"[DEBUG] Loading YOLO model: {self.model_path} on device: {self.device}", flush=True)
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print(f"[DEBUG] Model loaded successfully", flush=True)
        except Exception as e:
            print(f"[ERROR] Failed to load YOLO model: {e}", flush=True)
            raise
        rospy.loginfo(f'YOLOv8 loaded: {self.model_path} on {self.device}')
        # 发布检测到的物体池（包含位姿、类别ID和置信度）
        self.detected_objects_pub = rospy.Publisher('/detected_objects', DetectedObjectPool, queue_size=10)

        # 订阅三个话题,用于获得rgb图像、深度图、相机内参 注意这里的话题名字是在urdf中自己定义的
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
        
    # 世界转图像上的坐标,但这里的转换并不完全,只是转了一下坐标轴,由于焦距参数单位问题,没办法单独得到图像坐标(mm)
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
        
    # 直接从世界坐标到像素坐标,中间包含世界坐标转图像坐标的部分过程
    def world_to_pixel_coordinate(self, x_world, y_world, z_world):
        ps = PointStamped()
        ps.header.stamp = rospy.Time(0)
        ps.header.frame_id = 'world' 
        ps.point.x, ps.point.y, ps.point.z = x_world, y_world, z_world

        try:
            # 从 world 变换到 camera_color_optical_frame
            transform = self.tf_buffer.lookup_transform(
                'camera_color_optical_frame', 'world', rospy.Time(0), rospy.Duration(1.0)
            )
            cam_ps = tf2_geometry_msgs.do_transform_point(ps, transform)
            # 图像转像素坐标
            x_pixel = (cam_ps.point.x * self.fx) / cam_ps.point.z + self.cx
            y_pixel = (cam_ps.point.y * self.fy) / cam_ps.point.z + self.cy
            z_pixel = cam_ps.point.z
            return (x_pixel, y_pixel, z_pixel)
        except Exception as e:
            rospy.logwarn(f"TF transform failed: {e}")
            return None
    
    # 像素坐标转回世界坐标
    def pixel_to_world_coordinate(self, u, v, depth):
        # 像素坐标转相机坐标
        x_cam = (u - self.cx) * depth / self.fx
        y_cam = (v - self.cy) * depth / self.fy
        z_cam = depth

        ps = PointStamped()
        ps.header.stamp = rospy.Time(0)
        ps.header.frame_id = 'camera_color_optical_frame' 
        ps.point.x, ps.point.y, ps.point.z = x_cam, y_cam, z_cam

        try:
            # 从 camera_link 变换到 world
            transform = self.tf_buffer.lookup_transform(
                'world', 'camera_color_optical_frame', rospy.Time(0), rospy.Duration(1.0)
            )
            world_ps = tf2_geometry_msgs.do_transform_point(ps, transform)
            return (world_ps.point.x, world_ps.point.y, world_ps.point.z)
        except Exception as e:
            rospy.logwarn(f"TF transform failed: {e}")
            return None
    def process_frame(self):
        # 每次进入该函数时,会获取当前最新图像并处理
        with self.lock:
            if self.rgb_image is None or self.depth_image is None:
                return
            # 采用拷贝,避免在处理过程中图像被修改
            rgb = self.rgb_image.copy()
            depth = self.depth_image.copy()

        # 如果提供了测试图片路径,则优先使用文件中的图像（便于离线调试）
        if self.test_rgb_path not in (None, '', 'none', 'None'):
            test_rgb = cv2.imread(self.test_rgb_path, cv2.IMREAD_COLOR)
            if test_rgb is None:
                rospy.logwarn_throttle(5.0, f"Failed to read test_rgb_path: {self.test_rgb_path}")
                return
            rospy.loginfo_throttle(5.0, f"Using test RGB image from: {self.test_rgb_path}")
            rgb = test_rgb

        if self.test_depth_path not in (None, '', 'none', 'None'):
            test_depth = cv2.imread(self.test_depth_path, cv2.IMREAD_UNCHANGED)
            if test_depth is None:
                rospy.logwarn_throttle(5.0, f"Failed to read test_depth_path: {self.test_depth_path}")
                return
            rospy.loginfo_throttle(5.0, f"Using test depth image from: {self.test_depth_path}")
            depth = test_depth
        
        #urdf,camera_fixed_joint的旋转导致图像需要旋转180度
        rgb = cv2.rotate(rgb, cv2.ROTATE_180)
        depth = cv2.rotate(depth, cv2.ROTATE_180)
        
        # cam_set = self.image_to_cam_coordinate(319, 269, 1.5)
        # world_set = self.image_to_world_coordinate(319, 269, 1.5)
        # if cam_set is not None:
        #     rospy.loginfo(f"Cam coords at (320,240) depth 1.5m: {cam_set}")
        # if world_set is not None:
        #     rospy.loginfo(f"World coords at (320,240) depth 1.5m: {world_set}")

        # cam_set_back = self.world_to_image_coordinate(1.2, 0.0, 0.05)
        # if cam_set_back is not None:
        #     rospy.loginfo(f"World to Cam coords: {cam_set_back}")

        # image_set_back = self.world_to_pixel_coordinate(0.8, 0.65, 0.05)
        # if image_set_back is not None:
        #     rospy.loginfo(f"World to Pixel coords: {image_set_back}")

        # world_set_back = self.pixel_to_world_coordinate(image_set_back[0], image_set_back[1], image_set_back[2])
        # if world_set_back is not None:
        #     rospy.loginfo(f"Pixel to World coords: {world_set_back}")

        # return
        
        # 确保 RGB 是 uint8 且为 3 通道
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)
        
        if len(rgb.shape) == 2:  # 如果是灰度图,转为 BGR
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        elif rgb.shape[2] == 4:  # RGBA 转 BGR
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
        
        # 用模型推理这帧图像
        try:
            results = self.model(rgb, conf=self.conf_thres, verbose=False, device=self.device)
            result = results[0]
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"YOLO inference error: {e}")
            import traceback
            traceback.print_exc()
            return

        # Ultrayltics 的result对象自动带boxes这个属性 这个属性是监测框,内含xyxy,conf,cls
        if result is None or result.boxes is None or len(result.boxes) == 0:
            rospy.loginfo_throttle(5.0, "No detections")
            # 即使没有检测到物体,也显示图像
            cv2.imshow('RGB Image', rgb)
            cv2.waitKey(1)
            return

        boxes = result.boxes
        # 后续处理放在CPU上进行
        xyxy = boxes.xyxy.cpu().numpy() #框的左上、右下坐标
        rospy.loginfo_throttle(5.0, f"Detections: {xyxy.shape[0]} boxes")
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        detections = []
        detected_objects = []  # 存 DetectedObject 实例
        for box, score, cls_id in zip(xyxy, scores, classes):
            # if self.class_filter and cls_id not in self.class_filter:
            #     rospy.loginfo(f"Class {cls_id} not in filter")
            #     continue
            if score < self.conf_thres:
                rospy.loginfo(f"Score {score} less than conf_thres")
                continue
            
            # 取中心点
            u = int((box[0] + box[2]) / 2)
            v = int((box[1] + box[3]) / 2)

            # rospy.loginfo(f"u: {u}, v: {v}")
            if u < 0 or v < 0 or u >= depth.shape[1] or v >= depth.shape[0]:
                continue
            
            # 通过取出来的坐标去深度图里匹配对应的深度值
            # depth_value = depth[v, u]
            depth_value = 1.25
            point = self.pixel_to_world_coordinate(u, v, depth_value)
            rospy.loginfo(f"Pixel to World coords: {point}")
            if point is None:
                continue

            # 创建自定义消息,包含位姿、类别ID和置信度
            detected_obj = DetectedObject()
            detected_obj.pose = PoseStamped()
            if self.depth_header:
                detected_obj.pose.header = self.depth_header
            else:
                detected_obj.pose.header.stamp = rospy.Time.now()
                detected_obj.pose.header.frame_id = 'world'
            detected_obj.pose.pose.position.x, detected_obj.pose.pose.position.y, detected_obj.pose.pose.position.z = point
            detected_obj.pose.pose.orientation.w = 1.0 # 朝向默认不变
            detected_obj.class_id = int(cls_id)
            detected_obj.confidence = float(score)
            # self.detected_objects_pub.publish(detected_obj)
            detections.append((box, score, cls_id)) # 用于可视化,保存完整的检测信息
            detected_objects.append(detected_obj)

        if detected_objects:
            pool_msg = DetectedObjectPool()
            pool_msg.header.stamp = rospy.Time.now()
            pool_msg.header.frame_id = 'world'
            pool_msg.objects = detected_objects
            self.detected_objects_pub.publish(pool_msg)

        # 绘制检测框和标签
        for (box, score, cls_id) in detections:
            # 绘制矩形框
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 显示类别ID和置信度
            label = f"Class {cls_id}: {score:.2f}"
            # 获取文本大小以调整标签位置
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # 绘制标签背景
            cv2.rectangle(rgb, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), (0, 255, 0), -1)
            # 绘制标签文本
            cv2.putText(rgb, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 显示绘制了检测框的图像
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