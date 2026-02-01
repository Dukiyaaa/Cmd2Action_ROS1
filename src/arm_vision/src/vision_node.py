# arm_vision/src/vision_node.py
"""
主程序入口
"""
# 添加源代码目录到 Python 路径
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from detector import YOLODetector
from coordinate_transformer import CoordinateTransformer
from visualizer import Visualizer
import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from arm_vision.msg import DetectedObjectPool, DetectedObject
import cv2
import threading
from cv_bridge import CvBridge
import numpy as np

class VisionNode:
    def __init__(self):
        rospy.init_node('vision_node', anonymous=True)

        # 初始化组件
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        # 模型加载
        self.detector = YOLODetector(
            model_path=rospy.get_param('~model_path'),
            conf_thres=rospy.get_param('~conf', 0.45),
            device=rospy.get_param('~device', 'auto')
        )
        # 坐标转换
        self.transformer = CoordinateTransformer()
        # 可视化
        self.visualizer = Visualizer()

        # 参数
        self.test_rgb_path = rospy.get_param('~test_rgb_path', None)
        self.test_depth_path = rospy.get_param('~test_depth_path', None)

        # 存储数据
        self.rgb_image = None
        self.depth_image = None
        self.depth_header = None

        # 发布器
        self.detected_objects_pub = rospy.Publisher('/detected_objects', DetectedObjectPool, queue_size=10)

        # 订阅三个话题,用于获得rgb图像、深度图、相机内参 注意这里的话题名字是在urdf中自己定义的
        rospy.Subscriber('/camera/color/image_raw', Image, self._rgb_callback)
        rospy.Subscriber('/camera/depth/image_rect_raw', Image, self._depth_callback)
        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self._camera_info_callback)

        # 获取相机内参
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

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

    def _camera_info_callback(self, msg):
        if self.fx is None:
            self.fx = msg.K[0]
            self.fy = msg.K[4]
            self.cx = msg.K[2]
            self.cy = msg.K[5]
            self.transformer.set_camera_params(self.fx, self.fy, self.cx, self.cy)
            rospy.loginfo(f'Camera calibrated: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}')

    # 坐标线性回归优化器
    def correct_detection_value(self,detected_x, detected_y, k_x=1.025837, b_x=-0.009908, k_y=1.025445, b_y=0.001953):
        # x
        corrected_x = detected_x * k_x + b_x
        # y
        corrected_y = detected_y * k_y + b_y
        return corrected_x, corrected_y

    def process_frame(self):
        # 每次进入该函数时,会获取当前最新图像并处理
        with self.lock:
            if self.rgb_image is None or self.depth_image is None:
                return
            # 采用拷贝,避免在处理过程中图像被修改
            rgb = self.rgb_image.copy()
            depth = self.depth_image.copy()
        
        #urdf,camera_fixed_joint的旋转导致图像需要旋转180度
        rgb = cv2.rotate(rgb, cv2.ROTATE_180)
        depth = cv2.rotate(depth, cv2.ROTATE_180)
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

        # 检测
        detections = self.detector.detect(rgb)
        if not detections:
            rospy.loginfo_throttle(5.0, "No detections")
            # 即使没有检测到物体,也显示图像
            # cv2.imshow('RGB Image', rgb)
            # cv2.waitKey(1)
            return
        
        # 转换到世界坐标
        detected_objects = []
        for _, score, cls_id, u, v in detections:
            depth_value = depth[v, u] if u < depth.shape[1] and v < depth.shape[0] else 1.25
            depth_value = 1.25
            # rospy.loginfo(f'depth_value = {depth_value}')
            point = self.transformer.pixel_to_world_coordinate(u, v, depth_value)
            if point is None:
                continue
            rospy.loginfo(f"Pixel to World coords: {point}")
            # 坐标线性回归优化器
            corrected_x, corrected_y = self.correct_detection_value(point[0], point[1])
            point = (corrected_x, corrected_y, point[2])
            rospy.loginfo(f"Corrected World coords: {point}")
            
            obj = DetectedObject()
            obj.pose = PoseStamped()
            if self.depth_header:
                obj.pose.header = self.depth_header
            else:
                obj.pose.header.stamp = rospy.Time.now()
                obj.pose.header.frame_id = 'world'
            obj.pose.pose.position.x, obj.pose.pose.position.y, obj.pose.pose.position.z = point
            obj.pose.pose.orientation.w = 1.0
            obj.class_id = cls_id
            obj.confidence = score
            detected_objects.append(obj)

        if detected_objects:
            pool_msg = DetectedObjectPool()
            pool_msg.header.stamp = rospy.Time.now()
            pool_msg.header.frame_id = 'world'
            pool_msg.objects = detected_objects
            self.detected_objects_pub.publish(pool_msg)

        # 可视化
        rgb = self.visualizer.draw_detections(rgb, detections)
        # cv2.imshow('RGB Image', rgb)
        # cv2.imshow('Depth Image', depth)
        cv2.waitKey(1)

    def run(self):
        rate = rospy.Rate(10)
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
