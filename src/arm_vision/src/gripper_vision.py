# arm_vision/src/gripper_vision.py
"""
 gripper下的相机
"""
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import rospy
import cv2
import threading
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from coordinate_transformer import CoordinateTransformer
from std_msgs.msg import Float32
from arm_vision.msg import GripperObjectInfo
import numpy as np

from config import (
    GRIPPER_CENTER_U,
    GRIPPER_CENTER_V,
    GRIPPER_CAMERA_HEIGHT,
    VISION_RATE,
)

class GripperVision:
    def __init__(self):
        rospy.init_node('gripper_vision_node', anonymous=True)

        # 初始化组件
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        # 存储数据
        self.rgb_image = None
        self.depth_image = None
        self.depth_header = None

        self.object_info_pub = rospy.Publisher('/gripper_object_info', GripperObjectInfo, queue_size=10)

        rospy.Subscriber('/gripper_camera/color/image_raw', Image, self._rgb_callback)
        rospy.Subscriber('/gripper_camera/depth/image_rect_raw', Image, self._depth_callback)
        rospy.Subscriber('/gripper_camera/color/gripper_camera_info', CameraInfo, self._camera_info_callback)

        # 坐标转换
        self.transformer = CoordinateTransformer()
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
            rospy.loginfo(f'Gripper Camera calibrated: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}')

    def get_object_depth(self, u, v):
        if self.depth_image is None:
            return None
        if u < 0 or u >= self.depth_image.shape[1] or v < 0 or v >= self.depth_image.shape[0]:
            return None
        depth_value = self.depth_image[v, u]
        return depth_value
    
    def _extract_object_mask_from_depth(self, depth):
        """
        根据深度图提取桌面上物体的二值 mask
        返回:
            mask: uint8 二值图，物体区域为255，其他为0
            table_depth: 估计得到的桌面深度
        """
        if depth is None:
            return None, None

        depth_work = depth.copy()

        # 1. 过滤非法值
        invalid = ~((depth_work > 0) & np.isfinite(depth_work))
        depth_work[invalid] = 0

        valid_pixels = depth_work[depth_work > 0]
        if valid_pixels.size == 0:
            return None, None

        # 2. 用较大值估计桌面深度
        # 找到一个值，让85%的数据小于它，这个值视为桌面深度
        table_depth = np.percentile(valid_pixels, 85)

        # 3. 比桌面更近一截的区域，认为是物体
        depth_margin = 0.01  #比桌面至少近 1 cm的区域视为物体
        mask = np.zeros(depth_work.shape, dtype=np.uint8)
        mask[(depth_work > 0) & (depth_work < table_depth - depth_margin)] = 255

        return mask, table_depth

    def _get_largest_contour(self, mask, min_area=300):
        """
        从二值 mask 中提取最大轮廓
        返回:
            contour: 最大轮廓，若没有则返回 None
        """
        if mask is None:
            return None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) < min_area:
            return None

        return largest

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
        
        mask, table_depth = self._extract_object_mask_from_depth(depth)
        if mask is not None:
            rospy.loginfo(f"Estimated table depth: {table_depth:.4f}")
        
        contour = self._get_largest_contour(mask)
        rgb_vis = rgb.copy()

        if contour is not None:
            area = cv2.contourArea(contour)
            rospy.loginfo(f"Largest contour area: {area:.1f}")

            cv2.drawContours(rgb_vis, [contour], -1, (0, 255, 0), 2)
        
        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect
        rospy.loginfo(f"Raw rect angle: {angle:.2f}")

        box = cv2.boxPoints(rect)
        box = box.astype(int)
        cv2.drawContours(rgb_vis, [box], 0, (0, 0, 255), 2)

        # 将浮点数坐标转换为整数
        u = GRIPPER_CENTER_U
        v = GRIPPER_CENTER_V
        u_int = int(round(u))
        v_int = int(round(v))
        depth_value = self.get_object_depth(u_int, v_int)
        gripper_camera_height = GRIPPER_CAMERA_HEIGHT  # 反向计算出来
        object_height = gripper_camera_height - depth_value
        # pose_z = object_height / 2
        if depth_value is not None:
            rospy.loginfo(f"Gripper Depth at ({u_int},{v_int}): {depth_value}")
            rospy.loginfo(f"Object Height at ({u_int},{v_int}): {object_height}")
        else:
            rospy.logwarn(f"Failed to get depth value at ({u_int},{v_int})") 
        
        msg = GripperObjectInfo()
        msg.height = object_height
        msg.yaw = 0.0
        msg.has_yaw = False

        self.object_info_pub.publish(msg)
        # 可视化
        cv2.imshow('Gripper RGB Image', rgb_vis)

        depth_vis = depth.copy()
        depth_vis[~np.isfinite(depth_vis)] = 0
        depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)
        # cv2.imshow('Gripper Depth Image', depth_vis)

        if mask is not None:
            cv2.imshow('Object Mask From Depth', mask)

        cv2.waitKey(1)

    def run(self):
        rate = rospy.Rate(VISION_RATE)
        while not rospy.is_shutdown():
            self.process_frame()
            rate.sleep()

def main():
    try:
        gripper_vision = GripperVision()
        gripper_vision.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Gripper vision node interrupted")

if __name__ == '__main__':
    main()