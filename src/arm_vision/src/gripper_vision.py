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

        self.objects_height_pub = rospy.Publisher('/objects_height', Float32, queue_size=10)

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
        
        # 将浮点数坐标转换为整数
        u = 424.5
        v = 240.5
        u_int = int(round(u))
        v_int = int(round(v))
        depth_value = self.get_object_depth(u_int, v_int)
        gripper_camera_height = 0.2954  # 反向计算出来
        object_height = gripper_camera_height - depth_value
        # pose_z = object_height / 2
        if depth_value is not None:
            rospy.loginfo(f"Gripper Depth at ({u_int},{v_int}): {depth_value}")
            rospy.loginfo(f"Object Height at ({u_int},{v_int}): {object_height}")
        else:
            rospy.logwarn(f"Failed to get depth value at ({u_int},{v_int})") 
        
        self.objects_height_pub.publish(Float32(object_height)) 
        # 可视化
        cv2.imshow('Gripper RGB Image', rgb)
        cv2.imshow('Gripper Depth Image', depth)
        cv2.waitKey(1)

    def run(self):
        rate = rospy.Rate(10)
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