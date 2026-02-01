# arm_vision/src/gripper_vision.py
"""
 gripper下的相机
"""
import rospy
import cv2
import threading
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

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

        rospy.Subscriber('/gripper_camera/color/image_raw', Image, self._rgb_callback)
        rospy.Subscriber('/gripper_camera/depth/image_rect_raw', Image, self._depth_callback)
        rospy.Subscriber('/gripper_camera/color/camera_info', CameraInfo, self._camera_info_callback)

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