# arm_vision/src/coordinate_transformer.py
"""
坐标转换模块,在这个文件里做相机内参标定、坐标系转换
"""
import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped

class CoordinateTransformer:
    def __init__(self):
        # TF 变换工具：用于相机系 → 世界系坐标转换
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
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

    def set_camera_params(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    # 完整坐标转换：世界-相机-图像-像素,实际上最终要用的只有world_to_pixel_coordinate和pixel_to_world_coordinate
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