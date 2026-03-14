from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QGroupBox, QLabel, QTextEdit, QFrame,
    QFormLayout, QDoubleSpinBox, QPushButton, QSizePolicy,
    QListWidget, QListWidgetItem, QAbstractItemView
)
from PyQt5.QtCore import QTimer
import rospy
import rosgraph
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from arm_vision.msg import DetectedObjectPool
from PyQt5.QtGui import QImage, QPixmap, QFont
import numpy as np
import time

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cmd2Action Control Console")
        self.resize(1400, 800)

        self.init_ui()
        self.bind_signals()

        self.target_list_lock_until = 0.0
        # YOLO 类别映射
        self.class_names = {
            0: "blue_box",
            1: "green_cylinder",
            2: "red_box",
            3: "yellow_cylinder",
        }
        # 先不init_node，否则roscore没运行的话gui也起不来
        # rospy.init_node("qt_gui", anonymous=True)
        
        # move_to安全范围校验
        self.x_min, self.x_max = 0.3, 1.0
        self.y_min, self.y_max = -0.8, 0.8
        self.z_min = 0.4

        self.update_move_to_button_state()
        # 启动状态监测
        self.start_ros_monitor()
        # 启动图像订阅
        self.start_image_subscriber()
        # 启动视频刷新定时器（Qt主线程）
        self.start_video_timer()
        
    # 绑定按钮信号
    def bind_signals(self):
        self.btn_move_to.clicked.connect(self.on_move_to_clicked)
        self.btn_open_gripper.clicked.connect(self.on_open_gripper_clicked)
        self.btn_close_gripper.clicked.connect(self.on_close_gripper_clicked)

        # 点击列表目标自动填入坐标
        self.target_list.itemPressed.connect(self.on_target_selected)
        # 根据填入的xyz判断能否点move_to按钮
        self.spin_x.valueChanged.connect(self.update_move_to_button_state)
        self.spin_y.valueChanged.connect(self.update_move_to_button_state)
        self.spin_z.valueChanged.connect(self.update_move_to_button_state)
    
    # ros状态监测定时器
    def start_ros_monitor(self):
        self.ros_timer = QTimer()
        self.ros_timer.timeout.connect(self.check_ros_master)
        self.ros_timer.start(1000)   # 每 1000ms = 1秒检测一次
    
    # 监测定时器每隔一段时间进行监测
    def check_ros_master(self):
        try:
            master = rosgraph.Master('/qt_gui')
            master.getPid()

            self.label_ros_status.setText("ROS Master: 已连接")
            self.label_ros_status.setStyleSheet("color: green")

            # 初始化视觉节点需要的ros
            self.try_start_image_subscriber()

            # 获取当前系统中的 topic 信息
            pubs, subs, srvs = master.getSystemState()

            published_topics = set(topic for topic, _ in pubs)
            subscribed_topics = set(topic for topic, _ in subs)
            all_topics = published_topics | subscribed_topics

            # 1. Gazebo / 机械臂状态：根据话题"/joint_states"来判断
            if "/joint_states" in all_topics:
                self.label_gazebo_status.setText("Gazebo: 已连接")
                self.label_gazebo_status.setStyleSheet("color: green")
            else:
                self.label_gazebo_status.setText("Gazebo: 未连接")
                self.label_gazebo_status.setStyleSheet("color: red")

            # 2. Vision Node：根据话题"/detected_objects"和"/detected_object_pool"来判断
            if "/detected_objects" in all_topics or "/detected_object_pool" in all_topics:
                self.label_vision_status.setText("Vision Node: 已连接")
                self.label_vision_status.setStyleSheet("color: green")
            else:
                self.label_vision_status.setText("Vision Node: 未连接")
                self.label_vision_status.setStyleSheet("color: red")

            # 3. Main Node：根据话题"/llm_commands"和"/llm_user_input"来判断
            if "/llm_commands" in all_topics or "/llm_user_input" in all_topics:
                self.label_main_status.setText("Main Node: 已连接")
                self.label_main_status.setStyleSheet("color: green")
            else:
                self.label_main_status.setText("Main Node: 未连接")
                self.label_main_status.setStyleSheet("color: red")

        except Exception:
            self.label_ros_status.setText("ROS Master: 未连接")
            self.label_ros_status.setStyleSheet("color: red")

            self.label_gazebo_status.setText("Gazebo: 未连接")
            self.label_gazebo_status.setStyleSheet("color: red")

            self.label_vision_status.setText("Vision Node: 未连接")
            self.label_vision_status.setStyleSheet("color: red")

            self.label_main_status.setText("Main Node: 未连接")
            self.label_main_status.setStyleSheet("color: red")

    def get_move_to_invalid_reason(self, x, y, z):
        if x < self.x_min or x > self.x_max:
            return f"x out of range [{self.x_min:.2f}, {self.x_max:.2f}]"

        if y < self.y_min or y > self.y_max:
            return f"y out of range [{self.y_min:.2f}, {self.y_max:.2f}]"

        if z <= self.z_min:
            return f"z too low (must > {self.z_min:.2f})"

        return None

    def update_move_to_button_state(self):
        x = self.spin_x.value()
        y = self.spin_y.value()
        z = self.spin_z.value()

        reason = self.get_move_to_invalid_reason(x, y, z)

        if reason is None:
            self.btn_move_to.setEnabled(True)
            self.btn_move_to.setText("Move To")
            self.btn_move_to.setToolTip("Move to the specified safe pose")
        else:
            self.btn_move_to.setEnabled(False)
            self.btn_move_to.setText("Move To")
            self.btn_move_to.setToolTip(reason)

    def start_video_timer(self):
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_video_frames)
        self.video_timer.start(33)   # 约 30 FPS
    
    def update_video_frames(self):
        try:
            if self.latest_global_rgb is not None:
                self.show_cv_image(self.label_global, self.latest_global_rgb)

            if self.latest_gripper_rgb is not None:
                self.show_cv_image(self.label_gripper_rgb, self.latest_gripper_rgb)

            if self.latest_gripper_depth is not None:
                self.show_gray_image(self.label_gripper_depth, self.latest_gripper_depth)

            self.update_target_list()
        except Exception as e:
            print(f"update_video_frames error: {e}")
    # move_to按钮被按下后执行的函数
    def on_move_to_clicked(self):
        x = self.spin_x.value()
        y = self.spin_y.value()
        z = self.spin_z.value()

        if self.move_to_pub is None:
            self.log_text.append("Move To failed: /gui/move_to_pose publisher not ready")
            return

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"

        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z

        # 先给一个默认朝向，后面再扩展 yaw / quaternion
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        self.move_to_pub.publish(msg)

        self.log_text.append(
            f"Move To published -> /gui/move_to_pose : ({x:.3f}, {y:.3f}, {z:.3f})"
        )

    # 夹爪控制按钮被按下后执行的函数
    def on_open_gripper_clicked(self):
        self.log_text.append("Open Gripper command sent")

    def on_close_gripper_clicked(self):
        self.log_text.append("Close Gripper command sent")

    # 图像话题订阅及取图
    def start_image_subscriber(self):
        # 回调函数内只缓存最新图像，组件更新图像放到定时器里做
        self.bridge = CvBridge()

        # 订阅句柄
        self.image_sub = None
        self.gripper_rgb_sub = None
        self.gripper_depth_sub = None
        self.detected_objects_sub = None

        # 发布器句柄
        self.move_to_pub = None

        # 最新帧缓存
        self.latest_global_rgb = None
        self.latest_gripper_rgb = None
        self.latest_gripper_depth = None
        self.latest_detected_objects = None

        self.image_started = False
        
    def try_start_image_subscriber(self):
        if self.image_started:
            return

        try:
            rospy.init_node("qt_gui", anonymous=True, disable_signals=True)

            self.image_sub = rospy.Subscriber(
                "/camera/color/image_annotated",
                Image,
                self.image_callback
            )

            self.gripper_rgb_sub = rospy.Subscriber(
                "/gripper_camera/color/image_raw",
                Image,
                self.gripper_rgb_callback
            )
            
            self.gripper_depth_sub = rospy.Subscriber(
                "/gripper_camera/depth/image_rect_raw",
                Image,
                self.gripper_depth_callback
            )

            self.detected_objects_sub = rospy.Subscriber(
                "/detected_objects",
                DetectedObjectPool,
                self.detected_objects_callback
            )

            self.move_to_pub = rospy.Publisher(
                "/gui/move_to_pose",
                PoseStamped,
                queue_size=10
            )

            self.image_started = True
            self.log_text.append("Image subscriber started")

        except Exception as e:
            self.log_text.append(f"Image subscriber start failed: {e}")

    # 抽象一个图像显示函数
    def show_cv_image(self, label, rgb_image):
        try:
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width

            qt_image = QImage(
                rgb_image.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_RGB888
            )

            pixmap = QPixmap.fromImage(qt_image)

            label.setPixmap(
                pixmap.scaled(
                    label.width(),
                    label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )
            label.setAlignment(Qt.AlignCenter)

        except Exception as e:
            print(f"show_cv_image error: {e}")
    
    # 用于显示深度相机灰度图
    def show_gray_image(self, label, gray_image):
        try:
            height, width = gray_image.shape
            bytes_per_line = width

            qt_image = QImage(
                gray_image.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_Grayscale8
            )

            pixmap = QPixmap.fromImage(qt_image)

            label.setPixmap(
                pixmap.scaled(
                    label.width(),
                    label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )
            label.setAlignment(Qt.AlignCenter)

        except Exception as e:
            print(f"show_gray_image error: {e}")

    # 回调函数内只缓存最新图像    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            # rgb_image = cv2.rotate(rgb_image, cv2.ROTATE_180)

            self.latest_global_rgb = rgb_image.copy()

        except Exception as e:
            print(f"image_callback error: {e}")

    def gripper_rgb_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            rgb_image = cv2.rotate(rgb_image, cv2.ROTATE_180)

            self.latest_gripper_rgb = rgb_image.copy()

        except Exception as e:
            print(f"gripper_rgb_callback error: {e}")

    def gripper_depth_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            depth_image = depth_image.copy()

            # 更稳一点：把非有限值清掉
            depth_image[~np.isfinite(depth_image)] = 0

            depth_norm = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_uint8 = depth_norm.astype("uint8")
            depth_uint8 = cv2.rotate(depth_uint8, cv2.ROTATE_180)

            self.latest_gripper_depth = depth_uint8.copy()

        except Exception as e:
            print(f"gripper_depth_callback error: {e}")

    def on_target_selected(self, item):
        try:
            # 点击后锁住列表刷新 0.8 秒，避免刚选中就被刷新掉
            self.target_list_lock_until = time.time() + 0.8

            pos = item.data(Qt.UserRole)
            if pos is None:
                return

            x, y, z = pos

            self.spin_x.setValue(x)
            self.spin_y.setValue(y)
            self.spin_z.setValue(z)

            self.update_move_to_button_state()
            self.log_text.append(
                f"Target selected → MoveTo ({x:.3f}, {y:.3f}, {z:.3f})"
            )

        except Exception as e:
            print(f"on_target_selected error: {e}")
    
    def update_target_list_height(self):
        self.target_list.setFixedHeight(180)
    
    def update_target_list(self):
        try:
            # 用户刚点击过目标时，暂时不刷新列表
            if time.time() < self.target_list_lock_until:
                return

            objs = self.latest_detected_objects

            if objs is None:
                return

            self.target_list.clear()

            if len(objs) == 0:
                self.target_list.addItem("暂无目标")
                self.update_target_list_height()
                return

            for i, obj in enumerate(objs):
                cls_id = obj.class_id
                conf = obj.confidence

                name = self.class_names.get(cls_id, f"class_{cls_id}")

                x = obj.pose.pose.position.x
                y = obj.pose.pose.position.y
                z = obj.pose.pose.position.z

                text = f"{i+1:<2} {name:<16} conf={conf:>4.2f}   ({x:>5.2f}, {y:>5.2f}, {z:>5.2f})"
                item = QListWidgetItem(text)

                # 保存世界坐标
                item.setData(Qt.UserRole, (x, y, z))

                self.target_list.addItem(item)

            self.update_target_list_height()

        except Exception as e:
            print(f"update_target_list error: {e}")
        
    def detected_objects_callback(self, msg):
        try:
            # 只缓存，不改组件
            self.latest_detected_objects = list(msg.objects)
        except Exception as e:
            print(f"detected_objects_callback error: {e}")

    def init_ui(self):
        # 主窗口里的主控件，占整片区域
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 对于刚刚创建的主控件，让其布局定义为横向排列
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # 建立左侧组件widget--定义layout--加组件control_box--定义layout--加组件label
        left_panel = QVBoxLayout()

        # --- 系统状态框 ---
        status_box = QGroupBox("系统状态")
        status_layout = QVBoxLayout()

        self.label_ros_status = QLabel("ROS Master: 未连接")
        self.label_vision_status = QLabel("Vision Node: 未连接")
        self.label_main_status = QLabel("Main Node: 未连接")
        self.label_gazebo_status = QLabel("Gazebo: 未连接")

        status_layout.addWidget(self.label_ros_status)
        status_layout.addWidget(self.label_vision_status)
        status_layout.addWidget(self.label_main_status)
        status_layout.addWidget(self.label_gazebo_status)

        status_box.setLayout(status_layout)

        # --- 手动控制框 ---
        manual_box = QGroupBox("手动控制")
        manual_layout = QVBoxLayout()

        # 表单布局：放 x y z
        form_layout = QFormLayout()

        self.spin_x = QDoubleSpinBox()
        self.spin_y = QDoubleSpinBox()
        self.spin_z = QDoubleSpinBox()

        # 设置取值范围
        self.spin_x.setRange(-1.0, 1.0)
        self.spin_y.setRange(-1.0, 1.0)
        self.spin_z.setRange(-1.0, 1.0)

        # 设置小数位数
        self.spin_x.setDecimals(3)
        self.spin_y.setDecimals(3)
        self.spin_z.setDecimals(3)

        # 设置步长
        self.spin_x.setSingleStep(0.01)
        self.spin_y.setSingleStep(0.01)
        self.spin_z.setSingleStep(0.01)

        # 设置默认值
        self.spin_x.setValue(0.20)
        self.spin_y.setValue(0.00)
        self.spin_z.setValue(0.10)

        form_layout.addRow("X:", self.spin_x)
        form_layout.addRow("Y:", self.spin_y)
        form_layout.addRow("Z:", self.spin_z)

        # move_to 按钮
        self.btn_move_to = QPushButton("Move To")

        # 夹爪按钮
        self.btn_open_gripper = QPushButton("Open Gripper")
        self.btn_close_gripper = QPushButton("Close Gripper")

        manual_layout.addLayout(form_layout)
        manual_layout.addWidget(self.btn_move_to)
        manual_layout.addWidget(self.btn_open_gripper)
        manual_layout.addWidget(self.btn_close_gripper)

        manual_box.setLayout(manual_layout)

        # 加入左侧总布局
        left_panel.addWidget(status_box)
        left_panel.addWidget(manual_box)
        # 弹性空白
        left_panel.addStretch()

        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setFixedWidth(500)

        # ===== 中间视觉区 跟上面类似 =====
        center_panel = QVBoxLayout()

        vision_box = QGroupBox("视觉区")

        vision_outer_layout = QVBoxLayout()

        # ===== 当前目标栏 =====
        target_box = QGroupBox("当前检测目标")

        target_layout = QVBoxLayout()

        self.target_list = QListWidget()
        self.target_list.setFont(QFont("Monospace", 11))
        self.target_list.addItem("暂无目标")

        # 只允许单选
        self.target_list.setSelectionMode(QAbstractItemView.SingleSelection)

        # 每行更清楚一些
        self.target_list.setSpacing(2)

        # 先给一个较小默认高度，后面会动态调整
        self.target_list.setFixedHeight(180)

        target_layout.addWidget(self.target_list)
        target_box.setLayout(target_layout)
        target_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        vision_layout = QGridLayout()
        vision_layout.setRowStretch(0, 1)
        vision_layout.setRowStretch(1, 1)
        vision_layout.setColumnStretch(0, 1)
        vision_layout.setColumnStretch(1, 1)

        # ---------- 观察摄像机 ----------
        self.label_world = QLabel("观察摄像机\n(预留)")
        self.label_world.setAlignment(Qt.AlignCenter)
        self.label_world.setFrameShape(QFrame.Box)

        # ---------- 全局摄像机 ----------
        self.label_global = QLabel("全局摄像机 RGB\n/camera/color/image_annotated")
        self.label_global.setAlignment(Qt.AlignCenter)
        self.label_global.setFrameShape(QFrame.Box)

        # ---------- 夹爪RGB ----------
        self.label_gripper_rgb = QLabel("Gripper RGB\n/gripper_camera/color/image_raw")
        self.label_gripper_rgb.setAlignment(Qt.AlignCenter)
        self.label_gripper_rgb.setFrameShape(QFrame.Box)

        # ---------- 夹爪Depth ----------
        self.label_gripper_depth = QLabel("Gripper Depth\n/gripper_camera/depth/image_rect_raw")
        self.label_gripper_depth.setAlignment(Qt.AlignCenter)
        self.label_gripper_depth.setFrameShape(QFrame.Box)

        for label in [
            self.label_world,
            self.label_global,
            self.label_gripper_rgb,
            self.label_gripper_depth
        ]:
            label.setMinimumSize(320, 240)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            
        # 添加到2x2网格
        vision_layout.addWidget(self.label_world, 0, 0)
        vision_layout.addWidget(self.label_global, 0, 1)
        vision_layout.addWidget(self.label_gripper_rgb, 1, 0)
        vision_layout.addWidget(self.label_gripper_depth, 1, 1)

        # 检测目标列表
        vision_outer_layout.addWidget(target_box, 0)
        vision_outer_layout.addLayout(vision_layout, 1)

        vision_box.setLayout(vision_outer_layout)

        center_panel.addWidget(vision_box)

        center_widget = QWidget()
        center_widget.setLayout(center_panel)

        # ===== 右侧日志区 跟上面类似 =====
        right_panel = QVBoxLayout()

        log_box = QGroupBox("日志区")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.append("系统启动成功")
        # self.log_text.append("这里后面放任务日志、执行状态")

        log_layout.addWidget(self.log_text)
        log_box.setLayout(log_layout)

        right_panel.addWidget(log_box)

        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        right_widget.setFixedWidth(500)

        # 加入总布局
        main_layout.addWidget(left_widget, 0)
        main_layout.addWidget(center_widget, 1)
        main_layout.addWidget(right_widget, 0)