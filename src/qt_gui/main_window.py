from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGroupBox, QLabel, QTextEdit, QFrame
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cmd2Action Control Console")
        self.resize(1400, 800)

        self.init_ui()

    def init_ui(self):
        # 中央主控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 总体横向布局：左 中 右
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # ===== 左侧控制区 =====
        left_panel = QVBoxLayout()

        control_box = QGroupBox("控制区")
        control_layout = QVBoxLayout()
        control_layout.addWidget(QLabel("这里后面放：系统状态、手动控制、任务控制"))
        control_box.setLayout(control_layout)

        left_panel.addWidget(control_box)

        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setFixedWidth(300)

        # ===== 中间视觉区 =====
        center_panel = QVBoxLayout()

        vision_box = QGroupBox("视觉区")
        vision_layout = QVBoxLayout()

        vision_placeholder = QLabel("这里后面放相机画面")
        vision_placeholder.setAlignment(Qt.AlignCenter)
        vision_placeholder.setFrameShape(QFrame.Box)
        vision_placeholder.setMinimumHeight(400)

        vision_layout.addWidget(vision_placeholder)
        vision_box.setLayout(vision_layout)

        center_panel.addWidget(vision_box)

        center_widget = QWidget()
        center_widget.setLayout(center_panel)

        # ===== 右侧日志区 =====
        right_panel = QVBoxLayout()

        log_box = QGroupBox("日志区")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.append("系统启动成功")
        self.log_text.append("这里后面放任务日志、执行状态")

        log_layout.addWidget(self.log_text)
        log_box.setLayout(log_layout)

        right_panel.addWidget(log_box)

        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        right_widget.setFixedWidth(350)

        # 加入总布局
        main_layout.addWidget(left_widget)
        main_layout.addWidget(center_widget)
        main_layout.addWidget(right_widget)