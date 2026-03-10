# arm_vision/src/config.py

# ===== Global Camera / vision_node =====
# 全局视觉固定深度--用于vision_node
GLOBAL_FIXED_DEPTH = 1.25
# YOLO参数
VISION_CONF_THRESHOLD = 0.45
VISION_DEVICE = 'auto'

# 全局摄像头线性回归参数
WORLD_CORRECTION_KX = 1.025837
WORLD_CORRECTION_BX = -0.009908
WORLD_CORRECTION_KY = 1.025445
WORLD_CORRECTION_BY = 0.001953

# ===== Gripper Camera / gripper_vision =====
# gripper_vision--夹爪摄像机相关参数
# gripper 相机中心点
GRIPPER_CENTER_U = 424.5
GRIPPER_CENTER_V = 240.5

# gripper 相机相对参考平面的高度
GRIPPER_CAMERA_HEIGHT = 0.2954

# 视觉循环频率
VISION_RATE = 10

# 深度有效范围
MIN_VALID_DEPTH = 0.001
MAX_VALID_DEPTH = 1.0