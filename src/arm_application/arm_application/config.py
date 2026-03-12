# Cmd2Action/src/arm_application/arm_application/config.py

# scara_controller参数标准化
# ===== Controller Timing =====
CONTROLLER_INIT_WAIT = 1.0
JOINT_MOVE_DURATION = 1.5
MOVE_TO_DURATION = 3.0
OPEN_GRIPPER_DURATION = 0.5
CLOSE_GRIPPER_DURATION = 1.0
ALIGN_GRIPPER_DURATION = 1.0
RESET_DURATION = 3.0
GRIPPER_DOWN_DURATION = 1.0

# ===== Gripper Finger Positions =====
GRIPPER_OPEN_POS = (-0.02, 0.02, 0.02, -0.02)
GRIPPER_CLOSE_POS = (0.02, -0.02, -0.02, 0.02)

# ===== Reset Pose =====
RESET_THETA1 = 0.0
RESET_THETA2 = 0.0
RESET_D3 = 0.0
RESET_GRIPPER_ROLL = 0.0

# ===== Gripper Down =====
# 夹爪下降安全间距
GRIPPER_DOWN_SAFE_OFFSET = 0.206

# task_planner参数标准化
# ===== Planner Heights =====
PLANNER_SAFE_HEIGHT = 0.5
PICK_APPROACH_OFFSET = 0.188
PLACE_APPROACH_OFFSET = 0.20 + 0.05 - 0.016

# agent参数标准化
# ===== Agent Actions =====
ACTION_MOVE_TO = "move_to"
ACTION_PICK = "pick"
ACTION_PLACE = "place"
ACTION_PICK_PLACE = "pick_place"
ACTION_RESET = "reset"
ACTION_OPEN_GRIPPER = "open_gripper"
ACTION_CLOSE_GRIPPER = "close_gripper"
ACTION_CREATE = "create"
ACTION_DELETE = "delete"
ACTION_ALIGN_GRIPPER_ROLL = "align_gripper_roll"
ACTION_GRIPPER_DOWN = "gripper_down"

# ===== Agent Defaults =====
EMPTY_POSE = (-1.0, -1.0, -1.0)
INVALID_CLASS_ID = -1

# ===== Spawn Object Types =====
OBJECT_CLASS_BLUE_BOX = 0
OBJECT_CLASS_GREEN_CYLINDER = 1
OBJECT_CLASS_RED_BOX = 2
OBJECT_CLASS_YELLOW_CYLINDER = 3