# arm_application/msg/LLMCommands.msg

# 支持的动作类型
string action_type    # "pick", "place", "pick_place", "reset", "open_gripper", "close_gripper"
                      # "create", "delete" 这两个命令用来生成测试的物体,专门用来调试
# === 抓取目标 ===
int32 object_class_id # 物体类别 ID,-1 表示无（如 reset）
string object_name    # 语义名称,仅用于日志/调试（如 "red_block"）
float32 object_x      # 显式抓取坐标（若提供）
float32 object_y
float32 object_z

# === 放置目标 ===
int32 target_class_id # 放置目标类别 ID,-1 表示无
string target_name    # 语义名称,仅用于日志/调试（如 "blue_tray"）
float32 target_x      # 显式放置坐标（若提供）
float32 target_y
float32 target_z

# === 使用规则 ===
# - "pick":        使用 object_class_id 或 (object_x, object_y, object_z)
# - "place":       使用 target_class_id 或 (target_x, target_y, target_z)
# - "pick_place":  两者都需
# - "create":  使用object_name object_class_id (object_x, object_y, object_z)
# - "delete":  使用object_name
# - 其他动作:      忽略所有目标字段