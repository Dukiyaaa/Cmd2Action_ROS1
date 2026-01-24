# LLMCommands.msg
string action_type    # "pick", "place", "pick_place"
int32 target_class_id # -1 表示无目标（如 home）
string target_name    # 可选："red_block"（用于日志/调试）