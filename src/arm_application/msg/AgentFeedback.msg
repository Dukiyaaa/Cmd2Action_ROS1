# 用于agent执行一次大任务后,向llm反馈

string session_id       # 任务id
int32 step_id           # 步骤id

string action_type

bool success

string error_code
string message

bool retry_exhausted    # 是否重试

string object_name
int32 object_class_id

float32 object_x
float32 object_y
float32 object_z

bool done