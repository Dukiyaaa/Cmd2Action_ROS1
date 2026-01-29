cmake_minimum_required(VERSION 3.0.2)
project(arm_application)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  message_generation
)

# 声明 .msg 文件
add_message_files(
  FILES
  LLMCommands.msg
)

# 必须在 generate_messages() 之前调用
catkin_python_setup()

# 生成消息/服务的 Python/C++ 接口
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

# 声明运行时依赖
catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs message_runtime
)

# 安装可执行的 Python 脚本（仅入口节点）
catkin_install_python(PROGRAMS 
  arm_application/nodes/main_node.py
  DESTINATION  ${CATKIN_PACKAGE_BIN_DESTINATION}
)