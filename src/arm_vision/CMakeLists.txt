cmake_minimum_required(VERSION 3.0.2)
project(arm_vision)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  image_transport
  message_generation
)

## 添加消息文件
add_message_files(
  FILES
  DetectedObject.msg
  DetectedObjectPool.msg
)

## 生成消息
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

# Python 脚本不需要编译，但需要设置权限
catkin_install_python(
  PROGRAMS 
  src/my_vision.py 
  src/object_data_collector.py 
  src/detector.py 
  src/coordinate_transformer.py 
  src/visualizer.py 
  src/vision_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
