cmake_minimum_required(VERSION 3.0.2)
project(arm_vision)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  image_transport
)

catkin_package()

# Python 脚本不需要编译，但需要设置权限
catkin_install_python(PROGRAMS src/vision_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
