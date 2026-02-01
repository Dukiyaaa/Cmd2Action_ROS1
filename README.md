# Cmd2Action_ROS1

一个基于 ROS1 的 SCARA 机械臂智能抓取系统,集成了机械臂控制、计算机视觉和深度学习。包含机械臂描述（URDF/Xacro）、MoveIt 配置、Gazebo 仿真、YOLOv8 物体检测,以及 Python 应用节点（话题控制、物体生成、视觉伺服抓取、运动学求解、数据采集）。本项目在 Ubuntu 环境下运行,本文档面向 Ubuntu 用户详细说明环境准备、构建与运行方法、接口与配置、故障排查,以及给其他大模型理解本仓库的快速 Prompt。

---

## 1. 环境与依赖

- 操作系统：推荐 Ubuntu 20.04（ROS Noetic 官方支持）。
- 必需组件：
    - ROS Noetic（desktop-full 推荐,包含 Gazebo、Rviz、常用工具）
    - MoveIt（Noetic 对应版本）
    - Gazebo ROS 插件与 ros_control（`gazebo_ros`, `gazebo_ros_control`, `controller_manager`, `joint_state_controller`, `effort_controllers`, `position_controllers`）
    - Xacro（URDF 生成使用）
    - 计算机视觉组件：
        - `cv_bridge`（ROS-OpenCV 桥接）
        - `image_transport`（图像传输）
        - OpenCV Python (`python3-opencv`)
        - NumPy (`python3-numpy`)
    - 深度学习组件：
        - PyTorch（推荐 CUDA 版本以加速推理）
        - Ultralytics YOLOv8

### 1.1 安装 ROS Noetic 与 MoveIt（Ubuntu）

参考官方安装指南（简要示例）：

```bash
# 设置 ROS 软件源与密钥（略,参考 http://wiki.ros.org/noetic/Installation/Ubuntu ）
sudo apt update
sudo apt install -y ros-noetic-desktop-full

# 初始化 rosdep
sudo rosdep init || true
rosdep update

# 安装 MoveIt 及常用工具
sudo apt install -y ros-noetic-moveit ros-noetic-ros-control ros-noetic-ros-controllers \
    ros-noetic-gazebo-ros ros-noetic-gazebo-ros-control ros-noetic-joint-state-publisher \
    ros-noetic-joint-state-publisher-gui ros-noetic-robot-state-publisher ros-noetic-xacro \
    ros-noetic-cv-bridge ros-noetic-image-transport python3-opencv python3-numpy

# 安装深度学习依赖（推荐使用 pip）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
pip3 install ultralytics

# 配置环境（添加到 ~/.bashrc）
echo 'source /opt/ros/noetic/setup.bash' >> ~/.bashrc
source ~/.bashrc
```

---

## 2. 仓库结构与包说明

工程根目录：`Cmd2Action_ROS1/`

- `src/CMakeLists.txt`：顶层 catkin 工程入口,指向 Noetic 的 `toplevel.cmake`。
- `src/arm_description/`：机械臂描述与仿真显示。
    - `urdf/`：`scara.urdf.xacro`, `robot.ros_control.xacro` 等 Xacro/URDF 文件。
    - `config/controllers.yaml`：ros_control 控制器配置。
    - `launch/`：`display.launch`（Rviz 显示）、`gazebo.launch`（Gazebo 仿真）。
- `src/arm_moveit_config/`：MoveIt 自动生成配置。
    - `config/*.yaml`：规划器（OMPL/CHOMP/STOMP）、控制器、动力学、关节限速等。
    - `launch/`：`move_group.launch`, `moveit_rviz.launch`, `demo.launch`, `demo_gazebo.launch` 等。
- `src/arm_application/`：应用层 Python 节点。
    - `src/gazebo_box_display.py`：通过 Gazebo 服务生成/删除方块（SDF 字符串）。
    - `src/gazebo_cylinder_display.py`：通过 Gazebo 服务生成/删除圆柱体（SDF 字符串）。
    - `src/my_kinematics.py`：简易 SCARA DH 参数、FK/IK（含坐标系修正与可达判断）。
    - `src/simple_demo.py`：最小演示,生成方块并完成抓取、抬起、回零。
    - `src/my_scara_action.py`：随机抓取与放置循环,批量测试。
    - `CMakeLists.txt`：安装 Python 脚本为可执行,依赖 `rospy`, `std_msgs`, `sensor_msgs`, `geometry_msgs`, `gazebo_msgs`。
- `src/arm_vision/`：计算机视觉系统包。
    - `src/vision_node.py`：YOLOv8 物体检测节点,订阅 RGB+Depth 图像,发布检测到的物体 3D 位姿。
    - `src/my_vision.py`：视觉处理工具类,包含图像坐标到世界坐标的转换。
    - `src/object_data_collector.py`：数据集采集节点,在 Gazebo 中随机生成物体并采集 RGB 图像。
    - `model/best.pt`：训练好的 YOLOv8 模型权重文件。
    - `launch/vision.launch`：视觉系统启动文件。
    - `test_img/`：测试图像目录。
- `arm_vision_dataset/`：视觉训练数据集。
    - `images/`：包含数千张 Gazebo 仿真场景的 RGB 图像,用于 YOLOv8 模型训练。

---

## 3. 构建与安装

建议将仓库置于 catkin 工作空间的 `src/` 下。

```bash
# 创建工作空间（如无）
mkdir -p ~/ws_cmd2action/src
cd ~/ws_cmd2action/src

# 克隆仓库
git clone https://github.com/Dukiyaaa/Cmd2Action_ROS1.git

# 返回工作空间根目录并构建
cd ~/ws_cmd2action
catkin_make

# 载入工作空间环境
source devel/setup.bash
```

构建成功后,`arm_application` 中的 Python 脚本会安装为可运行的 ROS 节点,便于 `rosrun` 调用。

---

## 4. 运行方式

本项目支持三种典型模式：

### 4.1 Gazebo 仿真 + 应用节点（推荐）

1) 启动 Gazebo 仿真与控制器（任选其一）：

```bash
# 选项 A：使用描述包的 Gazebo 启动
roslaunch arm_description gazebo.launch

# 或 选项 B：使用 MoveIt 的 Gazebo 演示
roslaunch arm_moveit_config demo_gazebo.launch
```

2) 启动演示节点（在新的终端载入工作空间环境后）：

```bash
# 最小演示：生成一个方块,抓取并抬起,然后回零
rosrun arm_application simple_demo.py

# 随机抓取与放置循环：批量生成方块,执行 pick-and-place 并复位
rosrun arm_application my_scara_action.py
```

### 4.2 视觉伺服抓取系统

1) 启动完整的仿真环境（Gazebo + 视觉系统）：

```bash
# 启动 Gazebo 仿真（包含深度相机）
roslaunch arm_description gazebo.launch

# 在新终端启动视觉检测节点
roslaunch arm_vision vision.launch
```

2) 运行视觉辅助的抓取任务：

```bash
# 数据采集：生成随机物体场景并采集图像
rosrun arm_vision object_data_collector.py _output_dir:=/tmp/test_dataset _num_objects_min:=1 _num_objects_max:=3
```

### 4.3 MoveIt + Rviz 显示与交互（可选）

```bash
# 仅显示 URDF/Xacro 模型
roslaunch arm_description display.launch

# MoveIt + Rviz 交互（可用来查看规划、联动状态）
roslaunch arm_moveit_config demo.launch
```

注意：确保控制器话题名称与脚本中一致,例如：

- `/rotation1_position_controller/command`
- `/rotation2_position_controller/command`
- `/gripper_position_controller/command`
- `/finger1_position_controller/command`
- `/finger2_position_controller/command`
- `/finger3_position_controller/command`
- `/finger4_position_controller/command`

这些话题由 ros_control 的 position_controllers 提供,加载时机与命名由 `controllers.yaml` 和相应 launch 决定。

### 4.2 MoveIt + Rviz 显示与交互（可选）

```bash
# 仅显示 URDF/Xacro 模型
roslaunch arm_description display.launch

# MoveIt + Rviz 交互（可用来查看规划、联动状态）
roslaunch arm_moveit_config demo.launch
```

---

## 5. 话题与服务接口

### 5.1 应用节点输出话题（Float64）

- 手臂关节：
    - `/rotation1_position_controller/command`
    - `/rotation2_position_controller/command`
    - `/gripper_position_controller/command`
- 夹爪四指：
    - `/finger1_position_controller/command`
    - `/finger2_position_controller/command`
    - `/finger3_position_controller/command`
    - `/finger4_position_controller/command`

### 5.2 消费话题

- `/joint_states`（`sensor_msgs/JointState`）：用于读取当前关节状态（在 Gazebo 与 ros_control 正常运行时由控制器发布）。
- `/camera/color/image_raw`（`sensor_msgs/Image`）：RGB 图像流（由 Gazebo 仿真相机发布）。
- `/camera/depth/image_rect_raw`（`sensor_msgs/Image`）：深度图像流（由 Gazebo 仿真相机发布）。
- `/camera/color/camera_info`（`sensor_msgs/CameraInfo`）：相机内参信息。

### 5.3 发布话题

- `/detected_objects`（`geometry_msgs/PoseStamped`）：视觉系统检测到的物体 3D 位姿。

### 5.4 Gazebo 服务

- `/gazebo/spawn_sdf_model`（`gazebo_msgs/SpawnModel`）：生成 SDF 模型。
- `/gazebo/delete_model`（`gazebo_msgs/DeleteModel`）：删除模型。

`gazebo_box_display.py` 和 `gazebo_cylinder_display.py` 使用上述服务生成具有惯性、材质与接触参数的几何体,并支持指定位置、尺寸、颜色与质量。

---

## 6. 运动学说明（`my_kinematics.py`）

- 使用简化 DH 参数定义 SCARA：
    - 臂长：`a1=1.0`, `a2=0.8`
    - 竖向偏距：`d1=0.4`, `d2=0.1`
    - 竖向关节范围：`d3 ∈ [-0.5, 0.0]`
- 前向运动学：`forward_kinematics(theta1, theta2, d3)` 返回 4x4 变换矩阵。
- 逆运动学：`inverse_kinematics(x, y, z, elbow="down")` 返回 `(theta1, theta2, d3, reachable)`,含可达性判断与坐标系修正（代码中将目标平面坐标进行 `x, y = -y, x` 的旋转修正以匹配 URDF 初始朝向）。

该 IK 足以驱动示例任务,但未包含碰撞检测、速度/加速度限制与轨迹平滑；若需更复杂运动,请使用 MoveIt 的规划管线与 `FollowJointTrajectory` 控制器。

---

## 7. 视觉系统说明（`arm_vision`）

### 7.1 YOLOv8 物体检测

- 使用 Ultralytics YOLOv8 进行实时物体检测
- 支持 RGB+Depth 相机输入,输出 3D 物体位姿
- 默认模型：`best.pt`（可在启动时通过参数指定）
- 检测置信度阈值可配置（默认 0.45）
- 支持类别过滤（默认检测所有类别）

### 7.2 坐标变换

- `pixel_to_world(u, v, depth)`：将像素坐标 + 深度转换为相机坐标系下的 3D 坐标
- 自动处理深度单位转换（毫米→米）
- 集成相机内参校准信息
- 支持坐标系变换（相机→世界坐标系）

### 7.3 数据采集系统

- `object_data_collector.py`：自动生成随机物体场景并采集 RGB 图像
- 支持方块和圆柱体两种几何体
- 可配置物体数量范围（默认 1-3 个）
- 自动保存旋转校正后的图像（适配 URDF 相机朝向）
- 输出格式：`scene_XXXXXX.png`

---

## 7. 示例流程简介

### 7.1 `simple_demo.py`

1) 初始化 ROS 节点与控制器话题发布器；
2) 调用 Gazebo 服务生成一个小方块；
3) 通过 IK 计算抓取位姿,移动到目标,关闭夹爪,抬起方块；
4) 回到初始位置,打开夹爪；
5) 保持节点运行以便查看与调试。

### 7.2 `my_scara_action.py`

1) 初始化并打开夹爪；
2) 多轮随机测试：生成方块位置与放置位置,检查 IK 可达；
3) 执行 pick-and-place：下压抓取、抬起、移动到放置点、放下、抬起；
4) 复位并删除方块；
5) 轮次结束后继续下一轮,最终完成所有测试。

---

## 8. 故障排查

- Gazebo 服务不可用：确认已启动 `gazebo.launch` 或 `demo_gazebo.launch`,并等待 `/gazebo/spawn_sdf_model` 与 `/gazebo/delete_model` 服务就绪。
- 控制器话题未发布或命名不一致：检查 `arm_description/config/controllers.yaml` 与对应 `launch` 是否加载了 position_controllers,且话题名称与脚本一致。
- IK 返回不可达：检查目标坐标是否在臂展范围内（`r ∈ [|a1-a2|, a1+a2]`）,并且 `z` 满足竖向关节范围；注意坐标系修正的影响。
- 夹爪接触不稳定或未抓牢：适当调整方块的接触参数、摩擦与质量；在应用脚本中增加闭合时序和等待时间。
- MoveIt 显示异常或规划失败：确认 URDF/Xacro 与 SRDF 的一致性,确保 `move_group` 正常启动。
- 视觉节点启动失败：检查 PyTorch 和 ultralytics 是否正确安装,确保 CUDA 版本兼容。
- YOLO 检测无结果：确认相机话题正常发布,检查模型路径和置信度阈值设置。
- 深度坐标转换异常：验证相机内参是否正确获取,检查深度值单位（应为米或毫米会自动转换）。
- 数据采集图像异常：注意 URDF 中相机固定关节的旋转会影响图像朝向,节点已自动处理 180°旋转校正。

---

## 9. 扩展与改进建议

- 增加应用层 `launch`,一键启动 Gazebo + 控制器 + Demo 节点。
- 为 `my_kinematics.py` 增加单元测试,覆盖可达/不可达边界与数值稳定性。
- 迁移控制到 `FollowJointTrajectory` 接口,并使用 MoveIt 进行路径规划与执行,提升轨迹平滑与安全性。
- 完善 `README` 的控制器与关节名对照表,给出 URDF 关节映射到 controller 的说明。
- 增加 CI（例如基于 GitHub Actions 的语法检查与静态测试）。
- 扩展视觉系统：支持更多物体类别训练,增加实例分割功能。
- 实现视觉伺服控制：将视觉检测结果直接用于闭环抓取控制。
- 添加多相机支持和传感器融合功能。
- 集成强化学习：基于视觉反馈的智能抓取策略学习。

---

## 10. 许可证与署名

- `arm_description` 的 `package.xml` 与 `arm_moveit_config` 声明了公共依赖与 BSD 许可；请根据实际需求完善 `license` 与作者信息。

---

## 11. 给大模型的快速理解 Prompt

将以下 Prompt 提供给其他大模型,以便快速理解本仓库并给出针对性的帮助或分析：

```
你正在阅读一个 ROS1 工程仓库（Ubuntu 环境运行）,仓库名为 Cmd2Action_ROS1。这是一个集成了计算机视觉的 SCARA 机械臂智能抓取系统,包含四个核心包：

- arm_description：提供 SCARA 机械臂的 URDF/Xacro、ros_control 控制器配置、Gazebo 与 Rviz 的 launch。
- arm_moveit_config：MoveIt 自动生成的配置与多种规划管线的 launch（OMPL/CHOMP/STOMP）,以及 Rviz/MoveGroup。
- arm_application：Python 应用节点（rospy）,通过 position_controller 的话题控制关节、调用 Gazebo 服务生成/删除几何体（方块/圆柱体）,并包含简易的 FK/IK（my_kinematics.py）。
- arm_vision：计算机视觉系统,使用 YOLOv8 进行物体检测,包含数据采集和 3D 位姿估计。

关键接口：
- 机械臂控制话题（Float64）：/rotation1_position_controller/command, /rotation2_position_controller/command, /gripper_position_controller/command, /finger1_position_controller/command, /finger2_position_controller/command, /finger3_position_controller/command, /finger4_position_controller/command。
- 传感器话题：/joint_states（JointState）,/camera/color/image_raw（RGB）,/camera/depth/image_rect_raw（Depth）。
- 视觉输出话题：/detected_objects（PoseStamped,检测到的物体 3D 位姿）。
- Gazebo 服务：/gazebo/spawn_sdf_model 与 /gazebo/delete_model。

典型运行流程：
1) 启动 Gazebo 仿真（roslaunch arm_description gazebo.launch）。
2) 可选启动视觉系统（roslaunch arm_vision vision.launch）。
3) 启动应用节点进行抓取任务（rosrun arm_application simple_demo.py）或数据采集（rosrun arm_vision object_data_collector.py）。

```

---

## 12. 参考与链接

- ROS Noetic 安装与使用：http://wiki.ros.org/noetic
- MoveIt 文档：https://moveit.ros.org/
- Gazebo 与 ROS 集成：https://classic.gazebosim.org/tutorials?tut=ros_overview
