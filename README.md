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

本项目支持多种运行模式，推荐按照以下顺序启动系统：

### 4.1 标准启动流程

1) 启动 Gazebo 仿真环境（包含机械臂模型和控制器）：

```bash
roslaunch arm_description gazebo.launch
```

2) 启动视觉系统（在新的终端）：

```bash
roslaunch arm_vision vision.launch
```

3) 启动主节点（在新的终端）：

```bash
rosrun arm_application main_node.py
```

主节点会初始化 Agent 和 LLM 实例，等待接收 LLM 指令并执行相应的抓取和放置任务。

### 4.2 传统演示模式

```bash
# 最小演示：生成一个方块,抓取并抬起,然后回零
rosrun arm_application simple_demo.py

# 随机抓取与放置循环：批量生成方块,执行 pick-and-place 并复位
rosrun arm_application my_scara_action.py
```

### 4.3 视觉系统单独运行

```bash
# 数据采集：生成随机物体场景并采集图像
rosrun arm_vision object_data_collector.py _output_dir:=/tmp/test_dataset _num_objects_min:=1 _num_objects_max:=3
```

### 4.4 MoveIt + Rviz 显示与交互（可选）

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
- `/llm_commands`（`arm_application/LLMCommands`）：LLM 发送的指令消息，包含动作类型和参数。

### 5.3 发布话题

- `/detected_objects`（`geometry_msgs/PoseStamped`）：视觉系统检测到的物体 3D 位姿。
- `/detected_object_pool`（`arm_vision/DetectedObjectPool`）：视觉系统检测到的物体池，包含多个物体的信息。

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

## 7. 核心模块说明

### 7.1 Agent 模块

- **路径**：`src/arm_application/arm_application/agents/agent.py`
- **功能**：
  - 接收 LLM 指令（`/llm_commands` 话题）
  - 解析指令类型和参数
  - 处理不同类型的动作请求：`pick`、`place`、`pick_place`、`reset`、`open_gripper`/`close_gripper`、`create`、`delete`
  - 与视觉系统交互获取物体位置
  - 调用 TaskPlanner 生成动作序列
  - 执行动作序列

### 7.2 TaskPlanner 模块

- **路径**：`src/arm_application/arm_application/planners/task_planner.py`
- **功能**：
  - 根据任务描述生成动作序列
  - 实现不同动作的规划逻辑：`_plan_pick`、`_plan_place`、`_plan_pick_place`、`_plan_reset`、`_plan_open_gripper`/`_plan_close_gripper`

### 7.3 ScaraController 模块

- **路径**：`src/arm_application/arm_application/controllers/scara_controller.py`
- **功能**：
  - 实现机械臂的底层控制
  - 发布关节控制指令
  - 订阅关节状态信息
  - 提供高层控制接口：`move_to`、`open_gripper`、`close_gripper`、`reset`、`align_gripper_roll`

### 7.4 ObjectDetector 模块

- **路径**：`src/arm_application/arm_application/agents/object_detector.py`
- **功能**：
  - 订阅视觉系统发布的物体检测结果
  - 提供物体位置查询接口
  - 支持根据物体类别 ID 获取物体位置

## 8. 物体夹取和放置逻辑

### 8.1 抓取逻辑（Pick）

**流程**：
1. **指令接收**：Agent 接收 LLM 发送的 `pick` 类型指令
2. **位置获取**：
   - 优先使用指令中提供的显式坐标 (`object_x`, `object_y`, `object_z`)
   - 若未提供显式坐标，则通过视觉系统根据 `object_class_id` 获取物体位置
3. **动作规划**：TaskPlanner 生成抓取动作序列
4. **动作执行**：Agent 执行动作序列

**抓取动作序列**：
```python
[
    ("move_to", x, y, SAFE_HEIGHT),  # 移动到目标上方安全高度
    ("align_gripper_roll",),          # 对齐夹爪朝向
    ("move_to", x, y, above),         # 移动到目标上方合适高度
    ("close_gripper",),               # 关闭夹爪
    ("move_to", x, y, SAFE_HEIGHT)    # 抬起夹爪到安全高度
]
```
其中：
- `SAFE_HEIGHT = 0.5`（夹爪初始高度）
- `above = z + DIV`，`DIV = 0.187`（夹爪合适的下降位置）

### 8.2 放置逻辑（Place）

**流程**：
1. **指令接收**：Agent 接收 LLM 发送的 `place` 类型指令
2. **位置获取**：
   - 优先使用指令中提供的显式坐标 (`target_x`, `target_y`, `target_z`)
   - 若未提供显式坐标，则通过视觉系统根据 `target_class_id` 获取放置目标位置
3. **动作规划**：TaskPlanner 生成放置动作序列
4. **动作执行**：Agent 执行动作序列

**放置动作序列**：
```python
[
    ("move_to", x, y, SAFE_HEIGHT),  # 移动到目标上方安全高度
    ("align_gripper_roll",),          # 对齐夹爪朝向
    ("move_to", x, y, above),         # 移动到目标上方合适高度
    ("open_gripper",),                # 打开夹爪
    ("move_to", x, y, SAFE_HEIGHT)    # 抬起夹爪到安全高度
]
```
其中：
- `SAFE_HEIGHT = 0.5`（夹爪初始高度）
- `above = z + DIV`，`DIV = 0.23`（夹爪合适的下降位置）

### 8.3 抓取并放置逻辑（Pick_Place）

**流程**：
1. **指令接收**：Agent 接收 LLM 发送的 `pick_place` 类型指令
2. **位置获取**：
   - 获取物体位置（同 Pick 流程）
   - 获取放置目标位置（同 Place 流程）
3. **动作规划**：TaskPlanner 组合 Pick 和 Place 的动作序列
4. **动作执行**：Agent 执行组合动作序列

### 8.4 底层控制实现

**夹爪控制**：
- **打开夹爪**：
  ```python
  self.finger1_pub.publish(Float64(-0.02))
  self.finger2_pub.publish(Float64(0.02))
  self.finger3_pub.publish(Float64(0.02))
  self.finger4_pub.publish(Float64(-0.02))
  ```
- **关闭夹爪**：
  ```python
  self.finger1_pub.publish(Float64(0.02))
  self.finger2_pub.publish(Float64(-0.02))
  self.finger3_pub.publish(Float64(-0.02))
  self.finger4_pub.publish(Float64(0.02))
  ```

**位置控制**：
- 使用逆运动学计算关节角度：
  ```python
  theta1, theta2, d3, reachable = inverse_kinematics(x, y, z, elbow="down")
  ```
- 发布关节控制指令：
  ```python
  self.rotation1_pub.publish(Float64(theta1))
  self.rotation2_pub.publish(Float64(theta2))
  self.gripper_pub.publish(Float64(d3))
  ```

## 9. 系统架构

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│    LLM 节点     │────>│    Agent 节点   │────>│ TaskPlanner     │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  视觉节点        │────>│ ObjectDetector │     │ ScaraController │────> 机械臂执行
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## 10. 仿真环境交互

项目支持在 Gazebo 仿真环境中创建和删除物体：
- **创建物体**：
  - `create` 动作类型
  - 支持创建立方体（`object_class_id=0`）和圆柱体（`object_class_id=1`）
  - 通过 `BoxSpawner` 和 `CylinderSpawner` 实现
- **删除物体**：
  - `delete` 动作类型
  - 通过物体名称删除指定物体

## 11. 技术特点

1. **分层架构**：清晰的分层设计，从指令解析到动作规划再到底层控制
2. **多模式位置获取**：支持显式坐标和视觉识别两种方式获取物体位置
3. **模块化设计**：各功能模块解耦，便于扩展和维护
4. **仿真支持**：集成 Gazebo 仿真环境，支持物体创建和删除
5. **逆运动学**：使用逆运动学计算关节角度，实现笛卡尔空间的位置控制
6. **夹爪朝向对齐**：在抓取和放置前对齐夹爪朝向，提高操作精度

## 12. 运动学说明（`my_kinematics.py`）

- 使用简化 DH 参数定义 SCARA：
    - 臂长：`a1=1.0`, `a2=0.8`
    - 竖向偏距：`d1=0.4`, `d2=0.1`
    - 竖向关节范围：`d3 ∈ [-0.5, 0.0]`
- 前向运动学：`forward_kinematics(theta1, theta2, d3)` 返回 4x4 变换矩阵。
- 逆运动学：`inverse_kinematics(x, y, z, elbow="down")` 返回 `(theta1, theta2, d3, reachable)`,含可达性判断与坐标系修正（代码中将目标平面坐标进行 `x, y = -y, x` 的旋转修正以匹配 URDF 初始朝向）。

该 IK 足以驱动示例任务,但未包含碰撞检测、速度/加速度限制与轨迹平滑；若需更复杂运动,请使用 MoveIt 的规划管线与 `FollowJointTrajectory` 控制器。

---

## 13. 视觉系统说明（`arm_vision`）

### 13.1 YOLOv8 物体检测

- 使用 Ultralytics YOLOv8 进行实时物体检测
- 支持 RGB+Depth 相机输入,输出 3D 物体位姿
- 默认模型：`best.pt`（可在启动时通过参数指定）
- 检测置信度阈值可配置（默认 0.45）
- 支持类别过滤（默认检测所有类别）

### 13.2 坐标变换

- `pixel_to_world(u, v, depth)`：将像素坐标 + 深度转换为相机坐标系下的 3D 坐标
- 自动处理深度单位转换（毫米→米）
- 集成相机内参校准信息
- 支持坐标系变换（相机→世界坐标系）

### 13.3 数据采集系统

- `object_data_collector.py`：自动生成随机物体场景并采集 RGB 图像
- 支持方块和圆柱体两种几何体
- 可配置物体数量范围（默认 1-3 个）
- 自动保存旋转校正后的图像（适配 URDF 相机朝向）
- 输出格式：`scene_XXXXXX.png`

## 14. 示例流程简介

### 14.1 `simple_demo.py`

1) 初始化 ROS 节点与控制器话题发布器；
2) 调用 Gazebo 服务生成一个小方块；
3) 通过 IK 计算抓取位姿,移动到目标,关闭夹爪,抬起方块；
4) 回到初始位置,打开夹爪；
5) 保持节点运行以便查看与调试。

### 14.2 `my_scara_action.py`

1) 初始化并打开夹爪；
2) 多轮随机测试：生成方块位置与放置位置,检查 IK 可达；
3) 执行 pick-and-place：下压抓取、抬起、移动到放置点、放下、抬起；
4) 复位并删除方块；
5) 轮次结束后继续下一轮,最终完成所有测试。

### 14.3 `main_node.py`

1) 初始化 ROS 节点；
2) 创建 Agent 实例（核心控制逻辑）；
3) 创建 TongyiQianwenLLM 实例（处理自然语言指令）；
4) 进入 ROS 主循环，等待 LLM 指令；
5) 接收并处理 LLM 指令，执行相应的抓取和放置任务。

---

## 15. 故障排查

- Gazebo 服务不可用：确认已启动 `gazebo.launch` 或 `demo_gazebo.launch`,并等待 `/gazebo/spawn_sdf_model` 与 `/gazebo/delete_model` 服务就绪。
- 控制器话题未发布或命名不一致：检查 `arm_description/config/controllers.yaml` 与对应 `launch` 是否加载了 position_controllers,且话题名称与脚本一致。
- IK 返回不可达：检查目标坐标是否在臂展范围内（`r ∈ [|a1-a2|, a1+a2]`）,并且 `z` 满足竖向关节范围；注意坐标系修正的影响。
- 夹爪接触不稳定或未抓牢：适当调整方块的接触参数、摩擦与质量；在应用脚本中增加闭合时序和等待时间。
- MoveIt 显示异常或规划失败：确认 URDF/Xacro 与 SRDF 的一致性,确保 `move_group` 正常启动。
- 视觉节点启动失败：检查 PyTorch 和 ultralytics 是否正确安装,确保 CUDA 版本兼容。
- YOLO 检测无结果：确认相机话题正常发布,检查模型路径和置信度阈值设置。
- 深度坐标转换异常：验证相机内参是否正确获取,检查深度值单位（应为米或毫米会自动转换）。
- 数据采集图像异常：注意 URDF 中相机固定关节的旋转会影响图像朝向,节点已自动处理 180°旋转校正。
- LLM 指令无响应：检查 `llm_commands` 话题是否正确发布,确认 Agent 节点是否正常运行。
- 物体位置获取失败：检查视觉系统是否正常运行,确认物体是否在相机视野范围内。

---

## 16. 扩展与改进建议

- 增加应用层 `launch`,一键启动 Gazebo + 控制器 + Demo 节点。
- 为 `my_kinematics.py` 增加单元测试,覆盖可达/不可达边界与数值稳定性。
- 迁移控制到 `FollowJointTrajectory` 接口,并使用 MoveIt 进行路径规划与执行,提升轨迹平滑与安全性。
- 完善 `README` 的控制器与关节名对照表,给出 URDF 关节映射到 controller 的说明。
- 增加 CI（例如基于 GitHub Actions 的语法检查与静态测试）。
- 扩展视觉系统：支持更多物体类别训练,增加实例分割功能。
- 实现视觉伺服控制：将视觉检测结果直接用于闭环抓取控制。
- 添加多相机支持和传感器融合功能。
- 集成强化学习：基于视觉反馈的智能抓取策略学习。
- 增加碰撞检测：在运动规划中加入碰撞检测,避免机械臂与环境或其他物体碰撞。
- 优化夹爪控制：根据物体大小和形状自动调整夹爪闭合程度,添加力反馈。
- 扩展 LLM 接口：支持更复杂的自然语言指令,增加任务规划能力。

---

## 17. 许可证与署名

- `arm_description` 的 `package.xml` 与 `arm_moveit_config` 声明了公共依赖与 BSD 许可；请根据实际需求完善 `license` 与作者信息。

---

## 18. 给大模型的快速理解 Prompt

将以下 Prompt 提供给其他大模型,以便快速理解本仓库并给出针对性的帮助或分析：

```
你正在阅读一个 ROS1 工程仓库（Ubuntu 环境运行）,仓库名为 Cmd2Action_ROS1。这是一个集成了计算机视觉和大语言模型的 SCARA 机械臂智能抓取系统,包含四个核心包：

- arm_description：提供 SCARA 机械臂的 URDF/Xacro、ros_control 控制器配置、Gazebo 与 Rviz 的 launch。
- arm_moveit_config：MoveIt 自动生成的配置与多种规划管线的 launch（OMPL/CHOMP/STOMP）,以及 Rviz/MoveGroup。
- arm_application：Python 应用节点（rospy）,包含 Agent、TaskPlanner、ScaraController 等核心模块,通过 position_controller 的话题控制关节、调用 Gazebo 服务生成/删除几何体（方块/圆柱体）,并包含简易的 FK/IK（my_kinematics.py）。
- arm_vision：计算机视觉系统,使用 YOLOv8 进行物体检测,包含数据采集和 3D 位姿估计。

关键接口：
- 机械臂控制话题（Float64）：/rotation1_position_controller/command, /rotation2_position_controller/command, /gripper_position_controller/command, /finger1_position_controller/command, /finger2_position_controller/command, /finger3_position_controller/command, /finger4_position_controller/command。
- 传感器话题：/joint_states（JointState）,/camera/color/image_raw（RGB）,/camera/depth/image_rect_raw（Depth）。
- 视觉输出话题：/detected_objects（PoseStamped,检测到的物体 3D 位姿）,/detected_object_pool（DetectedObjectPool,检测到的物体池）。
- LLM 指令话题：/llm_commands（LLMCommands,LLM 发送的指令消息）。
- Gazebo 服务：/gazebo/spawn_sdf_model 与 /gazebo/delete_model。

典型运行流程：
1) 启动 Gazebo 仿真（roslaunch arm_description gazebo.launch）。
2) 启动视觉系统（roslaunch arm_vision vision.launch）。
3) 启动主节点（rosrun arm_application main_node.py）,等待 LLM 指令并执行相应的抓取和放置任务。

核心功能：
- 物体抓取和放置：支持通过显式坐标或视觉识别获取物体位置,执行抓取和放置操作。
- 仿真环境交互：支持在 Gazebo 中创建和删除物体。
- 视觉伺服：通过 YOLOv8 进行物体检测,提供 3D 物体位姿。
- LLM 接口：接收和解析自然语言指令,执行相应的任务。

```

---

## 12. 参考与链接

- ROS Noetic 安装与使用：http://wiki.ros.org/noetic
- MoveIt 文档：https://moveit.ros.org/
- Gazebo 与 ROS 集成：https://classic.gazebosim.org/tutorials?tut=ros_overview
