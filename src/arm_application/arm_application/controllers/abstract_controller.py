# arm_application/src/controllers/abstract_controller.py
"""
抽象基类,scara_controller由此继承,后续如果需要拓展为UR5的话也由此继承
机械臂的原子操作 如夹爪操作 臂操作
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional

class AbstractController(ABC):
    """
    所有机械臂控制器的抽象基类。
    定义了统一的接口,确保不同机械臂可以被同一套规划器调用。
    """
    @abstractmethod
    def move_to(self, x: float, y: float, z: float, duration: float = 1.0) -> bool:
        """
        移动末端执行器到指定世界坐标 (x, y, z)
        返回: 是否成功到达
        """
        pass

    @abstractmethod
    def open_gripper(self, duration: float = 1.0) -> None:
        """打开夹爪"""
        pass

    @abstractmethod
    def close_gripper(self, duration: float = 1.0) -> None:
        """关闭夹爪"""
        pass

    @abstractmethod
    def reset(self, duration: float = 3.0) -> None:
        """复位到初始姿态"""
        pass

    # 机械臂对齐朝向
    @abstractmethod
    def align_gripper_roll(self) -> None:
        """对齐工具（如夹爪）朝向"""
        pass