# arm_application/src/nodes/main_node.py
"""
验证重构的代码能否使用
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import sys
import os
import numpy as np

# 添加包路径（确保能 import arm_application）
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from controllers.scara_controller import ScaraController
from utils.gazebo_box_display import BoxSpawner
from utils.gazebo_cylinder_display import CylinderSpawner
from llm.llm import TongyiQianwenLLM
from agents.agent import Agent

def main():
    rospy.init_node('test_scara_controller', anonymous=True)
    rospy.loginfo("=== 启动 SCARA 控制器测试节点 ===")
    agent = Agent()
    llm = TongyiQianwenLLM()
    # box_spawner = BoxSpawner()
    # cylinder_spawner = CylinderSpawner()

    # # 生成方块名称
    # box_name = 'test_box'
    # # box_spawner.delete_entity(box_name)
    # box_x = 0.8
    # box_y = 0.3
    # box_z = 0.05
    # # 显示方块
    # success = box_spawner.display_test_box(
    #     box_pos=(box_x, box_y, box_z),
    #     box_name=box_name
    # )

    # # 生成圆柱名称
    # cylinder_name = 'test_cylinder'
    # # cylinder_spawner.delete_entity(cylinder_name)
    # cylinder_x = 0.8
    # cylinder_y = -0.3
    # cylinder_z = 0.05
    # # 显示方块
    # success = cylinder_spawner.display_test_cylinder(
    #     cyl_pos=(cylinder_x, cylinder_y, cylinder_z),
    #     cyl_name=cylinder_name
    # )
    try:
        # 实例化控制器
        # controller = ScaraController()
        # rospy.loginfo("控制器初始化成功")

        # # 测试 1: 复位
        # rospy.loginfo("\n-> 测试 1: 复位机械臂")
        # controller.reset(duration=2.0)

        # # 测试 2: 打开/关闭夹爪
        # rospy.loginfo("\n-> 测试 2: 夹爪开合")
        # controller.close_gripper(duration=1.0)
        # rospy.sleep(1.0)
        # controller.open_gripper(duration=1.0)
        # rospy.sleep(1.0)

        # # 测试 3: 移动到几个点
        # test_points = [
        #     (0.8, -0.8, 0.05),
        # ]

        # for i, (x, y, z) in enumerate(test_points, 1):
        #     rospy.loginfo(f"\n-> 测试 3.{i}: 移动到 ({x:.2f}, {y:.2f}, {z:.2f})")
        #     success = controller.move_to(x, y, z, duration=2.0)
        #     if success:
        #         rospy.loginfo("到达目标点")
        #         rospy.sleep(1.0)
        #     else:
        #         rospy.logwarn("目标点不可达")

        # # 测试 4: 对齐夹爪朝向（需先移动到某点再对齐）
        # rospy.loginfo("\n-> 测试 4: 对齐夹爪朝向")
        # # controller.move_to(0.6, 0.0, 0.1, duration=2.0)
        # controller.align_gripper_roll()
        # rospy.sleep(1.0)

        # # 最终复位
        # rospy.loginfo("\n-> 最终复位")
        # controller.reset(duration=2.0)

        # rospy.loginfo("\n所有测试完成！按 Ctrl+C 退出...")
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("程序被用户中断")
    except Exception as e:
        rospy.logerr(f"测试过程中发生错误: {e}")
        raise


if __name__ == '__main__':
    main()