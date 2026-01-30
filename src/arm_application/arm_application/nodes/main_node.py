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
    agent = Agent()
    llm = TongyiQianwenLLM()

    try:
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("程序被用户中断")
    except Exception as e:
        rospy.logerr(f"测试过程中发生错误: {e}")
        raise


if __name__ == '__main__':
    main()