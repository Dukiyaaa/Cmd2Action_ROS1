#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent模块,接受llm的输出,查阅视觉数据,向控制模块发送命令数据
"""

import sys
import os

# 添加源代码目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import rospy
from arm_vision.msg import DetectedObjectPool
from geometry_msgs.msg import PoseStamped
from arm_application.srv import AgentCommands

def main():
    try:
        rospy.init_node('task_agent')
        rospy.wait_for_service('/execute_task')
        task_client = rospy.ServiceProxy('/execute_task', AgentCommands)

        # 构造目标位姿
        target = PoseStamped()
        target.header.frame_id = "world"
        target.pose.position.x = 0.5
        target.pose.position.y = 0.5
        target.pose.position.z = 0.05
        target.pose.orientation.w = 1.0

        # 发送请求
        resp = task_client(target_pose=target, action_type="grasp")

        if resp.success:
            rospy.loginfo("✅ Task succeeded!")
        else:
            rospy.logwarn(f"❌ Task failed: {resp.message}")
        
    except rospy.ROSInterruptException:
        rospy.loginfo("程序被用户中断")


if __name__ == '__main__':
    main()
