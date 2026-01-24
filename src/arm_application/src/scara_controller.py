# src/scara_controller.py
#!/usr/bin/env python3

import sys
import os

# 添加源代码目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import rospy
from arm_application.srv import AgentCommands, AgentCommandsResponse
from geometry_msgs.msg import PoseStamped
from my_scara_action import ArmController

arm_controller = None
def handle_task(req):
    rospy.loginfo(f"收到任务: action='{req.action_type}', pos=({req.target_pose.pose.position.x:.2f}, {req.target_pose.pose.position.y:.2f}, {req.target_pose.pose.position.z:.2f})")
    target_x = req.target_pose.pose.position.x
    target_y = req.target_pose.pose.position.y
    target_z = req.target_pose.pose.position.z
    arm_controller.pick_and_place([target_x, target_y, target_z], [0.0, -1.8, 0.05 + 0.032])
    return AgentCommandsResponse(success=True, message="Mock success")

if __name__ == '__main__':
    rospy.init_node('scara_controller')
    arm_controller = ArmController()
    s = rospy.Service('/execute_task', AgentCommands, handle_task)
    rospy.loginfo("控制器已启动，等待任务...")
    rospy.spin()