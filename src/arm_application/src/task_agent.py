#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent 模块：接收 LLM 指令，查询视觉数据，执行 pick-and-place 任务
"""

import rospy
from arm_vision.msg import DetectedObjectPool
from geometry_msgs.msg import PoseStamped
from arm_application.srv import AgentCommands
from arm_application.msg import LLMCommands
import sys
import os

# 添加源代码目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

class ObjectDetector:
    def __init__(self):
        self.detected_objects = {}  # {class_id (int): (x, y, z)}
        self.sub = rospy.Subscriber('/detected_objects', DetectedObjectPool, self._callback)
        rospy.loginfo('[ObjectDetector] 初始化完成，等待视觉数据...')

    def _callback(self, msg):
        self.detected_objects.clear()
        for obj in msg.objects:
            self.detected_objects[obj.class_id] = (
                obj.pose.pose.position.x,
                obj.pose.pose.position.y,
                obj.pose.pose.position.z
            )

    def get_position(self, class_id):
        return self.detected_objects.get(class_id)

class TaskAgent:
    def __init__(self):
        # 初始化视觉检测器（单例）
        self.detector = ObjectDetector()
        
        # 等待控制器服务
        rospy.loginfo("等待 /execute_task 服务...")
        rospy.wait_for_service('/execute_task')
        self.task_client = rospy.ServiceProxy('/execute_task', AgentCommands)
        
        # 订阅 LLM 指令
        self.sub = rospy.Subscriber('/llm_commands', LLMCommands, self._llm_callback)
        rospy.loginfo("Agent 已启动，等待 LLM 指令...")

    def _llm_callback(self, msg):
        rospy.loginfo("收到 LLM 指令: action='%s', class_id=%d" % (msg.action_type, msg.target_class_id))
        
        # 仅处理 pick_place 类型（可根据需要扩展）
        if msg.action_type != "pick_place":
            rospy.logwarn("不支持的动作类型: %s" % msg.action_type)
            return

        # 查询目标物体位置
        pos = self.detector.get_position(msg.target_class_id)
        if pos is None:
            rospy.logerr("未检测到 class_id=%d 的物体！" % msg.target_class_id)
            return

        x, y, z = pos
        rospy.loginfo("目标位置: (%.3f, %.3f, %.3f)" % (x, y, z))

        # 构造 pick 和 place 位姿
        pick_pose = self._create_pose(x, y, z)
        
        # 安全放置位置
        place_pose = self._create_pose(0.0, -1.8, 0.05)

        # 发送任务请求
        try:
            resp = self.task_client(
                target_pose=[pick_pose, place_pose],
                action_type="pick_place"
            )
            if resp.success:
                rospy.loginfo("Pick-and-place 任务成功！")
            else:
                rospy.logerr("任务失败: %s" % resp.message)
        except rospy.ServiceException as e:
            rospy.logerr("服务调用异常: %s" % str(e))

    def _create_pose(self, x, y, z, frame_id="world"):
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0
        return pose


def main():
    rospy.init_node('task_agent')
    agent = TaskAgent()
    rospy.spin()  # 保持节点运行，监听 LLM 指令


if __name__ == '__main__':
    main()