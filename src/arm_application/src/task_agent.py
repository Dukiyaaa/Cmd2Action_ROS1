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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from arm_vision.msg import DetectedObjectPool


class ObjectDetector:
    """
    物体检测订阅器
    订阅 /detected_objects (DetectedObjectPool)，
    提取所有物体的 class_id 和对应的世界坐标 (x, y, z)
    """
    def __init__(self):
        """初始化检测器"""
        self.detected_objects = {}  # {class_id: (x, y, z)}
        self.is_processing = False  # 防止回调重入
        
        # 订阅检测结果话题
        self.sub = rospy.Subscriber('/detected_objects', DetectedObjectPool, self._detect_callback)
        
        rospy.loginfo('[ObjectDetector] 已初始化，等待物体检测数据...')

    def _detect_callback(self, msg):
        """处理 DetectedObjectPool 消息"""
        if self.is_processing:
            return

        # 清空上一帧的结果
        self.detected_objects.clear()

        # 遍历所有检测到的物体
        for obj in msg.objects:
            class_id = obj.class_id
            x = obj.pose.pose.position.x
            y = obj.pose.pose.position.y
            z = obj.pose.pose.position.z

            # 保存到字典（相同 class_id 会被覆盖，保留最后一个）
            self.detected_objects[class_id] = (x, y, z)

        # 可选：打印检测摘要（避免刷屏，可用 logdebug）
        # if self.detected_objects:
        #     ids = list(self.detected_objects.keys())
        #     rospy.loginfo(f"[ObjectDetector] 检测到 {len(ids)} 个物体 | class_ids: {ids}")

    def get_object_position(self, class_id):
        """
        根据 class_id 获取物体位置
        :param class_id: int, 物体类别ID
        :return: tuple (x, y, z) 或 None（未找到）
        """
        return self.detected_objects.get(class_id, None)

    def get_all_objects(self):
        """
        获取当前所有检测到的物体 {class_id: (x, y, z)}
        :return: dict 的副本
        """
        return self.detected_objects.copy()

    def has_object(self, class_id):
        """判断是否检测到指定 class_id 的物体"""
        return class_id in self.detected_objects


def main():
    try:
        rospy.init_node('task_agent')
        rospy.wait_for_service('/execute_task')
        task_client = rospy.ServiceProxy('/execute_task', AgentCommands)

        detector = ObjectDetector()
        try:
            rospy.wait_for_message('/detected_objects', DetectedObjectPool, timeout=5.0)
            rospy.loginfo("收到视觉数据")
        except rospy.ROSException:
            rospy.logerr("未收到检测数据，请检查视觉节点！")
            return

        target_class_id = 0
        position = detector.get_object_position(target_class_id)

        if position is not None:
            x, y, z = position
            rospy.loginfo(f"找到目标物体 (class_id={target_class_id}) 位置: ({x:.3f}, {y:.3f}, {z:.3f})")
            # 构造目标位姿
            pick_target = PoseStamped()
            pick_target.header.frame_id = "world"
            pick_target.pose.position.x = x
            pick_target.pose.position.y = y
            pick_target.pose.position.z = z
            pick_target.pose.orientation.w = 1.0

            place_target = PoseStamped()
            place_target.header.frame_id = "world"
            place_target.pose.position.x = 0.0
            place_target.pose.position.y = -1.8
            place_target.pose.position.z = 0.05
            place_target.pose.orientation.w = 1.0

            # 发送请求
            resp = task_client(target_pose=[pick_target, place_target], action_type="pick_place")

            if resp.success:
                rospy.loginfo("Task succeeded!")
            else:
                rospy.logwarn(f"Task failed: {resp.message}")
        else:
            rospy.logwarn(f"❌ 未检测到 class_id={target_class_id} 的物体")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("程序被用户中断")


if __name__ == '__main__':
    main()
