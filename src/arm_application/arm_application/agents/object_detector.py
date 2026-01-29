import rospy
from arm_vision.msg import DetectedObjectPool
from geometry_msgs.msg import PoseStamped

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