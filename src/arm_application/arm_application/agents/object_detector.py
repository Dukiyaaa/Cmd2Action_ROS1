import rospy
from arm_vision.msg import DetectedObjectPool
from geometry_msgs.msg import PoseStamped


class ObjectDetector:
    def __init__(self):
        # {class_id (int): [obj_info, obj_info, ...]}
        self.detected_objects = {}
        self.sub = rospy.Subscriber('/detected_objects', DetectedObjectPool, self._callback)
        rospy.loginfo('[ObjectDetector] 初始化完成,等待视觉数据...')

    def _callback(self, msg):
        self.detected_objects.clear()

        for obj in msg.objects:
            class_id = obj.class_id
            obj_info = {
                "position": (
                    obj.pose.pose.position.x,
                    obj.pose.pose.position.y,
                    obj.pose.pose.position.z
                ),
                "confidence": obj.confidence,
                "pose": obj.pose
            }

            if class_id not in self.detected_objects:
                self.detected_objects[class_id] = []

            self.detected_objects[class_id].append(obj_info)
    
    def get_objects(self, class_id):
        return self.detected_objects.get(class_id, [])
    
    def get_positions(self, class_id):
        objs = self.get_objects(class_id)
        return [obj["position"] for obj in objs]
    
    def get_best_position(self, class_id, strategy="nearest", ref_point=(0.0, 0.0, 0.0)):
        objs = self.get_objects(class_id)
        if not objs:
            return None

        if strategy == "nearest":
            def distance_sq(obj):
                x, y, z = obj["position"]
                rx, ry, rz = ref_point
                return (x - rx) ** 2 + (y - ry) ** 2 + (z - rz) ** 2

            best_obj = min(objs, key=distance_sq)
            rospy.logwarn(f"[ObjectDetector] 最近obj: {best_obj}")
            return best_obj["position"]

        elif strategy == "highest_confidence":
            best_obj = max(objs, key=lambda obj: obj["confidence"])
            return best_obj["position"]

        rospy.logwarn(f"[ObjectDetector] 未知策略: {strategy}")
        return None