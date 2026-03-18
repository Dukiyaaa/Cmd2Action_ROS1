import rospy
from arm_vision.msg import DetectedObjectPool
from geometry_msgs.msg import PoseStamped


class ObjectDetector:
    def __init__(self):
        # {class_id (int): [obj_info, obj_info, ...]}
        self.detected_objects = {}
        self.sub = rospy.Subscriber('/detected_objects', DetectedObjectPool, self._callback)
        rospy.loginfo('[ObjectDetector] ready')

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
            rospy.logdebug(
                f"[ObjectDetector] nearest object selected: "
                f"class_id={class_id}, position={best_obj['position']}, "
                f"confidence={best_obj['confidence']:.3f}"
            )
            return best_obj["position"]

        elif strategy == "highest_confidence":
            best_obj = max(objs, key=lambda obj: obj["confidence"])
            rospy.logdebug(
                f"[ObjectDetector] highest-confidence object selected: "
                f"class_id={class_id}, position={best_obj['position']}, "
                f"confidence={best_obj['confidence']:.3f}"
            )
            return best_obj["position"]

        rospy.logwarn(f"[ObjectDetector] unknown strategy: {strategy}")
        return None
    
    def exists_near_position(self, class_id, target_pos, tolerance=0.05):
        """
        判断某类物体是否仍然存在于目标位置附近

        Args:
            class_id (int): 物体类别
            target_pos (tuple): 目标位置 (x, y, z)
            tolerance (float): 允许的距离阈值（米）

        Returns:
            bool: True 表示该位置附近仍存在该类物体
        """

        objs = self.get_objects(class_id)
        if not objs:
            return False

        tx, ty, tz = target_pos
        tol_sq = tolerance * tolerance

        for obj in objs:
            x, y, z = obj["position"]

            dist_sq = (x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2
            if dist_sq < tol_sq:
                rospy.loginfo(f"x,y,z:{x,y,z},dist_sq:{dist_sq}")
                return True

        return False

    def exists_any_near_position(self, target_pos, tolerance=0.05):
        """
        判断任意类别物体是否仍然存在于目标位置附近

        Args:
            target_pos (tuple): 目标位置 (x, y, z)
            tolerance (float): 允许的距离阈值（米）

        Returns:
            bool: True 表示该位置附近仍存在某个物体
        """

        if not self.detected_objects:
            return False

        tx, ty, tz = target_pos
        tol_sq = tolerance * tolerance

        for class_id, objs in self.detected_objects.items():
            for obj in objs:
                x, y, z = obj["position"]
                dist_sq = (x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2

                if dist_sq < tol_sq:
                    rospy.loginfo(
                        f"[ObjectDetector] found object near target: "
                        f"class_id={class_id}, position={(x, y, z)}, dist_sq={dist_sq}"
                    )
                    return True

        return False

    # 根据坐标反查可能的id
    def infer_class_id_from_pose(self, pose, tolerance=0.05):
        """
        根据目标坐标，在当前检测结果中查找最近物体的 class_id

        Args:
            pose (tuple): 目标位置 (x, y, z)
            tolerance (float): 最大允许匹配距离（米）

        Returns:
            int | None: 匹配到的 class_id；如果没有足够近的目标则返回 None
        """
        if not self.detected_objects:
            return None

        px, py, pz = pose
        tol_sq = tolerance * tolerance

        best_dist_sq = float("inf")
        best_class_id = None

        for class_id, objs in self.detected_objects.items():
            for obj in objs:
                ox, oy, oz = obj["position"]
                dist_sq = (px - ox) ** 2 + (py - oy) ** 2 + (pz - oz) ** 2

                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_class_id = class_id

        if best_dist_sq <= tol_sq:
            rospy.loginfo(
                f"[ObjectDetector] inferred class_id={best_class_id} "
                f"for pose={pose}, dist_sq={best_dist_sq}"
            )
            return best_class_id

        rospy.logwarn(
            f"[ObjectDetector] no nearby object found for pose={pose}, "
            f"best_dist_sq={best_dist_sq}"
        )
        return None