# arm_vision/src/visualizer.py
"""
调试模块,可以在检测图上画框可视化
"""
import cv2

class Visualizer:
    def draw_detections(self, rgb_image, detections):
        """
        在图像上绘制检测框和标签
        detections: [(box, score, cls_id, u, v)]
        """
        for (box, score, cls_id, u, v) in detections:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class {cls_id}: {score:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(rgb_image, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), (0, 255, 0), -1)
            cv2.putText(rgb_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return rgb_image