# arm_vision/src/detector.py
"""
YOLO检测模块,在这个类里做模型初始化、推理,相应的参数由配置文件提供
"""
from ultralytics import YOLO
import torch

class YOLODetector:
    def __init__(self, model_path, device='auto', conf_thres=0.45):
        # 1. 先保存参数到实例变量
        self.model_path = model_path
        self.conf_thres = conf_thres

        # 2. 处理设备选择
        if device in (None, '', 'auto'):
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device_param

        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        self.conf_thres = conf_thres

    def detect(self, rgb_image):
        """
        输入 RGB 图像,返回检测结果列表：[box, score, class_id, center_u, center_v]
        仅返回uv,深度还需要进一步匹配
        """
        results = self.model(rgb_image, conf=self.conf_thres, verbose=False, device=self.device)
        # 有几张图片,result就有几项
        result = results[0]
        
        if result is None or result.boxes is None:
            return []
        
        # Ultrayltics 的result对象自动带boxes这个属性 这个属性是监测框,内含xyxy,conf,cls
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        detections = []
        for box, score, cls_id in zip(xyxy, scores, classes):
            if score < self.conf_thres:
                continue
            center_u = int((box[0] + box[2]) / 2)
            center_v = int((box[1] + box[3]) / 2)
            detections.append((box, score, cls_id, center_u, center_v))
        
        return detections