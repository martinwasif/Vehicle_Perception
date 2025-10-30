import cv2
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path='yolov8s.pt'):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, stream=True)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "class_id": cls,
                    "confidence": conf,
                    "label": self.model.names[cls]
                })
        return detections
