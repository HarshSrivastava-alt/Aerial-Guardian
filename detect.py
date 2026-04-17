from ultralytics import YOLO
from config import MODEL_PATH, IMG_SIZE, CONF_THRESHOLD

class Detector:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)

    def detect(self, frame):
        results = self.model(frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD)

        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:  # person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append([x1, y1, x2, y2])

        return detections
