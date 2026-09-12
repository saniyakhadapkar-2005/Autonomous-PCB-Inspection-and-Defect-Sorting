from pathlib import Path
from ultralytics import YOLO
class PCBDetector:
    def __init__(self, model_path="yolo11n.pt"):
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        print("? YOLO model loaded")
    def detect(self, image_path, confidence=0.25):
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self.model.predict(
            source=str(image_path),
            conf=confidence,
            save=False,
            verbose=False
        )
    def get_detections(self, image_path, confidence=0.25):
        results = self.detect(image_path, confidence)
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence_score = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                class_name = result.names[class_id]
                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence_score, 4),
                    "bbox": [round(value, 2) for value in bbox]
                })
        return detections
