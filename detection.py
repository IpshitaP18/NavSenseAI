"""
detection.py

This module handles object detection using the YOLOv8 pretrained model
from the Ultralytics library.

We wrap the model inside a small class (ObjectDetector) so that:
- The model is loaded only ONCE when the program starts.
  (Loading it again for every single frame would be very slow.)
- app.py can simply call `detector.detect(frame)` and get back clean,
  easy-to-use results instead of dealing with YOLO's raw output format.
"""

from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.5):
        """
        Load the YOLOv8 model.

        model_path:
            Path/name of the pretrained model.
            "yolov8n.pt" = YOLOv8 'nano' model -> smallest and fastest version.
            It is not the most accurate YOLOv8 model, but it runs in real time
            on a normal laptop CPU, which is exactly what we need here.

        confidence_threshold:
            Minimum confidence score (0 to 1) required for a detection to be
            considered valid. Detections below this score are ignored.
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame):
        """
        Run YOLOv8 detection on a single frame (one image from the webcam).

        Returns a list of detections. Each detection is a dictionary:
            {
                "class_name": str,        # e.g. "person"
                "confidence": float,      # e.g. 0.87
                "box": (x1, y1, x2, y2)   # bounding box corners, in pixels
            }
        """
        # verbose=False stops YOLO from printing extra logs for every frame
        results = self.model(frame, verbose=False)[0]

        detections = []

        for box in results.boxes:
            confidence = float(box.conf[0])

            if confidence < self.confidence_threshold:
                continue  # skip low-confidence / unreliable detections

            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]

            # box.xyxy gives coordinates as [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            detections.append({
                "class_name": class_name,
                "confidence": confidence,
                "box": (x1, y1, x2, y2)
            })

        return detections
