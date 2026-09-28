"""
app.py

Main entry point for the NavSense AI prototype (Phase 1).

Pipeline for this phase:
    Webcam (OpenCV) -> YOLOv8 (detection.py) -> Direction (direction.py) -> Display

Run this file to start the live webcam demo.
Press 'q' while the video window is focused to quit.
"""

import cv2

from detection import ObjectDetector
from direction import get_direction


def open_camera(camera_index=0):
    """
    Try to open the laptop webcam and handle common errors cleanly.

    camera_index=0 is almost always the built-in/default laptop webcam.
    cv2.CAP_DSHOW is used because it is the most reliable backend on Windows.
    """
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open the webcam. Please check that:\n"
            "  1. A webcam is physically connected / built-in and enabled.\n"
            "  2. No other application (Zoom, Teams, Camera app, etc.) is using it.\n"
            "  3. Windows camera privacy permissions allow desktop apps to use it."
        )

    return cap


def draw_detection(frame, class_name, confidence, direction, box):
    """
    Draws the bounding box and a text label for one detected object.

    Example label drawn on screen: "person 0.87 RIGHT"
    """
    x1, y1, x2, y2 = box

    # Draw the bounding box around the object
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Build the label text: "class_name confidence direction"
    label = f"{class_name} {confidence:.2f} {direction}"

    # Draw a filled background rectangle so the text is readable
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    label_y1 = max(y1 - text_h - 10, 0)  # keep label inside frame if box is near top
    cv2.rectangle(frame, (x1, label_y1), (x1 + text_w, y1), (0, 255, 0), -1)

    # Draw the label text itself
    cv2.putText(
        frame, label, (x1, y1 - 5 if y1 - 5 > 0 else y1 + text_h),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
    )


def main():
    print("Loading YOLOv8 model... (first run may take a little longer, "
          "since the model file gets downloaded automatically)")

    try:
        detector = ObjectDetector(model_path="yolov8n.pt", confidence_threshold=0.5)
    except Exception as e:
        print(f"Failed to load YOLOv8 model: {e}")
        return

    try:
        cap = open_camera(camera_index=0)
    except RuntimeError as e:
        print(f"Camera Error: {e}")
        return

    print("Webcam started successfully. Press 'q' in the video window to quit.")

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Warning: Failed to read a frame from the webcam. Retrying...")
            continue

        frame_width = frame.shape[1]

        # Run YOLOv8 detection on the current frame
        detections = detector.detect(frame)

        # For every detected object, work out its direction and draw it
        for det in detections:
            direction = get_direction(det["box"], frame_width)
            draw_detection(
                frame,
                det["class_name"],
                det["confidence"],
                direction,
                det["box"]
            )

        cv2.imshow("NavSense AI - Phase 1 Prototype", frame)

        # Wait 1ms for a key press; exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Always release the camera and close windows before exiting
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released. Program closed.")


if __name__ == "__main__":
    main()
