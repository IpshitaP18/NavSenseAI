"""
direction.py

This module decides whether a detected object is to the LEFT, CENTER,
or RIGHT of the camera's view.

Logic:
1. Take the bounding box of the object: (x1, y1, x2, y2)
2. Calculate its center x-coordinate:
       center_x = (x1 + x2) / 2
3. Split the frame width into 3 equal parts (thirds):
       [ LEFT third | CENTER third | RIGHT third ]
4. Check which third `center_x` falls into, and return that as the direction.

This is intentionally simple - no depth, no angles, no camera calibration.
That keeps it beginner-friendly and matches exactly what this phase needs.
"""


def get_direction(box, frame_width):
    """
    box:
        (x1, y1, x2, y2) - bounding box coordinates of the detected object,
        as returned by detection.py.
    frame_width:
        Width of the camera frame in pixels (frame.shape[1] in OpenCV).

    Returns:
        "LEFT", "CENTER", or "RIGHT"
    """
    x1, _, x2, _ = box

    # Center x-coordinate of the bounding box
    center_x = (x1 + x2) / 2

    # Boundaries that split the frame into three equal parts
    left_boundary = frame_width / 3
    right_boundary = 2 * frame_width / 3

    if center_x < left_boundary:
        return "LEFT"
    elif center_x > right_boundary:
        return "RIGHT"
    else:
        return "CENTER"
