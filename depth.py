"""
depth.py

Monocular depth estimation using MiDaS (Intel ISL), loaded via torch.hub.

Wrapped in a small class (DepthEstimator), the same way ObjectDetector wraps
YOLOv8, so the model is loaded only ONCE and then reused for every frame.
"""

import cv2
import numpy as np
import torch


class DepthEstimator:
    def __init__(self, model_type="MiDaS_small"):
        """
        Load the MiDaS model and its matching input transform.

        MiDaS_small is the fastest variant and is the right choice for a
        real-time webcam loop. "DPT_Hybrid" / "DPT_Large" are more accurate
        but much slower.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.model.to(self.device).eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if model_type in ("DPT_Large", "DPT_Hybrid"):
            self.transform = transforms.dpt_transform
        else:
            self.transform = transforms.small_transform

    def estimate(self, frame_bgr):
        """
        Run depth estimation on one OpenCV (BGR) frame.

        Returns a float32 array with the same height/width as the frame.
        MiDaS output is RELATIVE inverse depth: larger value = closer.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        batch = self.transform(rgb).to(self.device)

        with torch.inference_mode():
            prediction = self.model(batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        return prediction.cpu().numpy()

    @staticmethod
    def colorize(depth):
        """Turns a raw depth map into an RGB image (bright = near, dark = far)."""
        d_min, d_max = float(depth.min()), float(depth.max())
        normalized = (depth - d_min) / (d_max - d_min + 1e-6)
        gray = (normalized * 255).astype(np.uint8)
        colored_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)