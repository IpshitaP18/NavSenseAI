"""
priority.py

Ranks detected obstacles by collision risk using:
  - proximity  : smoothed depth score from distance.py (higher = closer)
  - centrality : how much the obstacle sits in the user's path (camera centre)

Risk = proximity * (center_floor + (1 - center_floor) * centrality)
"""

from distance import _iou  # reuse the IoU helper already used for tracking

RISK_HIGH, RISK_MEDIUM, RISK_LOW = "High", "Medium", "Low"


class ObstaclePrioritizer:
    def __init__(
        self,
        high_threshold=0.55,    # risk >= this -> High
        medium_threshold=0.30,  # risk >= this -> Medium, else Low
        center_floor=0.3,       # weight kept even for obstacles at the frame edge
        stickiness=0.08,        # margin a challenger needs to replace the current #1
        iou_threshold=0.3,      # overlap needed to treat boxes as the same obstacle
    ):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.center_floor = center_floor
        self.stickiness = stickiness
        self.iou_threshold = iou_threshold
        self._prev_top = None  # (class_name, box) of last frame's #1

    def rank(self, detections, scores, frame_width):
        """
        detections: list of dicts from ObjectDetector.detect()
        scores:     ObstacleDistanceEstimator.last_scores (same order)

        Returns a list sorted by risk (highest first). Each entry:
            {"index": i, "risk": float, "risk_level": str, "rank": 1-based}
        Obstacles without a depth score are skipped.
        """
        mid = frame_width / 2.0
        entries = []
        for i, (det, score) in enumerate(zip(detections, scores)):
            if score is None:
                continue
            proximity = min(max(score, 0.0), 1.0)

            x1, _, x2, _ = det["box"]
            if x1 <= mid <= x2:
                offset = 0.0  # box covers the centre line: directly in the path
            else:
                offset = min(abs(x1 - mid), abs(x2 - mid)) / mid
            centrality = max(0.0, 1.0 - offset)

            risk = proximity * (self.center_floor + (1 - self.center_floor) * centrality)
            entries.append({"index": i, "risk": risk})

        if not entries:
            self._prev_top = None
            return []

        entries.sort(key=lambda e: e["risk"], reverse=True)
        self._apply_stickiness(entries, detections)

        for rank, e in enumerate(entries, start=1):
            e["rank"] = rank
            if e["risk"] >= self.high_threshold:
                e["risk_level"] = RISK_HIGH
            elif e["risk"] >= self.medium_threshold:
                e["risk_level"] = RISK_MEDIUM
            else:
                e["risk_level"] = RISK_LOW

        top = detections[entries[0]["index"]]
        self._prev_top = (top["class_name"], top["box"])
        return entries

    def _apply_stickiness(self, entries, detections):
        """Keep last frame's #1 on top unless a challenger clearly beats it."""
        if self._prev_top is None:
            return
        prev_class, prev_box = self._prev_top
        for pos, e in enumerate(entries):
            det = detections[e["index"]]
            if det["class_name"] == prev_class and _iou(det["box"], prev_box) >= self.iou_threshold:
                if pos > 0 and entries[0]["risk"] - e["risk"] < self.stickiness:
                    entries.insert(0, entries.pop(pos))
                return