"""
distance.py

Fuses MiDaS depth with YOLOv8 bounding boxes to classify each obstacle as
Near / Medium / Far, with smoothing so the readings don't flicker.

NOTE: MiDaS gives RELATIVE depth (not metres). Labels therefore mean
"how close compared to the rest of the scene", not a physical distance.
"""

import numpy as np

NEAR, MEDIUM, FAR = "Near", "Medium", "Far"
LEVELS = [FAR, MEDIUM, NEAR]  # index 0..2, higher index = closer


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class ObstacleDistanceEstimator:
    def __init__(
        self,
        near_threshold=0.65,    # smoothed score >= this -> Near
        medium_threshold=0.40,  # smoothed score >= this -> Medium, else Far
        ema_alpha=0.4,          # per-object smoothing (lower = smoother, slower)
        range_alpha=0.1,        # smoothing of the scene depth range
        hysteresis=0.05,        # margin needed to change label
        iou_threshold=0.3,      # overlap needed to treat boxes as the same object
        max_missed=5,           # frames to remember an object that disappeared
        center_fraction=0.5,
           # use the central 50% of each box
    ):
        self.near_threshold = near_threshold
        self.medium_threshold = medium_threshold
        self.ema_alpha = ema_alpha
        self.range_alpha = range_alpha
        self.hysteresis = hysteresis
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.center_fraction = center_fraction

        self._tracks = []
        self._range = None
        self._last_depth = None
        self._norm = None
        self.last_scores=[]

    # ------------------------------------------------------------------
    def update(self, depth_map, detections):
        """
        depth_map:  raw array from DepthEstimator.estimate(), or None.
        detections: list of dicts from ObjectDetector.detect().

        Returns a list of labels ("Near"/"Medium"/"Far"), one per detection,
        in the same order. Entries are None if no depth map is available.
        """
        if depth_map is None:
            self.last_scores = [None] * len(detections)
            return [None] * len(detections)

        self._prepare_depth(depth_map)
        det_to_track, used = self._match(detections)

        labels, scores, new_tracks = [], [], []
        for di, det in enumerate(detections):
            raw = self._box_score(det["box"])
            if di in det_to_track:
                tr = self._tracks[det_to_track[di]]
                tr["score"] = self.ema_alpha * raw + (1 - self.ema_alpha) * tr["score"]
                tr["level"] = self._classify(tr["score"], tr["level"])
                tr["box"] = det["box"]
                tr["missed"] = 0
            else:
                tr = {
                    "class_name": det["class_name"],
                    "box": det["box"],
                    "score": raw,
                    "level": self._classify(raw, None),
                    "missed": 0,
                }
                new_tracks.append(tr)
            labels.append(LEVELS[tr["level"]])
            scores.append(tr["score"])

        # Keep recently-seen objects that weren't matched this frame
        survivors = []
        for ti, tr in enumerate(self._tracks):
            if ti in used:
                survivors.append(tr)
            else:
                tr["missed"] += 1
                if tr["missed"] <= self.max_missed:
                    survivors.append(tr)
        self._tracks = survivors + new_tracks
        self.last_scores = scores
        return labels

    # ------------------------------------------------------------------
    def _prepare_depth(self, depth_map):
        # Only re-normalise when a NEW depth map arrives
        if depth_map is self._last_depth:
            return
        lo, hi = np.percentile(depth_map[::4, ::4], (2, 98))
        if self._range is None:
            self._range = (lo, hi)
        else:
            b = self.range_alpha
            self._range = ((1 - b) * self._range[0] + b * lo,
                           (1 - b) * self._range[1] + b * hi)
        lo, hi = self._range
        self._norm = np.clip((depth_map - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        self._last_depth = depth_map

    def _box_score(self, box):
        h, w = self._norm.shape
        x1, y1, x2, y2 = box
        m = (1 - self.center_fraction) / 2
        bw, bh = x2 - x1, y2 - y1
        cx1, cx2 = int(max(0, x1 + bw * m)), int(min(w, x2 - bw * m))
        cy1, cy2 = int(max(0, y1 + bh * m)), int(min(h, y2 - bh * m))
        if cx2 <= cx1 or cy2 <= cy1:  # tiny box: fall back to the whole box
            cx1, cx2 = int(max(0, x1)), int(min(w, x2))
            cy1, cy2 = int(max(0, y1)), int(min(h, y2))
        if cx2 <= cx1 or cy2 <= cy1:
            return 0.0
        return float(np.median(self._norm[cy1:cy2, cx1:cx2]))

    def _classify(self, score, prev_level):
        cuts = [self.medium_threshold, self.near_threshold]
        if prev_level is None:
            return sum(score >= c for c in cuts)
        level, h = prev_level, self.hysteresis
        while level < 2 and score >= cuts[level] + h:
            level += 1
        while level > 0 and score < cuts[level - 1] - h:
            level -= 1
        return level

    def _match(self, detections):
        pairs = []
        for di, det in enumerate(detections):
            for ti, tr in enumerate(self._tracks):
                if tr["class_name"] != det["class_name"]:
                    continue
                iou = _iou(det["box"], tr["box"])
                if iou >= self.iou_threshold:
                    pairs.append((iou, di, ti))
        pairs.sort(reverse=True)
        det_to_track, used = {}, set()
        for _, di, ti in pairs:
            if di in det_to_track or ti in used:
                continue
            det_to_track[di] = ti
            used.add(ti)
        return det_to_track, used