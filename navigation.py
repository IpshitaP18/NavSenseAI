"""
navigation.py

Turns what NavSense AI already knows (object positions, Near/Medium/Far depth
labels, priority ranking) into ONE walking instruction:

    Stop | Slow Down | Keep Left | Keep Right | Path Clear

How it decides
--------------
1. The frame is split into three lanes: LEFT | CENTER | RIGHT (equal thirds).
   Every obstacle is assigned to ALL lanes its (slightly widened) box touches,
   so a wide object blocks every lane it spans, not just the one holding its
   centre point. The widening (`clearance`) keeps the user from being steered
   to hug the edge of an obstacle.
2. Only Near and Medium obstacles BLOCK a lane. Far obstacles are ignored for
   steering.
3. CENTER lane (the user's walking path):
      - nothing blocking                      -> Path Clear
      - Medium obstacle                       -> Slow Down
      - Near obstacle, a side lane is free    -> Keep Left / Keep Right
      - Near obstacle, no free side           -> Stop
      - extremely close obstacle (>= stop_score) -> Stop (too late to steer)
4. A side is only ever suggested if that lane has NO blocking obstacle.
   If both sides are blocked the answer is Stop / Slow Down, never a guess.
5. Output is stabilised so it doesn't flicker: Stop is instant, escalations
   need a couple of frames, relaxations need several, and a "Keep X"
   instruction is dropped immediately if side X becomes blocked.

Limits (be honest in the demo): depth is MiDaS RELATIVE depth, and the camera
only sees its field of view, so "side is clear" means "clear in the camera
image", not "clear in the real world".
"""

from distance import NEAR, MEDIUM, FAR

STOP = "Stop"
SLOW_DOWN = "Slow Down"
KEEP_LEFT = "Keep Left"
KEEP_RIGHT = "Keep Right"
PATH_CLEAR = "Path Clear"

_LANES = ("LEFT", "CENTER", "RIGHT")
_LEVEL = {FAR: 0, MEDIUM: 1, NEAR: 2}
_SEVERITY = {PATH_CLEAR: 0, SLOW_DOWN: 1, KEEP_LEFT: 2, KEEP_RIGHT: 2, STOP: 3}
_SIDE_OF = {KEEP_LEFT: "LEFT", KEEP_RIGHT: "RIGHT"}
_KEEP_FOR = {"LEFT": KEEP_LEFT, "RIGHT": KEEP_RIGHT}


class NavigationAdvisor:
    def __init__(
        self,
        stop_score=0.85,         # centre obstacle with smoothed depth score >= this -> Stop
        clearance=0.10,          # widen each box by this fraction per side before lane test
        min_lane_overlap=0.10,   # fraction of a lane a box must cover to block it
        escalate_frames=2,       # frames a MORE severe instruction must persist
        settle_frames=5,         # frames a LESS severe / sideways change must persist
        near_score=0.65,         # only used when depth is missing (match distance.py)
        medium_score=0.40,       # only used when depth is missing (match distance.py)
        fallback_gain=2.6,       # size -> pseudo-proximity when depth is missing
    ):
        self.stop_score = stop_score
        self.clearance = clearance
        self.min_lane_overlap = min_lane_overlap
        self.escalate_frames = escalate_frames
        self.settle_frames = settle_frames
        self.near_score = near_score
        self.medium_score = medium_score
        self.fallback_gain = fallback_gain

        self._current = None     # last committed result
        self._pending = None     # instruction waiting to be committed
        self._count = 0
        self._last_side = None   # last steering side, used to break ties

    # ------------------------------------------------------------------
    def advise(self, detections, distances, scores, frame_width, frame_height,
               ranking=None):
        """
        detections: ObjectDetector.detect() output
        distances:  ObstacleDistanceEstimator.update() output ("Near"/... or None)
        scores:     ObstacleDistanceEstimator.last_scores (same order)
        ranking:    ObstaclePrioritizer.rank() output (optional; only used to
                    pick which obstacle is named in the explanation)

        Returns:
            {
              "instruction": str,
              "severity": int,                       # 0 clear .. 3 stop
              "reason": str,                         # short plain-language text
              "focus": {"class_name", "distance"} or None,
              "safe_sides": {"LEFT": bool, "RIGHT": bool},
              "depth_available": bool,
            }
        """
        obstacles, depth_ok = self._build_obstacles(
            detections, distances, scores, frame_width, frame_height
        )
        rank_of = {r["index"]: r["rank"] for r in (ranking or [])}
        raw = self._decide(obstacles, rank_of, depth_ok)
        return self._stabilise(raw)

    def reset(self):
        self._current, self._pending, self._count, self._last_side = None, None, 0, None

    # ------------------------------------------------------------------
    def _build_obstacles(self, detections, distances, scores, fw, fh):
        obstacles, depth_ok = [], True
        for i, det in enumerate(detections):
            label = distances[i] if i < len(distances) else None
            score = scores[i] if i < len(scores) else None

            if label is None or score is None:
                # No depth for this object: fall back to apparent size.
                depth_ok = False
                x1, y1, x2, y2 = det["box"]
                area = max(0, x2 - x1) * max(0, y2 - y1) / max(fw * fh, 1)
                score = min(1.0, area * self.fallback_gain)
                label = (NEAR if score >= self.near_score
                         else MEDIUM if score >= self.medium_score else FAR)

            obstacles.append({
                "index": i,
                "class_name": det["class_name"],
                "label": label,
                "level": _LEVEL[label],
                "score": score,
                "lanes": self._lanes_for_box(det["box"], fw),
            })
        return obstacles, depth_ok

    def _lanes_for_box(self, box, frame_width):
        x1, _, x2, _ = box
        pad = (x2 - x1) * self.clearance
        x1, x2 = x1 - pad, x2 + pad
        lane_w = frame_width / 3.0
        # A thin box fully inside a lane must still count, so cap the
        # requirement at half the (padded) box width.
        need = min(self.min_lane_overlap * lane_w, 0.5 * (x2 - x1))
        lanes = set()
        for i, name in enumerate(_LANES):
            lo, hi = i * lane_w, (i + 1) * lane_w
            if min(x2, hi) - max(x1, lo) >= need > 0:
                lanes.add(name)
        return lanes

    # ------------------------------------------------------------------
    def _decide(self, obstacles, rank_of, depth_ok):
        def in_lane(lane):
            return [o for o in obstacles if lane in o["lanes"]]

        def blockers(lane):
            return [o for o in in_lane(lane) if o["level"] >= _LEVEL[MEDIUM]]

        def cost(lane):
            return max((o["score"] for o in in_lane(lane)), default=0.0)

        safe = {side: not blockers(side) for side in ("LEFT", "RIGHT")}
        center = blockers("CENTER")
        suffix = "" if depth_ok else " (estimated from size, depth unavailable)"

        # ---- Path clear -------------------------------------------------
        if not center:
            reason = "Nothing close in your path."
            far = [o for o in in_lane("CENTER")]
            if far:
                reason = f"{self._name(self._focus(far, rank_of))} far ahead, none close."
            side_near = [
                (lane, o) for lane in ("LEFT", "RIGHT") for o in blockers(lane)
                if o["level"] == _LEVEL[NEAR]
            ]
            if side_near:
                lane, o = side_near[0]
                reason += f" {self._name(o)} close on your {lane.lower()}."
            return self._result(PATH_CLEAR, reason + suffix, None, safe, depth_ok)

        focus = self._focus(center, rank_of)
        worst = max(center, key=lambda o: (o["level"], o["score"]))
        side_note = self._side_note(safe)

        # ---- Medium obstacle ahead -------------------------------------
        if worst["level"] < _LEVEL[NEAR]:
            reason = f"{self._name(focus)} ahead ({focus['label'].lower()}). {side_note}"
            return self._result(SLOW_DOWN, reason + suffix, focus, safe, depth_ok)

        # ---- Near obstacle ahead ---------------------------------------
        if worst["score"] >= self.stop_score:
            reason = f"{self._name(focus)} very close ahead. {side_note}"
            return self._result(STOP, reason + suffix, focus, safe, depth_ok)

        free = [s for s in ("LEFT", "RIGHT") if safe[s]]
        if not free:
            reason = f"{self._name(focus)} close ahead and both sides are blocked."
            return self._result(STOP, reason + suffix, focus, safe, depth_ok)

        # Prefer the emptier side; on a tie keep the side already in use.
        side = min(free, key=lambda s: (cost(s), s != self._last_side))
        reason = f"{self._name(focus)} close ahead. {side_note}"
        return self._result(_KEEP_FOR[side], reason + suffix, focus, safe, depth_ok)

    @staticmethod
    def _focus(obstacles, rank_of):
        """Obstacle to name: best priority rank, else closest."""
        return min(obstacles, key=lambda o: (rank_of.get(o["index"], 999), -o["score"]))

    @staticmethod
    def _name(o):
        return o["class_name"].capitalize()

    @staticmethod
    def _side_note(safe):
        if safe["LEFT"] and safe["RIGHT"]:
            return "Left and right are clear."
        if safe["LEFT"]:
            return "Left is clear, right is blocked."
        if safe["RIGHT"]:
            return "Right is clear, left is blocked."
        return "Both sides are blocked."

    @staticmethod
    def _result(instruction, reason, focus, safe, depth_ok):
        return {
            "instruction": instruction,
            "severity": _SEVERITY[instruction],
            "reason": reason,
            "focus": ({"class_name": focus["class_name"], "distance": focus["label"]}
                      if focus else None),
            "safe_sides": dict(safe),
            "depth_available": depth_ok,
        }

    # ------------------------------------------------------------------
    def _stabilise(self, raw):
        cur = self._current
        if cur is None:
            return self._commit(raw)

        if raw["instruction"] == cur["instruction"]:
            self._pending, self._count = None, 0
            return self._commit(raw)  # same instruction: refresh the text

        if raw["instruction"] == self._pending:
            self._count += 1
        else:
            self._pending, self._count = raw["instruction"], 1

        needed = (self.escalate_frames
                  if raw["severity"] > cur["severity"] else self.settle_frames)

        # Safety overrides: never wait to say Stop, and never keep telling the
        # user to steer into a side that has just become blocked.
        side = _SIDE_OF.get(cur["instruction"])
        forced = (raw["instruction"] == STOP
                  or (side is not None and not raw["safe_sides"][side]))

        if forced or self._count >= needed:
            return self._commit(raw)
        return cur

    def _commit(self, result):
        self._current = result
        self._pending, self._count = None, 0
        side = _SIDE_OF.get(result["instruction"])
        if side:
            self._last_side = side
        return result