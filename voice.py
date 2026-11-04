"""
voice.py

Speaks NavSense navigation instructions and obstacle warnings with pyttsx3.

Design
------
* pyttsx3 blocks while speaking, so all speech happens on ONE background
  worker thread that owns the engine. The camera loop never waits for audio.
* update() is called every frame. It decides WHAT to say and WHEN, applying
  per-message cooldowns so nothing is repeated constantly.
* Priorities: Stop (emergency) > Keep Left/Right > Slow Down / side-obstacle
  warning > Path Clear.
* Stop clears anything queued and interrupts whatever is being spoken
  (at the next word boundary). Other messages never interrupt anything; a
  newer pending message just replaces an older, less urgent one.
"""

import threading
import time
from collections import deque

try:
    import pyttsx3
except ImportError:          # app keeps working silently if it's missing
    pyttsx3 = None

P_STOP, P_STEER, P_WARN, P_INFO = 0, 1, 2, 3


class VoiceGuide:
    def __init__(
        self,
        rate=175,
        volume=1.0,
        stop_repeat=2.5,       # Stop keeps repeating this often while it persists
        steer_repeat=5.0,      # Keep Left / Keep Right repeat interval
        slow_repeat=6.0,       # Slow Down repeat interval
        clear_repeat=8.0,      # minimum gap between "Path clear" messages
        obstacle_repeat=8.0,   # per (object, side) warning interval
        min_gap=1.5,           # minimum gap between any two non-emergency repeats
    ):
        self.enabled = True
        self.available = pyttsx3 is not None
        self.rate, self.volume = rate, volume
        self.stop_repeat, self.steer_repeat = stop_repeat, steer_repeat
        self.slow_repeat, self.clear_repeat = slow_repeat, clear_repeat
        self.obstacle_repeat, self.min_gap = obstacle_repeat, min_gap

        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._interrupt = threading.Event()
        self._pending = None            # (priority, text)
        self._speaking = None           # priority currently being spoken
        self._last_spoken = {}          # cooldown key -> time
        self._last_any = 0.0
        self._last_instruction = None
        self.history = deque(maxlen=5)   # newest first: {"text", "priority", "time"}
        self.error = None

        if self.available:
            threading.Thread(target=self._worker, daemon=True).start()
    
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_enabled(self, on):
        self.enabled = bool(on)
        if not self.enabled:
            self.reset()

    def reset(self):
        """Forget history and silence anything queued (camera restart/off)."""
        with self._lock:
            self._pending = None
            if self._speaking is not None:
                self._interrupt.set()
        self._last_instruction = None
        self._last_spoken.clear()

    def snapshot(self):
        """Safe copy of voice state for the UI."""
        return {
            "enabled": self.enabled,
            "available": self.available,
            "error": self.error,
            "history": list(self.history),
        }
    
    def update(self, nav, rows=None):
        """
        nav:  NavigationAdvisor.advise() output
        rows: the per-object rows built in streamlit.py (sorted by rank)
        Call once per frame. Never blocks.
        """
        if not (self.enabled and self.available) or not nav:
            return

        now = time.monotonic()
        instr = nav["instruction"]
        focus = nav.get("focus")
        name = focus["class_name"] if focus else "Obstacle"
        prev = self._last_instruction
        changed = instr != prev
        self._last_instruction = instr
        spoke = False

        if instr == "Stop":
            gap = 1.0 if changed else self.stop_repeat
            if self._due("stop", now, gap):
                self._say(P_STOP, f"Stop. {name} very close ahead.", "stop", now)
            return  # nothing else may talk over an emergency

        if instr in ("Keep Left", "Keep Right"):
            if changed or self._due(instr, now, self.steer_repeat, gated=True):
                spoke = self._say(P_STEER, f"{name} ahead. {instr}.", instr, now)

        elif instr == "Slow Down":
            if changed or self._due(instr, now, self.slow_repeat, gated=True):
                spoke = self._say(P_WARN, f"Slow down. {name} ahead.", instr, now)

        elif instr == "Path Clear":
            # Only announce when it becomes clear after a warning, never on a loop.
            if changed and prev is not None and self._due("clear", now, self.clear_repeat):
                spoke = self._say(P_INFO, "Path clear.", "clear", now)

        # Side hazards the lane logic doesn't steer around (e.g. a person close
        # on your left while the path ahead is clear).
        if not spoke and rows:
            self._side_warning(rows, now)

    # ------------------------------------------------------------------
    # Decision helpers
    # ------------------------------------------------------------------
    def _side_warning(self, rows, now):
        top = rows[0]
        if top.get("rank") != 1 or top["direction"] == "CENTER":
            return
        if not (top.get("distance") == "Near" or top.get("risk_level") == "High"):
            return
        side = top["direction"].lower()
        key = f"obs:{top['class_name']}:{side}"
        if self._due(key, now, self.obstacle_repeat, gated=True):
            self._say(P_WARN, f"{top['class_name'].capitalize()} close on your {side}.", key, now)

    def _due(self, key, now, gap, gated=False):
        if now - self._last_spoken.get(key, -1e9) < gap:
            return False
        if gated and now - self._last_any < self.min_gap:
            return False
        return True

    def _say(self, priority, text, key, now):
        self._last_spoken[key] = now
        self._last_any = now
        with self._lock:
            if priority == P_STOP:
                self._pending = (priority, text)               # drop anything queued
                if self._speaking not in (None, P_STOP):
                    self._interrupt.set()                      # cut off lesser speech
            elif self._pending is None or priority <= self._pending[0]:
                self._pending = (priority, text)               # newest, most urgent wins
        self._wake.set()
        return True

    # ------------------------------------------------------------------
    # Worker thread (the only place the engine is touched)
    # ------------------------------------------------------------------
    def _on_word(self, name, location, length):
        if self._interrupt.is_set() and self._engine is not None:
            self._engine.stop()

    def _speak_once(self, text):
        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)
        engine.connect("started-word", self._on_word)
        self._engine = engine
        try:
            engine.say(text)
            engine.runAndWait()
        finally:
            self._engine = None
            del engine

    def _worker(self):
        # Windows: SAPI5 needs COM initialised on every thread that uses it.
        try:
            import comtypes
            comtypes.CoInitialize()
        except Exception:
            pass

        # Check once that a speech engine exists at all.
        try:
            probe = pyttsx3.init()
            del probe
        except Exception as e:
            self.available = False
            self.error = f"{type(e).__name__}: {e}"
            return

        while True:
            self._wake.wait()
            with self._lock:
                item, self._pending = self._pending, None
                self._wake.clear()
                if item:
                    self._interrupt.clear()
                    self._speaking = item[0]
            if not item:
                continue
            self.history.appendleft(
                {"text": item[1], "priority": item[0], "time": time.time()}
            )
            try:
                self._speak_once(item[1])
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
            finally:
                self._speaking = None
                self._interrupt.clear()