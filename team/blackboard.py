"""
blackboard.py — the team's message bus.

A thread-safe publish/subscribe blackboard. Every robot and the planner publish
messages (status reports, task events, assignments); subscribers receive everything.
All traffic is appended to `history` so a mission leaves an auditable "team chat"
transcript — this is the literal record of the robots "talking to each other".

Message: {t, from, topic, type, data}
  t     — mission clock (seconds, set by publisher)
  from  — sender id ("go2" | "drone" | "panda" | "planner" | "mission")
  topic — "status" | "task" | "assignment" | "event"
  type  — free-form string, e.g. "status_report", "task_complete", "plan"
  data  — arbitrary JSON-serializable dict
"""

from __future__ import annotations

import json
import threading
from collections import deque
from typing import Any, Callable


class Blackboard:
    def __init__(self, history_limit: int = 10000):
        self._lock = threading.RLock()
        self._subs: list[Callable[[dict], None]] = []
        self.history: deque[dict] = deque(maxlen=history_limit)
        self.clock = 0.0

    # -- publishing -------------------------------------------------------
    def publish(self, sender: str, topic: str, type_: str, data: dict[str, Any] | None = None):
        msg = {"t": round(self.clock, 3), "from": sender, "topic": topic, "type": type_, "data": data or {}}
        with self._lock:
            self.history.append(msg)
            subs = list(self._subs)
        for cb in subs:
            try:
                cb(msg)
            except Exception as e:  # a broken subscriber must not kill the bus
                print(f"[blackboard] subscriber error: {e}")
        return msg

    def set_clock(self, t: float):
        with self._lock:
            self.clock = t

    # -- subscribing ------------------------------------------------------
    def subscribe(self, callback: Callable[[dict], None]):
        with self._lock:
            self._subs.append(callback)

    # -- queries ----------------------------------------------------------
    def last_from(self, sender: str, topic: str | None = None) -> dict | None:
        with self._lock:
            for msg in reversed(self.history):
                if msg["from"] == sender and (topic is None or msg["topic"] == topic):
                    return msg
        return None

    def transcript(self) -> str:
        """Human-readable 'team chat' — the robots' conversation log."""
        lines = []
        with self._lock:
            for m in self.history:
                data = json.dumps(m["data"], default=str) if m["data"] else ""
                lines.append(f"[{m['t']:7.2f}s] {m['from']:>8} ({m['topic']}/{m['type']}): {data}")
        return "\n".join(lines)

    def save_transcript(self, path: str | __import__("pathlib").Path):
        from pathlib import Path as _P

        _P(path).write_text(self.transcript(), encoding="utf-8")
