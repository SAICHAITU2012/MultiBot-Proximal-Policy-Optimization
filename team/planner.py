"""
planner.py — the team's mission commander.

Planner is the interface: given (world_state, task_board) it returns assignments.
Two implementations:

* RuleBasedPlanner — deterministic, zero-dependency. Allocation rule:
    for each open task, pick the idle robot whose capabilities match and whose
    travel distance to the task is smallest. It logs its reasoning so the
    "team discussion" is auditable.
* LLMPlanner — sends the world state + task board as JSON to an LLM (local
  ollama server or any OpenAI-compatible endpoint), parses the returned
  assignments. NOT activated until the user chooses local-vs-API (Stage 6
  decision point); the HTTP plumbing is here so switching is one line.

Assignments: {robot_id: task_id}
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Protocol

# Robot capability table (mission-level; matches what each policy can actually do)
ROBOT_CAPABILITIES = {
    "go2": {"recon_ground", "push", "fetch_ground"},
    "drone": {"recon_air", "survey"},
    "panda": {"pick", "sort"},
}


class Planner(Protocol):
    name: str

    def plan(self, world_state: dict[str, Any], tasks: list[dict[str, Any]]) -> dict[str, str]:
        """Return {robot_id: task_id} for currently-assignable tasks."""
        ...


class RuleBasedPlanner:
    """Greedy nearest-capable allocator. Reliable, explainable, no dependencies."""

    name = "rule-based"

    def plan(self, world_state: dict, tasks: list[dict]) -> dict[str, str]:
        robots = world_state["robots"]
        assignments: dict[str, str] = {}
        busy = {r for r, st in robots.items() if st.get("task_id") is not None}
        open_tasks = [t for t in tasks if t["status"] == "open"]
        # highest priority first, then FIFO
        open_tasks.sort(key=lambda t: (-t.get("priority", 0), t.get("created_at", 0)))
        for task in open_tasks:
            need = task["required_capability"]
            candidates = [
                (rid, _dist(robots[rid]["pos"], task["location"]))
                for rid, st in robots.items()
                if rid not in busy and need in ROBOT_CAPABILITIES.get(rid, set())
            ]
            if not candidates:
                continue
            rid, d = min(candidates, key=lambda c: c[1])
            assignments[rid] = task["id"]
            busy.add(rid)
            task["assigned_to"] = rid  # caller persists this
            task["status"] = "assigned"
        return assignments


def _dist(a, b) -> float:
    """Euclidean distance over the shared leading dims (works for 2-D or 3-D)."""
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


class LLMPlanner:
    """
    LLM mission commander. Talks to any OpenAI-compatible /api/chat endpoint:
      - local ollama:  base_url="http://127.0.0.1:11434/v1"  (e.g. model="qwen2.5:7b")
      - hosted API:    base_url="https://api...", api_key="..." (user provides)
    The prompt contains the live world state + task board; the model must answer
    with strict JSON {robot_id: task_id}. We validate against capabilities and
    fall back to the rule-based planner for anything it gets wrong.
    """

    name = "llm"

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 temperature: float = 0.2, fallback: Planner | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.fallback = fallback or RuleBasedPlanner()
        self.last_reasoning: str = ""

    def plan(self, world_state: dict, tasks: list[dict]) -> dict[str, str]:
        prompt = self._build_prompt(world_state, tasks)
        try:
            content = self._chat(prompt)
        except Exception as e:
            self.last_reasoning = f"LLM call failed ({e}); falling back to rules."
            return self.fallback.plan(world_state, tasks)
        self.last_reasoning = content
        try:
            assignments = self._parse(content, world_state, tasks)
        except Exception:
            assignments = {}
        # validate: capabilities + busyness; fix via fallback for the rest
        valid = {
            rid: tid
            for rid, tid in assignments.items()
            if rid in world_state["robots"]
            and world_state["robots"][rid].get("task_id") is None
            and self._capable(rid, tasks, tid)
        }
        covered = set(valid.values())
        remaining = {
            t["id"]: t for t in tasks
            if t["status"] == "open" and t["id"] not in covered
        }
        if remaining:
            sub = self.fallback.plan(world_state, list(remaining.values()))
            for rid, tid in sub.items():
                valid.setdefault(rid, tid)
        return valid

    # -- internals ---------------------------------------------------------
    def _capable(self, rid: str, tasks: list[dict], tid: str) -> bool:
        for t in tasks:
            if t["id"] == tid:
                return t["required_capability"] in ROBOT_CAPABILITIES.get(rid, set())
        return False

    def _build_prompt(self, world_state: dict, tasks: list[dict]) -> str:
        return (
            "You are the mission commander of a heterogeneous robot team.\n"
            "Robots (id, capabilities, position, current task):\n"
            f"{json.dumps(self._robot_view(world_state), indent=1)}\n"
            "Open tasks (id, type, required capability, location, priority):\n"
            f"{json.dumps([{k: t[k] for k in ('id', 'type', 'required_capability', 'location', 'priority')} for t in tasks if t['status'] == 'open'], indent=1)}\n"
            "Assign every open task you can to the best idle robot (match capability, prefer shorter travel;"
            " one task per robot). Answer ONLY with a JSON object mapping robot_id to task_id, no prose.\n"
            "JSON:"
        )

    @staticmethod
    def _robot_view(world_state: dict) -> list[dict]:
        view = []
        for rid, st in world_state["robots"].items():
            view.append({
                "id": rid,
                "capabilities": sorted(ROBOT_CAPABILITIES.get(rid, set())),
                "position": [round(x, 3) for x in st["pos"]],
                "current_task": st.get("task_id"),
            })
        return view

    def _chat(self, prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]

    @staticmethod
    def _parse(content: str, world_state: dict, tasks: list[dict]) -> dict[str, str]:
        # extract the first JSON object from the reply
        start = content.index("{")
        end = content.rindex("}") + 1
        raw = json.loads(content[start:end])
        known_tasks = {t["id"] for t in tasks}
        return {str(k): str(v) for k, v in raw.items()
                if str(k) in world_state["robots"] and str(v) in known_tasks}
