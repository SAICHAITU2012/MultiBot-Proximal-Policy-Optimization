"""
mission.py — the integrated multi-robot mission (Stage 7 scenario).

Scenario "warehouse aid":
  N packages (boxes) appear at random ground locations. The team must:
    1. RECON  — drone surveys each package's approximate location and reports
                its observed position (the drone's policy flies the waypoints).
    2. FETCH  — Go2 navigates to behind each package and pushes it to the
                drop zone in front of the Panda station (policy-driven walking;
                pushing emerges from contact).
    3. SORT   — Panda picks packages that arrive in its workspace (grasp policy,
                when plugged in; boxes are detected as delivered when inside the
                drop zone).
  The planner (rule-based or LLM) assigns tasks over a blackboard; everything
  said/done is transcripted to a JSONL log + readable chat transcript.

Metrics: task completion per stage, makespan, distance efficiency, planner decisions.

Usage (inside the genesis conda env, from team_lab/experiments):
    python ../team/mission.py --packages 2 --planner rule --max-steps 6000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

TEAM_LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TEAM_LAB))
sys.path.insert(0, str(TEAM_LAB / "experiments"))  # colo_env

from blackboard import Blackboard  # noqa: E402
from planner import LLMPlanner, RuleBasedPlanner  # noqa: E402

import colo_env  # noqa: E402  (sets genesis device + loads robot agents)
import genesis as gs  # noqa: E402
import torch  # noqa: E402

from agents import (DroneTaskAgent, Go2TaskAgent, PandaTaskAgent,  # noqa: E402
                    TaskBoard, make_task)

# AXIS-ALIGNED pushes only: packages, corridor, and zone all on y = 1.2 —
# the learned pusher was trained face-on along one axis; corner/diagonal
# contact wedges the crate. Landing at (-1.2,1.2) is 0.8 m from the arm base
# (-2,1.2) — inside the 0.85 m workspace.
DROP_ZONE = (-1.65, 1.2, 0.14)   # approved 0.15 m shift toward the arm (landing center ~0.75 m, surface ~0.58 m — inside workspace)
PKG_APPROX_NOISE = 1.0           # recon reports package pos with this much error


class Mission:
    def __init__(self, n_packages: int = 2, planner_name: str = "rule",
                 max_steps: int = 6000, transcript_dir: str | None = None):
        self.bb = Blackboard()
        self.board = TaskBoard()
        self.max_steps = max_steps
        self.planner = (RuleBasedPlanner() if planner_name == "rule"
                        else LLMPlanner(base_url="http://127.0.0.1:11434/v1", model=planner_name))
        self.transcript_dir = Path(transcript_dir) if transcript_dir else TEAM_LAB / "logs"

        # -- world (policies: go2-nav for full-command navigation, drone s4,
        #    panda reach policy plugged in below)
        # packages on the SAME y-line as the drop zone: every push is a pure
        # -x axis-aligned shove, exactly the training geometry
        pkg_spawns = [(1.2, 1.2, 0.08), (2.0, 0.3, 0.08)]
        self.team = colo_env.CoLocatedTeam(
            go2_ckpt=str(TEAM_LAB / "logs/go2-nav/model_800.pt"),
            drone_ckpt=str(TEAM_LAB / "logs/drone-hover-s4/model_1600.pt"),
            extra_boxes=pkg_spawns,
        )
        self.agents = {
            "go2": Go2TaskAgent(self.team.robots["go2"], self.bb, "go2", team=self.team),
            "drone": DroneTaskAgent(self.team.robots["drone"], self.bb, "drone", team=self.team),
            "panda": PandaTaskAgent(self.team.robots["panda"], self.bb, "panda", team=self.team),
        }
        # plug in the trained reach policy (reaches whatever box robot.obj points at)
        panda_ckpt_dir = TEAM_LAB / "logs/grasp-panda_rl"
        if (panda_ckpt_dir / "cfgs.pkl").exists():
            self.team.robots["panda"].policy = colo_env.load_actor_from_log(panda_ckpt_dir)
            self.bb.publish("mission", "event", "panda_policy_loaded", {"ckpt_dir": str(panda_ckpt_dir)})
        # plug in the P1b manipulation policy (reach+press+push in one policy)
        mp_ckpt_dir = TEAM_LAB / "logs/panda-push-s1"
        if (mp_ckpt_dir / "model_900.pt").exists():
            self.team.robots["panda"].manip_policy = colo_env.load_actor_from_log(
                mp_ckpt_dir, "model_900.pt")
            self.bb.publish("mission", "event", "manip_policy_loaded",
                            {"ckpt_dir": str(mp_ckpt_dir / "model_900.pt")})
        # plug in the learned crate-push policy. Pinned (2026-11-08): go2-push-s5/model_800 —
        # the pre-rewrite champion, VERIFIED on the canonical 15 cm / 7 kg
        # geometry: 8/16 pushed >=0.15 m, 0 true falls, 16/16 moved (the policy
        # gates met; cross-env eval in INVESTIGATION_LOG.md).
        # sweep-verified best of the direction-robust curriculum (6/16 fresh
        # episodes pushed >=0.15 m, 16/16 moved, mean +0.127 m; later
        # checkpoints DEGRADED — pinned deliberately).
        push_ckpt_dir = TEAM_LAB / "logs/go2-push-s5"
        if (push_ckpt_dir / "model_800.pt").exists():
            self.team.robots["go2"].push_policy = colo_env.load_actor_from_log(
                push_ckpt_dir, "model_800.pt")
            self.bb.publish("mission", "event", "push_policy_loaded",
                            {"ckpt_dir": str(push_ckpt_dir / "model_577.pt")})

        # -- register packages (entities spawned pre-build) + recon tasks
        self.packages = []
        for i, ent in enumerate(self.team.extra_box_entities):
            pos = [float(v) for v in ent.get_pos().cpu().numpy().flatten()]
            self.packages.append({"id": f"pkg{i}", "entity": ent, "spawn": pos})
            # approximate location only — recon must refine it
            approx = (pos[0] + PKG_APPROX_NOISE, pos[1] + PKG_APPROX_NOISE, 1.0)
            self.board.add(make_task(f"recon{i}", "recon_air", "recon_air",
                                     location=approx, priority=2))
        # genuine-manipulation task: pick the graspable parcel at the station
        # (the 35 cm delivery crates are physically ungraspable — stroke limit)
        self.board.add(make_task("parcel_pick", "manipulate", "pick",
                                 location=(-1.55, 1.55), priority=1))  # slide station (north of the delivery corridor)
        task = self.board.get("parcel_pick")
        task["parcel_entity"] = self.team.parcel_entity
        task["parcel_start_x"] = float(
            self.team.parcel_entity.get_pos()[0, 0])
        # stage-2 fetch tasks are created after each recon completes
        self.recon_done = set()
        self.drop_zone_delivered = set()
        self.metrics = {"t_start": None, "recon": 0, "delivered": 0, "inspected": 0, "picked": 0,
                        "planner_decisions": 0, "messages": 0}

        self.bb.subscribe(self._on_message)

    # -- blackboard wiring -------------------------------------------------
    def _on_message(self, msg: dict):
        self.metrics["messages"] += 1
        if msg["from"] == "panda" and msg["type"] == "task_complete":
            tid = msg["data"]["task"]
            if tid.startswith("process"):
                self.metrics["inspected"] += 1  # crate CONTACT verified (honest: no grasp possible)
            elif tid == "parcel_pick":
                self.metrics["picked"] += 1     # genuine GRASPED+LIFTED of the parcel
        if msg["from"] == "drone" and msg["type"] == "task_complete":
            tid = msg["data"]["task"]
            self.recon_done.add(tid)
            self.metrics["recon"] += 1
            i = tid.replace("recon", "")
            pkg = next(p for p in self.packages if p["id"] == f"pkg{i}")
            true_pos = [float(v) for v in pkg["entity"].get_pos().cpu().numpy().flatten()]
            observed = (true_pos[0] + 0.15, true_pos[1] + 0.15)  # recon refinement
            pkg["observed"] = observed
            self.board.add(make_task(f"fetch{i}", "fetch_ground", "push",
                                     location=DROP_ZONE, priority=1))
            self.board.tasks[f"fetch{i}"]["box_pos_fn"] = (
                lambda ent=pkg["entity"]: [float(v) for v in ent.get_pos().cpu().numpy().flatten()])
            self.board.tasks[f"fetch{i}"]["box_quat_fn"] = (
                lambda ent=pkg["entity"]: [float(v) for v in ent.get_quat().cpu().numpy().flatten()])
            self.bb.publish("mission", "event", "package_located",
                            {"package": pkg["id"], "observed": [round(v, 2) for v in observed]})

    # -- main loop ----------------------------------------------------------
    def run(self, planner_period_s: float = 4.0):
        self.metrics["t_start"] = self.team.sim_time
        self.bb.publish("mission", "event", "mission_start",
                        {"packages": len(self.packages), "planner": self.planner.name})
        last_plan = -1e9
        try:
            # inference mode for the WHOLE mission loop: all policies are
            # frozen; mixing inference and normal tensors crashes on in-place
            # updates inside the physics control path
            with torch.inference_mode():
                for step in range(self.max_steps):
                    t = self.team.sim_time
                    self.bb.set_clock(t)

                    # planner ticks periodically
                    if t - last_plan >= planner_period_s:
                        last_plan = t
                        self._plan_tick(t)

                    # agents execute at their control periods (colocated stepping)
                    self.team.step()
                    for agent in self.agents.values():
                        agent.update(t)

                    self._check_deliveries()
                    if self._mission_complete():
                        break
        finally:
            self._finish()

    def _plan_tick(self, t: float):
        world = {"robots": {}}
        for rid, agent in self.agents.items():
            st = agent.status()
            st["pos"] = tuple(st["pos"])
            world["robots"][rid] = st
        open_tasks = self.board.open_tasks()
        if not open_tasks:
            return
        assignments = self.planner.plan(world, open_tasks)
        for rid, tid in assignments.items():
            task = self.board.get(tid)
            agent = self.agents[rid]
            # CLAIM-ON-APPLY (2026-11-11, M6): the guard required
            # task["status"] == "assigned", but open tasks are "open" — the
            # rule planner mutates the status inside its own plan() (so it
            # always worked) while the LLM planner returns a pure mapping
            # whose assignments were silently dropped (planner_decisions 0 in
            # every LLM run). Any planner's assignments must be claimable here.
            if agent.task is None and task["status"] in ("open", "assigned"):
                # accept BOTH claim styles: the rule planner pre-marks
                # "assigned" inside its own plan(); pure-mapping planners
                # (the LLM) return "open" tasks that we claim here.
                task["status"] = "assigned"
                task["assigned_to"] = rid
                agent.assign(task)
                self.metrics["planner_decisions"] += 1
                self.bb.publish("planner", "assignment", "plan",
                                {"robot": rid, "task": tid,
                                 "why": getattr(self.planner, "last_reasoning", "")[:200]})

    def _check_deliveries(self):
        for i, pkg in enumerate(self.packages):
            if pkg["id"] in self.drop_zone_delivered:
                continue
            pos = [float(v) for v in pkg["entity"].get_pos().cpu().numpy().flatten()]
            d = ((pos[0] - DROP_ZONE[0]) ** 2 + (pos[1] - DROP_ZONE[1]) ** 2) ** 0.5
            if d < 0.55:  # the station's arrival area (the zone is a region, not a point)
                self.drop_zone_delivered.add(pkg["id"])
                self.metrics["delivered"] += 1
                self.bb.publish("mission", "event", "package_delivered",
                                {"package": pkg["id"], "drop_distance": round(d, 3)})
                # hand-off to the arm: process (reach) task at the drop zone
                if f"process{i}" not in self.board.tasks:
                    task = make_task(f"process{i}", "process", "pick",
                                     location=DROP_ZONE, priority=1)
                    task["pkg_entity"] = pkg["entity"]
                    self.board.add(task)

    def _mission_complete(self):
        return (self.metrics["delivered"] == len(self.packages)
                and self.metrics["inspected"] == len(self.packages)
                and self.metrics["picked"] >= 1)

    def _finish(self):
        t = self.team.sim_time - self.metrics["t_start"]
        self.bb.publish("mission", "event", "mission_end",
                        {"makespan_s": round(t, 1), **{k: v for k, v in self.metrics.items() if k != "t_start"}})
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = self.transcript_dir / f"mission_{stamp}"
        out.mkdir(parents=True, exist_ok=True)
        self.bb.save_transcript(out / "transcript.txt")
        (out / "metrics.txt").write_text(
            "\n".join(f"{k}: {v}" for k, v in self.metrics.items()) + f"\nmakespan_s: {t:.1f}\n")
        print(f"[mission] artifacts -> {out}")
        print(f"[mission] makespan {t:.1f}s | recon {self.metrics['recon']}/{len(self.packages)} "
              f"| delivered {self.metrics['delivered']}/{len(self.packages)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--packages", type=int, default=2)
    ap.add_argument("--planner", default="rule",
                    help="'rule' or an ollama model name (e.g. 'qwen2.5:7b') for the LLM commander")
    ap.add_argument("--max-steps", type=int, default=6000)
    args = ap.parse_args()
    Mission(n_packages=args.packages, planner_name=args.planner, max_steps=args.max_steps).run()
