"""
diagnose_endgame.py — read-only probe of the Go2 push endgame.

Runs the REAL mission code path (team/mission.py + team/agents.py) but logs,
at 1 Hz, the exact geometry the Go2TaskAgent's push state machine computes:
dog pos/yaw, crate pos/yaw, crate->zone distance, the along/perp line test,
the stand-point distance and the state-machine flags. Nothing is monkeypatched
inside the agent's logic — the geometry is recomputed here from public state
with the same formulas, so the log shows what the agent is deciding on.

Usage (conda genesis, from team_lab/experiments):
    python diagnose_endgame.py --seconds 120 --packages 2
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

TEAM_LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TEAM_LAB))
sys.path.insert(0, str(TEAM_LAB / "team"))
sys.path.insert(0, str(TEAM_LAB / "experiments"))

import torch  # noqa: E402
from mission import Mission, DROP_ZONE  # noqa: E402


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--packages", type=int, default=2)
    ap.add_argument("--planner", default="rule")
    args = ap.parse_args()

    m = Mission(n_packages=args.packages, planner_name=args.planner,
                max_steps=int(args.seconds * 100 * 2))
    go2 = m.agents["go2"]
    max_steps = int(args.seconds * 100)

    m.metrics["t_start"] = m.team.sim_time
    t0 = time.time()
    last_plan = -1e9
    last_log = -1.0
    print(f"{'t':>6} {'dog_x':>7} {'dog_y':>7} {'z':>5} {'v':>5} {'yaw':>7} {'crate_x':>8} {'crate_y':>8} "
          f"{'cyaw':>6} {'box_d':>6} {'d_cr':>6} {'along':>6} {'perp':>6} {'d_std':>6} "
          f"{'inPos':>5} {'push':>4} {'ext':>3} cmd task")
    try:
        with torch.inference_mode():
            for step in range(max_steps):
                t = m.team.sim_time
                m.bb.set_clock(t)
                if t - last_plan >= 4.0:
                    last_plan = t
                    m._plan_tick(t)
                m.team.step()
                for agent in m.agents.values():
                    agent.update(t)
                m._check_deliveries()
                if m._mission_complete():
                    break
                if t - last_log >= 1.0:
                    last_log = t
                    _log(m, go2, t)
    finally:
        m._finish()
        print(f"[diag] wall {time.time() - t0:.1f}s for {m.team.sim_time:.0f}s sim")


def _log(m, go2, t):
    st = go2.robot.get_state()
    pos = st["pos"]
    yaw = go2._yaw_from_quat(st["quat"])
    # which package is the go2 working on?
    task = go2.task
    pkg = None
    if task is not None and "box_pos_fn" in task:
        raw = task["box_pos_fn"]()
        while isinstance(raw[0], (list, tuple)):
            raw = raw[0]
        pkg = (float(raw[0]), float(raw[1]))
    if pkg is None:
        print(f"{t:6.0f} {pos[0]:7.2f} {pos[1]:7.2f} {yaw:7.2f}  (no push task) "
              f"task={task['id'] if task else None}")
        return
    qfn = task.get("box_quat_fn")
    cyaw = None
    if qfn is not None:
        q = qfn()
        cyaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                          1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))
    gx, gy = DROP_ZONE[0], DROP_ZONE[1]
    box_d = math.dist(pkg, (gx, gy))
    czx, czy = pkg[0] - gx, pkg[1] - gy
    L = math.hypot(czx, czy) + 1e-9
    ldx, ldy = -czx / L, -czy / L
    wx, wy = pos[0] - pkg[0], pos[1] - pkg[1]
    along = wx * ldx + wy * ldy
    perp = abs(-wx * ldy + wy * ldx)
    stand = (pkg[0] - ldx * 0.6, pkg[1] - ldy * 0.6)
    d_std = math.dist((pos[0], pos[1]), stand)
    perp_gate = 0.20 if box_d < 0.8 else 0.35
    in_pos = (along < -0.05) and (perp < perp_gate)
    cmd = go2.robot.commands
    try:
        c = (f"({float(cmd[0]):.2f},{float(cmd[1]):.2f},{float(cmd[2]):.2f})"
             if cmd is not None else "None")
    except (TypeError, ValueError):
        c = f"bad{cmd}"
    lv = st.get('lin_vel_world') or [0.0, 0.0, 0.0]
    spd = math.hypot(lv[0], lv[1])
    print(f"{t:6.0f} {pos[0]:7.2f} {pos[1]:7.2f} {st['pos'][2]:5.2f} {spd:5.2f} {yaw:7.2f} {pkg[0]:8.2f} {pkg[1]:8.2f} "
          f"{cyaw if cyaw is not None else float('nan'):6.2f} {box_d:6.2f} "
          f"{math.dist((pos[0], pos[1]), pkg):6.2f} {along:6.2f} {perp:6.2f} "
          f"{d_std:6.2f} {str(in_pos)[0]:>5} {str(getattr(go2, '_pushing', False))[0]:>4} "
          f"{str(go2.robot.external_control)[0]:>3} {c} {task['id']} "
          f"phase={getattr(go2, '_phase', '-'):<9} del={m.metrics['delivered']}")


if __name__ == "__main__":
    main()
