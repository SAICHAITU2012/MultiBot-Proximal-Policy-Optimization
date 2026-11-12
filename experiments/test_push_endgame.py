"""
test_push_endgame.py — isolated, deterministic endgame test for the Go2 crate
push state machine (team/agents.py Go2TaskAgent).

Why: the mission is a 1200 s end-to-end run in which the push loop is one
component among six. Debugging it there is slow and confounded. This harness
runs ONLY the push loop, in the real mission scene (colo_env: same dog, same
policy, same 15 cm / 7 kg crate, same physics), with a synthetic zone, and
prints every state transition plus a per-cycle geometry trace.

It is also the regression test for the endgame: it asserts that the crate
reaches the zone from a standard mission start.

Usage:
    python test_push_endgame.py                 # standard case, 120 s budget
    python test_push_endgame.py --dist 2.7 --yaw 0.0 --seconds 120
    python test_push_endgame.py --verbose
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
import colo_env  # noqa: E402
import genesis as gs  # noqa: E402
from agents import Go2TaskAgent, make_task  # noqa: E402
from blackboard import Blackboard  # noqa: E402

# geometry: the crate starts east of the zone; the dog starts east of the crate
ZONE = (-1.65, 1.2)
CRATE0 = (1.20, 1.20)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class Recorder:
    """Records blackboard events and phase transitions in order."""

    def __init__(self, verbose: bool):
        self.events = []
        self.verbose = verbose
        self.last_phase = None
        self.cycles = 0
        self.finished = None

    def on_msg(self, msg):
        if msg["topic"] in ("event", "task"):
            self.events.append((msg["t"], msg["type"], msg["data"]))
            if msg["type"] == "push_stroke_start":
                self.cycles += 1
            if msg["type"] == "task_complete":
                self.finished = (msg["t"], msg["data"].get("result"))
        if self.verbose and msg["topic"] == "event":
            print(f"    [{msg['t']:7.2f}s] {msg['from']}: {msg['type']} {msg['data']}")

    def tick(self, t, agent):
        ph = getattr(agent, "_phase", None)
        if self.verbose and ph != self.last_phase:
            print(f"    [{t:7.2f}s] phase -> {ph}")
        self.last_phase = ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=float, default=2.7, help="crate->zone start distance")
    ap.add_argument("--yaw", type=float, default=0.0, help="crate start yaw (rad)")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--trace", type=float, default=0.0,
                    help="print a geometry row every N sim-seconds")
    args = ap.parse_args()

    with torch.inference_mode():
        _run(args)


def _run(args):
    bb = Blackboard()
    rec = Recorder(args.verbose)
    bb.subscribe(rec.on_msg)

    crate_xy = (ZONE[0] + args.dist, ZONE[1] + 0.0)
    team = colo_env.CoLocatedTeam(
        go2_ckpt=str(TEAM_LAB / "logs/go2-nav/model_800.pt"),
        drone_ckpt=str(TEAM_LAB / "logs/drone-hover-s4/model_1600.pt"),
        extra_boxes=[crate_xy],
    )
    crate = team.extra_box_entities[0]

    # place the crate exactly, with the requested yaw
    cq = torch.zeros((1, 7), dtype=gs.tc_float, device=gs.device)
    cq[0, 0], cq[0, 1], cq[0, 2] = crate_xy[0], crate_xy[1], 0.08
    cq[0, 3] = math.cos(args.yaw / 2.0)
    cq[0, 6] = math.sin(args.yaw / 2.0)
    crate.set_qpos(cq, zero_velocity=True)
    # the dog starts further east, facing west (the mission start geometry)
    dog = team.robots["go2"]
    dq = torch.zeros((1, 19), dtype=gs.tc_float, device=gs.device)
    dq[0, 0], dq[0, 1], dq[0, 2] = crate_xy[0] + 0.9, crate_xy[1], 0.42
    dq[0, 6] = 1.0                      # yaw = pi, facing -x
    dq[0, 7:19] = dog.default_dof_pos[0]
    dog.entity.set_qpos(dq, zero_velocity=True)
    dog.refresh()

    go2 = Go2TaskAgent(team.robots["go2"], bb, "go2", team=team)
    go2.robot.push_policy = colo_env.load_actor_from_log(
        TEAM_LAB / "logs/go2-push-s5", "model_800.pt")

    task = make_task("fetch", "fetch_ground", "push", location=(ZONE[0], ZONE[1], 0.0),
                     priority=1)
    task["box_pos_fn"] = lambda: [float(v) for v in crate.get_pos().cpu().numpy().flatten()]
    task["box_quat_fn"] = lambda: [float(v) for v in crate.get_quat().cpu().numpy().flatten()]
    go2.assign(task)

    t0 = time.time()
    steps = int(args.seconds / colo_env.SCENE_DT)
    last_log = -1.0
    tr_next = [0.0]
    delivered_t = None
    try:
        for _ in range(steps):
            t = team.sim_time
            bb.set_clock(t)
            team.step()
            go2.update(t)
            rec.tick(t, go2)
            cp = crate.get_pos()[0]
            d = math.hypot(float(cp[0]) - ZONE[0], float(cp[1]) - ZONE[1])
            if delivered_t is None and d < 0.55:
                delivered_t = t
                print(f"  *** DELIVERED at t={t:.2f}s (d={d:.3f})")
            if args.trace and t >= tr_next[0]:
                tr_next[0] = t + args.trace
                st = go2.robot.get_state()
                gp = st["pos"]
                yaw = go2._yaw_from_quat(st["quat"])
                cx, cy = float(cp[0]), float(cp[1])
                dcr = math.hypot(gp[0] - cx, gp[1] - cy)
                ux, uy = (ZONE[0] - cx) / max(d, 1e-9), (ZONE[1] - cy) / max(d, 1e-9)
                vx, vy = gp[0] - cx, gp[1] - cy
                along = vx * ux + vy * uy
                perp = abs(vx * uy - vy * ux)
                ye = wrap(math.atan2(uy, ux) - yaw)
                try:
                    cs = [round(float(v), 2) for v in list(go2.robot.commands[0])]
                except Exception:
                    cs = "?"
                print(f"    TR t={t:6.2f} {str(getattr(go2, '_phase', '-')):<9} "
                      f"dcr={dcr:4.2f} along={along:6.2f} perp={perp:4.2f} "
                      f"z={gp[2]:4.2f} ye={ye:6.2f} cmd={cs}")
            if t - last_log >= 5.0:
                last_log = t
                if not args.quiet:
                    gp = go2.robot.base_pos[0]
                    print(f"  t={t:6.1f} d={d:5.2f} dog=({float(gp[0]):5.2f},{float(gp[1]):5.2f},"
                          f"{float(gp[2]):.2f}) phase={str(getattr(go2, '_phase', '-')):<9}"
                          f" strokes={rec.cycles}")
            if go2.task is None:
                print(f"  task finished at t={t:.2f}s")
                break
    finally:
        cp = crate.get_pos()[0]
        d = math.hypot(float(cp[0]) - ZONE[0], float(cp[1]) - ZONE[1])
        print(f"\n[result] start_d={args.dist:.2f} final_d={d:.3f} "
              f"delivered={'YES' if delivered_t is not None else 'NO'} "
              f"t_delivered={delivered_t} strokes={rec.cycles} "
              f"wall={time.time() - t0:.1f}s")
        ev = [e[1] for e in rec.events]
        print(f"[events] stroke_start={ev.count('push_stroke_start')} "
              f"stroke_end={ev.count('push_stroke_end')} "
              f"timeout={ev.count('push_phase_timeout')} "
              f"replan={ev.count('push_replan')} "
              f"falls={ev.count('dog_fell_recovering')}")
        trav = [e[2].get("travelled") for e in rec.events if e[1] == "push_stroke_end"]
        if trav:
            print(f"[strokes] n={len(trav)} travel_m={[round(v or 0, 3) for v in trav]}")
        ok = delivered_t is not None
        print(f"[verdict] {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
