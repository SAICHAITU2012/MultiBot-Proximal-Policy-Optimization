"""
test_recovery.py — isolated test for the Go2 push RECOVERY primitive
(Go2TaskAgent._retreat_tick).

The recovery is "from wherever the stroke left the dog, get back to a clear
run-up behind the crate, on the crate->zone line, facing it". It is the single
most failure-prone part of the push loop (jamming, orbiting, wandering), and in
the full-mission harness its failures are indistinguishable from charge
failures or planner effects. Here it runs alone: the dog is placed at a chosen
bad pose next to the crate and only _retreat_tick drives it. No push policy, no
task logic, no planner.

A case PASSES when the charge gate opens:
    along <= -0.55, perp <= 0.30, 0.55 <= d_crate <= 1.05, |heading err| <= 0.60

Usage:
    python test_recovery.py              # run every case
    python test_recovery.py --case jam   # one case
    python test_recovery.py --verbose
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

ZONE = (-1.65, 1.2)
CRATE = (1.20, 1.20)

# dog start poses, as (d_crate, bearing_of_dog_from_crate_deg, yaw_deg)
#   0 deg  = dog is EAST of the crate (behind it, on the line)
#   180 deg = dog is WEST of the crate (in front: the wrong side)
CASES = {
    "behind_far":   (1.30, 0.0, 180.0),   # too far back, already aimed
    "behind_off":   (1.00, 40.0, 180.0),  # behind but off the line
    "behind_back":  (1.00, 0.0, 0.0),     # behind, aimed AWAY from the crate
    "jam":          (0.30, 10.0, 180.0),  # body against the crate, pressed in
    "jam_side":     (0.35, 75.0, 180.0),  # alongside the crate
    "wrong_side":   (0.60, 180.0, 0.0),   # between crate and zone
    "wrong_far":    (1.20, 170.0, 0.0),   # in front, well clear
    "side_right":   (0.80, 90.0, 270.0),  # due north of the crate
}


def run_all():
    """Run each case in a FRESH PROCESS.

    The locomotion policy carries internal history between steps, so a hard
    qpos teleport inside one sim leaves it stale and the next case starts in a
    transient. Measured: cases that pass alone (behind_back: 10.5 s) fail when
    run after another case in the same process. One process per case is the only
    way to get independent measurements.
    """
    import subprocess
    ok = 0
    for name in CASES:
        r = subprocess.run([sys.executable, __file__, "--case", name],
                           capture_output=True, text=True)
        line = [ln for ln in r.stdout.splitlines() if line_is_case(ln)]
        print(line[-1] if line else f"  {name:<12} ERROR")
        ok += 1 if r.returncode == 0 else 0
    print(f"\n[recovery] {ok}/{len(CASES)} cases reached the charge gate")
    return 0 if ok == len(CASES) else 1


def line_is_case(ln):
    return ln.strip().startswith(tuple(CASES)) and "dcr0=" in ln


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.case is None:
        return run_all()
    with torch.inference_mode():
        return _run(args)


def _run(args):
    bb = Blackboard()
    team = colo_env.CoLocatedTeam(
        go2_ckpt=str(TEAM_LAB / "logs/go2-nav/model_800.pt"),
        drone_ckpt=str(TEAM_LAB / "logs/drone-hover-s4/model_1600.pt"),
        extra_boxes=[CRATE],
    )
    crate = team.extra_box_entities[0]
    dog = team.robots["go2"]
    go2 = Go2TaskAgent(team.robots["go2"], bb, "go2", team=team)
    go2.robot.push_policy = colo_env.load_actor_from_log(
        TEAM_LAB / "logs/go2-push-s5", "model_800.pt")

    task = make_task("fetch", "fetch_ground", "push", location=(ZONE[0], ZONE[1], 0.0),
                     priority=1)
    task["box_pos_fn"] = lambda: [float(v) for v in crate.get_pos().cpu().numpy().flatten()]
    task["box_quat_fn"] = lambda: [float(v) for v in crate.get_quat().cpu().numpy().flatten()]
    go2.assign(task)
    go2._phase = "retreat"

    cases = {args.case: CASES[args.case]} if args.case else CASES
    results = {}
    for name, (dcr, bearing_deg, yaw_deg) in cases.items():
        # reset the crate: an earlier case's escape can nudge it, which would
        # silently change the next case's geometry (measured: identical cases
        # passed alone and failed in the suite purely from this pollution)
        cq = torch.zeros((1, 7), dtype=gs.tc_float, device=gs.device)
        cq[0, 0], cq[0, 1], cq[0, 2], cq[0, 3] = CRATE[0], CRATE[1], 0.08, 1.0
        crate.set_qpos(cq, zero_velocity=True)
        # place the dog
        a = math.radians(bearing_deg)
        px, py = CRATE[0] + dcr * math.cos(a), CRATE[1] + dcr * math.sin(a)
        dq = torch.zeros((1, 19), dtype=gs.tc_float, device=gs.device)
        dq[0, 0], dq[0, 1], dq[0, 2] = px, py, 0.30
        yaw = math.radians(yaw_deg)
        dq[0, 3], dq[0, 6] = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        dq[0, 7:19] = dog.default_dof_pos[0]
        dog.entity.set_qpos(dq, zero_velocity=True)
        dog.refresh()
        go2._reset_push_loop()

        t0sim = team.sim_time
        ok_at = None
        cpos_before = crate.get_pos()[0].cpu().numpy().copy()
        t0 = time.time()
        while team.sim_time - t0sim < args.seconds:
            t = team.sim_time
            bb.set_clock(t)
            team.step()
            st = go2.robot.get_state()
            pos = st["pos"]
            yaw_now = go2._yaw_from_quat(st["quat"])
            box, _ = go2._crate()
            g = go2._push_track(pos, box)
            ye = go2._wrap(math.atan2(g["u"][1], g["u"][0]) - yaw_now)
            ready = (g["along"] <= -0.55 and g["perp"] <= 0.30
                     and 0.55 <= g["d_crate"] <= 1.05 and abs(ye) <= 0.60)
            if args.verbose and int((team.sim_time - t0sim) * 2) % 2 == 0:
                if abs((team.sim_time - t0sim) % 1.0) < 0.02:
                    print(f"   [{team.sim_time - t0sim:5.2f}s] dcr={g['d_crate']:5.2f} "
                          f"along={g['along']:6.2f} perp={g['perp']:5.2f} "
                          f"ye={ye:6.2f} z={pos[2]:4.2f} ready={ready}")
            if ready:
                ok_at = team.sim_time - t0sim
                break
            go2._retreat_tick(pos, yaw_now, g, t)
        cpos_after = crate.get_pos()[0].cpu().numpy()
        moved = float(math.hypot(cpos_after[0] - cpos_before[0],
                                 cpos_after[1] - cpos_before[1]))
        results[name] = (ok_at, moved)
        status = f"ok in {ok_at:5.2f}s" if ok_at is not None else "FAIL (timeout)"
        print(f"  {name:<12} dcr0={dcr:4.2f} brg={bearing_deg:5.1f} yaw0={yaw_deg:5.0f}"
              f" -> {status:<16} crate_moved={moved:.3f}")

    n_ok = sum(1 for v in results.values() if v[0] is not None)
    print(f"\n[recovery] {n_ok}/{len(results)} cases reached the charge gate")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
