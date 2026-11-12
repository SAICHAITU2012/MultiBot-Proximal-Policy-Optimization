"""
probe_cmd_modes.py — physics probe of the Go2's command primitives in the
mission scene (colo_env, rigid dt 0.01, 15 cm / 7 kg crate).

Answers, with measured numbers, the questions the push endgame depends on:
  * reverse (vx < 0)         — does the dog actually back away from a crate?
  * slow forward (vx 0.15)   — can a slow body shove slide a 7 kg crate?
  * fast forward (vx 0.60)   — the learned-push command's shove distance/speed
  * strafe (vy)              — lateral mobility (the repositioning primitive)
  * in-place turn (wz)       — is a pure spin possible? (documented as ~dead)

Each case starts from the same geometry: crate at CRATE_XY, dog 0.55 m east of
it facing west (the mission's approach geometry). Measured with real physics.

Usage: python probe_cmd_modes.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

TEAM_LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TEAM_LAB / "experiments"))

import torch  # noqa: E402
import colo_env  # noqa: E402
import genesis as gs  # noqa: E402

CRATE_XY = (-1.0, 1.2)
DOG_OFFSET = 0.55          # dog base center is this far EAST of the crate
Z = 0.42


def make_team():
    return colo_env.CoLocatedTeam(
        go2_ckpt=str(TEAM_LAB / "logs/go2-nav/model_800.pt"),
        drone_ckpt=str(TEAM_LAB / "logs/drone-hover-s4/model_1600.pt"),
        extra_boxes=[CRATE_XY],
    )


def reset(team):
    """Place the crate at CRATE_XY and the dog DOG_OFFSET east of it, facing west."""
    dog = team.robots["go2"]
    crate = team.extra_box_entities[0]
    q = torch.zeros((1, 19), dtype=gs.tc_float, device=gs.device)
    q[0, 0] = CRATE_XY[0] + DOG_OFFSET
    q[0, 1] = CRATE_XY[1]
    q[0, 2] = Z
    q[0, 3] = 0.0   # w
    q[0, 4] = 0.0   # x
    q[0, 5] = 0.0   # y
    q[0, 6] = 1.0   # z -> yaw = pi  (facing -x, toward the crate)
    q[0, 7:19] = dog.default_dof_pos[0]
    dog.entity.set_qpos(q, zero_velocity=True)
    cq = torch.zeros((1, 7), dtype=gs.tc_float, device=gs.device)
    cq[0, 0] = CRATE_XY[0]
    cq[0, 1] = CRATE_XY[1]
    cq[0, 2] = 0.08
    cq[0, 3] = 1.0
    crate.set_qpos(cq, zero_velocity=True)
    dog.actions.zero_()
    dog.last_actions.zero_()
    dog.commands = torch.zeros((1, 3), dtype=gs.tc_float, device=gs.device)
    team.step()
    team.step()
    dog.refresh()

    return dog, crate


def run_case(team, name, cmd, seconds, hold_first=0.0):
    dog, crate = reset(team)
    p0 = [float(v) for v in dog.base_pos[0].cpu()]
    c0 = [float(v) for v in crate.get_pos()[0].cpu()]
    n = int(seconds / colo_env.SCENE_DT)
    hold = int(hold_first / colo_env.SCENE_DT)
    with torch.inference_mode():
        for i in range(n):
            if i < hold:
                dog.commands = torch.zeros((1, 3), dtype=gs.tc_float, device=gs.device)
            else:
                dog.commands = torch.tensor(cmd, dtype=gs.tc_float,
                                            device=gs.device).unsqueeze(0)
            team.step()
    dog.refresh()
    p1 = [float(v) for v in dog.base_pos[0].cpu()]
    c1 = [float(v) for v in crate.get_pos()[0].cpu()]
    q = dog.base_quat[0].cpu()
    yaw = math.atan2(float(2 * (q[0] * q[3] + q[1] * q[2])),
                     1 - 2 * float(q[2] ** 2 + q[3] ** 2))
    print(f"{name:<28} dog d=({p1[0]-p0[0]:+.3f},{p1[1]-p0[1]:+.3f}) "
          f"net={math.dist(p0[:2], p1[:2]):.3f}m  yaw0=3.14 yaw1={yaw:+.2f} "
          f"| crate d=({c1[0]-c0[0]:+.3f},{c1[1]-c0[1]:+.3f}) net={math.dist(c0[:2], c1[:2]):.3f}m "
          f"| dog->crate {math.dist(p1[:2], c1[:2]):.3f}m")
    return {"dog_net": math.dist(p0[:2], p1[:2]), "crate_net": math.dist(c0[:2], c1[:2]),
            "dog_pos": p1, "crate_pos": c1, "yaw": yaw}


def run_case_yaw(team, wz, seconds):
    dog, crate = reset(team)
    cq = torch.zeros((1, 7), dtype=gs.tc_float, device=gs.device)
    cq[0, 0], cq[0, 1], cq[0, 2], cq[0, 3] = 9.0, 9.0, 0.08, 1.0
    crate.set_qpos(cq, zero_velocity=True)
    q = dog.base_quat[0].cpu()
    y0 = math.atan2(float(2 * (q[0] * q[3] + q[1] * q[2])),
                    1 - 2 * float(q[2] ** 2 + q[3] ** 2))
    n = int(seconds / colo_env.SCENE_DT)
    for _ in range(n):
        dog.commands = torch.tensor((0.0, 0.0, wz), dtype=gs.tc_float,
                                    device=gs.device).unsqueeze(0)
        team.step()
    dog.refresh()
    q = dog.base_quat[0].cpu()
    y1 = math.atan2(float(2 * (q[0] * q[3] + q[1] * q[2])),
                    1 - 2 * float(q[2] ** 2 + q[3] ** 2))
    d = (y1 - y0 + math.pi) % (2 * math.pi) - math.pi
    return d / seconds


def run_case_free(team, name, cmd, seconds):
    """Same as run_case but with the crate moved far away (free floor)."""
    dog, crate = reset(team)
    cq = torch.zeros((1, 7), dtype=gs.tc_float, device=gs.device)
    cq[0, 0], cq[0, 1], cq[0, 2], cq[0, 3] = 9.0, 9.0, 0.08, 1.0
    crate.set_qpos(cq, zero_velocity=True)
    p0 = [float(v) for v in dog.base_pos[0].cpu()]
    n = int(seconds / colo_env.SCENE_DT)
    for _ in range(n):
        dog.commands = torch.tensor(cmd, dtype=gs.tc_float,
                                    device=gs.device).unsqueeze(0)
        team.step()
    dog.refresh()
    p1 = [float(v) for v in dog.base_pos[0].cpu()]
    return ((math.dist(p0[:2], p1[:2]) / seconds), name, p1)


def main():
    with torch.inference_mode():
        _main()


def _main():
    team = make_team()
    # settle: let the dog stand 1 s before each case (goto-like rest state)
    print("\n=== command primitive probe (crate 7 kg / 0.35 m, dog 0.55 m behind) ===")
    run_case(team, "rest (0,0,0) 1.5s", (0.0, 0.0, 0.0), 1.5)
    run_case(team, "reverse vx=-0.3 1.5s", (-0.3, 0.0, 0.0), 1.5)
    run_case(team, "reverse vx=-0.5 1.5s", (-0.5, 0.0, 0.0), 1.5)
    run_case(team, "strafe vy=+0.4 1.5s", (0.0, 0.4, 0.0), 1.5)
    run_case(team, "strafe vy=-0.4 1.5s", (0.0, -0.4, 0.0), 1.5)
    run_case(team, "spin wz=+0.8 1.5s", (0.0, 0.0, 0.8), 1.5)
    run_case(team, "spin wz=-0.8 1.5s", (0.0, 0.0, -0.8), 1.5)
    run_case(team, "creep-turn 0.15/+1.0 1.5s", (0.15, 0.0, 1.0), 1.5)
    run_case(team, "slow push vx=0.15 4s", (0.15, 0.0, 0.0), 4.0)
    run_case(team, "mid push vx=0.30 3s", (0.30, 0.0, 0.0), 3.0)
    run_case(team, "fast push vx=0.60 2s", (0.60, 0.0, 0.0), 2.0)
    # -- free-space speed calibration: the nav policy's LOW-SPEED DEADBAND
    #    (all "the dog froze" reports are small-command cases: measure it)
    print("\n=== free-space forward speed calibration (no crate ahead) ===")
    for v in (0.06, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60):
        r = run_case_free(team, f"vx={v:.2f} 2s", (v, 0.0, 0.0), 2.0)
        print(f"      commanded {v:.2f} m/s -> achieved {r[0]:+.3f} m/s  (ratio {r[0]/v:.2f})")
    # -- yaw calibration: is there a deadband on wz as well?
    print("\n=== free-space yaw rate calibration (pure spin) ===")
    for w in (0.15, 0.25, 0.35, 0.50, 0.80, 1.00):
        r = run_case_yaw(team, w, 1.5)
        print(f"      commanded wz {w:.2f} -> achieved {r:+.3f} rad/s")
    print("\n[probe] done")


if __name__ == "__main__":
    main()
