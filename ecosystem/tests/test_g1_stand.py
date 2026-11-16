#!/usr/bin/env python3
"""test_g1_stand.py — does the G1 standing policy still stand in Genesis 1.4.5?

Baseline to beat (measured in the original env, genesis 0.3.13):
    stood the full 10 s in 4/4 trials, base height 0.797 -> 0.792 m.

This test runs the ported policy with ZERO retraining and reports the same
numbers. A fall is base z < 0.40 m or |tilt| > 0.8 rad.
"""
import sys, math, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import genesis as gs

from robots.g1_stand import (G1Stand, G1StandPolicy, policy_params, load_env_args,
                             URDF, SIM_DT, DECIMATION)

PLANE = "urdf/plane/plane.urdf"  # resolved against the genesis assets dir
SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
SPAWN_Z = 0.80


def tilt(quat):
    w, x, y, z = quat.tolist()
    # angle between body z and world z
    up = 1 - 2 * (x * x + y * y)
    return math.acos(max(-1.0, min(1.0, up)))


def main():
    gs.init(backend=gs.cpu, precision="32", logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=SIM_DT, substeps=1),
        rigid_options=gs.options.RigidOptions(
            dt=SIM_DT, enable_collision=True, enable_joint_limit=True,
            enable_self_collision=False,
            constraint_solver=gs.constraint_solver.Newton, tolerance=1e-5),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.URDF(file=PLANE, fixed=True))
    ent = scene.add_entity(gs.morphs.URDF(file=str(URDF), pos=(0, 0, SPAWN_Z), fixed=False))
    scene.build()

    params = policy_params(load_env_args())
    g1 = G1Stand(ent, params, G1StandPolicy())
    print(f"joints mapped: {len(g1.dof_idx)}  kp[hip]={params['kp'][0]:.0f} "
          f"kd[hip]={params['kd'][0]:.0f} tau[hip]={params['tau_max'][0]:.0f}")

    steps = int(SECONDS / SIM_DT)
    hs, ts = [], []
    fell_at = None
    t0 = time.time()
    for s in range(steps):
        if s % DECIMATION == 0:
            g1.policy_step()
        g1.apply_torque()
        scene.step()
        if s % 20 == 0:                        # every 0.1 s
            pos, quat = g1.base()
            h, th = float(pos[2]), tilt(quat)
            hs.append(h); ts.append(th)
            if fell_at is None and (h < 0.40 or th > 0.8):
                fell_at = s * SIM_DT
        if s % 400 == 0:
            print(f"  t={s*SIM_DT:5.2f}s  z={float(g1.base()[0][2]):.3f}  tilt={tilt(g1.base()[1]):.3f}")

    wall = time.time() - t0
    h0, h_end, hmin = hs[0], hs[-1], min(hs)
    print("\n=== G1 STAND RESULT (genesis 1.4.5, no retraining) ===")
    print(f"  duration        : {SECONDS:.1f} s sim ({steps} steps, {wall:.1f} s wall, "
          f"{SECONDS/wall:.1f}x realtime)")
    print(f"  base height     : start {h0:.3f} m  end {h_end:.3f} m  min {hmin:.3f} m")
    print(f"  max tilt        : {max(ts)*180/math.pi:.1f} deg")
    print(f"  fell            : {fell_at if fell_at is not None else 'no'}")
    print(f"  baseline (0.3.13): 0.797 -> 0.792 m, no fall")
    ok = fell_at is None and h_end > 0.70
    print(f"  VERDICT         : {'PASS — port holds' if ok else 'FAIL — port does not hold'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
