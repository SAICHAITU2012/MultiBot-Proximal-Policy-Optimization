#!/usr/bin/env python3
"""test_g1_disturbance.py — Experiment B: can the FROZEN G1 standing policy
recover from a controlled external disturbance? (no retraining)

One trial per process (a fallen robot cannot be re-used for the next trial):

    python tests/test_g1_disturbance.py --sweep          # the full envelope
    python tests/test_g1_disturbance.py --trial 120 90   # one trial: F N for 0.25 s at 90 deg

Measured per trial:
  * initial stability  (pre-disturbance height mean/std)
  * max displacement   (horizontal base travel + max tilt)
  * recovery time      (after the impulse: back to nominal height +-3 % AND tilt <5 deg, held 2 s)
  * final height
  * fell (z < 0.40 m or tilt > 45 deg)

Disturbance = a real external wrench on the torso link for IMPULSE_S seconds.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PLANE = "urdf/plane/plane.urdf"
SPAWN_Z = 0.80
IMPULSE_S = 0.25
SETTLE_S = 3.0          # stand this long before the push
AFTER_S = 8.0           # observation window after the push
NOMINAL = 0.782         # the height the intact policy holds (measured, M1)
TILT_OK = math.radians(5.0)
TILT_FALL = math.radians(45.0)
Z_FALL = 0.40


def tilt(quat):
    w, x, y, z = quat.tolist()
    up = 1 - 2 * (x * x + y * y)
    return math.acos(max(-1.0, min(1.0, up)))


def one_trial(fmag: float, deg: float):
    import torch
    import genesis as gs
    from robots.g1_stand import (G1Stand, G1StandPolicy, policy_params, load_env_args,
                                URDF, SIM_DT, DECIMATION)
    from core.timing import SUBSTEPS

    gs.init(backend=gs.cpu, precision="32", logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=SIM_DT, substeps=SUBSTEPS),
        rigid_options=gs.options.RigidOptions(
            dt=SIM_DT, enable_collision=True, enable_joint_limit=True,
            enable_self_collision=False,
            constraint_solver=gs.constraint_solver.Newton, tolerance=1e-5),
        show_viewer=False)
    scene.add_entity(gs.morphs.URDF(file=PLANE, fixed=True))
    ent = scene.add_entity(gs.morphs.URDF(file=str(URDF), pos=(0, 0, SPAWN_Z), fixed=False))
    scene.build()

    g1 = G1Stand(ent, policy_params(load_env_args()), G1StandPolicy())
    torso = ent.get_link("torso_link")
    fx, fy = fmag * math.cos(math.radians(deg)), fmag * math.sin(math.radians(deg))
    force = torch.tensor([[fx, fy, 0.0]])

    steps_settle = int(SETTLE_S / SIM_DT)
    steps_impulse = int(IMPULSE_S / SIM_DT)
    steps_after = int(AFTER_S / SIM_DT)

    hs, ts, ds = [], [], []
    p0 = None
    fell = False
    rec_t = None
    ok_since = None

    for s in range(steps_settle + steps_impulse + steps_after):
        if s % DECIMATION == 0:
            g1.policy_step()
        g1.apply_torque()
        if steps_settle <= s < steps_settle + steps_impulse:
            torso.apply_external_force(force)      # renewed every sim step
        scene.step()

        pos, quat = g1.base()
        h, th = float(pos[2]), tilt(quat)
        if s == steps_settle - 1:
            p0 = (float(pos[0]), float(pos[1]))
        hs.append(h)
        ts.append(th)

        after = s >= steps_settle + steps_impulse
        if after and p0 is not None:
            d = math.hypot(float(pos[0]) - p0[0], float(pos[1]) - p0[1])
            ds.append(d)
            if h < Z_FALL or th > TILT_FALL:
                fell = True
            # recovery: nominal height +-3 %, tilt < 5 deg, held 2 s
            good = abs(h - NOMINAL) < 0.03 * NOMINAL and th < TILT_OK
            ok_since = (s * SIM_DT) if (good and ok_since is None) else (None if not good else ok_since)
            if ok_since is not None and rec_t is None and (s * SIM_DT) - ok_since >= 2.0:
                rec_t = round(ok_since - (steps_settle + steps_impulse) * SIM_DT, 2)

    settle_h = hs[:steps_settle]
    return {
        "force_N": fmag, "deg": deg, "impulse_Ns": round(fmag * IMPULSE_S, 1),
        "initial_height_mean": round(sum(settle_h) / len(settle_h), 4),
        "initial_height_std": round((sum((h - sum(settle_h) / len(settle_h)) ** 2
                                         for h in settle_h) / len(settle_h)) ** 0.5, 5),
        "max_displacement_m": round(max(ds), 3) if ds else 0.0,
        "max_tilt_deg": round(max(ts) * 180 / math.pi, 1),
        "recovery_s": rec_t,
        "final_height_m": round(hs[-1], 3),
        "fell": fell,
        "recovered": (rec_t is not None) and not fell,
    }


def sweep():
    grid = [(40, 0), (60, 0), (70, 0), (80, 0), (60, 90), (70, 90), (80, 90)]
    rows = []
    print(f"=== Experiment B: G1 disturbance recovery ({len(grid)} trials, "
          f"{IMPULSE_S}s impulse, frozen policy) ===\n")
    for f, d in grid:
        out = subprocess.run([sys.executable, __file__, "--trial", str(f), str(d)],
                             capture_output=True, text=True)
        line = [ln for ln in out.stdout.splitlines() if ln.startswith("TRIAL_JSON ")]
        if not line:
            print(f"  force {f:3d} N @ {d:2d} deg -> TRIAL FAILED (no result)")
            print("   ", (out.stderr or out.stdout)[-300:].replace("\n", " "))
            continue
        r = json.loads(line[0][len("TRIAL_JSON "):])
        rows.append(r)
        print(f"  {r['force_N']:3.0f} N @ {r['deg']:2.0f} deg ({r['impulse_Ns']:4.1f} Ns): "
              f"disp {r['max_displacement_m']:.2f} m  tilt {r['max_tilt_deg']:4.1f} deg  "
              f"recovery {str(r['recovery_s']):>5}s  final {r['final_height_m']:.3f} m  "
              f"{'FELL' if r['fell'] else ('RECOVERED' if r['recovered'] else 'unsettled')}")
    OUT = ROOT / "results"
    OUT.mkdir(exist_ok=True)
    (OUT / "m2_g1_disturbance.json").write_text(json.dumps(rows, indent=2))
    ok = sum(r["recovered"] for r in rows)
    print(f"\n  recovered {ok}/{len(rows)} trials | envelope stored in results/m2_g1_disturbance.json")
    return 0 if ok == len(rows) else (0 if ok > 0 else 1)


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        sys.exit(sweep())
    i = sys.argv.index("--trial")
    r = one_trial(float(sys.argv[i + 1]), float(sys.argv[i + 2]))
    print("TRIAL_JSON " + json.dumps(r))
