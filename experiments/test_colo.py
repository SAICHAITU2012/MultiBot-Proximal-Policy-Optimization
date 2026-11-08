"""Stage 5 smoke test: three policies (go2 walk + drone hover + panda hold) in one scene.

Runs 200 shared scene steps (2.0 s sim) with both trained policies active and
the panda holding its home pose.  Prints per-robot sanity every 50 steps and a
final PASS/FAIL verdict.

Pass criteria:
  * no NaN/Inf in any robot state at any step
  * go2 keeps standing/walking: base z in [0.15, 0.60] throughout
  * drone flies stably: max dist-to-target < 1.0 m, final < 0.3 m
  * panda stays sane: EE within 1.0 m of its base, all qpos finite

Usage:
  python test_colo.py [--steps 200] [--vis]
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import genesis as gs
from colo_env import CoLocatedTeam


def finite(state):
    def _walk(v):
        if isinstance(v, (list, tuple)):
            return all(_walk(x) for x in v)
        if isinstance(v, float):
            return math.isfinite(v)
        return True
    return all(_walk(v) for v in state.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--vis", action="store_true")
    args = p.parse_args()

    gs.init(backend=gs.cpu, precision="32", logging_level="warning", performance_mode=True)

    print("[test] building co-located team (go2 + drone + panda)...")
    team = CoLocatedTeam(show_viewer=args.vis)
    print("[test] robots:", {n: type(r).__name__ for n, r in team.robots.items()})
    for n, r in team.robots.items():
        assert r.policy is not None or n == "panda", f"{n} policy missing"
    print("[test] go2 actor loaded (45->512->256->128->12, elu); "
          "drone actor loaded (17->128->128->4, tanh); panda policy=None (hold pose)")

    history = {"go2_z": [], "drone_dist": [], "panda_ee_err": []}
    ok = True

    with torch.inference_mode():
        for t in range(args.steps):
            team.step()
            states = team.get_states()

            # NaN check on everything, every step
            for name, st in states.items():
                if not finite(st):
                    print(f"[test] FAIL: NaN/Inf in {name} state at step {t}: {st}")
                    ok = False

            go2_z = states["go2"]["base_z"]
            drone_dist = states["drone"]["dist_to_target"]
            panda_base = states["panda"]["base_pos"]
            panda_ee = states["panda"]["ee_pos"]
            ee_err = math.dist(panda_ee, panda_base)
            history["go2_z"].append(go2_z)
            history["drone_dist"].append(drone_dist)
            history["panda_ee_err"].append(ee_err)

            if (t + 1) % 50 == 0:
                print(
                    f"[t={t + 1:4d} | {team.sim_time:5.2f}s] "
                    f"go2: z={go2_z:.3f} vx={states['go2']['vx_body']:+.3f} | "
                    f"drone: z={states['drone']['pos'][2]:.3f} dist_to_target={drone_dist:.3f} | "
                    f"panda: ee=({panda_ee[0]:+.3f},{panda_ee[1]:+.3f},{panda_ee[2]:.3f}) "
                    f"ee_err={ee_err:.3f}"
                )

    # ---------------- verdict ----------------
    min_z, max_z = min(history["go2_z"]), max(history["go2_z"])
    max_dist, final_dist = max(history["drone_dist"]), history["drone_dist"][-1]
    max_ee_err = max(history["panda_ee_err"])

    checks = {
        "no NaN anywhere": ok,
        f"go2 base z kept in [0.15, 0.60] (actual min={min_z:.3f} max={max_z:.3f})": 0.15 <= min_z and max_z <= 0.60,
        f"drone max dist-to-target < 1.0 m (actual {max_dist:.3f})": max_dist < 1.0,
        f"drone final dist-to-target < 0.3 m (actual {final_dist:.3f})": final_dist < 0.3,
        f"panda EE within 1.0 m of base (actual max {max_ee_err:.3f})": max_ee_err < 1.0,
    }

    print("[test] ---- checks ----")
    for desc, passed in checks.items():
        print(f"[test] {'PASS' if passed else 'FAIL'}: {desc}")

    verdict = all(checks.values())
    print(f"[test] RESULT: {'PASS' if verdict else 'FAIL'}")
    gs.destroy()
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
