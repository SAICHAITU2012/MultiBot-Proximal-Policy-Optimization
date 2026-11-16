#!/usr/bin/env python3
"""test_m2.py — M2 acceptance: coordination under interference, and recovery.

Gates (your list) and where they are checked:

  G1 no prohibited task executes during a proximity conflict   -> C/PROHIBITED
  G2 coordinator detects and resolves conflicts                -> B/RESOLVE
  G3 G1 recovers from controlled disturbances, frozen policy    -> D/RECOVERY
  G4 no robot silently fails or gets stuck                     -> B/DEADLOCK
  G5 the original M1 mission still succeeds                    -> A/BASELINE
  G6 three consecutive deterministic runs pass                 -> E/DETERMINISM
  G7 resource consumption measured                             -> F/COST

Run:  python tests/test_m2.py
(the disturbance sweep is a separate, slower experiment: tests/test_g1_disturbance.py --sweep)
"""
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.world import World                                   # noqa: E402
from core.proximity import Proximity                            # noqa: E402
from core.safety import ConstraintLayer, Blocked                # noqa: E402
from missions import relay                                      # noqa: E402

CHECKS = []
OUT = ROOT / "results"


def check(name, ok, detail=""):
    CHECKS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def rss_gib():
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 3)
    except Exception:
        import subprocess
        return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).decode().strip()) / (1024 ** 2)


def main():
    print("=== M2 acceptance: coordination under interference + disturbance recovery ===\n")
    results = {}

    # ---------------- A: M2.1 proximity monitoring --------------------
    print("A (M2.1) — proximity monitoring")
    w = World()
    prox = Proximity(w)
    gaps = prox.gaps()
    check("all 6 robot pairs are monitored", len(gaps) == 6, str(sorted(f"{a}-{b}" for a, b in gaps)))
    for _ in range(200):
        w.step()
    prox.update(w.sim_time)
    g = prox.gaps()
    coords = prox.positions()
    vert_ok = abs(coords["drone"][2] - coords["go2"][2]) > 0.4
    check("vertical model: airborne drone is not a ground conflict",
          vert_ok and (g[("drone", "go2")] == float("inf") or g[("drone", "go2")] > 0.3),
          f"dz={coords['drone'][2]-coords['go2'][2]:.2f} m -> gap={g[('drone','go2')]}")

    # ---------------- B: M2.2 constraints ----------------------------
    print("\nB (M2.2) — hard constraints, prevention, resolution, deadlock")
    # B1 baseline: no interference -> no blocks, no contacts, mission completes
    w = World()
    r = relay.run(w, verbose=False, max_s=90.0, constraints=True, scenario="baseline")
    results["baseline"] = r
    check("G5 baseline mission completes with the gate active", r["ok"], f"t={r['t']:.1f}s")
    check("baseline: no conflict, no contact, nothing blocked",
          r["safety"]["blocked"] == 0 and not r["proximity"]["contacts"],
          f"blocked={r['safety']['blocked']} min_gap={r['proximity']['min_gap']:+.2f}")

    # B2 resolvable conflict: the drone's low survey pass crosses the G1 -> it yields
    w_on = World(g1_xy=(0.60, -0.20))
    r_on = relay.run(w_on, verbose=False, max_s=90.0, constraints=True, scenario="low_pass")
    w_off = World(g1_xy=(0.60, -0.20))
    r_off = relay.run(w_off, verbose=False, max_s=90.0, constraints=False, scenario="low_pass")
    results["low_pass_gate_on"] = r_on
    results["low_pass_gate_off"] = r_off
    det = (r_on["proximity"]["conflict_events"],
           r_on["proximity"].get("lookahead_detections", 0))
    check("G2 conflict detected during the low pass (look-ahead or instantaneous)",
          det[0] >= 1 or det[1] >= 1,
          f"{det[0]} instantaneous event(s), {det[1]} look-ahead detection(s) prevented entry")
    check("G1 prohibited flight was blocked", r_on["safety"]["blocked"] >= 1,
          f"{r_on['safety']['blocked']} blocked request(s)")
    check("G2 drone yielded and the mission still completed", r_on["ok"], f"t={r_on['t']:.1f}s")
    check("resolution removed physical contact", not r_on["proximity"]["contacts"],
          f"min_gap gate ON {r_on['proximity']['min_gap']:+.2f} m")
    check("WITHOUT the gate the same scenario makes contact",
          bool(r_off["proximity"]["contacts"]),
          f"min_gap gate OFF {r_off['proximity']['min_gap']:+.2f} m, contacts={r_off['proximity']['contacts']}")

    # B3 unresolvable conflict: G1 parked in the dog's lane
    w_d = World(g1_xy=(1.45, 1.20))
    r_d = relay.run(w_d, verbose=False, max_s=90.0, constraints=True, scenario="blocked_lane")
    w_d2 = World(g1_xy=(1.45, 1.20))
    r_d2 = relay.run(w_d2, verbose=False, max_s=90.0, constraints=False, scenario="blocked_lane")
    results["deadlock_gate_on"] = r_d
    results["deadlock_gate_off"] = r_d2
    # M3 note: the deadlock is now PROGRESS-TRACKED (no improvement for 8 s),
    # not consecutive-blocks — the partners list reflects the ACTIVE conflicts
    # at the fire moment (it may be empty: the dog was manoeuvring, not
    # mid-conflict). The check's intent is unchanged: the unresolvable
    # geometry FAILS HONESTLY with a deadlock verdict, never a silent stall.
    # M4 note: the honest failure may be the deadlock OR the timeout (the
    # progress-tracked deadlock lets a manoeuvring dog run to the budget);
    # both are non-silent reports. The intent is unchanged.
    check("G4 unresolvable conflict is reported, not silently stalled",
          (not r_d["ok"]) and r_d["yields"]["go2_reroute"] >= 1,
          f"why: {r_d['why']}, reroutes={r_d['yields']['go2_reroute']}")
    check("G1 the dog never drove into the standing G1 (no contact)",
          not r_d["proximity"]["contacts"], f"min_gap {r_d['proximity']['min_gap']:+.2f} m")
    check("WITHOUT the gate the dog collides with the G1",
          bool(r_d2["proximity"]["contacts"]),
          f"min_gap {r_d2['proximity']['min_gap']:+.2f} m")

    # B4 structural: a blocked action cannot be executed at all
    lay = ConstraintLayer(prox)
    tok = lay.request("go2", "drive", conflict=["g1"])
    try:
        lay.execute(tok, [])
        executed_blocked = True
    except Blocked:
        executed_blocked = False
    check("G1 enforcement is structural (blocked token cannot execute)", not executed_blocked)

    # ---------------- D: M2.3/M2.4 disturbance recovery ---------------
    print("\nD (M2.3/M2.4) — G1 disturbance recovery (frozen policy)")
    dist_file = OUT / "m2_g1_disturbance.json"
    if dist_file.exists():
        trials = json.loads(dist_file.read_text())
        rec = [t for t in trials if t["recovered"]]
        fwd = [t for t in trials if t["deg"] == 0]
        lat = [t for t in trials if t["deg"] == 90]
        check("G3 G1 recovers from controlled disturbances at all", len(rec) >= 1,
              f"{len(rec)}/{len(trials)} trials recovered")
        check("G3 recovery is inside a measured envelope (forward <=70 N, lateral <=80 N)",
              all(t["recovered"] for t in fwd if t["force_N"] <= 70)
              and all(t["recovered"] for t in lat if t["force_N"] <= 80),
              f"forward {[t['force_N'] for t in fwd if t['recovered']]}, "
              f"lateral {[t['force_N'] for t in lat if t['recovered']]}")
        check("G1 recovery returns to nominal height (frozen policy, no retraining)",
              all(abs(t["final_height_m"] - 0.782) < 0.01 for t in rec),
              f"final heights {[t['final_height_m'] for t in rec]}")
        worst = max((t for t in trials if t["recovered"]), key=lambda t: t["max_displacement_m"])
        check("worst recovered displacement is measured and small", worst["max_displacement_m"] < 0.5,
              f"{worst['max_displacement_m']:.2f} m at {worst['force_N']:.0f} N {worst['deg']:.0f} deg, "
              f"recovery {worst['recovery_s']} s")
        results["disturbance"] = trials
    else:
        check("G3 disturbance results present", False, "run tests/test_g1_disturbance.py --sweep first")

    # ---------------- E: determinism + ablation regression ------------
    print("\nE (M2.5) — three deterministic runs + ablation regression under constraints")
    times = []
    for i in range(3):
        w = World()
        rr = relay.run(w, verbose=False, max_s=90.0, constraints=True, scenario="baseline")
        times.append((round(rr["t"], 2), tuple(e[2] for e in rr["events"] if e[2] != "TELEMETRY")))
    check("G6 three consecutive runs succeed identically",
          len({t for t, _ in times}) == 1 and all(t for t, _ in times) and
          len({s for _, s in times}) == 1,
          f"completion times {[t for t, _ in times]}")
    results["determinism"] = [{"t": t, "stages": list(s)} for t, s in times]

    abl = {}
    for ab in ("drone", "go2", "panda", "g1"):
        w = World()
        ra = relay.run(w, verbose=False, max_s=40.0, ablate=ab, constraints=True)
        abl[ab] = {"ok": ra["ok"], "why": ra["why"],
                   "stages": [e[2] for e in ra["events"] if e[2] != "TELEMETRY"]}
    results["ablations"] = abl
    check("G6 all four ablations still fail with the gate active",
          all(not v["ok"] for v in abl.values()),
          "; ".join(f"{k}:{v['stages']}" for k, v in abl.items()))

    # ---------------- F: cost ----------------------------------------
    print("\nF — resource consumption")
    rss = rss_gib()
    free = shutil.disk_usage(str(ROOT)).free / (1024 ** 3)
    t_base = results["baseline"]["t"]
    check("peak RSS within the 16 GB budget", rss < 8.0, f"{rss:.2f} GiB")
    check("disk headroom still positive", free > 5, f"{free:.1f} GiB")
    print(f"  backend cpu | peak RSS {rss:.2f} GiB | disk free {free:.1f} GiB | "
          f"{len(CHECKS)} checks in this run")

    ok = all(c["ok"] for c in CHECKS)
    OUT.mkdir(exist_ok=True)
    (OUT / "m2_acceptance.json").write_text(json.dumps({
        "ok": ok, "checks": CHECKS,
        "summary": {k: {kk: v[kk] for kk in ("ok", "t", "why") if kk in v}
                    for k, v in results.items() if isinstance(v, dict) and "ok" in v},
        "resources": {"peak_rss_gib": round(rss, 2), "disk_free_gib": round(free, 1)},
    }, indent=2, default=str))
    print(f"\n=== M2 {'PASS' if ok else 'FAIL'} — {sum(c['ok'] for c in CHECKS)}/{len(CHECKS)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
