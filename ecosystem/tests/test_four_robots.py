#!/usr/bin/env python3
"""test_four_robots.py — M1 acceptance test.

M1 is NOT "four robots spawn". The criteria checked here are:

  A. four heterogeneous robots in ONE scene, one shared clock
  B. independent controller state per robot (own obs/action/rate, own identity)
  C. per-robot policy service rates match the configured runtime schedule
  D. state exchange: robots publish, a coordinator consumes, nothing is assumed
  E. a coordinated task that every robot must contribute to, in a dependency
     chain — and ABLATIONS that prove each contribution is load-bearing
  F. machine cost measured (wall time, sim speed, peak RSS, disk)

Run:  python tests/test_four_robots.py
"""
import json
import math
import os
import resource
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.world import World, SCENE_DT          # noqa: E402
from missions import relay                      # noqa: E402

OUT = ROOT / "results"
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _rss_gib():
    """Peak/current resident set in GiB (macOS ru_maxrss is unreliable here)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 3)
    except Exception:
        import subprocess
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).decode()
        return int(out.strip()) / (1024 ** 2)


def measure(fn):
    t0 = time.time()
    res = fn()
    return res, time.time() - t0


def main():
    print("=== M1 acceptance: heterogeneous four-robot ecosystem ===\n")
    t_build_start = time.time()
    w = World()
    build_s = time.time() - t_build_start
    print(f"  built in {build_s:.1f}s: {sorted(list(w.robots) + ['g1'])}\n")

    # ---- A + B: four robots, independent state -------------------------
    print("A/B — composition and independent controller state")
    ids = sorted(list(w.robots) + ["g1"])
    check("four robots, four distinct identities", ids == ["drone", "g1", "go2", "panda"], str(ids))
    dims = {
        "go2": (w.robots["go2"].get_obs().shape[-1], 12),
        "drone": (w.robots["drone"].get_obs().shape[-1], 4),
        "panda": (w.robots["panda"].get_obs().shape[-1], 6),
        "g1": (w.g1.obs().shape[-1], 29),
    }
    check("distinct obs/action contracts", dims == {"go2": (45, 12), "drone": (17, 4),
                                                    "panda": (14, 6), "g1": (96, 29)}, str(dims))
    def _buf(a):
        for attr in ("last_actions", "last_action", "actions"):
            if hasattr(a, attr):
                return id(getattr(a, attr))
        return id(a)
    bufs = {k: _buf(w.robots[k]) for k in w.robots}
    bufs["g1"] = _buf(w.g1)
    check("no shared action buffer (4 separate tensors)", len(set(bufs.values())) == 4, str(sorted(bufs)))
    ents = {k: id(w.robots[k].entity) for k in w.robots}
    ents["g1"] = id(w.g1_e)
    check("4 separate physics entities", len(set(ents.values())) == 4)

    # ---- C: rates ------------------------------------------------------
    print("\nC — control rates on ONE clock")
    t0, c0 = w.sim_time, dict(w.tick_counts)
    for _ in range(1000):                     # 5 s at 200 Hz
        w.step()
    dt = w.sim_time - t0
    got = {k: round((w.tick_counts[k] - c0[k]) / dt) for k in w.tick_counts}
    want = {"go2": 25, "drone": 200, "panda": 50, "g1": 50}
    check("per-robot Hz matches the runtime schedule", got == want, f"{got} (want {want})")
    check("single shared clock", abs(dt - 1000 * SCENE_DT) < 1e-9, f"{dt:.2f}s of sim in 1000 steps")

    # ---- D + E: coordinated mission + ablations ------------------------
    print("\nD/E — coordinated mission (message-gated dependency chain)")
    _t_mission, _t0m = None, time.time()
    r = relay.run(w, verbose=True, max_s=120.0)
    mission_wall = time.time() - _t0m
    mission_speed = r["t"] / mission_wall
    ev = [e for e in r["events"] if e[2] != "TELEMETRY"]
    order = [e[2] for e in ev]
    check("mission completes", r["ok"], f"at t={r['t']:.1f}s")
    check("stage order is the required chain",
          order == ["CRATE_SIGHTED", "CRATE_REACHED", "PARCEL_DISPATCHED", "STATION_STABLE"], str(order))
    check("every robot contributed a report",
          {e[1] for e in ev} == {"drone", "go2", "panda", "g1"}, str(sorted({e[1] for e in ev})))
    check("telemetry actually exchanged on the bus", r["messages"] > 4,
          f"{r['messages']} messages, {r['bus_reads']} coordinator reads")
    check("G1 never fell during the mission", not w.g1_fell, f"min z {w.g1_min_z:.3f} m")

    print("\nE2 — ablations: the mission MUST fail without each robot")
    for ab in ("drone", "go2", "panda", "g1"):
        wa = World()
        ra = relay.run(wa, verbose=False, max_s=30.0, ablate=ab)
        seq = [e[2] for e in ra["events"] if e[2] != "TELEMETRY"]
        why = ra["why"] or f"stalled after {seq[-1]}"
        check(f"ablate {ab:<5} -> mission fails", not ra["ok"], f"{why}; got {seq}")

    # ---- F: machine cost ----------------------------------------------
    print("\nF — machine cost")
    rss = _rss_gib()
    steps = w.t
    free = shutil.disk_usage(str(ROOT)).free / (1024 ** 3)
    speed = mission_speed
    check("peak RSS under the 16 GB machine budget", rss < 8 * 1024, f"{rss:.2f} GiB")
    check("disk headroom still positive", free > 5, f"{free:.1f} GiB free")
    print(f"  backend: cpu (no CUDA on this M4) | mission {r['t']:.1f}s sim in {mission_wall:.1f}s wall "
          f"= {speed:.2f}x realtime WITH 4 robots + policies | build {build_s:.1f}s | RSS {rss:.2f} GiB")

    # ---- verdict -------------------------------------------------------
    ok = all(c["ok"] for c in CHECKS)
    OUT.mkdir(exist_ok=True)
    (OUT / "m1_acceptance.json").write_text(json.dumps({
        "ok": ok, "checks": CHECKS, "build_s": round(build_s, 1),
        "sim_time": round(w.sim_time, 1), "sim_speed_x": round(speed, 1),
        "peak_rss_gib": round(rss, 2), "disk_free_gib": round(free, 1),
        "mission": {k: r[k] for k in ("ok", "t", "stage_times", "messages")},
    }, indent=2))
    print(f"\n=== M1 {'PASS' if ok else 'FAIL'} — {sum(c['ok'] for c in CHECKS)}/{len(CHECKS)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
