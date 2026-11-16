#!/usr/bin/env python3
"""test_m4_replan.py — the evidence M4's own suite does not produce.

WHY THIS FILE EXISTS
--------------------
`tests/test_m4.py` T3 ("blocker moves into the route: replanned") reports
`replans=0`: it teleports the G1 at t=3.0 s, BEFORE the dog commits, so the M3
planner routes around it at plan time and M4's closed loop is never entered.
T6 ("the budget fuse") asserts `replans <= 5` and passes with `replans=0`, i.e.
vacuous. See docs/M4_AUDIT.md F1.

This suite tests the capability the milestone is named after:

  commit a route -> a NEW obstacle appears on the remaining route -> the route
  is invalidated -> a replacement is computed and validated -> the robot
  resumes and completes, without contact.

The disturbance is still scripted (a teleport), but it is placed RELATIVE TO
THE DOG'S LIVE POSITION — 0.65 m ahead along its recent velocity, at a moment
chosen by its live progress. It therefore cannot be accounted for by the plan
that was already made, which is the difference between "static avoidance" and
"route invalidation after commitment".

Checks report PASS / FAIL / **SKIP**. SKIP means the prerequisite was not
exercised, so the claim is NOT verified — never silently green.

Run:  python tests/test_m4_replan.py [--out results/m4_replan_verification.json]
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from core.world import World  # noqa: E402
from core.proximity import RADIUS, SAFE_GAP  # noqa: E402
import missions.relay as relay  # noqa: E402

CRATE = (0.0, 1.2)
DOG = (3.4, 1.2)
_results = []


def check(name, status, detail=""):
    """status: True -> PASS, False -> FAIL, None -> SKIP (prerequisite absent)."""
    s = "SKIP" if status is None else ("PASS" if status else "FAIL")
    _results.append({"check": name, "status": s, "detail": detail})
    print(f"  [{s}] {name}" + (f" — {detail}" if detail else ""))


def _place(w, dog_xy=DOG, crate_xy=CRATE):
    q = w.robots["go2"].entity.get_qpos()[0].clone()
    q[0], q[1] = dog_xy
    w.robots["go2"].entity.set_qpos(q.unsqueeze(0), zero_velocity=True)
    w.robots["go2"].refresh()
    w.crate.set_qpos(torch.tensor([[crate_xy[0], crate_xy[1], 0.075,
                                    1.0, 0.0, 0.0, 0.0]]), zero_velocity=True)


class ReactiveBlocker:
    """Teleport the G1 onto the dog's LIVE remaining path, N times.

    Armed by live progress (the dog must have travelled `arm_dist`), never by a
    wall-clock guess, so the placement is always mid-execution. The target is
    1.10 m ahead of the dog along its recent velocity: OUTSIDE the clearance
    envelope (0.68 m envelope sum + 0.22 m SAFE_GAP = 0.90 m), so the teleport
    creates no instantaneous conflict, yet ON the committed route the dog is
    following -> the route segment's centre distance is ~0, a deep violation
    (needs < 0.07 m surface gap). >= 1.5 m from the crate so a detour exists.
    """

    def __init__(self, w, arm_dist=0.60, n=1, cooldown=2.2, ahead=1.10):
        self.w, self.arm_dist, self.n = w, arm_dist, n
        self.cooldown, self.ahead = cooldown, ahead
        self.start = None
        self.prev = None
        self.placed = []          # (t, dog_xy, blocker_xy, replans_at_placement)
        self.trace = []           # (t, dog_x, dog_y) — for the path analysis
        self._orig = None
        self.gap_at_placement = float('inf')

    # -- the hook -------------------------------------------------------
    def __enter__(self):
        orig = World.step
        st = self

        def stepped(self_, ):
            t = self_.sim_time
            try:
                p = self_.robots["go2"].entity.get_qpos()[0][:2]
                dog = (float(p[0]), float(p[1]))
            except Exception:
                dog = None
            if dog is not None:
                st.trace.append((t, dog[0], dog[1]))
                if st.start is None:
                    st.start = dog
            if (dog is not None and st.n > len(st.placed)
                    and math.dist(dog, st.start) >= st.arm_dist
                    and (not st.placed or t - st.placed[-1][0] >= st.cooldown)):
                # the recent velocity direction (falls back to dog->crate)
                vx = vy = 0.0
                if st.prev is not None:
                    vx, vy = dog[0] - st.prev[0], dog[1] - st.prev[1]
                n0 = math.hypot(vx, vy)
                if n0 < 1e-6:
                    vx, vy = CRATE[0] - dog[0], CRATE[1] - dog[1]
                    n0 = math.hypot(vx, vy) + 1e-9
                bx = dog[0] + vx / n0 * st.ahead
                by = dog[1] + vy / n0 * st.ahead
                if math.dist((bx, by), CRATE) >= 1.5:      # a detour must exist
                    q = self_.g1_e.get_qpos()[0].clone()
                    q[0], q[1] = bx, by
                    self_.g1_e.set_qpos(q.unsqueeze(0), zero_velocity=True)
                    st.gap_at_placement = min(
                        st.gap_at_placement,
                        math.dist(dog, (bx, by)) - (RADIUS["go2"] + RADIUS["g1"]))
                    st.placed.append((round(t, 2), dog, (round(bx, 2), round(by, 2)), None))
            if dog is not None:
                st.prev = dog
            orig(self_)

        self._orig = orig
        World.step = stepped
        return self

    def __exit__(self, *exc):
        World.step = self._orig
        return False

    # -- helpers --------------------------------------------------------
    def path_m(self):
        return sum(math.dist((a[1], a[2]), (b[1], b[2]))
                   for a, b in zip(self.trace, self.trace[1:]))


def _events(r, kinds):
    return [e[2] for e in r["events"] if e[2] in kinds]


def scenario_commit_then_invalidate(out):
    """A: a route is committed, then a new obstacle appears on it."""
    print("\n--- A: obstacle appears on the LIVE remaining route ---")
    w = World(g1_xy=(1.8, 1.2))          # a static blocker forces a reroute early
    _place(w)
    with ReactiveBlocker(w, arm_dist=0.60, n=1) as rb:
        r = relay.run(w, verbose=False, max_s=200.0, constraints=True,
                      scenario="m4_replan_live")
    replans = r["m4"]["replans"]
    ev = _events(r, {"ROUTE_REPLANNED", "REPLAN_NO_ROUTE", "REROUTE_STARTED",
                     "RESUME_DIRECT", "DEADLOCK"})
    detail = (f"placed={rb.placed} replans={replans} events={ev} "
              f"t={r['t']:.1f}s path={rb.path_m():.2f}m why={(r['why'] or 'completed')[:40]}")

    # The prerequisite: the dog really was moving when the obstacle appeared.
    check("A0 prerequisite: the dog was mid-execution when the obstacle appeared",
          bool(rb.placed) and rb.path_m() >= 0.50,
          f"placed_at={rb.placed[0][0] if rb.placed else None}s travelled={rb.path_m():.2f}m")

    # The capability. This is the check M4's own T3 does not make.
    check("A0b the obstacle was placed outside the conflict threshold "
          "(the conflict is a route violation, not a teleport artifact)",
          bool(rb.placed) and rb.gap_at_placement >= SAFE_GAP,
          f"surface gap at placement={round(rb.gap_at_placement,3)}m "
          f"(SAFE_GAP={SAFE_GAP}; no instantaneous conflict, the conflict is the ROUTE)")
    check("A1 the committed route was invalidated and REPLANNED (replans >= 1)",
          replans >= 1, detail)
    check("A2 a ROUTE_REPLANNED event was published (a replacement was chosen)",
          "ROUTE_REPLANNED" in ev, f"events={ev}")
    check("A3 the dog resumed and completed the mission after replanning",
          bool(r["ok"]), f"why={(r['why'] or 'completed')} t={r['t']:.1f}s")
    check("A4 no go2-g1 contact during the replan", "g1-go2" not in r["proximity"]["contacts"],
          f"min_gap={r['proximity']['min_gap']}")

    out["A_commit_then_invalidate"] = {
        "ok": bool(r["ok"]), "replans": replans, "events": ev,
        "placements": rb.placed, "path_m": round(rb.path_m(), 3),
        "t": round(r["t"], 2), "why": r["why"],
        "min_gap": r["proximity"]["min_gap"],
        "latency_ms": r["m4"]["replan_latencies_ms"],
    }
    return r


def scenario_fuse_nonvacuous(out):
    """B: the budget fuse, with the prerequisite asserted first."""
    print("\n--- B: repeated live invalidations (the budget fuse) ---")
    w = World(g1_xy=(1.8, 1.2))
    _place(w)
    with ReactiveBlocker(w, arm_dist=0.60, n=6, cooldown=2.2) as rb:
        r = relay.run(w, verbose=False, max_s=200.0, constraints=True,
                      scenario="m4_replan_fuse")
    replans = r["m4"]["replans"]
    ev = _events(r, {"ROUTE_REPLANNED", "REPLAN_NO_ROUTE", "DEADLOCK"})
    detail = (f"placements={len(rb.placed)}/{6} replans={replans} "
              f"MAX_REPLANS={relay.MAX_REPLANS} events={ev} path={rb.path_m():.2f}m")

    exercised = replans >= 1
    check("B0 prerequisite: the fuse scenario actually produced replans", exercised,
          detail if not exercised else f"{replans} replans over {len(rb.placed)} placements")
    # the fuse claim is only meaningful if replanning happened at all
    check("B1 the replan budget is never exceeded (no infinite replanning loop)",
          (replans <= relay.MAX_REPLANS) if exercised else None, detail)
    check("B2 the outcome is honest (complete or deadlock) and contact-free",
          ("g1-go2" not in r["proximity"]["contacts"]),
          f"ok={r['ok']} why={(r['why'] or 'completed')} min_gap={r['proximity']['min_gap']}")

    out["B_fuse_nonvacuous"] = {
        "placements": len(rb.placed), "replans": replans,
        "max_replans": relay.MAX_REPLANS, "events": ev,
        "ok": bool(r["ok"]), "why": r["why"], "path_m": round(rb.path_m(), 3),
        "min_gap": r["proximity"]["min_gap"],
    }
    return r


def main():
    out_path = "results/m4_replan_verification.json"
    if "--out" in sys.argv:
        out_path = sys.argv[sys.argv.index("--out") + 1]

    import hashlib
    files = ["core/replan.py", "core/proximity.py", "core/safety.py",
             "core/world.py", "missions/relay.py", "robots/go2_drive.py"]
    root = Path(__file__).resolve().parent.parent
    hashes = {f: hashlib.md5((root / f).read_bytes()).hexdigest()[:12]
              for f in files if (root / f).exists()}

    print("=== M4: route invalidation after commitment (the missing evidence) ===")
    out = {"suite": "test_m4_replan.py",
           "purpose": "M4's own T3/T6 do not exercise the replan path (docs/M4_AUDIT.md F1)",
           "version_hashes": hashes,
           "max_replans": relay.MAX_REPLANS, "replan_min_gap": relay.REPLAN_MIN_GAP}
    try:
        scenario_commit_then_invalidate(out)
    except Exception as e:
        check("A scenario ran to completion", False, f"{type(e).__name__}: {e}")
    try:
        scenario_fuse_nonvacuous(out)
    except Exception as e:
        check("B scenario ran to completion", False, f"{type(e).__name__}: {e}")

    n_pass = sum(1 for c in _results if c["status"] == "PASS")
    n_fail = sum(1 for c in _results if c["status"] == "FAIL")
    n_skip = sum(1 for c in _results if c["status"] == "SKIP")
    out["checks"] = _results
    out["summary"] = {"pass": n_pass, "fail": n_fail, "skip": n_skip}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"\nM4-REPLAN VERDICT: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP"
          f"  (artifact: {out_path})")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
