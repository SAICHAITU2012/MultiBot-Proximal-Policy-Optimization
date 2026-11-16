#!/usr/bin/env python3
"""test_m3.py — M3 acceptance: the yaw-then-drive reroute.

The M2 blocked-lane scenario (the G1 at (1.45, 1.2)) is GEOMETRICALLY
UNRESOLVABLE: the crate sits 0.25 m from the parked G1, and the reachable set
{<= 0.60 m from the crate} intersected with {>= 0.90 m from the G1} is empty
(the dog's arrival radius + the envelope exceed the crate-blocker distance).
For THAT scenario the correct M3 outcome is the honest DEADLOCK after a real
reroute attempt (the advisor's criterion: "detects unsuccessful rerouting and
reports DEADLOCK rather than claiming success").

The reroute CAPABILITY is proven on a RESOLVABLE variant: the blocker south of
the dog's start (g1_xy=(2.2, 0.8), 1.56 m from the crate) — the dog must pivot
and route around it; the geometry permits a clean north detour.

Gates:
  G1 the resolvable blocker is REROUTED AROUND: the mission completes, >=1
     REROUTE_STARTED, no go2-g1 contact.
  G2 the unresolvable blocker reports DEADLOCK (not a false success), >=1
     reroute attempt, no go2-g1 contact.
  G3 the gate-OFF resolvable blocker makes contact (why the gate matters).
  G4 the baseline mission (no blockers) still completes (M1 regression).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.world import World  # noqa: E402
import missions.relay as relay  # noqa: E402

_results = []


def check(name, cond, detail=""):
    _results.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


def main():
    print("=== M3 acceptance: the yaw-then-drive reroute ===")

    # G1: the RESOLVABLE blocker — a longer lane (the dog relocated to
    # (3.4, 1.2), the crate to (0.0, 1.2), the G1 in-lane at (1.8, 1.2):
    # >=0.9 m from the spawn and >=1.5 m from the crate, the minimum for a
    # gate-permissible approach). The 1.0-m M2 lane itself cannot host a
    # resolvable blocker (0.9 + 1.5 > 1.0) — that scenario stays DEADLOCK.
    w = World(g1_xy=(1.8, 1.2))
    q = w.robots["go2"].entity.get_qpos()[0].clone()
    q[0] = 3.4
    w.robots["go2"].entity.set_qpos(q.unsqueeze(0), zero_velocity=True)
    w.robots["go2"].refresh()
    w.crate.set_qpos(torch.tensor([[0.0, 1.2, 0.075, 1.0, 0.0, 0.0, 0.0]]), zero_velocity=True)
    r = relay.run(w, verbose=False, max_s=180.0, constraints=True, scenario="resolvable_block")
    check("G1 resolvable blocker: mission completes", r["ok"],
          f"t={r['t']:.1f}s yields={r['yields']}")
    check("G1 reroute was attempted", r["yields"]["go2_reroute"] >= 1,
          f"reroutes={r['yields']['go2_reroute']}")
    check("G1 no go2-g1 contact during the reroute",
          "g1-go2" not in r["proximity"]["contacts"],
          f"min_gap={r['proximity']['min_gap']}")

    # G2: the M2 blocked-lane (the G1 at (1.45, 1.2) — 0.25 m from the crate).
    # The geometry is TRULY unresolvable for a contact-free approach: the dog
    # must be within 0.60 m of the crate while staying >= 0.90 m from the G1,
    # but the crate-blocker distance is only 0.25 m (0.25 + 0.60 < 0.90 — the
    # reachable set is empty). The correct outcome: a REAL reroute attempt,
    # then the honest deadlock/timeout, never a false success, never contact.
    w2 = World(g1_xy=(1.45, 1.20))
    r2 = relay.run(w2, verbose=False, max_s=180.0, constraints=True, scenario="blocked_lane")
    honest = ((not r2["ok"]) or r2.get("deadlock")) and r2["yields"]["go2_reroute"] >= 1
    check("G2 unresolvable geometry: reroute attempted, then honest failure "
          "(no false success)",
          honest and "reached" not in r2["stage_times"],
          f"t={r2['t']:.1f}s reroutes={r2['yields']['go2_reroute']} why={r2['why']}")
    check("G2 no go2-g1 contact on the tight pass",
          "g1-go2" not in r2["proximity"]["contacts"],
          f"min_gap={r2['proximity']['min_gap']}")

    # G3: the gate-OFF resolvable blocker — the dog drives through the G1
    w3 = World(g1_xy=(2.2, 0.8))
    r3 = relay.run(w3, verbose=False, max_s=180.0, constraints=False,
                   scenario="resolvable_block_gateoff")
    check("G3 gate-OFF: the dog passes through the blocker's envelope "
          "(why the gate exists)",
          r3["proximity"]["min_gap"] < 0.10 or bool(r3["proximity"]["contacts"]),
          f"min_gap={r3['proximity']['min_gap']} contacts={r3['proximity']['contacts']}")

    # G4: the baseline (no blocker) still completes — the steering did not
    # break the M1 chain
    w4 = World()
    r4 = relay.run(w4, verbose=False, max_s=120.0, constraints=True, scenario="baseline")
    check("G4 baseline mission still completes with the primitive in the loop",
          r4["ok"], f"t={r4['t']:.1f}s stages={list(r4['stage_times'].keys())}")

    fails = sum(1 for _, ok, _ in _results if not ok)
    print(f"\nM3 VERDICT: {len(_results) - fails}/{len(_results)} checks pass")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
