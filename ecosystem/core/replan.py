#!/usr/bin/env python3
"""replan.py — M4: closed-loop route validation and replanning (M4.1).

M3's reroute was OPEN-LOOP: detect blocker -> plan 3 waypoints -> execute.
If the blocker moves after the plan, the remaining route may be invalid and
nothing notices. This module closes the loop:

  validate_route   every remaining leg checked against every other robot's
                   cylinder envelope (the same upright-cylinder model as
                   core/proximity.py), using SEGMENT-to-center distances (not
                   point checks — a leg can graze an envelope between ticks)
  plan_ladder      the N-blocker candidate ladder: both sides x lateral
                   offsets, the escape leg blended with the heading
  pick_route_ladder  validates each candidate from the CURRENT position and
                   returns the first valid one — or nothing (the honest
                   NO_GEOMETRIC_ROUTE input)
  (the M4-era plan_route/plan_candidates/pick_route trio was REMOVED in the
   M5 audit response: it was dead code superseded by the ladder, carried a
   divergent escape-length formula (the audit's F3), and would have been
   mistaken for live geometry — the audit's F2 lesson)

This is still NOT general-purpose dynamic obstacle avoidance: no velocity
obstacles, no trajectory prediction, no online sensing of unmapped objects.
It validates and replans against the robots the proximity layer can see.
"""
from __future__ import annotations

import math
import time

from core.proximity import RADIUS, SAFE_GAP


def _point_segment_dist(px, py, ax, ay, bx, by) -> float:
    """2D distance from point p to segment a-b."""
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab2 = abx * abx + aby * aby
    if ab2 < 1e-12:
        return math.hypot(apx, apy)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


def segment_clear(prox, robot: str, a, b, margin: float = 0.0):
    """True if segment a->b stays outside every other robot's envelope.

    Returns (True, None) or (False, [other, ...]) with the violated partners.
    Vertical spans must overlap for a pair to count (the drone above the G1 is
    not a conflict — the same rule as proximity.gaps)."""
    hits = []
    p_a = prox.positions()
    others = {k: v for k, v in p_a.items() if k != robot}
    z0, z1 = prox._span(robot, [a[0], a[1], a[2] if len(a) > 2 else 0.0])
    for o, p in others.items():
        oz0, oz1 = prox._span(o, p)
        if not (z0 < oz1 and oz0 < z1):
            continue
        d_start = math.hypot(a[0] - p[0], a[1] - p[1])
        d = _point_segment_dist(p[0], p[1], a[0], a[1], b[0], b[1])
        gap = d - prox.radius_of(robot) - prox.radius_of(o)
        if gap >= prox.safe_gap + margin:
            continue                     # comfortably clear
        # SHALLOW-DIP ESCAPE (mirrors the gate's per-tick escape clause): the
        # gate permits a transit whose local moves never worsen the gap; at
        # the segment level that reads as "the leg may dip below the better
        # endpoint distance, but not deeply". M3's successful reroute
        # transited at a 0.054 surface gap with a 0.17 dip; a straight-through
        # line dips 0.90. The 0.20-m dip tolerance separates them.
        d_end = math.hypot(b[0] - p[0], b[1] - p[1])
        dip = max(d_start, d_end) - d
        if gap > 0.02 and dip <= 0.20:
            continue                     # a shallow-dip escape transit
        hits.append(o)
    return (len(hits) == 0), hits


def validate_route(prox, robot: str, pos, wps, final, margin: float = 0.0):
    """Validate every remaining leg (pos->wp0->wp1->...->final).

    Returns (True, None) or (False, [violated partners on the first bad leg])."""
    a = [float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) > 2 else 0.0]
    for wp in list(wps) + [final]:
        ok, hits = segment_clear(prox, robot, a, [wp[0], wp[1], a[2]], margin)
        if not ok:
            return False, hits
        a = [float(wp[0]), float(wp[1]), a[2]]
    return True, None




def plan_ladder(pos, crate, blockers, heading, offsets=(0.9, 1.4, 2.0, 2.8)):
    """M5.2: the N-BLOCKER candidate ladder.

    For each side (+1 north-of-line / -1 south) and each lateral offset from
    the dog->crate line, a candidate route [escape?, detour, crate]. The
    escape leg fires when the dog is inside the nearest blocker's combined
    envelope, blended with the current heading (the M3 lesson: the first turn
    must stay within the creep arc's effective range — the heading-weighted
    blend keeps the target inside the arc's verified +-150 deg range). No
    hardcoded blocker count: `blockers` is a list of (x, y) positions, and
    every candidate is validated against ALL of them by the caller."""
    candidates = []
    hx, hy = math.cos(heading), math.sin(heading)
    for side in (1.0, -1.0):
        for off in offsets:
            wps = []
            if blockers:
                nearest = min(blockers, key=lambda b: math.hypot(b[0] - pos[0], b[1] - pos[1]))
                cur = math.hypot(nearest[0] - pos[0], nearest[1] - pos[1])
                from core.proximity import RADIUS, SAFE_GAP
                need = RADIUS["go2"] + 0.30 + SAFE_GAP + 0.15
                if cur < need:
                    ax, ay = pos[0] - nearest[0], pos[1] - nearest[1]
                    an = math.hypot(ax, ay) + 1e-9
                    away = (ax / an, ay / an)
                    # the heading-weighted blend: a target >150 deg off the
                    # heading exceeds the creep arc's verified range and the
                    # pivot wobbles at +-pi forever (the S2 boundary stall)
                    blend = (away[0] * 0.35 + hx * 0.65, away[1] * 0.35 + hy * 0.65)
                    bn = math.hypot(blend[0], blend[1]) + 1e-9
                    esc_len = need - cur + 0.55
                    wps.append([pos[0] + blend[0] / bn * esc_len,
                                pos[1] + blend[1] / bn * esc_len, 0.0])
            # the lateral detour: the midpoint + the perpendicular of the
            # dog->crate line, on `side`, at this ladder offset
            lx, ly = crate[0] - pos[0], crate[1] - pos[1]
            Ll = math.hypot(lx, ly) + 1e-9
            perp = (-ly / Ll, lx / Ll)
            mx, my = (pos[0] + crate[0]) / 2, (pos[1] + crate[1]) / 2
            wps.append([mx + perp[0] * side * off, my + perp[1] * side * off, 0.0])
            wps.append([crate[0], crate[1], 0.0])
            candidates.append({"side": side, "offset": off, "wps": wps})
    return candidates


def pick_route_ladder(prox, robot: str, pos, crate, blockers, heading):
    """M5.2: generate the ladder, validate each candidate against ALL
    blockers, return (route, meta, latency) for the first valid — or
    (None, None, latency) when the ladder is exhausted (the honest
    NO_GEOMETRIC_ROUTE diagnosis input)."""
    t0 = time.perf_counter()
    for cand in plan_ladder(pos, crate, blockers, heading):
        wps = cand["wps"][:-1]  # the last element is the crate itself
        ok, _ = validate_route(prox, robot, pos, wps, [crate[0], crate[1]])
        if ok:
            return wps, {"side": cand["side"], "offset": cand["offset"]}, \
                time.perf_counter() - t0
    return None, None, time.perf_counter() - t0
