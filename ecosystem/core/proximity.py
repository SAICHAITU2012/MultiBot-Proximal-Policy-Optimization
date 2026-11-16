#!/usr/bin/env python3
"""proximity.py — M2.1 robot-robot proximity monitoring.

Each robot is modelled as an UPRIGHT CYLINDER (footprint radius + body height).
The gap between a pair is the horizontal surface gap, but only when their
vertical spans actually overlap:

    horiz_gap(i,j) = hypot(dx, dy) - r_i - r_j
    overlap(i,j)   = z-spans intersect
    gap(i,j)       = horiz_gap if overlap else +inf

The vertical test is not cosmetic: a drone hovering at 1.3 m over a 0.8 m G1 is
not a conflict, and a purely spherical model wrongly calls it one (measured
during development: 8199 false blocks). Contact = horizontal_gap < 0 AND overlap.

This is monitoring only. It is NOT collision avoidance: no trajectory
prediction, no velocity obstacles, no replanning. The coordinator consumes these
reports (safety.py).
"""
from __future__ import annotations

import itertools
import math

# Upright-cylinder body envelope: (footprint radius, origin height when
# standing, body height). The ORIGIN IS NOT THE CENTRE: the G1's origin is its
# pelvis at 0.78 m, the Go2's is its base at 0.31 m, the Panda's is its pedestal
# mount at 0.40 m. Using a centred span put the G1's body at 0.38-1.18 m and
# reported a drone passing 0.2 m above its head as a -0.40 m penetration.
RADIUS = {"go2": 0.36, "drone": 0.16, "panda": 0.30, "g1": 0.32}
ORIGIN_REF = {"go2": 0.31, "drone": 0.00, "panda": 0.40, "g1": 0.78}
BODY_H = {"go2": 0.42, "drone": 0.12, "panda": 1.05, "g1": 0.80}

SAFE_GAP = 0.22        # enter conflict below this surface gap
CLEAR_HYST = 0.12      # leave conflict only above SAFE_GAP + CLEAR_HYST


class Proximity:
    def __init__(self, world, safe_gap=SAFE_GAP, hyst=CLEAR_HYST):
        self.w = world
        self.safe_gap = safe_gap
        self.hyst = hyst
        # M5.1: the registry FIRST — the min_gap init below calls _pairs(),
        # which reads this registry (an ordering bug shipped and caught in
        # the first M5 run: the AttributeError at _pairs)
        self.static_obstacles: dict[str, dict] = {}   # name -> {pos, radius, height}
        self.last_move_t: dict[str, float] = {}       # PER-OBSTACLE movement time
        self._last_pos: dict[str, list] = {}          # the movement detection state
        for name, ob in getattr(world, "static_obstacles", {}).items():
            self.register_static(name, ob["pos"], ob["radius"], ob["height"])
        self.active: set[tuple[str, str]] = set()
        self.events: list[tuple[float, str, str, float]] = []   # (t, kind, pair, gap)
        self.min_gap = {p: float("inf") for p in self._pairs()}
        self.contacts: set[tuple[str, str]] = set()
        self.reads = 0
        self.detections = 0        # look-ahead detections (prevented entries)

    def register_static(self, name: str, pos, radius: float, height: float):
        """Register an independent static obstacle (identity + geometry)."""
        self.static_obstacles[name] = {"pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                                       "radius": float(radius), "height": float(height)}
        self.min_gap = {p: float("inf") for p in self._pairs()}

    def update_static(self, name: str, pos):
        """Move a registered obstacle (its OWN identity and state only)."""
        if name in self.static_obstacles:
            self.static_obstacles[name]["pos"] = [float(pos[0]), float(pos[1]), float(pos[2])]

    def _bodies(self) -> dict:
        """Every tracked body: name -> (radius, origin_ref, height)."""
        out = {n: (RADIUS[n], ORIGIN_REF[n], BODY_H[n]) for n in RADIUS}
        for n, ob in self.static_obstacles.items():
            out[n] = (ob["radius"], ob["radius"], ob["height"])
        return out

    def radius_of(self, name: str) -> float:
        st = self.static_obstacles.get(name)
        return st["radius"] if st else RADIUS.get(name, 0.30)

    def _pairs(self):
        names = list(RADIUS) + list(self.static_obstacles)
        return [tuple(sorted(p)) for p in itertools.combinations(names, 2)]

    # -- geometry --------------------------------------------------------
    def positions(self) -> dict:
        w = self.w
        out = {
            "go2": w.robots["go2"].get_state()["pos"],
            "drone": w.robots["drone"].get_state()["pos"],
            "panda": w.robots["panda"].get_state()["base_pos"],
            "g1": list(w.g1_e.get_pos()[0].tolist()),
        }
        for n, ob in self.static_obstacles.items():
            out[n] = list(ob["pos"])
        return out

    def _span(self, robot, pos):
        """Vertical extent of the body in world z (feet/base .. top)."""
        if robot in self.static_obstacles:
            ob = self.static_obstacles[robot]
            return (pos[2] - ob["radius"], pos[2] + ob["height"])
        lo = pos[2] - ORIGIN_REF[robot]
        return (lo, lo + BODY_H[robot])

    def gaps(self) -> dict:
        """Surface gap per pair; +inf when the vertical spans do not overlap."""
        p = self.positions()
        out = {}
        for a, b in self._pairs():
            (az0, az1), (bz0, bz1) = self._span(a, p[a]), self._span(b, p[b])
            overlap = az0 < bz1 and bz0 < az1
            if not overlap:
                out[(a, b)] = float("inf")
                continue
            horiz = math.hypot(p[a][0] - p[b][0], p[a][1] - p[b][1])
            out[(a, b)] = horiz - self.radius_of(a) - self.radius_of(b)
        return out

    # -- monitoring ------------------------------------------------------
    def update(self, t: float) -> dict:
        self.reads += 1
        # M5.1: PER-OBSTACLE movement tracking (independent identity: one
        # blocker moving never touches another's movement state)
        g = self.gaps()
        for n, p in self.positions().items():
            prev = self._last_pos.get(n)
            if prev is not None and math.hypot(p[0] - prev[0], p[1] - prev[1]) > 0.05:
                self.last_move_t[n] = t
            self._last_pos[n] = [p[0], p[1]]
        for pair, gap in g.items():
            if gap == float("inf"):
                continue                     # no vertical overlap: not a pair in play
            self.min_gap[pair] = min(self.min_gap[pair], gap)
            if gap < 0:
                self.contacts.add(pair)
            if pair in self.active:
                if gap > self.safe_gap + self.hyst:
                    self.active.discard(pair)
                    self.events.append((t, "CLEAR", pair, gap))
            elif gap < self.safe_gap:
                self.active.add(pair)
                self.events.append((t, "ENTER", pair, gap))
        return g

    # -- look-ahead ------------------------------------------------------
    def violates_at(self, robot: str, pos) -> list[str]:
        """Which other robots' envelopes `pos` would enter (look-ahead test).

        The gate uses this on the PROSPECTIVE waypoint, so a robot is stopped
        before it enters an envelope rather than after. Measured need: a purely
        instantaneous check let the drone fly onto the G1's head, after which no
        yield could recover the run (23868 blocked ticks, deadlock).
        """
        if pos is None:
            return []
        others = self.positions()
        hit = []
        z0, z1 = self._span(robot, pos)
        for o, p in others.items():
            if o == robot:
                continue
            o0, o1 = self._span(o, p)
            if not (z0 < o1 and o0 < z1):
                continue
            gap = math.hypot(pos[0] - p[0], pos[1] - p[1]) - self.radius_of(robot) - self.radius_of(o)
            if gap < self.safe_gap:
                hit.append(o)
        if hit:
            self.detections += 1
        return sorted(hit)

    # -- queries ---------------------------------------------------------
    def partner_of(self, robot: str):
        return [p for p in self.active if robot in p]

    def min_gap_overall(self) -> float:
        vals = [v for v in self.min_gap.values() if v != float("inf")]
        return min(vals) if vals else float("inf")

    def report(self) -> dict:
        return {
            "pairs": len(self._pairs()),
            "conflict_events": len([e for e in self.events if e[1] == "ENTER"]),
            "clear_events": len([e for e in self.events if e[1] == "CLEAR"]),
            "min_gap_by_pair": {f"{a}-{b}": (round(v, 3) if v != float("inf") else None)
                                for (a, b), v in self.min_gap.items()},
            "min_gap": round(self.min_gap_overall(), 3),
            "contacts": sorted(f"{a}-{b}" for a, b in self.contacts),
            "lookahead_detections": self.detections,
            "reads": self.reads,
        }
