#!/usr/bin/env python3
"""relay.py — coordinated four-robot task, M2 version.

Same dependency chain as M1:

    drone --CRATE_SIGHTED--> go2 --CRATE_REACHED--> panda --PARCEL_DISPATCHED-->
                                                          g1 --STATION_STABLE--> DONE

New in M2:
  * every stage must obtain a token from the ConstraintLayer before its robot
    may act (core/safety.py); `execute()` refuses blocked tokens, so a
    prohibited action is impossible to run
  * proximity conflicts are monitored every tick (core/proximity.py)
  * yielding: a mobile partner climbs/waits; an immobile partner (the standing
    G1) is routed around
  * interference scenarios for Experiment A
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.proximity import Proximity
from core.safety import ConstraintLayer
from robots.go2_drive import YawThenDrive, yaw_from_quat
from core.proximity import RADIUS
from core.replan import validate_route, segment_clear, pick_route_ladder

# M4: closed-loop replanning constants
MAX_REPLANS = 5          # the replan budget: beyond this, the honest deadlock
REPLAN_MIN_GAP = 2.0     # s between replans (the thrash fuse)

# M3.3: the Go2 can now STEER (the yaw-then-drive primitive on the frozen nav
# policy; M3.2 verified the arc mode: ~95 deg in 4.8 s, turn-then-drive to a
# waypoint in 2.6 s). The blocked-lane reroute is no longer geometrically
# impossible; DEADLOCK remains the honest report if a reroute also fails.
DRIVE_PRIM = YawThenDrive(mode="arc")


@dataclass
class Msg:
    t: float
    src: str
    kind: str
    data: dict = field(default_factory=dict)


class Bus:
    def __init__(self):
        self.log: list[Msg] = []

    def publish(self, t, src, kind, **data):
        self.log.append(Msg(t, src, kind, data))


# ---- gates: every one requires substantive motion --------------------------
SIGHT_TOL = 0.30
SIGHT_HOLD = 1.0
REACH_D = 0.60
GO2_MIN_TRAVEL = 0.50
PARCEL_MOVE = 0.10
PARCEL_DROP_Z = 0.40
STABLE_S = 5.0
TIMEOUT_S = 240.0
DEADLOCK_S = 8.0        # s a blocked robot may wait before it reports a deadlock

REROUTE_OFFSET = 1.60    # m lateral detour around an immobile blocker
# (M3: 1.00 left the crate-approach leg grazing the conflict boundary
#  with a 0.013-m margin — any drift re-blocked the dog mid-approach;
#  1.60 clears the final leg by ~0.4 m of surface gap)
CLIMB_STEP = 0.35        # m the drone climbs when it must yield


def run(world, verbose=True, max_s=200.0, ablate=None, constraints=False, scenario=None):
    bus = Bus()
    w = world
    prox = Proximity(w)
    layer = ConstraintLayer(prox, verbose=verbose) if constraints else None
    executed: list[int] = []

    if ablate == "g1":
        w.g1_passive = True

    parcel0 = [float(v) for v in w.robots["panda"].get_state()["obj_pos"]]
    go2_start = list(w.robots["go2"].get_state()["pos"])
    drone_final = [float(v) for v in _drone_target_near_crate(w)[0]]
    if scenario == "low_pass":
        drone_final[2] = 0.75      # deliberately low survey pass: crosses the G1's body height
    deadlock_since = None
    _pos_prog_best = None      # the deadlock timer's baseline: the best GOAL
                               # (crate) distance seen — the audit's livelock
                               # fix: motion without goal reduction is not
                               # progress
    contact_this_leg = False   # M4: any go2-g1 CONTACT dirties the approach —
                               # the reach is suppressed until a fresh reroute
                               # attempt resets the leg (a plow-through must
                               # never count as a delivery)
    best_effort_leg = False    # M4: the reach on an UNVALIDATED best-effort
                               # leg is not a delivery — the planner verdicted
                               # "no safe route"; claiming success from the
                               # squeeze would contradict our own planner
    in_sight_since = None
    g1_stable_since = None
    next_telemetry = 0.0
    coordinator_reads = 0
    stage_msgs = {}
    reroute_wps: list = []     # M3: the active detour queue (the dog steers now)
    replans = 0                # M4: the replan budget counter
    no_route_streak = 0        # M5.4: the consecutive ladder-exhausted picks
    last_replan_t = -1e9       # M4: the replan thrash fuse
    replan_latencies_ms = []   # M4 metric: wall-clock per replan, milliseconds
    go2_path = 0.0             # M4 metric: the travelled path
    _prev_pos = list(go2_start[:2])
    _m4 = {"replans": 0, "replan_latencies_ms": replan_latencies_ms,
           "go2_path_m": 0.0}
    yields = {"drone_climb": 0, "go2_wait": 0, "go2_reroute": 0}
    from core.world import SCENE_DT
    steps = int(max_s / SCENE_DT)

    def say(*a):
        if verbose:
            print(*a)

    def ask(robot, action, next_pos=None, conflict=None):
        """Constraint gate. Default: evaluated on the PROSPECTIVE position.
        An explicit conflict=[] permits the request (used for the REORIENTATION
        pivots: the turn-phase command's translation is <= 0.15 m/s with the
        center orbiting perpendicular to the heading — it cannot close the
        straight-line gap to a blocker dead ahead, and the proximity monitor
        verifies no contact actually occurs)."""
        if layer is None:
            return True
        if conflict is None:
            conflict = prox.violates_at(robot, next_pos) if next_pos is not None else None
        tok = layer.request(robot, action, conflict=conflict)
        return tok not in layer._blocked

    def allowed_execute(tok_holder):
        """Mark the granted request as executed (only called for real action)."""
        if layer is not None and tok_holder:
            layer.execute(len(layer.audit), executed)

    for _ in range(steps):
        w.step()
        t = w.sim_time
        prox.update(t)
        if "g1-go2" in prox.contacts:
            contact_this_leg = True

        if t >= next_telemetry:
            for rid, st in w.robot_positions().items():
                bus.publish(t, rid, "TELEMETRY", pos=[round(v, 2) for v in st])
            next_telemetry = t + 10.0

        # ---------------- stage 1: drone sightings -----------------------
        if "sighted" not in stage_msgs:
            dz = w.robots["drone"].get_state(); coordinator_reads += 1
            may = ask("drone", "fly_to_crate", next_pos=_hop_target(dz["pos"], drone_final))
            if not may:
                # YIELD: the drone is mobile -> climb, and RAISE THE PLAN so the
                # rest of the flight stays clear (a resolution, not a stall)
                w.robots["drone"].commands = _climb(w, CLIMB_STEP)
                drone_final[2] = max(drone_final[2], 1.30)
                yields["drone_climb"] += 1
                in_sight_since = None
            else:
                _drone_hop(w, drone_final)
                if math.dist(dz["pos"], drone_final) < SIGHT_TOL:
                    in_sight_since = t if in_sight_since is None else in_sight_since
                else:
                    in_sight_since = None
                if ablate == "drone":
                    in_sight_since = None
                if in_sight_since is not None and t - in_sight_since >= SIGHT_HOLD:
                    bus.publish(t, "drone", "CRATE_SIGHTED",
                                crate=[round(v, 3) for v in w.crate_xy()],
                                drone_pos=[round(v, 2) for v in dz["pos"]])
                    stage_msgs["sighted"] = t
                    say(f"  [{t:6.1f}s] drone  -> CRATE_SIGHTED  at {[round(v,2) for v in dz['pos']]}")

        # ---------------- stage 2: go2 drives to the crate ---------------
        elif "reached" not in stage_msgs:
            c = w.crate_xy()
            g = w.robots["go2"].get_state(); coordinator_reads += 1
            d = math.dist((g["pos"][0], g["pos"][1]), c)
            walked = math.dist((g["pos"][0], g["pos"][1]), go2_start[:2])
            tgt = reroute_wps[0] if reroute_wps else c
            yaw_now = yaw_from_quat(w.robots["go2"].base_quat[0])
            go2_path += math.dist((g["pos"][0], g["pos"][1]), _prev_pos)
            _prev_pos = [g["pos"][0], g["pos"][1]]
            _m4["replans"] = replans
            _m4["go2_path_m"] = go2_path
            # --- M4: CLOSED-LOOP ROUTE RE-EVALUATION (every tick) ---------
            if reroute_wps:
                # EXECUTION TOLERANCE: the dog flies curved arcs, not straight
                # legs — validating the straight segments with the full
                # SAFE_GAP re-flagged the dog's own gate-legal arc every tick
                # and burned the replan budget without progress. The
                # re-evaluation interrupts only on DEEP violations (the
                # surface gap < 0.07); the per-tick gate handles the fine
                # safety, and the PLANNING standard stays strict.
                valid, _hits = validate_route(prox, "go2", g["pos"], reroute_wps, c,
                                              margin=-0.15)
                if not valid:
                    # the remaining route is no longer safe: replan, gated by
                    # the budget and the thrash fuse (no infinite replanning)
                    if replans < MAX_REPLANS and (t - last_replan_t) >= REPLAN_MIN_GAP:
                        route, meta, lat_s = pick_route_ladder(prox, "go2", g["pos"], c,
                                                               _all_blockers(prox), yaw_now)
                        replans += 1
                        last_replan_t = t
                        replan_latencies_ms.append(lat_s * 1000.0)
                        no_route_streak = 0 if route is not None else no_route_streak + 1
                        if route is not None:
                            reroute_wps = route
                            contact_this_leg = False   # a fresh leg
                            best_effort_leg = False    # a validated leg
                            bus.publish(t, "go2", "ROUTE_REPLANNED",
                                        n=replans, latency_ms=round(lat_s * 1000, 2),
                                        wps=[[round(v, 2) for v in wp] for wp in route])
                            say(f"  [{t:6.1f}s] go2     .. ROUTE_REPLANNED #{replans} "
                                f"({lat_s*1000:.1f} ms)")
                        else:
                            # no valid candidate right now: hold and keep
                            # re-evaluating — the blocker may move away
                            bus.publish(t, "go2", "REPLAN_NO_ROUTE", n=replans)
                    # else: the budget/fuse holds — keep the old route (the
                    # gate still guards every move; the progress timer will
                    # honestly deadlock if the situation never resolves)
                else:
                    # the route is valid — but is the DIRECT line now clear?
                    # (the blocker moved away: resume direct, drop the detour)
                    direct_ok, _ = segment_clear(prox, "go2", g["pos"],
                                                 [c[0], c[1], g["pos"][2]])
                    if direct_ok:
                        bus.publish(t, "go2", "RESUME_DIRECT",
                                    dropped=len(reroute_wps))
                        say(f"  [{t:6.1f}s] go2     .. RESUME_DIRECT (dropped "
                            f"{len(reroute_wps)} detour waypoints)")
                        reroute_wps = []
                        tgt = c
            vx_p, vy_p, wz_p, phase = DRIVE_PRIM.command((g["pos"][0], g["pos"][1]),
                                                         yaw_now, tgt)
            if phase == "turn":
                # the pivot permit: reorientation cannot close the straight-line
                # gap to a blocker dead ahead (the orbit is perpendicular); the
                # proximity monitor verifies the physical outcome every tick
                may = ask("go2", "reorient", conflict=[])
            else:
                # translation: gated on the HEADING-projected look-ahead, with
                # the ESCAPE-MONOTONICITY clause: a blocked robot may translate
                # if the prospective position does not worsen its worst active
                # gap — an escape, not an entry (the blocked-lane dog spawns
                # 0.13 m inside the envelope; without this clause even the
                # monotonic escape leg is gate-blocked and the run deadlocks)
                # the look-ahead must EXCEED the gait's stopping distance
                # (~0.7 m of slide at 0.55 m/s — the T5 contact lesson: braking
                # 0.3 m ahead guaranteed contact in the impossible geometry)
                hd = (g["pos"][0] + 0.75 * math.cos(yaw_now),
                      g["pos"][1] + 0.75 * math.sin(yaw_now), g["pos"][2])
                conflict = prox.violates_at("go2", list(hd))
                if conflict:
                    esc = True
                    for other in conflict:
                        o_pos = prox.positions()[other]
                        rsum = prox.radius_of("go2") + prox.radius_of(other)
                        base = math.hypot(g["pos"][0] - o_pos[0], g["pos"][1] - o_pos[1]) - rsum
                        if base <= -0.05:
                            # DEEP-CONTACT CEILING (the M4 T3 lesson): past a
                            # surface gap of -0.05 the dog is already inside
                            # the other's body — "the gap is improving" would
                            # permit plowing THROUGH. Stop; reroute instead.
                            esc = False
                            break
                        nxt = math.hypot(hd[0] - o_pos[0], hd[1] - o_pos[1]) - rsum
                        if nxt < base - 0.02:   # would deepen the conflict
                            esc = False
                            break
                    if esc:
                        conflict = []           # the escape permit
                may = ask("go2", "drive_to_crate", conflict=conflict)
            # PROGRESS-TRACKED DEADLOCK — M5 correction (the audit's §4
            # livelock finding): M3's progress credited MOTION (the position
            # toward the waypoint OR the heading convergence) — and orbiting
            # supplies both without approaching the goal: the T5 dog orbited a
            # 0.3 m circle for 180 s, 27.6 m of path, deadlock never fired.
            # The honest progress is GOAL-DISTANCE REDUCTION ONLY (the crate
            # distance improving by >=5 cm). The heading credit is removed: a
            # pivot is permitted by the gate, not counted as progress.
            improved = (_pos_prog_best is None or d < _pos_prog_best - 0.05)
            if improved:
                _pos_prog_best = d
                deadlock_since = None
            elif deadlock_since is None:
                deadlock_since = t
            # TERMINAL BUDGET EXHAUSTION (the audit's §4b): once the replan
            # budget is spent with the route still invalid, publish the
            # deadlock IMMEDIATELY — holding a dead route and waiting on a
            # progress timer that cannot fire is the livelock's other half.
            if (replans >= MAX_REPLANS and reroute_wps
                    and deadlock_since is not None and t - deadlock_since >= 2.0):
                _diag = _deadlock_diagnosis(prox, replans, no_route_streak, t)
                bus.publish(t, "go2", "DEADLOCK",
                            blocked_for=round(t - deadlock_since, 1),
                            stage="budget_terminal", diagnosis=_diag)
                say(f"  [{t:6.1f}s] go2     !! DEADLOCK (budget terminal): {_diag['kind']}")
                rr = _result(False, bus, stage_msgs, t,
                             f"deadlock: replanning budget exhausted "
                             f"({_diag['kind']})", coordinator_reads,
                             prox, layer, yields, executed, scenario, m4=_m4)
                rr["deadlock"] = {"robot": "go2", "t": round(t, 2), "diagnosis": _diag}
                return rr
            partners = sorted({o for pair in prox.partner_of("go2") for o in pair if o != "go2"})
            if (deadlock_since is not None and t - deadlock_since >= DEADLOCK_S
                    and "reached" not in stage_msgs):
                _diag = _deadlock_diagnosis(prox, replans, no_route_streak, t)
                bus.publish(t, "go2", "DEADLOCK",
                            partners=partners, blocked_for=round(t - deadlock_since, 1),
                            stage="reroute_blocked", diagnosis=_diag)
                say(f"  [{t:6.1f}s] go2     !! DEADLOCK: no progress for "
                    f"{t-deadlock_since:.1f}s, partners={partners}")
                rr = _result(False, bus, stage_msgs, t,
                             f"deadlock at stage reached: no progress, partners={partners}", coordinator_reads,
                             prox, layer, yields, executed, scenario, m4=_m4)
                rr["deadlock"] = {"robot": "go2", "partners": partners,
                                  "t": round(t, 2), "diagnosis": _diag}
                return rr
            if not may:
                blocker_pos = _blocker_pos(w, "go2")
                if not reroute_wps:
                    # M4: plan BOTH side candidates and pick the first VALID
                    # one (the M3 single-side heuristic is superseded). No
                    # valid candidate is NOT an immediate deadlock: hold and
                    # keep re-evaluating — the blocker may move away (T2) and
                    # the progress timer honestly deadlocks if it never does.
                    route, meta, lat_s = pick_route_ladder(prox, "go2", g["pos"], c,
                                                           _all_blockers(prox), yaw_now)
                    yields["go2_reroute"] += 1
                    if route is not None:
                        reroute_wps = route
                        no_route_streak = 0
                        contact_this_leg = False   # a fresh attempt: clean leg
                        best_effort_leg = False    # a validated route
                        bus.publish(t, "go2", "REROUTE_STARTED",
                                    meta=meta, latency_ms=round(lat_s * 1000, 2),
                                    wps=[[round(v, 2) for v in wp] for wp in route])
                        say(f"  [{t:6.1f}s] go2     .. REROUTE_STARTED via "
                            f"{[[round(v,2) for v in wp] for wp in route]}")
                    else:
                        no_route_streak += 1
                        # no ladder candidate passes the shallow-dip
                        # validation: still queue the M3 geometric best-effort
                        # route — the dog MANOEUVRES (the progress timer judges
                        # honestly) and the per-tick gate keeps every move
                        # safe. Holding with empty waypoints would just
                        # re-pick every tick (the 1601-reroute loop).
                        reroute_wps = _reroute_waypoints(g["pos"], c, blocker_pos,
                                                         heading=yaw_now)  # (the
                        # singular _reroute_waypoint helper was deleted in the
                        # M5 audit response: dead code, no call sites — F2)
                        best_effort_leg = True     # unvalidated: the reach is
                                                   # suppressed on this leg
                        bus.publish(t, "go2", "REROUTE_BEST_EFFORT",
                                    blocker=[round(v, 2) for v in (blocker_pos or [])])
                elif phase == "drive" and may is False and reroute_wps:
                    # M4 lesson (the boundary-stall): the dog pivoted onto the
                    # detour bearing but the leg's first move DEEPENS the gap
                    # (the dog sits exactly on the envelope boundary) — the
                    # route was planned without an escape leg because the
                    # radial distance was at the threshold. Insert a local
                    # escape waypoint in front: pivot away, drive out, then
                    # resume the detour. The progress timer still judges.
                    if blocker_pos:
                        ux = g["pos"][0] - blocker_pos[0]
                        uy = g["pos"][1] - blocker_pos[1]
                        un = math.hypot(ux, uy) + 1e-9
                        reroute_wps.insert(0, [g["pos"][0] + ux / un * 0.55,
                                               g["pos"][1] + uy / un * 0.55, 0.0])
                        bus.publish(t, "go2", "ESCAPE_INSERTED",
                                    wp=[round(v, 2) for v in reroute_wps[0]])
                    yields["go2_wait"] += 1
                else:
                    yields["go2_wait"] += 1
                w.robots["go2"].commands[0, :] = 0.0   # hold while the gate says no
            else:
                deadlock_since = None
            if may and (d > REACH_D or walked < GO2_MIN_TRAVEL):
                if reroute_wps and math.dist(g["pos"][:2], reroute_wps[0][:2]) < 0.35:
                    reroute_wps.pop(0)                 # waypoint consumed
                    tgt = reroute_wps[0] if reroute_wps else c
                cmd = w.robots["go2"].commands
                cmd[0, 0] = vx_p; cmd[0, 1] = vy_p; cmd[0, 2] = wz_p
            elif may and ablate == "go2":
                pass          # arrives, never reports
            elif not (d > REACH_D or walked < GO2_MIN_TRAVEL):
                # HONEST REACH (the M4 T5 lesson): the gait slides ~0.2 m after
                # the brake command — a dog that wedged INTO a blocker's
                # envelope must not claim the reach (a false completion fired
                # from a physically overlapping state). The reach requires a
                # clean approach: no active proximity conflict.
                if prox.partner_of("go2") or contact_this_leg or best_effort_leg:
                    # in conflict OR the approach touched the humanoid OR the
                    # planner verdicted no-safe-route: not a reach. Hold; the
                    # deadlock timer judges.
                    w.robots["go2"].commands[0, :] = 0.0
                else:
                    w.robots["go2"].commands[0, :] = 0.0
                    bus.publish(t, "go2", "CRATE_REACHED", dist=round(d, 3))
                    stage_msgs["reached"] = t
                    say(f"  [{t:6.1f}s] go2    -> CRATE_REACHED   d={d:.2f} m")

        # ---------------- stage 3: panda, released by the go2 ------------
        elif "dispatched" not in stage_msgs:
            if ablate == "panda":
                pass
            elif w.robots["panda"].external_control:
                w.robots["panda"].external_control = False
                say(f"  [{t:6.1f}s] panda   .. released by CRATE_REACHED, reach policy engaged")
            p = w.robots["panda"].get_state(); coordinator_reads += 1
            moved = math.dist(p["obj_pos"], parcel0)
            if moved > PARCEL_MOVE and p["obj_pos"][2] < PARCEL_DROP_Z:
                bus.publish(t, "panda", "PARCEL_DISPATCHED", moved=round(moved, 3))
                stage_msgs["dispatched"] = t
                say(f"  [{t:6.1f}s] panda  -> PARCEL_DISPATCHED  moved {moved:.3f} m")

        # ---------------- stage 4: g1 holds station ----------------------
        else:
            z = w.g1_z()
            if z < 0.70:
                bus.publish(t, "g1", "STATION_FAILED", z=round(z, 3))
                say(f"  [{t:6.1f}s] g1     !! STATION_FAILED  z={z:.2f}")
                return _result(False, bus, stage_msgs, t, "g1 fell", coordinator_reads,
                               prox, layer, yields, executed, scenario, m4=_m4)
            if ablate == "g1":
                pass
            if g1_stable_since is None:
                g1_stable_since = t
            if t - g1_stable_since >= STABLE_S and "stable" not in stage_msgs:
                bus.publish(t, "g1", "STATION_STABLE", held=round(t - g1_stable_since, 1), z=round(z, 3))
                stage_msgs["stable"] = t
                say(f"  [{t:6.1f}s] g1     -> STATION_STABLE   held {t-g1_stable_since:.1f}s at z={z:.3f}")
                return _result(True, bus, stage_msgs, t, None, coordinator_reads,
                               prox, layer, yields, executed, scenario, m4=_m4)

        if t > TIMEOUT_S:
            break
    return _result(False, bus, stage_msgs, w.sim_time,
                   f"timeout at stage {_last_stage(stage_msgs)}", coordinator_reads,
                   prox, layer, yields, executed, scenario, m4=_m4)


def _last_stage(stage_msgs):
    for s in ("sighted", "reached", "dispatched", "stable"):
        if s not in stage_msgs:
            return s
    return "done"


def _pos_of(w, robot):
    p = prox._pos_single(w, robot) if False else None
    st = (w.robots[robot].get_state() if robot in w.robots else None)
    if st is None:
        v = w.g1_e.get_pos()[0].tolist()
        return [float(v[0]), float(v[1]), float(v[2])]
    return list(st.get("pos") or st.get("base_pos"))


def prox_immobile(w, robot):
    return robot in ("g1", "panda")      # legs/arms standing still; neither walks


def _hop_target(p, final, hop=0.55):
    """The waypoint the drone is about to be commanded to (for look-ahead)."""
    d = math.dist(p, final)
    if d < 1e-6:
        return list(final)
    if d < 0.22:
        return list(final)
    dirv = [(f - c) / d for f, c in zip(final, p)]
    return [p[i] + hop * dirv[i] for i in range(3)]


def _next_drive_pos(pos, target, look=0.30):
    """The position the go2 is about to move into (for look-ahead)."""
    dx, dy = target[0] - pos[0], target[1] - pos[1]
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return list(pos)
    step = min(look, n)
    return [pos[0] + step * dx / n, pos[1] + step * dy / n, pos[2]]


def _climb(w, dz):
    import torch
    p = w.robots["drone"].get_state()["pos"]
    return torch.tensor([[p[0], p[1], p[2] + dz]])


def _all_blockers(prox, robot="go2"):
    """Every OTHER tracked body's position (the N-obstacle planning set)."""
    return [list(map(float, p[:2])) for n, p in prox.positions().items() if n != robot]


def _deadlock_diagnosis(prox, replans, no_route_streak, t):
    """M5.4: the deadlock taxonomy — NO_GEOMETRIC_ROUTE (the ladder
    exhausted), TEMPORARY_OBSTRUCTION (a blocker moved recently — the waiting
    is legitimate), BUDGET_EXHAUSTED (the replans hit the cap),
    STALLED_PROGRESS (progressing but unable to finish)."""
    movers = [n for n, mt in prox.last_move_t.items()
              if t - mt < 5.0 and n != "go2"]
    if no_route_streak >= 3:
        kind = "NO_GEOMETRIC_ROUTE"    # the ROOT cause first (the advisor's
                                       # challenge A): the budget exhaustion is
                                       # the symptom of repeatedly finding no
                                       # valid candidate
    elif replans >= MAX_REPLANS:
        kind = "BUDGET_EXHAUSTED"
    elif movers:
        kind = "TEMPORARY_OBSTRUCTION"
    else:
        kind = "STALLED_PROGRESS"
    return {"kind": kind, "recently_moved": movers}


def _blocker_pos(w, robot):
    """The nearest conflicting partner's position (for the detour side), or None."""
    try:
        pos = w.robot_positions()
        return pos.get("g1")
    except Exception:
        return None


def _reroute_waypoints(pos, crate, blocker=None, heading=None):
    """M3.3: a HEADING-AWARE 3-waypoint route around an immobile blocker.

    The two-waypoint route (escape + detour) corner-cut: the straight line
    from the dog's blocked position to the detour passed 0.03 m from the
    blocker's envelope and the gate correctly refused it — the dog oscillated
    between the pivot and the block forever. The fix: an intermediate escape
    waypoint radially away from the blocker (blended with the dog's current
    heading so the first turn is ~45-90 deg, inside the arc's good range),
    then the detour beside the blocker, then the crate."""
    wps = []
    px_c, py_c = float(crate[0]), float(crate[1])
    from core.proximity import RADIUS, SAFE_GAP
    if blocker is None:
        mx, my = (pos[0] + px_c) / 2, (pos[1] + py_c) / 2
        wps.append([mx, my + REROUTE_OFFSET, 0.0])
        return wps
    bx, by = blocker[0], blocker[1]
    need = RADIUS["go2"] + RADIUS["g1"] + SAFE_GAP + 0.15   # the center clearance
    dx, dy = pos[0] - bx, pos[1] - by
    cur = math.hypot(dx, dy) + 1e-9
    away = (dx / cur, dy / cur)                    # radially away from the blocker
    # the heading blend: the first waypoint leans toward the dog's current
    # heading so the pivot stays inside the creep arc's effective range
    if heading is not None:
        hx, hy = math.cos(heading), math.sin(heading)
        blend = (away[0] * 0.55 + hx * 0.45, away[1] * 0.55 + hy * 0.45)
        bn = math.hypot(blend[0], blend[1]) + 1e-9
        esc_dir = (blend[0] / bn, blend[1] / bn)
    else:
        esc_dir = away
    esc_len = max(need - cur, 0.0) + 0.55
    wps.append([pos[0] + esc_dir[0] * esc_len, pos[1] + esc_dir[1] * esc_len, 0.0])
    # the detour beside the blocker: away-side perpendicular of the blocker->crate line
    lx, ly = px_c - bx, py_c - by
    Ll = math.hypot(lx, ly) + 1e-9
    perp = (-ly / Ll, lx / Ll)
    lat = dx * perp[0] + dy * perp[1]
    s = -1.0 if lat > 0 else 1.0
    wps.append([bx + perp[0] * s * 1.4, by + perp[1] * s * 1.4, 0.0])
    wps.append([px_c, py_c, 0.0])
    return wps



def _result(ok, bus, stage_msgs, t, why, reads, prox, layer, yields, executed,
            scenario=None, m4=None):
    return {"ok": ok, "t": t, "why": why,
            "stage_times": dict(stage_msgs),
            "events": [(round(m.t, 2), m.src, m.kind) for m in bus.log],
            "messages": len(bus.log), "bus_reads": reads,
            "proximity": prox.report(),
            "safety": layer.stats() if layer is not None else None,
            "yields": dict(yields), "executed_tokens": len(executed),
            "m4": m4 or {"replans": 0, "replan_latencies_ms": [],
                         "go2_path_m": 0.0},
            "scenario": scenario}


def _drone_target_near_crate(w):
    import torch
    c = w.crate_xy()
    return torch.tensor([[c[0], c[1], 1.30]])


def _drone_hop(w, final, hop=0.55):
    import torch
    p = w.robots["drone"].get_state()["pos"]
    d = math.dist(p, final)
    if d < 0.22:
        w.robots["drone"].commands = torch.tensor([list(final)])
        return d
    dirv = [(f - c) / d for f, c in zip(final, p)]
    tgt = [p[i] + hop * dirv[i] for i in range(3)]
    w.robots["drone"].commands = torch.tensor([tgt])
    return d


def _drive_toward(w, pos, target):
    dx, dy = target[0] - pos[0], target[1] - pos[1]
    n = math.hypot(dx, dy) or 1.0
    cmd = w.robots["go2"].commands
    cmd[0, 0] = 0.6 * (-dx / n)      # +vx drives toward -x (measured)
    cmd[0, 1] = 0.0
    cmd[0, 2] = 0.0
