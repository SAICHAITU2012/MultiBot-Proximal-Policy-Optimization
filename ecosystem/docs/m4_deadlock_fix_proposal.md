# M4 deadlock fix — root cause and the minimal change (proposal, not yet validated)

**Status: NOT VALIDATED.** This document specifies the fix; it is deliberately not applied to
`missions/relay.py`, which was being edited by the milestone owner throughout this session
(hashes observed moving `ef988be8 → 6898ab60 → 276a7797`). Anything below must be applied and
re-measured by the owning session. No claim of a working fix is made here.

---

## 1. What is observed (current revision)

`tests/test_m4.py`, run on the current tree, scenario T5 (`World(g1_xy=(1.3,1.2))`, crate at
(1.2,1.2)):

```
why = "timeout at stage reached"     t = 180.0 s
travel = 27.57 m (ideal ~3.4 m)      replans = 5 (= MAX_REPLANS)   reroutes = 1   waits = 18
blocked_ticks = 19                   prox contacts = []           DEADLOCK events = none
```

A per-tick trace shows the dog confined to a ~0.3 m loop near (0.30, 0.45) — **orbiting, not
travelling**. The run ends on the `max_s` timeout, never on a deadlock verdict.

## 2. Two independent root causes (verified by source inspection)

### C1 — progress is measured against the wrong reference

```python
tgt = reroute_wps[0] if reroute_wps else c          # the TEMPORARY target
_pos_prog = math.dist(g["pos"][:2], (tgt[0], tgt[1]))
improved  = (_pos_prog < _pos_prog_best - 0.05) or (_herr < _herr_best - 0.15)
```

`tgt` is whatever waypoint the planner last inserted. A robot that orbits a *temporary*
waypoint keeps converging on that waypoint, so `improved` stays true while the mission goal
(the crate) is never approached. This is the mechanism, not a suspicion: replaying the
proposed rule on the recorded trajectory (§4) reproduces the livelock exactly.

### C2 — the stall timer is cleared whenever the gate *permits* motion

```python
if not may:                                          # blocked
    ...  ESCAPE_INSERTED ...
else:
    deadlock_since = None                            # <-- THE BUG
```

Every tick the robot is free to drive resets the timer. A livelock is, by definition, free
driving — so the timer can never mature. In T5 the gate blocked only **19 ticks out of
36 000**, which is enough to make the detector unreachable. Permission is not progress.

### C3 — budget exhaustion is not terminal

When `replans >= MAX_REPLANS` and the ladder returns no candidate, the code holds the old
route and waits for a progress timer that (per C2) cannot fire. The honest verdict must be
published at that moment.

## 3. The minimal change (three edits)

**Edit 1 — stable goal reference.** Replace the `tgt`-based progress with the mission goal,
monotone best-so-far, and credit a pivot's heading only while the goal distance is not
worsening:

```python
_goal_d    = math.dist((g["pos"][0], g["pos"][1]), (c[0], c[1]))
_goal_herr = abs((math.atan2(c[1]-g["pos"][1], c[0]-g["pos"][0])
                  - yaw_from_quat(w.robots["go2"].base_quat[0])
                  + math.pi) % (2.0*math.pi) - math.pi)
if _goal_best is None:
    improved = True
else:
    improved = (_goal_d < _goal_best - 0.05
                or (_goal_d <= _goal_best + 0.05
                    and _goal_herr < _goal_herr_best - 0.15))
if improved:
    _goal_best      = min(_goal_best, _goal_d)      if _goal_best      is not None else _goal_d
    _goal_herr_best = min(_goal_herr_best, _goal_herr) if _goal_herr_best is not None else _goal_herr
    deadlock_since  = None
elif deadlock_since is None:
    deadlock_since  = t
```

**Edit 2 — delete the permission reset** (§2 C2): the `else:` branch must *not* clear
`deadlock_since`. Reset it only on real plan milestones — a waypoint consumed, a
`ROUTE_REPLANNED` published, an escape inserted — so executing a plan counts as progress while
orbiting does not.

**Edit 3 — bound the escapes and make exhaustion terminal.**

```python
MAX_ESCAPES = 6
...
if ((replans >= MAX_REPLANS and no_route_streak >= 3) or escapes >= MAX_ESCAPES) \
        and "reached" not in stage_msgs and not reroute_wps:
    bus.publish(t, "go2", "DEADLOCK", classification=_diag["kind"], blockers=partners,
                detection_t=round(t, 2), escapes=escapes, replans=replans, ...)
    return _result(False, ...)      # terminal — not a timeout
```

## 4. Offline evidence that the rule is the right one

Replaying **Edit 1's rule only** against the recorded T5 trajectory (no simulation, pure
arithmetic on the per-tick positions):

| | value |
|---|---|
| observed | timeout at `max_s`, 27.57 m travelled (180 s run) |
| proposed rule fires at | **t = 12.46 s** |
| travel after that point | **82 % of the run** |
| wall-time saved | ~48 s per failing scenario (60 s run: 12.5 s instead of 60 s) |

So the rule separates "orbiting" from "approaching" on real data. It does not yet prove the
integrated system reports the verdict correctly — that is §6.

## 5. Regression risk to watch (why this must be measured, not assumed)

The dog *legitimately* moves away from the crate during an escape leg. With a stable-goal
monotone reference, a long escape can look like no progress. Mitigations already in the design:
reset on waypoint-consumed / `ROUTE_REPLANNED` / escape-inserted, and cap escapes. The
scenarios that must be re-run to prove this does not over-fire:

- `tests/test_m4_replan.py` — A (commit → invalidate → replan → resume, must still complete)
- `tests/test_m4.py` T1/T2/T4 — the completions that must not become false deadlocks
- `tests/test_m2.py` blocked-lane — the deliberate 8 s wait

## 6. Acceptance definition for closing this item

1. T5 publishes a **`DEADLOCK` event** (not a timeout) carrying `classification`, `blockers`,
   `detection_t`, `escapes`.
2. `detection_t` is materially below `max_s` (the trace predicts ~12–20 s; require < 60 s).
3. Travel for T5 drops from 27.57 m to the same order as the ideal path.
4. `tests/test_m4_replan.py` is still 9/9 (no false deadlocks), and M1/M2/M3 suites unchanged.
5. The artifact records the source hashes of the revision that produced it.

Differentiating the two outcomes matters: a **timeout** says the system stopped waiting; a
**deadlock verdict** says the system recognised it could not make valid progress. Only the
second is the capability M4 claims.

## 7. Why this is a proposal and not a patch

Two reasons, both recorded so the owner can judge them:

1. `missions/relay.py` was actively edited by the milestone owner during this session (three
   distinct hashes in twenty minutes). A context diff would be stale on arrival.
2. When the three edits above were applied to a frozen copy of the tree, instrumentation of the
   copy did not take effect in-process (an inserted `raise` at the top of `run()` did not fire,
   and results were byte-identical across two different patch states). That is a tooling
   anomaly in the snapshot, not evidence about the fix — but until it is explained, the patch
   is unmeasured, and an unmeasured patch must not be described as a fix.
