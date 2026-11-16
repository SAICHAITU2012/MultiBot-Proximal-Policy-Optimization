# M2 REPORT — coordination under interference, and recovery (2026-11-12)

**Verdict: M2 PASS — 21/21 acceptance checks**, with three consecutive identical
runs and the M1 mission unaffected.

M2 asked five questions in order. What follows answers each with a measured
number, including the places where the answer is "this fleet cannot do it".

---

## 1. The headline result: the G1's disturbance envelope (Experiment B)

The frozen standing policy (no retraining, no fine-tuning) was hit with a real
external wrench on its torso link for 0.25 s, then left alone.

| Force | Direction | Impulse | Max displacement | Max tilt | Recovery | Final height | Outcome |
|---|---|---|---|---|---|---|---|
| 40 N | forward | 10.0 N·s | 0.02 m | 2.2° | 0.00 s | 0.782 m | recovered |
| 60 N | forward | 15.0 N·s | 0.06 m | 5.8° | 0.44 s | 0.783 m | recovered |
| **70 N** | forward | **17.5 N·s** | **0.13 m** | **8.4°** | **1.23 s** | 0.783 m | recovered |
| **80 N** | forward | **20.0 N·s** | **1.53 m** | **80.9°** | — | 0.128 m | **FELL** |
| 60 N | lateral | 15.0 N·s | 0.05 m | 1.6° | 0.00 s | 0.782 m | recovered |
| 70 N | lateral | 17.5 N·s | 0.07 m | 1.6° | 0.00 s | 0.783 m | recovered |
| **80 N** | lateral | **20.0 N·s** | **0.09 m** | **1.7°** | **0.00 s** | 0.783 m | recovered |

Three things this establishes that "it stands for 10 s" did not:

1. **The policy actively recovers**, not merely tolerates: it returns to 0.782 m
   (±1 mm) after every disturbance inside the envelope, from its own control
   effort alone.
2. **The envelope is anisotropic and has a cliff.** Sagittally it recovers to
   17.5 N·s and fails at 20 N·s — a narrow boundary. Laterally it absorbs 20 N·s
   with 0.09 m of travel and essentially no recovery time, and it barely tilts.
   A push from the front is far more dangerous than the same push from the side.
3. **Beyond the envelope it fails completely, not gracefully**: 80 N forward
   ends at 0.128 m with 80.9° of tilt. There is no partial-recovery regime to
   rely on, which is exactly why the coordinator must keep other robots out of
   the G1's lane rather than count on it surviving a bump.

Post-fall the run is dead (a fallen humanoid does not get up with this policy —
recovery *from standing disturbances*, not from a fall). That distinction is
part of the claim, not a caveat to it.

---

## 2. Proximity monitoring (M2.1)

Six robot pairs, every tick, as **upright cylinders** — footprint radius plus a
vertical-overlap test, with per-kind body heights and origin references:

| robot | radius | body origin ref | body height |
|---|---|---|---|
| go2 | 0.36 m | 0.31 m | 0.42 m |
| drone | 0.16 m | 0.00 m | 0.12 m |
| panda | 0.30 m | 0.40 m | 1.05 m |
| g1 | 0.32 m | 0.78 m | 0.80 m |

Measured baseline: min surface gap **+0.60 m**, zero conflicts, zero contacts.

**A modelling error I had to fix, and why it matters.** My first version centred
each body on its origin. The G1's origin is its *pelvis* at 0.78 m, so its body
was placed at 0.38–1.18 m and a drone passing 0.2 m **above its head** was
reported as a −0.40 m penetration. A second version used a spherical distance,
which reported 8199 false conflicts for the same reason in reverse. Both are in
the log because they are the kind of error that turns a green dashboard into a
false claim. With the corrected spans, an airborne drone over a ground robot
gives `gap = inf` (verified: dz = 0.64 m → no conflict).

## 3. Hard constraints (M2.2) — what they actually do

Every stage must obtain a token; `execute()` **refuses** a blocked token, so a
prohibited action cannot run even by accident (verified: the test raises
`Blocked`).

The gate evaluates the **prospective waypoint**, not the current position. That
change was forced by measurement: with an instantaneous test, the drone flew
onto the G1's head and then no yield could recover the run — 23 868 blocked
ticks and a deadlock. Look-ahead prevents the entry instead of reacting to it.

| Scenario | Gate ON | Gate OFF |
|---|---|---|
| baseline | completes, 0 blocked, min gap **+0.60 m** | identical (nothing to prevent) |
| drone low survey pass over the G1 | **1 look-ahead detection**, 1 blocked flight, drone climbs and raises its plan, completes, min gap **+0.53 m**, no contact | completes, **min gap −0.40 m, contact drone↔g1** (flies through the body) |
| G1 parked in the dog's lane | **1601 blocked drive requests**, then an explicit `DEADLOCK` report, min gap **+0.05 m**, no contact, mission fails *with a reason* | completes with **min gap −0.40 m, contact g1↔go2** (drives through the standing humanoid) |

So the same mission "succeeds" in both columns — the difference is whether it
does so by passing through another robot. That is the whole case for the gate.

### The honest limit: this fleet has almost no lateral resolution

> **Superseded by M3 (2026-11-13).** The claim below is too strong: the strafe
> *does* move the dog (2.5 s vs 4.8 s for a 90° turn in `tests/test_m3_turn.py`),
> it is simply too inaccurate at large turns (a 180° strafe missed its arrival by
> 0.444 m). M3 therefore gave the dog a yaw-then-drive **arc** primitive and a
> heading-aware reroute, which converts the deadlock described here into a
> genuine resolution. Read this section as the M2 state of knowledge only.

The Go2's nav policy **cannot strafe** (measured deadband: commands below
≈0.15 give 0.006 m/s; it needs a run-up at ≈0.6). With two robots that need
0.68 m of centre separation and a 0.35 m-wide corridor, an immobile blocker in
the lane is **geometrically unresolvable** — the dog can only wait. So the
coordinator reports `DEADLOCK` after 8 s instead of pretending to route around
something it cannot. Adding a reroute means giving the dog a yaw-then-drive
primitive, which is a control milestone, not a coordination one.

**Proximity monitoring is not collision avoidance.** There is no trajectory
prediction, no velocity obstacle, no search over trajectories. What exists is a
look-ahead cylinder test, a block, and two deterministic yield strategies
(climb / wait). Nothing in this report claims general collision avoidance.

## 4. No silent failures (G4)

Every mission ends in exactly one of: the four-stage success chain, a named
stage timeout (`timeout at stage <name>`), a `STATION_FAILED` from the G1's own
height, or a `DEADLOCK` with the blocking partners and the wait duration. All
four sinks were exercised during M2. A robot that gets stuck produces a message
with an owner and a reason.

## 5. Regression: M1 is intact

| Check | Result |
|---|---|
| M1 acceptance suite (17 checks) | **PASS** (re-run after the M2 refactor) |
| M2 three consecutive baseline runs | **3/3 identical**: 10.12 s, same four stages |
| Ablations *with the gate active* | **4/4 still fail** — drone → nothing; go2 → 1 stage; panda → 2 stages; g1 → `STATION_FAILED` at 3 stages |

The constraint layer did not weaken the dependency chain: the ablations still
prove each robot is load-bearing.

## 6. Cost

| Metric | M1 | M2 |
|---|---|---|
| peak RSS | 2.93 GiB | **1.56 GiB** |
| mission time | 10.2 s | 10.12 s (unchanged) |
| sim speed | 15.1× | ~15× (same scene) |
| scene build | 14.1 s | 14.1 s |
| disk free | 29.1 GiB | **21.6 GiB (96 % used)** — see the risk note |

Disk fell to 21.6 GiB during M2 (repeated scene builds, subprocess trials, and
unrelated caches elsewhere on the machine: `~/.cache/uv` alone is 7.9 GB; the
Genesis shader cache is 276 MB). No checkpoint or asset was deleted. This
remains the standing blocker for multi-environment work (M5), as flagged in M0.

## 7. What M2 does not establish

- No general collision avoidance, and no claim of safety beyond the tested
  scenarios.
- **G1 recovers from pushes, not from falls.** Once it is down, it stays down.
- The disturbance envelope is measured for *torso* impacts of 0.25 s duration.
  Head, foot and sustained-contact disturbances are untested.
- The dog cannot manoeuvre around a blocker; only the drone exercises a
  spatial yield.
- Three robots are still scenario-configured rather than dynamically assigned;
  task reallocation under interference is M4's problem.

## 8. Files added in M2

```
ecosystem/core/proximity.py     cylinder-envelope monitoring + look-ahead test
ecosystem/core/safety.py        token-based gate (structural enforcement) + audit
ecosystem/missions/relay.py     constraints, yield strategies, deadlock reporting
ecosystem/tests/test_m2.py      M2 acceptance (21 checks)
ecosystem/tests/test_g1_disturbance.py  Experiment B, one trial per process
ecosystem/results/m2_acceptance.json
ecosystem/results/m2_g1_disturbance.json
```

## 9. Recommended next step (M3)

The measurements point at one thing before any fleet scaling:

**Give the Go2 a yaw-then-drive primitive** and re-run the blocked-lane
scenario. Today a stationary robot in the dog's lane converts a coordination
problem into a deadlock; with yaw the same conflict becomes resolvable, and the
coordinator's `reroute` strategy (already written and currently unused) becomes
testable rather than theoretical. That single primitive would turn M2's deadlock
case into a genuine resolution case — a much stronger claim than adding a fifth
robot.
