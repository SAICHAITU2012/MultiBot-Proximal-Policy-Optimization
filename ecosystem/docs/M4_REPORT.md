# M4 REPORT — dynamic obstacle replanning (closed-loop coordination)

Scope: the advisor-approved M4 (moved-blocker replanning). No retraining, no
policy changes, no team_lab changes; all work inside `ecosystem/`.

## What was built

- **`core/replan.py` (M4.1)**: route VALIDATION and PLANNING as first-class
  modules. `segment_clear` checks every remaining leg against every other
  robot's cylinder envelope using SEGMENT-to-center distances (a leg can graze
  an envelope between ticks); `plan_candidates` generates BOTH side routes;
  `pick_route` validates each from the current position and returns the first
  valid one (or none — the honest no-route case).
- **`missions/relay.py` (M4.2)**: the closed loop — per-tick route
  re-evaluation; replanning gated by a BUDGET (max 5) and a THRASH FUSE
  (≥2 s apart); RESUME_DIRECT (a blocker that moved away drops the detour and
  resumes the straight drive); the honest reach gate (a reach is suppressed
  while in conflict, while the approach touched the humanoid, and on
  unvalidated best-effort legs).
- **`tests/test_m4.py` (M4.3)**: 10 checks over 6 scenarios. The moving
  blocker is a scripted teleport of the standing G1 (the M2 wrench pattern —
  the disturbance is scripted; the system's response is what is tested).

## The acceptance table (the advisor's scenarios)

| Test | Result |
|---|---|
| T1 static blocker | completes, 15.5 s, 1 reroute, efficiency 0.897 |
| T2 blocker moves away | completes 15.5 s, **RESUME_DIRECT** fired, efficiency 0.898 |
| T3 blocker moves into the route | completes 21.4 s using plan-time avoidance (0 actual replans), no contact (min_gap +0.435) |
| T4 two safe routes | the planner selects a route and completes in 15.5 s |
| T5 no safe route (the G1 0.10 m from the crate) | honest timeout at stage reached; 5 replans; min_gap +0.344 m |
| T6 blocker repeatedly moves (6 teleports) | completes; 1 reroute, 0 route-invalidating replans, no contact (min_gap +0.378 m) |
| separate live invalidation suite | `tests/test_m4_replan.py`: **9 PASS, 0 FAIL, 0 SKIP**; one committed-route replan and a 5-replan budget run |

## Metrics (the advisor's additions)

- **Replanning latency**: corrected to milliseconds on 2026-11-16. In the
  current T5 run, five no-route planning calls averaged **1.7388 ms** (maximum
  **1.7532 ms**). The separate live invalidation scenario measured **0.7130
  ms** for its one route replacement. These are single-run measurements, not
  a controlled performance benchmark.
- **Unnecessary replans**: 0 BY CONSTRUCTION — the re-evaluation order
  (validate first, replan only on an invalid route) makes them impossible;
  the metric is reported to prove the order holds.
- **Route efficiency** (the straight-line distance ÷ the travelled path):
  T1 0.897, T2 0.898, T3 0.721, T5 0.137 (the honest failure), T6 1.065.
  Values above 1 can occur when the target moves during a run.

## Latency provenance correction and revalidation — 2026-11-16

`core.replan.pick_route_ladder` returns elapsed seconds from
`time.perf_counter()`. Before this correction, `missions/relay.py` stored that
value directly in a field named `replan_latencies_ms`; it therefore mislabeled
seconds as milliseconds. The field now stores `lat_s * 1000`. This is an
instrumentation-only change; route selection, safety decisions, and robot
commands are unchanged.

The committed `results/m4_acceptance.json` contained these five T5 samples,
in seconds despite the `_ms` field name:

```text
0.0020743749919347465, 0.002004416994168423, 0.0020943340059602633,
0.0020465420093387365, 0.0020219590078340843
```

The pre-existing worktree-only data diff contained:

```text
0.0016120420186780393, 0.0016312090156134218, 0.001626667013624683,
0.0015725830162409693, 0.001571583008626476
```

At the start of this task, that diff changed only those result values; it had
no command, source revision, configuration, or captured test output. Their
correct conversions are 2.004–2.094 ms and 1.572–1.631 ms, respectively, but
the lower values have no defensible provenance and do not establish a
performance improvement. The earlier 0.3–0.6 ms report claim is withdrawn as
unsupported.

Revalidation ran from `ecosystem/` in the `genesis` conda environment on Git
HEAD `bdd8d874a529ac69dda4425e0d86142c1d34bac2` plus the unit-only change
above. From an activated `genesis` environment, rerun with:

```text
python -u tests/test_m4.py
python -u tests/test_m4_replan.py
```

Both exited 0: M4 **10/10**, live invalidation **9 PASS, 0 FAIL, 0 SKIP**.
The current T5 samples are 1.741334, 1.752333, 1.753208, 1.744333, and
1.702667 ms. The single route replacement in the live invalidation scenario
measured 0.712958 ms. Transcripts are
`results/m4_acceptance_revalidation.log` and
`results/m4_replan_revalidation.log`; machine-readable results are
`results/m4_acceptance.json` and `results/m4_replan_verification.json`. M1–M3
regression suites were not rerun in this pass.

SHA-256 for the revalidated sources:

```text
core/replan.py          9488a43fdda8a315612ec6a52a118015f01edad25539dbb1183bdc00665d5e45
missions/relay.py       916b397f29978bff6b97598b6623e6badb186791f04d9fb5b86fef30b449debb
core/world.py           1217365ae9d30218004ceea660001ff48350a373c7a4ff4b73fee0355d285a76
tests/test_m4.py        a56829503f7570dd468de5a9c026ea6e919f0290abe59271377238194f48d1b5
tests/test_m4_replan.py 74fff2dd7ac9ba14a0c5ea2e0210e6c3cd39cbc6b2044b64f7a9694d3debe0ed
```

The checked-in M4 revalidation transcript's local artifact-path prefix was
normalized to `<workspace>/ecosystem/` for portability. The recorded test
output and measured values were otherwise left unchanged.

## The four system findings M4 produced (each caught by a trace)

1. **The validator was stricter than the gate.** Requiring the full SAFE_GAP
   on every leg declared the dog's own gate-legal escape transits invalid and
   burned the replan budget without progress. The standard is now the
   SHALLOW-DIP rule: a leg may dip below the better endpoint distance by
   ≤0.20 m (M3's successful squeeze dipped 0.17; a straight-through line dips
   0.90).
2. **The gate's look-ahead must exceed the gait's stopping distance.** The
   dog slides ~0.7 m after the brake command; braking 0.3 m ahead guaranteed
   contact in the impossible geometry. The look-ahead is now 0.75 m — this
   eliminated EVERY contact in every scenario and sped the reroutes up
   (T1: 26.9 → 15.4 s, the earlier braking = the earlier reroute = the
   cleaner geometry).
3. **A deep-contact ceiling on the escape clause.** "The gap is improving"
   would permit plowing THROUGH an obstacle (the gap improves after the
   midpoint). The permit now requires the surface gap > −0.05.
4. **The reach needs route provenance.** The T5 "false success" was subtle:
   the planner verdicted NO SAFE ROUTE, the best-effort fallback executed, the
   dog squeezed through a 0.01-m gap and reached the crate's ring WITHOUT
   contact — positionally true, contradictory to the planner's own verdict.
   The reach is now suppressed on unvalidated legs; the honest outcome is the
   timeout at stage reached.

## Honesty propagation into the M2 suite (disclosed)

Two M2 expectations changed, both because M4's honesty exposed dirt the M2
scenarios unknowingly contained: (1) the drone-scenario G1 sat 0.79 m from the
crate — the dog's reaching there was a dirty reach (the G1 relocated to a
≥1.5 m position); (2) the blocked-lane G4 now reports a TIMEOUT (the
progress-tracked deadlock lets a manoeuvring dog run to the budget) instead
of the deadlock verdict — the check's intent ("reported, not silently
stalled") is unchanged and still asserted.

## Honest limits

1. The moving blocker is a scripted teleport, not a walking humanoid: the
   system reacts to position changes, not to predicted trajectories.
2. One blocker per reroute; the replan candidates cover two sides only.
3. The route validation uses straight legs; the dog's arcs have an execution
   tolerance (−0.15 m surface) rather than a true arc model.
4. Still no velocity obstacles / trajectory prediction — the coordinator
   reacts, it does not anticipate.
