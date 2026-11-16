# M5 REPORT — multi-obstacle coordination

Scope: the advisor-approved M5.1–M5.5. No retraining, no policy changes, no
team_lab changes; all work inside `ecosystem/`.

## Current status — 2026-11-16

M5 is **open**. The last complete M5 acceptance run remains 15/16: the
original S5b measured push-out gate fails. A separate, feasible push-out
experiment has since passed five fixed seeds, but it does not replace the
original S5b gate. The M5 runtime, safety behavior, navigation policy, and
robot controllers were not changed. The `team_lab` M8 baseline remains
untouched.

The last complete M5 run used `python -u tests/test_m5.py` from `ecosystem/`
and exited 1 on the unchanged S5b gate. Its evidence is in
`results/m5_acceptance.json` and `results/m5_acceptance.log`. The subsequent
expected-infeasibility assertion in `tests/test_m5.py` evaluates to true
(maximum reachable center distance 0.70 m; required center clearance 0.88 m),
but that revised full suite has not completed, so no new M5 acceptance count
is claimed.

The M5 artifact records the portable command, ecosystem-relative working
directory, Python version, source hashes, and any configured path-override
variable names; it does not store a machine-specific executable or checkout
path. The checked-in M4/M5 transcript artifact-path prefixes were normalized
to `<workspace>/ecosystem/`; no other transcript content was edited for
portability.

## What was built

- **`core/proximity.py` (M5.1)**: a STATIC-OBSTACLE REGISTRY — N obstacles,
  each with independent identity, geometry and movement state
  (`register_static`/`update_static`, per-obstacle `last_move_t`).
  `positions()`/`gaps()`/`violates_at()`/`_span()` are registry-aware: every
  body (robot or obstacle) pairs with every other. **One blocker moving never
  touches another blocker's movement state** (the advisor's safeguard #2, now
  structural). `core/world.py` gains `extra_blockers=[(x, y), ...]` — real
  immovable column entities (0.35×0.35×0.4) registered into the layer.
- **`core/replan.py` (M5.2)**: the N-BLOCKER CANDIDATE LADDER — for each side
  and each lateral offset (0.9/1.4/2.0/2.8 m), a candidate route; every
  candidate validated against ALL obstacles from the current position;
  `pick_route_ladder` returns the first valid (or none — the honest
  NO_GEOMETRIC_ROUTE input). No hardcoded blocker count (the safeguard #1).
- **`missions/relay.py` (M5.3)**: both planning sites use the ladder; the
  deadlock verdict carries a DIAGNOSIS (M5.4): `NO_GEOMETRIC_ROUTE` (the
  ladder exhausted ≥3 consecutive picks) / `TEMPORARY_OBSTRUCTION` (a blocker
  moved within 5 s — the waiting is legitimate) / `BUDGET_EXHAUSTED` (the
  replans hit the cap) / `STALLED_PROGRESS` (progressing but unable to
  finish).

## The acceptance results

| Scenario | Result |
|---|---|
| S1 two in-line columns | PASS: completes in 23.4 s, one reroute, minimum monitored gap +0.244 m |
| S2 flanking columns | PASS for the expected failure: reports `NO_GEOMETRIC_ROUTE` at 19.8 s; minimum monitored gap +0.300 m; no column contact |
| S3 tight flanking | PASS: completes in 21.9 s; minimum monitored gap +0.257 m |
| S4 scripted G1 moves | PASS: completes in 15.5 s; minimum monitored gap +0.246 m. Scripted movement does not establish independent walking/navigation |
| S5a column 1.2 m from crate | PASS: completes in 20.0 s; minimum monitored gap +0.348 m |
| S5b column 0.1 m center-to-center from crate | **FAIL: strict push-out gate.** Crate displacement 0.2196 m; final center distance 0.1797 m (required ≥0.50 m); no `CRATE_REACHED`; deadlock `NO_GEOMETRIC_ROUTE` at 12.7 s |
| S5b Go2-to-column safety | PASS: no monitored Go2-column contact; minimum surface gap +0.580 m. Crate-to-column contact is not monitored by `Proximity` |
| S7 identity isolation | PASS: all four static-obstacle movement-state separation checks |
| M5.5 `plan_ladder` generation | 0.007039 ms (1 blocker), 0.007027 (2), 0.008160 (4), 0.012500 (8), 0.016959 (16) |

M5.5 values are arithmetic means over 200 calls per blocker count, with no
warm-up, on this run. They time candidate generation only; route validation
and simulation are excluded, so they are not end-to-end replanning latency.

**M5: the last complete run passed 15/16 checks; S5b failed, so M5 is not
closed.** The prior 12/12 claim and S5b push-out claim are superseded by this
corrected run. The later separate S5b.2 experiment is recorded below and in
`results/s5b2_feasible_push.json`; it is not counted in this table.

## S5b.2 separate feasible push-out experiment

The frozen Go2 push policy was evaluated in a separate reachable layout using
the checkpoint's saved 0.35 m crate, 7 kg mass, 1.20 m robot/crate start
separation, 50 Hz policy rate, and Genesis 1.4.5. The five fixed seeds
`11, 23, 37, 53, 71` all passed the preflight and push-out completion gates.
The crate moved 0.401–0.437 m along the planned direction; final center
distance was 1.402–1.551 m; measured Go2/blocker surface clearance stayed
0.358–0.649 m (required 0.22 m); continuous physical contact lasted
0.62–1.60 s. There were no falls, robot/blocker contacts, crate/blocker
contacts, or denied safety requests. Measured crate-footprint clearance
increased by at least 0.050 m in every trial.

This experiment's completion target is the explicit push-out gate, not the
policy's farther 2.20 m goal: `policy_target_reached` is false in all five
trials. The experiment demonstrates repeatable, bounded push-out in its own
feasible fixture. It does not make the original S5b trial pass or close M5.
The runner and full per-seed measurements are
`experiments/s5b2_feasible_push.py` and `results/s5b2_feasible_push.json`;
the prior 4/5 layout failure is retained separately as
`results/s5b2_feasible_push_geometry_v1_failed.json`.

## Regression status — 2026-11-16

- M1: **17/17 passed** (`results/m1_regression_2026-11-16.log`).
- M2: **21/21 passed** (`results/m2_regression_2026-11-16.log`).
- M3: attempted against a temporary extraction of Genesis World commit
  `1daee8d5210cdfdfb3e238c592bfabe7f7b11814`; import and scene setup reached
  the G1 configuration loader, then failed because the configured
  `Genesis-Humanoid/.../configs/env_args.yaml` is marked `dataless` by macOS
  and reads as zero bytes. `load_env_args()` therefore returned `None`, and
  the suite stopped before its acceptance checks. The original checkout was
  not modified (`results/m3_regression_attempt_2026-11-16.log`).
- M3-turn, M4, M4-replan, and the revised full M5 suite were not rerun after
  that shared G1 configuration prerequisite failed. Their Genesis worlds use
  the same frozen G1 configuration, so this environment cannot provide valid
  results for them. M4's previously recorded suite results remain documented
  in `docs/M4_REPORT.md`.

Consequently, this is not a fresh full M1–M5 regression sign-off. The current
5/5 push experiment and M1/M2 results are valid evidence, while M3 and the
remaining M4/M5 reruns still need a readable G1 configuration file.

## The geometric understanding M5 corrected

M4's earlier resolvability rule ("the blocker must be ≥1.5 m from the crate")
was derived assuming approach along the blocker-crate line. S5a demonstrates
one reachable 1.2 m geometry. The corrected S5b fixture places the 0.35 m
square crate and 0.35 m square column only 0.10 m center-to-center, so their
XY footprints initially overlap. This tests recovery from an initial overlap,
not passage through a narrow but non-overlapping gap. The run moved the crate
partway but did not clear the column or complete; this evidence does not
support the prior push-out claim or a general 0.3 m reachability threshold.
The crate is not a tracked body in the `Proximity` report, so crate-column
contact is not independently recorded. The strict push-out gate remains in
place for future work.

## The advisor's qualification, carried forward

"Zero unnecessary replans" is guaranteed by the validate-first design (the
re-evaluation replans only on an invalid route), not an independently
measured absence of all route changes. The distinction stands in this report.

## The M5 audit response (2026-11-13; historical)

The independent audit (`docs/M4_AUDIT.md` — a separate session) found:
F1 the M4 suite's T3/T6 were vacuous (the replan path never entered — fixed by
tests/test_m4_replan.py: 9/9 on THIS revision, replans=1 + the fuse at exactly
5, committed artifacts); F2 the dead _reroute_waypoint helper — REMOVED; F3 the
divergent escape-length formula — the dead plan_route/plan_candidates/pick_route
trio REMOVED (the ladder is the single planning implementation); F4 the missing
result artifacts — results/m4_acceptance.json and results/m5_acceptance.json now
written BY the suites; F5 the orbit livelock (motion without goal progress reset
the deadlock timer for 180 s, 27.6 m of path) — FIXED: the deadlock progress is
GOAL-DISTANCE reduction only, and the budget exhaustion is TERMINAL (the
deadlock publishes immediately instead of orbiting to the timeout). At that
time, the advisor's challenges A–D were documented as addressed. The
2026-11-16 evidence revalidation above supersedes the historical S5b push-out
claim; S7 verifies separate static-obstacle identities, not two walking
robots.

## Honest limits

1. The ladder's routes are straight legs; S2's flanking geometry exhausted
   the budget because no straight-leg candidate cleared both envelopes for
   the dog's arc execution — a richer route model (the curved/waypoint
   refinement) is the identified future work.
2. The static obstacles are immovable columns; a MOVABLE second blocker
   (another G1 instance) is the fleet-duplication phase's work.
3. The deadlock taxonomy's TEMPORARY_OBSTRUCTION relies on the obstacle
   movement tracking; a blocker that moves sub-threshold (<5 cm) is not
   detected as "recently moved".
4. The validation is reactive, not predictive (no velocity obstacles).
