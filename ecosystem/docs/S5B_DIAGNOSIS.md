# S5b diagnosis — reach and clearance are incompatible in this fixture

**Date:** 2026-11-16

**Status:** original S5b diagnosis preserved; a separate feasible push-out
experiment is now recorded below. M5 remains open. No mission, planner,
safety, or controller behavior was changed.

## Finding

The S5b run does not establish push-out capability. Its crate moves, but the
current safety contract leaves no Go2 position that both reaches the crate and
keeps the required clearance from the column. The planner then exhausts its
finite route-candidate ladder and terminates honestly.

The relevant geometry in `tests/test_m5.py` is:

- crate center: `(0.0, 1.2)` m;
- column center: `(0.1, 1.2)` m;
- both square footprints: `0.35 × 0.35` m.

The footprints overlap by 0.25 m in x and 0.35 m in y. Their vertical spans
also overlap. The safety monitor models the Go2 and column with center radii
0.36 m and 0.30 m and a required surface gap of 0.22 m. It therefore requires
their centers to remain at least `0.36 + 0.30 + 0.22 = 0.88 m` apart. The
mission's crate-reach distance is 0.60 m. Since the crate and column centers
are only 0.10 m apart, any Go2 position within reach of the crate is at most
`0.60 + 0.10 = 0.70 m` from the column. That is 0.18 m inside the required
center clearance. The current reach and safety conditions cannot both hold in
this fixture.

This is a reachability conflict under the configured circular safety envelope,
in addition to the overlapping square footprints at the start. A better route
search cannot make the two constraints compatible.

## What the recorded run shows

The M5 acceptance artifact at the `e0ee4c5` evidence revision records:

- run ended at 12.7 s with `NO_GEOMETRIC_ROUTE` and no `CRATE_REACHED`;
- crate displacement: 0.2196 m (required: at least 0.40 m);
- final crate/column center distance: 0.1797 m (required: at least 0.50 m);
- minimum crate/column center distance during the run: 0.0385 m;
- Go2/column minimum monitored surface gap: 0.58 m, with no monitored
  Go2/column contact;
- 2,540 safety requests: 1,008 allowed and 1,532 blocked; 1,530 wait ticks;
- 2.348 m Go2 path and one reroute attempt.

The contact monitor tracks robot-to-column contact. It does not track crate to
column contact, so the center-distance trace is the available crate/column
measurement in this run. The current gate passes only when the mission
completes, `CRATE_REACHED` occurs, crate displacement is at least 0.40 m, and
final center distance is at least 0.50 m. S5b fails that unchanged gate.

`NO_GEOMETRIC_ROUTE` means the eight candidates in `plan_ladder` (two sides by
four offsets, with straight waypoint legs) were rejected by route validation.
It does not prove that every possible continuous path was searched. In this
fixture, though, the empty safe-reach set is a separate geometric reason the
current task cannot complete under its stated constraints.

The reported `safety.executed: 0` counts calls successfully recorded through
`ConstraintLayer.execute()`. The relay's `allowed_execute` helper has no call
site, so this value does not count simulator motion commands. The crate and
Go2 path measurements show that the simulation did move.

## Separate follow-up result: S5b.2 feasible push-out

The original S5b fixture and its strict acceptance gate remain unchanged. A
separate experiment used the hash-pinned Go2 push checkpoint in a layout with
a reachable west-side contact pose and an eastward push that increases
blocker clearance. It matches the checkpoint's saved crate size (0.35 m),
height (0.25 m), mass (7 kg), 1.20 m robot/crate spawn separation, and 50 Hz
policy rate. The policy weights and runtime safety checks were not changed.

All five fixed seeds (`11, 23, 37, 53, 71`) passed. Across the trials:

- crate displacement along the push direction: **0.401–0.437 m**;
- final crate/column center distance: **1.402–1.551 m**;
- Go2/column minimum surface gap: **0.358–0.649 m**, above the 0.22 m
  requirement;
- measured continuous Go2/crate contact: **0.62–1.60 s**;
- falls, Go2/column contacts, crate/column contacts, and safety denials: **0**;
- crate-footprint edge clearance gain: **at least 0.050 m** in every trial.

The completion label means the experiment-specific push-out gate passed. The
policy's 2.20 m target was not reached in any trial (`policy_target_reached`
is false throughout). This is evidence for the separate feasible push-out
capability, not a successful run of original S5b and not M5 closure. The
configuration, source and checkpoint hashes, per-seed traces, and criteria
are in `results/s5b2_feasible_push.json`; the earlier 4/5 fixture attempt is
preserved in `results/s5b2_feasible_push_geometry_v1_failed.json`.

The added expected-infeasibility arithmetic for original S5b evaluates to a
0.18 m deficit (0.70 m maximum reachable center distance versus 0.88 m
required). The revised full M5 suite has not completed because its G1
`env_args.yaml` dependency is marked `dataless` on macOS and reads as zero
bytes. The standalone geometry evaluation passes, but no new full M5
acceptance result is claimed.

## Capability boundary

The ecosystem mission navigates the Go2 toward the crate center. It loads the
Go2 navigation policy and uses waypoint steering; it does not select an
object-contact pose, choose a push direction, or use the separate validated
Go2 push controller from `team_lab`. The current run therefore tests the
navigation coordinator under this geometry, not a dedicated manipulation
controller.

## Remaining follow-up work

| Experiment | Measurable acceptance criteria |
|---|---|
| Original S5b infeasible-start semantics | Keep the exact 0.10 m fixture and strict push-out gate. The geometric assertion is now present in `tests/test_m5.py`; complete a full M5 rerun when the frozen G1 config is readable. If runtime later gains an infeasibility verdict, test that it occurs before motion and never reports `CRATE_REACHED` or push success. |
| Test safe navigation to an adjacent crate | Use non-overlapping footprints and a fixture with a non-empty intersection between the 0.60 m crate-reach region and the region at least 0.88 m from the column. For axis-aligned 0.35 m squares, start with at least 0.40 m center separation. Require mission completion, `CRATE_REACHED`, zero Go2/column contact, and recorded minimum clearance. This is a navigation test, not a push-out claim. |
| Extend push-out distance and robustness | The five-seed feasible push-out gate passed. To claim the full saved policy goal or stronger manipulation capability, define a separate target-distance gate and test more seeds and initial crate poses; keep original S5b separate. |
| Improve route diagnosis | Log each candidate's rejected segment, blocker, and clearance. Compare the eight-candidate ladder result against an independent feasibility check so candidate exhaustion is never reported as a proof of general geometric impossibility. |

The next experiment should select one of these task meanings before changing
runtime behavior. The current S5b gate remains in force until a reviewed change
is supported by new evidence.
