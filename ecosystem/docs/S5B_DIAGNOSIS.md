# S5b diagnosis — reach and clearance are incompatible in this fixture

**Date:** 2026-11-16

**Status:** read-only diagnosis; M5 remains open. No mission, planner,
safety, or controller behavior was changed for this review.

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

## Capability boundary

The ecosystem mission navigates the Go2 toward the crate center. It loads the
Go2 navigation policy and uses waypoint steering; it does not select an
object-contact pose, choose a push direction, or use the separate validated
Go2 push controller from `team_lab`. The current run therefore tests the
navigation coordinator under this geometry, not a dedicated manipulation
controller.

## Candidate follow-up experiments

| Experiment | Measurable acceptance criteria |
|---|---|
| Keep the 0.10 m overlap as an infeasible-start case | A preflight check reports that no safe reachable pose exists before robot motion; the mission aborts with an explicit infeasible-geometry reason and never reports `CRATE_REACHED` or push success. |
| Test safe navigation to an adjacent crate | Use non-overlapping footprints and a fixture with a non-empty intersection between the 0.60 m crate-reach region and the region at least 0.88 m from the column. For axis-aligned 0.35 m squares, start with at least 0.40 m center separation. Require mission completion, `CRATE_REACHED`, zero Go2/column contact, and recorded minimum clearance. This is a navigation test, not a push-out claim. |
| Test actual push-out | Represent crate geometry and crate/column contact; choose a blocker placement and push direction with a verified safe contact pose and a collision-free path to it; use an explicit push controller. Keep the existing displacement and final-distance gates. For five fixed-seed trials, require 5/5 mission completions, at least 0.40 m crate displacement, at least 0.50 m final center distance, zero Go2/column contact, and logged crate/column footprint clearance and object-contact events. |
| Improve route diagnosis | Log each candidate's rejected segment, blocker, and clearance. Compare the eight-candidate ladder result against an independent feasibility check so candidate exhaustion is never reported as a proof of general geometric impossibility. |

The next experiment should select one of these task meanings before changing
runtime behavior. The current S5b gate remains in force until a reviewed change
is supported by new evidence.
