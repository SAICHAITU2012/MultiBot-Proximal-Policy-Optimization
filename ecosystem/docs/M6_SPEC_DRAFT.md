# M6 draft — multiple moving robots

**Status:** planning only, 2026-11-16. No M6 implementation has started.
Implementation depends on M5 closure. This document does not change runtime
code or the `team_lab` M8 baseline.

## Goal

Demonstrate two robots moving independently, maintaining separate identities
and state, sharing constrained space safely, responding to each other's
motion, and recovering from selected coordination failures. Use deterministic
coordination first; consider learned coordination only after it beats the
reproducible deterministic baseline.

## Readiness gate before M6.1

The current `World` and `Proximity` implementation each expose one G1 through
the singleton `g1_e` / `g1` fields and the identity `g1`. The G1 policy uses a
50 Hz standing controller with a zero velocity command. It demonstrates a
ported standing policy, not independent walking or navigation. Two standing
G1s therefore cannot meet M6's movement objective.

Before claiming multi-G1 coordination, a separately validated G1 locomotion
controller must pass five waypoint trials: each travels at least 1.0 m in the
commanded direction, stays above the existing fall threshold, and reports
completion, path, elapsed time, and minimum safety gap. Record the policy and
configuration hashes. Keep the standing controller and its validated baseline
unchanged. If this prerequisite is unavailable, defer multi-G1 movement claims
or choose an already mobile embodiment for an explicitly separate prototype.

## Proposed interfaces

- **World registry:** store each instance under a stable ID, such as
  `g1_0` and `g1_1`; each entry owns its simulator entity, policy/controller,
  command, tick counters, and controller memory. Do not share action history,
  low-pass state, or observations between instances.
- **State snapshot:** expose ID, simulation timestamp, base position and
  orientation, linear and angular velocity, joint position/velocity, and
  health/fall status. A coordinator reads snapshots and emits commands; it
  does not write torques directly to simulator entities.
- **Safety registry:** track both robot IDs as separate bodies with the same
  G1 envelope. Report per-pair distance, minimum gap, conflicts, and contacts.
- **Coordinator:** a deterministic `FleetCoordinator` consumes the complete
  state snapshot and assigned goals, then returns one namespaced command per
  robot. It must not assume a fixed robot count.
- **Clock:** keep the current shared 200 Hz physics clock. Service each G1
  policy at its validated 50 Hz rate (one policy update per four physics
  steps) and recompute torques on every physics step at 200 Hz.

## Milestones and acceptance

| Stage | Capability | Minimum acceptance evidence |
|---|---|---|
| M6.1 | Spawn two G1 instances with independent IDs | Both entities and controllers exist; commands and observations are namespaced; one instance's reset or update leaves the other's state unchanged |
| M6.2 | Independent state and timing | Over a 10 s run, each receives 500 policy updates at 50 Hz and 2,000 torque updates at 200 Hz; report per-instance tick counts, state traces, and command traces |
| M6.3 | Pairwise proximity and conflict handling | Continuously report pairwise gap; demonstrate a safety response before contact; zero robot-to-robot contact and zero falls |
| M6.4 | Shared constrained passage | Both robots complete a single-lane crossing in 5/5 fixed-seed trials, with no contact or fall; report order, waits, minimum gap, and completion time |
| M6.5 | Respond to motion after route commitment | Insert the second robot onto the first robot's committed route outside the instantaneous conflict threshold; assert route invalidation, at least one actual replan, validated replacement, resume, completion, and no contact |
| M6.6 | Deadlock detection and recovery | Start a reproducible mutual-blocking case; report the correct deadlock cause, execute a defined recovery, and show both robots resume or terminate honestly within a fixed budget |
| M6.7 | Held-out reliability | At least 9/10 held-out trials complete; zero contacts and falls; publish p50/p95 completion time, path length, minimum gap, replans, waits, and failures |

The 5/5 and 9/10 thresholds are proposed acceptance targets, not existing
results. Each scenario must distinguish scripted disturbance setup from the
system's measured response, following the M4 live-invalidation test pattern.

## Evidence and boundaries

Every run should record command, Git revision, source hashes, simulator
version, physics/control rates, seed, scenario geometry, policy hash, per-check
result, and raw transcript. Separate implemented, tested, and independently
verified claims. Simulation success is not hardware validation.

M6 begins only after M5 is closed and the locomotion readiness gate passes.
Do not begin with multi-agent RL, LLM planning, or world-model integration;
first establish a deterministic baseline and failure traces. Keep
SuperMimic/world-model research independent until an experiment demonstrates
measurable improvement over a simple baseline.
