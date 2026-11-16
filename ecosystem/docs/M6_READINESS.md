# M6 readiness review — locomotion prerequisite is unmet

**Date:** 2026-11-16

**Status:** assessment only. M6 implementation has not started; this review
does not change the standing controller or runtime code.

## Readiness assessment

| Prerequisite | Evidence reviewed | Status |
|---|---|---|
| M5 closure | The last complete strict acceptance result is 15/16. Original S5b still fails. Separate feasible push-out evidence passes 5/5, but does not replace that gate. | **Not met** |
| Fresh M1–M5 regression | M1 17/17 and M2 21/21 passed on 2026-11-16. M3 stopped before acceptance because the frozen G1 `env_args.yaml` is marked `dataless` and reads as zero bytes; M3-turn, M4, M4-replan and revised M5 were not rerun. | **Incomplete** |
| G1 standing | The Genesis 0.3.13 standing baseline held for 10 s in 4/4 trials. The Genesis 1.4.5 port held 10 s in the recorded test, with height 0.800 m to 0.782 m, maximum tilt 1.6°, and no fall. | **Met for standing only** |
| G1 locomotion | The port sends a zero velocity command. The reviewed Genesis-Humanoid and ecosystem evidence contains no validated G1 walking checkpoint or waypoint trials. `g1-calib` logs contain iteration-zero records; training code alone does not establish a working gait. | **Not met** |
| Multiple G1 instances | `World` and `Proximity` use singleton G1 fields and the fixed identity `g1`. There is no evidence of two independently controlled G1 instances. | **Not implemented** |
| M6 specification | `M6_SPEC_DRAFT.md` defines proposed interfaces and M6.1–M6.7 criteria. The draft still requires review and approval. | **Draft** |

The existing controller demonstrates balance while standing. It does not
demonstrate commanded translation, walking, waypoint navigation, or a second
G1's independent state. The M1 4-robot mission and the M5 scripted-G1 scenario
do not fill that gap.

## Gate before M6 implementation

Keep the existing standing controller and its evidence frozen. Close M5 using
its existing acceptance criteria, including the original S5b result and a
fresh full regression when the G1 configuration input is readable. The
separate five-seed S5b.2 result is useful capability evidence but is not an
M5 pass. Separately validate a G1 locomotion
controller in Genesis 1.4.5 with five fixed-seed waypoint trials. Each trial
must:

1. travel at least 1.0 m in the commanded direction;
2. remain above the existing fall threshold;
3. report completion, path length, elapsed simulation time, and minimum safety
   gap; and
4. record the controller policy and configuration hashes.

After those gates pass, review and approve the M6 draft. The current evidence
does not meet the M5 or locomotion gates. A Go2 multi-robot prototype could be
a separate experiment, but it would not establish multi-G1 locomotion
readiness.

## Proposed evidence after the gate passes

The M6 draft's targets remain proposals, not results:

- two G1s use stable, separate identities, policies, commands, observations,
  counters, and controller memory;
- over 10 s, each receives 500 policy updates at 50 Hz and 2,000 torque updates
  at 200 Hz;
- proximity checks prevent robot contact and falls;
- both complete a constrained-passage scenario in 5/5 fixed-seed trials;
- a moving robot inserted onto a committed path triggers an observed
  invalidation, replanning, validated replacement, and completion; and
- held-out reliability reaches at least 9/10 with zero contacts and falls,
  with timing, paths, clearances, replans, waits, and failures reported.

These tests depend on per-instance state and a validated locomotion controller.
They should follow a deterministic baseline before any learned coordination,
LLM planner, or world-model integration. SuperMimic and prediction experiments
remain separate research work until they beat a measured baseline.

All reviewed G1 evidence is simulation evidence. It does not establish hardware
readiness.
