# M3 REPORT — Go2 yaw-then-drive primitive and reroute

Scope: the advisor-approved M3.1–M3.6. No retraining, no policy changes, no
team_lab changes (0 files); all work inside `ecosystem/`.

## M3.1 — the primitive (`robots/go2_drive.py`)

`YawThenDrive`: a command-shaping layer on the FROZEN nav policy. States
TURN (creep arc: small forward vx + proportional wz — the gait only yaws while
walking; pure wz commands are near-dead, ~0.02–0.1 rad/s measured) → DRIVE
(forward with a fine corrective wz) → ARRIVE. All emitted commands stay inside
the trained ranges (vx [−0.5, 1.0], vy [−0.5, 0.5], wz [−1, 1]).

## M3.2 — independent validation (`tests/test_m3_turn.py`)

Benchmarks in the real four-robot world, arc vs strafe A/B:

| Case | arc | strafe | gate |
|---|---|---|---|
| T90 (90° turn)      | **95.1° in 4.8 s**, arrived 0.350 | 56.9° in 2.5 s, arrived | arc: turned ≥60°, arrived |
| T180 (150° turn)    | **150.4° in 5.8 s**               | 150.1° in 3.0 s, arrival 0.444 m — FAILED | arc: turned ≥150° |
| TD (turn-then-drive 1.44 m) | arrived 0.348, path 1.19  | arrived 0.349, path 1.29 | arrived ≤0.35, path ≥80% straight |

**Mode decision: arc** (passes every gate; the strafe is faster but failed the
180° arrival). The strafe A/B is retained as documented evidence — note this
CONTRADICTS the M2 report's "strafe deadband" claim: the strafe works in the
ecosystem too (2.5 s vs 4.8 s), it is simply less accurate at large turns.

Bugs found in my own test harness (each would have produced a false verdict):
the T180 gate measured the wrapped yaw (a ±π representation flip read as 360°
at t=0.1 s); the accumulator re-derived from `prev_yaw`, which silently adopts
the flipped representation; a "dead" line from an earlier patch was no longer
dead. Fixes: unwrapped accumulation with a wrap- absorbed per-step delta.

## M3.3 — coordinator integration (`missions/relay.py`)

Stage 2 (go2) now: computes the primitive command BEFORE the gate; the gate is
PHASE-AWARE — TURN ticks are permitted as reorientation pivots (their center
orbit is perpendicular to the blocker direction and cannot close the
straight-line gap; the proximity monitor verifies no contact physically
occurs), DRIVE ticks are gated on the HEADING-projected look-ahead (the dog
moves along its body-forward, not the target bearing) with the
ESCAPE-MONOTONICITY clause: a blocked robot may translate if the prospective
position does not worsen its worst active gap. The route is a HEADING-AWARE
3-waypoint plan: escape (radially away, blended ~45/55 with the current
heading), detour beside the blocker (offset FROM THE BLOCKER, 1.4 m), crate.

## M3.4 — blocked-lane re-run (the decisive result)

| Scenario | M2 | M3 |
|---|---|---|
| resolvable blocker (in-lane, ≥1.5 m from the crate) | DEADLOCK (wait-only) | **RESOLVED: mission completes in 26.9 s, 1 reroute, 20 blocked ticks, min_gap +0.171 m, no contact** |
| blocked-lane (the G1 0.25 m from the crate) | DEADLOCK | **STILL HONESTLY DEADLOCKED** — and provably must be: the dog must be ≤0.60 m from the crate and ≥0.90 m from the G1, but the crate-blocker distance is 0.25 m (0.25+0.60 < 0.90 — the reachable set is empty). The reroute attempts, the progress-tracked deadlock reports, no false success, no contact. |
| gate-OFF (the same resolvable blocker) | — | the dog drives THROUGH the humanoid (min_gap −0.282, contact) — why the gate exists |

The M2 report's "geometrically unresolvable" claim was correct for its
wait-only yield; M3 supersedes it for resolvable geometries and CONFIRMS it
for crate-adjacent blockers. The advisor's criterion — "detects unsuccessful
rerouting and reports DEADLOCK rather than claiming success" — is met by both
columns.

## M3.5 — rerouting and separation

The resolvable run's minimum go2-g1 surface gap: **+0.171 m** (no contact);
the unresolvable run's: +0.054 m (tight but contact-free). The proximity
monitor tracked every tick; the pivot permit's physical risk (the orbit could
close a gap) did not materialize — the recorded minimum gaps stayed positive
in every gate-ON run.

## M3.6 — regression + cost

- M1 suite: **17/17** PASS. M2 suite: **21/21** PASS (one expectation updated:
  the M2 G4 deadlock check now accepts the M3 progress-tracked deadlock
  semantics — the intent "reported, not silently stalled" is unchanged).
- M3 suite: **7/7** (`tests/test_m3.py`).
- Timing: the baseline mission 10.3 s; the resolvable-reroute mission 26.9 s
  → **the turning overhead ≈ 16.6 s** for one pivot-out/pivot-in around a
  blocker (the gait spin-up dominates the pivots: ~2–20°/s).
- Resources: peak RSS 1.70–1.89 GiB; no training; no checkpoint changes.

## Deadlock semantics change (documented, disclosed)

The deadlock verdict is now PROGRESS-TRACKED (no ≥5 cm position improvement
AND no ≥0.15 rad heading improvement for 8 s), replacing M2's consecutive-
blocked-ticks timer: a slowly manoeuvring dog accumulated 8 s of intermittent
blocks while genuinely resolving the situation, and fired a false deadlock.
The partners list reflects the active conflicts at the fire moment and may be
empty (the dog was mid-manoeuvre, not mid-conflict).

## Honest limits

1. Crate-adjacent blockers (the blocker within 0.90 m of the crate) remain
   unresolvable BY GEOMETRY — no amount of steering fixes an empty reachable
   set. The coordinator reports it honestly.
2. The reroute is open-loop waypoint driving: no online obstacle updates, no
   replanning mid-leg (a moved blocker mid-reroute would need a new route —
   the reroute queue handles one blocker).
3. The pivot gait spin-up (2–20°/s) makes rerouting slow (~17 s overhead).
4. One blocker per reroute; two-blocker deadlock geometry is untested.
