# M1 REPORT — heterogeneous four-robot ecosystem (2026-11-12)

**Verdict: M1 PASS — 17/17 acceptance checks**, with the existing two-package
mission still delivering **2/2** (unchanged regression).

Scope held to your instructions: nothing in `team_lab/` or `Genesis-Humanoid/`
was modified; no checkpoint was deleted or replaced; no training was run; no AMR
was introduced.

---

## 1. What M1 actually demonstrated

| Criterion (your wording) | Evidence |
|---|---|
| operate on a **shared simulation clock** | one `World.step()` at 0.005 s; 1000 steps = 5.00 s exactly; per-robot tick counts prove one clock drives all four |
| maintain **independent controller states** | 4 separate entities, 4 separate action buffers, 4 different obs/action contracts: go2 45→12, drone 17→4, panda 14→6, g1 96→29 |
| **exchange task/state information** | 8 bus messages per mission, 942 coordinator state reads; every stage transition is triggered by another robot's published report |
| demonstrate **one coordinated multi-robot task** | 4-stage dependency chain, completed 3/3 runs at t=10.2 s, plus 4 ablations that each make it FAIL |

The coordinated task, `missions/relay.py`:

```
drone --CRATE_SIGHTED--> go2 --CRATE_REACHED--> panda --PARCEL_DISPATCHED--> g1 --STATION_STABLE--> DONE
```

Each event is computed from that robot's own physics (position, joint state,
parcel displacement). The coordinator holds no robot state of its own; it reads
reports. **The panda is not even allowed to move until the go2 reports arrival** —
it holds its ready pose until released (`external_control`), so its task cannot
complete on its own schedule.

### The ablations are the real proof

| Disabled robot | Mission result | What the team got through |
|---|---|---|
| — (baseline) | **PASS at 10.2 s** | all four stages |
| drone | FAIL (timeout) | nothing — no crate ever sighted |
| go2 | FAIL (timeout) | `CRATE_SIGHTED` only |
| panda | FAIL (timeout) | `CRATE_SIGHTED`, `CRATE_REACHED` |
| g1 (physically falls) | **FAIL — `STATION_FAILED`** | 3 stages, then the G1's collapse is detected and reported |

Every robot's contribution is load-bearing. Four robots merely co-existing
would pass none of these.

---

## 2. The G1 port (no retraining) — the milestone's main technical result

The standing policy trained in Genesis **0.3.13** now stands in **1.4.5**.

Recovered contract, all read programmatically from the checkpoint's own
`env_args.yaml` (nothing hand-transcribed):

| Element | Value |
|---|---|
| actor | obs **96** → MLP(256,256,128, ReLU) → 29 → `mu` (29→29) |
| obs terms (order) | `last_action(29)`, `dof_pos(29)`, `dof_vel×0.1(29)`, `projected_gravity(3)`, `base_ang_vel×0.5(3)`, `commands(3)` |
| ctrl type | `DR_JOINT_POSITION_VELOCITY` — **torque control with explicit PD + low-passed velocity feed-forward** |
| law | `tgt=0.25a`; `v_des=(tgt−tgt_prev)/0.02`; `v_lp += 0.5(v_des−v_lp)`; `τ = kp(tgt+default−q) + kd(−q̇+0.9·v_lp)`, clamped to per-joint torque limits |
| gains | hip/knee kp 100 kd 20, ankle 30/6, waist 150/10, arms 30→10 |
| rate | dt 0.005, decimation 4 → 50 Hz policy; **torque recomputed every 0.005 s** |

Result vs the original measurements:

| | baseline (0.3.13) | port (1.4.5) |
|---|---|---|
| stood full 10 s | 4/4 | **yes** |
| base height | 0.797 → 0.792 m | **0.800 → 0.782 m** |
| max tilt | — | **1.6°** |
| in the 4-robot scene | — | **0.780 m held for the whole mission, no fall** |

Standing only. This says nothing about walking, navigation or manipulation — as
you noted, those stay separate future capabilities.

---

## 3. Three real defects found while building M1 (all fixed, all measured)

1. **Rigid timestep mismatch collapsed the G1.** The existing scene runs 0.02 s
   steps with 0.01 s rigid substeps — twice the G1's training step. At that
   cadence the port falls to z=0.13 m. Fix: one clock at **0.005 s**, with every
   robot serviced on a period derived from its own trained rate.
2. **The drone loses thrust if its rpm is not re-issued every sim step.** Ticking
   it every 4th step → immediate crash (z=0.012). Verified by isolation:
   period 1 hovers (z=1.01), period 4 falls.
3. **The panda completed its own task before the mission started.** It ran its
   reach policy from t=0 and had already displaced the parcel by the time the
   chain began, which made `PARCEL_DISPATCHED` unobservable. Fix: the arm idles
   until another robot releases it — which is also what makes the task
   coordinated rather than co-located.

Also corrected in my own test: an early run "passed" in 6.6 s because the go2's
arrival threshold (0.95 m) was looser than its starting distance (1.0 m) and the
parcel threshold (0.03 m) was crossed by noise. Every gate now requires
substantive motion: drone within 0.30 m **and holding 1 s**, go2 within 0.60 m
**and having walked ≥0.50 m**, parcel moved >0.10 m **and dropped below 0.40 m**,
G1 holding 5 s. That fake pass is exactly the failure mode you warned about.

---

## 4. Machine cost (measured, not estimated)

| Metric | Value |
|---|---|
| backend | CPU (no CUDA on this M4), `precision=32` |
| scene build | 14.1 s (4 robots: 2 URDF, 1 MJCF, 1 drone) |
| simulation speed | **15.1× realtime** during the mission (14.7 s sim in 1.0 s wall) with 4 robots + 4 policies |
| peak RSS | **2.93 GiB** of 16 GB |
| disk free | 29.1 GiB (unchanged; no new assets downloaded) |
| swap | unused |
| guard | `machine_guard.py` available and unchanged; sessions here were seconds-to-minutes, no long jobs launched |

---

## 5. Honest limitations of M1

- The chain is **sequential**, not concurrent: robots hand off, they do not work
  in parallel. Concurrent multi-task allocation is M4's job.
- **No inter-robot collision avoidance.** Robots share physics and can collide;
  nothing plans around another robot yet.
- **One scene, one instance of each robot.** Multi-instance spawning (N Go2s, a
  second drone) is M3.
- The G1 does **standing only** — it is a station-keeper, not a mobile agent.
- No AMR (deferred by your decision).
- The relay is a purpose-built M1 task, not the mission's 7 kg crate delivery;
  that validated pipeline remains in `team_lab` and was regression-tested, not
  rebuilt.

## 6. Reproduce

```bash
cd ~/Desktop/Genesis/ecosystem
conda activate genesis          # + machine_guard for long sessions

python tests/test_g1_stand.py 10        # G1 port vs its 0.792 m baseline
python tests/test_four_robots.py        # M1 acceptance (17 checks, ~90 s)
python tests/test_four_robots.py        # writes results/m1_acceptance.json

# regression on the frozen project (unchanged code):
cd ../team_lab/team && python mission.py --packages 2 --planner rule --max-steps 60000
# -> recon 2/2 | delivered 2/2
```

Artifacts: `results/m1_acceptance.json`, `results/m1_repeatability.json`.

## 7. Files added (all new, nothing existing touched)

```
ecosystem/
  core/world.py            one scene, 4 robots, one 0.005 s clock, per-robot rates
  robots/g1_stand.py       G1 standing port (exact PD + feed-forward contract)
  missions/relay.py        coordinated 4-stage task + message bus + ablations
  tests/test_g1_stand.py   port acceptance vs the 0.3.13 baseline
  tests/test_four_robots.py M1 acceptance (17 checks)
  docs/M1_REPORT.md        this file
  robot_registry.json      M0 machine-readable inventory
```

## 8. Recommended next step (M2)

Per-robot validation in the ecosystem context, before any scaling:

1. G1: measure how long it can hold, and its recovery after a small push
   (adds the missing "disturbance" dimension to the standing claim).
2. Go2: does the nav policy still track commands at 25 Hz with 3 other robots in
   the collision world? (rate was reduced to preserve validated behaviour —
   prove it didn't degrade.)
3. Panda: measure reach accuracy with the G1 occupying nearby space.
4. Add robot-robot proximity as a hard constraint in the coordinator (M2 item 3
   in the inventory's gap list) so the fleet cannot collide by construction.
