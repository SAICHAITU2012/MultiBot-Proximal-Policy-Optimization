# ASSET INVENTORY — M0 (2026-11-12)

Audit of every reusable asset before any ecosystem code was written. Every hash below was computed on this machine (`md5`, first 12 chars). Nothing was modified; this milestone is read-only by design.

**Machine:** Apple M4, 16 GB RAM, macOS, **32 GB free of 460 GB (93% used)**, swap unused.

---

## 1. Verdicts first (what the roadmap depends on)

| # | Finding | Consequence |
|---|---|---|
| **F1** | There are **two incompatible G1 stacks**, not one. | The Genesis G1 is the reusable one. The MuJoCo G1 cannot be dropped into a Genesis scene. |
| **F2** | The Genesis G1 standing policy is `obs 103 → MLP(256,256,128) → act 29`, and its URDF **is on disk**. | Port is plausible **without retraining** — same MLP loader already used for Go2/drone/Panda. Only the obs builder needs porting. Matches the "reuse first" mandate. |
| **F3** | `go2-push-s5/model_1000.pt` exists and is **not** the validated policy (`model_800.pt` is). | The registry pins by **hash**, never by "latest checkpoint". A "use the newest" convention would silently swap a verified policy for an unevaluated one. |
| **F4** | **ROS 2 is not installed** (`which ros2` → nothing). | The proposed ROS 2 middleware layer is a gap. Genesis runs in-process, so M1–M4 do not need it; revisit only if real hardware enters. |
| **F5** | Disk is at **93%** (32 GB free) and 66 GB of it is unpushed Genesis checkpoints. | **This is the one blocker for M1.** Multi-scene work + MuJoCo assets need room. Needs your approval to prune (destructive). |
| **F6** | `~/Desktop/humanoid` **does not exist** — the folder is `~/Desktop/humaniod` (typo). | Any script or doc referencing `Desktop/humanoid` will fail silently. |

---

## 2. Stack A — Genesis mission robots (engine 1.4.5, this repo)

Validated in the delivered mission; all five are PPO MLPs loadable through `colo_env.load_actor_from_log`.

The **Rate** column records the upstream policy/checkpoint cadence from the M0
asset audit. It is not the ecosystem's current runtime service schedule. In the
shared 200 Hz scene, the M1 runtime schedule is Go2 25 Hz, drone 200 Hz, and
Panda 50 Hz; the G1 stands at 50 Hz. See `docs/M1_REPORT.md` and
`core/timing.py` for runtime rates and physics-tick periods.

| Asset | Robot / capability | obs → act | Rate | md5 | Status |
|---|---|---|---|---|---|
| `logs/go2-nav/model_800.pt` | Go2 locomotion (velocity tracking) | 45 → 12 | 50 Hz | `7100a54f3313` | ✅ mission-verified |
| `logs/go2-push-s5/model_800.pt` | Go2 crate push (7 kg, 15 cm) | 45 → 12 | 20 Hz | `c56f741011b2` | ✅ 31/32 + 30/32 first-life, 0 true falls |
| `logs/drone-hover-s4/model_1600.pt` | Crazyflie hover / 3-D position | pos-err → 4 motors | 100 Hz | `d03cf15fa1fa` | ✅ recon 2/2 per mission |
| `logs/grasp-panda_rl/model_1500.pt` | Panda reach | reach obs → 7 joint targets + gripper | 100 Hz | `0e55df0cadbe` | ✅ picked = 1 |
| `logs/panda-push-s1/model_900.pt` | Panda reach + press + push (P1b) | push obs → arm targets | 100 Hz | `1cb4a6752e2d` | ✅ in-mission |
| `logs/go2-push-s5/model_1000.pt` | *unevaluated* | 45 → 12 | — | `b15c166c402b` | ⚠️ **do not use** (see F3) |

## 3. Stack B — Genesis humanoid (Genesis-Humanoid, engine pinned 0.3.13)

| Asset | Capability | Contract | md5 | Status |
|---|---|---|---|---|
| `Genesis-Humanoid/logs/g1-stand-s2/20261111_213159/checkpoints/checkpoint_final.pt` | G1 standing balance | **obs 103 → MLP(256,256,128) RELU → act 29** (`log_std(29)`, `mu.bias(29)`) | `79160cfe4725` | ✅ 4/4 stood the full 10 s, height 0.797 → 0.792 m |
| `Genesis-Humanoid/assets/robot/unitree_g1/g1_29dof_simple_collision.urdf` | G1 29-DOF body | URDF, 32 KB | — | on disk, loadable |

Training context recovered from the checkpoint's own configs: PPO, `clip 0.2`, `gamma 0.99`, `gae 0.95`, lr adaptive 1e-3, `dt 0.005`, FlatScene, command clips 0.2.

**Port risk (must be measured, not assumed):** the engine diff is 0.3.13 → 1.4.5. Contact solver, PD defaults and joint ordering can shift the gait. Plan: load the URDF in 1.4.5, rebuild the 103-D obs exactly, run a stand test, compare height/termination against the 0.797 m baseline. If it fails, the fallback is a short PPO fine-tune in 1.4.5 (not a from-scratch training).

## 4. Stack C — MuJoCo humanoid LfD (`~/Desktop/humaniod/humanoid_challenge`, MuJoCo 3.17)

A separate, self-contained project ("Humanoid-LfD — Teaching a Unitree G1 with your own hands", MIT, 12 GB incl. data/venv):

| Asset | Capability | Contract |
|---|---|---|
| `models/cup_bc_dagger1.pt` | Cup pick-place, BC + DAgger (15,110 transitions) | **obs 37 → act 8** |
| `models/cup_bc_5/25/100.pt` | BC ablation ladder | obs 37 → act 8 |
| `models/public_baseline_chunked.pt` | Chunked policy (source `lewislf/G1_Dex3_pick_white_cup_right8_1squeeze`, MIT) | **obs 56 → act 128** (16-step horizon × 8, exec 8, context 4) |
| `models/hand_mapper*.pt` (×4) | Phone-video → finger-joint retargeting | video features → hand joints |
| `vendor/mujoco_menagerie/` | G1 MJCF descriptions | MuJoCo XML |

**Incompatible with Genesis as-is:** different simulator, different observation semantics (hand-centric), different action space, and hand-only DOF. Reusing it inside the ecosystem would mean porting the env, not the weights. **Recommendation: keep Stack C as its own track** (it is a complete, published study), and use Stack B for the G1's role in the ecosystem.

## 5. Robot descriptions available

| Robot | File(s) | Notes |
|---|---|---|
| Go2, Crazyflie cf2x, Panda, plane | `genesis-world/genesis/assets/urdf/` | the exact assets the mission loads |
| G1 (29-DOF) | `Genesis-Humanoid/assets/robot/unitree_g1/g1_29dof_simple_collision.urdf` | matches the standing policy |
| G1 variants | `unitree_ros/robots/g1_description/`, `g1_d_description/`, `g1_with_brainco_hand/` | 29-DOF ± Dex hand variants |
| G1 MJCF | `humaniod/humanoid_challenge/vendor/mujoco_menagerie/` | MuJoCo only |

## 6. Infrastructure already built (do not rebuild)

| Component | File | Reuse in ecosystem |
|---|---|---|
| Machine guard (45 min sessions, 92 °C / 88 % memory pauses) | `team_lab/monitor/machine_guard.py` | as-is, mandatory |
| Message bus + auditable team log | `team_lab/team/blackboard.py` | as-is; becomes the multi-agent coordination substrate |
| Planners (rule + local LLM, capability validation) | `team_lab/team/planner.py` | extend to fleet-level |
| Mission loop, task board, delivery/recovery state machines | `team_lab/team/mission.py`, `agents.py` | pattern to generalise |
| Shared-scene harness with per-robot control rates | `team_lab/experiments/colo_env.py` | the base for multi-instance spawning |
| Mission recorder (real frames → mp4, timestamped overlay) | `team_lab/experiments/record_mission.py` | reuse for every milestone video |
| Policy loader (raw MLP from actor state dict) | `colo_env.load_actor_from_log` | loads Stack A **and** Stack B |

## 7. Gaps (nothing here is assumed to work)

1. **No G1 in the Genesis scene.** Only a standing policy in another engine version. *M1/M2 task.*
2. **No multi-instance spawning.** `colo_env` builds exactly one of each robot; robot identity, per-instance state and zone assignment do not exist. *M1 task.*
3. **No collision avoidance between robots.** Robots currently share physics but ignore each other. *M2/M3 task.*
4. **No AMR robot at all** (plan item E) — no model, no policy, no description. It is the largest genuine gap in the fleet plan.
5. **ROS 2 absent** (F4).
6. **Disk 93% full** (F5) — blocks multi-environment work.
7. **Capability-aware allocation is unproven at fleet scale.** The 1.5B commander already mis-attributes robots at 3 robots; at 8+ the guard matters more, not less.

## 8. Decisions I need from you (critical, per the execution rules)

1. **Disk (blocking):** approve pruning the 66 GB of unpushed checkpoints. Proposed keep-list: the 5 Stack-A policies + their `cfgs.pkl`, Stack-B G1, and the `s7/model_577` reference. Proposed delete: `go2-push-m0`, `m0b`, `r1`, `r2`, `r3`, `s1`–`s4`, `s6`, `s7` (whole dirs). Expected recovery ≈ 55 GB.
2. **G1 scope:** confirm G1's ecosystem role is **standing + station operations** (Stack B), with the MuJoCo hand-manipulation study kept as its own track rather than ported.
3. **AMR:** include a mobile base in M3, or defer it to a later phase? It requires a new model and a new policy — the only item that would force training.

## 9. Proposed M1 (next, no training)

1. Create `ecosystem/` as a **new** workspace; leave `team_lab` untouched and passing.
2. `RobotSpec` registry + `spawn_robot(kind, id, pose)` that builds N instances of any type with unique IDs and independent controller state.
3. Port the G1: load `g1_29dof_simple_collision.urdf` in 1.4.5, rebuild the 103-D observation, run the standing test, compare against 0.797 m.
4. Build the 4-robot scene (drone + Go2 + Panda + G1) with a shared clock and per-robot rates; assert no controller desync (the mission's existing timing invariant).
5. Regression: the existing 2-package mission must still deliver 2/2, unchanged.
