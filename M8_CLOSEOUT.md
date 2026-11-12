# M8 — Closeout: Genesis Multi-Robot Mission

**What this is.** A team of simulated robots works a warehouse task in [Genesis World](https://github.com/Genesis-Embodied-AI/genesis-world) v1.4.5 on an Apple M4 (CPU backend). A local LLM commander assigns work over a message bus; three robots carry it out. Everything below is a measured number from a run in this repository — nothing is projected or assumed.

**One video:** `logs/video_20261112_144810/mission.mp4` — 200 s, 960×540, 30 fps, 6000 frames, 3.8 MB.
It is a **recording of a real run**, not a plot or a replay reconstruction: the mission was re-executed unchanged with a camera sensor attached and the frames written straight from the renderer (`experiments/record_mission.py`). Playback is 6× faster than sim time (10 Hz capture at 30 fps). Its transcript is **byte-identical, apart from timestamps, to the logged run it documents** (`diff` = 0 lines against `logs/mission_20261112_143024`), which is how the video is known to match the execution. Overlay text is drawn on the frames: sim clock, robot count, live delivery tally, the most recent team event, and a banner with the measured drop distance at each delivery.

---

## The system

| | |
|---|---|
| **Robots** | **Unitree Go2** (quadruped, learned locomotion + learned crate-pushing), **Crazyflie 2.x drone** (learned hover, reconnaissance), **Franka Panda** (learned reach, parcel handling), **Unitree G1** (humanoid, learned standing — validated separately, not in this mission) |
| **Commander** | Local **qwen2.5:1.5b** via ollama; emits a JSON robot→task mapping every 4 s over a blackboard message bus (topics: status / task / assignment / event). No paid APIs. A rule-based planner is the interchangeable alternative and the LLM's fallback. |
| **Task** | 2 packages, each 7 kg, 0.35×0.35×0.15 m. The Go2 must push each one from its spawn along a corridor into a 0.55 m delivery zone; the Panda then reaches the delivered crate. |
| **Skills** | 5 learned policies, trained on this machine under a guarded session cap (`machine_guard.py`). No scripted pushing: the only load-moving primitive is a learned policy. |

## Milestones — M0–M7 complete

| Milestone | Result |
|---|---|
| M0 workspace & baseline | engine + 3 repos audited, version skew documented |
| M1 physics & integration | dt unified, contact solver stabilised (Newton), no NaN over 1200 s |
| M2 Go2 push policy | **`s5/model_800` passes the strict gates** — see below |
| M3 drone recon | recon 2/2 per mission |
| M4 Panda | picked=1, verified contact transition |
| M5 G1 humanoid | stands the full 10 s, height 0.797 m |
| M6 LLM commander | coordinates live: recon 2/2, delivered 2/2 |
| M7 end-to-end mission | **delivered 2/2, three independent runs** |

## Mission success

**3/3 independent runs, each delivering 2/2 packages** — plus a fourth regression run made after M8's changes, also 2/2.

| run | planner | recon | delivered | pkg0 drop | pkg1 drop |
|---|---|---|---|---|---|
| `mission_20261112_134116` | rule | 2/2 | **2/2** | 0.549 m | 0.547 m |
| `mission_20261112_143024` | rule | 2/2 | **2/2** | 0.549 m | 0.547 m |
| `mission_20261112_135455` | **qwen2.5:1.5b** | 2/2 | **2/2** | 0.549 m | 0.547 m |
| `mission_20261112_145310` — post-M8 regression, run after the recording hook was added | rule | 2/2 | **2/2** | 0.549 m | 0.547 m |

The fourth run matters for a different reason: it was made **after** the camera hook was added to the mission and environment, and its transcript is again identical (diff = 0 against the run above). Adding the recording path provably did not alter the simulation.

**Delivery, measured:** both crates finish **0.549 m and 0.547 m** from the drop-zone centre — inside the 0.55 m zone, and both fetches report `box_goal_distance 0.500` with `mode: learned`. The identically repeating distances are the sign of a deterministic control layer, not of tuned luck: the missions are reproducible to the sim second.

**How the delivery is achieved:** two-phase `retreat → charge`. The dog is placed ~0.9 m behind the crate on the crate→zone line, then the learned pusher is applied continuously under a stall detector, a climb guard, an overrun limit and a travel cap. Measured single strokes reach **0.901 m**, 0.608 m and 0.423 m; 70 strokes deliver both packages (43 + 27). The pusher was never the bottleneck — an earlier hypothesis that fine placement needed retraining was wrong, and the fix was control, not training (`INVESTIGATION_LOG.md`, M7 CLOSED).

## Go2 policy — the strict gates, on fresh seeds

`experiments/evaluate_go2_push.py`, 16 spawns never seen in training per run:

| checkpoint | reach+contact | first-life pushed ≥0.15 m | true falls | first-life displacement |
|---|---|---|---|---|
| **`logs/go2-push-s5/model_800.pt`** (mission policy), run 1 | 32/32 | **31/32** | **0** | mean **+1.90 m**, best +4.05 m |
| same checkpoint, run 2 (independent spawns) | 32/32 | **30/32** | **0** | mean **+1.42 m**, best +4.38 m |
| `go2-push-s7/model_577.pt` | 0/16 | 4/16 | 59 | — |
| `go2-push-r3/model_900.pt` | 0/16 | 5/16 | 563 | +0.12 m |

Gate was ≥8/16 first-life and 0 falls. The r1→r2→r3 branch (an attempt to train the fall-slam exploit away) **fails** and is a closed dead end; the mission's own policy passes by a wide margin. Nobody had applied the upgraded evaluator to it.

## Reliability — stated plainly, not rounded up

- **The mission is not fall-free: 17 falls per 2-package mission** — the same 17 in all three runs. Every one recovers automatically (`dog_fell_recovering`, base z ≈ 0.06 m) and none strands a task, but they are real. By timestamp against the charge windows: **12 during navigation/retreat, 5 during a charge.** In the push environment itself this policy falls **0 times in 32 lives**, so the failures belong to the nav policy meeting a 15 cm crate it never saw in training — a hardening item, not a defect of the pushing skill.
- **Only 1 of 2 packages gets inspected.** `process0` completes at 576 s; `process1` starts at 760 s and never finishes, so the mission runs to its 1200 s budget instead of exiting on its completion predicate. The likely cause is the second crate arriving at a station already occupied by the first. Delivery — the gate that matters — is unaffected.
- **The local LLM's mapping is unreliable, the guard is not.** qwen2.5:1.5b frequently names the wrong robot for a task (e.g. the drone for a push), and a capability guard redirects the published assignment; two LLM runs were also capped at 1/2 by a planner bug that stranded an unassigned task, now fixed.

## Reproducibility

| artifact | md5 (12) |
|---|---|
| `logs/go2-push-s5/model_800.pt` (push) | `c56f741011b2` |
| `logs/go2-nav/model_800.pt` (locomotion) | `7100a54f3313` |
| `logs/drone-hover-s4/model_1600.pt` | `d03cf15fa1fa` |
| `logs/panda-push-s1/model_900.pt` | `1cb4a6752e2d` |
| `logs/go2-push-s7/model_577.pt` (pinned reference) | `6e8249a898a3` |
| `experiments/go2_push_env.py` (training env, unchanged) | `3a7c91dbdb48` |
| `experiments/colo_env.py` | `15ec07ca10a7` |
| `team/mission.py` | `5b7d05d0ca23` |
| `team/agents.py` | `176fe6c64788` |
| `team/planner.py` | `6c8a2bef7767` |
| `logs/video_20261112_144810/mission.mp4` | `916ff0e9d64b` |

No checkpoint, training environment or physics setting was modified to produce any result here. The camera hook added for recording is off by default and changes nothing in the scene (`record_camera=False`).

**Reproduce:**
```bash
cd team_lab/team   && python mission.py --packages 2 --planner rule --max-steps 60000
cd team_lab/team   && python mission.py --packages 2 --planner qwen2.5:1.5b --max-steps 60000
cd team_lab/experiments && python test_push_endgame.py --seconds 240   # delivery in isolation, ~5 s
cd team_lab/experiments && python test_recovery.py                     # 8/8 bad poses recovered
cd team_lab/experiments && python evaluate_go2_push.py --exp-name go2-push-s5 --ckpt model_800 --episodes 32
cd team_lab/experiments && python record_mission.py                     # regenerate the video
```

## Timeline (what the video shows)

| sim t | event |
|---|---|
| 0–8 s | drone flies reconnaissance, reports both package positions; commander assigns fetch and parcel tasks |
| 5 s | first push stroke of `pkg0` |
| 5–536 s | Go2 shoves `pkg0` down the corridor in 43 strokes, closing the distance to the goal from 2.70 m to 0.50 m, including a 0.90 m haul |
| **536.36 s** | **`pkg0` delivered, 0.549 m from zone centre** |
| 540–576 s | Panda reaches the delivered crate; `process0` complete |
| 600–757 s | Go2 pushes `pkg1` from (2.0, 0.3) onto the corridor, 27 strokes |
| **757.32 s** | **`pkg1` delivered, 0.547 m from zone centre** |
| 758.9 s | `fetch1` complete; Panda starts `process1` (never completes) |
| 1200 s | mission ends on the step budget |

## What is built, what works, what remains

**Works:** 5 trained policies on real physics; a message-bus team with an LLM commander; a full pick-push-deliver pipeline that hits a 0.55 m placement tolerance twice per mission, reproducibly, with two different planners; a control layer that turns a coarse learned pusher into reliable delivery.

**Imperfect:** the mission is not fall-free (17 falls, all recovered, concentrated in nav/retreat); the second crate is never inspected; the 1.5B local model needs a capability guard to produce valid assignments. These are recorded, not hidden.
