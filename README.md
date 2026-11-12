# MultiBot-Proximal-Policy-Optimization

**A heterogeneous robot team that pushes 7 kg crates into a delivery zone, coordinated by a local LLM — learned skills (PPO), one shared physical scene, and every claim measured.**

Built on [Genesis World](https://github.com/Genesis-Embodied-AI/genesis-world) v1.4.5, trained and evaluated on a single Apple M4 (16 GB, CPU backend). No paid APIs; the commander is `qwen2.5:1.5b` running locally via ollama.

> **Result:** 2 of 2 packages delivered in **4 independent runs**, with the rule planner *and* the local LLM planner. Measured drop distances **0.549 m** and **0.547 m** against a 0.55 m zone. Deterministic to the sim second.

<p align="center"><em>logs/video_20261112_144810/mission.mp4 — a real recorded run, 200 s</em></p>

---

## The fleet

Three robots work the mission; a fourth is validated separately.

| Robot | Learned policy | Skill | Control |
|---|---|---|---|
| **Unitree Go2** | `go2-nav/model_800.pt` | velocity-tracking locomotion (walk / turn / strafe) | 50 Hz |
| **Unitree Go2** | `go2-push-s5/model_800.pt` | **push a 7 kg, 15 cm crate** — the only load-moving primitive in the system | 20 Hz |
| **Crazyflie 2.x** | `drone-hover-s4/model_1600.pt` | hold / step a 3-D position; reconnaissance | 100 Hz |
| **Franka Panda** | `grasp-panda_rl/model_1500.pt` + `panda-push-s1/model_900.pt` | reach, press, parcel handling | 100 Hz |
| **Unitree G1** | `g1-stand-s2/checkpoint_final.pt` | standing balance (10 s) | 50 Hz — *not in the mission* |

All policies are PPO-trained with `rsl_rl` under `monitor/machine_guard.py`, which caps each session at 45 min and pauses on CPU 92 °C / memory 88 %.

## How the team works

The policies are trained **individually** — one network, one robot, one task. The teamwork is a separate software layer:

```
planner.py    robot <-> task assignments every 4 s (rule-based, or local qwen2.5:1.5b)
blackboard.py message bus: status / task / assignment / event — also the auditable team log
agents.py     per-robot task state machines (recon waypoints, retreat->charge pushing, reach/press)
mission.py    task board, delivery detection, artifact logging
colo_env.py   one shared scene at 100 Hz; each robot steps its own policy at its own rate
```

The Go2 never *decides* to push. Its learned skill is "shove"; the mission layer decides **when, from where, for how long**, under a stall detector, a climb guard, an overrun limit and a travel cap.

## Measured results

| | |
|---|---|
| Mission (4 runs, both planners) | **delivered 2/2** each, drops 0.549 / 0.547 m |
| Push policy (fresh seeds, 32 envs x 2 runs) | **31/32** and **30/32** first-life pushes ≥ 0.15 m, **0 true falls**, mean +1.4–1.9 m per 12 s life |
| Best single push stroke | **0.901 m** of crate travel in one engagement |
| Drone recon | 2/2 packages located per mission |
| Panda | picked = 1, verified contact transition |
| G1 | stands the full 10 s, height 0.797 m |

## Known limitations (not hidden)

- **The mission is not fall-free:** 17 falls per run (12 during navigation/retreat, 5 during a charge). All recover automatically; none strands a task. The cause is the nav policy meeting a crate it never saw in training — 0 falls in 32 push-environment lives.
- **Only 1 of 2 crates is inspected** — the mission exits on its 1200 s step budget rather than its completion predicate, because the second crate arrives at a station already occupied by the first.
- The 1.5B local commander frequently names the wrong robot for a task; a capability guard redirects the published assignment.
- No hardware: nothing here has run on a physical robot.

## Reproduce

```bash
cd team/  && python mission.py --packages 2 --planner rule --max-steps 60000
cd team/  && python mission.py --packages 2 --planner qwen2.5:1.5b --max-steps 60000

cd experiments/ && python test_push_endgame.py --seconds 240        # delivery in isolation (~5 s)
cd experiments/ && python test_recovery.py                          # 8/8 bad poses recovered
cd experiments/ && python evaluate_go2_push.py --exp-name go2-push-s5 --ckpt model_800 --episodes 32
cd experiments/ && python record_mission.py                         # regenerate the video
```

## Documentation

- **[M8_CLOSEOUT.md](M8_CLOSEOUT.md)** — the one-page report: system, milestones, evidence, hashes, limitations
- **[INVESTIGATION_LOG.md](INVESTIGATION_LOG.md)** — the full research log, including every failed approach and why it failed
- **[MASTER_TRACKER.md](MASTER_TRACKER.md)** — milestone status and the defect list
- **[MISSION_PLAN.md](MISSION_PLAN.md)** — the original design

## Artifact integrity

The checkpoints published here are the ones every number above was measured with:

| file | md5 |
|---|---|
| `logs/go2-push-s5/model_800.pt` | `c56f741011b2` |
| `logs/go2-nav/model_800.pt` | `7100a54f3313` |
| `logs/drone-hover-s4/model_1600.pt` | `d03cf15fa1fa` |
| `logs/grasp-panda_rl/model_1500.pt` | — |
| `logs/panda-push-s1/model_900.pt` | `1cb4a6752e2d` |
| `logs/g1-stand-s2/checkpoint_final.pt` (G1, stands 10 s) | — |
| `logs/go2-push-s7/model_577.pt` (reference, fails gates) | `6e8249a898a3` |

Each checkpoint ships with its training config (`cfgs.pkl`) so it can be reloaded and re-evaluated.
