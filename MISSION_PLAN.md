# Mission Plan — From Single-Robot RL to a Coordinated Multi-Robot Team

**Machine contract:** Apple M4, 16 GB RAM. macOS throttles before hardware can be harmed —
our job is to *never reach sustained throttle*: every training session runs through
`team_lab/monitor/machine_guard.py` (pauses on thermal/load, hard session caps, frequent
checkpoints, `nice`-level scheduling). Sessions end cleanly and resume from checkpoints.

## The ladder (each stage ends with something that works)

| Stage | Goal | Robot(s) | Task(s) | Success criterion |
|---|---|---|---|---|
| 0 ✅ | Safe env + watchdog + smoke test | — | — | `import genesis` works; Go2 loads & steps |
| 1 | First trained skill | **Go2** | locomotion (`go2_train.py`, PPO) | walks forward, tracked command velocity, eval video |
| 2 | Aerial skill | **cf2x drone** | hover (`hover_train.py`, PPO) | hovers at target altitude, small position error |
| 3 | Manipulation skill | **Panda** | grasp (`grasp_train.py`, PPO + teacher) | picks cube, non-trivial success rate |
| 4 | Humanoid skill (hardest — expectations set: slow on this Mac) | **G1** | stand → walk progression (`Genesis-Humanoid`, own uv env, pinned Genesis 0.3.13) | stands stably first; walking stretch goal |
| 5 | Co-location | Go2 + drone + Panda | one shared scene; scripted coordination | all three simulated simultaneously, stable, no interference |
| 6 | Teamwork layer | all | (a) blackboard + task allocator; (b) LLM mission planner (local ollama or API — user to choose); (c) optional learned comms (MARL/CTDE) | robots receive missions, divide tasks, report status |
| 7 | Integrated mission | all | e.g. disaster-scenario: drone recon → Go2 fetch/push → Panda sort | scored metrics: task completion, interference, latency |

## What "robots talking, discussing, planning" means here (honest engineering)

Three cooperating layers, from literal to learned:
1. **Blackboard/DDS-style message bus** — every robot publishes state (position, battery-equivalent, current task, capabilities) and task events. This is the nervous system.
2. **Planner/Commander (LLM)** — reads the blackboard, *converses* (we log its reasoning: "Box A is heavy → Go2 is closer → assign Go2"), assigns tasks, rebalances on failure. Runs between control ticks (LLM latency ≫ 20 ms control loop).
3. **Learned coordination (MARL, optional stretch)** — shared-reward multi-agent training where each agent's observations include teammate state + a learned message vector (CTDE). This is the "policies that cooperate" part; emergent language is research-grade, we do structured messaging first.

## Ground rules for laptop safety (non-negotiable)

- All training launched **through** `Guard` (pauses on throttle/load, 45-min default session caps, checkpoint-resume).
- num_envs starts conservative (128–512); scale only if guard stays green.
- Long runs: background task + guard `--watch` CSV; we review health before extending.
- Disk check before checkpoints-heavy runs; logs rotate in `team_lab/logs/`.
- Nothing system-level is ever installed; everything stays inside conda/uv envs.

## Runtime environments

| env | engine | used by |
|---|---|---|
| conda `genesis` (py3.11) | genesis-world **1.4.5** (editable, our clone) | Stages 0–3, 5–7 (Go2, drone, Panda, team) |
| uv `.venv` inside `Genesis-Humanoid/` | genesis-world **0.3.13** (pinned by GH) | Stage 4 (G1) only |

## Session protocol

Every work session: (1) guard check → (2) resume/launch one stage → (3) review guard CSV +
metrics → (4) save checkpoints + log → (5) update INVESTIGATION_LOG.md. No stage advances
until the previous one's success criterion is met or we consciously re-scope it.
