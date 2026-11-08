# MANUAL — How to Run, Test, and Try Everything Yourself

Every command below was actually used in this project. Copy-paste ready.
All commands run from a terminal; each section states its working directory.

---

## 0. One-time setup (already done — for reference)

```bash
# conda env with the Genesis engine + RL library (exists: "genesis")
source /Users/saichaitu/miniconda3/etc/profile.d/conda.sh
conda create -n genesis python=3.11 -y
conda activate genesis
pip install torch rsl-rl-lib tensorboard
pip install -e /Users/saichaitu/Desktop/Genesis/genesis-world

# Genesis-Humanoid's own pinned venv (for the G1 only; exists)
cd /Users/saichaitu/Desktop/Genesis/Genesis-Humanoid && uv sync --package gs-env
```

## 1. Per-session setup (do this first in every new terminal)

```bash
source /Users/saichaitu/miniconda3/etc/profile.d/conda.sh
conda activate genesis
cd /Users/saichaitu/Desktop/Genesis/team_lab/experiments
```

## 2. Machine watchdog (run in a SEPARATE terminal before long jobs)

```bash
cd /Users/saichaitu/Desktop/Genesis/team_lab/experiments
python ../monitor/machine_guard.py --watch --interval 5 --csv ../logs/guard-live.csv
# one-shot check instead:
python ../monitor/machine_guard.py --check
```
Reads macOS thermal state (`pmset -g therm`); CPU_Speed_Limit < 100 = throttling.
All our trainers also have this built in — you never *need* this, but it's nice to watch.

## 3. Smoke test (is Genesis alive?)

```bash
python smoke_test.py cpu      # ~30 s; loads Go2, steps physics, prints PASS
python smoke_test.py metal    # GPU variant (slower for 1 env — CPU is our pick)
```

## 4. Go2 — walking

```bash
# TRAIN (11 min at 512 envs; ~10M env-steps; checkpoints every 25 iters)
python go2_train_guarded.py --exp-name go2-walking --fresh --backend cpu \
    --num-envs 512 --max-iterations 600 --session-minutes 45

# EVALUATE (20 episodes, headless metrics)
python evaluate_go2.py --exp-name go2-walking
```
Expected: `20/20 reach 20s timeout | vx error ~0.02 m/s | VERDICT: EXCELLENT`.
Checkpoints: `../logs/go2-walking/model_*.pt` · training curve: `tensorboard --logdir ../logs/go2-walking`

## 5. Go2 — full navigation (walk/stop/turn/strafe)

```bash
python go2_train_guarded.py --exp-name go2-nav --fresh --backend cpu \
    --num-envs 512 --max-iterations 800 --session-minutes 40 \
    --cmd-x="-0.5,1.0" --cmd-y="-0.5,0.5" --cmd-w="-1.0,1.0"
# NOTE the "=" syntax — argparse eats values starting with "-"

python evaluate_go2.py --exp-name go2-nav   # expect GOOD, |vx-cmd| ~0.09
```

## 6. cf2x drone — hover

```bash
python drone_train_guarded.py --exp-name drone-hover --fresh --backend cpu \
    --num-envs 512 --max-iterations 400 --session-minutes 45
python drone_train_guarded.py --exp-name drone-hover --resume --backend cpu \
    --num-envs 512 --max-iterations 2900 --session-minutes 44   # continue training

python evaluate_drone.py --exp-name drone-hover
```
Known behavior: stable hover, coarse target-tracking (transits dominate the mean;
converged error ~0.34 m). Checkpoints: `../logs/drone-hover/`.

## 7. Panda — precision reach

```bash
# TRAIN the reach policy (upstream task: keypoint alignment, box FIXED)
python grasp_train_guarded.py --exp-name grasp-panda --num-envs 512 \
    --max-iterations 1500 --session-minutes 40 --backend cpu

# EVALUATE (the honest metric: finger-to-box keypoint distance)
python evaluate_panda.py --exp-name grasp-panda
```
Expected: median ~0.084 m, 5/8 precise reaches, VERDICT GOOD.

## 8. Panda — manipulation-aware pushing (Priority 1b)

```bash
# TRAIN (free box! teaches contact + pushing; warm-started from reach)
python panda_push_train.py --exp-name panda-push-s1 --fresh --num-envs 512 \
    --max-iterations 900 --session-minutes 42

# EVALUATE (fresh-seed episodes: box displacement + verified contacts)
#   — the evaluation script is being finalized as part of P1b's deliverable
```
Checkpoints: `../logs/panda-push-s1/model_*.pt`. The reach checkpoint it
warm-started from is preserved untouched at `../logs/grasp-panda_rl/`.

## 9. G1 — standing (Genesis-Humanoid's own pinned venv)

```bash
cd /Users/saichaitu/Desktop/Genesis/Genesis-Humanoid
GH=$(pwd)
# train (stand curriculum: zero commands, learn balance first)
timeout 2700 nice -n 10 $GH/.venv/bin/python \
    /Users/saichaitu/Desktop/Genesis/team_lab/experiments/g1_train_guarded.py \
    --exp-name g1-stand-s2 --num-envs 256 --total-iterations 800 \
    --session-minutes 40 --stand
# evaluate standing (10 s, 4 envs — the inline eval from the sessions; falls
# expected after ~2-5 s at current training level)
```
IMPORTANT: run from the repo root; the script self-applies the headless GL patch
(`g1_headless_patch.py`) — no display needed.

## 10. Three robots in ONE scene (co-location)

```bash
cd /Users/saichaitu/Desktop/Genesis/team_lab/experiments
python test_colo.py    # Go2 walks + drone hovers + Panda holds, 600 steps, PASS/FAIL
```

## 11. THE FULL MISSION (the flagship demo)

```bash
cd /Users/saichaitu/Desktop/Genesis/team_lab/experiments
python ../team/mission.py --packages 2 --planner rule --max-steps 60000
```
Watch it work: the console prints the timeline; the full "team chat" transcript
and metrics land in `../logs/mission_<timestamp>/`:
- `transcript.txt` — every message the robots and planner exchanged
- `metrics.txt` — recon/delivered/inspected/picked counters + makespan

Options: `--packages 2` · `--planner rule` (an ollama model name instead of
"rule" activates the LLM commander — e.g. `--planner qwen2.5:7b`)

## 12. Where everything lives

| What | Where |
|---|---|
| All checkpoints | `team_lab/logs/<experiment>/model_*.pt` |
| Training configs (reproducibility) | `team_lab/logs/<experiment>/cfgs.pkl` |
| TensorBoard curves | `tensorboard --logdir team_lab/logs/<experiment>` |
| Mission transcripts + metrics | `team_lab/logs/mission_<timestamp>/` |
| Guard health CSVs | `team_lab/logs/<experiment>/guard.csv` |
| All trainers/evaluators | `team_lab/experiments/*.py` |
| Team layer (blackboard/planner/agents/mission) | `team_lab/team/*.py` |
| Evidence log (the project's history) | `INVESTIGATION_LOG.md` |
| Mission plan + safety rules | `team_lab/MISSION_PLAN.md` |

## 13. Troubleshooting quick reference

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: gs_env` (G1) | run from `Genesis-Humanoid/` root; the uv .pth is broken — scripts set PYTHONPATH themselves |
| Checkpoints all named model_0.pt | old sessions only; new trainers number correctly |
| Mission says `delivered 0/2` but no push_phase | check `transcript.txt` for `manip_*`/`push_*` events; geometry is layout-sensitive (see INVESTIGATION_LOG) |
| Everything is slow | check the guard CSV; close other heavy apps; 512 envs needs ~4 cores |
| A run crashed midway | all trainers checkpoint (save_interval) — relaunch with `--resume` (drone) or a fresh short run |

## 14. Safety rules (from MISSION_PLAN.md — non-negotiable)

1. One training job at a time.
2. Always through the guarded trainers (thermal pauses + session caps + checkpoints).
3. Never install system-level anything; conda/uv envs only.
4. Check `guard.csv` after long runs (throttle events = pause reasons).
