#!/usr/bin/env python3
"""
go2_train_guarded.py — Stage 1: Unitree Go2 locomotion training, laptop-safe edition.

Adapted from genesis-world/examples/locomotion/go2_train.py (upstream, MIT-licensed).
Differences from upstream, all deliberate:
  * backend selectable via --backend (cpu | gpu); upstream hardcodes gs.gpu (4096-env GPU job)
  * laptop-scale num_envs (default 256; upstream 4096)
  * machine_guard integration: pauses between iterations on thermal/load stress,
    hard session wall-clock cap, health CSV written next to logs
  * runner.learn() is called ONE iteration at a time so the guard runs between iterations
    (functionally equivalent: each call collects num_steps_per_env fresh transitions
    and does one PPO update; adaptive-LR and iteration counter persist on the runner)
  * does NOT wipe the log dir on start (upstream rmtree's it) — sessions append safely;
    pass --fresh to wipe
  * save_interval lowered (25) for more frequent checkpoints

Run from anywhere:
  python go2_train_guarded.py --num-envs 256 --max-iterations 60 --session-minutes 45
"""

import argparse
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

REPO = Path("/Users/saichaitu/Desktop/Genesis/genesis-world")
sys.path.insert(0, str(REPO / "examples" / "locomotion"))  # Go2Env + upstream cfgs
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "monitor"))  # machine_guard

import genesis as gs
from go2_env import Go2Env
from go2_train import get_cfgs, get_train_cfg
from machine_guard import Guard
from rsl_rl.runners import OnPolicyRunner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp-name", default="go2-walking")
    p.add_argument("-b", "--num-envs", type=int, default=256)
    p.add_argument("--max-iterations", type=int, default=60)
    p.add_argument("--session-minutes", type=float, default=45.0)
    p.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--cmd-x", default=None, help="override lin_vel_x_range, e.g. '-0.5,1.0'")
    p.add_argument("--cmd-y", default=None, help="override lin_vel_y_range, e.g. '-0.5,0.5'")
    p.add_argument("--cmd-w", default=None, help="override ang_vel_range, e.g. '-1.0,1.0'")
    p.add_argument("--fresh", action="store_true", help="wipe existing log dir")
    args = p.parse_args()

    log_dir = Path(__file__).resolve().parent.parent / "logs" / args.exp_name
    if args.fresh and log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[stage1] log_dir={log_dir}")

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    for key, arg in (("lin_vel_x_range", args.cmd_x), ("lin_vel_y_range", args.cmd_y), ("ang_vel_range", args.cmd_w)):
        if arg is not None:
            lo, hi = (float(v) for v in arg.split(","))
            command_cfg[key] = [lo, hi]
            print(f"[stage1] command_cfg[{key}] overridden to [{lo}, {hi}]")
    train_cfg = get_train_cfg(args.exp_name)
    train_cfg["save_interval"] = 25  # checkpoint every 25 iters (upstream: 100)

    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    print(f"[stage1] init backend={args.backend} num_envs={args.num_envs}")
    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.gpu, precision="32",
            logging_level="warning", performance_mode=True)

    env = Go2Env(num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
                 reward_cfg=reward_cfg, command_cfg=command_cfg)

    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)

    guard = Guard(session_minutes=args.session_minutes, log_csv=str(log_dir / "guard.csv"))
    t_start = time.time()
    done = 0
    try:
        for it in range(1, args.max_iterations + 1):
            guard.wait_until_safe()
            if guard.session_expired():
                print(f"[stage1] session cap reached after {done} iterations — checkpointed, exiting cleanly")
                break
            # rsl_rl 5.x derives iteration numbers from current_learning_iteration;
            # advance it manually so per-iteration learn() calls produce unique,
            # correctly-numbered checkpoints (model_<true_iter>.pt)
            runner.current_learning_iteration = it
            t0 = time.time()
            runner.learn(num_learning_iterations=1, init_at_random_ep_len=(it == 1))
            dt = time.time() - t0
            done = it
            elapsed = time.time() - t_start
            eta = elapsed / it * (args.max_iterations - it)
            print(f"[stage1] iter {it:4d}/{args.max_iterations}  {dt:6.1f}s/iter  "
                  f"elapsed {elapsed/60:5.1f}min  eta {eta/60:5.1f}min", flush=True)
    finally:
        guard.close()
        print(f"[stage1] DONE {done} iterations. Logs+checkpoints: {log_dir}")


if __name__ == "__main__":
    main()
