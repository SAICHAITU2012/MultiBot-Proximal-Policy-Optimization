#!/usr/bin/env python3
"""
grasp_train_guarded.py — Stage 3: Franka Panda grasping (RL stage), laptop-safe edition.

Adapted from genesis-world/examples/manipulation/grasp_train.py (upstream, MIT-licensed).
Task: 6-dim end-effector delta actions tracked by damped-least-squares IK at 100 Hz;
reward = keypoint distance between gripper fingers and box; 3 s episodes.
Upstream two-stage design (privileged RL teacher -> vision BC student) preserved:
this script runs stage 'rl'; stage 'bc' can be added later using the saved teacher.

Same laptop-safety pattern as the other guarded trainers.
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

REPO = Path("/Users/saichaitu/Desktop/Genesis/genesis-world")
sys.path.insert(0, str(REPO / "examples" / "manipulation"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "monitor"))

import genesis as gs
from grasp_env import GraspEnv
from grasp_train import get_task_cfgs, get_train_cfg
from machine_guard import Guard
from rsl_rl.runners import OnPolicyRunner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp-name", default="grasp")
    p.add_argument("-b", "--num-envs", type=int, default=512)
    p.add_argument("--max-iterations", type=int, default=400)
    p.add_argument("--session-minutes", type=float, default=45.0)
    p.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    args = p.parse_args()

    log_dir = Path(__file__).resolve().parent.parent / "logs" / f"{args.exp_name}_rl"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[stage3] log_dir={log_dir}")

    env_cfg, reward_scales, robot_cfg = get_task_cfgs()
    rl_train_cfg, bc_train_cfg = get_train_cfg(args.exp_name)
    rl_train_cfg["save_interval"] = 50
    env_cfg["num_envs"] = args.num_envs  # rl stage scales with envs (bc stage stays at 10)

    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump((env_cfg, reward_scales, robot_cfg, rl_train_cfg, bc_train_cfg), f)

    print(f"[stage3] init backend={args.backend} num_envs={args.num_envs}")
    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.gpu, precision="32",
            logging_level="warning", performance_mode=True)

    env = GraspEnv(env_cfg=env_cfg, reward_cfg=reward_scales, robot_cfg=robot_cfg,
                   show_viewer=False)

    runner = OnPolicyRunner(env, rl_train_cfg, str(log_dir), device=gs.device)

    guard = Guard(session_minutes=args.session_minutes, log_csv=str(log_dir / "guard.csv"))
    t_start = time.time()
    done = 0
    try:
        for it in range(1, args.max_iterations + 1):
            guard.wait_until_safe()
            if guard.session_expired():
                print(f"[stage3] session cap reached after {done} iterations — checkpointed, exiting cleanly")
                break
            runner.current_learning_iteration = it  # unique checkpoint names (see go2 trainer note)
            t0 = time.time()
            runner.learn(num_learning_iterations=1, init_at_random_ep_len=(it == 1))
            dt = time.time() - t0
            done = it
            elapsed = time.time() - t_start
            eta = elapsed / it * (args.max_iterations - it)
            print(f"[stage3] iter {it:4d}/{args.max_iterations}  {dt:6.1f}s/iter  "
                  f"elapsed {elapsed/60:5.1f}min  eta {eta/60:5.1f}min", flush=True)
    finally:
        guard.close()
        print(f"[stage3] DONE {done} iterations. Logs+checkpoints: {log_dir}")


if __name__ == "__main__":
    main()
