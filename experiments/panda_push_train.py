#!/usr/bin/env python3
"""
panda_push_train.py — Priority 1b: manipulation-aware arm training (guarded).

Warm-starts from the trained reach policy (grasp-panda_rl/model_1500.pt — the
observation/action dimensions are IDENTICAL, straight state_dict load) and
trains controlled contact + pushing with a FREE box in the upstream GraspEnv
world (arm base at the origin — the training frame, so the world-frame box
positions in the observations stay in-distribution by construction).

Advisor conditions honored:
  * fresh experiment dir — no existing checkpoint is overwritten
  * guarded training (thermal pauses, session cap, NaN-checked)
  * headless, checkpoints + logs saved for independent evaluation
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

REPO = Path("/Users/saichaitu/Desktop/Genesis/genesis-world")
sys.path.insert(0, str(REPO / "examples" / "manipulation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))                       # panda_push_env
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "monitor"))    # machine_guard

import torch

import genesis as gs
from panda_push_env import PandaPushEnv
from grasp_train import get_task_cfgs
from machine_guard import Guard
from rsl_rl.runners import OnPolicyRunner


def get_push_train_cfg(exp_name):
    cfg = {
        "algorithm": {
            "class_name": "PPO", "clip_param": 0.2, "desired_kl": 0.02,
            "entropy_coef": 0.01, "gamma": 0.99, "lam": 0.95,
            "learning_rate": 3e-4, "max_grad_norm": 1.0,
            "num_learning_epochs": 5, "num_mini_batches": 4,
            "schedule": "adaptive", "use_clipped_value_loss": True, "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel", "hidden_dims": [256, 256, 128], "activation": "relu",
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.6,
                                 "std_type": "scalar"},
        },
        "critic": {"class_name": "MLPModel", "hidden_dims": [256, 256, 128], "activation": "relu"},
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": 24,
        "save_interval": 50,
        "run_name": exp_name,
        "logger": "tensorboard",
    }
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp-name", default="panda-push-s1")
    p.add_argument("-b", "--num-envs", type=int, default=512)
    p.add_argument("--max-iterations", type=int, default=900)
    p.add_argument("--session-minutes", type=float, default=42.0)
    p.add_argument("--fresh", action="store_true")
    args = p.parse_args()

    log_dir = Path(__file__).resolve().parent.parent / "logs" / args.exp_name
    if args.fresh and log_dir.exists():
        import shutil
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[p1b] log_dir={log_dir}")

    env_cfg, reward_scales, robot_cfg = get_task_cfgs()
    # reward set for pushing (the keypoint reach reward is replaced)
    reward_scales = {
        "push_progress": 3.0,   # per-meter of +x box displacement (dt-scaled)
        "contact": 0.5,         # while touching the box
    }
    env_cfg["num_envs"] = args.num_envs

    train_cfg = get_push_train_cfg(args.exp_name)
    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump((env_cfg, reward_scales, robot_cfg, train_cfg), f)

    print(f"[p1b] init backend=cpu num_envs={args.num_envs}")
    gs.init(backend=gs.cpu, precision="32", logging_level="warning", performance_mode=True)

    env = PandaPushEnv(env_cfg=env_cfg, reward_cfg=reward_scales, robot_cfg=robot_cfg,
                       show_viewer=False)
    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)

    # WARM START: straight load from the reach policy (identical dims)
    walk_ckpt = Path(__file__).resolve().parent.parent / "logs/grasp-panda_rl/model_1500.pt"
    sd = torch.load(walk_ckpt, map_location=gs.device, weights_only=False)["actor_state_dict"]
    actor = runner.alg.actor if hasattr(runner.alg, "actor") else runner.alg._actor
    actor.load_state_dict(sd)
    print(f"[p1b] warm start from {walk_ckpt.name} applied (straight load)")

    guard = Guard(session_minutes=args.session_minutes, log_csv=str(log_dir / "guard.csv"))
    t_start = time.time()
    done = 0
    try:
        for it in range(1, args.max_iterations + 1):
            guard.wait_until_safe()
            if guard.session_expired():
                print(f"[p1b] session cap reached after {done} iterations")
                break
            runner.current_learning_iteration = it
            t0 = time.time()
            runner.learn(num_learning_iterations=1, init_at_random_ep_len=(it == 1))
            dt = time.time() - t0
            done = it
            elapsed = time.time() - t_start
            eta = elapsed / it * (args.max_iterations - it)
            print(f"[p1b] iter {it:4d}/{args.max_iterations}  {dt:6.1f}s/iter  "
                  f"elapsed {elapsed/60:5.1f}min  eta {eta/60:5.1f}min", flush=True)
    finally:
        guard.close()
        print(f"[p1b] DONE {done} iterations. Logs: {log_dir}")


if __name__ == "__main__":
    main()
