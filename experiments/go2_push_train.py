#!/usr/bin/env python3
"""
go2_push_train.py — Stage 6c: train the Go2 to PUSH a crate to a goal zone (PPO).

Learned skill replacing the failed hand-crafted push controller. Adapted from
go2_train_guarded.py with the push env/configs. Guarded like all our trainers.
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

REPO = Path("/Users/saichaitu/Desktop/Genesis/genesis-world")
sys.path.insert(0, str(REPO / "examples" / "locomotion"))  # reuse rsl helpers? not needed
sys.path.insert(0, str(Path(__file__).resolve().parent))    # go2_push_env
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "monitor"))

import genesis as gs
import torch
from go2_push_env import Go2PushEnv
from machine_guard import Guard
from rsl_rl.runners import OnPolicyRunner


def get_cfgs(fall_penalty=4.0):
    env_cfg = {
        "num_actions": 12,
        "default_joint_angles": {
            "FL_hip_joint": 0.0, "FR_hip_joint": 0.0, "RL_hip_joint": 0.0, "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8, "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5, "FR_calf_joint": -1.5, "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        "kp": 20.0,
        "kd": 0.5,
        "termination_if_roll_greater_than": 10,
        "termination_if_pitch_greater_than": 10,
        "base_init_pos": [2.2, 0.0, 0.42],
        "base_init_quat": [0.0, 0.0, 0.0, 1.0],  # wxyz: yaw 180°, facing the crate (-x)
        "episode_length_s": 12.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        "fall_penalty": fall_penalty,
    }
    obs_cfg = {
        "obs_scales": {
            "ang_vel": 0.25,
            "lin_vel": 2.0,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "crate_pos": 1.0,
        },
    }
    reward_cfg = {
        "base_height_target": 0.3,
        "progress_scale": 40.0,     # per-meter of crate progress (dt-scaled)
        "approach_scale": 10.0,     # per-meter of dog->crate approach (shaping)
        "proximity_sigma": 0.5,
        "reward_scales": {
            "push_progress": 3.0,
            "approach": 3.0,
            "proximity_bonus": 1.0,
            "lin_vel_z": -1.0,
            "base_height": -20.0,
            "action_rate": -0.005,
            "similar_to_default": -0.05,
        },
    }
    push_cfg = {
        "crate_size": [0.35, 0.35, 0.15],
        "crate_mass": 7.0,
        # the dog spawns BEHIND the crate relative to the goal, facing it:
        # crate 1.0 m ahead of the dog, goal 2.2 m beyond the crate — so the
        # learned skill is the pure push (circling/approach stays the mission
        # layer's job, where navigation already works)
        "crate_spawn": [0.65, 0.0, 0.08],
        "goal_xy": [-1.2, 0.0],
        "goal_radius": 3.1,
    }
    return env_cfg, obs_cfg, reward_cfg, push_cfg


def get_train_cfg(exp_name):
    return {
        "algorithm": {
            "class_name": "PPO", "clip_param": 0.2, "desired_kl": 0.02,
            "entropy_coef": 0.015, "gamma": 0.99, "lam": 0.95,
            "learning_rate": 0.001, "max_grad_norm": 1.0,
            "num_learning_epochs": 5, "num_mini_batches": 4,
            "schedule": "adaptive", "use_clipped_value_loss": True, "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu",
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        },
        "critic": {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu"},
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": 24,
        "save_interval": 50,
        "run_name": exp_name,
        "logger": "tensorboard",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp-name", default="go2-push")
    p.add_argument("-b", "--num-envs", type=int, default=512)
    p.add_argument("--max-iterations", type=int, default=800)
    p.add_argument("--session-minutes", type=float, default=42.0)
    p.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--warm-start", action="store_true", default=True)
    p.add_argument("--no-warm-start", dest="warm_start", action="store_false")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="continue from the newest checkpoint in the log dir")
    p.add_argument("--fall-penalty", type=float, default=4.0,
                   help="non-timeout termination penalty (default 4.0 = the historical value; "
                        "12.0 = the r2 fall-slam fix: break-even slam displacement 3.3 cm -> 10 cm)")
    p.add_argument("--warm-start-ckpt", default=None,
                   help="explicit warm-start checkpoint (default: the original go2-push-s7/model_577.pt)")
    args = p.parse_args()

    log_dir = Path(__file__).resolve().parent.parent / "logs" / args.exp_name
    if args.fresh and log_dir.exists():
        import shutil
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[push] log_dir={log_dir}")

    env_cfg, obs_cfg, reward_cfg, push_cfg = get_cfgs(fall_penalty=args.fall_penalty)
    train_cfg = get_train_cfg(args.exp_name)
    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump((env_cfg, obs_cfg, reward_cfg, push_cfg, train_cfg), f)

    print(f"[push] init backend={args.backend} num_envs={args.num_envs}")
    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.gpu, precision="32",
            logging_level="warning", performance_mode=True)

    env = Go2PushEnv(num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
                     reward_cfg=reward_cfg, push_cfg=push_cfg)
    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)

    resume_it = 0
    if args.resume:
        ckpts = sorted(log_dir.glob("model_*.pt"), key=lambda f: int(f.stem.split("_")[1]))
        if not ckpts:
            raise FileNotFoundError(f"--resume but no model_*.pt in {log_dir}")
        latest = ckpts[-1]
        runner.load(str(latest))
        resume_it = int(latest.stem.split("_")[1])
        print(f"[push] resuming from {latest.name} (iteration {resume_it})")

    # -- warm start: transplant the trained WALKING policy into the push actor.
    # The push obs = walking obs (45) + 4 crate/goal dims, so the first linear
    # layer gets the walking weights for the first 45 input columns and zeros
    # for the new columns; all deeper layers copy verbatim. The dog starts as
    # a competent forward walker (commands fixed at 0.3 m/s) and PPO only has
    # to learn the push adjustments on top.
    # P3b: warm start from the CURRENT BEST PUSHER (model_577) — it pushes
    # 16/16 at the mission geometry (while falling); the fall penalty now
    # shapes the stable gait. Straight load: the dims match exactly (49->49).
    walk_ckpt = (Path(args.warm_start_ckpt) if args.warm_start_ckpt else
                 Path("/Users/saichaitu/Desktop/Genesis/team_lab/logs/go2-push-s7/model_577.pt"))
    if args.warm_start and resume_it == 0 and walk_ckpt.exists():
        walk_sd = torch.load(walk_ckpt, map_location=gs.device, weights_only=False)["actor_state_dict"]
        actor = runner.alg.actor if hasattr(runner.alg, "actor") else runner.alg._actor
        push_sd = actor.state_dict()
        new_sd = {}
        for k, v in push_sd.items():
            if k.endswith("weight") and v.shape[1] == 49 and walk_sd[k].shape[1] == 45:
                w = torch.zeros_like(v)
                w[:, :45] = walk_sd[k]
                new_sd[k] = w
                print(f"[push] warm-start {k}: copied 45/49 input columns")
            elif k in walk_sd and walk_sd[k].shape == v.shape:
                new_sd[k] = walk_sd[k].clone()
            else:
                new_sd[k] = v
        actor.load_state_dict(new_sd)
        # copy the std parameter too (walking policy's learned exploration scale)
        print("[push] warm start from walking policy applied")

    guard = Guard(session_minutes=args.session_minutes, log_csv=str(log_dir / "guard.csv"))
    t_start = time.time()
    done = resume_it
    try:
        for it in range(resume_it + 1, args.max_iterations + 1):
            guard.wait_until_safe()
            if guard.session_expired():
                print(f"[push] session cap reached after {done} iterations")
                break
            runner.current_learning_iteration = it
            t0 = time.time()
            runner.learn(num_learning_iterations=1, init_at_random_ep_len=(it == 1))
            dt = time.time() - t0
            done = it
            elapsed = time.time() - t_start
            eta = elapsed / it * (args.max_iterations - it)
            print(f"[push] iter {it:4d}/{args.max_iterations}  {dt:6.1f}s/iter  "
                  f"elapsed {elapsed/60:5.1f}min  eta {eta/60:5.1f}min", flush=True)
    finally:
        guard.close()
        print(f"[push] DONE {done} iterations. Logs: {log_dir}")


if __name__ == "__main__":
    main()
