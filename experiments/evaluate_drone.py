#!/usr/bin/env python3
"""
evaluate_drone.py — headless evaluation of a trained cf2x hover policy.

Metrics over N parallel episodes (targets resampled every 3 s from a ±1 m box at z=1):
  * survival: episodes reaching the 15 s cap without crash/timeout-termination
  * position tracking: mean distance to the commanded 3-D target (steady state t > 3 s)
  * the 'at_target_threshold' from env cfg (0.1 m) as a success-rate-style metric

Usage:
  python evaluate_drone.py --exp-name drone-hover-s1 [--episodes 20]
"""

import pickle
import sys
from pathlib import Path

REPO = Path("/Users/saichaitu/Desktop/Genesis/genesis-world")
sys.path.insert(0, str(REPO / "examples" / "drone"))

import torch

import genesis as gs
from hover_env import HoverEnv
from rsl_rl.runners import OnPolicyRunner


def latest_ckpt(log_dir: Path) -> str:
    ckpts = sorted(log_dir.glob("model_*.pt"), key=lambda f: f.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"no model_*.pt in {log_dir}")
    return ckpts[-1].name


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp-name", default="drone-hover-s1")
    p.add_argument("--episodes", type=int, default=20)
    args = p.parse_args()

    log_dir = Path(__file__).resolve().parent.parent / "logs" / args.exp_name
    ckpt_name = latest_ckpt(log_dir)

    with open(log_dir / "cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}

    gs.init(backend=gs.cpu, logging_level="warning", performance_mode=True)
    env = HoverEnv(num_envs=args.episodes, env_cfg=env_cfg, obs_cfg=obs_cfg,
                   reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=False)

    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
    runner.load(str(log_dir / ckpt_name))
    policy = runner.get_inference_policy(device=gs.device)
    print(f"[eval-drone] exp={args.exp_name} ckpt={ckpt_name} episodes={args.episodes}")

    max_steps = int(env.max_episode_length)
    obs_dict = env.reset()
    alive = torch.ones(args.episodes, dtype=torch.bool, device=gs.device)
    deaths = {"timeout": 0, "crash": 0}
    dist_sum, dist_n, within_thresh = 0.0, 0, 0
    step = 0

    with torch.inference_mode():
        for t in range(max_steps + 1):
            if not alive.any():
                break
            actions = policy(obs_dict)
            obs_dict, _, dones, infos = env.step(actions)
            step = t
            for i in torch.nonzero(alive & dones).flatten().tolist():
                # HoverEnv termination: crash if tilt > 180 (never), ground < 0.1,
                # |x|>3, |y|>3, z>2 — anything before the time cap is a crash
                deaths["timeout" if t >= max_steps else "crash"] += 1
            alive &= ~dones
            m = alive
            if m.any() and t > int(3.0 / env.dt):
                target = env.commands[m]           # (n, 3) commanded position
                d = (env.base_pos[m] - target).norm(dim=1)
                dist_sum += d.sum().item()
                dist_n += int(m.sum())
                within_thresh += int((d < env.env_cfg["at_target_threshold"]).sum())

    print(f"[eval-drone] steps simulated: {step} of {max_steps}")
    print(f"[eval-drone] survived to 15 s cap: {deaths['timeout']}/{args.episodes}; crashes: {deaths['crash']}")
    if dist_n:
        mean_d = dist_sum / dist_n
        print(f"[eval-drone] mean distance to target: {mean_d:.3f} m")
        print(f"[eval-drone] samples within {env.env_cfg['at_target_threshold']} m of target: "
              f"{within_thresh}/{dist_n} ({100.0 * within_thresh / dist_n:.1f}%)")
        verdict = "EXCELLENT" if mean_d < 0.1 else "GOOD" if mean_d < 0.25 else "LEARNING" if mean_d < 0.6 else "POOR"
        print(f"[eval-drone] VERDICT: {verdict}")
    gs.destroy()


if __name__ == "__main__":
    main()
