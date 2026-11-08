#!/usr/bin/env python3
"""
evaluate_panda.py — headless evaluation of a trained Panda grasp policy.

The upstream reward is keypoint-based (finger keypoints vs object keypoints), so
the trained behavior is reach-and-grasp. Metrics over N parallel episodes
(3 s each at 100 Hz):
  * final finger-object keypoint distance (the training objective)
  * box lift: max height gain of the box above its spawn height
  * box displacement from spawn (did the grasp actually move it?)

Usage:
  python evaluate_panda.py --exp-name grasp-panda [--episodes 8] [--episodes-per-round 1]
"""

import argparse
import pickle
import sys
from pathlib import Path

REPO = Path("/Users/saichaitu/Desktop/Genesis/genesis-world")
sys.path.insert(0, str(REPO / "examples" / "manipulation"))

import torch

import genesis as gs
from grasp_env import GraspEnv
from rsl_rl.runners import OnPolicyRunner


def latest_ckpt(log_dir: Path) -> str:
    ckpts = sorted(log_dir.glob("model_*.pt"), key=lambda f: int(f.stem.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"no model_*.pt in {log_dir}")
    return ckpts[-1].name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp-name", default="grasp-panda")
    p.add_argument("--episodes", type=int, default=8)
    args = p.parse_args()

    log_dir = Path(__file__).resolve().parent.parent / "logs" / f"{args.exp_name}_rl"
    ckpt_name = latest_ckpt(log_dir)

    with open(log_dir / "cfgs.pkl", "rb") as f:
        env_cfg, reward_scales, robot_cfg, rl_train_cfg, bc_train_cfg = pickle.load(f)
    env_cfg["num_envs"] = args.episodes

    gs.init(backend=gs.cpu, logging_level="warning", performance_mode=True)
    env = GraspEnv(env_cfg=env_cfg, reward_cfg=reward_scales, robot_cfg=robot_cfg, show_viewer=False)

    runner = OnPolicyRunner(env, rl_train_cfg, str(log_dir), device=gs.device)
    runner.load(str(log_dir / ckpt_name))
    policy = runner.get_inference_policy(device=gs.device)
    print(f"[eval-panda] exp={args.exp_name} ckpt={ckpt_name} episodes={args.episodes}")

    obs_dict = env.get_observations()
    max_steps = int(env_cfg["episode_length_s"] / env_cfg["ctrl_dt"])  # 300
    spawn_z = env.object.get_pos()[:, 2].clone()
    spawn_xy = env.object.get_pos()[:, :2].clone()
    kp_dist_final, lift_max, disp_max = None, None, None

    with torch.inference_mode():
        for t in range(max_steps):
            actions = policy(obs_dict)
            obs_dict, _, _, _ = env.step(actions)
            obj = env.object.get_pos()
            lift = (obj[:, 2] - spawn_z).clamp(min=0)
            disp = (obj[:, :2] - spawn_xy).norm(dim=1)
            lift_max = lift.clone() if lift_max is None else torch.maximum(lift_max, lift)
            disp_max = disp.clone() if disp_max is None else torch.maximum(disp_max, disp)

    # The reward is exp(-summed keypoint distance) — invert it to get distance.
    # NOTE: upstream's RL stage trains REACH-to-box (fixed box); it does not
    # train lifting. Lift/displacement above are recorded for honesty: 0 lift
    # is EXPECTED for this reward design, not a failure.
    with torch.inference_mode():
        rew = env._reward_keypoints()
    kp_dist_final = -torch.log(rew.clamp(min=1e-6))  # summed keypoint distance (m)

    print(f"[eval-panda] final finger-object keypoint distance: mean {kp_dist_final.mean().item():.4f} m "
          f"median {kp_dist_final.median().item():.4f} m (summed over keypoints; ~0 = perfect reach)")
    print(f"[eval-panda] max box lift: mean {lift_max.mean().item()*100:.2f} cm (0 expected — reach task, box fixed)")
    print(f"[eval-panda] max box displacement: mean {disp_max.mean().item()*100:.2f} cm (0 expected)")
    reached = int((kp_dist_final < 0.10).sum())
    verdict = ("EXCELLENT" if reached >= args.episodes * 0.75 else
               "GOOD" if reached >= args.episodes * 0.4 else
               "LEARNING" if kp_dist_final.mean().item() < 0.3 else "POOR")
    print(f"[eval-panda] precise reaches (< 0.10 m): {reached}/{args.episodes}")
    print(f"[eval-panda] VERDICT: {verdict}")
    gs.destroy()


if __name__ == "__main__":
    main()
