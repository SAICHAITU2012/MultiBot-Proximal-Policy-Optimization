#!/usr/bin/env python3
"""
evaluate_go2.py — headless evaluation of a trained Go2 walking policy (v2).

Upstream go2_eval.py opens an interactive GUI and loops forever; this script instead
runs N parallel episodes headless and reports measurable quality metrics.

Lessons learned from v1 (kept here so the mistakes stay documented):
  * dones at exactly max_episode_length are TIMEOUTS (success), not falls — Go2Env
    merges terminated+truncated into one dones tensor.
  * Go2Env auto-resets a done env and zeroes episode_length_buf, so episode length
    must be recorded at the death step, not read after the loop.
  * metrics must skip steps where no env is alive (empty-tensor .mean() = NaN).

Metrics:
  * survival: episodes reaching the 20 s timeout vs falling early
  * velocity tracking: commanded vx = 0.5 m/s; mean |vx - 0.5| in steady state (t > 5 s)
  * lateral drift |vy| and turn rate |wz| (should be small for this task)

Usage:
  python evaluate_go2.py --exp-name go2-walking-s1 [--ckpt -1] [--episodes 20]
"""

import argparse
import pickle
import sys
from pathlib import Path

REPO = Path("/Users/saichaitu/Desktop/Genesis/genesis-world")
sys.path.insert(0, str(REPO / "examples" / "locomotion"))

import torch

import genesis as gs
from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner


def latest_ckpt(log_dir: Path) -> str:
    # rsl_rl 5.x under our per-iteration learn() loop names everything model_0.pt;
    # the surviving file is the final policy (overwritten after every update).
    ckpts = sorted(log_dir.glob("model_*.pt"), key=lambda f: f.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"no model_*.pt in {log_dir}")
    return ckpts[-1].name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp-name", default="go2-walking-s1")
    p.add_argument("--ckpt", default="-1", help="'-1' = latest file, or e.g. '600'")
    p.add_argument("--episodes", type=int, default=20)
    args = p.parse_args()

    log_dir = Path(__file__).resolve().parent.parent / "logs" / args.exp_name
    ckpt_name = latest_ckpt(log_dir) if args.ckpt == "-1" else f"model_{args.ckpt}.pt"

    with open(log_dir / "cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}  # eval doesn't need rewards

    gs.init(backend=gs.cpu, logging_level="warning", performance_mode=True)
    env = Go2Env(num_envs=args.episodes, env_cfg=env_cfg, obs_cfg=obs_cfg,
                 reward_cfg=reward_cfg, command_cfg=command_cfg)

    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
    runner.load(str(log_dir / ckpt_name))
    policy = runner.get_inference_policy(device=gs.device)
    print(f"[eval] exp={args.exp_name} ckpt={ckpt_name} episodes={args.episodes}")

    max_steps = int(env.max_episode_length)  # 1000 = 20 s at dt=0.02
    obs_dict = env.reset()
    alive = torch.ones(args.episodes, dtype=torch.bool, device=gs.device)
    death_step = torch.full((args.episodes,), -1, dtype=torch.int32, device=gs.device)
    timed_out, fell = 0, 0
    vx_err, vy_abs, wz_abs = [], [], []
    fall_z, fall_roll = [], []

    with torch.inference_mode():
        for t in range(max_steps + 1):
            if not alive.any():
                break
            actions = policy(obs_dict)
            obs_dict, _, dones, _ = env.step(actions)
            for i in torch.nonzero(alive & dones).flatten().tolist():
                death_step[i] = t
                if t >= max_steps:
                    timed_out += 1  # reached the 20 s cap: success
                else:
                    fell += 1
                    fall_z.append(env.base_pos[i, 2].item())
            alive &= ~dones
            m = alive
            if m.any() and t > int(5.0 / env.dt):  # steady-state window
                vx_err.append((env.base_lin_vel[m, 0] - 0.5).abs().mean().item())
                vy_abs.append(env.base_lin_vel[m, 1].abs().mean().item())
                wz_abs.append(env.base_ang_vel[m, 2].abs().mean().item())

    print(f"[eval] reached 20s timeout (success): {timed_out}/{args.episodes}")
    print(f"[eval] fell before timeout:            {fell}/{args.episodes}")
    if vx_err:
        print(f"[eval] vx tracking error |vx-0.5|: {sum(vx_err)/len(vx_err):.3f} m/s  (0 = perfect)")
        print(f"[eval] lateral drift |vy|:          {sum(vy_abs)/len(vy_abs):.3f} m/s  (lower = straighter)")
        print(f"[eval] turn rate |wz|:              {sum(wz_abs)/len(wz_abs):.3f} rad/s (lower = calmer)")
    if fell:
        print(f"[eval] falls: median death time {sorted(death_step.tolist())[max(0, fell//2)]} steps; "
              f"median z at fall {sorted(fall_z)[len(fall_z)//2] if fall_z else float('nan'):.2f} m")

    ok = timed_out == args.episodes and vx_err and sum(vx_err) / len(vx_err) < 0.1
    good = timed_out >= args.episodes * 0.7
    verdict = "EXCELLENT" if ok else "GOOD" if good else "LEARNING (needs more training)"
    print(f"[eval] VERDICT: {verdict}")
    gs.destroy()


if __name__ == "__main__":
    main()
