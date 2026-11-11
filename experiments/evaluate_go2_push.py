#!/usr/bin/env python3
"""
evaluate_go2_push.py — fresh-seed evaluation of a Go2 crate-push policy.

Per the advisor's evidence standard: fresh random spawns (never training
episodes), N parallel episodes, and the honest state evidence:
  REACHED  — the arm... (n/a for quadruped) the dog closes to contact distance
  CONTACT  — mask-verified dog<->crate contacts during the episode
  PUSHED   — the crate displaced >= 0.15 m from its spawn toward the goal

Usage:
  python evaluate_go2_push.py --exp-name go2-push-s7 --ckpt model_900 --episodes 16
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import torch

import genesis as gs
from go2_push_env import Go2PushEnv
from go2_push_train import get_cfgs, get_train_cfg
from rsl_rl.runners import OnPolicyRunner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp-name", default="go2-push-s7")
    p.add_argument("--ckpt", default="-1", help="-1 = highest-numbered checkpoint")
    p.add_argument("--episodes", type=int, default=16)
    args = p.parse_args()

    log_dir = Path(__file__).resolve().parent.parent / "logs" / args.exp_name
    with open(log_dir / "cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, push_cfg, train_cfg = pickle.load(f)

    gs.init(backend=gs.cpu, logging_level="error", performance_mode=True)
    env = Go2PushEnv(num_envs=args.episodes, env_cfg=env_cfg, obs_cfg=obs_cfg,
                     reward_cfg=reward_cfg, push_cfg=push_cfg, show_viewer=False)
    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)
    if args.ckpt == "-1":
        ckpts = sorted(log_dir.glob("model_*.pt"), key=lambda f: int(f.stem.split("_")[1]))
        ckpt = ckpts[-1].name
    else:
        ckpt = args.ckpt if args.ckpt.endswith(".pt") else f"{args.ckpt}.pt"
    runner.load(str(log_dir / ckpt))
    policy = runner.get_inference_policy(device=gs.device)
    print(f"[eval-push] exp={args.exp_name} ckpt={ckpt} episodes={args.episodes}")

    obs = env.reset()
    x0 = env.crate_pos[:, 0].clone()
    d0 = env.crate_dist.clone()
    max_disp = torch.zeros(args.episodes, device=gs.device)
    ever_contact = torch.zeros(args.episodes, dtype=torch.bool, device=gs.device)
    fall_steps = torch.zeros(args.episodes, dtype=torch.int, device=gs.device)
    # mission-relevant metrics (added 2026-11-08 after the fall-slam finding):
    # the aggregate max_disp accumulates across auto-reset lives, so a policy
    # that trips and body-slams every ~0.5 s can look "pushed" without ever
    # completing ONE clean engagement. first_disp = crate displacement at the
    # env's FIRST termination (one life); true_falls = pitch/roll/error
    # terminations (the auto-reset removes the fallen dog the same step, so a
    # z<0.15 counter never sees real falls).
    first_disp = torch.zeros(args.episodes, device=gs.device)
    first_life_done = torch.zeros(args.episodes, dtype=torch.bool, device=gs.device)
    true_falls = 0  # ALL non-timeout terminations (not just first per env)

    with torch.inference_mode():
        for t in range(600):  # 12 s episodes
            a = policy(obs)
            obs, r, d, extras = env.step(a)
            # the FALL MEASUREMENT: the base below 0.15 m (the standing gait's
            # base ~0.30-0.42; a fallen/tripped dog's base < 0.15)
            fall_steps += (env.base_pos[:, 2] < 0.15).to(torch.int)
            disp = (env.crate_pos[:, 0] - x0).abs()  # displacement along the push axis
            max_disp = torch.maximum(max_disp, disp)
            new_life = d.bool() & ~first_life_done
            falls_now = d.bool() & ~extras["time_outs"].bool()
            true_falls += int(falls_now.sum())
            if bool(new_life.any()):
                idx = new_life.nonzero(as_tuple=True)[0]
                first_disp[idx] = disp[idx].detach()
                first_life_done |= new_life
            c = env.robot.get_contacts(with_entity=env.crate)
            if c is not None and "valid_mask" in c:
                vm = c["valid_mask"].bool()
                if vm.any():
                    arm_lo, arm_hi = env.robot.geom_start, env.robot.geom_end
                    box_lo, box_hi = env.crate.geom_start, env.crate.geom_end
                    pair = torch.logical_or(
                        torch.logical_and(c["geom_a"] >= arm_lo, c["geom_a"] < arm_hi),
                        torch.logical_and(c["geom_b"] >= arm_lo, c["geom_b"] < arm_hi))
                    pair = torch.logical_and(pair, torch.logical_or(
                        torch.logical_and(c["geom_a"] >= box_lo, c["geom_a"] < box_hi),
                        torch.logical_and(c["geom_b"] >= box_lo, c["geom_b"] < box_hi)))
                    hit = (vm & pair).any(dim=1)
                    ever_contact |= hit
            if bool(d.all()):
                break

    moved = int((max_disp > 0.01).sum())
    pushed = int((max_disp >= 0.15).sum())
    # a policy that never terminates: its first life IS the whole window
    never_done = ~first_life_done
    if bool(never_done.any()):
        first_disp[never_done] = max_disp[never_done]
    first_pushed = int((first_disp >= 0.15).sum())
    print(f"[eval-push] exp={args.exp_name} ckpt={ckpt} episodes={args.episodes}")
    print(f"[eval-push] REACHED+CONTACT (verified): {int(ever_contact.sum())}/{args.episodes}")
    print(f"[eval-push] PUSHED (aggregate, across reset lives >= 0.15 m): {pushed}/{args.episodes}")
    print(f"[eval-push] FIRST-LIFE PUSHED (one clean engagement >= 0.15 m): {first_pushed}/{args.episodes}")
    print(f"[eval-push] TRUE FALLS (terminations): {true_falls} across {args.episodes} envs in 600 steps")
    print(f"[eval-push] moved at all (>1 cm): {moved}/{args.episodes}")
    print(f"[eval-push] max displacement: mean {float(max_disp.mean()):+.3f} m | "
          f"median {float(max_disp.median()):+.3f} m | best {float(max_disp.max()):+.3f} m")
    print(f"[eval-push] first-life displacement: mean {float(first_disp.mean()):+.3f} m | "
          f"best {float(first_disp.max()):+.3f} m")
    print(f"[eval-push] FALL STEPS per episode: {[int(v) for v in fall_steps]} "
          f"(NOTE: same-step auto-reset hides true falls — see TRUE FALLS)")
    verdict = ("STRONG" if pushed >= args.episodes * 0.5 else
               "WORKING" if pushed >= args.episodes * 0.25 else
               "EMERGING" if moved >= args.episodes * 0.5 else "NOT LEARNED")
    print(f"[eval-push] VERDICT (aggregate): {verdict} | first-life verdict: "
          f"{'STRONG' if first_pushed >= args.episodes * 0.5 else 'WORKING' if first_pushed >= args.episodes * 0.25 else 'EMERGING' if first_pushed > 0 else 'NONE'}")
    gs.destroy()


if __name__ == "__main__":
    main()
