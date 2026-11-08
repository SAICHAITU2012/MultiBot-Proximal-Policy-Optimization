#!/usr/bin/env python3
"""
drone_train_guarded.py — Stage 2: cf2x drone hover training, laptop-safe edition.

Adapted from genesis-world/examples/drone/hover_train.py (upstream, MIT-licensed).
Same deliberate differences as go2_train_guarded.py: selectable backend, laptop-scale
num_envs, machine_guard pauses + session cap, per-iteration learn() calls, log dir
preserved unless --fresh, save_interval lowered to 50.

Task: hover at a 3-D position target (resampled every 3 s from a ±1 m × ±1 m box
at z=1 m). Actions = 4 raw motor commands; reward = position tracking + smoothness
+ yaw + crash penalties (see get_cfgs in upstream hover_train.py).
"""

import argparse
import pickle
import shutil
import sys
import time
from pathlib import Path

REPO = Path("/Users/saichaitu/Desktop/Genesis/genesis-world")
sys.path.insert(0, str(REPO / "examples" / "drone"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "monitor"))

import genesis as gs
from hover_env import HoverEnv
from hover_train import get_cfgs, get_train_cfg
from machine_guard import Guard
from rsl_rl.runners import OnPolicyRunner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp-name", default="drone-hover")
    p.add_argument("-b", "--num-envs", type=int, default=512)
    p.add_argument("--max-iterations", type=int, default=400)
    p.add_argument("--session-minutes", type=float, default=45.0)
    p.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--resume", action="store_true",
                   help="continue from the newest checkpoint in the log dir")
    p.add_argument("--entropy-coef", type=float, default=None,
                   help="override algorithm.entropy_coef (exploration bonus; upstream 0.004)")
    p.add_argument("--desired-kl", type=float, default=None,
                   help="override algorithm.desired_kl (adaptive-LR KL target; upstream 0.01)")
    p.add_argument("--fresh", action="store_true")
    args = p.parse_args()

    log_dir = Path(__file__).resolve().parent.parent / "logs" / args.exp_name
    if args.fresh and log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[stage2] log_dir={log_dir}")

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name)
    train_cfg["save_interval"] = 50
    if args.entropy_coef is not None:
        train_cfg["algorithm"]["entropy_coef"] = args.entropy_coef
        print(f"[stage2] entropy_coef overridden to {args.entropy_coef}")
    if args.desired_kl is not None:
        train_cfg["algorithm"]["desired_kl"] = args.desired_kl
        print(f"[stage2] desired_kl overridden to {args.desired_kl}")

    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    print(f"[stage2] init backend={args.backend} num_envs={args.num_envs}")
    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.gpu, precision="32",
            logging_level="warning", performance_mode=True)

    env = HoverEnv(num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
                   reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=False)

    runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=gs.device)

    resume_it = 0
    if args.resume:
        ckpts = sorted(log_dir.glob("model_*.pt"), key=lambda f: int(f.stem.split("_")[1]))
        if not ckpts:
            raise FileNotFoundError(f"--resume requested but no model_*.pt in {log_dir}")
        latest = ckpts[-1]
        runner.load(str(latest))
        resume_it = int(latest.stem.split("_")[1])
        print(f"[stage2] resuming from {latest.name} (iteration {resume_it})")

    guard = Guard(session_minutes=args.session_minutes, log_csv=str(log_dir / "guard.csv"))
    t_start = time.time()
    done = resume_it
    try:
        for it in range(resume_it + 1, args.max_iterations + 1):
            guard.wait_until_safe()
            if guard.session_expired():
                print(f"[stage2] session cap reached after {done} iterations — checkpointed, exiting cleanly")
                break
            # advance rsl_rl's counter manually so checkpoints get unique names
            # (same fix as go2_train_guarded.py — see note there)
            runner.current_learning_iteration = it
            t0 = time.time()
            runner.learn(num_learning_iterations=1, init_at_random_ep_len=(it == 1))
            dt = time.time() - t0
            done = it
            elapsed = time.time() - t_start
            n = it - resume_it
            eta = elapsed / n * (args.max_iterations - it)
            print(f"[stage2] iter {it:4d}/{args.max_iterations}  {dt:6.1f}s/iter  "
                  f"elapsed {elapsed/60:5.1f}min  eta {eta/60:5.1f}min", flush=True)
    finally:
        guard.close()
        print(f"[stage2] DONE {done} iterations. Logs+checkpoints: {log_dir}")


if __name__ == "__main__":
    main()
