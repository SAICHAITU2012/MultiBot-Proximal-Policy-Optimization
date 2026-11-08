#!/usr/bin/env python3
"""
g1_train_guarded.py — Stage 4: Unitree G1 humanoid training (Genesis-Humanoid stack),
laptop-safe edition.

Wiring mirrors upstream examples/run_ppo_walking.py::train_policy with deliberate
differences:
  * runs in Genesis-Humanoid's own pinned venv (genesis-world 0.3.13) — launch with:
      GH/.venv/bin/python g1_train_guarded.py ...
  * applies g1_headless_patch (GL-free OffscreenRenderer stub + registry reward alias)
    BEFORE env creation, so training works without WindowServer access
  * torch device forced to CPU (no CUDA on this Mac)
  * laptop-scale num_envs (upstream: 8192 on L40s)
  * machine_guard wraps algorithm.train_one_iteration — pauses on thermal/load stress,
    raises SessionOver at the wall-clock cap; a final checkpoint is always saved
  * wandb disabled; logger uses stdout+csv only
  * after training, checkpoints are copied next to the saved configs so
    upstream evaluate_policy's glob (logs/<exp>/*, newest first) finds everything

Run from /Users/saichaitu/Desktop/Genesis/Genesis-Humanoid (relative logs/ paths).
"""

import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                                        # g1_headless_patch
sys.path.insert(0, str(HERE.parent / "monitor"))                     # machine_guard
sys.path.insert(0, "/Users/saichaitu/Desktop/Genesis/Genesis-Humanoid/examples")  # utils

import g1_headless_patch

g1_headless_patch.apply_headless_patch()          # BEFORE any env/scene creation
g1_headless_patch.apply_registry_reward_alias()   # pre-existing upstream name mismatch

import torch  # noqa: E402

from gs_agent.algos.config.registry import PPO_WALKING_MLP  # noqa: E402
from gs_agent.algos.ppo import PPO  # noqa: E402
from gs_agent.runners.config.registry import RUNNER_WALKING_MLP  # noqa: E402
from gs_agent.runners.onpolicy_runner import OnPolicyRunner  # noqa: E402
from gs_agent.utils.logger import configure as logger_configure  # noqa: E402
from gs_agent.wrappers.gs_env_wrapper import GenesisEnvWrapper  # noqa: E402
from gs_env.sim.envs import WalkingEnv  # noqa: E402
from gs_env.sim.envs.config.registry import EnvArgsRegistry  # noqa: E402
from machine_guard import Guard  # noqa: E402


class SessionOver(Exception):
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", default="g1-walk-s1")
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--total-iterations", type=int, default=300)
    p.add_argument("--session-minutes", type=float, default=40.0)
    p.add_argument("--stand", action="store_true",
                   help="stand-first curriculum: fix all commands to 0 (learn balance "
                        "before walking; walking is the stretch goal afterwards)")
    p.add_argument("--resume-dir", default=None,
                   help="load the newest checkpoint_*.pt from this experiment dir "
                        "under Genesis-Humanoid/logs/ before training")
    args = p.parse_args()

    device = torch.device("cpu")
    print(f"[stage4] exp={args.exp_name} num_envs={args.num_envs} iters={args.total_iterations} "
          f"device=cpu mode={'stand' if args.stand else 'walk'}")

    env_args = EnvArgsRegistry["g1_walk"]
    if args.stand:
        env_args = env_args.model_copy(update={
            "commands_range": ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        })
        print("[stage4] commands fixed to zero (stand curriculum)")
    env = WalkingEnv(args=env_args, num_envs=args.num_envs, show_viewer=False, device=device)
    wrapped_env = GenesisEnvWrapper(env, device=env.device)

    ppo = PPO(env=wrapped_env, cfg=PPO_WALKING_MLP, device=wrapped_env.device)

    if args.resume_dir:
        import glob
        cands = sorted(glob.glob(f"logs/{args.resume_dir}/*/checkpoints/checkpoint_*.pt"),
                       key=lambda f: f.stat().st_mtime)
        if not cands:
            raise FileNotFoundError(f"no checkpoints under logs/{args.resume_dir}")
        ckpt = torch.load(cands[-1], map_location="cpu", weights_only=False)
        ppo._actor.load_state_dict(ckpt["actor_state_dict"])
        ppo._critic.load_state_dict(ckpt["critic_state_dict"])
        print(f"[stage4] resumed actor+critic from {cands[-1]}")

    save_path = Path("logs") / args.exp_name
    runner_args = RUNNER_WALKING_MLP.model_copy(update={
        "total_iterations": args.total_iterations,
        "save_interval": 50,
        "save_path": save_path,
    })
    runner = OnPolicyRunner(algorithm=ppo, runner_args=runner_args, device=wrapped_env.device)

    # Logging + config artifacts (mirrors upstream train_policy)
    logger_folder = save_path / datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logger_configure(
        folder=str(logger_folder),
        format_strings=["stdout", "csv"],
        entity=None, project=None, exp_name=args.exp_name, mode="disabled",
    )
    config_dir = logger_folder / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    from utils import config_to_yaml  # examples/utils.py
    config_to_yaml(env_args, config_dir / "env_args.yaml")
    config_to_yaml(PPO_WALKING_MLP, config_dir / "algo_cfg.yaml")
    config_to_yaml(runner_args, config_dir / "runner_args.yaml")

    # Guard: intercept each training iteration
    guard = Guard(session_minutes=args.session_minutes, log_csv=str(logger_folder / "guard.csv"))
    orig_train_one_iteration = runner.algorithm.train_one_iteration

    def guarded_train_one_iteration(*a, **k):
        guard.wait_until_safe()
        if guard.session_expired():
            raise SessionOver()
        return orig_train_one_iteration(*a, **k)

    runner.algorithm.train_one_iteration = guarded_train_one_iteration

    t0 = time.time()
    completed = 0
    try:
        runner.train(metric_logger=logger)
        completed = args.total_iterations
    except SessionOver:
        print(f"[stage4] session cap reached — saving final checkpoint")
    finally:
        try:
            runner._save_checkpoint(Path("checkpoint_final.pt"))
        except Exception as e:  # noqa: BLE001
            print(f"[stage4] final checkpoint save failed: {e}")
        guard.close()
        # merge checkpoints next to configs so evaluate_policy finds them
        ckpt_src = runner.checkpoint_dir.resolve()
        ckpt_dst = logger_folder.resolve() / "checkpoints"
        if ckpt_src != ckpt_dst:
            ckpt_dst.mkdir(parents=True, exist_ok=True)
            for f in ckpt_src.glob("*.pt"):
                shutil.copy2(f, ckpt_dst / f.name)
        print(f"[stage4] DONE {completed}/{args.total_iterations} iters in {(time.time()-t0)/60:.1f} min. "
              f"Artifacts: {logger_folder}")


if __name__ == "__main__":
    main()
