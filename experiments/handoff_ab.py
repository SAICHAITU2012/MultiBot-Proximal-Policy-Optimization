#!/usr/bin/env python3
"""
handoff_ab.py — Priority 2 diagnostic: capture the Go2's freeze state during the
mission's re-approach, then replay it in (A) the training Go2Env and (B) the colo
mission scene. Verdict: if the dog moves in (A) but freezes in (B), the cause is
colo-scene-specific; if it freezes in both, the state/command mix is out of the
nav policy's training distribution (fix = navigation/training, not integration).
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, "/Users/saichaitu/Desktop/Genesis/genesis-world/examples/locomotion")
TL = Path("/Users/saichaitu/Desktop/Genesis/team_lab")
sys.path.insert(0, str(TL)); sys.path.insert(0, str(TL / "experiments")); sys.path.insert(0, str(TL / "team"))

import torch
import genesis as gs

import mission as M
from agents import make_task

gs.init(backend=gs.cpu, logging_level="error", performance_mode=True)
m = M.Mission(n_packages=1, planner_name="rule", max_steps=10)
go2 = m.team.robots["go2"]
ag = m.agents["go2"]
m._lp = -1e9

capture = None
freeze_ticks = 0
FREEZE_TICKS_NEED = 30  # 0.3 s of near-zero velocity under active commands

with torch.inference_mode():
    for step in range(24000):  # 240 s
        t = m.team.sim_time
        m.bb.set_clock(t)
        if t - m._lp >= 4.0:
            m._lp = t; m._plan_tick(t)
        m.team.step()
        for a in m.agents.values():
            a.update(t)
        m._check_deliveries()

        # freeze detection during the fetch re-approach (not pushing: the push
        # hover is the policy's known convergence behavior)
        cmd = go2.commands[0]
        cmd_mag = float(torch.norm(cmd[[0, 1]]))
        vel = go2.base_lin_vel[0]
        vel_mag = float(torch.norm(vel[:2]))
        if ag.task is not None and not getattr(ag, "_pushing", False) and cmd_mag > 0.15:
            if vel_mag < 0.03:
                freeze_ticks += 1
            else:
                freeze_ticks = 0
            if freeze_ticks >= FREEZE_TICKS_NEED and capture is None:
                st = go2.get_state()
                crate = m.packages[0]["entity"]
                capture = {
                    "t": t,
                    "qpos": go2.entity.get_qpos()[0].cpu().clone(),
                    "dof_vel": go2.entity.get_dofs_velocity().cpu().clone()[...,:18],
                    "commands": cmd.cpu().clone(),
                    "crate_pos": crate.get_pos()[0, :3].cpu().clone(),
                    "crate_quat": crate.get_quat()[0].cpu().clone(),
                    "phase": ag.state,
                    "task": ag.task["id"],
                }
                print(f"[CAPTURE] freeze at t={t:.1f}s task={capture['task']} "
                      f"cmd={[round(float(v),2) for v in cmd]} "
                      f"pos=({st['pos'][0]:.2f},{st['pos'][1]:.2f}) z={st['pos'][2]:.3f}")
                break
        if m._mission_complete():
            break

if capture is None:
    print("[CAPTURE] no freeze captured in 240 s")
    sys.exit(0)

pickle.dump(capture, open("../logs/handoff_capture.pkl", "wb"))
print("[CAPTURE] saved -> ../logs/handoff_capture.pkl")

# ---- (A) replay in the TRAINING nav env (go2_env.py's Go2Env) ----
print("\n[A] replay in the training Go2Env:")
from go2_env import Go2Env as TrainGo2Env
from go2_train import get_cfgs as go2_train_cfgs

tenv_cfg, tobs_cfg, treward_cfg, tcommand_cfg = go2_train_cfgs()
# match the mission's command ranges so the captured commands are settable
tcommand_cfg["lin_vel_x_range"] = [-0.5, 1.0]
tcommand_cfg["lin_vel_y_range"] = [-0.5, 0.5]
tcommand_cfg["ang_vel_range"] = [-1.0, 1.0]
tenv = TrainGo2Env(num_envs=1, env_cfg=tenv_cfg, obs_cfg=tobs_cfg,
                   reward_cfg=treward_cfg, command_cfg=tcommand_cfg)
tq = capture["qpos"].clone()
tq[2] = 0.42  # the training env's spawn height convention (base z)
tenv.robot.set_qpos(tq, zero_velocity=False)
tenv.robot.set_dofs_velocity(capture["dof_vel"][6:18], dofs_idx_local=tenv.motors_dof_idx)
tenv.commands[0] = capture["commands"]
movements = []
with torch.inference_mode():
    for t in range(300):  # 3 s
        # the nav policy: load once via the mission's saved nav checkpoint is
        # overkill — replay the COMMANDS through the env's own policy-free
        # dynamics by driving with the captured command (the policy tracking
        # is what we're testing; use the mission's nav policy directly)
        obs = tenv.get_observations()
        # use the mission's nav policy (go2-nav/model_800)
        if "navpol" not in dir():
            from rsl_rl.runners import OnPolicyRunner
            nrunner = OnPolicyRunner(tenv, {**{"algorithm": {"class_name": "PPO", "clip_param": 0.2, "desired_kl": 0.01, "entropy_coef": 0.01, "gamma": 0.99, "lam": 0.95, "learning_rate": 0.001, "max_grad_norm": 1.0, "num_learning_epochs": 5, "num_mini_batches": 4, "schedule": "adaptive", "use_clipped_value_loss": True, "value_loss_coef": 1.0},
                  "actor": {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu", "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}},
                  "critic": {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu"},
                  "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
                  "num_steps_per_env": 24, "save_interval": 1000, "run_name": "ab", "logger": "tensorboard"}},
                  "../logs/go2-nav", device=gs.device)
            nrunner.load("../logs/go2-nav/model_800.pt")
            navpol = nrunner.get_inference_policy(device=gs.device)
        actions = navpol(obs)
        obs, r, d, _ = tenv.step(actions)
        v = tenv.base_lin_vel[0]
        movements.append(float(torch.norm(v[:2])))
    avg_v = sum(movements[-100:]) / 100
    print(f"[A] avg |v_xy| over the last second: {avg_v:.3f} m/s -> "
          f"{'MOVES (policy responds to the state+command)' if avg_v > 0.05 else 'FROZEN (the state/command freezes the nav policy too)'}")
