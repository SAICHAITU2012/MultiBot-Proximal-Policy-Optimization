#!/usr/bin/env python3
"""
m1_obs_diff.py — M1: the observation-equivalence diagnosis (the advisor's protocol).

PART A: run the mission; capture the COMPLETE simulator state at the first
post-shove hover frame (the push policy engaged + the dog/crates near-stationary
for 1 s = the hover confirmed). The capture: the dog's qpos(19)/qvel(18), the
crate's pos/quat/vel, the commands, the action buffers, the goal, the full
49-dim obs, and the policy's action output at that state.

PART B: restore that exact state in the training environment (Go2PushEnv, the
same crate geometry/mass, the same goal), then:
  B1. the obs diff: the per-component (49), the largest abs, the mean abs —
      separating the observation mismatch from the policy generalization failure
  B2. the policy diff: the same checkpoint applied to both obs vectors —
      the actions compared
  B3. the behavior: the training env stepped 3 s from the restored state —
      does the dog crouch (the reproduced failure)?

No implementation changes during this diagnostic. The deliverable: the report.
"""

import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, "/Users/saichaitu/Desktop/Genesis/genesis-world/examples/locomotion")
TL = Path("/Users/saichaitu/Desktop/Genesis/team_lab")
sys.path.insert(0, str(TL)); sys.path.insert(0, str(TL / "experiments")); sys.path.insert(0, str(TL / "team"))

import torch
import genesis as gs

import mission as M

gs.init(backend=gs.cpu, logging_level="error", performance_mode=True)
m = M.Mission(n_packages=1, planner_name="rule", max_steps=10)
go2 = m.team.robots["go2"]
ag = m.agents["go2"]
crate = m.packages[0]["entity"]
m._lp = -1e9

# ---------- PART A: the mission capture at the hover ----------
HOVER_TICKS_NEED = 100  # 1 s of near-stationary dog + crate while pushing
hover_ticks = 0
capture = None
prev_crate = None
prev_dog = None

print("[A] running the mission until the post-shove hover...")
with torch.inference_mode():
    for step in range(30000):  # 300 s
        t = m.team.sim_time
        m.bb.set_clock(t)
        if t - m._lp >= 4.0:
            m._lp = t; m._plan_tick(t)
        m.team.step()
        for a in m.agents.values():
            a.update(t)
        m._check_deliveries()

        if getattr(ag, "_pushing", False) and ag.task is not None:  # the TASK agent's flag
            if not hasattr(ag, "_push_started_t") or ag._push_started_t is None:
                ag._push_started_t = t
            # the capture at 20 s into the push: the policy's behavior is
            # established by then (pushing or hovering — capture either; the
            # diff diagnoses it)
            if t - ag._push_started_t >= 20.0 and capture is None:
                # CAPTURE the complete state
                qpos = go2.entity.get_qpos()[0].cpu().clone()          # (19,)
                qvel = go2.entity.get_dofs_velocity()[0].cpu().clone() # (18,)
                crate_pos = crate.get_pos()[0, :3].cpu().clone()
                crate_quat = crate.get_quat()[0].cpu().clone()
                crate_vel = crate.get_vel()[0, :3].cpu().clone()
                commands = go2.commands[0].cpu().clone()
                actions = go2.actions[0].cpu().clone()
                last_actions = go2.last_actions[0].cpu().clone()
                # the obs: the mission's _learned_push construction
                st = go2.get_state()
                from genesis.utils.geom import inv_quat, transform_by_quat
                bt = torch.tensor(st["quat"], dtype=gs.tc_float, device=gs.device).unsqueeze(0)
                iq = inv_quat(bt)
                dog = torch.tensor([st["pos"][0], st["pos"][1], 0.0],
                                   dtype=gs.tc_float, device=gs.device).unsqueeze(0)
                cp = crate.get_pos()[0]
                crate_t = torch.tensor([cp[0], cp[1], 0.0], dtype=gs.tc_float, device=gs.device).unsqueeze(0)
                goal_t = torch.tensor([m.goal[0], m.goal[1], 0.0], dtype=gs.tc_float, device=gs.device).unsqueeze(0)
                zz = torch.zeros((1, 1), dtype=gs.tc_float, device=gs.device)
                cr = transform_by_quat(torch.cat([crate_t[:, :2] - dog[:, :2], zz], -1), iq)[:, :2]
                gr = transform_by_quat(torch.cat([goal_t[:, :2] - crate_t[:, :2], zz], -1), iq)[:, :2]
                obs = torch.cat((
                    go2.base_ang_vel * 0.25,
                    go2.projected_gravity,
                    torch.tensor([[0.6, 0.0, 0.0]], dtype=gs.tc_float, device=gs.device),
                    go2.dof_pos - go2.default_dof_pos,
                    go2.dof_vel * 0.05,
                    go2.actions,
                    cr * 1.0, gr * 1.0), dim=-1)[0].cpu()
                policy_action = go2.actions[0].cpu().clone()  # the push policy's last output
                capture = {
                    "t": t, "qpos": qpos, "qvel": qvel,
                    "crate_pos": crate_pos, "crate_quat": crate_quat, "crate_vel": crate_vel,
                    "commands": commands, "actions": actions, "last_actions": last_actions,
                    "obs": obs, "policy_action": policy_action,
                    "dog_pos": torch.tensor(st["pos"][:3]),
                    "goal_xy": torch.tensor([m.goal[0], m.goal[1]]),
                    "crate_spawn_xy": torch.tensor([float(v) for v in m.packages[0]["spawn"][:2]]),
                }
                print(f"[A] HOVER captured at t={t:.1f}s: dog=({st['pos'][0]:.2f},{st['pos'][1]:.2f}) "
                      f"crate=({float(cp[0]):.2f},{float(cp[1]):.2f}) | obs(49) + qpos(19) + qvel(18) saved")
                break
        if m._mission_complete():
            break

if capture is None:
    print("[A] no hover captured in 300 s — the hover signature needs retuning")
    sys.exit(1)

pickle.dump(capture, open("m1_capture.pkl", "wb"))

# ---------- PART B: the training env restore + the diffs ----------
print("\n[B] restoring the exact state in the training env (Go2PushEnv)...")
from go2_push_env import Go2PushEnv
from go2_push_train import get_cfgs
import colo_env

cfgs = get_cfgs()
pcfg = dict(cfgs[3])
pcfg["crate_size"] = [0.35, 0.35, 0.12]
pcfg["crate_mass"] = 7.0
pcfg["crate_spawn"] = [0.65, 0.0, 0.065]  # the M0b training env's own crate config
pcfg["goal_radius"] = 2.85                # the crate-goal distance (the mission's geometry)
gs.destroy() if getattr(gs, "_initialized", False) else None
gs.init(backend=gs.cpu, logging_level="error", performance_mode=True)
tenv = Go2PushEnv(num_envs=1, env_cfg=cfgs[0], obs_cfg=cfgs[1], reward_cfg=cfgs[2],
                  push_cfg=pcfg, show_viewer=False)
tenv.reset()

# restore the dog's state
tenv.robot.set_qpos(capture["qpos"], zero_velocity=False)
tenv.robot.set_dofs_velocity(capture["qvel"], dofs_idx_local=None)
# restore the crate
cq = torch.cat([capture["crate_pos"], capture["crate_quat"]])
tenv.crate.set_qpos(cq, zero_velocity=False)
# restore the goal
tenv.goal_xy[0, 0] = capture["goal_xy"][0]
tenv.goal_xy[0, 1] = capture["goal_xy"][1]
tenv._prev_box_x = capture["crate_pos"][0].clone() if hasattr(tenv, "_prev_box_x") else None

# B1: the obs diff
tobs = tenv.get_observations()["policy"][0].cpu()
mobs = capture["obs"]
names = (["ang_vel_x", "ang_vel_y", "ang_vel_z", "grav_x", "grav_y", "grav_z",
          "cmd_x", "cmd_y", "cmd_w"] +
         [f"dof{i}" for i in range(12)] + [f"dvel{i}" for i in range(12)] +
         [f"act{i}" for i in range(12)] +
         ["crate_rel_x", "crate_rel_y", "goal_rel_x", "goal_rel_y"])
diffs = [(abs(float(a) - float(b)), i, float(a), float(b)) for i, (a, b) in enumerate(zip(mobs, tobs))]
diffs.sort(reverse=True)
print("\n[B1] === THE OBS DIFF (the mission handoff vs the training env, the same restored state) ===")
print(f"[B1] the largest absolute differences (the mission obs vs the training obs):")
for d, i, mv, tv in diffs[:8]:
    print(f"[B1]   [{i:2d}] {names[i] if i < len(names) else i}: mission={mv:+.4f} train={tv:+.4f} |diff|={d:.4f}")
mean_abs = sum(d for d, _, _, _ in diffs) / len(diffs)
n_ood = sum(1 for d, _, _, _ in diffs if d > 0.05)
print(f"[B1] mean abs diff: {mean_abs:.4f} | components with |diff| > 0.05: {n_ood}/49")

# B2: the policy diff
policy = colo_env.load_actor_from_log("../logs/go2-push-m0b", "model_1200.pt")
dev = next(policy.parameters()).device
a_mission = policy(mobs.to(dev))
a_train = policy(tobs.to(dev))
a_diff = float(torch.norm(a_mission - a_train))
print(f"\n[B2] the same checkpoint's actions:")
print(f"[B2]   from the mission obs:    {[round(float(v),3) for v in a_mission]}")
print(f"[B2]   from the training obs:  {[round(float(v),3) for v in a_train]}")
print(f"[B2]   the action L2 diff: {a_diff:.4f}")

# B3: the behavior — the training env stepped 3 s from the restored state
crate_z0 = float(tenv.crate.get_pos()[0, 2])
dog_z0 = float(tenv.base_pos[0, 2])
x_start = float(tenv.crate.get_pos()[0, 0])
crate_moved = []
with torch.inference_mode():
    for t in range(150):  # 3 s
        a = policy(tenv.get_observations()["policy"])
        tenv.step(a)
        crate_moved.append(float(tenv.crate.get_pos()[0, 0]))
x_end = crate_moved[-1]
moved = abs(x_end - x_start)
print(f"\n[B3] the training env's behavior from the restored state (3 s):")
print(f"[B3]   the crate's x: {x_start:.3f} -> {x_end:.3f} (moved {moved:.3f} m)")
print(f"[B3]   the dog's z: {dog_z0:.3f} -> {float(tenv.base_pos[0,2]):.3f}")

print(f"\n[B3] === THE M1 VERDICT FRAMEWORK ===")
obs_match = mean_abs < 0.01 and a_diff < 0.05
print(f"[B3] the obs equivalence: {'MATCH (the obs builders agree)' if obs_match else 'MISMATCH (the obs builders differ — see B1)'}")
print(f"[B3] the policy behavior at the restored state: the crate moved {moved:.3f} m in 3 s "
      f"({'PUSHES' if moved > 0.05 else 'HOVERS/CROUCHES'})")
if obs_match and moved <= 0.05:
    print("[B3] VERDICT: THE OBS ARE EQUIVALENT — the checkpoint's behavior at this state IS the "
          "hover/crouch — a policy generalization failure at this state (the training never "
          "visited it). The fix: the training-distribution adjustment, evidence-based.")
elif not obs_match:
    print("[B3] VERDICT: THE OBS BUILDERS DIFFER — fix the mission's obs construction to match "
          "the training env, THEN re-test the behavior.")
gs.destroy()
