#!/usr/bin/env python3
"""
test_env_reset.py — PERMANENT REGRESSION TESTS for the two env bugs that broke
every Go2 push training run between Nov 6 and Nov 8, 2026 (see
INVESTIGATION_LOG.md, "ROOT CAUSE FOUND"):

  BUG 1 (buffer wipe): step() calls _reset_idx(reset_buf) every step; with an
       empty termination mask the old code still ran its reset path, which
       zeroed the env's obs/action buffers (last_actions, dof_vel,
       base_lin_vel, ...) BEFORE _update_observation. With
       simulate_action_latency=True the executed joint target was therefore
       ALWAYS default_dof_pos — the policy's actions were never executed (a
       statue) — and the policy's observations showed zero motion (blind).
  BUG 2 (z=0 spawn): the dog spawned at base z=0.0 — inside the floor — instead
       of the configured standing height (base_init_pos z=0.42, the upstream
       go2_env convention). Every episode began with a PD leg-extension launch
       and a pitch termination at step ~5-6.

Tests:
  T1  Reset with an EMPTY termination mask must not change any environment's
      velocity (dof velocities AND base linear velocity).
  T2  Reset must place the Go2 at the configured standing base height
      (env_cfg["base_init_pos"][2]).
  T3  50 zero-action steps from reset must produce ZERO terminations (the
      toss-loop detector) and the base must stay above the belly-collapse zone.

Usage:
  python test_env_reset.py                    # run the tests (expect all PASS)
  python test_env_reset.py --negative-control # monkeypatch the HISTORICAL buggy
                                              # reset; every test must FAIL
                                              # (proves the tests catch it)

Run this after ANY change to go2_push_env.py before trusting a training run.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/Users/saichaitu/Desktop/Genesis/genesis-world/examples/locomotion")
TL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TL / "experiments"))

import torch
import genesis as gs
from go2_push_env import Go2PushEnv
from go2_push_train import get_cfgs

N_ENVS = 2
STEPS = 50


def historical_buggy_reset_idx(self, envs_idx=None):
    """The pre-fix (Nov 6 20:48) _reset_idx body, verbatim behavior: no empty-mask
    guard, dog spawned at z=0. Used ONLY by --negative-control to prove the tests
    catch the historical regression."""
    import math
    n = self.num_envs if envs_idx is None else int(envs_idx.sum())
    theta = (torch.rand(n, device=gs.device) - 0.5) * (150.0 * math.pi / 180.0)
    goal = self.crate_spawn[None, :2] + (self.goal_radius + 0.3 * (torch.rand(n, 2, device=gs.device) - 0.5) * 2) * torch.stack(
        [torch.cos(theta), torch.sin(theta)], dim=1
    )
    dog_xy = self.crate_spawn[None, :2] - 1.0 * torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    dog_yaw = theta
    dog_quat = torch.stack([
        torch.cos(dog_yaw / 2), torch.zeros_like(dog_yaw),
        torch.zeros_like(dog_yaw), torch.sin(dog_yaw / 2),
    ], dim=1)
    dog_q = torch.cat([dog_xy, torch.zeros((n, 1), device=gs.device), dog_quat,
                       self.init_dof_pos.repeat(n, 1)], dim=1)
    if envs_idx is None:
        self.goal_xy.copy_(goal)
        self.robot.set_qpos(dog_q, zero_velocity=True)
    else:
        self.goal_xy[envs_idx] = goal
        self.robot.set_qpos(dog_q, envs_idx=envs_idx, zero_velocity=True)
    if n > 0:
        crate_q = self.crate_init_qpos.repeat(n, 1)
        self.crate.set_qpos(crate_q, envs_idx=envs_idx, zero_velocity=True)
    self.base_pos = self.robot.get_pos()
    self.base_quat = self.robot.get_quat()
    self.projected_gravity = self.projected_gravity  # (simplified: not needed for the tests)
    self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
    self.crate_pos = self.crate.get_pos()
    self.crate_quat = self.crate.get_quat()
    self.crate_dist = self._crate_goal_dist()
    self.dog_crate_dist = torch.norm(self.crate_pos[:, :2] - self.base_pos[:, :2], dim=1)
    self.base_lin_vel.zero_()
    self.base_ang_vel.zero_()
    self.dof_vel.zero_()
    self.actions.zero_()
    self.last_actions.zero_()
    self.last_dof_vel.zero_()
    if envs_idx is None:
        self.episode_length_buf.zero_()
        self.reset_buf.fill_(True)
    else:
        self.episode_length_buf.masked_fill_(envs_idx, 0)
        self.reset_buf.masked_fill_(envs_idx, True)


def make_env():
    env_cfg, obs_cfg, reward_cfg, push_cfg = get_cfgs()
    env = Go2PushEnv(num_envs=N_ENVS, env_cfg=env_cfg, obs_cfg=obs_cfg,
                     reward_cfg=reward_cfg, push_cfg=push_cfg, show_viewer=False)
    return env, env_cfg


def test_t1_empty_mask_preserves_state(env):
    """BUG 1 regression: an empty-mask reset must change NOTHING. Three layers:

    T1a the robot's actual velocity state (dofs + base),
    T1b the env's obs/action buffers (last_actions, actions, dof_vel,
        base_lin_vel) — the REAL historical wipe: the old code zeroed these
        every step, so with simulate_action_latency=True the executed joint
        target was always default_dof_pos (the policy's actions were never
        executed — a statue) and the policy's obs showed zero motion (blind),
    T1c end-to-end: a constant nonzero action must actually move the joints
        toward action*scale + default within a few steps."""
    n = env.num_envs
    all_dofs = torch.arange(env.robot.n_dofs, device=gs.device)
    pattern = torch.zeros(n, env.robot.n_dofs, device=gs.device)
    pattern[:, 0] = 0.3   # base vx
    pattern[:, 6:] = 0.5  # joint velocities
    env.robot.set_dofs_velocity(pattern, dofs_idx_local=all_dofs)
    v0_dofs = env.robot.get_dofs_velocity(all_dofs).clone()
    v0_base = env.robot.get_vel().clone()
    # seed the env buffers with nonzero values, as a mid-episode step would leave them
    env.actions.fill_(0.4)
    env.last_actions.fill_(0.4)
    env.dof_vel.fill_(0.6)
    env.base_lin_vel.fill_(0.2)
    env.base_ang_vel.fill_(0.1)
    empty = torch.zeros(n, dtype=torch.bool, device=gs.device)
    env._reset_idx(empty)  # the exact call step() makes when nothing terminated
    v1_dofs = env.robot.get_dofs_velocity(all_dofs)
    v1_base = env.robot.get_vel()
    a_ok = bool((env.actions == 0.4).all() and (env.last_actions == 0.4).all())
    v_ok = bool((env.dof_vel == 0.6).all() and (env.base_lin_vel == 0.2).all())
    dof_ok = torch.allclose(v0_dofs, v1_dofs, atol=1e-6)
    base_ok = torch.allclose(v0_base, v1_base, atol=1e-6)
    max_drift = float((v0_dofs - v1_dofs).abs().max())
    print(f"  T1a robot velocity preservation: dofs_ok={dof_ok} base_ok={base_ok} (max drift {max_drift:.6f})")
    print(f"  T1b env buffer preservation:     actions/last_actions ok={a_ok}, dof_vel/base_lin_vel ok={v_ok}")
    assert dof_ok and base_ok, \
        f"BUG 1 REGRESSION: an empty-mask reset changed the robot's velocities (max drift {max_drift:.6f})"
    assert a_ok and v_ok, \
        "BUG 1 REGRESSION: an empty-mask reset zeroed the env obs/action buffers " \
        "(with action latency this executes zero joint targets — the gait cannot run)"

    # T1c: end-to-end action execution (relative test — kp=20 has large static
    # error under body load, so compare against the zero-action baseline instead
    # of the ideal target; under BUG 1 the executed target is ALWAYS default, so
    # the two baselines are identical and the diff is ~0)
    def settle(action_val):
        env.reset()
        a = (torch.full((N_ENVS, env.num_actions), action_val, device=gs.device)
             if action_val != 0.0 else torch.zeros(N_ENVS, env.num_actions, device=gs.device))
        with torch.inference_mode():
            for _ in range(10):
                env.step(a)
        return env.dof_pos[0].clone()

    p_zero = settle(0.0)
    p_act = settle(0.5)
    diff = float((p_act - p_zero).abs().max())
    print(f"  T1c action executes: max joint-response difference between a 0.5 action and zero = {diff:.4f} rad")
    assert diff > 0.01, \
        f"BUG 1 REGRESSION: the policy's actions are not executed (0.5-action response identical to zero-action, diff {diff:.5f} rad)"


def test_t2_standing_spawn_height(env, env_cfg):
    """BUG 2 regression: reset must place the dog at the configured standing height."""
    env.reset()
    expected = float(env_cfg["base_init_pos"][2])
    got = env.base_pos[:, 2].clone()
    ok = bool(torch.all((got - expected).abs() < 1e-4))
    print(f"  T2 standing spawn height: expected z={expected:.3f}, got {got.tolist()} -> ok={ok}")
    assert ok, f"BUG 2 REGRESSION: reset spawn z {[round(float(v),3) for v in got]} != {expected:.3f}"


def test_t3_no_toss_loop(env):
    """BUG 2 behavioral detector: zero-action survival without terminations."""
    obs = env.reset()
    zero_a = torch.zeros(N_ENVS, env.num_actions, device=gs.device)
    terms = 0
    with torch.inference_mode():
        for t in range(STEPS):
            obs, r, d, extras = env.step(zero_a)
            terms += int(d.sum())
    z_min = float(env.base_pos[:, 2].min())
    print(f"  T3 zero-action survival ({STEPS} steps): terminations {terms} "
          f"(broken env: {N_ENVS}*{STEPS} by step ~6) | min base z {z_min:.3f}")
    assert terms == 0, f"BUG 2 REGRESSION: {terms} terminations under zero actions (toss-loop)"
    assert z_min > 0.18, f"BUG 2 REGRESSION: base collapsed to z={z_min:.3f} (belly zone)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--negative-control", action="store_true",
                   help="monkeypatch the historical buggy reset; every test must FAIL")
    args = p.parse_args()

    torch.manual_seed(7)
    gs.init(backend=gs.cpu, logging_level="error", performance_mode=True)
    env, env_cfg = make_env()

    results = {}
    tests = [
        ("T1 empty-mask state preservation", lambda: test_t1_empty_mask_preserves_state(env)),
        ("T2 standing spawn height", lambda: test_t2_standing_spawn_height(env, env_cfg)),
        ("T3 zero-action survival", lambda: test_t3_no_toss_loop(env)),
    ]
    if args.negative_control:
        Go2PushEnv._reset_idx = historical_buggy_reset_idx
        print("NEGATIVE CONTROL: historical buggy reset monkeypatched — "
              "every test below is EXPECTED to fail\n")
    for name, fn in tests:
        try:
            with torch.inference_mode():
                fn()
            results[name] = "PASS"
        except AssertionError as e:
            results[name] = f"FAIL: {e}"
    gs.destroy()

    print("\n=== RESULTS ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    fails = sum(1 for v in results.values() if v.startswith("FAIL"))
    if args.negative_control:
        print(f"\nNEGATIVE CONTROL: {fails}/3 failed as expected -> "
              f"{'TESTS VALID (they catch the historical bug)' if fails == 3 else 'WARNING: a test did NOT catch the bug!'}")
    else:
        print(f"\n{'ALL TESTS PASS — env reset is healthy' if fails == 0 else 'ENV RESET REGRESSION DETECTED — do not train!'}")
        sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
