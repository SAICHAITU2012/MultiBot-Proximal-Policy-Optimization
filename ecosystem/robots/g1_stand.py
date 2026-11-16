#!/usr/bin/env python3
"""g1_stand.py — Genesis 1.4.5 port of the G1 standing policy (no retraining).

Source of truth: Genesis-Humanoid/logs/g1-stand-s2/.../checkpoint_final.pt
  actor  : obs 96 -> MLP(256,256,128, relu) -> 29 -> mu(29)
  control: ctrl_type = DR_JOINT_POSITION_VELOCITY -> torque control with explicit PD
           plus a low-passed velocity feed-forward (leggedrobots._apply_dr_joint_pos_vel):
             tgt      = 0.25*a                              (OFFSET from default)
             v_des    = (tgt - tgt_prev)/control_dt
             v_lp    += alpha*(v_des - v_lp)                 alpha=0.5
             q_force  = kp*(tgt + default - dof_pos) + kd*(-dof_vel + ff*v_lp)
           clamped to per-joint torque limits; recomputed EVERY sim step.
  sim    : runtime dt and controller period come from core.timing
  obs    : [last_action(29), dof_pos(29), dof_vel*0.1(29),
            projected_gravity(3), base_ang_vel*0.5(3), commands(3)] = 96

Everything (joint order, gains, defaults, torque limits) is read from the
checkpoint's own env_args.yaml, so nothing is hand-transcribed.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import torch

from core.paths import G1_STAND_RUN_DIR, G1_URDF
from core.timing import CONTROL_HZ as ROBOT_CONTROL_HZ
from core.timing import CONTROL_PERIOD_STEPS, SCENE_DT

CFG_DIR = G1_STAND_RUN_DIR
URDF = G1_URDF

ACTION_SCALE = 0.25
DECIMATION = CONTROL_PERIOD_STEPS["g1"]
SIM_DT = SCENE_DT
CONTROL_HZ = ROBOT_CONTROL_HZ["g1"]
CONTROL_DT = SIM_DT * DECIMATION
OBS_DIM = 96
LOW_PASS_ALPHA = 0.5


# --------------------------------------------------------------------------- #
# config loading (yaml, from the checkpoint that produced the policy)
# --------------------------------------------------------------------------- #
def load_env_args(path: Path | None = None) -> dict:
    import yaml

    if path is None:
        path = CFG_DIR / "configs" / "env_args.yaml"

    class _L(yaml.SafeLoader):
        pass

    # the file carries `__tuple__` tags and python objects; keep only plain data
    _L.add_multi_constructor("tag:yaml.org,2002:python/object", lambda l, s, n: None)
    _L.add_constructor("__tuple__", lambda l, n: l.construct_sequence(n))
    with open(path) as f:
        return yaml.load(f, Loader=_L)


def _expand(per_prefix: dict, joint_names: list[str]) -> torch.Tensor:
    """Expand a {prefix: value} dict onto the ordered joint list.

    Joint names look like 'left_hip_pitch_joint'; the config keys are prefixes
    such as 'hip_pitch'. Longest matching prefix wins (so 'hip_yaw' beats
    'hip_yaw_x'-style ambiguity, and 'ankle_pitch' beats 'ankle')."""
    keys = sorted(per_prefix, key=len, reverse=True)
    out = []
    for jn in joint_names:
        hit = next((k for k in keys if k in jn), None)
        if hit is None:
            raise KeyError(f"no config entry for joint {jn!r}")
        out.append(float(per_prefix[hit]))
    return torch.tensor(out, dtype=torch.float32)


def policy_params(env_args: dict) -> dict:
    ra = env_args["robot_args"]
    jn = list(ra["dof_names"])
    assert ra.get("ctrl_type") == "DR_JOINT_POSITION_VELOCITY", ra.get("ctrl_type")
    assert int(ra.get("ctrl_freq", CONTROL_HZ)) == CONTROL_HZ
    assert int(ra.get("decimation", DECIMATION)) == DECIMATION
    return {
        "joint_names": jn,
        "default_dof_pos": _expand(ra["default_dof_pos"], jn),
        "kp": _expand(ra["dof_kp"], jn),
        "kd": _expand(ra["dof_kd"], jn),
        "tau_max": _expand(ra["dof_torque_limit"], jn),
        "ff": _expand(ra["feed_forward_ratio"], jn),
        "obs_scales": env_args.get("obs_scales", {}),
    }


# --------------------------------------------------------------------------- #
# policy
# --------------------------------------------------------------------------- #
class G1StandPolicy:
    """Loads the 4-layer actor + the final mu projection from the checkpoint."""

    def __init__(self, ckpt: Path | None = None, device="cpu"):
        ck = torch.load(ckpt or (CFG_DIR / "checkpoints" / "checkpoint_final.pt"),
                        map_location="cpu", weights_only=False)
        sd = ck["actor_state_dict"]
        self.w = {}
        for i in (0, 2, 4, 6):
            self.w[f"W{i}"] = sd[f"backbone._network.{i}.weight"].to(device)
            self.w[f"b{i}"] = sd[f"backbone._network.{i}.bias"].to(device)
        self.w["muW"] = sd["mu.weight"].to(device)
        self.w["mub"] = sd["mu.bias"].to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs
        for i in (0, 2, 4):
            x = torch.relu(x @ self.w[f"W{i}"].T + self.w[f"b{i}"])
        x = x @ self.w["W6"].T + self.w["b6"]
        return x @ self.w["muW"].T + self.w["mub"]


# --------------------------------------------------------------------------- #
# robot wrapper
# --------------------------------------------------------------------------- #
class G1Stand:
    """G1 in a Genesis 1.4.5 scene, driven exactly like the training env."""

    control_period_steps = DECIMATION      # 4 sim steps per policy step @ 200 Hz

    def __init__(self, entity, params: dict, policy: G1StandPolicy):
        self.entity = entity
        self.p = params
        self.policy = policy
        self.names = params["joint_names"]

        # map policy joint order -> simulator DOF indices
        by_name = {j.name: j for j in entity.joints if getattr(j, "type", None) != "FIXED"}
        idx = []
        for n in self.names:
            if n not in by_name:
                raise KeyError(f"joint {n!r} not found in URDF")
            j = by_name[n]
            idx.extend(j.dofs_idx_local)
        self.dof_idx = torch.tensor(idx, dtype=torch.long)
        assert len(idx) == len(self.names) == 29, f"expected 29 dofs, got {len(idx)}"

        self.default = params["default_dof_pos"]
        self.kp, self.kd, self.tau_max = params["kp"], params["kd"], params["tau_max"]
        self.ff = params["ff"]
        self.last_action = torch.zeros(29)
        self.commands = torch.zeros(3)      # standing: zero velocity command
        # velocity feed-forward state (exactly as leggedrobot_env.apply_action)
        self.last_target = torch.zeros(29)
        self.target_offset = torch.zeros(29)
        self.v_lp = torch.zeros(29)

        # settle at the default pose, zero velocity
        entity.set_dofs_position(self.default, self.dof_idx)
        entity.set_dofs_velocity(torch.zeros(29), self.dof_idx)

    # -- state ------------------------------------------------------------
    def dof_pos(self) -> torch.Tensor:
        return self.entity.get_dofs_position(self.dof_idx).flatten()

    def dof_vel(self) -> torch.Tensor:
        return self.entity.get_dofs_velocity(self.dof_idx).flatten()

    def base(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.entity.get_pos().flatten(), self.entity.get_quat().flatten())

    def obs(self) -> torch.Tensor:
        _, quat = self.base()
        g = torch.tensor([0.0, 0.0, -1.0])
        grav = _quat_apply(_quat_inv(quat), g)               # projected gravity
        ang = self.entity.get_ang().flatten()
        ang = _quat_apply(_quat_inv(quat), ang) * self.p["obs_scales"].get("base_ang_vel", 1.0)
        return torch.cat([
            self.last_action,
            self.dof_pos(),
            self.dof_vel() * self.p["obs_scales"].get("dof_vel", 1.0),
            grav,
            ang,
            self.commands,
        ])

    # -- control ----------------------------------------------------------
    def policy_step(self):
        """One 50 Hz policy step: obs -> action -> (position offset, v_ff low-pass)."""
        a = self.policy(self.obs())
        self.last_action = a.detach().clone()
        target = ACTION_SCALE * a
        v_des = (target - self.last_target) / CONTROL_DT
        self.v_lp = self.v_lp + LOW_PASS_ALPHA * (v_des - self.v_lp)
        self.last_target = target.detach().clone()
        self.target_offset = target.detach().clone()

    def apply_torque(self):
        """Recompute and apply the PD + feed-forward torque (every SIM step)."""
        force = (self.kp * (self.target_offset + self.default - self.dof_pos())
                 + self.kd * (-self.dof_vel() + self.ff * self.v_lp))
        force = torch.clamp(force, -self.tau_max, self.tau_max)
        self.entity.control_dofs_force(force, self.dof_idx)


# --------------------------------------------------------------------------- #
# quaternion helpers (torch, wxyz)
# --------------------------------------------------------------------------- #
def _quat_apply(q, v):
    w, x, y, z = q
    u = torch.stack([x, y, z])
    t = 2.0 * torch.cross(u, v, dim=0)
    return v + w * t + torch.cross(u, t, dim=0)


def _quat_inv(q):
    q = q.clone()
    q[1:] = -q[1:]
    return q
