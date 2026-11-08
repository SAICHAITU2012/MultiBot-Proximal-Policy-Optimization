"""
panda_push_env.py — Priority 1b: manipulation-aware arm training environment.

Teaches the arm what the reach policy never learned: making controlled physical
contact and MOVING a free object. Built directly on the upstream GraspEnv with
three deliberate changes:

  1. box_fixed = False          — the training box was IMMOVABLE (fixed); the
                                  policy could slam into it safely. A free box
                                  makes contact physically meaningful.
  2. push-progress reward       — per-step +x displacement of the box (the arm
                                  pushes away from itself, +x = away).
  3. contact reward             — mask-filtered ground-truth arm<->box contacts.

Everything else is upstream GraspEnv: the same arm, the same box size
(0.08 x 0.03 x 0.06), the same floor spawn (x 0.2-0.6, y +-0.25, z 0.025), the
same 14-dim observations (box position in the training world frame — the arm
base sits at the origin here, exactly like training). The reach checkpoint
warm-starts with a straight state_dict load (identical dimensions).
"""

import torch

import genesis as gs
from genesis.utils.geom import transform_by_quat, inv_quat

from grasp_env import GraspEnv


class PandaPushEnv(GraspEnv):
    def __init__(self, env_cfg, reward_cfg, robot_cfg, show_viewer=False):
        env_cfg = dict(env_cfg)
        env_cfg["box_fixed"] = False            # the entire point of 1b
        env_cfg["episode_length_s"] = 6.0       # longer than reach's 3 s
        # the floor spawn is z=0.025 — the training box (half-height 0.03) was
        # FIXED so the 5 mm floor penetration was harmless; a FREE box errors
        # the solver. Match the half-height exactly: 0.025.
        env_cfg["box_size"] = [0.08, 0.03, 0.05]
        super().__init__(env_cfg=env_cfg, reward_cfg=reward_cfg,
                         robot_cfg=robot_cfg, show_viewer=show_viewer)
        self._prev_box_x = self.object.get_pos()[:, 0].clone()

    def reset(self):
        out = super().reset()
        self._prev_box_x = self.object.get_pos()[:, 0].clone()
        return out

    def step(self, actions):
        out = super().step(actions)
        # keep prev-x in sync across per-env resets: a jump > 0.3 m is a
        # respawn teleport, not arm-induced motion
        cur = self.object.get_pos()[:, 0]
        jump = (cur - self._prev_box_x).abs() > 0.3
        self._prev_box_x[jump] = cur[jump]
        return out

    # -- rewards (registered via reward_scales keys, like upstream) ----------
    def _reward_push_progress(self):
        """Per-step +x displacement of the box (the arm pushes away from
        itself; +x is away). Teleports (respawns) contribute zero."""
        cur = self.object.get_pos()[:, 0]
        delta = cur - self._prev_box_x
        delta = torch.where(delta.abs() > 0.3, torch.zeros_like(delta), delta)
        self._prev_box_x = cur.clone()
        return delta

    def _reward_contact(self):
        """1.0 while the arm and the box are in verified contact (mask-filtered
        ground truth — the padded contact array is always non-empty)."""
        c = self.robot._robot_entity.get_contacts(with_entity=self.object)
        if c is None or "valid_mask" not in c:
            return torch.zeros(self.num_envs, dtype=gs.tc_float, device=gs.device)
        vm = c["valid_mask"].bool()  # (n_envs, n_slots)
        any_valid = vm.any(dim=1)
        # verify the pair actually involves an arm link and the box's link
        arm = self.robot._robot_entity
        arm_lo, arm_hi = arm.geom_start, arm.geom_end
        box_lo, box_hi = self.object.geom_start, self.object.geom_end
        pair_ok = torch.logical_or(
            torch.logical_and(c["geom_a"] >= arm_lo, c["geom_a"] < arm_hi),
            torch.logical_and(c["geom_b"] >= arm_lo, c["geom_b"] < arm_hi),
        )
        pair_ok = torch.logical_and(pair_ok, torch.logical_or(
            torch.logical_and(c["geom_a"] >= box_lo, c["geom_a"] < box_hi),
            torch.logical_and(c["geom_b"] >= box_lo, c["geom_b"] < box_hi),
        ))
        return (vm & pair_ok).any(dim=1).to(gs.tc_float)
