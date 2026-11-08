"""
go2_push_env.py — Go2 crate-pushing environment (RL-learned pushing).

Why this env exists: hand-crafted push controllers failed (8 iterations) because
the locomotion policy's yaw/lateral tracking error (~0.25 rad/s) makes precise
positioning around objects impractical. Contact-rich pushing is exactly the kind
of skill RL learns better than hand-coding — so we train it.

Adapted from genesis-world/examples/locomotion/go2_env.py (upstream, MIT), with:
  * a free heavy crate (0.35 x 0.35 x 0.25 m) and a goal zone
  * obs = 45 (walking obs) + 2 (crate position in base frame, xy) + 2 (goal
    minus crate, in base frame xy) = 49 dims
  * rewards: push progress (crate->goal distance decrease, dt-scaled), proximity
    bonus near the goal, plus the walking-quality penalties (upright posture,
    action smoothness, base height)
  * episode: 12 s; crate + dog reset each episode with small randomization

The dog spawns BEHIND the crate relative to the goal (the realistic mission
configuration after the fetch-navigation phase), so the policy learns the push
itself, not the approach.
"""

import math

import torch
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower


class Go2PushEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, push_cfg, show_viewer=False):
        self.num_envs: int = num_envs
        self.num_actions = env_cfg["num_actions"]
        self.cfg = env_cfg
        self.device = gs.device

        self.simulate_action_latency = True
        self.dt = 0.02  # 50 Hz control
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.push_cfg = push_cfg

        self.obs_scales: dict[str, float] = obs_cfg["obs_scales"]
        self.reward_scales: dict[str, float] = reward_cfg["reward_scales"]

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                max_collision_pairs=30,
                tolerance=1e-5,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5), camera_lookat=(0.0, 0.0, 0.5), camera_fov=40,
            ),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.env_cfg["base_init_pos"],
                quat=self.env_cfg["base_init_quat"],
            ),
        )

        # the crate to push. Mass matters: default density gives 18.4 kg vs the
        # dog's 15 kg — unbudgeable. ~7 kg post-build (set_mass; Box morph has
        # no density kwarg in v1.4.5): heavy enough that a walking gait doesn't
        # kick it flying, light enough for the dog to shove.
        self.crate = self.scene.add_entity(
            gs.morphs.Box(
                size=self.push_cfg["crate_size"],
                pos=self.push_cfg["crate_spawn"],
                fixed=False,
            ),
        )

        self.scene.build(n_envs=num_envs)

        # v1.4.5 has no morph-level density: set the mass after build
        self.crate.set_mass(self.push_cfg["crate_mass"])

        # names to indices (same as Go2Env)
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_int, device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)

        # initial states
        self.init_base_pos = torch.tensor(self.env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(self.env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        self.init_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][joint.name] for joint in self.robot.joints[1:]],
            dtype=gs.tc_float, device=gs.device,
        )
        self.init_qpos = torch.concatenate((self.init_base_pos, self.init_base_quat, self.init_dof_pos))
        self.init_projected_gravity = transform_by_quat(self.global_gravity, self.inv_base_init_quat)

        # crate/goal geometry. s6+: the GOAL direction is randomized per episode
        # (radius 2.2 +-0.3 around the crate, uniform angle) and the dog spawns
        # behind the crate RELATIVE to that goal — the policy learns to push
        # toward goals in any direction, not just dead ahead.
        self.crate_half = self.push_cfg["crate_size"][0] / 2
        self.crate_spawn = torch.tensor(self.push_cfg["crate_spawn"], dtype=gs.tc_float, device=gs.device)
        self.crate_init_qpos = torch.concatenate((
            self.crate_spawn.clone(),
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=gs.tc_float, device=gs.device),
        ))
        self.goal_radius = self.push_cfg.get("goal_radius", 2.2)

        # per-env goal buffer (resampled each reset)
        self.goal_xy = torch.empty((self.num_envs, 2), dtype=gs.tc_float, device=gs.device)

        # buffers
        self.base_lin_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_ang_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.projected_gravity = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.rew_buf = torch.empty((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.empty((self.num_envs,), dtype=gs.tc_int, device=gs.device)
        # fixed gentle forward command (walking-policy compatible; keeps the
        # dog walking into the crate so push shaping has signal from step 1)
        self.commands = torch.tensor([[0.3, 0.0, 0.0]], dtype=gs.tc_float, device=gs.device).repeat(
            self.num_envs, 1
        )
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device, dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.empty_like(self.actions)
        self.dof_vel = torch.empty_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_quat = torch.empty((self.num_envs, 4), dtype=gs.tc_float, device=gs.device)
        self.base_euler = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_float, device=gs.device,
        )
        self.crate_pos = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.crate_quat = torch.empty((self.num_envs, 4), dtype=gs.tc_float, device=gs.device)
        self.crate_dist = torch.empty((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.dog_crate_dist = torch.empty((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.extras = dict()

        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)

        self.reset()

    # crate -> goal horizontal distance
    def _crate_goal_dist(self):
        d = self.crate_pos[:, :2] - self.goal_xy
        return torch.norm(d, dim=1)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos[:, self.actions_dof_idx], slice(6, 18))
        self.scene.step()

        self.episode_length_buf += 1
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)
        self.crate_pos = self.crate.get_pos()
        self.crate_quat = self.crate.get_quat()

        # push-progress + approach bookkeeping (after the step)
        new_dist = self._crate_goal_dist()
        self.push_progress = self.crate_dist - new_dist  # >0 = moved toward goal
        self.crate_dist = new_dist
        new_dog_d = torch.norm(self.crate_pos[:, :2] - self.base_pos[:, :2], dim=1)
        self.approach_progress = self.dog_crate_dist - new_dog_d  # >0 = got closer
        self.dog_crate_dist = new_dog_d

        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= self.scene.get_error_envs_mask()
        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)
        # explicit fall penalty: without it, "stand still" (zero penalties) beats
        # "run and sometimes fall" in expected return, and PPO kills the walk
        self.rew_buf -= self.env_cfg["fall_penalty"] * self.reset_buf.float() * (1.0 - self.extras["time_outs"])

        self._reset_idx(self.reset_buf)
        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def _reset_idx(self, envs_idx=None):
        # BUG FIX (2026-11-08, root cause of the M0/M0b degenerate runs): two
        # defects entered with the s6-batch-spawn rewrite.
        # (1) An empty per-step reset called set_qpos(zero_velocity=True) on an
        #     empty envs_idx EVERY step, which zeroes ALL the robot's
        #     velocities in genesis 1.4.5 — the gait could never accumulate
        #     momentum (measured: v_xy == 0.0000 exactly, treadmill legs).
        # (2) The dog spawned at base z=0.0 (inside the floor) instead of the
        #     upstream standing height (go2_env spawns at init_base_pos z=0.42),
        #     so every episode began with a PD leg-extension launch and a
        #     pitch-termination at step ~5 — a 5-step toss-loop.
        # Any training/eval after the rewrite ran in this broken env; s7
        # session 1 (model_577, pre-rewrite) trained in the healthy env and its
        # pre-rewrite sweep (16/16 moved, 6/16 pushed) reproduces EXACTLY in
        # the fixed env.
        if envs_idx is not None and int(envs_idx.sum()) == 0:
            return
        # per-env randomized goal + matching dog spawn (behind crate, facing it).
        # s6 post-mortem: full-360 goal angles + a spawn bug (dog_q[0] applied to
        # ALL envs) made ~85% of envs unsolvable — zero learning in 1200 iters.
        # Fix: batch-correct spawns + a +-75 deg approach band matching the
        # mission's actual retry variation.
        n = self.num_envs if envs_idx is None else int(envs_idx.sum())
        theta = (torch.rand(n, device=gs.device) - 0.5) * (150.0 * math.pi / 180.0)
        goal = self.crate_spawn[None, :2] + (self.goal_radius + 0.3 * (torch.rand(n, 2, device=gs.device) - 0.5) * 2) * torch.stack(
            [torch.cos(theta), torch.sin(theta)], dim=1
        )
        dog_xy = self.crate_spawn[None, :2] - 1.0 * torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        dog_yaw = theta  # facing from behind the crate toward it
        dog_quat = torch.stack([
            torch.cos(dog_yaw / 2), torch.zeros_like(dog_yaw),
            torch.zeros_like(dog_yaw), torch.sin(dog_yaw / 2),
        ], dim=1)
        dog_z = torch.full((n, 1), float(self.init_base_pos[2]), device=gs.device)  # upstream standing height
        dog_q = torch.cat([dog_xy, dog_z, dog_quat,
                           self.init_dof_pos.repeat(n, 1)], dim=1)

        if envs_idx is None:
            self.goal_xy.copy_(goal)
            self.robot.set_qpos(dog_q, zero_velocity=True)  # BATCH: per-env spawns
        else:
            self.goal_xy[envs_idx] = goal
            self.robot.set_qpos(dog_q, envs_idx=envs_idx, zero_velocity=True)

        # crate reset with small randomization (position +-0.15 m, yaw +-0.4)
        if n > 0:
            crate_q = self.crate_init_qpos.repeat(n, 1)
            crate_q[:, :2] += gs_rand(
                torch.tensor([-0.15, -0.15], device=gs.device),
                torch.tensor([0.15, 0.15], device=gs.device),
                (n,),
            )
            yaw = gs_rand(
                torch.tensor([-0.4], device=gs.device), torch.tensor([0.4], device=gs.device), (n,)
            ).squeeze(-1)
            crate_q[:, 3] = torch.cos(yaw / 2)
            crate_q[:, 6] = torch.sin(yaw / 2)
            self.crate.set_qpos(crate_q, envs_idx=envs_idx, zero_velocity=True)

        # refresh ALL derived state from the actual per-env spawns (FK ran via
        # set_qpos without skip_forward); velocities start at zero
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_quat(self.base_quat))
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

        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            if envs_idx is None:
                value.zero_()
            else:
                value.masked_fill_(envs_idx, 0.0)

    def _update_observation(self):
        # crate position + goal direction in the ROBOT's base frame (xy only)
        inv_base_quat = inv_quat(self.base_quat)
        crate_rel = self.crate_pos[:, :2] - self.base_pos[:, :2]
        crate_rel_base = transform_by_quat(
            torch.cat([crate_rel, torch.zeros((self.num_envs, 1), device=gs.device)], dim=-1),
            inv_base_quat,
        )[:, :2]
        goal_rel = self.goal_xy - self.crate_pos[:, :2]
        goal_rel_base = transform_by_quat(
            torch.cat([goal_rel, torch.zeros((self.num_envs, 1), device=gs.device)], dim=-1),
            inv_base_quat,
        )[:, :2]
        self.obs_buf = torch.concatenate(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],                       # 3
                self.projected_gravity,                                               # 3
                self.commands * self.commands_scale,                                  # 3  (walking obs compat)
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],   # 12
                self.dof_vel * self.obs_scales["dof_vel"],                            # 12
                self.actions,                                                         # 12
                crate_rel_base * self.obs_scales["crate_pos"],                        # 2
                goal_rel_base * self.obs_scales["crate_pos"],                         # 2
            ),
            dim=-1,
        )
        # 49 dims: the walking policy's 45 + 4 push dims. Commands are FIXED at
        # a gentle forward pace (0.3 m/s) so a warm-started walking policy is
        # in-distribution and immediately contacts the crate.

    def reset(self):
        self._reset_idx()
        self._update_observation()
        return self.get_observations()

    # -- rewards -----------------------------------------------------------
    def _reward_push_progress(self):
        # per-step distance reduction crate->goal (positive = pushed toward goal)
        return self.push_progress * self.reward_cfg["progress_scale"]

    def _reward_approach(self):
        # shaping: per-step reduction of dog->crate distance (guides the dog to
        # contact; once in contact, push_progress dominates)
        return self.approach_progress * self.reward_cfg["approach_scale"]

    def _reward_proximity_bonus(self):
        # dense shaping near the goal so the final meters still pay off
        return torch.exp(-self.crate_dist / self.reward_cfg["proximity_sigma"])

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
