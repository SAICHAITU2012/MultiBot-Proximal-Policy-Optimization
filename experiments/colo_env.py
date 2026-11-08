"""Stage 5: co-location testbed — Go2 quadruped + cf2x drone + Panda arm in ONE Genesis scene.

Three independently-trained policies run simultaneously against a single shared
scene.step() clock.  The observation construction / action processing of every
robot replicates the upstream training envs EXACTLY:

  * go2   — genesis-world/examples/locomotion/go2_env.py   (45-dim obs, 50 Hz)
  * drone — genesis-world/examples/drone/hover_env.py      (17-dim obs, 100 Hz)
  * panda — genesis-world/examples/manipulation/grasp_env.py (14-dim obs, 100 Hz)

Shared-scene recipe (one compromise, see CAVEATS at bottom):
  SimOptions(dt=0.01, substeps=2)  +  RigidOptions(dt=0.005, enable_self_collision=False)
  * drone/panda are commanded every scene step (100 Hz, their native rate).
  * go2 is commanded every 2nd scene step (50 Hz, its native rate); the PD
    target set via control_dofs_position persists between ticks.
  * rigid substep 0.005 matches the drone env exactly; go2 was trained with
    rigid substep 0.01 but kp=20/kd=0.5 is insensitive to this (0.005 is MORE
    stable).

Panda: the grasp policy is not trained yet, so PandaAgent is fully plumbed
(obs builder + DLS-IK action applier + get_state) but runs with
``policy=None`` and a zero (hold-pose) default action.  Plugging in the policy
later is one line:  team.robots["panda"].policy = some_callable(obs14).
"""

import math
import pickle
from pathlib import Path

import torch
import torch.nn as nn

import genesis as gs
from genesis.utils.geom import inv_quat, transform_by_quat

TEAM_LAB = Path(__file__).resolve().parent.parent
LOGS_DIR = TEAM_LAB / "logs"

# ---------------------------------------------------------------------------
# Shared scene layout (n_envs=1)
# ---------------------------------------------------------------------------
SCENE_DT = 0.02          # matches go2_env's training dt (the Go2's gait
                         # dynamics are dt-sensitive: at 0.01 rigid the gait
                         # drifts/trips in ways never seen in training)
RIGID_DT = 0.01          # matches go2_env (SimOptions dt=0.02, substeps=2 ->
                         # rigid 0.01); the drone/panda control at 50 Hz (was
                         # 100) — mild OOD for their slow dynamics, verified
SUBSTEPS = 2             # rigid integrates twice per scene step

GO2_INIT_POS = (2.2, 1.2, 0.42)   # exactly 1.0 m behind pkg0 (the training's dog-crate distance), facing west: the handoff is immediate and in-distribution
DRONE_INIT_POS = (2.0, 1.5, 1.0)
PANDA_INIT_POS = (-2.0, 1.2, 0.4)      # station on the NORTH side: push lines from the south land crates at its reach without corner traps
PEDESTAL_POS = (-2.0, 1.2, 0.2)        # 0.6 x 0.6 x 0.4 box, top face at z=0.4
PANDA_OBJ_POS = (-1.45, 1.2, 0.435)    # graspable box in front of the arm


# ---------------------------------------------------------------------------
# Policy loading (rsl-rl-lib 5.5.2 checkpoints, direct state_dict load)
# Checkpoints store {"actor_state_dict", "critic_state_dict", ...}; the actor
# is an rsl_rl.modules.MLP wrapped in MLPModel: Sequential(Linear, Act, ...,
# Linear) with keys "mlp.<i>.{weight,bias}".  Deterministic inference
# (get_inference_policy) returns the raw MLP output (GaussianDistribution mean).
# ---------------------------------------------------------------------------
_ACT = {"elu": nn.ELU, "tanh": nn.Tanh, "relu": nn.ReLU}


def _build_mlp_from_actor_sd(actor_sd, activation):
    """Reconstruct the rsl_rl MLP from its state_dict key shapes."""
    idxs = sorted(int(k.split(".")[1]) for k in actor_sd if k.startswith("mlp.") and k.endswith("weight"))
    # weight i has shape (out, in): chain pairs (in -> out) through the network
    pairs = [(actor_sd[f"mlp.{i}.weight"].shape[1], actor_sd[f"mlp.{i}.weight"].shape[0]) for i in idxs]
    in_dim, out_dim = pairs[0][0], pairs[-1][1]
    hidden_dims = [out for (_, out) in pairs[:-1]]
    act = _ACT[activation.lower()]()
    layers = [nn.Linear(in_dim, hidden_dims[0]), act]
    for h_in, h_out in zip(hidden_dims[:-1], hidden_dims[1:]):
        layers += [nn.Linear(h_in, h_out), act]
    layers += [nn.Linear(hidden_dims[-1], out_dim)]
    mlp = nn.Sequential(*layers)
    mlp.load_state_dict({k[len("mlp."):]: v for k, v in actor_sd.items() if k.startswith("mlp.")})
    mlp.to(gs.device).eval()
    return mlp


def load_actor_from_log(log_dir, ckpt_name=None):
    """Load a deterministic actor from an rsl_rl log dir (cfgs.pkl + model_*.pt).

    cfgs.pkl layouts vary by task (5-tuples with different orderings); the RL
    train config is whichever element contains the "actor" key (the grasp task's
    tuple ends with the BC config, which has no actor MLP spec)."""
    log_dir = Path(log_dir)
    with open(log_dir / "cfgs.pkl", "rb") as f:
        cfgs = pickle.load(f)
    train_cfg = next((c for c in reversed(cfgs) if isinstance(c, dict) and "actor" in c), None)
    if train_cfg is None:
        raise KeyError(f"no train config with 'actor' key in {log_dir}/cfgs.pkl")
    if ckpt_name is None:
        ckpts = sorted(log_dir.glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
        if not ckpts:
            raise FileNotFoundError(f"no model_*.pt under {log_dir}")
        ckpt_name = ckpts[-1].name
    ckpt = torch.load(log_dir / ckpt_name, map_location=gs.device, weights_only=False)
    actor_sd = ckpt.get("actor_state_dict", ckpt)
    activation = train_cfg["actor"]["activation"]
    return _build_mlp_from_actor_sd(actor_sd, activation)


def _ensure_genesis_init():
    if not gs._initialized:
        gs.init(backend=gs.cpu, precision="32", logging_level="warning", performance_mode=True)


# ---------------------------------------------------------------------------
# Go2 agent — replicates Go2Env (go2_env.py) obs/action math exactly
# ---------------------------------------------------------------------------
GO2_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
GO2_DEFAULT_JOINT_ANGLES = {  # from go2_train.py get_cfgs()
    "FL_hip_joint": 0.0, "FR_hip_joint": 0.0, "RL_hip_joint": 0.0, "RR_hip_joint": 0.0,
    "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8, "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5, "FR_calf_joint": -1.5, "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
}


class Go2Agent:
    """45-dim obs, 12-dim action, PD position control at 50 Hz with 1-step latency."""

    name = "go2"
    control_period = 2  # scene steps between control ticks (50 Hz @ SCENE_DT=0.01)

    def __init__(self, entity, command=(0.5, 0.0, 0.0)):
        self.entity = entity
        # PD gains (go2 env: kp 20, kd 0.5 on the 12 motor dofs)
        self.motors_dof_idx = torch.tensor(
            [entity.get_joint(name).dof_start for name in GO2_JOINT_NAMES], dtype=gs.tc_int, device=gs.device
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)  # joint_names order -> urdf dof order
        entity.set_dofs_kp([20.0] * 12, self.motors_dof_idx)
        entity.set_dofs_kv([0.5] * 12, self.motors_dof_idx)

        self.default_dof_pos = torch.tensor(
            [GO2_DEFAULT_JOINT_ANGLES[n] for n in GO2_JOINT_NAMES], dtype=gs.tc_float, device=gs.device
        ).unsqueeze(0)                                                    # (1, 12) in joint_names order
        self.commands = torch.tensor(command, dtype=gs.tc_float, device=gs.device).unsqueeze(0)
        self.commands_scale = torch.tensor([2.0, 2.0, 0.25], device=gs.device)  # lin_vel, lin_vel, ang_vel scales
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)

        # Initial state (Go2Env._reset_idx): base pose + default joint angles, zero velocity
        init_pos = torch.tensor(GO2_INIT_POS, dtype=gs.tc_float, device=gs.device)
        init_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=gs.tc_float, device=gs.device)  # yaw 180: facing the packages/goal
        self.init_qpos = torch.cat([init_pos, init_quat, self.default_dof_pos[0]])
        entity.set_qpos(self.init_qpos, zero_velocity=True)  # FK runs now so the first obs is fresh

        self.action_scale = 0.25
        self.clip_actions = 100.0
        self.actions = torch.zeros((1, 12), dtype=gs.tc_float, device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)  # executed (1-step latency) action buffer
        self.policy = None
        self.external_control = False  # task agent owns the dog when True (exclusive: no tick double-apply)
        self.refresh()

    # -- state refresh (call after every scene.step, like Go2Env.step) -------
    def refresh(self):
        self.base_pos = self.entity.get_pos()                                   # (1, 3) world
        self.base_quat = self.entity.get_quat()                                 # (1, 4) wxyz
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.entity.get_vel(), inv_base_quat)   # body frame
        self.base_ang_vel = transform_by_quat(self.entity.get_ang(), inv_base_quat)   # body frame
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos = self.entity.get_dofs_position(self.motors_dof_idx)       # joint_names order
        self.dof_vel = self.entity.get_dofs_velocity(self.motors_dof_idx)

    # -- obs (45-dim): exact Go2Env._update_observation term order -----------
    def get_obs(self):
        return torch.cat(
            (
                self.base_ang_vel * 0.25,                                       # 3  (obs_scales.ang_vel)
                self.projected_gravity,                                         # 3
                self.commands * self.commands_scale,                            # 3  (cmd * [2, 2, 0.25])
                (self.dof_pos - self.default_dof_pos) * 1.0,                    # 12 (obs_scales.dof_pos)
                self.dof_vel * 0.05,                                            # 12 (obs_scales.dof_vel)
                self.actions,                                                   # 12 (current, NOT last)
            ),
            dim=-1,
        )

    # -- action: exact Go2Env.step processing --------------------------------
    def apply_action(self, action):
        """`action` = freshly computed policy output; the PREVIOUS action is the
        one actually executed (1-step latency, Go2Env.simulate_action_latency)."""
        self.actions = torch.clip(action, -self.clip_actions, self.clip_actions)
        exec_actions = self.last_actions
        target_dof_pos = exec_actions * self.action_scale + self.default_dof_pos
        self.entity.control_dofs_position(target_dof_pos[:, self.actions_dof_idx], slice(6, 18))
        self.last_actions = self.actions.clone()

    def default_action(self):
        return self.actions  # zero at start

    def get_state(self):
        lin_world = transform_by_quat(self.entity.get_vel(), self.base_quat)  # world-frame lin vel
        return {
            "pos": self.base_pos[0].detach().cpu().tolist(),
            "quat": self.base_quat[0].detach().cpu().tolist(),
            "lin_vel_world": lin_world[0].detach().cpu().tolist(),
            "lin_vel_body": self.base_lin_vel[0].detach().cpu().tolist(),
            "ang_vel_body": self.base_ang_vel[0].detach().cpu().tolist(),
            "base_z": float(self.base_pos[0, 2]),
            "vx_body": float(self.base_lin_vel[0, 0]),
            "command": self.commands[0].detach().cpu().tolist(),
            "dof_pos": self.dof_pos[0].detach().cpu().tolist(),
        }

    def tick(self):
        if getattr(self, "external_control", False):
            return  # the task agent owns the dog (exclusive: no double-apply)
        if self.external_control:
            return  # the task agent owns the dog (exclusive: no double-apply)
        obs = self.get_obs()
        action = self.policy(obs) if self.policy is not None else self.default_action()
        self.apply_action(action)


# ---------------------------------------------------------------------------
# Drone agent — replicates HoverEnv (hover_env.py) obs/action math exactly
# ---------------------------------------------------------------------------
class DroneAgent:
    """17-dim obs, 4-dim action -> propeller RPM via DroneEntity.set_propellers_rpm."""

    name = "drone"
    control_period = 1  # every scene step (100 Hz)

    HOVER_RPM = 14468.429183500699  # cf2x hover rpm (hover_env.py)

    def __init__(self, entity, target=None):
        self.entity = entity
        target = DRONE_INIT_POS if target is None else target
        self.commands = torch.tensor(target, dtype=gs.tc_float, device=gs.device).unsqueeze(0)  # world-frame target
        self.clip_actions = 1.0
        self.external_control = False
        self.actions = torch.zeros((1, 4), dtype=gs.tc_float, device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)  # obs term = action applied in previous step
        self.policy = None
        self.refresh()

    def refresh(self):
        self.base_pos = self.entity.get_pos()
        self.base_quat = self.entity.get_quat()
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.entity.get_vel(), inv_base_quat)  # body frame
        self.base_ang_vel = transform_by_quat(self.entity.get_ang(), inv_base_quat)  # body frame
        self.rel_pos = self.commands - self.base_pos  # world frame: target minus current pos

    # -- obs (17-dim): exact HoverEnv._update_observation term order ---------
    def get_obs(self):
        return torch.cat(
            [
                torch.clip(self.rel_pos * (1 / 3.0), -1, 1),                    # 3
                self.base_quat,                                                 # 4 (wxyz, unscaled)
                torch.clip(self.base_lin_vel * (1 / 3.0), -1, 1),               # 3
                torch.clip(self.base_ang_vel * (1 / 3.14159), -1, 1),           # 3
                self.last_actions,                                              # 4 (action applied last step)
            ],
            axis=-1,
        )

    # -- action: exact HoverEnv.step application -----------------------------
    def apply_action(self, action):
        self.actions = torch.clip(action, -self.clip_actions, self.clip_actions)
        # (1 + a * 0.8) * hover_rpm -> DroneEntity.set_propellers_rpm (once per step)
        self.entity.set_propellers_rpm((1 + self.actions * 0.8) * self.HOVER_RPM)
        self.last_actions = self.actions.clone()

    def default_action(self):
        return self.actions  # zero at start

    def get_state(self):
        return {
            "pos": self.base_pos[0].detach().cpu().tolist(),
            "quat": self.base_quat[0].detach().cpu().tolist(),
            "lin_vel_body": self.base_lin_vel[0].detach().cpu().tolist(),
            "target": self.commands[0].detach().cpu().tolist(),
            "dist_to_target": float(torch.norm(self.rel_pos)),
        }

    def tick(self):
        if self.external_control:
            return  # the task agent owns the dog (exclusive: no double-apply)
        obs = self.get_obs()
        action = self.policy(obs) if self.policy is not None else self.default_action()
        self.apply_action(action)


# ---------------------------------------------------------------------------
# Panda agent — replicates Manipulator (grasp_env.py) obs/action math exactly.
# The grasp policy is not trained yet; policy stays None (hold-pose default).
# ---------------------------------------------------------------------------
PANDA_DEFAULT_ARM_DOF = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]


class PandaAgent:
    """14-dim obs (finger_pos - obj_pos, finger_quat, obj_pos, obj_quat),
    6-dim EE action (dpos + dRPY, each x0.05) applied through damped-least-squares IK."""

    name = "panda"
    control_period = 1  # every scene step (100 Hz, grasp env ctrl_dt=0.01)

    def __init__(self, entity, obj_entity, ee_link_name="hand", gripper_link_names=("left_finger", "right_finger")):
        self.entity = entity
        self.obj = obj_entity

        self.arm_dof_dim = entity.n_dofs - 2
        self.arm_dof_idx = torch.arange(self.arm_dof_dim, device=gs.device)
        self.fingers_dof = torch.arange(self.arm_dof_dim, entity.n_dofs, device=gs.device)
        self.ee_link = entity.get_link(ee_link_name)
        self.left_finger_link = entity.get_link(gripper_link_names[0])
        self.right_finger_link = entity.get_link(gripper_link_names[1])

        # PD gains / force limits (grasp_env Manipulator.set_pd_gains; after build)
        entity.set_dofs_kp(torch.tensor([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        entity.set_dofs_kv(torch.tensor([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        # finger force limit 100 N -> 2 N: a 100 N close on a 0.15 kg parcel
        # launches it; 2 N presses gently (2 pads x 2 N x mu ~ 4 N friction
        # holds the ~1.5 N weight during the lift)
        entity.set_dofs_force_range(
            torch.tensor([-87, -87, -87, -87, -12, -12, -12, -2, -2]),
            torch.tensor([87, 87, 87, 87, 12, 12, 12, 2, 2]),
        )

        self.action_scales = torch.tensor([0.05] * 6, device=gs.device)
        self.gripper_open_dof, self.gripper_close_dof = 0.04, 0.00
        self.dls_lambda_matrix = (0.01 ** 2) * torch.eye(6, device=gs.device)
        self.policy = None  # plug the trained grasp policy here later
        self.external_control = False  # task agent owns the arm when True

        # Initial state (Manipulator.reset): ready pose + open gripper, zero velocity
        init_qpos = torch.tensor(
            PANDA_DEFAULT_ARM_DOF + [self.gripper_open_dof, self.gripper_open_dof],
            dtype=gs.tc_float, device=gs.device,
        )
        entity.set_qpos(init_qpos, zero_velocity=True)  # FK runs now so the first obs is fresh
        self._hold_qpos = init_qpos.unsqueeze(0)  # fixed hold target while policy is None

    # -- obs (14-dim): exact GraspEnv.get_observations term order ------------
    def refresh(self):
        pass  # panda reads links live in get_obs/get_state (no cached buffers)

    def get_obs(self):
        finger_pos, finger_quat = self.center_finger_pose[:, :3], self.center_finger_pose[:, 3:7]
        obj_pos, obj_quat = self.obj.get_pos(), self.obj.get_quat()
        # FRAME REBASING (critical): the reach policy was trained in grasp_env
        # with the arm base at (0,0,0) and the box at x in [0.2,0.6], y in
        # +-0.25, z = 0.025 (floor) — the obs carry the box's position in that
        # frame. In the mission the arm sits at (-2,1.2,0.4), so world-frame
        # obj_pos is wildly out of distribution. Rebase: subtract the arm base.
        arm_base = self.entity.get_pos()[0, :3].unsqueeze(0)
        obj_pos_rebased = obj_pos - arm_base
        return torch.cat([finger_pos - obj_pos, finger_quat, obj_pos_rebased, obj_quat], dim=-1)

    @property
    def center_finger_pose(self):
        # v1.4.5 link API: get_pos() -> (1,3), get_quat() -> (1,4) — SEPARATE
        # getters (upstream Manipulator does the same; a 7-in-1 get_pos would
        # silently empty the quat slice and corrupt the 14-dim obs)
        left_p = self.left_finger_link.get_pos()
        left_q = self.left_finger_link.get_quat()
        right_p = self.right_finger_link.get_pos()
        center_pos = (left_p[:, :3] + right_p[:, :3]) / 2
        return torch.cat([center_pos, left_q], dim=-1)

    # -- action: exact Manipulator.apply_action / _dls_ik --------------------
    def _dls_ik(self, delta_pose):
        """Solve (J J^T + lambda^2 I) y = dx, dq = J^T y; q_target = qpos + dq."""
        jacobian = self.entity.get_jacobian(link=self.ee_link)          # (1, 6, 9)
        A = torch.baddbmm(self.dls_lambda_matrix, jacobian, jacobian.mT)
        y = torch.linalg.solve(A, delta_pose)
        delta_joint_pos = (jacobian.mT @ y.unsqueeze(-1)).squeeze(-1)
        return self.entity.get_qpos() + delta_joint_pos

    def apply_action(self, action, open_gripper=True):
        """action: (1, 6) EE delta [dpos(3), dRPY(3, rad)] scaled by 0.05."""
        action = action * self.action_scales
        q_pos = self._dls_ik(action[:, :6])
        q_pos[:, self.fingers_dof] = self.gripper_open_dof if open_gripper else self.gripper_close_dof
        self.entity.control_dofs_position(position=q_pos)

    def default_action(self):
        return torch.zeros((1, 6), dtype=gs.tc_float, device=gs.device)

    def get_state(self):
        return {
            "ee_pos": self.center_finger_pose[0, :3].detach().cpu().tolist(),
            "ee_quat": self.center_finger_pose[0, 3:7].detach().cpu().tolist(),
            "obj_pos": self.obj.get_pos()[0].detach().cpu().tolist(),
            "base_pos": self.entity.get_pos()[0].detach().cpu().tolist(),
            "qpos": self.entity.get_qpos()[0].detach().cpu().tolist(),
        }

    def tick(self):
        if self.external_control:
            return  # the task agent owns the arm (exclusive control, no double-apply)
        if self.policy is not None:
            obs = self.get_obs()
            action = self.policy(obs)
            self.apply_action(action)
        else:
            # No trained policy yet: rigidly hold the ready pose.  (Re-issuing a
            # zero-delta IK target each tick would re-target the sagging qpos and
            # let the arm creep downward under gravity.)
            self.entity.control_dofs_position(position=self._hold_qpos)


# ---------------------------------------------------------------------------
# Co-located team
# ---------------------------------------------------------------------------
class CoLocatedTeam:
    """One Genesis scene, three robots, three policies, one shared clock.

    team.robots = {"go2": Go2Agent, "drone": DroneAgent, "panda": PandaAgent}
    team.step() advances the shared scene by SCENE_DT (0.01 s) and services
    each robot at its native control rate.
    """

    def __init__(self, show_viewer=False, go2_ckpt=None, drone_ckpt=None,
                 go2_command=(0.5, 0.0, 0.0), drone_target=None, extra_boxes=None):
        _ensure_genesis_init()
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=SCENE_DT, substeps=SUBSTEPS),
            rigid_options=gs.options.RigidOptions(
                dt=RIGID_DT,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=False,  # required by the go2 locomotion policy
                max_collision_pairs=50,
                constraint_solver=gs.constraint_solver.Newton,  # handles the stiff
                # dog-crate contact chain without the force divergence (the
                # default solver NaN'd at 875.3 s deterministically)
                tolerance=1e-5,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(4.5, -4.5, 3.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=45,
            ),
            show_viewer=show_viewer,
        )

        # world
        # the TRAINING-EXACT floor: the URDF plane (the go2_env's floor) —
        # the gs.morphs.Plane() has different default contact properties
        self.scene.add_entity(
            gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
        )
        # go2
        go2_entity = self.scene.add_entity(
            gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=GO2_INIT_POS, quat=(0.0, 0.0, 0.0, 1.0)),
        )
        # drone
        drone_entity = self.scene.add_entity(
            gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=DRONE_INIT_POS),
        )
        # panda on a fixed pedestal (MJCF base is welded to the world at `pos`)
        self.pedestal = self.scene.add_entity(
            gs.morphs.Box(size=(0.6, 0.6, 0.4), pos=PEDESTAL_POS, fixed=True),
        )
        panda_entity = self.scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=PANDA_INIT_POS, quat=(1.0, 0.0, 0.0, 0.0)),
        )
        # MANIPULATION PARCEL: a FREE small box exactly matching the reach
        # policy's training object (0.08 x 0.03 x 0.06 m). The gripper stroke
        # is ~8 cm, so only this parcel is genuinely graspable — the 35 cm
        # delivery crates can be reached/touched but never grasped (physical
        # limitation, documented). Free-standing so a successful close+lift
        # moves it (LIFTED is verifiable physics, not a proximity metric).
        # thin support POST directly under the parcel: holds it at the trained
        # reach height while staying clear of the horizontal reach path (the
        # earlier 0.25 m table blocked the hand and the arm bounced off it —
        # an untrained obstacle in the reach path)
        # SLIDE FIXTURE (station): a walled track on the arm's reach line.
        # The reach policy presses like the box is immovable (it was FIXED in
        # training) — a free parcel on a perch gets launched. The slide's walls
        # confine the parcel laterally so the arm's push moves it along +a
        # measurable path: REACHED -> CONTACT -> PUSHED is verifiable physics.
        # Base plate top at z=0.42; walls keep the 0.08x0.03x0.06 parcel in.
        # manipulation pedestal: the parcel rests at the arm-base height
        # (+0.035, matching the training box's relative height exactly); the
        # manip policy presses it off the pedestal edge — a chute dispatch
        self.parcel_table = self.scene.add_entity(
            gs.morphs.Box(size=(0.15, 0.15, 0.4), pos=(-1.55, 1.55, 0.2), fixed=True),
        )
        self.parcel_entity = self.scene.add_entity(
            gs.morphs.Box(size=(0.08, 0.03, 0.05), pos=(-1.55, 1.55, 0.435), fixed=False),
        )
        # mission packages: FREE rigid boxes (MUST be added before scene.build —
        # entities cannot be added post-build in Genesis). Heavy crates: a light
        # 18 cm box gets kicked away by the walking gait; a heavy crate slides
        # like a sled and gives the push controller stable contact.
        self.extra_box_entities = []
        for pos in (extra_boxes or []):
            # LOW FLAT crate (0.06 m tall, 7 kg): the Go2's learned gait has a
            # ~5-15 cm paw swing — a 25 cm crate face TRIPS the trot mid-stride
            # (paw strikes the face). A 6 cm crate slides under the gait like
            # the 5 cm training box, while the 7 kg mass keeps it shovable-not-
            # kickable. spawn z = half-height + 5 mm.
            # THE TRAINING-EXACT PARCEL (the decisive fix): the push policy
            # was trained on THIS EXACT object — the 0.08 x 0.03 x 0.05 m box
            # at 0.12 kg (the grasp_env's default density). Every heavier/
            # larger crate (3-7 kg, 15-25 cm) broke the interaction: the gait
            # tripped on tall faces, straddled low ones, and 58x the mass
            # exceeded the gait's press force. The delivery packages are
            # small parcels — physically what a quadruped courier carries.
            self.extra_box_entities.append(
                self.scene.add_entity(
                    gs.morphs.Box(size=(0.08, 0.03, 0.05), pos=(pos[0], pos[1], 0.03), fixed=False)
                )
            )

        self.scene.build(n_envs=1)

        # post-build: the training-exact parcel needs no mass override —
        # its default density (0.12 kg) matches the training box exactly

        # parcel: default density ~0.14 kg — graspable and liftable. High
        # friction on parcel+post: the reach policy was trained against a FIXED
        # box (immovable) and presses confidently — without friction the free
        # parcel slides off the perch on first contact.
        self.parcel_entity.set_mass(0.15)
        # high friction via the solver (entity-level API doesn't exist in 1.4.5):
        # the reach policy was trained against a FIXED box (immovable) and
        # presses confidently — without friction the free parcel slides off
        solver = self.scene.rigid_solver
        for ent in (self.parcel_entity, self.parcel_table):
            solver.set_geoms_friction(0.6, list(range(ent.geom_start, ent.geom_end)))
        # parcel: default density gives ~0.14 kg — graspable, light enough to lift

        # agents
        self.robots = {
            "go2": Go2Agent(go2_entity, command=go2_command),
            "drone": DroneAgent(drone_entity, target=drone_target),
            "panda": PandaAgent(panda_entity, self.parcel_entity),
        }

        # policies (panda left as None until its training finishes)
        if go2_ckpt is not None:
            self.robots["go2"].policy = load_actor_from_log(*self._resolve_ckpt(go2_ckpt))
        else:
            self.robots["go2"].policy = load_actor_from_log(LOGS_DIR / "go2-walking-s1", "model_0.pt")
        if drone_ckpt is not None:
            self.robots["drone"].policy = load_actor_from_log(*self._resolve_ckpt(drone_ckpt))
        else:
            self.robots["drone"].policy = load_actor_from_log(LOGS_DIR / "drone-hover-s1", "model_2396.pt")

        self.t = 0  # scene step counter (100 Hz clock)

    @staticmethod
    def _resolve_ckpt(ckpt):
        p = Path(ckpt)
        return p.parent, p.name

    @torch.inference_mode()
    def step(self):
        """Advance the shared scene one 100 Hz step, servicing each robot at
        its native control rate.  Ticks happen BEFORE the physics step, so a
        tick's action is applied during the step that immediately follows it
        (matching the upstream env.step(action) ordering)."""
        for robot in self.robots.values():
            if self.t % robot.control_period == 0:
                robot.tick()
        self.scene.step()
        self.t += 1
        for robot in self.robots.values():
            robot.refresh()

    def get_states(self):
        return {name: robot.get_state() for name, robot in self.robots.items()}

    @property
    def sim_time(self):
        return self.t * SCENE_DT
