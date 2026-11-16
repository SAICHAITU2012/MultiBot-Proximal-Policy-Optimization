#!/usr/bin/env python3
"""Repeatable safe-clearance push study using the pinned Go2 push actor.

The original S5b fixture and gate are untouched. This separate study uses a
geometrically reachable crate/column layout and matches the frozen policy's
training step (50 Hz actions, Genesis 0.02 s scene step, two 0.01 s substeps).
It does not change team_lab, ecosystem runtime behavior, or policy weights.

From ecosystem/ (with workspace paths available):
    python -u tests/test_s5b2_push.py
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.paths import TEAM_LAB_EXPERIMENTS, TEAM_LAB_LOGS, TEAM_LAB_ROOT  # noqa: E402
from core.proximity import RADIUS, SAFE_GAP, Proximity  # noqa: E402
from core.replan import validate_route  # noqa: E402
from core.safety import ConstraintLayer  # noqa: E402

for _path in (TEAM_LAB_EXPERIMENTS, TEAM_LAB_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import genesis as gs  # noqa: E402
import colo_env  # noqa: E402
from team.agents import Go2TaskAgent  # noqa: E402
from team.blackboard import Blackboard  # noqa: E402

PUSH_DIR = TEAM_LAB_LOGS / "go2-push-s5"
PUSH_MODEL = PUSH_DIR / "model_800.pt"
PUSH_MD5 = "c56f741011b2b2614279353abd702ee5"
PUSH_SHA256 = "123feb33d13d0361eac4ba09528eb79be855ee6048c5690522a9aac6d2348bf9"
PUSH_CFG_SHA256 = "41cd2b0f9c174932004e6748094828dc0ee44160b634483cd47f96f2842c957f"

PUSH_SCENE_DT = 0.02
PUSH_SUBSTEPS = 2
PUSH_POLICY_HZ = 50
PUSH_PHYSICS_DT = PUSH_SCENE_DT / PUSH_SUBSTEPS
MAX_SIM_TIME_S = 12.0
MAX_SPEED_BOUND_M_S = 2.0
CONTROL_LOOKAHEAD_M = MAX_SPEED_BOUND_M_S / PUSH_POLICY_HZ
PHYSICS_LOOKAHEAD_M = MAX_SPEED_BOUND_M_S * PUSH_PHYSICS_DT
TRACE_EVERY_STEPS = 5

CRATE_START = (0.0, 0.0)
# The column is 1.3 m from the nominal crate center. Its offset to the
# northwest keeps the eastward push moving away from it across the checkpoint's
# crate-position randomization, while leaving a safe approach from the west.
BLOCKER = (-0.5, 1.2)
CRATE_SIDE = 0.35
CRATE_HEIGHT = 0.25
CRATE_START_Z = 0.14
CRATE_MASS_KG = 7.0
ROBOT_SIDE_ANGLE = math.pi
PUSH_ANGLE = ROBOT_SIDE_ANGLE + math.pi
PUSH_DIR_XY = (math.cos(PUSH_ANGLE), math.sin(PUSH_ANGLE))
TARGET_DISTANCE_M = 2.20
TRAINING_SPAWN_DISTANCE_M = 1.20
LAUNCH_DISTANCE_M = 0.95
CONTACT_RADIUS_M = 0.50
EXPECTED_SEEDS = (11, 23, 37, 53, 71)


class _InactiveBody:
    """Far-away monitor-only body; these bodies are absent from the scene."""
    def __init__(self, pos):
        self.pos = [float(v) for v in pos]

    def get_state(self):
        return {"pos": self.pos, "base_pos": self.pos}


class _InactiveEntity:
    def __init__(self, pos):
        self.pos = torch.tensor([pos], dtype=gs.tc_float, device=gs.device)

    def get_pos(self):
        return self.pos


class _MonitorWorld:
    """Adapter for the shared Proximity layer around this small scene."""
    def __init__(self, go2, blocker):
        self.robots = {
            "go2": go2,
            "drone": _InactiveBody((-20.0, -20.0, 1.0)),
            "panda": _InactiveBody((-20.0, 20.0, 0.4)),
        }
        self.g1_e = _InactiveEntity((20.0, 20.0, 0.78))
        self.static_obstacles = {
            "blk0": {"pos": [blocker[0], blocker[1], 0.20],
                     "radius": 0.30, "height": 0.40}
        }


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_info() -> tuple[torch.nn.Module, dict]:
    if not PUSH_MODEL.is_file():
        raise FileNotFoundError(f"Pinned Go2 push checkpoint is missing: {PUSH_MODEL}")
    md5 = hashlib.md5(PUSH_MODEL.read_bytes()).hexdigest()
    sha = _sha256(PUSH_MODEL)
    if md5 != PUSH_MD5 or sha != PUSH_SHA256:
        raise RuntimeError(
            "Pinned Go2 push checkpoint hash mismatch; refusing to substitute another model "
            f"(md5={md5}, sha256={sha})"
        )
    cfg_path = PUSH_DIR / "cfgs.pkl"
    cfg_sha = _sha256(cfg_path)
    if cfg_sha != PUSH_CFG_SHA256:
        raise RuntimeError(
            "Pinned Go2 push training config hash mismatch; refusing to infer the policy "
            f"profile from a different run (sha256={cfg_sha})"
        )
    cfgs = pickle.loads(cfg_path.read_bytes())
    if not isinstance(cfgs, (tuple, list)) or len(cfgs) < 4:
        raise RuntimeError("Pinned Go2 push training config has an unexpected layout")
    env_cfg, _obs_cfg, _reward_cfg, push_cfg = cfgs[:4]
    crate_size = tuple(float(v) for v in push_cfg["crate_size"])
    crate_spawn = tuple(float(v) for v in push_cfg["crate_spawn"])
    base_spawn = tuple(float(v) for v in env_cfg["base_init_pos"])
    saved_goal = tuple(float(v) for v in push_cfg["goal_xy"])
    if crate_size != (0.35, 0.35, 0.25) or float(push_cfg["crate_mass"]) != 7.0:
        raise RuntimeError(f"Unexpected pinned checkpoint crate profile: {push_cfg}")
    if (abs(crate_spawn[0] - 1.0) > 1e-9 or abs(crate_spawn[1]) > 1e-9
            or abs(crate_spawn[2] - CRATE_START_Z) > 1e-9
            or abs(base_spawn[0] - 2.2) > 1e-9 or abs(base_spawn[1]) > 1e-9):
        raise RuntimeError(f"Unexpected pinned checkpoint spawn profile: {env_cfg}, {push_cfg}")
    if abs(math.dist(saved_goal, crate_spawn[:2]) - TARGET_DISTANCE_M) > 1e-9:
        raise RuntimeError(f"Experiment target distance differs from the saved config: {push_cfg}")
    if abs(math.dist(base_spawn[:2], crate_spawn[:2]) - TRAINING_SPAWN_DISTANCE_M) > 1e-9:
        raise RuntimeError(f"Experiment start distance differs from the saved config: {env_cfg}, {push_cfg}")
    if tuple(env_cfg["joint_names"]) != tuple(colo_env.GO2_JOINT_NAMES):
        raise RuntimeError("Go2 joint ordering differs between checkpoint config and Go2Agent")
    policy = colo_env.load_actor_from_log(PUSH_DIR, PUSH_MODEL.name)
    first = next(layer for layer in policy if isinstance(layer, torch.nn.Linear))
    last = next(layer for layer in reversed(policy) if isinstance(layer, torch.nn.Linear))
    if (first.in_features, last.out_features) != (49, 12):
        raise RuntimeError(
            "Pinned push actor architecture mismatch: "
            f"expected 49 observations/12 actions, got {first.in_features}/{last.out_features}"
        )
    return policy, {
        "asset": "team_lab/logs/go2-push-s5/model_800.pt",
        "md5": md5,
        "sha256": sha,
        "observation_dim": first.in_features,
        "action_dim": last.out_features,
        "policy_hz": PUSH_POLICY_HZ,
        "training_cfg_sha256": cfg_sha,
        "training_profile": {
            "crate_size_m": list(crate_size),
            "crate_mass_kg": float(push_cfg["crate_mass"]),
            "crate_spawn_xyz_m": list(crate_spawn),
            "base_init_xyz_m": list(base_spawn),
            "robot_crate_spawn_distance_m": round(math.dist(base_spawn[:2], crate_spawn[:2]), 6),
            "saved_goal_xy_m": list(saved_goal),
            "saved_goal_separation_m": round(math.dist(saved_goal, crate_spawn[:2]), 6),
            "joint_names_match_runtime_agent": True,
            "profile_source": "hash-pinned cfgs.pkl for model_800",
        },
    }


def _xy_at(radius: float, angle: float, origin=CRATE_START) -> tuple[float, float]:
    return (origin[0] + radius * math.cos(angle),
            origin[1] + radius * math.sin(angle))


def _signed_square_sat_gap(center_a, yaw_a, center_b, yaw_b, side=CRATE_SIDE):
    """Signed SAT separation/penetration margin for equal square footprints."""
    half = side / 2.0
    axes_a = ((math.cos(yaw_a), math.sin(yaw_a)),
              (-math.sin(yaw_a), math.cos(yaw_a)))
    axes_b = ((math.cos(yaw_b), math.sin(yaw_b)),
              (-math.sin(yaw_b), math.cos(yaw_b)))
    delta = (center_b[0] - center_a[0], center_b[1] - center_a[1])
    gaps = []
    for axis in (*axes_a, *axes_b):
        center_sep = abs(delta[0] * axis[0] + delta[1] * axis[1])
        ra = half * sum(abs(axis[0] * a[0] + axis[1] * a[1]) for a in axes_a)
        rb = half * sum(abs(axis[0] * b[0] + axis[1] * b[1]) for b in axes_b)
        gaps.append(center_sep - ra - rb)
    return max(gaps)


def _square_footprint_clearance(center_a, yaw_a, center_b, yaw_b, side=CRATE_SIDE):
    """Euclidean boundary distance for two disjoint square XY footprints."""
    sat_margin = _signed_square_sat_gap(center_a, yaw_a, center_b, yaw_b, side)
    if sat_margin < 0.0:
        return sat_margin

    def corners(center, yaw):
        axis_x = (math.cos(yaw), math.sin(yaw))
        axis_y = (-math.sin(yaw), math.cos(yaw))
        half = side / 2.0
        return [
            (center[0] + sx * half * axis_x[0] + sy * half * axis_y[0],
             center[1] + sx * half * axis_x[1] + sy * half * axis_y[1])
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]

    def point_segment_distance(point, start, end):
        vx, vy = end[0] - start[0], end[1] - start[1]
        length_sq = vx * vx + vy * vy
        t = 0.0 if length_sq == 0.0 else max(0.0, min(1.0,
            ((point[0] - start[0]) * vx + (point[1] - start[1]) * vy) / length_sq))
        return math.hypot(point[0] - (start[0] + t * vx),
                          point[1] - (start[1] + t * vy))

    corners_a = corners(center_a, yaw_a)
    corners_b = corners(center_b, yaw_b)
    edges_a = list(zip(corners_a, corners_a[1:] + corners_a[:1]))
    edges_b = list(zip(corners_b, corners_b[1:] + corners_b[:1]))
    distances = [point_segment_distance(point, *edge)
                 for point in corners_a for edge in edges_b]
    distances += [point_segment_distance(point, *edge)
                  for point in corners_b for edge in edges_a]
    return min(distances)


def _crate_yaw(quat) -> float:
    return math.atan2(
        2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
        1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
    )


def _push_out_gate(crate_xy, crate_yaw, start_crate, start_center_distance,
                   start_footprint_clearance) -> bool:
    delta = (crate_xy[0] - start_crate[0], crate_xy[1] - start_crate[1])
    along_push = delta[0] * PUSH_DIR_XY[0] + delta[1] * PUSH_DIR_XY[1]
    displacement = math.hypot(*delta)
    center_distance = math.dist(crate_xy, BLOCKER)
    sat_gap = _signed_square_sat_gap(crate_xy, crate_yaw, BLOCKER, 0.0)
    footprint_clearance = _square_footprint_clearance(crate_xy, crate_yaw, BLOCKER, 0.0)
    return all((
        along_push >= 0.40,
        displacement >= 0.40,
        center_distance >= 1.40,
        center_distance >= start_center_distance + 0.10,
        sat_gap >= 0.0,
        footprint_clearance >= start_footprint_clearance + 0.05,
    ))


def _contact_data(entity_a, entity_b):
    data = entity_a.get_contacts(with_entity=entity_b)
    forces = data["force_a"].reshape(-1, 3)
    force_norms = torch.linalg.norm(forces, dim=-1).detach().cpu().tolist()
    valid = data.get("valid_mask")
    if valid is not None:
        mask = valid.detach().cpu().reshape(-1).tolist()
        force_norms = [force for force, is_valid in zip(force_norms, mask) if is_valid]
    return len(force_norms), force_norms


def _scene_for_seed(spawn_xy, yaw, crate_xy, crate_yaw):
    """Checkpoint-profile Go2/crate physics plus one fixed 0.35 m column."""
    colo_env._ensure_genesis_init()
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=PUSH_SCENE_DT, substeps=PUSH_SUBSTEPS),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=False,
            max_collision_pairs=30,
            tolerance=1e-5,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -2.0, 2.5), camera_lookat=(0.0, 0.0, 0.4), camera_fov=40,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    go2_entity = scene.add_entity(gs.morphs.URDF(
        file="urdf/go2/urdf/go2.urdf", pos=(spawn_xy[0], spawn_xy[1], 0.42),
        quat=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
    ))
    crate = scene.add_entity(gs.morphs.Box(
        size=(CRATE_SIDE, CRATE_SIDE, CRATE_HEIGHT),
        pos=(crate_xy[0], crate_xy[1], CRATE_START_Z),
        quat=(math.cos(crate_yaw / 2), 0.0, 0.0, math.sin(crate_yaw / 2)),
        fixed=False,
    ))
    blocker = scene.add_entity(gs.morphs.Box(
        size=(CRATE_SIDE, CRATE_SIDE, 0.40),
        pos=(BLOCKER[0], BLOCKER[1], 0.20), fixed=True,
    ))
    scene.build(n_envs=1)
    crate.set_mass(CRATE_MASS_KG)
    go2 = colo_env.Go2Agent(go2_entity, command=(0.3, 0.0, 0.0))
    q = go2.entity.get_qpos()[0].clone()
    q[0], q[1], q[2] = float(spawn_xy[0]), float(spawn_xy[1]), 0.42
    q[3], q[4], q[5], q[6] = math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)
    q[7:19] = go2.default_dof_pos[0]
    go2.entity.set_qpos(q.unsqueeze(0), zero_velocity=True)
    go2.refresh()
    go2.external_control = True
    return scene, go2, crate, blocker


def _safe_conflicts(prox: Proximity) -> list[str]:
    conflicts = set()
    for pair, gap in prox.gaps().items():
        if "go2" in pair and math.isfinite(gap) and gap <= SAFE_GAP + CONTROL_LOOKAHEAD_M:
            conflicts.update(other for other in pair if other != "go2")
    return sorted(conflicts)


def _run_seed(seed: int, actor: torch.nn.Module) -> dict:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    crate_xy = (CRATE_START[0] + rng.uniform(-0.15, 0.15),
                CRATE_START[1] + rng.uniform(-0.15, 0.15))
    crate_yaw = rng.uniform(-0.4, 0.4)
    initial_crate_yaw = crate_yaw
    radial_offset = rng.uniform(-0.025, 0.025)
    lateral_offset = rng.uniform(-0.025, 0.025)
    yaw_offset = rng.uniform(-0.025, 0.025)
    side = (math.cos(ROBOT_SIDE_ANGLE), math.sin(ROBOT_SIDE_ANGLE))
    lateral = (-side[1], side[0])
    spawn_radius = TRAINING_SPAWN_DISTANCE_M + radial_offset
    spawn_xy = (crate_xy[0] + spawn_radius * side[0] + lateral_offset * lateral[0],
                crate_xy[1] + spawn_radius * side[1] + lateral_offset * lateral[1])
    robot_yaw = PUSH_ANGLE + yaw_offset
    launch_xy = _xy_at(LAUNCH_DISTANCE_M, ROBOT_SIDE_ANGLE, crate_xy)
    contact_xy = _xy_at(CONTACT_RADIUS_M, ROBOT_SIDE_ANGLE, crate_xy)
    goal_xy = (crate_xy[0] + TARGET_DISTANCE_M * PUSH_DIR_XY[0],
               crate_xy[1] + TARGET_DISTANCE_M * PUSH_DIR_XY[1])

    scene, go2, crate, blocker = _scene_for_seed(spawn_xy, robot_yaw, crate_xy, crate_yaw)
    monitor_world = _MonitorWorld(go2, BLOCKER)
    prox = Proximity(monitor_world)
    prox.update(0.0)
    layer = ConstraintLayer(prox)
    board = Blackboard()
    obs_builder = Go2TaskAgent(go2, board, "go2")
    start_crate = [float(v) for v in crate.get_pos()[0, :2].tolist()]
    start_center_distance = math.dist(start_crate, BLOCKER)
    start_sat_margin = _signed_square_sat_gap(start_crate, crate_yaw, BLOCKER, 0.0)
    start_footprint_clearance = _square_footprint_clearance(
        start_crate, crate_yaw, BLOCKER, 0.0)
    go2_start = go2.get_state()["pos"]
    route_ok, route_hits = validate_route(
        prox, "go2", go2_start,
        [[launch_xy[0], launch_xy[1], go2_start[2]]],
        [contact_xy[0], contact_xy[1], go2_start[2]],
    )
    dx, dy = side
    projection = max(CONTACT_RADIUS_M, min(spawn_radius,
        (BLOCKER[0] - crate_xy[0]) * dx + (BLOCKER[1] - crate_xy[1]) * dy))
    nearest_xy = _xy_at(projection, ROBOT_SIDE_ANGLE, crate_xy)
    min_center_distance = math.dist(nearest_xy, BLOCKER)
    min_surface_gap = min_center_distance - RADIUS["go2"] - 0.30
    initial_gap = prox.report()["min_gap_by_pair"].get("blk0-go2")
    preflight = {
        "route_to_launch_and_contact_clear": bool(route_ok),
        "route_rejected_by": route_hits or [],
        "go2_spawn_xy": [round(v, 6) for v in spawn_xy],
        "launch_xy": [round(v, 6) for v in launch_xy],
        "contact_xy": [round(v, 6) for v in contact_xy],
        "goal_xy": [round(v, 6) for v in goal_xy],
        "crate_blocker_center_distance_m": round(start_center_distance, 6),
        "crate_blocker_signed_sat_margin_m": round(start_sat_margin, 6),
        "crate_blocker_footprint_clearance_m": round(start_footprint_clearance, 6),
        "go2_blocker_initial_surface_gap_m": initial_gap,
        "go2_blocker_approach_min_center_distance_m": round(min_center_distance, 6),
        "go2_blocker_approach_min_surface_gap_m": round(min_surface_gap, 6),
        "go2_blocker_required_surface_gap_m": SAFE_GAP,
        "approach_surface_gap_after_safety_margin_m": round(min_surface_gap - SAFE_GAP, 6),
        "scene_dt_s": PUSH_SCENE_DT,
        "substeps": PUSH_SUBSTEPS,
        "policy_hz": PUSH_POLICY_HZ,
    }
    initial_goal_dist = math.dist(start_crate, goal_xy)
    # Training encodes goal - crate, so the vector shrinks as the crate
    # advances. Pass the fixed world goal instead of synthesizing a constant
    # 2.2 m vector at each policy step.
    obs = obs_builder._build_push_obs(go2.get_state(), start_crate, goal_xy)
    with torch.inference_mode():
        sample_action = actor(obs)
    obs_ok = tuple(obs.shape) == (1, 49) and tuple(sample_action.shape) == (1, 12)
    preflight["push_observation_shape"] = list(obs.shape)
    preflight["push_action_shape"] = list(sample_action.shape)
    preflight_ok = route_ok and min_surface_gap >= SAFE_GAP and obs_ok

    trace = []
    min_go2_gap = float("inf")
    min_crate_blocker_distance = float("inf")
    min_crate_sat_gap = float("inf")
    min_crate_footprint_clearance = float("inf")
    min_go2_z = float("inf")
    max_go2_z = -float("inf")
    contact_ticks = 0
    contact_events = 0
    contact_streak = 0
    max_contact_streak = 0
    max_contact_force = 0.0
    contact_force_samples = []
    contact_event_times = []
    previous_contact = False
    go2_blocker_contact = False
    crate_blocker_contact = False
    safety_stop = None if preflight_ok else "preflight_failed"
    task_complete_time = None
    sim_steps = 0
    if preflight_ok:
        with torch.inference_mode():
            for step in range(round(MAX_SIM_TIME_S / PUSH_SCENE_DT)):
                current_crate = [float(v) for v in crate.get_pos()[0, :2].tolist()]
                current_yaw = _crate_yaw(crate.get_quat()[0].tolist())
                if _push_out_gate(current_crate, current_yaw, start_crate,
                                  start_center_distance, start_footprint_clearance):
                    task_complete_time = step * PUSH_SCENE_DT
                    break

                conflicts = _safe_conflicts(prox)
                token = layer.request(
                    "go2", "s5b2_frozen_push_policy", conflict=conflicts,
                    sim_time_s=round(step * PUSH_SCENE_DT, 6),
                    reachable_disc_m=round(CONTROL_LOOKAHEAD_M, 6),
                )
                if token in layer._blocked:
                    safety_stop = "safety_layer_blocked"
                    break

                push_obs = obs_builder._build_push_obs(go2.get_state(), current_crate, goal_xy)
                action = actor(push_obs)
                if tuple(push_obs.shape) != (1, 49) or tuple(action.shape) != (1, 12):
                    safety_stop = "push_policy_io_shape_mismatch"
                    break
                go2.apply_action(action)
                layer.execute(token, [])

                scene.step()
                sim_steps += 1
                now = sim_steps * PUSH_SCENE_DT
                go2.refresh()
                prox.update(now)
                go2_state = go2.get_state()
                go2_z = float(go2_state["pos"][2])
                min_go2_z = min(min_go2_z, go2_z)
                max_go2_z = max(max_go2_z, go2_z)
                gap = prox.gaps().get(("blk0", "go2"), float("inf"))
                min_go2_gap = min(min_go2_gap, gap)

                crate_xy = [float(v) for v in crate.get_pos()[0, :2].tolist()]
                crate_distance = math.dist(crate_xy, BLOCKER)
                min_crate_blocker_distance = min(min_crate_blocker_distance, crate_distance)
                crate_quat = crate.get_quat()[0].tolist()
                crate_yaw = _crate_yaw(crate_quat)
                sat_gap = _signed_square_sat_gap(crate_xy, crate_yaw, BLOCKER, 0.0)
                footprint_clearance = _square_footprint_clearance(
                    crate_xy, crate_yaw, BLOCKER, 0.0)
                min_crate_sat_gap = min(min_crate_sat_gap, sat_gap)
                min_crate_footprint_clearance = min(
                    min_crate_footprint_clearance, footprint_clearance)

                n_contact, forces = _contact_data(go2.entity, crate)
                contact = n_contact > 0
                if contact and not previous_contact:
                    contact_events += 1
                    contact_event_times.append(round(now, 4))
                if contact:
                    contact_ticks += 1
                    contact_streak += 1
                    max_contact_streak = max(max_contact_streak, contact_streak)
                    contact_force_samples.extend(forces)
                    if forces:
                        max_contact_force = max(max_contact_force, max(forces))
                else:
                    contact_streak = 0
                previous_contact = contact

                go2_blocker_contact |= _contact_data(go2.entity, blocker)[0] > 0
                crate_blocker_contact |= _contact_data(crate, blocker)[0] > 0

                if step % TRACE_EVERY_STEPS == 0:
                    trace.append({
                        "sim_time_s": round(now, 4),
                        "go2_xy": [round(float(go2_state["pos"][0]), 4),
                                   round(float(go2_state["pos"][1]), 4)],
                        "go2_z_m": round(go2_z, 4),
                        "crate_xy": [round(crate_xy[0], 4), round(crate_xy[1], 4)],
                        "crate_blocker_center_distance_m": round(crate_distance, 4),
                        "crate_blocker_signed_sat_margin_m": round(sat_gap, 4),
                        "crate_blocker_footprint_clearance_m": round(footprint_clearance, 4),
                        "go2_blocker_surface_gap_m": round(gap, 4),
                        "go2_crate_contact": contact,
                        "go2_crate_contact_count": n_contact,
                    })

                if gap <= SAFE_GAP + PHYSICS_LOOKAHEAD_M:
                    safety_stop = "runtime_safety_margin"
                    break
                if _push_out_gate(crate_xy, crate_yaw, start_crate,
                                  start_center_distance, start_footprint_clearance):
                    task_complete_time = now
                    break

    end_crate = [float(v) for v in crate.get_pos()[0, :2].tolist()]
    end_blocker_distance = math.dist(end_crate, BLOCKER)
    end_quat = crate.get_quat()[0].tolist()
    end_yaw = _crate_yaw(end_quat)
    end_sat_gap = _signed_square_sat_gap(end_crate, end_yaw, BLOCKER, 0.0)
    end_footprint_clearance = _square_footprint_clearance(
        end_crate, end_yaw, BLOCKER, 0.0)
    displacement = math.dist(start_crate, end_crate)
    report = prox.report()
    pair_gap = report["min_gap_by_pair"].get("blk0-go2")
    if pair_gap is not None:
        min_go2_gap = min(min_go2_gap, pair_gap)
    policy_target_distance = math.dist(end_crate, goal_xy)
    policy_target_reached = policy_target_distance < 0.50
    task_completed = (task_complete_time is not None
                      and _push_out_gate(end_crate, end_yaw, start_crate,
                                         start_center_distance, start_footprint_clearance))
    longest_contact_s = max_contact_streak * PUSH_SCENE_DT
    clearance_improved = end_footprint_clearance >= start_footprint_clearance + 0.05
    passed = all((
        preflight_ok,
        task_completed,
        displacement >= 0.40,
        end_blocker_distance >= 0.50,
        end_blocker_distance >= start_center_distance + 0.10,
        end_blocker_distance >= 1.40,
        end_sat_gap >= 0.0,
        clearance_improved,
        contact_ticks > 0,
        longest_contact_s >= 0.10,
        min_go2_z >= 0.15,
        not go2_blocker_contact,
        not crate_blocker_contact,
        min_go2_gap >= SAFE_GAP,
        layer.stats()["blocked"] == 0,
    ))
    result = {
        "seed": seed,
        "passed": passed,
        "reason": safety_stop or ("push_out_complete" if task_completed else
                                   "policy_target_reached" if policy_target_reached else "episode_timeout"),
        "preflight": preflight,
        "simulated_time_s": round(sim_steps * PUSH_SCENE_DT, 4),
        "task_complete_time_s": None if task_complete_time is None else round(task_complete_time, 4),
        "crate": {
            "start_xy": [round(v, 6) for v in start_crate],
            "end_xy": [round(v, 6) for v in end_crate],
            "goal_xy": [round(v, 6) for v in goal_xy],
            "initial_yaw_rad": round(initial_crate_yaw, 6),
            "displacement_m": round(displacement, 6),
            "displacement_along_push_m": round(
                (end_crate[0] - start_crate[0]) * PUSH_DIR_XY[0]
                + (end_crate[1] - start_crate[1]) * PUSH_DIR_XY[1], 6),
            "policy_target_distance_end_m": round(policy_target_distance, 6),
            "start_blocker_center_distance_m": round(start_center_distance, 6),
            "end_blocker_center_distance_m": round(end_blocker_distance, 6),
            "minimum_blocker_center_distance_m": round(min_crate_blocker_distance, 6),
            "start_signed_sat_margin_m": round(start_sat_margin, 6),
            "end_signed_sat_margin_m": round(end_sat_gap, 6),
            "minimum_signed_sat_margin_m": round(min_crate_sat_gap, 6),
            "start_footprint_clearance_m": round(start_footprint_clearance, 6),
            "end_footprint_clearance_m": round(end_footprint_clearance, 6),
            "minimum_footprint_clearance_m": round(min_crate_footprint_clearance, 6),
        },
        "safety": {
            "go2_blocker_min_surface_gap_m": None if math.isinf(min_go2_gap) else round(min_go2_gap, 6),
            "required_surface_gap_m": SAFE_GAP,
            "go2_blocker_contact": go2_blocker_contact,
            "crate_blocker_contact": crate_blocker_contact,
            "stop_reason": safety_stop,
            "layer": layer.stats(),
        },
        "contact": {
            "physical_contact_events": contact_events,
            "contact_event_times_s": contact_event_times,
            "contact_physics_ticks": contact_ticks,
            "longest_continuous_contact_s": round(longest_contact_s, 6),
            "max_contact_force_n": round(max_contact_force, 6),
            "mean_contact_force_n": round(sum(contact_force_samples) / len(contact_force_samples), 6)
                if contact_force_samples else 0.0,
        },
        "locomotion": {
            "minimum_base_z_m": None if math.isinf(min_go2_z) else round(min_go2_z, 6),
            "maximum_base_z_m": None if max_go2_z == -float("inf") else round(max_go2_z, 6),
        },
        "task_complete": task_completed,
        "policy_target_reached": policy_target_reached,
        "acceptance": {
            "safe_route_preflight": preflight_ok,
            "push_out_completion_gate": task_completed,
            "crate_displacement_ge_0p40m": displacement >= 0.40,
            "final_center_distance_ge_0p50m": end_blocker_distance >= 0.50,
            "clearance_improved_by_ge_0p10m": end_blocker_distance >= start_center_distance + 0.10,
            "final_center_distance_ge_1p40m": end_blocker_distance >= 1.40,
            "footprints_remain_disjoint": end_sat_gap >= 0.0,
            "footprint_clearance_increased": clearance_improved,
            "measured_stable_contact_ge_0p10s": contact_ticks > 0 and longest_contact_s >= 0.10,
            "no_fall": min_go2_z >= 0.15,
            "zero_go2_blocker_contact": not go2_blocker_contact,
            "zero_crate_blocker_contact": not crate_blocker_contact,
            "minimum_go2_safety_gap_met": min_go2_gap >= SAFE_GAP,
            "no_safety_denials": layer.stats()["blocked"] == 0,
        },
        "trace": trace,
    }
    del obs_builder, board, prox, layer, monitor_world, go2, crate, blocker, scene
    gc.collect()
    return result


def _write_artifact(path: Path, artifact: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(artifact, indent=2, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, action="append",
                        help="run selected pilot seed(s); defaults to five fixed acceptance seeds")
    parser.add_argument("--max-s", type=float, default=MAX_SIM_TIME_S)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "s5b2_feasible_push.json")
    args = parser.parse_args(argv)
    if args.max_s <= 0:
        parser.error("--max-s must be positive")

    seeds = tuple(args.seed) if args.seed else EXPECTED_SEEDS
    full_suite = args.seed is None and abs(args.max_s - MAX_SIM_TIME_S) < 1e-9
    actor, model = _model_info()
    try:
        genesis_version = gs.__version__
    except AttributeError:
        genesis_version = "unknown"
    artifact = {
        "schema_version": 1,
        "experiment": "S5b.2 feasible clearance-increase push",
        "status": "running",
        "acceptance_suite": full_suite,
        "run": {
            "git_head": _git_head(),
            "python": sys.version.split()[0],
            "genesis": genesis_version,
            "scene_dt_s": PUSH_SCENE_DT,
            "substeps": PUSH_SUBSTEPS,
            "rigid_step_s": PUSH_PHYSICS_DT,
            "policy_hz": PUSH_POLICY_HZ,
            "max_sim_time_s": args.max_s,
            "seeds": list(seeds),
            "path_overrides": sorted(
                name for name in ("GENESIS_WORKSPACE_ROOT", "GENESIS_TEAM_LAB_ROOT",
                                  "GENESIS_HUMANOID_ROOT", "GENESIS_WORLD_ROOT")
                if os.environ.get(name)
            ),
        },
        "model": model,
        "configuration": {
            "crate_start_xy": CRATE_START,
            "crate_start_z_m": CRATE_START_Z,
            "crate_size_m": [CRATE_SIDE, CRATE_SIDE, CRATE_HEIGHT],
            "crate_mass_kg": CRATE_MASS_KG,
            "blocker_center_xy": BLOCKER,
            "blocker_size_m": [CRATE_SIDE, CRATE_SIDE, 0.40],
            "crate_xy_randomization_m": [-0.15, 0.15],
            "crate_yaw_randomization_rad": [-0.4, 0.4],
            "robot_contact_side_angle_deg": math.degrees(ROBOT_SIDE_ANGLE),
            "push_direction_deg": math.degrees(PUSH_ANGLE) % 360.0,
            "training_spawn_distance_m": TRAINING_SPAWN_DISTANCE_M,
            "policy_goal_distance_m": TARGET_DISTANCE_M,
            "push_out_completion": {
                "along_push_displacement_min_m": 0.40,
                "final_center_distance_min_m": 1.40,
                "center_distance_gain_min_m": 0.10,
                "signed_sat_margin_min_m": 0.0,
                "footprint_clearance_gain_min_m": 0.05,
            },
            "max_speed_reachability_bound_m_s": MAX_SPEED_BOUND_M_S,
            "goal_observation": "fixed world goal; goal-minus-crate vector shrinks during pushing, matching Go2PushEnv",
            "experiment_controller": "pinned policy actor using team_lab Go2TaskAgent's observation builder",
        },
        "acceptance_criteria": {
            "fixed_seed_trials": len(seeds),
            "required_passes": len(seeds),
            "each_trial": [
                "clear geometric route to the contact side under the existing 0.22 m surface-gap rule",
                "crate moves >= 0.40 m along the planned push direction",
                "push-out completion: center distance >= 1.40 m and >= start + 0.10 m",
                "non-overlapping footprints under SAT and >= 0.05 m increase in measured edge-to-edge distance",
                "measured Go2/crate rigid contact for >= 0.10 s",
                "no fall, no robot/column contact, no crate/column contact",
                "Go2/column surface gap >= 0.22 m and no denied safety requests",
            ],
        },
        "trials": [],
    }
    _write_artifact(args.output, artifact)
    for seed in seeds:
        started = time.perf_counter()
        result = _run_seed(seed, actor)
        result["wall_time_s"] = round(time.perf_counter() - started, 3)
        artifact["trials"].append(result)
        passed = sum(bool(t["passed"]) for t in artifact["trials"])
        artifact["summary"] = {
            "completed_trials": len(artifact["trials"]),
            "passed_trials": passed,
            "failed_trials": len(artifact["trials"]) - passed,
            "required_trials": len(seeds),
        }
        _write_artifact(args.output, artifact)
        print(f"seed {seed}: {'PASS' if result['passed'] else 'FAIL'} "
              f"disp={result['crate']['displacement_m']:.3f}m "
              f"final_center={result['crate']['end_blocker_center_distance_m']:.3f}m "
              f"Go2_gap={result['safety']['go2_blocker_min_surface_gap_m']}m "
              f"contact={result['contact']['longest_continuous_contact_s']:.3f}s "
              f"task={result['task_complete']} reason={result['reason']}")

    passed = sum(bool(t["passed"]) for t in artifact["trials"])
    artifact["summary"] = {
        "completed_trials": len(artifact["trials"]),
        "passed_trials": passed,
        "failed_trials": len(artifact["trials"]) - passed,
        "required_trials": len(seeds),
    }
    artifact["status"] = (
        "accepted" if full_suite and passed == len(seeds)
        else "pilot_passed" if not full_suite and passed == len(seeds)
        else "failed"
    )
    artifact["source_sha256"] = {
        "experiments/s5b2_feasible_push.py": _sha256(Path(__file__)),
        "core/proximity.py": _sha256(ROOT / "core" / "proximity.py"),
        "core/safety.py": _sha256(ROOT / "core" / "safety.py"),
        "core/replan.py": _sha256(ROOT / "core" / "replan.py"),
        "team_lab/experiments/colo_env.py": _sha256(TEAM_LAB_EXPERIMENTS / "colo_env.py"),
        "team_lab/team/agents.py": _sha256(TEAM_LAB_ROOT / "team" / "agents.py"),
        "team_lab/experiments/go2_push_env.py": _sha256(TEAM_LAB_EXPERIMENTS / "go2_push_env.py"),
    }
    _write_artifact(args.output, artifact)
    print(f"S5b.2: {passed}/{len(seeds)} trials passed; status={artifact['status']}; "
          f"artifact={args.output}")
    return 0 if passed == len(seeds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
