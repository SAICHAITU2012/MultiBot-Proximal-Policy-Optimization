#!/usr/bin/env python3
"""world.py — one Genesis 1.4.5 scene, FOUR heterogeneous robots, one clock.

    drone (crazyflie)  100 Hz   hover policy
    go2   (quadruped)   50 Hz   locomotion policy
    panda (arm)        100 Hz   reach policy
    g1    (humanoid)    50 Hz   standing policy  <- ported, no retraining

Reuses the existing agent wrappers read-only (`team_lab/experiments/colo_env.py`)
so the verified mission code is never modified; the G1 is the new component.

Each robot keeps its own observation buffers, action history and control rate.
`step()` advances one 100 Hz scene step and services every robot on its own
period from the SAME clock — no per-robot clocks, no wall-clock timing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

GENESIS = Path("/Users/saichaitu/Desktop/Genesis")
sys.path.insert(0, str(GENESIS / "team_lab" / "experiments"))
sys.path.insert(0, str(GENESIS / "ecosystem"))

import genesis as gs  # noqa: E402
import colo_env as CE  # noqa: E402  (reused, never modified)
from robots.g1_stand import (G1Stand, G1StandPolicy, policy_params, load_env_args,  # noqa: E402
                             URDF as G1_URDF)

LOGS = GENESIS / "team_lab" / "logs"
# ONE clock at the G1's native rigid step (0.005 s = 200 Hz). Every robot is
# serviced on its own period DERIVED FROM ITS TRAINED CONTROL RATE, so the
# 200 Hz clock does not change how fast any policy is actually driven:
#   go2   25 Hz  -> every 8 steps   (the rate the validated team_lab mission
#                                    actually ran it at: period 2 on a 50 Hz clock)
#   drone 200 Hz -> every step (measured: its propellers lose thrust if the
#                   rpm command is not re-issued each sim step)
#   panda 50 Hz -> every 4 steps
#   g1    50 Hz -> every 4 steps, torque recomputed EVERY step (200 Hz), which
#                 is what its PD was trained with
SCENE_DT = 0.005
SUBSTEPS = 1

CRATE_XY = (1.20, 1.20)
CRATE_SIZE = (0.35, 0.35, 0.15)
CRATE_MASS = 7.0
G1_SPAWN = (-1.30, 0.20, 0.80)     # clear of the panda and of the crate corridor
DRONE_START = (-0.60, -0.90, 1.00)   # ~2.9 m from the crate: the flight must be real
DRONE_SPAWN_Z = 1.00


class World:
    """Shared-scene, shared-clock, four-robot world."""

    def __init__(self, show_viewer: bool = False, record_camera: bool = False,
                 drone_target=None, go2_command=(0.0, 0.0, 0.0),
                 g1_xy=None, drone_start=None, extra_blockers=None):
        CE._ensure_genesis_init()
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=SCENE_DT, substeps=SUBSTEPS),
            rigid_options=gs.options.RigidOptions(
                dt=SCENE_DT, enable_collision=True, enable_joint_limit=True,
                enable_self_collision=False, max_collision_pairs=50,
                constraint_solver=gs.constraint_solver.Newton, tolerance=1e-5),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(4.5, -4.5, 3.5), camera_lookat=(0.0, 0.0, 0.5), camera_fov=45),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        go2_e = self.scene.add_entity(gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf", pos=CE.GO2_INIT_POS, quat=(0.0, 0.0, 0.0, 1.0)))
        _drone_pos = tuple(drone_start) if drone_start is not None else DRONE_START
        drone_e = self.scene.add_entity(gs.morphs.Drone(
            file="urdf/drones/cf2x.urdf", pos=_drone_pos))
        self.pedestal = self.scene.add_entity(gs.morphs.Box(size=(0.6, 0.6, 0.4), pos=CE.PEDESTAL_POS, fixed=True))
        panda_e = self.scene.add_entity(gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml", pos=CE.PANDA_INIT_POS, quat=(1.0, 0.0, 0.0, 0.0)))
        self.parcel_table = self.scene.add_entity(gs.morphs.Box(size=(0.15, 0.15, 0.4), pos=(-1.55, 1.55, 0.2), fixed=True))
        self.parcel = self.scene.add_entity(gs.morphs.Box(size=(0.08, 0.03, 0.05), pos=(-1.55, 1.55, 0.435)))
        self.crate = self.scene.add_entity(gs.morphs.Box(
            size=CRATE_SIZE, pos=(CRATE_XY[0], CRATE_XY[1], CRATE_SIZE[2] / 2 + 0.005)))

        # ---- the G1 (new) ------------------------------------------------
        _g1 = (G1_SPAWN[0], G1_SPAWN[1], G1_SPAWN[2]) if g1_xy is None else (g1_xy[0], g1_xy[1], G1_SPAWN[2])
        self.g1_e = self.scene.add_entity(gs.morphs.URDF(
            file=str(G1_URDF), pos=_g1, quat=(0.0, 0.0, 0.0, 1.0)))

        # ---- M5 static blocker columns (real obstacles: N per scene) ------
        self.static_obstacles = {}     # name -> {pos, radius, height} (the proximity registry data)
        self.blocker_entities = []
        for i, bxy in enumerate(extra_blockers or []):
            e = self.scene.add_entity(gs.morphs.Box(
                size=(0.35, 0.35, 0.4), pos=(bxy[0], bxy[1], 0.2), fixed=True))
            self.blocker_entities.append(e)
            self.static_obstacles[f"blk{i}"] = {
                "pos": [float(bxy[0]), float(bxy[1]), 0.20],
                "radius": 0.30, "height": 0.40}

        self.camera = None
        if record_camera:
            self.camera = self.scene.add_sensor(gs.sensors.RasterizerCameraOptions(
                res=CE.REC_CAM_RES, pos=CE.REC_CAM_POS, lookat=CE.REC_CAM_LOOKAT, fov=CE.REC_CAM_FOV))

        self.scene.build(n_envs=1)

        self.crate.set_mass(CRATE_MASS)
        self.parcel.set_mass(0.15)
        solver = self.scene.rigid_solver
        for ent in (self.parcel, self.parcel_table):
            solver.set_geoms_friction(0.6, list(range(ent.geom_start, ent.geom_end)))

        # ---- robots ------------------------------------------------------
        self.robots = {
            "go2": CE.Go2Agent(go2_e, command=go2_command),
            "drone": CE.DroneAgent(drone_e, target=drone_target or _drone_pos),
            "panda": CE.PandaAgent(panda_e, self.parcel),
        }
        self.robots["go2"].policy = CE.load_actor_from_log(LOGS / "go2-nav", "model_800.pt")
        self.robots["drone"].policy = CE.load_actor_from_log(LOGS / "drone-hover-s4", "model_1600.pt")
        self.robots["panda"].policy = CE.load_actor_from_log(LOGS / "grasp-panda_rl", "model_1500.pt")
        # the arm IDLES (holds its ready pose) until another robot's report
        # releases it — otherwise it completes its own task before the mission
        # even starts, which is co-location, not coordination
        self.robots["panda"].external_control = True

        # per-robot service periods, in 200 Hz clock steps
        self.period = {"go2": 8, "drone": 1, "panda": 4}   # drone rpm must be reapplied every step
        self.g1 = G1Stand(self.g1_e, policy_params(load_env_args()), G1StandPolicy())
        self.g1_period = 4          # 50 Hz policy
        self.g1_passive = False     # ablation hook: no policy -> it physically falls
        self.g1_fell = False
        self.g1_min_z = 1e9
        self.id = {k: k for k in self.robots}      # stable per-instance identity
        self.id["g1"] = "g1"

        self.t = 0                  # shared clock, in 100 Hz steps
        self.tick_counts = {k: 0 for k in (*self.robots, "g1")}   # audit: per-robot tick counts

    # ------------------------------------------------------------------ #
    @property
    def sim_time(self) -> float:
        return self.t * SCENE_DT

    @torch.inference_mode()
    def step(self):
        """One 100 Hz step: service every robot on its own period, then physics."""
        for name, r in self.robots.items():
            if self.t % self.period[name] == 0:
                r.tick()
                self.tick_counts[name] += 1
        if self.t % self.g1_period == 0 and not self.g1_passive:
            self.g1.policy_step()
            self.tick_counts["g1"] += 1
        self.g1.apply_torque()          # torque recomputed every sim step

        self.scene.step()
        self.t += 1

        for r in self.robots.values():
            r.refresh()
        z = float(self.g1_e.get_pos()[0, 2])
        self.g1_min_z = min(self.g1_min_z, z)
        if z < 0.40:
            self.g1_fell = True

    # ------------------------------------------------------------------ #
    def crate_xy(self) -> tuple[float, float]:
        p = self.crate.get_pos()[0]
        return float(p[0]), float(p[1])

    def g1_z(self) -> float:
        return float(self.g1_e.get_pos()[0, 2])

    def robot_positions(self) -> dict:
        out = {}
        for n, r in self.robots.items():
            st = r.get_state()
            out[n] = st.get("pos") or st.get("base_pos") or st.get("ee_pos")
        p = self.g1_e.get_pos()[0]
        out["g1"] = [float(p[0]), float(p[1]), float(p[2])]
        return out

    def render_frame(self):
        if self.camera is None:
            return None
        return CE.CoLocatedTeam.render_frame(self)
