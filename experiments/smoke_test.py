#!/usr/bin/env python3
"""Smoke test: init genesis on a backend, load plane + Unitree Go2, step 100 times."""
import sys
import time

import genesis as gs

backend_name = sys.argv[1] if len(sys.argv) > 1 else "cpu"
backend = gs.cpu if backend_name == "cpu" else gs.gpu

gs.init(backend=backend, precision="32", logging_level="warning")
print(f"[smoke] genesis initialized on {backend_name}")

scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01))
plane = scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(0, 0, 0.42)))
scene.build(n_envs=0)
print(f"[smoke] scene built: {robot.n_dofs} dofs, {robot.n_links} links")

# settle the robot with PD position control at default standing pose
joint_names = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
dofs_idx = [robot.get_joint(n).dofs_idx_local[0] for n in joint_names]
robot.set_dofs_kp([20.0] * 12, dofs_idx_local=dofs_idx)
robot.set_dofs_kv([0.5] * 12, dofs_idx_local=dofs_idx)
default_pos = [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5]
robot.control_dofs_position(default_pos, dofs_idx_local=dofs_idx)

t0 = time.time()
for i in range(100):
    scene.step()
dt = time.time() - t0

base = robot.get_link("base")
q, v = base.get_pos(), robot.get_dofs_velocity(dofs_idx_local=dofs_idx)
print(f"[smoke] {backend_name}: 100 steps in {dt:.2f}s ({100 / dt:.0f} steps/s)")
print(f"[smoke] base pos after 100 steps: x={q[0]:.3f} y={q[1]:.3f} z={q[2]:.3f} (started z=0.42)")
print(f"[smoke] |dof_vel| max: {abs(v).max():.3f}")
print("[smoke] PASS" if q[2] > 0.2 and abs(v).max() < 5.0 else "[smoke] CHECK: robot may have fallen")
gs.destroy()
