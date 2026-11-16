#!/usr/bin/env python3
"""go2_drive.py — M3.1: yaw-then-drive steering for the Go2, on top of the
FROZEN nav policy (no retraining; this is a command-shaping layer).

Why it exists (M2's honest limit): the relay's Go2 leg had NO steering —
`_drive_toward` walked forward on the spawn heading, and a stationary blocker
in the lane produced an unrecoverable DEADLOCK because the coordinator could
not turn the dog. The reroute waypoints existed; the manoeuvre did not.

Measured constraints this primitive respects (do not "optimize" them away):
  * pure yaw commands are nearly dead: wz alone yields ~0.02-0.1 rad/s of
    actual rotation (team_lab + ecosystem measurements) — the gait only turns
    while it is WALKING, so turning uses the CREEP ARC (small forward vx +
    proportional wz).
  * the trained command ranges are vx [-0.5, 1.0], vy [-0.5, 0.5],
    wz [-1, 1] (go2_env). Everything emitted here stays inside them.
  * team_lab measured a WORKING strafe (vy) on the same checkpoint; the
    ecosystem M2 measured a strafe deadband. Both modes are implemented;
    M3.2's benchmark decides which the ecosystem uses (mode="arc" default,
    mode="strafe" available for the A/B).

States: TURN (creep arc until the heading error < TOL) -> DRIVE (forward with
a small corrective wz) -> ARRIVE (stop inside RADIUS). The class is stateless
per call except for an optional drive gate: it recomputes everything from the
live pose each tick, so a moving target or a mid-course reroute needs no
special handling.
"""
from __future__ import annotations

import math


def wrap_angle(a: float) -> float:
    """Wrap to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(q) -> float:
    """Yaw from a wxyz quaternion (w, x, y, z)."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class YawThenDrive:
    """Command shaper: pose + target -> (vx, vy, wz) inside the trained ranges."""

    def __init__(self,
                 drive_vx: float = 0.55,
                 creep_vx: float = 0.15,
                 turn_wz: float = 1.0,
                 tol_rad: float = 0.30,
                 arrive_r: float = 0.35,
                 slow_r: float = 1.2,       # decelerate inside this radius
                 mode: str = "arc"):        # "arc" (creep-turn) or "strafe"
        self.drive_vx = drive_vx
        self.creep_vx = creep_vx
        self.turn_wz = turn_wz
        self.tol = tol_rad
        self.arrive_r = arrive_r
        self.slow_r = slow_r
        self.mode = mode

    def command(self, pos, yaw, target) -> tuple[float, float, float, str]:
        """Return (vx, vy, wz, phase) for the current pose/target.

        pos: (x, y) world; yaw: current heading (rad); target: (x, y) world.
        """
        px, py = float(pos[0]), float(pos[1])
        tx, ty = float(target[0]), float(target[1])
        dist = math.hypot(tx - px, ty - py)
        bearing = math.atan2(ty - py, tx - px)
        ye = wrap_angle(bearing - yaw)

        if dist < self.arrive_r:
            return 0.0, 0.0, 0.0, "arrive"

        if abs(ye) > self.tol:
            # ---- TURN: the gait only yaws while walking (the creep arc) ----
            wz = max(-self.turn_wz, min(self.turn_wz, 2.5 * ye))
            if self.mode == "strafe":
                # strafe toward the target's side while turning (vy is
                # trained to +-0.5; team_lab measured it working)
                vy = max(-0.5, min(0.5, 0.30 * (1.0 if ye > 0 else -1.0)))
                return self.creep_vx * 0.5, vy, wz, "turn"
            # arc mode: a small forward creep keeps the gait alive; cap the
            # forward component when the target is far off-axis so the arc
            # does not walk away from the line
            vx = self.creep_vx * max(0.25, math.cos(ye))
            return vx, 0.0, wz, "turn"

        # ---- DRIVE: aligned; decelerate near the target ----
        speed = self.drive_vx * min(1.0, dist / self.slow_r)
        vz = max(-self.turn_wz, min(self.turn_wz, 1.2 * ye))  # fine correction
        return speed, 0.0, vz, "drive"

    def waypoint_command(self, pos, yaw, waypoints, final):
        """Drive through a waypoint list, then to `final`. Returns
        (vx, vy, wz, phase, active_wp). A waypoint is consumed when the dog
        is within its capture radius (0.35 m)."""
        cap = 0.35
        while waypoints and math.hypot(waypoints[0][0] - float(pos[0]),
                                       waypoints[0][1] - float(pos[1])) < cap:
            waypoints.pop(0)
        tgt = waypoints[0] if waypoints else final
        vx, vy, wz, phase = self.command(pos, yaw, tgt)
        return vx, vy, wz, phase, (waypoints[0] if waypoints else None)
