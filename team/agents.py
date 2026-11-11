"""
agents.py — mission-level robot agents for the team layer.

These wrap the colo_env per-robot handles (which couple raw entity state to the
learned policies) with TASK EXECUTION state machines and blackboard reporting.
Separation of concerns:
    colo_env.Go2Agent   — policy I/O (obs build, action apply, PD, rates)
    team.agents.Go2TaskAgent — mission behaviors (goto, push) ON TOP of the policy

Honest capability envelope (from the trained policies as evaluated):
    go2:   walks forward at ~0.5 m/s, turns via yaw-rate commands; stopping at
           close range uses out-of-distribution zero commands (retrain with
           wider command ranges is Stage 6b backlog).
    drone: station-keeps to ~0.3 m of a commanded 3-D target; waypoint hopping
           with a 0.35 m arrival radius.
    panda: grasp policy (plugs in when trained); until then holds pose.
"""

from __future__ import annotations

import math
from typing import Any

# ---------------------------------------------------------------- tasks


def make_task(id: str, type: str, required_capability: str, location: tuple,
              priority: int = 0, created_at: float = 0.0) -> dict:
    return {
        "id": id, "type": type, "required_capability": required_capability,
        "location": tuple(location), "priority": priority, "created_at": created_at,
        "status": "open", "assigned_to": None, "result": None,
    }


class TaskBoard:
    def __init__(self):
        self.tasks: dict[str, dict] = {}

    def add(self, task: dict):
        self.tasks[task["id"]] = task

    def open_tasks(self) -> list[dict]:
        return [t for t in self.tasks.values() if t["status"] == "open"]

    def get(self, tid: str) -> dict | None:
        return self.tasks.get(tid)

    def complete(self, tid: str, result: Any = None):
        t = self.tasks.get(tid)
        if t:
            t["status"] = "done"
            t["result"] = result

    def fail(self, tid: str, reason: str = ""):
        t = self.tasks.get(tid)
        if t:
            t["status"] = "failed"
            t["result"] = reason


# ---------------------------------------------------------------- agents


class TaskAgent:
    """Base: owns a colo robot handle, a blackboard, and at most one task."""

    def __init__(self, robot, blackboard, robot_id: str, team=None):
        self.robot = robot
        self.bb = blackboard
        self.id = robot_id
        self.team = team  # colo team (for control-rate gating)
        self.task: dict | None = None
        self._last_status_t = -1e9

    # -- lifecycle ---------------------------------------------------------
    def assign(self, task: dict):
        self.task = task
        self._begin(task)
        self.bb.publish(self.id, "event", "task_started",
                        {"task": task["id"], "type": task["type"]})

    def _begin(self, task: dict):
        raise NotImplementedError

    def update(self, t: float):
        """Called every control period; advances task execution."""
        if self.task is not None:
            self._execute(t)
        if getattr(self, "FALL_RECOVERY", False):
            self._fall_recovery(t)
        if t - self._last_status_t >= 2.0:  # status every 2 s
            self._last_status_t = t
            self.bb.publish(self.id, "status", "status_report", self.status())

    def _execute(self, t: float):
        raise NotImplementedError

    def _finish(self, result: Any):
        tid = self.task["id"]
        self.task = None
        # restore the walking policy if the learned push parked it
        if getattr(self, "_saved_policy", None) is not None:
            self.robot.policy = self._saved_policy
            self._saved_policy = None
            self._pushing = False
        if getattr(self.robot, "external_control", False):
            self.robot.external_control = False  # hand the arm back to colo tick
        self.bb.publish(self.id, "task", "task_complete", {"task": tid, "result": result})

    def status(self) -> dict:
        st = self.robot.get_state()
        pos = st.get("pos", st.get("base_pos", st.get("ee_pos")))
        vel = st.get("lin_vel_world", st.get("lin_vel_body", [0.0, 0.0, 0.0]))
        return {
            "pos": [round(float(x), 3) for x in pos],
            "vel": [round(float(x), 3) for x in vel],
            "task_id": self.task["id"] if self.task else None,
            "task_type": self.task["type"] if self.task else None,
        }

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _yaw_from_quat(quat) -> float:
        """Yaw from a wxyz quaternion (w, x, y, z)."""
        w, x, y, z = quat
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _yaw_error(self, target_xy, pos, yaw) -> float:
        """Angle from current heading to (target - pos), wrapped to (-pi, pi].

        Wrapping matters: an unwrapped error at the ±pi branch cut flips sign
        every step and the yaw controller chatters at the antipode."""
        dx, dy = target_xy[0] - pos[0], target_xy[1] - pos[1]
        ye = math.atan2(dy, dx) - yaw
        return (ye + math.pi) % (2.0 * math.pi) - math.pi


def _dump_obs(robot, st, box, goal):
    """Replicates _learned_push's exact obs construction for the A/B diff."""
    import torch as _torch
    import genesis as _gs
    from genesis.utils.geom import inv_quat as _iq, transform_by_quat as _tbq
    bt = _torch.tensor(st["quat"], dtype=_gs.tc_float, device=_gs.device).unsqueeze(0)
    iq = _iq(bt)
    dog = _torch.tensor([st["pos"][0], st["pos"][1], 0.0], dtype=_gs.tc_float, device=_gs.device).unsqueeze(0)
    crate_t = _torch.tensor([box[0], box[1], 0.0], dtype=_gs.tc_float, device=_gs.device).unsqueeze(0)
    goal_t = _torch.tensor([goal[0], goal[1], 0.0], dtype=_gs.tc_float, device=_gs.device).unsqueeze(0)
    zz = _torch.zeros((1, 1), dtype=_gs.tc_float, device=_gs.device)
    cr = _tbq(_torch.cat([crate_t[:, :2] - dog[:, :2], zz], -1), iq)[:, :2]
    gr = _tbq(_torch.cat([goal_t[:, :2] - crate_t[:, :2], zz], -1), iq)[:, :2]
    return _torch.cat((
        robot.base_ang_vel * 0.25,
        robot.projected_gravity,
        _torch.tensor([[0.6, 0.0, 0.0]], dtype=_gs.tc_float, device=_gs.device),
        robot.dof_pos - robot.default_dof_pos,
        robot.dof_vel * 0.05,
        robot.actions,
        cr * 1.0, gr * 1.0), dim=-1)[0].cpu()


class Go2TaskAgent(TaskAgent):
    """Navigation by commanding the learned locomotion policy (vx, vy, wz)."""

    FALL_RECOVERY = True      # the base-class hook: recover from a fallen state
    STOP_RADIUS = 0.8          # m — within this, command (0,0,0)
    APPROACH_RADIUS = 0.4      # m — tight arrival at the push start point
    PUSH_STANDOFF = 0.45       # m behind the box before pushing
    PUSH_SPEED = 0.2           # m/s — gentle push (0.5 kicks the box away)
    MAX_YAW_CMD = 0.8

    def _fall_recovery(self, t: float):
        """Sim-only stand-up recovery primitive (the honest limitation: the real
        Go2 runs a dedicated recovery controller — outside this project's
        training scope; the simulation equivalent resets the base to standing
        IN PLACE, preserving xy and yaw, then holds a 2-s settle). Triggered
        when the base stays below 0.15 m for >2 s during any task. Mission run 1
        (2026-11-08) showed a fallen dog otherwise 'walks' (drags) under the nav
        policy for the rest of the mission — 865 s of dead time."""
        z = float(self.robot.base_pos[0, 2])
        if getattr(self, "_recovering", False):
            if t < self._recover_until:
                self.robot.commands = _cmd(0.0, 0.0, 0.0)
                self.robot.entity.control_dofs_position(
                    self.robot.default_dof_pos[0],
                    dofs_idx_local=self.robot.motors_dof_idx)
                return
            self._recovering = False
            self._fall_since = None
            self.bb.publish(self.id, "event", "dog_recovered",
                            {"method": "sim_stand_up_primitive",
                             "z_after": round(float(self.robot.base_pos[0, 2]), 3)})
            if self.task is not None:  # restart the current task's phase machine
                self._pushing = False
                self._settle_until = None
                self._stroke_best_d = None
                self._begin(self.task)
            return
        if z < 0.15:
            if getattr(self, "_fall_since", None) is None:
                self._fall_since = t
            elif t - self._fall_since > 2.0:
                st = self.robot.get_state()
                yaw = self._yaw_from_quat(st["quat"])
                q = self.robot.entity.get_qpos()[0].cpu().clone()
                q[2] = 0.42  # standing base height (the spawn convention)
                q[3] = math.cos(yaw / 2.0)
                q[4] = 0.0
                q[5] = 0.0
                q[6] = math.sin(yaw / 2.0)
                q[7:19] = self.robot.default_dof_pos[0]
                self.robot.entity.set_qpos(q.unsqueeze(0), zero_velocity=True)
                self._recovering = True
                self._recover_until = t + 2.0
                self.robot.external_control = False
                self.bb.publish(self.id, "event", "dog_fell_recovering",
                                {"z_before": round(z, 3), "t": round(t, 1)})

    def _begin(self, task: dict):
        self.phase = "approach"
        self.goal = task["location"]
        self._stall_since = 0.0
        self._backing_until = None
        self._route = None
        self._route_anchor = None
        self._pushing = False
        self._saved_policy = None
        self._best_box_d = None
        self._last_push_t = 0.0
        self._push_waypoints = None

    def _execute(self, t: float):
        st = self.robot.get_state()
        pos, yaw = st["pos"], self._yaw_from_quat(st["quat"])
        # any task carrying a live box position is a push/fetch: go behind the
        # box, then drive it to the goal (plain goto only when there's no box)
        if "box_pos_fn" in self.task:
            self._push(st, pos, yaw, t)
        else:
            self._goto(pos, yaw, self.goal, on_arrive=lambda: self._finish("arrived"))

    def _goto(self, pos, yaw, target_xy, on_arrive, stop_radius=None, speed=0.5):
        """Holonomic navigation: forward + strafe + turn SIMULTANEOUSLY.

        The nav policy was trained with vx in [-0.5, 1.0], vy in [-0.5, 0.5],
        wz in [-1, 1] — using vy (sidestepping) avoids the slow arcing that
        pure turn-then-walk controllers suffer with yaw-tracking error ~0.25."""
        sr = self.STOP_RADIUS if stop_radius is None else stop_radius
        dx, dy = target_xy[0] - pos[0], target_xy[1] - pos[1]
        d = math.hypot(dx, dy)
        if d < sr:
            self.robot.commands = _cmd(0.0, 0.0, 0.0)
            on_arrive()
            return
        bearing = math.atan2(dy, dx)
        ye = (bearing - yaw + math.pi) % (2 * math.pi) - math.pi
        v = speed * min(1.0, d / 1.0)  # decelerate over the last meter: the
        # walking gait at 0.5 m/s TRIPS at crate contact (front paws stop,
        # body pitches); slow arrivals match the training's at-rest starts
        if abs(ye) > math.pi / 2:
            # BEHIND-TARGET MOTION (the endgame lesson pair, 2026-11-08): pure
            # backward is dead (~0.01 m/s) and the forward creep-turn ARCS
            # meters off-line (the nav's yaw response is ~0.02-0.1 rad/s — the
            # endgame dog wandered 0.6 m north in 1000 s and never
            # re-contacted a crate sitting 1.6 cm outside delivery). The nav
            # policy's STRAFE (vy +-0.5) is fully trained and yaw-free:
            # strafe-dominant motion toward the target's side with a token
            # forward creep and the turn command — no dead modes.
            vx = 0.10
            vy = max(-0.5, min(0.5, 0.35 * (1.0 if ye > 0 else -1.0)))
            wz = max(-1.0, min(1.0, 1.5 * ye))
        else:
            vx = max(-0.5, min(1.0, v * math.cos(ye)))
            vy = max(-0.5, min(0.5, v * math.sin(ye)))
            wz = max(-1.0, min(1.0, 1.5 * ye))
        self.robot.commands = _cmd(vx, vy, wz)

    def _push(self, st, pos, yaw, t):
        goal = self.goal
        box_raw = self.task["box_pos_fn"]()  # live box position, set by the mission
        # flatten whatever shape arrived ((1,3) tensor, nested tolist, flat list)
        while isinstance(box_raw[0], (list, tuple)):
            box_raw = box_raw[0]
        box = (float(box_raw[0]), float(box_raw[1]))
        box_yaw = None
        qfn = self.task.get("box_quat_fn")
        if qfn is not None:
            q = qfn()  # wxyz
            box_yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                                 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))

        # NOTE on crate routing: waypoint legs were tried and REVERTED — the
        # push policy was trained with the goal ~2.2 m from the crate; a
        # near waypoint (0.58 m) is out of distribution and freezes it. The
        # straight-line geometry (mission spawn/zone placement) avoids the
        # pedestal corner instead. Mid-push stall recovery stays active.

        box_d = math.dist(box, (self.goal[0], self.goal[1]))
        if box_d < 0.3:
            self.robot.commands = _cmd(0.0, 0.0, 0.0)
            self._finish({"box_goal_distance": round(box_d, 3),
                          "mode": "learned" if self._pushing else "hand"})
            return

        if self.robot.push_policy is not None:
            import os as _os
            if _os.environ.get("PUSH_OBS_DUMP") and not getattr(self.robot, "_obs_dumped", False):
                import torch as _dump_torch
                _dump = {"obs": _dump_obs(self.robot, st, box, goal), "dog_qpos": st, "crate": [box[0], box[1]]}
                _dump_torch.save(_dump, "../logs/push_obs_dump.pt")
                self.robot._obs_dumped = True

            self._learned_push(st, pos, yaw, box, self.goal, box_d, t, box_yaw=box_yaw)
        else:
            self._hand_push(st, pos, yaw, t, box, self.goal, box_d)

    def _learned_push(self, st, pos, yaw, box, goal, box_d, t, box_yaw=None):
        """FACE-ON push: the dog walks straight at the crate from wherever it
        is and presses it away from itself. The delivery zone accepts the
        crate from any bearing (radius 0.55), so the push direction follows
        the dog's approach — no stand point behind the crate (the backward +
        spin re-approach was the nav policy's broken mode: the dog froze on
        every retry). The push policy (trained on the 12 cm crate) handles
        the contact and the pressing; the stall/retry loop handles jams."""
        import torch as _torch
        import genesis as _gs

        robot = self.robot
        bx, by = box[0], box[1]

        # STALL DETECTION: the crate's no-improvement for 20 s -> the task
        # fails honestly (the push policy can't move it from here)
        bx_d = math.dist((bx, by), (goal[0], goal[1]))
        if getattr(self, "_best_box_d", None) is None:
            self._best_box_d = bx_d
            self._last_stall_t = t
        elif bx_d < self._best_box_d - 0.1:
            self._best_box_d = bx_d
            self._last_stall_t = t
        elif t - self._last_stall_t > 45.0:
            # END-GAME PERSISTENCE (mission run 3 lesson, 2026-11-08): the crate
            # stalled 0.57 m from the zone (2 cm outside delivery) because this
            # timer killed the fetch mid-re-approach and nothing re-assigns a
            # finished task — 600 s of idle deadlock followed. A crate short of
            # the zone is never abandoned. Inside 0.9 m this timer also yields
            # nothing (continuous push — see the stroke-stall branch); outside,
            # cycle the stroke retry until the mission budget ends.
            self._best_box_d = None
            self._last_stall_t = t
            self._pushing = False
            self.robot.external_control = False
            self.bb.publish(self.id, "event", "push_retry",
                            {"task": self.task["id"], "stage": "task_stall",
                             "box_goal_distance": round(bx_d, 2)})
            return

        # ZONE-ALIGNED POSITIONING (runs at ANY distance — mission-seed-2 lesson:
        # a dog 0.3 m from the crate on the WRONG side is inside the old
        # d_crate>0.55 guard and pushes from the wrong bearing forever). The
        # dog must be BEHIND the crate (opposite the zone) AND within 0.15 m
        # of the crate->zone line; otherwise walk to the stand point 0.6 m
        # behind the crate, swinging wide when the straight path would clip
        # the crate (a repositioning bump shoves the crate the wrong way).
        d_crate = math.dist((pos[0], pos[1]), [bx, by])
        # POSITIONING runs ONLY between strokes (never mid-push: the seeds-2
        # lesson — a mid-push position test yanks control from the active
        # stroke as soon as the crate drifts, and the dog stops applying push
        # actions entirely). The stroke runs its full course; the stall/retry
        # ends it; THEN the positioning decides: realign, or creep to contact.
        if not getattr(self, "_pushing", False):
            if getattr(self, "_external_nav", True):
                self.robot.external_control = False  # the nav policy executes
            zx, zy = goal[0], goal[1]
            czx, czy = bx - zx, by - zy
            L = math.hypot(czx, czy) + 1e-9
            line_dx, line_dy = -czx / L, -czy / L     # the unit vector crate -> zone
            wx, wy = pos[0] - bx, pos[1] - by         # crate -> dog
            along = wx * line_dx + wy * line_dy       # <0 = behind the crate (correct side)
            perp = abs(-wx * line_dy + wy * line_dx)  # the lateral offset from the line
            stand = (bx - line_dx * 0.6, by - line_dy * 0.6)
            # HEADING IS PART OF POSITION (the -123.2-degree lesson): a dog
            # standing behind the crate but FACING AWAY passes every distance
            # test, then the settle starts a stroke whose first obs are OOD
            # (the crate behind) — geometry_lost fires instantly, the retry
            # returns to the same spot, forever. The heading must be part of
            # the in-position test so the goto (strafe-dominant + turn
            # branches) actually rotates the dog.
            if box_yaw is not None and abs(math.atan2(
                    math.sin(box_yaw), math.cos(box_yaw))) > math.radians(10):
                # YAW-AWARE FACE POINT (the stall probe, 2026-11-08): a crate
                # yawed 27 deg SHEDS the pushing dog — the straight push slides
                # along the yawed face until the dog clears the corner, then
                # the crate sits with ZERO contacts and the dog parks on the
                # zone side. Re-approach the EAST face's center (the face away
                # from the zone) with a 0.35 m standoff: the contact lands on
                # the face center and the push follows the face's inward
                # normal (<= 30 deg off the zone line — the virtual goal and
                # the delivery region absorb it).
                fy = box_yaw + math.pi  # the east-face outward normal
                stand = (bx + math.cos(fy) * (0.175 + 0.35),
                         by + math.sin(fy) * (0.175 + 0.35))
            d_stand = math.dist((pos[0], pos[1]), stand)
            # two-tier alignment: in the end-game (box within 0.8 m of the zone)
            # a 0.28 m lateral offset alone keeps the crate outside the 0.55 m
            # delivery circle — the final strokes must aim AT the zone point
            perp_gate = 0.20 if bx_d < 0.8 else 0.35
            in_position = (along < -0.05) and (perp < perp_gate)
            if not in_position and d_stand > 0.3:
                target = stand
                if d_crate < 0.9:
                    # the straight path would clip the crate: swing wide via a
                    # point 1.0 m from the crate, on the dog's side of the line
                    seg = (stand[0] - pos[0], stand[1] - pos[1])
                    Ls2 = seg[0] ** 2 + seg[1] ** 2 + 1e-9
                    tc = max(0.0, min(1.0, ((bx - pos[0]) * seg[0] + (by - pos[1]) * seg[1]) / Ls2))
                    closest = (pos[0] + tc * seg[0], pos[1] + tc * seg[1])
                    if math.dist(closest, (bx, by)) < 0.5:
                        side = 1.0 if (wx * (-line_dy) + wy * line_dx) >= 0 else -1.0
                        target = (bx - line_dy * side * 1.0, by + line_dx * side * 1.0)
                if getattr(self, "_stand_since", None) is None:
                    self._stand_since = t
                    self._stand_start = (pos[0], pos[1])
                elif (t - self._stand_since > 12.0
                      and math.dist((pos[0], pos[1]), self._stand_start) < 0.2):
                    self._stand_since = None  # stuck: fall back to face-on this tick
                    self._goto(pos, yaw, (bx, by), on_arrive=lambda: None,
                               stop_radius=0.5, speed=0.4 if d_crate > 0.9 else 0.15)
                    return
                self._goto(pos, yaw, target, on_arrive=lambda: None,
                           stop_radius=0.25, speed=0.4 if math.dist((pos[0], pos[1]), target) > 1.0 else 0.15)
                return
            # in position (or at the stand point): creep to contact range
            self._stand_since = None
            if d_crate > 0.55:
                self._goto(pos, yaw, (bx, by), on_arrive=lambda: None,
                           stop_radius=0.5, speed=0.4 if d_crate > 0.9 else 0.15)
                return

        # IN CONTACT RANGE: hand off to the push policy (the settle first:
        # the joints to default + a 1.5 s hold — the training spawn state)
        if not getattr(self, "_pushing", False):
            if getattr(self, "_settle_until", None) is None:
                self._settle_until = t + 1.5
                self.robot.commands = _cmd(0.0, 0.0, 0.0)
                return
            if t < self._settle_until:
                self.robot.commands = _cmd(0.0, 0.0, 0.0)
                self.robot.entity.control_dofs_position(
                    self.robot.default_dof_pos[0],
                    dofs_idx_local=self.robot.motors_dof_idx)
                return
            self._settle_until = None
            self._pushing = True
            self.robot.external_control = True  # the push policy owns the dog
            self.bb.publish(self.id, "event", "push_phase",
                            {"task": self.task["id"], "mode": "learned"})

        # THE PUSH: the learned policy presses forward. VIRTUAL GOAL (the s5-era
        # finding: goal_rel < 2 m freezes the policy — its training distribution
        # is 2.2±0.15 m): feed a moving virtual goal 2.2 m from the crate along
        # the crate->zone line so the push stays in-distribution through the
        # end-game. Delivery remains a REAL-zone entry event (the 0.55 m check
        # runs every tick; the crate never needs to stop).
        # DRIFT-COMPENSATED VIRTUAL GOAL: the pushing gait drifts the crate
        # NORTH by a systematic +0.2-0.3 m over a 2+ m haul (measured across
        # 6 mission seeds — an off-center-contact + gait-yaw bias). Aim the
        # virtual goal 0.25 m SOUTH of the zone so the drifted crate LANDS in
        # the delivery circle. The delivery check itself stays on the REAL,
        # unmodified zone.
        aim = (goal[0], goal[1] - 0.25)
        zg = math.dist((bx, by), aim)
        if zg > 1e-6:
            vgoal = (bx + (aim[0] - bx) / zg * 2.2, by + (aim[1] - by) / zg * 2.2)
        else:
            vgoal = (goal[0], goal[1])
        obs = self._build_push_obs(st, box, vgoal)
        action = robot.push_policy(obs)
        robot.apply_action(action)

        # the stall detection: the crate's no-improvement for 20 s -> the
        # back-off + the re-approach (the stroke cycle)
        new_d = math.dist((bx, by), (goal[0], goal[1]))
        if getattr(self, "_stroke_best_d", None) is None:
            self._stroke_best_d = new_d
            self._stroke_t = t
        elif new_d < self._stroke_best_d - 0.1:
            self._stroke_best_d = new_d
            self._stroke_t = t
        elif t - self._stroke_t > 20.0:
            self._stroke_best_d = None
            self._pushing = False
            self.robot.external_control = False
            self.bb.publish(self.id, "event", "push_retry",
                            {"task": self.task["id"],
                             "box_goal_distance": round(new_d, 2)})
            return

    def _build_push_obs(self, st, box, goal):
        """The push env's exact 49-dim observation from the current state."""
        import torch as _torch
        import genesis as _gs
        from genesis.utils.geom import inv_quat as _iq, transform_by_quat as _tbq
        robot = self.robot
        bt = _torch.tensor(st["quat"], dtype=_gs.tc_float, device=_gs.device).unsqueeze(0)
        iq = _iq(bt)
        dog = _torch.tensor([st["pos"][0], st["pos"][1], 0.0],
                            dtype=_gs.tc_float, device=_gs.device).unsqueeze(0)
        crate_t = _torch.tensor([box[0], box[1], 0.0], dtype=_gs.tc_float, device=_gs.device).unsqueeze(0)
        goal_t = _torch.tensor([goal[0], goal[1], 0.0], dtype=_gs.tc_float, device=_gs.device).unsqueeze(0)
        zz = _torch.zeros((1, 1), dtype=_gs.tc_float, device=_gs.device)
        cr = _tbq(_torch.cat([crate_t[:, :2] - dog[:, :2], zz], -1), iq)[:, :2]
        gr = _tbq(_torch.cat([goal_t[:, :2] - crate_t[:, :2], zz], -1), iq)[:, :2]
        return _torch.cat((
            robot.base_ang_vel * 0.25,
            robot.projected_gravity,
            _torch.tensor([[0.6, 0.0, 0.0]], dtype=_gs.tc_float, device=_gs.device),
            robot.dof_pos - robot.default_dof_pos,
            robot.dof_vel * 0.05,
            robot.actions,
            cr * 1.0, gr * 1.0), dim=-1)


    def _hand_push(self, st, pos, yaw, t, box, goal, box_d):
        """Fallback hand-crafted controller (used until the learned policy is plugged in)."""

        # stall back-off maneuver: reverse 1.5 s, then re-approach
        if self._backing_until is not None:
            if t < self._backing_until:
                self.robot.commands = _cmd(-0.5, 0.0, 0.0)
                return
            self._backing_until = None

        # side check: only press when the dog is on the goal-OPPOSITE side of
        # the crate (dog->box bearing roughly matches box->goal bearing)
        to_box = math.atan2(box[1] - pos[1], box[0] - pos[0])
        push_dir = math.atan2(goal[1] - box[1], goal[0] - box[0])
        press_angle = abs(((to_box - push_dir + math.pi) % (2 * math.pi)) - math.pi)

        if press_angle < math.radians(40) and math.dist((pos[0], pos[1]), box) < 0.75:
            # press: walk toward the GOAL — the crate is in between, so the
            # push emerges from contact. Slow speed = no kick-through.
            vw = st.get("lin_vel_world", [0, 0, 0])
            if math.hypot(vw[0], vw[1]) < 0.05 and box_d > 0.8:
                self._stall_since = (self._stall_since or 0.0) + 0.01
                if self._stall_since > 5.0:
                    self._backing_until = t + 1.5
                    self._stall_since = 0.0
                    self._route = None  # re-plan the arc after backing off
                    self.bb.publish(self.id, "event", "push_retry",
                                    {"task": self.task["id"], "box_goal_distance": round(box_d, 2)})
                    return
            self._goto(pos, yaw, (goal[0], goal[1]), on_arrive=lambda: None,
                       stop_radius=0.0, speed=0.2)
            return

        # wrong side: arc around the crate (radius 0.9 m, short way) to the
        # push-side approach point, then the press branch takes over
        if self._route is None or self._route_anchor is None or \
                math.dist(self._route_anchor, box) > 0.15:
            a0 = math.atan2(pos[1] - box[1], pos[0] - box[0])
            a1 = push_dir  # approach the crate FROM the push side: final arc
            # angle at which the approach point sits: opposite of push_dir
            a_target = push_dir + math.pi
            da = ((a_target - a0 + math.pi) % (2 * math.pi)) - math.pi
            n = 5
            self._route = [
                (box[0] + 0.9 * math.cos(a0 + da * k / n),
                 box[1] + 0.9 * math.sin(a0 + da * k / n))
                for k in range(1, n + 1)
            ]
            self._route.append((box[0] + 0.55 * math.cos(a_target),
                                box[1] + 0.55 * math.sin(a_target)))
            self._route_anchor = box
        while self._route and math.dist((pos[0], pos[1]), self._route[0]) < 0.4:
            self._route.pop(0)
        if self._route:
            self._goto(pos, yaw, self._route[0], on_arrive=lambda: None,
                       stop_radius=0.0, speed=0.5)


class DroneTaskAgent(TaskAgent):
    """Waypoint flight by commanding the learned hover policy's 3-D target."""

    ARRIVE_RADIUS = 0.35     # m — matches demonstrated station-keeping precision
    LOITER_S = 2.0

    def _begin(self, task: dict):
        self.waypoints = list(task.get("waypoints", [task["location"]]))
        self.phase = "fly"
        self._set_current_wp()

    def _set_current_wp(self):
        self.robot.commands = _vec3(self.waypoints[0])

    def _execute(self, t: float):
        if self.phase == "fly":
            st = self.robot.get_state()
            wp = self.waypoints[0]
            d = math.dist(st["pos"], wp)
            if d < self.ARRIVE_RADIUS:
                self.phase = "loiter"
                self._loiter_until = t + self.LOITER_S
                self.bb.publish(self.id, "event", "waypoint_reached",
                                {"task": self.task["id"], "wp": list(wp), "dist": round(d, 3)})
        elif self.phase == "loiter":
            if t >= self._loiter_until:
                self.waypoints.pop(0)
                if self.waypoints:
                    self.phase = "fly"
                    self._set_current_wp()
                else:
                    self._finish("survey_complete")


class PandaTaskAgent(TaskAgent):
    """Manipulation with HONEST state separation (advisor-mandated):

        REACHED  — fingertip midpoint within 0.10 m of the object SURFACE
                   (true AABB surface distance, not a naive center offset)
        CONTACT  — Genesis ground-truth contacts between arm links and the object
        GRASPED  — gripper closed while contact persists (parcel only: the
                   gripper stroke is ~8 cm, the 35 cm delivery crates are
                   PHYSICALLY ungraspable — documented limitation)
        LIFTED   — object center risen > 2 cm above its rest height (physics)
        PROCESSED— lift held 0.5 s

    The learned reach policy does REACHED/CONTACT. The lift is a SCRIPTED
    upward-IK primitive on top of the learned reach (lift-reward training is
    future work). Every state transition is published to the blackboard and
    recorded in the task history — nothing is inferred from proximity alone.
    """

    REACH_SURFACE = 0.10     # m, fingertip-to-surface for REACHED
    LIFT_HEIGHT = 0.02       # m, parcel rise for LIFTED
    LIFT_HOLD = 0.5          # s
    LIFT_DZ = 0.08           # scripted lift action (EE delta up, x0.05 scale)

    def _begin(self, task: dict):
        self.state = "idle"
        self._state_t = t0 = 0.0
        self._history = []
        self._lifted_since = None
        self._closing = False
        pkg_entity = task.get("pkg_entity")
        if task["type"] == "process" and pkg_entity is not None:
            self.robot.obj = pkg_entity       # inspect the delivered crate
            task["obj_half"] = (0.175, 0.175, 0.125)
        else:  # "manipulate": the graspable parcel at the station
            parcel = task.get("parcel_entity")
            if parcel is not None:
                self.robot.obj = parcel
            task["obj_half"] = (0.04, 0.015, 0.03)
            task["parcel_spawn_z"] = float(self.robot.obj.get_pos()[0, 2])
            # THE COMPLETION-CHECK KEY FIX (2026-11-11, M4 diagnosis): the
            # mission sets "parcel_start_x" while the completion check read
            # "parcel_spawn_xy" with a default that re-captured the CURRENT
            # position every call — disp was 0 forever and the task could
            # only complete via the z-drop path. Capture the spawn xy HERE.
            task["parcel_spawn_xy"] = (float(self.robot.obj.get_pos()[0, 0]),
                                       float(self.robot.obj.get_pos()[0, 1]))
        self._grasp_confirmed = False
        self._grasp_announced = False
        self.robot.external_control = True  # exclusive: no colo-tick double-apply
        self._set_state("reaching", 0.0)

    def _set_state(self, new: str, t: float):
        if new != self.state:
            self.state = new
            self._history.append({"state": new, "t": round(t, 2)})
            self.bb.publish(self.id, "event", f"manip_{new}",
                            {"task": self.task["id"], "object": type(self.robot.obj).__name__})

    # -- geometry helpers ---------------------------------------------------
    def _fingertip_mid(self):
        """True fingertip midpoint: finger link origin + R_finger @ (0,0,0.05)
        (the pad center offset from the panda.xml finger collision geoms)."""
        import torch as _torch
        from genesis.utils.geom import transform_by_quat as _tbq
        import genesis as _gs
        tips = []
        for link in (self.robot.left_finger_link, self.robot.right_finger_link):
            p = link.get_pos()
            q = link.get_quat()
            off = _tbq(_torch.tensor([[0.0, 0.0, 0.05]], dtype=_gs.tc_float, device=_gs.device), q)
            tips.append(p[:, :3] + off[:, :3])
        return (tips[0] + tips[1]) / 2

    def _surface_dist(self, tip, obj_entity, obj_half):
        """Distance from the fingertip midpoint to the object's axis-aligned
        bounding surface (per-axis AABB clamp) — the TRUE finger-to-surface
        distance, not a center-minus-constant approximation."""
        import torch as _torch
        import genesis as _gs
        c = obj_entity.get_pos()[0, :3]
        d = [max(abs(float(tip[0, i]) - float(c[i])) - obj_half[i], 0.0) for i in range(3)]
        return math.sqrt(sum(x * x for x in d))

    def _touching(self, obj_entity):
        """Ground-truth CONTACT: Genesis reports arm<->object contacts this step.

        get_contacts returns a FIXED-SIZE padded array (14 slots) that always
        has numel > 0 — slot contents include unrelated pairs (go2/floor...).
        Must apply valid_mask AND verify the pair actually involves an arm link
        and the object's link (probe-verified: slot 0 held plane<->go2 pairs)."""
        try:
            c = self.robot.entity.get_contacts(with_entity=obj_entity)
            if c is None or "link_a" not in c:
                return False
            vm = c["valid_mask"][0].bool()
            if not bool(vm.any()):
                return False
            arm_links = {l.idx for l in self.robot.entity.links}
            obj_links = {l.idx for l in obj_entity.links}
            for a, b in zip(c["link_a"][0][vm].tolist(), c["link_b"][0][vm].tolist()):
                if (a in arm_links and b in obj_links) or (a in obj_links and b in arm_links):
                    return True
            return False
        except Exception:
            return False

    def _lift_phase(self, t, obj, parcel):
        """Scripted lift: hold until the parcel rises > 2 cm, then PROCESSED."""
        if self.state not in ("lifting", "lifted"):
            return
        spawn_z = self.task.get("parcel_spawn_z")
        z = float(obj.get_pos()[0, 2])
        if z > spawn_z + self.LIFT_HEIGHT:
            if self._lifted_since is None:
                self._lifted_since = t
                self._set_state("lifted", t)
            elif t - self._lifted_since >= self.LIFT_HOLD:
                self._set_state("processed", t)
                self._finish({"states": self._history,
                              "lift_height_cm": round(100 * (z - spawn_z), 2)})
        else:
            self._lifted_since = None

    # -- execution ------------------------------------------------------------
    def _execute(self, t: float):
        if self.robot.policy is None:
            return  # hold pose until the policy is plugged in
        obj = self.robot.obj
        obj_half = self.task.get("obj_half", (0.04, 0.015, 0.03))
        parcel = self.task["type"] == "manipulate"

        if self.state in ("lifting", "lifted", "processed"):
            # the scripted lift primitive owns the arm here — running the reach
            # policy on the same ticks made the two controllers fight
            lift_action = _lift_action()
            self.robot.apply_action(lift_action, open_gripper=False)
            self._lift_phase(t, obj, parcel)
            return
        # NOTE: closing during the approach was tried and REVERTED — the policy
        # always saw open fingers (dof 0.04) in training; closed-finger joint
        # states are out of distribution and it retreats. Close only at real
        # CONTACT (mask-verified): the 100 Hz gripper traps the light parcel.
        tip = self._fingertip_mid()
        d_surf = self._surface_dist(tip, obj, obj_half)
        obs = self.robot.get_obs()
        action = self.robot.policy(obs)
        # terminal-velocity governor: the policy was trained against an
        # IMMOVABLE box, so its terminal approach slams — damp the deltas near
        # the object for a gentle, catchable contact (standard practice)
        if parcel and self.state in ("reaching",) and d_surf < 0.2:
            action = action * 0.25
        close_gripper = self._closing and parcel
        self.robot.apply_action(action, open_gripper=not close_gripper)

        touching = self._touching(obj)

        # PUSH-ALONG-SLIDE: the learned policy hovers ~0.1 m from the object
        # (its training convergence — the training box was immovable, so it
        # never learned to press), so once REACHED a scripted extension takes
        # over: press along the hand->object direction and drive the parcel to
        # whichever slide stop is opposite. Walls confine it: launch-proof.
        if (parcel and self.state in ("reached", "pushing")
                and getattr(self.robot, "manip_policy", None) is None):
            # LEGACY SCRIPTED PRESS — only when no trained manip policy exists.
            # (2026-11-11 M4 diagnosis: this path UNCONDITIONALLY returned for
            # every parcel, so the trained P1b policy below was DEAD CODE — the
            # arm pressed gently forever toward the OLD slide-era end stops
            # (x<=-1.60 / >=-1.18) that no longer exist, with no stall limit,
            # and parcel_pick could never complete: picked 0 in every run.)
            import torch as _torch
            import genesis as _gs
            tip = self._fingertip_mid()
            c = obj.get_pos()[0, :3]
            dvec = c - tip[0, :3]
            dvec = dvec / (dvec.norm() + 1e-9)  # press direction: hand -> object
            push_action = _torch.tensor(
                [[float(dvec[0]) * 0.1, float(dvec[1]) * 0.1, 0.0, 0.0, 0.0, 0.0]],
                dtype=_gs.tc_float, device=_gs.device)
            self.robot.apply_action(push_action, open_gripper=True)
            x_now = float(obj.get_pos()[0, 0])
            if x_now <= -1.60 or x_now >= -1.18:  # either end stop
                self._set_state("processed", t)
                self._finish({"states": self._history,
                              "slide_travel_m": round(abs(self._push_start_x - x_now), 3)})
            return
        # state transitions (all published; nothing inferred from proximity)
        if self.state in ("idle", "reaching") and d_surf < self.REACH_SURFACE:
            self._set_state("reached", t)
        if self.state in ("reached", "reaching") and touching:
            self._contact_t = t
            self._set_state("contact", t)

        if parcel:
            spawn_z = self.task.get("parcel_spawn_z")
            # P1b MANIPULATION POLICY: trained on the free box (reach+press+push
            # in one policy) — it drives the arm natively; the state machine
            # below verifies the physics and never infers success from proximity
            if getattr(self.robot, "manip_policy", None) is not None:
                obs = self.robot.get_obs()
                action = self.robot.manip_policy(obs)
                self.robot.apply_action(action, open_gripper=True)
                tip = self._fingertip_mid()
                d_surf = self._surface_dist(tip, obj, obj_half)
                if self.state in ("idle", "reaching") and d_surf < self.REACH_SURFACE:
                    self._set_state("reached", t)
                touching = self._touching(obj)
                if self.state in ("reached", "pushing") and touching:
                    self._set_state("contact", t)
                    self._contact_t = t
                # PRESS TIME-LIMIT: if the push stalls (the parcel jammed), back
                # off and retry — prevents the infinite press into a stopped
                # object (the contact-force NaN crash came from exactly that)
                if self.state == "pushing":
                    if not hasattr(self, "_pushing_since"):
                        self._pushing_since = t
                    elif t - self._pushing_since > 20.0:
                        self._pushing_since = t
                        self.robot.external_control = False
                        self._set_state("reaching", t)
                        self.bb.publish(self.id, "event", "manip_retry",
                                        {"task": self.task["id"]})
                        return
                # COMPLETION: the parcel pushed off the pedestal (the
                # horizontal displacement OR the z-drop — either is genuine
                # physics-verified manipulation). Without a completion check
                # the arm presses the parcel forever (the NaN crash came
                # from exactly that).
                p3 = obj.get_pos()[0, :2]
                sx, sy = self.task.get("parcel_spawn_xy", (float(p3[0]), float(p3[1])))
                disp = math.hypot(float(p3[0]) - sx, float(p3[1]) - sy)
                z_now = float(obj.get_pos()[0, 2])
                if disp >= 0.08 or z_now < 0.2:
                    self._set_state("processed", t)
                    self._finish({"states": self._history,
                                  "parcel_displacement_m": round(disp, 3),
                                  "fell_off_pedestal": z_now < 0.2})
                return
            # honest failure: the parcel fell off its perch (unreachable — the
            # policy's training distribution has the box at pedestal height)
            if z_now < spawn_z - 0.25:
                self._set_state("lost", t)
                self._finish({"states": self._history,
                              "note": "parcel knocked off perch; pick failed honestly"})
                return
            # contact -> close IMMEDIATELY (the free parcel escapes otherwise)
            if self.state in ("reached", "reaching") and touching:
                self._closing = True
                self._grasp_t = t
                self._set_state("grasped", t)
                self._set_state("lifting", t)  # scripted lift owns the arm now
            # during lift: confirm the grasp held (contact persists while closed)
            if self.state in ("lifting", "lifted"):
                if touching:
                    if not self._grasp_confirmed:
                        self._grasp_confirmed = True
                        self._grasp_hold_t = t
                    if self._grasp_confirmed and t - self._grasp_hold_t > 0.3 and not self._grasp_announced:
                        self._grasp_announced = True
                        self.bb.publish(self.id, "event", "grasp_confirmed",
                                        {"task": self.task["id"]})
                else:
                    self._grasp_confirmed = False
            self._lift_phase(t, obj, parcel)
            return

        # big-crate inspection: REACHED+CONTACT is the physical maximum
        # (gripper stroke 0.08 m < crate 0.35 m — GRASPED is impossible)
        if self.state == "contact" and t - self._contact_t > 0.2:
            self._finish({"states": self._history,
                          "note": "contact inspection complete; grasping physically "
                                  "impossible (gripper stroke 0.08 m < crate 0.35 m)"})


def _lift_action():
    import torch as _torch
    import genesis as _gs
    return _torch.tensor([[0.0, 0.0, 0.08, 0.0, 0.0, 0.0]], dtype=_gs.tc_float, device=_gs.device)


def _tip_x3():
    import genesis as _gs
    return _gs.tc_float  # placeholder replaced below by agent-side computation


def _push_action():
    # Scripted EE extension along -x (away from the arm, down the slide).
    import torch as _torch
    import genesis as _gs
    return _torch.tensor([[-0.1, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=_gs.tc_float, device=_gs.device)



# ---------------------------------------------------------------- utils


def _cmd(vx, vy, wz):
    import torch  # local import keeps module import cheap

    from colo_env import _ensure_genesis_init  # noqa: F401  (device setup)
    import genesis as gs
    return torch.tensor([vx, vy, wz], dtype=gs.tc_float, device=gs.device).unsqueeze(0)


def _vec3(xyz):
    import torch

    import genesis as gs
    return torch.tensor(xyz, dtype=gs.tc_float, device=gs.device).unsqueeze(0)
