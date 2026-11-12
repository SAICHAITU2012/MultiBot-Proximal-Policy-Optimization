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
        self._last_push_t = 0.0
        self._push_waypoints = None
        # closed-loop push state (see the CRATE PUSH section below)
        self._best_box_d = None
        self._progress_t = None
        self._route_t = -1e9
        self._route_progress_t = None
        self._best_route_d = None
        self._uncouple_until = 0.0
        self._settle_until = None
        self._shove_t0 = 0.0
        self._shove_d0 = 0.0
        self._reset_push_loop()

    def _execute(self, t: float):
        st = self.robot.get_state()
        pos, yaw = st["pos"], self._yaw_from_quat(st["quat"])
        # any task carrying a live box position is a push/fetch: go behind the
        # box, then drive it to the goal (plain goto only when there's no box)
        if "box_pos_fn" in self.task:
            self._push(st, pos, yaw, t)
        else:
            self._goto(pos, yaw, self.goal, on_arrive=lambda: self._finish("arrived"))

    def _goto(self, pos, yaw, target_xy, on_arrive, stop_radius=None, speed=0.5,
              turn_first=True):
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
        v = max(v, self.MIN_V)         # ...but NEVER below the nav policy's
        # low-speed deadband (measured: <0.2 m/s commands achieve ~0.006 m/s).
        # Gentle deceleration is a freeze; positioning is bang-bang instead.
        if turn_first and abs(ye) > 0.50:
            # SPIN IN PLACE FIRST. Turning at a near-maximal yaw rate WHILE
            # translating is what tipped the dog over during re-approach
            # (measured: 6 falls/package, every one at z 0.06-0.11 while
            # align was commanding vx~0.3 with wz near the +-1.0 clamp). A
            # stationary spin at full magnitude is the one turn primitive
            # measured stable (0.74 rad/s), so large errors are fixed first,
            # with no translation, exactly like the face phase.
            self.robot.commands = _cmd(0.0, 0.0, 1.0 if ye > 0 else -1.0)
            return
        vx = max(-0.5, min(1.0, v * math.cos(ye)))
        vy = max(-0.5, min(0.5, v * math.sin(ye)))
        wz = max(-0.55, min(0.55, 0.9 * ye))   # capped while translating
        self.robot.commands = _cmd(vx, vy, wz)

    # ===================================================================
    # CRATE PUSH — coarse-to-fine, closed-loop (the M7 endgame fix)
    # ===================================================================
    # Physics measured in the mission scene (probe_cmd_modes.py; 15 cm / 7 kg
    # canonical crate; rigid dt 0.01):
    #     vx  0.15 for 4 s   -> crate moves 0.002 m   (a slow lean does NOTHING)
    #     vx  0.30 for 3 s   -> crate moves 0.004 m
    #     vx  0.60 for 2 s   -> crate moves 0.289 m   (the shove that works)
    #     vx -0.30 for 1.5 s -> dog backs away 0.456 m (reverse is NOT dead)
    #     vy +-0.4 for 1.5 s -> dog strafes     0.41-0.59 m
    #     wz +-0.8 for 1.5 s -> 0.34 rad/s of yaw achieved
    #
    # CONCLUSION: the terminal 10 cm was never a policy-capability gap. The
    # learned pusher is the ONLY thing that moves a 7 kg crate, and it does so
    # reliably. What was missing is the CONTROL LAYER around it:
    #   (1) the old code ran the pusher OPEN-LOOP for 20-45 s, so the dog
    #       overran the crate, ended up between crate and zone, and shoved it
    #       AWAY — measured: 0.66 m -> 0.70 m of drift over 1150 s of
    #       "pushing" in mission_20261111_225336;
    #   (2) the wrong-side test only ran BETWEEN strokes, so a misaligned
    #       stroke was never aborted mid-flight;
    #   (3) there was no back-off primitive, so a dog touching the crate on
    #       the wrong side was wedged and orbited it forever (the observed
    #       permanent grind at 0.66 m).
    # THE FIX: a nudge cycle — uncouple, align precisely on the crate->zone
    # line, face the crate, settle, apply ONE bounded shove, then measure
    # again. Same learned skill, same physics; the loop supplies the feedback
    # the skill never had. Nothing about the acceptance gates is changed:
    # delivery is still the real crate reaching the real 0.55 m zone.
    HAUL_R = 0.9          # m — inside this box distance a stroke is "terminal"
    LAUNCH_D = 0.95      # m — run-up length used by the retreat phase
    STANDOFF = 0.65       # m — park behind the crate at the TRAINING crate
    #   spawn distance (go2_push_env: crate_spawn 0.65 m). Standing further
    #   back wastes a stroke on the run-up; standing closer makes the in-place
    #   facing turn sweep the dog's body into the crate, which locks the turn
    #   (measured: 177 s of a full-magnitude spin command producing 0.6 rad).
    CONTACT_D = 0.50      # m — dog->crate centre distance that means contact
    COUPLED_D = 0.60      # m — the reverse (a rescue, not a routine step) ends
    #   here, inside the alignment band so no lateral re-approach is needed.
    PERP_TOL = 0.20       # m — lateral tolerance on the crate->zone line.
    #   CALIBRATED against the nav primitive, not against the ideal: asked for
    #   0.15 m the dog overshoots each correction by 0.1-0.2 m (momentum carries
    #   it after "arrived") and limit-cycles forever — measured 80 s of align for
    #   a single stroke. 0.20 m is the tightest band the nav policy can actually
    #   hold, and it is still well inside the dog's ~0.25 m body width, so the
    #   contact remains square on the crate face.
    ALONG_MIN = -0.45     # m — "behind the crate": 0.45 m of run-up.
    #   MEASURED, and it settles a design question: strokes from a 0.65-0.76 m
    #   stand-off travel 0.35-0.38 m, but strokes re-fired from 0.18-0.30 m (a
    #   no-back-off "ratchet") travel 0.00-0.06 m and even netted -0.02 m. The
    #   learned pusher needs the run-up it was trained with. So the cycle must
    #   back off between strokes — the cost to minimise is the back-off, not the
    #   stand-off.
    FACE_ENTER = 0.40     # rad — turn if the heading error exceeds this
    FACE_EXIT = 0.25      # rad — stop turning below this. HYSTERESIS, and it
    #   must be WIDER than the settle's yaw disturbance: snapping the joints to
    #   the default pose rotates the body by ~0.15-0.25 rad, so a 0.15 rad exit
    #   band produced a permanent face<->settle livelock (measured: 190 s, one
    #   stroke, dog parked 0.58 m behind the crate perfectly on the line). A
    #   15 cm crate tolerates 0.40 rad of approach incidence before the contact
    #   becomes glancing.
    SETTLE_S = 0.4        # s — default-joint hold before a stroke
    STROKE_MAX_S = 6.0    # s — hard cap on one shove. A stroke is a HAUL, not a
    #   nudge: in contact the pusher moves the crate at ~0.145 m/s (0.289 m in
    #   2.0 s, probe_cmd_modes), so 6 s of unbroken contact is ~0.8 m for free.
    #   Stopping every 2 s to re-approach costs 8-15 s of fragile manoeuvring
    #   for each 0.25 m gained.
    PHASE_MAX_S = 8.0     # s — a phase that cannot finish re-plans (no locks)
    MIN_V = 0.30          # m/s — THE NAV POLICY'S LOW-SPEED DEADBAND floor.
    #   Measured (probe_cmd_modes.py, free floor, 2 s): commanded 0.06/0.10/
    #   0.15 m/s achieve 0.005/0.006/0.008 m/s — i.e. a gently decelerating
    #   controller is indistinguishable from a frozen dog. 0.20 -> 0.078
    #   (39%), 0.25 -> 0.164 (66%), 0.30 -> 0.268 (89%), 0.40 -> 0.453 (113%).
    #   Every historical "the dog freezes / parks / never re-contacts" report
    #   is this deadband. Translation therefore uses a FLOOR of 0.30 m/s and
    #   stops hard inside the tolerance (bang-bang positioning).
    FINISH_D = 0.50       # m — fetch done here (delivery fires at 0.55, so
    #                        this keeps a 5 cm safety margin inside the zone)

    # -- small helpers -----------------------------------------------------
    @staticmethod
    def _wrap(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))

    @staticmethod
    def _seg_dist(p, a, b) -> float:
        """Distance from point p to the segment a-b."""
        ax, ay = a
        bxx, byy = b
        dx, dy = bxx - ax, byy - ay
        L2 = dx * dx + dy * dy
        if L2 < 1e-9:
            return math.dist(p, a)
        t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
        return math.dist(p, (ax + t * dx, ay + t * dy))

    def _crate(self):
        """Live crate centre + yaw from the task's accessors."""
        raw = self.task["box_pos_fn"]()
        while isinstance(raw[0], (list, tuple)):
            raw = raw[0]
        box = (float(raw[0]), float(raw[1]))
        yaw = None
        qfn = self.task.get("box_quat_fn")
        if qfn is not None:
            q = qfn()  # wxyz
            yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                             1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))
        return box, yaw

    def _push_track(self, pos, box) -> dict:
        """The crate->zone line frame: u (crate->zone unit), along/perp of the
        dog relative to it, and the distances the state machine decides on."""
        gx, gy, = self.goal[0], self.goal[1]
        cx, cy = box[0] - gx, box[1] - gy
        L = math.hypot(cx, cy) + 1e-9
        u = (-cx / L, -cy / L)                       # crate -> zone
        wx, wy = pos[0] - box[0], pos[1] - box[1]    # crate -> dog
        return {
            "u": u,
            "box_d": L,
            "d_crate": math.hypot(wx, wy),
            "along": wx * u[0] + wy * u[1],          # < 0 = behind the crate
            "perp": abs(-wx * u[1] + wy * u[0]),     # lateral offset from the line
            "stand": (box[0] - u[0] * self.STANDOFF, box[1] - u[1] * self.STANDOFF),
        }

    def _aligned(self, g: dict) -> bool:
        """Good enough to shove straight at the zone: behind the crate on its
        centre line, at the training-distribution stand-off (the pusher trained
        with a 0.65 m crate spawn; 0.8 m back produces no contact inside one
        stroke, and inside the contact patch is the "slow lean" that moves
        nothing — measured 0.002 m)."""
        return (g["along"] <= self.ALONG_MIN
                and g["perp"] <= self.PERP_TOL
                and 0.55 <= g["d_crate"] <= 0.85)

    def _speed(self, st) -> float:
        lv = st.get("lin_vel_world") or [0.0, 0.0, 0.0]
        return math.hypot(lv[0], lv[1])

    # -- the push task entry point ----------------------------------------
    def _push(self, st, pos, yaw, t):
        """Drive the crate into the zone with a repeating CHARGE.

        The design is deliberately only two phases:

            retreat  — get a clear run-up behind the crate, roughly on the
                       crate->zone line (coarse: +-0.30 m)
            charge   — apply the learned pusher CONTINUOUSLY until the crate
                       stops responding, then retreat and charge again

        Why no align/face/settle/shove sequence: all four were implemented and
        measured, and each is defeated by a primitive that cannot do fine
        positioning. Measured on this robot, in this scene:

          * the nav policy overshoots every correction by 0.1-0.2 m (momentum
            carries it past "arrived"), so a 0.15 m lateral tolerance
            limit-cycles forever — 80 s of align for one stroke;
          * the "in-place" turn translates the dog 0.1-0.2 m, so turning costs
            position;
          * snapping to the default joints before a stroke rotates the body by
            0.15-0.25 rad, which livelocked face<->settle for 190 s.

        The pusher itself does not need any of that precision. It is a learned
        closed-loop controller: given a crate a short distance ahead and a goal
        direction, it walks in and drives the crate. A 0.9 m run-up both absorbs
        +-0.3 m of lateral error (the dog corrects on the way in) and is what
        produced the only large crate movements measured here (0.91 m in one
        stroke). So the control layer's job is reduced to the one thing it can
        do reliably — put the dog 0.9 m behind the crate — and the policy's job
        is left alone.

        Acceptance gates are untouched: delivery is still the real crate
        reaching the real 0.55 m zone.
        """
        box, _box_yaw = self._crate()
        box_d = math.dist(box, (self.goal[0], self.goal[1]))
        if box_d < self.FINISH_D:
            self._end_push({"box_goal_distance": round(box_d, 3),
                            "mode": "learned" if self.robot.push_policy is not None else "hand"})
            return
        if self.robot.push_policy is None:
            self._hand_push(st, pos, yaw, t, box, self.goal, box_d)
            return

        g = self._push_track(pos, box)

        # -- honest progress watchdog: a fetch that makes no progress for 120 s
        #    re-plans from scratch (it never abandons the task).
        if self._best_box_d is None or box_d < self._best_box_d - 0.05:
            self._best_box_d = box_d
            self._progress_t = t
        elif self._progress_t is not None and t - self._progress_t > 120.0:
            self._progress_t = t
            self._best_box_d = box_d
            self._reset_push_loop()
            self.bb.publish(self.id, "event", "push_replan",
                            {"task": self.task["id"], "box_goal_distance": round(box_d, 2)})

        if getattr(self, "_phase", None) == "charge":
            self._charge_tick(st, box, box_d, g, t)
            return

        # ---- retreat: build a run-up -----------------------------------
        # Heading gate 0.60 rad, and it must be enforced: a dog that arrives at
        # the launch point facing sideways never opens the gate otherwise
        # (measured: 190 s parked, staring east, crate behind it). But the
        # in-place turn moves the dog 0.1-0.2 m, so turning and lateral
        # positioning FIGHT. Handled by priority in _retreat_tick: the larger
        # residual wins and each rule yields the other room to finish.
        ye = self._wrap(math.atan2(g["u"][1], g["u"][0]) - yaw)
        ready = (g["along"] <= -0.55
                 and g["perp"] <= 0.30
                 and 0.55 <= g["d_crate"] <= 1.05
                 and abs(ye) <= 0.60)
        if ready:
            self._enter("charge", t)
            self._shove_t0 = t
            self._shove_d0 = box_d
            self._prog_d = None
            self._prog_t = t
            self._high_since = None
            self._pushing = True
            self.robot.external_control = True
            self.bb.publish(self.id, "event", "push_stroke_start",
                            {"task": self.task["id"],
                             "box_goal_distance": round(box_d, 2),
                             "perp": round(g["perp"], 3),
                             "along": round(g["along"], 2)})
            return
        self._retreat_tick(pos, yaw, g, t)

    def _charge_tick(self, st, box, box_d, g, t):
        """One continuous application of the learned pusher, with the abort
        conditions that the historical open-loop code was missing."""
        self.robot.external_control = True
        self._pushing = True
        aim = (self.goal[0], self.goal[1])
        zg = math.dist(box, aim)
        vgoal = ((box[0] + (aim[0] - box[0]) / zg * 2.2,
                  box[1] + (aim[1] - box[1]) / zg * 2.2) if zg > 1e-6 else aim)
        self.robot.apply_action(self.robot.push_policy(self._build_push_obs(st, box, vgoal)))

        travelled = self._shove_d0 - box_d
        # STALL DETECTOR — the piece the old open-loop code never had. Driving
        # the crate moves it ~0.145 m/s; when the contact goes glancing or the
        # crate wedges it stops dead, and an undetected stall means the dog
        # simply drives THROUGH/OVER the crate (measured: d_crate 0.09 m, base
        # z 0.40 m, then a fall).
        if self._prog_d is None or box_d < self._prog_d - 0.015:
            self._prog_d = box_d
            self._prog_t = t
        stalled = (t - self._prog_t) > 0.9
        # CLIMB GUARD: the crate is only 15 cm tall; a base well above walking
        # height means the dog is riding up it, not pushing it. The threshold has
        # to clear the NORMAL walking height, which reaches 0.30-0.36 m under
        # acceleration — a 0.33 m threshold aborted every charge within a second
        # (measured: 4 strokes, 0.001 m of crate travel, dog parked on the crate).
        # It must also PERSIST, so a gait transient cannot trip it.
        z_now = float(st["pos"][2])
        if z_now > 0.40:
            if self._high_since is None:
                self._high_since = t
        else:
            self._high_since = None
        climbing = (self._high_since is not None and t - self._high_since > 0.25)
        done = (box_d < self.FINISH_D
                or travelled >= 0.90
                or stalled
                or climbing
                or t - self._shove_t0 >= 6.0
                or g["along"] > -0.10)
        if done:
            self._pushing = False
            self.robot.external_control = False
            self._enter("retreat", t)
            self.bb.publish(self.id, "event", "push_stroke_end",
                            {"task": self.task["id"],
                             "travelled": round(travelled, 3),
                             "box_goal_distance": round(box_d, 2),
                             "secs": round(t - self._shove_t0, 2),
                             "stalled": bool(stalled),
                             "climbing": bool(climbing)})

    def _enter(self, phase, t):
        self._phase = phase
        self._phase_t0 = t

    def _reset_push_loop(self):
        self._phase = "retreat"
        self._phase_t0 = -1e9
        self._route = None
        self._route_t = -1e9
        self._best_route_d = None
        self._settle_until = None
        self._uncouple_until = 0.0
        self._prog_d = None
        self._prog_t = 0.0
        self._high_since = None
        self._jam_t0 = None
        self._jam_mode = 0
        self._jam_win = -1
        self._jam_d = None
        self._orbit_target = None
        self._orbit_t = -1e9
        self._esc_d = None
        self._esc_t = 0.0
        self._esc_until = None
        self._esc_sign = 1.0
        self._pushing = False
        self.robot.external_control = False

    def _retreat_tick(self, pos, yaw, g, t):
        """Recover a run-up behind the crate. Three rules, in order:

        1. WRONG SIDE (the dog is between the crate and the zone): it must go
           around, so orbit a ring at ~0.95 m around the crate — the only case
           that needs routing at all.
        2. TOO CLOSE: back off ALONG THE DOG'S OWN AXIS, away from the crate.
           Pure axial motion is the only movement near a 15 cm crate that cannot
           nudge it: the body sweeps the crate sideways on any curved path
           (measured — steering away from contact shoved the crate 0.19 m
           FURTHER from the zone while the dog crawled at 0.02 m/s).
        3. CLEAR: turn to face +u (the crate->zone direction) in place, and the
           charge gate opens.

        The dog's own push axis IS the crate->zone line (it just pushed along
        it), so an axial back-off both clears the crate and preserves the line —
        no lateral re-approach, which is what kept the old cycle stuck.
        """
        u = g["u"]
        bx = g["stand"][0] + u[0] * self.STANDOFF      # crate centre
        by = g["stand"][1] + u[1] * self.STANDOFF
        ye = self._wrap(math.atan2(u[1], u[0]) - yaw)  # heading error vs +u
        d = g["d_crate"]
        self.robot.external_control = False
        self._pushing = False

        # PATH CLEARANCE decides the route. A straight line to the launch point
        # is only usable if it passes clear of the crate; otherwise the dog must
        # orbit. Testing `along > 0.10` ("in front") instead was wrong: from
        # BESIDE the crate (along ~0, perp ~0.8 — the state both wrong-side
        # cases settle into) the straight path runs THROUGH the crate, so the dog
        # leaned on it and pushed it away at 0.02 m/s while the geometry never
        # changed.
        # ---- UNSTICK: a fixed 4-mode escape cycle ------------------------
        # Measured failure: the dog ends up body-on-crate (d_crate 0.3-0.45) and
        # STAYS there — 2240 s of a 2400 s mission parked at 0.42 m, and 200 s in
        # isolation, while every geometric rule reported "nothing to do here".
        # A single escape mode is not reliable (reverse tangles the front legs on
        # the crate; strafe alone can be blocked; a stationary turn is physically
        # blocked by the crate), so all four are cycled on a timer: pull off,
        # strafe one way, walk forward OVER the low crate, strafe back. At least
        # one mode always clears the jam, and cycling costs only 3.2 s.
        if d < 0.92:
            esc = ((-0.30, 0.0, 0.0), (0.0, 0.45, 0.0),
                   (0.35, 0.0, 0.0), (0.0, -0.45, 0.0))
            if self._jam_t0 is not None:
                # HILL-CLIMB over the four modes: hold whichever mode is gaining
                # ground, advance only when it stalls. A fixed rotation wastes 3
                # of every 4 windows (the forward mode drives the dog back onto
                # the crate right after the reverse mode pulled it off), which is
                # why the between-crate-and-zone case never escaped: its reverse
                # needs 0.8 s longer than one window to reach clearance.
                win = int((t - self._jam_t0) / 0.8)
                if win != self._jam_win:
                    if self._jam_d is not None and d > self._jam_d + 0.02:
                        pass                                    # keep the mode
                    else:
                        self._jam_mode = (self._jam_mode + 1) % 4
                    self._jam_win = win
                    self._jam_d = d
                    if win > 12:                            # gave up, re-route
                        self._jam_t0 = None
                        self._jam_mode = 0
                        self._jam_d = None
                if self._jam_t0 is not None:
                    self.robot.commands = _cmd(*esc[self._jam_mode])
                    return
            # only enter the cycle when CLOSE AND NOT MAKING PROGRESS — entering
            # on distance alone hijacks a dog that is legitimately orbiting in
            # toward the launch point, and the two then cancel out (measured: the
            # orbit cases that had just been fixed regressed to failure).
            if self._esc_d is None or d > self._esc_d + 0.04:
                self._esc_d, self._esc_t = d, t
            if t - self._esc_t > 1.0:
                self._jam_t0 = t
                self._jam_mode = 0
                self._jam_win = -1
                self._jam_d = None
                self._esc_d = None
                self.robot.commands = _cmd(*esc[0])
                return
        else:
            self._jam_t0 = None
            self._esc_d = None

        launch = (bx - u[0] * self.LAUNCH_D, by - u[1] * self.LAUNCH_D)
        if self._seg_dist((bx, by), (pos[0], pos[1]), launch) < 0.60:
            # ORBIT: step around a ring, at most 60 deg of bearing per tick,
            # with each step LATCHED until it is reached. Recomputing the ring
            # point every tick from the live bearing makes the target run away
            # from the dog (it is always 60 deg off the current angle), and the
            # dog random-walked around the crate without ever getting behind it
            # (measured: d_crate oscillating 0.42-1.08 m with along stuck near 0).
            if (self._orbit_target is None
                    or math.dist((pos[0], pos[1]), self._orbit_target) < 0.12
                    or t - self._orbit_t > 3.0):
                rx, ry = pos[0] - bx, pos[1] - by
                dd = math.hypot(rx, ry) + 1e-9
                cur = math.atan2(ry, rx)
                tgt = math.atan2(-u[1], -u[0])
                step = max(-1.05, min(1.05, self._wrap(tgt - cur)))
                a = cur + step
                r = max(0.78, min(dd, self.LAUNCH_D))
                self._orbit_target = (bx + r * math.cos(a), by + r * math.sin(a))
                self._orbit_t = t
            target = self._orbit_target
            # turn_first=False is ESSENTIAL here: a stationary turn with the body
            # against the crate is physically blocked by it (the documented
            # 0.6-rad-in-177-s lock), so an orbit that insists on facing first
            # never moves. Translating and turning together slides the dog off
            # the crate while it swings around.
            self._goto(pos, yaw, target, on_arrive=lambda: None,
                       stop_radius=0.09, speed=0.40, turn_first=False)
            return

        # A COMPLETE PARTITION, not two competing gates: the previous form was
        # "goto if perp>0.20 and |ye|<=0.60", "turn if |ye|>0.60 and perp<=0.55",
        # which left a hole at |ye|>0.60 AND perp>0.55 where NEITHER rule applied
        # and the dog sat in it indefinitely (measured: 25 s frozen at perp 0.55,
        # ye 0.9). Position first, always; heading only in the space position has
        # already fixed.
        if g["perp"] > 0.25 or d > 1.02:
            # turn_first=False: this policy is HOLONOMIC (forward + strafe), so it
            # closes lateral and yaw error simultaneously. Forcing a stationary
            # turn first is what made recovery and aiming mutually exclusive.
            self._goto(pos, yaw, launch, on_arrive=lambda: None,
                       stop_radius=0.09, speed=0.40, turn_first=False)
            return

        if abs(ye) > 0.60:
            self.robot.commands = _cmd(0.0, 0.0, 1.0 if ye > 0 else -1.0)
            return

        self.robot.commands = _cmd(0.0, 0.0, 0.0)

    def _end_push(self, result):
        self.robot.commands = _cmd(0.0, 0.0, 0.0)
        self.robot.external_control = False
        self._pushing = False
        self._reset_push_loop()
        self._finish(result)

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
