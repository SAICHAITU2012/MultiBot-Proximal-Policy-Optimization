#!/usr/bin/env python3
"""safety.py — M2.2 hard coordinator constraints.

The gate a task request must pass before a robot is allowed to act. Two rules:

  R1 (proximity)  a task that moves robot A may not START while A is in an
                  active proximity conflict with another robot.
  R2 (yielding)   a blocked robot may only proceed after the conflict is
                  RESOLVED — either the mobile partner yields (drone climbs) or,
                  if the blocker cannot move (the standing G1), the requester
                  re-routes around it.

Enforcement is structural, not by convention: `request()` returns a token and
`execute()` refuses any token that was not granted, so a blocked action cannot
be executed by accident. Every request and its verdict is written to an audit
log, which is what the tests assert on.
"""
from __future__ import annotations


class Blocked(Exception):
    pass


class ConstraintLayer:
    def __init__(self, proximity, verbose=False):
        self.prox = proximity
        self.verbose = verbose
        self.audit: list[dict] = []
        self._granted: set[int] = set()
        self._blocked: set[int] = set()
        self._n = 0

    # -- request / verdict ----------------------------------------------
    def request(self, robot: str, action: str, conflict=None, **detail) -> int:
        """Ask permission for `robot` to start `action`. Returns a token.

        `conflict` = partners the action would run into (look-ahead). When not
        given, the instantaneous proximity state is used."""
        self._n += 1
        tok = self._n
        if conflict is None:
            conflict = sorted({o for pair in self.prox.partner_of(robot) for o in pair if o != robot})
        verdict, reason = "ALLOW", ""
        if conflict:
            verdict = "BLOCK"
            reason = "would enter " + ",".join(conflict) + " safety envelope"
        self.audit.append({"n": tok, "robot": robot, "action": action,
                           "verdict": verdict, "reason": reason, **detail})
        (self._granted if verdict == "ALLOW" else self._blocked).add(tok)
        if self.verbose and verdict == "BLOCK":
            print(f"    [safety] BLOCK {robot}/{action}: {reason}")
        return tok

    def execute(self, tok: int, executed_log: list):
        """Record an executed action; refuse tokens without a grant."""
        if tok not in self._granted:
            raise Blocked(f"action token {tok} was not granted and must not execute")
        executed_log.append(tok)

    # -- yielding strategies --------------------------------------------
    def yield_action(self, robot: str, partner: str) -> str:
        """What the blocked robot (or its mobile partner) should do."""
        if partner == "g1":       # the humanoid stands still: go around it
            return "reroute"
        return "wait_and_altitude" if robot == "drone" else "wait"

    # -- stats -----------------------------------------------------------
    def stats(self) -> dict:
        blocked = [a for a in self.audit if a["verdict"] == "BLOCK"]
        return {
            "requests": len(self.audit),
            "allowed": len(self.audit) - len(blocked),
            "blocked": len(blocked),
            "blocked_by_pair": sorted({a["reason"].split("with ")[-1] for a in blocked}),
            "executed": len(self._granted & set()),
        }
