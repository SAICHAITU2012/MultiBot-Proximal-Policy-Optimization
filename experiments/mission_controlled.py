#!/usr/bin/env python3
"""
mission_controlled.py — the controlled mission test per the advisor's protocol:
the exact 15 cm crate config + the current M0b checkpoint (model_1200), with
full measurement instrumentation: the crate trajectory (5 s sampling), the
strokes, the retries, the falls, the net displacement, the delivery state.
No training/geometry changes during the test.
"""

import sys, math
from pathlib import Path

TL = Path("/Users/saichaitu/Desktop/Genesis/team_lab")
sys.path.insert(0, str(TL)); sys.path.insert(0, str(TL / "experiments")); sys.path.insert(0, str(TL / "team"))

import torch
import genesis as gs

import mission as M

gs.init(backend=gs.cpu, logging_level="error", performance_mode=True)
m = M.Mission(n_packages=1, planner_name="rule", max_steps=10)
crate = m.packages[0]["entity"]
go2 = m.team.robots["go2"]
ag = m.agents["go2"]
m._lp = -1e9

x_spawn = float(crate.get_pos()[0, 0])
print(f"[TEST] crate spawn x={x_spawn:.3f} | policy: go2-push-m0b/model_1200 | 15 cm crate, 7 kg")

samples = []
falls = 0
fall_state = False
strokes = retries = 0
last_event_count = 0

with torch.inference_mode():
    for step in range(60000):  # 600 s
        t = m.team.sim_time
        m.bb.set_clock(t)
        if t - m._lp >= 4.0:
            m._lp = t; m._plan_tick(t)
        m.team.step()
        for a in m.agents.values():
            a.update(t)
        m._check_deliveries()

        # instrumentation
        strokes = sum(1 for ev in m.bb.history if ev["type"] == "push_stroke_complete")
        retries = sum(1 for ev in m.bb.history if ev["type"] == "push_retry")
        z = float(go2.base_pos[0, 2])
        if z < 0.15 and not fall_state:
            fall_state = True; falls += 1
        elif z > 0.25 and fall_state:
            fall_state = False  # recovered (upright again)
        if step % 500 == 0:  # every 5 s
            dp = [float(v) for v in go2.base_pos[0].cpu().numpy().flatten()]
            cp = [float(v) for v in crate.get_pos().cpu().numpy().flatten()]
            samples.append((t, cp[0], cp[1], z))
            print(f"[TEST] t={t:6.1f}s crate=({cp[0]:.3f},{cp[1]:.3f}) dog=({dp[0]:.2f},{dp[1]:.2f}) dog_z={z:.3f} "
                  f"strokes={strokes} retries={retries} falls={falls}")

        if m._mission_complete():
            break

L = TL / "logs"
art = sorted(L.glob("mission_*"), key=lambda p: p.stat().st_mtime)[-1]
metrics = (art / "metrics.txt").read_text() if (art / "metrics.txt").exists() else "n/a"
print(f"\n[TEST] === FINAL (artifacts: {art.name}) ===")
print(metrics)
cp_final = [float(v) for v in crate.get_pos().cpu().numpy().flatten()]
net = math.hypot(cp_final[0] - x_spawn, cp_final[1] - float(m.packages[0]["spawn"][1]))
print(f"[TEST] NET crate displacement from spawn: {net:.3f} m (spawn x={x_spawn:.3f}, final x={cp_final[0]:.3f})")
print(f"[TEST] total strokes: {strokes} | retries: {retries} | falls: {falls}")
print(f"[TEST] delivered: {m.metrics['delivered']} | strokes total: {strokes}")
