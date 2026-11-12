#!/usr/bin/env python3
"""probe_camera.py — render real frames from the mission scene to verify framing
and measure per-frame render cost before a full recording."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "team"))
import numpy as np
from PIL import Image

import colo_env

OUT = Path(__file__).resolve().parent / "camera_probe"
OUT.mkdir(exist_ok=True)

team = colo_env.CoLocatedTeam(
    go2_ckpt=str(Path(__file__).resolve().parent.parent / "logs/go2-nav/model_800.pt"),
    drone_ckpt=str(Path(__file__).resolve().parent.parent / "logs/drone-hover-s4/model_1600.pt"),
    extra_boxes=[(1.2, 1.2, 0.08), (2.0, 0.3, 0.08)],
    record_camera=True,
)
assert team.camera is not None, "camera not created"

t0 = time.time()
for _ in range(50):
    team.step()
a = time.time() - t0
n = 0
t1 = time.time()
for _ in range(5):
    f = team.render_frame()
    n += 1
per = (time.time() - t1) / n
print(f"[probe] 50 steps (0.5 s sim) = {a:.2f}s | frame {f.shape} {f.dtype} | render {per*1000:.0f} ms/frame")
Image.fromarray(f).save(OUT / "frame_t050.png")

t2 = time.time()
for _ in range(3900):   # +39 s sim
    team.step()
print(f"[probe] 3900 steps = {time.time()-t2:.1f}s")
Image.fromarray(team.render_frame()).save(OUT / "frame_t40.png")
print("[probe] frames ->", OUT)
