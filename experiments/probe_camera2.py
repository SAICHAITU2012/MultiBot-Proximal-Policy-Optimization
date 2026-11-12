import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "team"))
import colo_env
team = colo_env.CoLocatedTeam(
    go2_ckpt=str(Path(__file__).resolve().parent.parent / "logs/go2-nav/model_800.pt"),
    drone_ckpt=str(Path(__file__).resolve().parent.parent / "logs/drone-hover-s4/model_1600.pt"),
    extra_boxes=[(1.2, 1.2, 0.08), (2.0, 0.3, 0.08)], record_camera=True)
for _ in range(50): team.step()
for i in range(4): team.render_frame()          # warm-up (shader compile)
t = time.time(); n = 10
for _ in range(n):
    team.step(); team.render_frame()
print(f"[probe2] steady-state {team.render_frame().shape}: {(time.time()-t)/n*1000:.0f} ms per (step+render)")
