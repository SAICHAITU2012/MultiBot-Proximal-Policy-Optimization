import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "team"))
from PIL import Image
import colo_env
L = Path(__file__).resolve().parent.parent / "logs"
CAND = {
 "A": ((0.30, -3.60, 4.40), (-0.10, 1.10, 0.30), 52),
 "B": ((2.90, -2.40, 3.90), (-0.40, 1.15, 0.30), 58),
 "C": ((0.20, -4.10, 5.00), (-0.10, 1.10, 0.25), 50),
}
which = sys.argv[1]
pos, look, fov = CAND[which]
colo_env.REC_CAM_POS, colo_env.REC_CAM_LOOKAT, colo_env.REC_CAM_FOV = pos, look, fov
team = colo_env.CoLocatedTeam(go2_ckpt=str(L/"go2-nav/model_800.pt"),
    drone_ckpt=str(L/"drone-hover-s4/model_1600.pt"),
    extra_boxes=[(1.2, 1.2, 0.08), (2.0, 0.3, 0.08)], record_camera=True)
for _ in range(120): team.step()     # 1.2 s: robots settle at spawn
out = Path(__file__).resolve().parent / "camera_probe"; out.mkdir(exist_ok=True)
Image.fromarray(team.render_frame()).save(out / f"cand_{which}.png")
print("saved", out / f"cand_{which}.png")
