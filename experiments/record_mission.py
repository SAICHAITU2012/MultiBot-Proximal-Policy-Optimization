#!/usr/bin/env python3
"""record_mission.py — record a REAL 2-package mission run to video (M8 closeout).

This is not a reconstruction. The mission is re-run unmodified (same scene, same
checkpoints, same planners) with a camera sensor attached, and the frames written
to the video are the actual rendered state at each captured sim step. The
recording run writes its own transcript, which is then diffed against the
previously logged run to prove the video matches a genuine execution.

Usage:
  python record_mission.py [--planner rule] [--packages 2] [--capture-every 10] [--fps 30]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "team"))

import av
import numpy as np
from PIL import Image, ImageDraw

import mission as mission_mod
from mission import Mission


def _md5(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", default="rule")
    ap.add_argument("--packages", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=60000)
    ap.add_argument("--capture-every", type=int, default=10, help="capture 1 frame every N scene steps (100 Hz)")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = HERE.parent / "logs" / f"video_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    vid_path = out_dir / "mission.mp4"

    m = Mission(n_packages=args.packages, planner_name=args.planner,
                max_steps=args.max_steps, record_camera=True)
    assert m.team.camera is not None, "recording camera was not created"

    container = av.open(str(vid_path), mode="w")
    stream = container.add_stream("libx264", rate=args.fps)
    stream.width, stream.height = 960, 540
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23"}          # quality/size; the scene is mostly flat colour

    state = {"n": 0, "frames": 0, "t_first": None, "t_last": None}
    # On-screen narration. The video must explain itself to someone who has never
    # seen the project, so each frame carries the sim clock, the delivery tally
    # and the most recent team event (assignment / stroke / delivery).
    nar = {"seen": 0, "line": "waiting for first task", "delivered_seen": 0,
           "banner_until": -1.0, "banner": ""}
    INTERESTING = ("plan", "task_started", "task_complete", "package_delivered",
                   "push_stroke_end", "recon")
    orig_step = m.team.step

    def step_and_capture():
        orig_step()
        if state["n"] % args.capture_every == 0:
            rgb = m.team.render_frame()
            img = Image.fromarray(rgb)
            draw = ImageDraw.Draw(img)
            ts = m.team.sim_time
            # -- newest interesting team event -> one narration line
            hist = list(m.bb.history)
            for msg in hist[nar["seen"]:]:
                if msg["topic"] in ("event", "task", "assignment") and \
                        any(k in msg["type"] for k in INTERESTING):
                    d = msg["data"]
                    keys = [k for k in ("task", "package", "result", "travelled",
                                        "box_goal_distance", "drop_distance", "robot") if k in d]
                    nar["line"] = f"{msg['from']} {msg['type']}" + (
                        "  " + " ".join(f"{k}={d[k]}" for k in keys)[:90] if keys else "")
                    if msg["type"] == "package_delivered":
                        nar["banner"] = (f"PACKAGE {str(d.get('package','')).upper()} DELIVERED"
                                         f"   drop distance {d.get('drop_distance')} m")
                        nar["banner_until"] = ts + 5.0
            nar["seen"] = len(hist)
            if m.metrics["delivered"] != nar["delivered_seen"]:
                nar["delivered_seen"] = m.metrics["delivered"]
            # -- draw
            draw.rectangle([0, 0, 960, 22], fill=(0, 0, 0))
            draw.text((8, 6), f"Genesis multi-robot mission  |  planner: {args.planner}  |  "
                              f"3 robots + commander  |  sim t={ts:7.2f}s", fill=(230, 230, 230))
            draw.rectangle([0, 516, 960, 540], fill=(0, 0, 0))
            draw.text((8, 524), f"delivered {m.metrics['delivered']}/{len(m.packages)}   "
                                f"recon {m.metrics['recon']}/{len(m.packages)}   |   {nar['line']}".ljust(4),
                      fill=(255, 255, 255))
            if ts < nar["banner_until"]:
                draw.rectangle([210, 40, 750, 86], fill=(0, 40, 0), outline=(120, 255, 120))
                draw.text((228, 54), nar["banner"], fill=(160, 255, 160))
            frame = av.VideoFrame.from_ndarray(np.asarray(img), format="rgb24")
            for pkt in stream.encode(frame):
                container.mux(pkt)
            state["frames"] += 1
            if state["t_first"] is None:
                state["t_first"] = ts
            state["t_last"] = ts
        state["n"] += 1

    m.team.step = step_and_capture      # instance attribute shadows the class method

    t0 = time.time()
    try:
        m.run()
    finally:
        for pkt in stream.encode():
            container.mux(pkt)
        container.close()
        wall = time.time() - t0

    manifest = {
        "kind": "recorded (real re-run with a camera sensor; NOT a replay reconstruction)",
        "planner": args.planner,
        "packages": args.packages,
        "capture_every_steps": args.capture_every,
        "capture_hz_sim": round(100.0 / args.capture_every, 2),
        "playback_fps": args.fps,
        "frames": state["frames"],
        "sim_time_first_frame_s": round(state["t_first"] or 0, 2),
        "sim_time_last_frame_s": round(state["t_last"] or 0, 2),
        "video_duration_s": round(state["frames"] / args.fps, 1),
        "playback_speed_vs_sim": round((state["t_last"] - state["t_first"]) / max(1e-9, state["frames"] / args.fps), 2),
        "wall_time_s": round(wall, 1),
        "mission_metrics": {k: v for k, v in m.metrics.items()},
        "files_md5": {
            "team/mission.py": _md5(HERE.parent / "team" / "mission.py"),
            "experiments/colo_env.py": _md5(HERE.parent / "experiments" / "colo_env.py"),
            "team/agents.py": _md5(HERE.parent / "team" / "agents.py"),
            "team/planner.py": _md5(HERE.parent / "team" / "planner.py"),
        },
        "video_md5": _md5(vid_path),
        "video_bytes": vid_path.stat().st_size,
    }
    (out_dir / "video_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[record] {vid_path}")
    print(f"[record] frames={state['frames']} fps={args.fps} "
          f"sim {state['t_first']:.1f}..{state['t_last']:.1f}s "
          f"video {state['frames']/args.fps:.1f}s wall {wall:.0f}s")
    print(f"[record] manifest -> {out_dir/'video_manifest.json'}")


if __name__ == "__main__":
    main()
