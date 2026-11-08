#!/usr/bin/env python3
"""
reward_audit.py — the reward decomposition audit (the advisor-approved M2 diagnostic).

v2 — corrects two v1 measurement bugs found after the first run:
  1. episode_sums are ZEROED by _reset_idx on every auto-reset
     (go2_push_env.py:314), so v1's end-of-run snapshot measured only the
     post-reset residual ("all components zero" was an artifact).
  2. v1 loaded go2-push-m0/model_577.pt — which is M0's OWN iteration 577
     (md5 differs from go2-push-s7/model_577.pt). The mission-pinned s7
     checkpoint was never audited. v2 audits the true pinned checkpoint.

Design (per the advisor's acceptance gates):
  * cfgs from logs/go2-push-m0/cfgs.pkl — the exact M0 training-time cfgs
    (verified: identical to the current source except crate_size/spawn-z,
    which ARE the mission geometry 12 cm / 0.65 m / goal 3.1 m).
  * checkpoints: go2-push-s7/model_577.pt (the mission-pinned push policy)
    and go2-push-m0/model_2270.pt (M0's final, the diagnosed degenerate).
  * policy path = colo_env.load_actor_from_log — the SAME loader the mission
    uses (plain actor + as_tensor conversion), not the runner's wrapper.
  * 16 parallel envs, torch.manual_seed(42) before EACH env build => the two
    checkpoints see IDENTICAL spawn sequences per env index (fresh seeds,
    never training episodes).
  * full 600 steps, no break at d; per-component contributions recorded by
    wrapping the env's reward functions (reset-proof); fall penalty tracked
    from d * (1 - time_outs); per-episode records closed on termination.

Deliverable: per-episode behavior + exact component accounting, then the
offline recalculation of candidate structures on the RECORDED trajectories
(the static accounting gate only — training dynamics need real training).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/Users/saichaitu/Desktop/Genesis/genesis-world/examples/locomotion")
TL = Path("/Users/saichaitu/Desktop/Genesis/team_lab")
sys.path.insert(0, str(TL)); sys.path.insert(0, str(TL / "experiments")); sys.path.insert(0, str(TL / "team"))

import torch
import genesis as gs

import colo_env
from go2_push_env import Go2PushEnv

SEED = 42
N_ENVS = 16
N_STEPS = 600  # one 12-s episode length; terminations inside = falls (pitch/roll)
DOWN_Z = 0.15  # the down-state threshold (standing gait base 0.30-0.42)

with open(TL / "logs" / "go2-push-m0" / "cfgs.pkl", "rb") as f:
    env_cfg, obs_cfg, reward_cfg, push_cfg, train_cfg = __import__("pickle").load(f)
print(f"[audit] cfgs from go2-push-m0/cfgs.pkl: crate={push_cfg['crate_size']} "
      f"spawn={push_cfg['crate_spawn']} goal_r={push_cfg.get('goal_radius')} "
      f"fall_penalty={env_cfg['fall_penalty']} ep_len={env_cfg['episode_length_s']}s")

CHECKPOINTS = {
    "s7_577 (mission-pinned)": TL / "logs" / "go2-push-s7" / "model_577.pt",
    "m0_2270 (M0 final)": TL / "logs" / "go2-push-m0" / "model_2270.pt",
}

all_records = {}

for ckpt_name, ckpt_path in CHECKPOINTS.items():
    torch.manual_seed(SEED)
    gs.init(backend=gs.cpu, logging_level="error", performance_mode=True)
    env = Go2PushEnv(num_envs=N_ENVS, env_cfg=env_cfg, obs_cfg=obs_cfg,
                     reward_cfg=reward_cfg, push_cfg=push_cfg, show_viewer=False)
    policy = colo_env.load_actor_from_log(str(ckpt_path.parent), ckpt_path.name)

    # --- reset-proof component recording: wrap each reward fn -------------
    comp_totals = {name: torch.zeros(N_ENVS, device=gs.device)
                   for name in env.reward_functions}
    orig_fns = dict(env.reward_functions)

    def _wrap(name, fn):
        def wrapped():
            r = fn() * env.reward_scales[name]
            comp_totals[name] += r.detach()
            return r
        return wrapped

    for name, fn in orig_fns.items():
        env.reward_functions[name] = _wrap(name, fn)

    obs = env.reset()  # single reset: the spawn sequence must stay identical across checkpoints
    obs = torch.as_tensor(obs["policy"].cpu().numpy()).to(gs.device)

    eps = [{  # per-env open episode record
        "comp": {name: 0.0 for name in orig_fns}, "fall_pen": 0.0, "steps": 0,
        "down_steps": 0, "z_min": 1e9, "z_sum": 0.0, "dog_crate_min": 1e9,
        "start_crate": None, "start_goal": None, "ret": 0.0,
    } for _ in range(N_ENVS)]
    closed = []

    def close(i, rec, end_crate, end_z):
        rec["end_z"] = end_z
        sc, sg = rec.pop("start_crate"), rec.pop("start_goal")
        d_goal = sg - sc[:2]  # goal is xy (2,), crate snapshots are xyz (3,)
        L = float(torch.norm(d_goal)) + 1e-9
        u = d_goal / L
        rec["crate_disp_toward_goal"] = float(torch.dot(end_crate[:2] - sc[:2], u))
        rec["crate_disp_abs"] = float(torch.norm(end_crate[:2] - sc[:2]))
        rec["z_mean"] = rec.pop("z_sum") / max(rec["steps"], 1)
        closed.append({"env": i, **rec})

    prev_crate = env.crate_pos.clone()
    prev_z = env.base_pos[:, 2].clone()
    with torch.inference_mode():
        for t in range(N_STEPS):
            a = policy(torch.as_tensor(obs.cpu().numpy()))
            obs_t, r, d, extras = env.step(a)
            obs = torch.as_tensor(obs_t["policy"].cpu().numpy()).to(gs.device)
            fall_pen = env_cfg["fall_penalty"] * d.float() * (1.0 - extras["time_outs"])
            to = extras["time_outs"].bool()

            for i in range(N_ENVS):
                rec = eps[i]
                cur_crate, cur_z = env.crate_pos[i].clone(), float(env.base_pos[i, 2])
                if rec["start_crate"] is None:  # first step of a fresh episode
                    rec["start_crate"] = prev_crate[i].clone()
                    rec["start_goal"] = env.goal_xy[i].clone()
                # accumulate using PREV-step state for terminated envs (they
                # were already reset inside step())
                src_crate, src_z = (prev_crate[i], prev_z[i]) if bool(d[i]) else (cur_crate, cur_z)
                rec["steps"] += 1
                rec["ret"] += float(r[i]) - float(fall_pen[i])
                rec["fall_pen"] += float(fall_pen[i])
                z = float(src_z)
                rec["down_steps"] += int(z < DOWN_Z)
                rec["z_min"] = min(rec["z_min"], z)
                rec["z_sum"] += z
                rec["dog_crate_min"] = min(rec["dog_crate_min"],
                                           float(torch.norm(src_crate[:2] - env.base_pos[i, :2])))
                for name in orig_fns:
                    rec["comp"][name] += float(comp_totals[name][i])
                if bool(d[i]):
                    close(i, rec, prev_crate[i], float(prev_z[i]))
                    eps[i] = {"comp": {name: 0.0 for name in orig_fns}, "fall_pen": 0.0,
                              "steps": 0, "down_steps": 0, "z_min": 1e9, "z_sum": 0.0,
                              "dog_crate_min": 1e9, "start_crate": cur_crate,
                              "start_goal": env.goal_xy[i].clone(), "ret": 0.0}
                    for name in orig_fns:
                        comp_totals[name][i] = 0.0
            prev_crate = env.crate_pos.clone()
            prev_z = env.base_pos[:, 2].clone()

    for i in range(N_ENVS):  # close still-open episodes (time-out at loop end)
        if eps[i]["steps"] > 0:
            close(i, eps[i], env.crate_pos[i], float(env.base_pos[i, 2]))

    for rec in closed:
        rec["comp_total"] = sum(rec["comp"].values())
        rec["return_check"] = rec["comp_total"] + rec["fall_pen"]

    all_records[ckpt_name] = closed
    bad = [r for r in closed if abs(r["return_check"] - r["ret"]) > 0.5]
    print(f"\n[audit] {ckpt_name}: {len(closed)} episodes closed | "
          f"accounting mismatches (return vs components+fall, >0.5): {len(bad)}")
    gs.destroy()

# ================= REPORT =================
print("\n=== THE REWARD AUDIT v2 (16 identical-seed episodes per checkpoint) ===")
for ckpt_name, recs in all_records.items():
    n = len(recs)
    pushed = sum(1 for r in recs if r["crate_disp_toward_goal"] >= 0.15)
    moved = sum(1 for r in recs if r["crate_disp_abs"] > 0.01)
    fell = sum(1 for r in recs if r["fall_pen"] < -0.01)
    reached = sum(1 for r in recs if r["dog_crate_min"] < 0.45)
    rets = sorted(r["ret"] for r in recs)
    print(f"\n--- {ckpt_name} ---")
    print(f"  REACHED(dog-crate<0.45m): {reached}/{n} | moved>1cm: {moved}/{n} | "
          f"PUSHED(>=15cm toward goal): {pushed}/{n} | episodes ending in a fall: {fell}/{n}")
    print(f"  returns: min {rets[0]:+.1f} | median {rets[n//2]:+.1f} | max {rets[-1]:+.1f}")
    comp_sum = {}
    for r in recs:
        for k, v in r["comp"].items():
            comp_sum[k] = comp_sum.get(k, 0.0) + v
    print(f"  component totals over {n} episodes (per-episode mean):")
    for k in sorted(comp_sum, key=lambda k: comp_sum[k]):
        print(f"    {k:20s}: {comp_sum[k]/n:+9.3f}")
    print(f"    {'fall_penalty':20s}: {sum(r['fall_pen'] for r in recs)/n:+9.3f}")
    print(f"  down-state steps (z<{DOWN_Z}): total {sum(r['down_steps'] for r in recs)} "
          f"| per-episode mean {sum(r['down_steps'] for r in recs)/n:.1f} of {N_STEPS}")
    print(f"  crate displacement toward goal: mean "
          f"{sum(r['crate_disp_toward_goal'] for r in recs)/n:+.3f} m | best "
          f"{max(r['crate_disp_toward_goal'] for r in recs):+.3f} m")

# pushing vs non-pushing episode economics under the current structure
print("\n=== EPISODE ECONOMICS: what the current structure pays each behavior ===")
for ckpt_name, recs in all_records.items():
    grp = {}
    for r in recs:
        g = "PUSHED" if r["crate_disp_toward_goal"] >= 0.15 else ("moved" if r["crate_disp_abs"] > 0.01 else "static")
        grp.setdefault(g, []).append(r["ret"])
    for g, v in sorted(grp.items()):
        print(f"  {ckpt_name} | {g:7s} n={len(v):2d}: return mean {sum(v)/len(v):+8.1f} "
              f"| range [{min(v):+.1f}, {max(v):+.1f}]")

# offline recalculation of candidate structures on the RECORDED trajectories
print("\n=== OFFLINE RECALCULATION (static accounting on recorded episodes only) ===")
print("  NOTE: this checks whether a candidate's accounting SEPARATES pushing from")
print("  degenerate episodes; it cannot predict training dynamics (needs real runs).")
CANDS = [
    ("current", None, None),
    ("A1: down -1.0 per down STEP (v1 proposal units)", lambda r: -1.0 * r["down_steps"], None),
    ("A2: down -2.0/s dt-scaled", lambda r: -2.0 * r["down_steps"] * (1.0 / 50.0), None),
    ("B: A2 + fall -1.0 (from -4.0)", lambda r: -2.0 * r["down_steps"] * (1.0 / 50.0) - 0.75 * r["fall_pen"], None),
]
for cand_name, adj, _ in CANDS:
    print(f"\n  {cand_name}:")
    for ckpt_name, recs in all_records.items():
        if adj is None:
            vals = [r["ret"] for r in recs]
        else:
            vals = [r["ret"] + adj(r) for r in recs]
        pg = [v for v, r in zip(vals, recs) if r["crate_disp_toward_goal"] >= 0.15]
        sg = [v for v, r in zip(vals, recs) if r["crate_disp_toward_goal"] < 0.15]
        pm = f"pushed mean {sum(pg)/len(pg):+8.1f}" if pg else "pushed mean      n/a"
        sm = f"not-pushed mean {sum(sg)/len(sg):+8.1f}" if sg else "not-pushed mean      n/a"
        gap = (sum(pg) / len(pg) - sum(sg) / len(sg)) if (pg and sg) else float("nan")
        print(f"    {ckpt_name}: {pm} | {sm} | gap {gap:+7.1f}"
              + ("  <-- gap NEGATIVE: structure still prefers not-pushing" if pg and sg and gap < 0 else ""))

out = TL / "experiments" / "reward_audit_v2_records.json"
json.dump(all_records, out.open("w"), indent=1, default=float)
print(f"\n[audit] per-episode records saved: {out}")
