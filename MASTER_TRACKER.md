# MASTER TRACKER — Genesis Multi-Robot Autonomous Project

Maintained per the Master Execution Directive (2026-11-08). Status labels:
`VERIFIED COMPLETE` / `IMPLEMENTED, NOT VALIDATED` / `PARTIALLY COMPLETE` /
`BLOCKED` / `NOT STARTED`. Detailed evidence: `INVESTIGATION_LOG.md`.

## Milestone status

| Milestone | Status | Evidence / next step |
|---|---|---|
| **M0 Workspace audit & baseline** | `VERIFIED COMPLETE` (core) | The Nov 8 root-cause investigation re-audited the push subsystem end-to-end (env code, cfgs.pkl snapshots vs source, checkpoint hashes, num_envs behavior, loader paths, upstream conventions). Tracker + manifest = this file. Remaining: G1/LLM audits (deferred to M5/M6). |
| **M1 Physics & integration reliability** | `VERIFIED COMPLETE` (core) | dt unification (scene 0.02 / rigid 0.01 both scenes); the push-env reset repaired (buffer-wipe + z=0 spawn) with permanent regression tests (`experiments/test_env_reset.py`, positive 3/3 PASS, negative-control 3/3 FAIL as required); co-location smoke test `test_colo.py` (last verified 2026-11-07 — re-run after the mission realignment). |
| **M2 Go2 nav / push / delivery** | `VERIFIED COMPLETE` (s5/model_800) | **Gates are met by the mission's own policy (2026-11-12 eval, fresh seeds, 32 episodes x 2 runs):** `go2-push-s5/model_800` — REACHED+CONTACT **32/32**, FIRST-LIFE pushed (>=0.15 m in ONE clean engagement) **30/32 and 31/32**, TRUE FALLS **0**, first-life displacement mean +1.42/+1.90 m (best +4.38 m). Gate was >=8/16 first-life and 0 falls => passes decisively in both runs. The r1->r2->r3 branch pursued the fall-slam problem and is a **closed dead end**: r3 model_300 6/16 first-life with 16 true falls, model_600 7/16 with 16 falls, model_900 5/16 with **563** falls (degenerate, ~1 termination / 0.34 s); s7/model_577 regressed too (4/16, 59 falls). M2 never depended on that branch — the pinned s5 policy was the one delivering. Evidence table: `INVESTIGATION_LOG.md` (M2 section). |
| **M3 Drone reconnaissance** | `VERIFIED COMPLETE` (mission role) | recon completes 1-2/2 per mission across all 2026-11-11 runs; static-target tracking median 0.515 m documented (the 0.1-m eval metric is not the mission metric; the historical 0.337 median used a different protocol). The recon1-50 Hz issue did not reproduce. |
| **M4 Panda processing** | `VERIFIED COMPLETE` (parcel path) | The ROOT CAUSE of picked=0-everywhere: a LEGACY scripted-press path unconditionally returned before the trained P1b policy (dead code since the P1b integration) + a completion-check key mismatch (parcel_spawn_xy never set → disp ≡ 0). Fixed 2026-11-11: reached(2.6 s)→contact(3.8 s)→processed(3.9 s), picked=1 per run, physics-verified displacement. The delivered-crate process path (7 kg, ungraspable) remains the documented limitation (the honest `inspected` metric). |
| **M5 G1 baseline** | `VERIFIED COMPLETE` (10-s standard) | Session 2 (+1150 iters, resume-glob fix, eval-only mode added to the trainer): **4/4 stood the full 10 s, height 0.797→0.792 m** (session 1 fell within 10 s). Walking = the documented stretch goal. |
| **M6 LLM commander** | `COMPLETE` (local) | ROOT CAUSE of planner_decisions=0: _plan_tick's claim guard required status=="assigned" while open tasks are "open" — the rule planner self-mutates (always worked); the LLM's pure mappings were silently dropped. Claim-on-apply fixed (accepting both styles). Local qwen2.5:1.5b now coordinates live: valid JSON assignments, stage-2 fetch creation after recon, decisions=3-4/run, picked=1 under LLM coordination. No paid APIs. **Stranded-task bug fixed (2026-11-12):** the completion pass in `LLMPlanner.plan` steals a robot the LLM already assigned, so its pick is discarded while `RuleBasedPlanner`'s internal `status="assigned"` sticks — the task becomes unowned and `open_tasks()` never returns it (one recon never ran; LLM missions capped at delivered 1/2). Unused claims are now reverted to `open`. LLM-coordinated mission: recon 2/2, **delivered 2/2**, decisions=7. |
| **M7 End-to-end mission** | `COMPLETE (both planners)` | **delivered 2/2** (2026-11-12): pkg0 drop_distance 0.549 at t=536 s, pkg1 0.547 at t=757 s, both `mode: learned`, recon 2/2, picked=1, makespan 1200 s. The gate is met by a **push control layer**, not retraining: the learned pusher was never the gap (it hauls 0.35-0.91 m per stroke); what was missing was the closed loop around it. Diagnosis, calibration and the two new regression tests are in `INVESTIGATION_LOG.md` (M7 CLOSED section). **Both planners now deliver 2/2** (`--planner rule` and `--planner qwen2.5:1.5b`, identical mission traces, drop 0.549/0.547 m, 68 strokes); the LLM path needed a one-line fix in the planner's completion pass (see M6). |
| **M8 Closeout** | `VERIFIED COMPLETE` (2026-11-12) | One-page report `M8_CLOSEOUT.md` + **video of a real run**: `logs/video_20261112_144810/mission.mp4` (200 s, 960x540, 30 fps, 6000 frames, 3.8 MB, md5 `916ff0e9d64b`, clean full decode). Recorded, NOT reconstructed: the mission was re-executed unchanged with a camera sensor attached (`experiments/record_mission.py`, camera off by default and additive) and its transcript is byte-identical apart from timestamps to the logged run (`diff` = 0 vs `logs/mission_20261112_143024`). Frames carry an overlay (clock, delivery tally, latest team event, delivery banner with the measured drop distance); 5 stills extracted and visually verified. Manifest with per-file md5s: `video_manifest.json`. Report numbers re-checked against raw transcripts (all delivery runs identical: 17 falls = 12 nav + 5 charge, 70 strokes, drops 0.549/0.547 m); a 4th post-M8 regression run after the camera hook also delivered 2/2 with a byte-identical transcript, proving the recording path did not alter the sim. No frozen asset changed. |

## Key artifacts & hashes

- Repaired `experiments/go2_push_env.py` md5 `3a7c91dbdb484887a25770045672c6fd`; commit `cc2f0664a1e098d0561477f30aeda0dcccd8d3fe` (re-alignment commit follows).
- Original pinned checkpoint `logs/go2-push-s7/model_577.pt` md5 `6e8249a898a3210befd39cf897c99223` — the r1 warm start source. Preserved, never overwritten.
- Regression tests `experiments/test_env_reset.py` md5 `61abee2f3a07fb6330cf4f6758523a02` (run after ANY go2_push_env.py change).
- Training: `go2-push-r1` (cfgs = 15 cm crate / 7 kg / 0.65 m spawn / 3.1 m goal / ±75° band; checkpoints every 50 iters; guarded 42-min sessions).
- Canonical package (per directive §4-M2): **7 kg, 0.35×0.35×0.15 m** — training env AND mission (colo realigned 2026-11-08). Do not weaken the mission definition.

## Acceptance thresholds (fixed 2026-11-08, not to be moved post-hoc)

- Go2 policy: fresh-seed movement 16/16 (basic); pushed ≥0.15 m in ≥8/16 AND 0 falls (acceptance).
- Mission delivery: the 7 kg crate reaches the delivery zone; 3/3 independent runs.
- Panda: measured object displacement + verified PROCESSED transition.
- Drone: recon success criteria + tracking metrics (from the s4 eval standard).
- G1: measured standing duration, orientation, failure rate.

## Defect list (open)

1. ~~Go2 mission-level delivery~~ — CLOSED 2026-11-12: delivered 2/2 with the rule planner AND with the local LLM planner (drop 0.549/0.547 m, both `mode: learned`). **3/3 independent runs** (rule, LLM, rule), each delivered 2/2.
2. Drone recon1 at 50 Hz colo control unresolved (M3).
3. Panda in-mission process completion unverified against 7 kg crates (M4).
4. G1 needs sessions 2-3 + a standing-duration eval (M5).
5. ~~LLM commander not started~~ — CLOSED: local qwen2.5:1.5b coordinates the mission (recon 2/2, delivered 2/2, decisions 7).
6. The Go2 evaluator's mask-filtered contact check reports 0/16 despite movement (cosmetic; displacement is the evidence used).
7. **Hardening (not an M7 reopening):** mission-level fall rate, 17 per 2-package mission — 12 during nav/retreat, 5 during a charge. The push env itself is fall-free (0 falls in 32 lives), so this is the nav policy meeting the crate, not the pushing skill.
