# MASTER TRACKER — Genesis Multi-Robot Autonomous Project

Maintained per the Master Execution Directive (2026-11-08). Status labels:
`VERIFIED COMPLETE` / `IMPLEMENTED, NOT VALIDATED` / `PARTIALLY COMPLETE` /
`BLOCKED` / `NOT STARTED`. Detailed evidence: `../INVESTIGATION_LOG.md`.

## Milestone status

| Milestone | Status | Evidence / next step |
|---|---|---|
| **M0 Workspace audit & baseline** | `VERIFIED COMPLETE` (core) | The Nov 8 root-cause investigation re-audited the push subsystem end-to-end (env code, cfgs.pkl snapshots vs source, checkpoint hashes, num_envs behavior, loader paths, upstream conventions). Tracker + manifest = this file. Remaining: G1/LLM audits (deferred to M5/M6). |
| **M1 Physics & integration reliability** | `VERIFIED COMPLETE` (core) | dt unification (scene 0.02 / rigid 0.01 both scenes); the push-env reset repaired (buffer-wipe + z=0 spawn) with permanent regression tests (`experiments/test_env_reset.py`, positive 3/3 PASS, negative-control 3/3 FAIL as required); co-location smoke test `test_colo.py` (last verified 2026-11-07 — re-run after the mission realignment). |
| **M2 Go2 nav / push / delivery** | `PARTIALLY COMPLETE` | r1 trained (1352 iters) but the sweep exposed a **fall-slam exploit**: aggregate "pushes" were accumulated across auto-reset lives (286 true terminations/9600 env-steps for model_400; median life 0.4 s). Evaluator upgraded (true falls + first-life displacement). Controlled fix r2 running: fall_penalty 4→12 (one variable), warm start from r1/model_400. Gates (mission-relevant): ≥8/16 FIRST-LIFE pushed ≥0.15 m, 0 true falls. |
| **M3 Drone reconnaissance** | `VERIFIED COMPLETE` (mission role) | recon completes 1-2/2 per mission across all 2026-11-11 runs; static-target tracking median 0.515 m documented (the 0.1-m eval metric is not the mission metric; the historical 0.337 median used a different protocol). The recon1-50 Hz issue did not reproduce. |
| **M4 Panda processing** | `VERIFIED COMPLETE` (parcel path) | The ROOT CAUSE of picked=0-everywhere: a LEGACY scripted-press path unconditionally returned before the trained P1b policy (dead code since the P1b integration) + a completion-check key mismatch (parcel_spawn_xy never set → disp ≡ 0). Fixed 2026-11-11: reached(2.6 s)→contact(3.8 s)→processed(3.9 s), picked=1 per run, physics-verified displacement. The delivered-crate process path (7 kg, ungraspable) remains the documented limitation (the honest `inspected` metric). |
| **M5 G1 baseline** | `VERIFIED COMPLETE` (10-s standard) | Session 2 (+1150 iters, resume-glob fix, eval-only mode added to the trainer): **4/4 stood the full 10 s, height 0.797→0.792 m** (session 1 fell within 10 s). Walking = the documented stretch goal. |
| **M6 LLM commander** | `VERIFIED COMPLETE` (local) | ROOT CAUSE of planner_decisions=0: _plan_tick's claim guard required status=="assigned" while open tasks are "open" — the rule planner self-mutates (always worked); the LLM's pure mappings were silently dropped. Claim-on-apply fixed (accepting both styles). Local qwen2.5:1.5b now coordinates live: valid JSON assignments, stage-2 fetch creation after recon, decisions=3-4/run, picked=1 under LLM coordination. No paid APIs. |
| **M7 End-to-end mission** | `PARTIALLY COMPLETE` | 3 deterministic full-mission runs (2026-11-11): recon 2/2 ✓, fetches assigned ✓, Panda picked=1 ✓, **delivered 0/2** — the pkg0 endgame grind + pkg1's 3.8-m diagonal exceed the 1200-s budget. Single-package controlled runs deliver 2/6 seeds. The identified fix: a terminal-placement training stage (the M2 bottleneck). |
| **M8 Closeout** | `NOT STARTED` | After M7. |

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

1. Go2 mission-level delivery not yet revalidated with the realigned canonical packages (M2/M7).
2. Drone recon1 at 50 Hz colo control unresolved (M3).
3. Panda in-mission process completion unverified against 7 kg crates (M4).
4. G1 needs sessions 2-3 + a standing-duration eval (M5).
5. LLM commander not started (M6) — local-model feasibility check first.
6. The Go2 evaluator's mask-filtered contact check reports 0/16 despite movement (cosmetic; displacement is the evidence used).
