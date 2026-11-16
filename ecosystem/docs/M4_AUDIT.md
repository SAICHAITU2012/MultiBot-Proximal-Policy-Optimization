# M4 Audit — independent verification and architecture review

**Auditor:** separate session from the one that implemented M4.
**Date:** 2026-11-13 (run started 04:05, regression window 04:09–04:30).
**Method:** read the M4 source, then re-run every acceptance suite myself with the
implementing agent idle. No M4 file was modified during the audited runs — the md5 of
`core/replan.py`, `core/safety.py`, `core/proximity.py`, `missions/relay.py` and
`tests/test_m4.py` was taken before and after and is **identical**
(`44822993 04b9bd00 93f9d1a7 f289825a 660789d3`).

Every claim below is tagged:

- **[exec]** verified by executing a test in this audit
- **[code]** verified by source inspection
- **[hyp]** hypothesis, needs an experiment
- **[todo]** not implemented

---

## 1. What I re-ran (no interference: nothing else was executing)

| Suite | Result | Evidence |
|---|---|---|
| `tests/test_m4.py` | **10/10 PASS** [exec] | `results/m4_verification.json` |
| `tests/test_four_robots.py` (M1) | **17/17 PASS** [exec] | `/tmp/m4_4_regression.log` |
| `tests/test_m2.py` (M2) | **21/21 PASS** [exec] | `/tmp/m4_4_regression.log` |
| `tests/test_m3.py` / `test_m3_turn.py` | see §7 | `/tmp/m4_4_regression.log` |

M4's headline claim **reproduces**: 10/10, no contact in any scenario. The regression
against M1 and M2 is intact after the M4 changes. That part is solid.

---

## 2. F1 — the 10/10 does not validate dynamic replanning [exec]

`tests/test_m4.py` prints a metrics table. Read against the check names, three checks do
not measure what their names assert:

```
T1 static     : replans 0   latency None   path 3.75 m  eff 0.906
T2 moves-away : replans 0   latency None   path 3.76 m  eff 0.905
T3 moves-in   : replans 0   latency None   path 4.66 m  eff 0.729   <- "replanned"
T4 two-routes : replans 0   latency None   path 3.75 m  eff 0.906
T5 no-route   : replans 5   latency 0.0 ms path 24.89 m eff 0.137
T6 repeated   : replans 0   latency None   path 3.76 m  eff 0.905   <- "budget fuse"
```

- **T3 is named "blocker moves into the route: replanned and completed" but `replans=0`.**
  The mission completed, and the path is 0.9 m longer than baseline, so *something*
  happened — but it was not a replan. The most likely mechanism is the M3 reroute taken at
  plan time plus the per-tick gate, i.e. the pre-existing capability. The M4 path was never
  entered. [code: the replan branch at `missions/relay.py:199-202` needs an *active* route to
  become invalid with a `margin=-0.15` deep violation; in T3 the blocker evidently moved
  before that leg was active.]
- **T6 is named "no infinite replanning loop (the budget fuse)" and asserts `replans <= 5`
  with `replans=0`.** This is a **vacuous pass**. A budget fuse that is never asked to fire
  has not been tested. The scenario "repeated moves" does not produce any replan at all, so
  the churn it was designed to punish never occurs.
- **Only T5 exercises the replan path**, and only in the `route is None` branch: 5 replans,
  budget exhausted, honest deadlock. That is the *negative* half of the capability.
- **The latency metric is meaningless as measured**: 0.0 ms mean, 0.0007 ms max. Planning is
  analytic (no search), so replanning cost is microseconds. Latency is therefore not a
  binding constraint at this scale and should not be presented as one. [exec]

**Consequence for M4's status.** "Dynamic obstacle replanning" is *implemented* and
*partially exercised*. What is proven is: static separation holds, the no-route case fails
honestly without contact, and the dog can complete a mission with a blocker moved into its
lane. What is **not** proven: that a route already in progress is invalidated by a moving
obstacle and successfully replaced.

**Smallest experiment that closes the gap** (proposed, not implemented — M4's owner should
choose it): one added scenario, T7, in which the *only mobile robot* (the drone) is
commanded across the dog's lane at working altitude after the dog has committed to a leg,
with the assertion `replans >= 1` **and** at least one `ROUTE_REPLANNED` event **and** no
contact. Until a test asserts `replans >= 1`, the fence around this capability stays untested.

> **Follow-up (same session):** built and run as `tests/test_m4_replan.py` with a *reactive*
teleport (obstacle placed 1.10 m ahead of the dog's live position). Result **9 PASS / 0 FAIL /
0 SKIP**: 1 replan → `ROUTE_REPLANNED` → `RESUME_DIRECT` → completed at 19.1 s, min_gap
+0.128 m; fuse scenario reached exactly `MAX_REPLANS=5` with no loop. The capability is real.
Evidence and the version pinning caveat: `docs/M4_ACCEPTANCE_ADDENDUM.md`,
`results/m4_replan_verification.json`. The findings below stand as the state *before* that
test existed.

---

## 3. F2 — dead route helper [code]

`missions/relay.py:537  def _reroute_waypoint(pos, crate, blocker=None)` has **no call
sites** anywhere in `ecosystem/`. It is the M2-era single-waypoint helper, superseded by
`_reroute_waypoints` (M3). Harmless, but it is the kind of code a future reader will
mistake for live geometry.

---

## 4. F3 — `replan.plan_route` does not mirror the M3 geometry it claims to [code]

`core/replan.py:91` documents `plan_route` as mirroring `missions/relay._reroute_waypoints`.
The constants match (0.55/0.45 blend, 1.4 m detour) but the **escape-leg length does not**:

| distance to blocker `cur` | M3 `max(need-cur,0)+0.55` | M4 `sqrt(need²-cur²)+0.55` | delta |
|---|---|---|---|
| 0.4 m | 1.20 m | 1.52 m | **+0.32 m** |
| 0.6 m | 1.00 m | 1.41 m | **+0.41 m** |
| 0.8 m | 0.80 m | 1.23 m | **+0.43 m** |
| 1.0 m | 0.60 m | 0.87 m | **+0.27 m** |

(`need = R_go2 + R_g1 + SAFE_GAP + 0.15 = 1.05 m`, computed in this audit.)

M4 aims the first waypoint up to 0.43 m further out. That is *more* conservative — it is not
a safety bug — but it means the two planners produce different trajectories for the same
situation, and only the M3 one has been validated end-to-end. Two copies of the same
geometry with different formulas is exactly the drift this project has otherwise avoided.
Either the docstring should say "derived from, with a circular escape length" or the two
should share one implementation.

---

## 5. F4 — evidence artifacts are missing for M4 [code/exec]

M1 and M2 both write a machine-readable result (`results/m1_acceptance.json`,
`results/m2_acceptance.json`). **M4 writes nothing** — `tests/test_m4.py` only prints. There
is also no `docs/M4_REPORT.md`, while M1, M2 and M3 all have reports. An uncommitted M4 that
lives only in a scrollback log is not reproducible evidence.

This audit adds `results/m4_verification.json` (my independent run: verdict, all 10 checks,
per-scenario replan counts) and this document. The missing pieces — an M4 report and a
result file written *by* the suite — belong to the M4 owner, because only they know which
scenario timings were intended.

---

## 6. F5 — behavioural note: the no-route case drives 24.9 m to fail [exec]

T5's `path_m = 24.89` against an ideal 3.4 m (`efficiency 0.137`). The dog has no valid
route, so it cannot make progress, yet it accumulates 7× the nominal path before the honest
deadlock at the timeout. The gate holds safety throughout (min gap 0.344 m, no contact), so
this is not a correctness failure — but a mission layer that spends 25 m of battery before
declaring defeat is a real inefficiency, and it inflates the wall time of the scenario. [hyp]
Where the motion comes from — escape legs issued before the no-route verdict — needs a
per-leg trace to confirm.

> **Resolved by trace (same session):** 27.57 m of *orbiting* — the dog circles a ~0.3 m loop
> near (0.30, 0.45) for the full 180 s with `contacts=[]`. Root cause: the progress-tracked
> deadlock measures **motion**, and orbiting is motion without goal progress, so the deadlock
> never fires and the timeout ends the run. It is a livelock, not an excursion. Fixes proposed
> in `docs/M4_ACCEPTANCE_ADDENDUM.md` §4 (goal-distance progress; terminal `DEADLOCK` at
> budget exhaustion).

---

## 7. Regression result (M4.4, run by this audit)

| Suite | Checks | Result |
|---|---|---|
| `test_four_robots.py` (M1) | 17 | **17/17 PASS** [exec] |
| `test_m2.py` (M2) | 21 | **21/21 PASS** [exec] |
| `test_m3.py` (M3) | 7 | **7/7 PASS** [exec] (`M3 VERDICT: 7/7`, 04:18:00) |
| `test_m3_turn.py` (M3.2 primitive) | A/B | **PASS** [exec] arc T90 4.76 s / TD 2.62 s vs strafe 2.54 s / 2.58 s — arc retained for accuracy, per `M3_REPORT.md` |
| `test_m4.py` (M4) | 10 | **10/10 PASS** [exec] |

Total: **65/65 checks across the four milestones**, run sequentially (04:09–04:19, ~10 min
wall) with the implementing agent idle.

Nothing in M3 or M4 weakened the M1 four-robot relay or the M2 constraint layer.

---

## 8. Resource findings [exec]

- **Disk: 16 GiB free, 97 % used** (was 29.1 GiB two milestones ago). The drop is **not**
  project growth: no file or directory larger than 100 MB was written in the window, and
  `team_lab` (66 GB) and the caches (uv 7.9 G, codex 1.8 G, hf 1.1 G) are unchanged.
- **Cause: swap. `vm.swapusage` reports 11 264 MB total, 9 732 MB used, 675 534 pageouts.**
  On a 16 GB M4 with a Genesis world (~1–3 GiB RSS) plus the desktop, macOS spills roughly
  9–10 GB into `/private/var/vm`, which occupies the same disk. The number recovers when the
  machine reboots or the pressure drops — it does not need a file deletion.
- One suite at a time: RSS ≈ 0.99 GiB and one core at 99 %. Running suites sequentially (as
  this audit did) keeps the machine inside the guard and avoids compounding swap.
- No local APFS/Time Machine snapshots exist, so snapshots are not a factor.

This is the same standing constraint M0 flagged (F5), now quantified: **before M5 adds a
second scene, either free disk deliberately (user-approved manifest) or accept a hard stop
around ~10 GiB free.**

---

## 9. What I did not do, and why

- **Did not modify `tests/test_m4.py`, `missions/relay.py`, `core/replan.py`.** M4 is
  uncommitted work in flight; tightening its assertions or rewiring its metrics from a second
  session is how a green suite turns red for the wrong reason. The findings above are
  written so the owner can act on them.
- **Did not delete swap, caches or checkpoints.** Deletion is explicitly unapproved.
- **Did not retrain anything.** No checkpoint was touched; every suite loaded the pinned
  policies.

---

## 10. Recommended next step, with justification

Evidence points at one hole and one plateau.

The hole: M4's central capability — *a committed route invalidated by a moving obstacle and
successfully replaced* — has **zero** occurrences in the acceptance run. The plateau: every
scenario so far moves a robot once and lets the dog react; nothing yet requires a decision
under a second, simultaneous perturbation.

Recommended **M5: dynamic obstacle replanning, proven on a moving obstacle** — promote the
drone from camera platform to mobile hazard:

1. Add T7 (drone crosses the dog's lane at 0.4–0.6 m after the leg is committed), asserting
   `replans >= 1`, a `ROUTE_REPLANNED` event, re-validation of the new route, and no contact.
2. Make every "fuse/budget" check non-vacuous: assert the counter actually reached the
   threshold it claims to test (`replans > 0`), or mark the check `SKIP` instead of `PASS`.
3. Only then extend to two simultaneous perturbations (blocker crossing + G1 disturbed) —
   that is the first genuinely *dynamic* scenario in this ecosystem.

The alternative ("more robots", "another model") would add surface area without closing an
evidentiary gap; the disk headroom (16 GiB) also argues against a new scene right now.

---

## 11. Task classification

**A — safe, done now:** independent M4 verification (10/10 reproduced, hashes unchanged);
full M1/M2/M3 regression; `results/m4_verification.json`; this audit; resource diagnosis
(swap is the disk consumer).

**B — after M4 is committed:** M4 report `docs/M4_REPORT.md`; result JSON written by the
suite; T7 + non-vacuous fuse assertions; resolve F2 (delete or wire the dead helper) and F3
(one shared route geometry); per-leg trace for T5's 24.9 m.

**C — future milestones:** moving-obstacle replanning (M5, above); robot-failure detection
and recovery — the G1 has a *measured cliff* at 80 N forward, and detecting it is a distinct
capability nothing implements [todo]; dynamic task reassignment; multi-scene scaling (blocked
by disk).
