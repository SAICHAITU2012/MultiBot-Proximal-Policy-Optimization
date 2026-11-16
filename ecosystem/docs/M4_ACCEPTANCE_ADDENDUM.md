# M4 acceptance addendum — the replan path, actually exercised

**Status of this document:** independent evidence + two proposed code fixes. It does **not**
modify M4's implementation or its suite. Companion to `docs/M4_AUDIT.md` (finding F1) and
`results/m4_replan_verification.json`.

---

## 1. The gap, restated precisely

`tests/test_m4.py` T3 is named "blocker moves into the route: replanned and completed" but
runs with `replans=0`. Reading the scenario: it teleports the G1 at **t = 3.0 s**, before the
dog has committed to a leg. The M3 planner then sees the blocker at plan time and routes
around it — static avoidance, not replanning. T6 ("the budget fuse") asserts `replans <= 5`
and also runs with `replans=0`, so it passes without the fuse ever being asked to fire.

So M4's *closed loop* — route invalidated after commitment → replacement computed, validated,
resumed — had zero recorded occurrences.

## 2. What now proves it

`tests/test_m4_replan.py` (new, additive). Its disturbance is a teleport placed **relative to
the dog's live position**: 1.10 m ahead along its recent velocity (a 0.42 m surface gap at
placement — outside `SAFE_GAP`, so the teleport itself creates no conflict), armed by live
progress (the dog must have travelled ≥ 0.60 m), so the obstacle cannot be accounted for by
the plan already made. Placement is always ≥ 1.5 m from the crate so a detour exists.

**Result: 9 PASS, 0 FAIL, 0 SKIP** — deterministic across two runs.

| Check | Measurement |
|---|---|
| A0 prerequisite: dog mid-execution at placement | placed at t=4.0 s, 5.08 m travelled |
| A0b placement outside the conflict threshold | surface gap **0.42 m** at placement (SAFE_GAP 0.22) |
| **A1 the committed route was invalidated and replanned** | **replans = 1** |
| A2 a replacement route was published | `ROUTE_REPLANNED`, then `RESUME_DIRECT` |
| A3 dog resumed and completed | **completed, t = 19.1 s** |
| A4 no go2-g1 contact | **min_gap +0.128 m** |
| B0 fuse scenario produced replans | 6 live placements → **5 replans** |
| B1 budget never exceeded | `replans = MAX_REPLANS = 5`, no loop |
| B2 honest outcome, contact-free | completed, min_gap **+0.068 m** |

That is the capability M4 is named after: **commit → invalidate → replan → validate → resume →
complete, without contact.** It works.

A rejected first attempt is worth recording: placing the obstacle at 0.65 m (inside the
0.68 m envelope sum) produced `min_gap = −0.321` and a genuine `g1-go2` contact — the
proximity layer caught it, and the contact was my scenario's construction error, not the
system's. The final placement (1.10 m) is the honest test.

## 3. Version pinning — this evidence belongs to a revision, not to "M4"

The suite ran against a frozen snapshot of M4 (hashes in the artifact):

```
core/replan.py    e5b83956    core/proximity.py 87cf02ce    core/safety.py  04b9bd00
core/world.py     3a71f87e    missions/relay.py  ef988be8    robots/go2_drive 58a4a202
```

While this audit ran, the M4 owner continued editing: `missions/relay.py` moved
`ef988be8 → 6898ab60` and `tests/test_m4.py` moved `660789d3 → 94caec1e`. **The evidence
above is therefore pinned to `relay.py ef988be8`.** Before M4 is called closed, re-run
`python tests/test_m4_replan.py` against the final revision and commit the refreshed
artifact. A green result on a superseded file proves nothing about the current one.

## 4. T5's 24.9 m, diagnosed

Reproduced with a per-tick position trace (`World(g1_xy=(1.3,1.2))`, crate at (1.2,1.2)):

```
outcome: ok=False  why=timeout at stage reached  t=180.0 s
travel : 27.57 m   replans=5 (the budget)   reroutes=1   waits=18
safety : blocked_ticks=19   contacts=[]   (no collision anywhere)
```

The path is not a stray excursion: the dog **orbits a ~0.3 m circle near (0.30, 0.45) for the
whole 180 s**, ~0.15 m/s, while the crate sits at (1.2, 1.2). Two consequences:

1. **Root cause.** The "progress-tracked deadlock" (M3) tests *motion*, and orbiting supplies
   motion without approaching the goal. So the deadlock never fires and the run ends on the
   180 s timeout instead — a **livelock**, honestly reported but 3× more expensive than it
   needs to be.
2. **Proposed fix (not applied — owner's file).** (a) Measure progress as reduction in
   goal distance / stage progress, not path length; (b) make budget exhaustion terminal: once
   `replans == MAX_REPLANS` with no valid candidate, publish `DEADLOCK` immediately instead of
   holding the old route and waiting on a progress timer that cannot fire.

## 5. Route geometry — the divergence is now *inside* `replan.py`

`core/replan.py` contains **both** formulas: `plan_route` uses
`L = sqrt(need² − cur²) + 0.55` (line 111) while `plan_candidates` uses
`esc_len = need − cur + 0.55` (line 148). The second matches `relay._reroute_waypoints`
(the M3 geometry that has been validated end-to-end); the first does not, by up to 0.43 m.
Whichever path a given replan takes, it gets a different first waypoint.

Proposal (one change, no behaviour change to the validated path): make `plan_route` call the
same expression as `plan_candidates`, then delete the dead `relay._reroute_waypoint`
(no call sites). Both are one-line edits; neither should be made from this session.

## 6. To close M4

1. Re-run `tests/test_m4_replan.py` on the final revision; commit the artifact with the hashes.
2. Strengthen the owner's suite so it cannot go green vacuously: T3 asserts `replans >= 1`,
   T6 asserts `replans > 0` before judging the budget, and `check()` gains `SKIP`
   (prerequisite absent ≠ PASS).
3. Write `docs/M4_REPORT.md` with numbers re-derived from the artifact, not copied.
4. Apply §4 and §5 fixes, then re-run M1/M2/M3/M4 + this suite for the closeout regression.

Only then is M5 (moving-obstacle replanning with the drone as a mobile hazard) worth starting —
and its design must distinguish, as a measured property, "the obstacle was not known at plan
time" (§2 A0/A0b are the template) from "a scripted reroute the plan already contained".
