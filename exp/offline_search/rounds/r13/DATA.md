# R13 dev data (coordinator)

## r13_devrec_50: B-pool dev recordings (launched 2026-10-04 11:31 CDT)

**Location:** `/home/weiland/trace_runs/os_closed_loop/r13_devrec_50/`. Build notes: `rounds/r13/devrec/HANDBACK.md`.

**Design:**
- 28 arms, 100 episodes each: 10 inits per task on the `r11_dev_k10_size50` selection.
- All 7 arms of a cell use the same inits, so they are paired.
- B-pool only and disjoint from the served subsets.

**Cells:** `pi05_l10_50`, `pi05_spatial_50`, `groot_l10_50`, `groot_spatial_50`.

**Arms per cell** (all are frozen `R11Knob` artifacts, referenced byte-identically):

| arm suffix | method | targets |
|---|---|---|
| `off_shadow` | knob off | — |
| `periodic_irNN` | R11 periodic | .25, .32, .40 |
| `distance_irNN` | R11 distance (offline-calibrated threshold) | .25, .32, .40 |

**What is recorded:**
- Every arm records looks under `runs/<arm>/server_*/looks/` (reader: `closed_loop/look_recording.read_looks`). Each look has keys of both cameras, robot state, the executed action, all 16 neighbours with distances and weights, and the full guard state.
- No tokens or images are recorded.
- The `off_shadow` arms also hold `shadow_policy`, the full policy chunk at every real look. These are policy-vs-cache labels on cache-driven states; served actions are unchanged.
- Decision logs are under `runs/<arm>/server_*/decisions_*.jsonl`. Journals and summaries are under `runs/<arm>/`.

**Hosts:** GR00T cells run on the local RTX 4090. π0.5 cells run on the local 4090, or on h100 if it frees up. Each cell's arms run on one host.

**Status:** arms appear as `runs/<arm>/summary.json` once collected. Check them with:

```
ls /home/weiland/trace_runs/os_closed_loop/r13_devrec_50/runs/*/summary.json
```
