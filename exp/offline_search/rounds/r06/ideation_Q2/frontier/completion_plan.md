# Provisional frontier-completion plan

**Not scheduled or executed. Finalize after all eight pilot cells arrive.** This frozen plan contains 43 configurations × 500 episodes = 21,500 episodes; at most 6 configurations per cell. 10 use existing code and have emit_arms input specs; 33 need a small deployment adapter. No profiler or controller code is built here. Existing full-run results remain the only frontier evidence.

## Placement evidence, not frontier evidence

Three complete profiler cell tables were available at the single frozen read; the path list and hashes are in `pilot_placement_paths.json` / `pilot_placement_provenance.json`. The other five cells use only existing 500-episode gaps for provisional placement. There is no polling. These 60 episodes/cohort have only 20 distinct task/init clusters; a favorable point is not a final success claim.

| Cell | Dose cohort | Episodes / init clusters | SR | Owner IR |
|---|---|---|---:|---:|
| groot_spatial_50 | A | 60 / 20 | 0.917 | 0.075 |
| groot_spatial_50 | dose125 | 60 / 20 | 0.950 | 0.129 |
| groot_spatial_50 | dose25 | 60 / 20 | 0.917 | 0.183 |
| groot_spatial_50 | dose50 | 60 / 20 | 0.917 | 0.289 |
| groot_spatial_50 | P10 | 60 / 20 | 0.900 | 0.510 |
| groot_spatial_50 | B | 60 / 20 | 0.933 | 0.117 |
| pi05_l10_50 | A | 60 / 20 | 0.717 | 0.077 |
| pi05_l10_50 | dose125 | 60 / 20 | 0.850 | 0.130 |
| pi05_l10_50 | dose25 | 60 / 20 | 0.900 | 0.189 |
| pi05_l10_50 | dose50 | 60 / 20 | 0.867 | 0.289 |
| pi05_l10_50 | P10 | 60 / 20 | 0.950 | 0.504 |
| pi05_l10_50 | B | 60 / 20 | 0.900 | 0.170 |
| pi05_spatial_50 | A | 60 / 20 | 0.733 | 0.077 |
| pi05_spatial_50 | dose125 | 60 / 20 | 0.800 | 0.128 |
| pi05_spatial_50 | dose25 | 60 / 20 | 0.833 | 0.187 |
| pi05_spatial_50 | dose50 | 60 / 20 | 0.933 | 0.297 |
| pi05_spatial_50 | P10 | 60 / 20 | 1.000 | 0.514 |
| pi05_spatial_50 | B | 60 / 20 | 0.833 | 0.170 |

For pi05 l10-50 and spatial-50, no measured sub-P10 fixed dose is within 2 pp of that pilot’s P10 point; rho=.35/.45 targets the missing high-budget interval. GR00T spatial-50 is much more optimistic in the pilot than the complete 500-init record; retain both low and moderate budgets rather than concluding that A is already sufficient. For the dense libraries, look below/around B, not only above it: existing cache methods already approach or exceed pure-policy SR. No SR is extrapolated from these pilot point estimates. Placement costs use the request-denominated owner ledger to match this frontier; P3 P10 can be slightly above .5 because episode termination interrupts the two-request commitment. It is not silently substituted for native L=10.

## Per-cell configurations


### pi05_l10_50 — 6 × 500 episodes

| Policy | Code | Cost placement | Paired comparison / hypothesis |
|---|---|---|---|
| B + extra anchor dose 0.25 | needs new deployment adapter | 0.261 forecast | versus B and both policy references; isolate added calls. More calls may close the remaining SR gap; no numerical SR prediction is identified |
| B + extra anchor dose 0.5 | needs new deployment adapter | 0.341 forecast | versus B and both policy references; isolate added calls. More calls may close the remaining SR gap; no numerical SR prediction is identified |
| B + extra anchor dose 0.75 | needs new deployment adapter | 0.420 forecast | versus B and both policy references; isolate added calls. More calls may close the remaining SR gap; no numerical SR prediction is identified |
| risk lottery rho=0.35, 10 controls | needs new deployment adapter | 0.350 forecast | versus B, nearest uniform/periodic budget, and both pure references; allocation versus cost. Risk allocation may improve SR at a matched budget; no positive gain assumed |
| risk lottery rho=0.45, 10 controls | needs new deployment adapter | 0.450 forecast | versus B, nearest uniform/periodic budget, and both pure references; allocation versus cost. Risk allocation may improve SR at a matched budget; no positive gain assumed |
| B + cap of four consecutive HIT requests | existing code; emit spec supplied | measure | versus source method, B and both policy references. guard_only retains B guards; cap can force an early vision anchor; not a fixed anchor dose |

### pi05_l10_500 — 5 × 500 episodes

| Policy | Code | Cost placement | Paired comparison / hypothesis |
|---|---|---|---|
| B + extra anchor dose 0.125 | needs new deployment adapter | 0.202 forecast | versus B and both policy references; isolate added calls. More calls may close the remaining SR gap; no numerical SR prediction is identified |
| B + extra anchor dose 0.25 | needs new deployment adapter | 0.244 forecast | versus B and both policy references; isolate added calls. More calls may close the remaining SR gap; no numerical SR prediction is identified |
| risk lottery rho=0.18, 10 controls | needs new deployment adapter | 0.180 forecast | versus B, nearest uniform/periodic budget, and both pure references; allocation versus cost. Risk allocation may improve SR at a matched budget; no positive gain assumed |
| risk lottery rho=0.24, 10 controls | needs new deployment adapter | 0.240 forecast | versus B, nearest uniform/periodic budget, and both pure references; allocation versus cost. Risk allocation may improve SR at a matched budget; no positive gain assumed |
| K7 tail confirmation | existing code; emit spec supplied | measure | versus source method, B and both policy references. confirm the observed .906 @ .197 point; do not assume that winning replicate is its true SR |

### pi05_spatial_50 — 6 × 500 episodes

| Policy | Code | Cost placement | Paired comparison / hypothesis |
|---|---|---|---|
| B + extra anchor dose 0.25 | needs new deployment adapter | 0.230 forecast | versus B and both policy references; isolate added calls. More calls may close the remaining SR gap; no numerical SR prediction is identified |
| B + extra anchor dose 0.5 | needs new deployment adapter | 0.320 forecast | versus B and both policy references; isolate added calls. More calls may close the remaining SR gap; no numerical SR prediction is identified |
| B + extra anchor dose 0.75 | needs new deployment adapter | 0.410 forecast | versus B and both policy references; isolate added calls. More calls may close the remaining SR gap; no numerical SR prediction is identified |
| risk lottery rho=0.35, 10 controls | needs new deployment adapter | 0.350 forecast | versus B, nearest uniform/periodic budget, and both pure references; allocation versus cost. Risk allocation may improve SR at a matched budget; no positive gain assumed |
| risk lottery rho=0.45, 10 controls | needs new deployment adapter | 0.450 forecast | versus B, nearest uniform/periodic budget, and both pure references; allocation versus cost. Risk allocation may improve SR at a matched budget; no positive gain assumed |
| R3 confidence quantile h=.70 confirmation | existing code; emit spec supplied | measure | versus source method, B and both policy references. existing .980 point is close to .986 L10; .70 targets HIT share, not SR |

### pi05_spatial_500 — 5 × 500 episodes

| Policy | Code | Cost placement | Paired comparison / hypothesis |
|---|---|---|---|
| A + extra anchor dose 0.03125 | needs new deployment adapter | 0.091 forecast | versus A, B and both policy references; isolate added calls. More calls may close the remaining SR gap; no numerical SR prediction is identified |
| A + extra anchor dose 0.0625 | needs new deployment adapter | 0.104 forecast | versus A, B and both policy references; isolate added calls. More calls may close the remaining SR gap; no numerical SR prediction is identified |
| risk lottery rho=0.085, 10 controls | needs new deployment adapter | 0.085 forecast | versus B, nearest uniform/periodic budget, and both pure references; allocation versus cost. Risk allocation may improve SR at a matched budget; no positive gain assumed |
| risk lottery rho=0.105, 10 controls | needs new deployment adapter | 0.105 forecast | versus B, nearest uniform/periodic budget, and both pure references; allocation versus cost. Risk allocation may improve SR at a matched budget; no positive gain assumed |
| A cached commitment 15 controls | existing code; emit spec supplied | measure | versus source method, B and both policy references. test below the cheap A10 frontier; no MISS and no policy-tail inference |

### groot_l10_50 — 6 × 500 episodes

| Policy | Code | Cost placement | Paired comparison / hypothesis |
|---|---|---|---|
| B + extra anchor dose 0.25 | needs new deployment adapter | 0.291 forecast | versus B and both policy references; isolate added calls. More calls may close the remaining SR gap; no numerical SR prediction is identified |
| B + extra anchor dose 0.5 | needs new deployment adapter | 0.360 forecast | versus B and both policy references; isolate added calls. More calls may close the remaining SR gap; no numerical SR prediction is identified |
| B + extra anchor dose 0.75 | needs new deployment adapter | 0.430 forecast | versus B and both policy references; isolate added calls. More calls may close the remaining SR gap; no numerical SR prediction is identified |
| risk lottery rho=0.35, 10 controls | needs new deployment adapter | 0.350 forecast | versus B, nearest uniform/periodic budget, and both pure references; allocation versus cost. Risk allocation may improve SR at a matched budget; no positive gain assumed |
| risk lottery rho=0.45, 10 controls | needs new deployment adapter | 0.450 forecast | versus B, nearest uniform/periodic budget, and both pure references; allocation versus cost. Risk allocation may improve SR at a matched budget; no positive gain assumed |
| CycleTail every 2 anchors, 10 controls | existing code; emit spec supplied | measure | versus source method, B and both policy references. existing GR00T A-periodic controller; no B guards; includes a first-anchor policy call |

### groot_l10_500 — 5 × 500 episodes

| Policy | Code | Cost placement | Paired comparison / hypothesis |
|---|---|---|---|
| A + extra anchor dose 0.125 | needs new deployment adapter | 0.128 forecast | versus A, B and both policy references; isolate added calls. More calls may close the remaining SR gap; no numerical SR prediction is identified |
| A + extra anchor dose 0.25 | needs new deployment adapter | 0.181 forecast | versus A, B and both policy references; isolate added calls. More calls may close the remaining SR gap; no numerical SR prediction is identified |
| risk lottery rho=0.13, 10 controls | needs new deployment adapter | 0.130 forecast | versus B, nearest uniform/periodic budget, and both pure references; allocation versus cost. Risk allocation may improve SR at a matched budget; no positive gain assumed |
| risk lottery rho=0.18, 10 controls | needs new deployment adapter | 0.180 forecast | versus B, nearest uniform/periodic budget, and both pure references; allocation versus cost. Risk allocation may improve SR at a matched budget; no positive gain assumed |
| CycleTail every 8 anchors, 10 controls | existing code; emit spec supplied | measure | versus source method, B and both policy references. existing GR00T A-periodic controller; no B guards; includes a first-anchor policy call |

### groot_spatial_50 — 5 × 500 episodes

| Policy | Code | Cost placement | Paired comparison / hypothesis |
|---|---|---|---|
| B + extra anchor dose 0.125 | needs new deployment adapter | 0.191 forecast | versus B and both policy references; isolate added calls. More calls may close the remaining SR gap; no numerical SR prediction is identified |
| B + extra anchor dose 0.25 | needs new deployment adapter | 0.235 forecast | versus B and both policy references; isolate added calls. More calls may close the remaining SR gap; no numerical SR prediction is identified |
| risk lottery rho=0.12, 10 controls | needs new deployment adapter | 0.120 forecast | versus B, nearest uniform/periodic budget, and both pure references; allocation versus cost. Risk allocation may improve SR at a matched budget; no positive gain assumed |
| risk lottery rho=0.2, 10 controls | needs new deployment adapter | 0.200 forecast | versus B, nearest uniform/periodic budget, and both pure references; allocation versus cost. Risk allocation may improve SR at a matched budget; no positive gain assumed |
| CycleTail every 3 anchors, 10 controls | existing code; emit spec supplied | measure | versus source method, B and both policy references. existing GR00T A-periodic controller; no B guards; includes a first-anchor policy call |

### groot_spatial_500 — 5 × 500 episodes

| Policy | Code | Cost placement | Paired comparison / hypothesis |
|---|---|---|---|
| risk lottery rho=0.065, 15 controls | needs new deployment adapter | 0.065 forecast | versus B, nearest uniform/periodic budget, and both pure references; allocation versus cost. Risk allocation may improve SR at a matched budget; no positive gain assumed |
| risk lottery rho=0.09, 15 controls | needs new deployment adapter | 0.090 forecast | versus B, nearest uniform/periodic budget, and both pure references; allocation versus cost. Risk allocation may improve SR at a matched budget; no positive gain assumed |
| A15 confirmation | existing code; emit spec supplied | measure | versus source method, B and both policy references. validate the low-IR region already above the policy point estimate |
| phase-particle pure-cache confirmation | existing code; emit spec supplied | measure | versus source method, B and both policy references. validate the low-IR region already above the policy point estimate |
| CycleTail every 8 anchors, 15 controls | existing code; emit spec supplied | measure | versus source method, B and both policy references. compare low-dose committed policy rescue against A15; native L15 SR is unmeasured |

## Exact serving recipes and code availability

**B plus fixed additional dose d (new adapter):** at each genuine vision anchor, compute B’s original proposal/guard. If B mandates policy, call it. Otherwise call iff a private hash uniform variate is below d. Preserve B’s current cache/return hooks, 10-control policy lifecycle tail, retrieval state invalidation, and early looks. Do not inspect a shadow action to choose the source. Hash `(manifest SHA, task, original init, replicate, anchor index, "Q2-extra-call-v1")` as compact UTF-8 JSON, take the first 64 SHA256 bits big-endian divided by 2^64. Use a separate policy-noise RNG stream. This is an extra-call probability on B-cache opportunities, not total m; guard state feedback makes the cost forecast approximate. A+d uses the identical Bernoulli mechanism with A and no mandatory guards. Fixed d is an experimental placement parameter, not the eventual owner interface.

**Risk target rho (new adapter):** use `../PREREG.md`’s library-only weights and 80-iteration budget solve, with the exact per-task dose-lottery weights emitted in `completion_risk_allocations.csv` / `needs_code_configs.json`. At episode start sample one of {0,.125,.25,.5,1} from those weights using the frozen `Q2-episode-dose-v1` hash domain; hold that Bernoulli anchor policy for the episode. No validation-success or LIBERO task-name threshold enters the allocation. The default is 10-control commitment. The GR00T spatial-500 low-cost extension explicitly uses 15 controls (H=16 permits three R=5 blocks), and its library anchor opportunities are recomputed as mean ceil(episode_decisions/3), not borrowed from the 10-control estimate. This extension is a new proposed policy version and has no P3-identifiable 15-control SR. The owner knob stays rho; alpha is an internal guard calibration parameter, not another user control. The fallback library risk has not demonstrated a portable SR guarantee. Q1 may replace it only before outcome inspection for the new arm, with a verified library/metric mapping; a score chosen using pilot outcomes needs independent final evaluation.

**Existing code:** B is `q1_commit.judge:CommitJudge` (pi05) or `p1_groot_commit.judge:GrootCommitJudge` (GR00T). `--os-judge guard_only --os-judge-cap 4` retains guards and adds a consecutive-HIT cap; cap counts request slots including blind tails, can force early vision, and is not “every four anchors.” GR00T `q2_groot.judge:CycleTail(cycle_k=k, tail_blocks=1 or 2)` is an A-periodic controller with no B guards; it is not B+periodic rescue. Other supplied existing specs copy the tested R3 quantile / K7 configurations or change `BlindAWM` cached-tail budget from 1 to 2. Exact kwargs, client overrides, full_model, cost_ledger, and plugin arguments are in `emit_arms_existing.json`.

**Two shortcuts are unavailable:** `--os-judge periodic:k` ignores force-MISS guard flags and uses a shared server decision clock in blind mode (`closed_loop/plugin.py:853,873`); it does not implement B+d. P3 v2 `blind_shadow=False, resample_p=0` still computes shadow policy at every vision anchor (`v2_engine.py:79`). `enabled=False` bypasses its profiling/randomization path. Thus “P3 fixed dose with no shadow” is not an existing deployable configuration, and no such emit spec is claimed here.

## Pairing, validation, and finalization

Each proposed arm evaluates tasks0–9 × inits0–49 once, using `eval500_manifest.json`, identical environment seed/reset and matching policy-noise seed where meaningful. Pair comparisons by task/init and preserve any repeated blocks as clusters. Existing B and policy runs are retrospective references; pairing IDs alone does not make historical harness/seed differences disappear. If strict contemporaneous baseline replication is required, substitute B/P10 replays for lower-priority proposed configurations within the six-per-cell cap rather than silently enlarging the campaign. Use 2 pp SR NI against **both separately reported L=5/L=10 references**, exact McNemar equality diagnostics, and a valid paired difference bound. 500 episodes may not certify a 2 pp loss when discordance is high; retain “inconclusive” and report the point frontier plus intervals. The supplied plan does not promise a crossing or universal transfer.

When all pilot cells are available, finalize once: retain the existing near-crossing-family confirmation in each cell; use pilot fixed-dose knots only to bracket the P10−.02 crossing; place two rho values within the bracket (at one-third and two-thirds of the bracket in measured IR, rounded to the nearest .01). If no sub-P10 dose reaches it, use the [.5-dose,P10] interval; if A already reaches it, use the [A,.125-dose] interval but retain an intermediate historical-gap check when pilot and full-run evidence disagree. Choose up to three extra-B doses whose constant-cadence forecast lies closest to those budgets and their midpoint, from {1/16,1/8,1/4,1/2,3/4}; deduplicate. Dense-library 15-control candidates remain a separately labeled low-cost extension. Freeze the final file/hash before its 500-episode outcomes; do not turn the pilot’s best seed/task into the final success estimate.

## Artifacts and reproduction

`emit_arms_existing.json` is the input-spec list for existing classes only. `<RUN>` placeholders must be replaced by the coordinator; no emit_arms invocation was made. Copy `eval500_manifest.json` to `<RUN>/manifests/eval500.json` in a future authorized experiment. `needs_code_configs.json` is a design specification, not executable emit input. `emit_specs_static_validation.json` records AST class checks and source hashes; runtime/fitting/rollout validation is unperformed.

```bash
taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/offline_search/rounds/r06/ideation_Q2/frontier/prepare_completion_plan.py
```
