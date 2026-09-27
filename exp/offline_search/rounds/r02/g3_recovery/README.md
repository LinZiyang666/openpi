# R2 family G3 — closed-loop mechanisms: V6 `stuck_recovery`, V7 `drift_calibrated_confidence`

Both are **wrappers** around any base selector (G1 AWM, G2 PCA-cosine, or the stand-ins here). A wrapper does not
re-implement retrieval: it asks the base for the scores of ALL admissible candidates (the hook below) and redoes the
top-k + kernel synthesis itself, so it can exclude candidates (recovery), force a member (blend) and compute
features (confidence) over any base. With every mechanism off the wrapper reproduces `base.query()` bit for bit
(`tools/contract_check.py`).

Files: `wrappers.py` (V6 `StuckRecovery`, V7 `DriftCalibratedConfidence`, `Passthrough`), `g3_core.py` (hook
access, selection / synthesis, library tables, thresholds, LOEO pseudo-queries, PAV / 10-bin maps), `standins.py`
(stand-in bases with the hook), `batch.json` (+ `_make_batch.py`), `tools/` (contract check, subset runner,
analysis), `smoke/` (smoke transcripts). Arrays / run outputs: `/home/weiland/trace_runs/offline_search_store/derived/r02/g3_recovery/`.

## Base contract (what G1 / G2 must add so G3 can wrap them)

After `fit()` the base must provide:

| member | contract |
|---|---|
| `os_score_all(q) -> (rows, scores, aux)` | **Required** (the name `score_all` is accepted too; then only the first two tuple elements are used and aux only if the 3rd is a dict). `rows` int[C]: library rows (into `os_library`) of **all admissible candidates** of the query (task filter, step-0 alignment etc. already applied; C ≥ 1). `scores` float[C]: the base's ranking score, higher = better, finite — the base's own top-k must be `argsort(-scores, stable)[:synth_k]` (ties → lower position). `aux` dict of per-candidate float[C] arrays; **V7 requires** `"vis_v0"` and `"vis_v1"` (per-camera visual similarity in the base's own representation, higher = more similar; e.g. task-centred PCA cosine, or minus the vision-block distance for AWM) or a single `"vis"`. Must be **stateless** (a pure function of `q` and the fit): V7 calls it on library pseudo-queries during `fit`. It may read any QueryView field (`key_v0/1`, `rs`, `step`, `task_id`, `prev_hit`, `prev_a_exec`, `hist_*`); pseudo-queries have no `raw_state` (NaN) and no tokens. |
| `os_library: str` | the library `rows` index (= the base's `Result.library`: `"current"`, `"bpool_cs"`, `"bpool_all"`). |
| `os_fit_library: str` | the library its fitted parameters (PCA basis, metric, scales) come from; default = `os_library`. G3's thresholds and V7's calibration are fitted on it ("calibration data follows fit_data"). |
| `synth_k: int`, `synth_T: float` | the base's synthesis: kernel mean of the top `synth_k` with weights `exp(-(S_0 - S_i) / synth_T)` (`math.inf` = plain mean, `synth_k = 1` = the top-1 row served as is). |
| `os_synth(q, rows, w) -> (H, 32) float32` | *optional*: the base's own synthesis of given ordered rows / unnormalized weights (e.g. G1's V3 norm-preserving mean + gripper hysteresis). Without it G3 uses `g3_core.synth` (weighted mean of the full chunks, float64 accumulation). |
| `os_confidence(q, rows, scores, aux, served_rows, w) -> float` | *optional*: the base's confidence for a given served set (used by V6 with `conf="base"`; else the top-1 score). |
| `os_uses_prev: bool` | *optional* (default True): False if `os_score_all` ignores `prev_hit` / `prev_a_exec` / `hist_*` (halves V7's calibration calls). |

`reset()` of the wrapper calls `base.reset()`. The base instance lives inside the wrapper (deep-copied per
connection by the plugin; pickled inside the fit artifact — the wrapper's `__reduce__` imports the base module first).

Check a new base (must print `"PASS": true`):
```bash
taskset -c <cpus> .venv/bin/python exp/offline_search/rounds/r02/g3_recovery/tools/contract_check.py \
    --base exp/offline_search/rounds/r02/g1_awm/<file>.py:<Class> --base-kwargs '{...}' --cell groot_l10_cache --episodes 10
```

## V6 `StuckRecovery` (B-P2) — exact algorithm

Per decision (state per episode in `self._s`, reset in `reset()`):
1. `rows, S, aux = base.os_score_all(q)`.
2. **Anchor** (the previous decision's served row): the candidate-library row whose chunk equals `prev_a_exec[:, :7]`
   bit for bit (offline cache arms: the recorded B0 pick when it is in the candidate library; online: an R3 top-1
   row), else this wrapper's own previous top-1 (online with a synthesized action; `anchor="self"` forces this).
3. **Detector** (online data only; thresholds from the fit library):
   `still_t = |rs_t − rs_{t−1}|[:8] < m_thr` (10th pct of the library's per-decision motion) **and**
   `min_cam cos(key_t − m_task, key_{t−1} − m_task) ≥ c_thr` (95th pct of the same cosine between consecutive
   library decisions; `m_task` = the fit library's task-mean raw key); `stuck_n` = consecutive still decisions;
   `overtime = step / median library ep_len of the task`; `terminal` = anchor has no `next` (after a HIT);
   `lag = step − mean library step of the base top-5`; **stuck := stuck_n ≥ 2 OR terminal OR (overtime > 1 AND lag > 5)**.
4. **Recovery** (after a HIT only; `esc_n` = stuck decisions since the last reset; reset after 2 consecutive decisions
   with `stuck_n = 0`): level = min(3, 1 + (esc_n − 1) // 2), i.e. R1 on the 1st–2nd stuck decision, R2 on the
   3rd–4th, R3 from the 5th.
   R1: exclude the library episodes of the anchors of the last 3 decisions, re-select with the base scores.
   R2: additionally keep only candidates with library progress ≤ `ref_prog − 0.2`; `ref_prog` = progress of the anchor
   when the stuck spell began, frozen for the spell (otherwise every R2 decision would rewind again).
   R3: the top-1 of the R2 set, served as the library row (`Result.action = None`).
   If a restriction empties the candidate set it is relaxed (`relaxed` extra: 1 rewind dropped, 2 exclusion dropped).
5. **Blend** (after a HIT, not while recovering, anchor has a `next`): `next(anchor)` joins the top-k with the top
   weight (w = w_0 = 1): if it is already in the top-k its weight is raised, else it replaces the k-th member
   (k = 1 bases: appended, 50/50). The next anchor is still the S-top-1 of the selection (topk[0]), so tracking
   cannot self-perpetuate.
6. Confidence: `conf="base"` (base's `os_confidence`, default), `"score"` (S of the top-1) or `"v7"` (V7's).

Options (all encoded in the name): `blend=False` / `recover=False`; `anchor="self"`; `blend_w=<beta>` (next gets the
fixed final share beta, the top-(k−1) others 1 − beta — measured worse than the default "top weight", see results);
`ot_still=True` (the overtime-and-lag clause counts only while stuck_n ≥ 1 — halves the STUCK rate on l10, see
results); `lag_thr`, `stuck_thr`, `m_pct`, `c_pct`, `excl_win`, `rewind`, `esc`.

Extras: `regime, motion, vself, still, stuck_n, overtime, terminal, lag, stuck, level, esc_n, relaxed, n_excl, n_cand,
blend, nx_in, w_nx, anchor, anchor_src (1 exec match / 0 self), anchor_ep, top1, top1_ep, top1_step, top1_prog,
k_sel, k_eff, dnn, disp, vis, cont, zsum, pred_err` (NaN-valued ones are left out = NaN in the npz).

## V7 `DriftCalibratedConfidence` (B-P3) — exact algorithm

Base selection unchanged (same err as the base). Features of the served set / candidate pool:
- stale (after a HIT) and step 0: `−d_nn` (min rs[:8] distance to the candidates), `−disp` (mean pairwise RMS of the
  top-5 heads, σ units), `−overtime`, `vis = max vis_v0 + max vis_v1` (base aux), `−min(stuck_n, 5)`, `−|lag|`;
- fresh (after a MISS): `−cont` (RMS of served head − previous chunk's tail [5:10, :7], σ units), `−disp`, `vis`.
`zsum = Σ sign · (f − μ_f) / σ_f`; `pred_err = map_regime(zsum)` (isotonic PAV, non-increasing, linear between block
centroids; or 10 quantile bins + weighted PAV); **confidence = −pred_err + 1e-6 · zsum** (ties of flat pieces broken
by the z-sum, so within a regime the ranking is exactly the z-sum's; across regimes the unit is predicted err).
μ, σ and the three maps (step 0 / fresh / stale) are fitted on **library LOEO pseudo-queries** of the fit library:
every step-0 row, plus ≤ `ncal` = 3000 random rows with step ≥ 1 posed once as stale (`prev_hit=True`) and once as
fresh (`prev_hit=False`, the previous chunk = the library episode's own previous row); candidates of the same episode
are excluded; GT = the row's own action; stuck_n / overtime / lag from the library episode's own history. Failed
library episodes are included when the library has them (10× libraries: 13 / 64 / 44 / 73 of 500 episodes failed in
π0.5-sp / π0.5-l10 / GR00T-sp / GR00T-l10; the `current` libraries are 100 % successful — so the current-fit
calibration has no failed-episode drifted states).

## Variants (`batch.json`, 14 entries; `python3 _make_batch.py` rewrites it)

Stand-in base `standins.py:VZSKernel` (r01 M9c vision z-sum, st = 1, kernel-8 T = 0.5, step-0 alignment; current:
raw keys, big: the r01 big-library PCA-128 codes) — for lib ∈ {current (fit current), big (fit big)}:

| name (lib = current; `big` analogous with `G3sb_vzs_big_pca128_...`) | what |
|---|---|
| `G3sb_vzs_current_raw_st1_k8_T0p5_al0` | the base alone (reference for every paired delta) |
| `V6sr_bl1rc0__<base>` | V6 blend only, anchor = executed row (offline blend effect) |
| `V6sr_bl1rc0_aself__<base>` | V6 blend only, anchor = own previous top-1 (the closed-loop semantics) |
| `V6sr_bl1rc1__<base>` | V6 full (blend + detector + recovery); flags in extras; its offline err is NOT a verdict |
| `V6sr_bl1rc1_ots__<base>` | V6 full with `ot_still=True` (recommended CL3 detector, see results) |
| `V7dc_iso__<base>` / `V7dc_bin10__<base>` | V7 confidence, isotonic / 10-bin map |

**Over G1 / G2 once they land** (after their base passes `tools/contract_check.py`), per G1/G2 variant to wrap:
```json
{"method": "exp/offline_search/rounds/r02/g3_recovery/wrappers.py:StuckRecovery",
 "kwargs": {"base": "exp/offline_search/rounds/r02/g1_awm/<file>.py:<Class>", "base_kwargs": {<the G1 variant's kwargs>}},
 "family": "g3_recovery", "cells": "all", "subsample": null, "allow_gpu_fit": false}
```
(+ `"recover": false` for the blend-only row, `DriftCalibratedConfidence` with the same `base` / `base_kwargs` for V7).
The wrapper inherits the base's library and fit data (`os_library`, `os_fit_library`), so wrapping the "current, fit
current" base variant gives the mandatory current-only V6/V7 variant, and wrapping the big one the 10× variant.
Names are `<V6sr_bl1rc1|V7dc_iso|...>__<base name>`, so they never collide across bases. CL3 (closed loop) =
`StuckRecovery` around the CL2 AWM config with V3 on (V3 must then be exposed through `os_synth`, see the contract).

## Closed loop (plugin)

`--os-method exp/offline_search/rounds/r02/g3_recovery/wrappers.py:StuckRecovery --os-kwargs '{"base": ..., "base_kwargs": {...}}'`.
Per-episode state is in the instance (`self._s`) and reset in `reset()`; the plugin's per-connection deepcopy shares
every fitted array (`clone_mode = deepcopy_shared_arrays`); the fit pickles (`--os-fit-artifact`; 102 MB for GR00T-sp
big incl. two copies of the 10× action table). ⚠ Run the prefit with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`
(the servers run single-threaded numpy): an artifact fitted with multi-threaded BLAS differs from an offline fit in
the last bit of float64 fit statistics (observed: 3.5 % of decisions with last-bit score differences, top-k / served
action identical); single-threaded prefit → bit-exact.
In the pure-cache loop the executed chunk is V6's own synthesized mean, so the anchor is always the own previous top-1
(or, after R3, the served library row, matched exactly).

## Results (development numbers; the full run is the coordinator's)

Cells π0.5-sp / π0.5-l10 / GR00T-sp / GR00T-l10. **20 % subset** = every 5th episode (100 episodes / cell; stale
decisions 2913 / 8265 / 2687 / 7794), same runner code path as the full run (`tools/subset_launch.py`, outputs in
`derived/r02/g3_recovery/sub5/`, report `sub5/analysis.txt` from `tools/g3_analysis.py`). Stand-in base = VZSKernel.
Library scale: **current** = 49–50 episodes, 1018 / 2640 / 1063 / 2645 entries, stand-in key 262 KB/entry (raw keys);
**10×** = 500 episodes, 10909 / 29472 / 11751 / 29631 entries, stand-in 1056 B/entry (2 × PCA-128 + rs) + 33.5 MB
PCA bases / suite. The wrappers add **46 B/entry** (next, episode, step, progress, rs[:8] — the last shared with any
base that stores rs) and **2.6 MB / suite** fixed (task-mean raw keys of both cameras for the visual self-change);
V7's maps / scales are < 10 KB.

Stand-in base itself (err): 10× stale .522 / .482 / .463 / .464 (step 0 .222 / .154 / .264 / .164, fresh .322 / .281 /
.370 / .346); current stale .583 / .548 / .556 / .530. (≈ B-P1 of the ideation: .524 / .454 / .442 / .469 on 10×.)

### V6 blend (offline effect; stale decisions, wrapper − base, paired, 95 % episode-bootstrap CI)

| lib | anchor | π0.5-sp | π0.5-l10 | GR00T-sp | GR00T-l10 |
|---|---|---|---|---|---|
| 10× | exec (π0.5: no exec match in bpool_cs → self) | −.0035 [−.0052, −.0019] | +.0004 [−.0031, +.0043] | −.0021 [−.0047, +.0003] | −.0001 [−.0077, +.0067] |
| 10× | self (closed-loop semantics) | −.0035 (same) | +.0004 (same) | −.0048 [−.0087, −.0016] | −.0027 [−.0051, −.0002] |
| current | exec (= recorded B0 pick) | −.0049 [−.0108, +.0001] | −.0015 [−.0036, +.0004] | −.0036 [−.0067, −.0013] | −.0029 [−.0069, +.0006] |
| current | self | −.0014 [−.0028, −.0001] | +.0026 [−.0004, +.0063] | −.0030 [−.0057, −.0004] | −.0023 [−.0065, +.0013] |

The blend applies on 74–96 % of stale decisions; next(served) is already in the base's top-8 in 41–98 % of them (so
"top weight" only raises its weight: w_nx p50 .18–.45). The gain sits mostly in episodes B0 failed (current, exec
anchor: −.010 / −.002 / −.007 / −.004 there).
**Below the ideation's −.008…−.023 and below B-P2's offline kill line (≤ −.005 on 10× stale) in every cell.**
Diagnostics: (i) on a plain mean-5 base (the ideation's synthesis) the same blend is −.001…−.004 (CI mostly < 0) —
the ideation's larger number came from double-counting next(served) when it is already in the top-4 (its weight
becomes 2/5); (ii) a fixed final share `blend_w` = .25 / .4 for next(served) is never better than the top-weight blend
(Δ vs base −.003…+.013) — the kernel weights beat a forced share. Gripper-sign-faithful err moves the same way (10× blend
stale-gs .533 / .492 / .475 / .480 vs base .537 / .493 / .478 / .479).

### V6 detector (stale decisions; the base's err on flagged / unflagged decisions)

| 10× | π0.5-sp | π0.5-l10 | GR00T-sp | GR00T-l10 |
|---|---|---|---|---|
| still (motion < m10 ∧ vis-cos ≥ c95) | .145 | .151 | .147 | .091 |
| stuck_n ≥ 2 | .110 (.71 / .50) | .134 (.60 / .46) | .116 (.85 / .41) | .079 (.54 / .46) |
| terminal served row | .151 (.87 / .46) | .046 (.67 / .47) | .209 (.82 / .37) | .056 (.60 / .46) |
| overtime > 1 | .306 | .359 | .244 | .339 |
| lag > 5 | .229 | .428 | .156 | .373 |
| overtime ∧ lag | .216 (.85 / .43) | .310 (.64 / .41) | .148 (.89 / .39) | .272 (.62 / .41) |
| **STUCK** (spec) | **.305 (.83 / .39)** | **.365 (.63 / .40)** | **.241 (.83 / .35)** | **.312 (.62 / .39)** |
| R1 / R2 / R3 share | .188 / .031 / .085 | .230 / .017 / .117 | .118 / .021 / .102 | .232 / .010 / .071 |

Current library: STUCK .314 / .423 / .260 / .382 (err flagged .91 / .69 / 1.07 / .69 vs unflagged .43 / .44 / .38 /
.43). The flags pick the lost states (err ×2 of the rest). The overtime-∧-lag clause alone flags 15–36 % of stale
decisions, mostly while the robot moves: on l10 it doubles the STUCK rate. Replay of the recorded flags
(`tools/detector_variants.py`, `sub5/detector_variants.txt`): with `ot_still=True` (that clause only while stuck_n ≥ 1)
STUCK drops to .257 / .185 / .215 / .133 (10×) with the same flagged err (.81 / .62 / .81 / .59) and R1 only
.154 / .061 / .095 / .060 — **recommended for CL3** (fewer trajectory switches while the arm is moving). Offline err of
full V6 (recovery on) is not a verdict (the recorded states do not respond): 10× stale .557 / .518 / .444 / .476 vs
base .522 / .482 / .463 / .464.

### V7 confidence (AURC; stale = cache step ≥ 1, fresh = inf step ≥ 1, step 0 = inf step 0; pooled p = .5)

| | stale | fresh | step 0 | pooled own | pooled rank-norm | pooled opt |
|---|---|---|---|---|---|---|
| 10× base's own | .379 .373 .357 .361 | .247 .209 .325 .273 | .199 .131 .224 .148 | .303 .278 .340 .313 | .313 .291 .341 .317 | .208 .184 .231 .217 |
| 10× **V7 iso** | **.314 .312 .302 .324** | **.206 .177 .261 .225** | .185 .117 .228 .138 | **.246 .226 .278 .266** | .260 .245 .281 .275 | |
| current base's own | .427 .392 .382 .416 | .347 .304 .370 .365 | .260 .215 .363 .222 | .384 .342 .375 .390 | .387 .348 .376 .391 | .259 .246 .273 .272 |
| current **V7 iso** | **.361 .369 .349 .380** | **.274 .258 .318 .314** | .244 .199 .370 .263 | **.313 .304 .331 .343** | .318 .313 .333 .347 | |

`bin10` = iso ± .001. V7 beats the base's own confidence in every cell for stale (−.02…−.07) and fresh
(−.03…−.07) and is below M8's B0-score stale AURC (.333 / .322 / .323 / .346, R1 full run) in 4/4 cells on 10×
(B-P3 kill criterion 1 passed on the subset); its pooled one-threshold AURC is *better* than regime-wise rank
normalisation (by .002–.019; kill criterion 2 passed): the predicted-err scale is shared by the regimes. Step 0 is
mixed: GR00T current (50 calibration rows) .370 / .263 vs the base's .363 / .222 — the base's top-1 similarity ranks
step 0 better than {d_nn, disp, vis, lag} there. In-sample calibration (library LOEO): corr(z-sum, err) −.35…−.54
stale, −.70…−.76 fresh.

### Closed-loop readiness (plugin)

`tools/cl_evidence.sh` = plugin `selftest` (real per-connection stack on recorded keys, `--os-log-inputs`) + plugin
`verify_logs` (offline harness replay of the LOGGED online inputs, i.e. with V6's own synthesized chunks as history):

| run | decisions | online == offline (topk / scores / conf / synth / extras incl. stuck_n, esc_n, level, anchor) | executed == selected | state-machine coverage |
|---|---|---|---|---|
| V6 current, π0.5-sp | 604 | 1.0 all | 604 / 604 | R1 114, R2 14, R3 53, blend 403, terminal 84 |
| V6 10×, GR00T-sp, **fit artifact** | 548 | 1.0 all | 548 / 548 | R1 49, R2 8, R3 24, blend 447 |
| V6 current, GR00T-l10, fit artifact | 887 | 1.0 all | 887 / 887 | R1 259, R2 6, R3 4, blend 606 |
| V6 10×, π0.5-l10, fit artifact | 1071 | 1.0 all | 1071 / 1071 | R1 298, R2 17, R3 137, blend 607 |
| V7 current, π0.5-l10 | 1071 | 1.0 all | 1071 / 1071 | (confidence incl. stuck_n) |

The selftest's own offline check can only compare the first decision of each episode for a synthesized-action method
(history diverges from the recording); verify_logs covers every decision. `tools/contract_check.py`: Passthrough ==
base.query() bit for bit (top-k, synthesized action, confidence) for VZSKernel current / 10× and for B0KernelHook
(kernel / mean) — whose `query()` is the untouched r01 M4 code.

### Cost

- **ms/query** (single thread, one process pinned to one core, machine shared with other agents; `derived/.../timing/overhead.txt`):
  wrapper overhead on top of the base's `os_score_all` p50 **1.29 ms (V6) / 1.10 ms (V7)** on GR00T-l10 10× (3138
  candidates / task), 1.24 / 0.98 ms on π0.5-l10 10×, 0.93 / 0.81 ms on π0.5-sp current. The biggest parts: the visual
  self-change (two 32768-d centred cosines vs the previous decision, ~0.22 ms), the base's own `os_confidence` (V6,
  stand-in: ~0.12 ms), feature / top-k / synthesis bookkeeping. Stand-in base alone: 5.3–5.8 ms on 10× (its PCA-128
  projection of the raw keys dominates — a stand-in cost; G1/G2 bases are ~1 ms, so wrapper + base ≈ 2–2.5 ms).
  Harness smokes (3 in parallel, loaded machine): `smoke/SUMMARY.txt`.
- **fit**: base fit + tables / thresholds 2–35 s; V7 adds the LOEO calibration (≤ 3000 × 2 + #episodes pseudo-queries;
  1 call each when the base declares `os_uses_prev = False`): 6–10 s current / 30–65 s 10× with the stand-in.
- **fit artifact** (plugin pickle): 102 MB GR00T-sp 10×, 155 MB π0.5-l10 10× — dominated by two copies of the 10×
  action table (the base's and the wrapper's) and the stand-in's PCA bases; 711 MB GR00T-l10 current (the stand-in's
  raw 262 KB keys).

### Caveats

- Stand-in base only. Over G1 / G2 the blend / flag / confidence numbers must be re-measured (the coordinator's run);
  the mechanisms do not depend on the base beyond the contract.
- The `current` libraries have no failed episodes, so the current-fit V7 calibration never sees library drifted
  states (the 10× ones do: 13–73 failed episodes); still, current V7 beats the base's confidence in 4/4 cells.
- V7's step-0 map at current size is fitted on 49–50 pseudo-queries (ranking within step 0 = the z-sum's anyway).
- Recovery (R1–R3) cannot be evaluated offline; only its triggers can (flags + err there). The anchor of the offline
  cache arms is the recorded B0 pick when it is in the candidate library (π0.5 bpool_cs never contains it → self
  anchor); in the closed loop it is always V6's own previous top-1.
- Offline full-V6 stale err is worse than the base by construction (it serves non-best candidates while stuck):
  10× +.035 / +.036 / −.019 / +.012, current +.039 / +.040 / +.031 / +.026 (GR00T-sp 10× the only improvement).

## Files

```
g3_recovery/
  wrappers.py        V6 StuckRecovery, V7 DriftCalibratedConfidence, Passthrough (G3Wrapper)
  g3_core.py         hook access, top-k / kernel / synthesis, Tables, thresholds helpers, PAV / 10-bin, PseudoQuery
  standins.py        VZSKernel (r01 M9c + hook + kernel-8), B0KernelHook (r01 M4 + hook)
  batch.json         14 variants (_make_batch.py)
  _smoke_all.sh      harness smokes -> smoke/*.txt, smoke/SUMMARY.txt
  tools/contract_check.py     Passthrough(base) == base.query()  (run it on every new base)
  tools/cl_evidence.sh        plugin selftest + verify_logs (+ optional single-thread prefit artifact)
  tools/subset_job.py, subset_launch.py   every-n-th-episode runs through run.run_cell
  tools/g3_analysis.py        regime err, blend effect (paired CI), detector flags, V7 AURC / pooled AURC
  tools/detector_variants.py  replay the detector / escalation on recorded extras for other clause settings
derived/r02/g3_recovery/ (not in the repo): sub5/ (+ analysis.txt, detector_variants.txt), sub5_mean5/, cl/
  (selftest inputs + verify reports, evidence_summary.jsonl), contract/final_summary.jsonl, timing/, smoke/
```
