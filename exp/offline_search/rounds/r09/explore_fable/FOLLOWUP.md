# R9 explore_fable — follow-up (coordinator tasks of 2026-10-01 21:1x CDT)

Inputs read after my PROPOSALS.md was frozen: `explore_astra/{REPORT.md,PROPOSALS.md,methods.py,inference.py,
confirmation_specs.json}` and the discovery journals of `r09_astra_confirmation` (π0.5 Spatial-50 arms only; no
LIBERO-10 result existed locally, see `PREDICTION_L10.md` timestamp 21:20:32 CDT). Nothing from inits 30–49 was read.

## 1. Why the half-residual head gains and my grown library does not

Both use the same labels (deferred policy shadow at cache look decisions). Three differences, with evidence.

**(a) It is not the offline gap.** On the same held-out decisions (pure-cache arm, inits 20–29; `tools` session,
`DATA_ANALYSIS.md` metrics) the *grown* library is closer to the policy than the half residual, yet it lost
closed loop while the residual won on Spatial-50:

| cell | served | residual half | residual full | grown (full replacement) | grown half-blend, motion only |
|---|---|---|---|---|---|
| π0.5 L10-50 | .412 | .292 | .222 | .254 | .299 |
| GR00T L10-50 | .525 | .354 | .244 | .293 | .369 |
| π0.5 Sp-50 | .518 | .335 | .212 | .289 | .358 |

Per task the half residual lowers the gap uniformly (every task −25% to −40%), including tasks where the policy
is weaker than the cache. So "distance to the policy" ranks these two methods the wrong way; the explanation must
be dynamical.

**(b) Trajectory consistency: the residual corrects the cache's own chunk; the grown library switches sources.**
From the R4 server logs of my confirmation run (π0.5 L10-50, 100 pairs each):

| arm | gripper sign toggles / 100 controls | top-1 neighbour changes source episode between consecutive looks |
|---|---|---|
| pure cache A | 1.92 | 43% |
| grown, 6,763 rows | 2.27 (+18%) | 50% |
| grown, 73,945 rows | 2.35 (+22%) | **63%** |

The grown retrieval draws 72–97% of its weight from rows of many different cache episodes (effective source
episodes among the 16 neighbours 3.3 → 5.0 → 5.9), so consecutive looks follow different recordings more often
and the gripper toggles more — exactly the chattering signature R8 identified as the harm of looking every 5 (A5:
+25–50% gripper toggles, −6 to −8 pp). The residual head leaves retrieval, neighbours and commitment untouched
(switch rate unchanged by construction) and adds a smooth, ridge-shrunk function of (keys, state, cached chunk).

**(c) Gripper and strength.** The grown rows replace the gripper with the policy's (grip disagreement to the
shadow .104 → .058); the residual never touches it. Half strength keeps the corrected chunk inside the cache's
own neighbourhood (shrinkage toward the demo manifold); the grown library replaces it fully. Where the policy is
*worse* than the demos (π0.5 L10-50 tasks 2 and 3: P10 .83/.90 vs cache .90/.97), full replacement followed the
policy down (−40–50 pp); a half-strength motion-only correction should lose far less there (prediction below).

**(d) Cell.** Spatial-50's cache gap is dominated by one task (6: .37 → .87 under the residual, 79% of its gap
recovered) and no task was hurt; the residual recovered 65% of the overall P10 gap. Long-horizon LIBERO-10 adds
compounding and grasp precision; my grown library's per-task pattern there was help on 0/4/9, harm on 2/3.

**Prediction for the running LIBERO-10-50 residual arms** (per task, written at 21:20 CDT before any result):
`PREDICTION_L10.md`. Headline: π0.5 L10-50 residual_half .76–.82 (point .79, +4 to +10 pp over the paired cache
arm; gains on tasks 0, 4, 6, 7, 9; no gain or small loss on 2, 3; ≥ 8 pp below P10); GR00T L10-50 .70–.78 (point
.74, +5 to +12 pp; gains on 0, 7, 9, 4, 6).

## 2. Combination: residual everywhere + calls only on the hardest tasks (`r09_fable_combo`, prepared, not launched)

Design: astra's frozen `ResidualCache` (blend .5, motion only, 10-control commit) serves every task; an
independent per-anchor call coin with task-dependent probability `task_p[t]` adds policy calls (with CU's policy
tail lifecycle) on the k hardest tasks ranked label-free by the per-task shadow gap on inits 0–29 (`out/task_alloc/`):
π0.5 Sp-50 top-2 = tasks {6, 9}; π0.5 L10-50 top-3 = {0, 9, 7}; GR00T L10-50 top-3 = {7, 9, 0}. Two doses: p = .5
at anchors (≈ ρ .29 on those tasks) and p = 1 (pure policy on those tasks). Code: `tools/combo_method.py::ResidualTaskCalls`
(subclass of R8 `AnchorCalls`; only the base and `_assignment` differ; 15/15 unit tests pass; CPU plugin selftest with `--blind --policy-tail --judge guard_only` on the π0.5 L10-50 artifact: PASS, 48 decisions, 8 forced MISSes, 8 policy tails, duplicate requests rejected, 0 preflight rejections).

**Dose-simulator estimate, π0.5 Spatial-50** (residual outcomes from astra's discovery arm on the easy tasks,
R8-main call-arm outcomes on the hard tasks, paired on the same 300 pairs; cross-topology for the hard tasks):

| hard set | hard-task controller | combo SR @ IR | vs residual alone .933 @ .078 |
|---|---|---|---|
| {6, 9} | CU-like calls (ρ .3) | .950 @ .124 | +1.7 pp [−0.3, +3.7] |
| {6, 9} | pure policy | .960 @ .162 | +2.7 [+0.7, +4.7] |
| {6, 9, 0} | CU-like calls | .953 @ .143 | +2.0 [0.0, +4.3] |
| {6, 9, 0} | pure policy | .970 @ .194 | **+3.7 [+1.7, +6.0]** |

So on Spatial-50 the residual already captures most of what calls bought (CU .927 @ .304 is below residual alone
.933 @ .078); calls on the two hardest tasks add 2–4 pp for +.05–.12 IR, reaching .96–.97 vs P10 .993. For
LIBERO-10-50 no residual outcome exists yet; using my prediction (.79) as the easy-task baseline and R8 CU/P10 on
{0, 9, 7}: combo p=.5 ≈ .83 @ .15, combo p=1 ≈ .86 @ .20 (both ≈ CU's .88 @ .30 at half the cost) — to be
replaced by the dose simulator once the residual L10 arms land.

**Arms emitted** (12, `r09_fable_combo/arms.json`, manifest = the same discovery300 pairs, prefits in `fits/`,
dependency plan passes locally: 96 files, 20.2 GiB logical, mostly already on h100): per cell in
{π0.5 Sp-50, π0.5 L10-50, GR00T L10-50}: `r9f_combo_<cell>_cache`, `_res` (same-fleet controls),
`_res_top_p05`, `_res_top_p1`. Expected wall time ≈ 12 × 4–5 min ≈ 1 h on 48 workers.

**Not launched**: timan107 is reserved for the sol coder's `r09_p1` arms. Command block (run only when the
coordinator frees timan107; `tools/combo_method.py` must first be present in the h100 tree at
`/data/oscl_h100/openpi/exp/offline_search/rounds/r09/explore_fable/tools/{__init__.py,combo_method.py}` together with
`explore_fable/__init__.py` — a new directory, so it cannot disturb running servers):

```bash
cd /home/weiland/projects/openpi
P=(taskset -c 22-25 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src .venv/bin/python)
C=/home/weiland/trace_runs/os_closed_loop/r09_fable_combo
ARMS=(r9f_combo_pi05_spatial_50_cache r9f_combo_pi05_spatial_50_res r9f_combo_pi05_spatial_50_res_top_p05 r9f_combo_pi05_spatial_50_res_top_p1 \
      r9f_combo_pi05_l10_50_cache r9f_combo_pi05_l10_50_res r9f_combo_pi05_l10_50_res_top_p05 r9f_combo_pi05_l10_50_res_top_p1 \
      r9f_combo_groot_l10_50_cache r9f_combo_groot_l10_50_res r9f_combo_groot_l10_50_res_top_p05 r9f_combo_groot_l10_50_res_top_p1)
# 1. deploy the one new module (new directory; no existing file touched) -- adapt to the coordinator's deployment helper
#    push: exp/offline_search/rounds/r09/explore_fable/__init__.py, tools/__init__.py, tools/combo_method.py  -> same relative paths under /data/oscl_h100/openpi
# 2. plan + concurrent sync (reserved port 23195)
WORKER_HOST=timan107 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control plan "$C" "${ARMS[@]}"
WORKER_HOST=timan107 SYNC_PORT=23195 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control sync --concurrent "$C" "${ARMS[@]}"
# 3. chain (full-model arms need 9 GB per port for pi0.5, 8 GB for GR00T; 4 ports)
tmux new -d -s r9f_combo "cd /home/weiland/projects/openpi && WORKER_HOST=timan107 PORTS=23240,23241,23242,23243 WPS=12 MAX_ATTEMPTS=2 POLL_SECONDS=30 \
  OSCL_MANIFEST=$C/manifests/discovery300.json ${P[*]} -m exp.offline_search.closed_loop.ops.h100.control chain $C ${ARMS[*]} > $C/chain_console.log 2>&1"
# 4. analysis (paired, per task): adapt tools/confirm_grown.py --run-root $C --ref r9f_combo_<cell>_res <arms of that cell>
```

Decision rule (preregistered here): adopt `res_top_p05` for a cell if it beats `_res` by ≥ 2 pp paired at IR ≤ .15
and is within 3 pp of P10; adopt `res_top_p1` only if it reaches P10 − 2 pp at IR ≤ .20. Otherwise the residual
alone is the cell's operating point.

## 3. New ideas suggested by the corrector result (ranked; prepared where cheap, not run)

1. **Per-task strength chosen label-free (prepare next).** The half strength is a global compromise: tasks where
   the cache is far from the policy and the policy is good (large shadow gap, large P10−cache) can take blend 1.0;
   tasks where the policy is no better than the demos should take 0–.25. Label-free rule: blend_t = clip(gap_t /
   median_gap − .5, 0, 1) with gap_t the per-task mean shadow gap of the calibration episodes. Evidence: offline the
   full residual halves the gap again (π0.5 L10-50 .292 → .222) and astra's Spatial result hurt no task, so the risk
   is confined to tasks where P10 < cache (π0.5 L10-50 tasks 2, 3). Prepare as `ResidualCache` with per-task blend
   (10-line subclass) after the L10 residual results show whether tasks 2/3 lose at blend .5.
2. **Residual on the gripper channel, gated.** The corrector leaves the largest remaining disagreement untouched
   (grip .10–.18 vs floor ≤ .06; my student reduced it 60–80% offline). Risk: gripper timing is the chattering
   channel (§1b). Safe variant: correct the gripper only with hysteresis (flip only if the predicted policy
   sign is held for ≥ 3 of 5 executed controls) and only on tasks where the cache's gripper disagreement is in the
   top half. Needs a new head (the ridge target must include channel 6); not prepared.
3. **Keep the corrected anchor for one extra blind block on dense libraries** — the residual already corrects the
   tail; R7/R8's gated extension was safe only on 500-demo libraries. Low priority (IR −.01 at best).
4. **Do not** combine the residual with the grown library or the student: the evidence says the gain comes from a
   smooth correction that preserves the cache's own trajectory; replacing retrieval loses it.

## 4. Status of my earlier proposals after astra's result

- P1 (per-task call budget) stands and is now the *second* layer on top of the residual (combo above); on
  Spatial-50 its marginal value shrinks (+2–4 pp) because the residual removes most of the task-6 deficit.
- P2 (per-task library choice with grown rows) is superseded by the residual head: same labels, smaller
  trajectory disruption, confirmed closed loop on one cell with holdout. Keep the grown-library tool as a diagnostic.
