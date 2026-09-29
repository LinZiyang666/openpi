# Configuration C controller handback

## Current state: R6-C-v2 with SELECTION §7b (2026-09-29, read this first)

This section supersedes the v1 text further down wherever they disagree. The v1 text covers the all-call cooldown, its
feasibility table, `dryrun_summary.json`, `replays/`, `test_results.json`, `packaging_tests.json`, `prefit_manifest.json` and
`delivery_manifest.json`, and all of it is stale. The pipeline, B-val audit, recording plan and coordinator sequence below
are still valid. All runs in this pass were CPU-only on 14-17,58-61 with at most 4 processes. There was no rollout,
server, git or remote access.

### What changed

- **v2 (SELECTION §7).** A one-free-anchor cooldown follows **stall-triggered calls only**. R and uniform lottery calls
  have no cooldown. `cooldown_scope='all'` is kept for tests and is refused for deployment. The coins are keyed
  `R6-C-v2/<cell>`.
- **§7b (this pass; version string unchanged).** A `slow_ambiguous` anchor draws the R or uniform lottery exactly as an
  `ok` anchor does: same p, same keyed coin. Its extra LOOK is scheduled only when that lottery does not call.
  `slow_confirmed` handling (p=1 unless cooled, then a cooldown) is unchanged. So are A identity at p≡0, the coin
  keys, the commit/tail and GR00T gripper handling, and the logged fields. No log key was added.
- **One rule for model and deployment.** The rule lives in `stall_bridge.py`: `call_probability`, `scheduled_look`,
  `starts_cooldown` and `AMBIGUOUS_RULE='lottery_then_look'`. `methods.py:CalibratedRescue.query` uses it.
  `CalibratedRescue._load` refuses any calibration that does not declare that rule, so the pre-§7b `dryrun_v2`
  calibrations are now refused.
- **Cost model (`budget.py`).** A call now changes the cadence (no LOOK) and therefore the stall tracker's later
  windows, so the cadence is no longer one fixed anchor list.
  - `replay_cadence` builds an exact DAG. Each node is (recorded row, the tracker's last W+1 observations, last extra
    LOOK still inside the W·L cap, cooled).
  - `expected_cost` propagates exact reach probabilities through it with the shared rule.
  - The solver is unchanged when the 65-point grid is monotone, which holds for all 64 dry-run grids. If a grid is
    not monotone, the fallback takes the grid-supremum ceiling and the smallest-parameter crossing; it never
    triggered.
  - `fit_calibration.py` records `ambiguous_rule` and `cadence_nodes`. It also stores the calibrated-ceiling
    pseudo-target per mode and placement (key = its 12-digit value) for §7b's `C.max`.
- **Test and driver scripts.** `dryrun_all.py --suffix`, `run_replays.sh` (TAG/DRYRUN, plus a new `stall30` case),
  `replay_test.py` (§7b checks and DAG-path agreement), `check_results.py --tag` and `check_packaging.py --tag` were
  updated. `old_rule_regression.py` is new.
- **Bug caught before any kept artifact.** History slicing was wrong for W=3 tasks. `old_rule_regression.py` caught it,
  and it was fixed before any artifact was kept.

### Feasibility, v2 + §7b dry run (`dryrun_summary_v2b.json`, `dryrun_v2b.log`)

This is a code test only: pilot A, block 0, init 0, 10 episodes per cell (`DRYRUN_TEST_INITS`), c1 = .152 (π0.5) and
.148 (GR00T). The command is `"${Q1_PY[@]}" -m "$Q1_MOD.dryrun_all" --suffix v2b`, and the output goes to
`/tmp/q1_method_c_fits/<cell>/dryrun_v2b/`. Floors and ceilings are identical for uniform and R. Feasibility at
ρ = .18 / .30 / .45 is written uniform | R.

| Cell | No stall: floor / ceiling | No stall: .18 .30 .45 | Stall: floor / ceiling | Stall: .18 .30 .45 | v2 stall ceiling |
|---|---|---|---|---|---|
| pi05_l10_50 | .0768 / .5053 | YYY \| YYY | .1241 / .4758 | YYY \| YYY | .3975 |
| pi05_l10_500 | .0767 / .5046 | YYY \| YYY | .0947 / .4899 | YYY \| YYY | .4823 |
| pi05_spatial_50 | .0780 / .5129 | YYY \| YYY | .0990 / .4974 | YYY \| YYY | .4513 |
| pi05_spatial_500 | .0788 / .5183 | YYY \| YYY | .0924 / .5106 | YYY \| YYY | .5090 |
| groot_l10_50 | .0745 / .5031 | YYY \| YYY | .1658 / .4258 | YY**n** \| YY**n** | .3924 |
| groot_l10_500 | .0748 / .5057 | YYY \| YYY | .1168 / .4797 | YYY \| YYY | .4440 |
| groot_spatial_50 | .0766 / .5177 | YYY \| YYY | .0986 / .5035 | YYY \| YYY | .4658 |
| groot_spatial_500 | .0759 / .5129 | YYY \| YYY | .0759 / .5129 | YYY \| YYY | .5000 |

- **Unchanged from v2.** The no-stall columns and every stall floor are unchanged from v2. §7b only affects ambiguous
  anchors with p>0.
- **The one infeasible target.** On these test inits only **GR00T L10-50 C.45 (stall)** is infeasible, with a ceiling
  of .426. π0.5 L10-50 C.45 became feasible (ceiling .476, previously .397).
- **Size of the model.** Modeled extra LOOKs per episode fall as ρ rises; for example, π0.5 L10-50 R goes 3.68 → 2.59
  → 0.69. DAG sizes are 111–761 nodes per cell, and each cell fits in ≤ 27 s.

### Test results (all PASS)

- **`old_rule_regression_v2b.log`.** With the pre-§7b rule patched in, the DAG reproduces `dryrun_summary_v2.json`
  exactly: 8 cells, 866 values, max |Δ| 1.4e-14. That covers floors, ceilings, parameters, predicted IR, calls,
  anchors and LOOKs.
- **Plugin replays: 16/16 PASS.** The cases are A, floor, uniform, R, R_reverse, R45, stall (.18) and stall30 (.30),
  for pi05_l10_50 and groot_l10_50. The logs are `replay_v2b_*.log` and the outputs are in
  `/tmp/q1_method_c_fits/replays_v2b/`.
  - Every fresh anchor's p, call, os_reason and extra LOOK is re-derived from its logged state and nominal p.
  - Every realized episode is one path of the budget DAG built from the same recorded keys, with the same stall
    states, cooldowns and LOOK eligibility. That holds for **2,738 anchors**.
  - The stall replays hit 63 ambiguous anchors, with **16 ambiguous lottery calls** and 35 extra LOOKs.
  - All 12 no-stall replays are **bit-identical to `replays_v2`**.
- **`check_results_v2b.log` → `test_results_v2b.json`.** A, floor and R_reverse identities hold. The results below
  also pass.
  - Exhaustive branch enumeration versus the DAG: windowed and full-history trackers, stall and all scopes, uniform
    and R, 192 values, max |Δ| 3e-14.
  - An always-ambiguous path never calls at p=0, respects the LOOK cap, and calls at every anchor with zero LOOKs at
    p=1.
  - 1,000 keyed seeds per cell:
    - no-stall rate: .18 within 1.2 SE;
    - uniform versus `RiskLottery` difference: < .01;
    - stall rate at .30 with real Q3 statuses: IR, calls and LOOKs match the DAG within 1.6 SE over 48 comparisons.
  - Loader gates: the pre-§7b calibration is rejected, GR00T L10-50 C.45 is refused as infeasible, and its `C.max`
    ceiling key loads with a saturated parameter.
  - The B-val exclusion masks and the deployment firewall hold.
- **`check_packaging_v2b.log` → `packaging_tests_v2b.json` and `prefit_manifest_v2b.json`.**
  - Retrieval identity holds.
  - The GR00T judge is the same function as before.
  - The reset and pool gates hold.
  - All 20 prefits reload with kwargs equal to the current specs.
  - `emit_specs` and `render_specs` (roots from `rendered_locations.json`) reproduce every production spec file
    exactly.
  - All 44 C rows carry the R6-C-v2 kwargs.
  - The dropped baseline log keys are the same set as v2.
- **Reverse control (`negative_control.py` → `negative_control_v2b.log`).** A controller patched back to the pre-§7b
  rule fails `replay_test` at its p re-derivation check.
- **Current file hashes.** They are in `delivery_manifest_v2b.json`. The pre-change sources of the edited files are
  kept in `/tmp/q1_method_c_fits/pre7b_sources/` for diffing.

**Superseded expectations (they encoded the old ambiguous=p0 rule).** The old `check_results` asserted that an
always-ambiguous path has zero calls at p=1. It also enumerated ambiguous anchors with p=0 on a fixed cadence. Both
were replaced by the §7b expectations above. No other test expectation changed.

### Which artifacts are current

- **Production specs.** `recording_prefit_specs.json`, `validation_prefit_specs.json`, `smoke_prefit_specs.json`,
  `emit_arms_c32.json` and `emit_arms_smoke2.json` already carry the v2 kwargs. §7b changes no kwarg, and the emitter
  and renderer reproduce them exactly, so **none were regenerated**. The `*_test_v2.json` files are scratch-root test
  renders.
- **Prefits.**
  - Current recording prefits: `/tmp/q1_method_c_fits/final_recording_prefits/`, 8 files with final CAL paths.
  - Current U and Bmech prefits: **regenerated** into `/tmp/q1_method_c_fits/prefits_v2b/` (`prefit_baselines_v2b.log`).
  - `/tmp/q1_method_c_fits/prefits/` is stale: its U pickles carry `R6-C-v1/<cell>` keys and its recording pickles
    carry scratch paths.
- **Calibrations.** The current code-test calibrations are `/tmp/q1_method_c_fits/<cell>/dryrun_v2b/`. `dryrun_v2` and
  earlier are pre-§7b and are refused by the controller.

### Coordinator: still to do before recording and prefit

1. **Recording.** The recordings (A-cohort `Profile`) are unaffected by §7b. Stage the B pools and record as described
   below.
2. **Production calibration.** Run `fit_calibration` for production **with this code**. A calibration without
   `ambiguous_rule` cannot be prefitted.
3. **Infeasible targets.** If a production target is still infeasible (the dry-run candidate is GR00T L10-50 C.45),
   report it as infeasible. Add an explicit `C.max` spec row with `rho = float(<ceiling key>)` from that calibration.
   `emit_specs` does not generate that row.
4. **U prefits.** Copy U prefits from `prefits_v2b` (they have no placeholders), not from `prefits`.
5. **Smoke acceptance.** Update it to the new rules:
   - the cooldown applies only after stall calls;
   - ambiguous anchors may make lottery calls (os_reason 62);
   - an extra LOOK happens only on an ambiguous anchor that did not call.

---

# v1 handback (partly superseded; see the section above)

Built on 2026-09-29 under `ideation_Q1/method_c/`; scratch/artifacts are exclusively `/tmp/q1_method_c_fits/`. No rollout, model inference, server, worker, chain, tmux operation, git command, or remote write was run. Two remote inspections used read-only `tether exec`. All Python execution used repository `.venv/bin/python`, CPUs `14-17,58-61`, one numerical-library thread and hidden CUDA. At most two Python processes overlapped.

## Read this before scheduling

**The requested one-free-anchor cooldown after every call conflicts with the selected high budgets.** The controller implements that literal requirement. Existing `CacheDose`/`RiskLottery` and C10's lifecycle tail do **not** impose this additional free-anchor cooldown. For long recordings without extra LOOKs, C's maximum owner IR is `c1/2 + (1-c1)/4`, or .288 for π0.5 and .287 for GR00T. Finite episode endpoints and stall LOOKs are modeled explicitly, rather than replaced by that approximation.

In the **pilot-only code dry run**, all eight .18 targets are feasible; all eight .45 targets are infeasible. Among the requested sparse arms, three R.30, all four C.30, and all four C.45 are infeasible on these recordings: **11 of the 20 C-controller configurations**. R.30 on GR00T Spatial-50 is feasible because of finite-episode endpoints. These are test-code results, not production calibration decisions. The new B-val calibration must report its own bounds. A requested target outside its feasible interval raises an error at prefit; it is never silently clamped or replaced.

The supplied 32 specs preserve SELECTION's requested identities. **Do not launch an infeasible arm or label a lower-cost arm C.45.** A revised interpretation of cooldown or revised target values is necessary before the selected high-budget campaign can generally run. Removing the cooldown would change this controller's declared semantics and requires a new calibrated version. No such amendment was made here.

**B pools are absent on the remote client.** Read-only inspection found neither `/scratch/zixuans8/openpi_trace/exp/common/data/db_init/libero/libero_10` nor `.../libero_spatial`; see `remote_pool_audit.txt`. The coordinator must copy the audited pools before recording. The driver already accepts them; no worker or driver patch is needed.

## Delivered files and artifacts

| Item | Location / status |
|---|---|
| Deployment controller | `methods.py:CalibratedRescue` |
| Exact row LOEO reconstruction | `precompute_r.py`, `common.py`; all eight banks built |
| Shadow calibration and cost solver | `fit_calibration.py`, `budget.py` |
| Q3 bridge / inactive stub | `stall_bridge.py`; real Q3 module is now present and was exercised |
| B-val membership audit and specs | `prepare_recordings.py`, `bval_membership_audit.json`, `calibration_manifest_*.json`, `recording_exclusions.json` under each scratch cell |
| Recording-only exclusion adapter | `recording.py:ExcludedRecordingA` |
| 80-episode recording plan | `emit_calibration_recordings.json`, `calibration_schedule.json`, two `bval_pool_*.yaml` |
| Remote launcher adapter | `run_arm_bval.sh`; optional `chain_calibration.patch` changes only a run-specific copy's launcher pathname |
| 32 validation / 2 smoke specs | `emit_arms_c32.json`, `emit_arms_smoke2.json`, `eval500_manifest.json`, `smoke4_manifest.json` |
| Fit preparation | `prefit.py`; 20 checked prefits in `/tmp/q1_method_c_fits/prefits/`; paths/hashes in `prefit_manifest.json` |
| Replays and tests | `replay_test.py`, `run_replays.sh`, `check_results.py`, `check_packaging.py`; `test_results.json`, `packaging_tests.json` both PASS |
| Code-test calibration | `/tmp/q1_method_c_fits/<cell>/dryrun_stall/`, all **DRYRUN_TEST_INITS**; `dryrun_summary.json` |
| B pool archive | `/tmp/q1_method_c_fits/bval_pools.tar`, 20 original `.init` files |

Each `/tmp/q1_method_c_fits/<cell>/` contains `r_bank.json` and `r_bank.npz`. Copy those files together. `r_bank_manifest.json` gives hashes and an exact per-cell command. Deployed A source artifacts and library directories remain absolute, read-only paths on this server. The fitted projections, metrics, actions, row identities and kernel settings are content-fingerprinted; row arrays and library action/identity/manifest files are SHA256-bound. This is a fingerprint of the effective serving bank/fit, not a hash of every historical image/token file.

| Cell | Bank rows | Distinct source episodes |
|---|---:|---:|
| pi05_l10_50 | 2,640 | 50 |
| pi05_l10_500 | 29,472 | 500 |
| pi05_spatial_50 | 1,018 | 49 |
| pi05_spatial_500 | 10,909 | 500 |
| groot_l10_50 | 2,645 | 50 |
| groot_l10_500 | 29,631 | 500 |
| groot_spatial_50 | 1,063 | 50 |
| groot_spatial_500 | 11,751 | 500 |

All **89,129** row residuals were computed in **126.98 s** summed per-cell wall time (`precompute.log`, `r_bank_manifest.json`). The nominal π0.5 Spatial-50 bank contains 49 distinct source-file identities; all aliases of a source are excluded together.

## Exact algorithm and interface

`CalibratedRescue(rho, placement='R', stall_model_path=None, calibration_path=..., random_seed=0, randomization_key='R6-C-v1')` wraps the existing, frozen A fit. No deployment policy shadow, resampling, profiler or simulator-state access is introduced.

1. Let `b=manifest.exec_steps`, `L=min(manifest.H,2*b)` and use the manifest's valid action coordinates. For each coordinate, take population SD over every library row's first `b` action steps; a SD at most `1e-12` is replaced by 1. For every row j, run A's scalar projection/distance, its early regime when the stored step is zero, exact top-k tie rule, kernel and float32 action mixing. Exclude every candidate belonging to the same acquisition source episode, including aliases. `r_j` is RMS of the `L`-step reconstruction error divided by those SDs. Metric/projection fits stay frozen: this is conditional LOEO, not refitted LOEO.
2. At a free vision anchor, first feed the raw observation to `StallTracker.observe`, then execute A's unchanged retrieval. `R(s)=sum_i w_i*r_i` using its actual rows/weights. Set `Ehat=a+b_R*R`. No action or gripper coordinate is rewritten.
3. Calibration label E is the same-observation primary shadow/cache RMS on the same `L` steps and scales. Fit nonnegative intercept and slope by NNLS on weighted design/labels. Each recorded task/episode has equal mass; its anchors share that mass equally. At most ten recordings, one per recorded task; this LIBERO plan uses ten. No success label enters the fit or selected init list.
4. Let t index free anchors, including additional LOOKs. A previous-anchor call forces `p_t=0`. Outside that cooldown: `slow_confirmed` sets `p_t=1`; `slow_ambiguous` sets `p_t=0`; otherwise p is d (uniform) or `min(1,lambda*Ehat)` (R). The SHA256 keyed coin is exactly Q2's `uniform(key,seed,task,original_init,decision_step,'dose-anchor')`; it is independent of UID and arrival order. A slow-ambiguous anchor schedules a LOOK on the next request after `min(b,L)` controls, limited to once per `W*L` controls. It uses `LookReason(8,...)`, the existing B blind-LOOK veto mechanism.
5. A call sets only the force-MISS verdict and records policy-tail provenance. The plugin runs the policy, serves the first five controls, then exactly one five-control tail through unchanged `CommitJudge.policy_tail_step` lifecycle validation. GR00T inherits this very same function through `GrootCommitJudge`; negative normalized gripper is still closed. Hold is 1. The next anchor retrieves using actual executed policy-tail history. C then applies its additional one-free-anchor cooldown.
6. Replay Q3 over the recorded raw-key metric codes, including diagnostic codes at blind decisions. This determines extra LOOK cadence and mandatory stall statuses. For each cadence, integrate cooldown exactly: unconditional call probability `u_t=(1-u_(t-1))*q_t`, with `u_-1=0` and q equal to the nominal/stall probability above. For episode e, modeled owner IR is `[c1*A_e+(1-c1)*sum_t u_t]/N_e`, where N is the number of nominal five-control requests. Average these episode ratios with equal task/episode weights. Solve d or lambda by 80 bisections; endpoints and 65 monotonicity-check points establish the feasible interval. Targets outside it are explicit infeasible results. Library-length Q2 uniform budgets are also saved as diagnostics. The additional cooldown is why d is generally larger than Q2's uncooldowned anchor dose.

The solver includes mandatory stall calls, their cooldown, and extra LOOK vision cost. It holds the observations and episode lengths fixed: calls may change both in a rollout, so actual IR must still pass SELECTION's ±.02 check. This is not an SR/OPE estimator. Owner shares are supplied to the fitter (`.152` / `.148` here), rather than encoded in the controller.

The current plugin transport executes five controls/request and supports one tail. The wrapper explicitly refuses a bank interface other than `b=5,L=10`; for another robot, port the commit transport and executed-control timestamps, then use its manifest action mask/horizon and measured c1. The statistical R/stall calibration recipe has no LIBERO phase, distance, gripper, task-length or action-error threshold. Online control time here is `decision_step*5`, valid for the verified nonterminal request contract; variable-duration control needs an actual timestamp adapter.

Decision logs include `os_c_R`, `os_c_Ehat`, actual `os_c_p`, nominal p, coin, call, cooldown, extra-LOOK, control index, anchor ordinal and stall diagnostics. Stall codes: inactive=0, ok=1, slow_confirmed=2, slow_ambiguous=3. `os_c_fresh=0` and `os_c_carried_anchor_score=1` identify carried scores on cache/policy tails; their p/call are zero. Force-MISS reasons are 62 (random) and 63 (stall). The plugin's existing 40-scalar log limit can displace baseline blind diagnostics; retrieval/actions are unchanged (`packaging_tests.json` lists displaced keys).

## Non-test selection and bank overlap

Lowest B-val indices by task 0..9, chosen from the frozen split without consulting outcomes:

| Suite | Original B indices |
|---|---|
| LIBERO-10 | 8, 8, 4, 2, 9, 9, 11, 0, 9, 2 |
| Spatial | 9, 6, 5, 0, 11, 9, 9, 0, 9, 2 |

All 20 selected reset-state byte hashes are saved. The B and official test pools were compared by actual state contents and have zero identical states; indices alone do not identify a pool. The existing driver's `load_apool_digest` successfully rehashed and accepted both local 500-state B records in the packaging test. Remote inspection found the B directories missing, not a digest match.

Every 500-episode bank contains exactly one source episode for each selected B-val task/init: **exclude that entire source during recording**. GR00T L10-50 also contains selected task 1 / B init 8, which is excluded. The other three 50-episode banks have no selected-state overlap. The frozen deployed metric and full deployment bank are retained; only recording-time retrieval candidates are excluded. Thus recording calibration remains conditional on the original representation, and all 500-bank B states were acquisition states. It is not an independently acquired bank test.

π0.5 500 membership is explicit in `episodes.json.orig_init_state_idx`; π0.5 current comes from the historical cache pools, whose states were matched to `protected_in_train`. GR00T collection uses B states with `episode_id=task_id*50+episode_idx`, no init remapping (`launch_collection.sh`, `collect_util.py`, `report_collection.py`). The Spatial store has null original-init IDs, and its surviving lane result JSONs contain only the last 36 explicit rows after resharding. The audit recovers IDs using that documented collector contract, checks every bank task/episode identity, and checks every surviving explicit lane result; it does **not** assume that a null means non-overlap. Evidence paths/hashes and recovered IDs are in each exclusion artifact.

The production fitter additionally requires accepted client reset telemetry and rejects reset hashes that differ from the frozen B-val selection. It verifies the catalog's pool, manifest, exclusions, R-bank and fitted-metric binding. `DRYRUN_TEST_INITS` calibrations are rejected by the deployment loader. CPU tests inject dry-run objects only in memory and write no deployable test-calibration pickle.

## Commands actually run and tests

All commands below are from `/home/weiland/projects/openpi`. This shell array is shorthand for the prefix used on every Python command:

```bash
Q1_PY=(taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
Q1_MOD=exp.offline_search.rounds.r06.ideation_Q1.method_c
Q1_DIR=exp/offline_search/rounds/r06/ideation_Q1/method_c
"${Q1_PY[@]}" -m "$Q1_MOD.precompute_r" --cell all
"${Q1_PY[@]}" -m "$Q1_MOD.prepare_recordings"
"${Q1_PY[@]}" -m "$Q1_MOD.emit_specs"
"${Q1_PY[@]}" -m "$Q1_MOD.dryrun_all"
bash "$Q1_DIR/run_replays.sh"
"${Q1_PY[@]}" -m "$Q1_MOD.check_results"
bash "$Q1_DIR/prefit_recordings.sh"
"${Q1_PY[@]}" -m "$Q1_MOD.check_packaging"
```

Existing calibrations/replay output directories are intentionally not overwritten. For a fresh dry-run reproduction, change the dry-run output suffix or use a new scratch destination; never relabel it NONTEST_BVAL. `recording_prefit_specs.json` and `validation_prefit_specs.json` are filled **scratch-path** versions used to test prefitting, not production run roots. U/Bmech prefits were generated sequentially with `prefit --spec .../validation_prefit_specs.json --name <arm>` for the 12 names in `baseline_prefit_names.txt`.

Measured checks:

- 12 real-plugin/fake-policy replay cases, four recorded episodes each: **4,656 decisions, 417 verified policy tails, 19 scheduled extra LOOK flags**. No simulator or policy model ran. A versus zero-call C matches all action bytes, retrieval top-k/scores/confidence/weights, HIT/source and vision schedules, on both models. New diagnostic fields are additive, so whole log files are not byte-identical.
- Reversing episode arrival and renaming UIDs preserves the complete keyed R action/source sequence. Every non-final call's next action equals the original policy chunk shifted by five on **all 32 coordinates**, followed by vision. GR00T's inherited judge-tail function and gripper predicate were checked directly.
- Missing-module stub and unconfigured-stall paths pass. Exhaustive short-path enumeration matches the cooldown expectation with confirmed and ambiguous states. Synthetic repeated ambiguity respects the W·L LOOK cap; the real Q3 tracker generated extra LOOKs in both model replays.
- 1,000 keyed seeds per cell on all eight calibration cadences: R realized IR **.179774–.180181** for target .18; uniform **.179868–.180232**. Every value is within five Monte Carlo SEs of .18. C-uniform versus the actual `RiskLottery` episode-assignment code differs by at most **.001957 IR**. Library-length versus recording-length endpoint weighting explains a small deterministic difference. **Rates agree at a feasible budget; schedules are not identical**, since RiskLottery has episode mixtures and no C cooldown.
- All 80 source-exclusion candidate masks checked against real library queries. All eight test-init calibrations are rejected for deployment. Matching reset attestation accepted; wrong-pool reset rejected. Twenty prefits reload with exact method/kwargs identities; all 32 validation constructors/commit flags checked.

The dry-run calibration uses **only A, block 0, init 0**, ten episodes per cell. It tests code, not transfer or final calibration. Bounds below are from `dryrun_summary.json`:

| Cell | No-stall floor / ceiling | With-stall floor / ceiling |
|---|---|---|
| pi05_l10_50 | .076801 / .292181 | .124061 / .269626 |
| pi05_l10_500 | .076701 / .296854 | .094671 / .293240 |
| pi05_spatial_50 | .077966 / .299574 | .099022 / .288337 |
| pi05_spatial_500 | .078776 / .305390 | .092445 / .305736 |
| groot_l10_50 | .074454 / .291068 | .165805 / .287740 |
| groot_l10_500 | .074848 / .293631 | .116788 / .289190 |
| groot_spatial_50 | .076615 / .303791 | .098636 / .298580 |
| groot_spatial_500 | .075908 / .308359 | .075908 / .299940 |

## Coordinator sequence: recording, fit, smoke, validation

**These deployment/rollout commands have not been executed.** Keep the coordinator's existing P3 streaming/fence deployment; Q1 does not replace any profiler implementation. First copy the audited B pools and two records, plus the new wrapper. Tether staging is under `/tmp` because direct pushes to `/scratch` are unsupported:

```bash
tether push --force /tmp/q1_method_c_fits/bval_pools.tar timan107:/tmp/q1_c_bval_pools.tar
tether push --force "$Q1_DIR/bval_pool_l10.yaml" timan107:/tmp/q1_c_bval_l10.yaml
tether push --force "$Q1_DIR/bval_pool_spatial.yaml" timan107:/tmp/q1_c_bval_spatial.yaml
tether push --force "$Q1_DIR/run_arm_bval.sh" timan107:/tmp/q1_c_run_arm_bval.sh
tether exec timan107 -- bash -lc 'set -e; D=/scratch/zixuans8/openpi_trace/exp/common/data/db_init/libero; I=/scratch/zixuans8/openpi_trace/os_cl; mkdir -p "$D" "$I/q1_c_bval"; tar -xf /tmp/q1_c_bval_pools.tar -C "$D"; cp /tmp/q1_c_bval_l10.yaml "$I/q1_c_bval/bval_pool_l10.yaml"; cp /tmp/q1_c_bval_spatial.yaml "$I/q1_c_bval/bval_pool_spatial.yaml"; cp /tmp/q1_c_run_arm_bval.sh "$I/run_arm_c_bval.sh"; sha256sum "$D"/libero_10/*.init "$D"/libero_spatial/*.init'
```

Verify these 20 digests against `bval_pool_*.yaml`. The driver rehashes at launch as well. No `.pruned_init` may appear in those directories. The wrapper accepts only `r6c_cal_*` arms with an exact manifest and appends the B pool arguments **after** the stock launcher's default A-pool flags. Python argparse's last value wins. `run_gtp` passes that explicitly bound directory to the workers. Local code paths are `closed_loop/ops/remote/run_arm.sh`, `run_gtp_subset.py`, `exp/gate_threshold_pareto/run_gtp.py`, `examples/libero/worker_entry.py` and `p3_profiling/run_gtp_v2.py`. Remote stock-launcher/subset SHA256s were inspected and are in `remote_pool_audit.txt`.

Choose a fresh recording `<RUN>` and durable artifact `<CAL>`. Copy each cell's `r_bank.json`, `r_bank.npz`, and `recording_exclusions.json` into `<CAL>/<cell>/`; copy the ten-pair manifests unchanged into `<RUN>/manifests/`. Fill `<RUN>`/`<CAL>` in `emit_calibration_recordings.json` and save `<RUN>/arms_in.json`. Preserve `calibration_schedule.json` as `<RUN>/schedule_v2.json`; its `client_env` supplies environment seed 1603. Refit against these **final paths** (pickle kwargs must match exactly), then copy the prefits to `<RUN>/fits/`:

```bash
# Repeat for the eight r6c_cal_<cell> names, sequentially.
"${Q1_PY[@]}" -m "$Q1_MOD.prefit" --spec '<RUN>/arms_in.json' --name r6c_cal_pi05_l10_50 --out /tmp/q1_method_c_fits/final_recording_prefits
# Coordinator copies generated .pkl/.json pairs to <RUN>/fits/.
"${Q1_PY[@]}" -m exp.offline_search.closed_loop.ops.emit_arms --run-root '<RUN>' --spec '<RUN>/arms_in.json'
```

If `<CAL>` remains `/tmp/q1_method_c_fits`, the eight existing recording prefits can be copied directly: changing the spec's run-root/manifest path does not alter kwargs, while changing `<CAL>` does. Use durable storage before a long campaign.

For P3 chain integration, copy `p3_profiling/chain_p3.sh` into `<RUN>/chain_calibration.sh` and apply `chain_calibration.patch` in that run directory. It changes only `run_arm_v2.sh` to `run_arm_c_bval.sh`; shared scripts remain untouched. The coordinator then uses its normal sync/server allocation/chain procedure for the eight recording arms. Without that optional patch, the exact per-arm remote invocation is:

```bash
cp exp/offline_search/rounds/r06/p3_profiling/chain_p3.sh '<RUN>/chain_calibration.sh'
patch -d '<RUN>' -p0 < "$Q1_DIR/chain_calibration.patch"
# Run by the coordinator after its usual server setup; no unspecified defaults.
# P3_ENV_SEED=1603, P3_SNAPSHOT_DIR and OSCL_MANIFEST come from client_plan.
bash /scratch/zixuans8/openpi_trace/os_cl/run_arm_c_bval.sh '<full-suite>' '<r6c_cal_cell>' '<servers>' '<workers-per-server>' '<remote-arm-output>'
```

Once a recording arm finishes, produce strict tables, then fit both no-stall and stall budget maps. `<CELL>` uses `spatial`; `<P3_CELL>`/Q3 artifact names use `sp`:

```bash
"${Q1_PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.read_v2 --run-root '<RUN>' --arms 'r6c_cal_<CELL>' --client-root '<RUN>/runs' --require-stage-counts --require-snapshots --out '<RUN>/tables/<CELL>'
"${Q1_PY[@]}" -m "$Q1_MOD.fit_calibration" --tables '<RUN>/tables/<CELL>' --bank '<CAL>/<CELL>/r_bank.json' --out '/tmp/q1_method_c_fits/<CELL>/calibrated' --bval-manifest "$Q1_DIR/calibration_manifest_<CELL>.json" --client-root '<RUN>/runs' --stall-model-path '<STALL>/<P3_CELL>' --c1 '<.152-or-.148>' --rho .18 .30 .45
```

Copy the resulting `calibrated/` directory as a unit into `<CAL>/<CELL>/calibrated/`. No `--dry-run` is permitted here. Inspect status, all feasible intervals, stall/LOOK counts and coefficients; an infeasible target stops its prefit. The current fitter restricts its writes to the Q1 scratch/owned directory, so its `--out` above intentionally uses scratch even when final `<CAL>` is elsewhere.

Fill `<RUN>`, `<CAL>`, `<STALL>` in the validation specs. Keep Q3's four Bmech classes/kwargs and Q2's eight original `RiskLottery(allocation='uniform')` controls. U deliberately retains Q2's original uniform episode mixture, as SELECTION specifies; the C class also supports direct uniform placement for calibration/replay. U's calibration is the existing hash-bound library-only Q2 artifact. Prefit each feasible configuration with `prefit.py`, then stage matching `.pkl/.json` files and manifests and emit through the ordinary ops emitter. The existing 12 U/Bmech prefits are reusable, since their kwargs have no `<CAL>`/`<STALL>` substitutions. C prefits do not yet exist.

The two-arm smoke uses **π0.5 L10-500 C.18 and GR00T L10-500 C.18**, tasks 0,1 × official measurement inits 0,1 = **four episodes each**. Use `emit_arms_smoke2.json`, final non-test calibration, real Q3 models, and `smoke4_manifest.json`. No P3/shadow at deployment. If a production C.18 target is infeasible, do not launch even the smoke under that label. Acceptance: four unique accepted pairs, correct artifact hashes/kwargs, every call followed by one exact policy tail and fresh vision, one free-anchor cooldown, correct stall/extra-LOOK logs, no shadow forwards, and reconciled `(N,V,M)` owner cost. Four episodes are not an SR or budget-precision test.

After that plumbing check and resolution of any infeasible targets, the full specs define 32 × 500 measurement episodes, paired with the existing A/B replicates and pure L10/L5. Preserve SELECTION's SR/IR acceptance rules; this handback changes no scientific outcome rule. In particular, disagreement prediction is not a causal claim that R placement improves SR.

Remaining work belongs to the coordinator: stage pools/artifacts, collect the 80 non-test recordings, run production calibration, resolve the cooldown/high-budget incompatibility, and run smoke/validation. No closed-loop success, production cost calibration, remote deployment, or cross-robot performance is claimed by these CPU tests.
