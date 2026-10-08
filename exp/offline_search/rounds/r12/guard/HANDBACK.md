# R12 sol: H2 guard and M2 look-half

H2/M2 are implemented with five fresh B-dev roots (15 arms) under `pilots/r12_guard_fix1_*`. The prior guard roots remain untouched. No chains, actual fleet syncs or test-A work were launched; only the bounded local engagement smoke below is authorized.

## Engagement fix (sol_fix.md)

The retained evidence does **not** reproduce the reported argv rewrite: all **32 startup records** from the eight completed arms have `--os-lookhalf` equal to the constructor and startup method. All **34,359 decisions** have `execution_audit=false`; none encodes wrist first. `../lookhalf/FIX_FINDINGS.json` lists the exact source logs and per-arm counts. Camera planning correctly invalidated memory without a client acknowledgment, but the old pilot used the unchanged LIBERO worker, which never supplied those acknowledgments. Zero one-camera looks therefore did not test the intended acquisition methods.

`closed_loop/lookhalf_config.py` now validates exact flag values at emitted-arm planning, after relocation, in final server command construction, in startup parsing and against the loaded prefit. Missing/empty/unknown/duplicate values, missing constructor selection, mismatches, and misconfigured aliased R12 classes fail loudly before model serving. The planner also asserts method preservation across relocation; local fit relocation replaces only the exact artifact token, never a substring of an unrelated value. No method-token rewrite was found or claimed repaired.

For active methods, `control.chain` sets `OSCL_EXECUTION_AUDIT=1` and the B-attested driver selects `closed_loop.execution_worker`. This opt-in worker runs the unchanged LIBERO rollout through proxies that count **actual successful environment steps**, exclude initial settling, and send `executed_steps`, `exec_ok`, and sequential `decision_id` on each infer request. Partial/failed execution is reported as such and conservatively completes acquisition. An active R12 server refuses missing or malformed receipts **before any encoding**. Explicit off and non-R12 workers retain their existing path.

`closed_loop/test_r12_pipeline.py` reads actual freshly emitted arms, runs plan and sync with only external transport mocked, then checks the **entire final server argv**, not a pre-plan approximation. It covers every local arm (17 look-half + 15 guard), all seven method variants through remote dry sync, recording contracts/SHA values, real startup parsing, bad configuration, actual/partial execution counts, worker entry selection, and prefit mismatch. Content hashes are calculated from real assets and cached only while inode/size/mtime/ctime stay unchanged. Evidence: `../lookhalf/FIX_VERIFICATION.json` and its test logs: **127 CPU checks passed** (98 pipeline/regression checks, 2 coordinator environment checks, 14 look-half and 13 guard selftests).

Bounded live evidence is being finalized in `../lookhalf/smokes/fix4/ENGAGEMENT.json`. `live_smoke.py` uses real pi05 weights and real LIBERO B-dev simulator prefixes, one server and one worker at a time, the same two held-out B inits for each method, at most 150 controls per episode and 25 minutes per method. Unchanged bf16 GPU vision / float32 CPU policy placement keeps physical GPU use below 4096 MiB; a 3100 MiB torch allocator cap leaves room for CUDA/EGL contexts. The parent checks `nvidia-smi`, monitors total physical memory and stops only its owned PIDs. These truncated engagement checks do not estimate SR or all-GPU numerical equivalence. The first attempt (`smokes/fix1`) stopped before inference because the smoke client had not loaded its bundle. The second (`smokes/fix2`) confirmed audited wrist-first requests, then was stopped by owned PID because this AVX2 CPU emulates bf16 policy GEMMs too slowly. The third attempt (`smokes/fix3`) exposed an attention-mask dtype mismatch in the smoke-only float32 adapter; no pilot source or fit changed. All attempts remain preserved; the final smoke uses the trained policy weights in float32 on CPU, with the genuine GPU encoder unchanged.


## Entry point and behavior

`exp.offline_search.closed_loop.guard:R12Guard` accepts `method=off|h2|m2`, `knob_method=off|periodic`, the existing whole-episode library selection, and `target=.32` for the periodic cell. Use the matching `--os-lookhalf METHOD`, `--os-blind --os-judge guard_only`, and the emitted `--os-fit-artifact`. Active guard fitting at serving startup is refused: use the explicit CPU builder. Off fitting retains the ordinary R10/R11 builder.

Absent opt-in flags retain the historical action and decision-log bytes. Explicit off delegates directly to R10/R11 without extra guard diagnostics or state updates. H2 changes only the layer-two verdict and removes the old no-progress blind veto. Its decision reaches R11 before the knob evaluates it, so guard calls retain priority and reset the periodic call gap through the ordinary accepted history. Full cache retrieval, synthesis, distance correction, and lifecycle policy-tail behavior remain unchanged.

At every active look, W retrieval supplies seven agentview-free risk features. These are wrist nearest distance divided by a frozen library replay median, decision step/100, lag relative to W's donor episode length, mean XYZ speed over the last two look intervals, direction reversals over three intervals, log(1 + commanded-vs-realized displacement residual divided by its B-library median), and progress advance clipped to [-5, 8]. As in Opus's actual procedure, advance compares the current W donor with the **previous served donor's progress**: F for full looks and WL for accepted half looks. The displacement feature uses both actual executed five-control heads from accepted history, including policy calls and tails. Missing startup history uses frozen library feature means, with the original special handling for mismatch and advance; H2 cannot call before decision four.

H2 calls at step >= 4 iff `(advance <= .5 and risk >= theta_a) or risk >= theta_b`. Coefficients are fitted from B-library data; theta values are frozen per-cell B-dev calibration from `opus/out/guard_m2.json`, explicitly labelled in every artifact. No pilot result can refit them. H2 looks always encode both cameras.

M2 plans wrist first whenever valid remembered agentview exists, except at step zero or a planned periodic call. A previous call does not force the following look full. H2 is evaluated before a wrist-first request commits anything; an H2 call requests agentview completion before ordinary full retrieval. Otherwise, lag above eight, unsupported lag cadence, or consistency above .10 completes the agentview. There are no M1 suspicion completions. Consistency uses uncorrected W and WL chunks, the deployed cache's sigma, and the first five gripper signs. An accepted half look serves WL with the wrist-only corrector head and W distance attenuation. It retains genuine agentview memory and installs the corrected WL action as the authoritative blind tail. Every policy call has both cameras.

`NeedCamera` commits neither guard, knob nor cache anchor. Completion refreshes the request-owned stage-one prefix and CP1 keys exactly once. The second query uses the already prepared H2 decision, then commits the ordinary cache/guard/knob transaction once. Duplicate completed queries are idempotent; fitted arrays are read-only across connection clones. H2 features, risk, branch flags, thresholds and decision are recorded at every look in `lookhalf.h2` and scalar `os_h2_*` extras. The recorder traverses the guard adapter to retain actual donor rows, weights and deployed guard memos.

Implementation changes are confined to `closed_loop/guard.py`, narrow opt-in CLI dispatch in `closed_loop/plugin.py`, the recorder's wrapper traversal in `closed_loop/look_recording.py`, and this new directory. Existing `lookhalf.py`, M1 fits, pilot roots, and launch scripts were not edited.

## Library build and verification

`build.py` fits the logistic risk model afresh with Opus's `lib_table`/`fit_logit` procedure: size-50 B-library LOEO rows from L10 and Spatial, pooled per model, target divergence above the pooled 75th percentile, lambda .01 and forty Newton iterations. The action-to-state ridge map is rebuilt from the B parent library, as in Opus's code. Audited library reads and SHA-bound library replay sources are recorded in `risk_model.json`; coefficient references are assertion inputs only. No dev recording or outcome is read by the risk builder. Threshold constants are the separate, explicitly labelled dev calibration input.

The fitting rows are **1,866 pi05 / 1,899 GR00T**. Maximum coefficient/mean/std differences against `opus/out/guard_lib.json` are **4.44e-16 / 5.55e-17**; the builder asserts equality within 1e-10. It preserves the attested R10 full fit, the selected-library Opus W/WL metrics, wrist head and distance scale from the completed look-half build, and the existing library-calibrated R11 periodic fit. It copies none of these artifacts back into the running look-half tree. Off/H2/M2 artifacts are emitted for all four size-50 model/suite cells, pi05 Spatial-200, and the pi05 L10-50 periodic cell. Spatial-200 is an additional prepared fit, not one of this pilot's cells.

Evidence:

- `SELFTEST.json` / `selftest.log`: **13 tests**. **720 exact off decisions** across five physical cells plus the periodic cell compare rows, scores, action bytes, confidence, extras and guard progress against R10/R11. **120 active full-look comparisons** retain the full cache/corrector result exactly. Tests cover H2 priority over the knob, removal of the old blind veto, retries, completion without mutation, frozen decisions across completion, hidden-camera poisoning, WL/wrist-head serving, authoritative tails, unchanged gripper/padding, lag/consistency/knob completion, call memory, connection isolation, all fifteen B contracts, recording, ledger identities, and coordinator environment/port validation.
- `regression.log`: **33 tests** passed, including the historical default action/decision-log byte proof and existing CPU recording and camera-dispatch tests.
- `REPLAY.json` / `replay.log`: **21,229 recorded-dev looks over ten arms** reproduce Opus's H2/M2 aggregate metrics in `guard_m2.json` and the earlier risk-only guard/schedule in `guard_final.json`, with zero aggregate difference at 1e-12 tolerance. Rebuilt risks differ by at most 5.55e-16. One decision is at an exact calibrated threshold and differs below that tolerance; aggregate reproduction uses the reference's inclusive tie and records the disagreement. All feature transforms match the published table exactly. Across **512 SHA-verified raw B looks**, **381 eligible history checks** have feature error at most 7.33e-15. Real serving regime-two W retrieval has **100% top-1 agreement**, distance error at most 6.39e-6 and risk error at most **2.69e-7**, passing the 1e-6 numerical risk tolerance.
- `BUILD.json`, `risk_model.json`, `artifacts/*/build.json`, `pilot_roots.json`, and each `ROOT.json` bind fit and selection provenance. The shared bounded live engagement smoke is documented above; the library/replay evidence remains unchanged.

## Prose/replay normalization discrepancy

**Serving follows the requested SPEC normalization, so its decisions cannot be claimed identical to Opus's published replay at numerical tolerance.** Opus's prose says library median, but `guard_table.py` divides W distance by the **dev arm's median**. The frozen risk coefficients and thresholds are otherwise identical. The replay check first reproduces that published normalization, then separately quantifies the required library-normalized result; it does not silently refit or substitute the dev median in serving.

| Cell | Published H2 calls/look | Library-normalized calls/look | Different decisions | M2 share with library median/cache sigma |
| --- | ---: | ---: | ---: | ---: |
| pi05 L10-50 | .23577 | .21388 | 70 / 3198 | .35053 |
| pi05 Spatial-50 | .17042 | .14148 | 36 / 1244 | .35450 |
| GR00T L10-50 | .32063 | .32092 | 1 / 3384 | .22340 |
| GR00T Spatial-50 | .12951 | .11967 | 12 / 1220 | .37377 |
| pi05 L10-50 periodic .32 | .15048 | .12454 | 35 / 1349 | .22535 |

The last column is still replay on deployed trajectories, not a closed-loop pilot result. Opus's published consistency score also uses the whole parent library's `sig_r`; serving uses the selected deployed cache's sigma as SPEC requests. `REPLAY.json` labels both choices and gives the additional Spatial-200 / GR00T knob checks. The real-wrist numerical check uses regime two, as Opus's dev feature table does; serving retains the specified post-MISS continuity regime when applicable. New M2 trajectories also use their actually served previous donor for progress. These are practical limits of the open-loop reproduction claim.

## Costs and pilot roots

The existing compute-share ledger is reused: pi05 .076 per encoded camera plus .848 for policy; GR00T .074 per camera plus .852 for policy. A full or completed policy call costs 1; blind reuse costs 0; at most one completion is charged. The legacy ledger remains a comparison field. Accepted decisions and look shards retain camera counts, availability, acquisition and completion reason. Collection emits `lookhalf_cost_ledger` and compute-share IR; diagnostic encoder work remains excluded.

GUARD.md section 8 takes precedence over the alternative five-cell list in the task note. Each root has deployed off, two-camera H2, and M2, with **100 episodes per arm / K=10 per task**. Selections are exact byte copies of `r11_dev_k10_size50` selections and are checked disjoint from the served subset. Four roots have knob off; the fifth uses the existing periodic .32 calibration.

| Root under `pilots/` | Knob | Arms |
| --- | --- | --- |
| `r12_guard_fix1_pi05_l10_50` | off | off, h2, m2 |
| `r12_guard_fix1_pi05_spatial_50` | off | off, h2, m2 |
| `r12_guard_fix1_groot_l10_50` | off | off, h2, m2 |
| `r12_guard_fix1_groot_spatial_50` | off | off, h2, m2 |
| `r12_guard_fix1_pi05_l10_50_periodic_ir32` | periodic .32 | off, h2, m2 |

Every arm is configured for local serving, one policy-tail block, B-only look recording and 56-pixel wire thumbnails. GR00T also records exact-input joint/single encoder diagnostics every twenty looks through the existing adapter. `guard_readout.json` currently reports pending with zero episodes; no rollout success or IR result is invented.

After collection, `report.py` checks accepted complete attempts only, reports SR, total and guard call rates, calls in failing episodes, mean first guard call in failing episodes, failed episodes without a guard call, one-camera share, planning time, numerical diagnostics and the collected ledger. It reports the frozen pilot checks: calls within .03 of deployed, a stop flag if first guard calls in failing episodes average at least three decisions later, and M2 share within .10 of the frozen expectation. No-call failures are exposed separately rather than assigned a fake first-call step.

## Exact coordinator commands

These commands are handed back, **not executed by sol**. Install the updated audited B worker using the command below before any active pilot. An older worker is refused at the first active request. Use an idle local GPU and worker fleet; the prior look-half pilot was stopped by the coordinator. The guard script reads listener state and refuses **sync and chain** while ports 23100–23103 are listening; it never signals or stops their processes. The controller's existing port and fleet ownership checks remain in force.

`LOCAL_SERVER_COUNT` and `PORTS` are honored from the environment. The default is **four servers** and ports **23120,23121,23122,23123**; if PORTS is unset, the script derives the requested number from LOCAL_SERVER_COUNT starting at 23120. The number of ports must equal the server count. Reserved ports 23100–23103 are rejected for every action. WPS defaults to **16 per server**, giving four servers × 16 workers. Set both count and ports when using a custom list.

```bash
cd /home/weiland/projects/openpi
export SERVER_HOST=local WORKER_HOST="${WORKER_HOST:-timan107}"
export LOCAL_SERVER_COUNT="${LOCAL_SERVER_COUNT:-4}"
# Optional example for four explicitly chosen free ports:
# export PORTS=23120,23121,23122,23123
export WPS="${WPS:-16}"
export LOCAL_SERVER_CPUS=22-37,66-81
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src
unset OSCL_MANIFEST OSCL_TASKS OSCL_EPISODES OSCL_INIT_POOL

# Required once on an idle worker island; shared with look-half.
taskset -c 22-37,66-81 env CUDA_VISIBLE_DEVICES= .venv/bin/python -B \
  -m exp.offline_search.closed_loop.ops.h100.control setup --worker-only

bash exp/offline_search/rounds/r12/guard/coordinator_commands.sh plan
bash exp/offline_search/rounds/r12/guard/coordinator_commands.sh sync
bash exp/offline_search/rounds/r12/guard/coordinator_commands.sh chain
bash exp/offline_search/rounds/r12/guard/coordinator_commands.sh report
```

The script names all fifteen arms explicitly and runs roots sequentially. `chain` already collects. To repeat only collection after an interrupted collection phase:

```bash
bash exp/offline_search/rounds/r12/guard/coordinator_commands.sh collect
bash exp/offline_search/rounds/r12/guard/coordinator_commands.sh report
```

CPU verification and reproduction:

```bash
cd /home/weiland/projects/openpi
P=(taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python -B)
"${P[@]}" -m exp.offline_search.rounds.r12.guard.selftest
"${P[@]}" -m exp.offline_search.rounds.r12.guard.replay
"${P[@]}" -m pytest -q -p no:cacheprovider \
  --basetemp=exp/offline_search/rounds/r12/guard/scratch/pytest_coordinator \
  exp/offline_search/closed_loop/test_look_recording.py \
  exp/offline_search/rounds/r07/c2_wrist/test_dispatch.py
```

Explicit B-library risk/artifact rebuild, only before starting guard pilots:

```bash
"${P[@]}" -m exp.offline_search.rounds.r12.guard.build
# prepare.py creates fresh roots and deliberately refuses existing roots.
# Initial root creation, already completed for this handback:
# "${P[@]}" -m exp.offline_search.rounds.r12.guard.prepare
```

Rebuilding preserves the old look-half tree. Do not rebuild artifacts already in use by a guard chain. No test-A command is included. All CPU work uses cores 22–37,66–81 and single-threaded BLAS; no git, destructive directory removal, broad process killing or external messaging was used.

Fresh-root regeneration (use a new prefix after the existing emission):

```bash
"${P[@]}" -m exp.offline_search.rounds.r12.guard.prepare --prefix r12_guard_fix1
"${P[@]}" -m pytest -q -p no:cacheprovider \
  --basetemp=exp/offline_search/rounds/r12/guard/scratch/pytest_fix_coordinator \
  exp/offline_search/closed_loop/test_r12_pipeline.py
```

The coordinator script now names only `r12_guard_fix1_*` roots and all fifteen arms, honors `LOCAL_SERVER_COUNT`/`PORTS`, and defaults to four servers × WPS 16. Run look-half and guard command sets sequentially on the same fleet. Off, H2 and M2 use the identical corrected planning and contract path.

## Guard startup fix (sol_fix2)

`closed_loop/lookhalf_config.py::validate_fitted` now compares `guard_method` for `r12_guard` and retains `lookhalf_method` for look-half. Missing selections/fields, mismatches and aliased imports remain fail-closed. The startup, guard camera/query, diagnostics, recording-contract and ledger paths contain no sibling comparison against the inherited guard `lookhalf_method='off'`.

`closed_loop/test_r12_pipeline.py` regressions: `test_frozen_r12_pickles_match_fix1_arm_selection` **35 passed** (18 guard + 17 look-half); `test_frozen_h2_pickle_refuses_m2_selection` **1 passed** (also rejects off/absent selection); `test_aliased_r12_classes_require_matching_explicit_selection` **2 passed**; updated `test_fitted_alias_and_method_mismatch_fail` **1 passed**. Before the fix, the artifact regression reproduced **12 failures / 23 passes**. All 32 referenced fits use exact flags read from fix1 `arms.json`; the three prepared guard Spatial-200 fits have no emitted arms and use corresponding fix1 guard selections.

Full CPU checks: **101 passed, zero failures** — `test_r12_pipeline.py` **49**, `test_look_recording.py` **25**, look-half selftest **14**, guard selftest **13**. All ran via pytest on cores `22-37,66-81`, CUDA hidden and single-threaded BLAS. Coordinator report tests used temporary metadata copies; selftest entry-point evidence files were not rewritten. Log: `/tmp/r12_sol_fix2_5djog8ci/full_cpu_final.log`.

Frozen artifacts, fit hashes, arm/root definitions and their inventories are unchanged. Live look-half summaries changed concurrently: GR00T Spatial-50 updated at 18:41:12 CDT, before the full CPU suite started, and pi05 Spatial-200 gained a summary. No task code writes those summaries. No servers, real syncs/chains, smokes or remote operations were launched.

Exact changed source files for the coordinator to push to h100:

- `exp/offline_search/closed_loop/lookhalf_config.py`
- `exp/offline_search/closed_loop/test_r12_pipeline.py`

The only documentation edit is this appended handback section.
