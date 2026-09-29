# Q2 deployment adapters

Implemented and CPU-tested under `adapters/`. All 43 proposed arm specs and matching prefits are available. No server, simulator, new evaluation episode, GPU inference, worker, chain, port, remote host, or git operation was invoked. Shared code and run roots were not edited.

## HANDBACK

- **Full input specs:** [`adapters/emit_arms_all43.json`](adapters/emit_arms_all43.json), exactly the 43 plan names. Every arm explicitly carries `--os-root /home/weiland/trace_runs/offline_search_store`, `--os-no-shadow-native`, `full_model=true`, and `cost_ledger=true`. `<RUN>` placeholders remain for the coordinator. These are emit_arms inputs; no run root has been emitted.
- **Deployment classes:** [`methods.py`](adapters/methods.py): `ExtraDosePi05`, `ExtraDoseGroot`, `CacheDose`, `RiskLottery`, under module `exp.offline_search.rounds.r06.ideation_Q2.frontier.adapters.methods`. The extra `CacheDose` class implements the plan's four A-plus-fixed-dose arms as well as the lottery's base.
- **Prefits:** `/tmp/q2_adapter_fits/<arm>.pkl`, 43 deployment artifacts totaling **2,390,025,613 bytes**. [`prefit_manifest.json`](adapters/prefit_manifest.json) contains exact method/kwargs/cell identities, hashes, original deployed-fit provenance, and solved per-task lottery probabilities. Copy only the 43 artifacts in this manifest; the same scratch directory also contains test fits. No fit was written to a run root.
- **Prefit commands:** [`prefit_all43.sh`](adapters/prefit_all43.sh), [`prefit_commands.json`](adapters/prefit_commands.json). All executed successfully. Commands are sequential CPU processes with the required affinity/thread limits. They reuse verified deployed A/B fits when only serving or randomization changes; they do not run a new metric fit for each dose.
- **Two-arm, eight-episode smoke:** [`SMOKE_RECIPE.md`](adapters/SMOKE_RECIPE.md), [`emit_arms_smoke2.json`](adapters/emit_arms_smoke2.json), and [`smoke4_manifest.json`](adapters/smoke4_manifest.json). One pi05 B+.5 arm and one GR00T rho=.35 lottery, four episodes each. **Recipe prepared, not run.** Model transport, real latency, closed-loop effects, and SR remain unverified.
- **Validation:** [`test_results.json`](adapters/test_results.json), [`replay_rates.csv`](adapters/replay_rates.csv), and `adapters/replay_checked/commands.json`. All **56 CPU plugin replay cases passed**, totaling **59,946 decision requests** over repeated recorded streams. There are **1,120 stream replays**, not 1,120 new or independent episodes. The same 20 recorded task/init pairs per cell are reused across cases and library sizes.

### Required plan annotations before finalization

1. The latest instruction explicitly requires **10 controls** for the lottery. This supersedes the earlier 15-control GR00T Spatial-500 extension. Both existing arm names remain in the 43-arm file, but now use A10 and a one-block policy tail. [`plan_changes.json`](adapters/plan_changes.json) records the amendment. `rho=.09` is feasible. **`rho=.065` clamps to A's library-predicted IR floor `.07535392732533401`**, with dose zero; it cannot attain .065 at this cadence. The fit and query logs explicitly report clamping. It is an A-equivalent control, not an unobserved cheap crossing.
2. **The original pi05 `r6q2_pi05_spatial_500_A15` proposal was mistaken.** `harness/dims.py` and the stored chunks have H=10, not H=50. `BlindAWM(budget=2, serving="anchor_tail")` requests vision when its tail is exhausted at 10 controls. Its supplied spec is executable but is effectively an A10 duplicate; it cannot test a 15-control same-chunk cache. The requested name/configuration is retained rather than silently replacing it with a different controller. Drop or replace this redundant slot at the already-planned finalization. The existing GR00T A15 / CycleTail15 arms remain valid because GR00T H=16.
3. These three annotations do not change the remaining arm names or the other dose/rho positions. All positions remain provisional until the owner finalizes the complete pilot's placement curves. No pilot outcomes were read for this adapter implementation.

## Serving contract

### Deployed B plus extra dose

`ExtraDosePi05(_ExtraDose, CommitJudge)` and `ExtraDoseGroot(_ExtraDose, GrootCommitJudge)` preserve the deployed B constructor kwargs, retrieval, guards, state transitions, cache invalidation, and policy-tail hook. The emitted base kwargs exactly equal the audited `r5q1_c10_p_*` / `r6p1_c10_g_*` specs; the prefit builder checks this equality.

At a genuine vision query, first execute the original B `query`. If its `os_force_miss` is true, keep the mandatory call and original reason. Otherwise call iff the keyed uniform is below d. An added call sets `os_force_miss=1`, `os_reason=61`; it does not overwrite B's `os_flags` or guard/progress memos. The plugin commits the actual MISS to history, invalidates the cache anchor, and serves the policy. B's existing saved policy-gate anchor and C10 lifecycle then serve exactly the following five-control tail. The GR00T subclass retains `GrootCommitJudge`'s existing gripper-sign correction; no action coordinate is changed by the adapter.

`d=0` returns the superclass `Result` immediately, without coin generation, extra fields, or changed B query state. CPU replay matched deployed B in every one of the eight cells. Results include identical retrieval rows, scores, confidence, actions, guard extras, vision/miss schedule, and served chunks, excluding metadata/timing fields in plugin log comparisons.

### A plus fixed dose / risk lottery

`CacheDose` directly subclasses **`BlindAWM`**, using the exact A kwargs, so it preserves A's retrieval tie rule. It does not use B with `guards=False`: the P1 handback documents a few historical A/B-off tie differences. A policy MISS saves a separate gate record before the plugin invalidates A's cache anchor. `policy_tail_step` invokes the existing **`CommitJudge.policy_tail_step` lifecycle code** through an ephemeral facade with `monitor="off"` and `base=self`; it is not a new copy of the lifecycle logic. No facade or cyclic object is retained in connection state.

`RiskLottery` uses that same A query and policy-tail transport. On reset, select a single episode dose from the frozen adjacent-knot distribution. At every vision anchor use a separate independent keyed uniform with that dose. Never reselect the dose based on state, guards, apparent progress, number of prior calls, episode duration, or terminal outcome. `allocation="uniform"` sets equal task risk weights before solving the same budget; this option is tested although the current 43-arm plan contains no uniform-lottery arm.

`rho=0` returns the exact A query result without new query extras and serves no policy calls. A still pays its vision cost: zero requested IR is infeasible and the fit records the clamp. The zero-dose assignment is deterministic (dose 0, assignment probability 1) and inferable from the startup kwargs. Nonzero-rho arms, including ones clamped to dose 0, log their assignment explicitly.

Both adapters require `--os-blind --os-policy-tail --os-policy-tail-blocks 1 --os-judge guard_only`, `replan_steps=5`, and a full model. These are present in all 33 new deployment specs. No P3 Profile object, shadow policy call, K4 resample, snapshot hook, sequential importance weighting, or GPU retrieval is used. The only new query work is hashing, a scalar decision, and diagnostics in the existing logger. Real CPU latency overhead has not been measured; replay timings are not deployment latency measurements.

## Frozen budget and seeds

[`calibration.json`](adapters/calibration.json) contains eight libraries × ten task rows, copied from the pre-existing **library-only** Q2 fallback `library_quality.json`, with SHA256 provenance. Per task: q is `state_loeo_distance`, h is mean stored episode decision count, and a is mean `ceil(episode_decisions/2)`. `load_calibration` verifies the actual bank's manifest, robot-state, task-ID, and episode-array hashes and recomputes h/a before accepting a fit. No SR or evaluation outcome enters this file.

The solve in [`budget.py`](adapters/budget.py) is the preregistered rule: default equal task traffic; replace zero risks with machine epsilon times the largest positive risk (all zero => uniform); normalize q by its traffic-weighted mean; `p_t=min(1,lambda*w_t)`; solve the traffic-weighted predicted owner cost using **80 bisection iterations**. Clamp below the A floor or above every-anchor cost and retain the infeasibility flag. Mixtures use adjacent knots **{0, .125, .25, .5, 1}** and exactly reproduce expected dose p_t. All 144 risk/uniform/target checks matched `pilot_q2.allocation`.

Current c1 values `.152` / `.148` are the owner's accounting convention carried in calibration metadata, **not newly measured library latencies**. On another robot, record stage timings on library inputs, supply that measured c1 and valid episode lengths/risks, and freeze the new calibration file/hash. The allocation algorithm has no fitted LIBERO task thresholds; the serving adapter here intentionally implements the installed five-control, ten-control-commit contract. A weak library risk score does not certify SR, and trajectory-length changes can move realized IR away from the library forecast.

Exact randomization version is `Q2-deploy-v1`. Hash the compact ASCII JSON array

```
["Q2-deploy-v1", randomization_key, random_seed, task_id, original_init, decision_step, domain]
```

with SHA256. Interpret its first eight bytes as a big-endian integer, drop the low 11 bits, and multiply by 2^-53. This produces an exactly representable uniform in [0,1), avoiding the possibility that float division of a full 64-bit integer rounds to 1. The domains are `extra-anchor` for B's extra calls, `dose-anchor` for A's anchor coins, and `episode-dose` with decision step −1 for the episode lottery. The inverse CDF uses right-side search so zero-mass endpoints are skipped.

All supplied deployment specs use `random_seed=26092901` and a key combining the saved eval500 manifest SHA with the cell. The key is frozen in the kwargs. No UID, worker, arrival order, attempt count, or eventual outcome is hashed; retries of the same original init reproduce the intervention schedule. For independent replicate blocks, change **`random_seed` in kwargs and generate a matching prefit**. Changing only `--os-seed` does not change these private intervention coins. This is a declared implementation version of the earlier recipe: it uses actual decision index rather than an order-dependent anchor counter. Neither the global NumPy/Python RNG nor the policy's RNG is consumed.

## Logging and cost

The existing plugin logs these numeric `extras` at every relevant vision anchor. Their `os_` prefix preserves them ahead of the mixed logger's 40-scalar cap; end-to-end replay checked their presence.

| Field | Meaning |
|---|---|
| `os_q2_base_miss`, `os_q2_eligible` | B's original forced verdict and eligibility for an extra coin |
| `os_q2_coin`, `os_q2_dose`, `os_q2_extra` | anchor uniform, applied dose, added MISS flag |
| `os_q2_p_call` | actual conditional call propensity: 1 at mandatory B anchors, otherwise dose |
| `os_q2_episode_dose`, `os_q2_episode_propensity`, `os_q2_episode_coin` | episode assignment, its probability, and assignment uniform |
| `os_q2_task_p` | library-solved expected task dose |
| `os_q2_rho`, `os_q2_predicted_IR`, `os_q2_clamped`, `os_q2_uniform` | requested budget, library forecast, feasibility, and allocator |
| `os_q2_seed`, `os_q2_version` | intervention seed and version marker |

Join the first anchor (`step=0`) to the accepted episode using `(uid, attempt, task_id, init)`. The dose/propensity is repeated on later anchors as an audit. The plugin terminal `ev=episode` schema is unchanged. At exact zero settings the original query extras are unchanged, preserving the identity test.

Policy inference happens only through the existing served-MISS path. Thus the ordinary ledger prices `c1*V/N+(1-c1)*M/N`, with zero charge for blind cache and policy tails. B-extra's d is a probability **on B-cache opportunities**, not total M/V. The lottery's conditional rate in a small replay depends on which episode doses were sampled; do not compare it blindly to the task-marginal p_t.

## Measured CPU validation

Replayed the real `PluginStrategy`, `PluginJudge`, `PluginStorage`, connection wrapper, blind transport, and `CacheOrchestrator` using recorded keys/states and recorded `a_inf` chunks as the fake policy response on MISS. No observations respond to the newly served actions. Tests therefore establish deployment mechanics, **not closed-loop SR or causal call benefit**. Git startup provenance was explicitly disabled in the test process; shared files were not patched.

- **56/56 plugin replay cases passed**; 59,946 decision requests, 9,265 non-final MISS tails checked against the exact original chunk's rows 5..9. Every such tail is followed by vision; no 10-control adapter served a second blind tail. Fake stage-1 calls exactly equal vision counts; tail logs have no vision/MISS timing charge.
- Zero-dose identity passed in **all eight cells** for A and B, over **16,600 paired replay decisions** combined. Four additional repeat comparisons reversed recorded episode order and changed UID prefixes; selected sources, actions, probabilities and retrieval outputs stayed identical.
- **8,676 full `Result` comparisons** on recorded all-vision query streams matched exactly, including every extras field and array dtype/bytes. This exercises GR00T's sign-correction branch, not just ordinary alternating anchors: **56 terminal-closed and 20 terminal-open** cases had the correct terminal flag.
- **163,840 synthetic episode-lottery assignments**, 144 allocation checks, six invalid-configuration refusals, private RNG reproducibility/isolation, and all 43 fit/spec identities and safe per-connection clones passed.
- Each ordinary replay uses all ten tasks and original inits 0/49 (20 recorded streams). Success labels are not used. `replay_checked/commands.json` records every command and exit status. Per-arm call-rate results are in `replay_rates.csv`.

For B plus d=.5, the realized **extra-call rate on eligible anchors** was:

| Cell | Extra / eligible | Realized | Nominal |
|---|---:|---:|---:|
| pi05 LIBERO-10 50 | 245 / 458 | .5349 | .5 |
| pi05 LIBERO-10 500 | 236 / 463 | .5097 | .5 |
| pi05 Spatial 50 | 102 / 201 | .5075 | .5 |
| pi05 Spatial 500 | 101 / 202 | .5000 | .5 |
| GR00T LIBERO-10 50 | 228 / 448 | .5089 | .5 |
| GR00T LIBERO-10 500 | 230 / 449 | .5122 | .5 |
| GR00T Spatial 50 | 101 / 197 | .5127 | .5 |
| GR00T Spatial 500 | 102 / 210 | .4857 | .5 |

The rho=.2 risk replays also passed exact keyed-coin/propensity checks. Their largest absolute conditional count z-score was **2.341**; finite replay rates are not expected to equal nominal probabilities exactly. The preregistered software sanity bound in the test is six conditional standard deviations; this is not a success-rate hypothesis test.

## Reproduction

Executed from `/home/weiland/projects/openpi`. Every Python invocation uses `.venv/bin/python`, the requested affinity, one BLAS/OMP thread, and an empty CUDA device list. `run_replays` uses three subprocesses plus its coordinator, at most four Python processes.

```bash
taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r06.ideation_Q2.frontier.adapters.prepare_adapters
bash exp/offline_search/rounds/r06/ideation_Q2/frontier/adapters/prefit_all43.sh
taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r06.ideation_Q2.frontier.adapters.prepare_replays
taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src TMPDIR=exp/offline_search/rounds/r06/ideation_Q2/frontier/adapters/.tmp .venv/bin/python -m exp.offline_search.rounds.r06.ideation_Q2.frontier.adapters.run_replays --out exp/offline_search/rounds/r06/ideation_Q2/frontier/adapters/replay_checked
taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src TMPDIR=exp/offline_search/rounds/r06/ideation_Q2/frontier/adapters/.tmp MPLCONFIGDIR=exp/offline_search/rounds/r06/ideation_Q2/frontier/adapters/.mplconfig .venv/bin/python -m exp.offline_search.rounds.r06.ideation_Q2.frontier.adapters.check_adapters --replays exp/offline_search/rounds/r06/ideation_Q2/frontier/adapters/replay_checked
```

Replay outputs deliberately refuse overwrite; use a new owned output directory to repeat. Logs: `prefit_all43.log`, `prepare_replays.log`, `check_adapters.log`. The earlier two-case probe in `replay_probe/` was also successful but is not added to the main test counts.

`prefit.py` can fit directly from the library with `--fresh` when the requested output artifact does not yet exist. The delivered artifacts use verified deployed fits: base method/kwargs/cell and source digest are checked, then only new randomization settings or known serving-only budget/cycle settings are installed. Direct fresh numerical refitting was not separately tested in this turn. The plugin's own artifact identity check remains active at startup.

Sources read: `rounds/r05/q1_commit/judge.py` and its HANDBACK, `rounds/r06/p1_groot_commit/HANDBACK.md` and judge, `closed_loop/README.md`, the existing blind/commit/plugin implementation and replay helpers, `harness/dims.py`, the Q2 preregistration and frozen library fallback, and audited emitted arm specs. The two plan limits above come from these actual contracts, not an inference from pilot outcomes.
