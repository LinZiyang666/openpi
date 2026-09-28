# K5 hand-back: randomized single-landmark CALL/CACHE identification

Implemented the experiment overlay, logging, four arm specifications and an offline estimator. No learned controller, lambda selection, policy/action blending, library changes or deployment table was added. All shared edits were developed in `dev/`, compared against `before/`, and installed by same-directory temporary file plus atomic `os.replace`. No GPU, server, port, LIBERO worker, chain, remote host, git command or `tests/review_tests` was used. All Python processes ran on CPUs **26–29,70–73**, with OMP/OpenBLAS/MKL threads 1 and CUDA disabled.

## Installed files and provenance

Paths below are relative to the repository. Shared replacements:

| File | Atomic install UTC | SHA256 |
|---|---|---|
| `exp/offline_search/closed_loop/plugin.py` | 2026-09-27T20:06:24.803540+00:00 | `5bd0c878774ccd79fc79d1e53cee167f9e5bbca039af2b473c9e5dea8bc89f9a` |
| `exp/offline_search/closed_loop/selftest.py` | 2026-09-27T20:06:24.806427+00:00 | `58b3e54750ce8ed5ffd2fffd26e62e6c1fcc1041ea9187cd4d5924368c1bb954` |
| `exp/offline_search/closed_loop/verify_logs.py` | 2026-09-27T20:06:24.808715+00:00 | `5ba206f439c46ecb21a1c0e4956e7be4d9476773a246735b11af3feffa91a50f` |

`plugin.py` adds opt-in CLI validation, episode assignment/reset, post-judge override, JSONL and NPZ records. `selftest.py` adds `--rand-seed` / `--rand-replicate` with the same fake policy path; `verify_logs.py` independently reconstructs the ordinary verdict from guard inputs and then audits assignment, opportunity, context, actual verdict and truthful run length. Existing behavior is retained when the new flags are absent.

New top-level files under `exp/offline_search/rounds/r04/k5_rand/` (UTC modification times; new files were available before their first use):

| File | Modified UTC | SHA256 |
|---|---|---|
| `__init__.py` | 2026-09-27T20:13:25.021126+00:00 | `5d0d137854ad77df57b4d338eb9e493ca1812adc83163cfbb16f1d89bf8b960c` |
| `arms_rand.json` | 2026-09-27T20:01:03.521450+00:00 | `fe13655ff189369dcaf10c6207e07a67250c364964795aecef26a2e391fc5567` |
| `check_ledger.py` | 2026-09-27T20:09:32.785945+00:00 | `8bb410e6d5610e8b7ed6471b04c4f424f2fdc5ff75ee5620bbadff04b8e37b3a` |
| `check_parity.py` | 2026-09-27T19:57:22.148157+00:00 | `7bcde07e1ada92247c6432be251cd06eaaa773ba5421ae5772c31141f5723903` |
| `check_replay_estimator.py` | 2026-09-27T20:07:43.090852+00:00 | `e2cd01a135975a6824aabe11fcae9cfdbdf5f8a7114632c9c577f824083e4749` |
| `estimate.py` | 2026-09-27T20:09:32.767945+00:00 | `fb65354b10577899e735316c8ae7e51e7307771f5223a435a935b38e135790d4` |
| `final_audit.py` | 2026-09-27T20:11:43.656049+00:00 | `b15aa33e3e3981aa39f20044f820181b1f4ca69ffdcc5ca238db927008113e2d` |
| `install.py` | 2026-09-27T20:06:24.743782+00:00 | `4d506fc8ff90666ff2b78cec28e60be7086f1e89fbe5b09fdac4726107d2fb8c` |
| `launch_test.py` | 2026-09-27T19:57:22.147157+00:00 | `82e03dc95ab3a09ebb18207955c4298e5780c8c800a34408f4036e25ff129546` |
| `make_existing_checks.py` | 2026-09-27T20:06:24.819782+00:00 | `2c5c886a40daec8646caaf32e553c5c9515e92c9a7bc62425ca7732ef51a95fe` |
| `overlay.py` | 2026-09-27T19:56:32.199081+00:00 | `8c5f74020b90d2dde822a1ae5a1d0496535f26ac8c7d31733f3c44b3bb11021b` |
| `prepare_arms.py` | 2026-09-27T20:01:03.471450+00:00 | `95a436435705bc52ef895c02610e3fb18ba185214ebbf0b079961be3aebb01ac` |
| `run_all_checks.sh` | 2026-09-27T20:13:25.023126+00:00 | `15ffc40d4f5cc62e2fc3f8741ee52d8986b7b45d8eef248b7b4749df0db67b26` |
| `run_random_replays.py` | 2026-09-27T20:03:04.968585+00:00 | `cc5e5da3347dc3de6ba9c4a1764e6f1353f8e262f2cedfd61310df8d4e882045` |
| `summarize_checks.py` | 2026-09-27T20:12:37.325090+00:00 | `cab51da2bf423d2cc7988277884efb145dd2e5fc0739b5b8e916193e660dc68c` |
| `test_overlay.py` | 2026-09-27T20:03:04.966585+00:00 | `eb247356dfd58a33084065182b0830390eaf26ce331c39769266695d487bd22b` |
| `validate_estimator.py` | 2026-09-27T20:05:18.923720+00:00 | `80dd72b2cc1aaed23078ab9d81273419972a914505cee885db82033375b70619` |

`overlay.py` implements assignment/context/state; `estimate.py` implements ingestion/statistics; `arms_rand.json` and `prepare_arms.py` prepare reviewed configurations and reuse verified fits. Remaining Python/shell files are verification or installation helpers. `before/` contains the three original files, `dev/` the final candidate copies and adapted recipes, and `results/` the evidence. `results/install_times.json` records original and installed hashes; `results/deliverable_hashes.json` records added source timestamps/hashes. `results/file_inventory.json` lists every K5 artifact, including this hand-back, with timestamp and SHA256 (excluding the inventory itself).

## Exact switches, identity and behavior

Enable with both:

```text
--os-rand-seed 20260927 --os-rand-replicate 1
# Complementary arm:
--os-rand-seed 20260927 --os-rand-replicate 2
```

Keep `--os-judge guard_only`, default cap 0, judge burst 1, step0 `judge`, and guard-only MixedJudge (`guards=true`, `events=none`, method burst 0). The overlay explicitly refuses `--os-blind`, other cells than `pi05_l10_cache`, incompatible judge settings and non-guard MixedJudge configurations. Either randomization flag alone fails with a clear error. Randomization also enables R4 logs; all supplied arms explicitly include `--os-log-r4`.

Identity is **the original init**, not the connection, arrival order, attempt, task UID, EpisodeView store index or `--os-seed`. `exp/gate_threshold_pareto/run_gtp.py` constructs `EpisodeTask.orig_init_state_idx` (episode index for the ordinary 50-init A-pool; original mapped index for subset pools). `WebsocketClientPolicy.episode_start` sends `extra_metadata` as `__extra__`; the existing server `_ConnPolicy._osp_episode_start` passes it to `PluginSession.client_episode_start`. At `_begin`, the plugin's lazy episode reset before the first method query, task ID comes from the library task map and is checked against metadata; init comes from **`extra_metadata.orig_init_state_idx`**. A missing/negative init fails rather than using order/connection. `episode_id` is deliberately not a fallback.

A private `random.Random` seeded from SHA256 of `causal_rescue_credit:v1:<experiment_seed>:<task_id>:<init>` draws two independent bits: class 1/3 and CALL/CACHE. Replicate 2 flips only treatment. There is no global RNG consumption. Assignments are identical across library scales for the same seed; estimates must still be separate per scale. `trial_id` contains seed/task/init/replicate; arm/tag identify the library scale.

After the ordinary verdict, each baseline MISS increments the episode opportunity counter. Only the assigned first/third opportunity is `eligible=true`. CALL keeps the policy MISS path; CACHE changes the actual verdict to HIT and serves that decision's ordinary proposal, with no new retrieval or synthesis. Subsequent decisions run the ordinary controller on the actual history. `on_executed` appends the actual HIT/MISS and chunk, and the full MixedJudge replay confirms correct fresh/stale regime, gripper and history inputs. Unreached landmarks do not change executed behavior. The count resets per episode and is isolated per connection.

Every randomized decision row contains:

- `trial_id`, `experiment_seed`, `replicate`, task/init;
- `eligible` (the one assigned opportunity, not every baseline MISS), `opportunity_index` (cumulative baseline MISS count, including this decision), `landmark_class` (1 or 3), `assigned_treatment`, `propensity=0.5`;
- `baseline_verdict` and `actual_verdict` (`HIT` / `MISS`), plus ordinary `hit`, `src`, guard extras, and `judge=randomized:CALL|CACHE` at the landmark;
- `context` and numeric `context_values`, captured before applying the treatment: selected top-1 library progress `<0.5`/`>=0.5`; previously executed normalized gripper `[4,6]` (<0 open, >=0 closed for pi05); confidence `>-0.3`/`<=-0.3`; age since first observed `stuck_n>=2` OR `noprog_n>=2`, binned `before`/`0-9`/`>=10`.

At step zero there is no previously executed gripper: its bin is null, never invented. Such an exposed episode remains in ITT/parent estimates and is counted as an unknown context; it is not fit into a 48-bin child table. Current-decision stall evidence gives age 0. Progress means retrieved library progress, not simulator success.

All R4 fields remain present, including **actual `served_head` on every HIT and MISS**, `vision`, `src`, `miss_k`, and stage timings. Episode records add assignment, `exposed` and `baseline_miss_opportunities`; startup records identify randomization configuration. With `--os-log-inputs`, the NPZ carries a `randomization` JSON string per decision for the verifier. Guard-forced MISS extras intentionally remain visible on an overridden CACHE row; `actual_verdict`/`hit` describe what executed.

## Arms and fit reuse

`arms_rand.json` is emit_arms format with literal `<RUN>` fit placeholders. It contains exactly four arms, all full-model, five controls/request, `cost_ledger: true`, guard-only MixedJudge, and `--os-log-r4`:

| Arm | Library | Replicate | R3 source fit |
|---|---|---|---|
| `r4k5_p_l10_g50_r1` | 50 | 1 | `r3mx_p_l10_g.pkl` |
| `r4k5_p_l10_g50_r2` | 50 | 2 | same |
| `r4k5_p_l10_g500_r1` | 500 | 1 | `r3mx_p_l10_g500.pkl` |
| `r4k5_p_l10_g500_r2` | 500 | 2 | same |

Both use the exact method string `exp/offline_search/rounds/r03/h3_judge/judge.py:MixedJudge`, base string `exp/offline_search/rounds/r02/g1_awm/awm.py:AWM`, `guards:true`, `events:"none"`, cell `pi05_l10_cache`. g50 uses `base_kwargs:{"lib":"current","kref":5}`; g500 uses **`base_kwargs:{}`** exactly. Do not substitute equivalent-looking kwargs: the pickle contract uses exact dictionaries/spec strings/cell.

`prepare_arms.py --run-root RUN` verifies those three metadata fields before copying each original R3 pickle atomically into `RUN/fits/`, verifies SHA256, resolves placeholders, and invokes the existing emitter. It never starts a server. `--smoke` selects only the g50 replicate pair. `_reuse_fit` is preparation metadata, not a new emitter field. All four emitted arms were checked under `/tmp/k5_arms_check`; see `results/emitted_arms.json` and `results/fit_audit.json`.

Fit files under `/home/weiland/trace_runs/os_closed_loop/r03_mx/fits/`:

| Fit | Bytes | SHA256 |
|---|---:|---|
| `r3mx_p_l10_g.pkl` | 32,602,497 | `a6fe33ee8dda8843e693880d224b0c7aff38ea48356692d58c97a34975ccfcd8` |
| `r3mx_p_l10_g500.pkl` | 141,226,847 | `dfe51ea22257e877d4036933a8c7b60655e4ac0bf66ca948f4c9b76898cbf52f` |

No new fit was trained for these arms. The experiment adds no per-entry representation; synthesis and candidate libraries are unchanged. The R3 l10 libraries are 50/500 episodes, 2,640/29,472 entries, MixedJudge representation 626 B/entry (R3/ideation specification), versus the deployed pi05 l10 pickle reference ~1,103 MB. The fit byte sizes/hashes above were measured here. Outcome-derived estimator tables are **borrowed big-library information** beyond the deployed action library; they are research outputs only.

## Estimator and interpretation

`estimate.py` accepts two complementary replicate arms, separately per library scale, plus optional `--baseline ARM` entries and multiple `--run-root` paths. It reads accepted terminal journal attempts (`accepted`, status done/failed, no error), joins matching server attempts, deduplicates identical `(uid,step)` records, and checks contiguous decisions, episode-end totals, action/head fields, assignment/context/opportunity consistency and treatment compliance. Missing accepted logs, conflicting duplicates, incomplete replicate pairs and differing controller/library configurations fail clearly. It never substitutes rejected attempts or silently drops unmatched inits.

`--input` accepts a collect summary or KPI JSON as a locator for its raw run root; those aggregates do not contain enough information to infer randomization/context/cluster effects alone. Raw logs must remain. `--episodes-out` writes the explicit `causal_rescue_credit.episodes.v1` per-episode export, which can subsequently be used with `--input` independently of the raw root. `--arms REP1 REP2` is always explicit.

Outputs include ITT over **all assigned episodes**, ITT by class, exposed parent effects for each class, all 48 class/context children, exposure/compliance and task/class/context balance tables. Unexposed episodes retain their assigned treatment in ITT. Conditional effects only use the context recorded before the eligible intervention. Discordant pair exposure and unknown-context counts are reported.

For each population S, use Horvitz–Thompson inverse-propensity means:
`mean_CALL = sum(1[CALL]*outcome/.5)/|S|`, `mean_CACHE = sum(1[CACHE]*outcome/.5)/|S|`; report CALL minus CACHE for Y=success, N=all requests, M=all actual MISSes, including the entire subsequent episode. These are not arm-name contrasts. HT arm means in small imbalanced subgroups need not be bounded by ordinary outcome ranges.

Bootstrap resamples **(task, original init)** clusters with both replicates retained, recomputes subgroup denominators and parent/child effects, and reports pointwise percentile 95% intervals (default 2,000 draws). Child shrinkage is `n/(n+100)` toward its exposed class parent, where n is the number of exposed episode records in that child (not decisions); bootstrap draws recompute n and shrinkage. Both treatment labels and at least two init clusters are required for a supported cell. Unsupported cells have null shrunk estimates/intervals and retain baseline CALL. This is an explicit minimal support rule, not a claim that two clusters give useful precision. Shrunk intervals are research summaries, not validated controller-selection bounds or simultaneous 48-cell coverage guarantees.

Cost outputs apply the stipulated owner basis:
`ΔCρ = .848*ΔM + (.152−ρ)*ΔN`, at **ρ_g50=.3234160220826887**, **ρ_g500=.23821622358554875**. The same bootstrap gives their intervals. Those historical operating points were re-derived from raw baselines: g50 N=32,967, M=6,664, SR=.740; g500 N=29,340, M=2,983, SR=.864 (500 episodes each). Optional baseline comparisons use matching inits and clustered intervals but are explicitly descriptive, not randomized comparisons against historical controls.

The existing R4 cost ledger also works on all four new replay logs (342 vision decisions per arm, K=10, L=5, no ledger warnings). It uses the installed K3 cost table; that basis is distinct from the requested `.152/.848` causal cost formula. Fake-policy timing fields are not GPU latency measurements.

## Offline verification: final installed results

All runs passed after final installation; no shared source changed afterwards.

| Verification | Final observed result |
|---|---|
| Existing K2 serving matrix | 21 PASS, 963 decisions |
| Existing K1 blind serving matrix | 10 PASS, 480 decisions |
| Existing K4 plugin matrix | 4 PASS, 156 decisions |
| Total existing plugin selftests | **35 PASS, 1,599 decisions** |
| New real-MixedJudge replay matrix | **4 PASS, 1,368 decisions**, 16 episodes |
| New injected lifecycle/history test | **104 decisions**, 12 episodes, 4 interleaved connections; 10 exposed, 2 unexposed; all history checks pass |
| Legacy pure/mixed/R4 JSONL parity | **byte-identical**, respectively 63,349 / 91,773 / 145,754 bytes; 45 rows each |
| Assignment stability | All 500 pairs identical across 3 fresh processes (different Python hash seeds), reversed arrival order and concurrent assignment calls; all 500 replicate flips complementary |
| Assignment balance, replicate 1 | first/CALL 119; first/CACHE 121; third/CALL 130; third/CACHE 130 (class 240/260, treatment 249/251) |
| Invalid configuration / identity | 6 option combinations rejected; missing original init rejected; blind mode has an explicit error |
| Raw ingestion / report input modes | 16 synthetic episodes, 16 duplicate decision rows ignored, rejected retries excluded; raw/collect/KPI/export results equal; conflicting duplicates and gaps rejected |
| Real-layout ingestion | Both historical guard arms: 500 episodes each; installed replay logs: 8 episodes / 684 decisions per library with optional historical baseline join |
| Forged-log / cross-scale audit | Four forged opportunity logs rejected; accidental g50/g500 replicate pairing rejected |

Existing recipes were copied into owned `dev/existing_{k2,k1,k4}.sh`, changing CPU affinity, cold-store/output paths and the test loader only. They did not write into K1/K2/K4 directories. K2's separate 41-decision ProbeHist logged-input replay has every equality field 1.0 and executed/selected 41/41. Evidence: `results/existing_summary.json` and per-test JSON copies; bulky arrays/logs are under `/tmp/k5_existing_installed`, with K4's extra artifact root recorded there. The summary helper initially expected a `PASS` key in the old verifier format; it was fixed to check that format's equality fields and rerun successfully. No plugin test failed.

Randomized installed replay details:

| Arm | Decisions | Eligible CALL / CACHE | MISSes |
|---|---:|---:|---:|
| g50 r1 | 342 | 3 / 1 | 117 |
| g50 r2 | 342 | 1 / 3 | 114 |
| g500 r1 | 342 | 3 / 1 | 105 |
| g500 r2 | 342 | 1 / 3 | 106 |

Every HIT executed its exact ordinary cache proposal; every MISS executed the recorded full-policy stand-in. Every online/offline topk, scores, confidence, library, synthesis and extras comparison is 1.0; verdict and run violations are zero in all four. All four guard reasons occurred. The injected test additionally makes landmarks coincide with step-zero and guard-forced MISSes, checks baseline resumption on the next decision, reverses interleaving order, resets the same identities with changed UIDs/attempts, and ends third-landmark episodes early. These are fixed-observation replays, not causal rollout outcomes.

Estimator Monte Carlo: **200 planted + 200 null datasets**, each 500 init clusters × two replicates, 499 bootstrap draws. A deterministic case recovers Δ(Y,N,M)=(1,4,2) exactly; parent/child shrinkage and unsupported cells are checked.

| Case | Metric | True ITT | Mean estimate | 95% coverage |
|---|---|---:|---:|---:|
| planted | ΔY | .096 | .09373 | .940 |
| planted | ΔN | −2.4 | −2.41659 | .960 |
| planted | ΔM | −.8 | −.80095 | .915 |
| planted | ΔC_g50 / ΔC_g500 | −.267002 / −.471481 | −.264963 / −.470856 | .935 / .935 |
| null | ΔY | 0 | .00058 | .955 |
| null | ΔN | 0 | −.05336 | .960 |
| null | ΔM | 0 | −.00336 | .970 |
| null | ΔC_g50 / ΔC_g500 | 0 / 0 | .006297 / .001751 | .940 / .945 |

Planted exposed effects are (.12, −3, −1), with .8 exposure probability, hence the smaller ITT values. These coverage checks concern unshrunk ITT intervals. See `results/estimator_validation.json` for exact numbers and the synthetic fixture path.

### Exact executed commands

From `/home/weiland/projects/openpi`, define this shell array (every Python command below expands to the full required affinity/thread/CUDA prefix):

```bash
PY=(taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python)
K5=exp/offline_search/rounds/r04/k5_rand
"${PY[@]}" "$K5/prepare_arms.py" --run-root /tmp/k5_arms_check
"${PY[@]}" "$K5/install.py"
"${PY[@]}" "$K5/make_existing_checks.py"
bash "$K5/dev/existing_k2.sh"
bash "$K5/dev/existing_k1.sh"
bash "$K5/dev/existing_k4.sh"
"${PY[@]}" "$K5/test_overlay.py" --source installed --out /tmp/k5_overlay_installed
"${PY[@]}" "$K5/check_parity.py" installed
"${PY[@]}" "$K5/run_random_replays.py" installed
"${PY[@]}" "$K5/validate_estimator.py"
"${PY[@]}" "$K5/check_replay_estimator.py"
"${PY[@]}" "$K5/check_ledger.py"
"${PY[@]}" "$K5/final_audit.py"
"${PY[@]}" "$K5/summarize_checks.py"
```

Outputs were redirected to the matching `results/*.log` files. The development checks used `dev` instead of `installed`, before installation. `results/replay_commands_{dev,installed}.json` contains every expanded arm replay command. The adapted shell scripts contain all existing matrix commands. `run_all_checks.sh` combines the final verification commands sequentially, refuses existing fixed artifact directories to prevent blind-log appends, and passes `bash -n`; the combined wrapper itself was not run because its constituent commands had already run. Preserve/move the old K5 `/tmp` directories or use fresh adapted paths before a full rerun.

## Coordinator smoke recipe and next steps

The following operational commands are **for the coordinator; K5 did not execute them**. Use a distinct smoke run root because Cartesian-subset `.DONE` markers can otherwise skip a later full run with the same arm name. The two-arm smoke uses tasks 0,1 × inits 0,1: four episodes per arm, eight total. Coordinator chooses its own authorized free port/server CPUs and GPU budget. Keep five controls/request, full stage 1 and K=10; no blind, K2, wrist-only or other cost-engine composition in this identification trial.

```bash
# Set these to coordinator-owned paths/resources before running.
SMOKE_RUN=/home/weiland/trace_runs/os_closed_loop/r04_k5_smoke
K5=exp/offline_search/rounds/r04/k5_rand
# Use the prefixed PY array above for CPU-only preparation.
"${PY[@]}" "$K5/prepare_arms.py" --run-root "$SMOKE_RUN" --smoke
bash exp/offline_search/closed_loop/ops/sync_remote.sh "$SMOKE_RUN" r4k5_p_l10_g50_r1 r4k5_p_l10_g50_r2
# PORTS and SERVER_CPUS must already name coordinator-authorized resources.
unset OSCL_MANIFEST
PORTS="$PORTS" SERVER_CPUS="$SERVER_CPUS" OSCL_EPISODES=0,1 OSCL_TASKS=0,1 WPS=2 bash exp/offline_search/closed_loop/ops/chain.sh "$SMOKE_RUN" r4k5_p_l10_g50_r1 r4k5_p_l10_g50_r2
"${PY[@]}" "$K5/estimate.py" --run-root "$SMOKE_RUN"   --arms r4k5_p_l10_g50_r1 r4k5_p_l10_g50_r2 --boot 2000   --out "$SMOKE_RUN/k5_smoke_estimate.json" --episodes-out "$SMOKE_RUN/k5_smoke_episodes.json"
```

Check accepted pairs, one-or-zero eligible decisions per episode, complementary assignment, no compliance errors, valid served heads and complete R4 ledger fields. The smoke is for serving/instrumentation only; four init clusters cannot rank an SR effect. If it passes, prepare **a separate full run root** without `--smoke`, sync all four emitted arms using the coordinator's normal procedure, unset subset/manifest environment variables, and run the 500-init pair at each scale. The local serving plugin imports the new `k5_rand/overlay.py`; any separate server checkout must include it along with the three installed shared files. No client protocol or remote worker change is required.

After full collection, estimate separately (repeat with g500 arm names and baseline):

```bash
"${PY[@]}" "$K5/estimate.py" --run-root "$RUN"   --run-root /home/weiland/trace_runs/os_closed_loop/r03_mx   --arms r4k5_p_l10_g50_r1 r4k5_p_l10_g50_r2 --baseline r3mx_p_l10_g   --boot 2000 --out "$RUN/k5_g50_estimate.json" --episodes-out "$RUN/k5_g50_episodes.json"
```

Real GPU inference/MISS execution, websocket/simulator integration, concurrent live rollout reproducibility, realized causal SR/cost effects and throughput remain **unverified by K5**. Assignment reproducibility is guaranteed by the stable identity key; stochastic policy trajectories can still differ across servers/replicates. No runtime success/cost improvement is claimed. The raw-log estimator audit was exercised on fake-policy replays and real historical layouts; the journal labels constructed for the replay integration check are explicitly synthetic, not newly measured LIBERO outcomes.
