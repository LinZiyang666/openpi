# K3 cost engine hand-back

Completed 2026-09-27, local time CDT. No git command, simulator, LIBERO worker, remote host, live closed-loop chain, or real server/port was used. All Python work used `taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. At most one GPU process ran; each exited after its benchmark. No OOM occurred.

## Installed files and ownership

Shared files were developed in `k3_cost/dev/`, compiled/tested there, then installed with same-directory temporary files and `os.replace`:

| File under `exp/offline_search/` | Installed (CDT) | Change |
|---|---|---|
| `closed_loop/stage_overrides.py` | 2026-09-27 14:13:16 | New opt-in dummy cache / wrist stage override, coordinator payload propagation, startup logging hook, MISS-K preflight |
| `closed_loop/serve_pi05.py` | 2026-09-27 14:13:16 | Consume cost flags before plugin parsing; install hooks before model construction |
| `closed_loop/serve_groot.py` | 2026-09-27 14:13:16 | Log live K; reject bundle/head K mismatch before model load |
| `closed_loop/ops/start_server.sh` | 2026-09-27 14:13:16 | `GROOT_DENOISING_STEPS`, default 8 |
| `closed_loop/ops/cost_table.json` | 2026-09-27 14:20:32 | Measured ms/shares, K4-compatible schema, complete samples/provenance |

SHA256s, before images, and install audit are `results/install.json`, `results/cost_install.json`, `dev/before_*`. No `plugin.py`, `src/`, harness, profile, or earlier-round source was edited.

New owned sources: `method.py`, `bench_pi05.py`, `bench_groot.py`, `make_cost_table.py`, `make_arms.py`, `arms_r4.json`, `prefit.sh`, `prefit_arms.sh`, `check_{overrides,methods,arms,blind_wrist,launcher,startup}.py`, `run_selftests.sh`, `check_wrist_serving.sh`, `install.py`, `verify_final.py`, `INTEGRATION.md`, this file, development copies and result files. Large fit pickles are outside the repository, in `/home/weiland/trace_runs/offline_search_store/derived/r04/k3_cost/fits/` and `/home/weiland/trace_runs/os_closed_loop/r04_cost/fits/`.

## Switches and exact integration

π0.5 reduced MISS steps use the EXISTING `_miss_steps()` path. Arm fields:

```json
{
  "full_model": true,
  "yaml_patch": {
    "miss": {"num_steps": 2, "evidence_dir": "/home/weiland/trace_runs/os_closed_loop/r04_cost/evidence/ARM"},
    "write_policy": {"type": "never"}
  }
}
```

`miss.evidence_dir` is mandatory in `_miss_errors`; a patch containing only `num_steps` is insufficient unless K4 fills the directory. Keep the stored library's `denoise_schedule` unchanged. `trace`, shadow teacher, `warm_reset`, and routing must be absent/disabled. Existing trace-dual-derived cache YAMLs already satisfy these after the emitter removes `trace`.

GR00T additionally uses `GROOT_DENOISING_STEPS=2` (or `server_env: {"GROOT_DENOISING_STEPS":"2"}`); unset remains 8. K4's installed chain derives this from `yaml_patch.miss.num_steps`. Both the server preflight and existing interceptor reject mismatches. π0.5 default remains 10.

Stage options go in `plugin_args`, consumed by the server wrapper:

- `--os-stage1-mode dummy_cached`: complete original CP1 keys/prefix, cache the exact preprocessed **-1** dummy image's embedding. Each batch shape is cached once per model/weight-version/dtype/device. Changed pixels or a nonmasked dummy fail explicitly. No camera offsets change.
- `--os-stage1-mode wrist_only --os-no-shadow-native --os-tokens off`: only wrist tower on warmed HIT decisions; dummy is cached. The stage output owns a clone of the current preprocessed base image. Split, reordering, rebatching and `.to()` preserve it. Only stage 2 entry completes its tower, after CP1 has decided MISS; the prefix and normal stage 2/3 computation are then exact. No per-connection or global pending-image stash is used.
- **`--os-pack-prefix` is refused.** The optional 256-masked-token packing helper is retained only for measurement. Bfloat16 action parity failed, so no deployable arm or ledger mode credits packing.
- No stage flag means the model methods are untouched. Generated launcher commands with options unset are byte-identical to the original for both models, apart from generated timestamp/path normalization in the fixture. Required startup fields are intentionally additive.

The startup hook wraps `PluginRuntime.emit` before `plugin.install`: `stage1_mode`, `miss_steps`, `prefix_packing`. This works with K2's current plugin; **no further plugin field/edit is required**. K2's installed decision code resolves `miss_k` from the actual interceptor/head.

Wrist key/plugin contract: fixed CP1 layout, `vision_0` is a zero placeholder, `vision_1` is the exact pooled wrist key, robot state unchanged. The method must declare `camera_mode="wrist_only"`; the server rejects other methods. No native shadow or token consumer is permitted on the missing camera. `WristView` makes the legacy two-camera diagnostic/guard interface observe wrist in both slots, while the metric itself contains wrist only once: **64 wrist PCs + state8 = 72 dimensions**. V7 pseudo-queries and visual-still thresholds are refitted in that space. Stock raw query widths need no plugin changes.

## Artifacts and arm specs

`arms_r4.json`: **26 runnable specs**, validated with K4's installed `emit_arms.py` and real `load_cache_config`, including strict artifact spec/kwargs/cell checks.

- Batch two: existing l10 `g500`, `g`, `perk5`, spatial guard-only, all K2; supplemental spatial guard500 and l10 periodic5/500; spatial/l10 pure-inference K2 at seeds 1101 and 1102 (four baselines).
- Batch four: spatial/l10 × current/big × dummy/wrist, with and without K1 phase-particle B2 blindness (16 arms). The blind arms use `--os-blind`, `noprog_span`, and all gates. **B2 is prespecified, not a measured winner**; substitute the batch-three winner and refit when kwargs change.
- No packed-prefix arms. Default evidence/run root is `/home/weiland/trace_runs/os_closed_loop/r04_cost`.

Prefits completed for all 26 arms' required artifacts, reusing unchanged R3 artifacts where appropriate. `prefit_arms.sh` contains the exact per-artifact commands and skips existing files; `check_arms.py` validates those files instead of blindly trusting their presence. `prefit.sh` alone builds the four new nonblind wrist fits. Regenerate run-local paths with:

```bash
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m exp.offline_search.rounds.r04.k3_cost.make_arms --run-root /home/weiland/trace_runs/os_closed_loop/r04_cost
bash exp/offline_search/rounds/r04/k3_cost/prefit.sh
bash exp/offline_search/rounds/r04/k3_cost/prefit_arms.sh
```

Same-library wrist fits, no borrowing from 500 to current; PCA uses that same library's stored wrist pooled keys (camera deletion is exact). All metric, early-code, V7 and guard fits are rebuilt.

| Suite/library | Episodes / rows | Actual fitted pickle bytes | V7 calibration calls | Fit seconds |
|---|---:|---:|---:|---:|
| spatial/current | 49 / 1,018 | 15,389,123 | 1,987 | 2.626 |
| spatial/bpool_cs | 500 / 10,909 | 50,372,731 | 6,500 | 15.657 |
| l10/current | 50 / 2,640 | 21,125,076 | 5,230 | 7.517 |
| l10/bpool_cs | 500 / 29,472 | 116,011,290 | 6,500 | 42.537 |

`WristAWM` retrieval representation is 324 B/entry; `WristMixedJudge` reports 370 B/entry including inherited guard bookkeeping. These exclude action payload and fixed PCA/metric blocks; the table gives actual complete serialized artifacts. Compare the owner nominal deployed π0.5 pickles of **431 MB spatial / 1,103 MB l10**. The large l10 pickle retains standard padded actions and auxiliary calibration arrays; it is not a claim of a 324-B complete deployment. `results/method_checks.json` includes all fitted thresholds and calibration details.

## GPU parity and measured costs

RTX 4090, B=1, serving mixed bfloat16/float32, one CPU/BLAS thread, 5 warmups / 30 timed repetitions, median CUDA-event elapsed time. This includes CPU launch gaps in the eager stage; it excludes transports, retrieval and simulator work. **Current concurrent π0.5 disables compilation; the current GR00T launcher omits `--compile-stage1`. Neither measured serving path uses CUDA graphs.** These measured eager numbers must not be mixed with the historical graph constants 10.26/27.69/29.57 ms.

π0.5 exact stored observations: 12 inputs from two tasks per suite, episode start / next decision / final decision. Fixed noise, K2 and K10; dummy-cached and completed wrist paths each have **24/24 bitwise-identical full action chunks**, max absolute delta 0. All 12 full dummy prefixes/keys, completed wrist prefixes, and wrist keys match bitwise. Installed hooks plus coordinator split/rebatch/.to pass **4/4** additional prefix/action checks. Dummy/wrist action parity needs no tolerance.

Optional packing changes actions on every tested input: maximum absolute differences **0.011698544 (K2)** and **0.014065892 (K10)**. It is rejected for serving. Native re-encoding versus historical stored wrist keys has absolute differences up to 1.0 on these raw, unnormalized pooled features; the wrist override and current full-stage baseline agree bitwise on every input. Thus this establishes override parity, **not universal bitwise reproduction of the historical stored encoder path**.

| π0.5 stage | Median ms |
|---|---:|
| Stage 1 full | 65.783119 |
| Stage 1 dummy-cached | 44.967089 |
| Stage 1 wrist-only | 23.888945 |
| Base-camera completion after wrist MISS | 21.591600 |
| Stage 2 unpacked | 33.421633 |
| Stage 2 packed, nondeployable | 27.947440 |
| Stage 3 K1 / K2 / K5 / K10 | 37.271040 / 71.429184 / 180.868690 / 364.189178 |

Full K10 total **463.393930 ms**; shares **0.141959389 / 0.072123588 / 0.785917023**. Stage 3 full-loop average **36.418918 ms/step**. Use direct K measurements for auditing; the required ledger scales only stage 3 linearly.

| GR00T suite | Stage 1 | Stage 2 | Stage 3 K8 | K8 average per step |
|---|---:|---:|---:|---:|
| spatial | 31.548928 | 29.790720 | 175.959534 | 21.994942 |
| l10 | 33.511936 | 28.405760 | 201.045502 | 25.130688 |

GR00T full-loop upstream and explicit-noise split agree bitwise for **K1/K2/K4/K8 in both suites (8/8)**. The flat ledger entry averages the two suite medians: **32.530432 / 29.098240 / 188.502518 ms**, shares **0.130053482 / 0.116331913 / 0.753614605**, K8 average **23.562815 ms/step**. Each suite's prompt, token count, full samples and direct K timings remain under `models.groot.by_suite`.

Peak reserved memory: π0.5 **7.234375 GiB**; GR00T spatial **5.382813 GiB**, l10 **5.380859 GiB**. Each job checked >=14 GiB free and set a 9.5-GiB allocator cap. No OOM or GPU process remains.

Cost schema: `models.MODEL.full={s1,s2,s3,k}`, `models.MODEL.modes.MODE={s1,s2,s3,k,miss_s1_extra}`, `full_cost_ms`, all ms. **Wrist MISS must add `miss_s1_extra`**; the key-only cost is insufficient. K4 ledger passed all 8 mode/HIT/MISS combinations, full-reference IR=1 and L10 IR=.5 identities. For illustration only, current measured-cost K2 all-MISS ratios: full π0.5 .37126638, dummy .32634557, wrist .32745361; these are cost identities, not SR or closed-loop outcomes.

## Exact verification commands and final results

Run from `/home/weiland/projects/openpi`. All shell drivers prefix each Python command with the assigned taskset/environment. Results listed below are the final passing runs.

```bash
# CPU structure/default/launch safety, real fit reset/camera isolation and batch-four lifecycle
 taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m exp.offline_search.rounds.r04.k3_cost.check_overrides
 taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m exp.offline_search.rounds.r04.k3_cost.check_launcher
 taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m exp.offline_search.rounds.r04.k3_cost.check_methods
 taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m exp.offline_search.rounds.r04.k3_cost.check_blind_wrist
 taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m exp.offline_search.rounds.r04.k3_cost.check_arms
 taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m exp.offline_search.rounds.r04.k3_cost.check_startup
 bash exp/offline_search/rounds/r04/k3_cost/run_selftests.sh final
 bash exp/offline_search/rounds/r04/k3_cost/check_wrist_serving.sh
 taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m exp.offline_search.rounds.r04.k3_cost.verify_final

# GPU jobs ran SEQUENTIALLY, with admission checked again inside each process
 taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m exp.offline_search.rounds.r04.k3_cost.bench_pi05
 taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/home/weiland/projects/openpi_ext/third_party/gr00t_n15:/home/weiland/projects/openpi_ext/third_party/gr00t_n15/examples/Libero:/home/weiland/projects/openpi:/home/weiland/projects/openpi/src:/home/weiland/projects/openpi/packages/openpi-client/src /home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/bin/python -m exp.offline_search.rounds.r04.k3_cost.bench_groot --suite spatial
 taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/home/weiland/projects/openpi_ext/third_party/gr00t_n15:/home/weiland/projects/openpi_ext/third_party/gr00t_n15/examples/Libero:/home/weiland/projects/openpi:/home/weiland/projects/openpi/src:/home/weiland/projects/openpi/packages/openpi-client/src /home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/bin/python -m exp.offline_search.rounds.r04.k3_cost.bench_groot --suite l10
 taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m exp.offline_search.rounds.r04.k3_cost.make_cost_table
```

- `final_audit.json`: **PASS**; 26 arms, 8 ledger cases, 309 default / 844 wrist serving decisions, 16 Python files compile, installed hashes match, no OOM marker. All shell drivers also pass `bash -n`.
- `override_checks.json`: **11/11** structural/lifecycle/default/validation checks.
- `launcher/checks.json`: **4/4** groups; default command equality on both models, K2 forwarding, four invalid K values rejected. `tmux` and `ss` were stubbed: no server or port was touched.
- `method_checks.json`: **128 queries**, including **48 after-MISS cases**, four same-library fits; reset/reversed-episode order bitwise; deleted camera access raises if attempted; all pass.
- `blind_wrist_checks.json`: four fitted suite/scale compositions; step0, after-MISS and reset force look; a budget-only diagnostic confirms 16-member vision-free continuation. Actual arms retain all gates.
- `arm_checks.json`: **26/26** emitted configs and all **22** fit-using arm references validated; current π0.5 `_miss_steps()` K10/K2 and GR00T match/mismatch guards pass.
- `startup_check.json`: actual fitted PluginRuntime startup logs wrist_only, miss_steps=2, prefix_packing=false.
- Existing selftests before install, after install, and final: π0.5 HIT **80 decisions**, π0.5 all-MISS **77**, GR00T HIT **74**, GR00T all-MISS **78**, four episodes each; **309 final decisions, all PASS**, before/final core metrics equal. See `final_selftests.json` and individual `final_*/selftest_report.json`.
- New wrist mixed selftests: spatial current/big **80 each (41 HIT,39 MISS)**; l10 current/big **342 each (171 HIT,171 MISS)**, four episodes each; **844 decisions**, all PASS, online versus offline mini-store bitwise and served/MISS-policy execution checks pass. See `wrist_*/selftest_report.json`.
- GPU final evidence: `results/pi05.json`, `results/groot_spatial.json`, `results/groot_l10.json`; cost table contains copies plus source hashes. Final π0.5 timing/parity includes the coordinator's actual `return_intermediates=True` path.

## Coordinator next steps / limits

1. Emit the supplied specs into the selected run root using K4's installed emitter. If moving the run root, regenerate `arms_r4.json` first; evidence and new fit paths are explicit, not literal `<RUN>` placeholders. All cache/MISS arms request full weights.
2. Coordinate 500-init paired runs and any remote script/config propagation; none was launched here. Pure-inference baselines have no meaningful library size and belong on both panels.
3. Compare dummy/full and wrist/full at each scale. Wrist changes retrieval and therefore can change success and MISS rates; no success preservation is claimed. Dummy/wrist **MISS policy actions** passed exact parity.
4. Charge the wrist completion cost and keep current-eager versus historical-graph denominators separate. Batch-1 timing does not establish multi-client throughput or multi-batch latency. GR00T flat constants average two prompt shapes; suite-specific measurements are available.
5. Do not enable prefix packing: it failed the mandatory numeric gate. No CUDA-graph override parity is claimed because neither current launcher uses graphs; enabling graphs or changing the backend requires new parity/timing.
6. Batch-four B2 is a ready, fitted candidate; select the final blind budget/method after batch three. K1 owns that method choice and its source; changing kwargs requires matching new prefits.
