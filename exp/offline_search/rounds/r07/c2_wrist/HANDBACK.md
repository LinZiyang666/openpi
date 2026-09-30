# C2 — SW and SF+SW, π0.5

Implementation, CPU verification, eight arm specs and eight fitted arms are delivered. **Real π0.5 GPU parity remains a coordinator prerequisite**: local `nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader` returned exit 9, “couldn't communicate with the NVIDIA driver.” No GPU process was started. `parity_pi05.py` checks admission again immediately before loading one model and stops unless a GPU has at least 12288 MiB free. It performs twelve recorded-observation comparisons, without serving, simulation or timing benchmarks.

GR00T SW is unavailable: there is no validated one-camera encoder path. The new plugin flag is rejected for GR00T, and enabled StageWrist refuses a GR00T fit. No GR00T encoder implementation was built.

## Files and installs

All new implementation and delivery files are in this directory. `FILES.sha256` records their hashes and the two shared files. `FITS.sha256` records the sixteen original bank/arm fit files under `/tmp/r7_C2/fits/`. `VERIFICATION.json` preserves the replay and integration counts.

Shared files were developed in `/tmp/r7_C2/dev/`, tested there, then installed with a temporary file in `closed_loop/` and one `mv` per revision. Final installed versions:

| File | Final install time (CDT) | SHA256 |
|---|---|---|
| `closed_loop/plugin.py` | 2026-09-30 01:37:21.232866775 -05:00 | `45a807a6b9ab6479d693c0ae95be125dec05547dee2ff228d1c68d5270b3e308` |
| `closed_loop/stage_overrides.py` | 2026-09-30 01:23:48.396190055 -05:00 | `e65e8dc0630519cbb86ea953db8b3a15ff47fe93c7cd0e2b5e98722de33ce226` |

Original shared-file snapshots and install-time records remain in `/tmp/r7_C2/`. No other coder's files, earlier-round files, source, launchers or operations tools were edited.

## Method and server contract

Method: `exp.offline_search.rounds.r07.c2_wrist.method:StageWrist`.

SW kwargs: `enabled=true`, `base_spec=exp.offline_search.rounds.r04.k1_blind.blind_awm:BlindAWM`, `base_kwargs={lib: current|big, kref: 5|8, serving: anchor_tail, budget: 1, gates: budget_only}`, plus `base_fit`, `wrist_fit`, `stage_fit` as specified in `arms_in.json`. Dense `lib="big"` is the existing constructor's accepted spelling; it resolves to `bpool_cs` for π0.5. The exact deployed A artifacts are read from R6's provenance catalog, not refitted.

SF+SW uses the same method with `base_spec=exp.offline_search.rounds.r07.c1_follow.methods:StageFollow` and adds `extend_blocks=1`, `stage_gate=true`, `state_valve=true`, `stages_path=<RUN>/fits/stages_pi05_<cell>.pkl` to `base_kwargs`. The validated C1 subclass inherits the frozen A retrieval arrays, then runs `finish_follow_fit`. SW temporarily substitutes the independent wrist metric fields during that subclass's ordinary query; SF keeps ownership of its anchor, extension plan, state valve and blind lifecycle. Full metric fields are restored in a `finally` block. Policy chunks are never extended.

Required plugin flags:

```text
--os-root /home/weiland/trace_runs/offline_search_store
--os-no-shadow-native --os-tokens off --os-blind --os-request-cameras
--os-fit-artifact <RUN>/fits/<arm>.pkl
```

Do not combine with a static `--os-stage1-mode wrist_only|dummy_cached`; the new flag installs the per-request override. Full-model loading is specified for every arm so a future MISS can complete the missing camera and run the ordinary policy. These A/SF arms have no call judge by default. The usual mixed judge can still produce a wrist-origin MISS.

`next_camera_mode` is a method-owned, per-connection declaration; `set_camera_mode` receives the selected mode before stage 1. Episode start, unresolved lifecycle/audit, periodic/capped forced LOOKs and method hard-stage vetoes use full vision. Query and blind decisions update the next plan. Eligibility uses C1's StageTable: every member, including zero-weight members, has a real same-episode chain; its learned mode agrees with the issued gripper command through the next anchor's head; no event-near row occurs in that interval. The successful-library p95 displacement valve also keeps a deviating state hard. There are no task/suite names, dimensions or five-control offsets in this decision logic; geometry comes from the library manifest. The encoder adapter explicitly supports only the already validated canonical π0.5 LIBERO layout.

`CameraRequest` carries mode and work audit with the observation, before the coordinator stacks it. Dynamic stage 1 encodes each request at B=1 to avoid changing bf16 GEMM shape and therefore keys. The full branch calls stock stage 1 and seeds the masked-dummy cache from its existing output; a later wrist branch runs only the wrist image tower. Missing base images and audit references travel with deferred stage outputs through split, mixed/reordered stack, and `.to()`. Stage 2 and capture complete each missing row at B=1, then call the original policy stage. The shared model never stores the next camera mode or pending request images. Arrays of fitted data remain shared/read-only; connection histories and plans are isolated. Arm `server_env` explicitly sets `BATCHING_MAX_BATCH_SIZE=1`; no latency advantage from batching is claimed.

Absent `--os-request-cameras`, no new encoder hook is installed and no new decision/startup fields are emitted. `enabled=false` delegates query and blind decisions to the original base, including extras.

## Arm specs and prefits

`arms_in.json` is the exact eight-row `emit_arms` input with `<RUN>` placeholders. SW is the four-cell profile variant; SW and SF+SW are the planned four-cell evaluation variants, subject to coordinator pruning. Each row preserves five-control client requests and today's A retrieval settings.

| Cell | Arms | Candidate / kref | Successful valve episodes / supported anchors | Row p95 radius |
|---|---|---|---|---|
| π0.5 L10-50 | `r7_sw_pi05_l10_50`, `r7_sf_sw_pi05_l10_50` | current / 5 | 50 / 1172 | 0.3987586889618458 |
| π0.5 L10-500 | `r7_sw_pi05_l10_500`, `r7_sf_sw_pi05_l10_500` | bpool_cs / 8 | 436 / 10569 | 0.26430627971008475 |
| π0.5 Spatial-50 | `r7_sw_pi05_sp_50`, `r7_sf_sw_pi05_sp_50` | current / 5 | 49 / 390 | 0.34230940510579827 |
| π0.5 Spatial-500 | `r7_sw_pi05_sp_500`, `r7_sf_sw_pi05_sp_500` | bpool_cs / 8 | 487 / 4521 | 0.2726420403719471 |

Every task's wrist Wf is 72×72, independently action-supervised in `[wrist PCA-64, valid state]`; full A Wf remains 136×136. Same PCA recipe, library, k=16, kref, ridge/nearest-action-pair recipe, fresh-MISS continuity and kernel. No submatrix of the full fitted metric is used. C1 owns stage calibration; its frozen-fit candidate-LOEO limitation applies here as well.

All Python commands run from `/home/weiland/projects/openpi` with this exact CPU prefix (shell array used only to shorten the commands below):

```bash
PY=(taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
"${PY[@]}" -m exp.offline_search.rounds.r07.c2_wrist.specs
"${PY[@]}" -m exp.offline_search.rounds.r07.c2_wrist.prefit
```

The executed prefit command above produced all eight arm fits and both banks per cell. For independently rebuilding each cell, these exact commands each produce **both** its SW and SF+SW arms:

```bash
"${PY[@]}" -m exp.offline_search.rounds.r07.c2_wrist.prefit --cells l10_50
"${PY[@]}" -m exp.offline_search.rounds.r07.c2_wrist.prefit --cells l10_500
"${PY[@]}" -m exp.offline_search.rounds.r07.c2_wrist.prefit --cells sp_50
"${PY[@]}" -m exp.offline_search.rounds.r07.c2_wrist.prefit --cells sp_500
```

Outputs: `/tmp/r7_C2/fits/{r7_sw_pi05_<cell>,r7_sf_sw_pi05_<cell>,wrist_pi05_<cell>,stages_pi05_<cell>}.pkl`. `prefit_report.json` records sizes, calibration counts and timings. Fits are scratch-only. Before coordinator copy, repackage the exact arm metadata for the concrete destination; copying the original arm pickle alone would fail the plugin's kwargs check:

```bash
"${PY[@]}" -m exp.offline_search.rounds.r07.c2_wrist.relocate --target-run /home/weiland/trace_runs/os_closed_loop/r07_wrist
```

Copy the sixteen `.pkl` files in `/tmp/r7_C2/relocated/` to that run's `fits/`, and use its `arms_in.json` with `emit_arms`. Eight relocated arms were successfully loaded through PluginRuntime using target kwargs. The source A artifacts remain unchanged and must remain available for future refitting; a loaded serving artifact does not reopen them.

## Tests and exact counts

Commands executed, after applying the CPU prefix above:

```bash
bash exp/offline_search/rounds/r07/c2_wrist/test_commands.sh after
"${PY[@]}" -m unittest exp.offline_search.rounds.r07.c2_wrist.test_dispatch exp.offline_search.rounds.r07.c2_wrist.test_method
"${PY[@]}" /tmp/r7_C2/existing_override_checks.py
"${PY[@]}" -m exp.offline_search.rounds.r07.c2_wrist.replay
"${PY[@]}" -m exp.offline_search.rounds.r07.c2_wrist.verify
"${PY[@]}" -m exp.offline_search.rounds.r07.c2_wrist.specs --run-root /tmp/r7_C2 --out /tmp/r7_C2/arms_resolved.json
"${PY[@]}" -m exp.offline_search.closed_loop.ops.emit_arms --run-root /tmp/r7_C2/emitted --spec /tmp/r7_C2/arms_resolved.json
"${PY[@]}" -m exp.offline_search.rounds.r07.c2_wrist.check_specs
```

`test_commands.sh` contains the full exact existing selftest invocations. The corresponding four before-install runs used the same arguments and `before_<model>_<mode>` output directories while the originals were installed. The legacy override script is a copy of R4 `check_overrides.py` whose import points to the installed shared module and whose output directory is relocated to `/tmp/r7_C2/`; no earlier-round file was modified.

- **15/15 new CPU tests pass:** real key-builder wrist key equality; all completed Stage1Output fields; omission of base tower work; request-local mode declaration before forward; mandatory first/forced full looks; mixed/reordered split/stack/`.to()`; actual completion audits; unknown/terminal/transition/state/issued-command vetoes; independent 72-D fits; all five owner-cost cases and refusal of a missing completion.
- **11/11 existing R4 override checks pass.**
- **Four before/after existing plugin selftests pass:** π0.5 cache 80, GR00T cache 74, π0.5 all-MISS 77, GR00T all-MISS 78 decisions; four episodes each. All 309 actions, semantic decision rows, verdicts and non-timing archive fields match exactly across installs. Wall-clock timestamps and measured execution durations are excluded. 816 archive fields compared exactly.
- **Two existing blind plugin tests pass:** 48 decisions each, 32 vision, 16 blind, 8 MISS; two interleaved connections and four episodes each. Each test rejects four duplicates, four invalid blind candidates, four failed output preflights and four partial-control requests without corrupting histories.
- **Disabled method identity passes on 5173 recorded decisions / 120 episodes / eight cell×dataset groups** from B-val and P3 A_r0, including actions, blind/LOOK decisions, unchanged judge inputs and extras. Enabled SW/SF+SW wrist retrieval equals the independently fitted standalone wrist retriever on **1235 wrist anchors** (top-k, scores, confidence and action).
- **Eight arm configs emit successfully; eight public arm bank loads and eight relocated serving artifacts validate.** GR00T flag refusal is tested. CPU parsing also checks every owned Python script, including the pending GPU parity script.

Recorded replay holds original states/executed histories fixed. Its routing/cost counts are diagnostics, not closed-loop SR or measured GPU savings. No B-val or P3 outcomes fit the wrist metric or stage thresholds. No servers, workers, chains, tmux, ports, remote hosts, git commands, or training-process operations were run.

## Telemetry and cost

Result/blind extras: `os_sw_next_camera` (0 full, 1 wrist), `os_sw_reason`, `os_sw_mode_mass`, `os_sw_unanimous`, `os_sw_event_mass`, `os_sw_unknown_mass`, `os_sw_rows_to_event`, `os_sw_valve`, `os_sw_radius`, `os_sw_target_blocks`.

Reason codes: 0 admitted; 1 absent/invalid anchor or lifecycle; 2 unknown/nonunanimous/current-command mismatch; 3 missing successor; 4 mode change or event neighbourhood; 5 unsupported/nonfinite/outside state valve; 6 C1 hard LOOK veto. First failure takes precedence.

Plugin-computed extras: `os_sw_camera`, `os_sw_look_cost`, `os_sw_completion_cost`, `os_sw_call_cost`, `os_sw_decision_cost`. Per-decision fields: `camera_mode` (`full|wrist_only|blind`), `stage1_mode` (actual vision mode, full placeholder when blind), `camera_completion`, `camera_completion_calls`, `camera_stage1_calls`, `owner_cost`, `camera_cost_assumption`. Startup: `request_cameras`, `stage1_mode=per_request`, `camera_cost_basis`.

Essential composed SF fields are placed ahead of inherited diagnostics: `os_sf_granted`, `os_sf_structural`, `os_sf_stage_ok`, `os_sf_cap`, `os_sf_delta`, `os_sf_radius`, `os_sf_valve_fire`, `os_sf_look`. Actual costs plus those eight and the ten SW decision scalars fit the 24-scalar pure-cache log cap. Blind extras also retain inherited diagnostics. Library fits/large arrays are not decision scalars.

Owner cost per request = `.152 full_look + .055198 wrist_look + .049890 actual_completion_calls + .848 policy_call`; blind costs zero. Wrist/completion prices are explicitly the R4 proportional-latency **assumption**. The request audit, not the next-mode forecast, counts completion tower calls. Failed requests are rejected by the strict cost audit rather than assigned a complete policy cost.

The default operations table uses another eager latency basis. **Use `owner_costs.json` explicitly** when reporting these arms, and audit the actual camera counters:

```bash
"${PY[@]}" -m exp.offline_search.rounds.r07.c2_wrist.cost --log-dir <RUN>/runs/<arm> --out /tmp/r7_C2/<arm>_owner_cost.json
"${PY[@]}" -m exp.offline_search.closed_loop.ops.kpi --run-root <RUN> <arm> --cost-table exp/offline_search/rounds/r07/c2_wrist/owner_costs.json
```

IR from these tools is per five nominal controls; terminal requests can execute fewer. Retain the existing actual-control ledger for that denominator. There is no new latency measurement or third-person-only/GR00T camera price.

## Coordinator next

First run the guarded real parity script, with the CPU prefix but **without** empty CUDA_VISIBLE_DEVICES:

```bash
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r07.c2_wrist.parity_pi05 --out /tmp/r7_C2/gpu_parity.json
```

Required PASS: full dispatch equals stock full; wrist pooled key equals full's wrist key; all completed policy input fields equal full; unmodified K10 same-noise policy outputs equal full, across twelve recorded observations and deferred split/rebatch/`.to()` handling. A failure stops SW profiling. CPU stand-ins do not establish numerical GPU parity.

Then relocate/copy fits, emit the eight specs, and run a coordinator-owned non-test smoke with batch size 1. Include a mixed-judge smoke that obtains a wrist-origin MISS and requires completion_calls=1 before the policy action; full-origin MISS requires zero completions. Audit cost.py, resets/interleaving, camera modes, stage/valve reasons and exact executed-control bookkeeping. Profile SW × four π0.5 cells on the selected non-test B-val initial states (at most two per task), before selecting evaluation arms. SF+SW E=1 is ready for the later frozen evaluation list; changing SF's cap requires matching arm kwargs/refit metadata. No rollout claim is made by this hand-back.
