# R12HalfEverywhere handback

Implemented frozen DESIGN_HALF_EVERYWHERE §1–§8. Both tasks retain the deployed stack. Wrist tasks use ablation `wrist_sub`, AWM post-MISS continuity, wrist LOEO heads and per-task `scale_sub`, unchanged H2 features/coefficients/thresholds, and request-owned completion. Cold π0.5 looks encode only the wrist, including step zero; masked dummy slots require no tower call. H2 calls complete the agentview before committing retrieval/guard/knob once. Periodic planned calls acquire both cameras. Mode counts and compute-share receipts are checked and collected.

Files/functions:

- `closed_loop/half_everywhere.py`: `R12HalfEverywhere`, `_SubCache`, `_TaskJudge`; independent per-connection history/anchor bookkeeping.
- `closed_loop/half_everywhere_{config,cameras,ledger}.py`: SHA-bound tables, cold single-image acquisition/completion, receipt and per-mode accounting.
- `half/build.py::build_cell,fit_wrist`: table-independent B-only heads/scales; `fit_path`/`library_boundary`, complete LOEO donor exclusion and fold evidence.
- `half/knob.py::calibrate`: frozen R11 replay with table prices; `half/prepare.py::prepare`: tables, gaps, hashes, B emission, A staging and zero-wrist skips. `install_test_a` is coordinator-only.
- Atomic shared edits: `closed_loop/plugin.py`, `lookhalf_config.py`, `ops/collect.py`, `ops/h100/assets.py`, `ops/h100/node.py`. Existing H2/ablation/deployed artifacts were preserved; source hashes are in `VERIFICATION.json`.

CPU verification: **331 tests passed**, zero failures. `test_half.py` has **59**: (a) deployed byte equivalence 6; (b) disabled wrist ablation chunks/tails 5; (c) identical-input H2 equivalence 5; (d) R11 gaps 3; (e) acquisition/engagement/continuity/receipts 11; (f) LOEO/fold/read audits 5; (g) exact local/H100 argv 20 cases / 24 arm commands; (h) binding refusal and standalone remote validation 2; emission/rebinding tests 2. Existing regression counts: cache ablation 71, pipeline 50, test-A definitions 26, recording 25, look-half 14, H2 13, devset 38, camera dispatch 8, bounded collection 27. Evidence: `CPU_HALF.xml`, `CPU_REGRESSION.xml`, `ARGV_VERIFICATION.json`, `VERIFICATION.json`. Transports were mocked, plans and legacy definition snapshots stayed in `/tmp`; no servers, smokes, real sync/chains or remote operations ran. The unchanged collector splits GR00T L10 archives at the **700,000,000-byte** bound.

All five cells have ten fitted wrist heads and ten scales. The dry emission is explicitly **provisional** under `half/provisional/`: 5 B roots / 7 arms and 5 staged A roots. π0.5 Spatial-50 was skipped (no wrist tasks); periodic gap **1.0164**. All-both calibration exactly reproduces **2.6125 / 1.0657 / .4386**. Final production roots have not been emitted; the final camera JSON is the only pending input. No further fitting depends on it.

From `/home/weiland/projects/openpi`, this **single command** consumes the final JSON, writes per-cell tables, recalibrates the gap, rebinds hashes, emits `half/pilots/r12_half_<cell>` and stages `half/test_a_staged/r12_test_half_<cell>` (plus periodic). Zero-wrist cells are reported and skipped. Existing emitted destinations are refused.

```bash
taskset -c 22-37,66-81 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m exp.offline_search.rounds.r12.half.prepare --camera-choice exp/offline_search/rounds/r12/opus/out/design/camera_choice_screen_camdev.json
```

Runtime push: **H100** needs the four new `half_everywhere*.py` runtime files, updated `plugin.py`, `lookhalf_config.py`, and `ops/h100/node.py` (also the standalone helper at `/data/oscl_h100/node.py`); retain the existing R12 guard/look-half/stage override and R10/R11 dependencies. Updated `assets.py` and `collect.py` run on the local coordinator. **timan107/timan108** need their existing audited `execution_worker.py`, `worker_pool.py`, `devset.py`, `ops/remote/run_gtp_subset.py`, and `ops/h100/run_arm.sh`; these were not changed. Refresh each selected island's helper/snapshot with normal `setup --worker-only` if needed. `setup` with `SERVER_HOST=h100` installs both the H100 runtime and selected worker snapshot. Normal sync transfers bound tables, frozen artifacts, payloads, configs and pool contracts; no fitting corpus is needed online.

Exact B commands, for the coordinator after the runtime/fleet is ready (GR00T defaults to half + same-batch deployed off):

```bash
S=exp/offline_search/rounds/r12/half/coordinator_commands.sh
for action in plan sync chain; do
  for cell in pi05_l10_50 pi05_spatial_50 pi05_spatial_200 pi05_l10_50_periodic_ir32; do
    SERVER_HOST=local WORKER_HOST=timan107 PORTS=23170,23171,23172,23173 LOCAL_SERVER_COUNT=4 WPS=16 bash "$S" "$action" B "$cell" half
  done
  for cell in groot_spatial_50 groot_l10_50; do
    SERVER_HOST=h100 WORKER_HOST=timan108 PORTS=23370,23371,23372,23373 WPS=8 bash "$S" "$action" B "$cell" half off
  done
done
```

A remains staged, recording off, exact R11 `eval500.json` bytes, 500 episodes/arm. The coordinator installs later with `python -m exp.offline_search.rounds.r12.half.prepare --install-test-a`. For each cell that passes the frozen B stop rules, use the same explicit host/port environment above and:

```bash
bash "$S" plan A "$cell" half
bash "$S" sync A "$cell" half
bash "$S" chain A "$cell" half
```

Reproduce CPU checks with the affinity/environment prefix from the final-table command and `-m pytest -q -p no:cacheprovider exp/offline_search/rounds/r12/half/test_half.py`, then `-m exp.offline_search.rounds.r12.half.cpu_suite`.
