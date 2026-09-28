This directory is an isolated retrieval experiment. It does not install a serving plugin or modify a server. Start with `REPORT.md`; `audit.json` distinguishes completed measurements from the strict numerical-parity gate.

- `gpu_awm.py`: fixed-shape device buffers, batched AWM, guard-only MixedJudge confidence/features, in-place prepared-row append, compact verdict packet.
- `stage1_adapter.py`: capture-safe stage 1 with the same image towers, embeddings and output fields; replaces stock's host-list zero-mask construction.
- `benchmark.py`: real QueryView parity, batch-1/batch-8 eager/graph timing, append-after-capture check.
- `bench_stage1.py`: actual checkpoint, images, stock key builder and CPU method versus the isolated GPU path; stage-only, separate and combined graphs; paired increments and changed-input replay checks.
- `common.py`: exact pickle loading, query selection, GPU admission and memory watchdog.
- `run_jobs.py`: one GPU child at a time, fresh processes, headroom scheduling, retained rejected attempts. It waits on GPU headroom without holding a GPU context itself.
- `numerical_probe.py`: CPU-only float32/float64 early-distance and tie diagnosis.
- `report.py`, `audit.py`: aggregate only successful runs started after `FINAL_CODE_TIME.txt`, and check their implementation hashes. Earlier development evidence remains beside the final evidence and is excluded from final tables.

Run from `/home/weiland/projects/openpi`. The serial scheduler records every expanded child command in `final_commands.json`:

```bash
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r04/k9_gpu_retrieval/run_jobs.py
```

For a single module measurement (the script checks nvidia-smi before GPU work):

```bash
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src TMPDIR=/tmp/k9_scratch JAX_PLATFORMS=cpu .venv/bin/python exp/offline_search/rounds/r04/k9_gpu_retrieval/benchmark.py --config pi05_l10_500_MixedJudge --precision float32 --run manual
```

`GPUAWM.from_pickle(path, library=None, precision="float32")` returns CPU buffers; move the module to the admitted GPU before capture. Pure AWM needs the matching read-only `LibraryView` for exact progress/length/next metadata. MixedJudge already carries those tables. `precision="float64"` promotes runtime projection/distance arithmetic; the fitted artifacts remain the original float32 fits, and action accumulation remains float32.

`forward` arguments are, in order:

| Input | Shape / type |
| --- | --- |
| pooled_key_v0, pooled_key_v1 | B×32768; already on GPU; preserve deployed pooling rounding, then cast to f32 |
| rs | B×32; valid first 8 coordinates; CPU-equivalent f32 cast is inside the module |
| task_id, step, prev_hit | B int64 each; task 0–9, prev_hit 0=MISS, 1=HIT, -1=unknown |
| prev_a_exec | B×10×32; actual previous executed chunk, including unexecuted tail |
| prev_key_v0, prev_key_v1 | B×32768; previous real vision keys |
| prev_rs | B×32; previous state |
| stuck_prev | B int64; previous stuck count, zero at episode reset |

Allocate one input set per graph slot and update values with `copy_`; do not replace addresses. `benchmark.inputs` and `benchmark.capture` are concrete examples. `forward` returns GPU tensors including action, all 16 row IDs, scores, kernel weights, AWM confidence/extras, and (for MixedJudge) V7 confidence/features and the next stuck count. `verdict_packet` packs the small host-facing fields; it does not make the final guard/HIT decision. Episode lifecycle, progress/burst memos, global quantile state and execution acknowledgement are caller responsibilities.

`append_prepared(task, fields)` is maintenance outside graph execution. All names in `module.row_fields` are required, with N leading rows and the correct trailing shapes. Prepare new rows with the frozen fit, use ascending new IDs, and serialize maintenance against every graph reading those buffers. The method uses `copy_` into spare capacity and publishes validity last. It rejects exhaustion; growing/replacing buffers requires recapture.

Measurements exclude serving installation, RPC, logging, native shadow search and simulation. See the report for numerical mismatches and the design-only integration plan.
