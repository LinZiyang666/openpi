# Executed CPU-only workflow

All commands were run from `/home/weiland/projects/openpi` with this prefix:

```sh
taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= .venv/bin/python
```

Modules were executed in this order, with output redirected into the correspondingly named `.log` in this directory:

```text
-m exp.offline_search.rounds.r11.astra.experiment --concurrency 4
-m exp.offline_search.rounds.r11.astra.quicklook
-m exp.offline_search.rounds.r11.astra.analyze --concurrency 4
-m exp.offline_search.rounds.r11.astra.selfcheck
-m exp.offline_search.rounds.r11.astra.crosscheck
-m exp.offline_search.rounds.r11.astra.inspect_results
-m exp.offline_search.rounds.r11.astra.supplement
-m exp.offline_search.rounds.r11.astra.freeze
-m exp.offline_search.rounds.r11.astra.selfcheck
-m exp.offline_search.rounds.r11.astra.crosscheck
-m exp.offline_search.rounds.r11.astra.write_docs
-m exp.offline_search.rounds.r11.astra.finalize
```

The repeated self-check verifies the documented endpoint clarification. The repeated cross-check binds the final available shared IR-model version; neither refits a knob. `freeze.py` intentionally refuses to overwrite `freeze.json`. Its first attempted execution encountered a duplicate dictionary-key construction before emitting any frozen payload; that local reporting bug was corrected before the successful timestamped freeze. All successful scientific computations and data checks are retained.

`experiment.py` regenerates the held-out scores and final pooled predictors. `analyze.py` generates cost curves and calibrations. `supplement.py` checks robust risk proxies. `freeze.py` emits the proposed arm grid and prospective forecasts. `write_docs.py` emits every numeric REPORT statement from those recorded payloads. `finalize.py` checks the deliverables and creates the final manifest.

The scripts deliberately install `boundary.py` to reject every `os_closed_loop` read (stricter than the brief's permitted exceptions), all replay-directory reads, non-library store reads, network activity and writes outside this astra directory. The fixed R10 PCA, fitted cache and CV artifacts are B-library-derived inputs, recorded by hash. The raw library arrays are read-only. No GPU, policy forward, simulator, server, model worker, git command or deployment was used. Thread pools only parallelize local NumPy analysis under the one pinned CPU process.

The prediction freeze binds the original numerical payloads. Re-running a producer can change provenance timestamps and thus hashes, even when its numbers are deterministic; retain this frozen package before intentionally regenerating it. The checks and document readers can be rerun without changing the frozen forecasts.
