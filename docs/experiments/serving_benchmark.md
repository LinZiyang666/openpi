# Serving Benchmark — Throughput / Latency Runbook

Automated probe for the concurrent-serving optimization layer (plan
[`logs/concurrent_serving_optimization_plan.log.md`](../../logs/concurrent_serving_optimization_plan.log.md) §4.7).
Surfaces the throughput vs latency Pareto curve under different worker
counts, request rates, multi-bundle densities, and batching coordinator
parameters.

## Layout (per [`artifact_layout.md`](artifact_layout.md))

```
exp/serving_benchmark/
├── driver.py            # WebSocket client driver
├── sweep.py             # Cell-grid runner (consumes config/*.yaml)
├── collect.py           # GPU / CPU sampling + post-sweep collation
├── plot.py              # Pareto / batch-size figure helpers
├── gpu_microbench.py    # Mode 0 — direct model.sample_actions sweep
├── config/*.yaml       # sweep grids (one per mode)
├── data/<run_id>/       # CSV + log artifacts
└── analysis/<run_id>/   # rendered PNGs
```

## Modes

| Mode | Config | Purpose |
|------|--------|---------|
| **0** GPU direct microbench | `gpu_microbench.yaml` | Theoretical upper bound — bypasses server / wrappers, calls `model.sample_actions` for batch_size ∈ {1,2,4,8,16,32}. Audit report §B.3 #4 alignment. |
| **1** sparse → dense workers | `sparse_to_dense.yaml` | Per-worker rate fixed; sweep worker count to find concurrency ceiling. |
| **2** request frequency sweep | `freq_sweep.yaml` | Worker count fixed at 8; sweep per-worker request rate to surface back-pressure onset. |
| **3** multi-bundle density | `yaml_density.yaml` | Verify M2 + M3 pool memory savings under K-bundle load. |
| **4** batch window tuning | `batch_window.yaml` | Sweep `(max_batch_size, max_wait_ms)` to pin the BatchingCoordinator default. |

## Driver workflow (Modes 1-4)

1. Start the optimized server (Phase 5 makes `--concurrent` the default):
   ```bash
   python scripts/serve_policy.py --env LIBERO --port 8000 \
       --cache_config exp/.../some.yaml
   ```
2. In a second shell, start metrics sampling (GPU / CPU):
   ```bash
   python -m exp.serving_benchmark.collect sample --run-id 2026-05-23_baseline
   ```
3. In a third shell, kick off the sweep:
   ```bash
   python -m exp.serving_benchmark.sweep \
       --config exp/serving_benchmark/config/sparse_to_dense.yaml \
       --run-id 2026-05-23_baseline
   ```
4. After the sweep finishes, stop the sampler (`touch
   exp/serving_benchmark/data/2026-05-23_baseline/.sb_stop`) and collate:
   ```bash
   python -m exp.serving_benchmark.collect collate --run-id 2026-05-23_baseline
   ```
5. Render figures:
   ```bash
   python -m exp.serving_benchmark.plot --run-id 2026-05-23_baseline --mode pareto
   ```

## Driver workflow (Mode 0)

Mode 0 is server-less; run directly against a checkpoint:
```bash
python -m exp.serving_benchmark.gpu_microbench \
    --checkpoint <path-to-trained-checkpoint> \
    --config-name pi05_libero \
    --batch-sizes 1,2,4,8,16,32 \
    --output exp/serving_benchmark/data/<run_id>/gpu_microbench.csv
python -m exp.serving_benchmark.plot --run-id <run_id> --mode gpu_microbench
```

## Outputs

* `data/<run_id>/sweep_summary.csv` — per-cell aggregate (throughput, p50/95/99 latency)
* `data/<run_id>/<cell_id>/latency.csv` — per-request rows
* `data/<run_id>/gpu.log` — `nvidia-smi dmon -s u` raw output (1 Hz)
* `data/<run_id>/cpu.csv` — per-core utilisation (1 Hz)
* `data/<run_id>/cell_metrics.csv` — post-`collate` merged table
* `analysis/<run_id>/pareto.png` — throughput vs latency p95 scatter
* `analysis/<run_id>/gpu_microbench.png` — Mode 0 batch-size curves

## Notes

* `driver.py` reproduces a LIBERO-shaped dummy obs (zeros). Targeting a
  non-LIBERO checkpoint requires editing `_make_dummy_obs` to match the
  expected schema.
* The driver does **not** mutate server-side BatchingCoordinator
  parameters between cells in Mode 4 — change them on the server (or
  restart the server with new defaults) between cells and re-run the
  same sweep with a fresh `--run-id`.
* The benchmark is intentionally light on dependencies (no torch, no
  openpi heavy imports in `driver.py` / `collect.py`) so the sampler
  process does not steal CPU from the real workload.
