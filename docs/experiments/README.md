# docs/experiments/

Experiment run-books: step-by-step pipelines for each experiment family.

| File | Description |
|------|-------------|
| [artifact_layout.md](artifact_layout.md) | Canonical `exp/<experiment>/{config,data,analysis}/` layout rules — where new files go, tracking policy, `.gitignore` exceptions |
| [conductor_tutorial.md](conductor_tutorial.md) | **实验编排教程（重点：如何编写 driver 策略）**：用新框架跑大规模评测 — 编写 `ExperimentStrategy`（`plan` 构造 TaskGraph/Stage/CalibrationArtifact + `on_stage_begin`/`on_stage_complete`/`on_resume`）、复用/自写 `EpisodeRunner`、用 `ConductorDriver` + `WorkerAgent` 启动（按 server 分配 worker / 断点续跑 / 重试 / 监控）、调度语义契约、常见模式（纯 eval / 共享 / 历史 warmup）、测试。取代旧 `main.py --num-workers` / `run_phase` 编排 |
| [cp1_cache.md](cp1_cache.md) | CP1 Cache experiment guide: artifact building, calibration, YAML generation, 3-phase experiment execution, result analysis |
| [temporal_prune.md](temporal_prune.md) | Temporal Prune experiment pipeline |
| [trajectory_deviation.md](trajectory_deviation.md) \[[EN](trajectory_deviation.en.md)\] | Trajectory Deviation experiment runbook: Step 1a→1b→2→3→4 pipeline, parallelism rules, tunables |
| [warm_start_sweep.md](warm_start_sweep.md) \[[EN](warm_start_sweep.en.md)\] | Warm Start sweep runbook: 3 keybuilder × 3 start_t under always-hit + always_warm_start, artifact rebuild, 3-server parallel run, recovery/loss analysis |
| [serving_benchmark.md](serving_benchmark.md) | Serving throughput/latency benchmark runbook (Phase 6 M7 from `logs/concurrent_serving_optimization_plan.log.md`): 5 modes (GPU microbench / sparse→dense / freq sweep / yaml density / batch window) + driver/sweep/collect/plot tooling |

Back to [docs index](../README.md).
