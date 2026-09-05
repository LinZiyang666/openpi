# docs/experiments/

Experiment run-books: step-by-step pipelines for each experiment family.

| File | Description |
|------|-------------|
| [artifact_layout.md](artifact_layout.md) | Canonical `exp/<experiment>/{config,data,analysis}/` layout rules — where new files go, tracking policy, `.gitignore` exceptions |
| [conductor_tutorial.md](conductor_tutorial.md) \[[EN](conductor_tutorial.en.md)\] | **实验编排教程（重点：如何编写 driver 策略）**：用新框架跑大规模评测 — 编写 `ExperimentStrategy`（`plan` 构造 TaskGraph/Stage/CalibrationArtifact + `on_stage_begin`/`on_stage_complete`/`on_resume`）、复用/自写 `EpisodeRunner`、用 `ConductorDriver` + `WorkerAgent` 启动（按 server 分配 worker / 断点续跑 / 重试 / 监控）、调度语义契约、常见模式（纯 eval / 共享 / 历史 warmup）、测试。取代旧 `main.py --num-workers` / `run_phase` 编排 |
| [cp1_cache.md](cp1_cache.md) \[[EN](cp1_cache.en.md)\] | CP1 Cache experiment guide: artifact building, calibration, YAML generation, 3-phase experiment execution, result analysis |
| [temporal_prune.md](temporal_prune.md) \[[EN](temporal_prune.en.md)\] | Temporal Prune experiment pipeline |
| [llm_layer_extract.md](llm_layer_extract.md) \[[EN](llm_layer_extract.en.md)\] | CP1 LLM Layer Extract 端到端 runbook：数据采集 → Step 2 build pkl（带 tokenizer self-check）→ YAML 模板（A/B 两种 reducer）→ run_cache_experiments → 结果分析 → manual parity verify |
| [trajectory_deviation.md](trajectory_deviation.md) \[[EN](trajectory_deviation.en.md)\] | Trajectory Deviation experiment runbook: Step 1a→1b→2→3→4 pipeline, parallelism rules, tunables |
| [warm_start_sweep.md](warm_start_sweep.md) \[[EN](warm_start_sweep.en.md)\] | Warm Start sweep runbook: 3 keybuilder × 3 start_t under always-hit + always_warm_start, artifact rebuild, 3-server parallel run, recovery/loss analysis |
| [serving_benchmark.md](serving_benchmark.md) | Serving throughput/latency benchmark runbook (Phase 6 M7 from `logs/archive/concurrent_serving_optimization_plan.log.md`): 5 modes (GPU microbench / sparse→dense / freq sweep / yaml density / batch window) + driver/sweep/collect/plot tooling |
| [actioncache_baseline.md](actioncache_baseline.md) | ActionCache 式 post-backbone（CP2）基线 runbook：从 CP1 库逐条建 CP2 库（backbone 输出 → 稀疏三值投影 500 维，id/payload/链边不变）→ 验证器 + parity 门 → shadow 表（GST K=1 IR 寻址切点）→ `export_arms` 出臂 yaml（N_hit∈{0,1} = FULL_HIT / WARM@0.1，单阈值、suite 内无 task 过滤）→ `run_gtp.py --checkpoint cp2` 评测 → CP2 开销实测 → 聚合 / 与 `exp/rit_pareto` 前沿的两侧分层 bootstrap 对照 |
| [weighted_sum.md](weighted_sum.md) \[[EN](weighted_sum.en.md)\] | Weighted-sum 两层检索校准 + 权重搜索 runbook：Phase 1 离线为每 (keybuilder, 模态) 选 Layer-1 归一化方法+参数（LOEO query×全库分布，J=mag_sep+β·intra_spread−λ·sat 出 shortlist）；Phase 2 conductor 纯-eval（always_hit）找有用模态+搜权重，含 init-state 防泄漏硬门；prompt_emb 已退出 |

Back to [docs index](../README.md).
