# docs/

Project architecture and usage guides. For implementation logs, see [`logs/README.md`](../logs/README.md).

> **AGENT: READ FIRST** — Project rules are in [`WORKING_AGREEMENT.md`](../WORKING_AGREEMENT.md) (workflow §2, documentation §4, subsystem rules §8). This file is a navigation index only — it does not define rules.

Bilingual companions use `.en.md` (English) or `.zh.md` (Chinese); they are folded under the primary entry in each subdirectory's index and do not occupy their own row here.

---

## Directory Layout

```
docs/
├── reference/        # Project reference (structure, architecture, deployment)
├── architecture/     # Cache system specs and workflow diagrams
├── cache/            # Cache system user guides (tutorial, migration, components)
├── experiments/      # Experiment run-books (CP1, temporal prune, trajectory deviation)
├── data_collection/  # Data collection (HDF5 schema, --collect flag)
├── deployment/       # Deployment / simulator setup (ALOHA, LIBERO)
├── papers/           # Related-work bibliographies (inference cache literature, etc.)
└── upstream/         # Original upstream openpi docs (remote inference, docker, norm stats)
```

Each subdirectory has its own `README.md` index listing the docs inside.

## Reading Paths

### First time in this fork

1. Read [`../WORKING_AGREEMENT.md`](../WORKING_AGREEMENT.md) for project rules and development workflow.
2. Read [reference/openpi.md](reference/openpi.md) for project structure, model architecture, code paths, and deployment options.

### Working on the inference pipeline or cache system

1. [reference/openpi.md](reference/openpi.md) — locate the PyTorch model and policy code paths.
2. [architecture/cache_system.md](architecture/cache_system.md) — CP1/CP2/CP3 design and integration boundary.
3. [cache/tutorial.md](cache/tutorial.md) — hands-on component guide, YAML config, testing patterns.
4. [`../logs/archive/step1.log`](../logs/archive/step1.log) and [`../logs/archive/step2.log`](../logs/archive/step2.log) — staged API and timing implementation history.

### Working on data collection

1. [data_collection/guide.md](data_collection/guide.md) — user-facing workflow and HDF5 layout.
2. [`../logs/archive/step3_data_collection.log`](../logs/archive/step3_data_collection.log) — hook points, wrapper ordering, compatibility decisions.

### Working on cache experiments

1. [experiments/cp1_cache.md](experiments/cp1_cache.md) — full experiment pipeline (artifact building, YAML generation, experiment execution, result analysis).
2. [`../logs/archive/cache_experiment_plan.log.md`](../logs/archive/cache_experiment_plan.log.md) — experiment design rationale.

### Working on training, deployment, or environment setup

1. [reference/openpi.md](reference/openpi.md) — configs, transforms, and deployment modes.
2. Then the relevant deployment guide under [deployment/](deployment/) or [upstream/](upstream/).

---

## Section Indexes

### [reference/](reference/)

| File | Description |
|------|-------------|
| [reference/openpi.md](reference/openpi.md) | Project structure, model architecture, Pi0 vs Pi0.5 differences, code paths, training configs, deployment, hardware |

### [architecture/](architecture/)

| File | Description |
|------|-------------|
| [architecture/cache_system.md](architecture/cache_system.md) \[[ZH](architecture/cache_system.zh.md)\] | Cache system spec: 3-stage pipeline, CP1/CP2/CP3 checkpoints, interceptor pattern, component design; §5.6 SimilarityJudge purity contract refinement (no write to storage; read-only via PayloadView allowed); §5.10 Search Session — Cross-Step Score Memo (opt-in per-episode score memoization, mutation contract, lock-free derivation); §5.11 PayloadView (read-only Judge-side facade + ForkPolicy); §5.12 Verdict Factor System (OnlineExtractor / OfflineWriter protocols + factor registry + CompositeJudge pipeline + LibraryStats + payload.factors schema + `all_nan_fallback` bootstrap + ⚠ multi-worker normalizer cold-start amplification operational gotcha — pre-launch self-check `per_worker_verdict_budget × (1 - max_factor_nan_rate) > window_size`); §5.13 Wire-Level Observability + Warmup Preload Protocol (`__hit_meta__` response field, `fetch_dump` / `preload_normalizer_buffer` / `unload_warmup_buffer` ctrls + `load_cache_config.yaml_id`, `CurrentCacheBundle.yaml_id`, `WarmupPool` lifecycle, server-owned warmup dump root with `.resolve()` allowlist, relation to §5.6 purity + §5.12 cold-start amplification); §5.14 Gate-Research Per-Step Collection (`__collect_meta__` sibling + `CheckResult.searched`, opt-in inline collector — wire byte-identical when off; robot_state default, vision standalone-only, raw out of scope; client-side ndarray→list codec; two row kinds incl. `episode_summary` provenance; canonical `compute_global_episode_id` via `EpisodeTask.extra["num_trials_per_task"]`; `summarize_gate_log` verdict-only tally); §5.5 gate roster refresh + G0a additive lifecycle hooks (`record_verdict` verdict-feedback fed by `check()` on every return path + `on_episode_start(task_key)` broadcast, both guarded/signature-filtered so legacy gates and the wire are unchanged) + `score_hysteresis` server-side N1 gate (+ Stage 3b N4 V2 injection branch via optional `L`; `L=None`=N1 latency profile, `L: 6`=N4 SR profile, reuses the `record_verdict` hit_type hook, zero orchestrator/wire change). Chinese companion frozen at 2026-04-03 |
| [architecture/cache_workflow.md](architecture/cache_workflow.md) | End-to-end workflow diagrams: startup, single inference with CP1/CP3, episode lifecycle, storage layer, YAML mapping, design principles |
| [architecture/experiment_conductor.md](architecture/experiment_conductor.md) \[[EN](architecture/experiment_conductor.en.md)\] | 两层实验编排框架（worker/agent/driver + 机制/策略分离）：episode 级中央队列 + worker pull 消除等待泡沫；yaml 亲和（warmup 放松 / eval 收紧）/ 永不空转 / warmup→eval barrier；账本断点续跑 + server 自愈；重试分类 / agent 监督 / 聚合监控；按 server 分配 worker；server 端点支持 --replicas 公共端口（router 已对 fetch_dump fan-out + 拼接）/ 多独立端点、server 协议不动。设计见 [`logs/archive/client_conductor_two_layer_refactor.log.md`](../logs/archive/client_conductor_two_layer_refactor.log.md) |

### [cache/](cache/)

| File | Description |
|------|-------------|
| [cache/tutorial.md](cache/tutorial.md) | Complete tutorial: glossary, all components (KeyBuilder/Gate/Judge/SearchStrategy/Backend), YAML config, registration, testing; Search Session score-memo usage (lifecycle through interceptor → orchestrator, mutation contract, manual usage example, `force_legacy_path()` parity escape hatch) |
| [cache/migration.md](cache/migration.md) \[[EN](cache/migration.en.md)\] | Cache framework migration guide: how to adapt the cache system for non-Pi0.5 models |
| [cache/temporal_prune.md](cache/temporal_prune.md) \[[EN](cache/temporal_prune.en.md)\] | Temporal Prune KeyBuilder 使用指南：两步架构、参数配置、Reducer 选择、离线 Artifact 构建、生命周期 |
| [cache/llm_layer_extract.md](cache/llm_layer_extract.md) \[[EN](cache/llm_layer_extract.en.md)\] | CP1 LLM Layer Extract KeyBuilder 使用指南：两步架构（LayerExtractor + PrefixReducer）、attach_model 注入、离线 Stage 1 重建契约（重 tokenize + tokenizer self-check）、在线/离线 parity test |
| [cache/verdict_factor_judge.md](cache/verdict_factor_judge.md) \[[EN](cache/verdict_factor_judge.en.md)\] | **2026-05-07 重构 G1 APPROVED**：5 因子 → **17 因子扁平化** (`<descriptor>_<source>_<channel>` + `topk_action_variance`)；4 desc 改名 (`dir→direction`, `curv_radius→dispersion`, `cum_disp→path_length`)；judge **4 层正交架构** (Normalization → Factor → Calibration → Composer)；**no cold-start** (启动 fail-fast，废除 `cold_start_strategy` / `all_nan_fallback` / `sentinel`)；wire schema_version=2 (`factor_outputs.{raw, calibrated, composer_score}`)；详细方案见 [`logs/archive/verdict_factor_judge_refactor.log.md`](../logs/archive/verdict_factor_judge_refactor.log.md) |

### [experiments/](experiments/)

| File | Description |
|------|-------------|
| [experiments/artifact_layout.md](experiments/artifact_layout.md) | Canonical `exp/<experiment>/{config,data,analysis}/` layout rules — where new files go, tracking policy, `.gitignore` exceptions; §7 verdict-factor enrichment (B2 `--factors-yaml` flag, `library_stats` field, legacy fallback) |
| [experiments/conductor_tutorial.md](experiments/conductor_tutorial.md) \[[EN](experiments/conductor_tutorial.en.md)\] | **实验编排教程（重点：如何编写 driver 策略）**：用新 conductor 框架跑大规模评测 — 编写 `ExperimentStrategy`、复用/自写 `EpisodeRunner`、`ConductorDriver` + `WorkerAgent` 启动（按 server 分配 worker / 断点续跑 / 重试 / 监控）、调度语义、常见模式、测试；C1 `--non-concurrent` 保留原始单连接结构但当前为 sdpa 数值。取代旧 `main.py --num-workers` / `run_phase` 编排 |
| [experiments/cp1_cache.md](experiments/cp1_cache.md) \[[EN](experiments/cp1_cache.en.md)\] | CP1 Cache experiment guide: artifact building, calibration, YAML generation, 3-phase experiment execution, result analysis |
| [experiments/cache_latency_bench.md](experiments/cache_latency_bench.md) | Cache CP1 `check()` 六段延迟**回放基准** runbook：不加载模型、按真实 cache yaml 组装真 `CacheOrchestrator`、用 H5 真实 trajectory 逐 step 回放驱动 cache 真实工作、`SystemTimer` 探针记录每请求每部件延迟（per-step CSV + 聚合 json）；支持 threshold/always_hit/always_warm_start/composite(offline-calib + enriched pkl)；已知偏差：CPU build 段无 D2H / 仅驱动 CP1 / 读 action-history 的 composite judge 非逐调用等价。代码 `exp/cache_latency_bench/`，设计 `logs/archive/cache_latency_bench_plan.log.md` |
| [experiments/temporal_prune.md](experiments/temporal_prune.md) \[[EN](experiments/temporal_prune.en.md)\] | Temporal Prune experiment pipeline |
| [experiments/llm_layer_extract.md](experiments/llm_layer_extract.md) \[[EN](experiments/llm_layer_extract.en.md)\] | CP1 LLM Layer Extract 端到端 runbook：数据采集 → Step 2 build pkl（带 tokenizer self-check）→ YAML 模板（A/B 两种 reducer）→ run_cache_experiments → 结果分析 → manual parity verify |
| [experiments/trajectory_deviation.md](experiments/trajectory_deviation.md) \[[EN](experiments/trajectory_deviation.en.md)\] | Trajectory Deviation experiment runbook: Step 1a→1b→2→3→4 pipeline, parallelism rules, tunables |
| [experiments/warm_start_sweep.md](experiments/warm_start_sweep.md) \[[EN](experiments/warm_start_sweep.en.md)\] | Warm Start sweep runbook: 3 keybuilder × 3 start_t under always-hit + always_warm_start, artifact rebuild, 3-server parallel run, recovery/loss analysis |
| [experiments/serving_benchmark.md](experiments/serving_benchmark.md) | Throughput/latency benchmark runbook (Phase 6 M7 of `logs/archive/concurrent_serving_optimization_plan.log.md`): 5 modes (GPU microbench / sparse→dense / freq sweep / yaml density / batch window), driver/sweep/collect/plot tooling under `exp/serving_benchmark/` |
| [experiments/weighted_sum.md](experiments/weighted_sum.md) \[[EN](experiments/weighted_sum.en.md)\] | Weighted-sum two-layer calibration + weight-search runbook: Phase 1 offline picks the Layer-1 normalizer method+params per (keybuilder, modality) from the real LOEO query×library distribution (J = mag_sep + β·intra_spread − λ·sat shortlist); Phase 2 conductor pure-eval (always_hit) finds useful modalities + searches weights with the init-state leak guard; prompt_emb dropped. §3 Trajectory 扩展：在 18 个 Phase-2 best base（per-keybuilder top2+2nd-worst ∪ top10，去重）上叠加 depth{3,4,5,6} 多步检索，复用 `build_eval_config` + 老递减 `trajectory_weights` + depth-1 基线复用（`emit_trajectory_yamls.py` / `plot_trajectory_results.py`，零改 src/） |

### [data_collection/](data_collection/)

| File | Description |
|------|-------------|
| [data_collection/guide.md](data_collection/guide.md) | HDF5 data collection via `--collect` flag, schema and directory layout; **gate-research per-step collection** (`--collect-gate-dir` / `--export-collect-meta`, `__collect_meta__`, per-step verdict + `episode_summary` row schema, conductor `extra` contract, `summarize_gate_log`) |

### [deployment/](deployment/)

| File | Description |
|------|-------------|
| [deployment/aloha_sim.md](deployment/aloha_sim.md) | ALOHA Sim remote inference (WSL2 client + remote GPU); concurrent vs non-concurrent modes, with C1 described as raw single-connection structure under current sdpa numerics |
| [deployment/libero.md](deployment/libero.md) | LIBERO remote inference and simulator environment setup (WSL2 client + remote GPU); concurrent vs non-concurrent modes, with C1 described as raw single-connection structure under current sdpa numerics |
| → [experiments/conductor_tutorial.md](experiments/conductor_tutorial.md) | 并发 server 起法（`--concurrent` / `--replicas` 公共端口 / 多独立端点）+ 调优 + C1/C2 + troubleshooting **已并入**该端到端教程（原 `concurrent_serving.md`）；与 client 编排（写 driver 策略）合为一篇 |

### [papers/](papers/)

| File | Description |
|------|-------------|
| [papers/inference_cache_related_work.md](papers/inference_cache_related_work.md) | Related-work bibliography for inference caching / retrieval-augmented control in robotics, organized by proximity to our cache system (RT-Cache, VINN, VLA-Cache, BAC, RTC, Behavior Retrieval, etc.) |
| [papers/cloud_edge_deployment.md](papers/cloud_edge_deployment.md) | Cloud/edge deployment, brain-cerebellum split, fleet serving, compute/energy efficiency — deployment-context motivation for inference cache |
| [papers/paper_workbench.md](papers/paper_workbench.md) | Paper workbench: idea → method → story → experiments, living document |

### [upstream/](upstream/) — Original upstream openpi docs

| File | Description |
|------|-------------|
| [upstream/remote_inference.md](upstream/remote_inference.md) | General WebSocket remote inference setup |
| [upstream/docker.md](upstream/docker.md) | Docker installation and container usage |
| [upstream/norm_stats.md](upstream/norm_stats.md) | Normalization statistics: reuse, recompute, asset_id mapping |
