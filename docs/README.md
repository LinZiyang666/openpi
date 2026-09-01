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
├── theory/           # Formal results (Markov inheritance law and proofs)
├── papers/           # Related-work bibliographies (inference cache literature, etc.)
├── iclr/             # ICLR 2027 submission workdocs (TIER outline, experiment designs)
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
| [reference/openpi.md](reference/openpi.md) | Project structure, model architecture, Pi0 vs Pi0.5 differences, code paths, training configs (incl. the `pi05_robocasa` inference config), deployment, hardware |

### [architecture/](architecture/)

| File | Description |
|------|-------------|
| [architecture/cache_system.md](architecture/cache_system.md) \[[ZH](architecture/cache_system.zh.md)\] | Cache system spec: 3-stage pipeline, CP1/CP2/CP3 checkpoints, interceptor pattern, component design; §5.6 SimilarityJudge purity contract refinement (no write to storage; read-only via PayloadView allowed); §5.10 Search Session — Cross-Step Score Memo (opt-in per-episode score memoization, mutation contract, lock-free derivation); §5.11 PayloadView (read-only Judge-side facade + ForkPolicy); §5.12 Verdict Factor System (OnlineExtractor / OfflineWriter protocols + factor registry + CompositeJudge pipeline + LibraryStats + payload.factors schema + `all_nan_fallback` bootstrap + ⚠ multi-worker normalizer cold-start amplification operational gotcha — pre-launch self-check `per_worker_verdict_budget × (1 - max_factor_nan_rate) > window_size`); §5.13 Wire-Level Observability + Warmup Preload Protocol (`__hit_meta__` response field, `fetch_dump` / `preload_normalizer_buffer` / `unload_warmup_buffer` ctrls + `load_cache_config.yaml_id`, `CurrentCacheBundle.yaml_id`, `WarmupPool` lifecycle, server-owned warmup dump root with `.resolve()` allowlist, relation to §5.6 purity + §5.12 cold-start amplification); §5.14 Gate-Research Per-Step Collection (`__collect_meta__` sibling + `CheckResult.searched`, opt-in inline collector — wire byte-identical when off; robot_state default, vision standalone-only, raw out of scope; client-side ndarray→list codec; two row kinds incl. `episode_summary` provenance; canonical `compute_global_episode_id` via `EpisodeTask.extra["num_trials_per_task"]`; `summarize_gate_log` verdict-only tally); §5.5 gate roster refresh + G0a additive lifecycle hooks (`record_verdict` verdict-feedback fed by `check()` on every return path + `on_episode_start(task_key)` broadcast, both guarded/signature-filtered so legacy gates and the wire are unchanged) + `score_hysteresis` server-side N1 gate (+ Stage 3b N4 V2 injection branch via optional `L`; `L=None`=N1 latency profile, `L: 6`=N4 SR profile, reuses the `record_verdict` hit_type hook, zero orchestrator/wire change) + `follow_winner` server-side N2 lockstep blind-replay gate (Stage 4a; adds the **third** gate execution state — *blind replay* — via the optional docstring-only `replay_target()` hook + the orchestrator `not should_search` branch walking `PayloadView.walk_next` and returning `FULL_HIT × searched=False`; requires an in_memory backend; `walk_next` empty/exception falls through once to skip → `searched=False` MISS unlocks the gate, a single fail-safe contract against a permanent lock); §5.6 Phase 5 `failure_aware_gate` β₂·u_t kinematic term (gate-internal online-factor + D⁺-only ZScoreNormalization; `u_t_factor` config, NaN→margin-only degrade, `export_factor_outputs` opt-in wire-invariant by default); §5.4 learned projection (TRACER Phase 6): `ProjectionKeyBuilder` optional per-field linear head `z=xWᵀ` on cosine fields, trained offline on action-compatibility via masked InfoNCE (`fit(loss="masked")`); identity default = value-for-value non-regression, backend stays cosine; projected-artifact rebuild + re-calibration are the offline execution pipeline (planned, see the Phase-6 plan); §5.15 external executor hooks + routing（ablation study）：CP1 hit/miss 槽执行体替换的 additive 默认惰性 hook（`hit_executor`/`miss_executor`，WARM_START/prefill fail-fast）+ `CacheConfig.routing` 正向 allowlist（cp1-only/无 warm_tiers/depth-1/write-never，yaml 加载即拒）+ `SidecarExecutor` 有界连接 fail-closed sidecar 客户端；routing 随 bundle 热切换逐臂下发；§5.16 MLP Router Verdict Layer (X14 online-RL baseline)：`mlp_router` judge 在 verdict 槽按 query_keys 采样执行臂（对库侧全屏蔽），三臂映射 teacher→MISS / student→payloadless FULL_HIT(`hit_override=True`，零 fetch) / cache→强制回放(`hit_override=False`)；加法式 `JudgeResult/CheckResult.hit_override` 三态 + `router_outputs` 冻结 schema（`decision_idx` = 服务器权威步坐标，client step_idx 随 replan_steps 跨步不可作 join 键）；`judge_accepts_query_keys` 构建期签名探测条件注入（Composite / dump-wrapped legacy 调用不变）；`on_episode_end`/`on_task_end` 的 finally 广播驱动分片终结（write-never 走 decline 早返回，故必须 finally）；encoder 算子序 `raw→normalize→Q(fp16)` + 单线程 forward ⇒ trainer 逐位复算 behavior logprob；批完整性权威 = shard manifest 而非 journal. Chinese companion frozen at 2026-04-03；§5.17 GR00T N1.5 两阶段路径（RoboCasa365 跨场景线）：平行实现而非共用基类（`interceptor.py:75` 模块级 `import jax`，GR00T venv 导不进）；切点 = 视觉 token 散射进语言序列后、Qwen3 第 0 层前的 `input_embeds`，HIT 跳过 Qwen3 12 层 + action head，无第三阶段；三条静默失效点（图像 token 偏移随 prompt 长度浮动 ⇒ 必须掩码定位 / `LayerNorm` 在 autocast fp32 名单上 max|Δ|=1.4e-2 / inference tensor 在 context 内 `.cpu()` 逃不掉）；两处共享缝：`_CP1BaseKeyBuilder._slice()` 可覆写 + `InMemoryBackend.artifact_meta` 经 `CacheStorage` facade 暴露（`load_artifact` 原本只校验 `vector_dims`，同维库可静默错配）；§5.19 Text-IVF prompt 桶索引 + 指令段定界池化（text 模态字节相同建桶 / 精确→最近两段式收桶 / `text_ivf_knn` 替代 task_key 圈定 / `prompt_masked_pool`+`prompt_instruction_span` 旋钮与 `PROMPT_POOL_KNOB_BUILDERS` allowlist / artifact `prompt_pool` 元信息启动期强绑定 / 桶索引=派生结构 eager-at-load + 原子懒重建）；§5.6 judge 表标注 dispatch-surface 线术语：`threshold`+网格搜索 = **GST**（Grid-Searched Threshold），`dispatch_surface` = **RIT**（Risk-Indexed Threshold） |
| [architecture/cache_workflow.md](architecture/cache_workflow.md) | End-to-end workflow diagrams: startup, single inference with CP1/CP3, episode lifecycle, storage layer, YAML mapping, design principles |
| [architecture/experiment_conductor.md](architecture/experiment_conductor.md) \[[EN](architecture/experiment_conductor.en.md)\] | 两层实验编排框架（worker/agent/driver + 机制/策略分离）：episode 级中央队列 + worker pull 消除等待泡沫；yaml 亲和（warmup 放松 / eval 收紧）/ 永不空转 / warmup→eval barrier；账本断点续跑 + server 自愈；重试分类 / agent 监督 / 聚合监控；按 server 分配 worker；server 端点支持 --replicas 公共端口（router 已对 fetch_dump fan-out + 拼接）/ 多独立端点、server 协议不动；`sharding.shard_eval_stage` 把一个 **eval** yaml 摊成每台 server 一个兄弟 stage，让单臂相位也吃满整池（对 warmup 直接 raise——分片会静默切碎标定）。设计见 [`logs/archive/client_conductor_two_layer_refactor.log.md`](../logs/archive/client_conductor_two_layer_refactor.log.md) |

### [cache/](cache/)

| File | Description |
|------|-------------|
| [cache/tutorial.md](cache/tutorial.md) | Complete tutorial: glossary, all components (KeyBuilder/Gate/Judge/SearchStrategy/Backend), YAML config, registration, testing; Search Session score-memo usage (lifecycle through interceptor → orchestrator, mutation contract, manual usage example, `force_legacy_path()` parity escape hatch); §7 `dynamic_depth_knn` (TRACER Phase 1 / M3) per-step adaptive trajectory depth; §4 `projection` (TRACER Phase 2 / M1) outcome-compatible projection key builder (identity default); §6/§7 `dual_retrieval_knn` + `failure_aware_gate` (TRACER Phase 3 / M2) failure-aware dual-pool retrieval + sigmoid gate (degenerate == fixed-depth strategy + ThresholdJudge); §6 Phase 5 `failure_aware_gate` β₂·u_t kinematic activation (`u_t_factor` + `export_factor_outputs` opt-in; offline `MarginGateCalibrator`); §6 `mlp_router` (X14 online-RL baseline) judge 注册 + CP1-only yaml 配方（`arms` ts/tc/tsc、`weights_path` xor `constant_arm`、student 臂 ⇔ `routing.hit_to` 双向、`mode: sample` 需 temperature+seed、`dump_dir` 空即零 I/O）；§18 `cp1_groot_*`（GR00T N1.5 四个 pool builder：掩码定位三段图像 token、三相机需 `vision_2`、`cp1_` 前缀承载两条 validate 检查、reduce 前统一 fp32）；§4/§7 text-IVF：定界池化旋钮 + `text_ivf_knn` 配方与离线重建命令 |
| [cache/migration.md](cache/migration.md) \[[EN](cache/migration.en.md)\] | Cache framework migration guide: how to adapt the cache system for non-Pi0.5 models; §8 records GR00T N1.5 as the first model to complete the path |
| [cache/temporal_prune.md](cache/temporal_prune.md) \[[EN](cache/temporal_prune.en.md)\] | Temporal Prune KeyBuilder 使用指南：两步架构、参数配置、Reducer 选择、离线 Artifact 构建、生命周期 |
| [cache/llm_layer_extract.md](cache/llm_layer_extract.md) \[[EN](cache/llm_layer_extract.en.md)\] | CP1 LLM Layer Extract KeyBuilder 使用指南：两步架构（LayerExtractor + PrefixReducer）、attach_model 注入、离线 Stage 1 重建契约（重 tokenize + tokenizer self-check）、在线/离线 parity test |
| [cache/verdict_factor_judge.md](cache/verdict_factor_judge.md) \[[EN](cache/verdict_factor_judge.en.md)\] | **2026-05-07 重构 G1 APPROVED**：5 因子 → **17 因子扁平化** (`<descriptor>_<source>_<channel>` + `topk_action_variance`)；4 desc 改名 (`dir→direction`, `curv_radius→dispersion`, `cum_disp→path_length`)；judge **4 层正交架构** (Normalization → Factor → Calibration → Composer)；**no cold-start** (启动 fail-fast，废除 `cold_start_strategy` / `all_nan_fallback` / `sentinel`)；wire schema_version=2 (`factor_outputs.{raw, calibrated, composer_score}`)；详细方案见 [`logs/archive/verdict_factor_judge_refactor.log.md`](../logs/archive/verdict_factor_judge_refactor.log.md) |

### [theory/](theory/)

| File | Description |
|------|-------------|
| [theory/markov_inheritance.md](theory/markov_inheritance.md) | **马尔可夫继承定律：形式化与证明**（纯数学，自成一体，零实验内容；实证对照见配对文档 history_verdict.md）。核心链条（推论 4.1，对数损失）`0 ≤ R*(k)−R*(k,h) = I(a*;h|k) ≤ I(a*;o|k)`。引理 1（无记忆 teacher ⇒ 标签条件独立 `I(a*;h|o)=0`）；引理 2（去噪上界：历史能贡献的信息被 key 有损度封死）+ 命题 2.2（上界紧且**双向非蕴含**：A1/A2 既推不出历史无用也推不出有用）+ 引理 2.3（上界随 key 粗化单调变松）+ 备注 2.4（**剂量-反应非定理**，附反例）；命题 3（**阶段分解**：Φ=φ(h) 时为精确恒等式；备注 4.2 论在线/oracle 阶段变量）；定理 4（风险单调性对一切损失、恒等式**限对数损失**；备注 5.1 方向规则：改善⇒I>0、无改善⇏I=0）+ 命题 4.2（**固定算子上历史效果符号不定**，双向构造）；命题 5（**选择条件化 = collider 重注**，两分支闭式 0→1 bit；备注 6.1 一般选择机制；**备注 6.2 聚合去随机化 = 另一超越-teacher 机制，故非唯一**）；命题 6（**标量化分离**：帧间 dynamics 特征非 `{s_l}` 的函数 + 单调打分不可实现全局差分阈值）。§8 假设清单（A0/A1/A2 + 经验条件 (4.1)）与六条不主张；§9 主结果一览 |
| [theory/history_verdict.md](theory/history_verdict.md) | **历史帧的价值判决：继承定律在本系统的实证定位**（与 markov_inheritance.md 配对的解释性文档；两份合读即自足）。**§3 实例化定理**（命题 7–11 + 推论 8.1/9.1/10.1，编号接续数学文档）：本方法确切公式（pool 降维 key → 逐模态 cos/L2 → zscore-tanh → 模态加权 → 历史非负加权）逐阶段对应命题；**命题 7**（zscore-tanh 零信息增删，但校准参与融合排序）、**命题 8**（单帧决策充分 ⇒ 非负历史加权期望增量非正 + 严格受损构造；**条件定理**，前提待测）、**命题 9**（**ε-决策充分预算：任何历史方法的期望增量 ≤ 单帧 regret E[ε₀]**，"足够优质⇒挖不出"由此定量化为可测预算）、**命题 10**（margin 逐候选 no-flip 证书；保守统一证书在生产权重下恒不触发）、**命题 11**（**反定理：任意接近完美的准确率仍可有严格正增量**，"足够优质"必须用 regret 定义）、推论 8.1/10.1（修复须滞后边际反超 ∧ 决策有益；E3 的 ADR 非该交集的测量）。含 2 幅 mermaid 图（历史价值地图：两来路三闸门 / 判决解释链）。现象：强配置上加历史全负（top-10 Δ −6.4pp 10/10、171 形状空集、l10 d1 最优）。定价：**E1** B/C 算子残差小幅低于 A（+2.9%~+8.8%；算子层证据，非互信息证明）→ **E1-O** 0/8 过预注册门槛、最严对齐 cell 精确 0.00%、ε=0.10 两 cell 下界正但低于门槛（与阶段错位补偿一致；oracle 进度 + 非等价判定，不升级为等式）→ **E1-C** Δ 特征 spatial k=1 即 +9%、k=5 至 +12.4%，生产打分不可表（与命题 6 方向一致；特征×聚合混杂待分解）→ **E3** 近似无混叠（ADR 0.24%/2.79% vs 随机 47–48%，12/12）⇒ 选择通道本地近似关闭（ADR 非可修复占比的测量）→ **E4** 已试旋钮均不兑现、闭环净负（−3.47pp）→ **E5** d3-trough = winner's curse。gate 线旁证（AUC 0.973–0.986，N4 胜出）= 历史的正确岗位。含逐通道定价表、**预算总闸审计清单**（增益 ≤ E[ε₀] + 阶段坐标/滞后修复/换算子/winner's curse）、五个遗留开口（**regret 预算测量 / 连续权重域穷尽** / E1-C 两对照 / 在线阶段版 E1-O / 记忆假肢）、主张-依据对照表、边界声明 |

### [experiments/](experiments/)

| File | Description |
|------|-------------|
| [experiments/artifact_layout.md](experiments/artifact_layout.md) | Canonical `exp/<experiment>/{config,data,analysis}/` layout rules — where new files go, tracking policy, `.gitignore` exceptions; **§1.1 实验族**（`exp/<family>/<experiment>/`，族目录只放 README + 子实验、单层嵌套、测试 basename 须全局唯一；现有族 `exp/ablation_study/`）；**§1.2 登记目录**（registry — 非实验的第三类 `exp/` 条目：禁 `config/`/`data/` 两槽、台账目录禁名 `data/`（否则被 §3 忽略吞掉）、只存指针与校验和不存字节；**准 `analysis/` 且必须组织为 `analysis/<任务>/` —— 任务层强制 + 每任务必带 `MANIFEST.json`（逐文件 sha256 + source）+ 收编是复制不是移动**（移动会打断实验报告的相对路径引用）；设计依据落 `logs/`；现有登记目录 `exp/data_authority/` = 实验数据集的权威副本台账 + 收编的分析产物）；§7 verdict-factor enrichment (B2 `--factors-yaml` flag, `library_stats` field, legacy fallback) |
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
| [data_collection/guide.md](data_collection/guide.md) | HDF5 data collection via `--collect` flag, schema and directory layout; **gate-research per-step collection** (`--collect-gate-dir` / `--export-collect-meta`, `__collect_meta__`, per-step verdict + `episode_summary` row schema, conductor `extra` contract, `summarize_gate_log`); **RoboCasa365 teacher-library collection** (one-server-one-worker topology, run-plan artifact, audit + deterministic manifest, binomial completion rule) |

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
| [papers/paper_workbench.md](papers/paper_workbench.md) | Paper workbench: idea → method → story → experiments, living document（⚠ ENGRAM 旧叙事，已被 TIER 方向取代，待重写；现行论文工作文档见 [iclr/](iclr/)） |

### [iclr/](iclr/)

ICLR 2027 投稿（TIER: experience-tiered inference）论文工作文档。⚠ 2026-08-22 起旧版冻结为 `.old`（owner 大改中，新版文件落地后替换下表）。

| File | Description |
|------|-------------|
| [iclr/paper_rethink_discussion.md](iclr/paper_rethink_discussion.md) | 现行讨论纪要（2026-08-22 起，已整体重构）：放弃旧 TIER 叙事回归 VLA cache 本身；novelty 叙事 / R(ε) 定义与数学 / 工作提纲 v0.2 / 实验对照表（E0–E13）/ defense 弹药库 / 部署生命周期 / deck 结构 |
| [iclr/actioncache_response_plan.md](iclr/actioncache_response_plan.md) | ActionCache（concurrent work）攻防方案（2026-08-26）：核实（地板 45–53% vs 14%、端到端 1.26×、LIBERO 被 NFE=1 裸基线支配）/ ICLR 规则豁免 / 写作定位与三层攻防分工 / 12 条弹药 / 四臂对照（Arm1=激活 CP2，核心 ~3,000 ep）与预注册分支 |
| [iclr/redundancy_structure_fig.html](iclr/redundancy_structure_fig.html) | §5.3 冗余结构三 panel 草图（真数据自包含页 + PNG）；讲法与裁决见纪要 §6.2 |
| [iclr/tier_paper_outline.old.md](iclr/tier_paper_outline.old.md) \[[ZH](iclr/tier_paper_outline.zh.old.md)\] | **旧版（待重写取代）** TIER 论文提纲 v2：thesis「库的价值在索引不在 payload」、9 页结构/float 台账/appendix 预算/3 贡献；4 审稿人对抗评审 32 findings 裁决修订（裁决日志在文末）；scope lock=无 history 项、Markov 继承线独立成文；文末 Q&A rebuttal 弹药库（Q1 trained-router 质疑三层回应） |
| [iclr/tier_experiment_designs.old.md](iclr/tier_experiment_designs.old.md) | **旧版（待重写取代）** 实验设计全卡 X1–X13（目的/设计/产出/判读分支含预注册负结果结论）+ 前置基建清单 + 波 0–3 执行顺序 |

### [upstream/](upstream/) — Original upstream openpi docs

| File | Description |
|------|-------------|
| [upstream/remote_inference.md](upstream/remote_inference.md) | General WebSocket remote inference setup |
| [upstream/docker.md](upstream/docker.md) | Docker installation and container usage |
| [upstream/norm_stats.md](upstream/norm_stats.md) | Normalization statistics: reuse, recompute, asset_id mapping |
