# Weighted-Sum 两层重构 + 两阶段校准/权重搜索实验

> **Status**: `G2 APPROVED` (2026-05-25 / weighted_sum two-layer refactor code review)
> **Level**: L3（新增搜索层归一化子系统 + config schema 变更 + 新实验管线 + conductor 策略，跨模块）
> **Authority**: Execution
> **关联**: [`docs/architecture/cache_system.md`](../docs/architecture/cache_system.md) §5.x search/normalization、[`docs/experiments/conductor_tutorial.md`](../docs/experiments/conductor_tutorial.md)、[`docs/experiments/artifact_layout.md`](../docs/experiments/artifact_layout.md)、现有 phase1 weight-grid 模板 `exp/common/analysis/phase1/libero_spatial/`

---

## 1. 背景与失败根因

`weighted_score_sum` 路线（`InMemoryBackend._search_weighted_score_sum`，`src/openpi/cache/backends/in_memory_backend.py:560-630`）此前几乎完全失败。代码层面定位到三个叠加根因：

1. **校准分布来源错误**：`exp/common/calibrate_score_sum_stats.py:69-82` 采样的是**库内 entry×entry 随机对**的相似度来估 p5/p95，但检索时真实分布是 **query→全库**。两者分布不同，拟合出的归一化参数与实际搜索分数不匹配。
2. **percentile 在高基线相似度模态上塌缩**：vision/prompt 走 cosine，同任务/跨任务 separation 极小（mean-pool 把判别信号洗没了），数值上 `p5≈p95`，触发 `denom≈0 → s=0.5` 兜底（`in_memory_backend.py:614-615`）→ 该模态对所有候选贡献同一常数，**完全失去判别力、权重被白白浪费**，搜索被少数校准尚可的模态主导。
3. **校准/归一化与 keybuilder 强耦合却不分层**：不同 keybuilder（mean_pool / spatial / max_pool / clip）对同一模态产出的分布形状差异巨大，但旧代码只有一种 percentile 方法、且"raw→[0,1]→percentile→加权和"全部焊死在 `_search_weighted_score_sum` 一个函数里，无法按 (模态, keybuilder) 选不同方法/参数。

补充观察（owner 提供）：cosine 分数高度聚集，常常**要到小数点后好几位才有区别**。这意味着归一化既要解决"分布来源 + 塌缩"，还要能把这种极窄高基线带**拉伸到数值上足以区分**。

---

## 2. 目标与非目标

### 目标
- 把 weighted_sum 重构成**两层正交架构**（与现有 verdict judge 的 4 层架构同构，是已验证的设计先例）：
  - **Layer-1 校准/归一化层**：可插拔的 per-(模态, keybuilder) normalizer，把底层 raw similarity → 归一化值；参数由 Phase 1 数据拟合。
  - **Layer-2 搜索/融合层**：对归一化值做加权和；Phase 2 搜权重。
- 设计一套**两阶段实验**：Phase 1 离线收集真实 query×全库分数分布、为每个 (模态, keybuilder) 数据驱动地选归一化方法+参数；Phase 2 找有用模态 + 搜最优权重分配。
- 实验 server/client 一律走 `src/openpi/conductor/` 新基础设施。

### 非目标
- 不动 `weighted_rrf` 路线的语义（仅在共享 `_batch_field_scores` 等 helper 上复用）。
- 不动 server WebSocket 协议、不破坏 C1（non-concurrent bit-identical）/ C2（runtime write-frozen）。
- 不新采集数据（见 §5 决策：LOEO 复用现有 50 episode）。
- 不在本轮把 Qdrant 后端纳入 score_sum（现状仅 `in_memory` 支持，保持）。

---

## 3. 已锁定的设计决策（owner 确认）

| # | 决策 | 取值 |
|---|------|------|
| D1 | Layer-1 归一化方法 | **方法库 + 数据驱动自动选择**；候选全部为**单调、保幅（保序+保留相对间距/幅值结构）变换**：仿射(min-max) / z-score / 单调非线性(logit、−log(1−cos)、power/gamma) / L2 的 exp(−d/τ)。**显式排除 rank/经验CDF**——它把分布均匀化、只剩名次、退化成 weighted RRF，违背 weighted_sum"聚合分数反映底层模态相似度幅值"的初衷。 |
| D2 | 校准分布定义 | **真实 query × 全库 全部相似度**。query = 现有库 entry 本身；打分时**过滤掉 query 自己所属的那条链/episode**（LOEO by chain）以消除 self-match 与链内近邻污染，忠实还原线上"库中无当前 episode、但有同任务其它 episode"。 |
| D3 | 校准计算位置 | **离线脚本**。Phase 1 纯离线（无需起 server / conductor）。conductor 只在 Phase 2 eval 用；若将来要补 held-out 才用 conductor 采集。 |
| D4 | 任务套件 / 数据 | **libero_spatial + 复用现有**。对齐模板 `exp/common/analysis/phase1/libero_spatial/`。 |
| D5 | 非线性 | **仿射 + 单调非线性自动选择**：饱和 cosine 靠非线性把稠密区拉开，但严格单调保幅。 |
| D6 | 记忆 | 本任务全程不写 auto-memory（owner 指令）。 |
| D7 | prompt_emb | **直接退出实验**：任务内近常量(task-ID 化)、双峰、对步级判别无用。YAML 经 `keys.prompt_emb.enabled: false`（即默认值，已核实 `config.py:1069/1274` 不进 enabled_fields/fusion_weights）屏蔽；校准与权重搜索均不含 prompt_emb。 |

---

## 4. 数据核查结论（前置，已完成）

- 6 个 libero_spatial 库 artifact（`cp1_mean_pool`/`cp1_max_pool`/`cp1_spatial_pool_16`/`cp1_spatial_pool_64`/`clip_vit_b_32`/`clip_vit_l_14`）均 **1018 entries / 10 tasks / 全 CP1**，带 `intermediates {0.1…0.9}`、`prev_ids`(链)、`step_idx`、`library_stats`。
- 字段：CP1 库 `query_keys` 含 `vision_0/1/2 + prompt_emb + robot_state`（vision dim 随 keybuilder：mean/max=2048、spatial16=32768、spatial64=8192）；CLIP 库仅 `vision_0/1`（512 / 768）+ prompt_emb + robot_state。
- 原始 HDF5（50 ep）每步含 `vision_0/1/2 (256,2048)`、`prompt_emb (200,2048)`、`robot_state (32)`、原图 → 足以为任意 keybuilder 重建 key。
- **缺口已解（见 D2）**：原无独立 held-out query；采用 LOEO（过滤 query 自己这条链）后零额外采集即可。
- **`vision_2` 休眠**：CP1 库 `query_keys` 有 `vision_2` 但 `vector_dims` 未列 → 当前 `_iter_active_fields` 不激活它。Phase 2 找有用模态时可作 CP1 候选（需补 `vector_dims`，列为可选项、非关键路径）。
- **候选模态集（D7 后）**：`{vision_0, vision_1, robot_state}` + CP1 可选 `vision_2`；**prompt_emb 已移除**。CLIP 库本就无 vision_2。

---

## 5. Part A — 代码重构（两层）

### 5.1 新增：Layer-1 归一化子系统

**新文件** `src/openpi/cache/components/score_normalizers.py`

- `ScoreNormalizer`（Protocol）：`__call__(self, raw: torch.Tensor) -> torch.Tensor`。输入是某 field 的底层几何分数张量 `[N]`（cosine 时为 cos∈[-1,1]，l2 时为距离 d∈[0,∞)），输出归一化后 `[N]`，**保证对相似度单调递增**。
- 固定方向预映射约定：cosine 用 `x=cos`（高=更像）；l2 用 `x=-d`（高=更像）；`exp_l2` 方法直接吃 `d`。所有方法内部按 `sim_type` 处理方向。
- **方法 × sim_type 划分（写死，registry 校验）**：`logit` / `neg_log_one_minus` / `power` 为 **cosine-only**（专治近 1 饱和带）；`l2` 字段（robot_state）仅允许 `{affine_clip, zscore, exp_l2}`；`affine_clip` / `zscore` 两 sim_type 通用。method 与 field 的 sim_type 不兼容时 fail-fast。
- 具体 normalizer（dataclass，持拟合参数；均单调保幅）：
  - `AffineClipNormalizer(lo, hi)`：`clip((x-lo)/(hi-lo), 0, 1)`。泛化旧 percentile（lo=p_a、hi=p_b）。带内纯线性。
  - `ZScoreNormalizer(mu, sigma, squash)`：`(x-mu)/sigma` + **强制有界 squash**（默认 `0.5*(tanh(z)+1)`→[0,1]；`squash` 语义在测试里锁定）。仿射核。
  - `LogitNormalizer(lo, hi, eps)`：先 `u=clip((x-lo)/(hi-lo),eps,1-eps)` 再 `logit(u)` 再仿射回 [0,1]。把近 1 稠密区拉开。
  - `NegLogOneMinusNormalizer(eps, scale)`：`-log(1-x+eps)` 再仿射。专治 cosine 近 1 饱和。
  - `PowerNormalizer(lo, hi, gamma)`：`((x-lo)/(hi-lo))**gamma`。gamma 调稠密区分辨率。
  - `ExpL2Normalizer(tau)`：`exp(-d/tau)`（L2 映射；τ 由真实分布拟合）。
  - `LegacyPercentileNormalizer(sim_type, tau, p5, p95)`：**仅向后兼容**——忠实复刻旧 `percentile`：先 `(cos+1)/2`(cosine) 或 `exp(-d/τ)`(l2) 映到 [0,1]，再 `clip((s-p5)/(p95-p5),0,1)`，含 `denom≤0→0.5` 兜底。**不进 Phase 1 候选库**，仅用于加载旧 YAML/`calibration.json`。
  - `DirectionUnifyNormalizer(sim_type, tau)`：**仅 backend/spec 级向后兼容**（`type:none` 路径）——cosine→`(cos+1)/2`、l2→`exp(-d/τ)`，无归一化裁剪。**不进 Phase 1 候选库**，且 **不是合法生产 YAML**（config 级 `weighted_score_sum_knn` 仍 reject `type:none`，见 §5.3）——仅服务于现有 backend spec 级单测。
- `SCORE_NORMALIZER_REGISTRY: dict[str, type]`，每类提供：
  - `name`、`supported_sim_types`
  - `fit_from_scores(scores: np.ndarray, sim_type, *, robust_pct=(1,99)) -> "ScoreNormalizer"`（离线拟合参数）
  - `params_to_dict()` / `from_params_dict(d)`（序列化到/自 calibration artifact 与 QuerySpec）
- `build_field_normalizers(active_fields, score_normalization: dict | None, field_similarity: dict) -> dict[str, ScoreNormalizer]`：运行时工厂。
  - `type=="per_field"`：按 `fields[field]={"method","params"}` 构造；**normalizer 自带 tau 等全部参数（Layer-1 全拥 raw→归一化映射）**，`field_similarity.to_similarity` 在此路径被忽略，`field_similarity.type` 仅用于告知 raw 分数类型。
  - `type=="percentile"`：每 field 构造 `LegacyPercentileNormalizer`，tau 取自 `field_similarity.to_similarity`（复刻旧值）。
  - `type=="none"` / 缺省：**不是恒等**——构造 `DirectionUnifyNormalizer`（cosine→`(cos+1)/2`、l2→`exp(-d/τ)`，τ 取自 `field_similarity`），忠实保留旧 `weighted_score_sum` 无归一化语义（旧 l2 测试期望 `exp(-d/0.334717)`）。

> 关键不变量（写入 docstring + 测试）：所有 normalizer **全局单调非减、仅在未饱和区间严格递增**（裁剪/饱和段允许平台——刻意的有界设计，不要求全实数域严格单调）；**不做经验CDF/均匀化**；对"近 1 极窄带"输入要能把带拉开（测试用合成饱和分布断言 normalized std 显著放大、且 ≠ 等距 rank）；**所有 Phase 1 候选 normalizer 输出 ∈ [0,1]**（ZScore 必带有界 squash；logit/neglog/power 末端 affine-clip 到 [0,1]）——保证 Layer-2 加权和与 §6.4 的 J 在同尺度可比，杜绝无界 normalizer 靠尺度而非判别力支配。

### 5.2 改造：后端搜索路径变薄

`in_memory_backend.py`：
- `_search_weighted_score_sum`（:560-630）改为：构造 `normalizers = build_field_normalizers(active, spec.score_normalization, spec.field_similarity)`；逐 field `raw, mask = _batch_field_scores(...)` → `s = normalizers[field](raw)` → `final_scores += weight * s * mask`。**移除**内联的 `(cos+1)/2` / `exp(-d/tau)` / percentile（这些下沉进 normalizer 方法）。
- 轨迹路径一致性：`_compute_level_scores`（:826-857）与 legacy `_batch_step_scores`→`_search_weighted_score_sum`（:1054-1055）共用同一归一化。**性能修正**：normalizers 在 `_search_with_trajectory` / `_search_with_trajectory_legacy` 入口**构造一次**、向下层透传，**不在每层每 query 重复 build**（否则链深 L 重复 L 次）。
- `_batch_field_scores` 契约不变（仍返回 raw cos / raw d + mask）。

### 5.3 config 层变更

`src/openpi/cache/config.py`：
- `ScoreNormalizationConfig`（:309-313）扩展：
  - `type: str`：`"none" | "per_field" | "percentile"`。`"percentile"` 保留为**向后兼容**：每 field 走 `LegacyPercentileNormalizer`（§5.1）——**关键修正**：旧 p5/p95 是在 `(cos+1)/2`/`exp(-d/τ)` **映射后** [0,1] 空间算的（核实 `in_memory_backend.py:597-615` + `calibrate_score_sum_stats.py:76`），**不是 raw cos 空间**，故不可简单套 `AffineClip` 到 raw cos。旧 `calibration.json` 与旧 YAML 不变即可加载。
  - `fields: dict[str, dict]`：`per_field` 形如 `{"vision_0": {"method":"logit","params":{...}}, ...}`。
- 校验（:1259-1283，现状 `:1261` 对 `weighted_score_sum_knn` 已 reject `None`/`type==none`）：
  - **保持 reject `type==none`/`None`**：生产 YAML 必须用 `per_field` 或 `percentile`（`type:none`=无归一化=失败模式，仅 backend/spec 级单测可用，**不是合法 YAML**——消解 §5.1 与本节的矛盾）。
  - **新增 `per_field` 分支**（现状只处理 percentile）：要求 `fields` 覆盖所有 weight>0 的 enabled field（镜像 percentile 的覆盖检查 `:1271-1283`）；每 field `method` ∈ registry、`method` 与该 field `field_similarity.type` 兼容（§5.1 划分）、`params` 含该 method 必需键（affine: lo/hi；exp_l2: tau；logit: lo/hi/eps；…），否则 fail-fast `ConfigValidationError`。
  - tau 归属：`per_field` 用 params.tau；`percentile` 用 `field_similarity.to_similarity.tau`。
- `_score_norm_to_dict`（:2318 附近）透传新 schema 到 `QuerySpec.score_normalization`。
- 工厂 `weighted_score_sum_knn` 分支（:2396-2408）不变（仍传 `score_normalization` dict）。

### 5.4 受影响文件清单

| 文件 | 改动 |
|------|------|
| `src/openpi/cache/components/score_normalizers.py` | **新建**：Protocol + 6 候选 normalizer（affine/zscore/logit/neg_log/power/exp_l2）+ 2 兼容 normalizer（LegacyPercentile/DirectionUnify）+ registry + builder |
| `src/openpi/cache/backends/in_memory_backend.py` | `_search_weighted_score_sum` / `_compute_level_scores` / `_batch_step_scores` 改用 builder（入口单次构造透传）；删内联归一化；更新模块 + 方法 docstring（移除内联 (cos+1)/2 / exp(-d/τ) 描述）|
| `src/openpi/cache/config.py` | `ScoreNormalizationConfig` schema 扩展 + 校验 + `_score_norm_to_dict` |
| `src/openpi/cache/storage_types.py` | `QuerySpec.score_normalization` 注释更新（dict schema 不变，值结构变）|

向后兼容：`weighted_rrf` / `qdrant_*` / `fusion_method=None` 全不受影响；旧 `percentile` calibration 仍可加载。

### 5.5 文档与索引 + commit 边界（L3 必需，WA §2.1 / §4）

**文档更新**（WA §2.1：L3 强制架构文档更新）：
- `docs/architecture/cache_system.md` §5.x search/normalization — 增 Layer-1 两层归一化子系统（`ScoreNormalizer` 协议 + registry + per-(模态,keybuilder) 选择 + [0,1] 有界契约 + 单调保幅/排除 rank 的设计理由）。
- `docs/cache/tutorial.md` — SearchStrategy/Backend 段增 `weighted_score_sum_knn` 的 `score_normalization.per_field {method,params}` schema + `percentile` 兼容 + `type:none` 仅 backend 级说明。
- `docs/experiments/weighted_sum.md`（**新建 runbook**）— Phase 1 离线校准 + Phase 1.5 代理 + Phase 2 conductor 权重搜索端到端。
- grep `docs/` 中 `calibrate_score_sum_stats` 引用（如 `docs/experiments/cp1_cache.md` / `docs/reference/openpi.md`）→ 指向新校准脚本。
- **索引同步（constitutional §4）**：上述 `docs/` 改动**同 commit** 更新 `docs/README.md` **与 `docs/experiments/README.md`**（新建 runbook 的本地索引）；`logs/` 状态变更同步 `logs/README.md`。

**commit 边界 / 跟踪（WA §4 + artifact_layout）**：
- 入库：`.py` / YAML / analysis / plan / docs；`data/**` 默认 gitignore。
- **需新增 `.gitignore` 例外**（小 JSON 真值源，复现必需）：① `calibration_normalizers.json`（现有 `**/calibration.json` glob 不覆盖此文件名）；② `exp/common/data/db/libero_cache/libero_spatial_init_map.json`（**核实当前未被 git 跟踪**，是 §7 防 init-state 泄漏硬门的输入）。
- ⚠ `.gitignore` 修改属高危 git 操作（execution_authority §7），commit 时需 **owner 逐次显式同意**；`emit_yamls` 在 `init_map` 缺失时 **fail-fast**，绝不静默跳过泄漏门。

---

## 6. Part B — 校准方法（离线）

**新文件** `exp/common/calibrate_score_normalizers.py`（取代 `calibrate_score_sum_stats.py`；旧脚本保留至迁移完成后再决定删除）。

输入：库 artifact（6 pkl）；输出：`calibration_normalizers.json`（per keybuilder per field：`{method, params, sim_type, diagnostics}`）+ 诊断图到 `analysis/`。

每 (keybuilder, field) 流程（field 取库 `vector_dims` 的活跃字段，**排除 prompt_emb(D7) 与休眠 vision_2**）：
1. **分组**：按 entry `trajectory_id`（确认已填充，形如 `episode_XXXX`）分组为 episode；Code 阶段 assert 全 entry trajectory_id 非空，否则 fail-fast（见 §9）。
2. **真实分布采集（D2）**：对每个 query entry q，对**不在 q 所属链**的全库 entry 算 raw 几何分数（cosine→cos，l2→d），汇集成该 field 的 pooled raw-score 分布；同时记录每对的 same-task / cross-task 标签（`payload.task_key`）。规模：1018 query × ~1000 lib ≈ 1e6 对/field，GPU 批量 cosine 可行；spatial16(32768-d) 用 fp32 分块。
3. **逐方法拟合**：对该 field 兼容的 method（§5.1 划分）`fit_from_scores` 得参数，apply 到 pooled 分布。
4. **诊断指标（不做单一最优裁决）**：因所有候选都单调，**rank-based 指标（AUC 等）对单调变换不变、无法区分方法**，故用**幅值结构 + 类内保真**双指标：
   - `mag_sep = mean(s_norm|same-task) − mean(s_norm|cross-task)`（跨类幅值分离）。
   - `intra_spread = mean_class std(s_norm)`（**类内梯度保真**——防止近二值/阶跃解把任务内幅值压平，这正是 weighted_sum 要保留的信号；纯靠 mag_sep 会反向奖励近二值解）。
   - `sat = frac(s_norm∈{0,1})`（饱和裁剪比例）。
   - 综合诊断分 `J = mag_sep + β·intra_spread − λ·sat`（β,λ 默认 0.5，可调）。**所有候选输出 ∈ [0,1]**（§5.1 不变量），故 J 各项尺度可比、不会被无界 normalizer 靠尺度刷高。
5. **输出 shortlist 而非赌单一最优**：每 field 落 **top-2 候选 (method,params)** + 全诊断。**最终方法由 Phase 2 真实任务成功率裁决**——避免用内在代理指标选"唯一最优"重蹈旧校准（代理指标）失败覆辙。
6. **落盘**：`calibration_normalizers.json` 写 per (keybuilder, field) 的 shortlist + diagnostics(mag_sep/intra_spread/sat/p1/p99 等)。
7. **副产物**：各 field 的 `mag_sep` 给出**模态判别力排序**，为 Phase 2 有用模态提供先验。
8. **离线检索质量代理（Phase 1.5，高信号零 GPU）**：用已选 normalizer 对 held-out query（LOEO）跑 weighted_sum 检索 top-1，统计 **same-task 命中率 / `|Δstep|` 时序邻近 / 动作块 MSE vs GT**。这比 Phase 2 always_hit 纯回放 SR 信号强得多，用来**先给权重配置/模态组合排序**，再把 GPU 预算花在前排候选上。LOEO 分组 = 按 `trajectory_id`（确认已填充，形如 `episode_XXXX`）。

---

## 7. Part C — 两阶段实验

新实验包 `exp/weighted_sum/`（遵 `artifact_layout.md`：`config/ data/ analysis/` + 根放代码）。

```
exp/weighted_sum/
  __init__.py
  weight_search_strategy.py              # ExperimentStrategy（纯 eval，skip_warmup）— 根目录（artifact_layout 规则，不建 strategies/ 子包）
  run_phase2.py                          # 构造 strategy + ConductorDriver.run
  summarize.py                           # journal(jsonl) → per-yaml success_rate JSON（供 plot_phase2 消费）
  init_holdout.py                        # held-out init 防泄漏（缺 init_map fail-fast；leak guard 在 episode 构造期）
  emit_yamls.py                          # 生成 eval YAML：per_field normalizer + 权重格 + write_policy:{type:never}(C2)；prompt_emb 屏蔽
  config/
    phase2_isolation/   # 单模态隔离 yaml（找有用模态）
    phase2_grid/        # 有用模态上的权重网格 yaml
  data/
    calibration_normalizers.json         # Phase 1 产物（需新增 .gitignore 例外，见 §5.5）
    phase2/<keybuilder>/experiment_state.json
  analysis/
    plot_phase1_calibration.py           # 各 (field,kb,method) raw vs normalized 分布 + mag_sep 柱状
    plot_phase2_results.py               # success_rate vs 权重配置（仿 phase1_results.png）
    phase1_calibration.md / phase2.md
```

### Phase 1（校准，离线）
- 1a 数据：复用现有 50-ep HDF5 + 6 库 pkl，**无需采集**（D2/D4）。
- 1b 库：已建好；可选小重建为 CP1 激活 `vision_2`（补 `vector_dims`），非关键路径。
- 1c 校准：跑 `calibrate_score_normalizers.py` → `calibration_normalizers.json` + 诊断图。**纯离线、无 GPU server**。

### Phase 2（找有用模态 + 搜权重，conductor eval）
- **Judge = always_hit**（纯 cache replay，隔离检索质量，与 phase1 weight-grid 模板一致），`search_strategy.type = weighted_score_sum_knn` + 校准好的 per_field normalizer。
- **C2 write-frozen**：emit 的每个 eval YAML 必须含 `write_policy: {type: never}`（默认 `on_any_miss` 会在 server load/`load_cache_config` fail-fast，`config.py:403`）。strategy/config 测试覆盖。
- **候选模态**：`{vision_0, vision_1, robot_state}`（+CP1 可选 vision_2）；**prompt_emb 经 `keys.prompt_emb.enabled:false` 屏蔽**。
- **2a 模态隔离 + 方法定档**：每次只给一个模态非零权重，并对该模态 Phase 1 **shortlist 各候选 method** 各跑一次 eval → 同时定"有用模态集合"与"每模态最终 normalizer 方法"；与 Phase 1 `mag_sep` 先验交叉验证。
- **2b 权重搜索**：在有用模态（各自已定档方法）上做权重网格（粗→细），找最优分配。
- **⚠ 防 init-state 泄漏（硬约束）**：Phase 2 eval 的 `orig_init_state_idx` **必须与建库 50 episode 所用的集合不相交**（建库每任务仅用 ~5/50 init，剩 ~45/任务可用）。否则 live rollout 从同 init 起会与库内孪生 episode 早期逐帧近同 → 平凡命中、SR 虚高。`emit_yamls.py`/episode 列表从 `libero_spatial_init_map.json` 读出已用 `orig_init_state_idx` 并排除。
- **先代理后 eval**：用 Phase 1.5 离线检索质量代理（§6.8）预排序权重/模态组合，只对前排候选跑 GPU eval，省预算。
- 编排：`WeightSearchStrategy` 为**纯 eval 模式**（`consumes_calib_id=None`、不建 warmup stage，参照 `WarmupEvalStrategy(skip_warmup=True)`）；`EpisodeRunner` 直接复用 `examples/libero/episode_runner.py`；server 用 `serve_policy.py --concurrent`（或 `--replicas N`）；driver/agent/journal/retry/monitor 全复用 conductor。
- **运维**：固定 `--seed`；server 拓扑参 `reference_device_topology`（单 server ≤3 replica）；**预算估算**：设 K 个有用模态、每模态 shortlist≤2 方法、2a 隔离 ~ (K×2) 配置、2b 网格粗扫 ~ O(数十) 配置，每配置取 held-out init 子集（如每任务 20 ep × 10 = 200 ep）→ 量级随确认的网格在 Code 前补精确 wall-clock。
- 分析：`plot_phase2_results.py` 出 success_rate × 权重配置（按 keybuilder 分组），对齐 `exp/common/analysis/phase1/libero_spatial/` 风格。

---

## 8. 测试策略（WA §6）

新增/修改测试：
- `tests/cache/test_score_normalizers.py`（新）：
  - 每个 normalizer 的**单调性**：全局单调非减；未饱和区间严格递增（饱和/裁剪段允许平台）。
  - **保幅 / 非均匀化**：合成"近 1 极窄带"分布，断言 normalized std 显著放大、且与等距 rank 输出的差异超阈值（证明不是 RRF/CDF）。
  - 序列化 round-trip（`params_to_dict`→`from_params_dict`）。
  - registry / builder 的缺省回退（`type=none`→`DirectionUnify` 方向映射，**非恒等**：l2 须出 `exp(-d/τ)`、cosine 出 `(cos+1)/2`）。
  - **输出有界 [0,1]**：所有 registry 候选 normalizer 在随机输入上输出 ∈ [0,1]（含 ZScore 默认 squash）；锁定 squash 语义。
- `tests/cache/test_in_memory_backend_experiment.py`：现有 2 个 score_sum 测试（`type:none` l2 + `percentile`）**保持不变**，作 legacy 路径 **bit-identical 回归基线**（LegacyPercentile/DirectionUnify 须让它们零修改通过）；**新增** 两层 `per_field` 路径用例（产出预期排序 + 轨迹层归一化一致 + 入口单次构造）。
- `tests/cache/test_search_strategy_experiment.py`（改）：新 schema 透传。
- `tests/cache/test_config_*`（改/新）：`per_field` schema 校验（fields 覆盖 + method∈registry + sim_type 兼容 + params 必需键）；`weighted_score_sum_knn` 对 `type:none`/`None` **仍 reject**；旧 `{p5,p95}` percentile 向后兼容。
- `tests/exp/test_calibrate_score_normalizers.py`（新）：tiny 合成库上，方法选择能在饱和带选出"拉开带"的非线性方法、在均匀带选仿射；LOEO 过滤生效（同链 entry 不进分布）。
- `tests/weighted_sum/test_weight_search_strategy.py`（新）：`plan()` 纯逻辑（stage/依赖/纯 eval 结构 + `validate()`）；`emit_yamls` 产出的 eval YAML 含 `write_policy:{type:never}`（C2）且 `orig_init_state_idx` 与建库集合不相交；init_map 缺失时 fail-fast。

§6 Verify：`uv run pytest` 全绿；推理路径变更跑 staged API 测试。

---

## 9. 风险登记

| 风险 | 影响 | 缓解 |
|------|------|------|
| LOEO 链/episode 归组 | 校准分布污染 | **已降级**：entry `trajectory_id`(形如 `episode_XXXX`) 与 `id`(`episode_XXXX:step`) 均确认填充 → 直接按 `trajectory_id` 分组；Code 阶段加 assert 全 entry 有非空 trajectory_id，否则 fail-fast |
| Phase 2 init-state 泄漏 | live rollout 命中库内孪生 → SR 虚高不泛化 | **硬约束**：Phase 2 eval `orig_init_state_idx` 与建库集合不相交（从 `libero_spatial_init_map.json` 排除已用 init）|
| always_hit 纯回放 SR 触地板、信号弱 | 难区分权重/模态配置 | Phase 1.5 离线检索质量代理（same-task/Δstep/动作 MSE）先排序，再 GPU eval；必要时备选 warm_start judge |
| 50 ep 数据偏小，方法选择过拟合 | 选错归一化方法 | 用 robust 百分位拟合；selection 用 same/cross-task 标签；诊断图人工复核；plan 中记"小样本"caveat |
| logit/neglog 在 cos=1 处发散 | NaN/inf | 统一 `eps` clamp；fp32 计算（HDF5 是 fp16）|
| 旧 `calibration.json`(percentile) 兼容 | 破坏既有 yaml | 显式兼容路径 + 专门测试 |
| 选择指标用 rank-based 会对单调变换不变 | 无法区分方法、选择失效 | §6.4 改用**幅值+类内保真双指标**(mag_sep+β·intra_spread−λ·sat) **且只出 shortlist**，最终方法由 Phase 2 真实成功率裁决 |
| prompt_emb 任务内近常量、双峰 | affine/zscore 单带假设被打爆 | **D7：直接退出实验**，校准与搜索均不含 |
| 离线拟合与运行时 raw 分数漂移 | normalizer 参数不匹配 | 离线用与运行时同款 torch fp32 `F.cosine_similarity`/`torch.norm`，加 parity 单测 |
| 归一化改变绝对分数尺度 | 吃 `cp1_score` 的 Judge/Gate 阈值失准 | Phase 2 用 always_hit 不依赖绝对分数；后续上线需重标定下游阈值，记入文档 |
| `percentile` 旧 calib 误用 raw-vs-映射后空间 | 破坏旧 YAML 复现 | `LegacyPercentileNormalizer` 忠实复刻旧两步 + 专门 parity 测试 |
| `vision_2` 激活需改 vector_dims/重建 | 范围蔓延 | 列为可选、非关键路径，默认不动 |
| C2 frozen | normalizer 运行时只读、无写库 → 无冲突 | 仅确认无写路径 |
| 性能：每 query 构造 normalizer | 检索变慢 | builder 仅构造 dataclass（O(field 数)）；apply 全张量化，无 per-entry Python 循环 |

---

## 10. 默认项（随 G1 APPROVED 锁定）

- 实验包路径 `exp/weighted_sum/`（vs 并入 `exp/common`）。
- Phase 2 Judge = `always_hit`（隔离检索质量）。
- normalizer registry 方法集（§5.1 六种）与 selection 指标 λ=0.5。
- 是否在 Phase 1 激活 CP1 `vision_2`（默认否）。
- 旧 `calibrate_score_sum_stats.py` 迁移完成后是否删除（默认保留到 Phase 2 跑通）。

---

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-05-25 20:13 CDT

- [Blocking] [Concern] `weighted_score_sum_knn` 的 `score_normalization.type` 未知值未 fail-fast，且运行时会静默退回 `DirectionUnifyNormalizer` 旧失败模式 — reasoning: `src/openpi/cache/config.py:1261-1344` 只处理 `none` / `percentile` / `per_field`，没有对其它 `type` 加错误；同时 `src/openpi/cache/components/score_normalizers.py:540-541` 把未识别类型当作 direction-unify back-compat。G2 探针构造 `ScoreNormalizationConfig(type="typo", fields={...exp_l2...})` 后 `validate_cache_config(cfg)` 输出 `accepted_unknown_type`，说明生产 YAML typo 会绕过 §5.3 "必须用 per_field 或 percentile" 的约束，并在 backend 里恢复到 `type:none` 等价路径。修复：validator 对 `score_normalization.type not in {"per_field","percentile","none"}` 明确报 `ConfigValidationError`，并给 config 测试加 unknown-type 用例；运行时 builder 也应对非兼容路径未知类型 raise，而不是静默 fallback。
- [Blocking] [Concern] §5.5/§7 锁定的两个小 JSON 真源仍被 `.gitignore` 忽略，复现所需输入不会随实现入库 — reasoning: 计划要求为 `exp/weighted_sum/data/calibration_normalizers.json` 与 `exp/common/data/db/libero_cache/libero_spatial_init_map.json` 新增 `.gitignore` 例外；但 G2 执行 `git check-ignore -v exp/weighted_sum/data/calibration_normalizers.json exp/common/data/db/libero_cache/libero_spatial_init_map.json` 返回二者都命中 `.gitignore:6:exp/**/data/**`，且当前工作树没有 `.gitignore` 修改。这样 Phase-1 normalizer artifact 与 Phase-2 init-state leak guard 的源数据仍不能被跟踪，违反已批准的 commit boundary。修复：按计划添加精确 `.gitignore` 例外并确保需要跟踪的小 JSON 能被 `git status` 看见；若暂不生成 `calibration_normalizers.json`，也至少要提交 ignore 例外和可复现生成/缺失 fail-fast 路径。

Verification:
- `PYTHONPATH=. uv run pytest tests/cache/test_score_normalizers.py tests/cache/test_in_memory_backend_experiment.py tests/cache/test_config.py tests/exp/test_calibrate_score_normalizers.py tests/weighted_sum/test_weight_search_strategy.py -q` → 162 passed, 8 warnings.
- G2 config probe: unknown `score_normalization.type="typo"` currently accepted.
- G2 ignore probe: both required JSON paths still ignored by `.gitignore:6`.

### G2 Round 1 — Executor — 2026-05-25

- [Item 1 / unknown-type fail-fast] Accepted — 三处收口：① `config.py` validator 的 `weighted_score_sum_knn` 分支加 `else`，对 `score_normalization.type ∉ {per_field, percentile, none}` 报 `ConfigValidationError`（`none` 仍单独 reject）；② `score_normalizers.build_field_normalizers` 把未知 `type` 从静默 `DirectionUnifyNormalizer` 改为 `raise ValueError`（`none` 仍走 DirectionUnify 的 backend/spec 级兼容）；③ `in_memory_backend` 两条轨迹入口仅 `fusion_method=="weighted_score_sum"` 才构造 normalizer（避免 RRF 路径无谓构造或对 stray type 误 raise）。新增 `test_score_sum_rejects_unknown_type`（config）+ `test_build_unknown_type_raises_not_silent_fallback`（builder）两个回归测试。
- [Item 2 / .gitignore 真源未跟踪] Accepted — `.gitignore` 按 plan §5.5 加精确例外（此 §7 高危操作经 owner "依法修改" 指令授权，已显式标注）：`!exp/weighted_sum/data/calibration_normalizers.json` + `!exp/common/data/db/` + `!exp/common/data/db/libero_cache/` + `!exp/common/data/db/libero_cache/libero_spatial_init_map.json`。`git check-ignore -v` 确认二者命中的是 `!` 负规则（已解禁），`git status` 现可见 `?? .../libero_spatial_init_map.json`（待 commit 由 owner stage，符合"不擅自 git add"）。`calibration_normalizers.json` 尚未生成（Phase 1 跑时产出），ignore 例外已就位使其届时可跟踪；缺失 `init_map` 的 fail-fast 路径在 `init_holdout.load_used_inits`（含 `test_load_used_inits_missing_fails_fast`）。

### G2 Round 2 — Reviewer — APPROVED — 2026-05-25 20:22 CDT

- [Resolved] Unknown `score_normalization.type` now fails fast in both config validation and runtime builder paths — evidence: `config.py` now has an explicit unknown-type `else` under `weighted_score_sum_knn`; `score_normalizers.build_field_normalizers` raises for unknown types while preserving `type:"none"` only for backend/spec compatibility. Independent G2 probe with `ScoreNormalizationConfig(type="typo", ...)` now raises `ConfigValidationError: unknown score_normalization.type 'typo'`.
- [Resolved] Required small JSON source-of-truth paths are unignored without exposing large run data — evidence: `git check-ignore -v exp/weighted_sum/data/calibration_normalizers.json exp/common/data/db/libero_cache/libero_spatial_init_map.json` now reports the `.gitignore` `!` exceptions; `git ls-files --others --exclude-standard exp/common/data/db` shows only `libero_spatial_init_map.json`, while representative HDF5/MP4 files still match `.gitignore:6:exp/**/data/**`.

Verification:
- `PYTHONPATH=. uv run pytest tests/cache/test_score_normalizers.py tests/cache/test_in_memory_backend_experiment.py tests/cache/test_config.py tests/exp/test_calibrate_score_normalizers.py tests/weighted_sum/test_weight_search_strategy.py -q` → 164 passed, 8 warnings.
- Direct unknown-type config probe → rejected with `ConfigValidationError`.
