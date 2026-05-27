# Verdict Factor Judge 使用指南

> ⚠️ **2026-05-07 — 重构 (G1 APPROVED Round 4) 已 land**：
>
> - **5 → 17 因子扁平化**：`f1a_a / f1a_t / f1b_a / f1b_t / f2` **全部废除**；新名 `<descriptor>_<source>_<channel>`：
>   - 4 descriptor：`jerk` / `direction` (旧 `dir`) / `dispersion` (旧 `curv_radius`) / `path_length` (旧 `cum_disp`)
>   - 4 变体 = 2 source × 2 channel：`online | offline` × `action | state` → 16 个；外加 `topk_action_variance`
> - **Judge 4 层正交架构**：Normalization → Factor → Calibration → Composer，每层 yaml 独立可插拔
> - **No cold-start**：第 1 / 第 3 层启动时校准数据**必备**，缺失即 fail-fast；旧 `cold_start_strategy: force_miss/passthrough/lenient` + `all_nan_fallback` + `JudgeResult.factor_outputs.sentinel` 全部移除
> - **诊断 schema_version=2**：`factor_outputs.{raw, calibrated, composer_score}`（旧 `norm` / `score` / `sentinel` 字段废除）
> - 完整设计 + 决策见 [`logs/verdict_factor_judge_refactor.log.md`](../../logs/archive/verdict_factor_judge_refactor.log.md)（G1 APPROVED Round 4，2026-05-07）。
> - 历史设计文档（重构前）见 `logs/old_verdict_factor_*.log.md` 8 份归档。

> **前置知识**：阅读 [tutorial.md](tutorial.md) §6 了解 Judge 组件基础，§10 了解 YAML 配置；阅读 [../architecture/cache_system.md](../architecture/cache_system.md) §5.12 / §5.13 了解 verdict factor 系统的架构契约。

---

## 1. 概述（refactor 4 层架构）

`CompositeJudge` 把命中决策拆成 **4 个正交、可插拔的层**：

```
                    ┌──────────────────────────────────────┐
原始 action / state │ 第 1 层  Normalization               │
─────────────────►  │   ZScoreNormalization                │
                    │   stats_source: offline (LibraryStats)│
                    └──────────────┬───────────────────────┘
                                   │ normalized data 注入
                                   ▼
SearchResultLite[]  ┌──────────────────────────────────────┐
PayloadView         │ 第 2 层  17 个 Factor                │
HistoryView      ──►│   每因子 Factor.extract(ctx) -> raw   │
                    │   FactorContext = {results, view,    │
                    │     history, normalization}          │
                    └──────────────┬───────────────────────┘
                                   │ raw factor dict
                                   ▼
                    ┌──────────────────────────────────────┐
                    │ 第 3 层  Calibration (per-key)        │
                    │   PercentileRollingCalibration       │
                    │   samples_source: offline | warmup   │
                    │   bind_keys() 启动时 fail-fast        │
                    └──────────────┬───────────────────────┘
                                   │ calibrated dict in [0, 1]
                                   ▼
                    ┌──────────────────────────────────────┐
                    │ 第 4 层  Composer                     │
                    │   declared_dependencies 实例属性     │
                    │   compose(calibrated, *, winner_id)  │
                    │   子类自带 fallback 逻辑             │
                    └──────────────┬───────────────────────┘
                                   │
                                   ▼
                    JudgeResult(hit_type, winner_id, start_t,
                                factor_outputs={schema_version:2,
                                                raw, calibrated,
                                                composer_score})
```

**关键不变量**：
- 每层互不知道彼此实现（只通过约定 dataclass 通讯）。
- NaN 唯一合法来源 = 因子物理边界（`history` 不够 P 步、`walk_next` 走完 < F 步、winner 数据缺失）；**不**来自 calibration cold-start（启动时 buffer 必满）+ 不来自 normalization σ 缺失（启动 fail-fast）。
- 空 `results` → CompositeJudge **直接** MISS（不交给 composer，避免 winner_id=None + WARM_START 的歧义）。
- composer 子类**自决** NaN 处理：`WeightedSumWithWarmFallbackComposer` 在所有非零权重 key 都 NaN 时 → WARM_START（承接旧 `all_nan_fallback` 语义）。

---

## 2. 17 因子扁平化总览

每个因子是一个独立 registry 名；按 4 维笛卡尔 `descriptor × source × channel` 展开 16 + 1 个 topk = 17。

**4 个运动学描述子（公式不变，名字部分改）**：

| 新描述子 | 旧描述子 | 取向 | 公式 | 含义 |
|---|---|---|---|---|
| `jerk` | `jerk` | risky | `median \|Δ²a\|`（z-score 后 active-DOF 上 median，再 mean） | 加速度变化幅度，高 → 不平滑 |
| `direction` | `dir` | safe  | `mean cos(v[t], v[t+1])` | 方向一致性，高 → 平滑 |
| `dispersion` | `curv_radius` | non_monotonic | `mean ‖p[t] − centroid‖` | 窗口几何弥散度（中等 = 圆弧，极小 = 停滞，极大 = 大幅直线） |
| `path_length` | `cum_disp` | non_monotonic | `sum ‖p[t+1] − p[t]‖` | 累积路径长度（小 + 低 jerk = 静止；大 = 快速移动） |

**4 个变体维度（每 descriptor × 4 = 16 个独立 factor）**：

| 维度 | 取值 | 含义 |
|---|---|---|
| source | `online` | verdict 时算（拼 `[history[-P:], winner, walk_next(F)]` splice + 调 `ctx.normalization.normalize_*`） |
| source | `offline` | artifact build 时由 `OfflineWriter.compute_for_episode` 离线算并写入 `payload.factors`；online verdict 时直接读 `winner.payload.factors` |
| channel | `action` | 取链上 `payload.action_chunk[0]` 序列 |
| channel | `state` | 取链上 `query_keys["robot_state"]` 序列 |

**第 17 个因子**：

| 注册名 | 类 | 数据源 | requires_chain_walk | required_top_k |
|---|---|---|---|---|
| `topk_action_variance` | `TopkActionVariance` | top-K 候选的 `action_chunk[0]` per-DOF 方差（candidate-local active mask） | False | K |

**Online 因子统一 splice**（`<descriptor>_online_<channel>`，共 8 个，`requires_chain_walk=True`）：

```
splice = [history[-P:],   winner,   walk_next(F).<channel>]
              ^               ^         ^
              past            anchor    future
              P 个执行点       winner    F 个沿 chain 下游
```

- `online_action`：history.actions[-P:] + winner.action_chunk[0] + walk_next(F).payload.action_chunk[0]
- `online_state`： history.states[-P:]  + winner.query_keys["robot_state"] + walk_next(F).query_keys["robot_state"]
- splice 第 t 步的 state 与 action 来自**同一次推理**（chain entry t 的 query_keys + payload）。

**Offline 因子**（`<descriptor>_offline_<channel>`，共 8 个，`requires_chain_walk=False`）：

- `OfflineWriter.compute_for_episode(entries, library_stats)` 在 artifact build 时滑窗算每 entry 的描述子，写入 `entry.payload.factors[<key>]`。
- Verdict 时只读 `winner.payload.factors[<key>]`，没有 chain walk。
- key 模板：`<descriptor>_offline_<channel>__p<P>_f<F>`。

**capability 约束**：含任意 `requires_chain_walk=True` 因子的 yaml → `backend.type` 必须 == `"in_memory"`（唯一暴露 `fetch_entry` 的 backend）。yaml load 时静态校验。

`risky` / `safe` Composer 自动按取向 flip 贡献分；`non_monotonic` 必须在 `composer.directions` 显式指定 `"high"` / `"low"` / `"range:[lo,hi]"`，否则 validator 拒收。

---

## 3. yaml 配置示例（refactor 4 层 schema）

```yaml
checkpoints:
  cp1:
    enabled: true
    gate: { type: always_search }

    judge:
      type: composite

      # ── 第 1 层 ──
      normalization:
        type: zscore
        params: {}
        stats_source:
          type: offline                                 # 当前唯一可选；warmup 通道未实现
          # offline 模式：σ + active_mask 自动从 backend.load_artifact 的 library_stats 读

      # ── 第 2 层 ──（17 因子任意子集）
      factors:
        - type: jerk_online_state
          params:
            windows:
              - { past: 5, future: 5 }
              - { past: 7, future: 7 }
        - type: dispersion_offline_state
          params:
            windows: [{ past: 5, future: 5 }]
        - type: topk_action_variance
          params: { K: 5 }

      # ── 第 3 层 ──
      calibration:
        type: percentile_rolling
        params: { window_size: 50 }
        samples_source:
          type: warmup                                   # | offline
          # warmup: 从 WarmupPool[eval_yaml_id] 读（sibling warmup yaml 必须先跑过）
          # offline 替代：
          # offline: { path: data/calibration/spatial16_v2.jsonl, format: jsonl }

      # ── 第 4 层 ──
      # ComposerConfig 是扁平 schema —— ``weights`` / ``tier_thresholds`` /
      # ``warm_start_t`` / ``warm_fallback_start_t`` / ``directions`` /
      # ``per_factor_thresholds`` 直接挂在 composer 顶层，**不在** ``params`` 下面。
      composer:
        type: weighted_sum_with_warm_fallback           # | weighted_sum / and / or
        weights:
          jerk_online_state__p5_f5:        1.0
          jerk_online_state__p7_f7:        1.0
          dispersion_offline_state__p5_f5: 1.0
          topk_action_variance:            0.5
        tier_thresholds:
          full_hit:    0.30
          warm_start:  0.10                             # 可选；触发常规 warm tier
        warm_start_t: 0.7                               # 与 tier_thresholds.warm_start 配对
        warm_fallback_start_t: 0.7                      # 子类内部 fallback (旧 all_nan_fallback 等价)
        directions:                                     # non_monotonic key 必填
          dispersion_offline_state__p5_f5: "range:[0.3, 0.7]"

      export_factor_outputs: true                       # 默认 false；true 时 schema_version=2
```

**关键 validator 规则**（`config.py` 在 yaml load 时执行）：

1. `judge.type=="composite"` 必须含 `normalization` / `factors` / `calibration` / `composer` 四块。
2. 每个 `factors[].type` 必须 ∈ 17 注册名。旧 5 名 (`f1a_a`/`f1a_t`/`f1b_a`/`f1b_t`/`f2`) 加载时 `Unknown factor name` reject。
3. composer 实例的 `declared_dependencies` 必须 ⊆ Layer 2 union key 集合。
4. 含 `requires_chain_walk=True` 因子时，`backend.type=="in_memory"` 必须成立。
5. non_monotonic key 进 composer 时 `composer.params.directions` 必须显式覆盖。
6. `normalization.stats_source.type` 当前必须 == `"offline"`；`calibration.samples_source.type ∈ {"offline", "warmup"}`。
7. 旧字段 `judge.normalizer` / `judge.all_nan_fallback` / `judge.cold_start_strategy` 在 yaml 中出现 → load 时报错并提示"legacy schema, rewrite to 4-layer"。

---

## 3.1 多因子 / Online+Offline 组合 yaml

17 因子是**扁平**的：每个 `<descriptor>_<source>_<channel>` 是一个独立 registry 名。任何子集都可以在同一份 yaml 的 `factors:` 列表里组合，只要每条 entry 各占一行 + 在 `composer.weights`（或 `per_factor_thresholds`）里给对应的 `__p<P>_f<F>` key 分配权重。

**关键合同**：

- `factors:` 是个列表，每条元素 `{type: <registry-name>, params: {...}}`
- 即便是同一个 descriptor 的 online + offline 版本也是**两条独立 entry**（registry 名不同，互相独立）
- 同一类型的多窗口可以放在**一条** entry 的 `params.windows: [...]` 里（17 因子的每一个原生支持多窗口）
- composer 的 `weights` / `per_factor_thresholds` 必须 reference 由这些 factor 的 `describe(params)` 推出来的所有 key（多窗口 → 多 key，每窗口一条）

### 3.1.1 多 descriptor 组合（单一 source × channel）

例：state 通道、online 来源，同时启用 jerk + direction 两个 descriptor，每个一个窗口。

```yaml
factors:
  - type: jerk_online_state
    params:
      windows: [{past: 5, future: 5}]
  - type: direction_online_state
    params:
      windows: [{past: 5, future: 5}]

composer:
  type: weighted_sum
  weights:
    jerk_online_state__p5_f5:      1.0      # risky → 自动 1-v
    direction_online_state__p5_f5: 1.0      # safe  → 直接 v
  tier_thresholds: { full_hit: 0.5 }
```

### 3.1.2 同一 descriptor 同时用 online + offline

例：jerk 在 state 通道上同时启用 online（运行时拼 splice 算）+ offline（artifact build 时按 chain 算好读出来）—— 两者是两个独立 factor entry，**不能合并**到一条。

```yaml
factors:
  - type: jerk_online_state              # 运行时 splice [history[-5:], winner, walk_next(5)]
    params:
      windows: [{past: 5, future: 5}]
  - type: jerk_offline_state             # 离线 build 时按 chain 滑窗写 payload.factors
    params:
      windows: [{past: 5, future: 5}]

composer:
  type: weighted_sum_with_warm_fallback
  weights:
    jerk_online_state__p5_f5:  1.0
    jerk_offline_state__p5_f5: 1.0       # 与 online key 同窗口但 source 不同 → 不同 key
  tier_thresholds: { full_hit: 0.5 }
  warm_fallback_start_t: 0.7             # 全 NaN（chain 走完 + offline 没写）→ WARM_START
```

> 两个 key 的物理含义：`jerk_online_state__p5_f5` 是"verdict 时拼 history+walk_next 算的 jerk"；`jerk_offline_state__p5_f5` 是"artifact build 时这个 entry 在它自己 chain 上 (P=5, F=5) 窗口的 jerk"。它们对应**同一物理量的两种估计**，calibration 会用各自独立的 percentile 桶。

### 3.1.3 同一 factor 多窗口 + 多 factor 多 source × channel 全开

例：4 desc × 2 source × 2 channel = 16 + topk = 17 全开，spatial16 上跑全空间扫描。

```yaml
factors:
  # 8 个 online 因子，每个 2 个窗口
  - { type: jerk_online_action,        params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: jerk_online_state,         params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: direction_online_action,   params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: direction_online_state,    params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: dispersion_online_action,  params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: dispersion_online_state,   params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: path_length_online_action, params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: path_length_online_state,  params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  # 8 个 offline 因子，同窗口
  - { type: jerk_offline_action,        params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: jerk_offline_state,         params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: direction_offline_action,   params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: direction_offline_state,    params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: dispersion_offline_action,  params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: dispersion_offline_state,   params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: path_length_offline_action, params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  - { type: path_length_offline_state,  params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
  # 第 17 个：top-K 候选共识方差（无窗口、单 key）
  - { type: topk_action_variance, params: { K: 5 } }

composer:
  type: weighted_sum
  weights:
    # 16 desc 因子 × 2 窗口 = 32 个 key（uniform 1.0）+ topk 单独权重
    jerk_online_action__p5_f5:        1.0
    jerk_online_action__p7_f7:        1.0
    jerk_online_state__p5_f5:         1.0
    jerk_online_state__p7_f7:         1.0
    # ... (省略其余 28 个 desc key，写法同上 uniform 1.0) ...
    topk_action_variance:             0.5   # 不同权重也 OK
  tier_thresholds: { full_hit: 0.30 }
  directions:                                 # non_monotonic key (dispersion / path_length) 必须给方向
    dispersion_online_state__p5_f5:  "range:[0.3, 0.7]"
    path_length_online_state__p5_f5: "high"
    # ... 所有 non_monotonic key 都要列 ...
```

### 3.1.4 多 factor 走 AndGate / OrGate

`weighted_sum` 用 `weights`；`and` / `or` 用 `per_factor_thresholds`，每 key 独立判定通过。

```yaml
factors:
  - { type: jerk_online_state,        params: { windows: [{past: 5, future: 5}] } }
  - { type: dispersion_offline_state, params: { windows: [{past: 5, future: 5}] } }

composer:
  type: and          # 两个 key 都要过自己的阈值才 FULL_HIT
  per_factor_thresholds:
    jerk_online_state__p5_f5:        0.3   # risky → v <= 0.3
    dispersion_offline_state__p5_f5: 0.5   # non_monotonic → 用 directions 决定通过条件
  directions:
    dispersion_offline_state__p5_f5: "range:[0.3, 0.7]"
  warm_start_t: 0.7    # 可选：通过即 WARM_START（CP1-only）
```

### 3.1.5 Warmup yaml 必须 over-collect 上述所有 key

每个 eval yaml 都需要一份 sibling warmup yaml 来填 Layer 3 calibration 的 rolling buffer。warmup 的 `judge.dump.factors` 必须**至少**覆盖 eval composite 用到的所有 key 集合（否则 eval `bind_keys` fail-fast）：

```yaml
# <eval_yaml_id>__warmup.yaml 中（generator 通常自动产；手写见下）
checkpoints:
  cp1:
    judge:
      type: always_warm_start
      start_t: 0.7
      dump:
        deferred: true
        config_id: <eval_yaml_id>__warmup
        factors:
          # ⚠ 必须覆盖 eval factors 的全部 key —— 一对一镜像 eval 的 factors 列表，
          #    或者直接 over-collect 17 因子全集（推荐：之后切换 eval 因子组合无需重跑 warmup）
          - { type: jerk_online_state,        params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
          - { type: jerk_offline_state,       params: { windows: [{past: 5, future: 5}, {past: 7, future: 7}] } }
          # ... (其余 15 个) ...
          - { type: topk_action_variance, params: { K: 5 } }
```

实际写 warmup 不用手动维护—— `exp/verdict_factor_judge/v2_spec.py:build_warmup_yaml(cfg_id, eval_yaml_id, eval_factors=...)` 接受 eval factors 列表，自动 union 出 dump factor superset（每 (descriptor, source, channel) 把 eval 的窗口 + `_W_UNION_DEFAULT` 默认窗口都收进去）。

---

## 4. 因子精确数学表达式

> 4 个 descriptor 公式与重构前完全一致；只重命名 + 扁平化。formula source: `src/openpi/cache/components/factors/_descriptor_kernel.py`。

### 4.1 通用记号

| 符号 | 含义 |
|---|---|
| `seq` | 单窗口在 active subspace 上的 z-scored 点序列，shape `[W, D_act]` |
| `v[t]` | `seq[t+1] - seq[t]`（一阶差分），shape `[W-1, D_act]` |
| `j[t]` | `seq[t+1] - 2 seq[t] + seq[t-1]`（二阶差分），shape `[W-2, D_act]` |
| `D_act` | LibraryStats 的 active mask 上为 True 的维度数（z-score 后剔除 padded DOF） |

z-score：第 1 层 Normalization 做 `seq / sigma.clamp_min(eps=0.01)` 然后选 active mask；factor 层只看 `seq` 已经 normalized。

### 4.2 4 个 descriptor 公式

#### `jerk` (risky)
```
jerk = mean_DOF(   median_t(|j[t]|)   )
```
按时间维取 median（吸收 gripper 的单帧 spike）后按 DOF 取 mean。NaN 条件：`j` 为空（W < 3）。

#### `direction` (safe)
```
direction = mean_t(  cos(v[t], v[t+1])  )    // 仅在两端 norm > 0 的项上
```
相邻速度向量的余弦均值。NaN 条件：`v` 长度 < 2（W < 3）；所有相邻速度对至少有一端为零。

#### `dispersion` (non_monotonic)
```
dispersion = mean_t(  ‖seq[t] - centroid(seq)‖  )
```
窗口几何弥散度（点到质心的平均距离）。NaN 条件：W < 2。

#### `path_length` (non_monotonic)
```
path_length = sum_t(  ‖seq[t+1] - seq[t]‖  )
```
累积步长。NaN 条件：W < 2。

### 4.3 17 因子各自的 splice 形状

#### Online (8 因子)
```
seq = [history[-P:], winner, walk_next(F).<channel>]      # 长度 P + 1 + F
```
- channel = `action`：history.actions[-P:] + winner.action_chunk[0] + walk_next 链上 action_chunk[0]
- channel = `state`： history.states[-P:]  + winner.query_keys["robot_state"] + walk_next 链上 robot_state

NaN 物理边界：
- `len(history) < P`（episode 早期）
- `walk_next` 走完 < F 个（chain 末端 / fork detected）
- winner 缺 robot_state（state 通道）

#### Offline (8 因子)
verdict 时直接读 `winner.payload.factors[<key>]`（artifact build 时已写）。
- 离线 build 时：扫整个 chain，对每 entry × 每 (P, F) 窗口在 chain 上做滑窗算 descriptor，结果写入 `entry.payload.factors`。
- 边界：窗口超出 chain 两端时该 (entry, 窗口) 的 key 写 NaN。

#### `topk_action_variance`（第 17 个）
```
var = mean(  per_DOF_var(  [r.payload.action_chunk[0] for r in results[:K]]  )  )
                                                                ↑
                                                  candidate-local active mask
                                                  (per-DOF var > 1e-8)
```
NaN 条件：`results < K`（搜索没返回足够候选）；候选池每 DOF 都一致（pool 完全同意 → 没有 active dim）。

不调 Layer 1 Normalization：变量是 candidate-pool 方差，scale-invariant；Pi0.5 padded DOF 全 0 自动通过 candidate-local mask 剔除。

---

## 5. 全周期流程

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1   Build artifact (one-off, GPU 重活)                  │
│   build_in_memory_cache_artifact.py: HDF5 → .pkl            │
│   含 entries + LibraryStats（σ + active_mask）              │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│ Step 2   Enrich pkl with offline factors (秒级 smoke)        │
│   build_in_memory_cache_artifact.py enrich-existing-pkl     │
│   --input old.pkl --factors-yaml factors.yaml --output new.pkl│
│   ↑ 新增 17 因子 keys 进 payload.factors，不重算 σ          │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│ Step 3   Warmup yaml run (collect calibration samples)       │
│   <eval>__warmup.yaml + DumpingJudge                         │
│   → JSONL of factor raw values                               │
│   → preload_normalizer_buffer ctrl → WarmupPool[eval_yaml_id]│
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│ Step 4   Eval yaml run                                       │
│   load_cache_config: validator + _build_calibration 拉      │
│      WarmupPool[eval_yaml_id] / offline jsonl 灌满 buffer    │
│      → bind_keys fail-fast 校验每 key 样本数 ≥ window_size   │
│   每 verdict: Layer 1 → Factor → Calibration → Composer     │
│   每 verdict: 可选 export_factor_outputs 写 schema_v=2 jsonl │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Step 1+2: 构建 / 增强 Artifact

### 6.1 从 HDF5 build（首次）
```bash
uv run python -m exp.common.build_in_memory_cache_artifact \
  --data-dir exp/common/data/db_init/libero_cache/libero_spatial \
  --builder-type cp1_spatial_pool_16 \
  --output exp/warm_start/data/spatial16/cp1_spatial_pool_16.pkl \
  --workers -1 \
  --factors-yaml my_factors.yaml             # 同时 enrich 17 因子 keys
```

### 6.2 已有 pkl 增量加因子（秒级 smoke）
```bash
uv run python -m exp.common.build_in_memory_cache_artifact enrich-existing-pkl \
  --input  exp/warm_start/data/spatial16/cp1_spatial_pool_16.pkl \
  --factors-yaml my_factors.yaml \
  --output exp/warm_start/data/spatial16/cp1_spatial_pool_16_v2.pkl
```
关键点（plan §16 B6.5）：
- 复用 input pkl 的 `library_stats`（**不重算 σ**），smoke 几秒完成
- 旧 `payload.factors` keys 保留（additive merge），新 17 因子 keys 加入
- 新 pkl 与 input pkl 平行存在，input 不被覆盖（rollback friendly）

### 6.3 `factors.yaml` 形状（offline 因子专用）
```yaml
factors:
  - type: jerk_offline_action
    params:
      windows:
        - {past: 1, future: 1}
        - {past: 5, future: 5}
  - type: dispersion_offline_state
    params:
      windows: [{past: 5, future: 5}]
```
仅接受 8 个 offline 因子（`<descriptor>_offline_<channel>`）；online + topk 没 `compute_for_episode` 接口，传它们会被 `_load_offline_writers_from_yaml` 显式 reject。

---

## 7. Step 3+4: yaml 配置 + 跑推理

完整 4 层 yaml 已在 §3 给出。本节列字段细节。

### 7.1 第 1 层 Normalization
```yaml
normalization:
  type: zscore                  # 唯一注册 type
  params: {}                    # ZScoreNormalization 当前无 param
  stats_source:
    type: offline               # 唯一可选；warmup 通道 deferred
```
启动校验：`backend.library_stats` 必须可达（`backend.type == "in_memory"`）。

### 7.2 第 2 层 Factors
每 factor 一项 `{type, params}`。typical params：
- 8 online + 8 offline：`windows: [{past, future}, ...]`
- `topk_action_variance`：`K: int >= 2`

### 7.3 第 3 层 Calibration
```yaml
calibration:
  type: percentile_rolling      # 唯一注册 type
  params:
    window_size: 50             # rolling buffer 大小
  samples_source:
    type: warmup                # | offline
    # warmup: 从 WarmupPool[eval_yaml_id] 拉（sibling warmup yaml 必须先跑）
    # offline 替代：
    # offline:
    #   path: data/calib/spatial16_v2.jsonl
    #   format: jsonl           # | pkl
```
`bind_keys` 时 fail-fast 检查：每个第 2 层 union key 必须在 samples 中有 ≥ window_size 个非 NaN 样本，否则 yaml load 失败。

### 7.4 第 4 层 Composer

> **schema 形状**：`ComposerConfig` 是扁平结构 —— 所有关键字段（`weights` / `tier_thresholds` / `per_factor_thresholds` / `warm_start_t` / `warm_fallback_start_t` / `directions`）直接挂在 `composer:` 顶层，**不在** `composer.params:` 下面。yaml parser 会忽略并 warning 任何 `composer.params.*`，validator 随后多半会因为缺 `tier_thresholds.full_hit` 等必填字段而 reject。

注册的 4 个子类：

| type | 关键字段（顶层） | hit 条件 |
|---|---|---|
| `weighted_sum` | `weights` / `tier_thresholds: {full_hit, warm_start?}` / `warm_start_t` | 加权（取向 flip 后）求和 ≥ full_hit → FULL_HIT；≥ warm_start → WARM_START（CP1-only） |
| `weighted_sum_with_warm_fallback` | 上述 + `warm_fallback_start_t` | 同上；当所有非零权重 key 都 NaN → WARM_START @ warm_fallback_start_t（承接旧 `all_nan_fallback`）。**`warm_fallback_start_t` 也是 WARM_START 发射路径，与 `warm_start_t` 一样仅 CP1 支持** |
| `and` | `per_factor_thresholds` / `warm_start_t?` | 每 key 都过阈值 → FULL_HIT（或 WARM_START 如配 `warm_start_t`，CP1-only）|
| `or` | 同上 | 任一 key 过阈值 → FULL_HIT / WARM_START |

`directions` 字段：`non_monotonic` orientation 的 key 必须在 `directions` 中显式给 `"high"` / `"low"` / `"range:[lo, hi]"`，否则 yaml load 时 reject。

> **CP3 限制**（plan §3.6 / validator §13.3 规则 5c）：CP3 没有 `intermediates` payload 可以 resume，所以**任何** WARM_START 发射路径在 CP3 上都被 yaml validator reject —— 这包括 `warm_start_t`（常规 warm tier）以及 `weighted_sum_with_warm_fallback` 子类的 `warm_fallback_start_t`（all-NaN fallback）。CP3 上的 composite judge 必须省略这两个字段。

### 7.5 诊断字段 `factor_outputs` (schema_version=2)

YAML 顶层 `judge.export_factor_outputs: true` 时，每 verdict 在 `JudgeResult.factor_outputs` 写：
```python
{
  "schema_version": 2,
  "raw":             {key: float | None},     # Layer 2 raw
  "calibrated":      {key: float | None},     # Layer 3 输出
  "composer_score":  float | None,            # Layer 4 内部分数
}
```
NaN 在 wire 上转 None（JSON-strict 兼容）。`hit_type / winner_id / start_t` 仍在 `JudgeResult` 顶层（不进 `factor_outputs`）。

---

## 8. 自定义扩展（如何编写自己的因子 / Calibration / Composer）

> 本节是 step-by-step 教程：从写代码 → 注册 → yaml 引用 → 单元测试 → 启动 server。每一节末尾的 yaml 片段直接可粘贴到 §3 的完整 4 层 yaml 模板里。

### 8.1 新增第 2 层 Factor

#### 第 1 步：决定因子的 capability 与命名

| 决策点 | 取值规则 | 影响 |
|---|---|---|
| 命名 | 推荐沿用 `<descriptor>_<source>_<channel>` 模式（见 §2），自定义因子可自由命名但要避开 17 因子保留名 | 注册名也是 yaml `factors[].type` 的字面量 |
| `requires_chain_walk` | 调用 `ctx.view.walk_prev / walk_next` 即 True | yaml load 时若 backend 不支持 `fetch_entry` 即 reject |
| `required_top_k` | 因子至少需要的 top-K 候选数；不需要就 0 | CompositeJudge 取所有 factor 的 max 反向喂给 search strategy 的 `min_top_k_hint` |
| 描述子 key 与取向 | `"safe"`（高 = 好）/ `"risky"`（高 = 坏）/ `"non_monotonic"`（必须在 composer 配 `directions`） | composer 自动按取向 flip（safe → 直接用，risky → `1-v`） |

#### 第 2 步：实现类

放在一个会被 `factors/registry.py` import 的模块下（最简：直接放 `factors/online.py` 末尾，或新建 `factors/my_pack.py` 然后在 `registry.py` 加一行 import）。完整骨架：

```python
# src/openpi/cache/components/factors/my_pack.py

"""My custom factor — describe what physical signal this captures."""

from __future__ import annotations

import math

import torch

from openpi.cache.components.factors.base import FactorContext
from openpi.cache.components.factors.registry import register


@register("my_action_burstiness")
class MyActionBurstiness:
    """Mean burst-ratio over the last K executed actions.

    Returns a single risky-orientation key:
        my_action_burstiness__k<K>: float in [0, 1]   (or NaN at boundary)
    """

    # ---- class-level capability flags ----
    requires_chain_walk: bool = False     # only reads history.actions
    required_top_k:      int = 0          # not a candidate-pool factor

    # ---- constructor ----
    def __init__(self, *, K: int) -> None:
        if K < 2:
            raise ValueError(f"MyActionBurstiness K must be >= 2, got {K}")
        self.K = int(K)
        self.descriptor_orientations = self.__class__.describe({"K": K})

    # ---- pure metadata classmethod (validator calls this WITHOUT instantiating) ----
    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        K = int(params["K"])
        return {f"my_action_burstiness__k{K}": "risky"}

    # ---- per-verdict extraction ----
    def extract(self, ctx: FactorContext) -> dict[str, float]:
        key = next(iter(self.descriptor_orientations))
        if len(ctx.history.actions) < self.K:
            return {key: float("nan")}                       # boundary

        # Pull last K actions, normalize via Layer 1 (z-score).
        seq = torch.stack(
            [torch.as_tensor(a, dtype=torch.float32) for a in ctx.history.actions[-self.K:]],
            dim=0,
        )                                                    # [K, A]
        normed = ctx.normalization.normalize_action(seq)     # [K, A_active]
        if normed.shape[-1] == 0:
            return {key: float("nan")}                       # empty active mask

        # Burst ratio = fraction of consecutive-step diffs whose magnitude
        # exceeds the median diff magnitude. Risky: high = bursty.
        diffs = (normed[1:] - normed[:-1]).norm(dim=-1)      # [K-1]
        if diffs.numel() < 2:
            return {key: float("nan")}
        median = float(diffs.median())
        ratio = float((diffs > median).float().mean())
        return {key: ratio}
```

#### 第 3 步：让 registry 在 import 时拿到它

在 `factors/registry.py` 末尾加一行 import：

```python
from openpi.cache.components.factors import my_pack  # noqa: F401
```

任何 `from openpi.cache.components.factors import registry` 都会触发这行 import，`@register` 副作用立即把因子注册到全局 registry。

#### 第 4 步：yaml 中引用

```yaml
checkpoints:
  cp1:
    judge:
      type: composite
      normalization: { type: zscore, params: {}, stats_source: { type: offline } }
      factors:
        - type: my_action_burstiness
          params: { K: 5 }
      calibration:
        type: percentile_rolling
        params: { window_size: 50 }
        samples_source:
          type: warmup            # warmup yaml 必须 over-collect 这个 key
      composer:
        type: weighted_sum
        weights: { my_action_burstiness__k5: 1.0 }
        tier_thresholds: { full_hit: 0.7 }   # 高 burst-ratio = miss → 高 1-v = full_hit
```

#### 第 5 步：单元测试

```python
# tests/cache/components/factors/test_my_action_burstiness.py
import math
import pytest
import torch

from openpi.cache.components.factors.base import FactorContext, HistoryView, LibraryStats
from openpi.cache.components.factors.my_pack import MyActionBurstiness
from openpi.cache.components.factors.normalization import ZScoreNormalization
from openpi.cache.storage_types import SearchResultLite
from openpi.cache.types import CheckpointID

def _ctx(history_actions):
    a = torch.ones(2, dtype=torch.float32)
    s = torch.ones(2, dtype=torch.float32)
    ls = LibraryStats(
        action_sigma=a, action_active_mask=torch.ones(2, dtype=torch.bool),
        state_sigma=s, state_active_mask=torch.ones(2, dtype=torch.bool),
    )
    return FactorContext(
        results=[SearchResultLite(id="w", score=1.0, checkpoint_id=CheckpointID.CP1)],
        view=None, normalization=ZScoreNormalization(ls),
        history=HistoryView(
            actions=[torch.tensor(a, dtype=torch.float32) for a in history_actions],
            states=[],
        ),
    )

def test_K_below_2_rejected():
    with pytest.raises(ValueError, match=">= 2"):
        MyActionBurstiness(K=1)

def test_history_too_short_emits_nan():
    f = MyActionBurstiness(K=5)
    out = f.extract(_ctx([[0.0, 0.0]] * 3))
    assert math.isnan(out["my_action_burstiness__k5"])

def test_describe_classmethod_is_pure():
    """Validator calls describe() WITHOUT instantiating — the map must be derivable from params alone."""
    out = MyActionBurstiness.describe({"K": 5})
    assert out == {"my_action_burstiness__k5": "risky"}
```

#### 8.1 关键合同（fail-loud 列表）

- `extract` 返回 dict 的 key 集合必须 == `self.descriptor_orientations.keys()` —— CompositeJudge `__call__` 内做 key contract assertion，不一致即 raise（不是悄悄 NaN）
- `describe(params)` 是 classmethod，validator 在 yaml load 时**不实例化**就调它，所以**不能**依赖 `self`、不能 IO、不能查 library_stats
- 任何物理边界（history 不够 / walk 走完 / 退化退化分母）→ 返回 `float("nan")`，**不要** raise；composer 子类负责 NaN 处理
- 不要在 factor 内自己做 z-score；用 `ctx.normalization.normalize_action / normalize_state`（Layer 1 owns z-score）

---

### 8.2 新增第 3 层 Calibration

继承 `Calibration` Protocol（`factors/calibrations/base.py`）：实现 `__init__(samples) / bind_keys(keys) / __call__(raw) / on_episode_start()`。

```python
# src/openpi/cache/components/factors/calibrations/my_zscore.py

"""Z-score-on-factors calibration: per-key (v - μ) / σ over the warmup
samples. Demonstration of an alternative to PercentileRollingCalibration."""

from __future__ import annotations

import math

from openpi.cache.components.factors.base import CalibrationSamples


class ZScoreOnFactorsCalibration:
    """Each verdict's factor value is mapped to its z-score against the
    bound key's warmup-sample distribution. NaN inputs propagate."""

    def __init__(self, samples: CalibrationSamples) -> None:
        if samples is None:
            raise ValueError("ZScoreOnFactorsCalibration requires CalibrationSamples")
        self._samples = samples
        self._stats: dict[str, tuple[float, float]] = {}   # key -> (mu, sigma)

    def bind_keys(self, keys: list[str]) -> None:
        # bind_keys is the FAIL-FAST hook (plan §6.3): every Layer-2 union
        # key must have enough non-NaN samples here, otherwise raise.
        for k in keys:
            samples = self._samples.samples.get(k)
            if samples is None:
                raise KeyError(f"calibration source missing key {k!r}")
            non_nan = [v for v in samples if not math.isnan(v)]
            if len(non_nan) < 30:
                raise ValueError(
                    f"key {k!r}: only {len(non_nan)} samples, need >= 30"
                )
            n = len(non_nan)
            mu = sum(non_nan) / n
            var = sum((x - mu) ** 2 for x in non_nan) / n
            sigma = max(var ** 0.5, 1e-6)
            self._stats[k] = (mu, sigma)

    def __call__(self, raw: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in raw.items():
            if math.isnan(v) or k not in self._stats:
                out[k] = float("nan")
                continue
            mu, sigma = self._stats[k]
            out[k] = (float(v) - mu) / sigma
        return out

    def on_episode_start(self) -> None:
        return None     # stats are immutable post-bind_keys
```

注册（写在子类上方，加到 `calibrations/__init__.py`）：

```python
# src/openpi/cache/components/factors/calibrations/__init__.py
from openpi.cache.components.factors.calibrations.my_zscore import (
    ZScoreOnFactorsCalibration,
)
```

`_build_calibration` 加一个分支（`config.py`）：

```python
if cfg.type == "z_score_on_factors":
    return ZScoreOnFactorsCalibration(samples, **dict(cfg.params))
```

yaml 引用：

```yaml
calibration:
  type: z_score_on_factors
  params: {}                         # forwarded as kwargs (this class has none)
  samples_source:
    type: warmup                     # warmup pool fills CalibrationSamples
```

⚠ 关键合同：`bind_keys` 是**唯一**的 fail-fast 入口（startup 时调）；`__call__` 接到未知 key 必须返回 NaN（不要 raise，CompositeJudge 上游已做 key contract 校验）。Plan §6.3 / §6.5 原则 3 严守 no cold-start。

---

### 8.3 新增第 4 层 Composer

继承 `Composer` Protocol（`composers/base.py`）：构造时计算 `self.declared_dependencies` 实例属性、实现 `bind_orientations(orientations)` 和 `compose(calibrated, *, winner_id)`。

```python
# src/openpi/cache/components/factors/composers/my_max.py

"""Max-only composer: max calibrated value across all weighted keys.
Useful when any single high-signal factor should suffice for hit."""

from __future__ import annotations

import math
from typing import Optional

from openpi.cache.components.judge import HitType, JudgeResult


class MaxComposer:
    """JudgeResult based on max(calibrated[k] for non-zero-weight keys).

    safe key: contributes calibrated[k]
    risky key: contributes 1 - calibrated[k]
    non_monotonic: requires `directions[k]` per orientation contract.
    """

    declared_dependencies: set[str]

    def __init__(
        self,
        *,
        weights: dict[str, float],
        full_hit_threshold: float,
        warm_start_threshold: Optional[float] = None,
        warm_start_t: Optional[float] = None,
        directions: Optional[dict[str, str]] = None,
    ) -> None:
        self._weights = dict(weights)
        self._full = float(full_hit_threshold)
        self._warm = warm_start_threshold
        self._warm_t = warm_start_t
        self._directions = dict(directions or {})
        self._orientations: dict[str, str] = {}
        # Layer 4 contract — declare what factor keys we need
        self.declared_dependencies = {k for k, w in self._weights.items() if w != 0.0}

    def bind_orientations(self, orientations: dict[str, str]) -> None:
        self._orientations = dict(orientations)
        missing = [
            k for k, ori in self._orientations.items()
            if ori == "non_monotonic" and self._weights.get(k, 0.0) != 0.0
            and k not in self._directions
        ]
        if missing:
            raise ValueError(f"non_monotonic keys missing directions: {sorted(missing)}")

    def compose(
        self,
        calibrated: dict[str, float],
        *,
        winner_id: str,
    ) -> JudgeResult:
        contribs = []
        for k, w in self._weights.items():
            if w == 0.0:
                continue
            v = calibrated.get(k, float("nan"))
            if math.isnan(v):
                continue
            ori = self._orientations.get(k)
            if ori == "safe":
                contribs.append(v)
            elif ori == "risky":
                contribs.append(1.0 - v)
            else:
                # non_monotonic — handle per direction (omitted for brevity)
                contribs.append(v)
        if not contribs:
            return JudgeResult(HitType.MISS, composer_score=None)
        score = max(contribs)
        if score >= self._full:
            return JudgeResult(HitType.FULL_HIT, winner_id=winner_id, composer_score=score)
        if self._warm is not None and self._warm_t is not None and score >= self._warm:
            return JudgeResult(
                HitType.WARM_START, winner_id=winner_id,
                start_t=self._warm_t, composer_score=score,
            )
        return JudgeResult(HitType.MISS, composer_score=score)
```

注册到 `composers/__init__.py` 的 re-export 表 + `_build_composer` 加分支：

```python
# composers/__init__.py
from openpi.cache.components.factors.composers.my_max import MaxComposer

# config.py _build_composer
if cfg.type == "max":
    return MaxComposer(
        weights=cfg.weights,
        full_hit_threshold=cfg.tier_thresholds["full_hit"],
        warm_start_threshold=cfg.tier_thresholds.get("warm_start"),
        warm_start_t=cfg.warm_start_t,
        directions=cfg.directions,
    )
```

yaml 引用：

```yaml
composer:
  type: max
  weights:
    jerk_online_state__p5_f5: 1.0
    direction_online_state__p5_f5: 1.0
  tier_thresholds: { full_hit: 0.7, warm_start: 0.4 }
  warm_start_t: 0.7
```

⚠ 关键合同：
- `declared_dependencies` 是**实例属性**（不是 classmethod），构造时算出。CompositeJudge `__init__` 静态校验 `composer.declared_dependencies ⊆ Layer 2 union key 集`
- WARM_START 发射路径都受 plan §3.6 / §13.3 规则 5c **CP1-only** 约束 —— 任何输出 `HitType.WARM_START` 的 composer 子类，其 yaml 配置都不能挂在 cp3 下（validator 会 reject `warm_start_t` 在 cp3）
- 空 `calibrated` / 全 NaN 时建议返回 `JudgeResult(HitType.MISS)`；如果你的 composer 想在那种情况下走 WARM_START 路径，参考 `WeightedSumWithWarmFallbackComposer` 的实现 + 走 `composer.warm_fallback_start_t` yaml 字段（也是 CP1-only）

---

### 8.4 新增 OfflineWriter（让 offline 因子能写 artifact）

`Factor` 子类同时实现 `required_payload_fields()` + `compute_for_episode(entries, library_stats)` 即满足 `OfflineWriter` Protocol（duck typing）。`exp/common/factor_postprocess.py:_load_offline_writers_from_yaml` 通过 `hasattr(cls, 'compute_for_episode')` 自动发现。

完整骨架（一个写 entry-chain 上 action 平均速度的 demo）：

```python
# src/openpi/cache/components/factors/my_pack.py (extends 8.1)

@register("mean_speed_offline_action")
class MeanSpeedOfflineAction:
    """Per-entry mean velocity magnitude over a (P, F) chain window."""

    requires_chain_walk: bool = False     # online path reads payload.factors
    required_top_k:      int = 0

    def __init__(self, *, windows: list[dict]) -> None:
        from openpi.cache.components.factors.base import normalize_windows
        self._windows = normalize_windows(windows)
        self.descriptor_orientations = self.__class__.describe({"windows": windows})

    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        from openpi.cache.components.factors.base import normalize_windows
        return {
            f"mean_speed_offline_action__p{p}_f{f}": "non_monotonic"
            for (p, f) in normalize_windows(params["windows"])
        }

    # ---- ONLINE path: read what the OfflineWriter wrote ----
    def extract(self, ctx) -> dict[str, float]:
        keys = list(self.descriptor_orientations)
        if not ctx.results:
            return {k: float("nan") for k in keys}
        winner = ctx.view.get(ctx.results[0].id)
        if winner.factors is None:
            return {k: float("nan") for k in keys}
        return {k: float(winner.factors.get(k, float("nan"))) for k in keys}

    # ---- OFFLINE path: writer surface ----
    def required_payload_fields(self) -> set[str]:
        return set()                                    # uses existing schema

    def compute_for_episode(
        self, entries, library_stats,
    ) -> list[dict[str, float]]:
        import torch
        keys = list(self.descriptor_orientations)
        T = len(entries)
        if T == 0:
            return []
        seq = torch.stack(
            [torch.as_tensor(e.payload.action_chunk[0], dtype=torch.float32) for e in entries],
            dim=0,
        )                                               # [T, A]
        sigma = library_stats.action_sigma.clamp_min(0.01)
        seq_norm = seq / sigma                          # [T, A]
        active = library_stats.action_active_mask
        pts = seq_norm[..., active]                     # [T, A_active]
        out: list[dict[str, float]] = []
        for k_idx in range(T):
            row: dict[str, float] = {}
            for (P, F) in self._windows:
                lo, hi = k_idx - P, k_idx + F
                if lo < 0 or hi >= T:
                    row[f"mean_speed_offline_action__p{P}_f{F}"] = float("nan")
                    continue
                w = pts[lo:hi + 1]
                v = (w[1:] - w[:-1]).norm(dim=-1)       # [W-1]
                row[f"mean_speed_offline_action__p{P}_f{F}"] = (
                    float(v.mean()) if v.numel() else float("nan")
                )
            out.append(row)
        return out
```

`exp/common/factor_postprocess.py` 找到这个类的方式：通过 `_load_offline_writers_from_yaml` 走 registry，检查 `hasattr(cls, 'compute_for_episode')`，所以**无需注册**第二处。

**写 artifact**：

```bash
uv run python -m exp.common.build_in_memory_cache_artifact enrich-existing-pkl \
  --input  exp/warm_start/data/spatial16/cp1_spatial_pool_16.pkl \
  --factors-yaml my_factors.yaml \
  --output exp/warm_start/data/spatial16/cp1_spatial_pool_16_v2.pkl

# my_factors.yaml
factors:
  - type: mean_speed_offline_action
    params:
      windows: [{past: 1, future: 1}, {past: 5, future: 5}]
```

CLI 内部只接受 8 个 offline 因子（registry name + `hasattr(compute_for_episode)`）；尝试用 online 因子或 topk 立刻 reject。完整 e2e smoke 测试见 `tests/exp/common/test_build_enrich_existing_pkl.py`。

---

### 8.5 编写流程 checklist（五步法）

1. **想清楚物理含义** —— 因子捕捉哪个信号？safe / risky / non_monotonic？需要 chain walk 吗？需要 top-K 吗？
2. **写类 + `@register("name")`** —— `__init__` / `describe(params)` / `extract(ctx)` 三个 method，外加 capability flags
3. **加 import** —— `factors/registry.py` 末尾加一行 `from . import my_pack` 让 `@register` side-effect 在 registry 模块加载时触发
4. **写 yaml + smoke** —— 复制 §3 完整 yaml 模板，把 `factors` / `calibration` / `composer` 改成新名字；本地跑 `load_cache_config` 看是否过 validator
5. **写测试** —— 至少覆盖 happy path + 物理边界（NaN 出口）+ classmethod `describe(params)` 返回值与实例 `descriptor_orientations` 一致

---

## 9. 与重构前差异速查

| 维度 | 旧 | 新 |
|---|---|---|
| 因子数 | 5 (`f1a_a/f1a_t/f1b_a/f1b_t/f2`) | 17 (`<desc>_<source>_<channel>` × 16 + `topk_action_variance`) |
| descriptor 名 | `jerk / dir / curv_radius / cum_disp` | `jerk / direction / dispersion / path_length` |
| 架构 | 因子 + Normalizer + Composer + 框架级 fallback | 4 层正交 (Normalization / Factor / Calibration / Composer) |
| Cold-start | `cold_start_strategy: force_miss/passthrough/lenient` | **删除** — 启动 fail-fast |
| `all_nan_fallback` yaml | `{type: warm_start, start_t}` | **删除** — 由 `WeightedSumWithWarmFallbackComposer` 子类承接 |
| `factor_outputs` 字段 | `{raw, norm, score, sentinel}` | `{raw, calibrated, composer_score, schema_version=2}` |
| OfflineWriter 签名 | `compute_for_episode(entries, library_stats)` | 不变 |
| OnlineExtractor Protocol | 多参数 `extract(results, view, history, cached_data)` | 改 `Factor.extract(ctx: FactorContext)` 单参数 dataclass |
| Composer dependency check | classmethod `declared_dependencies(params)` | 实例属性 `composer.declared_dependencies` |
| 旧 wire 协议 | `__hit_meta__["factor_outputs"]` 同 | schema_version=2，旧 client 见到字段缺失 = v1 |

完整设计 + 决策史见 [`logs/verdict_factor_judge_refactor.log.md`](../../logs/archive/verdict_factor_judge_refactor.log.md)（G1 APPROVED Round 4，2026-05-07）。
