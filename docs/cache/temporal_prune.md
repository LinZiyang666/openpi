# Temporal Prune KeyBuilder 使用指南

> **前置知识**: 阅读 [tutorial.md](tutorial.md) §4 了解 KeyBuilder 组件基础。
>
> **设计文档**: 完整方案设计见 [`logs/archive/redundant_token_prune_plan.log.md`](../../logs/archive/redundant_token_prune_plan.log.md)

---

## 1. 概述

`CP1TemporalPruneKeyBuilder` 是一个两步式 KeyBuilder，在现有 vision token 池化之前插入一个 **时间冗余剪枝** 阶段：

```
原始 vision token          Step 1: 时间剪枝              Step 2: Token 池化
[256, 2048]  ──────►  [K, 2048] (K<256)  ──────►  [output_dim]
                      去除跨帧静态 token              可插拔 reducer
```

**核心思想**: SigLIP 产生的 256 个 vision token 中，大部分是跨时间步几乎不变的静态背景 token。通过比较相邻帧之间的 cosine 变化，识别并去除这些冗余 token，使最终 cache key 更聚焦于动作相关的视觉区域。

**适用场景**: CP1 checkpoint 的 vision token 降维，在线推理和离线 artifact 构建均可使用。

---

## 2. 快速开始

### 2.1 构建离线 Artifact

```bash
uv run python exp/common/build_in_memory_cache_artifact.py \
    --data-dir exp/common/data/db/libero_cache/libero_spatial \
    --builder-type cp1_temporal_prune \
    --reducer-type mean_pool \
    --prune-window-size 4 \
    --temporal-keep-ratio 0.5 \
    --output exp/common/data/cache_artifacts/libero_spatial/cp1_temporal_prune_mean.pkl
```

### 2.2 配置在线推理 YAML

```yaml
# cache_temporal_prune.yaml
enabled: true

key_builder:
  type: cp1_temporal_prune
  prune_window_size: 4          # 时间窗口帧数（最小 2）
  temporal_keep_ratio: 0.5      # 保留变化最大的 50% token
  reducer:
    type: mean_pool             # Step 2 降维策略

keys:
  vision_0: { enabled: true, weight: 1.0 }
  vision_1: { enabled: true, weight: 1.0 }
  robot_state: { enabled: true, weight: 0.5 }
  prompt_emb: { enabled: true, weight: 0.3 }

backend:
  type: in_memory
  vector_dims:
    vision_0: 2048              # 必须与 reducer output_dim 一致
    vision_1: 2048
    prompt_emb: 2048
    robot_state: 32
  in_memory:
    preload_path: exp/common/data/cache_artifacts/libero_spatial/cp1_temporal_prune_mean.pkl

checkpoints:
  cp1:
    enabled: true               # cp1_temporal_prune 要求 CP1 启用
    judge:
      type: threshold
      threshold: 0.98
    search_strategy:
      type: weighted_rrf_knn
      top_k: 1
  cp3:
    enabled: true
    judge:
      type: threshold
      threshold: 0.95
    search_strategy:
      type: weighted_rrf_knn
      top_k: 1
```

### 2.3 启动推理服务

```bash
uv run python scripts/serve_policy.py \
    --env LIBERO \
    --cache_config cache_temporal_prune.yaml
```

---

## 3. 两步架构详解

### 3.1 Step 1: Temporal Pruning（时间冗余剪枝）

**作用范围**: 仅 vision 模态（vision_0 / vision_1 / vision_2），每个摄像头独立处理。prompt_emb 和 robot_state 不经过此步骤。

**算法**:
1. 维护一个 FIFO 历史窗口（W 帧），每次 CP1 collect 时 push 当前帧
2. 窗口满时，对每个 token 位置计算 temporal score = 相邻帧平均 cosine 变化
3. 按 temporal score 保留变化最大的 top `keep_ratio` 个 token
4. 输出 `PruneResult`（包含保留的 token、原始位置索引、temporal score）

**退化模式**: 窗口未满时（episode 初始阶段），跳过剪枝，全部 256 token 直接传给 Step 2。输出维度保持一致。

### 3.2 Step 2: Token Reduction（可插拔降维）

Step 2 由 `TokenReducer` 协议定义，可自由替换实现：

| Reducer | 参数 | 输出维度 | 说明 |
|---------|------|----------|------|
| `mean_pool` | — | 2048 | 时间+token 全平均，最简单 baseline |
| `max_pool` | — | 2048 | 时间平均后 token 维 per-dim max |
| `spatial_pool` | `output_tokens` | output_tokens * 2048 | 填回 16x16 网格 -> adaptive avg pool |
| `task_scoring` | `select_k`, `temperature` | 2048 | 用 cos(token, prompt_emb) 选 top-K 再加权池化 |

---

## 4. 参数说明

### 4.1 KeyBuilder 参数

| 参数 | 类型 | 默认值 | 约束 | 说明 |
|------|------|--------|------|------|
| `prune_window_size` | int | 4 | >= 2 | 时间窗口帧数。temporal scoring 需要至少 2 帧计算相邻帧差 |
| `temporal_keep_ratio` | float | 0.5 | (0, 1] | 保留 token 比例。0.5 = 从 256 个中保留 128 个 |

### 4.2 Reducer 参数

| 参数 | 适用 reducer | 类型 | 默认值 | 约束 | 说明 |
|------|-------------|------|--------|------|------|
| `output_tokens` | spatial_pool | int | 16 | >= 1, 完美平方数 | 输出 token 数，决定 pool 后网格大小 |
| `select_k` | task_scoring | int | 32 | >= 1 | 按 task score 选择的 top-K token 数 |
| `temperature` | task_scoring | float | 1.0 | > 0 | softmax 温度，越小越集中 |

### 4.3 vector_dims 与 reducer 的对应关系

配置 YAML 时，`backend.vector_dims` 中 vision 字段的维度 **必须** 与 reducer 输出维度一致，否则 config 校验会报错：

| reducer.type | vision 字段维度 |
|-------------|---------------|
| `mean_pool` | 2048 |
| `max_pool` | 2048 |
| `task_scoring` | 2048 |
| `spatial_pool` | output_tokens * 2048（如 output_tokens=16 → 32768） |

---

## 5. Reducer 选择指南

| 场景 | 推荐 reducer | 理由 |
|------|-------------|------|
| 初始 baseline 实验 | `mean_pool` | 最简单，易于与不剪枝的 `cp1_mean_pool` 直接对比 |
| 需要保留空间信息 | `spatial_pool` | 通过填回网格保留 token 的空间位置关系 |
| 混合任务数据库 | `task_scoring` | 利用 prompt_emb 区分"动态且相关" vs "动态但无关"的 token |
| 单任务数据库 | `mean_pool` 或 `max_pool` | prompt_emb 区分度低，task scoring 优势不大 |

---

## 6. 离线 Artifact Builder CLI 参考

```bash
uv run python exp/common/build_in_memory_cache_artifact.py \
    --data-dir <HDF5 数据目录> \
    --builder-type cp1_temporal_prune \
    --output <输出 .pkl 路径> \
    --reducer-type <mean_pool|max_pool|spatial_pool|task_scoring> \
    --output-tokens <int>              # spatial_pool 专用，默认 16 \
    --select-k <int>                   # task_scoring 专用，默认 32 \
    --temperature <float>              # task_scoring 专用，默认 1.0 \
    --prune-window-size <int>          # 默认 4 \
    --temporal-keep-ratio <float>      # 默认 0.5 \
    --workers <int>                    # 0=全部 CPU, -1=串行模式
```

**重要**: 离线 artifact 和在线推理必须使用 **完全相同的 reducer 参数**，否则 key 语义不一致。artifact 元数据中会记录 `reducer_params` 字段，便于人工核对。

### 6.1 批量构建示例

```bash
# 对比不同 reducer
for rt in mean_pool max_pool; do
    uv run python exp/common/build_in_memory_cache_artifact.py \
        --data-dir exp/common/data/db/libero_cache/libero_spatial \
        --builder-type cp1_temporal_prune \
        --reducer-type $rt \
        --output exp/common/data/cache_artifacts/libero_spatial/cp1_tp_${rt}.pkl
done

# 对比不同 keep_ratio
for kr in 0.25 0.5 0.75; do
    uv run python exp/common/build_in_memory_cache_artifact.py \
        --data-dir exp/common/data/db/libero_cache/libero_spatial \
        --builder-type cp1_temporal_prune \
        --reducer-type mean_pool \
        --temporal-keep-ratio $kr \
        --output exp/common/data/cache_artifacts/libero_spatial/cp1_tp_mean_kr${kr}.pkl
done
```

---

## 7. 生命周期与有状态行为

与现有无状态 KeyBuilder（如 `cp1_mean_pool`）不同，`CP1TemporalPruneKeyBuilder` 是 **有状态的**，维护跨步历史缓冲区：

| 事件 | 行为 |
|------|------|
| `collect(CP1, ...)` | push 当前帧 vision token 到历史缓冲区（clone 存储） |
| `collect(CP3, ...)` | **不** push 历史（避免 CP1+CP3 同帧双计） |
| `build(CP1/CP3)` | 从历史窗口取数据 → prune → reduce → CPU key |
| `clear()` | 只清 per-cycle cache，**不清** 历史缓冲区 |
| `on_episode_start()` | 重置历史缓冲区（由 Orchestrator 自动广播） |

**离线 artifact 构建**: `_process_episode()` 在每个 episode 开始时调用 `on_episode_start()`，builder 跨步保持，每步 `collect → build → clear`，历史自然积累。

---

## 8. 与现有 KeyBuilder 对比

| 特性 | `cp1_mean_pool` 等 | `cp1_temporal_prune` |
|------|-------------------|---------------------|
| vision token 处理 | 全量 256 token → 直接池化 | 先剪枝 → 再池化 |
| 有状态 | 否 | 是（历史缓冲区） |
| episode 边界 | 无需处理 | 需要 `on_episode_start()` 重置 |
| Step 2 可插拔 | 固定（mean/max/spatial） | 可选 4 种 reducer |
| 离线 artifact 构建 | 逐步独立 | 跨步连续（保持 builder 实例） |

---

## 9. 模块文件一览

| 文件 | 内容 |
|------|------|
| `src/openpi/cache/components/token_reducer.py` | PruneResult、TokenReducer Protocol、4 个 reducer 实现 |
| `src/openpi/cache/components/key_builder.py` | _VisionHistoryBuffer、CP1TemporalPruneKeyBuilder |
| `src/openpi/cache/config.py` | ReducerConfig、KeyBuilderConfig 扩展、_build_reducer 工厂、校验规则 |
| `src/openpi/cache/orchestrator.py` | on_episode_start 广播到 key_builder |
| `exp/common/build_in_memory_cache_artifact.py` | 离线 artifact 构建（含 cp1_temporal_prune 支持） |
| `tests/cache/components/test_temporal_prune.py` | 46 个测试用例 |

---

## 10. 常见问题

### Q: 为什么 `prune_window_size` 最小是 2？

temporal scoring 计算的是 **相邻帧之间** 的 cosine 变化。只有 1 帧时没有"相邻帧"，无法计算变化度，会产生 NaN。

### Q: episode 开始的前几步 key 质量是否会下降？

窗口未满时会跳过剪枝（PruneResult.pruned=False），全部 256 token 直接传给 reducer。输出维度与正常模式一致，但 key 内容等价于不剪枝的版本。这是设计选择：宁可退化为无剪枝，也不做不完整的 temporal scoring。

### Q: 如何扩展新的 reducer？

1. 在 `token_reducer.py` 中实现 `TokenReducer` 协议（`reduce()` 方法和 `output_dim` 属性）
2. 在 `config.py` 的 `_build_reducer()` 工厂中添加分支
3. 在 `config.py` 的 `validate_cache_config()` 中添加参数校验（如有新参数）
4. 在 `_valid_reducer_types` 集合中注册
5. 在 `exp/common/build_in_memory_cache_artifact.py` 的 `_build_artifact_reducer()` 中添加分支
6. 编写测试

### Q: 离线和在线的 key 会不会不一致？

只要使用相同的 reducer 参数，两者产生的 key 是确定性一致的。artifact 元数据中记录了 `reducer_params`，可以人工核对。config 层会校验 reducer 输出维度与 `backend.vector_dims` 是否匹配，不一致时启动即报错。
