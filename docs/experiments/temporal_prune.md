# Temporal Prune 实验教程

> **前置知识**:
> - [../cache/temporal_prune.md](../cache/temporal_prune.md) — KeyBuilder 组件和参数说明
> - [cp1_cache.md](cp1_cache.md) — 通用 CP1 实验流程

---

## 1. 实验概述

本实验在 `cp1_temporal_prune` KeyBuilder 上做网格搜索，探索 temporal pruning 参数对 cache 检索效果的影响。

### 1.1 实验网格

| 维度 | 取值 | 说明 |
|------|------|------|
| reducer | mean_pool, max_pool | Step 2 池化策略 |
| prune_window_size | 3, 4, 5, 6 | 时间窗口帧数 |
| temporal_keep_ratio | 0.25, 0.5, 0.75 | 保留 token 比例 |
| weights | WA, WB | 检索融合权重（见下） |

**权重配置**:

| ID | vision_0 | vision_1 | robot_state | 特点 |
|----|----------|----------|-------------|------|
| WA | 0.1 | 0.1 | 0.8 | robot_state 主导 |
| WB | 0.5 | 0.25 | 0.25 | vision 主导 |

### 1.2 总量

- **Artifact**: 2 × 4 × 3 = **24 个 .pkl**
- **YAML**: 24 × 2 = **48 个**
- **搜索策略**: 全部使用 `weighted_rrf_knn`, `top_k=1`, `step_filter=all`, `trajectory_depth=1`

---

## 2. Step 1: 构建 Artifact

### 2.1 一键生成构建命令

```bash
uv run exp/temporal_prune/generate_temporal_prune_yamls.py \
    --print-artifact-commands \
    --data-dir exp/common/data/db/libero_cache/libero_spatial \
    --artifact-dir exp/common/data/cache_artifacts/libero_spatial/temporal_prune
```

这会输出所有 24 条 `build_in_memory_cache_artifact.py` 命令。

### 2.2 执行构建

将输出的命令保存为脚本并运行：

```bash
uv run exp/temporal_prune/generate_temporal_prune_yamls.py \
    --print-artifact-commands \
    --data-dir exp/common/data/db/libero_cache/libero_spatial \
    --artifact-dir exp/common/data/cache_artifacts/libero_spatial/temporal_prune \
    > /tmp/build_tp_artifacts.sh

bash /tmp/build_tp_artifacts.sh 2>&1 | tee logs/build_tp_artifacts.log
```

> **注意**: 使用 `--workers -1`（串行模式）避免 WSL2 下 ProcessPoolExecutor fork 问题。24 个 artifact 串行构建需要一定时间。

### 2.3 验证 Artifact

```bash
ls exp/common/data/cache_artifacts/libero_spatial/temporal_prune/*.pkl | wc -l
# 应输出 24
```

期望的文件命名：

```
cp1_tp_mean_3w_025kr.pkl   cp1_tp_max_3w_025kr.pkl
cp1_tp_mean_3w_05kr.pkl    cp1_tp_max_3w_05kr.pkl
cp1_tp_mean_3w_075kr.pkl   cp1_tp_max_3w_075kr.pkl
cp1_tp_mean_4w_025kr.pkl   cp1_tp_max_4w_025kr.pkl
...                        ...
cp1_tp_mean_6w_075kr.pkl   cp1_tp_max_6w_075kr.pkl
```

---

## 3. Step 2: 生成 YAML 配置

```bash
uv run exp/temporal_prune/generate_temporal_prune_yamls.py \
    --artifact-dir exp/common/data/cache_artifacts/libero_spatial/temporal_prune \
    --output-dir exp/temporal_prune
```

验证：

```bash
ls exp/temporal_prune/config/*.yaml | wc -l
# 应输出 48
```

文件命名示例：

```
tp_run_001_mean_3w_025kr_wa.yaml   # mean_pool, window=3, kr=0.25, 权重A
tp_run_002_mean_3w_025kr_wb.yaml   # mean_pool, window=3, kr=0.25, 权重B
...
tp_run_048_max_6w_075kr_wb.yaml    # max_pool, window=6, kr=0.75, 权重B
```

---

## 4. Step 3: 启动 GPU 服务器

在 GPU 机器上启动推理服务：

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache_config cache.yaml \
    --env LIBERO \
    --port 8000 \
    --stage1_device cuda:0 \
    --stage2_device meta \
    --stage3_device meta \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

> `--concurrent` 模式允许通过 WebSocket 动态切换 cache 配置，无需重启服务器。

---

## 5. Step 4: 运行实验

在评估端执行：

```bash
uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/temporal_prune \
    --episodes-per-run 5 \
    --num-workers 5 \
    --host <GPU_HOST> --port <GPU_PORT> \
    --task-suite libero_spatial \
    --seed 42 \
    --conda-env libero_sim
```

### 5.1 断点续跑

实验支持断点续跑。如果中断，直接重新运行相同命令即可，已完成的 run 会自动跳过：

```bash
uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/temporal_prune \
    --resume \
    ...  # 其余参数同上
```

进度保存在 `exp/temporal_prune/data/experiment_state.json`。

---

## 6. Step 5: 分析结果

```bash
uv run exp/common/analyze_cache_results.py \
    --state-file exp/temporal_prune/data/experiment_state.json \
    --output exp/temporal_prune/config/analysis.json
```

分析维度：
- 按 reducer 分组：mean_pool vs max_pool 哪个更好？
- 按 keep_ratio 分组：0.25 / 0.5 / 0.75 的最优点在哪？
- 按 window 分组：3 / 4 / 5 / 6 是否有单调趋势？
- 按权重分组：WA（robot_state 主导）vs WB（vision 主导）

---

## 7. 文件一览

| 文件 | 用途 |
|------|------|
| `exp/temporal_prune/generate_temporal_prune_yamls.py` | 生成 24 artifact 构建命令 + 48 YAML 配置 |
| `exp/common/build_in_memory_cache_artifact.py` | 构建 .pkl artifact（已有） |
| `exp/common/run_cache_experiments.py` | 执行实验（已有） |
| `exp/common/analyze_cache_results.py` | 分析结果（已有） |
| `exp/temporal_prune/config/` | YAML 配置和实验状态 |
| `exp/common/data/cache_artifacts/libero_spatial/temporal_prune/` | .pkl artifact 文件 |
