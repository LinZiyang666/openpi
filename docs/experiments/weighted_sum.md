# Weighted-Sum 校准 + 权重搜索 runbook（libero_spatial）

> 两层 `weighted_score_sum` 检索的端到端实验：**Phase 1** 离线为每个 (keybuilder, 模态) 选 Layer-1 归一化方法+参数；**Phase 2** 用 conductor 纯-eval 找有用模态 + 搜最优权重。
> 设计依据见 [`logs/archive/weighted_sum_two_layer_refactor.log.md`](../../logs/archive/weighted_sum_two_layer_refactor.log.md)、架构见 [`cache_system.md` §5.8.1](../architecture/cache_system.md)、编排见 [`conductor_tutorial.md`](conductor_tutorial.md)。

---

## 0. 背景

旧 `weighted_score_sum` 因校准失败（库内随机对分布 + percentile 在高基线 cosine 上塌缩成常数）几乎不可用。重构把它拆成 **Layer-1 归一化**（可插拔、单调保幅、`[0,1]`、排除 rank）+ **Layer-2 加权和**，并用真实 query×全库分布数据驱动地选归一化方法。`prompt_emb` 已退出实验（任务内近常量），候选模态 = `{vision_0, vision_1, robot_state}`（CP1 另可选 `vision_2`）。

数据复用现有 `exp/common/data/cache_artifacts/libero_spatial/` 的 6 个库 artifact（1018 entries / 10 tasks / 全 CP1，带 `trajectory_id`），零额外采集。

## 1. Phase 1 — 离线校准（无 GPU/server）

```bash
uv run exp/common/calibrate_score_normalizers.py \
    --artifact-dir exp/common/data/cache_artifacts/libero_spatial \
    --output exp/weighted_sum/data/libero_spatial/phase1/calibration_normalizers.json \
    --max-queries 300
```

- **LOEO**：每个 entry 当 query，按 `trajectory_id` 过滤掉自己这条 episode 后对全库打分 → 真实 query×全库 raw 分布（消除 self-match / 链内近邻污染，忠实还原线上）。
- 对每个 (stem, field∈vector_dims 去 prompt_emb) 拟合所有兼容候选 normalizer，按 **`J = mag_sep + β·intra_spread − λ·sat`**（幅值结构指标；rank 指标对单调变换不变故不可用）排出 **top-2 shortlist**，记 `selected` = top-1。
- 输出按 **artifact stem** 分组（两个 CLIP 变体 `builder_type` 都是 `clip`，会碰撞）：`{stem: {builder_type, vector_dims, fields: {field: {sim_type, shortlist, selected}}}}`。

诊断图：

```bash
uv run exp/weighted_sum/analysis/plot_phase1_calibration.py \
    --calibration exp/weighted_sum/data/libero_spatial/phase1/calibration_normalizers.json
```

> 最终方法**不**由 J 单独裁决——shortlist 交给 Phase 2 用真实任务成功率定档。

## 2. Phase 2 — 权重搜索（conductor 纯-eval）

### 2.1 生成 eval YAML

```bash
uv run exp/weighted_sum/emit_yamls.py \
    --calibration exp/weighted_sum/data/libero_spatial/phase1/calibration_normalizers.json \
    --stem cp1_spatial_pool_16 \
    --preload-path exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl \
    --output-dir exp/weighted_sum/config/phase2/libero_spatial --mode both
```

每个 YAML：`weighted_score_sum_knn` + Phase-1 选定的 `score_normalization.type: per_field`、`judge.type: always_hit`（纯回放隔离检索质量）、`keys.prompt_emb.enabled: false`、**`write_policy.type: never`**（C2 write-frozen，否则 server load fail-fast）。`--mode`：`isolation`（单模态找有用模态）/ `grid`（有用模态权重网格）/ `both`。

### 2.2 起 server（见 conductor_tutorial §1）

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero --policy.dir=<ckpt> --port 8001 \
    --cache_config exp/weighted_sum/config/phase2/libero_spatial/<any>.yaml
```

### 2.3 跑评测（防 init-state 泄漏）

```bash
uv run exp/weighted_sum/run_phase2.py \
    --yaml-dir exp/weighted_sum/config/phase2/libero_spatial \
    --init-map exp/common/data/db/libero_cache/libero_spatial_init_map.json \
    --journal exp/weighted_sum/data/libero_spatial/phase2/journal.jsonl \
    --servers <host>:8001 --task-ids 0-9 --eval-trials 20
```

> ⚠ **init-state 泄漏硬门**：建库每任务只用了 ~5/50 个 init。`run_phase2` 经 `init_holdout` 从 `init_map` 读出已用 `orig_init_state_idx` 并**只从剩余 held-out init 取 episode**；`init_map` 缺失即 fail-fast，绝不静默跳过。

### 2.4 聚合 + 分析

先把 conductor journal 聚合成 per-yaml success_rate，再绘图：

```bash
uv run exp/weighted_sum/summarize.py \
    --journal exp/weighted_sum/data/libero_spatial/phase2/journal.jsonl \
    --out exp/weighted_sum/data/libero_spatial/phase2/results.json

uv run exp/weighted_sum/analysis/plot_phase2_results.py \
    --results exp/weighted_sum/data/libero_spatial/phase2/results.json
```

按 keybuilder 分组画 success_rate × 权重配置（对齐 `exp/common/analysis/phase1/libero_spatial` 风格）。2a 隔离结果与 Phase-1 `mag_sep` 先验交叉验证定"有用模态集合 + 每模态最终方法"；2b 在有用模态上 粗→细 网格搜最优权重。

## 3. Trajectory 扩展（depth>1 多步检索）

在 Phase-2 选出的最优配置之上叠加 trajectory search（多步 query 历史聚合），方法学对照老
trajectory 实验对 Phase 1 的做法。设计与决策见 [`logs/archive/weighted_sum_trajectory_search.log.md`](../../logs/archive/weighted_sum_trajectory_search.log.md)。

**Base 选取（18 个，去重）**：① per-keybuilder（4 个 CP1 keybuilder 各取 top1+top2+倒数第二，
倒数第二取正规权重网格、同 `zscore`，排除 `__norm2`/`iso_`）；② 全实验 top10。两组重叠 4 个
（spatial_16/max_pool 的 top1+top2 都在 top10），并集 18。depth ∈ {3,4,5,6}，`trajectory_weights`
复用老递减方案；depth-1 基线复用现有 `data/libero_spatial/phase2/all_results.csv` 的 SR（同 jupyter 机可比）。

```bash
# 1) 生成 72 份 trajectory yaml（18 base × 4 depth），自带去重断言 + schema 自检
PYTHONPATH=. uv run exp/weighted_sum/emit_trajectory_yamls.py
#    → exp/weighted_sum/config/trajectory/<base_id>__d{depth}.yaml

# 2) 起 server（jupyter，--replicas + HOME 见 devices.md / 上方 §2.2）+ tether expose
# 3) 跑评测（timan107，每 yaml 100 ep = 10 task × --eval-trials 10）
PYTHONPATH=. uv run exp/weighted_sum/run_phase2.py \
    --yaml-dir exp/weighted_sum/config/trajectory/libero_spatial \
    --init-map exp/common/data/db/libero_cache/libero_spatial_init_map.json \
    --journal  exp/weighted_sum/data/libero_spatial/trajectory/journal.jsonl \
    --servers <host>:<port> --task-ids 0-9 --eval-trials 10 --workers 48 --gpus 8

# 4) 聚合 + 分析（合并 depth-1 基线 + depth 3/4/5/6，算 Δ vs 单步）
uv run exp/weighted_sum/summarize.py \
    --journal exp/weighted_sum/data/libero_spatial/trajectory/journal.jsonl \
    --out     exp/weighted_sum/data/libero_spatial/trajectory/results.json
PYTHONPATH=. uv run exp/weighted_sum/analysis/plot_trajectory_results.py \
    --results  exp/weighted_sum/data/libero_spatial/trajectory/results.json \
    --baseline exp/weighted_sum/data/libero_spatial/phase2/all_results.csv
```

> trajectory 只支持 `InMemoryBackend`；`run_phase2` / conductor / 并发 server 对 trajectory 透明，
> 零改动。`emit_trajectory_yamls.py` 复用 `emit_yamls.build_eval_config`（已含 trajectory 形参）。

## 4. 文件

| 路径 | 作用 |
|------|------|
| `exp/common/calibrate_score_normalizers.py` | Phase 1 离线校准 |
| `exp/weighted_sum/emit_yamls.py` | Phase 2 eval YAML 生成（C2/per_field/prompt_emb mask）|
| `exp/weighted_sum/emit_trajectory_yamls.py` | Trajectory 扩展：18 base × depth 生成（复用 `build_eval_config`）|
| `exp/weighted_sum/init_holdout.py` | held-out init 防泄漏 |
| `exp/weighted_sum/weight_search_strategy.py` | conductor 纯-eval 策略 |
| `exp/weighted_sum/run_phase2.py` | Phase 2 / Trajectory 评测入口（通用 yaml-dir runner）|
| `exp/weighted_sum/summarize.py` | journal → per-yaml success_rate JSON |
| `exp/weighted_sum/analysis/` | Phase 1/2 + trajectory 绘图（`plot_trajectory_results.py`）|
