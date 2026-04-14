# CP1 Cache 实验运行指南

> 本文档基于 `cache_cp1_impl_plan.log.md` 中的实验设计，提供完整的运行步骤。

---

## 网络拓扑

```
┌─────────────────────────┐         frp 隧道          ┌────────────────────────┐
│  GPU 服务器 (无公网IP)    │ ◄─────────────────────── │  LIBERO 评估端          │
│  serve_policy.py         │   155.98.36.13:9000       │  run_cache_experiments  │
│  监听 localhost:8000     │   → localhost:8000        │  examples/libero/main   │
└─────────────────────────┘                           └────────────────────────┘
```

- **GPU 服务器**: 运行模型推理，监听 `0.0.0.0:8000`
- **评估端**: 运行 LIBERO 环境 + 实验控制器，通过 frp 连接 `155.98.36.13:9000`
- 两端都需要本仓库代码和 `uv` 环境

---

## 前提条件

1. 两端已完成 `GIT_LFS_SKIP_SMUDGE=1 uv sync`
2. GPU 服务器上有 Pi0.5 checkpoint（默认路径 `gs://openpi-assets/checkpoints/pi05_base`）
3. 评估端上有 LIBERO benchmark 数据
4. 已收集的 HDF5 演示数据在 `data/db/libero_cache/libero_spatial/`（50 个 episode）
5. frp 隧道已配置：`155.98.36.13:9000` → GPU 服务器 `localhost:8000`

---

## Step 0: 验证 frp 隧道

在评估端运行：

```bash
curl http://155.98.36.13:9000/healthz
# 预期输出: OK
```

如果服务器未启动，会连接失败。先启动 Step 1 的服务器再测试。

---

## Step 1: 构建 Cache Artifact（评估端或 GPU 服务器）

将 HDF5 演示数据转换为 4 种降维方式的 `.pkl` 向量索引文件。

```bash
# 在有 data/db/libero_cache/libero_spatial/*.h5 的机器上运行

mkdir -p data/cache_artifacts/libero_spatial

# CP1 系列 (从 stage1 prefix_embs 降维)
for bt in cp1_mean_pool cp1_spatial_pool_16 cp1_spatial_pool_64 cp1_max_pool; do
    uv run exp/cache_experiment/build_in_memory_cache_artifact.py \
        --data-dir data/db/libero_cache/libero_spatial \
        --builder-type $bt \
        --output data/cache_artifacts/libero_spatial/${bt}.pkl
    echo "Done: $bt"
done

# CLIP ViT-B-32 (从原始图片编码)
uv run exp/cache_experiment/build_clip_cache_artifact.py \
    --data-dir data/db/libero_cache/libero_spatial \
    --clip-model ViT-B-32 \
    --clip-pretrained openai \
    --output data/cache_artifacts/libero_spatial/clip_vit_b_32.pkl \
    --device cuda \
    --batch-size 64 \
    --fields vision_0,vision_1,vision_2,prompt_emb,robot_state
```

产物：
```
data/cache_artifacts/libero_spatial/
├── cp1_mean_pool.pkl          # A: mean pool → 2048d
├── cp1_spatial_pool_16.pkl    # B1: 4×4 spatial → 32768d
├── cp1_spatial_pool_64.pkl    # B2: 2×2 spatial → 8192d
├── cp1_max_pool.pkl           # C: max pool → 2048d
└── clip_vit_b_32.pkl          # D: CLIP ViT-B-32 → 512d
```

**注意**: 这些 `.pkl` 文件需要在 GPU 服务器上可访问（因为 `serve_policy.py` 加载它们）。如果在评估端构建，需要 scp 到 GPU 服务器。

---

## Step 2: 校准 Score Sum 统计量（评估端或 GPU 服务器）

为 `weighted_score_sum` 融合策略计算每个字段的 p5/p95 百分位统计。

```bash
uv run exp/cache_experiment/calibrate_score_sum_stats.py \
    --artifact-dir data/cache_artifacts/libero_spatial \
    --output data/cache_artifacts/libero_spatial/calibration.json \
    --num-pairs 50000 \
    --seed 42
```

产物：`data/cache_artifacts/libero_spatial/calibration.json`

检查输出中的 separation 警告——如果某个字段的 same-task vs cross-task 分离度 < 0.05，该字段区分度差。

---

## Step 3: 生成 Phase 1 实验 YAML 配置

共 10 种 combo（5 降维 × 2 融合），但当前 `SKIP_SCORE_SUM = True`（在 `generate_cache_run_yamls.py` 第 66 行），实际生成 **5 combo × 8 权重 = 40 个 YAML**。如需恢复 Score Sum 系列，将 `SKIP_SCORE_SUM` 改为 `False` 可生成全部 80 个。

```bash
uv run exp/cache_experiment/generate_cache_run_yamls.py \
    --phase 1 \
    --artifact-dir data/cache_artifacts/libero_spatial \
    --calibration-file data/cache_artifacts/libero_spatial/calibration.json \
    --output-dir configs/cache_runs
```

产物：
```
configs/cache_runs/phase1/
├── phase1_run_001_a_rrf_w1.yaml
├── phase1_run_002_a_rrf_w2.yaml
├── ...
├── ...
└── phase1_run_040_d_rrf_w8.yaml   # D: CLIP ViT-B-32
                                    # 共 40 个文件 (SKIP_SCORE_SUM=True)
                                    # 若 SKIP_SCORE_SUM=False 则为 80 个
```

**重要**: 生成的 YAML 中 `preload_path` 指向 `data/cache_artifacts/libero_spatial/` 的绝对路径。确保 GPU 服务器上的路径一致，或在生成后手动修改路径。如果两端路径不同，在 GPU 服务器端生成 YAML 或修改 `--artifact-dir` 使路径匹配 GPU 服务器的文件系统。

---

## Step 4: 启动 GPU 服务器（GPU 服务器端）

在 GPU 服务器上启动策略服务，使用 `--concurrent` 模式以支持动态配置切换。

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache_config cache.yaml \
    --env LIBERO \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

关键参数说明：
- `--concurrent`: **必须开启**。实验运行器通过 WebSocket 控制消息动态切换 cache 配置，只有 concurrent 模式支持此功能
- `--cache_config cache.yaml`: 初始 cache 配置（服务器启动后由实验运行器通过 `load_cache_config` 控制消息动态切换为每个 run 的 YAML）
- `--env LIBERO`: LIBERO 环境
- `policy:checkpoint --policy.config pi05_libero --policy.dir ...`: 指定本地 Pi0.5 LIBERO checkpoint 路径，避免从 GCS 下载

验证服务器就绪：
```bash
curl http://localhost:8000/healthz
# 输出: OK
```

---

## Step 5: 运行 Phase 1 实验（评估端）

### 5a. 完整运行（当前 40 个配置 × 10 task × 5 episodes = 2000 episodes）

```bash
uv run exp/cache_experiment/run_cache_experiments.py \
    --yaml-dir configs/cache_runs/phase1 \
    --episodes-per-run 5 \
    --num-workers 5 \
    --host 155.98.36.13 \
    --port 9000 \
    --task-suite libero_spatial \
    --seed 42 \
    --conda-env libero_sim
```

参数说明：
- `--host 155.98.36.13 --port 9000`: 通过 frp 隧道连接 GPU 服务器
- `--episodes-per-run 5`: 每个 task 跑 5 个 episode（与数据收集时一致）
- `--num-workers 5`: 每个 task 开 5 个并发 worker（需要 `--concurrent` 服务器）
- `--task-suite libero_spatial`: 10 个 task 的 suite
- `--seed 42`: 固定随机种子（数据收集时用默认 seed=7，实验评估用不同种子避免过拟合）
- `--conda-env libero_sim`: 使用 conda 环境运行 LIBERO 评估（`main.py` 需要 LIBERO 依赖，不在 uv 环境中）

### 5b. 只运行部分配置（调试用）

```bash
# 只运行第 1~8 个配置（即第一个 combo 的所有权重）
uv run exp/cache_experiment/run_cache_experiments.py \
    --yaml-dir configs/cache_runs/phase1 \
    --episodes-per-run 5 \
    --num-workers 5 \
    --host 155.98.36.13 \
    --port 9000 \
    --task-suite libero_spatial \
    --seed 42 \
    --conda-env libero_sim \
    --runs 1-8
```

### 5c. 断点续跑

实验运行器在每个 task 完成后持久化进度。如果中断（Ctrl+C、崩溃、网络断开），用 `--resume` 从上次位置继续：

```bash
uv run exp/cache_experiment/run_cache_experiments.py \
    --yaml-dir configs/cache_runs/phase1 \
    --episodes-per-run 5 \
    --num-workers 5 \
    --host 155.98.36.13 \
    --port 9000 \
    --task-suite libero_spatial \
    --seed 42 \
    --conda-env libero_sim \
    --resume
```

**注意**: `--resume` 会严格校验 `--episodes-per-run` 和 `--task-suite` 必须与之前一致，否则报错。

### 5d. 运行状态

进度保存在 `configs/cache_runs/phase1/experiment_state.json`。可以直接查看：

```bash
# 查看当前进度概要
python3 -c "
import json
states = json.load(open('configs/cache_runs/phase1/experiment_state.json'))
done = sum(1 for s in states if s['status'] == 'done')
running = sum(1 for s in states if s['status'] == 'running')
failed = sum(1 for s in states if s['status'] == 'failed')
pending = sum(1 for s in states if s['status'] == 'pending')
print(f'Done: {done}, Running: {running}, Failed: {failed}, Pending: {pending}')
for s in states:
    if s['status'] == 'done':
        print(f'  {s[\"run_id\"]}: success_rate={s[\"success_rate\"]:.4f}')
"
```

每个 run 的详细日志在 `configs/cache_runs/phase1/<run_id>.log`。

---

## Step 6: 分析 Phase 1 结果

```bash
uv run exp/cache_experiment/analyze_cache_results.py \
    --state-file configs/cache_runs/phase1/experiment_state.json \
    --output configs/cache_runs/phase1/analysis.json
```

输出：
- 排名表：所有配置按 success_rate 排序
- 每个 combo 的最优权重
- **Top 3 combo**：进入 Phase 1.5 的候选

检查 `configs/cache_runs/phase1/analysis.json` 中的 `top3` 字段。

---

## Step 7: 生成 Phase 1.5 配置

围绕 Phase 1 的 Top 3 combo 做细粒度权重搜索。

```bash
uv run exp/cache_experiment/generate_cache_run_yamls.py \
    --phase 1.5 \
    --artifact-dir data/cache_artifacts/libero_spatial \
    --calibration-file data/cache_artifacts/libero_spatial/calibration.json \
    --phase1-analysis configs/cache_runs/phase1/analysis.json \
    --output-dir configs/cache_runs
```

产物：`configs/cache_runs/phase1_5/` 下约 45 个 YAML。

---

## Step 8: 运行 Phase 1.5 实验

```bash
uv run exp/cache_experiment/run_cache_experiments.py \
    --yaml-dir configs/cache_runs/phase1_5 \
    --episodes-per-run 5 \
    --num-workers 5 \
    --host 155.98.36.13 \
    --port 9000 \
    --task-suite libero_spatial \
    --seed 42 \
    --conda-env libero_sim
```

断点续跑同理加 `--resume`。

---

## Step 9: 分析 Phase 1.5 结果

```bash
uv run exp/cache_experiment/analyze_cache_results.py \
    --state-file configs/cache_runs/phase1_5/experiment_state.json \
    --output configs/cache_runs/phase1_5/analysis.json
```

---

## Step 10: 生成 Phase 2 配置（加入 prompt_emb）

```bash
uv run exp/cache_experiment/generate_cache_run_yamls.py \
    --phase 2 \
    --artifact-dir data/cache_artifacts/libero_spatial \
    --calibration-file data/cache_artifacts/libero_spatial/calibration.json \
    --phase1-5-analysis configs/cache_runs/phase1_5/analysis.json \
    --output-dir configs/cache_runs
```

产物：`configs/cache_runs/phase2/` 下约 3 个 YAML（prompt_emb 权重 0.0 / 0.1 / 0.2）。

---

## Step 11: 运行 Phase 2 实验

```bash
uv run exp/cache_experiment/run_cache_experiments.py \
    --yaml-dir configs/cache_runs/phase2 \
    --episodes-per-run 5 \
    --num-workers 5 \
    --host 155.98.36.13 \
    --port 9000 \
    --task-suite libero_spatial \
    --seed 42 \
    --conda-env libero_sim
```

---

## Step 12: 分析 Phase 2 最终结果

```bash
uv run exp/cache_experiment/analyze_cache_results.py \
    --state-file configs/cache_runs/phase2/experiment_state.json \
    --output configs/cache_runs/phase2/analysis.json
```

`analysis.json` 中的 `best` 字段即为最终最优配置。

---

## 完整流水线时间估算

| Phase | 配置数 | Task 数 | Episodes/Task | 总 Episodes | 预计时长 |
|-------|--------|---------|---------------|-------------|---------|
| 1     | 40 (SKIP_SCORE_SUM) / 80 (full) | 10 | 5 | 2,000 / 4,000 | 取决于单 episode 时间 |
| 1.5   | ~45    | 10      | 5             | ~2,250      | — |
| 2     | 3      | 10      | 5             | 150         | — |

单个 episode 时间取决于 LIBERO task 的 max_steps（libero_spatial: 220 步）和推理延迟。

---

## 故障排查

### 服务器连接失败

```bash
# 检查 frp 隧道
curl http://155.98.36.13:9000/healthz

# 检查 GPU 服务器本地
curl http://localhost:8000/healthz
```

### load_cache_config 报错

- 确认服务器使用了 `--concurrent` 启动
- 确认 YAML 中的 `preload_path` 在 GPU 服务器上存在
- 查看 GPU 服务器终端的错误日志

### 实验中断后的恢复

```bash
# 查看哪些 run 没完成
python3 -c "
import json
states = json.load(open('configs/cache_runs/phase1/experiment_state.json'))
for s in states:
    if s['status'] != 'done':
        remaining = sum(1 for v in s['task_progress'].values() if v != 'done')
        print(f'{s[\"run_id\"]}: status={s[\"status\"]}, remaining_tasks={remaining}')
"

# 继续运行
uv run exp/cache_experiment/run_cache_experiments.py \
    --yaml-dir configs/cache_runs/phase1 \
    --episodes-per-run 5 \
    --num-workers 5 \
    --host 155.98.36.13 \
    --port 9000 \
    --task-suite libero_spatial \
    --seed 42 \
    --conda-env libero_sim \
    --resume
```

### 单个 task 手动测试

不通过实验运行器，直接跑一个 task 验证环境：

```bash
MUJOCO_GL=egl conda run --no-capture-output -n libero_sim python examples/libero/main.py \
    --host 155.98.36.13 \
    --port 9000 \
    --task-suite-name libero_spatial \
    --num-trials-per-task 2 \
    --num-workers 1 \
    --task-ids 0
```

### 多 worker 手动测试

验证 concurrent 模式下多个 worker 都能正常工作：

```bash
MUJOCO_GL=egl conda run --no-capture-output -n libero_sim python examples/libero/main.py \
    --host 155.98.36.13 \
    --port 9000 \
    --task-suite-name libero_spatial \
    --num-trials-per-task 2 \
    --num-workers 5 \
    --task-ids 0 1 2 3 4 \
    --seed 42
```

> **注意**: `--num-workers` 不应超过 `--task-ids` 的数量，否则多余的 worker 会因队列为空立即退出（任务分配粒度是 task 级别，不是 episode 级别）。

> **注意**: `main.py` 依赖 LIBERO 环境，必须用 `conda run -n libero_sim` 而非 `uv run`。
> 实验运行器 `run_cache_experiments.py` 本身通过 `uv run` 启动（只需 `msgpack`/`websockets`），
> 但内部通过 `--conda-env libero_sim` 参数用 conda 调用 `main.py`。

---

## 文件依赖关系总览

```
data/db/libero_cache/libero_spatial/*.h5           ← 原始 HDF5 演示数据
    │
    ├──▶ build_in_memory_cache_artifact.py         (CP1 系列: A/B1/B2/C)
    ├──▶ build_clip_cache_artifact.py              (CLIP 系列: D)
    ▼
data/cache_artifacts/libero_spatial/*.pkl          ← 向量索引 artifact
    │
    ├──▶ calibrate_score_sum_stats.py
    │       ▼
    │   calibration.json                           ← 百分位统计
    │       │
    ├───────┤
    ▼       ▼
generate_cache_run_yamls.py --phase 1
    ▼
configs/cache_runs/phase1/*.yaml                   ← 64 个实验配置
    │
    ├──▶ serve_policy.py --concurrent              (GPU 服务器加载 artifact + YAML)
    │
    ▼
run_cache_experiments.py                           (评估端执行)
    ▼
experiment_state.json                              ← 实验进度+结果
    │
    ▼ analyze_cache_results.py
analysis.json                                      ← 排名 + Top 3
    │
    ▼ generate_cache_run_yamls.py --phase 1.5
configs/cache_runs/phase1_5/*.yaml → ... → analysis.json
    │
    ▼ generate_cache_run_yamls.py --phase 2
configs/cache_runs/phase2/*.yaml → ... → analysis.json (最终结果)
```
