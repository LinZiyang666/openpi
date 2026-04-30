> Status: Plan
> Date: 2026-04-21
> Level: L1

# Phase1 libero_10 执行命令清单

60 个 yaml 分 3 个 batch，三服务器 / 三客户端并行：

- `exp/common/config/phase1/libero_10/batch1/` — 20 份（001–020，a×8 + b1×8 + b2_w1..w4）
- `exp/common/config/phase1/libero_10/batch2/` — 20 份（021–040，b2_w5..w8 + c×8 + d×8）
- `exp/common/config/phase1/libero_10/batch3/` — 20 份（041–060，prompt_emb 验证 5 builder × 4 权重档位 p1..p4）

在 `exp/common/data/cache_artifacts/libero_10/*.pkl` 上执行；runner 通过 WebSocket `load_cache_config` 为每个 run 热切换 cache bundle。

实验在 LIBERO 默认 `pruned_init` 上跑，每 sub task 10 个 init（`--episodes-per-run 10`）。

## 端口映射

沿用 trajectory_deviation step2 的约定：server 本地端口 + 1000 = frp 公网端口。

| batch | server 本地端口 | frp 公网端口 | 种子 yaml |
|---|---:|---:|---|
| `batch1` | `7998` | `8998` | `phase1_run_001_a_rrf_w1.yaml` |
| `batch2` | `7999` | `8999` | `phase1_run_021_b2_rrf_w5.yaml` |
| `batch3` | `8000` | `9000` | `phase1_run_041_a_rrf_p1.yaml` |

公网 host：`155.98.36.13`。

---

## 前置检查

客户端机器确认三个入口健康：

```bash
curl http://155.98.36.13:8998/healthz
curl http://155.98.36.13:8999/healthz
curl http://155.98.36.13:9000/healthz
```

三个都应返回：`OK`

确认 pkl 齐全：

```bash
ls -la exp/common/data/cache_artifacts/libero_10/
# 期望：clip_vit_b_32.pkl / clip_vit_l_14.pkl / cp1_mean_pool.pkl /
#       cp1_max_pool.pkl / cp1_spatial_pool_16.pkl / cp1_spatial_pool_64.pkl
```

---

## 1. 服务器命令（GPU 服务器侧，三终端）

每个 batch 一个独立 server；`--cache-config` 传该 batch 的第一个 yaml 作为种子（runner 会 WebSocket 热切换后续配置）。

### Server A: batch1, local port 7998

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/common/config/phase1/libero_10/batch1/phase1_run_001_a_rrf_w1.yaml \
    --env LIBERO \
    --port 7998 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

### Server B: batch2, local port 7999

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/common/config/phase1/libero_10/batch2/phase1_run_021_b2_rrf_w5.yaml \
    --env LIBERO \
    --port 7999 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

### Server C: batch3, local port 8000

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/common/config/phase1/libero_10/batch3/phase1_run_041_a_rrf_p1.yaml \
    --env LIBERO \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

> 每个 client 只能指向一个独立 server（`load_cache_config` 不能跨 batch 并发），所以三个 batch 的 client 必须各自对应一台 server，不要复用端口。

---

## 2. 客户端 Runner 命令（LIBERO eval 主机，三终端）

`libero` 不在 uv venv 里，必须通过 `--conda-env` 走 conda（绝对路径时 runner 自动用 `conda run -p`）。下面以 `/scratch/zixuans8/libero_sim` 为例；按名字装的 env 改成 `--conda-env libero_sim` 即可。

### Client 1: batch1 via frp port 8998

```bash
uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/common/config/phase1/libero_10/batch1 \
    --episodes-per-run 10 \
    --num-workers 5 \
    --host 155.98.36.13 --port 8998 \
    --task-suite libero_10 \
    --seed 42 \
    --conda-env /scratch/zixuans8/libero_sim \
    --resume
```

### Client 2: batch2 via frp port 8999

```bash
uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/common/config/phase1/libero_10/batch2 \
    --episodes-per-run 10 \
    --num-workers 5 \
    --host 155.98.36.13 --port 8999 \
    --task-suite libero_10 \
    --seed 42 \
    --conda-env /scratch/zixuans8/libero_sim \
    --resume
```

### Client 3: batch3 via frp port 9000

```bash
uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/common/config/phase1/libero_10/batch3 \
    --episodes-per-run 10 \
    --num-workers 5 \
    --host 155.98.36.13 --port 9000 \
    --task-suite libero_10 \
    --seed 42 \
    --conda-env /scratch/zixuans8/libero_sim \
    --resume
```

每个 client 跑 20 run × 10 task × 10 episode = 2000 episodes。

---

## 3. 可选：只跑部分 run / task

只跑某几个 run（runner 自带过滤，下标是 yaml 目录内的相对序号，不是全局 run id）：

```bash
--runs '3-5,10'
```

只跑某几个 task：

```bash
--task-ids '0,3,5'
```

指定 CUDA device：

```bash
--cuda '0'
```

---

## 4. 结果位置

每个 run 输出写到与 yaml 同目录，文件名前缀与 yaml 对应：

- `phase1_run_NNN_<group>_rrf_<tag>.log` — stdout 日志
- `phase1_run_NNN_<group>_rrf_<tag>.episode_results.json` — 逐 episode 成功/失败
- `phase1_run_NNN_<group>_rrf_<tag>.timing.csv` — 每步耗时（若 timer.enabled）
- `experiment_state.json` — runner 断点续跑状态（`--resume` 读取）

`--log-dir` 可把 stdout 日志另外集中到其他目录。

---

## 5. 关键 Caveat

### 5.1 init-state

实验用 LIBERO 默认 `pruned_init`（每 task 50 条，runner 默认流程）。pkl artifact 是由 `exp/common/data/db_init/libero_cache/libero_10/` 子集（每 task 5 条）构建的，所以客户端侧 `--episodes-per-run 10` 抽到的 init 大多数不在 pkl 构建集中；命中率主要靠 step-level key 在其他 trajectory 上的泛化，而不是 init 一一对应。这是本次实验的既定设计，不是 bug。

### 5.2 prompt_emb 维度

Batch 3 的 d 组（CLIP ViT-B-32）里 prompt_emb 维度写的是 `2048`，不是 CLIP text encoder 的 512。原因：所有 KeyBuilder 的 prompt_emb 都来自 paligemma prefix_embs 的 mean pool（`clip_prompt_key_from_tokens`），与 CLIP 视觉分支独立。

### 5.3 Batch 3 权重档位

- `p1`: v0=0.15, v1=0.15, prompt=0.10, robot=0.60（prompt 最小）
- `p2`: v0=0.50, v1=0.00, prompt=0.50, robot=0.00（prompt 最大）
- `p3`: v0=0.33, v1=0.00, prompt=0.34, robot=0.33（三均衡）
- `p4`: v0=0.15, v1=0.15, prompt=0.20, robot=0.50

---

## 6. 失败时的恢复

断点续跑（task 粒度）：

```bash
# 再次执行同一条命令即可；--resume 读取 experiment_state.json 跳过已完成 run
uv run exp/common/run_cache_experiments.py ... --resume
```

重跑某个 run（不续跑）：

```bash
# 删掉该 run 在 experiment_state.json 里的记录；或用 --runs 只点名这一个
--runs '17'
```

服务器重启后所有已完成的 run 仍然算完成（结果落盘），客户端带 `--resume` 跳过即可。
