> Status: Plan
> Date: 2026-04-22
> Level: L1

# Trajectory libero_10 执行命令清单

60 个 yaml 分 3 个 batch，一个 depth 一个 batch，三服务器 / 三客户端并行：

- `exp/common/config/trajectory/libero_10/batch1/` — 20 份（001–020，全部 `trajectory_depth=4`）
- `exp/common/config/trajectory/libero_10/batch2/` — 20 份（021–040，全部 `trajectory_depth=5`）
- `exp/common/config/trajectory/libero_10/batch3/` — 20 份（041–060，全部 `trajectory_depth=6`）

每个 batch 覆盖同一组 20 个 `(key_builder, weight)` 组合 —— 每个 key_builder 按 libero_10 Phase1 排名选出 top-1 / top-2 / top-3 / 2nd-worst 共 4 个 weight：

| kb | top-1 | top-2 | top-3 | 2nd-worst |
|---|---|---|---|---|
| `a`  (cp1_mean_pool)       | w5 | w8 | w3 | w2 |
| `b1` (cp1_spatial_pool_16) | w5 | w3 | w8 | w2 |
| `b2` (cp1_spatial_pool_64) | w5 | w8 | w3 | w2 |
| `c`  (cp1_max_pool)        | w5 | w8 | w3 | w1 |
| `d`  (clip ViT-B/32)       | w6 | w7 | w5 | w8 |

复用 `exp/common/data/cache_artifacts/libero_10/*.pkl`（与 Phase1 libero_10 同一套 artifact）；runner 通过 WebSocket `load_cache_config` 为每个 run 热切换 cache bundle。

实验在 LIBERO 默认 `pruned_init` 上跑，每 sub task 10 个 init（`--episodes-per-run 10`）。

## 端口映射

沿用 Phase1 libero_10 的约定（server 本地端口 + 1000 = frp 公网端口）。若仍在跑 Phase1 须选不同端口避开冲突。

| batch | trajectory_depth | server 本地端口 | frp 公网端口 | 种子 yaml |
|---|---:|---:|---:|---|
| `batch1` | 4 | `7998` | `8998` | `traj_d4_001_a_rrf_w5.yaml` |
| `batch2` | 5 | `7999` | `8999` | `traj_d5_021_a_rrf_w5.yaml` |
| `batch3` | 6 | `8000` | `9000` | `traj_d6_041_a_rrf_w5.yaml` |

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

数据落盘目录（runner 第一次写入时自动创建，提前 ls 确认存在也无害）：

```bash
ls exp/common/data/trajectory/libero_10/batch1 \
   exp/common/data/trajectory/libero_10/batch2 \
   exp/common/data/trajectory/libero_10/batch3
```

---

## 1. 服务器命令（GPU 服务器侧，三终端）

每个 batch 一个独立 server；`--cache-config` 传该 batch 的第一个 yaml 作为种子（runner 会 WebSocket 热切换后续配置）。

### Server A: batch1 (depth=4), local port 7998

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/common/config/trajectory/libero_10/batch1/traj_d4_001_a_rrf_w5.yaml \
    --env LIBERO \
    --port 7998 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

### Server B: batch2 (depth=5), local port 7999

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/common/config/trajectory/libero_10/batch2/traj_d5_021_a_rrf_w5.yaml \
    --env LIBERO \
    --port 7999 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

### Server C: batch3 (depth=6), local port 8000

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/common/config/trajectory/libero_10/batch3/traj_d6_041_a_rrf_w5.yaml \
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

**`--state-path` 与 `--log-dir` 显式指向 `data/`**，否则 runner 默认把结果写回 `--yaml-dir`，实验产物又会落在 `config/` 下（这是 Phase1 libero_10 踩过的坑，已修）。

### Client 1: batch1 (depth=4) via frp port 8998

```bash
uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/common/config/trajectory/libero_10/batch1 \
    --state-path exp/common/data/trajectory/libero_10/batch1/experiment_state.json \
    --log-dir exp/common/data/trajectory/libero_10/batch1 \
    --episodes-per-run 10 \
    --num-workers 5 \
    --host 155.98.36.13 --port 8998 \
    --task-suite libero_10 \
    --seed 42 \
    --conda-env /scratch/zixuans8/libero_sim \
    --resume
```

### Client 2: batch2 (depth=5) via frp port 8999

```bash
uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/common/config/trajectory/libero_10/batch2 \
    --state-path exp/common/data/trajectory/libero_10/batch2/experiment_state.json \
    --log-dir exp/common/data/trajectory/libero_10/batch2 \
    --episodes-per-run 10 \
    --num-workers 5 \
    --host 155.98.36.13 --port 8999 \
    --task-suite libero_10 \
    --seed 42 \
    --conda-env /scratch/zixuans8/libero_sim \
    --resume
```

### Client 3: batch3 (depth=6) via frp port 9000

```bash
uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/common/config/trajectory/libero_10/batch3 \
    --state-path exp/common/data/trajectory/libero_10/batch3/experiment_state.json \
    --log-dir exp/common/data/trajectory/libero_10/batch3 \
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

按上面的 `--state-path` / `--log-dir` 设置，每个 run 写到 `exp/common/data/trajectory/libero_10/batch{N}/`：

- `traj_dX_NNN_<kb>_rrf_<wid>.log` — stdout 日志
- `traj_dX_NNN_<kb>_rrf_<wid>.episode_results.json` — 逐 episode 成功/失败
- `traj_dX_NNN_<kb>_rrf_<wid>.timing.csv` — 每步耗时（若 timer.enabled）
- `experiment_state.json` — runner 断点续跑状态（`--resume` 读取）
- `cache_eval_results.json` — runner 汇总 success_rate

---

## 5. 失败时的恢复

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
