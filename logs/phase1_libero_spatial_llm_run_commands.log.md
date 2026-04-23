> Status: Plan
> Date: 2026-04-23
> Level: L1

# Phase1 libero_spatial_llm 执行命令清单

196 个 yaml 分 6 个 batch，六服务器 / 六客户端并行：

- `exp/common/config/phase1/libero_spatial_llm/batch1/` — 32 份（001–032，w1/w2 sweep）
- `exp/common/config/phase1/libero_spatial_llm/batch2/` — 32 份（033–064，w3/w4 sweep）
- `exp/common/config/phase1/libero_spatial_llm/batch3/` — 32 份（065–096，w5/w6 sweep）
- `exp/common/config/phase1/libero_spatial_llm/batch4/` — 32 份（097–128，w7/w8 sweep）
- `exp/common/config/phase1/libero_spatial_llm/batch5/` — 32 份（129–160，p1/p2 prompt_emb 权重档）
- `exp/common/config/phase1/libero_spatial_llm/batch6/` — 36 份（161–192 p3/p4 + 193–196 prefix_mean_pool × l0..l3）

每 batch 内遍历顺序：`reducer → extract_layer → weight_tag`，对应 4 reducer × 4 layer × 2 weight = 32；batch6 额外挂 4 份 prefix_mean_pool。

Runner 通过 WebSocket `load_cache_config` 为每个 run 热切换 cache bundle；server 启动时 `--cache-config` 只是种子。

在 `exp/common/data/cache_artifacts/libero_spatial/llm_layer_extract/` 下的 20 个 pkl 上执行（5 reducer × 4 layer）。

## 端口映射

沿用 phase1_libero_10 的约定：server 本地端口 + 1000 = frp 公网端口（`155.98.36.13`）。

| batch | server 本地端口 | frp 公网端口 | 种子 yaml |
|---|---:|---:|---|
| `batch1` | `7998` | `8998` | `phase1_run_001_a_l0_rrf_w1.yaml` |
| `batch2` | `7999` | `8999` | `phase1_run_033_a_l0_rrf_w3.yaml` |
| `batch3` | `8000` | `9000` | `phase1_run_065_a_l0_rrf_w5.yaml` |
| `batch4` | `8001` | `9001` | `phase1_run_097_a_l0_rrf_w7.yaml` |
| `batch5` | `8002` | `9002` | `phase1_run_129_a_l0_rrf_p1.yaml` |
| `batch6` | `8003` | `9003` | `phase1_run_161_a_l0_rrf_p3.yaml` |

公网 host：`155.98.36.13`。若 frp 上 9001–9003 未开，提前让运维或本人在 frp config 里加 6 条 tcp 映射。

---

## 前置检查

客户端机器确认六个入口健康：

```bash
for p in 8998 8999 9000 9001 9002 9003; do
    printf 'port %s: ' "$p"
    curl -s "http://155.98.36.13:${p}/healthz" || echo
done
```

六个都应返回：`OK`

确认 pkl 齐全（远程 GPU 服务器 home 下，用 tar 解压后产物）：

```bash
ls exp/common/data/cache_artifacts/libero_spatial/llm_layer_extract/ | wc -l
# 期望：20（5 reducer × 4 layer）
```

---

## 1. 服务器命令（GPU 服务器侧，六终端）

每个 batch 一个独立 server；`--cache-config` 传该 batch 的第一个 yaml 作种子（runner 会 WebSocket 热切换后续配置）。

### Server A: batch1, local port 7998

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/common/config/phase1/libero_spatial_llm/batch1/phase1_run_001_a_l0_rrf_w1.yaml \
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
    --cache-config exp/common/config/phase1/libero_spatial_llm/batch2/phase1_run_033_a_l0_rrf_w3.yaml \
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
    --cache-config exp/common/config/phase1/libero_spatial_llm/batch3/phase1_run_065_a_l0_rrf_w5.yaml \
    --env LIBERO \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

### Server D: batch4, local port 8001

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/common/config/phase1/libero_spatial_llm/batch4/phase1_run_097_a_l0_rrf_w7.yaml \
    --env LIBERO \
    --port 8001 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

### Server E: batch5, local port 8002

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/common/config/phase1/libero_spatial_llm/batch5/phase1_run_129_a_l0_rrf_p1.yaml \
    --env LIBERO \
    --port 8002 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

### Server F: batch6, local port 8003

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/common/config/phase1/libero_spatial_llm/batch6/phase1_run_161_a_l0_rrf_p3.yaml \
    --env LIBERO \
    --port 8003 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

> 每个 client 只能指向一个独立 server（`load_cache_config` 不能跨 batch 并发），所以六个 batch 的 client 必须各自对应一台 server，不要复用端口。

---

## 2. 客户端 Runner 命令（LIBERO eval 主机，六终端）

`libero` 不在 uv venv 里，必须通过 `--conda-env` 走 conda（绝对路径时 runner 自动用 `conda run -p`）。下面以 `/scratch/zixuans8/libero_sim` 为例；按名字装的 env 改成 `--conda-env libero_sim` 即可。

### Client 1: batch1 via frp port 8998

```bash
uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/common/config/phase1/libero_spatial_llm/batch1 \
    --episodes-per-run 10 \
    --num-workers 5 \
    --host 155.98.36.13 --port 8998 \
    --task-suite libero_spatial \
    --seed 42 \
    --conda-env /scratch/zixuans8/libero_sim \
    --resume
```

### Client 2: batch2 via frp port 8999

```bash
uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/common/config/phase1/libero_spatial_llm/batch2 \
    --episodes-per-run 10 \
    --num-workers 5 \
    --host 155.98.36.13 --port 8999 \
    --task-suite libero_spatial \
    --seed 42 \
    --conda-env /scratch/zixuans8/libero_sim \
    --resume
```

### Client 3: batch3 via frp port 9000

```bash
uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/common/config/phase1/libero_spatial_llm/batch3 \
    --episodes-per-run 10 \
    --num-workers 5 \
    --host 155.98.36.13 --port 9000 \
    --task-suite libero_spatial \
    --seed 42 \
    --conda-env /scratch/zixuans8/libero_sim \
    --resume
```

### Client 4: batch4 via frp port 9001

```bash
uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/common/config/phase1/libero_spatial_llm/batch4 \
    --episodes-per-run 10 \
    --num-workers 5 \
    --host 155.98.36.13 --port 9001 \
    --task-suite libero_spatial \
    --seed 42 \
    --conda-env /scratch/zixuans8/libero_sim \
    --resume
```

### Client 5: batch5 via frp port 9002

```bash
uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/common/config/phase1/libero_spatial_llm/batch5 \
    --episodes-per-run 10 \
    --num-workers 5 \
    --host 155.98.36.13 --port 9002 \
    --task-suite libero_spatial \
    --seed 42 \
    --conda-env /scratch/zixuans8/libero_sim \
    --resume
```

### Client 6: batch6 via frp port 9003

```bash
uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/common/config/phase1/libero_spatial_llm/batch6 \
    --episodes-per-run 10 \
    --num-workers 5 \
    --host 155.98.36.13 --port 9003 \
    --task-suite libero_spatial \
    --seed 42 \
    --conda-env /scratch/zixuans8/libero_sim \
    --resume
```

Client 1–5 各跑 32 run × 10 task × 10 episode = 3200 episodes；Client 6 跑 36 run × 10 task × 10 episode = 3600 episodes（含 4 份 prefix_mean_pool 单字段 baseline）。

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

指定 CUDA device（客户端侧）：

```bash
--cuda '0'
```

---

## 4. 结果位置

每个 run 输出写到与 yaml 同目录，文件名前缀与 yaml 对应：

- `phase1_run_NNN_<group>_l<layer>_rrf_<tag>.log` — stdout 日志
- `phase1_run_NNN_<group>_l<layer>_rrf_<tag>.episode_results.json` — 逐 episode 成功/失败
- `phase1_run_NNN_<group>_l<layer>_rrf_<tag>.timing.csv` — 每步耗时（若 `timer.enabled=true`）
- `experiment_state.json` — runner 断点续跑状态（`--resume` 读取）

`--log-dir` 可把 stdout 日志另外集中到其他目录。

---

## 5. 关键 Caveat

### 5.1 Reducer ↔ group tag 对应

文件名里的 group tag 与 `prefix_reducer.type` 的一一映射（沿用 phase1_libero_10 的字母约定扩展）：

| tag | reducer | 源 legacy |
|---|---|---|
| `a`  | `per_modality_mean_pool`         | `cp1_mean_pool` |
| `b1` | `per_modality_spatial_pool_16`   | `cp1_spatial_pool_16` |
| `b2` | `per_modality_spatial_pool_4`    | `cp1_spatial_pool_64`（alias） |
| `c`  | `per_modality_max_pool`          | `cp1_max_pool` |
| `e`  | `prefix_mean_pool`               | 无（新增，LLM-only 单字段 baseline） |

### 5.2 extract_layer 集合

每份 yaml 的 `key_builder.extract_layer ∈ {0, 1, 2, 3}`，对应 PaliGemma language_model 的前 4 层 hidden state（depth=18）。pkl artifact 和 yaml 一一对应：`preload_path` 里 `cp1_llm_l{L}_{reducer}.pkl` 的 `L` 必须与 yaml 的 `extract_layer` 相同。生成器已对齐，不要手改。

### 5.3 prefix_mean_pool 特殊性

`prefix_mean_pool` 只 emit `vision_0` 单字段（全 prefix masked mean），`vision_1/2/prompt_emb` 必须 disabled，否则 `validate_cache_config()` 立即 fail。batch6 的 4 份 `..._e_l{0..3}_rrf_const.yaml` 固定权重 `vision_0=1.0, robot_state=1.0`，不参与 weight sweep。

### 5.4 init-state

实验用 LIBERO 默认 `pruned_init`（每 task 50 条，runner 默认流程）。pkl artifact 由 `exp/common/data/db/libero_cache/libero_spatial/` 的 50 个成功 episode 构建，客户端侧 `--episodes-per-run 10` 抽到的 init 大多数不在 pkl 构建集中；命中率主要靠 step-level key 在其他 trajectory 上的泛化。

### 5.5 spatial_pool_16 的在线代价

`per_modality_spatial_pool_16` 每个 vision 字段 dim=32768，三相机合计 98304 维做 cosine similarity；若 in-memory backend 用 `brute_force` 索引，单步 search 延迟可能显著高于 `per_modality_mean_pool`（2048 维）。若发现延迟异常，可在 batch3/batch5/batch6 里单独跳过 `b1` run。

### 5.6 frp 端口准备

batch4–6 使用公网 9001–9003，若 frp 服务端尚未开这三个 tcp 映射，server 启动后客户端会收到连接拒绝。确认 `frpc.toml` / `frps.toml` 里 `[tcp-9001] ... remote_port=9001` 等配置齐全后再开始。

---

## 6. 执行次序（推荐）

1. 客户端 host 先跑前置 `curl` 健康检查六端口（不通则调 frp）。
2. GPU 服务器开六个 tmux / screen 面板，同时起 6 个 `serve_policy.py`（§1）。
3. 客户端 host 开六个 tmux / screen 面板，同时起 6 个 `run_cache_experiments.py`（§2）。
4. 监控：每个 run 约 5–10 min（10 episode），总预计 32 × 10min / 并行 ≈ 5–6 小时；batch6 多 4 run，预计 5.5–6.5 小时。
5. 所有 client 结束后，运行结果分析（后续补充 §分析脚本清单）。
