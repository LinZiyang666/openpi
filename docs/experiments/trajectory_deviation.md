# Trajectory Deviation 实验运行指南

> 基于 `logs/trajectory_deviation_experiment_plan.log.md` 与 `logs/trajectory_deviation_corrective_experiment.log.md`，对齐当前 `exp/trajectory_deviation/` 重构后的脚本路径与 CLI。英文版：[trajectory_deviation.en.md](trajectory_deviation.en.md)。

---

## 实验总览

对每个 cache 配置（`clip_w7_d4` / `spatial16_w8_d4` / `max_pool_w3_d5`）回答两个问题：

1. Cache 把 inference 推离 GT 轨迹的程度，是否可以被 **deviate score** 量化？
2. 对 deviate score 排名前 k 的 cycle，若将 env teleport 回 GT `sim_state` + 纯 cache rollout，能否恢复到成功？

完整流水线：

```
Step 1a  baseline cache 实验        → cache_eval_results.json（含 success flag）
Step 1b-pre dump_step1a_failed_inits → per-task .init + step1b_filter.json
Step 1b  run_step1b_gt.py           → GT HDF5（AlwaysSkip，建议 max_pool inference bundle）
Step 2   compute_deviate_scores     → deviate_score_{cfg}.json（Phase1 M 次 + Phase2 1 次 + Phase3 aggregate）
Step 3   run_step3_per_cycle_policy → results.jsonl（每个 cfg 一个服务端 + 客户端）
         merge_step3_cfgs           → summary.csv（跨 cfg 聚合）
```

## 网络拓扑（与 CP1 实验一致）

```
┌─────────────────────────┐        frp 隧道/内网         ┌────────────────────────┐
│  GPU 服务器 (无公网IP)    │ ◄──────────────────────────│  LIBERO 评估端          │
│  serve_policy.py         │   155.98.36.13:9000         │  exp/trajectory_       │
│  监听 localhost:8000     │   → localhost:8000          │  deviation/*.py         │
│  --concurrent           │                              │  examples/libero/main   │
└─────────────────────────┘                              └────────────────────────┘
```

- **GPU 服务器**：承载 Pi0.5 + cache backend，整条流水线只起一次。
- **评估端**：跑 libero env + 所有 `exp/trajectory_deviation/*.py` driver；`main.py` 子进程通过 `--conda-env libero_sim` 在 conda 里起。
- 网络连通性与 CP1 完全一样——如果你已经能跑 `cp1_cache.md` 的 Step 4 / 5，这里直接复用。

---

## 并行性一览（先看这张表再决定怎么跑）

| 阶段 | 可并行的维度 | 不能并行的维度 | 备注 |
|------|-------------|--------------|------|
| Step 1a | task 级（`run_cache_experiments.py --num-workers`） | 不同 YAML 的 runs 串行 | 各 run 共用同一 server，server 必须 `--concurrent` |
| Dump failed inits | — | 单进程 | 纯离线 JSON → tensor 切片，秒级 |
| Step 1b GT 采集 | **不能并行** | 每个 unit 都是独立 `main.py` 子进程 + libero env；当前 runner 使用串行 `run()` 而非 `parallel_run()` | 强行多开会抢 GPU/显存；同一 AlwaysSkip bundle 切一次即可（建议 `inference_max_pool_w3_d5.yaml`，避免加载 CLIP） |
| Step 2 Phase 1 / Phase 2 | episode × sample（`--num-workers`） | 不同 cfg 串行；同 cfg 的 Phase1 → Phase2 串行 | Driver 在阶段切换时 `send_load_cache_config` 重载 server bundle，并行会互相抢改 |
| Step 2 Phase 3 (aggregate) | — | 单进程，纯 numpy | 每个 cfg 独立输出 |
| Step 3 per-cycle | episode × (τ, n)（`--num-workers` ≤ 5，MuJoCo EGL 上限） | 同一 server 内不同 cfg 串行 | 推荐三 cfg 三 server 并行；每个 client 进程只跑 `--cfg` 中的一个 |

> **能并行就开** `--num-workers`；**不能并行的地方（cfg 切换、Step 1b）强行多开会产生脏数据**——server bundle 被抢着改、HDF5 被覆盖。

---

## 前置条件

1. 两端已完成 `GIT_LFS_SKIP_SMUDGE=1 uv sync`。
2. GPU 服务器上有 Pi0.5 LIBERO checkpoint，本地路径默认 `$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch`（用 `--policy.dir` 指向它，避免每次启动都从 GCS 拉）。
3. 评估端已装 `libero_sim` conda 环境。
4. 已按 [cp1_cache.md](cp1_cache.md) 的 Step 1–2 构建好三类 cache artifact + calibration：
   - CLIP 对应 `clip_vit_b_32.pkl`（`clip_w7_d4.yaml` / `inference_clip_w7_d4.yaml` 引用）
   - Spatial-pool-16 对应 `cp1_spatial_pool_16.pkl`（`spatial16_w8_d4.yaml` / `inference_spatial16_w8_d4.yaml`）
   - Max-pool 对应 `cp1_max_pool.pkl`（`max_pool_w3_d5.yaml` / `inference_max_pool_w3_d5.yaml`）
5. `configs/cache_runs/deviate_exp/` 下 6 个 YAML（3 对 `inference_*.yaml` + 真实 cache YAML）已就位，`preload_path` 指向 GPU 服务器文件系统上的 artifact。

---

## Step 0：启动 GPU 服务器（一次到位）

一个 server 贯穿整条流水线，因此 flag 一次配齐——后续每个 driver 会自己调 `send_load_cache_config` 切 bundle，server 不重启。

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache_config configs/cache_runs/deviate_exp/inference_max_pool_w3_d5.yaml \
    --env LIBERO \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

关键参数说明：

| Flag | 含义 | 本实验必须开吗 |
|------|------|---------------|
| `--concurrent` | 打开多连接模式，允许多 worker 同时 WebSocket + 允许 `load_cache_config` 控制消息 | **必须**。Step 2/3 的 `--num-workers>1`、以及 driver 切 bundle 都依赖它 |
| `--cache_config` | 启动时的初始 bundle | 任选一个 AlwaysSkip bundle 即可，建议填 GT bundle（`inference_max_pool_w3_d5.yaml`），这样 Step 1b 可以用 `--skip-config-switch` 且不会为 GT 采集加载 CLIP |
| `--env LIBERO` | 选择环境分支 | 本实验固定 LIBERO，必须 |
| `--port 8000` | 监听端口 | 与评估端 driver 的 `--port` 对齐（若走 frp 隧道，评估端填映射后的外网端口，例如 `9000`） |
| `policy:checkpoint --policy.config pi05_libero --policy.dir <本地路径>` | 使用本地 checkpoint 避免 GCS 下载 | 强烈建议填本地路径 |

验证 server 就绪（在评估端）：

```bash
curl http://<server-host>:<server-port>/healthz
# 输出: OK
```

> Step 0 的启动命令与 `cp1_cache.md` Step 4 一致；如果你之前跑过 CP1，直接复用即可。

---

## Step 1a：Baseline Cache 评估

目的：拿到 per-episode 的 `success` flag，定位"cache 失败"的 episode。

```bash
uv run exp/cache_experiment/run_cache_experiments.py \
    --yaml-dir configs/cache_runs/deviate_exp \
    --task-suite libero_spatial \
    --host <server-host> --port <server-port> \
    --episodes-per-run 50 \
    --num-workers 5 \
    --seed 42 \
    --conda-env libero_sim \
    --state-path data/deviation_experiment/step1a_state.json
```

参数说明与调节建议：

| Flag | 含义 | 调节建议 |
|------|-----|---------|
| `--yaml-dir` | 要跑的 YAML 集合。driver 会对 dir 下每个 `*.yaml` 顺序起一次 run | 用 `configs/cache_runs/deviate_exp`（6 个 YAML；3 个 `inference_*` 其实是 AlwaysSkip 对照，评估端跑它们也没问题，只是全部 success_rate 会很接近 pure inference） |
| `--episodes-per-run` | 每个 task 跑多少 episode。LIBERO 每个 task 有 50 个 init，默认全跑 | 想要更干净的失败集就 50；只想快速过流程可以 10 |
| `--num-workers` | 每个 run 内部的 task 级并行度。workers 共用一个 server | server `--concurrent` 打开后 4 比较安全；卡多、显存宽松可以调到 8 |
| `--seed` | 传给 `main.py` 的随机种子，控制 libero env 的随机性 | 固定 42 便于复现；和数据采集的 seed=7 区分，避免过拟合 |
| `--conda-env` | `main.py` 必须在 libero conda 环境跑 | 固定 `libero_sim` |
| `--state-path` | 断点续跑状态文件 | 指向实验目录下一个稳定位置，后续加 `--resume` 即可接着跑 |

产物：默认写到 `<yaml-dir>/cache_eval_results.json`。把它拷到或软链到实验根目录，供 Step 1b-pre 读取：

```bash
cp configs/cache_runs/deviate_exp/cache_eval_results.json \
   data/deviation_experiment/cache_eval_results.json
```

---

## Step 1b-pre：Dump Failed Inits（离线单进程，秒级）

```bash
uv run python scripts/dump_step1a_failed_inits.py \
    --step1a-results data/deviation_experiment/cache_eval_results.json \
    --task-suite libero_spatial \
    --out-dir data/deviation_experiment/inits
```

参数说明：

| Flag | 含义 | 调节建议 |
|------|-----|---------|
| `--step1a-results` | Step 1a 聚合 JSON | 必须是经过 `_aggregate_episode_results` 去重后的版本（`cache_eval_results.json`），不要用单个 run 的 episode 日志 |
| `--task-suite` | LIBERO suite 名，用于查 task.name 与 init states | 与 Step 1a 保持一致 |
| `--out-dir` | 产物目录 | 建议放实验根 `data/deviation_experiment/inits`，后续几步都引用它 |
| `--no-torch` | 跳过写 `.init` tensor，仅写 JSON | CI smoke 用；正常运行不要加 |

产物：
- `<out>/<task_name>.init` — 仅失败 init 的 `(K, 92)` tensor 子集
- `<out>/<task_name>.init_map.json` — subset → orig 索引映射
- `<out>/step1b_filter.json` — 下一步的 `--episode-filter`

---

## Step 1b：GT 轨迹采集（**串行，必须用 AlwaysSkip**）

Step 1b 只需要一个 **AlwaysSkip / pure inference** bundle；Step 2 的三个配置必须共享同一批 GT，否则 deviate_score 的分母失去可比性。具体用哪个 `inference_*.yaml` 不影响 GT 动作语义，因为 gate 是 `always_skip`，不会读 cache action。建议使用 `inference_max_pool_w3_d5.yaml`：它不触发 CLIP key builder，避免 Step 1b 为无用的 CLIP 模型占用显存。Runner 启动时调一次 `send_load_cache_config`，然后所有 unit 串行跑 `main.py`（每个 unit 单独起一个 libero env 子进程）。

```bash
uv run python -m exp.trajectory_deviation.run_step1b_gt \
    --inits-dir data/deviation_experiment/inits \
    --out-dir data/deviation_experiment/gt \
    --task-suite libero_spatial \
    --host <server-host> --port <server-port> \
    --inference-yaml configs/cache_runs/deviate_exp/inference_max_pool_w3_d5.yaml \
    --seed 7 \
    --conda-env libero_sim \
    --resume
```

参数说明：

| Flag | 含义 | 调节建议 |
|------|-----|---------|
| `--inits-dir` | Step 1b-pre 的产物目录 | 必填；driver 会读 `step1b_filter.json` |
| `--out-dir` | GT HDF5 根目录（传给 `main.py --save-trajectory-dir`） | 与 Step 2 `--gt-dir` 保持一致；一般 `data/deviation_experiment/gt` |
| `--task-suite` | LIBERO suite 名 | 与 Step 1a/1b-pre 一致 |
| `--host / --port` | server 地址 | 评估端这里填 frp 外网端口；server 本机则 `localhost:8000` |
| `--seed` | 传给 `main.py` 的 seed | **固定 7**（plan §9.2 默认值），Step 2 background 噪声建模默认此 seed；换 seed 会让 Phase 1 的 M 次采样与 GT 分布错位 |
| `--conda-env` | libero conda 环境 | 固定 `libero_sim` |
| `--inference-yaml` | GT bundle YAML 路径 | 必须是 `gate.type: always_skip` 的 pure inference bundle；建议 `configs/cache_runs/deviate_exp/inference_max_pool_w3_d5.yaml`，避免加载 CLIP。不要换成真实 cache YAML（如 `clip_w7_d4.yaml` / `max_pool_w3_d5.yaml`） |
| `--state-path` | runner 状态文件 | 默认 `<inits-dir>/step1b_state.json`；断点续跑配合 `--resume` |
| `--resume` | 断点续跑 | 调 runner 时几乎必开 |
| `--skip-config-switch` | 不调 `load_cache_config`，假定 server 已经在正确 bundle | 多 GPU 分片、手动切过 bundle 时才用；默认留着让 runner 自己切 |

调节建议：

- **想加快？** 不能通过并行加快（runner 本身串行）；能做的只有：增加 `max_retries` 之外的不做任何事情。如果真的要并行化 Step 1b，需要重构 runner 用 `parallel_run` + 多 libero env 子进程，plan 里没有授权。
- **某个 unit 老 fail？** 调大 `max_retries`（当前默认 2）或手动用 `examples/libero/main.py` 单跑那个 `(task_id, orig_init)` 复现调试。
- 产物：`gt/task_{id}/episode_{subset_idx}.h5`。

---

## Step 2：Deviate Score（每个 cfg 内部并行，cfg 之间串行）

一次跑所有 3 个 cfg（driver 自己按序切 bundle）：

```bash
uv run python -m exp.trajectory_deviation.compute_deviate_scores \
    --configs clip_w7_d4 spatial16_w8_d4 max_pool_w3_d5 \
    --gt-dir data/deviation_experiment/gt \
    --out-dir data/deviation_experiment/deviate_scores \
    --M 5 \
    --num-workers 5 \
    --host <server-host> --port <server-port> \
    --floor 0.1 \
    --config-fail-results data/deviation_experiment/cache_eval_results_cache_fail.json \
    --resume
```

每个 cfg 的内部顺序：

```
load inference_{cfg}.yaml → Phase1Runner  (parallel, M × episodes)   → bg_{cfg}.jsonl
load {cfg}.yaml           → Phase2Runner  (parallel, 1 × episodes)   → cache_{cfg}.jsonl
offline Phase3 aggregate                                              → deviate_score_{cfg}.json
```

参数详细说明与调节：

| Flag | 默认 | 含义 | 调节建议 |
|------|------|-----|---------|
| `--configs` | — | 要跑的 cache 配置 ID 列表（不带 `.yaml` 后缀，driver 自己拼 `inference_{cfg}.yaml` / `{cfg}.yaml`） | 三者全跑，对比才有意义；调试可以只传一个 |
| `--gt-dir` | — | Step 1b 产物根 | 必须与 Step 1b `--out-dir` 一致 |
| `--out-dir` | — | 所有 jsonl + state + `deviate_score_*.json` 的根 | 固定 `data/deviation_experiment/deviate_scores` |
| `--M` | 20 | Phase 1 的随机采样次数，用于估计 "背景 L2 噪声" | **这是 deviate_score 的信噪比旋钮**。M 越大 background L2 估计越稳但 Phase 1 时间线性增长。plan §10.2 参考 20；快速过流程用 10；想锁定统计显著性可以 30 |
| `--num-workers` | 4 | 同 cfg 内 worker 数（server 端必须 `--concurrent`） | 受 server GPU 容量限制；4 通常安全；卡够大可以 8 |
| `--host / --port` | `localhost` / `8000` | server 连接 | 与 Step 0 对齐 |
| `--floor` | 0.1 | deviate_score 的分母下限 `max(bg_l2, floor)`，防止除零放大 | **不建议改**（plan §10.2 凭经验定死）；若 M 很小且 bg_l2 常常低于 0.1，可以调到 0.05 观察分布变化 |
| `--config-yaml-dir` | `configs/cache_runs/deviate_exp` | YAML 解析根 | 除非要换实验，否则不动 |
| `--config-fail-results` | off | 可选 Step 1a 结果 JSON；开启后每个 cfg 只跑该 cfg 自己失败过的 `(task_id, orig_init_state_idx)` | 推荐填 `data/deviation_experiment/cache_eval_results_cache_fail.json`，避免把已经成功的 init 也拿去算 deviate score；不填则三个 cfg 共用完整成功 GT 集 |
| `--skip-config-switch` | off | 不调 `load_cache_config`，假定 server 已正确 | 多 server 分片时用（一台 server 跑 clip，一台跑 spatial16，三个 client 指向各自 server，每个都加此 flag） |
| `--include-failed-gt` | off | 保留 `success=False` 的 GT episode | 默认剔除（Step 1b 里推不到目标的 episode 不携带 recovery 信号）；仅在做噪声分析时开 |
| `--include-unknown-gt` | off | 保留旧版无 `success` attr 的 HDF5 | 仅兼容归档数据；重跑 Step 1b 即可避免 |
| `--resume` | off | 按 state 文件续跑 | 几乎总开 |

调节实战：

- **想快速看到结果？** `--M 10 --num-workers 4 --configs clip_w7_d4`：单 cfg + 一半采样，跑完就能看 deviate_score_clip_w7_d4.json 分布。
- **想做统计比较？** `--M 20` 以上；三个 cfg 全跑；`--num-workers` 视 server 能力尽量调高。
- **Phase1 / Phase2 bundle 跑错了？** 通常是你并行起了多个 driver 指同一 server（它们抢着调 `load_cache_config`）。正确做法：单进程串行 cfg，或每个 cfg 独享一台 server + `--skip-config-switch`。

⚠️ **不要** 并排起两个 `compute_deviate_scores` 进程对同一 server——bundle 会被抢改，Phase 1 可能读到 Phase 2 的 bundle，输出的 deviate_score 会看起来全在 1.0 附近（毫无意义）。

---

## Step 3：Per-Cycle Policy（推荐三 cfg 三 server 完全并行）

> `run_step3_per_cycle_policy` 把"每个推理 cycle 是否绕过 cache"的决定交给 client，通过 `__gate_decision__` 信号驱动 `ClientControlledGate`。详细重构记录见 [`logs/trajectory_deviation_step3_redesign.log.md`](../../logs/trajectory_deviation_step3_redesign.log.md)。
>
> Episode 列表权威来源：`deviate_score_{cfg}.json` 的 keys（`task_X/episode_Y`，其中 `Y` 是 Step 1b 的 subset_init_state_idx）。这些 key 本身就是该 cfg 在 Step 1a 下的失败子集，Step 3 不再需要 `--cache-eval-results` / `--config-fail-results`。

### 前置条件

1. 已有三份 Step 2 产物：`deviate_score_clip_w7_d4.json` / `deviate_score_spatial16_w8_d4.json` / `deviate_score_max_pool_w3_d5.json`（按 [Step 2 并行流程](../../logs/trajectory_deviation_step2_parallel_commands.log.md) 合入 `data/deviation_experiment/deviate_scores/`）。
2. 已有 Step 1b pruned init states：`data/deviation_experiment/inits/`。
3. `configs/cache_runs/deviate_exp/step3_{cfg}.yaml` 已就位；三份 YAML 的 `checkpoints.cp1.gate.type: client_controlled`，其他字段与对应 `{cfg}.yaml` 一致。
4. 三个 server 沿用 Step 2 的端口映射：

    | config | server 本机端口 | frp 外网端口 |
    |---|---:|---:|
    | `clip_w7_d4` | `7998` | `8998` |
    | `spatial16_w8_d4` | `7999` | `8999` |
    | `max_pool_w3_d5` | `8000` | `9000` |

    Public host：`155.98.36.13`

### Server 命令

若 Step 2 的三个 server 还在，**无需重启**——`run_step3_per_cycle_policy` 会在启动时调用一次 `send_load_cache_config` 把 bundle 切到 `step3_{cfg}.yaml`。若已停机，按下列命令拉起（任一 `--cache-config` 都可，runner 自己会切）。

#### Server A：clip，本机端口 7998

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config configs/cache_runs/deviate_exp/step3_clip_w7_d4.yaml \
    --env LIBERO \
    --port 7998 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

#### Server B：spatial16，本机端口 7999

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config configs/cache_runs/deviate_exp/step3_spatial16_w8_d4.yaml \
    --env LIBERO \
    --port 7999 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

#### Server C：max_pool，本机端口 8000

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config configs/cache_runs/deviate_exp/step3_max_pool_w3_d5.yaml \
    --env LIBERO \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

健康检查（在评估端）：

```bash
curl http://155.98.36.13:8998/healthz
curl http://155.98.36.13:8999/healthz
curl http://155.98.36.13:9000/healthz
```

每个应输出 `OK`。

### Client 命令

评估端 / LIBERO 主机上三个终端并行起三个 client 进程。每个 client 绑定一个 cfg 和对应 server，`--num-workers 5`（MuJoCo EGL 上限）；`--tau-grid` / `--n-grid` 逗号分隔。Client 侧可用 GPU `6–7`，按 cfg pin 一下避免互踩。

#### Client 1：clip via frp port 8998（gpu 0）

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python -m exp.trajectory_deviation.run_step3_per_cycle_policy \
    --cfg clip_w7_d4 \
    --host 155.98.36.13 --port 8998 \
    --yaml configs/cache_runs/deviate_exp/step3_clip_w7_d4.yaml \
    --deviate-score-json data/deviation_experiment/deviate_scores/deviate_score_clip_w7_d4.json \
    --init-states-dir data/deviation_experiment/inits \
    --out-dir data/deviation_experiment/step3/clip_w7_d4 \
    --task-suite-name libero_spatial \
    --tau-grid 3,5,7,10 \
    --n-grid 1,2,3,5,10 \
    --num-workers 5 \
    --resume
```

#### Client 2：spatial16 via frp port 8999（gpu 1）

```bash
CUDA_VISIBLE_DEVICES=1 \
uv run python -m exp.trajectory_deviation.run_step3_per_cycle_policy \
    --cfg spatial16_w8_d4 \
    --host 155.98.36.13 --port 8999 \
    --yaml configs/cache_runs/deviate_exp/step3_spatial16_w8_d4.yaml \
    --deviate-score-json data/deviation_experiment/deviate_scores/deviate_score_spatial16_w8_d4.json \
    --init-states-dir data/deviation_experiment/inits \
    --out-dir data/deviation_experiment/step3/spatial16_w8_d4 \
    --task-suite-name libero_spatial \
    --tau-grid 3,5,7,10 \
    --n-grid 1,2,3,5,10 \
    --num-workers 5 \
    --resume
```

#### Client 3：max_pool via frp port 9000（gpu 2）

```bash
CUDA_VISIBLE_DEVICES=2 \
uv run python -m exp.trajectory_deviation.run_step3_per_cycle_policy \
    --cfg max_pool_w3_d5 \
    --host 155.98.36.13 --port 9000 \
    --yaml configs/cache_runs/deviate_exp/step3_max_pool_w3_d5.yaml \
    --deviate-score-json data/deviation_experiment/deviate_scores/deviate_score_max_pool_w3_d5.json \
    --init-states-dir data/deviation_experiment/inits \
    --out-dir data/deviation_experiment/step3/max_pool_w3_d5 \
    --task-suite-name libero_spatial \
    --tau-grid 3,5,7,10 \
    --n-grid 1,2,3,5,10 \
    --num-workers 5 \
    --resume
```

参数详细说明与调节：

| Flag | 默认 | 含义 | 调节建议 |
|------|------|-----|---------|
| `--cfg` | — | 要跑的单一 cache 配置 ID | 必填；一个 client 进程只跑一个 cfg |
| `--host / --port` | — | server 地址（评估端用 frp 外网端口） | 必填；三 client 指向不同 server，避免 `load_cache_config` 竞争 |
| `--yaml` | — | Step 3 cache YAML，`gate.type: client_controlled` | 必填；路径必须在 server 端文件系统可见 |
| `--deviate-score-json` | — | Step 2 的 `deviate_score_{cfg}.json`，其 keys 即 Step 3 的 episode 列表 | 必填；空 JSON 会报 `has no episodes` |
| `--init-states-dir` | — | Step 1b-pre 产物目录（含 `{task}.init` / `{task}.init_map.json`） | 必填 |
| `--out-dir` | — | 输出根：`run_state.json` + `results.jsonl` | 一般 `data/deviation_experiment/step3/{cfg}` |
| `--task-suite-name` | `libero_spatial` | 与 Step 1a/1b 保持一致；决定 `_MAX_STEPS_BY_SUITE` | 与前面的 step 保持一致 |
| `--tau-grid` | `3,5,7,10` | deviate_score 阈值网格（标量，非 burst 长度） | **核心旋钮之一**：τ 小 = 触发更频繁的 search；τ 大 = 更多 skip |
| `--n-grid` | `1,2,3,5,10` | 首次触发 search 后连续走 cache 的 cycle 数（burst 长度） | **另一核心旋钮**：n 大 = search 后让 cache "吃一段"再复位；n=1 = 每次都重新判断 |
| `--replan-steps` | 5 | 每个 inference cycle 内真正执行的 env.step 数 | 与 `examples/libero/main.py` 一致，默认 5；降低可提高控制分辨率但加大调用压力 |
| `--num-steps-wait` | 10 | env reset 后先 no-op 的 env.step 数（libero "物体掉落稳定"）| 与 libero 主脚本一致，默认不改 |
| `--resize-size` | 224 | 策略输入图像分辨率 | 固定 |
| `--num-workers` | 1 | 同 cfg 内 client 并行度 | **最多 5**（MuJoCo EGL 限制，runner 会 hard fail 拦截超限） |
| `--max-cycles-safety` | 5 | 在 `ceil(max_env_steps/replan_steps)` 之外额外留几圈保险丝 | 默认即可 |
| `--experiment-tag` | `trajectory_deviation_step3` | 透传到 `client.episode_start(experiment=...)` | 方便 server 侧日志区分 |
| `--skip-load-cache-config` | off | 跳过启动时的 `send_load_cache_config` | 仅在 server 已手动切好 bundle 时用 |
| `--resume` | off | 按 `run_state.json` 续跑 | 几乎总开 |

`(τ, n)` 组合数 = `len(tau_grid) × len(n_grid)`：默认 4×5=20 种；乘以 episodes × 3 cfgs 就是总 unit 数。调节建议：

- **先探路**：`--tau-grid 5 --n-grid 1,5 --cfg clip_w7_d4 --num-workers 1`，2 组合 × 单 cfg × 单 worker，验证 gate 信号注入 / success 语义。
- **正式跑**：默认 4×5 + 3 cfg，全并行；参考 Step 2 实测三台 server 各 ~3h 完成 150+ 个 episode 的 inference-only 反馈。
- **显存 / GPU 不够？** 优先降 `--num-workers`（≤5），其次缩 `--tau-grid`。

⚠️ 同一 server 上不要起多个 `run_step3_per_cycle_policy`——`load_cache_config` 会竞争，bundle 写时序混乱会导致 gate 信号被丢弃。三 cfg 必须各跑一台 server。

### 合并三 cfg 结果

三个 client 跑完后，用 `merge_step3_cfgs` 把 JSONL 聚合成一张 `(cfg, τ, n)` 级的 CSV。脚本会按 `(cfg, ep, τ, n)` 去重（retry 写入时后者覆盖前者）。

```bash
uv run python -m exp.trajectory_deviation.merge_step3_cfgs \
    --jsonl data/deviation_experiment/step3/clip_w7_d4/results.jsonl \
    --jsonl data/deviation_experiment/step3/spatial16_w8_d4/results.jsonl \
    --jsonl data/deviation_experiment/step3/max_pool_w3_d5/results.jsonl \
    --out data/deviation_experiment/step3/summary.csv
```

产物 `summary.csv` 字段：`cfg, tau, n, episodes, success_rate, mean_inference_ratio, std_inference_ratio`（`std` 为总体标准差 `ddof=0`）。

### 快速验证

```bash
wc -l data/deviation_experiment/step3/*/results.jsonl
```

每个 cfg 的行数应 ≈ `len(episodes(cfg)) × |τ_grid| × |n_grid|`（例：clip_w7_d4 ≈ 159 × 4 × 5 = 3180 行；含 resume retry 时略多，由 merge 去重）。

---

## 参数调节速查表（总结）

按"重要性 × 常改"排序，方便你在实验设计阶段决定哪些要 sweep：

### 核心"实验旋钮"

| 参数 | 所在 Step | 作用 | 典型取值 |
|------|----------|------|---------|
| `--configs` | Step 2/3 | 要对比的 cache 配置 | `clip_w7_d4 spatial16_w8_d4 max_pool_w3_d5`（全部） |
| `--M` (Phase 1) | Step 2 | 背景 L2 噪声估计的采样数 | 10 / 20 / 30 |
| `--tau-grid` | Step 3 | deviate_score 判 search 阈值（逗号分隔） | `3,5,7,10`（默认） |
| `--n-grid` | Step 3 | 首次 search 后连续 skip 的 cycle 数（burst 长度） | `1,2,3,5,10`（默认） |

### 并发 / 性能

| 参数 | 所在 Step | 作用 | 调节建议 |
|------|----------|------|---------|
| `--num-workers` | Step 1a/2 | 同 cfg 内 worker 数 | 受 server GPU 限制；4–5 通常安全 |
| `--num-workers` | Step 3 | 同 cfg 内 LIBERO env 并发 | **最多 5**（MuJoCo EGL 硬上限）|

### 实验锚点（plan 锁死，不轻易改）

| 参数 | 值 | 理由 |
|------|-----|-----|
| GT bundle | `inference_max_pool_w3_d5.yaml` | 三 cfg 共享同一 GT；使用 AlwaysSkip 即可，max-pool inference bundle 不加载 CLIP |
| `--floor` (deviate_score) | 0.1 | plan §10.2 经验值，防 bg_l2≈0 爆分母 |
| `--seed` (Step 1b) | 7 | plan §9.2 默认，Phase 1 统计建立在此 seed 上 |
| `--seed` (Step 1a) | 42 | 与数据采集 seed=7 区分，避免过拟合 |
| LIBERO max_steps | 220 | libero_spatial 硬限 |

---

## 典型目录布局

```
data/deviation_experiment/
├── cache_eval_results.json                # Step 1a 聚合
├── inits/
│   ├── <task>.init
│   ├── <task>.init_map.json
│   └── step1b_filter.json
├── gt/task_{id}/episode_{subset_idx}.h5   # 客户端 GT
├── deviate_scores/
│   ├── bg_{cfg}.jsonl
│   ├── cache_{cfg}.jsonl
│   ├── phase1_state_{cfg}.json
│   ├── phase2_state_{cfg}.json
│   └── deviate_score_{cfg}.json
└── step3/
    ├── clip_w7_d4/
    │   ├── run_state.json
    │   └── results.jsonl
    ├── spatial16_w8_d4/
    │   ├── run_state.json
    │   └── results.jsonl
    ├── max_pool_w3_d5/
    │   ├── run_state.json
    │   └── results.jsonl
    └── summary.csv                     # merge_step3_cfgs 产物
```

---

## 故障排查

| 症状 | 原因 | 处理 |
|------|------|------|
| Step 3 报 `has no episodes` | `deviate_score_{cfg}.json` 为空或路径写错 | 确认 Step 2 产物已复制到 `data/deviation_experiment/deviate_scores/`，文件名 `deviate_score_{cfg}.json` 与 `--cfg` 对齐 |
| Step 3 起不来：`--num-workers=N exceeds MuJoCo EGL cap` | 评估端单进程 libero env 上限 5 | 把 `--num-workers` 降到 ≤5；想更高并发只能再开一台评估主机 |
| Step 2 deviate_score 全部 ≈ 1.0 | Phase 1 / Phase 2 读到了同一个 bundle | 多进程抢了 `send_load_cache_config`；停掉并行 driver，单进程跑；或每 cfg 独立 server + `--skip-config-switch` |
| Step 1b 大量 `inference_failed=True` | server 挂了 / 不在指定的 AlwaysSkip GT bundle | 检查 server 进程；若你加了 `--skip-config-switch`，确认已手动 `load_cache_config` 到 `inference_max_pool_w3_d5.yaml` 或等价 AlwaysSkip bundle |
| Step 1a `main.py` 抱怨 `VIRTUAL_ENV` | uv 环境变量泄漏到 conda 子进程 | `_build_subprocess_cmd` 应已清理；若复现请检查自定义 `--conda-env` wrapper |
| `load_cache_config` 报错 | server 未加 `--concurrent`；或 YAML 的 `preload_path` 在 server 端不存在 | 重启 server 加 `--concurrent`；核对 artifact 在 GPU 服务器上的路径 |

---

## 交叉引用

- Server 启动的更多上下文、artifact 构建、CP1 实验：[cp1_cache.md](cp1_cache.md)
- Cache 系统组件、YAML 字段含义：[../cache/tutorial.md](../cache/tutorial.md)
- 远程推理的网络拓扑：[../deployment/libero.md](../deployment/libero.md)
- 原始实验设计与评审：[../../logs/trajectory_deviation_experiment_plan.log.md](../../logs/trajectory_deviation_experiment_plan.log.md)、[../../logs/trajectory_deviation_corrective_experiment.log.md](../../logs/trajectory_deviation_corrective_experiment.log.md)

---

## 附录：单机 6+2 GPU 同机部署（实验性，待研究）

> **状态：留档待研究，先别按这版跑。**
>
> 2026-04-16 在 `timan107` 上试过这套方案：把三台 server 全部塞进同一台 8 卡机（gpu 0–5 给 server，gpu 6–7 给 client），server 走 `serve_policy.py` 的 `--stage1_device / --stage2_device / --stage3_device` 做 Pi0.5 三阶段切分，client 直连 `127.0.0.1`。结果 `max_pool_w3_d5` 那条 client 在 `[parallel retry 1/2] 3000 failed units` 全跪，原因尚未排查（怀疑 MuJoCo EGL 与 server CUDA 上下文在同机互相挤压，或 init_states 路径 / Server C 启动状态的问题）。
>
> 下面的命令保留是为了后续复现/排查，不要直接当主流程跑。要跑也请先：
>
> 1. 确认三台 server 都拉起来了（`nvidia-smi` 看 gpu 0–5 全部有占用）；
> 2. `curl http://127.0.0.1:7998/healthz` 三个端口都 OK；
> 3. 先用 `--num-workers 1` 单 worker 跑 1 个 unit 验证通路，再逐步加并发。

### 附录·GPU 分配

| 角色 | 绑定 | 物理 GPU（`CUDA_VISIBLE_DEVICES`） | stage1 / stage2 / stage3 | 说明 |
|---|---|:---:|:---:|---|
| Server A | `clip_w7_d4` | `0,1` | `cuda:0` / `cuda:1` / `cuda:1` | stage1 独占物理 gpu0；stage2+3 合占物理 gpu1 |
| Server B | `spatial16_w8_d4` | `2,3` | `cuda:0` / `cuda:1` / `cuda:1` | stage1 → 物理 gpu2；stage2+3 → 物理 gpu3 |
| Server C | `max_pool_w3_d5` | `4,5` | `cuda:0` / `cuda:1` / `cuda:1` | stage1 → 物理 gpu4；stage2+3 → 物理 gpu5 |
| Client 1 | clip → 127.0.0.1:7998 | `6` | — | 与 Client 2 共享 gpu6 |
| Client 2 | spatial16 → 127.0.0.1:7999 | `6` | — | 与 Client 1 共享 gpu6 |
| Client 3 | max_pool → 127.0.0.1:8000 | `7` | — | 独占 gpu7 |

`scripts/serve_policy.py` 支持 `--stage{1,2,3}_device`（Pi0.5 三阶段切分：stage1 vision / stage2 LLM backbone / stage3 action head）。`CUDA_VISIBLE_DEVICES=N,N+1` 把物理两卡重映射为进程内的 `cuda:0`/`cuda:1`，所以三台 server 内部的 stage device 串完全一致。分阶段放置要求传入 `--cache_config`（已满足；否则 `serve_policy` 会拒启，见 `scripts/serve_policy.py:163-176`）。

### 附录·Server 命令（GPU 切分版）

```bash
# Server A：clip，物理 gpu 0+1
CUDA_VISIBLE_DEVICES=0,1 \
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config configs/cache_runs/deviate_exp/step3_clip_w7_d4.yaml \
    --env LIBERO --port 7998 \
    --stage1_device cuda:0 --stage2_device cuda:1 --stage3_device cuda:1 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"

# Server B：spatial16,物理 gpu 2+3
CUDA_VISIBLE_DEVICES=2,3 \
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config configs/cache_runs/deviate_exp/step3_spatial16_w8_d4.yaml \
    --env LIBERO --port 7999 \
    --stage1_device cuda:0 --stage2_device cuda:1 --stage3_device cuda:1 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"

# Server C：max_pool，物理 gpu 4+5
CUDA_VISIBLE_DEVICES=4,5 \
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config configs/cache_runs/deviate_exp/step3_max_pool_w3_d5.yaml \
    --env LIBERO --port 8000 \
    --stage1_device cuda:0 --stage2_device cuda:1 --stage3_device cuda:1 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

健康检查：

```bash
curl http://127.0.0.1:7998/healthz
curl http://127.0.0.1:7999/healthz
curl http://127.0.0.1:8000/healthz
```

### 附录·Client 命令（127.0.0.1 同机直连版）

```bash
# Client 1：clip → 127.0.0.1:7998 on gpu6
CUDA_VISIBLE_DEVICES=6 \
uv run python -m exp.trajectory_deviation.run_step3_per_cycle_policy \
    --cfg clip_w7_d4 \
    --host 127.0.0.1 --port 7998 \
    --yaml configs/cache_runs/deviate_exp/step3_clip_w7_d4.yaml \
    --deviate-score-json data/deviation_experiment/deviate_scores/deviate_score_clip_w7_d4.json \
    --init-states-dir data/deviation_experiment/inits \
    --out-dir data/deviation_experiment/step3/clip_w7_d4 \
    --task-suite-name libero_spatial \
    --tau-grid 3,5,7,10 --n-grid 1,2,3,5,10 \
    --num-workers 5 --resume

# Client 2：spatial16 → 127.0.0.1:7999 on gpu6
CUDA_VISIBLE_DEVICES=6 \
uv run python -m exp.trajectory_deviation.run_step3_per_cycle_policy \
    --cfg spatial16_w8_d4 \
    --host 127.0.0.1 --port 7999 \
    --yaml configs/cache_runs/deviate_exp/step3_spatial16_w8_d4.yaml \
    --deviate-score-json data/deviation_experiment/deviate_scores/deviate_score_spatial16_w8_d4.json \
    --init-states-dir data/deviation_experiment/inits \
    --out-dir data/deviation_experiment/step3/spatial16_w8_d4 \
    --task-suite-name libero_spatial \
    --tau-grid 3,5,7,10 --n-grid 1,2,3,5,10 \
    --num-workers 5 --resume

# Client 3：max_pool → 127.0.0.1:8000 on gpu7
CUDA_VISIBLE_DEVICES=7 \
uv run python -m exp.trajectory_deviation.run_step3_per_cycle_policy \
    --cfg max_pool_w3_d5 \
    --host 127.0.0.1 --port 8000 \
    --yaml configs/cache_runs/deviate_exp/step3_max_pool_w3_d5.yaml \
    --deviate-score-json data/deviation_experiment/deviate_scores/deviate_score_max_pool_w3_d5.json \
    --init-states-dir data/deviation_experiment/inits \
    --out-dir data/deviation_experiment/step3/max_pool_w3_d5 \
    --task-suite-name libero_spatial \
    --tau-grid 3,5,7,10 --n-grid 1,2,3,5,10 \
    --num-workers 5 --resume
```

### 附录·已知问题（待排查）

- 2026-04-16 timan107 实测：Client 3（max_pool）`[parallel retry 1/2] 3000 failed units` 全跪，时间窗约 5 分钟（`02:46:48 → 02:51:44`）。需查 `data/deviation_experiment/step3/max_pool_w3_d5/run_state.json` 里任一 failed unit 的 `result.error` 才能定性。
- 同机 5 worker MuJoCo EGL + server CUDA 上下文是否互相挤压未验证。
- Server C 当时是否真的起来、`curl /healthz` 是否过——日志缺失。
