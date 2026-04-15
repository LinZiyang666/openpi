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
Step 3   run_spawn_experiment       → spawn_state_{cfg}.json + spawn_aggregate.csv
Step 4   analyze_deviation_results  → figures/*.png
```

## 网络拓扑（与 CP1 实验一致）

```
┌─────────────────────────┐        frp 隧道/内网         ┌────────────────────────┐
│  GPU 服务器 (无公网IP)    │ ◄──────────────────────────│  LIBERO 评估端          │
│  serve_policy.py         │   155.98.36.13:9000         │  exp/trajectory_       │
│  监听 localhost:8000     │   → localhost:8000          │  deviation/*.py         │
│  --concurrent --collect  │                              │  examples/libero/main   │
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
| Step 3 Spawn | episode × (s,n,k)（`--num-workers`） | 不同 cfg 串行 | SpawnRunner 与 BaselineRunner 同 cfg 内串行 |
| Step 4 分析 | — | 单进程 matplotlib（`Agg`） | 无需 X server |

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
    --collect \
    --collect-dir data/deviation_experiment/collected \
    --collect-images \
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
| `--collect` + `--collect-dir` | 把每个 episode 的 intermediates（含 `clean_action`）写 HDF5 | **必须**。Step 3 的 prefill 只认 server 端的 `clean_action`；不开 collect 会在 Step 3 直接 `FileNotFoundError` |
| `--collect-images` | 额外保存原始 image（CLIP KeyBuilder 需要） | 只要当前或后续会切到 `key_builder.type: clip` 的 bundle 就必须开；本实验包含 `clip_w7_d4`，所以建议 server 全程开着 |
| `--env LIBERO` | 选择环境分支 | 本实验固定 LIBERO，必须 |
| `--port 8000` | 监听端口 | 与评估端 driver 的 `--port` 对齐（若走 frp 隧道，评估端填映射后的外网端口，例如 `9000`） |
| `policy:checkpoint --policy.config pi05_libero --policy.dir <本地路径>` | 使用本地 checkpoint 避免 GCS 下载 | 强烈建议填本地路径 |

验证 server 就绪（在评估端）：

```bash
curl http://<server-host>:<server-port>/healthz
# 输出: OK
```

> Step 0 的启动命令和 `cp1_cache.md` Step 4 基本相同，只多了 `--collect --collect-dir --collect-images`——如果你之前跑过 CP1，直接加这三个 flag 重启即可。

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
- 产物：`gt/task_{id}/episode_{subset_idx}.h5`，server 端镜像到 `collected/libero_spatial/task_{id}/episode_{subset_idx}.h5`。

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

## Step 3：Spawn 纠偏实验（每个 cfg 内部并行，cfg 之间串行）

如果 Step 2 使用了 `--config-fail-results`，Step 3 不需要额外参数：`run_spawn_experiment` 只会根据 `deviate_score_{cfg}.json` 里存在的 episode 生成 spawn units。若复用旧 `--out-dir` + `--resume`，runner/aggregate 也会按当前 score 文件过滤旧 state 中的额外 episode；不过正式实验仍建议用干净的 spawn 输出目录。

```bash
uv run python -m exp.trajectory_deviation.run_spawn_experiment \
    --gt-dir data/deviation_experiment/gt \
    --collected-dir data/deviation_experiment/collected \
    --task-suite-name libero_spatial \
    --deviate-score-dir data/deviation_experiment/deviate_scores \
    --out-dir data/deviation_experiment/spawn \
    --configs clip_w7_d4 spatial16_w8_d4 max_pool_w3_d5 \
    --n-grid 1 3 5 10 20 \
    --k-grid 1 3 5 \
    --max-spawn-env-steps 300 \
    --num-workers 4 \
    --baselines random equidistant \
    --random-seed 0 \
    --host <server-host> --port <server-port> \
    --resume
```

参数详细说明与调节：

| Flag | 默认 | 含义 | 调节建议 |
|------|------|-----|---------|
| `--gt-dir` | — | 客户端 GT HDF5 根（Step 1b 产物） | 必填 |
| `--collected-dir` | — | **server 端** `--collect-dir`，脚本只读 `clean_action` 做 prefill | 必填；评估端与 server 不共盘时，先把 server 的 `collected/` 同步过来，或把 driver 挪到 server 上跑 |
| `--task-suite-name` | — | 与 Step 1a/1b 完全一致 | 必填 |
| `--deviate-score-dir` | — | Step 2 输出根，driver 会读 `deviate_score_{cfg}.json` | 必填 |
| `--out-dir` | — | spawn state + aggregate CSV 输出根 | 一般 `data/deviation_experiment/spawn` |
| `--configs` | — | 要跑的 cfg 列表（与 Step 2 保持一致） | 通常三者全跑 |
| `--D` | 从 cfg 名 `_dN` 自动解析 | 轨迹 depth，决定 prefill 读 server HDF5 的历史长度 | 除非你知道 artifact 的 depth 与 cfg 名不一致，否则别填 |
| `--n-grid` | `1 3 5 10 20` | teleport 前沿 GT 推进的 cycle 数 | **核心旋钮之一**：n 小 = 贴近 crisis 才 teleport（最严格恢复测试）；n 大 = 提前让 GT 先带一段路再把控制权交给 cache。想简化改成 `1 5 20` 即可得到粗粒度曲线 |
| `--k-grid` | `1 3 5` | 每 episode 取 deviate_score top-k 个点（每个独立 spawn） | **另一核心旋钮**：k=1 只看最严重一点；k=5 覆盖多点。想做 ablation 可以传 `1 2 3 5 7` |
| `--max-spawn-env-steps` | 300 | teleport 后单 unit 最多跑多少 env.step | libero 一个 episode ≤220 步，300 留 ~30% margin；资源紧张可以降到 220，但碰到边界的 unit 会判负需要 |
| `--num-workers` | 4 | 同 cfg 内并行度 | 与 Step 2 相同的经验值 |
| `--baselines` | `[]` | SpawnRunner 之后额外跑的对照策略，可选 `random` / `equidistant` | 想论证 "top-k deviate 确实有用"，建议至少加 `random`；完整实验三者都开 |
| `--random-seed` | 0 | `BaselineRunner` 的随机 k 抽样 seed | 固定；想做 seed 敏感性分析可以跑 0/1/2 三次 |
| `--skip-config-switch` | off | 不调 `load_cache_config` | 多 server 分片才用 |
| `--config-yaml-dir` | `configs/cache_runs/deviate_exp` | YAML 根 | 一般不动 |
| `--resume` | off | 断点续跑 | 几乎总开 |

`(n, k)` 组合数等于 `len(n-grid) × len(k-grid)`：默认 5×3=15 种；乘以 episodes × cfgs，是 Step 3 最费时的部分。调节建议：

- **先探路**：`--n-grid 1 5 --k-grid 1 3 --configs clip_w7_d4 --baselines ""`，4 组合 × 单 cfg，先验证 teleport / prefill 跑得通。
- **正式跑**：默认的 5×3 + 3 configs + `random equidistant` 两 baseline，才能得到 plan §12 想要的四张图。
- **显存不够？** 先降 `--num-workers`，再考虑缩 `--n-grid` / `--k-grid`。

⚠️ 同样，不要对同一 server 并行起多个 `run_spawn_experiment`——和 Step 2 一样会抢 bundle。

---

## Step 4：分析与绘图（单进程，秒级）

```bash
uv run python -m exp.trajectory_deviation.analyze_deviation_results \
    --deviate-score-dir data/deviation_experiment/deviate_scores \
    --spawn-csv data/deviation_experiment/spawn/spawn_aggregate.csv \
    --out-dir data/deviation_experiment/figures \
    --configs clip_w7_d4 spatial16_w8_d4 max_pool_w3_d5 \
    --n-threshold 3
```

参数说明：

| Flag | 默认 | 含义 | 调节建议 |
|------|------|-----|---------|
| `--deviate-score-dir` | — | Step 2 输出根 | 必填 |
| `--spawn-csv` | — | Step 3 聚合 CSV | 必填 |
| `--out-dir` | — | 图片输出目录 | `data/deviation_experiment/figures` |
| `--configs` | — | 要画的 cfg，必须出现在 CSV / JSON 里 | 通常三者全画 |
| `--n-threshold` | 3 | "close-in spawn" 的 cutoff，定义 "真 failure cycle" 的标准——只统计 `n ≤ threshold` 下失败的 spawn | **这是 ROC 图的灵敏度旋钮**：threshold 越小越严格（只有近距离 teleport 还失败才算"真 failure"），ROC 越稀疏。默认 3；想要更宽松的 coverage 曲线可以调到 5 |

每个 cfg 输出四张图：deviate_score 直方图、top-k 覆盖率（ROC 样式）、`(n, k_idx)` 成功率热力图、最佳 cell 下 `top-k vs random vs equidistant` 对比。

---

## 参数调节速查表（总结）

按"重要性 × 常改"排序，方便你在实验设计阶段决定哪些要 sweep：

### 核心"实验旋钮"

| 参数 | 所在 Step | 作用 | 典型取值 |
|------|----------|------|---------|
| `--configs` | Step 2/3/4 | 要对比的 cache 配置 | `clip_w7_d4 spatial16_w8_d4 max_pool_w3_d5`（全部） |
| `--M` (Phase 1) | Step 2 | 背景 L2 噪声估计的采样数 | 10 / 20 / 30 |
| `--n-grid` | Step 3 | teleport 前推进几个 cycle | `1 3 5 10 20`（默认）/ `1 5 20`（粗粒度） |
| `--k-grid` | Step 3 | 每 episode 取 top-k 个 deviate 点 | `1 3 5`（默认）/ `1 2 3 5 7`（ablation） |
| `--baselines` | Step 3 | 对照策略 | `random equidistant`（完整实验必开） |
| `--n-threshold` | Step 4 | ROC 图 "真 failure" cutoff | 3（默认）/ 5（更宽松） |

### 并发 / 性能

| 参数 | 所在 Step | 作用 | 调节建议 |
|------|----------|------|---------|
| `--num-workers` | Step 1a/2/3 | 同 cfg 内 worker 数 | 受 server GPU 限制；4 通常安全，8 看卡 |
| `--max-spawn-env-steps` | Step 3 | 单 spawn unit 的 env.step 预算 | 300（默认）；libero 上限 220 + 30% margin |

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
├── collected/libero_spatial/task_{id}/episode_{subset_idx}.h5  # server 侧 clean_action
├── deviate_scores/
│   ├── bg_{cfg}.jsonl
│   ├── cache_{cfg}.jsonl
│   ├── phase1_state_{cfg}.json
│   ├── phase2_state_{cfg}.json
│   └── deviate_score_{cfg}.json
├── spawn/
│   ├── spawn_state_{cfg}.json
│   ├── spawn_state_random_{cfg}.json
│   ├── spawn_state_equidistant_{cfg}.json
│   └── spawn_aggregate.csv
└── figures/{hist,coverage,heatmap,strategy}_{cfg}.png
```

---

## 故障排查

| 症状 | 原因 | 处理 |
|------|------|------|
| Step 3 `FileNotFoundError: collected/.../clean_action` | server 启动漏了 `--collect` 或 `--collect-images` | 按 Step 0 完整 flag 重启；失败 unit 带 `--resume` 重跑 |
| Step 2 deviate_score 全部 ≈ 1.0 | Phase 1 / Phase 2 读到了同一个 bundle | 多进程抢了 `send_load_cache_config`；停掉并行 driver，单进程跑；或每 cfg 独立 server + `--skip-config-switch` |
| Step 1b 大量 `inference_failed=True` | server 挂了 / 不在指定的 AlwaysSkip GT bundle | 检查 server 进程；若你加了 `--skip-config-switch`，确认已手动 `load_cache_config` 到 `inference_max_pool_w3_d5.yaml` 或等价 AlwaysSkip bundle |
| Step 4 coverage 曲线为空 | 该 cfg 在 `n ≤ n_threshold` 下没有失败 spawn | 调大 `--n-threshold`，或检查 Step 3 是否真的跑完整个 `n-grid` |
| Step 1a `main.py` 抱怨 `VIRTUAL_ENV` | uv 环境变量泄漏到 conda 子进程 | `_build_subprocess_cmd` 应已清理；若复现请检查自定义 `--conda-env` wrapper |
| `load_cache_config` 报错 | server 未加 `--concurrent`；或 YAML 的 `preload_path` 在 server 端不存在 | 重启 server 加 `--concurrent`；核对 artifact 在 GPU 服务器上的路径 |

---

## 交叉引用

- Server 启动的更多上下文、artifact 构建、CP1 实验：[cp1_cache.md](cp1_cache.md)
- Cache 系统组件、YAML 字段含义：[../cache/tutorial.md](../cache/tutorial.md)
- 远程推理的网络拓扑：[../deployment/libero.md](../deployment/libero.md)
- 原始实验设计与评审：[../../logs/trajectory_deviation_experiment_plan.log.md](../../logs/trajectory_deviation_experiment_plan.log.md)、[../../logs/trajectory_deviation_corrective_experiment.log.md](../../logs/trajectory_deviation_corrective_experiment.log.md)
