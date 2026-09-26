# π0.5 × MetaWorld MT50 接入 warm reset 一等公民框架 — 实现记录

> 状态：`Verified — 已提交`（2026-09-25 16:20 CDT 起；Authority: Execution；级别 L2；两轮真实冒烟准入通过、无库自产手算逐位一致、CPU 全量 Verify 无新增失败；owner 2026-09-25 19:57 CDT 指示提交推送；正式运行 6 臂 × 50 任务 × 20 集已于 19:4x 从冻结快照 `/data/openpi_mw` 开跑）。
> **owner 2026-09-25 16:20 CDT 当次豁免 plan / G1**：直接实现，本文件记录实现与验证；G2 / 提交待 owner 安排（本轮不 `git add` / commit / push）。
> 上位：[`step_diag_metaworld_selfstart_plan.log.md`](step_diag_metaworld_selfstart_plan.log.md)（已 Superseded；其 §1–§2 的冒烟结论与仿真约定为本实现的权威来源）、[`warm_reset_migration_study.log.md`](warm_reset_migration_study.log.md) §9、框架指南 [`docs/cache/warm_reset_experiments.md`](../docs/cache/warm_reset_experiments.md)。
> 分工：框架侧（`src/openpi/cache/**`、`src/openpi/serving/**`、`scripts/serve_policy.py`、`exp/warm_reset/**`）由另一 agent 同期扩展（按 bundle 的 MISS 步数 `miss:`、无库自产 `warm_reset.trigger: always`、入口 env 注册钩子），本线不改其文件。

## 1. 目标与臂

新 benchmark：π0.5（RLinf SFT checkpoint `/data/ckpt/pi05_metaworld_rlinf`）在 MetaWorld MT50（50 任务）上，跑在 warm reset 框架（`python -m exp.warm_reset.run`）上。**不建 cache 库、不收集 cache。**

| 臂 | 起点 | N | 每决策 NFE |
|---|---|---|---|
| `full` | 噪声 | 10 | 10 |
| `plain_k2` | 噪声 | 2 | 2 |
| `plain_k1` | 噪声 | 1 | 1 |
| `selfwarmreset_t0.2` | 自产 K=10 在 T=0.2 的快照 | 2 | 10+2 |
| `selfresetfinal_t0.2` | 自产 K=10 最终动作 | 2 | 10+2 |
| `selfmidfinal_t0.2` | 自产 K=10 最终动作，入口 0.9 | 2 | 10+2 |

规模：50 任务 × 20 集（idx 0..19）× 6 臂 = 6,000 集。配对身份 = (任务, idx, bench seed 7)。

## 2. 实现（本线文件）

| 文件 | 内容 |
|---|---|
| `src/openpi/policies/metaworld_policy.py`（新） | `MetaworldInputs`：只 `base_0_rgb` 为真图，两腕位零图、mask False，state 转 float32 透传，prompt 透传；`MetaworldOutputs`：`actions[:, :4]`；`make_metaworld_example` |
| `src/openpi/training/config.py` | 新增 `_MetaworldGroup` 与**仅推理**条目 `pi05_metaworld`：`Pi0Config(pi05=True, action_horizon=5, discrete_state_input=False)`（max_token_len 取 pi05 默认 200）、`SimpleDataConfig(assets=AssetsConfig(asset_id="metaworld_mt50"))`；quantile 由 `create_base_config` 按 PI05 打开（照 `pi05_robocasa` 写法，不设 repo_id / 训练参数） |
| `exp/metaworld/tasks.py`（新） | 50 任务表（RLinf prompt 原文 + 难度组 28/11/6/5；行序 = metaworld 3.0.0 `MT50().train_classes`，位置即 `task_id`）；常量 `BENCH_SEED=7`、`SETTLE_STEPS=15`、`MAX_POLICY_STEPS=160`、`REPLAN_STEPS=5`、相机 `corner2`/id 2/`(0.75, 0.075, 0.7)`、480×480、`EPISODES_PER_TASK=20`；`export_tasks` 产出入口 `prepare --tasks` 的清单 `[{task_id, name, init_indices}]`（`name` = env 名，`init_indices` = `MT1.train_tasks` 下标，限 0..49） |
| `exp/metaworld/env.py`（新） | 仿真约定：`make_env`（`MT1(task, seed)` + `set_task(train_tasks[idx])`，每集新建 env；`MT1` 对象按 (任务, seed) 进程内缓存——它只是 seed 抽样出的任务数据，构造约 1.2 s / 集，缓存后 env 构造约 0.07 s，初始观测 / 首帧与不缓存逐位相同；`corner2` 断言为相机 id 2 后改 `cam_pos`）、`render_image`（`[::-1, ::-1]` + contiguous）、`reset_and_settle`（15 步零动作）、`run_episode`（每次推理开环执行 5 步、≤160 策略步、任一步 `info["success"]` 即成功并结束；动作形状不符直接报错）；metaworld 延迟导入 |
| `exp/metaworld/episode_runner.py`（新） | conductor `MetaworldEpisodeRunner`（照 `LiberoEpisodeRunner`）：同 (server, bundle) 复用连接、异常即丢连接；`episode_start(experiment=metaworld_mt50, task=<env 名>, episode_id=episode_idx, extra={task_id, orig_init_state_idx, task_uid, attempt, seed})`；每次推理一行 per_step（`step_idx` = 发出推理时的策略步 0,5,10,…；`hit_type`/`start_t` 等 `__hit_meta__` 标量白名单 + `warm_reset_{n_steps,decision_nfe,continuation_nfe,self_direct_nfe,self_seed}`；无 hit meta 时 `hit_type=None` 照样记行）；每集一行 `_kind: episode_summary`（步数、决策数、结束原因、推理耗时） |
| `exp/metaworld/worker_entry.py`（新） | `python -m exp.metaworld.worker_entry`，在 `~/metaworld_sim` venv 中运行 |
| `exp/metaworld/warm_reset_env.py`（新） | 入口所需的全部 MetaWorld 事实：`ENV = EnvSpec("pi05_metaworld", "pi05", "metaworld_mt50", 5, 32, 4, 10, "pi05_v1", (0.2,), (1, 2), "weilandserver", "weilandserver")`、`ARMS`（六臂）、`EPISODES_PER_TASK`、`export_tasks`、`validate_rollout`（seed 必须 7、replan 必须 5、无 init 池）、`spawn_worker`（`WorkerAgent` 的 spawn 函数：用模拟器 venv 解释器、清 `VIRTUAL_ENV/PYTHONPATH/PYTHONHOME`、`PYTHONPATH=<repo>:<repo>/src`、`MUJOCO_GL=egl`、`CUDA_VISIBLE_DEVICES` 与 `MUJOCO_EGL_DEVICE_ID` 都取 worker 的 GPU、`start_new_session=True`） |
| `tests/exp/metaworld/`（新） | 见 §4 |

核对（2026-09-25 16:25 CDT）：任务表与参考实现（`/home/weiland/.claude/jobs/3f6cef91/tmp/mw/mw_client.py`）50 条 prompt / 难度逐条一致；`make_env` + 沉降后的观测与首帧图像与参考实现构造逐位相同；同 (任务, idx, seed) 两个进程初始观测与首帧 sha256 相同。`pi05_metaworld` 的输入 / 输出变换链（含 checkpoint 真 norm stats 的 quantile Normalize / Unnormalize 与 model transforms）与参考 server 的 `MWData` 链在同一观测 / 动作上逐位相同。

## 3. 入口注册

框架侧钩子（`exp/warm_reset/envs.py`，见 [`warm_reset_framework_completion.log.md`](warm_reset_framework_completion.log.md)）约定：`exp/<包>/warm_reset_env.py` 在导入时 `register_env(EntryEnv(..., adapter=<BenchmarkAdapter 子类>))`，入口按约定自动发现。`exp/metaworld/warm_reset_env.py` 改为：

- `MetaworldAdapter`（无字段的 frozen dataclass，两实例相等 ⇒ 重复导入 / reload 重注册同一环境不冲突）：
  - `validate_rollout`：seed 必须 7（MT1 bench seed）、replan 必须 5、不接受 init 池；
  - `export_tasks`：`--task-names`（逗号）优先，否则 `--task-ids`（`all` = 50 个，表序）；`init_indices` 限 0..49；
  - `experiment` = `metaworld_mt50`（固定 ⇒ 同身份的自产噪声跨运行共同）；
  - `episode_extra` = `{task_name}`，并在 plan 时核对清单 `task_id` 等于 MT50 规范 id（worker 端再交叉核对一次）；
  - `identity` = `{seed: rollout.seed}`（受信身份含 bench seed，worker 送错 seed 即 `identity_mismatch` 拒收，测试覆盖）；
  - `pairing` = `{init_idx, env_seed}`（入口另加任务名 ⇒ 配对身份 (任务, idx, seed)）；
  - `worker_agent`：`WorkerAgent(spawn_fn=partial(spawn_worker, worker_python=...))`，`agent --worker-python` 覆盖默认 `~/metaworld_sim/bin/python`；拒 `--conda-env` / `--pinned-objects`。
- `ENV = register_env(EntryEnv("pi05_metaworld", policy="pi05", benchmark="metaworld_mt50", action_horizon=5, k_full=10, schedule_id="pi05_v1", default_arms=ARMS))`。原先的 step_diag `EnvSpec` 行与模块级 `export_tasks` / `validate_rollout` 并入适配器后删除。
- 框架测试 `tests/exp/warm_reset/test_entry_arms.py::test_repo_plugins_import` 导入本插件无错误；本插件导入不拉 metaworld / mujoco / torch / jax（子进程测试）。

臂表达：`--arms full,plain_k2,plain_k1`（每 bundle `miss` 块）+ `--arms selfwarmreset_t0.2,selfresetfinal_t0.2,selfmidfinal_t0.2 --self-trigger always`（无库自产），全部无库，不需 `--base-yaml`；`--arms all` 即六臂。生成的 yaml 核对：`full`/`plain_k2`/`plain_k1` 的 `miss.num_steps` = 10/2/1；三个自产臂 `trigger: always`、`start_t: 0.2`、`num_steps: remaining`（=2），`selfwarmreset` 快照起点 entry 1.0、`selfresetfinal` 最终动作 entry 1.0、`selfmidfinal` 最终动作 entry 0.9。

## 4. 测试

`tests/exp/metaworld/`（CPU，41 项，均通过；另 `tests/exp/warm_reset` 全部通过，其 `test_repo_plugins_import` 导入本插件无错误）：
- `test_policy_config.py`：配置（pi05、H=5、action_dim 32、无离散 state、max_token_len 200、不导入 metaworld）、asset_id `metaworld_mt50` 与 quantile、变换链类型、输入键 / mask / 零腕位、CHW float 图解析、4 维切片、合成 quantile 统计下整条输入链（224 图、200 token、state 归一化后补零到 32）。
- `test_env.py`：任务表（50、难度 28/11/6/5、push-back 原文、首尾任务）、清单导出与入口 `validate_tasks` 兼容、越界 / 重复 / 未知任务拒绝；假环境上相机断言、180° 翻转 + contiguous、请求格式、15 步零动作沉降、160 步上限与 0,5,…,155 决策间隔、成功即停（块中途）、terminated 不算成功、错误动作形状 / replan 拒绝；真模拟器（`~/metaworld_sim` 子进程两次）同身份初始观测相同、同进程内重复同身份（复用缓存 MT1）相同、不同 idx / 任务不同（反向对照）。
- `test_runner.py`：身份 / episode_start 参数、per_step 行（步距、hit meta 拷贝含 `miss_nfe` 与 `warm_reset_*`、NaN→None、numpy 标量→Python、可 `allow_nan=False` 序列化）、无 hit meta 仍计决策、连接复用 / bundle 切换 / 换 server 重连、失败丢连接并 episode_end、task_name 交叉校验、spawn 命令与环境、worker_entry 不导入模拟器。
- `test_entry.py`：经插件钩子注册（`get_env` 即 `ENV`、无插件导入错误、同环境重注册被接受）、插件导入不拉 metaworld / mujoco / torch / jax（子进程）、六臂类型（3 miss + 3 self_only）与步数（10/2/1，自产 N=2）、适配器（按名 / 按 id / all 导出、rollout 校验、experiment、extra 与 task_id 规范性、identity、pairing）、CLI `tasks` + `prepare --arms all --self-trigger always` 无 base yaml（yaml 内容、EpisodeTask experiment / extra）、agent 使用 MetaWorld spawn 与 `--worker-python`、拒 `--conda-env`；**端到端**：六臂计划的每个派发 episode 经 CPU 桩服务栈（evidence wrapper + interceptor）由真实 `MetaworldEpisodeRunner` + `run_episode`（假模拟器）驱动、按 driver 规则盖章后 `admit` 全部准入且实测 NFE = 决策数 × 10/2/1/12；worker 送错 bench seed ⇒ 全部 `identity_mismatch` 拒收。

**§6 CPU 全量 Verify**（2026-09-25 18:5x CDT，标准命令 `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP/MKL/OPENBLAS_NUM_THREADS=2 uv run --no-sync --with pytest-xdist --with pytest==9.0.2 pytest -n 24 --dist loadfile -q`，日志 `/tmp/wr_mw/verify_final.log`）→ **7425 passed、95 skipped、21 failed、49 errors**（235 s）。21 个失败恰为既有清单：`test_ws2_evidence_runner` ×2、`test_groot_concurrent_serving::test_frozen_commands_pass_the_new_guards` ×2、`test_prebuilt_matrix_backend` ×2、`tests/review_tests` 15 个（只看 pytest 汇总里的 id，未读其内容）；49 个错误来自既有的 3 个收集错误文件（`test_bench_groot_stages.py`、`test_review_cache_prune_g2.py`、`online_rit_g2/test_round2_boundaries.py`）。无新增失败。

## 5. 冒烟与运行

### 5.1 怎么跑（weilandserver 本机：server + driver + agent / worker）

```bash
cd ~/projects/openpi
R=/data/wr_mw/runs; EV=/data/wr_mw/evidence   # 运行目录 / 服务端证据根（绝对路径）
# 1. server（π0.5 concurrent；无库，不需 --cache-config，臂 yaml 由 driver 热加载）
tmux new -d -s wr_mw_srv23192 "CUDA_VISIBLE_DEVICES=0 uv run --no-sync scripts/serve_policy.py --concurrent --port 23192 \
  policy:checkpoint --policy.config pi05_metaworld --policy.dir /data/ckpt/pi05_metaworld_rlinf > /data/wr_mw/logs/server_23192.log 2>&1"
# 2. 任务清单（纯 Python，主 venv 即可；正式 = 全部 50 任务 × idx 0..19）
uv run --no-sync python -m exp.warm_reset.run tasks --env pi05_metaworld --episodes 20 --out $R/formal_tasks.json
# 3. prepare（六臂全无库：--arms all + --self-trigger always，无 --base-yaml；seed 必须 7、replan 必须 5）
uv run --no-sync python -m exp.warm_reset.run prepare --env pi05_metaworld --tasks $R/formal_tasks.json \
  --arms all --self-trigger always --servers 127.0.0.1:23192 --server-evidence-root $EV \
  --namespace wr_mw_formal_01 --out $R/formal_01 --replan-steps 5 --seed 7
# 4. driver 与 agent（agent 在主 venv 跑，worker 用 ~/metaworld_sim 解释器，--worker-python 可覆盖）
tmux new -d -s wr_mw_drv "uv run --no-sync python -m exp.warm_reset.run run --run-dir $R/formal_01 --bind-host 127.0.0.1 --port 23193 --concurrency 6"
tmux new -d -s wr_mw_agent "uv run --no-sync python -m exp.warm_reset.run agent --run-dir $R/formal_01 --server 127.0.0.1:23192 \
  --driver-host 127.0.0.1 --driver-port 23193 --gpus 0 --workers-per-gpu 6 --prefix wr_mw_wls"
# 5. driver 退出后停 agent（按 PID 发 TERM，会连带 worker 进程组），再 admit / 分析
uv run --no-sync python -m exp.warm_reset.run admit --run-dir $R/formal_01
uv run --no-sync python -m exp.warm_reset.analysis --run-dir $R/formal_01 --out-json ... --out-md ...
```

注意：driver 结束时下发 shutdown，worker 正常退出后 agent 会把它们当「死亡」重启（重启的 worker 连不上已退出的 driver，只是重试日志）；这是标准 agent 的既有行为（指南「driver 完成后用 Ctrl-C 停止 agent」），不是崩溃。

### 5.2 冒烟（2026-09-25，wls 4090；server 起前空闲 11.95 GB，server 占约 7.6 GB；6 worker 本机，`--concurrency 6`）

两次运行 `admit` 均 **exit 0、`ok: true`、全部 episode 准入、无 global problem**，全部 attempt = 1、无 error。每决策实测 NFE（服务端证据 `measured_total_nfe` / worker 决策数）与 worker 行一致：

| 臂 | hit type | NFE / 决策 | smoke_01 成功（reach + push-back，idx 0,1） | smoke_02 成功（button-press + drawer-close + door-open，idx 0,1） |
|---|---|---|---|---|
| `full` | MISS（`miss_nfe` 10） | 10.0 | 1/4 | 6/6 |
| `plain_k2` | MISS（2） | 2.0 | 1/4 | 6/6 |
| `plain_k1` | MISS（1） | 1.0 | 1/4 | 5/6（button-press idx 1 到 160 步上限） |
| `selfwarmreset_t0.2` | SELF_ONLY，start_t 0.2 | 12.0 | 0/4 | 6/6 |
| `selfresetfinal_t0.2` | SELF_ONLY，start_t 0.2 | 12.0 | 0/4 | 6/6 |
| `selfmidfinal_t0.2` | SELF_ONLY，start_t 0.2 | 12.0 | 0/4 | 4/6（button-press 两集都到上限） |

- 运行目录：`/data/wr_mw/runs/smoke_01`（24 集，17:21:51–17:24:47 CDT）、`/data/wr_mw/runs/smoke_02`（36 集，18:38:10–18:40:55 CDT）；服务端证据 `/data/wr_mw/evidence/<token>`；日志 `/data/wr_mw/logs/`。
- 吞吐：smoke_01 约 8.2 集/分钟（多为 160 步满长的失败集），smoke_02 约 13 集/分钟。每决策推理耗时（worker 端、含合批等待，6 连接并发）：full 1.18 s、plain_k2 0.72 s、plain_k1 0.60 s、自产臂 1.73–2.02 s。单集平均耗时 11–44 s。
- smoke_01 的两个任务偏难（参考冒烟里 push-back 属 hard 组），4 集样本不足以判断；smoke_02 换成参考冒烟中 full 2/2 成功的三个 easy 任务，自产臂能成功，只有 `selfmidfinal` 在 button-press 两集失败（见 §5.3 手算核对）。

### 5.3 无库自产路径手算核对（真 checkpoint，GPU）

目的：smoke_01 三个自产臂 0/4、smoke_02 `selfmidfinal` 在 button-press 两集失败（full 成功），排除自产路径实现错误。做法（一次性脚本 `/tmp/wr_mw/self_arm_check.py`，不入库；2026-09-25 18:4x CDT，server 已停，GPU 空闲 11.96 GB）：真 checkpoint + `pi05_metaworld` 配置建 policy；取一条真 MetaWorld 观测（button-press idx 0 沉降后，`/tmp/wr_mw/obs_button_press_0.npz`）；对 smoke_02 冻结的六个臂 yaml（证据目录改到 /tmp）逐一建与 server 相同的 `InferenceInterceptor` + evidence wrapper 栈、`episode_start` 后 `infer` 一次；再用同一 stage-2 手算：`private_noise(self_seed, (H=5, D=32))` 起 K=10 全程（`run_stage3(return_intermediates, save_timesteps=(0.2,))`），取 T=0.2 快照（`selfwarmreset`）或最终动作（`selfresetfinal` / `selfmidfinal`），按入口 t0 = 1.0 / 1.0 / 0.9 走 2 步显式 Euler（dt = −t0/2），经同一输出变换（quantile Unnormalize + `[:, :4]`）。

结果（`/tmp/wr_mw/self_check/report.json`）：三个自产臂 served 动作与手算 **max|Δ| = 0.0（逐位相等）**；meta 的网格与计划一致（`t = [1.0, 0.5]`、`dt = −0.5`；`selfmidfinal` 为 `[0.9, 0.45]`、`dt = −0.45`；`n_steps = 2`）；起点形状取自模型配置 `[1, 5, 32]` float32；三臂 hit type `SELF_ONLY`，`full` / `plain_k2` / `plain_k1` 为 MISS 且 `miss_nfe` = 10 / 2 / 1，输出均为 `[5, 4]`。自产臂与其自身 K=10 动作的差（反归一化后 max|Δ| ≈ 1.24–1.27）主要落在超出 [−1, 1]、会被环境裁剪的维度上。**结论：无库自产路径按臂定义执行；smoke 中的失败是样本太少时的结局差异，不是实现错误。**是否有真实臂效应要看正式规模（每臂 1,000 集）。

## 6. 未决 / 风险

- 正式规模（50 任务 × 20 集 × 6 臂 = 6,000 集）未启动，待 owner 安排；按冒烟吞吐（6 worker、1 server）约 8–13 集/分钟 ⇒ 约 8–12 小时，可加 worker / 第二个 server（显存允许时）缩短。
- 冒烟结局样本很小（每臂 4–6 集），不作结论；§5.3 已排除自产路径实现错误。
- wls 显存：π0.5 server 约 7.6 GB；本线冒烟时空闲 11.95 GB（满足「需求 + 4 GB」）。冒烟后 server / agent / worker 已按 PID 停止。
- 本线文件未提交（owner 未授权 commit）；`logs/README.md` 索引未改（按分工由上级会话同步）。
