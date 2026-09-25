# Warm reset 多臂实验入口

`python -m exp.warm_reset.run` 复用正式 concurrent server、conductor 和标准
LIBERO / RoboCasa worker。它提供任务清单导出、臂 YAML 生成、多臂调度及证据准入，
不再依赖 step_diag 专用 server 或 monkey patch。

本页命令中的大写变量由操作者设置为实际路径/地址；在仓库根目录运行。
`WR_RUN` 是一个尚不存在的运行目录，之后把整个目录复制到 worker 主机的同一路径
（或使用共享目录）。`WR_BASE` 是对应环境已经能跑通的 warm-start YAML：
CP1 `always_search`、`write_policy: {type: never}`、in-memory 冻结库，
`preload_path` 必须是各服务端可访问的绝对路径。库必须包含请求的任务、快照和匹配的
schedule；加载时仍会执行生产库校验。生成器保留检索配置，将 judge 固定为该臂的
`always_warm_start`，并冻结已展开的环境变量。不同机器的 library 路径需要一致。

## LIBERO：生成并运行

先在安装了 LIBERO 的模拟器环境导出真实任务语言。这一步不会加载模型或运行 episode。
以下例子用 π0.5 / libero_10、任务 0 和 1，每任务 2 个原始 init state：

```bash
conda run -n "$WR_LIBERO_ENV" python -m exp.warm_reset.run tasks \
  --env pi05_libero_10 --task-ids 0,1 --episodes 2 --out "$WR_TASKS"

uv run --no-sync python -m exp.warm_reset.run prepare \
  --env pi05_libero_10 --base-yaml "$WR_BASE" --tasks "$WR_TASKS" \
  --arms warm_t0.2,warmreset_t0.2,resetfinal_t0.2,selfwarmreset_t0.2,selfresetfinal_t0.2 \
  --servers "$WR_SERVER:8000" --server-evidence-root "$WR_SERVER_EVIDENCE" \
  --namespace warmreset_smoke_01 --out "$WR_RUN" --replan-steps 5 --seed 7
```

`prepare` 输出服务端本次运行的证据目录：`<WR_SERVER_EVIDENCE>/<随机 token>`。
`plan.json` 保存任务身份、rollout 参数、schedule/K、每臂规格摘要和 YAML 文本/哈希；
`yamls/` 中是可直接检查的配置。可用 `--arms all` 选择该环境已有协议的 warm 家族：
RoboCasa 为 macro-13 与 self-13 的 warm 臂并集，LIBERO 为 self-start 臂表。
`full` / `plain_k*` 不属于本入口；额外 reset/shoot 变体可以用显式逗号列表指定。

在模型服务端启动生产服务（π0.5 默认 concurrent，显式写出便于检查）：

```bash
uv run --no-sync scripts/serve_policy.py --concurrent --port 8000 \
  --cache-config "$WR_BASE" policy:checkpoint \
  --policy.config pi05_libero --policy.dir "$WR_CHECKPOINT"
```

在 driver 主机运行；另开终端/主机运行 agent。一个 server 可以同时处理多个 bundle，
`--concurrency` 控制每个服务端同时激活的臂数，agent 的 worker 数控制并发 episode 数。
若有多个独立 server，prepare 的 `--servers` 用逗号分隔，每个端点都要有 agent 绑定。

```bash
uv run --no-sync python -m exp.warm_reset.run run \
  --run-dir "$WR_RUN" --bind-host 0.0.0.0 --port 9100 --concurrency 4

uv run --no-sync python -m exp.warm_reset.run agent \
  --run-dir "$WR_RUN" --server "$WR_SERVER:8000" \
  --driver-host "$WR_DRIVER" --driver-port 9100 \
  --gpus 0 --workers-per-gpu 4 --prefix libero_host_a --conda-env "$WR_LIBERO_ENV"
```

若 LIBERO 模拟器是 venv 而不是 conda 环境（如 weilandserver 的 `~/libero_sim`），省略
`--conda-env`：agent 会直接启动 `PATH` 上的 `python`，因此用主 venv 的解释器运行 agent，
并让 `PATH` 先指向模拟器 venv、`PYTHONPATH` 含仓库根与 `src`：

```bash
env -u VIRTUAL_ENV PATH="$WR_LIBERO_VENV/bin:/usr/bin:/bin" PYTHONPATH="$PWD:$PWD/src" MUJOCO_GL=egl \
  .venv/bin/python -m exp.warm_reset.run agent \
  --run-dir "$WR_RUN" --server "$WR_SERVER:8000" \
  --driver-host "$WR_DRIVER" --driver-port 9100 --gpus 0 --workers-per-gpu 4 --prefix libero_host_a
```

多台 agent 的 `--prefix` 必须不同。driver 完成后，用 Ctrl-C 停止 agent，入口会清理其
worker 子进程。默认使用 LIBERO 自带 init pool；自定义完整 pool 通过 prepare 的
`--init-states-dir` 指定 worker 本地路径。索引始终是任务清单的 `init_indices`，
不把裁剪后的子集位置当作原始身份。`tasks --init-offset` 可以选择原始 pool 的一段。
清单 `episode_idx` 是本次清单位置，`orig_init_state_idx` 保留原始索引。

## GR00T 和 RoboCasa

GR00T LIBERO：prepare / tasks 改为 `groot_libero_10` 或 `groot_libero_spatial`，
base YAML 的 schedule 必须为 `groot_n15_k8_v1`。可用
`--arms warmreset_t0.75_n1,selfwarmreset_t0.75_n1,resetfinal_t0.5_n2,selfresetfinal_t0.5_n2`
或 `--arms all`。agent 自动用 `resize_size=256`，π0.5 则为 224；两者均使用标准
LIBERO worker。GR00T 服务在其现有依赖环境中启动：

```bash
"$WR_GROOT_PYTHON" -m exp.libero_groot.serve_groot_libero \
  --checkpoint "$WR_CHECKPOINT" --port 8000 --denoising-steps 8 \
  --cache-config "$WR_BASE" --concurrent --allow-dynamic-bundles
```

RoboCasa 清单使用 canonical env name；显式选择 seed 段、layout 和 style：

```bash
uv run --no-sync python -m exp.warm_reset.run tasks \
  --env groot_rc --task-names OpenDrawer,CloseFridge --episodes 2 --out "$WR_TASKS"

uv run --no-sync python -m exp.warm_reset.run prepare \
  --env groot_rc --base-yaml "$WR_BASE" --tasks "$WR_TASKS" \
  --arms warmreset_t0.75,selfwarmreset_t0.75,midshoot_t0.75,selfmidshoot_t0.75 \
  --servers "$WR_SERVER:8000" --server-evidence-root "$WR_SERVER_EVIDENCE" \
  --namespace warmreset_rc_smoke_01 --out "$WR_RUN" \
  --base-seed 3000000 --layout 1 --style 1 --replan-steps 5

"$WR_GROOT_PYTHON" -m exp.robocasa365.serve_groot_n15 \
  --checkpoint "$WR_CHECKPOINT" --port 8000 --cache-config "$WR_BASE" \
  --concurrent --allow-dynamic-bundles
```

RoboCasa GR00T 使用 K=4 / `groot_n15_k4_v1`，并通过标准 `groot_tp` teacher adapter
运行。π0.5 RoboCasa 则选择 `pi05_rc`、π0.5 的臂 ID 和 K=10；服务命令使用上面的
`serve_policy.py`，改为 `--policy.config pi05_robocasa` 和对应 checkpoint。
RoboCasa worker 通过现有 island-A spawn 路径启动，driver 命令与 LIBERO 相同：

```bash
uv run --no-sync python -m exp.warm_reset.run agent \
  --run-dir "$WR_RUN" --server "$WR_SERVER:8000" \
  --driver-host "$WR_DRIVER" --driver-port 9100 --gpus 0 --workers-per-gpu 4 \
  --prefix robocasa_host_a --worker-python "$WR_RC_PYTHON" --robocasa-cwd "$WR_RC_CWD" \
  --egl-lib-dir "$WR_EGL_LIB" --egl-vendor-dir "$WR_EGL_VENDOR" --max-cached-envs 1
```

本入口使用普通未固定对象的 RoboCasa rollout；不会隐式套用 pinned-object 实验协议。
`base_seed + orig_init_state_idx` 是实际模拟器 seed。不同任务/臂共享相同原始索引列表；
self-start 的私有噪声在各臂共享 namespace 和标准 episode 身份。

## 准入和退出码

driver 把实际任务图、整份运行清单的哈希及 `run_id` 写入 `execution.json`；journal 和 driver stamped
worker 记录分别写入 `journal.jsonl`、`per_step.jsonl`。将各模型服务端本次 token
目录的**全部** JSONL 收集到 driver 主机后运行；如果本机可直接读取服务端路径，省略
`--evidence-dir` 即使用 prepare 输出的路径：

```bash
uv run --no-sync python -m exp.warm_reset.run admit \
  --run-dir "$WR_RUN" --evidence-dir "$WR_COPIED_SERVER_EVIDENCE"
```

多个服务端目录可以重复 `--evidence-dir`。输出完整 `admission.json` 和逐臂汇总；
退出码 0 表示准入全部通过，2 表示拒收。缺 episode、错误终局、运行清单/YAML 漂移、重复决策、
身份不符、服务端计数/网格不符或 JSONL 损坏均拒收。每集仅核算 accepted terminal 的
attempt；之前已接受但未终局的重试记录保留在原始文件中，不混入本次计数。
任务失败但完整结束（`success=false`、无 error）仍可准入，计入成功率分母。

warm reset 臂必须闭合三路证据，报告 `measured_total_nfe`；self 臂包括 K+N 的实测总数。
普通 `warm_t*` 是 `worker_reference`：仅验证 worker/journal 的完整性和 WARM_START，
无新增服务端计数，NFE 明确为 `null`，不能当作实测 NFE 证明。
`admitted` 和 `successes` 同时给出，拒收的 episode 不进入计数或成功数。

run 完成后仍须执行 admit。run 对错误/缺失终局返回 2，对中断返回 130；模拟器正常
失败不等于基础设施错误。本入口不跨 driver 重启复用旧运行目录：prepare 拒绝已有目录，
run 拒绝已有 execution。中断后保留证据、创建新运行；这防止重启后的 attempt=1 与旧证据
混淆。正在运行的 YAML 不允许修改。

本轮验证了 CPU 数值/边界测试和真实本地 TCP 多 bundle 闭合；真实 checkpoint 的 GPU
数值对等及模拟器部署冒烟需要在对应运行环境执行，不据此声称吞吐或实验成功率。
