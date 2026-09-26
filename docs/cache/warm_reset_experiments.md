# Warm reset 多臂实验入口

`python -m exp.warm_reset.run` 复用正式 concurrent server、conductor 和标准
LIBERO / RoboCasa worker。它提供任务清单导出、臂 YAML 生成、多臂调度及证据准入，
不再依赖 step_diag 专用 server 或 monkey patch；`python -m exp.warm_reset.analysis`
对准入后的运行做配对统计。支持的臂：warm 家族（精确续跑、缓存 / 自产 warm reset 与 shoot）、
`full`、`plain_k<k>` 与无库自产；环境来自可扩展的注册表（新 benchmark 的钩子见文末）。

本页命令中的大写变量由操作者设置为实际路径/地址；在仓库根目录运行。
`WR_RUN` 是一个尚不存在的运行目录，之后把整个目录复制到 worker 主机的同一路径
（或使用共享目录）。`WR_BASE` 是对应环境已经能跑通的 warm-start YAML：
CP1 `always_search`、`write_policy: {type: never}`、in-memory 冻结库，
`preload_path` 必须是各服务端可访问的绝对路径。库必须包含请求的任务、快照和匹配的
schedule；加载时仍会执行生产库校验。生成器保留检索配置，将 judge 固定为该臂的
`always_warm_start`，并冻结已展开的环境变量。不同机器的 library 路径需要一致。
只含 `full` / `plain_k*` / 无库自产臂的运行不需要 `WR_BASE`（这些臂的 yaml 由无库模板生成）。

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
`full` / `plain_k<k>`、无库自产臂和额外 reset/shoot 变体用显式逗号列表指定，见下节。

## 臂类型与入口参数

一次 `prepare` 可以混合下列臂；`plan.json` 的每个臂记录冻结其类型（`kind`，缺省 = warm）。

| 臂 | 例子 | 入口参数 | 生成的 yaml | 需要库 / `--base-yaml` |
|---|---|---|---|---|
| 精确续跑（ours） | `warm_t0.2`、`warm_t0.875` | `--arms warm_t0.2` | 基线 yaml，judge `always_warm_start`，无块 | 是 |
| 缓存 warm reset / shoot | `warmreset_t0.2`、`midreset_t0.75_n1` | `--arms warmreset_t0.2` | 基线 + `warm_reset` 块（`source: cache`） | 是 |
| 带库自产 | `selfresetfinal_t0.2` | `--arms selfresetfinal_t0.2`（`--self-trigger verdict`，缺省） | 基线 + `warm_reset` 块（`source: self`），由 WARM_START 判决触发 | 是 |
| 无库自产 | `selfresetfinal_t0.2` | 同上 + `--self-trigger always` | 无库模板 + `warm_reset` 块（`trigger: always`、`start_t`） | 否 |
| `full` | `full` | `--arms full` | 无库模板 + `miss: {num_steps: K}` | 否 |
| `plain_k<k>` | `plain_k2` | `--arms plain_k2`（GR00T 另需 `--k-servers`） | 无库模板 + `miss: {num_steps: k}` | 否 |
| 钉物体 RoboCasa（任意臂） | PnP 任务 | `--pinned-objects <manifest>`（prepare 与 agent 都给） | 臂 yaml 不变；pin 三键写入每个 EpisodeTask | 视臂而定 |

「无库模板」不启用任何 checkpoint、没有 `preload_path`，server 热加载它不读任何库文件；
`--self-trigger always` 作用于本次运行的全部 self 臂。只含无库臂的运行可以省略
`--base-yaml`。例如 π0.5 LIBERO 的 plain / full + 无库自产（MetaWorld 也是这一形态）：

```bash
uv run --no-sync python -m exp.warm_reset.run prepare \
  --env pi05_libero_10 --tasks "$WR_TASKS" --self-trigger always \
  --arms full,plain_k2,selfwarmreset_t0.2,selfresetfinal_t0.2,selfmidfinal_t0.2 \
  --servers "$WR_SERVER:8000" --server-evidence-root "$WR_SERVER_EVIDENCE" \
  --namespace wr_libero_selfonly_01 --out "$WR_RUN" --replan-steps 5 --seed 7
```

π0.5 的 MISS 步数按 bundle 生效：一个 concurrent server 同时服务 `full`、`plain_k1`、
`plain_k2` 和全部 warm 臂，MISS 合批桶按步数分开（`("miss", None, k)`）。每个 MISS 决策
的 `__hit_meta__["miss_nfe"]` 与服务端证据行都记录执行步数，准入核对它等于臂的 k。
不写 `miss` 块的 yaml 行为不变（仍读模块常量 `_NUM_STEPS`，step_diag 的进程级设置照旧有效）。

### GR00T：每个 K 一个端点

GR00T 的 MISS 步数是进程级的 `--denoising-steps`，server 会拒绝步数与活 head 不符的
`miss` bundle。`plain_k<k>`（k ≠ K）需要另起 `--denoising-steps k` 的 server，并在 prepare
里用 `--k-servers k=host:port[,k=host:port...]` 声明；`--servers` 仍是跑 K 步的端点（warm
臂、自产臂、`full`）。计划里每个臂冻结 `endpoints`，driver 只把臂派到这些端点（同组多个
端点时按臂轮转），准入拒收派错端点的 episode。每个端点各起一个 agent（`--server` 取该端点）。

```bash
uv run --no-sync python -m exp.warm_reset.run prepare \
  --env groot_libero_10 --base-yaml "$WR_BASE" --tasks "$WR_TASKS" \
  --arms full,plain_k1,plain_k2,warm_t0.875,warmreset_t0.75_n1,selfwarmreset_t0.75_n1 \
  --servers "$WR_SERVER:23180" --k-servers "1=$WR_SERVER:23182,2=$WR_SERVER:23183" \
  --server-evidence-root "$WR_SERVER_EVIDENCE" --namespace wr_groot_l10_01 --out "$WR_RUN" \
  --replan-steps 5 --seed 7

"$WR_GROOT_PYTHON" -m exp.libero_groot.serve_groot_libero \
  --checkpoint "$WR_CHECKPOINT" --port 23182 --denoising-steps 1 --concurrent --allow-dynamic-bundles
```

K 端点可以同时服务 `full` 与 warm 臂（`full` 的 `miss.num_steps` = K）。`plain_k*` 端点不需要
`--cache-config`：计划的 yaml 由 driver 热加载，无库。

### RoboCasa：钉物体与稳定 experiment id

PnP 线与 step_diag 一样钉物体：`prepare --pinned-objects exp/robocasa365/config/pnp_pinned_objects.json`
把 manifest 的 `pin_id` 与本次任务的槽位表冻结进 `plan.json`（`pin_id` / `pins`），每个
EpisodeTask 带 `pin_id` / `pin_task_id` / `pinned_objects`；agent 必须给同一 manifest 的本机
副本（`agent --pinned-objects`，`pin_id` 不符即拒），worker 再逐集核对。准入的受信身份含
`pin_id` / `pin_task_id`，分析的配对身份含 `pin_id`。未钉的计划拒绝带 `--pinned-objects`
的 agent，钉住的计划拒绝不带它的 agent。

RoboCasa 的 episode `experiment` 缺省为 `warm_reset_<token>`（每次运行不同），因此不同运行的
自产臂不共享噪声。`prepare --experiment-id <稳定 id>`（`[A-Za-z0-9._-]`）把它固定下来：分段
运行、重跑同一 (task, init, attempt, decision) 时自产 seed 相同。LIBERO 的 experiment 固定为
suite 名，不接受该参数。

### LIBERO：init pool 摘要

给了 `--init-states-dir` 时 `plan.json` 记录 `init_pool_sha256`：prepare 主机能读到该目录就
按 step_diag 的 `sha256_tree` 计算（再给 `--init-pool-sha256` 必须相等）；目录只在 worker 主机时
用 `--init-pool-sha256` 传入（在有同一 pool 的主机上 `python -c "from exp.step_diag.envs import
sha256_tree; print(sha256_tree('<pool>'))"`）；两者都没有时记 `null`，prepare 输出一条警告，分析
改用 `path:<目录>` 作 pool 身份。

## 启动服务

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

未给 `--pinned-objects` 时是普通未固定对象的 RoboCasa rollout（见上节的钉物体用法）。
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
无库自产臂的决策是 `SELF_ONLY`（worker 行与服务端行都是），`start_t` 为臂的值，同样按 K+N 计价。
`full` / `plain_k*` 的决策是 MISS（`start_t` 为 null），服务端行的 `miss_nfe` 必须等于臂的步数，
逐集输出 `miss_nfe`，`measured_total_nfe` 为 MISS 步数之和；旧 server 忽略 `miss` 块时行里没有
`miss_nfe`，按 `miss_nfe_missing` 拒收。
普通 `warm_t*` 是 `worker_reference`：仅验证 worker/journal 的完整性和 WARM_START，
无新增服务端计数，NFE 明确为 `null`，不能当作实测 NFE 证明。
`admitted` 和 `successes` 同时给出，拒收的 episode 不进入计数或成功数。

run 完成后仍须执行 admit。run 对错误/缺失终局返回 2，对中断返回 130；模拟器正常
失败不等于基础设施错误。本入口不跨 driver 重启复用旧运行目录：prepare 拒绝已有目录，
run 拒绝已有 execution。中断后保留证据、创建新运行；这防止重启后的 attempt=1 与旧证据
混淆。正在运行的 YAML 不允许修改。

## 分析：配对统计

多次运行（同一环境的分段，例如按任务段拆开的 GR00T LIBERO）各自 `admit` 后合并分析：

```bash
uv run --no-sync python -m exp.warm_reset.analysis \
  --run-dir "$WR_RUN_A" --run-dir "$WR_RUN_B" \
  --out-json wr_libero10.json --out-md wr_libero10.md --success-length-out wr_libero10_len.json
```

适配器读 `plan.json`（重新核对 yaml）、`execution.json` 与 `admission.json`（必须是该 execution 的），
按配对身份（任务、环境适配器给出的字段：LIBERO 为 init 索引 / seed / pool，RoboCasa 为
init 索引 / seed / lane / pin / layout / style，以及 env id）组织每集结局、实测 NFE 与决策数，
统计直接调用 `exp.step_diag.analysis.warm_variants` 与 `success_length` 的函数，输出同形状的
JSON / Markdown：`{"m<m>": 面板}`（每个等续跑 NFE 面板含 `full` 与该面板的臂，逐任务 cells /
paired_deltas / comparison_problems，外加 `macro`），以及 `{"decisions": {...}}` 的成功集长度
（journal 没有环境步数，只输出决策数单位）。不同环境的运行拒绝合并；同一臂在两次运行里冻结
得不同（不计 token / 证据目录 / namespace）、或运行间 rollout 契约不同，都报出并排除出 macro。

框架 cell 默认不与 step_diag cell 配对。`--step-diag-root` / `--step-diag-server-rows` /
`--step-diag-arms` 可把 step_diag 臂作为 `step_diag:<arm>` cell 载入对照，只有加
`--allow-cross-framework` 才计算同臂的「框架 − step_diag」差（标 `[cross-framework]`，永不进
macro）；π0.5 合批数值差与自产噪声公式不同（见迁移调研）使两者不是同一实现。

## 新 benchmark 的注册钩子

入口的环境来自 `exp.warm_reset.envs` 注册表；step_diag 的六个环境已按原 id 注册。新 benchmark
在自己的包里放 `exp/<包名>/warm_reset_env.py`，导入时调用 `register_env`：

```python
from exp.warm_reset.envs import BenchmarkAdapter, EntryEnv, register_env

class MyAdapter(BenchmarkAdapter):
    name = "my_benchmark"
    def validate_rollout(self, env, rollout): ...          # 除 replan_steps 外的 rollout 参数
    def export_tasks(self, env, *, task_ids, task_names, episodes, init_offset): ...  # tasks 子命令
    def experiment(self, env, manifest): ...               # EpisodeTask.experiment（自产 seed 键之一）
    def episode_extra(self, env, manifest, task): ...      # worker 需要的 EpisodeTask.extra
    def identity(self, env, manifest, episode): ...        # worker 在 episode_start 发出、准入可信的额外身份键
    def pairing(self, env, manifest, episode): ...         # 分析的配对身份字段（init_idx、env_seed 等）
    def worker_agent(self, env, manifest, *, specs, driver_host, driver_port, options): ...  # WorkerAgent

register_env(EntryEnv(env_id="pi05_mybench", policy="pi05", benchmark="mybench", action_horizon=5,
                      k_full=10, schedule_id="pi05_v1", adapter=MyAdapter(), default_arms=(...)))
```

`get_env` / `env_ids` 在首次遇到未注册的 id（以及 CLI 列选项）时导入全部
`exp/*/warm_reset_env.py`，此后 `tasks` / `prepare` / `run` / `agent` / `admit` / `analysis`
都认得新 id，无需改 `exp/warm_reset`。插件模块必须轻量（模块级不导入模拟器或模型，driver 与
准入跑在主 venv）；导入失败的插件只在请求它的 id 时报错，不影响其他环境。已有 step_diag
`EnvSpec` 行时可用 `entry_env_from_spec(spec, adapter, default_arms)`。`policy` 必须是
`pi05` 或 `groot`（服务端执行体族），worker 的 `WorkerSpec` 由入口按计划生成后交给
`worker_agent`（`options` 含 agent 的 `conda_env` / `rc` / `pinned_objects` 参数），适配器可直接用
或自行改写。

本轮验证了 CPU 数值/边界测试和真实本地 TCP 多 bundle 闭合；真实 checkpoint 的 GPU
数值对等及模拟器部署冒烟需要在对应运行环境执行，不据此声称吞吐或实验成功率。
`miss` 块的真实模型 B=1 对等（每 bundle 的 `plain_k` / `full` 与 step_diag 进程级 plain 臂逐位相等）
见 `tests/cache/warm_reset/test_miss_parity_manual.py` 与
[`logs/warm_reset_framework_completion.log.md`](../../logs/warm_reset_framework_completion.log.md)。
