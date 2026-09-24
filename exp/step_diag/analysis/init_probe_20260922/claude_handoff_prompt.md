# 发给 Claude 的交接 prompt：缓存初始化敏感性先行实验

请接手 Codex 已启动、为让出设备而中止的 `init_probe_20260922` 实验。**先排入你现有的设备队列，不抢占当前实验；资源轮到它时再运行。目标是补完原定 40 集，校验、分析并交付中英文结果和直接可看的 PNG，然后释放本实验资源。不要擅自扩成 50/500 集，也不要重做已经完成的 warmreset/resetfinal 成功率阶梯。**

请先按 `logs/session_handoff.md` §0 初始化，并读其要求的权威运行文档。继续遵守 server 只在 server 节点、worker 只在 timan、tether 同步、先验证再放量的纪律。直接读本机工作区；相关改动尚未 commit。此次不运行 git 命令，不 commit/push，不发 agentchat 消息，不给我 ZIP。交接中旧端口/GPU 只是上轮配置，不代表现在空闲；以你的实时设备队列为准。只做本实验的必要恢复，不改其他实验代码或进程。

## 1. 问题与冻结设计

我们想知道：为什么缓存重置方法在 π0.5 上明显提高成功率，在 GR00T 上却没有同等提升？本实验只检验一个机制假说——**完整噪声端重启后，不同缓存初始化造成的动作差异，经过各次前向是收缩还是放大？** 不预设 GR00T 会抹去缓存，也不把动作差异大小当作动作质量。

这是固定观测的反事实诊断：环境始终由原有 full 策略执行，GR00T 4 步、π0.5 10 步。在指定决策处复用同一 stage-2 条件，额外跑不同初始化；这些额外动作不控制环境。

冻结参数：

- 模型：GR00T N1.5 RoboCasa target-posttraining checkpoint-60000；π0.5 RoboCasa PyTorch checkpoint，路径见下文。
- 任务：`CloseFridge`（main）、`PickPlaceCounterToStove`（pnp）。每模型每任务 10 集，总计 40 集。
- `experiment_id=init_probe_pilot_20260922`，`arm_id=shadow`。这个 shadow 名称是复用服务/评测外壳，实际矩阵以 `probe_contract.json` 为准，原 shadow 大矩阵已由 wrapper 替换。
- `base_seed=2000000`，每任务 `init_idx=0..9`，即环境种子 2000000..2000009；layout/style=1，replan=5；PnP 使用原冻结 pin 表。
- 决策索引：`0,5,15,30,60`，只取 episode 实际到达的索引，不强行让成功结束的 episode 继续。
- 缓存噪声水平固定 **T=0**，即缓存的最终动作。这不是原 warmreset 的中间快照扫描。
- **N=1、2 独立比较**，同一初始化用于两个 N。从共同的噪声水平 t=1 出发到 t=0；GR00T 原生坐标 s=0→1，π 原生 t=1→0。
- 初始化四种：检索到的缓存最终动作；同任务、排除检索获胜轨迹后均匀抽取另一条轨迹的一个缓存 entry；全零；私有生成器的高斯噪声。随机缓存是 uniform entry，不是 uniform trajectory。
- 每个观测保存 teacher 动作、四种初始化的 N+1 个状态、原始观测数组、身份和 SHA256。主指标是相对检索缓存的 `output RMSE / input RMSE`，仅前 5 个动作 × 前 12 个有效归一化维度。先集内平均再集间平均，不把多个观测当作独立 episode。
- 私有随机生成器和 `torch.random.fork_rng` 隔离辅助前向；检查全局 RNG、缓存载荷未变，同输入重复前向一致。

## 2. 当前进度与停止状态

2026-09-22 20:59 CDT，Codex 已停止自己的两个服务器、两个远端 worker 及调度器，并确认端口和 GPU 占用释放。没有自动重启。**这描述的是停止当时，不是对当前机器空闲状态的保证。**

| 模型 | 任务 | 已完成并核验 | 要补 |
|---|---|---|---|
| GR00T | CloseFridge | 10/10，idx 0..9 | 0 |
| GR00T | PickPlaceCounterToStove | 1/10，idx 0 | 9，idx 1..9 |
| π0.5 | CloseFridge | 1/10，idx 0 | 9，idx 1..9 |
| π0.5 | PickPlaceCounterToStove | 0/10，尚未启动 | 10，idx 0..9 |

合计完成 **12/40**，剩余 **28**。有效观测：GR00T 55，π 5。两个模型各还有 4 个来自未完成集的观测，不能当作完成集证据混入。

先读以下本地文件了解已冻结结果：

```
exp/step_diag/data/init_probe_20260922/pilot/controller_status.json
exp/step_diag/analysis/init_probe_20260922/partial_results.zh.md
exp/step_diag/analysis/init_probe_20260922/partial_results.en.md
exp/step_diag/analysis/init_probe_20260922/partial_initialization_sensitivity.png
```

初步 CloseFridge 的同任务另一轨迹缓存对照：GR00T（10 集）N1 最终 0.0848，N2 第一步 0.5185、最终 0.3169；π（仅 1 集）分别 0.1474、0.5218、0.9232。**这只是待复核线索，不是要复现的预设结论。** 两步零初始化对照分别 0.4447、0.4763，差距明显依赖对照类型。不同模型后续访问的观测、缓存库不同；本实验不能直接证明成功率变化的因果来源。

## 3. 代码、数据和设施

本机工作区（weilandserver）：`/home/weiland/projects/openpi`。

本次新增/使用的代码：

```
exp/step_diag/serve_init_probe.py
exp/step_diag/analysis/analyze_init_probe.py
exp/step_diag/ops/run_init_probe_pilot.py
tests/exp/step_diag/test_init_probe.py
exp/step_diag/data/figures/plot_init_probe_partial.py
```

`serve_init_probe.py` 是进程内 wrapper，不修改生产模型。GR00T 使用原 `staged.denoise_loop`，每次重新构造会被处理过程修改的 head 输入；π 使用现有 `warm_variant_stage3(..., variant="reset_final")` 并记录实际逐步状态。它将 `envs.git_commit` 在进程内替换成工作区来源标识，因此自身启动不会执行 git。

原始模型服务端和 probe 证据已在本地：

```
exp/step_diag/data/init_probe_20260922/pilot/{groot,pi05}/server/
exp/step_diag/data/init_probe_20260922/pilot/{groot,pi05}/probe/
exp/step_diag/data/init_probe_20260922/pilot/{groot,pi05}/server.log
exp/step_diag/data/init_probe_20260922/pilot/{groot,pi05}/client_stopped/
exp/step_diag/data/init_probe_20260922/pilot/{groot,pi05}/partial_summary.json
```

`client_stopped/` 是停止时从 timan107 拉取并核 SHA 的 **journal + launch 快照**，应保持不变，后面用它确定原分段哪些 episode 真正完成。运行计划、per-step 和完整 worker 产物仍在远端原目录。

远端 worker 节点和目录：

```
node: timan107
repo: /scratch/zixuans8/step_diag/openpi
python: /scratch/zixuans8/Isaac-GR00T/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python
env: /scratch/zixuans8/step_diag/openpi/exp/step_diag/config/rc_timan107.env
out root: /scratch/zixuans8/step_diag/openpi/exp/step_diag/data/init_probe_20260922/pilot
worker artifacts: <out root>/{groot_tp,pi05}/shadow/
launcher: exp/step_diag/ops/run_rc_cell.sh
pin: exp/robocasa365/config/pnp_pinned_objects.json
```

上轮为本机 server + timan107 worker：GR00T 端口 23158，worker GPU1；π 端口 23147，worker GPU2；公网 endpoint 是 `ziyanglin.com:<port>`。本机 4090 为 48GB 改装卡。GR00T 约 6GB 显存，π 约 8GB；缓存加载还会占较多主机 RAM。π 缓存较大，上轮加载约 8 分钟，安静不等于卡住，须等端口实际监听再起 worker。是否仍用这些 GPU/端口，必须服从你的实时排队结果。

本机模型与 Python：

```
GR00T code: /home/weiland/gr00t_n15
GR00T python: /home/weiland/gr00t_n15_venv/.venv/bin/python
GR00T checkpoint: /home/weiland/ckpt_n15_robocasa_tp/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000
pi05 python: /home/weiland/projects/openpi/.venv/bin/python
pi05 checkpoint: /home/weiland/ckpt_pi05_robocasa_pytorch
GR00T cache: /data/robocasa365_cache/cache_artifacts_w13/groot_tp_spatial_pool_16_w13_full.pkl
pi05 cache: /data/robocasa365_cache/cache_artifacts_w13/pi05_spatial_pool_16_w13_full.pkl
configs: exp/step_diag/config/arms/{groot_rc,pi05_rc}/shadow.yaml
```

## 4. 恢复必须处理的三个坑

**不要原样运行 `run_init_probe_pilot.py`。** 它是首次启动的调度器，没有做好本次中断后的恢复：会把旧失败/中止 cell 日志判成失败；旧 `probe_contract.json` 存在会使服务拒绝重启；分析默认假设单段完整数据。

**不要删除旧 contract 或覆盖旧数组来绕过检查。** `serve_init_probe.py` 故意拒绝覆盖，数组名由 `(task_uid, attempt, decision_idx)` 得出。Conductor 重启后 attempt 可能又从 1 开始，旧中断 episode 的这些键会与重跑撞名。仅按 uid+attempt 合并，也会把旧中断片段冒充新完成集的一部分。

**不要把剩余 9 集作为 `--episodes 9` 直接传原 launcher。** 原 `shadow` 评测冻结每任务 10 集，而且 run-plan 会校验。应保留原 10 集计划和 journal，让 conductor 自动跳过已完成的集。

## 5. 推荐恢复方案：新服务分段 + 原 worker journal 续跑

先实现/核对必要的恢复和合并逻辑，不改推理矩阵。推荐：

1. 为续跑服务创建全新的本地目录，例如
   `exp/step_diag/data/init_probe_20260922/resume_r1/{groot,pi05}/{probe,server}`。
   使用新的 server launch-id。保留原 `experiment_id`、缓存配置、checkpoint 和其他实验参数。
2. 从旧 `probe/probe_contract.json` 读取原服务器 `argv`，只替换输出目录和 server launch-id。新 manifest 的 `config_sha` 必须与同模型旧值一致，再起 worker。不要手抄 hash，不要跳过 stamp 校验。
3. Worker 仍用远端原 out root 和同样的 experiment/arm/task/seed/server endpoint，保留 journal/run_plan。原 driver 会跳过 accepted 的 done/failed 终态，继续缺失集。只启动 GR00T pnp、π main、π pnp；不重跑 GR00T main。两个任务共用一个模型 server 时串行运行，每 server 一个 worker。
4. 原 tmux tag 是 `initprobe`，续跑改新 tag（如 `initprobe_r1`）以保留旧日志；tag 不改变实验身份。如果同一 endpoint 尚未轮到空闲，就等待。不要为抢跑改 endpoint 导致原 run-plan 身份不匹配；若不得不迁机/换端口，先明确处理配置与计划迁移，不能删除旧证据后冒充原地恢复。
5. 对第一集恢复结果及时核对 uid、config_sha、完整执行步数、probe 数组和 terminal，再继续。若换了拓扑或修改 wrapper，先另设 smoke 验证；不要把 smoke 混入正式 40 集。

旧 config_sha（仅用于核对，启动时仍程序读取 manifest）：

```
groot: c55ae08580b94896a7d6c2ab221aa565c3ac7f0f317a67e99adc19e8f4beed00
pi05:  694b9fb7e95352492ae0bd9ca01f8c482c7839022466f8c6f844a0c9d7e1ccd1
```

config_sha 绑定配置合同，不包含 runtime source hash；输出目录和 server launch-id 不在合同内。**所以 sha 相同并不保证代码没有变，仍要保存并比对关键源文件内容。** 旧 wrapper 的 SHA256 是 `bf446a53ac5b5f7bdb293aa2b78424382260a105541a6725d0728d1c1b18fbf8`。分析器后来只增加了显式部分样本模式、比值复算和采样覆盖校验，未改变实际推理。

原模型启动环境，可在新分段输出目录下复用；从项目根目录执行：

```bash
# GR00T: use the saved contract argv after substituting fresh output paths and launch-id.
PYTHONPATH=/home/weiland/gr00t_n15:/home/weiland/projects/openpi/src:/home/weiland/projects/openpi:/home/weiland/projects/openpi/packages/openpi-client/src \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
/home/weiland/gr00t_n15_venv/.venv/bin/python -u -m exp.step_diag.serve_init_probe \
  --policy groot --probe-out exp/step_diag/data/init_probe_20260922/resume_r1/groot/probe -- \
  --benchmark rc --mode shadow --env-id groot_rc --arm-id shadow \
  --experiment-id init_probe_pilot_20260922 \
  --diag-out exp/step_diag/data/init_probe_20260922/resume_r1/groot/server \
  --launch-id resume1_23158 --cache-config exp/step_diag/config/arms/groot_rc/shadow.yaml -- \
  --checkpoint /home/weiland/ckpt_n15_robocasa_tp/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000 \
  --port 23158

# pi0.5: same condition, new evidence segment.
PYTHONPATH=/home/weiland/projects/openpi/src:/home/weiland/projects/openpi:/home/weiland/projects/openpi/packages/openpi-client/src \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OPENPI_SERVER_GPU_MEMORY_LOCK=0 PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python -u -m exp.step_diag.serve_init_probe \
  --policy pi05 --probe-out exp/step_diag/data/init_probe_20260922/resume_r1/pi05/probe -- \
  --mode shadow --env-id pi05_rc --arm-id shadow --experiment-id init_probe_pilot_20260922 \
  --diag-out exp/step_diag/data/init_probe_20260922/resume_r1/pi05/server \
  --launch-id resume1_23147 --policy-dir /home/weiland/ckpt_pi05_robocasa_pytorch \
  --cache-config exp/step_diag/config/arms/pi05_rc/shadow.yaml -- \
  --port 23147 policy:checkpoint --policy.config pi05_robocasa --policy.dir /home/weiland/ckpt_pi05_robocasa_pytorch
```

这些是已用过的服务入口，不是让你现在占设备。请用你已初始化的监控/排队设施运行，先确认端口空闲与服务就绪。

Worker 命令模板（在 timan107 上、经 tether 执行；变量由实际 model/队列和本机读取的 config_sha 填入）：

```bash
SD_EXP=init_probe_pilot_20260922 SD_BASE_SEED=2000000 \
SD_RC_ENV=/scratch/zixuans8/step_diag/openpi/exp/step_diag/config/rc_timan107.env \
SD_OUT_ROOT=/scratch/zixuans8/step_diag/openpi/exp/step_diag/data/init_probe_20260922/pilot \
SD_GPUS=<allocated_worker_gpu> \
bash /scratch/zixuans8/step_diag/openpi/exp/step_diag/ops/run_rc_cell.sh \
  <groot_tp_or_pi05> shadow <main_or_pnp> <original_server_endpoint> \
  <task> 10 - <config_sha_read_from_manifest> initprobe_r1
```

三格依次对应：`groot_tp / pnp / PickPlaceCounterToStove / ziyanglin.com:23158`；`pi05 / main / CloseFridge / ziyanglin.com:23147`；`pi05 / pnp / PickPlaceCounterToStove / ziyanglin.com:23147`。

## 6. 合并规则与验收

请在新 `merged/` 输出目录合并，保留 pilot 和 resume 的全部原始证据。**不要把目录递归拼起来就分析。**

- 旧 pilot 分段只能纳入 `client_stopped/` 当时 accepted 终态对应的 12 集。旧的 8 个未完成观测即使 uid+attempt 后来被重用，也永不进入最终数据。
- 新分段只纳入续跑中新完成、与新 server 终态和 worker launch/driver 身份一致的 episode；不能只用最后 journal 的 uid+attempt 去筛旧分段。
- 最终 client 集只保留一份最新累积 journal（以及需要的 launch 清单），不要再把旧 journal 快照追加进去造成重复计数。
- Probe 行、数组、server rows 按已验收分段复制；必要时在数组相对路径加入 segment 前缀，保持原数组内容/hash不变。记录 `merge_provenance.json`，保留每段 contract、manifest、来源与纳入/排除清单。服务端 manifest 的合同相同而 runtime 可能不同，保留来源，不伪造单段运行。
- 原分析器当前接受一个 probe/server/client 目录，不直接支持多段拼接，故需要先构建经身份核对的 merged 视图，或实现等价的显式分段分析；不能直接把它指向全部原始目录。
- 最终每模型两任务各 **10 个唯一 `(task, env_seed, init_idx)`**，全体 **40**，成功和失败 episode 都纳入；每集只计一个真正完成的执行。某集任务失败不应为了成功率重跑。
- 核查 full 实际步数、配置/模型/缓存身份、所有按计划可到达的决策观测、数组 SHA256、有限值、同输入重复、RNG/载荷不变、N1/N2 同初始化、首步 Euler 恒等式。用原始数组重算比值。
- 用正式完整模式分析，**最终交付不能开 `--allow-incomplete` 冒充完成**。该开关仅用于本轮中止后的部分分析。

可复用分析入口，假定 merged 视图按 `<merged>/<policy>/{probe,server,client}` 布局：

```bash
CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/init_probe_mpl PYTHONPATH=src:. \
.venv/bin/python -B -m exp.step_diag.analysis.analyze_init_probe \
  --probe-dir <merged_policy_dir>/probe --client-dir <merged_policy_dir>/client \
  --out-json <merged_policy_dir>/summary.json \
  --out-figure exp/step_diag/analysis/init_probe_20260922/<policy>_complete.png
```

现有回归测试：本轮推理 wrapper 加现有 warm 变体最初 32 passed；最近 CPU 分析完整性测试 3 passed。必要修改之后重跑相关测试，不要宣称全仓测试已通过：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -B -m pytest -q \
  tests/exp/step_diag/test_init_probe.py \
  tests/exp/step_diag/test_groot_warm_variants.py \
  tests/exp/step_diag/test_warm_variants.py
```

## 7. 交付和资源纪律

完成后输出中英文两份报告及 PNG，放在 `exp/step_diag/analysis/init_probe_20260922/`。保留现有 partial 报告为历史，不把部分结果覆盖成仿佛从未中断。报告同时呈现同任务缓存、零、高斯三种对照，N1 和 N2 的首步/最终结果，按 episode 统计不确定性；不要只挑支持假说的格子。单集退化 bootstrap 区间不代表确定性。

需要明确区分：缓存初始化敏感性、与 full 动作的接近程度、真实任务成功率是三个问题。本实验直接测第一项，不替代已有成功率实验；同模型内部固定观测，跨模型并非同一整段观测轨迹。

传输只走 tether；timan `/scratch` 不在 pull/push 允许根目录，须 `/tmp/sdiag/` 中转。小文件顺序传输，上轮并发 pull 遇到 `too_many_in_flight`；逐文件 SHA 对账，不要交付压缩包。本次不碰别人的图、日志、服务与 tmux。资源完成后只停止本实验的精确进程/会话，不做全局 `pkill` 或关闭别人使用的 MPS。

运行时 cron 负责定期状态巡检，Monitor 负责就绪/终态/错误触发，遵守你的 handoff 规范。Codex 上轮没有原生 Cron/Monitor 工具，使用了前台调度器；该调度器现已停止，不能假设仍有人监控。

**停止条件：原定 40 集全部有唯一终态，原始证据校验完成，中英文报告和 PNG 交付，自己占用的设备释放。若结果不支持“GR00T 更强收缩缓存差异”，如实报告，不追加实验追求预期结果。**
