# 在跑实验迁往 warm reset 一等公民框架的调研（含 MetaWorld 建议）

> 状态：`Design Only`（2026-09-25 调研，Authority: Execution；只读调研，零代码改动）。
> 上位：[`warm_continuation_first_class_plan.log.md`](warm_continuation_first_class_plan.log.md)（L3，G2 APPROVED 2026-09-25 11:58 CDT；其 §15 为 Verify / GPU 对等 / 冒烟记录）；step_diag 线 [`step_vs_warmstart_diagnostics_plan.log.md`](step_vs_warmstart_diagnostics_plan.log.md)、[`step_diag_libero_selfstart_plan.log.md`](step_diag_libero_selfstart_plan.log.md)。
> 数据快照：两队列 `queue_state.json` 于 2026-09-25 12:24 CDT 只读统计；代码引用均为当日工作树（含他线未提交改动）实读。未触碰在跑队列、状态文件、server、tmux 与远端主机。

## 0. 结论先行

1. **在跑的两条 step_diag 队列不要中途迁移。** π0.5 两条线已全部完成（RC 39/39、LIBERO 180/180 cell），剩下的只有 GR00T：RC 133 个 cell 未完（其中 89 个在钉物体的 PnP 线，入口目前不支持钉物体；且大多数臂已部分完成，迁移会让同一臂横跨两套实现），让 `sdq` 跑完。
2. **GR00T LIBERO（573/580 cell 未开始）是唯一值得迁的存量。** 条件是先补两件事：入口支持 `full` / `plain_k1` / `plain_k2`（GR00T 用「每个 K 一个进程」即可，exp 层 ~1 天），以及 step_diag 队列与入口之间的显存 / 内存预留机制（否则两边互不可见、会超卖主机）。GR00T 生产路径是 infer 锁串行、B=1，续跑与 step_diag 逐位相同（K=8 真模型对等已通过），迁移不引入合批数值漂移；收益来自一个进程服务多 worker 带来的显存 / RSS 摊薄与仿真重叠，**必须先实测 t_inf 与多 worker 吞吐**再决定。
3. **新入口今天就能跑的**：π0.5 K=10 与 GR00T K=4 / K=8 的全部 warm reset 臂（缓存 / 自产 × 快照 / 最终动作 × reset / shoot，含 GR00T `_n<N>` 解耦臂），以及精确 `warm_t*` 基线（仅 worker 级证据）；LIBERO（含自定义 init 池）与未钉物体的 RoboCasa。**跑不了的**：`full` / `plain_k*`、钉物体 RoboCasa、没有缓存库的自产臂、MetaWorld（缺整条接线）。
4. **自产臂没有库就跑不了，两套框架都一样**：生产路径与 step_diag 都要求 WARM_START 判决带一条 payload（至少要它的快照形状与 K）。无库自产有两条路：A「形状锚」库（1 条合成条目 + `placeholder` key builder + `task_scoped: false`，零 src 改动，~0.5 天）；B 在 `warm_reset` 块加「无检索自产」触发方式（新 L3，~2–3 天含 G1/G2）。
5. **`plain_k*` 在 π0.5 生产 server 上表达不了**：MISS 步数写死为 `interceptor._NUM_STEPS = 10`（`src/openpi/cache/interceptor.py:93`），step_diag 靠进程级猴补丁（`serve_diag_pi05.py:141-144`）。要上生产需「按 bundle 钉 MISS 步数」（L2，~1.5 天含评审），或临时用专用进程。
6. **MetaWorld 建议建在新框架上**（§9）：它没有历史数据要配对；两套框架都得新写 MetaWorld 的策略配置、episode runner / worker；wls 目前只剩约 12 GB 显存，step_diag「一臂一进程」需要 5 × 7.8 GB，新框架一个并发进程即可服务全部 5 个臂。前置三件：MetaWorld 接线（两边都要）、无库自产（先走 A，长期走 B）、`plain_k2`（按 bundle 钉 MISS 步数）。
7. **证据链不能直接喂给现有分析脚本**：自产 seed 公式、driver 侧文件形态、配对身份字段都不同。需要一个 exp 层适配器（~1 天），把 `admission.json` + 计划身份转成 `warm_variants` / `success_length` 需要的「配对身份 → 结局 / NFE / 长度」表，统计函数（`_boot_delta` / `_paired`）直接复用。
8. **生产结果不与 step_diag cell 混池**（计划 Q8）。π0.5 并发合批与 B=1 有数值差（真模型 B=4 实测 max|Δ| = 5.36e-3）。GR00T 没有合批，这条对 GR00T 只剩「自产噪声公式不同」一项：缓存臂续跑逐位相同，配对按 init 身份仍然成立，但自产臂之间的共同随机数不跨框架成立。

## 1. 现状快照

### 1.1 两条在跑队列（2026-09-25 12:24 CDT）

| 队列 | teacher | done | running | pending | 臂数 | 备注 |
|---|---|---|---|---|---|---|
| RC `sdq`（`/data/step_diag_self13/`） | π0.5 | 39 | 0 | 0 | 3 | 完成 |
| | GR00T | 140 | 13 | 120 | 21 | 未完 133 个 cell：PnP（钉物体）89、main 44；涉及 19 个臂，多数臂已部分完成 |
| LIBERO `sdlq`（`/data/step_diag_libero_self/`） | π0.5 | 180 | 0 | 0 | 9 | 完成 |
| | GR00T | 7 | 0 | 573 | 29 | 只完成 `groot_libero_spatial` 的 `warmreset_t0.75_n1` task 0–3、`warmreset_t0.5_n2` task 0–2 |

cell = teacher × arm × task（LIBERO 另 × env），每 cell 50 集、一台 server、一个 worker（`self13_queue.py:110-119`、`libero_queue.py:101-110`、`envs.py:231,246`）。GR00T LIBERO 在任何 RC job 还 pending 时被挡住（`libero_queue.py:166`），所以 RC 跑完之前它不会开跑。

### 1.2 step_diag 的运行模型（迁移时要对照的部分）

- **一 cell 一进程一连接**：π0.5 强制 `--non-concurrent`（`serve_diag_pi05.py:81-84`），GR00T RC 非并发（`serve_groot.sh:7`），`run_diag` 拒绝每 server 多于 1 个 worker（`run_diag.py:219-220`），LIBERO 容量 1（`run_libero_diag.py:358`）。
- **优先级与跨队列预算**：RC π0.5 > π0.5 LIBERO > RC GR00T > GR00T LIBERO，由两个队列互读对方 `queue_state.json` 的 `slots` 实现（`libero_queue.py:132-176`、`self13_queue.py:159-210`、`:393-398`）；每台主机按 `COST × 槽数 ≤ gpu_budget / ram_budget` 记账（h100 76 GB / 185 GB，wls 44 GB / 220 GB；`self13_queue.py:51-72`）。两进程之间没有锁，可能短暂超订。
- **跨机**：wls 本地 `bash -lc`，其余主机一律 `tether exec`（`self13_queue.py:94-104`）；server 在主机 tmux `sdsrv<port>`，cell 的 driver + agent 在 worker 主机私有 tmux socket `-L sdiag` 里（`run_rc_cell.sh:39-48`、`run_lib_cell.sh:30-44`）。队列负责起 / 停 server、重试（`MAX_TRIES=3`）、server 丢失后回 pending、从 journal 续跑。
- **实测时长**（queue log 中位数，含 60 s 轮询）：RC π0.5 约 28 min / 50 集 cell；RC GR00T 47（h100）/ 62（wls）min；LIBERO π0.5 spatial 8 / 13 min，libero_10 15 / 30 min。

### 1.3 新入口 `exp/warm_reset` 的运行模型

- **一次运行 = 一个 env × 若干臂 × 任务 × init**，`prepare` 冻结 YAML、任务身份、rollout 参数与端点，`run` 起 `ConductorDriver`，`agent` 在 worker 主机起标准 LIBERO / RoboCasa worker，`admit` 三路证据准入（`docs/cache/warm_reset_experiments.md`）。
- **server 由操作者手动起**（π0.5 `serve_policy.py` 并发；GR00T `serve_groot_{n15,libero} --concurrent --allow-dynamic-bundles`），一进程多 bundle，臂经 `load_cache_config` 热加载。
- **分配粒度是臂**：`assign_servers` 把整个臂（yaml）放到一台 server（`src/openpi/conductor/driver.py:62-120`），worker 绑定到 server；`--concurrency` = 每台 server 同时激活的臂数（`scheduler.eval_concurrency`）。未用 `sharding.shard_eval_stage`（把一个臂摊到多台 server）。
- **不跨 driver 重启续跑**：`execution.json` 以 `open("x")` 抢占，已有 journal 即拒（`exp/warm_reset/conductor.py:104-110`）；中断后要新建运行目录。
- **证据落在 server 本机**的 `<evidence_root>/<token>/`；远端 server（h100）须先拉回再 `admit --evidence-dir`。

## 2. 能力矩阵：新入口今天能跑什么

| 臂族 | π0.5 LIBERO | π0.5 RoboCasa | GR00T LIBERO（K=8） | GR00T RoboCasa（K=4） | 证据 |
|---|---|---|---|---|---|
| 缓存 warm reset（warmreset / resetfinal / midreset* / midfinal*） | 可 | 可（仅未钉物体） | 可（`_n1/_n2` 显式 N） | 可（仅未钉物体） | 服务端 + worker，实测 NFE |
| 自产 warm reset（self*） | 可（需库） | 可（需库，未钉） | 可（需库） | 可（需库，未钉） | 同上，K+N 计价、seed 可重算 |
| shoot（warmshoot / midshoot / midshoot50） | 只有 cache `warmshoot`（step_diag 无 π0.5 自产 shoot） | 同左 | step_diag 未定义 LIBERO shoot | 可，含 self 版 | 同上 |
| 精确 `warm_t*` | 可 | 可 | 可 | 可 | 仅 worker_reference，NFE 为 null |
| `full` / `plain_k*` | 不可 | 不可 | 不可 | 不可 | — |
| 钉物体 RoboCasa（PnP 线） | — | 不可 | — | 不可 | — |
| 无库自产 | 不可 | 不可 | 不可 | 不可 | — |

真实模型逐位对等（B=1）已通过：π0.5 K=10（`pi05_libero_10`）与 GR00T K=8（`groot_libero_10`）；**GR00T K=4（RoboCasa）的 manual 对等尚未跑**（本机无 RoboCasa wire 观测 npz），迁 RC 之前应补。

## 3. 差距逐项

### 3.1 `full` / `plain_k*` 在入口之外

- 入口拒绝：臂 ID 正则要求 `_t`（`exp/warm_reset/plan.py:43`），`trusted_expected` 要求 worker 行 `hit_type == WARM_START`（`admit.py:96`）。
- **π0.5**：`full` 可以用 `always_skip` gate 的 yaml 表达（MISS 走 K=10）；`plain_k<k>` 不行——生产 MISS 步数是模块常量 `_NUM_STEPS = PI05_V1.num_steps`（`interceptor.py:93`，MISS 调用 `:623`、`:2010-2017`），step_diag 用进程级 `_icpt._NUM_STEPS = exec_steps` 猴补丁（`serve_diag_pi05.py:141-144`），这与「一进程多 bundle」根本冲突。上生产需新特性「按 bundle 钉 MISS 步数」：yaml 加一个可选字段（缺省 = 10，逐位不变），interceptor 与 coordinator 的 MISS 分支按 bundle 取值；Pi05 批处理桶键已含 `num_steps`（`batching_coordinator.py:92-98`），合批不受影响。L2，~1.5 天含 G1/G2。
- **GR00T**：MISS 步数 = 活 head 的 `num_inference_timesteps`，即进程级 `--denoising-steps`。`plain_k1` / `plain_k2` 可以各起一个生产进程（`--denoising-steps k`，MISS-only yaml），但**不能与 K=8 的 warm 臂同进程**（活 schedule 守卫会拒绝 K=8 库）。exp 层改动：`arm_block` 接受 full / plain、计划里按 K 分端点、准入按 MISS 口径（~1 天）。
- **证据缺口**：生产 MISS 的 `__hit_meta__` 不带执行步数；step_diag 的 plain 臂准入要求服务端行 `executed_steps == m`（`aggregate_arms.py:367-370`）。上生产后 plain 臂的步数只能由进程启动参数证明（或给 MISS 加一个加法 wire 字段，属上面的 L2 范围）。

### 3.2 自产臂仍需库 / WARM_START 命中

自产臂在两套实现里都挂在 WARM_START 分支上：

- 生产：`Pi05WarmResetExecutor.run` 要 `cp_result.payload`（核 `denoising_num_steps == K`）与快照 `snapshot_x`（只用它的形状 / dtype / 设备生成私有噪声并整形结果；`src/openpi/cache/warm_reset/pi05.py:228-251`）；GR00T 同理取 `payload.intermediates[start_t]` 作 `like`（`groot.py:232-245`）。
- step_diag：自产臂必须 `--cache-config`（`serve_diag_pi05.py:44,68`），`_self_start` 同样以缓存快照为 `like`（`exp/step_diag/pi05.py:333-359`）。

也就是说，**库的内容对自产臂的数值没有贡献，只贡献「有一次 WARM_START 判决」与「快照形状 / K」**。无库自产的两条路：

- **A. 形状锚库（零 src 改动）**：yaml 用 `key_builder: placeholder`（只用 32 维 robot_state，`components/key_builder.py:103-140`）、`search_strategy.task_scoped: false`、`top_k: 1`、`gate: always_search`、`judge: always_warm_start`；in-memory 库只放 1 条合成条目：payload `action_chunk` 与 `intermediates[start_t]` 形状恰为模型动作 `[H, D]`（π0.5 MetaWorld：H=5、D=32），`denoising_num_steps = K`、`schedule_id = pi05_v1`。检索成本可忽略，每步必得 WARM_START，自产执行体照常跑，证据（自产证明、K+N 计价、seed 重算）全部有效。代价：`hit_type = WARM_START` 与 `winner_id` 是占位含义，必须在实验文档里写明；条目形状写错会让噪声形状跟着错（冒烟时核对 `self_direct_nfe == K` 与动作维度）。~0.5 天（exp 层写一个生成器 + 冒烟）。
- **B. 无检索自产触发（正式特性）**：`warm_reset` 块加 `trigger: always`（缺省 = `verdict`，即今天的行为），仅允许 `source: self`；interceptor 在块为 always 时不调 orchestrator，按 `effective_denoise_schedule` 与块里声明的 `start_t` 直接走自产 + 续跑，`like` 形状取自模型配置；证据的 `hit_type` 记一个新值（如 `SELF_ONLY`），准入相应分支。触及 config / 两个 interceptor / 证据准入，L3，~2–3 天含 G1/G2。优点是语义干净、没有伪检索。

### 3.3 shoot 臂

入口已支持：π0.5 只有缓存 `warmshoot`（step_diag 本就没有 `selfwarmshoot`，`envs.py:110`）；GR00T 的 `warmshoot / midshoot / midshoot50` 及其 self 版在 `arm_block` 里经 `GROOT_SHOOT_MODES` 放行（`exp/warm_reset/plan.py:54-58`）。step_diag 只在 RC 上允许 shoot（`envs.py:497-498`），与入口一致无差距。RC GR00T 还剩的 shoot 相关 cell 共 52 个。

### 3.4 GR00T K=8 的 N 解耦臂

已支持：`<mode>_t<s>_n<N>` → judge `start_t: s` + 显式 `num_steps: N`（`plan.py:45-77`），网格与 step_diag `split_warm_steps` 一致；真模型 K=8 逐位对等覆盖了全部 `_n1/_n2` 缓存与自产臂。

### 3.5 精确 `warm_t*` 基线

入口把它当 `worker_reference`：只核 worker / journal 完整性与 WARM_START，服务端不写证据（无块 ⇒ 无 wrapper），NFE 报 null（`admit.py:103-104`、`:218-221`）。若分析需要精确续跑的实测步数，要么接受「步数 = remaining」的推定，要么在无块路径上也挂一个只读证据 wrapper（会改变「缺块逐位不变」的装配形态，不建议）。

## 4. 调度 / 优先级 / 预算 / 跨机：入口 conductor 模型 vs step_diag 队列

| 需求 | step_diag 队列 | 新入口 | 差距与补法 |
|---|---|---|---|
| 工作单元 | cell（臂 × 任务 × 50 集），一进程一 worker | 一次运行 = 一 env 的全部臂 × 任务 × init，一进程多臂多 worker | 粒度更粗，吞吐更高；失败重跑要整个运行目录（见续跑行） |
| 优先级 | 四级跨队列优先级 | 无 | 同一 env 内由 `--concurrency` 与臂顺序决定；跨 env / 跨 teacher 只能靠操作者排程或按主机划分 |
| 预算 | 每主机 GPU / RAM 记账，两队列互读 | 无 | 入口的 server 对 step_diag 队列不可见 ⇒ 与队列共存时必须加一个外部预留（队列读一个预留文件从 budget 扣除），或整台主机只归一边（~0.5 天，改 step_diag ops，属他线代码） |
| server 生命周期 | 自动起停、就绪探测、丢失回 pending | 手动起（tmux），无健康检查 | 小型 launcher（按计划 `servers` 起 / 探活 / 记录 PID），~0.5 天 |
| 跨机 | `tether exec` 起 server 与 worker | agent 在 worker 主机手动起；运行目录需复制到 worker 主机同路径 | 可复用队列的 `sh()` 模式写一个 wrapper；证据目录要从 h100 拉回（tether pull） |
| 多 server | cell 绑 server | 整臂绑 server；worker 绑 server | 臂数少于 server 数时要用 `shard_eval_stage` 把臂摊开，入口未接 |
| 续跑 | journal ep 级续跑，同目录重启 | 拒绝复用运行目录 | 长跑（GR00T LIBERO 约 28,650 集）必须按「任务段」拆多个运行，或给入口加「同计划续跑」（~1 天）；分析层要能合并多个运行目录 |
| 重试 | 队列 3 次 + conductor 自身重试 | conductor 自身重试 | 足够 |

## 5. 证据 / 准入 vs 现有分析脚本

- **字段能对上的**：`executed_steps` ↔ `warm_reset.continuation_nfe` / `n_steps`；`n_stage3_calls`、`self_start`、`self_seed`、`self_direct_nfe`（嵌在 `warm_reset` 内）；`hit_type / start_t / schedule_id / decision_idx / status`；finalize 的 `n_decisions / terminal / outcome`；`init_idx` ↔ `identity.orig_init_state_idx`；RC `env_seed` ↔ `identity.seed`；`session_id` ↔ `(conn_id, episode_seq)`。
- **对不上 / 缺的**：
  - 自产 seed 公式不同：step_diag `noise_seed(experiment_id, env_id, task, (env_seed, init_idx, pool_sha), attempt, idx, "self")`（`exp/step_diag/recorder.py:70-73`、`:339-348`），生产 `stable_digest_int(namespace, experiment, task, orig_init_state_idx, attempt, idx, "self")`。`aggregate_arms.expected_self_seed` 会把每个生产自产决策判为 `self_seed_mismatch`；生产准入用自己的受信重算，不能套 step_diag 的。
  - 生产服务端行没有 `env_id / config_sha / client_stamp / manifest（checkpoint 与库 sha、h_exec）/ arrays / lane / pin_id / layout / style / init_pool_sha256`，LIBERO 身份里没有 env seed（`admit.py:106-116`）。
  - driver 侧形态不同：入口一次运行只有一个 `journal.jsonl` 与 `per_step.jsonl`（覆盖全部臂）+ `plan.json` / `execution.json`；没有逐臂 `launch_*.json`，没有 `episode_summary` 行 ⇒ 没有精确 `n_env_steps`（`success_length.py:74-176` 需要它，只能用决策数近似 `5(n−1) < n_env ≤ 5n`）。
  - `task_uid` 内嵌 `yaml_id`（每臂不同），跨臂配对要走计划里的 `(task, orig_init_state_idx, rollout seed / pool)`。
  - `admission.json` 只有逐集 `admitted / success / attempt / n_decisions / NFE / problems` 与逐臂汇总，没有配对身份。
- **建议适配器**（exp 层，~1 天，L1）：读 `plan.json` + `execution.json` + `admission.json`，按计划任务重建 `PAIR_IDENTITY_KEYS`（`aggregate_arms.py:405-412`）的等价物（task、init_idx、env_seed、pool、env_id），输出「配对身份 → 结局、实测 NFE、决策数」表，直接调用 `warm_variants._boot_delta / _paired` 与 `success_length` 的统计函数，不经 `cell_admission`（准入已由 `episode_problems` 完成）。适配器应拒绝把生产 cell 与 step_diag cell 放进同一个配对比较，除非该比较被显式标为「跨框架核对」。

## 6. 配对身份连续性

| 维度 | step_diag | 新入口 | 能否与已采数据配对 |
|---|---|---|---|
| LIBERO init | A 池 `exp/common/data/db_init/libero/<suite>_apool`，`orig_init_state_idx = i`（0..49），env seed 7，replan 5（`run_libero_diag.py:157-181`、`run_lib_cell.sh:25`） | `prepare --init-states-dir <apool> --seed 7 --replan-steps 5` + 任务清单 `init_indices 0..49`（worker `init_state_index_mode="orig"` 按目录内索引取） | 可以，但要一次交叉核对：同一 (task, idx) 两边渲染出的首帧 / 初始 sim state 相同；入口不记 `init_pool_sha256`，应在 `plan.json` 里补记 |
| RoboCasa seed | `base_seed = 2_000_000`，`env_seed = base_seed + init_idx`，layout 1 / style 1 / replan 5（`run_diag.py:139-140,242-243`） | `--base-seed 2000000 --layout 1 --style 1 --replan-steps 5` | main 线可以。初始状态是 seed 的纯函数；「每次推理新抽的」只是流匹配噪声（`exp/robocasa365/episode_runner.py:25-28`），缓存 warm 臂恒为 WARM_START、不抽噪声，因此 B=1 下（GR00T 生产路径即如此）同库同 checkpoint 的缓存臂轨迹可逐集复现（仍受 GPU kernel 非确定性约束，未实测）；自产臂与 full / plain 只能 init 级配对 |
| RoboCasa 钉物体 | PnP 线钉（`pnp_pinned_objects.json`，`pin_id 4d13ac5e…`，`run_diag.py:117-122,222-223,302-303`），main 线禁钉 | 不支持：strategy 不写 `PIN_EXTRA_KEYS`，agent 不传 `--pinned-objects`；RoboCasa runner 两边不对称即拒（`exp/robocasa365/episode_runner.py:461-515`） | PnP 线不能迁；补法：strategy 按任务写 pin 三键、agent 传 `pinned_objects_path`、身份与准入加 `pin_id`（exp 层 ~0.5–1 天） |
| RC `task_id` | `DEFAULT_EVAL_TASKS` 中的位置（`run_diag.py:73,104`） | 任务清单中的位置 | 不影响仿真；分析按 task 名配对 |
| experiment | LIBERO = benchmark；RC = experiment_id | LIBERO = benchmark；**RC = `warm_reset_<token>`**（`conductor.py:49-51`） | RC 自产 seed 因此随运行而变：同一 namespace 的两次 RC 运行不共享噪声。若要跨运行共同随机数，应把 RC experiment 改为稳定值（exp 层一行，需确认 RC runner 对 experiment 的其他用途） |
| 自产噪声 | `noise_seed(...)` | `EpisodeDigestSeed(namespace, keys)` | **不可能相等**（计划 F7）：迁移后的自产臂与已采自产臂只在 init 级配对，不再共享噪声 |
| 缓存臂数值 | B=1 | π0.5 并发 B>1（max|Δ| 5.36e-3）；GR00T B=1（infer 锁） | π0.5 不混池；GR00T 续跑逐位相同（K=8 真模型已证），K=4 待补 manual 对等 |
| attempt | 进 seed | 进 seed | 一致 |

## 7. 吞吐

- **π0.5**：step_diag 每进程 B=1、一连接，launch-bound（4090 上 GPU 利用 6–15%），每进程 7.8 GB 显存 + 31 GB RSS，wls 约 5 个进程封顶。生产并发一个进程（≤3 replica）承载全部臂与 worker，stage 1/2 跨臂合批、stage 3 按网格键合批；自产臂每决策两次 stage-3 提交（K 步自产桶 + N 步续跑桶）。既有实测：wls 本机 16 worker 18.7 ep/min（GPU 86%），4 worker 6.15 ep/min；a100 单进程约 12 inf/s、3 replica 26–29 inf/s。预期同主机并发 episode 数从 ~5 提到 16–48，吞吐 3–5 倍直至 GPU 饱和（计划 §10.4）；π0.5 线已跑完，这条收益主要落在 MetaWorld。
- **GR00T**：生产非 trace 路径是进程级 infer 锁串行（计划 F1），没有合批。单进程上限约 `1 / t_inf` 决策每秒，相对现状增益约 `min(W, (t_inf + t_sim + t_rtt) / t_inf)`；另一半收益是资源：现在 h100 被 RAM 卡在 7 个 GR00T 进程（7 × 26 GB RSS），wls 被显存卡在 6 个（6 × 6.4 GB），一个生产进程服务 W 个 worker 只付一份 RSS / 显存。RoboCasa 仿真重（RC GR00T 一集约 1 min，推理只占一小部分），增益大于 LIBERO。**t_inf 与多 worker 吞吐尚未实测**，是迁移 GR00T LIBERO 前的必做项（1 任务 × 若干集 × 3 臂、W ∈ {1, 4, 8}，读 `dump_metrics` 与 ep/min）。
- 旧的 broker 延迟教训仍适用：worker 与 server 同机或走公网直连（`ziyanglin.com:231xx`），不要走 broker。

## 8. 分阶段迁移路径与工作量

| 阶段 | 内容 | 级别 | 工作量 | 前置 / 产出 |
|---|---|---|---|---|
| S0 | 在跑队列不动：`sdq` 跑完 RC GR00T；`sdlq` 的 GR00T LIBERO 暂不开跑或只跑 `full / plain_k*` | — | 0 | 保持现有结果的单一实现 |
| S1 | GR00T 吞吐实测：生产 `serve_groot_libero --concurrent --allow-dynamic-bundles --denoising-steps 8`，入口跑 1 任务 × 2–4 集 × 3 臂，W ∈ {1, 4, 8}；同时补 GR00T K=4 manual 对等（需一个 RC wire 观测） | 运维 | 0.5 天 | 决定 S3 是否值得 |
| S2 | 入口补齐：① `full / plain_k*`（GR00T 每 K 一个进程；π0.5 暂不）② 与 step_diag 队列共存的主机预留（队列读预留文件扣 budget）③ server launcher（起 / 探活 / PID）④ 同计划续跑或多运行合并 ⑤ `plan.json` 补记 `init_pool_sha256` | L1–L2（exp；② 触及 step_diag ops） | 2–3 天 | S1 结论为「值得」 |
| S3 | GR00T LIBERO 全部 29 臂 × 2 env × 10 任务 × 50 集迁入口（7 个已完成 cell 保留或重跑，按整臂整 env 迁，不在臂内混实现） | 运维 | 视吞吐 | S2 |
| S4 | 分析适配器（§5） | L1 | 1 天 | 可与 S2 并行 |
| S5 | RoboCasa 钉物体支持（strategy pin 三键 + agent `--pinned-objects` + 准入 `pin_id`）；RC experiment 改稳定值 | L1 | 0.5–1 天 | 以后的 RC 轮次 |
| S6 | π0.5 `plain_k*`：按 bundle 钉 MISS 步数 | L2 | ~1.5 天 | MetaWorld / 以后的 π0.5 轮次 |
| S7 | 无库自产：A 形状锚库（先用）→ B `trigger: always` 正式特性 | A: exp L1；B: L3 | A 0.5 天；B 2–3 天 | MetaWorld |

## 9. MetaWorld（π0.5 MT50）建议

**现状（本工作树，2026-09-25 12:2x CDT 只读核查）**：checkpoint `/data/ckpt/pi05_metaworld_rlinf` 在（PyTorch `model.safetensors` 7.47 GB，`metadata.pt` 记 `action_horizon 5`、`action_dim 32`、`num_steps 10`、`pi05 True`；`assets/metaworld_mt50/norm_stats.json` 为 4 维 state / action），`~/metaworld_sim`（py3.11，metaworld 3.0.0、mujoco 3.14、`openpi_client` editable，无 torch）在。仓库里**没有**任何 MetaWorld 代码：无 `pi05_metaworld` TrainConfig 与策略变换、无 `ENVS` 条目、无 episode runner / worker、无 serving 路径（`serve_diag_pi05` 只认 `pi05_robocasa / pi05_libero`，`serve_diag_pi05.py:170`）。`metadata.pt` 的 `action_env_dim 7` 与 4 维 norm stats 不一致、`simulator: libero` 疑为 RLinf 默认值，接线时要核对。

**两个选项的对比**：

| | step_diag | 新框架（生产并发 server + conductor） |
|---|---|---|
| MetaWorld 接线（策略配置、runner / worker、任务清单） | 需要 | 需要（同一份工作；conductor `EpisodeRunner` 是可复用形态） |
| `full` | 原生（`--mode full`） | yaml `always_skip`（K=10），入口需接受 full |
| `plain_k2` | 原生（`--exec-steps 2`） | 需 S6（按 bundle 钉 MISS 步数），或临时专用进程 |
| 3 个自产臂（无库） | 同样被卡（自产要 `--cache-config`） | 同样被卡 |
| 显存 | 一臂一进程：5 × 7.8 GB ≈ 39 GB，wls 目前只剩约 12 GB，要么等队列、要么上 h100 | 一个并发进程（~8–9 GB）服务 5 个臂 |
| 吞吐 | B=1 单连接 | 合批，3–5 倍（§7） |
| 证据 | step_diag 行 + 现有分析 | 服务端 JSONL + `admit` + 需 §5 适配器 |
| 历史配对 | 无（新 benchmark） | 无（新 benchmark），不存在混池问题 |

**建议：建在新框架上。** 理由：没有历史数据要对齐，框架一致性只要求 5 个臂同框架；接线工作两边相同；显存与吞吐决定性地偏向一个并发进程；自产臂在两边都要补无库路径。落地顺序：

1. MetaWorld 接线：`pi05_metaworld` TrainConfig + 策略变换（4 维 state / action 与 32 维模型动作的 pad / 截断、相机键）、conductor `EpisodeRunner` + `worker_entry`（`~/metaworld_sim`）、MT50 任务清单与 seed / init 协议、入口的 env 表条目（建议给入口自己的 env 表，不再依赖 `exp.step_diag.envs`，避免改在跑实验的模块）。
2. 无库自产：先用 §3.2 A 形状锚库（H=5、D=32、K=10、`start_t: 0.2`），冒烟核对 `self_direct_nfe == 10`、`continuation_nfe` = 2（selfwarmreset / selfresetfinal，reset 1.0，remaining）与 selfmidfinal（reset 0.9）；线要长期保留再走 B。
3. `full` 走 `always_skip` yaml；`plain_k2` 走 S6。若 S6 来不及，`plain_k2` 可临时由一个专用进程服务（进程级 K），但它与其余 4 臂的合批状态不同，报告里须注明。
4. `admit` 的 worker 步距检查假设 decision 间隔 = replan；MetaWorld 的 action_horizon 5 与 replan 取值要在 runner 里对齐。

## 10. 不做 / 风险

- 不改 `exp/step_diag/**`、不同步岛树、不动在跑队列；本调研零代码改动。
- 共存风险：入口的 server 对 step_diag 队列不可见，任何在 wls / h100 上与队列并行的入口运行都要先有 S2②的预留或整机划分，否则会把队列按 budget 起的 server 挤到 OOM。
- GR00T K=4 的真模型逐位声明仍缺证据（S1 补）。
- 形状锚库是权宜之计：`hit_type` / `winner_id` 的占位含义必须在实验文档与结果里写明，不能被读成「检索命中」。
