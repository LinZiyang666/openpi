# Warm reset 续跑族一等公民化（生产 cache / serving / conductor 接入）— 计划

> 状态：`Verified`（2026-09-24 立项；owner 当日给出需求与三条追加约束 §1.2；**G1 APPROVED 2026-09-25 00:50 CDT**（Round 1 / Round 2 修订后放行，G1 Review Log 已按执行法 §3.1 定稿时删除）→ Code → **G2 APPROVED 2026-09-25 11:58 CDT** → §6 Verify 与按需 GPU 对等、serving 冒烟完成（§15，2026-09-25 13:05 CDT）→ owner 2026-09-25 指示分次提交并推送（框架 / 实验入口 / 记录三次提交））。级别 **L3**（跨 `src/openpi/cache`、`src/openpi/serving`、GR00T cache 路径、serving 入口与证据链；新增子系统 + 修改公共接口）。Authority: Execution。
> 上位：step_diag 线（[`step_vs_warmstart_diagnostics_plan.log.md`](step_vs_warmstart_diagnostics_plan.log.md)、[`step_diag_libero_selfstart_plan.log.md`](step_diag_libero_selfstart_plan.log.md)）；owner 2026-09-24 需求及同日三条追加约束（§1.2）。
> 本文件引用的 `file:line` 均为 HEAD `5483183` 工作树实读核验结果。

## 0. 摘要（供无对话上下文的 G1 审查方）

step_diag 的 warm reset 续跑族（(T, N, t) 变体：`warmreset` / `resetfinal` / `midfinal(50)` / `midreset(50)` / `warmshoot` / `midshoot(50)` 及其 `self*` 自产起点版本）目前只存在于实验层：`exp/step_diag` 以猴补丁改写 `model.run_stage3_from` / `runner.run_stage3_from`，每个臂一个进程、一个连接（π0.5 `--non-concurrent`；GR00T RC 非并发；GR00T LIBERO 虽并发但 launcher 每端点 1 worker）。本计划把它做成生产路径的一等功能：

1. **配置**：yaml 新增可选顶层块 `warm_reset:`（缺省 = 不存在 = 今天的精确续跑，逐位不变）。它只描述「WARM_START 判决之后 stage 3 怎么续」——起点来源（缓存快照 / 缓存最终动作 / 自产快照 / 自产最终动作）、t 网格（reset 到 entry_t，或 shoot 不重置）、步数 N。judge、检索、key builder、backend、orchestrator、conductor、worker 一概不感知。
2. **计划解析 → stage-3 执行**：完全镜像现有 WARM_START 链（判决 `start_t` → orchestrator 校验 payload → interceptor 取 `(start_x, start_t, num_steps)` → `run_stage3_from` / `Stage3WarmStartPayload` → 合批 `("warm_start", start_t, num_steps)`）：interceptor 在同一分支里解析出冻结的 `WarmResetPlan` → 调用与 `run_stage3_from` 并列的续跑入口 → coordinator 走与 `Stage3WarmStartPayload` 并列的 `Stage3WarmResetPayload`，按网格键 `grid_key` 合批。新入口自己承担旧入口 `run_stage3_from` 内的运行时守卫（GR00T：autocast session、库 schedule = plan schedule = 活 head schedule；π0.5：plan / payload 的 K 与 `PI05_V1` 一致），且全部检查先于任何去噪调用（G1 R1 B1）。
3. **证据**：自产直推与续跑都由 warm reset 自有的循环执行，步数由循环内的调用计数器**实测**（不回填预算），经合批 adapter 按请求回传（G1 R1 B2）；执行体把实测的续跑步数、续跑 / 自产调用次数、逐步 t、自产 seed 与直推步数放进 `__hit_meta__["warm_reset"]`；一个外层证据 wrapper 把每个决策写成服务端 JSONL，每集以一条 `finalize` 闭合记录收尾；纯函数准入检查器只接受**受信的期望输入**（worker per_step 给出的期望决策数、journal 的 accepted 终局、下发的 yaml / 规格身份），复刻 step_diag `cell_admission` 的完整性与三条规则（等续跑 NFE、自产证明、K+N 计价），缺尾 / 缺中间 / 重复 / 终局不一致一律拒收（G1 R1 B3）。worker 不改。
4. **范围**：π0.5（K=10）直连与并发合批全支持；GR00T（RoboCasa K=4、LIBERO K=8）支持非并发与并发 infer-lock 多连接、多 bundle；**GR00T 非 trace 路径本来就没有合批 coordinator**（§3.4-F1），v1 不新建。
5. **对等性**：对 step_diag 的每个臂（π0.5 K=10、GR00T RC K=4、GR00T LIBERO K=8，含全部自产臂）在 B=1 上逐位复现（π0.5 为 float32 t 序列 + 输出；GR00T 为 (bucket, dt) 序列 + 输出）。B>1 并发只验收路由 / 身份 / 计数并记录数值偏差，不作逐位声明；真实模型的逐位声明以 manual GPU 对等证据为前提（G1 R1 N1）。

## 1. 目标与约束

### 1.1 目标（owner 2026-09-24）

让 step_diag 的 warm reset (T, N, t) 续跑族成为主框架（cache / serving / conductor）的一等公民：这些臂能跑在生产并发 server（`scripts/serve_policy.py` 并发模式、一个端点多 bundle 按 yaml 选择、`BatchingCoordinator` 合批）与 conductor（`ConductorDriver` / `WorkerAgent`、一台 server 多 worker、`run_size_eval` 式「一臂 = 一 yaml」）上，而不是实验专用的一连接一进程 server。

### 1.2 owner 追加约束（2026-09-24，绑定）

1. warm reset 族**必须同时**把两种起点来源作为一等选项：缓存起点（检索条目的快照 / 最终动作）与自产起点（本决策上的一次直接完整推理，取其 start_t 快照或最终动作，私有确定性 seed）。每个缓存 warm reset 变体都有其自产对应臂，与 `exp/step_diag`（`SELF_VARIANT_MODES` / `SELF_SHOOT_MODES`）完全一致。
2. **与现有 WARM_START（精确续跑）相同的接入方式**：沿判决 `start_t` → orchestrator / interceptor → stage-3 入口 → batching coordinator 的既有结构镜像（判决 / 配置 → 续跑规格 → stage-3 执行），不另造平行通路。
3. 遵循既有设计逻辑：对其他组件解耦且透明；新行为经 interceptor / wrapper / hook / strategy 对象接入（WA §2.5、§3.1）；judge、检索、key builder、backend、conductor、worker、现有 yaml 不感知且不改；规格缺省时每条代码路径与今天逐位相同。计划须写明哪些组件不动、以什么测试保证（本计划 §4.9、§9.3）。

### 1.3 记法（owner）

- **T**：起点动作所处的流时间（π0.5 约定：1 = 噪声、0 = 干净；GR00T 原生时间 τ = 1 − t）。缓存快照臂 T = 判决 `start_t`（π0.5）或 1 − `start_t`（GR00T 原生）；最终动作臂 T = 0。
- **N**：实际执行的 Euler 步数（续跑 NFE）。
- **t**：逐步喂给模型的流时间序列（本文一律按 π0.5 约定书写，GR00T 另注原生 τ 与 bucket）。
- 只有「从缓存快照按原网格精确续跑」叫 **warm start**（我们的方法，yaml 无 `warm_reset` 块）；其余一律叫 **warm reset**。

## 2. 范围与非目标

**范围内**

- `warm_reset` yaml 块的 schema、解析、静态校验（`src/openpi/cache/config.py`）。
- 模型无关的计划解析、自产 seed 策略、会话状态、证据 wrapper 与准入检查器（新包 `src/openpi/cache/warm_reset/`）。
- π0.5 执行体：直连路径 + coordinator 路径（新 payload 类型 + `Pi05StageBatcher` 分支）。
- GR00T 执行体：`GrootCacheInterceptor` 的 WARM_START 分支（非并发与并发 infer-lock 两种 serving 形态，含 `--allow-dynamic-bundles`）。
- 装配：`scripts/serve_policy.py::_wrap_policy`、`exp/robocasa365/serve_groot_n15.py`、`exp/libero_groot/serve_groot_libero.py` 的缓存栈构造点（后两者是生产实验实际使用的 GR00T serving 入口，住在 exp/ 属历史布局）。
- 对等性测试、隔离测试、架构文档。

**非目标（本计划不做）**

- 不改 `exp/step_diag/**`（在跑实验 `sdq` / `sdlq` 的代码与对等性参考实现，测试只读导入）。
- 不改 judge / gate / search strategy / key builder / backend / `CacheStorage` / `CacheOrchestrator` / trace 包 / conductor / worker（`examples/libero/episode_runner.py`、`exp/robocasa365/episode_runner.py`）/ `WebsocketPolicyServer` / `replica_proxy` / `pi0_pytorch.py` / `groot/staged.py` / `groot/batcher.py`。
- 不给 GR00T 非 trace 路径新建合批 coordinator（§3.4-F1，开放问题 Q3）。
- 不支持 `warm_reset` 与 trace 模式、CP2-only 配置、`online_rit` judge、routing（sidecar 执行器）、X15 `shadow_teacher` 同时使用（v1 在加载期与构造期都响亮拒绝，§4.2.3）。
- 不做 step_diag 臂的 exp 侧 conductor strategy、arm→yaml emitter、统计分析与正式轮排程（后续 exp 计划，开放问题 Q6）；本计划只保证其所需的机制与证据齐备。
- 等 NFE 参照臂 `plain_k<k>`（减步 MISS）不在 warm reset 族内，生产 serving 目前也没有按 bundle 钉 MISS 步数的机制；`full` 可用 `always_skip` gate 的 yaml 表达（开放问题 Q9）。

## 3. 现状核验

### 3.1 精确 WARM_START 在生产路径上的流向（π0.5）

| 段 | 位置 | 行为 |
|---|---|---|
| 判决 | `always_warm_start` judge；`config.py:3057-3091` 校验 `start_t` ∈ schedule 可恢复点并 round4 回写 | 每步 WARM_START，携带 `start_t` |
| orchestrator | `orchestrator.py:1240-1311` | 取 winner payload，按 `artifact_meta.schedule_id` / `payload.schedule_id` 解析 `schedule_from_id`，`payload.validate_for_warm_start(schedule, start_t)`，返回 `CheckResult(hit_type, payload, start_t, score, entry_id, ...)`（`orchestrator.py:126-166`） |
| interceptor | `interceptor.py:1883-1907` | `validate_for_warm_start(PI05_V1, start_t)`；`start_x = payload.intermediates[start_t].to(stage3_device)[None]`；`_run_stage3_from = self._stage3_from_fn or self._model.run_stage3_from`；在 `stage3_warm` probe 内调用 `(stage2, start_x, start_t, num_steps=payload.denoising_num_steps)` |
| stage-3 入口 | `pi0_pytorch.py:704-769` `run_stage3_from(stage2, start_x, start_t, *, num_steps=10) -> Stage3Output` | `n = _warm_start_num_steps(start_t, num_steps)`（`pi0_pytorch.py:185-198`，`floor(start_t*K+0.5)`）；float32 设备张量上重放 `K−n` 次 `timestep += -1/K` 作为起点；n 步 `x += dt*v`，`dt = -1/K` |
| coordinator 绑定 | `interceptor.py:438-446` + `_make_warm_start_via_coordinator` `interceptor.py:1014-1034` | 构造 `Stage3WarmStartPayload(stage2_out, start_x[H,D], start_t, num_steps)` 交 `submit_to_stage(3, bundle_id, ...)` |
| payload / 合批 | `batching_core.py:108-122`；`Pi05StageBatcher.bucket_key` `batching_coordinator.py:92-98` → `("warm_start", start_t, num_steps)`；`_group_stage3_requests` `batching_core.py:1080-1094` 只接受两种 payload；`_run_stage3_bucket` `batching_core.py:1107-1128` 非 MISS 即 `run_stage3_warm` | `Pi05StageBatcher.run_stage3_warm` `batching_coordinator.py:182-202` 堆叠后调 `model.run_stage3_from` |
| wire | `_build_hit_meta` `interceptor.py:861-939`；尾部调用 `interceptor.py:2039-2044` | `hit_type / start_t / winner_id / cp1_score / checkpoint / score / searched`（+ 可选加法字段） |

GR00T 对应链：`GrootCacheInterceptor._get_action_impl` `groot/interceptor.py:373-497`，WARM_START 分支 `:422-449` 取 `schedule = self._library_schedule(payload)`（`:1165-1191`，库 stamp 与条目 `denoising_num_steps` 必须一致），在 `runner.session()` 内 `run_stage2_llm` 后调 `runner.run_stage3_from(stage2, payload.intermediates[start_t], start_t, schedule=schedule)`（`groot/staged.py:843-908`：`schedule != live_schedule()` 即拒；`denoise_loop(noise=start_x, num_steps=K, start_index=snapshot_index(start_t))`，`steps_run = remaining_steps(start_t)`）。上游循环转写 `denoise_loop` `groot/staged.py:1002-1055`：τ_i = `i/float(N)`，bucket = `int(τ_i * num_timestep_buckets)`，`dt = 1.0/N`，每步 `step_fn(...).clone()`。

### 3.2 step_diag 参考实现（必须逐位复现的语义）

- 臂表与常量：`exp/step_diag/envs.py:88-90` `WARM_VARIANT_MODES`；`:96` `MID_ENTRY_T = {"pi05": 0.9, "groot": 0.75}`；`:100-101` `MID_ENTRY_T_BY_VARIANT`（mid50 = 0.5）；`:110` `SELF_VARIANT_MODES`（无 `selfwarmshoot`）；`:115` `SHOOT_ENTRY_T = {"overshoot": 1.0, "mid_shoot": 0.75, "mid_shoot50": 0.5}`；`:116-117` `GROOT_SHOOT_MODES` / `SELF_SHOOT_MODES`；`:126-138` `split_warm_steps` / `warm_steps_of`（GR00T LIBERO `_n<N>` 把 N 与快照解耦）；`:221-227` `MACRO13_ARMS_BY_POLICY`；`:232-241` `SELF13_ARMS_BY_POLICY`；`:255` `GROOT_LIBERO_START_T_BY_N = {1: 0.75, 2: 0.5}`；`:259-264` `LIBERO_SELF_ARMS_BY_POLICY`。
- π0.5：`exp/step_diag/pi05.py:72-125` `warm_variant_stage3`：`n = _warm_start_num_steps(start_t, K)`；reset 类 `timestep = tensor(1.0)`、`dt = tensor(-1.0/n)`；mid 类 `timestep = tensor(entry)`、`dt = tensor(-entry/n)`；overshoot `timestep` = 从 1.0 累加 `K−n` 次 `tensor(-1.0/K)`、`dt = tensor(-1.0/n)`；每步经 `model.denoise_step`。`Pi05DiagInterceptor`（`:128-387`）以实例猴补丁把执行路径的 `run_stage3_from` 换成变体（`:195-207`）；最终动作起点从 `orch._storage.fetch_payload(entry_id).action_chunk` 取并 `.to(device, dtype).reshape(like)`（`:361-372`）；自产起点 `_self_start`（`:333-359`）：`make_noise(seed, (H,D))[None].to(like.device)`，`model.run_stage3(stage2, noise=, num_steps=K, return_intermediates=True, save_timesteps=(start_t,))`，取 `intermediates[start_t]` 或 `action_chunk`，直推步数从 `executed_steps` 扣除并记为 `self_direct_nfe`。
- GR00T：`exp/step_diag/groot.py:132-182` `groot_warm_variant_stage3`：`n = remaining_steps(start_t)` 或显式 `num_steps`；reset/resetfinal → `_staged.denoise_loop(noise=start, num_steps=n, start_index=0)`；mid 类 → `mid_denoise_loop(entry_t=1.0 − MID…["groot"])`（`:86-107`：`dt = (1.0 − entry_t)/n`，τ_i = `entry_t + i*dt`）；shoot 类 → `shoot_denoise_loop(t0=float(start_t), dt=SHOOT_ENTRY_T/n)`（`:110-129`，τ_i = `t0 + i*dt`）；外包 `runner._timer.measure("stage3_warm")` 与 `validate_data`，返回 `GrootStage3Output(steps_run=n)`。`groot_self_start`（`:60-83`）：`make_noise(seed)` float32 → `denoise_loop(num_steps=K, start_index=0, on_step=…)` 在 `snapshot_index(start_t)` 处抓 `x_in`，或取最终动作。`install_warm_variant`（`:185-229`）换掉 `runner.run_stage3_from`；最终动作起点取 `cp1.payload.action_chunk`。
- seed 与证据：`exp/step_diag/recorder.py:56-59` `stable_digest_int`（`"|".join(str(p))` 的 sha256 前 8 字节 little-endian & 0x7FFF…）；`:70-73` `noise_seed(experiment_id, env_id, task, env_seed, attempt, decision_idx, sample)`；`:76-80` `make_noise`（CPU 私有 `torch.Generator`，float32）；`:339-348` `DiagSession.self_start_seed` = `noise_seed(..., (env_seed, init_idx, init_pool_sha256), attempt, decision_idx, "self")`。
- 准入：`exp/step_diag/analysis/aggregate_arms.py:207-216` `expected_self_seed`（从**期望身份**重算，不从被检行）；`:219-235` `self_start_problems`（`self_start is True`、seed 相等、`self_direct_nfe == K`；缓存臂不得带三字段）；`:238-397` `cell_admission`（决策 idx 连续、`status ok`、`schedule_id` 与 `start_t` 符合臂、`hit_type == WARM_START`、`executed_steps == m`、`n_stage3_calls == 1`、`episode_{continuation,self_start,total}_nfe`）。

### 3.3 并发 serving 与 conductor 现状

- π0.5 并发：`serve_policy.py:998-1059` 每进程一个 `BatchingCoordinator`（`:1021-1038`），每连接由 `_connection_policy_factory` → `_wrap_policy(..., eager=True, shared_cache=..., bundle_id=...)`（`:549-765`）构造独立 `InferenceInterceptor`；bundle 分支 `:596-663`（`get_current_cache_bundle(bundle_id)`），启动 yaml 分支 `:664-737`。`WebsocketPolicyServer` 在 `select_bundle` / 首个 `episode_start{bundle_id}` 时懒绑定，换 bundle 即重建整栈并对旧栈 `on_task_end`（`websocket_policy_server.py:600-626`）；`load_cache_config` ctrl 走通用 `load_cache_config` 校验并建共享 storage（`:747-857`，`:790`）。同 `preload_path` 的多臂经 `BackendPool` 共享一份库（`config.py:4232-4250`）。
- conductor：一臂 = 一 yaml，`PureCacheEvalStrategy.on_stage_begin` 以 `ctl.load_cache_config(yaml_content, yaml_id, bundle_id)` 下发（`exp/ablation_study/cache_size/run_size_eval.py:63-109`，`on_stage_begin` `:102-109`），worker 按 `EpisodeTask.bundle_id`（`src/openpi/conductor/task.py:94`）`select_bundle`；RoboCasa 侧同构 `Ws2ArmStrategy`（`exp/robocasa365/run_ws_search2.py:139`，`:190-195`）对 `serve_groot_n15 --concurrent --allow-dynamic-bundles`。注意 `validate_pure_cache_arms` 强制 `judge.type == always_hit`（`run_size_eval.py:112-131`），不能直接驱动 `always_warm_start` 臂。
- GR00T 并发：`serve_groot_n15.py:502-628` / `serve_groot_libero.py:495-678`，非 trace 时每连接 `_InferLockedPolicy(Adapter(GrootCacheInterceptor(...)), lock)`（`serve_groot_n15.py:399-425`：整段 `infer` 进程级锁串行）；**只有 trace 模式**才起 `BatchingCore(GrootStageBatcher)`（`serve_groot_n15.py:258-281`、`serve_groot_libero.py:469-492`）。
- per_step：worker 端按**白名单**从 `__hit_meta__` 抄字段——LIBERO `_hit_row`（`examples/libero/episode_runner.py:58-112`），RoboCasa 只抄 5 个字段（`exp/robocasa365/episode_runner.py:593-610`）。`episode_start` 元数据：LIBERO conductor `{task_id, orig_init_state_idx, [run_id,batch_id,weights_version], task_uid, attempt}`（`episode_runner.py:114-142`、`:217-222`），`experiment = task.experiment`、`task` = 语言描述；RoboCasa `{task_uid, attempt, task_id, orig_init_state_idx, seed, …pin/layout}`（`exp/robocasa365/episode_runner.py:559-572`）。
- 现状容量：每个 step_diag 服务进程 π0.5 7.8 GB 显存 + 31 GB RSS、GR00T 6.4 GB + 21 GB（`exp/step_diag/ops/self13_queue.py:49` 实测）；主机预算 wls 44 GB / 220 GB、h100 76 GB / 175 GB（`:53`、`:61`）。

### 3.4 使集成比表面更难的发现

- **F1 GR00T 非 trace 路径没有合批**：生产 GR00T 并发 = 多连接 + 进程级 infer 锁串行；coordinator 只在 trace 模式存在，且 trace 每步跑 full + 全部 warm 档，不能当吞吐方案。⇒ GR00T 臂上生产 server 的收益来自「一进程多连接多 bundle + 仿真与推理重叠 + RSS 摊薄」，不是合批（§10.4）。给非 trace GR00T 加合批要动整段 infer 锁契约（视觉塔 CUDA-graph 静态缓冲、transform 链非线程安全，见 `docs/architecture/cache_system.md` §5.17），属另一个 L3（Q3）。
- **F2 worker 的 per_step 是白名单**：任何新加在 `__hit_meta__` 的字段都会被两个标准 runner 丢掉。owner 要求 worker 不改 ⇒ 权威证据必须在**服务端**落盘，按 `(task_uid, attempt, decision_idx)` 与 journal / per_step 对齐（§4.7）。
- **F3 旧 server 静默忽略未知 yaml 键**：`_dict_to_dataclass` 对未知键只 `logger.warning` 后跳过（`config.py:1070-1074`）。一个 warm reset 臂的 yaml 若被未更新代码的 server（例如未同步的岛树）加载，会**静默跑成精确续跑**且 `hit_meta` 完全正常。⇒ 准入必须要求每个决策的正向 `warm_reset` 证据（含 `spec_digest`），不能只看 `hit_type`（R1）。
- **F4 名字冲突**：`continuation_spec` / `ContinuationSpec` 已被 online RIT 占用（`orchestrator.py:749-752`、`components/online_rit.py:653`、GR00T interceptor 多处 `hasattr(orch, "continuation_spec")`）。⇒ 新概念统一命名 `warm_reset`，绝不使用 `continuation_spec`。
- **F5 逐位对等依赖浮点表达式顺序**：π0.5 的 t 是设备 float32 累加（shoot 起点须按 `run_stage3_from` 的方式重放）；GR00T mid 的 `dt` 须按 `(1.0 − (1.0 − entry))/N` 计算、τ_i 用 `τ0 + i*dt`；而 entry = 1 的 reset 在 step_diag 里走的是上游转写 `denoise_loop`（τ_i = `i/float(N)`），两式对一般 N 不保证逐位相同 ⇒ 执行体必须按 kind 分派到与 step_diag 相同的三条循环（§4.3.3）。
- **F6 B>1 合批数值 ≠ B=1**：批大小不同的 GEMM 不逐位一致（现有精确 WARM_START 同理）。⇒ 对等性在 B=1 上逐位成立；并发合批产出的闭环结果不得与 step_diag（B=1）cell 混池（R3、Q8）。
- **F7 自产 seed 无法在不改 worker 的前提下等于 step_diag 的 seed**：`noise_seed` 身份含 `experiment_id` / `env_id`（DiagSpec，server CLI）与 `(env_seed, init_idx, init_pool_sha256)`；LIBERO 标准 runner 不发 env seed 与池 SHA。⇒ 生产 seed 用可配置的「命名空间 + episode_start 身份键」配方；与 step_diag 的逐位对等在「同一 seed 整数」前提下验证（Q2）。
- **F8 coordinator 子类陷阱**：若新 payload 继承 `Stage3WarmStartPayload`，任何不认识它的 adapter（如 `GrootStageBatcher.bucket_key` `groot/batcher.py:248-278`、`_run_stage3_bucket` 的非 MISS 即 warm 分派）会**静默当精确续跑执行**。⇒ 新 payload 必须是**并列类型**，核心分派对不支持的 adapter 响亮失败（§4.5.2）。
- **F9 合批栈叠的设备一致性**：adapter 对桶内请求逐行 `torch.stack` 后 `.to(device)`（参照 `Pi05StageBatcher.run_stage3_miss` `batching_coordinator.py:148`）；同桶其他请求的输入在 stage-3 设备上，私有噪声若留在 CPU 会在栈叠时设备不一致而失败 ⇒ 自产噪声提交前先 `.to(stage3_device)`（G1 R1 修订后自产直推走 `Stage3WarmResetPayload` 而非 MISS payload，约束不变）。
- **F10 trace 可由 CLI 单独开启**（`--trace-out`，`serve_policy.py:474-480` `validate_effective_trace`）⇒ 只在 yaml 校验里拒绝不够，interceptor 构造期也要拒绝。
- **F11 GR00T LIBERO adapter 不转发 `on_task_end`**（仅 trace 时挂 `close_trace_episode`，`exp/libero_groot/policy_adapter.py:161-162`）⇒ 连接中途断开时该连接未结束 episode 的证据行留在缓冲区丢失（R7）。
- **F12 `run_size_eval` 不能原样驱动 warm 臂**（`always_hit` 强制，§3.3）⇒ conductor 侧需要一个 exp strategy（后续计划，Q6）；conductor 核心本身无需改。
- **F13 GR00T 运行时守卫住在旧入口里面**（G1 R1 B1）：`_library_schedule`（`groot/interceptor.py:1165-1191`）只比较库 stamp 与条目 K；与活 head 的比较（`schedule != self.live_schedule()`）和 autocast session 断言 `_require_session`（`groot/staged.py:447-456`）都在 `run_stage3_from` 内（`:865-872`）。直接调用 `denoise_loop` / 自写循环会绕过这两道守卫；`runner.run_stage3(noise=...)` 虽断言 session（`:810`），却用 `head.num_inference_timesteps` 而不接收库 schedule。⇒ 新入口必须自带这两道守卫（§4.3.3、§4.6）。
- **F14 π0.5 完整推理接口不返回执行步数**（G1 R1 B2）：`Stage3Output` 只有 `action_chunk` / `intermediates`（`pi0_pytorch.py:98-110`），`run_stage3`（`:644-702`）与 `Pi05StageBatcher.run_stage3_miss`（`batching_coordinator.py:140-180`）拆包后都没有计数；step_diag 的 `self_direct_nfe` 是包装 `model.denoise_step` 实例属性、按线程计出来的（`exp/step_diag/pi05.py:173-181`、`:333-359`），不是回填预算。合批时计数又只能按「循环迭代数」归属每行，不能把整桶 B×K 记给单请求。改旧 MISS 接口返回计数会改变缺块旧路径 ⇒ 自产直推改走 warm reset 自有的计数循环（§4.4、§4.5.2）。
- **F15 逐行盖 `success` 不是完整性标记**（G1 R1 B3）：`PerStepWriter.flush_episode`（`per_step_recorder.py:105-123`）只给缓冲行盖 `success` 再写出，丢掉尾部若干行后剩余行仍「连续、终局」。完整性必须由独立来源给出：driver 已给每条 per_step 行盖 `task_uid` / `attempt` / `accepted` / `yaml_id` / `run_id`（`src/openpi/conductor/driver.py:338-358`），据此可在 exp 侧得到每个 (task_uid, attempt) 的期望决策数 ⇒ 检查器接受受信期望输入，服务端另写 `finalize` 闭合记录（§4.7）。

## 4. 设计

### 4.1 总览：镜像 WARM_START 的三段

| 段 | 精确 WARM_START（不变） | warm reset（新增，并列） |
|---|---|---|
| 配置 | judge `start_t`（+ `denoise_schedule`） | 同一 judge 不变 + 顶层 `warm_reset:` 块（`WarmResetConfig`） |
| 判决 | `always_warm_start` / `warm_tiers` 等 → WARM_START(`start_t`) | **完全相同**（judge 不感知） |
| orchestrator | 取 payload、`validate_for_warm_start(schedule, start_t)` | **完全相同**（orchestrator 不感知；快照在 `start_t` 必须存在，最终动作 / 自产臂亦然，与 step_diag 一致） |
| interceptor 取参 | `(start_x, start_t, num_steps=payload.denoising_num_steps)` | 同一分支：`plan = resolve_plan(spec, schedule, start_t)`（冻结、可哈希） + 起点 `start_x`（缓存快照 / 缓存最终动作 / 自产快照 / 自产最终动作） |
| 运行时守卫 | 旧入口内：GR00T `_require_session` + 库 schedule = 活 schedule（`groot/staged.py:865-872`）；π0.5 `validate_for_warm_start(PI05_V1, start_t)` | 新入口内自带同等守卫并先于任何去噪调用（§4.3.2、§4.3.3） |
| stage-3 入口 | `model.run_stage3_from` / `runner.run_stage3_from` | 续跑 `run_pi05_continuation(model, stage2, x, plan)` / `run_groot_continuation(runner, stage2, x, plan, schedule=)`；自产直推 `run_pi05_self_start(model, stage2, noise, plans)`（逐行列表；直连传 `[self_plan]`）/ `groot_self_start(runner, stage2, noise, self_plan, schedule=)`；全部返回实测 steps_run |
| coordinator 绑定 | `_stage3_from_fn = _make_warm_start_via_coordinator(...)` | `_stage3_warm_reset_fn = _make_warm_reset_via_coordinator(...)`（仅块存在时构造；续跑与自产直推共用） |
| payload / 合批键 | `Stage3WarmStartPayload` → `("warm_start", start_t, num_steps)` | `Stage3WarmResetPayload` → 续跑 `("warm_reset", plan.grid_key())`、自产直推 `("warm_reset_self", self_plan.grid_key())` |
| adapter | `Pi05StageBatcher.run_stage3_warm` → `run_stage3_from` | `Pi05StageBatcher.run_stage3_warm_reset` → `run_pi05_continuation` / `run_pi05_self_start`（逐行回传实测步数） |
| wire | `__hit_meta__`（旧字段） | 旧字段不变 + 加法字段 `__hit_meta__["warm_reset"]`（仅块存在且本步执行了续跑时出现） |

新增的只有一个观测层：证据 wrapper（与 `PolicyRecorder`、`_InferLockedPolicy` 同类的外层包装），负责会话身份、决策序号、服务端 JSONL 与每集 `finalize` 闭合记录；它不参与执行。

### 4.2 yaml schema：`warm_reset:` 块

#### 4.2.1 结构

```yaml
warm_reset:
  start:
    source: cache        # cache | self       （必填）
    point: snapshot      # snapshot | final   （必填）snapshot = 判决 start_t 处的快照；final = 最终动作 (T = 0)
  grid:
    kind: reset          # reset | shoot      （必填）
    entry_t: 1.0         # reset 必填：起点被声明所处的流时间（π0.5 约定，所有 schedule 通用），∈ (0, 1]
    # step_budget: 1.0   # shoot 必填：dt = step_budget / N（π0.5 约定），∈ (0, 1]；起点留在自身 T，不重置
  num_steps: remaining   # remaining | 整数 N（1 ≤ N ≤ K）；remaining = schedule.remaining_steps(start_t)
  self_seed:             # 仅 source: self（必填）
    namespace: sdiag_libero_self_prod     # 实验级命名空间；同一实验的各自产臂必须相同（共同随机数配对）
    identity_keys: [experiment, task, orig_init_state_idx, attempt]   # 缺省即此列表
  evidence_dir: /data/warm_reset_evidence/<run>   # 必填：服务端证据 JSONL 目录（server 本地路径）
```

数据类（`config.py` 新增并登记进 `_CONFIG_TYPES` `config.py:959-996`，顶层挂 `CacheConfig.warm_reset: Optional[WarmResetConfig] = None`，与 `routing: Optional[RoutingConfig] = None` 同形）：`WarmResetStartConfig(source, point)`、`WarmResetGridConfig(kind, entry_t=None, step_budget=None)`、`WarmResetSelfSeedConfig(namespace="", identity_keys=[...缺省...])`、`WarmResetConfig(start, grid, num_steps="remaining", self_seed=None, evidence_dir="")`。必填项在数据类里以 `None` / 空串为缺省，由校验器报缺（与 `ShadowTeacherConfig.path` 同一做法，`config.py:736-753`、`:2093-2098`），不让隐式缺省决定实验语义。

#### 4.2.2 数值约定

- `entry_t` / `step_budget` 一律按 π0.5 约定书写（1 = 噪声），与 owner 的 T/N/t 记法一致；GR00T 执行体换算为原生 τ（§4.3.3）。同一 yaml 里 judge 的 `start_t` 仍是 schedule 原生时间（GR00T 0.75 = T 0.25）——两种约定并存是既有事实，文档与示例逐字标注（Q1）。
- 不存在 `kind: exact`：精确续跑 = 不写 `warm_reset` 块。这保证「块缺省 ⇔ 今天的精确路径」这一条性质不可能被一个等价配置绕开。

#### 4.2.3 静态校验（`validate_cache_config` 内新增 `_warm_reset_errors(config, check_files)`）

1. 块存在时，配置必须能发出 WARM_START（复用 `_config_emits_warm_start` `config.py:3675`）；否则是死配置。
2. 拒绝组合：`trace.enabled`、`shadow_teacher.enabled`、`routing` 非空、任一启用 `cp2` 检查点、任一检查点 judge 为 `online_rit`（其反馈契约依赖精确续跑的第一步）。
3. `start.source ∈ {cache, self}`、`start.point ∈ {snapshot, final}`；`grid.kind ∈ {reset, shoot}`；reset 必须且只能给 `entry_t ∈ (0, 1]`，shoot 必须且只能给 `step_budget ∈ (0, 1]`；`point: final` 与 `kind: shoot` 互斥（T = 0 无「自身流时间」可留）。
4. `num_steps` 为 `"remaining"` 或整数 `1 ≤ N ≤ effective_denoise_schedule(config).num_steps`（π0.5 缺省 `PI05_V1` K=10；GR00T warm 配方本就强制写 `denoise_schedule`，`groot/load_guard.py:238`）。
5. `source: self` ⇒ `self_seed.namespace` 非空、`identity_keys` 非空且不含 `task_uid`（它内嵌 `yaml_id`，会让同一决策在不同臂抽到不同噪声，破坏自产臂间的共同随机数配对）；`source: cache` ⇒ 不得写 `self_seed`。
6. `evidence_dir` 非空；`check_files=True` 时最近的已存在祖先目录须可写（`load_cache_config` 与 `load_cache_config` ctrl 都在加载期执行，失败即 error ack，不等到第一个连接）。

### 4.3 计划解析与两模型的 t 网格

所有新入口都验证 plan 的整数类型：K / N / 非 None 的 capture_index 均不接受 bool；批量入口要求 B ≥ 1，逐行 plan 与 batch 维一致。解析与入口守卫均采用同一验证函数，避免 resolver 正确但直接构造 plan 绕过检查。

#### 4.3.1 `WarmResetSpec` / `WarmResetPlan`（`warm_reset/types.py`，jax-free、无模型依赖）

- `WarmResetSpec.from_config(cfg)`：冻结视图（`source, point, kind, level, num_steps|None, seed_namespace, seed_keys, evidence_dir`；`level` = reset 的 `entry_t` 或 shoot 的 `step_budget`）；`digest()` = 规范 JSON 的 sha256（证据与准入用）。
- `resolve_plan(spec, schedule, start_t) -> WarmResetPlan`：**无论 N 是否显式**，先调用 `schedule.snapshot_index(start_t)`（`types.py:132-148`，非可恢复点抛 `ValueError`）——显式 N 不跳过可恢复点检查（G1 R1 B1；orchestrator 的 `validate_for_warm_start` 虽已在 `storage_types.py:152-153` 做过同一检查，公开函数不依赖调用方）；`K = schedule.num_steps`；`N = schedule.remaining_steps(start_t)`（`types.py:150-152`）或显式值（再次核对 `1 ≤ N ≤ K`）。π0.5 的 `remaining_steps` 与 step_diag 所用 `_warm_start_num_steps(start_t, 10)` 对 `PI05_V1` 全部 9 个可恢复点相等（测试逐点钉住）。
- `resolve_self_plan(spec, schedule, start_t) -> SelfStartPlan`（仅 `source: self`）：`SelfStartPlan(schedule_id, direction, k, start_t, capture_index | None)`；先无条件执行 `index = schedule.snapshot_index(start_t)`（final 臂也检查判决可恢复点），再取 `capture_index = index`（`point: snapshot`）或 None（`point: final`）。start_t 保留用于公开入口的独立校验；非 None 的 capture_index 必须等于该映射。`grid_key()` = `(schedule_id, k)`——自产直推网格只由 K 决定，start_t / 抓取位置不进键，因此各自产臂同桶合批；adapter 必须传入全部逐行 plans，不能只传桶首 plan（§4.5.2）。π0.5 的 `snapshot_index(t)` 与 `_stage3_with_intermediates` 所用 `round((1.0 − t) * K)`（`pi0_pytorch.py:795-798`）对 9 个可恢复点相等（测试钉住）。
- `WarmResetPlan`（frozen dataclass）：`schedule_id, direction, k, start_t, source, point, kind, level, n_steps`；`grid_key()` = `(schedule_id, k, kind, level, n_steps, start_t if kind == "shoot" else None)`——网格只由这些量决定：reset 网格与 `start_t` 无关、shoot 网格从 `start_t` 出发；`source` / `point` 不影响网格，因而缓存臂与自产臂、快照臂与最终动作臂在同网格时共享合批桶。键里带 `schedule_id` 与 `k`，满足「GR00T RC(K=4) 与 LIBERO(K=8) 共用代码时主键必须带步数」。
- `WarmResetPlan.flow_times()`：逐步喂给模型的 t（π0.5 约定）的主机侧重放，用于证据；π0.5 以 CPU float32 张量按与设备相同的加法序列重放（IEEE 单次加法与设备无关，测试钉住与桩模型实收的 t 逐位一致），GR00T 为 python float。

#### 4.3.2 π0.5（`DIRECTION_DESC`，K=10）

`run_pi05_continuation(model, stage2, start_x, plan) -> WarmResetStage3Output`（`Stage3Output` 的 dataclass 子类，加 `steps_run: int`；下游只读 `.action_chunk` 与 `getattr(stage3, "intermediates", None)`）：

- 守卫（先于任何 `denoise_step`）：公开入口检查 `plan.schedule_id == PI05_V1.schedule_id`、`plan.direction == DIRECTION_DESC`、`plan.k == PI05_V1.num_steps`，执行 `PI05_V1.snapshot_index(plan.start_t)`，并验证 N 为非 bool 的整数且 `1 ≤ N ≤ K`；否则 `ValueError`。π0.5 没有活步数概念（`interceptor.py:93`）；拥有 payload 的 executor 另在提交前核对 `cp_result.payload.denoising_num_steps == plan.k`，不让公开入口依赖其签名中不存在的 payload。
- 实测计数：循环不直接调用 `model.denoise_step`，而是调用一个**调用内局部**的计数闭包 `step = _CountedStep(model.denoise_step)`（每次调用 +1、原样转发参数与返回值）；`steps_run = step.calls`。不在共享模型实例上打补丁（线程安全，缺块路径不受影响）；循环次数由模块内 `_loop_steps(plan) -> int`（= `plan.n_steps`）给出，测试可在执行层把它注入为 N±1，以验证证据反映实际值（B2）。合批时计数器对整个批的一次循环计数，每行的 `steps_run` = 该行经过的循环迭代数（= 计数器值），不是 B×N。
- reset：`timestep = torch.tensor(level, float32, device)`；`dt = torch.tensor(-level / N, float32, device)`。`level = 1.0` 时与 step_diag 的 `tensor(1.0)` / `tensor(-1.0/n)` 为同一 double → 同一 float32；mid 类与 `tensor(entry)` / `tensor(-entry/n)` 同式。
- shoot：`grid = torch.tensor(-1.0 / K)`，`timestep = torch.tensor(1.0)`，累加 `K − _warm_start_num_steps(start_t, K)` 次 `timestep = timestep + grid`（与 `run_stage3_from` `pi0_pytorch.py:755-756` 及 step_diag overshoot 同一重放；重放次数取快照自身位置，与 N 无关）；`dt = torch.tensor(-level / N)`。
- 循环 `_loop_steps(plan)` 次：`v = step(state, prefix_pad_masks, past_key_values, x, timestep.expand(B))`；`x = x + dt * v`；`timestep = timestep + dt`；`steps_run = step.calls`。`step` 转发到 `model.denoise_step`（实例属性解析，与 step_diag 相同的被调函数），数值与直接调用逐位相同。

`run_pi05_self_start(model, stage2, noise, plans: Sequence[SelfStartPlan]) -> list[Pi05SelfStartOutput]`：noise 为 `[B,H,D]`，`len(plans) == B`，stage2 的 batch 维也为 B；入口在第一次去噪前检查全部 plans 属于 `PI05_V1`（schedule_id / direction / K）、同一 grid_key、start_t 可恢复，且每个非 None 的 capture_index 等于其 `snapshot_index(start_t)`。逐语句复刻 `_stage3_with_intermediates`（`pi0_pytorch.py:771-818`）：`dt = tensor(-1.0 / K)`、`timestep = tensor(1.0)`；循环 `_self_loop_steps(plans[0])` 次，每步调用前为满足 `step_idx == plans[i].capture_index` 的行抓 `x[i:i+1].clone()`，随后整个 batch 调一次计数闭包 step、更新 x 和 timestep。返回按输入顺序排列的 B 个 `Pi05SelfStartOutput(action_chunk=x[i:i+1], snapshot=snapshot_i, steps_run=step.calls)`；final 行 snapshot 为 None，快照行若没有实际抓到指定步则抛错，不能退回 final。每行计数相同，不乘 B。直连绑定以 `[self_plan]` 调用并取结果 `[0]`；合批 adapter 原样传入 `[p.plan for p in payloads]` 并返回结果列表，不再拆一次。所有输出保留单位 batch 维。

#### 4.3.3 GR00T（`DIRECTION_ASC`，K=4 / 8）

`run_groot_continuation(runner, stage2, start_x, plan, *, schedule) -> GrootStage3Output`（`warm_reset/groot.py`；`start_x` 维度 2 时补 batch 维，同 step_diag）。

**入口守卫（F13，G1 R1 B1；全部先于任何 `process_backbone_output` / 去噪调用，任一不满足即 `RuntimeError` / `ValueError`，且不改 `groot/staged.py`）：**

1. autocast session：`runner._require_session("warm_reset_continuation")`（`groot/staged.py:447-456`；与 step_diag 访问 `runner._model` / `runner._head_inputs` / `runner._timer` 同级的包内私有访问，`noqa: SLF001`）。
2. 库 schedule 与活 head：`schedule == runner.live_schedule()`（`:420-422`；消息同 `run_stage3_from` `:867-872`），每次调用时现取，因此装配后活 K 被改动也会被拦下。
3. plan 与库 schedule：`plan.schedule_id == schedule.schedule_id` 且 `plan.k == schedule.num_steps` 且 `plan.direction == DIRECTION_ASC`。
4. 起点可恢复：`schedule.snapshot_index(plan.start_t)`（显式 N、自产 final 亦然）；自产 plan 的 capture_index 为 None（final）或等于该映射的非 bool 整数。

续跑还须验证 N 为非 bool 的整数且 `1 ≤ N ≤ K`。公开入口检查手工构造的 plan，不只依赖 resolver。

实测计数：`step_fn` 传入调用内局部计数闭包 `_CountedStep(staged.denoise_step)`（`denoise_loop` 与 `grid_denoise_loop` 都接受 `step_fn`，`staged.py:1010`），`steps_run = step.calls`，循环次数同样经可注入的 `_loop_steps(plan)`。计数闭包原样转发，数值不变。

按 kind 分派到与 step_diag **相同的三条循环**（F5）：

| kind / 条件 | 循环 | τ_i（原生） | dt（原生） |
|---|---|---|---|
| reset，`level == 1.0` | `openpi.cache.groot.staged.denoise_loop(head, runner._head_inputs(stage2), stage2.action_inputs, noise=start_x, num_steps=N, start_index=0)`（上游转写本身） | `i / float(N)` | `1.0 / N` |
| reset，`level < 1.0` | `grid_denoise_loop(..., tau0=1.0 − level, dt=(1.0 − tau0) / N)` | `tau0 + i*dt` | `(1.0 − (1.0 − level)) / N`（表达式顺序同 `mid_denoise_loop`） |
| shoot | `grid_denoise_loop(..., tau0=float(start_t), dt=level / N)` | `tau0 + i*dt`（越过干净端） | `level / N` |

`grid_denoise_loop` 是 step_diag `mid_denoise_loop` / `shoot_denoise_loop` 的同一段语句（`process_backbone_output` → `state_encoder` → 每步 `torch.full((B,), int(τ_i * head.num_timestep_buckets))` → `step_fn(head, vl, state_features, embodiment_id, actions, timesteps, dt).clone()`，`step_fn` 缺省 `staged.denoise_step`），放在 `warm_reset/groot.py`，**不改** `groot/staged.py`（其头注释要求改动即重跑 G0-C 等价门）。整体包在 `runner._timer.measure("stage3_warm")` 内，结束后 `runner._model.validate_data(_batch_feature({"action_pred": ...}), backbone_outputs, is_training=False)`，返回 `GrootStage3Output(action_pred, start_t, steps_run=step.calls)`（实测，非 `plan.n_steps`）。证据记录 τ_i、bucket_i、dt 与换算后的 π0.5 约定 t_i = 1 − τ_i。

`groot_self_start(runner, stage2, noise, self_plan, *, schedule) -> (x_snapshot | action_final, steps_run)`：先过同样的守卫 1–4 及 capture_index 映射检查（final 也检查 start_t），再调用 `runner.run_stage3(stage2, noise=noise, on_step=observer)`（`staged.py:793-841`）。observer 在 `step == capture_index` 时抓 `x_in.detach().clone()`，并对每次回调计数——`denoise_loop` 每步恰回调一次（`staged.py:1053-1054`），因此回调次数即实测 self_direct_nfe。快照请求没有抓到指定步即抛错；final 请求取返回的 action_pred。

#### 4.3.4 step_diag 臂 → `warm_reset` 规格（迁移对照表，亦是对等性测试的枚举表）

π0.5（K=10，judge `always_warm_start start_t: T`，T ∈ {0.1, 0.2, 0.3}，N = remaining = 1/2/3）：

| step_diag 臂 | start | grid | num_steps | t（T=0.2） |
|---|---|---|---|---|
| `warm_t{T}`（ours） | 无块 | — | — | 0.2, 0.1 |
| `warmreset_t{T}` / `selfwarmreset_t{T}` | cache / self · snapshot | reset 1.0 | remaining | 1, 0.5 |
| `resetfinal_t{T}` / `selfresetfinal_t{T}` | cache / self · final | reset 1.0 | remaining | 1, 0.5 |
| `midfinal_t{T}` / `selfmidfinal_t{T}` | cache / self · final | reset 0.9 | remaining | 0.9, 0.45 |
| `midfinal50_t{T}` / `selfmidfinal50_t{T}` | cache / self · final | reset 0.5 | remaining | 0.5, 0.25 |
| `midreset_t{T}` / `selfmidreset_t{T}` | cache / self · snapshot | reset 0.9 | remaining | 0.9, 0.45 |
| `midreset50_t{T}` / `selfmidreset50_t{T}` | cache / self · snapshot | reset 0.5 | remaining | 0.5, 0.25 |
| `warmshoot_t{T}`（无自产版） | cache · snapshot | shoot 1.0 | remaining | 0.2, −0.3 |

GR00T RoboCasa（K=4，`denoise_schedule: groot_n15_k4_v1`，judge 原生 `start_t` s ∈ {0.75, 0.5} ⇒ T = 0.25 / 0.5，N = remaining = 1 / 2）：

| step_diag 臂（s） | start | grid | t（π0.5 约定）s=0.75 ／ s=0.5 | τ（原生） |
|---|---|---|---|---|
| `warm_t{s}`（ours） | 无块 | — | 0.25 ／ 0.5, 0.25 | 0.75 ／ 0.5, 0.75 |
| `warmreset` / `resetfinal`（+ `self`） | snapshot / final | reset 1.0 | 1 ／ 1, 0.5 | 0 ／ 0, 0.5 |
| `midreset` / `midfinal`（+ `self`） | snapshot / final | reset 0.75 | 0.75 ／ 0.75, 0.375 | 0.25 ／ 0.25, 0.625 |
| `midreset50` / `midfinal50`（+ `self`） | snapshot / final | reset 0.5 | 0.5 ／ 0.5, 0.25 | 0.5 ／ 0.5, 0.75 |
| `warmshoot`（+ `selfwarmshoot`） | snapshot | shoot 1.0 | 0.25 ／ 0.5, 0.0 | 0.75 ／ 0.5, 1.0 |
| `midshoot`（+ `selfmidshoot`） | snapshot | shoot 0.75 | 0.25 ／ 0.5, 0.125 | 0.75 ／ 0.5, 0.875 |
| `midshoot50`（+ `selfmidshoot50`） | snapshot | shoot 0.5 | 0.25 ／ — | 0.75 ／ — |

GR00T LIBERO（K=8，`groot_n15_k8_v1`）：`<mode>_t<s>_n<N>` ⇒ judge `start_t: s`（`_n1` 取 s=0.75、`_n2` 取 s=0.5，`envs.py:255`），`num_steps: N`（显式），grid 同 RC 同名 mode（warmreset / midreset / midreset50 / resetfinal / midfinal / midfinal50 及 self 版）；t 序列与上表逐项相同（owner 裁定的「与 K=4 同 (T, N, t)」）。`warm_t0.875` / `warm_t0.75` 为无块精确续跑。

`warm_t{T}` 与同 `start_t` 的所有 warm reset 臂共用同一个检索配置（judge 块逐字相同），只差 `warm_reset` 块——与 step_diag「共用 `warm_t<T>.yaml`」一致。

### 4.4 起点来源与自产 seed

- **cache · snapshot**：`payload.intermediates[start_t]`，即现有 WARM_START 分支已经取出的 `start_x`（π0.5：`.to(stage3_device)[None]`；GR00T：原样传入，与 `groot/interceptor.py:436` 相同）。
- **cache · final**：`cp1_result.payload.action_chunk.to(device=like.device, dtype=like.dtype).reshape(like.shape)`，`like` = 上面的快照张量。与 step_diag 数值相同（step_diag π0.5 经 `orch._storage.fetch_payload(entry_id)` 再取一次同一条目，GR00T 直接读 `cp1.payload`）。
- **self · snapshot / final**：本决策的 stage-2 句柄上跑一次 K 步完整推理，噪声 = `private_noise(seed, (H, D))`（CPU 私有 `torch.Generator`、float32，与 `recorder.make_noise` 同一实现），全局 RNG 不动。
  - π0.5：噪声 `[None].to(device=like.device)`；K = payload.denoising_num_steps = self_plan.k = PI05_V1.num_steps。自产使用 §4.3.2 的计数循环：直连绑定调用 `run_pi05_self_start(model, stage2, noise, [self_plan])[0]`；coordinator 提交 `Stage3WarmResetPayload(x=noise[H,D], plan=self_plan)`，在 `("warm_reset_self", (schedule_id, K))` 桶中执行一次 K 步循环，函数接收桶内全部逐行 plans。每行拿回自己的快照或最终动作与实测步数，再 `.to(like.device, like.dtype).reshape(like.shape)`。旧 model.run_stage3 / Stage3MissPayload 不改；代价是自产不再与普通 MISS 同桶，其他自产请求仍可合批。
  - GR00T：`groot_self_start(runner, stage2, noise, self_plan, schedule=schedule)`（§4.3.3：先过 session / 库 = 活 schedule / plan 守卫，再 `runner.run_stage3(noise=, on_step=observer)`），`observer` 在 `capture_index` 步抓 `x_in.detach().clone()` 并计回调次数 = 实测 `self_direct_nfe`。
  - 计数边界：`WarmResetSession.begin_decision` 一次性清零本请求的计数与步数累加器。executor 调用的两个绑定分别由请求局部计数 wrapper 包住：每次进入续跑 / 自产绑定时相应 calls +1，成功返回时累加该请求输出的实测 steps_run；不能仅在整个 executor 外固定加一。`n_stage3_calls = continuation_calls`，自产调用单列 self_start_calls；continuation_nfe / self_direct_nfe 为相应调用实测步数之和，多次调用不能只保留末次结果。调用异常时该决策为 error，不产生可准入的成功计数；既无全局模型补丁，也不跨请求共用计数器。π0.5 另注册 timer probe stage3_self_start。
- **seed 策略**（strategy 对象）：`SelfSeedPolicy` 协议 `seed(identity: Mapping, decision_idx: int) -> int`；v1 唯一实现 `EpisodeDigestSeed(namespace, keys)` = `stable_digest_int(namespace, *[identity[k] for k in keys], decision_idx, "self")`（`stable_digest_int` 与 `recorder.py:56-59` 逐字节同算法，放 src）。`identity` = `{"experiment", "task", "episode_id"}` ∪ `episode_start.extra_metadata`；缺任何一个键在 `begin_episode` 即抛错（第一个决策之前失败，不静默退化）。缺省键 `[experiment, task, orig_init_state_idx, attempt]` 在 LIBERO 与 RoboCasa 两个标准 runner 的 `episode_start` 中都存在（§3.3）。不含 `yaml_id` / `bundle_id` ⇒ 同一实验里各自产臂在同一 (episode, decision) 抽到同一噪声（共同随机数配对，与 step_diag 的性质相同）；含 `attempt` ⇒ 重试得到新噪声（同 step_diag）。

### 4.5 π0.5 执行：直连与 coordinator

#### 4.5.1 interceptor 改动（`src/openpi/cache/interceptor.py`，全部在「块存在」时才生效）

- 构造参数末尾加 `warm_reset: Optional[Any] = None`（`Pi05WarmResetExecutor`，由装配点注入，interceptor 本身不 import `openpi.cache.warm_reset`，与 `shadow_teacher` / `trace` 注入同形）。非 None 时：与 `trace`、`hit_executor` / `miss_executor`、`shadow_teacher`、CP2-only（`self._cp2_only`）同时出现即 `ValueError`（F10 的构造期兜底）；源为 self 时注册 probe `stage3_self_start`；构造 `self._stage3_warm_reset_fn`：有 coordinator 时 = `_make_warm_reset_via_coordinator(coordinator, bundle_id)`，否则 = `warm_reset.direct_runner(self._model)`（执行体返回绑定了模型的分派函数：`WarmResetPlan` → `run_pi05_continuation`、`SelfStartPlan` → `run_pi05_self_start`，interceptor 因此无需 import `warm_reset` 包）——与 `_stage3_from_fn` 的两种绑定（`interceptor.py:438-449`）同构。
- WARM_START 分支（`interceptor.py:1883-1907`）：validate_for_warm_start 与 start_x 的取法不变；`if self._warm_reset is None:` 走原语句，否则 `stage3, warm_reset_meta = self._warm_reset.run(stage2=stage2, cp_result=cp1_result, snapshot_x=start_x, run_stage3=self._stage3_warm_reset_fn, timer=self._timer)`。续跑本体在 stage3_warm probe 内，自产直推在 stage3_self_start probe 内；executor 在提交前完成 plan / payload 的 schedule 与 K 核对，§4.4 的调用计数 wrapper 包住实际绑定。每决策计数只由 begin_decision 清零，执行体不重复清零掩盖先前调用。
- `_build_hit_meta`（`interceptor.py:861-939`）加关键字参数 `warm_reset: Optional[dict] = None`，非 None 才写入 `meta["warm_reset"]`；尾部调用（`:2039-2044`）传入 `warm_reset_meta`（分支前初始化为 None）。其余调用点不传，wire 字节不变。

#### 4.5.2 coordinator（`src/openpi/serving/batching_core.py`、`batching_coordinator.py`）

- `batching_core.py` 新增并列 dataclass `Stage3WarmResetPayload(stage2_out, x[H,D], plan, ready_events=())`（`plan` 为 `WarmResetPlan`（续跑，`x` = 起点）或 `SelfStartPlan`（自产直推，`x` = 私有噪声）；**不继承** `Stage3WarmStartPayload`，F8）；`Stage3InitPayload` Union 加入它；`_group_stage3_requests`（`:1080-1094`）的类型白名单加入它；`_run_stage3_bucket`（`:1107-1128`）加分支：`isinstance(p0, Stage3WarmResetPayload)` → `run = getattr(self._batcher, "run_stage3_warm_reset", None)`，缺失即 `TypeError`（per-bucket 故障隔离使其只失败本桶请求，响亮而非静默）；`StageBatcher` 协议 docstring 注明该方法为可选。本模块仍 jax-free（在 GR00T 隔离测试名单内）。
- 3bucket 指标（G1 R1 N2）：两处记录（`:785-803` bucket-first、`:1061-1073` 通用循环）目前只读 payload 顶层的 `start_t` / `num_steps`，新 payload 会落成 −1 且无网格信息。对新类型改为：`mode` = `"warm_reset"` / `"warm_reset_self"`，`start_t` / `num_steps` 取自 `plan`（续跑 `plan.start_t` / `plan.n_steps`；自产 `start_t = -1.0`、`num_steps = plan.k`），并加法记录 `grid_key`（规范化为 JSON 可序列化列表）与本桶实测 `steps_run`（adapter 返回的计数器值）。旧两类 payload 的记录字段与取值逐字不变。
- `batching_coordinator.py`：re-export 新类型；`Pi05StageBatcher.bucket_key` 在原分支前识别 `Stage3WarmResetPayload`：续跑键 `("warm_reset", plan.grid_key())`，自产键 `("warm_reset_self", plan.grid_key())`，旧分支逐字不变。`run_stage3_warm_reset(payloads)` 先检查全部 plan 类型与 grid_key 同类同值，再逐行检查有效性，stack stage2 / x 后懒导入执行函数。续跑调用 `run_pi05_continuation(..., payloads[0].plan)` 并按行切回 `WarmResetStage3Output(action_chunk[i:i+1], steps_run=out.steps_run)`；reset 桶虽可混 start_t / source / point，实际网格完全相同，逐请求元数据仍从各自 plan 构造。自产调用 `run_pi05_self_start(..., [p.plan for p in payloads])`，直接返回 B 个逐行输出；请求顺序与抓取位置一一对应。每行 steps_run 是本行经过的实测迭代数，不是 B×N。
- `_make_warm_reset_via_coordinator(coordinator, bundle_id)`（interceptor 静态方法，与 `_make_warm_start_via_coordinator` 并列）：`(stage2, x, plan) -> submit_to_stage(3, bundle_id, Stage3WarmResetPayload(stage2_out=stage2, x=x.squeeze(0), plan=plan, ready_events=record_ready_events(x.device)))`。`ready_events` 沿用 trace 路径为「生产线程写入的输入」建立的协议（最终动作 / 自产噪声都是在请求线程上新生成或搬运的张量）。
- 自产臂一次决策 = 两次 stage-3 提交（自产桶 K 步 + 续跑桶 N 步），各自与其他连接同键请求合批；两次都返回实测步数。

### 4.6 GR00T 执行

- `GrootCacheInterceptor.__init__`（`groot/interceptor.py:152-164`）加 `warm_reset: Optional[Any] = None`；非 None 时与 `trace`、CP2-only 同现即 `ValueError`；orchestrator 的 CP1 judge 为 `online_rit`（`continuation_spec` 非 None）时亦拒（配置层已拒，此处兜底）。
- WARM_START 分支（`:422-449`）：`schedule = self._library_schedule(payload)` 与 `run_stage2_llm` 不变；`if self._warm_reset is None:` 原语句逐字不动，否则 `out, warm_reset_meta = self._warm_reset.run(runner=self._runner, stage2=stage2, cp_result=cp1_result, schedule=schedule)`（在同一 `runner.session()` 内）。库 schedule 由 interceptor 的 `_library_schedule` 给出，但**不**以它代替活值检查：`groot_self_start` 与 `run_groot_continuation` 各自在第一次去噪调用之前执行 §4.3.3 的守卫 1–4（session、库 = 活 schedule、plan = 库 schedule、起点可恢复），即旧入口 `run_stage3_from` 内守卫的等价物；执行体再核对 `self_plan` 与 `plan` 的 `schedule_id` / `k` 一致（G1 R1 B1）。`_build_hit_meta`（`:295-347`）加 `warm_reset: Optional[dict] = None`，行为与 `online_rit` 加法字段相同。
- 本 interceptor 模块不 import `openpi.cache.warm_reset`（执行体由入口注入），GR00T 导入链因此不可能触到 π0.5 执行体。
- serving 形态：非并发（RC `_build_served_policy`、LIBERO 非并发分支）与并发 infer-lock（两入口 `cache_factory` 非 trace 分支，含 `--allow-dynamic-bundles`）。无合批（F1）。

### 4.7 证据

#### 4.7.1 两层

1. **wire 加法字段** `__hit_meta__["warm_reset"]`（只在本决策执行续跑时出现）：共同字段为 `schema="warm_reset_meta_v1", spec_digest, kind, source, point, level, start_t, schedule_id, k, n_steps`（计划值）、`continuation_nfe, n_stage3_calls, self_start_calls`（§4.4 的实测值）、`t, dt, decision_nfe`；GR00T 另有 tau / bucket。仅自产臂增加 `self_start: True, self_seed: int, self_direct_nfe: int`；缓存臂不写这三个键，self_start_calls 为整数 0，decision_nfe = continuation_nfe。自产臂 decision_nfe = continuation_nfe + self_direct_nfe。计数不由预算回填。
2. **服务端证据 JSONL**（权威，F2）：证据 wrapper 每个请求写一行决策行，每集末尾写一条 `finalize` 闭合行，episode 批量落盘，复用 `openpi.serving.per_step_recorder.PerStepWriter(path, stamp_success=True)`（`per_step_recorder.py:67`；`flush_episode(success)` 给整集行盖 `success`，连接中断时 `close()` 以 `success=None` 尾刷）。文件 `<evidence_dir>/warm_reset_<yaml_id>_<host>_<pid>_<conn_id>.jsonl`，每连接一个文件，无跨连接锁。

#### 4.7.2 行 schema（`warm_reset_evidence_v1`）

- **公共字段**（两种行都有）：`schema, row_kind(decision|finalize), conn_id, episode_seq, bundle_id, yaml_id, yaml_sha256, spec_digest, schedule_id, k, family(pi05|groot), experiment, task, episode_id, task_uid, attempt, identity, success`。identity 为 session 在 episode_start 冻结的规范身份（experiment / task / episode_id 加已发送 extra_metadata），缓存臂也记录；seed 只选 seed_keys 对应的子集。同名身份字段若与 extra_metadata 冲突，begin_episode 拒绝，不能覆盖。conn_id 为连接 uuid，episode_seq 从 0 起；yaml_sha256 是所加载文本摘要，success 由终局盖章。
- **决策行**：`decision_idx, status(ok|error), hit_type, start_t, winner_id, warm_reset{…同 4.7.1…}|null, error|null, wall_ms`。
- **finalize 行**：`n_decisions`（服务端本集决策计数器，独立于行数）、`terminal`（`on_episode_end` 为 True；`on_task_end` / 关闭时尚有未结束 episode 为 False）、`outcome`（`on_episode_end(success)` 收到的布尔值；非终局为 None）。finalize 行与本集决策行在**同一次** `flush_episode` 中写出（`PerStepWriter._commit` 一次 `write`），任何尾部截断都会先丢 finalize 行 ⇒ 必然被判为缺闭合（fail-closed）。

决策序号在请求**入口**分配（每个请求一行，异常写 `status: error` 行后原样重抛），序号不会有空洞；执行体从同一个 `WarmResetSession` 读「当前决策序号 + episode 身份」算 seed——wrapper 与执行体共享一个会话对象，由同一个 builder 构造（单一事实来源）。

#### 4.7.3 准入检查器（`warm_reset/evidence.py`，纯函数；G1 R1 B3）

**受信输入契约**：检查器不从被检行推断任何期望值；调用方（exp 分析层）必须提供 `ExpectedEpisode`（frozen dataclass）：

| 字段 | 受信来源（由 exp 层解析，不在 src） |
|---|---|
| `task_uid`, `attempt` | 本次运行选中的唯一 accepted 终局：accepted is True、status 为 done / failed、error 为空、success 为 bool；与派发的 EpisodeTask 核对。缺失或多个冲突 accepted 终局在 exp 层拒收，不构造 ExpectedEpisode。 |
| `outcome` | 同一 accepted 终局的 `success`（bool） |
| `n_decisions` | exp 层先固定 accepted terminal 的 run_id / yaml_id / task_uid / attempt，从同一 run 的 driver per_step 中取相符且 `accepted is True`、带 hit_type 键的决策行；校验 success 与终局一致、step_idx 为无重复的合法环境步索引后计数。driver 强制盖 run_id / yaml_id / accepted，但 task_uid / attempt / success 使用 setdefault，不能把它们当作已核验身份。不得混入其他 run、静默去重或以服务端剩余行数回填；来源缺失或冲突则调用方拒收。两个 runner 的 step_idx 是环境步编号，不能当作连续 decision_idx。 |
| `yaml_id` / `bundle_id` | yaml_id 从 accepted terminal / 已派发 EpisodeTask 与 per_step 相互核对；bundle_id 从该 EpisodeTask 独立取出，不假定两者相等。 |
| `spec_digest`, `yaml_sha256`（可选） | 调用方对自己下发的 yaml 文本计算（`WarmResetSpec.from_config(...).digest()`、文本 sha256） |
| `schedule_id`, `k`, `start_t` | 臂定义（yaml 的 `denoise_schedule` / judge `start_t`） |
| `identity` | 由派发的 EpisodeTask 与冻结的环境任务定义重建实际 episode_start 身份：experiment = task.experiment、episode_id = task.episode_idx（不是 LIBERO per_step 的全局 episode_id）；LIBERO task 为该任务语言描述，RoboCasa 为 canonical task name；另含 orig_init_state_idx / attempt 及 self_seed.keys 所需的已发送字段。EpisodeTask 没有字符串 task 属性。不能从被检服务端行读取期望身份。 |
| `spec` | 由下发 yaml 构造的 `WarmResetSpec`（决定期望 N、source、seed 策略） |

`episode_problems(rows, *, expected) -> {problems: Counter, continuation_nfe, self_start_nfe, total_nfe}`：调用方先限定本次运行的证据目录与 accepted terminal 来源，再收集该 (task_uid, attempt) 在该范围全部文件中的所有行；不得在不同 run 复用同一未分区 evidence_dir，因为 task_uid / attempt 可重复。先验证 expected.n_decisions / attempt / K 等计数为非 bool 整数、outcome 为 bool、spec_digest 等于 expected.spec.digest()；所有行的 schema / row_kind、身份、外层 spec / schedule 都要检查。只有 decision 行检查内层 warm_reset 与执行字段；finalize 没有内层，按闭合规则检查。未知行类型、缺必要字段或坏类型一律形成问题码，不跳过，不通过字符串 / 布尔强转修复。检查职责：

| 类别 | 问题码 | 规则 |
|---|---|---|
| 存在 / 闭合 | `server_evidence_missing` | 无任何行（含旧 server 未写证据的情形） |
| | `finalize_missing` / `duplicate_finalize` | finalize 行恰一条 |
| | `duplicate_session` | 所有行来自同一 `(conn_id, episode_seq)`（同一 attempt 出现两个服务端会话即拒，不猜哪个被 driver 接受） |
| | `non_terminal` | finalize `terminal is True` 且 `outcome` 为 bool |
| | `outcome_mismatch` | finalize `outcome` 与各行盖章 `success` 均等于 `expected.outcome` |
| 完整性 | `no_decisions` | `expected.n_decisions ≥ 1` |
| | `decision_count_mismatch` | finalize `n_decisions == expected.n_decisions` |
| | `decision_gap` / `duplicate_decision` | 决策行的 `decision_idx` 多重集合恰为 `{0, …, expected.n_decisions − 1}`（缺中间、缺尾、重复、越界都命中） |
| | `decision_error` | 每条决策行 `status == ok` |
| 身份 | `identity_mismatch` | 每行 task_uid / attempt / yaml_id / bundle_id 等于期望；experiment / task / episode_id 及行内 identity 中预期的键值须与 expected.identity 一致（含缓存臂）；期望给出 yaml_sha256 时也必须相等。未知额外元数据可保留，必要身份键不可缺失。 |
| | `spec_mismatch` | 每行外层 spec_digest 等于 expected.spec_digest；仅 decision 行再检查内层 warm_reset.spec_digest 相等。finalize 不要求内层。 |
| | `schedule_mismatch` | 每行外层 schedule_id / k 等于期望；仅 decision 行检查内层 schedule_id / k、内层及外层 start_t 等于 expected.start_t。 |
| 执行 | `hit_type_mismatch` | 每个决策 `hit_type == WARM_START` |
| | `warm_reset_missing` | WARM_START 决策行必须带内层 `warm_reset`（旧 server 或未接线即命中） |
| | `steps_mismatch` | `continuation_nfe` = 由 `expected.spec` 与 schedule 解析出的 N（与行内 `n_steps` 也相等） |
| | `extra_stage3_calls` | `n_stage3_calls == 1` |
| 自产证明 | `self_start_missing` / `extra_self_start_calls` | 自产臂：`self_start is True` 且 `self_start_calls == 1` |
| | `self_seed_mismatch` | `self_seed` = 按 `expected.spec` 的 seed 策略、`expected.identity` 与 `decision_idx` 重算的值 |
| | `self_direct_nfe_mismatch` | `self_direct_nfe == expected.k` |
| | `self_start_on_cache_arm` | 缓存臂不得带 self_start / self_seed / self_direct_nfe 三个键，self_start_calls 必须是整数 0；自产臂三个键都必须存在且类型正确。 |
| 计价 | `decision_nfe_mismatch`；总量为输出 | 每个 decision_nfe 等于本行实测续跑与自产步数之和（缓存臂自产成本 0）；各步数、调用数、idx 必须为非 bool 的非负整数。总量为逐行实测值之和；缺计数或完整性 / 身份 / 闭合不合格时总量为 None，不输出可误用的完整 episode 成本。任一问题码非零均不准入。 |

问题码与 step_diag `cell_admission`（`aggregate_arms.py:238-397`）同名者语义相同；生产证据增加 finalize_missing、duplicate_session、duplicate_decision、spec_mismatch、warm_reset_missing、extra_self_start_calls、identity_mismatch、no_decisions、decision_nfe_mismatch。非法 schema / row_kind / 类型分别记 schema_mismatch / invalid_row / invalid_field，受信期望自身不自洽记 invalid_expected。任何问题码非零均不准入。journal / per_step 解析、配对与统计仍属 exp，不进 src。

### 4.8 装配

- 构造器：`warm_reset/pi05.py::build_pi05_warm_reset(config, *, bundle_id, yaml_id, yaml_path) -> WarmResetParts | None` 与 `warm_reset/groot.py::build_groot_warm_reset(...)`；块缺省返回 None，调用点于是什么都不做。`WarmResetParts(spec, session, executor, wrap)`，`wrap(policy)` 返回证据 wrapper（π0.5：`WarmResetEvidencePolicy`，包 `infer`；GR00T：`GrootWarmResetEvidencePolicy`，包 `get_action`；两者显式实现 `on_episode_start / on_episode_end / on_task_end` 后转发，其余经 `__getattr__` 委托，保持 server 的 `hasattr` 探测面不变，同 `_InferLockedPolicy` 的做法）。wrapper 在 `on_episode_start` 推进 `episode_seq` 并交给 `session.begin_episode`；每个请求入口 `session.begin_decision()`；`on_episode_end(success)` 写 `finalize`（`terminal: True`，`n_decisions = session.end_episode()`，`outcome = success`）后 `flush_episode(success)`；`on_task_end` 时若有未结束 episode，写 `terminal: False` 的 `finalize` 后 `close()`（§4.7.2）。
- `scripts/serve_policy.py::_wrap_policy`：bundle 分支（`:596-663`）与启动 yaml 分支（`:664-737`）各加三行：build → `InferenceInterceptor(..., warm_reset=parts.executor if parts else None)` → `if parts: policy = parts.wrap(policy)`；wrapper 位于 interceptor 与 `PolicyRecorder`（`:762-763`）之间。`--cache` 无配置分支不变。
- `exp/robocasa365/serve_groot_n15.py`：`_build_served_policy`（`:284-396`）缓存栈与 `_build_concurrent_factory` 非 trace 分支（`:606-610`）接线，包裹次序 `_InferLockedPolicy(GrootPolicyAdapter(wrapper(interceptor)), lock)`；RIT shadow 分支（`:354-383`）与 trace 分支遇到带块配置即拒绝（一行 `refuse_warm_reset(config, where=...)`）。
- `exp/libero_groot/serve_groot_libero.py`：`cache_factory` 非 trace 分支（`:652-656`）与非并发缓存栈（`:988-993`）接线；`_build_shadow_factory`（`:208`）、`_build_loto_factory`（`:295`）与 trace 分支拒绝带块配置。

动态 bundle 的证据身份必须取注册对象的真实 yaml_id 与连接选中的 bundle_id，二者独立传给 builder。π0.5 已持有 bundle；两个 GR00T 入口在现有 `_resolve_bundle(..., provenance=...)` 中为 warm reset 装配补传 bundle.yaml_id，与已有 config_path 一同传递，不能用 bundle_id 替代 yaml_id。缺块装配与旧 trace 的身份处理保持原样；测试通过真实 factory 覆盖 yaml_id != bundle_id。

### 4.9 解耦与「不感知」保证

| 组件 | 是否改动 | 保证手段 |
|---|---|---|
| judges / gates / search strategies / key builders / factors（`src/openpi/cache/components/**`） | 否 | 静态测试：这些模块不 import `openpi.cache.warm_reset`；既有测试集原样通过 |
| `CacheOrchestrator`、`CheckResult`、`CacheStorage`、backends、`BackendPool` | 否 | 同上；warm reset 判决与精确续跑走同一 `check()` 返回 |
| trace 包、online RIT | 否 | 组合被拒（配置 + 构造期），无共享代码路径 |
| conductor（`src/openpi/conductor/**`）、worker runner（`examples/libero/episode_runner.py`、`exp/robocasa365/episode_runner.py`） | 否 | 臂只是 yaml；证据在服务端；静态测试 + 既有 `tests/conductor/**` 原样通过 |
| `WebsocketPolicyServer`、`replica_proxy`、`pi0_pytorch.py`、`groot/staged.py`、`groot/batcher.py` | 否 | 同上；`GrootStageBatcher` 对新 payload 响亮失败（F8），v1 也从不提交 |
| 现有 yaml | 否 | 缺块 ⇒ `config.warm_reset is None` ⇒ 各装配点得到 None ⇒ interceptor 走原分支；「缺块逐位不变」测试（§9.3） |
| `exp/step_diag/**` | 否 | 在跑实验走各自岛树（`/data/openpi_sdiag`、`/data/openpi_sdlib`，`serve_pi05.sh:54` 以 `PYTHONPATH=$REPO/src:$REPO` 优先于本工作树的 editable 安装）；测试只读导入 |

改动集中在：`config.py`（schema + 校验）、两个 interceptor 的构造参数 / WARM_START 分支 / `_build_hit_meta`、`batching_core.py` + `batching_coordinator.py`（并列 payload 与分派）、三处装配点，以及新包 `warm_reset/`。

## 5. 改动文件清单

**新增（src）**

| 文件 | 内容 |
|---|---|
| `src/openpi/cache/warm_reset/__init__.py` | 公共名导出（不导入 `pi05` / `groot` 子模块） |
| `src/openpi/cache/warm_reset/types.py` | `WarmResetSpec`、`WarmResetPlan`、`resolve_plan`、`SelfStartPlan`、`resolve_self_plan`、`SelfSeedPolicy`、`EpisodeDigestSeed`、`stable_digest_int`、`private_noise`（jax-free，只依赖 torch 与 `openpi.cache.types`） |
| `src/openpi/cache/warm_reset/runtime.py` | `WarmResetSession`（per 连接：`conn_id`、`episode_seq`、身份、决策序号、每决策调用计数、当前 seed）、计数闭包 `_CountedStep`、`WarmResetParts`、通用 `_build(config, executor_factory, wrapper_factory, ...)`、`refuse_warm_reset` |
| `src/openpi/cache/warm_reset/evidence.py` | `WarmResetEvidencePolicy`、`GrootWarmResetEvidencePolicy`（决策行 + `finalize` 闭合行）、行 schema、`ExpectedEpisode`、`decision_problems` / `episode_problems` |
| `src/openpi/cache/warm_reset/pi05.py` | `WarmResetStage3Output`、`Pi05SelfStartOutput`、`run_pi05_continuation`、`run_pi05_self_start`（入口守卫 + 实测计数）、`Pi05WarmResetExecutor`、`build_pi05_warm_reset` |
| `src/openpi/cache/warm_reset/groot.py` | `grid_denoise_loop`、`run_groot_continuation`、`groot_self_start`（二者入口自带 session / 活 schedule / plan 守卫 + 实测计数）、`GrootWarmResetExecutor`、`build_groot_warm_reset`（jax-free） |

**修改（src / 入口）**

| 文件 | 改动 |
|---|---|
| `src/openpi/cache/config.py` | 四个数据类 + `_CONFIG_TYPES` 登记 + `CacheConfig.warm_reset` + `_warm_reset_errors` 接入 `validate_cache_config` |
| `src/openpi/cache/interceptor.py` | 构造参数 + 组合拒绝 + `_stage3_warm_reset_fn` 绑定 + `_make_warm_reset_via_coordinator` + WARM_START 分支二选一 + `_build_hit_meta(warm_reset=)` |
| `src/openpi/cache/groot/interceptor.py` | 构造参数 + 组合拒绝 + WARM_START 分支二选一 + `_build_hit_meta(warm_reset=)` |
| `src/openpi/serving/batching_core.py` | `Stage3WarmResetPayload`、Union、分组白名单、`_run_stage3_bucket` 分派、新类型的 3bucket 指标（mode / plan 取值 / `grid_key` / 实测 `steps_run`）、协议 docstring |
| `src/openpi/serving/batching_coordinator.py` | re-export、`Pi05StageBatcher.bucket_key` 分支、`run_stage3_warm_reset`（续跑 / 自产直推，逐行回传实测步数） |
| `scripts/serve_policy.py` | `_wrap_policy` 两个分支接线 |
| `exp/robocasa365/serve_groot_n15.py` | 两个缓存栈接线；RIT shadow / trace 拒绝 |
| `exp/libero_groot/serve_groot_libero.py` | 两个缓存栈接线；shadow / LOTO / trace 拒绝 |

**测试（新增为主，§9）**：`tests/cache/warm_reset/`（新目录）下 config、plan、pi0.5 对等、coordinator、interceptor、evidence、隔离；`tests/cache/groot/test_warm_reset_groot.py`；GR00T 入口的 factory 测试（新文件，沿用 `tests/robocasa365/test_groot_concurrent_serving.py`、`tests/libero_groot/test_dynamic_bundle_guards.py` 的桩法）；manual GPU 测试两份。唯一修改的既有测试文件：`tests/cache/groot/test_import_isolation.py`（把 `warm_reset/{__init__,types,runtime,evidence,groot}.py` 加入 `GUARDED_FILES` / `TRANSITIVE_ROOTS`，`openpi.cache.warm_reset.pi05` 加入 `PI05_ONLY_MODULES`）。

**文档**：`docs/architecture/cache_system.md`、`docs/cache/tutorial.md`、`docs/README.md`（§13）；`logs/README.md`（本行与后续状态）。

## 6. 接口

### 6.1 修改的现有接口（现签名已核验）

| 接口 | 现签名（file:line） | 改动 |
|---|---|---|
| `InferenceInterceptor.__init__` | `(self, policy, timer=None, orchestrator=None, eager=False, collect_images=False, stage_config=None, coordinator=None, bundle_id="default", export_collect_meta=False, collect_fields=("robot_state",), collect_kb_id="", hit_executor=None, miss_executor=None, shadow_teacher=None, trace=None) -> None`（`interceptor.py:219-236`） | 末尾加 `warm_reset: Optional[Any] = None`；None 时构造逻辑逐字不变 |
| `InferenceInterceptor._build_hit_meta` | `@staticmethod (cp1_result, arm_executed=None, checkpoint=None, library_sha256=None) -> dict`（`interceptor.py:861-867`） | 加 `warm_reset: Optional[dict] = None`（None 不写键） |
| `GrootCacheInterceptor.__init__` | `(self, policy, runner, *, orchestrator=None, timer=None, trace=None, coordinator=None, bundle_id="default", trace_vision_fields=None, model_lock=None) -> None`（`groot/interceptor.py:152-164`） | 加 `warm_reset: Optional[Any] = None` |
| `GrootCacheInterceptor._build_hit_meta` | `@staticmethod (cp1_result, *, checkpoint=None, library_sha256=None, online_rit=None) -> dict`（`groot/interceptor.py:295-302`） | 加 `warm_reset: Optional[dict] = None` |
| `Pi05StageBatcher.bucket_key` | `@staticmethod (payload) -> Hashable`：MISS `("miss", None, num_steps)`、WARM `("warm_start", start_t, num_steps)`、其他 `("unknown", None, None)`（`batching_coordinator.py:92-98`） | `Stage3WarmResetPayload`：续跑 `("warm_reset", plan.grid_key())`、自产直推 `("warm_reset_self", plan.grid_key())`；原三支逐字不变 |
| `BatchingCore._group_stage3_requests` / `_run_stage3_bucket` | `batching_core.py:1080-1094` / `:1107-1128` | 接受并分派 `Stage3WarmResetPayload`（adapter 缺 `run_stage3_warm_reset` 即 `TypeError`）；原两类分派不变 |
| 3bucket 指标记录 | `batching_core.py:784-803`、`:1061-1073`（只读 payload 顶层 `start_t` / `num_steps`） | 新类型取 plan 值并加法记录 `grid_key`、`steps_run`；旧两类记录逐字不变 |
| `CacheConfig` | `config.py:797-824` | 加 `warm_reset: Optional[WarmResetConfig] = None` |
| `validate_cache_config` | `(config: CacheConfig, *, check_files: bool = True) -> None`（`config.py:2068`） | 追加 `_warm_reset_errors` 结果；签名不变 |
| `_wrap_policy` | `(base_policy, args, *, quiet=False, eager=False, shared_cache=None, stage_config=None, bundle_id="default")`（`serve_policy.py:549-558`） | 签名不变，内部接线 |

未改但被依赖（核验过的签名）：`PI0Pytorch.run_stage3(stage2, *, noise=None, num_steps=10, return_intermediates=False, save_timesteps=(0.7, 0.5, 0.3)) -> Stage3Output`（`pi0_pytorch.py:644-702`）；`PI0Pytorch.denoise_step(state, prefix_pad_masks, past_key_values, x_t, timestep)`（`:831-871`）；`_warm_start_num_steps(start_t: float, num_steps: int) -> int`（`:185-198`）；`GrootStagedRunner.run_stage3(stage2, *, noise=None, on_step=None) -> GrootStage3Output`（`groot/staged.py:793-841`，`noise=None` 且给 `on_step` 抛 `ValueError`）；`staged.denoise_loop(action_head, backbone_output, action_input, *, noise, num_steps, start_index=0, step_fn=denoise_step, on_step=None)`（`:1002-1055`）；`GrootStage3Output(action_pred, start_t, steps_run, first_step_input=None, first_step_x=None)`（`:213-230`）；`DenoiseSchedule.remaining_steps(t) -> int`（`types.py:150-152`，非可恢复点经 `snapshot_index` 抛 `ValueError`）；`PerStepWriter(path, stamp_success=False)` / `begin_episode()` / `write_row(row)` / `flush_episode(success=_MISSING) -> int` / `close()`（`per_step_recorder.py:67-165`）；`GrootStagedRunner.live_schedule() -> DenoiseSchedule`（`groot/staged.py:420-422`，每次按活 `num_inference_timesteps` 现算）；`GrootStagedRunner._require_session(stage: str) -> None`（`:447-456`，autocast 非 bf16 抛 `RuntimeError`）；`DenoiseSchedule.snapshot_index(t) -> int`（`types.py:132-148`）；`CachePayload.validate_for_warm_start(schedule, start_t) -> None`（`storage_types.py:122-170`，内含 `snapshot_index` 检查）；`ConductorDriver.handle_result` 对 per_step 行的盖章（`src/openpi/conductor/driver.py:338-358`：`success` / `task_uid` / `attempt` 用 `setdefault`，`accepted` / `yaml_id` / `run_id` 强制赋值）。

### 6.2 新增接口

```python
# openpi.cache.warm_reset.types
@dataclass(frozen=True)
class WarmResetSpec:
    source: str; point: str; kind: str; level: float
    num_steps: int | None                 # None = remaining
    seed_namespace: str | None; seed_keys: tuple[str, ...]; evidence_dir: str
    @classmethod
    def from_config(cls, cfg: "WarmResetConfig") -> "WarmResetSpec": ...
    def digest(self) -> str: ...

@dataclass(frozen=True)
class WarmResetPlan:
    schedule_id: str; direction: str; k: int; start_t: float
    source: str; point: str; kind: str; level: float; n_steps: int
    def grid_key(self) -> tuple: ...
    def flow_times(self) -> tuple[float, ...]: ...

def resolve_plan(spec: WarmResetSpec, schedule: DenoiseSchedule, start_t: float) -> WarmResetPlan: ...  # ValueError; always checks snapshot_index(start_t)

@dataclass(frozen=True)
class SelfStartPlan:
    schedule_id: str; direction: str; k: int; start_t: float; capture_index: int | None  # None = final; start_t is still validated
    def grid_key(self) -> tuple: ...                                     # (schedule_id, k)
def resolve_self_plan(spec: WarmResetSpec, schedule: DenoiseSchedule, start_t: float) -> SelfStartPlan: ...  # ValueError
class SelfSeedPolicy(Protocol):
    def seed(self, identity: Mapping[str, Any], decision_idx: int) -> int: ...
@dataclass(frozen=True)
class EpisodeDigestSeed:                   # implements SelfSeedPolicy
    namespace: str; keys: tuple[str, ...]
def stable_digest_int(*parts: Any) -> int: ...
def private_noise(seed: int, shape: Sequence[int]) -> torch.Tensor: ...   # CPU float32

# openpi.cache.warm_reset.runtime
class WarmResetSession:
    def begin_episode(self, *, experiment: str, task: str, episode_id: int, extra_metadata: dict | None) -> None: ...
    def begin_decision(self) -> int: ...     # also zeroes the per-decision call counters
    def end_episode(self) -> int: ...        # returns the server-side n_decisions for the finalize row
    def self_seed(self) -> int: ...          # RuntimeError outside an episode / decision
    def count_continuation_call(self) -> None: ...
    def count_self_start_call(self) -> None: ...
    identity: Mapping[str, Any]; decision_idx: int | None; episode_seq: int; conn_id: str
    continuation_calls: int; self_start_calls: int
@dataclass(frozen=True)
class WarmResetParts:
    spec: WarmResetSpec; session: WarmResetSession; executor: Any; wrap: Callable[[Any], Any]
def refuse_warm_reset(config: "CacheConfig", *, where: str) -> None: ...   # ConfigValidationError if the block is present

# openpi.cache.warm_reset.pi05
@dataclass
class WarmResetStage3Output(Stage3Output):
    steps_run: int = 0                       # measured by the in-call step counter
@dataclass
class Pi05SelfStartOutput:
    action_chunk: torch.Tensor; snapshot: torch.Tensor | None; steps_run: int   # measured
def run_pi05_continuation(model, stage2: Stage2Output, start_x: torch.Tensor, plan: WarmResetPlan) -> WarmResetStage3Output: ...  # ValueError on plan/schedule mismatch before any step
def run_pi05_self_start(model, stage2: Stage2Output, noise: torch.Tensor, plans: Sequence[SelfStartPlan]) -> list[Pi05SelfStartOutput]: ...  # one plan and one unit-batch output per row; B=1 uses [plan]
class Pi05WarmResetExecutor:
    def direct_runner(self, model) -> Callable[[Stage2Output, torch.Tensor, WarmResetPlan | SelfStartPlan], WarmResetStage3Output | Pi05SelfStartOutput]: ...
    def run(self, *, stage2, cp_result, snapshot_x, run_stage3, timer) -> tuple[Stage3Output, dict]: ...
def build_pi05_warm_reset(config, *, bundle_id: str, yaml_id: str | None, yaml_path: str | None) -> WarmResetParts | None: ...

# openpi.cache.warm_reset.groot
def grid_denoise_loop(action_head, backbone_output, action_input, *, start, tau0: float, dt: float, num_steps: int, step_fn=None) -> torch.Tensor: ...
def run_groot_continuation(runner, stage2, start_x: torch.Tensor, plan: WarmResetPlan, *, schedule: DenoiseSchedule) -> GrootStage3Output: ...
    # guards before any denoise call: runner._require_session; schedule == runner.live_schedule();
    # plan.schedule_id / k / direction == schedule; schedule.snapshot_index(plan.start_t). steps_run = measured.
def groot_self_start(runner, stage2, noise: torch.Tensor, plan: SelfStartPlan, *, schedule: DenoiseSchedule) -> tuple[torch.Tensor, int]: ...
    # same guards 1-4 plus capture_index mapping; then runner.run_stage3(noise=, on_step=observer)
class GrootWarmResetExecutor:
    def run(self, *, runner, stage2, cp_result, schedule) -> tuple[GrootStage3Output, dict]: ...
def build_groot_warm_reset(config, *, bundle_id: str, yaml_id: str | None, yaml_path: str | None) -> WarmResetParts | None: ...

# openpi.cache.warm_reset.evidence
class WarmResetEvidencePolicy: ...          # infer(obs, **kw) + lifecycle forwarding
class GrootWarmResetEvidencePolicy: ...     # get_action(obs) + lifecycle forwarding
@dataclass(frozen=True)
class ExpectedEpisode:                      # trusted inputs only (journal / per_step / dispatched yaml); never from the rows
    task_uid: str; attempt: int; outcome: bool; n_decisions: int
    yaml_id: str; bundle_id: str; spec: WarmResetSpec; spec_digest: str; yaml_sha256: str | None
    schedule_id: str; k: int; start_t: float; identity: Mapping[str, Any]
def decision_problems(row: dict, *, expected: ExpectedEpisode) -> list[str]: ...
def episode_problems(rows: list[dict], *, expected: ExpectedEpisode) -> dict: ...   # {problems, continuation_nfe, self_start_nfe, total_nfe}

# openpi.serving.batching_core
@dataclass
class Stage3WarmResetPayload:               # sibling of Stage3WarmStartPayload, not a subclass
    stage2_out: Any; x: torch.Tensor; plan: Hashable; ready_events: tuple = ()
# Pi05StageBatcher.run_stage3_warm_reset(self, payloads: list) -> list[WarmResetStage3Output | Pi05SelfStartOutput]
```

## 7. 集成点

1. **yaml 解析**：`load_cache_config`（`config.py:1215-1244`）→ `_dict_to_dataclass` 经 `_CONFIG_TYPES` 递归建出 `WarmResetConfig` → `validate_cache_config` 执行 §4.2.3；`load_cache_config` ctrl（`websocket_policy_server.py:790`）走同一路径，GR00T 动态 bundle 再过 `validate_groot_cache_config`（不改）。
2. **随 WARM_START 判决**：judge / orchestrator 不变；interceptor WARM_START 分支（π0.5 `interceptor.py:1883-1907`，GR00T `groot/interceptor.py:422-449`）在块存在时把 `(spec, schedule, cp1_result.start_t)` 解析为 `WarmResetPlan` 并执行；MISS / FULL_HIT 分支完全不碰。
3. **coordinator**：`Stage3WarmResetPayload` + `("warm_reset", grid_key)`（续跑）/ `("warm_reset_self", (schedule_id, K))`（自产直推）；两者都由 `Pi05StageBatcher.run_stage3_warm_reset` 执行并逐行回传实测步数；旧 MISS / WARM_START payload 不参与。
4. **stage-3 入口**：π0.5 `run_pi05_continuation` / `run_pi05_self_start`（直连或 adapter 内，入口守卫 + 实测计数）；GR00T `run_groot_continuation` / `groot_self_start`（interceptor 内，session 已开，入口自带 session / 活 schedule / plan 守卫）。
5. **自产 seed**：`WarmResetSession` 由证据 wrapper 在 `on_episode_start` / 每请求入口推进，执行体读取；策略对象 `EpisodeDigestSeed`。
6. **证据**：`__hit_meta__["warm_reset"]`（wire，加法）+ 服务端 JSONL（wrapper，决策行 + 每集 `finalize` 行）；exp 分析层从 journal（accepted 终局的 attempt / success）与 driver 盖章的 per_step（期望决策数、yaml_id）构造 `ExpectedEpisode`，`episode_problems` 在此受信输入上核验完整性、闭合、身份与 step_diag 准入三规则（§4.7.3）。
7. **conductor**：零改动。臂 = 含 `warm_reset` 块的 yaml，经既有 `ctl.load_cache_config(yaml_content, yaml_id, bundle_id)` 下发、worker 按 `bundle_id` `select_bundle`。

## 8. 向后兼容

- **现有 yaml**：无 `warm_reset` 键 ⇒ `CacheConfig.warm_reset is None`；`_warm_reset_errors` 对 None 返回空；三处装配点的 builder 返回 None ⇒ interceptor 以 `warm_reset=None` 构造 ⇒ WARM_START 分支执行原语句（原样保留，不重排、不改缩进外的任何字符）；`__hit_meta__` 字典逐键逐值相同；无 wrapper、无新 probe、无新文件。整份 `CacheConfig` 的摘要当前没有任何生产路径在算（已核：只有 `compute_surface_retrieval_contract` 摘 `key_builder` / `keys` 子段，`config.py:922-925`），新字段不改变任何既有身份摘要。
- **精确 WARM_START 与普通 MISS**：`Stage3WarmStartPayload` / `Stage3MissPayload` 的类型、字段、桶键与 `run_stage3_warm` / `run_stage3_miss` 逐字不变；`run_stage3_from` / `run_stage3` / `Stage3Output` 不改。自产直推的实测计数走 warm reset 自有的循环与 payload（F14），因此不需要改旧接口的返回类型。
- **旧客户端 / 旧 worker**：只会多看到一个它们不读的 `hit_meta` 键（仅 warm reset yaml）。
- **旧 server 加载新 yaml**（F3）：会静默忽略块——不能靠兼容性兜住，只能靠准入（R1）。
- **在跑实验**：`exp/step_diag/**` 不改；两个队列的 server 由岛树启动且 `PYTHONPATH` 指向岛树（`self13_queue.py:53`、`:61`；`libero_queue.py:51-52`；`serve_pi05.sh:54`），本工作树的改动不会被它们加载；实现期间不同步任何岛树（`/data/*`）。

## 9. 测试策略

全部 CPU 测试以 `CUDA_VISIBLE_DEVICES=""` 可跑，进 CPU 全量；GPU 测试仅 `@pytest.mark.manual`，不进 CPU 全量，按需在正确 venv 手动执行（π0.5：主 venv；GR00T：孤岛 B + gr00t worktree 于 `PYTHONPATH`，并核对 skip 数以防 `importorskip` 静默跳过）。**口径（G1 R1 N1）**：真实模型上的任何「逐位」声明只以相应 manual GPU 对等测试的通过记录为证据；CPU 桩通过、或 manual 测试被 skip，都不能替代该证据。

### 9.1 B=1 逐位对等（与 step_diag）

- **π0.5 K=10**（`tests/cache/warm_reset/test_parity_pi05.py`，CPU 桩）：枚举 `WARM_VARIANT_MODES ∪ SELF_VARIANT_MODES` × T ∈ {0.1, 0.2, 0.3}（从 `exp.step_diag.envs` 只读导入，单一事实来源）；桩模型的 `denoise_step` 对 (x, t) 非线性并记录 t 的 float32 位模式。断言：(a) 新 `run_pi05_continuation` 与 `exp.step_diag.pi05.warm_variant_stage3` 的 t 位序列与输出 `torch.equal`；(b) 最终动作起点与 step_diag 取法相同；(c) 给定同一 seed 整数，新 `run_pi05_self_start` 的快照 / 最终动作与 `Pi05DiagInterceptor._self_start`（经 `model.run_stage3(return_intermediates=True, save_timesteps=(t,))`）`torch.equal`，续跑输出亦然；(d) 经完整 `InferenceInterceptor`（沿用 `tests.cache.test_interceptor.FakePolicy`）与经 `Pi05DiagInterceptor`（其 `DiagRecorder` 写入 tmp）跑同一决策，执行动作逐位相同；(e) `PI05_V1.snapshot_index(t) == round((1.0 − t) * 10)` 与 `remaining_steps(t) == _warm_start_num_steps(t, 10)` 对 9 个可恢复点成立。
- **GR00T RC K=4 与 LIBERO K=8**（`tests/cache/groot/test_warm_reset_groot.py`，CPU 桩）：枚举 `serve_diag_groot.GROOT_VARIANT_MODES` × s ∈ {0.75, 0.5}，以及 `LIBERO_SELF_ARMS_BY_POLICY["groot"]` 的全部 `_n1/_n2` 臂；桩 action head 记录每步 (bucket, dt) 与输入。断言新 `run_groot_continuation` 与 `groot_warm_variant_stage3(num_steps=…)` 的 (bucket, dt) 序列、输入序列与输出逐位相等；同 seed 下新 `groot_self_start` 与 step_diag `groot_self_start` 逐位相等（快照与最终动作两种）。
- **覆盖性**：测试内的「arm id → spec」映射（即 §4.3.4 表）对 `MACRO13_ARMS_BY_POLICY`、`SELF13_ARMS_BY_POLICY`、`LIBERO_SELF_ARMS_BY_POLICY`、`VAR500_ARMS`、`XSEED_ARMS_BY_POLICY` 中每个 warm 家族臂都有定义（缺一即失败）；`warm_t*` 映射为「无块」。
- **计划解析表**：`resolve_plan` / `resolve_self_plan` 对上述每臂给出的 (kind, source, point, N, capture_index, `flow_times`) 与 §4.3.4 表一致。
- **manual GPU（B=1）**（`tests/cache/warm_reset/test_parity_manual.py`、`tests/cache/groot/test_warm_reset_groot_manual.py`）：真实 checkpoint + 一条真实 wire 观测（沿用 step_diag 的 `SD_OBSERVATION` npz 约定），每个环境每臂：新路径 vs step_diag 函数 B=1 逐位；π0.5 coordinator 路径在「桶内仅一个请求」时与直连逐位（续跑桶与自产桶各一）。这是本计划唯一的真实模型逐位声明。

### 9.2 B>1 并发验收（不作逐位声明，G1 R1 N1）

- **CPU 桩（确定性，可逐位）**：B=3 的续跑桶与自产桶逐行等于单行直连——这只验证 adapter 的行对齐、按行快照与计数回传逻辑，桩是逐行独立的确定函数。
- **manual GPU**：B=4 同桶结果与逐行 B=1 的 max|Δ|、各行 `steps_run`、行对齐（每行只拿到自己请求的快照 / 最终动作）记录进测试输出；断言只针对路由、身份与计数，数值偏差只记录（F6）。
- **serving 冒烟**（§10.3）同样只核对路由 / 身份 / 计数，并记录相对单连接参考的数值偏差。

### 9.3 缺块逐位不变（「不感知」的测试保证）

- config：无块 yaml ⇒ `warm_reset is None`；带块 yaml 的数据类往返；`spec.digest()` 稳定。
- π0.5 interceptor：`warm_reset=None` 时，对 FULL_HIT / WARM_START / MISS 三种判决记录模型与 coordinator 的全部调用（方法名 + 参数），与冻结期望逐项相等（WARM_START 必须恰为 `run_stage3_from(stage2, start_x, start_t, num_steps=10)` 或 `Stage3WarmStartPayload`）；`__hit_meta__` 与冻结期望字典相等；不注册 `stage3_self_start` probe；模型实例上没有任何属性被替换（`vars(model)` 前后键集相等）。
- GR00T interceptor：同上，WARM_START 恰为 `runner.run_stage3_from(stage2, intermediates[start_t], start_t, schedule=…)`。
- coordinator：`Stage3MissPayload` / `Stage3WarmStartPayload` 的 `bucket_key`、分派参数、返回类型与 HEAD 相同；这两类的 3bucket 指标记录逐字段相同。
- 装配：无块 yaml 时 `_wrap_policy` 返回的对象类型与 HEAD 相同（`InferenceInterceptor`，或 `--record` 时的 `PolicyRecorder`）；GR00T 两入口 factory 返回的栈类型不变。
- 静态解耦：AST 扫描 `src/openpi/cache/components/**`、`orchestrator.py`、`cache_storage.py`、`backends/**`、`src/openpi/conductor/**`、`src/openpi/serving/websocket_policy_server.py`、`examples/libero/**`、`exp/robocasa365/episode_runner.py`，断言均不 import `openpi.cache.warm_reset`；GR00T 隔离测试名单扩充（§5）。
- 既有测试集原样通过：`tests/cache/**`（含 `test_interceptor.py`、`test_serving_optimization.py`、`trace/**`、`groot/**`）、`tests/serving/**`、`tests/conductor/**`、`tests/libero_groot/**`、`tests/robocasa365/**`、`tests/exp/step_diag/**`。

### 9.4 功能与拒绝

- **Round 2 接口正反例**：自产 B=3 同桶请求依次抓 t=0.1、t=0.3、final（capture_index 为 9 / 7 / None），使用不同噪声；与各自 B=1 参考比较并置换行序再验证，所有输出保持 `[1,H,D]`，计数各为 K。plans 长度或 stage2 / noise batch 维不符、非桶首 plan 非法、K 与 PI05_V1 不符、final 的 start_t 非法、capture_index 与 start_t 错配，均在首次去噪前拒绝；两模型均覆盖手工坏 plan。
- **受信输入适配样例**：用真实 runner 字段形状的 CPU fixture 展示 ExpectedEpisode 的构造：LIBERO 语言描述与 wire episode_idx（对照不同值的 per_step 全局 episode_id）、RoboCasa canonical name、yaml_id 与 bundle_id 不同；同 uid / attempt 的其他 run 行不能并入，保留的 stale attempt / success、重复 step_idx、缺终局、accepted terminal.error 非空或多个冲突 accepted 终局均拒收。样例仅在测试层，正式 exp 分析器仍为后续工作。

- **yaml 校验**：§4.2.3 每条规则各一个负例 + 正例（含 `task_uid` 进 identity、N > K、final+shoot、CP2、online_rit、trace、routing、shadow_teacher、evidence_dir 不可写）。
- **GR00T 运行时守卫（G1 R1 B1）**：计数桩 action head 断言以下每种情形都在**第一次去噪调用之前**抛错（桩的 `denoise_step` / `process_backbone_output` 调用数为 0）：(1) 库 schedule K=4、活 head K=8；(2) 装配后把活 head 的 `num_inference_timesteps` 改掉（执行体已构造、第二次调用时拒绝）；(3) 不在 `runner.session()` 内调用（autocast 非 bf16）；(4) 显式 N 配非可恢复的 `start_t`（`resolve_plan` / 入口守卫各一例）；(5) `plan` 与传入 schedule 的 `schedule_id` / `k` 不一致；(6) `groot_self_start` 在 (1)–(3) 下同样先拒。π0.5 对应：plan 的 schedule 非 `PI05_V1`、`payload.denoising_num_steps != plan.k` 时在第一次 `denoise_step` 之前拒绝。
- **实测计数链路（G1 R1 B2）**：执行 → 输出 → hit_meta → wrapper → episode_problems 全链路注入，不直接篡改 JSON：(1) π0.5 的 `_loop_steps` 返回 N−1 / N+1，计数同向变化并触发 steps_mismatch；(2) `_self_loop_steps` 返回 K−1 / K+1，使用 final 或仍能抓到的早期 snapshot 验证 self_direct_nfe_mismatch；未抓到要求的 snapshot 则 error，不得回退；(3) 多调用一次续跑绑定，calls = 2 且累计步数 = 2N，触发 extra_stage3_calls；(4) 多调 / 漏调自产绑定被拒，多调时累计 2K；(5) B=3 每行计数为 K / N，非 3K / 3N；(6) π0.5 直连与 coordinator 各走一遍 (1)–(4)，双连接计数互不影响；(7) GR00T 在实际循环 / runner 的测试替身中少执行或多执行一步，分别核验 step_fn 与 on_step 计数贯穿证据；不能用改变 live K 代替故障注入，因为那会先触发 schedule 守卫。
- **coordinator**：同网格的缓存 / 自产 / 快照 / 最终动作臂同续跑桶；N、entry、schedule、shoot 的 `start_t` 不同即分桶；不同 T、不同 point 的自产直推同自产桶且各行拿回自己的快照；`GrootStageBatcher` 收到新 payload 时该桶响亮失败且不影响同批其他桶；噪声设备放置（F9）；新类型的 3bucket 记录带 `grid_key` 与 `steps_run`，旧类型记录不变（G1 R1 N2）。
- **自产 seed**：同一 (identity, decision_idx) 两次运行相同；`decision_idx` / `attempt` / `task` 变化即变；`yaml_id` / `bundle_id` 变化不变；缺 identity 键在 `begin_episode` 报错；全局 RNG（CPU 与 CUDA 状态）前后相等。
- **构造期拒绝**：π0.5 `warm_reset` + `trace` / 执行器 / `shadow_teacher` / CP2；GR00T `warm_reset` + `trace` / CP2 / online RIT。
- **证据写出**：每请求一行、序号连续、身份字段、`success` 盖章、每集恰一条 `finalize`（与决策行同一次 flush）、`on_task_end` 时 `terminal: False`、异常写 error 行并重抛。
- **证据准入（G1 R1 B3）**：分别构造缓存 / 自产、成功 / 失败的合法 10 决策 episode 为正控，finalize 不带 warm_reset，缓存 metadata 不带自产三字段，四种组合均准入。负例逐项覆盖：空集、期望 n=0、删尾 2 个 decision 但保留 finalize / success（decision_gap）、finalize.n_decisions 与期望不符（decision_count_mismatch）、中间缺行、重复 / 越界 idx、缺 / 重复 finalize、非终局、终局或行盖章与受信 outcome 不同、重复 session、task_uid / attempt / yaml_id / bundle_id / yaml_sha256 / wire identity 任一错配、内外 spec / schedule / start_t 冲突、缺 warm_reset、非 WARM_START、自产 seed / K / calls 错配、缓存臂带自产字段、error 行、错误 schema / row_kind / bool 冒充计数、decision_nfe 与实测和不符。所有负例不准入；完整性失败不能按剩余行给出完整 episode NFE；π0.5 K=10,N=2 与 GR00T K=8,N=1 均验证 K+N 计价。
- **装配**：带块 yaml 在 `_wrap_policy` 两个分支、GR00T 两入口 4 个缓存栈构造点上得到「wrapper + 注入执行体」；RIT shadow / LOTO / trace 分支对带块 yaml 报错。

## 10. 迁移 / 上线

### 10.1 实施顺序

1. G1 通过后在本工作树实现（owner 若要求，可在 worktree 中进行）；不同步岛树、不动在跑队列。
2. §6 Verify：全量只跑 CPU、不碰 GPU 测试（owner 2026-09-24 定的标准命令：`CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 uv run --no-sync --with pytest-xdist --with pytest==9.0.2 pytest -n 24 --dist loadfile`），列出 HEAD 既有失败；本改动触及推理路径（WA §2.7），另按需点名跑 §9.1 的两份 manual GPU 对等性测试（B=1 逐位）与 §9.2 的 manual B>1 记录，不进全量；真实模型逐位结论只引用这两份 manual 测试的通过记录（N1）。
3. serving 冒烟（§10.3）。
4. 闭环：后续 exp 计划（Q6）写 arm emitter 与 conductor strategy，smoke 实验 id → 正式轮。

### 10.2 一个 warm reset 臂如何变成 yaml + conductor 臂

1. yaml = 该环境的 `warm_t<start_t>.yaml`（judge `always_warm_start`；GR00T 带 `denoise_schedule`）原样 + 按 §4.3.4 表追加 `warm_reset` 块（`evidence_dir` 为 server 本地路径；自产臂写同一实验命名空间）。
2. strategy 在 `on_stage_begin` 用 `ctl.load_cache_config(yaml_content=..., yaml_id=<arm>, bundle_id=<arm>)` 下发（`run_size_eval.py:102-109` 同式）；π0.5 对 `serve_policy`（并发缺省，可 `--replicas ≤3`），GR00T 对 `serve_groot_n15` / `serve_groot_libero --concurrent --allow-dynamic-bundles`（LIBERO 另 `--denoising-steps 8`）。同一 env 的多臂共享一份库（`BackendPool`）。
3. `EpisodeTask.bundle_id = <arm>`，worker 自行 `select_bundle`；W 个 worker × M 个臂共享一个进程。
4. 事后：journal（结局）+ per_step（worker）+ 服务端证据 JSONL → `episode_problems` 准入 → exp 分析层统计。

例（π0.5 LIBERO `selfresetfinal_t0.2`）：

```yaml
# … exp/step_diag/config/arms/pi05_libero_10/warm_t0.2.yaml 的全部内容（judge: always_warm_start, start_t: 0.2）…
warm_reset:
  start: {source: self, point: final}
  grid: {kind: reset, entry_t: 1.0}
  num_steps: remaining
  self_seed: {namespace: sdiag_libero_self_prod}
  evidence_dir: /data/warm_reset_evidence/libero_self_prod
```

例（GR00T LIBERO `midreset_t0.75_n1`）：`denoise_schedule: groot_n15_k8_v1`、judge `start_t: 0.75` + `warm_reset: {start: {source: cache, point: snapshot}, grid: {kind: reset, entry_t: 0.75}, num_steps: 1, evidence_dir: …}`。

### 10.3 冒烟长什么样

- **serving 冒烟（本计划交付物内，manual）**：weilandserver 4090，在 23100-23199 段中挑空闲块（先 `ss` 侦察，不动他人 tmux）起 π0.5 LIBERO 并发 `serve_policy`；经 ctrl 载入 4 个 bundle（`warm_t0.2` 精确、`warmreset_t0.2`、`resetfinal_t0.2`、`selfresetfinal_t0.2`）；8 个合成客户端（每 bundle 2 个）回放录制的 wire 观测若干集。验收只核对路由 / 身份 / 计数（G1 R1 N1；此时 stage 1/2/3 都可能 B>1，不要求逐位）：精确臂 `hit_meta` 无 `warm_reset` 键、每决策走 `Stage3WarmStartPayload`；三个 warm reset bundle 的证据行齐全并通过 `episode_problems`（以回放脚本自身记录的每集决策数与结局构造 `ExpectedEpisode`）、`continuation_nfe = 2`、`n_stage3_calls = 1`、自产臂 `self_direct_nfe = 10`、`self_start_calls = 1` 且 seed 可由身份重算；`dump_metrics` 的 3bucket 记录出现 `warm_reset` 桶（warmreset / resetfinal / self 共享同一 `grid_key`）与 `warm_reset_self` 桶，`steps_run` 分别为 2 与 10（N2）；每个 bundle 另取同一观测序列在单连接下的动作作参考，**记录** max|Δ|，不设逐位门槛；记录吞吐。GR00T：孤岛 B 上 `serve_groot_libero --concurrent --allow-dynamic-bundles --denoising-steps 8`，载入 `warm_t0.875`、`midreset_t0.75_n1`、`selfmidreset_t0.75_n1` 同样核对（GR00T 推理在 infer 锁下串行，无合批桶可读）。
- **闭环冒烟（后续 exp 计划）**：smoke 实验 id、每臂 1 任务 × 2–4 集，准入全绿后才开正式轮。

### 10.4 吞吐预期与理由

- **现状上限**：一臂一进程一连接。π0.5 进程 7.8 GB 显存 + 31 GB RSS、GR00T 6.4 GB + 21 GB（`self13_queue.py:49`）；wls（44 GB / 220 GB）最多约 5 个 π0.5 进程（显存先到顶）、h100（76 GB / 175 GB）约 5 个（RSS 先到顶）⇒ 每台主机同时只有约 5 个 π0.5 episode 在跑，且每个进程是 B=1、launch-bound（4090 上 GPU 利用率 6–15%）。
- **π0.5 生产并发**：一个进程（≤3 replica）承载全部臂的全部 worker，stage 1/2 跨臂合批，stage 3 按网格键合批；warm reset 臂 stage 3 只有 N=1–3 步，决策成本主要在已合批的 stage 1/2；自产臂的 K 步直推在 `warm_reset_self` 桶里与其他连接的自产直推合批（不与普通 MISS 同桶，B2 修订的代价；warm 臂 judge 恒发 WARM_START，普通 MISS 本就稀少）。既有实测：a100 单进程约 12 inf/s、3 replica 约 24–29 inf/s（LIBERO 闭环，N≈48 worker）；wls 本机 16 worker 18.7 ep/min（GPU 86%）对 4 worker 6.15 ep/min。据此预期同主机并发 episode 数从约 5 提到 16–48，吞吐约 3–5 倍直至 GPU 饱和；以 §10.3 冒烟实测为准，不作承诺。
- **GR00T**：非 trace 路径推理仍进程内串行（F1），收益来自 (a) RSS/显存只付一份（一个进程服务所有臂与 worker）、(b) 其他连接的仿真步进与网络往返和本连接推理重叠。单进程上限约 `1 / t_inf` 决策每秒，相对现状的增益约 `min(W, (t_inf + t_sim + t_rtt) / t_inf)`；RoboCasa 仿真重、增益大于 LIBERO。需实测 t_inf 后再定 worker 数；若不够，合批属 Q3。

## 11. 风险登记

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| R1 | 旧代码 server（未同步岛树等）静默忽略 `warm_reset` 块，跑成精确续跑（F3） | 臂名与语义不符，结论全错且无报错 | 准入强制每决策正向证据 + `spec_digest`；冒烟核对；部署清单要求 server 树提交号与本计划实现提交一致 |
| R2 | 浮点表达式顺序漂移导致与 step_diag 不再逐位（F5） | 「一等公民化」后语义悄悄变化 | 按 kind 分派到与 step_diag 相同的三条 GR00T 循环；π0.5 同式构造 dt/起点；逐位对等测试覆盖全部臂 |
| R3 | 并发合批数值 ≠ B=1（F6） | 生产结果与 step_diag cell 不可逐位复现 | 对等性声明限定 B=1；分析层禁止与 step_diag cell 混池（Q8）；manual 测试记录 B>1 偏差量级 |
| R4 | 合批桶碎片化、自产臂两次排队、自产直推不再与普通 MISS 同桶 | 延迟上升、吞吐不及预期 | 网格键排除 source/point、自产键排除抓取步以最大化共享；冒烟读 3bucket 指标（`grid_key` / `steps_run`，N2）；`BATCHING_MAX_WAIT_MS` 可调 |
| R5 | wrapper 与执行体的决策序号 / 身份不一致 | 自产 seed 错位，自产证明失败 | 单一 `WarmResetSession` 对象、入口分配序号；准入按期望身份重算；专项测试 |
| R6 | GR00T 孤岛导入链被新模块污染（jax / π0.5 模块） | server 启动即失败 | interceptor 不 import warm_reset；GR00T 侧模块进 `GUARDED_FILES` / `TRANSITIVE_ROOTS`，`warm_reset.pi05` 进 `PI05_ONLY_MODULES` |
| R7 | GR00T LIBERO 连接中断时未结束 episode 的证据丢失（F11） | 该 attempt 证据缺失 | 准入把它判为缺证据（fail-closed）；conductor 重试产生新 attempt；不为此改 adapter |
| R8 | 实现期影响在跑队列 | 实验被打断或混入新代码 | 队列 server 走岛树且 `PYTHONPATH` 优先（已核）；不同步 `/data/*`、不动 tmux；需要时在 worktree 实现 |
| R9 | 多档 judge（`warm_tiers`）与显式 N 组合的语义歧义 | 不同档用同一 N | 语义写入文档：显式 N 对所有档相同，`remaining` 随档变化；测试覆盖 |
| R10 | 证据 I/O 与目录权限 | 连接建栈失败 | 加载期检查可写；行小、按集批量写、每连接一个文件 |
| R11 | 新 payload 被不支持的 adapter 执行（F8） | 静默精确续跑 | 并列类型 + 缺方法即 `TypeError`；测试 |
| R12 | 自产臂 stage-3 计算翻倍、stage-2 句柄在两次提交之间常驻 | 显存 / 延迟压力 | 与精确 WARM_START 相同的句柄生命周期；冒烟记录显存；worker 数按实测调 |
| R13 | 新 GR00T 入口绕过旧入口内的 session / 活 schedule 守卫（F13，G1 R1 B1） | autocast 漏开（LayerNorm max\|Δ\|≈1.4e-2）或 K=4 库在 K=8 head 上续跑，数值静默错误 | `run_groot_continuation` / `groot_self_start` 入口自带守卫 1–4 且先于任何去噪调用；每次调用现取活 schedule；§9.4 六类拒绝测试断言零去噪调用 |
| R14 | 证据计数回填预算而非实测（F14，G1 R1 B2） | 少跑 / 多跑一步或多一次续跑仍被准入 | 计数闭包与调用计数在执行层产生、经 adapter 逐行回传；故障注入测试走完整链路（§9.4） |
| R15 | 证据尾部丢失 / 会话重复仍「看起来完整」（F15，G1 R1 B3） | 少算决策、错绑结局 | `finalize` 闭合行与决策行同一次写出；期望决策数与结局只取自 journal / driver 盖章的 per_step；重复会话、缺尾、缺中间、结局不一致全部拒收（§4.7.3、§9.4） |

## 12. 开放问题（owner）

| # | 问题 | 建议 |
|---|---|---|
| Q1 | `entry_t` / `step_budget` 用哪种时间约定？ | 一律 π0.5 约定（与 owner T/N/t 记法一致），judge 的 `start_t` 保持 schedule 原生；文档与示例逐字标注，证据同时记两种 |
| Q2 | 生产自产 seed 是否必须等于 step_diag 的 `noise_seed`？ | 否。标准 worker 不发 env seed / 池 SHA，强求需改 worker（违背约束 3）；用「命名空间 + episode_start 身份键」，对等性以同 seed 整数验证 |
| Q3 | GR00T 非 trace 路径是否要合批？ | 本计划不做；先以 infer-lock 并发冒烟测 t_inf 与吞吐，不够再单开 L3（需重议整段 infer 锁契约） |
| Q4 | trace 模式是否支持 warm reset？ | v1 拒绝；确有需要时在 trace 变体表里加 warm reset 臂，另立计划 |
| Q5 | 是否让 worker 的 per_step 抄 `hit_meta["warm_reset"]`？ | 否（遵守约束 3）；服务端证据为权威，wire 字段留作可选观测 |
| Q6 | step_diag 臂的 exp 侧 arm emitter + conductor strategy + 分析是否并入本计划？ | 另立后续 exp 计划（L1/L2），本 L3 只交付机制与证据，G1 审查面更聚焦 |
| Q7 | 显式 N 的上界是否保留 N ≤ K？ | 保留（沿用 step_diag LIBERO 规则，续跑不应贵于完整推理，计价口径清楚） |
| Q8 | 生产（可能合批）结果能否与 step_diag（B=1）结果混池？ | 不混池；若要跨框架比较，先跑一个交叉核对臂 |
| Q9 | 等 NFE 参照臂（`plain_k<k>`）如何上生产？ | 不在本计划；`full` 可用 `always_skip` gate 的 yaml 表达，`plain_k` 需要单独的「按 bundle 钉 MISS 步数」特性 |
| Q10 | 块名 `warm_reset`（避开 online RIT 的 `continuation_spec`）是否认可？ | 建议认可 |

## 13. 架构文档更新（随代码同一提交）

- `docs/architecture/cache_system.md`：新增 §5.22「Warm reset 续跑族（`warm_reset`）」——§4.1 的镜像表、yaml 块、计划解析与两模型网格公式（含 F5 的表达式顺序要求）、起点四种来源与自产 seed、新入口自带的运行时守卫（GR00T session / 活 schedule，F13）、实测计数链路（F14）、拒绝组合、证据两层（决策行 + `finalize`）与受信输入的准入契约及问题码（F15）、「缺块逐位不变」；§9.X BatchingCoordinator 段落补 `Stage3WarmResetPayload` 与 `("warm_reset", grid_key)` / `("warm_reset_self", …)` 桶、新类型 3bucket 指标字段、可选 adapter 方法与响亮失败；§5.17 补一句：GR00T 的 warm reset 走 infer-lock 路径，无合批。
- `docs/cache/tutorial.md`：§10 yaml 参考加入 `warm_reset` 块（字段、约定、校验规则、§4.3.4 对照表摘要）；§12 Interceptor 补 WARM_START 二选一。
- `docs/README.md`：上述两条目的描述与日期同步（WA §4 索引同步）。
- `logs/README.md`：本计划状态随阶段更新。

## 14. 实现记录（G2 交付，2026-09-25 02:15 CDT）

**变更集**（工作树，未暂存、未提交；审查方先前暂存的是 G1 阶段的计划与索引基线）：

| 文件 | 内容 |
|---|---|
| `src/openpi/cache/warm_reset/{__init__,types,runtime,evidence,pi05,groot}.py`（新） | §4 / §6.2 全部新接口：冻结 spec / plan 与校验（手工构造的 plan 也过同一校验）、GR00T 网格、自产 seed 与私有噪声；会话（身份、决策序号、调用与步数计数）、计数 wrapper、`refuse_warm_reset`；证据 wrapper（决策行 + 同一次落盘的 `finalize` 行）、`ExpectedEpisode`、`decision_problems` / `episode_problems`；π0.5 / GR00T 续跑与自产入口（入口守卫 + 实测步数）、执行体与 builder |
| `src/openpi/cache/config.py` | 四个数据类、`CacheConfig.warm_reset`、`_warm_reset_errors` 接入 `validate_cache_config` |
| `src/openpi/cache/interceptor.py`、`src/openpi/cache/groot/interceptor.py` | 构造参数 + 组合拒绝、stage-3 绑定、WARM_START 分支二选一（块缺省时原语句逐字保留）、`_build_hit_meta(warm_reset=)` |
| `src/openpi/serving/batching_core.py`、`batching_coordinator.py` | 并列 `Stage3WarmResetPayload`、分组 / 分派（不支持的 adapter 该桶 `TypeError`）、两个新桶键、`run_stage3_warm_reset`、3bucket 指标新字段 |
| `scripts/serve_policy.py`、`exp/robocasa365/serve_groot_n15.py`、`exp/libero_groot/serve_groot_libero.py` | 装配接线（π0.5 两分支、GR00T 四个缓存栈构造点）；RIT shadow / LOTO / trace 分支拒绝带块配置；`_resolve_bundle` 传出 bundle 的 `yaml_id` |
| `docs/architecture/cache_system.md`、`docs/cache/tutorial.md`、`docs/README.md` | §13 文档更新 |
| 测试：`tests/cache/warm_reset/`（新，含 `_arms.py` 臂表）、`tests/cache/groot/test_warm_reset_groot.py`（新）、两份 manual GPU 对等测试（新）、`tests/cache/groot/test_import_isolation.py`（名单扩充） | §9 |

**计划符合性 — 偏差**（执行方已向 owner 报告，待 G2 审查确认）：
1. `src/openpi/cache/trace/runtime.py` 加一行 `"WarmResetConfig": ("evidence_dir",)` 到 `TWIN_READONLY_PATH_FIELDS`：计划写 trace 包不改，但既有测试 `test_config_trace.py::test_every_path_like_config_field_is_classified` 要求每个路径型配置字段都登记（与 `ShadowTeacherConfig.path` 同做法）；warm_reset 与 trace 互斥，无运行时影响。
2. `exp/robocasa365/serve_groot_n15.py::_build_served_policy` 补一行 `from openpi.cache.groot.staged import GrootStagedRunner`：HEAD 上该单连接 `--cache-config` 路径只在 trace 分支里导入 `GrootStagedRunner`，非 trace 时即 `UnboundLocalError`（既有缺陷）；不补则计划的第一个 GR00T 构造点无法运行。
3. 接口增补（计划 §4.4 的累计实测步数所需）：`WarmResetSession.in_episode` / `add_continuation_steps` / `add_self_start_steps` / `continuation_nfe` / `self_direct_nfe`；公开辅助 `runtime.decision_meta`、`types.native_grid` / `groot_loop_grid`；`run_pi05_continuation` 额外拒绝起点 batch 与 stage-2 不一致。
4. 计划未定的边界（均 fail-closed）：episode 外的请求抛 `RuntimeError` 且不写行；未收到 `episode_end` 又来 `episode_start` 时把前一集记为非终局；启动 yaml 无 `yaml_id` 时证据文件名用 `none`。
5. GR00T `_loop_steps` 故障注入钩子（测试用，生产恒等于 N）。

**本地测试（advisory，非 §6 Verify；CPU-only，`uv run --no-sync --with pytest-xdist --with pytest==9.0.2 pytest -n 24 --dist loadfile`）**：
- 新测试（`tests/cache/warm_reset`、`tests/cache/groot/test_warm_reset_groot*.py`、`test_import_isolation.py`、`test_config_trace.py`）：655 passed、2 skipped（两份 manual）。
- 受影响套件（`tests/cache tests/serving tests/conductor tests/libero_groot tests/robocasa365 tests/exp/step_diag`，忽略 HEAD 即收集失败的 `test_bench_groot_stages.py`）：**3813 passed、60 skipped、4 failed**（执行方复跑，02:14 CDT，116 s）；4 项均为既有：`test_ws2_evidence_runner` ×2、`test_groot_concurrent_serving::test_frozen_commands_pass_the_new_guards` ×2（全量时测试污染，单独跑通过）。
- 全仓（实现方跑）：7172 passed、94 skipped、21 failed = 6 项既有基线 + 15 项 `tests/review_tests/` 他线旧探针（与本次改动前的全量结果同一组）。
- manual GPU 对等测试（π0.5 主 venv；GR00T 孤岛 B，须核 skip 数）两份已写、**未运行**（owner：GPU 测试只按需；GPU 当前被实验占用）。真模型上的逐位对等以它们在 §6 Verify 按需跑通为准；§10.3 serving 冒烟亦未做。

## 15. G2 放行后的执行方复核与 Verify（2026-09-25 12:02 CDT 起）

G2 Round 2 APPROVED 后，按 owner 常设规则（审查方的直接修改不整体照收，执行方逐块复核、改正确有缺陷处）复核了暂存区外属于本计划的全部 hunk，再做 §6 Verify、按需 GPU 对等与 §10.3 冒烟。暂存区保持审查方留下的原样（cached diff sha256 `8cd98f3f…` 全程核对不变）；执行方改动全部留在工作树。

### 15.1 审查方直接修改的复核结论

| 对象 | 结论 | 依据 |
|---|---|---|
| B1 `evidence.py` 准入核对 source / point / kind / level / t / dt（GR00T 加 tau / bucket）、family、成功行不得带 error、必要字段齐全 | 接受 | 期望值与执行体写入值同源：π0.5 `dt` 执行体写 `float(torch.tensor(-level/N, float32))`，检查器用 `struct` float32 取整，二者逐位相同；`t` 双方都是 `plan.flow_times()`；GR00T `tau` / `dt` 双方都是 `native_grid(plan)`（原生时间）。JSON 往返对 python float 精确。正控（π0.5 缓存 / 自产 × 成败、GR00T K=8 自产 K+N 计价、新增的入口端到端正控）全部准入，无误拒 |
| B2 `ExpectedEpisode` 与身份 / 规格先自检，坏 `conn_id` / `episode_seq` 记 `invalid_field` 不抛异常 | 接受 | 缓存臂 `WarmResetSpec.from_config` 给出 `seed_namespace=None`、`seed_keys=()`，与新增的「缓存臂不得带 seed 配置」一致；自产臂 seed 键须在期望身份内 |
| B3 GR00T 两入口 `_check_batch` 在任何 head 调用前核 batch | 接受 | 字段名与 `GrootStage2Output`（`backbone_features` / `attention_mask` / `action_inputs[state, embodiment_id]`）一致；执行体传入的起点 / 噪声均为单位 batch；真模型 GR00T K=8 对等测试通过（§15.4） |
| 新入口 `exp/warm_reset/{plan,conductor,admit,run}.py` | 一处缺陷已改（15.2 D1），其余接受 | 逐项核对：LIBERO 决策 `step_idx` 自首个真实决策起为 0、replan、2·replan…（`examples/libero/main.py:_run_episode` 的 `num_steps_wait` 之后归零），RoboCasa 同为 0 起步长 replan，入口的步距完整性检查成立；两 runner 的 `episode_start` 身份字段、journal 只记终局、重试 per_step 行按 attempt 过滤、`load_cache_config` 的 `yaml_content` 落临时文件不删（`yaml_sha256` 可算）均与代码一致 |
| `tests/exp/warm_reset/test_entry.py` | 补一条正控（15.2 T1） | 交付的入口测试只有负例；审查方所述「真实 TCP 端到端」探针不在交付测试集内，`admit()` 的成功路径在可提交的测试里没有覆盖 |
| 文档 `docs/cache/warm_reset_experiments.md`、架构 / 教程补段、两个 README 行 | 接受（附一条使用说明，15.6） | 命令逐条核对：GR00T 两入口的 `--cache-config / --concurrent / --allow-dynamic-bundles / --denoising-steps` 均存在；π0.5 `serve_policy` 的 `--concurrent --cache-config` 由冒烟实测可用（§15.5）；`tasks` 子命令在 `~/libero_sim`（py3.8）下实测可跑 |

### 15.2 执行方修复（工作树，未暂存）

- **D1（缺陷）`exp/warm_reset/conductor.py::worker_agent`**：原写死 `env={"MUJOCO_EGL_DEVICE_ID": "0"}`，而 `_default_spawn` 设 `CUDA_VISIBLE_DEVICES=<gpu>`。robosuite（`~/libero_sim/.../robosuite/utils/binding_utils.py:29-35`）断言 `MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES`，因此 `--gpus` 只要不是 `0`（timan107 / timan108 的多卡 agent 正是这种情形），每个 LIBERO worker 在导入时即断言失败。改为 `env={"MUJOCO_EGL_DEVICE_ID": gpu}`，与既有 `exp/ablation_study/cache_size/run_size_eval.py:557` 同一约定（RoboCasa spawn 不读 `spec.env`，不受影响）。测试 `test_workers_use_existing_islands_with_correct_policy_knobs` 加断言：每个 worker 的 EGL 设备号等于其 `gpu_id`。
- **T1（缺测）`tests/exp/warm_reset/test_entry.py::test_complete_run_is_admitted_with_measured_nfe`**：CPU 正控，覆盖 `admit()` 成功路径：`prepare` 冻结计划 → `build_driver` 写 `execution.json` → 按派发的 `EpisodeTask` 用真实 `build_pi05_warm_reset` + `InferenceInterceptor` + 证据 wrapper 跑每集 3 个决策（`episode_start` 元数据用标准 LIBERO runner 的 `_episode_extra_metadata`）→ 按 driver 盖章格式写 journal / per_step → `admit()` 全部准入；`warmreset_t0.2` 实测总 NFE 12、`selfresetfinal_t0.2` 72（2 集 × 3 决策 × N=2 / K+N=12），成功数各 1。
- 两处改动 `ruff check` 通过；`tests/exp/warm_reset/test_entry.py` 36 passed。

### 15.3 §6 Verify（CPU-only 全量，2026-09-25 12:09–12:13 CDT）

命令：`CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run --no-sync --with pytest-xdist --with pytest==9.0.2 pytest -n 24 --dist loadfile --durations=30 -q`（日志 `/tmp/wr_smoke/verify.log`）。

结果：**7268 passed、94 skipped、21 failed、49 errors（3 个文件的收集 / 夹具错误）**，236 s。本计划的全部测试（`tests/cache/warm_reset/**`、`tests/cache/groot/test_warm_reset_groot.py`、`test_import_isolation.py`、`tests/exp/warm_reset/**`）无失败。21 项失败逐一归类，无一与本计划相关：

- 既有 4 项：`test_ws2_evidence_runner::test_hit_meta_rows_identical_across_runners` ×2；`test_groot_concurrent_serving::test_frozen_commands_pass_the_new_guards` ×2（全量污染，单跑通过）。
- 既有 2 项：`tests/exp/test_prebuilt_matrix_backend.py::test_cosine_fast_path_bit_identical`、`::test_fast_path_robust_to_candidate_reordering`（gather 与 stack 的 `torch.equal` 不成立）。已在 `git archive HEAD` 的干净导出（`5483183`）上以同一 venv 复跑，同样 2 failed，确认与本次改动无关（该测试及其导入的 `in_memory_backend.py` / `prebuilt_matrix_backend.py` 相对 HEAD 均未改）。§14 所称「6 项既有基线」即这 6 项。
- `tests/review_tests/` 15 项（只报 id，不读内容）：`test_cache_size_g2.py` ×3、`test_groot_robocasa_g2.py::test_run_one_closes_env_when_inference_fails`、`test_keybuilder_g2_independent.py::test_existing_script_cli_without_pythonpath`、`test_n1_serverside_stage1c_g2.py::test_warm_start_downgrade_feeds_final_miss_verdict_to_gate`、`test_rl_router_g2_contracts.py` ×2、`test_tracer_phase7_g2_report.py::test_polished_plan_and_roadmap_have_nonstale_status`、`test_warmstart_w1_g2.py::test_certification_rejects_capture_in_measurement_trace`、`test_ws2_g2_round1.py` ×4、`test_x0_g2_review.py::test_kitchen_selects_completed_set_before_order`。数量与 §14 实现方全量所记 15 项一致。
- 收集 / 夹具错误（既有）：`tests/robocasa365/test_bench_groot_stages.py`（24 条，模块缺属性）、`tests/review_tests/test_review_cache_prune_g2.py`（24 条，`KeyError: 'REVIEW_SCRATCH'`）、`tests/review_tests/online_rit_g2/test_round2_boundaries.py`（import file mismatch）。
- 终稿复跑（文档 / 日志改完后，13:06–13:11 CDT，同一命令，日志 `/tmp/wr_smoke/verify2.log`）：**7269 passed、94 skipped、21 failed、49 errors**；失败集合与首跑逐项相同，错误文件相同（多收集的 1 项来自新增日志文件）。

### 15.4 manual GPU 对等（B=1 逐位；本计划唯一的真实模型逐位证据）

- **GR00T LIBERO K=8**（孤岛 B `~/gr00t_n15_venv` + gr00t worktree 于 `PYTHONPATH`；`SD_ENV_ID=groot_libero_10`，`SD_CKPT=~/ckpt_n15_libero_10`，`SD_OBSERVATION=/tmp/sdiag/obs/obs_groot_libero_10.npz`，`--run-manual -rs`）：**1 passed、0 skipped**，82 s（12:14–12:16 CDT）。覆盖 `LIBERO_SELF_ARMS_BY_POLICY["groot"]` 全部 `_n1/_n2` 缓存与自产臂：`run_groot_continuation` 与 `groot_warm_variant_stage3` 逐位相等且 `steps_run == N`，自产起点与 step_diag `groot_self_start` 逐位相等且实测步数 = 8。
- **π0.5 K=10**（主 `.venv`；`SD_ENV_ID=pi05_libero_10`，`SD_CKPT=~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch`，`SD_OBSERVATION=/tmp/sdiag/obs/obs_pi05_libero_10.npz`）：**1 passed、0 skipped**，188 s（12:16–12:19 CDT）。T ∈ {0.1, 0.2, 0.3} × `WARM_VARIANT_MODES ∪ SELF_VARIANT_MODES` 全部臂：新入口与 step_diag 函数逐位相等；桶内单请求的 coordinator adapter 与直连逐位相等；自产直推快照 / 最终动作逐位相等。B=4（续跑桶 + 自产桶）路由、逐行抓取与计数断言通过，数值偏差只记录：**max|Δ| = 5.36e-3**（对逐行 B=1）。
- **未跑：GR00T RoboCasa K=4**。本机没有 RoboCasa 的生产 wire 观测 npz（`/tmp/sdiag/obs/` 只有 LIBERO 两个 suite），K=4 的真实模型逐位声明仍缺证据；CPU 桩对等覆盖不变。
- 运行前均核 `nvidia-smi` 空闲显存（均为 11.96 GB，≥ 需求 + 4 GB 余量；实测峰值总占用 GR00T 42.5 GB、π0.5 44.4 GB / 49.1 GB），未影响其他进程。

### 15.5 §10.3 真实 serving 冒烟（π0.5 LIBERO，weilandserver 4090）

按 `docs/cache/warm_reset_experiments.md` 的命令走完 `tasks → prepare → run → agent → admit`，真实 checkpoint（`pi05_libero_pytorch`）、真实库（`exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl`，基于 `exp/step_diag/config/arms/pi05_libero_10/warm_t0.2.yaml`，仅把 `preload_path` 改为绝对路径）、真实 LIBERO 模拟器（`~/libero_sim`，EGL）。libero_10 task 0（`put both the alphabet soup and the tomato sauce in the basket`），原始 init 0 / 1，seed 7，replan 5。server `scripts/serve_policy.py --concurrent --port 23120 --cache-config …`（单 replica，`OPENPI_SERVER_GPU_MEMORY_LOCK=0`），driver 端口 23121、`--concurrency 3`，agent 2 个 worker（GPU 0）。GPU 时段：第一轮 12:50–12:57 CDT，第二轮 12:58–13:05 CDT（server 常驻约 8.0 GB）；此前 12:19 起的一次启动在收到协调方「`mwsmoke` 运行期间暂停 GPU 步骤」通知后未分配显存即停止，12:50 `mwsmoke` 结束后再开。

| 轮次 | 臂 | 集 | 成功 | admit | 决策数 | 实测续跑 NFE | 实测自产 NFE | 总 NFE |
|---|---|---|---|---|---|---|---|---|
| 1 | `warm_t0.2`（精确，worker_reference） | 2 | 2 | 准入 | 55 / 91 | — | — | null（按设计） |
| 1 | `warmreset_t0.2` | 2 | 2 | 准入 | 54 / 64 | 108 / 128 | 0 | 236 |
| 1 | `selfresetfinal_t0.2` | 2 | 2 | 准入 | 58 / 64 | 116 / 128 | 580 / 640 | 1464 |
| 2 | `warmreset_t0.2` | 2 | 2 | 准入 | 97 / 64 | 194 / 128 | 0 | 322 |
| 2 | `selfresetfinal_t0.2` | 2 | 2 | 准入 | 54 / 55 | 108 / 110 | 540 / 550 | 1308 |
| 2 | `selfwarmreset_t0.2` | 2 | 2 | 准入 | 61 / 63 | 122 / 126 | 610 / 630 | 1488 |

- 两轮 driver 均 `episodes_done 6 / failed 0`、退出码 0；`admit` 两轮均 `ok: true`、退出码 0、无 global problems。
- 服务端逐决策（warm reset 臂决策行第一轮 240、第二轮 394，全部 `status: ok`、`hit_type: WARM_START`）：`continuation_nfe = 2`、`n_stage3_calls = 1`、`t = [1.0, 0.5]`；自产臂 `self_direct_nfe = 10`、`self_start_calls = 1`、`decision_nfe = 12`；缓存臂 `decision_nfe = 2`、不带自产三字段；每集恰一条 `finalize`（`terminal: true`，`outcome` 与 journal 一致）。自产 seed 由准入按受信身份重算通过。
- 共同随机数：第二轮两个自产臂在同一 (episode, decision) 上共有 109 个决策，seed 全部相同。
- 合批（第二轮 `OPENPI_MONITOR_LEVEL=BASIC`，`dump_metrics`）：`3bucket` 记录出现 `warm_reset` 桶，`grid_key = ["pi05_v1", 10, "reset", 1.0, 2, null]`、`steps_run = 2`（386 个单请求桶 + 4 个 size 2 桶）与 `warm_reset_self` 桶，`grid_key = ["pi05_v1", 10]`、`steps_run = 10`（225 + 4 个 size 2 桶）；缓存 / 自产、快照 / 最终动作臂共用同一续跑网格键，不同抓取位置的自产臂共用自产桶，符合 §4.5.2 / N2。
- 延迟不代表性能：服务端每决策 wall 中位 723 ms、p90 1225 ms（第一轮），GPU 同时被 6 个 step_diag GR00T server 占用（MPS 共享、利用率 57%）；本冒烟不作吞吐结论。
- 清理：只停自己起的 tmux（`wr_srv23120` / `wr_drv23121` / `wr_agent`），逐一核对 PID 已退出、端口 23120 / 23121 已关闭，显存回到启动前的 36.55 GB。冒烟产物在 `/tmp/wr_smoke/`（运行目录 `run1` / `run2`、服务端证据 `server_ev/`、`run2_3bucket.json`、各日志）。

### 15.6 未覆盖与使用说明

- 未做：GR00T RoboCasa K=4 manual 对等（缺 RC wire 观测）；GR00T serving 冒烟（`serve_groot_libero --concurrent --allow-dynamic-bundles --denoising-steps 8` + `warm_t0.875` / `midreset_t0.75_n1` / `selfmidreset_t0.75_n1`）；RoboCasa 入口冒烟。三者都应在迁移任何 GR00T / RoboCasa 实验之前补上（见 `logs/warm_reset_migration_study.log.md` §8 S1）。
- 文档补充（执行方）：`docs/cache/warm_reset_experiments.md` 原只给 conda 形式的 LIBERO agent 命令；本机与 timan107 的 LIBERO 模拟器是 venv，`conda run -p <venv>` 不可用。补了一段 venv 用法（主 venv 解释器跑 agent，`PATH` 先指向模拟器 venv、`PYTHONPATH` 含仓库根与 `src`），即本次冒烟实际使用的方式；`docs/README.md` 该行描述同步。
- 冒烟同时验证了：`tasks` 子命令在 py3.8 模拟器 venv 下可跑；`prepare` 对真实 step_diag 基线 yaml 的整份 dataclass 往返（生成的 YAML 物化了全部缺省字段，语义与基线相同，只多 `warm_reset` 块）可被 server 热加载。

## Review Log


### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-09-25 11:26 CDT

审查对象：本计划获批设计对应的工作树实现（基于 HEAD `5483183`），不包含并行的 step_diag / trace_dual 工作。Authority: Review。owner 已授权审查后暂存开发者版本，再直接修复，审查方的修改全部保留在暂存区外；本轮依该授权执行。

- [Blocking] [Concern] B1：证据准入未核对实际描述的起点和网格。`evidence._decision_content_problems` 对 `source/point/kind/level/t/dt` 缺失或错配均放行；将缓存 reset 行改为 self / final / shoot、把 dt 改为 -0.1，仍返回空 problems 和完整 NFE。`family` 错配或 `status=ok` 同时有 error 亦可通过。—— reasoning: §4.7 要求必要字段、类型和规格身份闭合；只比较 spec_digest 不能证明行内描述与受信规格一致。独立探针已复现。
- [Blocking] [Concern] B2：坏类型没有稳定拒收。`conn_id=[]` 或 `episode_seq={}` 在加入 sessions 集合时抛 TypeError；`ExpectedEpisode.identity=None` 也直接抛错；expected.attempt 与 expected.identity.attempt 冲突时仍可准入一套同样矛盾的行。—— reasoning: §4.7.3 明确要求 invalid_field / invalid_expected，分析器需要逐集拒收而非被坏行中断；受信输入自身也应先校验一致性。独立探针已复现。
- [Blocking] [Concern] B3：GR00T 两个公开入口缺少 batch 校验。stage2 B=1、start/noise B=2 时，已执行 head.process_backbone_output 才在张量运算中失败。—— reasoning: §4.3 的公共入口要求 B≥1 和输入 batch 一致；应在任何 head 调用前拒绝，不能依赖下游广播或拼接报错。独立探针已复现。
- [Non-blocking] [Concern] N1：manual GPU 对等与 §10.3 真实 serving 冒烟未执行，不能声称真实 checkpoint 逐位对等或给出吞吐结论。CPU 8 连接 / 4 bundle、真实 BatchingCoordinator、三次决策的独立集成探针通过（精确续跑 + cache snapshot / final + self final），涵盖跨 bundle 合批与证据准入，但没有模拟器或 GPU。
- [Non-blocking] [Question] N2：owner 本轮追加“直接用 concurrent server 和 conductor 跑 warmreset 家族一批臂”。原 §2/Q6 把 arm emitter、专用 strategy 和正式分析器列为后续。现有 `run_size_eval` 的 always_hit 守卫拒绝 warmreset；`run_ablation_eval` 要求 routing；GR00T LIBERO `run_conductor` 要求 warm sweep index。这些现成 CLI 不能无配置适配直接替代本族实验入口。已请求明确此次验收是否包含实验入口，框架能力与现成实验命令分开验收。

Checklist：
1. 计划符合性：NEEDS REVISION。执行／合批／注入边界符合；B1–B3 未完成约定的拒收与入口守卫。§14 的 trace 路径字段登记、RC 缺失 import 修复和会话辅助接口属于必要且局部的偏差，接受。
2. 测试：NEEDS REVISION。独立复跑新增 655 passed / 2 manual skipped；受影响套件 3816 passed / 60 skipped / 4 failed。4 项为报告中的既有失败：两项 ws2 行字段断言（相关源码和测试与 HEAD 相同），两项 gr00t.__spec__ 测试污染（单独重跑通过）。新增独立边界探针 20 项失败；另 8 连接 / 4 bundle 集成探针通过。
3. 文档与索引：通过。架构 §5.22、coordinator、教程与 docs/logs 索引均更新；本轮审查及后续修复另记并保持未暂存。
4. 回归：已有 CPU 证据未发现旧精确续跑行为变化；B1–B3 修复后仍须复验。judge、orchestrator、worker、conductor、原 staged 模型实现不修改。

暂存边界：已仅暂存本计划的开发者实现、测试和文档，排除 `tests/review_tests/` 及其他实验修改。以下 owner 授权修复及最终复验记录均留在工作树，不更新该索引快照。

### Owner 范围补充与修复方案 — 2026-09-25 11:36 CDT

owner 回答「本轮补齐可直接运行的实验入口」，覆盖原计划 Q6 的后续工作边界。本轮增加 `exp/warm_reset/` 独立入口：由已有基准环境/臂定义生成静态 YAML 与运行清单，经标准 concurrent server 动态 bundle + conductor 调度 LIBERO / RoboCasa、π0.5 / GR00T 多臂；复用标准 worker，不修改 conductor、worker 或原 step_diag 实验逻辑。

入口分为 prepare / run / agent / admit：prepare 固化臂规格、YAML 内容及哈希、任务语言/名称与 init-state 身份、调度端点及 rollout 参数；run 使用独立且不可复用的运行目录，记录实际任务图、driver run_id、journal 与 accepted per-step；admit 从这些受信资料构造 ExpectedEpisode，对服务端证据逐集准入并汇总实测 NFE。路径和资源由操作者参数提供，不启动 GPU 服务或替用户选择正式实验数据。LIBERO 与 RoboCasa 的 worker 参数分别复用既有 spawn 路径。

验证追加：CLI 生成及错误参数测试、调度/两种 worker 参数映射、伪造/缺失/重复/旧 attempt 的准入拒收；真实本地 TCP conductor + concurrent WebSocket + CPU 模型的多连接、多 YAML、episode 生命周期闭合。GPU checkpoint 与真实模拟器仍保留为部署环境验证，不声称已运行。新增和修复继续保持 unstaged，开发者 snapshot 不再更动。

### G2 Round 2 — Reviewer — APPROVED — 2026-09-25 11:58 CDT

本轮依据 owner「暂存开发者版本后直接修复直到可以放行；我的修改保持 unstaged」及「本轮补齐可直接运行的实验入口」的明确授权完成。此授权覆盖本轮审查方直接修复与扩展实验入口的流程例外，不改动并行 step_diag / trace_dual 工作。

**阻塞项闭合**：
- B1 已修复：证据准入核对必要字段、source/point/kind/level、完整 t/dt（GR00T 同核 tau）、family 与成功行的 error 状态；列表/字典递归类型严格比较，不能把 bool 当 int。π0.5 dt 按 float32 实际值重建。
- B2 已修复：ExpectedEpisode 与 identity/spec 的类型及身份一致性先校验；坏 conn_id/episode_seq 不再触发 unhashable 异常，按 invalid_field 拒收。拒收集不产出可用 NFE。
- B3 已修复：GR00T continuation/self 入口在任何 head 调用前检查 B≥1，起点/噪声、backbone_features、attention_mask、state 和 embodiment_id 的 batch 一致。
- N2 依 owner 范围补充闭合：新增 `exp/warm_reset/{plan,conductor,admit,run}.py`，CLI 为 tasks / prepare / run / agent / admit。静态臂 YAML 对齐既有环境/臂表，支持 π0.5 K=10、GR00T K=4/K=8、LIBERO spatial/10 和 RoboCasa；通过生产动态 bundle 与标准 WorkerAgent/runner 执行，核心 conductor/worker 无修改。RoboCasa GR00T 映射为标准 `groot_tp` teacher；LIBERO resize 分别为 224/256。

**入口的证据契约**：prepare 固化任务语言/名称、原始 init 索引、rollout 参数、端点、YAML 原文及哈希；run 将完整 manifest 哈希、实际任务图及 driver run_id 写入 execution，再持久化 journal/per-step。admit 只选择 accepted terminal 的 attempt（允许保留旧的 accepted 非终局重试记录），以该 attempt 的 worker 决策数和受信清单构造 ExpectedEpisode，再核服务端所有同行。任务缺失、清单漂移、重复/坏行、身份/计数不符均拒收。普通 exact warm 基线明示 worker_reference，NFE 为 null；不伪装成服务端实测证据。每次 prepare 使用独立 token/bundle/evidence 目录，拒绝覆盖旧运行，跨 driver 崩溃重启需新运行目录。

**§4 检查表**：
1. **计划一致性：PASS**。保留获批框架接口及无块时原执行路径；三项修复补足原计划边界。实验入口是本轮 owner 明确追加的 Q6 范围，置于独立 exp 包，不侵入生产机制。
2. **测试覆盖与通过：PASS（已列明环境边界）**。最终定向 CPU/边界/装配/数值/隔离/GR00T staged/入口集：756 passed、2 skipped，`/tmp/warm-g2-delivery-tests.log`。新增入口的最终重试与 manifest 哈希校验及真实 TCP 测试：36 passed，`/tmp/warm-g2-entry-and-wire-final.log`。后者以真实 WebSocketPolicyServer concurrent factory、YAML 热加载、ConductorDriver 和 8 个 WorkerLoop 运行 4 bundle × 2 episode × 3 decision，并由新 admit 命令使用真实 journal/per-step/server JSONL 闭合；cache reset 总 NFE=12，自产 final 总 NFE=72，exact reference 不报实测 NFE。只有模型、缓存内容和模拟器 rollout 用 CPU 替身。新增 exp 文件/测试的 Ruff check 及 `git diff --check` 通过。
3. **文档与索引：PASS**。新增 `docs/cache/warm_reset_experiments.md`，提供两类模型、两个基准的服务/prepare/driver/agent/admit 命令及边界；tutorial/architecture 和 docs/logs README 已同步。
4. **回归：PASS（无新增失败）**。受影响子系统全套复跑：3884 passed、60 skipped、4 failed，`/tmp/warm-g2-final-regression.log`。4 项与开发者版本及独立首轮完全相同：`test_ws2_evidence_runner::test_hit_meta_rows_identical_across_runners` ×2（HEAD 的旧白名单断言）和 `test_groot_concurrent_serving::test_frozen_commands_pass_the_new_guards` ×2（全套测试污染使 gr00t.__spec__ 为 None，单独通过）。仍按原运行排除 HEAD 即收集失败的 `test_bench_groot_stages.py`。全套结束后最后两项 exp 边界增补以所述 36 项最终测试覆盖；未重复无关套件。

- [Non-blocking] [Concern] N1 保留：两份 manual GPU 对等、真实 checkpoint 与模拟器端到端冒烟未运行；本次不声称真实 GPU 逐位对等、吞吐或实验成功率。GR00T production concurrent 仍使用既有 infer-lock，不把多连接解释为 GR00T 模型合批。
- [Non-blocking] [Concern] 实验入口当前使用普通未固定对象的 RoboCasa rollout，不隐式启用 pinned-object 协议；准备库/任务/路径与上述约束不匹配时应调整明确的运行参数，不能绕过准入。

**暂存边界**：开发者原始 37 路径快照已暂存，cached binary diff sha256 始终为 `8cd98f3fc2a4d63837b427db62fc89b05da4710b2e537b33f109a6924b931115`。B1–B3 修复、新增实验入口/测试、文档和本 Review Log 均留在暂存区外；私有 review_tests 未进入 index。未 commit / push，未运行 GPU 实验。

**Verdict: APPROVED**（当前工作树，包含 owner 授权的本轮修复与新增入口）。

### G2 Round 2 — Executor — 2026-09-25 CDT

G2 已 APPROVED；以下为执行方对 G2 两轮全部审查项的逐项回应（owner 常设规则：审查方直接修改须经执行方复核），复核细节、Verify、GPU 对等与冒烟记录见正文 §15。执行方改动均留在工作树、未暂存；暂存区保持审查方快照不变。

- Round 1 B1（证据准入未核起点与网格）：Accepted — 审查方修复经复核正确：π0.5 `dt` 以 float32 取整重建，与执行体写入逐位一致；`t` / GR00T `tau` / `dt` 与执行体同源（`plan.flow_times()` / `native_grid(plan)`）；正控全部准入、负控全部拒收。未再改动。
- Round 1 B2（坏类型未稳定拒收、受信期望未自检）：Accepted — 复核确认缓存臂 `seed_namespace=None / seed_keys=()` 满足新自检，坏 `conn_id` / `episode_seq` 记 `invalid_field` 而非抛异常。未再改动。
- Round 1 B3（GR00T 入口缺 batch 校验）：Accepted — `_check_batch` 字段与 `GrootStage2Output` 一致，先于任何 head 调用；真模型 GR00T K=8 manual 对等通过（§15.4）。未再改动。
- Round 1 N1 / Round 2 N1（manual GPU 对等与真实 serving 冒烟未执行）：Accepted — 已执行：π0.5 K=10 与 GR00T K=8 manual 对等均 1 passed / 0 skipped，B=4 偏差 max|Δ| = 5.36e-3 仅记录（§15.4）；π0.5 LIBERO 真实 serving 冒烟见 §15.5。GR00T RoboCasa K=4 manual 对等与 GR00T serving 冒烟未跑（本机无 RC wire 观测；GPU 被在跑实验与协调方冒烟占用），真实模型 K=4 逐位声明仍缺证据。
- Round 1 N2（实验入口是否在本次验收内）：Accepted — owner 已扩大范围；入口经逐项复核发现一处缺陷并已修复：`worker_agent` 写死 `MUJOCO_EGL_DEVICE_ID="0"`，与 `CUDA_VISIBLE_DEVICES=<gpu>` 组合时 robosuite 断言失败（任何非 0 号 GPU 上的 LIBERO worker 启动即死），改为按 worker 的 `gpu`（§15.2 D1）；另补 `admit()` 成功路径的 CPU 正控测试（§15.2 T1），因交付测试集中只有负例。
- Round 2 Concern（入口 RoboCasa 未启用钉物体协议）：Accepted（作为已知限制记录，不在本计划内实现）— 复核确认 strategy 不写 pin 三键、agent 不传 `--pinned-objects`，因此 step_diag RC 的 PnP 线（钉物体）目前不能经入口复现；补法与工作量见 `logs/warm_reset_migration_study.log.md` §6、§8 S5。
