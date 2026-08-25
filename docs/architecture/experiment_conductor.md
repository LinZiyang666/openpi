# 实验编排框架（Experiment Conductor）

> 通用、与实验无关的 **episode 级编排引擎**，支持跨 GPU / 机器 / server 的大规模评测。
> 机制（mechanism）在 `src/openpi/conductor/`，策略（policy）在 `exp/`。
> 设计决策与评审记录见 [`logs/archive/client_conductor_two_layer_refactor.log.md`](../../logs/archive/client_conductor_two_layer_refactor.log.md)。

---

## 1. 动机

旧 client（`examples/libero/main.py` 进程内线程 + `exp/` 下各 driver 脚本）有四个问题：派发粒度是 yaml→subtask，后期 worker 闲置形成**等待泡沫**；断点续跑 / 重试 / 监控在各脚本各自简陋；进程内线程模型无法跨卡跨机；调度与实验语义耦合、无法复用。

本框架把调度机制与实验语义彻底分离，并把派发粒度降到 **episode 级**。

## 2. 三层架构

```
driver (1 个,中枢)         src/openpi/conductor/driver.py
  • EpisodeScheduler 中央 episode 队列 + 亲和贪心
  • TaskGraph 执行引擎 (stage 生命周期 begin/complete)
  • Journal 账本 / 断点续跑    • Retry 分类 + HealthAggregator + Monitor
  • 到各 server 的 ctl 连接池   • 装载一个 ExperimentStrategy
      ▲ pull EpisodeTask / ▼ report(progress|result)   (worker→driver: TCP, msgpack)
agent (每机常驻)            src/openpi/conductor/agent.py
  • 按 GPU/EGL slot fork & 监督 worker 进程,死则重启 (仅生命周期,不转发消息)
worker (进程,绑 1 GPU)      src/openpi/conductor/worker.py
  • WorkerLoop: pull → EpisodeRunner.run → report  (直连 driver pull 端口)
  • EpisodeRunner 接口; LiberoEpisodeRunner 复用 main._run_episode
      ▲ 原 WebSocket 推理协议(不动)
server1 / server2          (M2 多 bundle + WarmupPool, 已支持)
```

> **实现简化（vs plan §3/§5.3）**：worker **直连 driver** 的 pull 端口；agent 只在本机 fork+监督 worker（不做 plan 原图设想的 worker↔agent↔driver 消息转发）。agent 自身崩溃时，driver 通过连接断检测回队其 worker 的 in-flight episode，无消息丢失。这比两段链路更简单，且功能等价。

- **worker 无脑**：只接 `EpisodeTask`、执行、回报，不持调度策略、不解释 `phase`。
- **server 协议不动**：复用现有 `infer` / `__ctrl__`（`load_cache_config` / `select_bundle` / `fetch_dump` / `preload_normalizer_buffer` / `unload_warmup_buffer`）。

## 3. 机制 / 策略分离

| | 核心机制（`src/openpi/conductor`） | 策略（`exp/`，`ExperimentStrategy`） |
|---|---|---|
| 职责 | 调度 / pull / 亲和 / 账本续跑 / 重试 / 监控 / ctl 池 | 每个 yaml 的 stage 序列、barrier 处做什么、episode 任务构造 |
| 是否知道 warmup/eval | **不知道**，只认 stage 依赖 + 生命周期钩子 | **知道**，warmup/fetch/preload/eval 都在策略里 |

`src/openpi/conductor/` **不得** import `exp.*` 或 LIBERO；实验语义经 `ExperimentStrategy` / `EpisodeRunner` 接口注入。

## 4. 核心数据结构（`task.py`）

- `EpisodeTask`：派发单元。`task_uid` 确定性派生 `f"{yaml_id}:{phase}:{task_id}:{episode_idx}"`（续跑幂等匹配账本）。`extra` 承载自由 per-task 元数据；**gate 采集 producer contract**：经 `LiberoEpisodeRunner` 运行的 strategy 必须 stamp `extra["num_trials_per_task"]` 为本 stage 的 per-phase trial 数（warmup/eval 不同），runner 据此推 canonical `episode_id`、缺失即 fail-fast（不回退 worker 默认值），使 conductor 与 standalone 的 id 一致。
- **gate 采集回传**：worker 侧精简采集（默认 `robot_state`）作为额外 key 内联进 `EpisodeResult.per_step_rows`，随 result 经 msgpack wire 中央回传 driver（**无 NFS、无 protocol version 变更**）；vision 因单 episode 帧 vs 64 MiB 上限而 standalone-only。详见 [数据收集指南](../data_collection/guide.md#gate-research-per-step-collection-distinct-from---collect)。
- `Stage`：一组 episode + `phase` + 所属 `server` + 可选 `produces_calib_id` / `consumes_calib_id`。warmup 的重试/续跑原子单位是 stage。
- `TaskGraph`：`Stage` + `StageDependency` + `CalibrationArtifact`；`validate()` 拒绝悬空 calib 引用与依赖环。
- `StageContext`（`strategy.py`）：线程安全黑板，承载 warmup→eval 的 calibration buffer 交接。
- `CalibrationArtifact`：把"校准数据"与"产出它的 warmup stage"解耦，支持**共享**（一对多消费者）与**历史源**（`source=historical_file`）。`cleanup_id` 约束 warmup dump 命名为 `<cleanup_id>__warmup`，使清理走现有 `unload_warmup_buffer(cleanup_id)`，**不改 server 协议**。

## 5. 调度算法（`scheduler.py`）

- **归属（静态）**：`assign_servers` 把每个 yaml 整体分配到一个 server（一个 yaml 的全部 stage 同 server，因 `WarmupPool` 是 server 进程级状态），按 episode 总数均衡，共享 calib 的 yaml co-locate。
  - ⚠ **这是默认放置，不是硬约束**。策略层的 `plan()` 决定 stage 集合，核心从不要求「一个 yaml 一个 stage」。
    `src/openpi/conductor/sharding.py` 的 `shard_eval_stage` 把**一个 eval yaml 摊成每台 server 一个兄弟 stage**，
    于是**单臂相位也能吃满整池**（否则它只用 1/N 容量——实测占 libero_spatial 全程的 25.2%）。
    活化上限按 `(server, phase)` 计数、`make_task_uid` 不含 server，所以这既不与上限冲突、
    也不破坏 resume 幂等（分片数变了照样按 uid 命中 journal）。
    **只对 eval 有效**：warmup stage 发布标定产物，分片会让 N 次 `fetch_dump` 各拿 1/N、
    N 次 `ctx.publish` 互相覆盖，且**不会报错** —— helper 因此对 warmup 直接 raise。
- **激活（动态）+ 亲和**：per server 限制同时激活的 yaml 数——**warmup 放松**（默认 ≤2，填 barrier 空隙）、**eval 收紧**（默认 1，省显存）。
- **永不空转**：该 server 有任意 ready episode 时 pull 必返回。
- **barrier 门控**：downstream stage 在 upstream 全 done + 其 `on_stage_complete` 返回前保持 blocked。

## 6. stage 生命周期与数据流（`driver.py` + `strategy.py`）

核心对每个 stage 保证顺序：`upstream.on_stage_complete → downstream.on_stage_begin → downstream episodes ready → done → downstream.on_stage_complete`。`begin/complete` 在**独立线程**执行，不阻塞 worker pull（plan §6.4）。warmup→eval 典型流：

1. warmup `on_stage_begin`：`load_cache_config(warmup_yaml)`（driver 在此**前**已 `unload_warmup_buffer(cleanup_id)` 清旧 dump）
2. warmup episodes 跑 → DumpingJudge 落盘
3. warmup `on_stage_complete`：`fetch_dump` + 聚合 → `ctx.publish(calib_id, buffer)`
4. eval `on_stage_begin`：**先** `preload_normalizer_buffer` **再** `load_cache_config(eval_yaml)`（config 对缺 WarmupPool fail-fast）
5. eval `on_stage_complete`：`unload_warmup_buffer`

**Fan-out**：消费同一 `calib_id` 的多个 eval stage 各自在自己 server 的 `on_stage_begin` 用共享 `ctx` buffer preload——天然 fan-out 到每个相关 server。

## 7. 断点续跑（`journal.py`，plan §8）

- **账本**：每个终态 episode 追加一行 JSONL；重启回放，`done` 跳过、`failed` 按重试重入队。
- **warmup 原子性**：warmup 只做 **stage 级** 续跑/重试（重跑前清 `<cleanup_id>__warmup`），避免重复 append 污染 buffer；仅 eval 做 episode 级。
- **server 自愈**（plan §8.3）：现有协议无 WarmupPool 探测帧，故取 **(B) 无条件重建**。续跑（driver 重启）时 warmup stage 本就从头重跑（warmup 不 episode-journal），天然重建；driver 另以 `_resuming`（journal 非空）门控，对 eval stage 在 `on_stage_begin` 前调 `strategy.on_resume` 清旧 server pool。**一阶用 "journal 非空 == 可能失效" 近似 §8.3 的失效判定**（正常首跑不触发）。

## 8. 重试与健康

- **重试分类**（`is_retriable_error`）：网络/超时/crash → 可重试；`ConfigValidationError` 等致命 → 不重试、标记 yaml 失败。
- **断连回队**：worker crash → driver 连接断 → 该连接 in-flight episode 自动回队（eval 回队 / warmup stage 作废）。这是本 build 的主失联检测。
- **agent 监督**：worker 进程死则本机重启（`poll()` 检测进程退出）。
- **episode 墙钟超时**：driver `requeue_timed_out` 周期回收 dispatch 超 `episode_timeout_s`（默认 1800s）仍未回报的 episode——**卡在 infer 不退出的 worker 也被回收**（plan §9.2；eval 回队、warmup 整 stage 作废）。
- **stale-result fence**：`EpisodeTask`/`EpisodeResult` 带 `attempt`，scheduler 每次 dispatch 递增该 uid 的 generation；超时回队 + 重 dispatch 后，旧 worker 迟到的 result（低 attempt）被 `mark_result` 拒绝，不会污染当前 dispatch。剩余 server-side dump 残留（旧 worker 在 rerun 后 append）作为已知边角（1800s 超时下旧 worker 实际已卡死、其 result 已被 fence）；完整 dump 隔离列为后续。
- **健康观测**：`HealthAggregator` 汇总 per-worker 状态/吞吐，`stale_workers`（基于 progress/result 的 `last_ts`）作 best-effort 失联提示。无独立 heartbeat wire，主动心跳由连接活性 + episode 超时替代。

## 9. 监控（`monitor.py`）

`Monitor` 渲染**聚合视图**（全局 done/total/SR + worker 运行数 + 总吞吐），而非 96 行 tqdm（head-less 友好）；per-worker 明细经 `health.snapshot()` 按需取。

## 10. 部署约束

- **server 端点可为 `--replicas` 单公共端口（router）或独立单进程端点**：`replica_proxy` 已把 `fetch_dump` 改为 **aggregate**（fan-out 到所有 child + 拼接各 replica 的 warmup dump 切片，见 `merge_dump_replies`），故 warmup→eval 的 dump 经 router 也完整，`--replicas N` 单公共端口对 conductor 透明（driver 注册一个 endpoint）。仍可注册多个独立 `--concurrent` 端点用于按 server 细粒度分配 worker。
- **EGL 上限**：agent 按 (机器, 卡) 配额 fork，沿用单卡 ≤15 worker。

## 11. 模块与测试

| 模块 | 职责 |
|------|------|
| `task.py` | 数据结构 + TaskGraph |
| `protocol.py` | msgpack-over-TCP wire（length-prefix framing + `protocol_version`） |
| `scheduler.py` | 调度状态机 |
| `journal.py` | 账本续跑 |
| `strategy.py` | `ExperimentStrategy` ABC + `StageContext` |
| `worker.py` | `EpisodeRunner` ABC + `WorkerLoop` |
| `driver.py` | 引擎主循环 + pull 服务 + ctl 池 + 归属/重试 |
| `agent.py` | worker fork / 监督 / 重启 |
| `health.py` / `monitor.py` | 健康聚合 / 聚合渲染 |

测试在 `tests/conductor/`（CI 全 fake，无 GPU）+ `tests/exp/test_warmup_eval_strategy.py`。端到端真 LIBERO + 真 server 为 manual。

## 12. 扩展

- **新实验** → 写一个 `ExperimentStrategy`（`plan` 产出 TaskGraph + `on_stage_begin/complete` 编排控制帧），放 `exp/`。示例：[`exp/verdict_factor_judge/strategies/warmup_eval_strategy.py`](../../exp/verdict_factor_judge/strategies/warmup_eval_strategy.py)。
- **新环境** → 实现 `EpisodeRunner`（`run(task, report)`）。示例：[`examples/libero/episode_runner.py`](../../examples/libero/episode_runner.py)（复用 `main._run_episode`）。
- 调度 / 续跑 / 重试 / 监控基础设施零重写。
