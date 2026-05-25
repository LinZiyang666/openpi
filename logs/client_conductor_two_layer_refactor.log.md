# Client 编排基础设施两层重构 — 实施计划

- **Status**: In Progress（G1 APPROVED / G2 APPROVED 2026-05-25 / §6 Verify done 1646 pass / 文档更新进行中）
- **Level**: L3（跨模块新子系统 + 架构文档）
- **Authority**: Execution（plan 作者）
- **Date**: 2026-05-25
- **关联**: 取代/上移 `exp/verdict_factor_judge/common/run_phase.py`、`phase5/runner.py`、`phase5/g5_warmup_libero10_driver.py` 的调度逻辑；复用现有 server WebSocket 协议（`src/openpi/serving/websocket_policy_server.py`，**不改动**）。

---

## 1. 背景与动机

### 1.1 现状
- **worker 执行内核**：`examples/libero/main.py`。进程内多 worker 是**线程**，共享单卡 EGL 渲染，`--num-workers` 单卡上限 15（`main.py:1001`）。整进程钉死一个 `CUDA_VISIBLE_DEVICES`（`:1143`）。
- **任务分配**：进程内用共享 `task_queue` 抢占（`_eval_concurrent:725-781`），但派发粒度是 **yaml → task_id（subtask）**，由上层 driver（`run_phase.py` 7 步、`phase5/runner.py` 4-mode）按 yaml 串行编排。
- **调度逻辑与实验语义耦合**：warmup→eval barrier、`fetch_dump`、`preload_normalizer_buffer`、yaml 切换全部硬编码在各 driver 脚本里，每个实验重写一遍。

### 1.2 问题
1. **等待泡沫**：一个 yaml 跑到尾声，先做完的 worker 闲置，必须等整个 yaml 结束、进入下一个 yaml 才复工。subtask/yaml 之间有空隙。
2. **无统一基础设施**：断点续跑、报错重试、per-worker 健康/吞吐监控在现有脚本里缺失或各自简陋实现。
3. **单机单卡天花板**：进程内线程模型无法跨卡/跨机扩展；`48+48` 这种 server 级分配无从表达。
4. **调度 ⊥ 策略 不分离**：实验编排无法复用。

### 1.3 目标
- **三物理层**：`worker`（执行）/ `agent`（每机常驻，本地 fork+心跳）/ `driver`（中枢）。
- **driver 内部机制/策略分离**：通用引擎核心 + 可编程 `ExperimentStrategy`。
- **episode 级无空隙调度**：中央队列，worker pull，先做完先取，subtask/yaml 间无缝。
- **yaml 亲和 + 单 server yaml 数最少**（软约束，永不空转优先）。
- **完美断点续跑**（账本 + server 自愈）+ **报错重试**。
- **按 server 分配 worker**（如 48→server1、48→server2）。
- **per-worker 监控**：进度条、吞吐、健康。

---

## 2. 设计原则（解耦）

| 原则 | 落地 |
|------|------|
| **机制在 `src/`，策略在 `exp/`** | 通用编排引擎进 `src/openpi/conductor/`；具体实验剧本（verdict warmup/eval）作为 `ExperimentStrategy` 插件留 `exp/`。 |
| **worker 无脑** | worker 只收带控制信息的 episode 任务并执行+回报，不持调度策略，不知 warmup/eval 之分（仅透传 `phase` 标签）。 |
| **server 协议不动** | 复用现有 `infer` / `__ctrl__`（`load_cache_config`/`select_bundle`/`fetch_dump`/`preload_normalizer_buffer`/`unload_warmup_buffer`）。重构纯 client 侧。 |
| **执行内核可插拔** | worker 进程 = 通用 pull/report 循环 + 可插拔 `EpisodeRunner`；LIBERO 是其一实现，从 `main.py` 抽出。 |
| **barrier 由原语拼出** | driver 核心提供「stage 依赖 + stage 完成回调」抽象原语，warmup→eval barrier 由策略用原语表达，核心不含实验语义。 |

---

## 3. 架构总览

```
┌─ driver (1 个,中枢) ─ src/openpi/conductor/driver.py ─────────────┐
│  • EpisodeScheduler:中央 episode 队列(ready/blocked) + 亲和贪心       │
│  • TaskGraph 执行引擎:stage 依赖 + barrier 完成回调                   │
│  • Journal:账本持久化 / 断点续跑                                       │
│  • Retry + HealthAggregator + 到各 server 的 ctl 连接池               │
│  • 加载一个 ExperimentStrategy(实验剧本)                             │
└───────────────▲ pull EpisodeTask / ▼ report(progress|result|health)──┘
                │  (agent↔driver: 跨机 TCP, msgpack)
┌─ agent (每台 client 机 1 个,常驻) ─ src/openpi/conductor/agent.py ┐
│  • 按 driver 指令在本机指定 GPU/EGL slot fork & 管理 worker 进程      │
│  • 心跳聚合 + crash/超时检测 → 本地重启 worker;转发 pull/report      │
└───────────────▲ (worker↔agent: 本机 IPC) ───────────────────────────┘
┌─ worker (进程,绑 1 GPU 的 1 EGL slot) ─ src/openpi/conductor/worker.py ┐
│  • WorkerLoop:pull task → EpisodeRunner.run(task) → report           │
│  • EpisodeRunner 接口;LIBERO 实现连"指定 server"跑 infer            │
└───────────────▲ 原 WebSocket 推理协议(不动) ────────────────────────┘
                ▼
            server1 / server2  (M2 多 bundle + WarmupPool, 已支持)
```

**stage 生命周期数据流（G1R1 Item 2 修订）**：核心对每个 stage 保证 `on_stage_begin → episodes → on_stage_complete` 顺序，且 downstream.begin 在 upstream.complete 之后。以 warmup→eval 为例：
1. warmup stage `on_stage_begin`：`ctl.load_cache_config(warmup_yaml)`；
2. warmup episodes 跑完（DumpingJudge 落盘）；
3. warmup `on_stage_complete`：`buf = aggregate(ctl.fetch_dump(f"{cleanup_id}__warmup"))` → `ctx.publish(calib_id, buf)`；
4. eval stage `on_stage_begin`：`ctl.preload_normalizer_buffer(eval_id, ctx.buffer[calib_id])` **再** `ctl.load_cache_config(eval_yaml)`（顺序关键，config 对缺 WarmupPool fail-fast）→ eval episode 此刻才转 ready、可 pull；
5. eval `on_stage_complete`：`ctl.unload_warmup_buffer(eval_id)` 清理。

---

## 4. 模块划分与文件清单

### 4.1 新增：`src/openpi/conductor/`（通用机制）
| 文件 | 职责 |
|------|------|
| `__init__.py` | 包导出 |
| `task.py` | `EpisodeTask` / `TaskGraph` / `Stage` / `StageDependency` 数据结构（§5.1-5.2） |
| `strategy.py` | `ExperimentStrategy` ABC（§6.1）+ 注册机制 |
| `scheduler.py` | `EpisodeScheduler`：队列 + 亲和贪心 + barrier 门控（§7） |
| `driver.py` | `ConductorDriver`：引擎主循环、pull 服务端、ctl 连接池、strategy 装载 |
| `agent.py` | `WorkerAgent`：本机 worker 生命周期、心跳、转发 |
| `worker.py` | `WorkerLoop` + `EpisodeRunner` ABC（§6.2） |
| `journal.py` | `Journal`：账本持久化 + 断点续跑查询（§8） |
| `protocol.py` | worker↔agent↔driver wire 消息 schema + 编解码（msgpack）（§5.3） |
| `health.py` | `HealthAggregator` + 超时/心跳判定（§9.2） |
| `monitor.py` | per-worker 进度/吞吐聚合 + 渲染（§10） |

### 4.2 改动：`examples/libero/`
- 从 `main.py` **抽出** episode 执行内核（env 创建、wait-phase、infer 循环、closed-loop 计时、per-step 记录）为 `LiberoEpisodeRunner`（实现 `EpisodeRunner`），放 `examples/libero/episode_runner.py`。
- `main.py` 保留**单机 standalone 入口**（向后兼容，内部改为调用 `LiberoEpisodeRunner` + 一个进程内 mini-driver），不破坏现有 CLI。

### 4.3 新增：`exp/verdict_factor_judge/strategies/`（实验策略）
- `warmup_eval_strategy.py`：把 `run_phase.py` 的 7 步表达为 `ExperimentStrategy`（warmup stage →barrier(fetch+preload)→ eval stage）。
- `historical_warmup_strategy.py`：g5 历史重跑策略。
- 现有 `run_phase.py` / `phase5/runner.py` / `g5_warmup_libero10_driver.py`：迁移后成为「构造 strategy + 启动 driver」的薄入口（§11.2）。

### 4.4 文档（L3 要求）
- `docs/architecture/experiment_conductor.md`（新）+ `docs/README.md` 同步。

**解耦边界声明**：`src/openpi/conductor/` 不得 import `exp.*`、不得 import LIBERO；只依赖 `openpi_client`（ctl）+ 标准库。LIBERO/verdict 语义全部经 `EpisodeRunner` / `ExperimentStrategy` 接口注入。

---

## 5. 核心数据结构与 schema

### 5.1 `EpisodeTask`（driver→worker 派发单元）
```python
@dataclass(frozen=True)
class EpisodeTask:
    task_uid: str            # 确定性派生 f"{yaml_id}:{phase}:{task_id}:{episode_idx}"
                             # (续跑时同 episode 生成同 uid,与账本幂等匹配;自审 P15)
    yaml_id: str             # 实验配置标识
    phase: str               # "warmup" | "eval"(透传,worker 不解释)
    experiment: str          # task_suite_name(实验种类)
    task_id: int             # LIBERO task 下标
    episode_idx: int         # 该 task 内 episode 序号
    orig_init_state_idx: int # 初始状态映射(沿用现有 filter 语义)
    server_host: str         # 该 episode 所属 yaml 的归属 server
    server_port: int
    bundle_id: str           # M2 select_bundle 目标
    extra: dict              # 策略自定义控制信息(开放扩展)
```

### 5.2 `TaskGraph` / `Stage` / `StageDependency` / `StageContext`
- `Stage`：一组 `EpisodeTask` + `stage_id`（如 `<yaml_id>:warmup`）+ 所属 server + `phase` + 可选 `produces_calib_id` / `consumes_calib_id`（§5.5）。
- `StageDependency`：`upstream_stage → downstream_stage`。核心据此在 upstream 全 done 后依次跑 `on_stage_complete`(upstream) → `on_stage_begin`(downstream)，**不再用单一 `barrier: Callable`**，而是 §6.1 的 begin/complete 生命周期钩子（G1R1 Item 2）。
- `StageContext`（`ctx`，G1R1 Item 2）：stage 间黑板，`publish(calib_id, buffer)` / `buffer[calib_id]` 承载 warmup→eval 的 calibration buffer 交接；跨 server 时由核心负责 fan-out（§7）。
- `TaskGraph`：由策略 `plan()` 产出，核心据此维护每个 stage 的 ready/blocked。

### 5.3 wire 消息（msgpack；worker↔agent↔driver）
| 方向 | 类型 | 字段 |
|------|------|------|
| worker→driver | `pull` | `worker_id`, `server_host`（worker 所属 server，用于亲和） |
| driver→worker | `assign` | `EpisodeTask` \| `{none, backoff_ms}`（无 ready 任务时） |
| worker→driver | `report.progress` | `worker_id`, `task_uid`, `step`, `actions_per_s`, `hit_type` |
| worker→driver | `report.result` | `task_uid`, `success`, `n_steps`, `error`(\|None), `per_step_rows`（§11.4 随结果回传） |
| worker→agent | `heartbeat` | `worker_id`, `ts`, `state`(running/idle), `current_task_uid` |
| driver→worker | `shutdown` | 优雅退出 |

**传输（G1R1 Item 5 定稿）**：msgpack-over-TCP，length-prefixed framing，每条消息带 `protocol_version`；worker/agent 断线指数退避重连；`pull`/`report` 以 `task_uid` 幂等（重复 report 安全）。连接断时该连接在跑 episode 由 agent 心跳/driver 超时回队。

### 5.4 账本（journal）
- 追加式 JSONL：`{task_uid, yaml_id, phase, status: done|failed, success, ts}`。
- driver 启动时回放账本 → 已 `done` 的 episode 不再入队（幂等续跑）。

### 5.5 `CalibrationArtifact`（G1R1 Item 3 — warmup/校准数据一等实体）
把"校准数据"与"产出它的 warmup stage"解耦，支撑共享与历史源（reviewer 指出 1 eval↔1 sibling warmup 不足以表达 phase5 G3/G5）：
```python
@dataclass(frozen=True)
class CalibrationArtifact:
    calib_id: str                 # 独立标识(非派生自 eval_yaml_id)
    source: Literal["warmup_stage", "historical_file"]
    warmup_stage_id: str | None   # source=warmup_stage:产出它的 stage_id
    historical_path: str | None   # source=historical_file:phase3/phase4 raw jsonl
    cleanup_id: str | None        # warmup dump 的清理 id(source=warmup_stage 必填):
                                  # 约束 dump 命名 f"{cleanup_id}__warmup"(现有 server 派生
                                  # 规则的形状),清理走 unload_warmup_buffer(cleanup_id),
                                  # 不改 server 协议(G1R2 Item 1)。historical_file 源无 dump→None
```
- **依赖模型**：eval stage 依赖一个 **`calib_id`**（一对多——一个 artifact 可被多个 eval stage 消费），而非 1:1 sibling warmup。
- **共享（phase5 G3）**：多 eval cell 共享一个 `calib_id` → 其 warmup 只跑一次，buffer 经 `ctx` 分发给所有消费者。
- **历史（phase5 G5）**：`source=historical_file`，无 warmup stage；eval stage 的 `on_stage_begin` 直接从 `historical_path` 聚合 buffer 后 preload。
- **清理（G1R2 Item 1 — 不改 server 协议）**：现有 `unload_warmup_buffer(id)` 只接受一个 id 并在 server 端派生 `<id>__warmup.jsonl` 删除，wire **不接受任意 dump 名**。故约束 warmup dump 必须命名 `<cleanup_id>__warmup`：dump 文件清理走 `unload_warmup_buffer(cleanup_id)`，fetch 用 `f"{cleanup_id}__warmup"`；各消费 eval 的 WarmupPool entry 另按各自 `eval_yaml_id` 调 `unload_warmup_buffer(eval_yaml_id)`（删 pool entry）。1:1 场景 `cleanup_id = eval_yaml_id`（即现状）；共享 G3 用统一 `cleanup_id`。完全落在现有协议内。co-location/fan-out 见 §7。

---

## 6. 接口定义

### 6.1 `ExperimentStrategy`（策略层；exp/ 实现）

**stage 生命周期（G1R1 Item 2）**：核心对每个 stage 保证严格顺序
`upstream.on_stage_complete → 本 stage.on_stage_begin → 本 stage episode 入 ready → 全 done → 本 stage.on_stage_complete`。
`begin` 是 stage **开始前**的有序 setup control-op 原语（补齐 reviewer 指出的"只有完成 barrier、缺 setup"缺口）；`ctx`（`StageContext`，§5.2）是 stage 间黑板，承载 warmup→eval 的 calibration buffer 交接。

```python
class ExperimentStrategy(ABC):
    @abstractmethod
    def plan(self, yamls: list[str], server_assignment: dict[str, ServerEndpoint]) -> TaskGraph:
        """产出任务图:每个 yaml 的 stage 序列 + stage 间依赖;每个 stage 携带其
        setup/teardown 所需控制信息(warmup_yaml / eval_yaml / 依赖的 calib_id 等)。"""

    def on_stage_begin(self, stage: Stage, ctl: WebsocketClientPolicy, ctx: StageContext) -> None:
        """(G1R1 Item 2) 该 stage 的 episode 入 ready *前* 的有序 setup。典型:
        - warmup stage: ctl.load_cache_config(warmup_yaml, yaml_id=warmup_id)
        - eval stage:   先 ctl.preload_normalizer_buffer(eval_id, ctx.buffer[calib_id])
                        再 ctl.load_cache_config(eval_yaml, yaml_id=eval_id)
                        (顺序关键:config 加载对缺 WarmupPool fail-fast)
        默认 no-op。"""

    def on_stage_complete(self, stage: Stage, ctl: WebsocketClientPolicy, ctx: StageContext) -> None:
        """该 stage 全 episode done *后* 的 handoff/teardown:
        - warmup stage: buf = aggregate(ctl.fetch_dump(f"{cleanup_id}__warmup")); ctx.publish(calib_id, buf)
                        (写入 ctx 供下游 eval stage 的 on_stage_begin 消费)
        - eval stage:   ctl.unload_warmup_buffer(eval_id) 清理
        默认 no-op。"""

    def on_resume(self, stage: Stage, ctl: WebsocketClientPolicy, ctx: StageContext) -> None:
        """被触发式 server 自愈回调(§8.3):核心判定该 stage 的 server 端前置状态
        (WarmupPool)可能失效时调用,策略据此重建(重跑 warmup + preload)。判定走
        (B) 无条件重建 / (D) instance-id 优化,见 §8.3。默认 no-op。"""
```

### 6.2 `EpisodeRunner`（执行层；libero 实现）
```python
class EpisodeRunner(ABC):
    @abstractmethod
    def run(self, task: EpisodeTask, report: ProgressCallback) -> EpisodeResult:
        """连 task.server_host:port,select_bundle(task.bundle_id),跑一个 episode,
        周期性调 report(step, actions_per_s, hit_type);返回 success/n_steps。"""
```

### 6.3 driver 提供给策略的服务
- `ctl_for_server(server) -> WebsocketClientPolicy`：到某 server 的控制连接（连接池，复用）。
- stage 完成/续跑回调点（见 6.1）。

### 6.4 并发执行模型与连接生命周期（自审 P5/P6/P7）
- **barrier 回调不阻塞调度**：`on_stage_complete`（fetch_dump 拉大文件 + preload）可能耗时，必须在**独立线程**执行；期间 downstream stage 保持 blocked，调度主循环继续服务其他 server/yaml 的 pull。
- **worker 连接跨 episode 复用**：`EpisodeRunner` 持有到 `(server, bundle)` 的 WebSocket，连续 episode 复用（episode_start/episode_end 管边界，沿用现有单连接跑多 episode 语义）；仅当下一个 task 的 server/bundle 变化时才重连 + `select_bundle`。
- **bundle 切换成本**：跨 yaml 取任务 = `select_bundle` 切 bundle（触发 server 端 on_task_end/on_task_begin + lazy wrapper 构造）——这是 §7 yaml 亲和软约束的另一动机（亲和降低切换频率）。

---

## 7. 调度器算法（`EpisodeScheduler`）

**两个正交概念（自审 P1 厘清）**：
- **归属（静态，核心）**：driver 启动时把每个 yaml 整体分配到一个 server（一个 yaml 的全部 stage 归同一 server，因 WarmupPool 是 server 进程级状态）。归属决定"哪些 yaml 属于这个 server"，运行期不变（除非动态再平衡，见 R12）。均衡度量（自审 P4）：按 yaml 的 **episode 总数**（task 数 × trials，含 warmup+eval）一阶均衡 + yaml 数尽量均摊。
- **激活（动态，核心）**：在某 server 已归属的 yaml 中，此刻有多少个 yaml 的 stage 处于 ready/in-progress。亲和的目标是压低"同时激活 yaml 数"。
- **calibration co-location（G1R1 Item 3）**：消费同一 `calib_id`（§5.5）的 eval stage **优先归属同一 server**（避免 buffer 跨 server）。若因均衡无法 co-locate，driver 把该 `calib_id` 的 derived buffer **fan-out preload 到每个拥有依赖 eval stage 的 server**（WarmupPool per-process，每 server 各 preload 一份）；历史源（G5）同理 preload 到所有相关 server。

**worker 绑定（自审 P3）**：worker 启动绑定到一个 server（按配额 48/48），只取该 server 归属 yaml 的 episode；**不跨 server 偷活**（需连该 server 推理 + WarmupPool 在该 server）。代价是末期负载不均（见 R12）。

**pull 决策（优先级，高→低）**：
1. **永不空转**：该 server 有任何 ready episode 时，pull 必返回（全空才回 `none + backoff_ms`，指数退避；自审 P10）。
2. **yaml 亲和（软约束）**：ready episode 中优先返回**已激活 yaml**的 episode，维持激活集合最小。
3. **激活新 yaml**：仅当已激活 yaml 的 ready episode 取尽且仍有空闲 worker，才把一个新（已归属）yaml 的首 stage 转 ready。
4. **barrier 门控**：downstream stage 在 upstream 全 done + barrier 回调返回前保持 blocked，永不被 pull。

**自审暴露的张力（已纳入风险）**：
- **R8 warmup 低并行度（owner 定稿 2026-05-25）**：warmup stage 的 episode 数远少于 worker 数（如 2 vs 48），barrier 期大量 worker 无 ready 任务。**定稿:亲和强度分阶段**——**warmup 阶段放松**（允许同 server 同时 ≤2 个 yaml 并行 warmup，默认上限可配，填满 barrier/低并行空隙），**eval 阶段收紧**（严格单 yaml 亲和）。依据:显存压力主要来自 eval（长连接 + KV 累积），warmup 短小;故 eval 守"yaml 数最少"省显存，warmup 放宽换利用率。
- **R12 末期不均**：静态归属下某 server 先清空、其 worker 闲置。一阶实现接受；动态再平衡（迁未开工 yaml 到空闲 server，需重跑 warmup）列为后续。

---

## 8. 断点续跑（账本 + server 自愈）

### 8.1 driver 账本
- `task_uid` 确定性派生（§5.1），续跑时同一 episode 生成同 uid，与账本幂等匹配。
- driver 收到 `report.result` 后追加写 journal（done/failed）。重启回放：done 跳过、failed 按重试策略重入队。
- 幂等安全：driver 崩在写账本前 / result 丢失 → 该 episode 重跑（见 R3）。

### 8.2 warmup 的续跑粒度（自审 P8，必须解决）
warmup episode 的 factor 值由 server 端 DumpingJudge **append** 落盘；若按 episode 级重跑某 warmup episode，dump 重复 append → 聚合 buffer 偏斜。对策（plan 选定）：**warmup 的原子单位是 stage，不做 episode 级**——无论是 driver 重启**续跑**还是运行中的**重试**（§9.1），只要某 warmup stage 未整体完成，就先 `unload_warmup_buffer(cleanup_id)`（现有协议删 `<cleanup_id>__warmup` dump，§5.5）再整段重跑；**仅 eval stage 做 episode 级续跑/重试**。续跑与运行中 retry 在此**同一口径**（G1R1 Item 1）。**cleanup_id 生命周期（G1R4 建议）**：每次 warmup-stage 启动/重跑**前**清 `<cleanup_id>__warmup`；最后一个依赖该 calib 的 eval 消费完后清。

### 8.3 server 自愈（自审 P9 — owner 定稿 2026-05-25）
运行中/续跑时若 server 重启，其 WarmupPool / loaded bundle 丢失；现有协议无"查询 WarmupPool"控制帧，driver 无法主动探测。**定稿 = (B) 无条件重建保底 + (D) 可选 instance-id 优化**。放弃 (A) reactive（其正确性依赖"server 校准缺失错误可被 worker 可靠区分"这一未核实前提，不作为定稿），放弃 (C) 加 `query_warmup` 帧（违背"server 协议不动"）。

- **正确性基线 (B)**：续跑 / 判定 WarmupPool 可能失效时，对未整体完成的 eval stage **无条件**先重跑该 yaml 的 warmup stage + 重新 `preload_normalizer_buffer`（preload 幂等覆盖）。**不依赖任何未核实前提、server 零改动**。
- **效率优化 (D，可选，需 owner 批准 server metadata 纯增量)**：server 握手 metadata 增 `server_instance_id`（启动 nonce，向后兼容、不碰任何控制帧/响应语义）；driver 账本记录每个 (server, yaml_id) 关联的 instance_id。续跑时 instance_id 未变 → WarmupPool 必在 → **跳过重建**（零浪费）；变化或缺失 → 退回 (B)。
- **关键性质**：正确性**不依赖** (D)；server 一行不改时主路径 (B) 仍成立，只是每次续跑多跑一次未完成 yaml 的 warmup。

`strategy.on_resume`（§6.1）定位为"被触发式重建"：核心判定需重建后调用，策略执行重跑 warmup + preload；机制（判定/触发）在核心，**重建动作在策略**（解耦）。

---

## 9. 重试与健康

### 9.1 重试
- **可重试**（网络/超时/server 不可达/worker crash），**按 phase 区分粒度（G1R1 Item 1）**：
  - **eval episode**（无服务端副作用）：单 episode `task_uid` 回队，换 worker 重试，默认上限 3。
  - **warmup episode**（有副作用——DumpingJudge 已 append dump 行，单 episode 重跑会重复 append、污染 buffer）：**不单独回队**；作废**整个 warmup stage**（`unload_warmup_buffer(cleanup_id)` 删 `<cleanup_id>__warmup` dump，§5.5；+ 重置 stage 状态）后整段重跑。即 warmup 的重试原子单位是 **stage**，与 §8.2 续跑同口径。
- **致命**（`ConfigValidationError` 等配置错）：不重试，标记该 yaml 失败并告警，其余 yaml 继续。
- 分类依据：错误类型 + worker report 的 `error` 字段。

### 9.2 健康
- agent 定期收 worker `heartbeat`；**心跳超时**或 **episode 墙钟超时**（按 task `max_steps` 推算上限）→ agent 杀掉并本地重启 worker，其在跑 episode 回队（计一次可重试失败）。
- driver `HealthAggregator` 汇总 per-worker：`state`(running/idle/dead)、当前 task、进度、`actions/s`（滑窗，沿用现有 `_update_rate` 逻辑）。
- **agent 健康（自审 P11）**：agent 是本机所有 worker 的中转单点。driver 监控 agent 心跳；agent 失联 → 该机全部在跑 episode 回队 + 告警，等待 agent 恢复或人工介入。

---

## 10. 监控
- `monitor.py` 聚合 worker `report.progress` → per-worker 进度条 + 吞吐 + 健康 + 全局汇总（episodes done/total、success rate、各 server 在跑 yaml 数）。
- **大规模渲染（自审 P16）**：96 worker 用 96 行 tqdm 不现实，且 driver 常在无 TTY 后台。默认输出**聚合视图**（每 server 一行汇总 + 全局 done/total/SR/各 server 激活 yaml 数）；per-worker 明细经可选 verbose 或单独查询；非 TTY 时退化为周期性日志行。

---

## 11. 集成与迁移

### 11.1 与 server 对接（不改 server）
- worker `EpisodeRunner` 复用 `WebsocketClientPolicy.infer` + `select_bundle`。
- 策略 stage setup/teardown（§6.1）复用 `load_cache_config`/`fetch_dump`/`preload_normalizer_buffer`/`unload_warmup_buffer`（经 driver ctl 连接池）。
- **(G1R1 Item 4) conductor 直连单进程 server 端点，不经 `replica_proxy` 路由器**：`ServerEndpoint` 必须是单个推理进程（其直连/loopback 端口）。理由：`fetch_dump` 在 `replica_proxy` 是 **sticky**、`load_cache_config`/`preload`/`unload` 是 **broadcast**，而 DumpingJudge 落盘与 WarmupPool 都是 **per-replica-process** 的——经 `replica_proxy` 的 `fetch_dump` 只读到一个 child replica 的 partial dump，再 broadcast partial buffer，语义错误。
- driver 的 worker→server 显式分配**替代** `replica_proxy` 的横向扩展角色：多副本 = 向 driver 注册**多个独立单进程 server 端点**（各占一端口），由 driver 直接 1:1 fetch/preload，无 partial。
- **(G1R2 Item 2) endpoint 类型做成显式部署不变量，而非运行时探测**：现有 `replica_proxy` 转发 child backend 的 metadata，routed 端点与 direct concurrent child 均显示 `{"concurrent": True}`，**无法在不改 proxy 协议的前提下区分**。故本编排路径把"server 端点必须是直连单进程（非 replica 公共端口）"定为**用户配置不变量 + 部署契约**：driver 的端点列表由用户显式提供，架构文档/部署文档明示此约束，driver **不自动探测、不运行时拒绝**（探测不可靠）。保证由部署纪律 + 文档承担（见 R13）。
- **⚠ 已撤销（owner override 2026-05-25，见 §16.6）**：上述"不经 `replica_proxy`、多副本=多独立端点、endpoint 必须单进程"约束已废除——`replica_proxy` 的 `fetch_dump` 改为 **aggregate**（fan-out + `merge_dump_replies` 拼接各 replica 切片），`--replicas N` 单公共端口对 conductor 透明，driver 注册一个 endpoint 即可；多独立端点降级为可选（按 server 细粒度分配 worker 时用）。

### 11.2 现有实验迁移
- `run_phase.py` 7 步 → `WarmupEvalStrategy`：`on_stage_begin`(warmup: load warmup_yaml) / `on_stage_complete`(warmup: fetch+aggregate→publish calib) / `on_stage_begin`(eval: preload→load eval_yaml) / `on_stage_complete`(eval: unload)。
- **phase5 G3 共享 warmup（G1R1 Item 3）**：多 eval cell 的 `consumes_calib_id` 指向同一 `CalibrationArtifact`（`source=warmup_stage`）；该 warmup 只跑一次，buffer 经 `ctx` 分发 + 按 §7 co-location/fan-out preload 到相关 server。
- **phase5 G5 历史 warmup（G1R1 Item 3）**：eval cell 的 `calib_id.source=historical_file`，无 warmup stage；`on_stage_begin` 从 `historical_path`（phase3/phase4 raw）聚合后 preload。
- `phase5/runner.py` 的 lazy warmup / emit-eval → 同一策略的 `plan` 变体（懒触发改为 stage 依赖）。
- 旧入口脚本改为薄封装（构造 args → strategy → `ConductorDriver.run()`），**保留 CLI 兼容**一个过渡期。

### 11.3 `examples/libero/main.py` 兼容（自审 P12 厘清）
- standalone = **进程内薄循环直驱 `LiberoEpisodeRunner`**（线程池 + 共享 task_queue，沿用现有 `_eval_concurrent` 结构），**不**起 agent/网络、**不**复用 `ConductorDriver`（避免单机背三层进程通信）。仅共享 `EpisodeRunner` 执行内核，调度走简化路径。现有 `--num-workers` 等 CLI 不破坏。

### 11.4 per-step log 跨机收集（自审 P13）
worker 分布多机，B1.2 临时分片散在各机本地磁盘。plan 选定：worker 在 `report.result` 时**随结果回传该 episode 的 per-step rows**（每 episode 几十~上百行，量小），driver 统一落盘 + 按全局键 `(yaml_id, task_id, episode_idx, step)` 排序归并；不依赖共享 NFS。

---

## 12. 测试策略

| 层 | 测试 | 是否需 GPU/server |
|----|------|-------------------|
| 调度器 | 亲和贪心、永不空转、yaml 数最少、barrier 门控（纯逻辑，fake TaskGraph） | 否（CI） |
| 账本 | done 幂等跳过、failed 重入队、续跑回放 | 否（CI） |
| 重试/健康 | 可重试 vs 致命分类、超时回队、心跳判定 | 否（CI） |
| 协议 | wire 消息 msgpack round-trip | 否（CI） |
| 集成 | `FakeEpisodeRunner` + `FakeServer` 端到端跑 mock 实验，验证无空隙/续跑/重试/barrier | 否（CI，全 fake） |
| **R1/R2 正确性专项（G1R2 Item 3）** | 见下方 6 项，逐一 test-pin R1/R2 的高风险修复 | 否（CI，fake） |
| 端到端 | 真 LIBERO + 真 server 多机 | `@pytest.mark.manual` |

**R1/R2 正确性专项测试（G1R2 Item 3）**——每项独立 test，钉死对应修订：
1. **warmup-stage 作废**：warmup episode 失败 → 整 stage 作废重跑（含 `unload_warmup_buffer(cleanup_id)`），而非单 episode retry（§9.1）。
2. **preload-before-load 顺序**：eval stage `on_stage_begin` 必须先 `preload_normalizer_buffer` 再 `load_cache_config`，乱序应触发 fail-fast（§6.1）。
3. **共享 `CalibrationArtifact` co-location/fan-out**：多 eval 共享一 calib_id 时 buffer 分发到所有相关 server（§7）。
4. **historical-file 校准**：`source=historical_file` 的 stage 跳过 warmup、直接从 `historical_path` 聚合 preload（§5.5）。
5. **cleanup-id ↔ dump 命名映射**：dump 命名 `<cleanup_id>__warmup` 与 `unload_warmup_buffer(cleanup_id)` 删除一致（§5.5）。
6. **~~replica-proxy 端点不变量~~（§16.6 撤销）**：原计划测"端点单进程契约"已废；改为测 `replica_proxy` 的 `fetch_dump` fan-out 聚合（classify→aggregate + `merge_dump_replies` 拼接/跳过缺失/全缺失报错 + 端到端 fan-out），见 `tests/serving/test_replica_proxy.py`。

- 新增测试置于 `tests/conductor/`（执行侧），**不触碰** `tests/review_tests/`。

---

## 13. 风险登记

| # | 风险 | 缓解 |
|---|------|------|
| R1 | worker↔agent↔driver 通信可靠性（跨机断连） | 心跳 + 超时回队 + 幂等 task_uid；pull 失败 backoff 重试 |
| R2 | server 重启致 WarmupPool 丢失，续跑结果失真 | 同 R10（§8.3 (B) 无条件重建保底，不依赖探测） |
| R3 | 账本与实际状态不一致（崩在写账本前/后） | 账本追加式 + 幂等；result 先到 driver 再写账本；重复 done 安全 |
| R4 | 亲和贪心退化（极端任务分布下仍多 yaml 并存） | 软约束 + 可配置每 server yaml 软上限；监控暴露并存 yaml 数 |
| R5 | 迁移破坏现有 verdict 实验结果可比性 | 保留旧入口过渡期；策略输出与旧 7 步逐字段对齐；迁移前后**相同 seed 下对拍** episode 集合 / warmup buffer keys / SR 判定一致（自审 P14） |
| R6 | EGL 单卡 worker 上限（15/卡）被 agent fork 超配 | agent 按 (机器,卡) 配额 fork，沿用 15/卡上限并暴露为配置 |
| R7 | `src/openpi/conductor/` 误依赖 exp/libero（破坏解耦） | import 边界单测（禁止 import exp.*/libero）；§4 边界声明 |
| R8 | warmup 低并行度致 barrier 期 worker 空转 | §7：barrier 期可预激活同 server 下一 yaml warmup（上限可配）；或接受 warmup 期低利用 |
| R9 | warmup episode 重跑致 server dump 重复污染 buffer | §8.2：warmup 只做 stage 级续跑/重试，重跑前 `unload_warmup_buffer(cleanup_id)` 清 `<cleanup_id>__warmup`（现有协议，§5.5） |
| R10 | server 重启致 WarmupPool 丢失，且无探测帧 | §8.3 定稿 (B) 无条件重建保底（不依赖未核实前提、server 零改动）+ (D) instance-id 优化（可选，需 owner 批准 metadata 增量） |
| R11 | agent 单点崩溃致本机 worker 全失联 | §9.2：driver 监控 agent 心跳，失联则该机 episode 回队 + 告警 |
| R12 | 末期 yaml→server 静态归属致负载不均 | 一阶接受；动态再平衡列后续（迁移需重跑 warmup） |
| R13 | 经 replica_proxy 时 fetch_dump sticky → partial warmup buffer | §11.1：endpoint 必须直连单进程，定为**用户配置不变量 + 部署契约**（现有 metadata 无法区分 routed/direct child，driver 不探测）；多副本 = 多个独立端点由 driver 1:1 fetch/preload。**⚠ 2026-05-25 owner override 撤销（§16.6）**：fetch_dump 改 aggregate + `merge_dump_replies` 拼接，--replicas 单公共端口透明 |

---

## 14. 实施分步（里程碑）

- **M1** `task.py` + `protocol.py` + `strategy.py`/`worker.py` 接口骨架 + schema 单测。
- **M2** `scheduler.py`（亲和/无空隙/barrier）+ `journal.py` + 单测（CI 全覆盖，无 GPU）。
- **M3** `driver.py` 引擎主循环 + ctl 连接池 + pull 服务端；`agent.py` fork/心跳；`worker.py` 循环。
- **M4** `LiberoEpisodeRunner`（从 main.py 抽出）+ `WarmupEvalStrategy`；`FakeEpisodeRunner`/`FakeServer` 集成测试。
- **M5** 重试 + 健康 + 监控渲染。
- **M6** 迁移 run_phase/phase5/g5 为策略 + 旧入口薄封装 + 一致性对拍。
- **M7** 架构文档 + `docs/README.md` 同步 + 全量 Verify。

> 每个里程碑独立可测；M1-M2 是纯逻辑核心，优先在 CI 锁死调度/续跑正确性。

---

## 15. 待 G1 讨论的开放点
1. ~~wire 传输选型~~ **已定稿（G1R1 Item 5）→ §5.3**：**msgpack-over-TCP**（复用 `msgpack_numpy`，零新重依赖）；length-prefixed framing；消息带 `protocol_version`；worker/agent 断线**指数退避重连**；pull/report 以 `task_uid` 幂等。失败语义：连接断 → 该连接在跑 episode 由 agent 心跳/driver 超时回队（warmup 按 §9.1 走 stage 级作废）。跨机可观测性经 agent 心跳聚合到 driver。
2. `examples/libero/main.py` 兼容期：保留多久、是否最终弃用 standalone 进程内多线程路径。
3. 旧 verdict 入口（run_phase/phase5/g5）迁移是否在本次 L3 一次性完成，还是分 PR（plan 倾向：核心引擎 + libero runner + WarmupEvalStrategy 本次完成；g5/phase5 变体迁移可作为后续）。
> **P9 server 自愈（§8.3）与 P2 warmup 利用率（§7 R8）已由 owner 定稿（2026-05-25）**，不再开放：P9 = (B) 无条件重建保底 + (D) 可选 instance-id 优化、放弃 (A)/(C)；P2 = warmup 放松 / eval 收紧的分阶段亲和。详见对应正文。其中 (D) 涉及 server metadata 纯增量字段，**若 owner 坚持 server 零改动则仅用 (B)，正确性不受影响**。reviewer 仍可对定稿本身提出异议。

---

## 16. 实现偏离与一阶简化（实现期记录，待 G2 审）

实现期相对已 APPROVED plan 的偏离/简化，按执行权限 §4 在此 flag：

1. **agent 链路简化（vs §3/§5.3）**：worker **直连 driver** 的 pull 端口；agent 仅本机 fork+监督 worker（`poll()` 检测进程退出即重启），不做原图设想的 worker↔agent↔driver 消息转发。agent 崩溃由 driver 连接断检测 + in-flight 回队覆盖，功能等价且少一跳。`heartbeat` wire 消息（§5.3）相应**移除**（未接通的死接口）。
2. **episode 墙钟超时已实现，独立 heartbeat wire 简化（vs §9.2）**：driver `requeue_timed_out`（`episode_timeout_s` 默认 1800s）回收卡死 worker 的 in-flight episode（满足 §9.2 的 episode 墙钟超时回队 — G2R1 Blocking 2）；worker crash 由连接断回队。独立 heartbeat wire 未实现，主动心跳由连接活性 + episode 超时替代；`stale_workers` 作 best-effort 观测。
3. **server 自愈 (B) 失效判定近似（§8.3）**：driver 用 `_resuming`（journal 非空）门控 `on_resume`，以"续跑 == WarmupPool 可能失效"近似 §8.3 判定（正常首跑不触发）。续跑时 warmup stage 本就从头重跑（warmup 不 episode-journal），天然满足 (B) 的无条件重建。
4. **warmup 超时回队的 stale 防护（G2R3，owner 定稿 A）**：实现 **result fence**——`EpisodeTask`/`EpisodeResult` 带 `attempt`，scheduler 每 uid dispatch generation 递增，`mark_result` 拒绝 superseded attempt 的迟到 result（worker-reported 才 fence；driver 主动回队 `attempt=None`）。超时回队 + 重 dispatch 同 uid 后，旧 worker 的迟到 result 不会被当作当前 dispatch 接受（修 G2R3 探针）。**剩余 server-side dump 残留**（旧 worker 在 rerun 后 DumpingJudge append 被 fetch 读到）作为已知边角接受：`episode_timeout_s` 默认 1800s，超时的 warmup worker 实际几乎已卡死（不再 append），且其 result 已被 fence。完整 dump 隔离（generation-scoped bundle/dump 或 agent worker-kill）列为后续 owner-approved 增强。
5. **后续项（按 §15.3 倾向）**：g5/phase5 策略迁移、co-location 跨 server fan-out 多 server 实战、standalone `main.py` mini-driver 改造未在本次落地；本次交付核心引擎 + `LiberoEpisodeRunner` + `WarmupEvalStrategy`(1:1)。
6. **(owner override 2026-05-25 — 撤销 G1R1 Item 4 / R13 的 replica_proxy 约束)**：owner 行使项目主权（WA §7），判定"conductor 不经 `replica_proxy`、多副本=多独立端点"是错误回避，要求修 `replica_proxy` 本身而非绕过。已实现：`replica_proxy` 把 `fetch_dump` 从 **sticky** 改为 **aggregate**——fan-out 到所有 child + `merge_dump_replies` 拼接各 replica 的 warmup dump 切片（每 child 的 DumpingJudge 只落自己路由到的 episode，合并即整机完整 dump，消除 partial）；`bundle`/`preload`/`unload` 本就 broadcast、infer sticky，故 warmup→eval 全链经 router 完整。**结果**：`--replicas N` 单公共端口对 conductor 透明（driver 注册一个 endpoint），§11.1 的"必须直连单进程"不再是约束（多独立端点降级为"按 server 细粒度分配 worker"的可选方式）。owner 指示按主权直接落地（不走 G1/G2）；改动仅 proxy 路由+merge，**不动 server wire 协议**（C2/write-frozen 不受影响）；新增 `merge_dump_replies` + 单元/端到端测试，live 文档（experiment_conductor §10 / tutorial §1.3·§10 / docs/README）已同步。

偏离 1-5 均与硬约束（零 server 改动 / 无空隙调度 / warmup 原子）一致，是实现期发现的简化/定稿；第 6 条为 G2 后 owner 主权变更（仅 proxy 路由，C2 不受影响）。

---

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-05-25 14:44 CDT

- [Blocking] [Concern] Stage lifecycle hook failures can retry forever, including fatal config errors — reasoning: the approved plan says `ConfigValidationError` and other configuration mistakes are fatal and must not be retried (§9.1), but `ConductorDriver.drive_stages_once` catches every `on_stage_begin` / `on_stage_complete` exception and resets the stage to pending again. Real `WarmupEvalStrategy.on_stage_begin` calls `load_cache_config`, so a bad yaml or missing WarmupPool fails in the stage hook path, never reaches `is_retriable_error`, and can leave the driver looping forever. An independent probe that raised `RuntimeError("ConfigValidationError: bad config")` from `on_stage_begin` remained `setup_pending` / `all_done=False` after repeated `drive_stages_once` calls.
- [Blocking] [Concern] The heartbeat / episode wall-clock timeout requirement is not implemented — reasoning: approved plan §9.2 requires heartbeat timeout or episode wall-clock timeout to kill/restart a worker and requeue its in-flight episode. The implementation demotes this to crash-only connection-drop requeue plus best-effort `stale_workers`; a process that stays alive with an open socket but blocks inside infer keeps its task in-flight forever. Either implement the timeout/heartbeat path, or get an explicit approved scope change that narrows the G2 acceptance target.
- [Blocking] [Concern] `logs/README.md` was not synchronized with the modified active log — reasoning: Working Agreement §4 requires the corresponding README to be updated whenever a `logs/` file changes. The plan is now G2 code-review material, but `logs/README.md` still lists `client_conductor_two_layer_refactor.log.md` as `Plan` / `待 G1`, so the index is stale.
- [Blocking] [Concern] New empty `__init__.py` files violate the module-docstring rule — reasoning: Working Agreement §3.2 requires every new file to have a module-level docstring. `examples/__init__.py`, `examples/libero/__init__.py`, and `tests/conductor/__init__.py` are new zero-byte files.

### G2 Round 2 — Executor — 2026-05-25

4 个 Blocking 均 well-founded，逐条接受修订。§6 Verify：全量 1645 passed / 5 skipped / 13 failed（13 全为 pre-existing JAX/网络环境失败，与 baseline 一致），零回归；ruff check 全过、已格式化。

- Accepted（Blocking 1 — hook 失败可无限重试含致命 config 错误）— `drive_stages_once` 的 setup/complete except 改为 `except Exception as exc` 并用 `is_retriable_error(repr(exc))` 分类；scheduler `mark_setup_failed`/`mark_complete_failed` 加 `fatal` 参数 + `setup_attempts` 计数：**fatal（ConfigValidationError 等）或超 `max_setup_retries` → 标 FAILED + `_cascade_fail` 下游**（不再无限重试，plan §9.1），可重试 → 回滚 SETUP_PENDING（bounded 自恢复）。reviewer 的 `RuntimeError("ConfigValidationError: ...")` 探针现 → 立即 FAILED + `all_done()` 成立。新增 `test_fatal_setup_hook_error_fails_fast_not_infinite_retry` + `test_setup_retry_exhaustion_fails_stage`。
- Accepted（Blocking 2 — heartbeat / episode 墙钟超时未实现）— 实现 **episode 墙钟超时**（满足 §9.2 的 "OR episode wall-clock timeout"）：scheduler 记 dispatch ts + 新增 `requeue_timed_out(timeout_s)`，driver 加 `episode_timeout_s`（默认 1800s）并在 `_stage_loop` 周期调用——**卡在 infer 不退出的 worker 的 in-flight episode 被回收**（eval 回队 / warmup 整 stage 作废）。独立 heartbeat wire 仍简化（主动心跳由连接活性 + episode 超时替代），已在 plan §16.2 + docs §8 明确。新增 `test_requeue_timed_out_requeues_stuck_episode`。
- Accepted（Blocking 3 — logs/README 未同步）— `logs/README.md` 的 plan 行 Status 由 `Plan / 待 G1` 更新为 `In Progress (G1 APPROVED 2026-05-25 / Post-G1 polish / §4 Code + 3 轮自审完成 / G2 进行中)`（WA §4）。
- Accepted（Blocking 4 — 空 `__init__.py` 缺 module docstring）— `examples/__init__.py`、`examples/libero/__init__.py`、`tests/conductor/__init__.py` 均补 module-level docstring（WA §3.2）。

修订后重入 G2，请审。

### G2 Round 3 — Reviewer — NEEDS REVISION — 2026-05-25 14:57 CDT

- [Blocking] [Concern] Timed-out warmup work is requeued without fencing or killing the old attempt, so stale side effects can still contaminate the rerun — reasoning: G2R2 added `requeue_timed_out`, which fixes the "in-flight forever" hang, but it does not terminate the old worker/connection and the protocol still has no dispatch attempt id. For warmup stages this matters because `DumpingJudge` appends on the server: after timeout the stage is reset and the same `task_uid` can be dispatched again; if the old worker later writes/reports, `mark_result` can accept that stale result as the current dispatch. A reviewer probe timed out a warmup task, redispatched the same uid, then delivered the old uid's result and the stage moved to `COMPLETE_PENDING`. This reopens the duplicate/stale append risk that G1/R1 fixed. The fix needs either actual worker kill before warmup rerun, or an attempt/generation fence that rejects stale results and prevents stale server-side dump rows from being used.

### G2 Round 4 — Executor — 2026-05-25

Blocking well-founded（reviewer 探针准确复现 stale-result 污染）。接受并修订。§6 Verify：全量 1646 passed / 5 skipped / 13 failed（13 全为 pre-existing JAX/网络环境失败，与 baseline 一致），ruff 全过、已格式化，零回归。

- Accepted（G2R3 — 超时回队的 warmup 无 fence，stale result/dump 污染）— 实现 reviewer 路线的 **result fence（"reject stale results"）**：`EpisodeTask`/`EpisodeResult` 加 `attempt`；scheduler 每 uid 维护 dispatch generation（`_dispatch_gen`），`next_task` 递增并 `dataclasses.replace(task, attempt=gen)`；`mark_result` 对 **worker-reported** result 检查 `attempt == 当前 generation`——超时回队 + 重 dispatch 同 uid 后，旧 worker 的迟到 result（低 attempt）被拒绝（driver 主动超时/断连回队传 `attempt=None`，不 fence）。reviewer 探针（超时→重 dispatch→旧 result→COMPLETE_PENDING）现被 fence 阻止。worker 经 `dataclasses.replace` 回传 `task.attempt`，driver `handle_result` 透传给 `mark_result`。新增 `test_stale_attempt_result_fenced_after_requeue`。
  reviewer 路线的另一半 "prevents stale server-side dump rows from being used"（旧 worker 在 rerun 后 `DumpingJudge` append 被 `fetch_dump` 读到）经 **owner 定稿取方案 A（result fence + 文档化残留）**：`episode_timeout_s` 默认 1800s 下，超时的 warmup worker 实际几乎已卡死（不再 append），且其 result 已被 fence；完整 dump 隔离（generation-scoped bundle/dump 或 agent worker-kill）作为后续 owner-approved 增强。已记入 plan §16.4 + docs §8。

修订后重入 G2，请审。

### G2 Round 5 — Reviewer — APPROVED — 2026-05-25 15:19 CDT

- [Non-blocking] [Suggestion] Keep generation-scoped dump isolation or agent-side worker kill as a follow-up hardening item — reasoning: G2R4 closes the actionable stale-result hole with an attempt fence, and §16.4 explicitly records the remaining server-side dump residual as owner-accepted scope. This is not a G2 blocker for the current delivery.

### G2 Round 6 — Reviewer — NEEDS REVISION — 2026-05-25 16:21 CDT

- [Blocking] [Concern] `replica_proxy.fetch_dump` aggregate can still return a partial warmup dump on replica error or backend failure — reasoning: `merge_dump_replies` skips every `{"__ack__": "error"}` whenever at least one replica returned content, and `_aggregate` drops backend exceptions / malformed replies before calling the merge. That is only safe for the explicit "dump not found" case where a replica had no routed warmup episodes; it is unsafe for `warmup_dump_root not configured`, invalid id, backend disconnect, or any failed child, because the proxy then returns a successful `fetch_dump` containing only survivor slices. A reviewer probe with one successful slice and one `{"__ack__": "error", "msg": "warmup_dump_root not configured"}` returned `{"__ack__": "fetch_dump", ...}`. This reintroduces the partial calibration error the owner override is meant to eliminate. The aggregate path should surface non-missing errors and missing/failed backend replies as an error, while still allowing explicit per-child `dump not found` to be skipped.
- [Non-blocking] [Suggestion] Update the remaining active index link to the deleted concurrent-serving guide — reasoning: `docs/deployment/concurrent_serving.md` is deleted and the content is documented as merged into `docs/experiments/conductor_tutorial.md`, but `logs/README.md` still links `concurrent_serving_optimization_plan.log.md` to `../docs/deployment/concurrent_serving.md`. This is advisory because it does not affect runtime correctness, but it should be cleaned before commit to avoid a stale active-index link.

### G2 Round 7 — Executor — 2026-05-25

- Accepted（Blocking — `merge_dump_replies` 仍可返回 partial dump）— 审查者正确。原实现两个缺陷：(a) 只要任一 replica 有 content 就跳过**所有** `{"__ack__":"error"}`，把致命错误（`warmup_dump_root not configured` / `invalid warmup_yaml_id` / bad request）当良性 missing 一并跳过；(b) `_aggregate` 在 fan-out（`return_exceptions=True`）后只收成功解码的 bytes，**异常/掉线的 backend 被静默丢弃**，survivor-only 切片被当整机 dump 返回。只有 server `_handle_fetch_dump` 的 `dump not found`（`not candidate.exists()`，该 replica 无路由 warmup episode）是良性可跳过。**修复**：(1) 新增契约常量 `FETCH_DUMP_NOT_FOUND_MSG`（注释引用 server handler；`test_fetch_dump_not_found_msg_matches_server_contract` 用 inspect 守护跨进程字符串不漂移）；(2) `merge_dump_replies(replies, *, n_expected)` 重写为三类判定——benign missing 跳过 / 致命（非 missing msg、malformed reply、success 无 bytes content）累积上报 / **dropped backend** 用 `n_expected - len(replies) > 0` 检测；任一致命或 dropped → `{"__ack__":"error","msg":"fetch_dump would be a partial dump: ..."}`；全 benign-missing 仍返回 "no dump on any replica"；(3) `_aggregate` 传 `n_expected=len(self._backends)`。遵守 owner "仅改 proxy、不动 server wire" 约束（server 零改动，靠契约常量 + 守护测试）。新增/更新 7 测试：fatal-not-skipped / dropped-backend-partial / malformed-reply / benign-mix / all-benign / 契约守护 / 端到端 `test_proxy_fetch_dump_dropped_backend_is_error`（真实 fan-out 下断一个 backend → error）。`tests/serving/test_replica_proxy.py` 28 passed。
- Accepted（Non-blocking — logs/README stale link）— `logs/README.md` 第 57 行 `concurrent_serving_optimization_plan.log.md` 的 operator guide 链接仍指向已删除的 `docs/deployment/concurrent_serving.md`，改为指向并入后的 `docs/experiments/conductor_tutorial.md`。

### G2 Round 8 — Reviewer — APPROVED — 2026-05-25 16:35 CDT

- [Non-blocking] [Suggestion] Keep the `FETCH_DUMP_NOT_FOUND_MSG` contract test with the proxy tests — reasoning: the aggregate path now correctly fails closed on non-missing child errors and dropped backends while preserving benign missing slices. The string guard is the right lightweight protection since the server wire protocol remains unchanged.

### G2 Round 9 — Executor — 2026-05-25

- Acknowledged（Non-blocking — 保留契约测试）— 同意。`test_fetch_dump_not_found_msg_matches_server_contract` 保持在 `tests/serving/test_replica_proxy.py` 与 proxy 测试同处；server wire 协议未变，inspect 字符串守护是匹配该不变量的轻量保护，无代码改动。G2 APPROVED（R8），进入 §6 Verify → §7 Commit → §8 Push。
