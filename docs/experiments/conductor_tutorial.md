# 并发服务 + 实验编排教程

> 端到端 how-to：如何**起并发 inference server** + 用新的**实验编排框架**（`src/openpi/conductor/`）跑大规模评测。**重点讲如何编写 driver 的独立策略部分**（`ExperimentStrategy`）。
> 设计依据见 [`docs/architecture/experiment_conductor.md`](../architecture/experiment_conductor.md)、[`docs/architecture/cache_system.md`](../architecture/cache_system.md) §9.X、`logs/client_conductor_two_layer_refactor.log.md`、`logs/concurrent_serving_optimization_plan.log.md`。

---

## 0. 它取代了什么

- **Server 侧**：post-Phase-5 server 托管一份 base policy，用 `BatchingCoordinator` 把多连接的 stage1/2/3 forward 跨连接 batch，吃满 GPU；多份 cache YAML 可作"bundle"并排加载，每连接绑一个 bundle。
- **Client 侧（新）**：旧的 `examples/libero/main.py --num-workers N`（进程内多线程、单卡 ≤15）+ `run_phase.py` 硬编码 7 步编排，被**实验编排框架**取代——你只写一个 `ExperimentStrategy`，调度/续跑/重试/监控/亲和全部由通用引擎负责。

```
driver (中枢)   ← 你装载一个 ExperimentStrategy
  └ EpisodeScheduler / Journal / Retry / Monitor / 到各 server 的 ctl 连接池
agent (每机)    ← fork & 监督本机 worker 进程（绑 GPU/EGL slot）
worker (进程)   ← pull episode → EpisodeRunner.run → report；直连 server 跑 infer
```

**核心心智模型**：你**不写**调度/pull/续跑/重试/监控（通用机制）；你**只写** (1) 一个 `ExperimentStrategy`（实验剧本）；(2) 复用或实现一个 `EpisodeRunner`（怎么跑一个 episode）。`src/openpi/conductor/` **不依赖** `exp.*` 或 LIBERO。

---

## 1. 起并发 server

### 1.1 `--concurrent`（默认）

```bash
python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero --policy.dir=<checkpoint dir> \
    --port 8000 --cache_config <some.yaml>
```

- 一个 server 进程，多 client 连接共享同一份 GPU 权重。
- `BatchingCoordinator` 起三个 stage worker 线程，把并发请求合批 forward。默认 `max_batch_size=8`、`max_wait_ms=10`（调优见 §8）。
- `--cache_config` 启动时加载一次；后续 bundle 经 `__ctrl__: load_cache_config` 注入（策略在 `on_stage_begin` 做）。
- **C2 write-frozen**：backend 启动后冻结，运行时 `insert/delete/load_artifact` 等抛 `BackendFrozenError`；YAML 的 `write_policy` **必须 `"never"`**，否则启动即 `ConfigValidationError`（fail-fast，不静默中和）。cache artifact 离线构建。

### 1.2 `--non-concurrent`（baseline / 极速）

```bash
python scripts/serve_policy.py ... --non-concurrent --cache_config <yaml>
```

- 单连接 server（后续连接被拒，WS 1013）。无 `BatchingCoordinator` / lazy lifecycle / bundle indirection，保留 C1 原始单连接结构；数值匹配当前 sdpa 模型，不等同于历史 pre-Phase-5 eager baseline。用于量单连接延迟上界。C2 仍生效。

### 1.3 多副本：`--replicas N` 单公共端口，或多个独立 `--concurrent` 端点

单进程被 **GIL 串行化 CUDA kernel-launch** 限制（pi05_libero ~12 inf/s），所以扩展是**进程级**。两种方式都受 conductor 支持：

> **A) `--replicas N` + 单公共端口（推荐）**：`serve_policy.py --replicas N` 在一个公共端口后起 N 个 child 推理进程，由 `replica_proxy` router 横向扩展。router 对 conductor **透明**：infer 按 least-connections sticky、bundle/`preload`/`unload` broadcast 到所有 child，而 **`fetch_dump` 已是 aggregate**——fan-out 到所有 child 并**拼接各 replica 的 warmup dump 切片**为整机 dump（每个 child 的 DumpingJudge 只落自己路由到的 episode，router 经 `merge_dump_replies` 合并后无 partial）。driver 只需注册**一个** `ServerEndpoint`（公共端口）。
>
> **B) 多个独立 `--concurrent` 单进程端点**：各占一个端口（同 GPU 多进程 / 多 GPU / 多机），作为**多个独立 `ServerEndpoint`** 注册给 driver，由 driver 的 worker→server 分配调度。适合需要按 server 细粒度分配 worker（如 server1→48 worker、server2→48 worker）的场景。
>
> 两者可混用（如 2 台机各 `--replicas 3`，注册 2 个公共端点）。
>
> **跨 server 非均分 worker（如 16/48）**：两台 server 算力/显存不对等时（如一台被别的任务占着），用 `run_phase2 --server-workers "16,48"` 把 16 个 worker 绑到 `servers[0]`、48 个绑到 `servers[1]`（长度须等于 `--servers` 端点数，和覆盖 `--workers`）。同一比例同时作为 `server_capacities` 传给 `ConductorDriver`；`assign_servers` 按 `weight / capacity` 放置 yaml（16:48 → ~1:3 的 episode 落到各 server），使两台**同时收工**，不让 worker 少的那台成瓶颈。worker 是**严格亲和**的（绑某端点的 worker 只跑分到该端点的 yaml），所以 worker 分配与 yaml 放置必须按同一比例，二者由该参数一并设好。留空 = 均匀轮询 + 等容量（旧行为，对应上面 B 的 48/48 对等例子）。

GPU 显存 ≈ `进程数 × ~7.5 GB`（每进程一份权重）；40 GB A100 上每 GPU ≤3 进程（再多 per-GPU 算力饱和，见 §8.3）。

---

## 2. 两个你要实现的接口

| 接口 | 在哪 | 你负责 |
|------|------|--------|
| `ExperimentStrategy` | `exp/...`（策略层） | 实验剧本：`plan()` 产出 stage 图 + `on_stage_begin/complete/on_resume` 编排控制帧 |
| `EpisodeRunner` | `examples/...`（执行层） | 跑一个 episode（连 server、infer、回报）。LIBERO 已有现成实现可复用 |

driver 核心保证的顺序（你只需依赖）：`upstream.on_stage_complete → downstream.on_stage_begin → downstream episodes ready → 全 done → downstream.on_stage_complete`。`on_stage_begin/complete` 在**独立线程**执行（不阻塞 worker pull）。

---

## 3. 编写 `ExperimentStrategy`（核心）

接口（`src/openpi/conductor/strategy.py`）：

```python
class ExperimentStrategy(abc.ABC):
    @abc.abstractmethod
    def plan(self, yamls: list[str], server_assignment: dict[str, ServerEndpoint]) -> TaskGraph: ...
    def on_stage_begin(self, stage, ctl, ctx): ...      # stage 开始前的有序 setup（default no-op）
    def on_stage_complete(self, stage, ctl, ctx): ...   # stage 全 done 后的 handoff/teardown
    def on_resume(self, stage, ctl, ctx): ...           # 被触发式 server 自愈
```

- `ctl` 是到该 stage 所属 server 的 `WebsocketClientPolicy`（driver 连接池给你，**复用**）。
- `ctx` 是 `StageContext`（线程安全黑板），把 warmup 产出的 calibration buffer 交给下游 eval stage。

### 3.1 `plan()`：构造 stage 图

产出 `TaskGraph`：一组 `Stage`（每个含若干 `EpisodeTask`）+ stage 间依赖 + `CalibrationArtifact`。

```python
from openpi.conductor import task as T

def plan(self, yamls, server_assignment):
    g = T.TaskGraph()
    for yaml_id in yamls:
        server = server_assignment[yaml_id]            # driver 算好的 yaml→server 归属
        g.add_calibration(T.CalibrationArtifact(        # warmup→eval 的 calib（解耦"数据"与"产出它的 stage"）
            calib_id=yaml_id, source="warmup_stage",
            warmup_stage_id=f"{yaml_id}:warmup", cleanup_id=yaml_id))
        g.add_stage(T.Stage(                            # eval stage（消费 calib）
            stage_id=f"{yaml_id}:eval", yaml_id=yaml_id, phase="eval", server=server,
            episodes=self._episodes(yaml_id, "eval", server),
            consumes_calib_id=yaml_id, setup={"eval_yaml": f"/cfg/{yaml_id}.yaml"}))
        g.add_stage(T.Stage(                            # warmup stage（产出 calib）
            stage_id=f"{yaml_id}:warmup", yaml_id=yaml_id, phase="warmup", server=server,
            episodes=self._episodes(yaml_id, "warmup", server),
            produces_calib_id=yaml_id, setup={"warmup_yaml": f"/cfg/{yaml_id}__warmup.yaml"}))
        g.add_dependency(f"{yaml_id}:warmup", f"{yaml_id}:eval")  # warmup→eval barrier
    return g
```

episode 列表（`task_uid` 必须**确定性派生**，续跑才能幂等匹配账本）：

```python
def _episodes(self, yaml_id, phase, server):
    return [
        T.EpisodeTask(
            task_uid=T.make_task_uid(yaml_id, phase, task_id, ep),  # 确定性
            yaml_id=yaml_id, phase=phase, experiment=self._task_suite,
            task_id=task_id, episode_idx=ep, orig_init_state_idx=ep,
            server_host=server.host, server_port=server.port, bundle_id=yaml_id)
        for task_id in self._task_ids for ep in range(self._trials[phase])
    ]
```

**要点**：`phase` 是不透明标签（worker 不解释）；核心只区分 **warmup（有 server 副作用，stage 原子重试）** 与 **eval（幂等，episode 级重试）**。`setup` 字典是你给自己用的（yaml 路径等）。driver 启动时 `TaskGraph.validate()`（拒绝悬空 calib / 依赖环）。

### 3.2 `on_stage_begin()`：stage 开始前的有序 setup

```python
def on_stage_begin(self, stage, ctl, ctx):
    if stage.phase == "warmup":
        ctl.load_cache_config(yaml_content=_read(stage.setup["warmup_yaml"]),
                              yaml_id=f"{stage.yaml_id}__warmup")
    else:  # eval
        buf = ctx.get(stage.consumes_calib_id) or {}
        if buf:
            ctl.preload_normalizer_buffer(stage.yaml_id, buf)   # 先 preload
        ctl.load_cache_config(yaml_content=_read(stage.setup["eval_yaml"]),
                              yaml_id=stage.yaml_id)            # 再 load eval
```

> ⚠ **eval 必须先 `preload_normalizer_buffer` 再 `load_cache_config`**：config 加载对缺失的 WarmupPool 是 fail-fast。

### 3.3 `on_stage_complete()`：stage 完成后的 handoff

```python
def on_stage_complete(self, stage, ctl, ctx):
    if stage.phase == "warmup":
        content = ctl.fetch_dump(f"{stage.yaml_id}__warmup")   # 取回 server 落盘的 dump
        buf = aggregate_dump(content)                          # 聚合成 {factor_key: [floats]}
        ctx.publish(stage.produces_calib_id, buf)              # 交给下游 eval 的 on_stage_begin
    else:
        ctl.unload_warmup_buffer(stage.yaml_id)                # 清理 WarmupPool + dump
```

**这就是旧 `run_phase.py` 7 步在新框架里的样子**——你不再写 spawn worker、等进程、切 yaml、断点续跑。完整实现见 [`exp/verdict_factor_judge/strategies/warmup_eval_strategy.py`](../../exp/verdict_factor_judge/strategies/warmup_eval_strategy.py)。

### 3.4 `on_resume()`：server 自愈（可选）

续跑时（driver 重启，journal 非空）eval setup 前核心会调它，让你清掉可能失效的 server 端 WarmupPool：

```python
def on_resume(self, stage, ctl, ctx):
    if stage.phase == "eval":
        with contextlib.suppress(Exception):
            ctl.unload_warmup_buffer(stage.yaml_id)  # 丢旧 pool，on_stage_begin 会重新 preload
```

### 3.5 cleanup_id / dump 命名约束（务必遵守）

现有 server `unload_warmup_buffer(id)` 只接受一个 id 并派生 `<id>__warmup.jsonl`。所以 warmup dump **必须**命名 `<cleanup_id>__warmup`，`CalibrationArtifact.cleanup_id` 就是这个 id。1:1 场景 `cleanup_id = eval_yaml_id`；共享场景用统一 `cleanup_id`。全程落在现有 server 协议内（**不改 server**）。

---

## 4. `EpisodeRunner`：复用或自写

LIBERO 已有现成实现 [`examples/libero/episode_runner.py`](../../examples/libero/episode_runner.py)（复用经验证的 `main._run_episode`），多数实验**直接用它**。它负责：连 `task.server`、`select_bundle(task.bundle_id)`、跑一个 episode、周期 `report(step, actions_per_s, hit_type)`、回传结果 + per-step `__hit_meta__` 行；连接跨 episode 复用。

自写新环境 runner（接口 `src/openpi/conductor/worker.py`）：

```python
class MyEpisodeRunner(EpisodeRunner):
    def run(self, task, report) -> EpisodeResult:
        client = self._ensure_client(task)          # 连 task.server + select_bundle
        ...                                          # reset env, infer loop, report(...)
        return EpisodeResult(task.task_uid, success=done, n_steps=n, per_step_rows=rows)
    def close(self): ...
```

> worker 是"无脑"的：不解释 `phase`、不持调度策略。`attempt` 字段由 worker 自动回传（超时回队后的 stale-result fence），你不用管。

---

## 5. 启动实验：`ConductorDriver` + worker

### 5.1 单机 / 进程内（开发调试）

```python
from openpi.conductor import ConductorDriver, ServerEndpoint
from examples.libero.episode_runner import default_client_factory

driver = ConductorDriver(
    MyStrategy(task_ids=range(10), warmup_trials=2, eval_trials=10, ...),
    yaml_weights={"cell_a": 100, "cell_b": 100},    # 每 yaml episode 权重（server 均衡用）
    servers=[ServerEndpoint("server1.host", 8001), ServerEndpoint("server1.host", 8002)],
    journal_path="run.jsonl",                        # 断点续跑账本
    ctl_factory=default_client_factory,              # driver 用它建到各 server 的控制连接
    episode_timeout_s=1800,                          # episode 墙钟超时回队
)
driver.run()   # 跑到所有 stage done 返回
```

### 5.2 跨机 / 多 server（生产）

- **按 server 分配 worker**（48 连 server1、48 连 server2）：每台 client 机起一个 `WorkerAgent`，按 `WorkerSpec(worker_id, server_key, gpu_id)` 在本机 fork worker 进程（绑 GPU/EGL slot，单卡 ≤15）；worker 直连 driver pull 端口。
- **断点续跑**：重启 driver（同 `journal_path`）→ 已完成 episode 自动跳过，warmup stage 整体重跑（stage 原子）。
- **监控**：传 `Monitor(scheduler=...)`，`render()` 给聚合视图（done/total/SR + 各 server worker 数 + 吞吐）。

```python
from openpi.conductor import WorkerAgent, WorkerSpec
specs = [WorkerSpec(f"w{i}", server_key="server1.host:8001", gpu_id=str(i % 8)) for i in range(48)]
WorkerAgent(specs, driver_host="driver.host", driver_port=9000).run()  # 每台 client 机
```

---

## 6. 调度语义（策略作者需知道的契约）

引擎替你保证（写策略时可依赖）：

- **永不空转 + yaml 亲和**：worker 做完立刻领下一个；同 server 优先维持"激活 yaml 集合最小"（warmup 与 eval **默认均 ≤2 并行**填 barrier/straggler 空隙——第 2 个 yaml 仅在第 1 个 ready episode 取尽且 worker 空闲时激活；同 keybuilder yaml 经 BackendPool 共享 backend，几乎不增显存。设 `eval_concurrency=1` 可换最省显存但末尾有空转气泡）。**subtask/yaml 间无等待泡沫**。
- **warmup→eval barrier**：eval episode 在其 warmup stage 全 done + 你的 `on_stage_complete` 返回前不会被派发。
- **warmup 原子**：warmup 失败/超时 → 整 stage 作废重跑（不会 episode 级重跑而重复污染 server dump）；eval 失败 → 单 episode 回队。
- **retry 分类**：网络/超时/crash → 可重试；`ConfigValidationError` 等致命 → 不重试、标 stage FAILED 并级联下游。
- **stale-result fence**：超时回队 + 重 dispatch 后，旧 worker 迟到的 result（低 `attempt`）被拒。
- **co-location / fan-out**：消费同一 `calib_id` 的 eval stage 优先同 server；跨 server 时引擎把 buffer fan-out preload 到每个相关 server（你只 `ctx.publish` 一次）。

---

## 7. 常见模式

- **纯 eval（无 warmup）**：`plan()` 不建 warmup stage、eval 的 `consumes_calib_id=None`（参考 `WarmupEvalStrategy(skip_warmup=True)`）。
- **共享 warmup（phase5 G3）**：多 eval stage 的 `consumes_calib_id` 指向同一 `CalibrationArtifact` → warmup 只跑一次，buffer 经 `ctx` 分发。
- **历史 warmup（phase5 G5）**：`CalibrationArtifact(source="historical_file", historical_path=...)`，无 warmup stage；eval 的 `on_stage_begin` 直接从文件聚合后 preload。

---

## 8. Server 调优

### 8.1 Batch 参数（`max_batch_size`, `max_wait_ms`）

coordinator 在凑满 `max_batch_size` 或 `max_wait_ms` 超时时发一个 stage batch。启动时用环境变量设（无需改代码），或运行时热切换：

```bash
BATCHING_MAX_BATCH_SIZE=32 BATCHING_MAX_WAIT_MS=25 python scripts/serve_policy.py ...
# 运行时热切（对一个端点）：
python exp/serving_benchmark/dump_mem.py --host <ip> --port 8001 set-batch --max-batch-size 32 --max-wait-ms 25
```

LIBERO 闭环下 batch 很少填满（每窗口请求少），长 `max_wait_ms` 主要是**纯延迟税**。实测 sweet spot `max_wait_ms=25`（10ms 太激进 batch 不成形；>50ms 只加延迟），`max_batch_size=32` 是安全上限。

### 8.2 CPU 线程过订阅

cache 搜索路径（`InMemoryBackend.search` 的 `cosine_similarity`）释放 GIL 落入 BLAS，多并发 worker 易过订阅。启动前设 `OMP_NUM_THREADS` / `MKL_NUM_THREADS` 为 `cpu_count() // 典型并发 worker 数`。

### 8.3 每 server 推荐配置（自动调优实测）

`pi05_libero` + phase5 cache mix（FULL_HIT/WARM_START/MISS），每 GPU 起 **3 个独立 `--concurrent` 端点**：

| Server | 端点数 | client workers | `max_wait_ms` | 吞吐 |
|--------|:------:|:--------------:|:-------------:|:----:|
| a100 (A100-40GB) | 3 | 48 | 25 | ~24 inf/s |
| jupyter (H200) | 3 | 48 | 25 | ~31 inf/s |
| fleet (a100+jupyter) | 3+3 | 48+48 | 25 | ~48-51 inf/s |

单进程 baseline ≈12 inf/s → 3 端点 ≈2.4×、fleet ≈4.3×。吞吐天花板是**闭环推理延迟**（stage3 denoise ~1s/call 占 client wall-clock ~86%），非 batch/queue。每 GPU 不超过 3 端点（再多 per-GPU 算力饱和）。

重新调优（换模型/GPU/cache mix）用 `exp/serving_benchmark/autotune_workers.py`（geometric bracket → golden-section → USL fit）确定每端点最优 worker 数 + `max_wait_ms`。完整 benchmark 见 [`serving_benchmark.md`](serving_benchmark.md)。

---

## 9. 测试你的策略

- **`plan()` 纯逻辑测**（无 GPU/server）：构造策略 → `plan(yamls, assignment)` → 断言 stage/依赖/calib 结构 + `validate()` 通过。见 [`tests/exp/test_warmup_eval_strategy.py`](../../tests/exp/test_warmup_eval_strategy.py)。
- **端到端集成测**：`FakeEpisodeRunner` + fake ctl（不连真 server）跑 mock 实验，验证无空隙/续跑/重试/barrier。见 `tests/conductor/test_integration.py` + `conftest.py`。

---

## 10. 硬约束与部署约束（不可违反）

- **C1 — non-concurrent 原始单连接结构**：`--non-concurrent` 路径无 coordinator/bundle/lazy；数值匹配当前 sdpa 模型，不与历史 eager baseline 直接 bit-identical（用于量延迟上界）。
- **C2 — runtime write-frozen**：backend 启动后冻结，`write_policy` 必须 `"never"`（否则启动 fail-fast `ConfigValidationError`）；cache artifact 离线构建（`exp/common/factor_postprocess.py`）。
- **server 端点 = `--replicas` 公共端口 或 独立 `--concurrent` 端点**（§1.3）：`replica_proxy` 的 `fetch_dump` 已 fan-out + 拼接各 replica 切片（`merge_dump_replies`），warmup→eval dump 经 router 完整，故两种都受支持。
- **单卡 ≤15 worker**（MuJoCo EGL 上下文上限）：`WorkerAgent` 按 (机器, 卡) 配额 fork。

---

## 11. Troubleshooting

- **`BackendFrozenError: backend is frozen`**：运行时有人写 backend。常见：自定义代码直接 `storage.batch_insert(...)`（移到离线）；`write_policy` 非 `never` 漏过校验（查启动/`load_cache_config` 日志的 `ConfigValidationError`）；同 fingerprint 第二次 `load_cache_config`（被 pool 拦，返回已冻结 backend，无实际二次 load）。
- **`select_bundle: unknown bundle_id`**：`select_bundle("foo")` 前没 `load_cache_config(..., bundle_id="foo")`。先 load，或用 `"default"`（最近一次 `load_cache_config` 隐式填充）。
- **`select_bundle or episode_start{bundle_id} required before infer`**：infer 前没绑 bundle 且无 `"default"` slot。先发 `select_bundle` 或带 `bundle_id` 的 `episode_start`。
- **吞吐低于 Mode 0 baseline**：CPU 线程过订阅（§8.2）；误传 `--non-concurrent`；worker 请求率太稀疏（coordinator 只能 batch 它在 `max_wait_ms` 内看到的）；大 pkl 的 CP1 搜索慢（用 M7 driver 看 `cp1_search` 计时列）。

---

## 12. 参数全集（Parameter Reference）

> 之前散落在源码 docstring 里、用户文档未收录的参数。批量调优/排查时对照本表。Batch 参数详见 §8.1，调度语义见 §6。

### 12.1 调度器 `EpisodeScheduler`（经 `ConductorDriver(scheduler_kwargs={...})` 传）

| 参数 | 默认 | 含义 |
|---|---|---|
| `eval_concurrency` | `2` | 每 server 同时**激活**的 eval yaml 数上限。`2`（默认，2026-05-26 起；原为 1）让当前 yaml 末尾 straggler 收尾时下一个 yaml 提前激活、空闲 worker 立刻领活，**消除 barrier 气泡**（实测 util 从末尾掉 0–10% 变为持平 98–100%）——同 keybuilder 相邻 yaml 共享 backend（BackendPool fingerprint），库不重复加载、几乎不增显存。设 `1` 换最省显存（eval 长连接 + KV 是显存主源），代价是末尾空转气泡 |
| `warmup_concurrency` | `2` | 每 server 同时激活的 warmup yaml 数上限。warmup episode 远少于 worker（如 2 vs 48），放宽填 barrier 空隙换利用率 |
| `max_episode_retries` | `3` | 单个 eval episode 可重试次数（网络/超时/crash 类可重试错误）|
| `max_warmup_stage_retries` | `3` | warmup stage **整体**重试次数（warmup 原子：失败先 `unload_warmup_buffer` 再整段重跑，不做 episode 级）|
| `max_setup_retries` | `3` | `on_stage_begin`/`on_stage_complete` hook 重试次数；超限或致命（`ConfigValidationError`）→ 标 stage FAILED 并级联下游 |

### 12.2 `ConductorDriver`

| 参数 | 默认 | 含义 |
|---|---|---|
| `episode_timeout_s` | `1800.0` | episode 墙钟超时；卡在 infer 不退的 worker 的 in-flight episode 被回收（eval 回队 / warmup 整 stage 作废）|
| `bind_host` | `"127.0.0.1"` | driver pull-server 绑定 host（worker 连这里取任务）|
| `bind_port` | `0` | `0`=随机端口 |
| `scheduler_kwargs` | `None` | 透传给 `EpisodeScheduler`（设 §12.1）|
| `colocation` | `None` | `{yaml_id: server_key}` 强制归属（co-location）|
| `poll_s`（`run()` 参数）| `0.05` | 主循环 poll 间隔（秒）|

### 12.3 `WorkerAgent` / `WorkerSpec`

| 参数 | 默认 | 含义 |
|---|---|---|
| `poll_s`（WorkerAgent）| `1.0` | worker 监管轮询间隔（检测并重启死 worker）|
| `conda_env`（WorkerSpec）| `""` | 设则 worker 经 `conda run -p <env>` 启（隔离解释器，如有 libero+openpi_client 的 LIBERO sim env）；空=用 driver 自己的 python |
| `worker_module` | `examples.libero.worker_entry` | worker `python -m` 目标模块 |
| `gpu_id` | — | 该 worker 的 `CUDA_VISIBLE_DEVICES`（EGL slot 绑定）|

### 12.4 `run_phase2` CLI（weighted_sum 实验入口，作为一个 driver 范例）

| flag | 默认 | 含义 |
|---|---|---|
| `--yaml-dir` | （必填）| 一批 cache yaml 的目录 |
| `--init-map` | （必填）| `libero_*_init_map.json`（held-out 防泄漏）|
| `--journal` | （必填）| 续跑账本 jsonl |
| `--servers` | （必填）| 逗号分隔 `host:port` 端点 |
| `--task-ids` | `0-9` | LIBERO task 选择 |
| `--eval-trials` | `20` | 每 task 的 held-out trial 数 |
| `--task-suite` | `libero_spatial` | task suite |
| `--total-inits` | `50` | 全集 init 数（held-out 计算用）|
| `--episode-timeout-s` | `1800` | → driver `episode_timeout_s` |
| `--workers` | `48` | 本机 worker 进程数 |
| `--gpus` | `8` | round-robin worker 的 GPU 数（EGL slot；单卡 ≤15 worker，§10）|
| `--conda-env` | `""` | → `WorkerSpec.conda_env` |
| `--bind-host` | `127.0.0.1` | → driver `bind_host` |

> ⚠ `run_phase2` 暂未暴露 `eval_concurrency` 等 `scheduler_kwargs`（用默认 1/2）。要调需在代码里给 `ConductorDriver` 传 `scheduler_kwargs={"eval_concurrency": 2}`，或加一个 CLI flag。

### 12.5 `serve_policy.py` CLI

| flag | 默认 | 含义 |
|---|---|---|
| `--port` | `8000` | 监听端口 |
| `--replicas` | `1` | 单公共端口后的并发副本进程数（`>1` 启 `replica_proxy` router，per-connection 路由 + broadcast bundle/preload + aggregate fetch_dump）|
| `--replica-spawn-batch` | `0` | `replicas>1` 时分批 spawn：每批并发起这么多子进程、等其加载+bind 完再起下一批（`0`=一次性全起；大库分批防同时加载撑爆）|
| `--concurrent` / `--non-concurrent` | `True` | 并发多 client + 动态 bundle 热切（默认）；`--non-concurrent`=C1 原始单连接极速基线（无 coordinator/bundle/lazy；当前 sdpa 数值）|
| `--cache-config` | `None` | 启动时加载的 cache yaml |
| `--record` / `--collect` / `--collect-dir` / `--collect-images` / `--cache` | … | 录制 / 采集建库相关（采集即生成 h5：`collection_policy` 抽 vision/prompt embedding 存 float16）|

### 12.6 Server 环境变量

| env | 默认 | 含义 |
|---|---|---|
| `OPENPI_SERVER_GPU_MEMORY_LOCK` | `1` | `1`=锁住已 reserve 的 GPU 显存块不还系统（防共享机被抢/防碎片）；`0`=旧的 release-to-driver 行为 |
| `PYTORCH_CUDA_ALLOC_CONF` | — | torch CUDA 分配器配置；常用 `expandable_segments:True` 减碎片 |
| `BATCHING_MAX_BATCH_SIZE` | `8` | coordinator 一个 stage batch 的 episode 上限（§8.1）|
| `BATCHING_MAX_WAIT_MS` | `10` | 凑批超时；LIBERO 闭环很少填满，长=纯延迟税，实测 sweet spot `25`（§8.1）|
| `BATCHING_STAGE1/2/3_WORKERS` | `1` | 各 stage 的 batching worker 线程数（stage3 可多线程 + 多 CUDA stream 并发 denoise 提吞吐）|
| `OPENPI_DISABLE_STAGE_STREAMS` | （空）| `=1` 关 per-stage CUDA stream（stage 间走 interceptor host-side CP 逻辑会强制 sync，stream 收益有限时可关）|
| `OPENPI_STAGE3_BUCKET_FIRST` | （空）| `=1` 用 stage3 先分桶循环；默认 generic pull-then-group（实测 a100 吞吐更好）|
| `OPENPI_MONITOR_LEVEL` | （空）| 监控埋点级别（见 `serving/monitor.py`）|
| `OPENPI_MONITOR_AUTOFLUSH_DIR` | `""` | 监控指标自动 flush 目录 |
| `TORCHINDUCTOR_CACHE_DIR` | — | torch.compile inductor 编译缓存目录 |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | — | BLAS 线程数；多并发 worker 防 CPU 过订阅，设 `cpu_count() // 并发 worker 数`（§8.2）|

---

## See also

- [`docs/architecture/experiment_conductor.md`](../architecture/experiment_conductor.md) — 编排框架架构
- [`docs/architecture/cache_system.md`](../architecture/cache_system.md) §9.X — C1/C2、BackendPool、BundleDispatcher、BatchingCoordinator 设计
- [`docs/experiments/serving_benchmark.md`](serving_benchmark.md) — M7 吞吐 benchmark runbook
- 参考实现：策略 [`warmup_eval_strategy.py`](../../exp/verdict_factor_judge/strategies/warmup_eval_strategy.py)、执行器 [`episode_runner.py`](../../examples/libero/episode_runner.py)、worker 入口 [`worker_entry.py`](../../examples/libero/worker_entry.py)
