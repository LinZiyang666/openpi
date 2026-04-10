# Concurrent Inference Plan

> **Status:** Plan
> **Date:** 2026-04-05

## 1. 背景

当前推理系统是完全串行的：一个 `main.py` 客户端连接一个 `serve_policy.py` 服务端，逐 task、逐 episode 执行。LIBERO 的 5 个 task suite 各含 10 个 task × 50 episodes = 2500 个 episode，只能排队跑。

**目标：** 多个仿真环境并发连接同一个服务端，各自独立跑 forward + cache 流水线，共享同一份 GPU 模型权重和 Qdrant DB。不做 batch 推理，每个连接独立走完整推理路径，GPU 通过 CUDA stream 自动串行 kernel。

## 2. 架构

```
Worker 0 ──→ WebSocket ──→ InferenceInterceptor_0 + Orchestrator_0 ─┐
Worker 1 ──→ WebSocket ──→ InferenceInterceptor_1 + Orchestrator_1 ─┤──→ 共享 Pi0Policy (GPU)
Worker 2 ──→ WebSocket ──→ InferenceInterceptor_2 + Orchestrator_2 ─┘        ↕
                                                                        共享 Qdrant DB
```

- **模型只加载一次**，所有连接共享同一个 `Policy` 对象（引用传递）
- **每个连接**创建独立的 InferenceInterceptor + CacheOrchestrator + SystemTimer
- **Qdrant** 天然支持多客户端并发读写
- **Cache 数据不隔离**（设计上就是共享的，相似观测可以跨连接命中）

## 3. 改动清单

### 3.1 `src/openpi/cache/timing.py` — 加 `quiet` 参数

**改动量：** 2 行

- `SystemTimer.__init__` 加 `quiet: bool = False` 参数
- `_print_summary()` 开头加 `if self._quiet: return`

**效果：** `quiet=True` 时不打印 timing 表格到 stdout，CSV 输出不受影响。`quiet=False`（默认）行为完全不变。

### 3.2 `src/openpi/serving/websocket_policy_server.py` — 支持并发

**改动量：** ~30 行

**构造函数新增参数：**

```python
def __init__(
    self,
    policy,
    host="0.0.0.0",
    port=None,
    metadata=None,
    concurrent=False,                      # 新增
    connection_policy_factory=None,        # 新增
) -> None:
```

- `concurrent=False`：是否允许多连接
- `connection_policy_factory`：`Callable[[BasePolicy], BasePolicy]`，并发模式下每个连接调用一次，传入 base policy，返回该连接的 wrapper stack

**`_handler` 改动：**

```
非并发模式：
  - 新增 _active_connection 标志
  - 第二个连接进来时直接 close(1013, "Server is in single-connection mode")
  - conn_policy = self._policy（和现在一样）

并发模式：
  - conn_policy = self._connection_policy_factory(self._policy)
  - 连接关闭时 conn_policy 自然被 GC

两种模式共同：
  - policy.infer(obs) 改为 await asyncio.to_thread(conn_policy.infer, obs)
  - 其余 episode_start/end 等控制消息不变，只是操作对象从 self._policy 变为 conn_policy
```

**`asyncio.to_thread` 的必要性：** 当前 `policy.infer()` 是同步阻塞调用，会冻结 event loop。改为 `to_thread` 后，多个连接可以被同时 accept 和处理。即使非并发模式也受益（health check 不会被阻塞）。

### 3.3 `scripts/serve_policy.py` — 加 `--concurrent` 参数

**改动量：** ~30 行

**Args 新增：**

```python
concurrent: bool = False   # 启用并发连接模式
```

**`main()` 逻辑分支：**

```
concurrent=False（默认）：
  → 现有行为完全不变，启动时创建完整 wrapper chain，传给 server

concurrent=True：
  → 启动时只创建 base policy（不包 wrapper）
  → 定义 factory 闭包，捕获 args 和 cache_config
  → factory 内部调用 build_cache_components() 创建独立组件集，quiet=True
  → 传 base_policy + factory 给 server
  → metadata 加入 {"concurrent": True}
```

**Factory 闭包：**

```python
def _make_connection_policy(base_policy):
    p = base_policy
    if args.cache_config is not None:
        cfg = load_cache_config(args.cache_config)
        components = build_cache_components(cfg, quiet=True)
        orchestrator = CacheOrchestrator(...)
        p = InferenceInterceptor(p, timer=components["timer"], orchestrator=orchestrator)
    elif args.cache:
        timer = SystemTimer(enabled=True, quiet=True)
        p = InferenceInterceptor(p, timer=timer)
    if args.record:
        p = PolicyRecorder(p, "policy_records")
    if args.collect:
        collector = EpisodeDataCollector(base_dir=args.collect_dir)
        p = CollectionPolicy(p, collector)
    return p
```

**注意：** `InferenceInterceptor.__init__` 内部调用 `torch.compile`，但 PyTorch 会按函数 identity 缓存编译结果，所以多个 Interceptor 指向同一个 model 时不会重复编译。

**orchestrator 日志：** gate/judge 的日志用的是 `logger.info`（标准 logging），并发模式下通过 `build_cache_components` 传 `quiet=True` 控制。具体做法：在 `build_cache_components` 中接受 `quiet` 参数，当 `quiet=True` 时将 orchestrator 的 logger level 设为 WARNING（只需一行 `logging.getLogger("openpi.cache.orchestrator").setLevel(logging.WARNING)`），或者在 CacheOrchestrator 中加 `quiet` 参数来跳过 gate/judge 的 info 日志。

### 3.4 `examples/libero/main.py` — 加 `--num-workers` 并发

**改动量：** ~80 行（主要是新增 `_eval_concurrent` 函数）

**Args 新增：**

```python
num_workers: int = 1   # 并发 worker 数，1=串行（默认）
```

**主入口分流：**

```python
def eval_libero(args):
    # ... seed, task_suite, max_steps 初始化不变 ...
    if args.num_workers <= 1:
        _eval_serial(args, task_suite, num_tasks, max_steps)
    else:
        _eval_concurrent(args, task_suite, num_tasks, max_steps)
```

**`_eval_serial`：** 将现有 `eval_libero` 的循环体原封不动搬进来，行为零变化。

**`_eval_concurrent` 设计：**

```python
def _eval_concurrent(args, task_suite, num_tasks, max_steps):
    # 1. 服务端兼容性检查
    probe = WebsocketClientPolicy(args.host, args.port)
    meta = probe.get_server_metadata()
    if not meta.get("concurrent", False):
        raise RuntimeError(
            "Server does not support concurrent mode. "
            "Start server with --concurrent."
        )
    probe._ws.close()

    # 2. 任务队列（task 级粒度）
    task_queue = queue.Queue()
    for task_id in range(num_tasks):
        task_queue.put(task_id)

    # 3. 共享状态
    lock = threading.Lock()
    counters = {"episodes": 0, "successes": 0}
    pbar = tqdm.tqdm(total=num_tasks * args.num_trials_per_task, desc="Eval")

    # 4. Worker 函数
    def worker():
        client = WebsocketClientPolicy(args.host, args.port)
        while True:
            try:
                task_id = task_queue.get_nowait()
            except queue.Empty:
                break
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, desc = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            for ep_idx in range(args.num_trials_per_task):
                gid = task_id * args.num_trials_per_task + ep_idx
                client.episode_start(experiment=args.task_suite_name,
                                     task=str(desc), episode_id=gid)
                success = _run_episode(env, client, initial_states[ep_idx],
                                       desc, args, max_steps)
                client.episode_end(success=success)
                with lock:
                    counters["episodes"] += 1
                    if success:
                        counters["successes"] += 1
                    pbar.update(1)
                    rate = counters["successes"] / counters["episodes"]
                    pbar.set_postfix(sr=f"{rate:.1%}")
            env.close()
        client._ws.close()

    # 5. 启动
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = [pool.submit(worker) for _ in range(args.num_workers)]
        for f in as_completed(futures):
            f.result()
    pbar.close()
    logging.info(f"Done: {counters['successes']}/{counters['episodes']} "
                 f"({counters['successes']/counters['episodes']:.1%})")
```

**`_run_episode` 辅助函数：** 从现有 episode 循环中提取，去掉视频录制和 display 逻辑，返回 `bool`。串行模式也可以复用这个函数。

**任务分配与种子：**
- 任务队列按 `[0, 1, ..., num_tasks-1]` 顺序填充
- 5 个 worker 时：前 5 个 task 被立即取走，谁先完成就取下一个
- 每个 task 的 `env.seed(args.seed)` 和 `initial_states` 都与串行一致，结果可复现
- `global_episode_id = task_id * num_trials_per_task + episode_idx`，确定性计算，不依赖执行顺序

**并发时禁用的功能：**
- 视频录制（不调用 `_save_video`）
- 渲染窗口（不调用 `cv2.imshow`）
- per-task/per-episode logging（避免交错输出）
- 只保留一个全局进度条 + 成功率

## 4. 线程安全分析

| 组件 | 并发安全性 | 说明 |
|------|-----------|------|
| Pi0Policy (GPU model) | 安全 | PyTorch eval 模式下 forward pass 线程安全，CUDA stream 串行 kernel |
| InferenceInterceptor | 安全 | 每个连接独立实例，无共享可变状态 |
| CacheOrchestrator | 安全 | 每个连接独立实例 |
| SystemTimer | 安全 | 每个连接独立实例 |
| Qdrant Client | 安全 | 原生支持多客户端并发 |
| task_queue (客户端) | 安全 | `queue.Queue` 线程安全 |
| counters + pbar (客户端) | 安全 | `threading.Lock` 保护 |
| asyncio event loop (服务端) | 安全 | `to_thread` 不阻塞 loop，handler 协程正常调度 |

## 5. 兼容性保证

- `concurrent=False`（默认）：服务端行为与改动前完全一致
- `num_workers=1`（默认）：客户端行为与改动前完全一致
- 并发客户端 → 非并发服务端：客户端检测 metadata 后立即报错
- 非并发服务端 → 第二个连接：服务端发 close frame (code=1013) 拒绝
- 不修改任何现有类的公开接口，只新增参数（都有默认值）

## 6. 实施顺序

1. **timing.py** — 加 `quiet` 参数（最小、最独立）
2. **websocket_policy_server.py** — 并发连接支持 + `asyncio.to_thread`
3. **serve_policy.py** — `--concurrent` 参数 + factory 闭包
4. **main.py** — `--num-workers` + 线程池 + 进度条

## 7. 验证方式

1. `num_workers=1` 跑一轮，确认结果与改动前一致
2. `--concurrent` 启动服务端，`num_workers=1` 连接，确认功能正常
3. 非并发服务端 + `num_workers=2` 客户端，确认报错信息
4. `--concurrent` + `num_workers=3`，确认多连接并行、进度条正常、无交错输出
5. 对比串行和并发模式下相同 task 的成功率，验证可复现性
