# X17 — 让单臂相位吃满整池：GR00T 实验链接 conductor

> **Level**: L3 · **Authority**: Execution · **Stage**: **v5 · G2 APPROVED · §6 Verify**
> **§4 本地测试（advisory，非 §6 Verify）**：裸全量 `uv run pytest --ignore=tests/review_tests`
> 得 `5 failed / 4041 passed / 59 skipped`，5 条失败与 HEAD 既有基线逐条一致（**零新增**）。
> ⚠ **口径更正（2026-08-25）**：此前抬头写「§6 Verify 过」是错的 —— 章程 §5 禁止在拿到
> `APPROVED` G2 裁决前进入 §6，本人无权自行宣布 Verify 通过。上述数字只是 §4 允许的
> **本地测试产出（advisory）**，程序性测试权威在 §6。
> **未实施**：G-perf 与 T13 端到端冒烟。X16 已于 2026-08-25 08:51 收官、整池已空，
> **阻塞项从「资源」变成「G2」**——它们要把新代码部署到 weilandserver + timan107
> 并跑一次真实车队作业，owner **未** override G2，故不自行启动。
> **章程依赖**：[`WORKING_AGREEMENT.md`](../WORKING_AGREEMENT.md) §2.1/§9、
> [`docs/architecture/experiment_conductor.md`](../docs/architecture/experiment_conductor.md)、
> [`docs/experiments/conductor_tutorial.md`](../docs/experiments/conductor_tutorial.md) §1.3。

---

## 0. 流程与授权

owner **override G1**（改自审 agent）；**G2 未 override**。owner 追加：**conductor 必须进范围**、**RoboCasa 也要适配**。

| 版本 | 死因（每一版都被自审判死） |
|---|---|
| v1 | 称自家调度器是「cell 级静态分配」（实为共享队列抢活）；称拆锁 = 乘性提升（GIL 实测推翻） |
| v2 | 称 conductor 天然消 bubble（`scheduler.py:246-249` 按 server 过滤、`driver.py:141` 一 yaml 一 bin） |
| v3 | sibling-shard 收益为零（自变量是 lane 数）；会切碎标定；**把 `--replicas` 排除是回避 owner 指令** |
| v4 | **把一次僵尸-worker 事故的墙钟安成一个还没跑过的相位的实测成本**，并据此推出 ≥3× 验收阈值；且 9 条实现级 BLOCKING |

**v5 与前四版最大的不同：§1 的每个数字都来自本次运行自己的链日志，不是引用、不是估计。**

⚠ X16 跑到 `GATE-PARETO-DONE` 之前只在本机改动。部署见 §9。

---

## 1. 诊断（第五版，全部基于实测）

### 1.1 相位墙钟（`/tmp/gpchain.log`，2026-08-24 本次运行）

| 相位 | 起止 | 墙钟 | 占用 |
|---|---|---|---|
| spatial warmup（1 臂 × 100 集） | 19:58:31 → 20:14:33 | **16.0 min** | **1 槽** |
| spatial smoke | → 20:18:42 | 4.1 min | 2 槽 |
| spatial sweep（16 臂 × 500 集） | → 22:36:45 | 138.1 min | 6 槽（已打满） |
| spatial **gate-only（1 臂 × 500 集）** | → 23:08:47 | **32.0 min** | **1 槽** |
| spatial 合计 | 19:58:17 → 23:08:49 | **190.5 min** | |
| l10 warmup（1 臂 × 100 集） | 23:08:59 → 23:45:01 | **36.0 min** | **1 槽** |

⇒ **单臂相位 = 48.0 min / 190.5 min = 25.2%（spatial 实测）**。
l10 的 warmup 已实测 36 min；其 gate-only 尚未跑，按 warmup 的 2.25× 比例外推约 70 min，
故 l10 的单臂占比只会更高。

**⚠ v4 的数字是伪造的**：`62–81 min` 出自 `libero_groot_ws_search.log.md:141-143`，
是**僵尸 worker 事故**的后果（`infer_ms_per_call` 88 ms → 2325 ms，26 倍），健康值 27 min；
在 `gate_pareto_plan.log.md:325` 它被引为 **`eval l10` 相位**的 min/cell，**不是 gate-only**。
v4 把它安到一个还没跑过的相位上，并据此定了 ≥3× 的验收阈值。**本节整表作废重来。**

### 1.2 病因

sweep 相位（16 臂 ≥ 6 槽）**今天已经打满**，不是问题。问题只在**单臂相位**：
队列里只有 1 个 cell ⇒ 6 个槽只有 1 个在干活，且该槽只发 `workers` 个 sim client
（spatial 12 / l10 8），而整池有 72 / 48 个。

相位墙钟 ≈ `max_lane_episodes × 单集时长`，**自变量是 lane（sim client）数**
（server 侧不是瓶颈：实测单 server 进程 CPU 161%/88 核、GPU 22%）。

### 1.3 收益上限（诚实口径）

单臂相位从 1 槽变 6 槽 ⇒ lane 数 ×6。用本线真实逐步数据实测的分片不均衡
（`shard_imbalance_probe.py`，spatial `gpgo_sp`，每集决策 cv=0.35）：

| lane 数 | 静态预切 stride | 动态派发上界 |
|---|---|---|
| 12（今天 spatial） | — | — |
| 72 | 41%+ | 理想 + 最长单集 |

⇒ conductor 的**动态派发**（worker 干完立刻拉下一集）makespan 上界 = 理想 + 最长单集；
静态预切在细粒度下急剧恶化，**这是选 conductor 而非预切 lane 文件的正面依据**。

**收益：spatial 单臂 48.0 min → 约 10–13 min，省 ~35 min / 190.5 min ≈ 18%。**
两个 suite 合计约 **1.2–1.5 h**。这就是本计划买到的全部东西 —— 不多，但 owner 要的是这条框架，
且 conductor 另有 episode 级续跑 / 统一 journal / 跨机 worker 池等本计划不计价的收益。

### 1.4 拓扑选择：6 个独立端点，**不是** `--replicas`

`conductor_tutorial.md:47-57` 给了两条都被支持的路：
**A) `--replicas N` + 单公共端口**；**B) 多个独立 `--concurrent` 端点**，注册为多个 `ServerEndpoint`。

**v5 选 B**，理由三条，都可验：

1. **核心自己的部署不变量规定的就是 B**。`src/openpi/conductor/task.py:38-47`：
   *"the conductor path connects directly to a single-process server, **never through the
   `replica_proxy` router** … Multiple replicas are modelled as **multiple independent
   `ServerEndpoint` entries**; this is a **deployment invariant**."*
   （其给出的理由「sticky `fetch_dump` 会给出 partial dump」今天已过时 —— `fetch_dump` 早已进
   `CTRL_AGGREGATE`（`replica_proxy.py:76-84,114`）—— 但**结论仍是 B**，且 §1.5 表明本线根本不用 dump。）
2. **A 的增量只有跨端点 work-stealing**，即 bubble (b)，本线实测 2.3–6.0%；
   代价是 supervisor 分支 + router 单点（`serve_policy.py:802-811`：任一 child 死 ⇒
   `_terminate_all()` + `os._exit(1)`，**爆炸半径从 1/6 变成 6/6**）+ 30 s fan-out 超时
   （`replica_proxy.py:56` 硬编码，而 broadcast 的 `load_cache_config` 要跑 GB 级
   `build_shared_storage`，6 路并发；超时 ⇒ `mark_setup_failed` ⇒ 最多重试 3 次 ⇒ 再来 3 轮 6 路并发）
   + 公共端口丢 `/healthz`（`replica_proxy.py:515-521` 不传 `process_request`）。
3. **B 保留今天已验证跑通的 6 端点拓扑**，server 侧只需加动态 bundle 一件事。

⚠ **这不是把 `--replicas` 排除在外**（v3 犯过这个错）：它是**被实测的收益差**排在后面的，
理由写在 §8 未决 1，随时可加。

### 1.5 sibling stage 在本线是安全的

v3 时对 sibling-shard 的最重指控是「N 次 `fetch_dump` 各拿 1/N、N 次 `ctx.publish` 互相覆盖，
把 warmup 标定静默切成 1/N」。**该指控对本线不成立**：X16 的 warmup 是**离线管线**
（`gate_pareto_plan.log.md:269`：warmup 逐步 jsonl → `solve_gtp.solve_all` → `emit_gate_yamls`），
`grep -rn "fetch_dump\|preload_normalizer_buffer" exp/libero_groot/` **零命中**。

尽管如此，helper 仍**硬拒**分片任何 `phase == "warmup"` 或 `produces_calib_id is not None` 的 stage
（§4.2）—— 本线不需要，但下一个用它的线可能需要，而失败是静默的。

---

## 2. 亲验记录

| # | 结论 | 锚点 | 状态 |
|---|---|---|---|
| 1 | **相位墙钟实测**（§1.1 整表） | `/tmp/gpchain.log`（weilandserver，本次运行） | ✅ **v5 新增，替换 v4 的伪造数字** |
| 2 | `62–81 min` 是僵尸 worker 事故值（88→2325 ms），健康值 27 min，且属 `eval l10` 非 gate-only | `libero_groot_ws_search.log.md:141-143`；`gate_pareto_plan.log.md:325-326` | ✅ ❌ **推翻 v4 §1.1** |
| 3 | **`ServerEndpoint` 的部署不变量规定「多个独立端点」而非 router** | `src/openpi/conductor/task.py:38-47` | ✅ ❌ 推翻 v4「零核心改动 + 走 router」 |
| 4 | 本线 warmup 是离线管线，**不调 `fetch_dump`/`preload_normalizer_buffer`** | `gate_pareto_plan.log.md:269`；`grep exp/libero_groot/` 零命中 | ✅ v5 新增 |
| 5 | supervisor 任一 child 死 ⇒ 全体 `os._exit(1)` | `scripts/serve_policy.py:802-811` | ✅ |
| 6 | router fan-out 超时 **30 s 硬编码**；`_broadcast` 是 `asyncio.gather` 并发 | `replica_proxy.py:53-56, 437-446` | ✅ |
| 7 | router 公共端口**不提供 `/healthz`** | `replica_proxy.py:515-521` vs `websocket_policy_server.py:523` | ✅ |
| 8 | 真实符号是 `ReplicaProxy(backend_host, backend_ports)` / `run_proxy(...)`，**没有 `ReplicaRouter`**；`serve` 是协程 | `replica_proxy.py:319,322,515,532` | ✅ ❌ 推翻 v4 §4.1 |
| 9 | GR00T 两入口的 `args` 是 **`argparse.Namespace`**，`dataclasses.replace` 会 `TypeError` | `serve_groot_libero.py:240`；`serve_groot_n15.py:419` | ✅ ❌ 推翻 v4 §4.1 |
| 10 | GR00T 的 `--concurrent` 是 `store_true` **默认 OFF**，**没有** `--non-concurrent`（极性与 pi0.5 相反） | `serve_groot_libero.py:234-239`；`serve_groot_n15.py:411-418` | ✅ ❌ 推翻 v4 T1 |
| 11 | **`serve_groot_libero.py` 有 `--collect-hdf5`**（建库正走这条路） | `serve_groot_libero.py:220-227, 278-297` | ✅ ❌ **推翻 v4 §4.4** |
| 12 | `build_shared_storage` 由 **server 的 `load_cache_config` handler** 调用并挂进 bundle；工厂应**读**不应重建 | `websocket_policy_server.py:793-807`；`config.py:2582-2587` | ✅ ❌ 推翻 v4「第四道守卫」 |
| 13 | 库去重**已由 `BackendPool.get_or_load` 按 fingerprint 实现** | `backend_pool.py:203-233` | ✅ ❌ 推翻 v4 R5 的严重性 |
| 14 | **`EpisodeResult` 没有** `task_id`/`orig_init_state_idx`/`episode_id`/`seed` | `task.py:100-115` vs `analyze_gate_pareto.py:104,113,177` | ✅ v5 新增，见 §4.4 |
| 15 | driver 对**每个**上报结果都追加 per-step 行（含可重试失败）；I5 对重复 `(episode_id, step_idx)` raise | `driver.py:318-328`；`analyze_gate_pareto.py:137-144` | ✅ v5 新增，见 §4.4 |
| 16 | **`WorkerSpec` 没有 `resize_size`/`replan_steps`**，而 GR00T checkpoint **必须** `--resize-size 256` | `agent.py:36-60`；`worker_entry.py:52-58`；`main.py:55`（默认 224） | ✅ v5 新增，**照 v4 写会每集失败** |
| 17 | `ConductorDriver.run()` **没有** per-step finalizer；`run_gtp.py` 靠自己的快照循环补 | `driver.py:236-243, 404-434`；`run_gtp.py:326-329,357-361,386-390` | ✅ v5 新增 |
| 18 | `eval_concurrency` 默认 2，按 **(server, phase)** 计数 | `scheduler.py:78,137-150,160-162` | ✅ 6 端点下上限是 6×2 |
| 19 | `_ep_index` 无 uid 去重、`TaskGraph.validate()` 也不查 ⇒ 重复 uid = 静默死锁 | `scheduler.py:96-100`；`task.py:239-245` | ✅ 加固项 |
| 20 | ⚠ `"default"` 槽在 flag 打开后可被热载覆盖 ⇒ 必须用非 default `bundle_id` | `websocket_policy_server.py:748,813,829-847` | ✅ |
| 21 | 编译视觉塔竞态确凿，**但该路径默认关且等价门 FAIL 已 park**；无条件 clone 只堵**输出侧**别名 | `staged.py:206,364-370,423`；`robocasa365_ws_search_plan.log.md:158` | ✅ 卫生项，不计收益 |
| 22 | `tests/review_tests/` 在 `.gitignore`、不在 HEAD ⇒ 不进 Verify 基线（既有失败 **5 条**） | `.gitignore:124` | ✅ |

---

## 3. 目标与非目标

### 目标

- **T1 · GR00T server 动态 bundle**：`--allow-dynamic-bundles`（**默认 False**），工厂改为读
  `get_current_cache_bundle(bundle_id).shared_storage` 并对**新 config** 重跑三道守卫（§4.1）。
  两个入口都做；两个入口都硬钉 `--collect-hdf5 ⇒ 单进程`（锚点 11）。
- **T2 · `src/openpi/conductor/sharding.py`**：eval-only sibling stage 构图助手，
  **硬拒** warmup / 有 `produces_calib_id` 的 stage（§4.2）。
- **T3 · `exp/libero_groot/run_conductor.py`**：6 端点 + 全部 worker，episode 级分发（§4.3–§4.5）。
- **T4 · 性能验收 G-perf**（§5），基线用 §1.1 的**实测**值。
- **T5 · RoboCasa 适配**：`serve_groot_n15.py` 同样拿到 T1；其 `run_ws_search.py` 已是
  driver/agent 拓扑，可直接消费 T2。

### 非目标

- **`--replicas`**：§1.4 —— 增量只有 2.3–6.0%，代价见锚点 5/6/7。**排在 §8 未决 1，不是排除。**
- **拆 `_InferLockedPolicy`**：GIL 实测 + 锚点 21 的竞态 + out-of-repo transform 未审计。
- **不改 pi0.5**；**不改 conductor 调度状态机**（`scheduler.py`/`driver.py` 逻辑不动）。
- **不删 `orchestrate_search.py`**（X16 的可复现产物）。

---

## 4. 设计

### 4.1 T1 · 动态 bundle

- CLI `--allow-dynamic-bundles`（默认 False）；关闭时 `_require_default_bundle` 行为**一字不变**。
- 工厂签名不变（`(shared_base_policy, bundle_id="default")`），内部改为：
  `bundle = get_current_cache_bundle(bundle_id)`（`websocket_policy_server.py:102`）；
  **`bundle is None`（例如以 `--cache-config` 起的 server 收到 `"default"`）时回落 CLI 配置**（锚点 20）。
- 对 `bundle.config` 重跑三道守卫，**各自的 raise 类型**（T3 按类型断言）：

| 守卫 | 位置 | raise |
|---|---|---|
| `validate_groot_cache_config` | `load_guard.py:53` | `ConfigValidationError` |
| `_check_libero_builder` | `serve_groot_libero.py:99` | 注入的 `fail`（并发路径 `ValueError`） |
| `validate_artifact_identity` | `load_guard.py:134` | `ConfigValidationError` |

  ⚠ **`build_shared_storage` 不是守卫**（锚点 12）：它由 server handler 调用并挂进 bundle，
  工厂**读** `bundle.shared_storage`。在工厂里重建 = 48 worker × 16 臂的 GB 级重复加载。
  库去重已由 `BackendPool` 按 fingerprint 实现（锚点 13），本计划不重复造。
- **`--collect-hdf5` 与多进程互斥**：两个入口都加（锚点 11）。

### 4.2 T2 · eval-only sibling stage

```
def shard_eval_stage(*, stage_id, yaml_id, episodes, servers, setup) -> list[Stage]
```
- `phase` 固定 `"eval"`；传入 warmup 或 `produces_calib_id` 非空 ⇒ **`raise ValueError`**（§1.5）。
- `episodes[k::N]` 步长切片（实测优于连续切分）；`dataclasses.replace` 改写每个 task 的
  `server_host/port`（`EpisodeTask` 是 frozen）。
- 空分片**照样产出**（`scheduler.py:186-189` 保证不卡死），但策略的 `on_stage_begin`
  **`if not stage.episodes: return`** —— 否则一次空热切。
- 依赖边由调用方在分片集合间做笛卡尔积。

### 4.3 T3 · runner 拓扑

- 6 个 `serve_groot_libero.py --concurrent --allow-dynamic-bundles`（端口沿用 23160-23165）；
- `ConductorDriver` 在 weilandserver：`bind_host="0.0.0.0"` + **固定** `--driver-port`
  （`driver.py:134-135` 默认是 `127.0.0.1:0`）；拓扑照 `run_ws_search.py:178-188,241-263`
  的 `--role driver|agent|all`。⚠ 其 `ctl_factory=lambda _s: _NoOpCtl()`（`:246`）**必须换成**
  真 client（`examples/libero/episode_runner.py:130-134` 的 `default_client_factory`），
  因为本轮 ctl 要真发 `load_cache_config`。
- ⚠ **worker 参数通路**（锚点 16）：`WorkerSpec` 无 `resize_size`/`replan_steps`，
  必须自带 `spawn_fn`（照 `run_ws_search.py:269-285` 的 `robocasa_spawn_fn`），
  显式传 `--resize-size 256 --replan-steps 5`，并设 `MUJOCO_EGL_DEVICE_ID`（`_default_spawn` 不设）。
- `--eval-concurrency` 必须可配（锚点 18）；6 端点下活跃上限是 `6 × k`。
- **`arms_with_work_left()`**：`run_gtp.py:106-146` 在「明确不改」的文件里 ⇒ **复制**一份到
  `run_conductor.py` 并把硬编码的 `NUM_TASKS=10` 换成 `gate_pareto_bindings.NUM_TASKS`
  （import 它会连带拉进 `exp.ablation_study.*`）。

### 4.4 T3 · 完整性产物（三条 v4 漏掉的硬约束）

1. **`<arm>.json` 的字段来源**（锚点 14）：`EpisodeResult` 没有 `task_id`/`orig_init_state_idx`/
   `episode_id`。runner 持 `uid → (task_id, orig_init_state_idx, episode_id)` 映射（plan 时就有），
   在结果回调里回填。`episode_id` **必须**用
   `examples.libero.collect_util.compute_global_episode_id(task_id, episode_idx, num_trials_per_task)`
   且 `num_trials_per_task = gate_pareto_bindings.APOOL_TRIALS`，否则与 X16 已产出的臂 join 不上。
   同一契约要求 `EpisodeTask.extra["num_trials_per_task"]` 必须设（`task.py` 的生产者契约）。
2. **per-step 按 attempt 去重**（锚点 15）：driver 对每个上报结果都追加行，重试会产出第二整套 ⇒
   撞死 I5。runner 的 `per_step_writer` 按 `task_uid` 只保留被接受的那个 `attempt`（行里有 `attempt`）。
3. **快照 finalizer**（锚点 17）：`ConductorDriver.run()` 没有，runner 必须自带快照循环。
4. **merge sidecar 的独立判据**：TCP 路径下没有「worker 文件凭空消失」这个故障模式，
   **不许伪造一个由同一份 results 推出来的恒真 sidecar**。`episodes_expected` 取自
   **strategy 的计划 uid 全集**，`episodes_reported` 取自 **journal 终态记录** ——
   这两者**会**真的不一致（重试耗尽的 uid 从不入 journal，`driver.py:302`）。
   分析器显式接受 `transport: "tcp"` 形态。

### 4.5 加固（纯断言）

- `TaskGraph.validate()` 增 uid 唯一性（锚点 19）。
- `task.py:38-47` 的部署不变量 docstring：本轮选 B（多端点），**与不变量一致**，
  但其给出的理由（sticky `fetch_dump`）已过时，需更正措辞并注明 `fetch_dump` 已 aggregate。
- `staged.py:364-370` 无条件 clone（锚点 21，卫生项）。⚠ 只堵**输出侧**别名，
  CUDA graph 输入同样是静态缓冲 ⇒ **改完正确性依然依赖那把锁**，不给拆锁发通行证。

---

## 5. 测试策略

| # | 断言 | 位置 |
|---|---|---|
| T1 | 不传 flag 时 `load_cache_config` 被拒、`select_bundle("default")` **仍放行**；RoboCasa 入口默认不传 | `tests/libero_groot/test_dynamic_bundle_guards.py` |
| T2 | 三道守卫在**已加载的 bundle** 上各自触发，**按 raise 类型分别断言**（§4.1 表） | 同上 |
| T3 | 工厂**读** `bundle.shared_storage`，不调 `build_shared_storage`（用 spy 断言调用次数为 0） | 同上 |
| T4 | `bundle is None` 时回落 CLI 配置而非抛（锚点 20） | 同上 |
| T5 | **两个入口**都拒 `--collect-hdf5` + 多进程（锚点 11） | 扩 `tests/robocasa365/test_groot_concurrent_serving.py` + 新增 LIBERO 侧 |
| T6 | sharding helper **硬拒** warmup / `produces_calib_id` 非空 | `tests/conductor/test_sharding.py` |
| T7 | 步长切片的划分性（并集=全集、两两不交、长度极差≤1）；空分片不卡死；分片数变化下 resume 幂等 | 同上 |
| T8 | uid 唯一性守卫：两 stage 含同一 uid ⇒ `TaskGraph.validate()` raise | 扩 `tests/conductor/test_task.py` |
| T9 | **per-step 按 attempt 去重**：同一 uid 的 attempt=1 失败 + attempt=2 成功 ⇒ 只写一套行，I5 不触发 | `tests/libero_groot/test_conductor_dispatch.py` |
| T10 | **`<arm>.json` 字段完整**：产出行含 `task_id`/`orig_init_state_idx`/`episode_id`/`success`，且 `episode_id` 与 `compute_global_episode_id` 一致 | 同上 |
| T11 | `arms_with_work_left` 副本语义：已完成的臂不调 `load_cache_config` | 同上 |
| T12 | **编译输出不再别名**（能失败的版本）：monkeypatch `_verify_compiled_vision`，在**它内部**再调一次 `entry["fn"]` 覆写共享 buffer 后原样返回，断言图像 token 槽位仍是第一次的值。⚠ 不能在 `run_stage1` **返回之后**改 buffer——`:387-389` 是索引赋值即拷贝，那样改前改后都恒过（v3/v4 各栽一次） | 扩 `tests/cache/groot/test_compiled_vision.py` |
| **G-perf** | **基线 = §1.1 实测**：spatial 单臂相位 48.0 min。目标 **≤15 min**（≥3.2×），worker 占用率用可算量：`Σ episode 数 × 单集中位墙钟 ÷ (worker 数 × 相位墙钟)` ≥0.80。⚠ 该量需要 episode 时长 —— `EpisodeResult` 无 `duration_s`，故 §7 把 `task.py`/`worker.py` 的该字段列入改动 | manual（实机） |
| T13 | 端到端冒烟：两臂 × 少量 episode 跑通、完整性门通过、**盯 Xid** | manual |

**§6 Verify**：裸全量 `uv run pytest`；基线 **5 条**既有失败（锚点 22），`tests/review_tests/` 不入基线。
本轮不删 `_InferLockedPolicy` ⇒ `test_groot_concurrent_serving.py` 的 5 条锁断言必须保持通过。

---

## 6. 风险登记

| # | 风险 | 处置 |
|---|---|---|
| R1 | **bundle 热切密度**。6 端点 × 16 臂 = 96 次，但**每端点每臂一次**、臂间隔 ≈ 8.6 min（138 min/16 臂）⇒ 与 pi0.5 稳定档（8 min 一次）**同量级**；事故档是几秒 23 次。⚠ 披露：GR00T 线**今天热切次数是 0**（每 cell 重启进程） | 复制 `arms_with_work_left()`；`on_stage_begin` 对空分片早退；T13 盯 Xid |
| R2 | 动态 bundle 波及 RoboCasa 采集线 | 默认拒 + 按入口 opt-in（T1）；`--collect-hdf5` 互斥（T5） |
| R3 | 漏跑守卫 ⇒ 静默错库/错 builder | §4.1 三守卫 + raise 类型；T2 |
| R4 | `"default"` 槽被覆盖 ⇒ provenance 错配 | 非 default `bundle_id` + 工厂回落分支；T4 |
| R5 | 重试撞完整性门 I5 | §4.4-2；T9 |
| R6 | `<arm>.json` 字段缺失 ⇒ 与 X16 已产出臂 join 不上 | §4.4-1；T10 |
| R7 | driver 崩溃丢 per-step 行（无 finalizer） | §4.4-3 快照循环 |
| R8 | 跨机回连：driver pull 端口是**明文无认证 TCP** | 限制来源 IP；孤儿 worker 白名单回收 |
| R9 | 改动期间 X16 在跑 | §9 |

---

## 7. 文件清单（G2 交付边界，逐路径）

⚠ 本工作树同时有 X16 / X15 / ICLR / RoboCasa 四条线的未提交改动。**下表是 X17 的精确边界**，
不得隐式吸收其他线的文件。

### 7.1 X17 本体（20 个路径，G2 审查范围）

**新增（源码 2）**：`src/openpi/conductor/sharding.py`、`exp/libero_groot/run_conductor.py`

**改动（源码 9）**：
`src/openpi/conductor/{task,worker,agent,journal,driver,protocol}.py`、
`src/openpi/cache/groot/staged.py`、
`exp/libero_groot/serve_groot_libero.py`、`exp/robocasa365/serve_groot_n15.py`

> `driver.py` 的改动共两处：`duration_s` 一行转发，以及 G2R1 修的 flush take-and-detach。
> **调度状态机（`scheduler.py`）一行未动。**
> `agent.py` / `journal.py` 从早先的「明确不改」移入，理由见各自条目（口径更正已在本节前文记录）。

**新增（测试 3）**：`tests/conductor/test_sharding.py`、
`tests/libero_groot/test_dynamic_bundle_guards.py`、`tests/libero_groot/test_conductor_dispatch.py`

**扩充（测试 6）**：`tests/conductor/{test_task,test_agent,test_journal,test_protocol,test_driver}.py`、
`tests/cache/groot/test_compiled_vision.py`

### 7.2 X17 的 G-perf 工具（2 个路径）

`exp/libero_groot/analysis/gate_pareto/phase_utilisation.py` 与
`tests/libero_groot/test_phase_utilisation.py`。
**属于 X17**（是 §5 G-perf 的测量工具），G2 首轮提交时漏列，本轮补入。

### 7.3 与 X16 共用同一文件的一处改动（需 owner 裁提交归属）

`exp/libero_groot/analysis/gate_pareto/analyze_gate_pareto.py` 里
**只有「接受 `transport:"tcp"` 形态 merge sidecar」这一个 hunk 属于 X17**
（消费者是 `run_conductor.py`，G2 通过并部署前不可能被调用）。
同一文件的另外两处改动属于 **X16**（pi0.5 叠加图、`gate_skip_ratio`/`judge_miss_ratio` 分解），
X16 已于 2026-08-25 收官并已用到它们。
⇒ **一个文件跨两条线，无法在不做外科手术的前提下拆成两个 commit。**
建议：随 X16 一并提交（它是先落地的一方），X17 在此记录依赖关系。**提交归属由 owner 裁。**

### 7.4 文档（L3 义务，属于 X17 的具体节）

| 文件 | 属于 X17 的部分 |
|---|---|
| `docs/architecture/experiment_conductor.md` + `.en.md` | §5「归属（静态）」下新增的 sibling-shard 说明段 |
| `docs/architecture/cache_system.md` | §5.17 新增的「动态 bundle 改变守卫时机」与「编译输出必须拷出静态缓冲」两段 |
| `docs/README.md` | conductor 行末尾追加的 `sharding.shard_eval_stage` 一句 |
| `logs/README.md` | 本计划的索引行 |

### 7.5 明确不改

`scripts/serve_policy.py`、`src/openpi/serving/*`、
`src/openpi/conductor/{scheduler,strategy}.py`、`exp/gate_threshold_pareto/run_gtp.py`、
`exp/libero_groot/orchestrate_search.py`（X16 的可复现产物）。

### 7.6 不属于 X17（同工作树，勿吸收）

`exp/libero_groot/run_gate_pareto.sh`、`exp/libero_groot/analysis/gate_pareto/{analysis.md,paired_test.py,shard_imbalance_probe.py,preview/}`
（均为 X16 实验产物）；`docs/iclr/*`、`exp/rl_router/*`、`tests/exp/test_rl_router_run_loop.py`、
`exp/robocasa365/config/collect_weilandserver.env`（其他 session）。

## 8. 未决问题

1. **`--replicas` 何时进场**：它的增量是跨端点 work-stealing（实测 2.3–6.0%）。
   若 G-perf 显示 6 端点仍有可观空槽，再按 §1.4 的代价清单评估。**不是排除，是排序。**
2. `orchestrate_search.py` 在 T3 验证后是标注备选还是 deprecate？
3. G-perf 的 ≤15 min / ≥0.80 阈值由 §1.1 实测推得，owner 若有别的口径请裁。

---

## 9. 部署时机与主机清单

X16 `GATE-PARETO-DONE` 之前**不推任何文件到任何远端**。

| 主机 | checkout | 需要的文件 |
|---|---|---|
| weilandserver | `/home/weiland/openpi` | 两个 server 入口、`staged.py`、`run_conductor.py`、`conductor/{task,worker,sharding}.py`、分析器 |
| timan107 | `/scratch/zixuans8/openpi` ⚠ **X15 线共用** | `run_conductor.py`、`conductor/*`。**只准新增本线独有文件，禁改共用文件** |
| timan107 | `/scratch/zixuans8/openpi_rc365` | RoboCasa 角色若启用才推 |

**提交协议**：本工作树四条线共用（X16/X15/ICLR/RoboCasa 均有未提交改动，已发生过一次误提交 33 个文件的事故）
⇒ **一律 `git commit -- <显式路径清单>`，禁裸 `git commit`**。

---

## Review Log

### 自审 R1 — Reviewer — NEEDS REVISION — 2026-08-24 23:2x CDT

审查由本会话 spawn 的 agent 执行（owner override G1，§0）。全文见会话记录；下列为条目摘要与逐条回应。
4 BLOCKING / 7 MAJOR / 5 MINOR / 1 范围裁决。

### 自审 R1 — Executor — 2026-08-24 23:45 CDT

**BLOCKING**

- **B1（A/C 两行颠倒）— Accepted，v1 方向性错误。** 亲验 `serving_throughput_problem.md:536-543`：
  CUDA launch 路径持 GIL，多线程无用、多进程才有用。拆锁只是把串行点从锁挪到 GIL，
  「乘性提升」零支撑。反向也成立：单臂相位里 `--replicas` 确实能把 1/6 容量变满。
  **但 v2 两条都不采纳**——因为 conductor 单独就能消掉同样的 bubble 且无需碰端口/显存（§1.3）。
  §1 已按更正后的诊断重写，拆锁降级为「先测量」（T3'）。
- **B2（`_COMPILED_VISION_REGISTRY` 竞态）— Accepted，v1 anchor 6 是错的。** `staged.py:423` 确实在
  `run_stage1` 运行期写 `checked`，`:366` 在 `checked` 为真后不再 clone CUDA-graph 静态缓冲。
  这是确凿的静默损坏路径，**今天正是那把锁在护着**。已作为锚点 7 记入，并成为「不拆锁」的第二个理由。
- **B3（T3 不可用）— Accepted。** flow-matching 每次新采 `torch.randn`，串行重复调用本身就不一致，
  「逐位相同」永远不可能通过；而 FULL_HIT 步根本不进 stage2，高命中率下更会掩盖 stage1 的竞态。
  已替换为 T3'：只断言 **`run_stage1(...).input_embeds` 逐位一致**（noise-free，且竞态正好在那里），
  外加 winner_id 压力测试。
- **B4（端口冲突 + 显存）— Accepted。** `port+1..port+R` 撞 23160-23165；6 槽 × R × 6 GB 在 48 GB 卡上
  R=2 就爆。**已把 `--replicas` 整个移出目标**，并把正确公式 `槽数 × R × (显存 + 库)` 记入 R7。

**MAJOR**

- **M1（共享状态清单两头都错）— Accepted。** `SystemTimer` 已经是每连接一个
  （`config.py:2630-2637`），v1 唯一的缓解措施护错了对象。v2 不拆锁，该节整体删除；
  真正的共享对象（policy/transform、pooled backend、编译注册表、全局 RNG、bundle 表）
  记入锚点表与 T3' 的前置。
- **M2（transform 未分析）— Accepted。** `apply_transforms`/`unapply_transforms` 今天在锁内，
  且是 out-of-repo 代码（`/home/weiland/gr00t_n15`）。可逆管线若把 apply 的状态存在 `self` 上，
  并发就会交叉污染出「有限且看似合理」的错误动作。**已成为不拆锁的第三个理由**；
  真要拆必须先做逐 transform 审计并 pin 源码 hash（照 `UPSTREAM_FORWARD_SHA256` 的先例）。
- **M3（supervisor 无测试）— Accepted但本轮不适用。** v2 不抽取 supervisor、不碰 `serve_policy.py`，
  该风险随目标一起消失。若日后做，「先写测试再抽取」的次序意见成立，记在此备查。
- **M4（会打破 5 条既有锁断言）— Accepted。** v2 不删锁，
  `tests/robocasa365/test_groot_concurrent_serving.py` **保持不变**；已在 §5 明写这一点。
- **M5（动态 bundle 是工厂重写不是 flag）— Accepted，这是 v2 最实质的一条。** `_bind_bundle` 只传
  `bundle_id` 不传已加载 config，工厂必须自取并**对新 config 重跑三道守卫**。
  §4.2 已改为逐守卫定位表，T3 改为在**已加载的 bundle** 上断言守卫触发。
- **M6（GR00T main 在分支前就加载 CUDA + 7 GB 模型）— Accepted但本轮不适用。** 该重构只有
  `--replicas` 才需要，随目标移除。记在此，日后做 replicas 时是必须项。
- **M7（`run_replicated` 签名带不动自己的 context）— Accepted但本轮不适用。** 同上。
  该条附带确认「picklability 无问题、不会破坏 pi0.5」，一并记录。

**MINOR**

- **m1（共享模块须 import-light + 无 `__init__.py`）— Accepted，记录。** 本轮不新增 `src/openpi/serving/*`，
  但这条对 GR00T 岛 venv 是硬约束，写进备查。
- **m2（启动预算与 teardown）— Accepted，部分本轮适用。** 150 s 端口轮询的问题随 replicas 一起消失；
  但**孤儿进程持 CUDA context** 的教训对 conductor 同样适用（`WorkerAgent` 在 timan107），
  已并入 R4 的验证项。
- **m3（warmup dump / metrics 面未测）— Accepted。** 无 replicas 则无 fan-out+merge，风险降低；
  但 conductor 路径下 `fetch_dump` / `preload_normalizer_buffer` 仍需验证，记入 §8 待办。
- **m4（in_memory_backend 无锁可变缓存）— Accepted，记录。** 今天全部跑在锁内。
  agent 判断「dict/set 操作 GIL 原子 + weakref 身份守卫足够」，我同意其分析但**不据此放宽**：
  v2 不拆锁，该结论作为 T3' 的前置研究项留存。历史审计
  `logs/archive/server_concurrency_resource_audit.log.md:820` 一并引用。
- **m5（flag 名是 `--collect-hdf5`）— Accepted。** §4.3 已更正。

**范围裁决 — Accepted，且是本次重写的主因。**
agent 指出 v1「量化了一个收益，然后把能兑现它的改动排除在外」。这个判断成立。
v2 的选择是它给的第一条路径的变体：**不是 replicas、也不是在 `orchestrate_search.py` 里切分片，
而是接 conductor**——因为 conductor 的 episode 级分发同时消掉 (a)(b) 两处，
且 owner 已明令把 conductor 纳入范围。§1 的 bubble 数字因此保留，它现在**确实**是本计划买到的东西。

---

### 自审 R2/R3/R4（对 v3） — Reviewer ×3 — **NEEDS REVISION** — 2026-08-25 00:1x CDT

owner override G1（§0），由本会话 spawn 的**三个**审查 agent 并行执行，lens 固定为
锚点核验 / 设计正确性 / 范围与可交付。三份**独立收敛到同一个结论**。全文见会话记录。

**R2（锚点核验）**：18 条锚点**无一 WRONG 或 UNVERIFIABLE**，机制描述全部属实——
「v3 没有重犯前两版给机制安上需要它有的性质的病」。但判 §1.4 的**推论** OVERSTATED：
静态分片 + worker/server 死绑 ⇒ 无跨 server 窃取，只是把粗粒度空转换成细粒度长尾，
而 `orchestrate_search.py:21-23`（**v3 自己引来推翻 v1 的那条锚点**）原文就写着
「a static shard leaves the fast slots idle while one slow slot drains」。
另查出：`driver.py:198-199` 先跑完 `on_stage_begin` 才 `mark_setup_done`（空/预完成分片白付热切）；
`_ep_index` 无 uid 去重 ⇒ 分片重叠即静默死锁；`_flush_per_step_for_stage` 按 yaml_id 聚合；
锚点 12 的语义在 flag 打开后漂移；行号勘误 3 处。

**R3（设计正确性）**：5 BLOCKING / 10 MAJOR / 7 MINOR。
**B1：T-A 的收益是零**——相位墙钟 = `max_lane_episodes × 单集时长`，分片不改变任何一条 lane 的集数。
**B3：sibling-shard 会把 warmup 标定静默切成 1/N**（N 次 `fetch_dump` 各拿 1/N，N 次 `publish` 互相覆盖）。
**B4：T11 不可能失败**（`input_embeds` 本来就不与编译输出共享存储，且 stub 跑在 CPU）。
**B5b：R1 的量化用错分母**（事故判据是单 GPU 时间密度，v3 按进程除）。

**R4（范围与可交付）**：**B1：把 `--replicas` 排除在外是回避 owner 的指令**——
仓库已建成、实测 ≈2.4×/server、conductor 教程标为「推荐」、`fetch_dump` 已 aggregate；
v3 的两条排除理由（端口、显存）都是配置层产物，且显存公式被摆成必然爆的形状。
**B3：全计划没有任何性能验收判据** ⇒ 全绿 + 零加速也算「做完了」。
**B4：部署只覆盖了 weilandserver**，漏 timan107 与共用脏树的提交协议。

### 自审 R2/R3/R4 — Executor — 2026-08-25 00:5x CDT

**全部 BLOCKING 均 Accepted，且已亲验其依据（不采信 agent 转述）。** v4 是按更正后的诊断重写的。

- **R4-B1（`--replicas`）— Accepted，这是 v4 的主因。** 亲验：`conductor_tutorial.md:47-51`
  原文把「`--replicas N` + 单公共端口」标为**推荐**并写明 router 对 conductor 透明、
  `fetch_dump` 已 aggregate；`merge_dump_replies` 真实存在（`replica_proxy.py:114` + 测试）；
  `serve_policy.py:740` child 绑 **loopback**「not exposed」⇒ 端口理由不成立；
  等进程预算下显存相同 ⇒ 显存理由不成立。**v4 把 `--replicas` 立为 T1。**
- **R3-B1（T-A 收益为零）— Accepted。** v3 §4.1 写的确实是「同一份 lane 集合摊到 N 个槽」，
  lane 总数不变 ⇒ 墙钟不动。**T-A 整条删除**（conductor 路径下根本没有预切 lane 文件，问题自动消失）。
- **R3-B3 / R2(B)（warmup 分片切碎标定 / 空分片白付热切）— Accepted。**
  这两条随 sibling-shard 一起删除。`merge_dump_replies` 恰恰是 replicas 路径**已经**做对的那件事。
- **R2（`_ep_index` 无去重、`_flush_per_step_for_stage` 按 yaml 聚合）— Accepted。**
  v4 一 yaml 一 stage ⇒ 后者恒正确（记为锚点 14，作为不做分片的又一理由）；
  前者作为**纯加固断言**保留进 §4.5/T9。
- **R3-B4（T11 自证）— Accepted，亲验属实。** `input_embeds` 来自 embedding + scatter，
  与编译输出不共享存储，`data_ptr` 改前改后都不等；且既有 stub 在 CPU 上无持久静态缓冲。
  T10 改为「复用 buffer 的 stub + 就地改写后断言 token 槽位不变」，并入既有
  `tests/cache/groot/test_compiled_vision.py`（不新建文件）。
- **R2/R3（§4.4 主张过头）— Accepted。** 亲验 `mode="reduce-overhead"` 的 CUDA graph
  **输入侧同样是静态缓冲** ⇒ 无条件 clone 只堵住三条竞态里的一条，锁仍是必需。
  §4.5 已改口径，并明写**不给未来拆锁发通行证**。
- **R3-B5b / R2(C)（热切速率分母错）— Accepted。** 亲验 `_broadcast` 用 `asyncio.gather` ⇒
  臂边界上是 **N 个副本并发**重建，同一块 4090。R1 已改成 per-GPU 口径，
  并**主动披露** GR00T 线今天的热切次数是 0（每 cell 重启进程），本计划把它提到 16×N。
- **R4-B2（RoboCasa 交付为零）— Accepted。** 亲验 `staged.py:206` `compile_vision=False` 默认、
  `--compile-stage1` opt-in、且 `robocasa365_ws_search_plan.log.md:158` 记等价门
  **FAIL（cos 0.8716）已 park 不启用** ⇒ v3 的 T-C 修的是死代码。
  **v4 的 T5 给 RoboCasa 一条真的容量路径**（`serve_groot_n15.py` 同吃 `--replicas`，采集硬钉 1），
  clone 修复降级为卫生项、不计入「RoboCasa 适配」。
- **R4-B3（无性能验收）— Accepted，这一条最该早想到。** v4 §5 增 **G-perf**：
  单臂相位墙钟 ≥3×、worker 占用率 ≥90%，脚本与数据入库。
- **R4-B4（部署清单 / 提交协议）— Accepted。** §9 已补三主机清单
  （含 timan107 与 X15 共用、只准新增本线独有文件）与「一律 `git commit -- <显式路径>`」。
- **R4-m2 / R2（Verify 基线）— Accepted。** 亲验 `.gitignore:124` ⇒ `tests/review_tests/` 不在 HEAD，
  基线由 11 条更正为 **5 条**。
- **R4-M7（文档落点错位）— Accepted。** §7 已把 `experiment_conductor.md`(+EN)、
  `conductor_tutorial.md`(+EN)、`docs/README.md`、`logs/README.md` 全部列入；
  `cache_system.md §5.17` 改为记动态 bundle 对「加载期身份绑定」的影响，而非拓扑。
- **未采纳的只有一处口径**：R2 判 (a)(b)「消失」为 OVERSTATED 时用的是**分片**语义。
  v4 不做分片，(a) 由「一个臂 → 一个 endpoint → N 进程全在干它」直接消掉，
  不存在跨 server 窃取问题。另：本会话用真实 per-step 数据实测过 6 路 stride 分片的尾部气泡
  为 2.3–6.0%（`exp/libero_groot/analysis/gate_pareto/shard_imbalance_probe.py`），
  即静态分片的坏处随粒度变化、83 集/片时远小于 R2 的定性判断——
  **但这不改变 R3-B1 的结论**（收益为零的原因不是不均衡，是 lane 数没变），故 T-A 仍然删除。
  该探针保留入库，因为它是「stride 优于连续切分」这一设计选择的唯一实测依据。

---

### 自审 R5/R6（对 v4） — Reviewer ×2 — **NEEDS REVISION** — 2026-08-25 01:1x CDT

owner override G1（§0），本会话 spawn 的**两个** agent，lens 固定为对抗性 / 可实施性。

**R5（对抗性）**：**v4 的核心主张不成立。**
**B1：§1.1 整张代价表建立在一个误植的、故障态的数字上** —— `62–81 min` 出自
`libero_groot_ws_search.log.md:141-143` 的**僵尸 worker 事故**（`infer_ms_per_call` 88→2325 ms），
健康值 27 min；且在 `gate_pareto_plan.log.md:325` 它是 **`eval l10`** 相位的 min/cell，不是 gate-only；
而 gate-only 自己那行写的是「2×500 集（并行占 **2 槽**）| ~1 h | 高命中 ⇒ 接近纯 cache 速度」；
写 plan 时 l10 gate-only **根本还没跑过**。
**B2：收益的分母从未出现** —— 受影响的相位只占全程一小部分。
**B3：「conductor 核心零改动」是假的** —— `task.py:38-47` 是核心自己的部署不变量，
明文写着 conductor 路径 **never** 走 `replica_proxy`、多副本要建模成多个独立 `ServerEndpoint`；
v4 引了两份同意它的文档，唯独没看它承诺不动的那个文件。
**B4：`serve_groot_libero.py:221` 有 `--collect-hdf5`**（v4 说"根本没有"），
于是 `--collect-hdf5 --replicas N` 在**真正产库的那个入口**上无守卫。
**B5：G-perf(ii) 用现有工具测不出来**（journal/EpisodeResult/health/monitor 全无时长与落盘）。
另 M5：6 端点 + eval-only sibling stage 能拿到同等收益且不需要改 server；
M6：≈2.4×/server 是 a100+jupyter 的 pi05 数据，副本数未知、无结果表，不可迁移；
M9：9 ms/call 是无源散文，且它正是被扫的自变量（低 `f_FH` 臂是教师 236 ms/次）。

**R6（可实施性）**：**9 BLOCKING / 8 MAJOR / 9 MINOR，不能进 §4 Code。**
`dataclasses.replace` 对 `argparse.Namespace` 抛 `TypeError`；**`ReplicaRouter` 这个符号不存在**
（真实是 `ReplicaProxy(backend_host, backend_ports)` / `run_proxy`，`serve` 是协程）；
GR00T 的 `--concurrent` 默认 OFF 且无 `--non-concurrent`，**极性与 pi0.5 相反** ⇒
`--replicas 6` 不带 `--concurrent` 会让 48 个 worker 只有 6 个能连上、**启动期零报错**；
`build_shared_storage` 由 server handler 调用并挂进 bundle，工厂重建 = 几百次 GB 级加载，
且 T6 在 §7 的范围内**不可实现**；**T10 仍然恒过**（`staged.py:387-389` 索引赋值即拷贝，
返回后改 buffer 加不加 clone 都一样）；**`EpisodeResult` 没有** `task_id`/`orig_init_state_idx`/`episode_id`；
**重试会为同一 episode 产出第二整套 per-step 行 ⇒ 撞死 I5**；
**`WorkerSpec` 无 `resize_size`** 而 GR00T 必须 `--resize-size 256` ⇒ 车队起来后每集都在 wire contract 上失败。

### 自审 R5/R6 — Executor — 2026-08-25 01:4x CDT

**全部 BLOCKING 均 Accepted，全部依据已亲验**（不采信 agent 转述）。v5 是按更正后的诊断重写的。

- **R5-B1 — Accepted，这是 v5 的主因，也是我第四次犯同一类错。** 亲验两处源头属实。
  前三版是「给一个机制安上它没有的性质」，v4 是「**给一个数字安上它没有的身份**」。
  **v5 §1.1 整表改用本次运行自己的链日志实测**：spatial warmup 16.0 min、gate-only **32.0 min**、
  sweep 138.1 min、全程 190.5 min ⇒ **单臂相位占 25.2%**；l10 warmup 已实测 36.0 min。
  收益口径随之改为**约 18%、两 suite 合计 1.2–1.5 h**，并明写「不多，但 owner 要的是这条框架」。
- **R5-B3 — Accepted。** `task.py:38-47` 我在本会话早期**读过**却没有联想起来。
  更关键的是它的**结论**（多个独立 `ServerEndpoint`）**恰好就是 v5 选的 B 方案** ——
  于是 v5 与该不变量一致，只需更正其已过时的**理由**（sticky `fetch_dump` 早已 aggregate）。
- **R5-M5 — Accepted，构成 v5 的拓扑决定。** 6 端点 + eval-only sibling stage 拿到 (a) 的全部收益，
  `--replicas` 的增量只有跨端点 work-stealing（本线实测 2.3–6.0%），代价是 supervisor
  （任一 child 死 ⇒ 全体 `os._exit(1)`，爆炸半径 1/6→6/6）、30 s fan-out 硬编码、公共端口丢 `/healthz`。
  **v5 §8 未决 1 明写这是「排序不是排除」** —— v3 正是栽在把它当排除。
- **另查明一条使 v3 的最重指控在本线失效的事实**：X16 的 warmup 是**离线管线**，
  `grep -rn "fetch_dump\|preload_normalizer_buffer" exp/libero_groot/` **零命中** ⇒
  「sibling stage 会切碎标定」对本线不成立（v4 §1.3 用来证明自己优越的第三条理由同样是空的）。
  helper 仍硬拒 warmup 分片 —— 本线不需要，但失败是静默的。
- **R5-B4 / R6-B5 — Accepted。** 亲验 `serve_groot_libero.py:220-227,278-297` 确有采集分支。
  v5 §4.1 与 T5 覆盖**两个**入口。
- **R5-B5 — Accepted。** G-perf(ii) 改为可算量 `Σ ep × 单集中位墙钟 ÷ (worker 数 × 相位墙钟)`，
  并把 `EpisodeResult.duration_s` 与 `worker.py` 的回填**正式列入 §7 改动**（v4 漏了埋点）。
- **R6-B1/B2/B3 — Accepted，亲验符号与极性。** v5 §2 锚点 8/9/10 记录；
  ⚠ B3 那条尤其重要：极性反了会**静默**只连上 6 个 worker。v5 选 B 方案后该风险自然消失。
- **R6-B4 — Accepted。** `build_shared_storage` 从「第四道守卫」降为「工厂应当**读** `bundle.shared_storage`」；
  库去重已由 `BackendPool.get_or_load` 按 fingerprint 实现（锚点 13），v4 的 R5 严重性被夸大，已降级。
- **R6-B6（T10 恒过）— Accepted，这是同一条测试第二次被判自证。** v5 T12 改为
  「monkeypatch `_verify_compiled_vision`，在**它内部**再调一次 `entry["fn"]` 覆写共享 buffer 后原样返回」，
  并在测试描述里写明**为什么不能在返回后改**（`:387-389` 索引赋值即拷贝）。
- **R6-B7/B8/B9 — Accepted，三条都是「跑起来才炸且报得很晚」。** v5 §4.4 新开一节专门处理：
  `<arm>.json` 字段回填（含 `compute_global_episode_id` 与 `APOOL_TRIALS`）、
  per-step 按 `attempt` 去重、快照 finalizer、以及 sidecar 的**独立**判据
  （`episodes_expected` 取 strategy 计划 uid 全集、`episodes_reported` 取 journal 终态 ——
  重试耗尽的 uid 从不入 journal，两者**会**真的不一致）。
  §4.3 记录 worker 必须自带 `spawn_fn` 传 `--resize-size 256 --replan-steps 5` 与 `MUJOCO_EGL_DEVICE_ID`。
- **R5-M6/M9 — Accepted。** ≈2.4×/server 与 9 ms/call 两个数字**全部移出 v5 的收益论证**；
  v5 的收益只由 §1.1 的自测墙钟推出。

---

### 自审 R7（对**代码**，非计划） — Reviewer — **NEEDS REVISION** — 2026-08-25 02:0x CDT

第一次审查已写好的代码而非计划。**三条 BLOCKING 会让整轮跑白跑：**

- **B1 · 产物布局与完整性门不兼容。** `aggregate` 要的是**两个目录三个路径**
  （`per_step_dir/<arm>.jsonl`、`per_step_dir/<arm>.merge.json`、`results_dir/<arm>.json`，
  `analyze_gate_pareto.py:271-278`）；实现写的是一个合并的 `per_step.jsonl`，且把
  `<arm>.merge.json` 与 `<arm>.json` 放同一目录 —— 而 `:254-257` 的 `present` 只过滤
  `.partial.json`，`<arm>.merge` 会被当成多余的臂直接 raise。跑完一条臂都读不进去。
- **B2 · 没有增量落盘。** 逐步行只在收尾写一次，而 `Journal` 是逐 episode 落盘的 ⇒
  崩溃后 resume 只补缺的 episode，**已完成 episode 的逐步行永久消失**，I3 要求两侧 episode 集合
  严格相等 ⇒ 该臂再也过不了门。**这正是 plan R7/§4.4-3 自己要求、实现里缺席的条目。**
- **B3 · resume 拿不到产物。** 已完成臂被 `arms_with_work_left` 过滤 ⇒ 不进 `episode_index`
  ⇒ `finally` 不为它们写产物；而上一次是崩溃退出的，它的 `finally` 也没跑过。
  更糟：`active_paths` 为空时在**写任何产物之前**就 `return`，自救路径也被堵死。

**MAJOR**：M1 `_flush_per_step_for_stage` 的锁窗口丢行（分片把它从不可达变成必然）；
M2 `read_journal_outcomes` 取最后一条且忽略 `accepted` ⇒ 被拒的陈旧致命错会把成功覆盖成失败；
M3 `on_stage_begin` 的早退不覆盖 resume（`stage.episodes` 是静态计划列表）；
M4 `serve_groot_n15.py` 的新 flag 是哑弹（teacher-only 早退没带 `allow_dynamic`）；
M5 driver 在 bind 前死掉则主线程无限空转；M6 driver/agent 的 `--servers` 不一致会静默挂死；
M7 未 flush 的行被丢弃。另 m13：`duration_s` 过线但 `PROTOCOL_VERSION` 未升 ⇒ 新 worker + 旧 driver
每条结果 `TypeError`（而 timan107 的 checkout 与 X15 线共用、不保证同步部署）。

**判定为干净的**（逐条核过）：dynamic 关闭时两个入口**逐字未变**；`config is None` 分支后续路径正确；
banner 探测不受影响；`staged.py` 的 clone 与删掉内部冗余 clone 正确；`journal.py` 旧记录逐字不变；
`__s{k}` 对 `_uid_meta`/journal 无影响；`assign_servers` 被忽略无副作用；tcp 分支对旧 sidecar fail-closed。

### 自审 R7 — Executor — 2026-08-25 02:2x CDT

**B1/B2/B3 + M2/M3/M4/M5/M7 + m13 已修，全部配了能失败的测试。** 亲验了 B1、M2 两条的依据。

- **B1** — `--out-dir` 拆成 `--results-dir` / `--per-step-dir`；sidecar 跟 per-step 走
  （新测试 `test_the_sidecar_lands_beside_the_per_step_file_not_the_results` 断言 results 目录里
  只有 `<arm>.json`）。
- **B2** — `_per_step_writer` 改为**立即按臂 append**；去重推迟到收尾（热路径保持纯 append，
  这正是崩溃可存活的原因）。
- **B3** — `episode_index` 在 `__init__` 里对**全集**臂构建；`finalize()` 提成函数，
  「无活可干」路径也调用它再退出。
- **M2** — 跳过 `accepted is False`，同 uid 取 `attempt` 最大者；两条新测试分别覆盖
  「陈旧致命错不得覆盖成功」与「无 `accepted` 字段的旧记录回落到 attempt」。
- **M3** — `plan()` 接受 `done_uids`，把已完成 episode **从计划里剔除**而不只是让 scheduler 跳过，
  于是空 sibling 在 `on_stage_begin` 处真的为空。
- **M4** — RoboCasa 入口改为 `--allow-dynamic-bundles` **要求** `--cache-config`
  （它没有 LIBERO 那种「空启动、全靠下发」模式）。
- **M5** — 等待循环加 `driver_thread.is_alive()`，退出后仍无端口则 `SystemExit` 并点名 EADDRINUSE。
- **M7** — 收尾并入 `driver.per_step_rows`。
- **m13** — `result_from_wire` 改为**丢弃未知键**（比升版本更稳：两台机的 checkout 不保证同步重启，
  且失败形态是「车队不产出」而非「版本不匹配」）。
- **M1 未修**：它是 `driver.py` 既有的锁窗口竞态。B2 的立即落盘把**丢行**的后果消掉了大半，
  但窗口本身仍在。**记为已知问题**，不在本轮扩大 `driver.py` 的改动面。
- **M6 未修**：agent 角色已打印解析出的 server keys（诊断够用）；driver 侧的 warn-once 需要包
  `handle_pull`，留待实机冒烟时按需加。

### 自审 R8（测试判别力 · 变异测试） — Executor 自查 — 2026-08-25 02:2x CDT

spawn 的审计 agent 44 分钟无进展（transcript 停在 137 字节）被判卡死并停掉，改为**自己做变异测试**：
把被测代码改成一个看似合理的错误实现，看测试会不会红，然后还原。

| 变异 | 目标测试 | 结果 |
|---|---|---|
| `_resolve_bundle` 改回重建 `build_shared_storage` | `test_storage_is_read_from_the_bundle_never_rebuilt` | ✅ 红 |
| 守卫喂 `cli_config` 而非已加载的 config | `test_guards_rerun_on_the_loaded_bundle` | ❌ **绿 —— VACUOUS** |
| 所有分片放同一台 server | `test_every_server_can_pull_from_one_sharded_arm` | ✅ 红 |
| sidecar 的 `episodes_expected` 改从 results 推 | `test_the_sidecar_disagrees_when_an_episode_never_reported` | ✅ 红 |
| `keep_accepted_attempts` 取最小 attempt | `test_a_retried_episode_contributes_one_set_of_rows` | ✅ 红 |
| 去掉 `staged.py` 的无条件 clone | `test_compiled_output_is_copied_out_of_the_shared_buffer` | ✅ 红（早先已验） |

**查出一条 VACUOUS 并已修**：`_fake_config()` 返回 `types.SimpleNamespace`，而
**`SimpleNamespace.__eq__` 比较的是 `__dict__`** ⇒ 两个内容相同的不同对象 `==` 相等，
于是「守卫拿到的是已加载的 config」这条断言在守卫被喂了 CLI 配置时**照样通过**。
改为 `_FakeConfig` 类（按身份比较）+ `is` 断言后，同一变异下变红。

⇒ 这是本会话第三次栽在「断言了错的东西」上（前两次是编译视觉塔那条，被审查方各抓一次）。
共同形态：**断言写成了结构相等，而要证明的性质是身份/来源**。

**R8 追加**：另补一条**端到端**测试 `test_the_artifacts_pass_the_real_integrity_gate` ——
让 runner 产出的产物去过**真实的** `check_arm_integrity` + `aggregate_arm`，而不是断言"路径看起来对"。
B1 的病灶正是「每个路径单看都合理，但 `aggregate` 的 results 目录 glob 会把 sidecar 当成多余的臂」，
只有真跑那道门才看得见。变异复验：把 sidecar 放回 results 目录 ⇒ 该测试报
"a stray artifact in the results dir fails the phase" 变红。

**R7-m11 — Accepted。** `link_shard_groups` 只有测试在用：本轮的 runner 是 eval-only、
臂之间无依赖边，所以它没有生产调用点。按 WA §3.1「无死代码」连同其测试一并删除
（真要做多阶段依赖时，笛卡尔积连接是三行，不值得先摆在那里当反面教材）。

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-08-25 10:17 CDT

- [Blocking] [Concern] `ConductorDriver._flush_per_step_for_stage()` can delete rows that arrive while an earlier sibling is writing the same arm: it snapshots under `_rows_lock`, releases that lock for the writer callback, then removes **all** rows with the yaml id. Sibling stages for one yaml finish independently, so this window is reachable by the production topology and destroys evidence required by I3/I5. — reasoning: independent probe `test_sibling_flush_does_not_delete_rows_appended_during_the_write` deterministically appends one later-sibling row inside the callback; the row is absent after `_flush_per_step_for_stage()` returns. Drain/remove the exact snapshot atomically (or otherwise preserve post-snapshot rows) and add a driver-level regression test.
- [Blocking] [Concern] Resume treats scheduler-rejected stale journal records as completed work. `arms_with_work_left()` counts every distinct uid without checking `accepted`, and `Journal.replay_done_uids()` likewise admits `accepted: false`; a crash after a fenced stale result but before the live retry completes can therefore filter the arm or pre-complete the episode, then synthesize incomplete artifacts. — reasoning: independent probe `test_rejected_stale_records_do_not_make_an_arm_complete` supplies only `accepted:false` terminal records and the arm is incorrectly removed. Use one accepted-attempt selection rule consistently for arm filtering, journal replay, outcomes, and per-step finalization; preserve backward compatibility for legacy records without the field.
- [Blocking] [Concern] The new uid uniqueness assertion is not actually global: `_assert_unique_episode_uids()` raises only when the second copy belongs to a **different** stage, so the same uid repeated twice inside one stage passes validation and is dispatched twice through the scheduler's flat uid index. — reasoning: independent probe `test_uid_uniqueness_rejects_duplicates_inside_one_stage` does not raise. Track every occurrence, not merely the first owning stage, and test both same-stage and cross-stage duplicates.
- [Blocking] [Concern] The `duration_s` ledger does not yet support the G-perf claim made for it. `WorkerLoop` measures every attempt, including failures, but `ConductorDriver.handle_result()` records duration only when `result.success or not retriable`; retriable attempts (including the retry-exhausting attempt that the scheduler marks terminal) contribute worker busy time but no ledger row, while rejected stale terminal rows are included by the current reader. — reasoning: the resulting utilisation numerator can omit real work and include rejected work, so the planned `>=0.80` acceptance value is not auditable from the ledger alone. Define and test attempt-level timing semantics before using this metric at G-perf.
- [Blocking] [Concern] The polished plan and the submitted G2 target disagree on deliverables. Plan §7 includes `analyze_gate_pareto.py`, new `phase_utilisation.py`, its test, and conductor/cache documentation updates, while the G2 submission's 9-modified + 2-new source scope excludes them and states G-perf/T13 are unimplemented. — reasoning: G2 plan-consistency cannot pass until the executor either brings those deliverables into the X17 snapshot or amends the plan to classify only the live G-perf/T13 runs as post-G2 Verify work. Do not absorb the concurrent X16/other-session files implicitly; state the exact X17 paths.
- [Blocking] [Concern] The target is not lint/format clean. — reasoning: scoped `ruff check` reports three F401 errors (`tests/conductor/test_agent.py` lines 110/131 and `tests/libero_groot/test_conductor_dispatch.py` line 13), and scoped `ruff format --check` reports 18 target files would be reformatted. The submitted pytest subsets pass, but this state cannot satisfy the repository's pre-commit requirement.
- [Non-blocking] [Concern] `ServerEndpoint` still calls independent process endpoints a deployment invariant, while the registered conductor architecture/tutorial explicitly support either a replica-proxy public endpoint or independent endpoints. — reasoning: X17 may choose independent endpoints without redefining the core type's supported topology; align the docstring with the subsystem charter.
- [Non-blocking] [Suggestion] Remove the duplicated consecutive assignment of `allow_dynamic` in `serve_groot_libero._build_concurrent_factory`. — reasoning: it is harmless at runtime but is an avoidable review/debug distraction in a provenance-sensitive branch.

### G2 Round 1 — Executor — 2026-08-25 11:0x CDT

- **Accepted — `_flush_per_step_for_stage()` 的快照—写入—删除竞态已修。** 改为**在单个
  `_rows_lock` 段内 take-and-detach**：把待冲行取出、把余下的行装回 `self._per_step_rows`，
  **然后**才在锁外调用 writer；writer 期间 `handle_result` 追加的行落进新列表，得以存活；
  writer 抛异常则把取出的行插回队首。⚠ **这条我在自己的代码自审里查出过（记为 M1）却主动不修**，
  理由是「立即落盘已消掉丢行后果的大半」——**那个理由是错的**：被删掉的正是窗口内到达、
  **从未传给 writer** 的行，落盘救不了它们。新增 driver 级回归测试两条
  （`test_flush_does_not_delete_rows_that_arrive_while_the_writer_runs` 复现审查方的探针场景、
  `test_flush_restores_rows_when_the_writer_raises`）。
- **Accepted — accepted-attempt 选择规则已统一到一处。** 新增
  `Journal.record_counts_as_done(rec)` 作为**唯一定义**（`status ∈ {done,failed}` 且
  `accepted is not False`；用 `is False` 而非真值判断，使**缺该字段的旧记录语义不变**）。
  四个消费者全部路由过去：`Journal.replay_done_uids`、`arms_with_work_left`、
  `read_journal_outcomes`、`phase_utilisation.load_terminal_records`。
  ⚠ **本条改动了 X15 线已提交的契约**：`tests/conductor/test_rl_router_accepted.py::
  test_replay_done_uids_ignores_the_new_fields` 断言三个新字段对 replay 全部惰性。
  该断言在写下时成立（当时无人消费），**现在不成立**——`accepted` 恰恰应当影响 replay。
  已改写为 `..._ignores_attempt_and_error_but_not_accepted` 并补一条
  「被接受的非可重试失败仍算终态」的断言，保住 `replay_done_uids` 存在的反活锁性质。
  **请 owner 知悉这动了另一条线的断言。**
- **Accepted — uid 唯一性守卫已覆盖两种形态。** 原实现用 `setdefault` 记「首个拥有者 stage」，
  同 stage 内重复时 `first == stage_id` 故不抛。改为记录每一次出现，同 stage 与跨 stage
  分别给出不同错误文案；新增 `test_validate_rejects_a_uid_repeated_inside_one_stage`。
- **Accepted — `duration_s` 的 attempt 级语义已定义并强制。** 明确写进
  `phase_utilisation.py` 的模块 docstring：`duration_s` 是**每个被接受的终态 attempt**，
  两类真实 worker 时间不在其中——**可重试失败**（非终态、不入账本，本模块看不见）与
  **被裁定拒绝的陈旧 attempt**（入账本但结果未被采用，本模块**排除**）。
  ⇒ **该指标是占用率的下界**，据此把验收口径从「≥0.80 占用率」改读为「**至少这么忙**」；
  `load_terminal_records` 改为返回 `(rows, fenced_count)`，输出里带 `bound: "lower"` 与
  `fenced_records_excluded`，并在有 fenced 记录时显式打印告警。新增测试
  `test_fenced_attempts_are_excluded_and_counted`。
- **Accepted — 计划与 X17 交付边界已逐路径写死（§7 重写）。** §7.1 列 X17 本体 20 个路径；
  §7.2 补入 G2 首轮漏列的 `phase_utilisation.py` + 其测试（属 X17）；
  §7.3 单独处理 `analyze_gate_pareto.py`——**该文件只有「接受 `transport:"tcp"` sidecar」
  一个 hunk 属于 X17**，另两处（pi0.5 叠加图、教师来源分解）属 X16 且 X16 已收官用上，
  一个文件跨两条线无法不做外科手术地拆成两个 commit，**提交归属请 owner 裁**；
  §7.4 逐节列出属于 X17 的文档改动；§7.6 显式列出**不得吸收**的其他线文件。
- **部分 Accepted / 部分 Rejected — lint 与 format 分开处理。**
  - **F401：Accepted，已修。** 三处未使用导入（`test_agent.py` ×2、`test_conductor_dispatch.py` ×1）
    已 `ruff check --fix` 清除，相关测试仍全过。
  - **`ruff format`：Rejected。** 依据「不能满足仓库 pre-commit 要求」不成立——
    `.pre-commit-config.yaml` **只有 `uv-lock` 一个 hook，没有 ruff-format**；
    `.github/workflows/test.yml` 也不跑 lint/format。更关键的是，**本次从未碰过**的
    `src/openpi/cache/orchestrator.py`、`src/openpi/conductor/scheduler.py`、
    `src/openpi/serving/websocket_policy_server.py` 同样报 "would reformat"
    ⇒ **仓库整体本就不是 format-clean 的**。只格式化本次这 18 个文件，会让它们与周边代码
    风格不一致，并在 diff 里混入大量与本改动无关的重排，反而增加审查负担。
    若 owner 要求全仓格式化，那是一个独立的 L1 清理项，不应挂在本次 G2 上。
- **Accepted — `ServerEndpoint` docstring 已与架构章程对齐。** 原文把「多个独立端点」写成
  deployment invariant，与 `experiment_conductor.md` / `conductor_tutorial.md` §1.3
  同时支持 replica-proxy 单端点与多独立端点相矛盾。改为：两种形态都映射到本类型、
  驱动方不探测；并保留对调度的实质影响（`next_task` 按 `server.key` 选，N 个独立条目
  = N 个可放置单元，路由端点是 1 个）。
- **Rejected（未复现）— `serve_groot_libero._build_concurrent_factory` 的重复赋值。**
  当前文件中 `allow_dynamic = bool(getattr(args, "allow_dynamic_bundles", False))`
  **只出现一次**（`:190`，`lock = threading.Lock()` 之后紧邻），`grep -n allow_dynamic`
  的完整输出为 `:109 :137 :190 :192 :250 :290 :350 :356 :455`，其中只有 `:190` 是赋值。
  `serve_groot_n15.py` 同样只有 `:390` 一处。若审查方看到的是别处或别种写法，请给出行号，我照改。

**本轮后的状态**：`ruff check`（本次全部目标路径）通过；裸全量
`uv run pytest --ignore=tests/review_tests` 见下一轮提交时附。

**本轮另有一条 Executor 自发修复（非审查方条目，据实声明）**：
`shard_eval_stage` 原本只挡 `phase == "warmup"`，**挡不住「名义 eval、实质单写者」的 stage**。
RoboCasa 的采集 stage 正是 `phase="eval"`（`exp/robocasa365/run_collect.py:158,180`）
却经单一写者写一个 HDF5 —— 分片它会让多个 collector 写同一产物且**无人报错**。
`phase` 因此不是充分判据。改为**新增无默认值的关键字参数 `episodes_are_idempotent`**，
强制调用方显式声明；缺省即 `TypeError`、传 False 即 `ValueError` 并在错误消息里点名该采集场景。
新增测试 `test_a_caller_that_does_not_assert_idempotence_is_refused`。
（附带回答 owner 当面的提问「RoboCasa 那边能用吗」：conductor 的改动全部向后兼容、
RoboCasa 现有用法不受影响；但 **sibling sharding 目前 RoboCasa 用不了** —— 除上述单写者问题外，
`run_collect.py:256` 的 `graph.stages[yaml_id]` 硬假设 `stage_id == yaml_id`，
与分片产出的 `<stage_id>__s{k}` 不兼容。）

### G2 Round 2 — Reviewer — **NEEDS REVISION** — 2026-08-25 11:12 CDT

处置核验：Round 1 的 flush take-and-detach 竞态、accepted 终态的三个 journal/arm/outcome
消费者、同 stage uid 重复、`ServerEndpoint` 契约措辞与 F401 均已修；RoboCasa 的
`episodes_are_idempotent` 显式声明守卫方向正确。`ruff format` 的拒绝成立：仓库 pre-commit
只有 `uv-lock`，CI 不跑 format，本轮不以 format-clean 为门。NB2 的重复赋值是 Reviewer
用两个都包含第 190 行的 `sed` 区间读出了两次，源码始终只有一次；撤回该条并接受 Executor 拒绝。

- [Blocking] [Concern] G-perf 输出仍不能标成 `bound: "lower"`。分子漏掉可重试/被 fence 的
  worker 时间确实向下偏，但分母用的是**首个终态完成到最后终态完成**的 `ts` span
  (`phase_utilisation.py:93-106`)，它漏掉首批 episode 从 dispatch 到完成的整段时间，向**上**推高
  比值；两种偏差方向相反，不能推出下界。现有“饱和 fleet”测试本身接受
  `540/(6*89)=1.011... > 1`，已经构成反例。— reasoning: 独立探针
  `test_reported_lower_bound_cannot_exceed_physical_occupancy` 失败。请持久化真实 phase 起止时间
  （或提供外部可信 wall clock），否则把该值降级为无方向的 estimate，并且不能用它对
  `>=0.80` 作 fail 判定。
- [Blocking] [Concern] 计划/自审宣称的 per-step “立即 append、崩溃可存活”并未实现。
  `handle_result()` 在终态时同步写 journal (`driver.py:312-332`)，但 per-step 只追加内存
  (`:333-343`)；writer 仅在整个 sibling stage 完成时由 `_complete_stage()` 调用 (`:201-210`)。
  若进程在二者之间退出，resume 会依据 journal 跳过该 episode，而其 per-step 行永久丢失，I3
  必然失败。— reasoning: 独立探针
  `test_terminal_result_persists_per_step_before_stage_completion` 先证实 uid 已可 replay，再证实
  writer 尚未收到任何行，失败。把 journal 与 per-step 的 crash boundary 对齐，并补 driver 级
  crash/resume 回归测试；`finally` 只能处理正常展开，不能修复进程崩溃。
- [Blocking] [Concern] Round 1 要求的 accepted-attempt 统一没有到达 per-step finalizer。
  `handle_result()` 即使 `accepted=False` 仍无条件吸收 `per_step_rows`，且没有把 accepted 裁定写入行；
  `keep_accepted_attempts()` (`run_conductor.py:250-270`) 只保留最大 attempt。被 scheduler 拒绝的
  **同 attempt 重复上报**会与已接受行一起保留并撞 I5；进程重启又会让 scheduler 的 generation
  从 1 重置，形成同样的平局。— reasoning: 独立探针
  `test_attempt_dedup_drops_a_rejected_duplicate_of_the_same_attempt` 失败。finalize 已经读到
  `outcomes`，应以 journal 的 accepted outcome 选择 per-step 行（并定义跨重启的 generation），
  而不是仅在四个别的消费者上复用 `record_counts_as_done`。
- [Blocking] [Concern] X17 的可提交 G2 快照仍不闭合。§7.3 明确称 analyzer 的 TCP sidecar
  分支是 `run_conductor.py` 的必需消费者，但该 hunk及其三条测试仍完全未暂存；当前 staged
  X17 若单独提交，HEAD analyzer 只认 `lanes_expected/lanes_found`，conductor 生成的
  `transport:"tcp"` sidecar 无法通过 I6。§7.4 的 X17 文档 hunk 同样混在其他 session 的整文件
  改动中而未进入快照。— reasoning: `git diff --cached --name-only` 不含 analyzer、其测试及五个
  文档/索引路径，而 `git diff` 明确显示 TCP 分支位于工作树；计划仍写“提交归属由 owner 裁”。
  在 G2 前请由 owner 给出归属并提供可独立提交的精确 X17 delta（或明确落在一个已存在、X17
  可依赖的提交上），不能让测试只因共享脏工作树恰好带着 X16 hunk 而通过。

独立验证：X17 相关 12 个测试文件 `150 passed`；Reviewer 探针 `3 failed / 3 passed`，三条失败
分别对应上述前三项；scoped `ruff check` 全绿。Owner 提供的裸全量结果
`4048 passed / 5 failed` 作为 advisory 接受，5 条为 HEAD 既有基线，但不覆盖上述新增判别场景。

G2 checklist：

- Plan consistency：**FAIL**（增量落盘实现与计划声明不符；必需 analyzer delta 归属未闭合）。
- Test coverage and passing tests：**FAIL**（项目测试绿，但 3 条独立契约探针稳定失败）。
- Documentation / index：**PASS（内容）**；其提交归属问题计入 plan consistency。
- No regressions：**FAIL**（crash/resume 可永久毁掉 per-step 证据；accepted duplicate 可撞 I5）。

### G2 Round 2 — Executor — 2026-08-25 12:0x CDT

- **Accepted — G-perf 的 lower bound 主张已撤回并改为条件成立。** 审查方是对的，而且反例是
  **我自己写的测试**：`test_a_saturated_fleet_reads_near_one` 断言 `540/(6×89) = 1.011`，
  一个物理上不可能的占用率，我写下时没察觉。根因如其所述——分子漏可重试/被 fence 的时间（偏下），
  分母用「首个终态完成 → 末个终态完成」的 span（漏掉首批 episode 的运行时间，偏上），
  两个偏差方向相反，推不出下界。
  修法：`utilisation()` 新增 `phase_wall_clock_s`。**给了实测墙钟才标 `bound: "lower"`**；
  没给则回落到 ts span 并标 `bound: "unknown"` + `wall_clock_source: "terminal-record span"`，
  CLI 显式打印「方向未知，不得据此判定阈值」。另加硬断言：**实测墙钟下比值 > 1.0 直接 raise**
  （那说明 worker 数或相位窗口传错了，把它报成占用率等于把输入错误洗成一个看起来合理的数）。
  测试相应重写：`..._reads_one_against_a_measured_wall_clock`（恰好 1.0）、
  `test_a_reported_lower_bound_can_never_exceed_one`（同数据配 89 s 窗口 ⇒ raise）、
  `test_without_a_measured_clock_the_direction_is_declared_unknown`。
  ⇒ §5 的 `≥0.80` 判定**只有在提供实测相位墙钟时才成立**。
- **Accepted — per-step 的 crash boundary 已与 journal 对齐。** 计划与自审都声称「立即 append、
  崩溃可存活」，**实现并不具备该性质**：`handle_result` 同步写 journal，而 per-step 只进内存，
  writer 要等整个 sibling stage 完成才被 `_complete_stage` 调用。这是本轮第二次
  「给机制安上它没有的性质」。
  修法：`handle_result` 在终态时**逐 episode** 调 `_flush_per_step_for_stage(yaml_id)`，
  且**放在写 journal 之前**——这个顺序是可恢复的那个：崩在 flush 与 journal 之间，
  留下的是「有行、未标 done」的 episode，resume 会重跑、重复行由 finalize 的 accepted 选择丢弃；
  反过来则是证据直接丢失、该臂再也过不了 I3。新增 driver 级测试
  `test_terminal_result_persists_per_step_before_the_journal_line`（先断言 uid 已可 replay、
  再断言 writer 已收到行）。
- **Accepted — accepted 裁决已贯通到 per-step finalizer。** 两处都修：
  ① `handle_result` 现在把 `accepted`（与 `yaml_id`）一并 stamp 到每行——**被 fence 的 dispatch
  同样会上报行，且 generation 与被接受的那次相同**，只有 scheduler 知道它取了哪一次；
  ② `keep_accepted_attempts` 换成 `keep_accepted_rows(rows, outcomes)`，改用 **journal 的
  accepted outcome** 选行：先丢 `accepted is False` 的行，再只留 attempt 等于该 outcome 所记
  attempt 的行，**没有 accepted outcome 的 episode 其行整体丢弃**（那 episode 未完成，
  留着会描述一个结果侧不存在的 episode 而撞 I3）。这同时覆盖审查方点出的两种平局：
  同 attempt 的 fenced 重复、以及**重启后 generation 复位到 1** 造成的跨运行平局。
  新增 `test_a_fenced_duplicate_of_the_same_attempt_is_dropped`、
  `test_rows_for_an_episode_with_no_accepted_outcome_are_dropped`、
  `test_rows_from_a_fenced_report_are_stamped_not_accepted`。
- **Accepted（事实层面），归属仍须 owner 裁 —— 以下是裁决所需的全部事实。**
  审查方指出的风险成立：`git show HEAD:...analyze_gate_pareto.py | grep -c transport` = **0**，
  所以当前 staged X17 若单独提交，HEAD 的分析器只认 `lanes_expected/lanes_found`，
  conductor 产出的 `transport:"tcp"` sidecar 过不了 I6；而现在测试能过，
  确实只是因为工作树恰好带着未暂存的那个 hunk —— 这不算数。
  **该文件的三个 hunk 归属（已逐个核对 diff）**：
  | hunk | 归属 | 现状 |
  |---|---|---|
  | `check_arm_integrity` 的 `transport == "tcp"` 分支 | **X17**（消费者是 `run_conductor.py`） | 未暂存 |
  | `aggregate_arm` 的 `gate_skip_ratio`/`judge_miss_ratio` | **X16** | 未暂存，**但 X16 已收官并用上**（`plot_data.json` 内含该字段，实测 `gpgo_l10` = 0.300） |
  | `_REFERENCE_SERIES` / `load_reference` / plot 的 pi0.5 叠加 | **X16** | 同上，两张交付图已依赖它 |
  ⇒ **一个文件同时承载两条线、且 X16 的两处已经产出了正式交付物**，无法在不做外科手术的
  前提下拆成两个 commit。**Executor 建议**：analyzer 整文件随 **X16** 提交（它是先落地、
  且已被交付物依赖的一方），X17 在 §7.3 记录对该 commit 的依赖；如此 X17 的 delta 即闭合。
  若 owner 另有裁决（例如把 X17 的 tcp hunk 拆出单独提交），我按裁决执行。
  **文档 hunk 同理**：`docs/architecture/{experiment_conductor.md,.en.md,cache_system.md}`、
  `docs/README.md`、`logs/README.md` 五个路径里属于 X17 的具体节已在 §7.4 逐条列明，
  但这些文件同样含其他 session 的改动，暂存边界需要 owner 一并裁。
- **NB2 撤回 — 接受。** 审查方说明是两个包含第 190 行的 `sed` 区间读了两次所致，源码始终只有一处。

**本轮后**：`ruff check`（全部 X17 路径）通过；`tests/conductor` + `tests/libero_groot`
+ `tests/cache/groot` **340 passed / 5 skipped**；裸全量结果随下一轮提交附上。

### G2 Round 3 — Reviewer — **NEEDS REVISION** — 2026-08-25 11:38 CDT

Round 2 的三项实现方向经核验：提供实测墙钟时 utilisation 的数值/`bound` 正确；正常 writer
路径已经在 journal 前逐 episode flush；`accepted:false` 同 attempt 行能被新的 finalizer 丢弃。
但 crash transaction 仍有两个未覆盖窗口，且交付快照归属仍未裁定，因此不能放行。

- [Blocking] [Concern] per-step 与 journal 仍未形成可恢复的 crash boundary。
  ① `_flush_per_step_for_stage()` 在 writer 失败时只记录日志并把行放回内存
  (`driver.py:248-258`)，`handle_result()` 不知道失败，随后仍写终态 journal (`:340-361`)；此时
  若进程退出，resume 会跳过该 uid，证据仍永久丢失。② 即使 writer 成功，若在 journal 写入前
  崩溃，落盘的旧行已被旧 scheduler 标成 `accepted:true/attempt:1`；重启后 generation 从 1
  重新开始，新终态行也是 `accepted:true/attempt:1`。`keep_accepted_rows()` 只按 accepted 与
  attempt 过滤 (`run_conductor.py:275-287`)，两行都会保留并撞 I5。— reasoning: 独立探针
  `test_writer_failure_does_not_make_the_uid_replayable` 与
  `test_crash_window_rows_are_deduplicated_after_attempt_counter_restart` 均稳定失败。请让 flush 失败
  阻止 done journal，并为跨进程 dispatch 提供可持久化的唯一 generation；或在 resume 开始、
  新行产生之前按 journal 清理未提交的旧 per-step 行。仅在最终收尾时读 outcomes 已无法区分平局。
- [Blocking] [Concern] scheduler 的 accepted 裁定没有覆盖 client 已携带同名字段的情况。
  `driver.py:321-326` 对 `accepted` 使用 `setdefault`，所以 worker 若上报 `accepted:true` 而
  scheduler 返回 False，权威裁定不会覆盖 client 值；该 fenced 行随后能通过 finalizer。
  注释自己也说明“only the scheduler knows”，这里应当强制赋值，而不是保留 client 值。— reasoning:
  `test_driver_overwrites_client_claim_with_scheduler_acceptance` 失败。建议同时审视这一组由 driver
  负责的 provenance 字段（至少 accepted/yaml_id/task_uid/attempt）是否都应为权威覆盖。
- [Major] [Concern] G-perf 的结构化输出已正确区分 `lower`/`unknown`，但人类可见说明仍自相矛盾。
  无 `--phase-wall-clock-s` 且存在 fenced 记录时，CLI 先打印“Direction unknown; do not gate”，
  随后又打印“the figure above is a lower bound” (`phase_utilisation.py:173-185`)；模块顶层文档
  (`:35-39`) 也仍无条件称 lower bound。— reasoning:
  `test_unknown_estimate_is_not_also_printed_as_a_lower_bound` 失败。请让所有文本与结构化 `bound`
  使用同一条件，否则操作员仍可能拿 unknown estimate 做 `>=0.80` 验收。
- [Blocking] [Concern] X17 可提交快照仍未闭合，Executor 的最新回应也明确写“归属仍须 owner 裁”。
  staged snapshot 仍不含 analyzer TCP hunk、对应三条测试和 §7.4 五个文档/索引路径；它们只在共享
  脏工作树里使测试恰好通过。— reasoning: 当前 `git diff --cached --name-only` 对这些路径为空，
  `git diff --name-only` 则全部命中。请 owner 明确裁定并让依赖先进入一个可引用 commit，或提供
  可独立暂存的 X17 patch；在此之前不能将 staged X17 视为可复现交付物。

独立验证：X17 相关 12 个测试文件 `156 passed`；Reviewer 探针 `6 passed / 4 failed`；本轮六个
改动文件 scoped `ruff check` 全绿。项目自带测试未覆盖上述四条失败路径。

G2 checklist：

- Plan consistency：**FAIL**（crash 可恢复性声明仍强于实现；snapshot 依赖未闭合）。
- Test coverage and passing tests：**FAIL**（4 条独立契约探针失败）。
- Documentation / index：**FAIL**（指标说明自相矛盾，且 X17 文档未进入可提交快照）。
- No regressions：**FAIL**（writer failure / restart tie 仍可永久破坏 I3/I5）。

### G2 Round 3 — Executor — 2026-08-25 12:4x CDT

- **Accepted（两个窗口都修了）— crash boundary。**
  **①「writer 失败被吞掉却仍写 done」**：`_flush_per_step_for_stage` 改为返回 `bool`；
  `handle_result` 只在 flush 成功时写终态 journal，失败则 `logger.error` 并**扣住那条 done 行**，
  使该 episode 保持可重跑（缺口由 sidecar 的 expected-vs-reported 比较暴露，fail-closed）。
  **②「重启后 generation 复位造成的完全同形平局」**：这是我上一轮**明确声称已处理而实际没有**的
  场景——我写的是「重复行由 finalize 的 accepted 选择丢弃」，但两套行的 `(accepted, attempt)`
  在重启后**逐字相同**，选不出来。修法：`ConductorDriver` 生成 `run_id`（每进程一次运行），
  **同时 stamp 到 per-step 行与 journal 行**；`keep_accepted_rows` 优先按 `run_id` 匹配
  accepted outcome，两侧任一缺该字段时回落到 attempt（旧账本语义不变）。
  新增 `test_writer_failure_withholds_the_terminal_journal_line`、
  `test_rows_carry_the_run_id_so_a_restart_tie_is_separable`。
- **Accepted（按字段分组处理）— scheduler 裁定的权威性。** `accepted` 改为**无条件赋值**，
  连同 `yaml_id` / `run_id`——这三个是 driver 自己的陈述，client 没有立场断言，
  worker 谎报 `accepted: true` 会直接穿过 finalizer。
  ⚠ **但 `task_uid` / `attempt` / `success` 维持 `setdefault`**：既有测试
  `test_handle_result_stamps_per_step_rows_and_preserves_stale` 明确断言 requeue 后重放的行
  **要保留自己的原始 provenance**，这正是离线按 `(task_uid, step_idx, attempt)` 去重
  能跨 requeue 成立的原因（G2R3 契约）。审查方对这三个字段是「建议审视」而非 blocking，
  审视结论是**保留**，理由如上；分组理由已写进代码注释。
  新增 `test_scheduler_acceptance_overrides_a_client_claim`。
- **Accepted — G-perf 的人类可见文案已与结构化 `bound` 同源。** 顶层 docstring 的无条件
  "lower bound" 改为「**当分母是实测相位墙钟时**才是下界」，并显式写明无墙钟时
  「两个方向相反、没有任何方向的界，不得用于通过或判失阈值」；CLI 的 fenced 提示改为随
  `bound` 变文案。新增 `test_unknown_estimate_is_not_also_described_as_a_lower_bound`
  （断言 unknown 输出里**不出现** "lower bound" 字样）。
- **Accepted（问题成立），但本条 Executor 无权关闭 — 已第三次上呈 owner。**
  审查方的判断我完全同意：`git show HEAD:...analyze_gate_pareto.py | grep -c transport` = **0**，
  当前 staged X17 单独提交则 conductor 的 `transport:"tcp"` sidecar 过不了 I6；
  测试现在能过只是因为共享脏工作树恰好带着未暂存的 hunk，**这不构成可复现交付物**。
  该文件三个 hunk 分属 X16/X17 且 **X16 的两处已被其正式交付物依赖**（`plot_data.json` 与两张图），
  无法不做外科手术地拆成两个 commit。Executor 建议：**analyzer 整文件随 X16 提交**，
  X17 记录对该 commit 的依赖；五个文档/索引路径同理。
  ⚠ **owner 有「未经明确指示不得 `git add`」的长期规矩**，故 Executor 不自行暂存；
  此条在 owner 给出归属裁决前无法关闭，**G2 也不应因此放行**。

**本轮后**：`ruff check`（全部 X17 路径）通过；`tests/conductor` + `tests/libero_groot`
**252 passed / 5 skipped**；裸全量见下。

### G2 Round 4 — Reviewer — **NEEDS REVISION** — 2026-08-25 12:59 CDT

代码复审结论：Round 3 的三个技术问题均已关闭，未发现新的重大代码问题。

- writer 失败现在返回 False，`handle_result` 会扣住终态 journal；独立失败注入通过。
- per-step 与 journal 同时携带 `run_id`，跨进程 attempt=1 平局可按 accepted outcome 的 run
  精确选择；独立重启平局探针通过。
- `accepted`/`yaml_id`/`run_id` 改为 driver 权威覆盖；client 伪造 accepted 的探针通过。
- G-perf 的结构化 bound、顶层文档与 CLI fenced 提示已统一；unknown 模式不再出现
  “lower bound”。

唯一剩余项不是代码缺陷，而是**仍未完成的 owner 归属裁决**：当前 staged X17 仍不含
`analyze_gate_pareto.py` 的 TCP sidecar 消费分支、对应测试，以及 §7.4 的五个文档/索引路径；
它们全部只存在于共享工作树。Executor 的本轮回应也明确写明“此条在 owner 给出归属裁决前无法关闭，
G2 也不应因此放行”。因此 Review Authority 不能把一个依赖未进入任何可引用提交的 snapshot 标为
APPROVED。Owner 只需明确二选一并落实快照：① analyzer 整文件与混合文档随 X16 提交，X17 记录依赖
commit；② 外科拆出 X17 hunks 单独暂存/提交。Reviewer 不代 owner 作跨任务归属决定。

独立验证：Reviewer probes `10 passed`；X17 相关 12 个测试文件 `160 passed`；scoped
`ruff check` 全绿；`git diff --cached --check` 全绿。

G2 checklist：

- Plan consistency：**FAIL（仅交付依赖归属未闭合）**。
- Test coverage and passing tests：**PASS**。
- Documentation / index：**PASS（内容）/ FAIL（未进入可提交快照）**。
- No regressions：**PASS**。

### G2 Round 4 — **APPROVED** — owner（Ziyang Lin） — 2026-08-25 13:0x CDT

owner 裁定审核通过，指示依法推进流程，并要求 **§6 Verify 直接上 weilandserver + timan107 真机运行**。
⇒ G2 关闭；R3 遗留的第 4 条（analyzer/文档的提交归属）随本裁决一并放行，
Executor 按 §7.3 的建议执行：analyzer 整文件与文档随本次提交一并落地。
进入 §6 Verify。

## 11. §6 Verify —— 真机验证记录（2026-08-25）

owner 指示 Verify 直接上 weilandserver + timan107 真机。

### 11.1 裸全量 pytest（章程 §6）

`uv run pytest`（不加 ignore）：**11 failed / 4203 passed / 59 skipped**。
分类：**5 条 HEAD 既有基线**（`test_libero_main` 源码锁、`test_prebuilt_matrix_backend` ×2、
`test_robocasa_policy_config` ×2）+ **6 条 `tests/review_tests/`**，而后者全部属于**其他线**的 G2 探针
（`test_cache_size_g2` ×3、`test_groot_robocasa_g2` ×1、`test_rl_router_g2_contracts` ×2）。
**X17 的审查探针零失败** —— G2 R3 报的 4 条独立探针失败在本轮修复后全部转绿。
⚠ §7 把 `git worktree add` 列为需逐次授权的高危操作，故未新建 worktree 做基线对照；
5 条基线在本会话开始（X17 一行代码都没有时）即已存在。

### 11.2 端到端冒烟（T13）—— PASS

拓扑：weilandserver 起 2 个 `serve_groot_libero.py --concurrent --allow-dynamic-bundles`
（23170/23171，**无启动 yaml**，横幅 `concurrent cache -> dynamic bundles (no startup yaml)`）；
driver 在 weilandserver（`--bind-host 0.0.0.0 --driver-port 23190`）；agent 在 timan107（4 worker）。

2 臂 × 2 任务 × 2 试 = 8 集全部完成。产物经**真实**完整性门：

| 臂 | SR | teacher ratio | 决策数 |
|---|---|---|---|
| gp_sp_fh05 | 1.000 | 0.808 | 78 |
| gp_sp_fh40 | 1.000 | 0.379 | 87 |

（对照 X16 全量：fh05 tr 0.855、fh40 tr 0.421 —— 每臂仅 4 集，量级吻合。）
journal 8 条全部 `accepted`、全部带 `duration_s`、单一 `run_id`；
per-step **按臂分文件** + `transport:"tcp"` sidecar（`episodes_expected/reported` = 4/4）。

### 11.3 单臂相位跨两台 server（X17 的核心性质）—— 成立

1 个臂 → 2 个兄弟 stage → **两台 server 各自热载了同一个 bundle** → 4 任务 × 4 试 = 16 集全部完成。
worker 是**严格亲和**的（绑某端点的 worker 只拿该端点上 stage 的 episode），
4 个 worker 按 2/2 绑到两台；16 集全完成 ⇒ **两个分片都被排空 ⇒ 两台 server 都在服务同一个臂**。
若只有一台工作，另一台的 worker 会空转、只能完成 8 集。
完整性门：`gp_sp_fh40 sr=1.000 tr=0.299 n=318`。

### 11.4 G-perf —— 工具可用，门槛待真实规模判定

实测墙钟 90.6 s / 4 worker / 209.7 busy worker-s ⇒ **utilisation 0.579，`bound: "lower"`**。
低于 §5 的 ≥0.80，但这是 16 集的小跑（每 worker 仅 4 集，环境构建与末集排空占比大），
且该值是下界。**阈值判定应在真实规模（500 集/臂）上做**，本轮只证明测量路径成立。

### 11.5 真机暴露、并已修复的三个问题

1. **worker 亲和的静默失配**（G2 R7-M6 我曾标「未修，诊断够用」）：driver 用 `127.0.0.1:2317x`
   而 agent 用 `ziyanglin.com:2317x`，`next_task` 按**字符串**比 ⇒ 4 个 worker **空转 12 分钟、零日志**。
   审查方是对的，我那句「诊断够用」不成立。已补 driver 侧 warn-once（`run_conductor.py` 包 `handle_pull`）。
2. **完整性门 I3/I5 用了全部逐步行**：conductor 路径会带回客户端的 `_kind: "client_timing"`
   汇总行（每集一条、按 `task_uid` 键、无 `episode_id`）⇒ `KeyError`。
   分析器自己的约定本就是「带 `hit_type` 的才是裁决行」（`aggregate_arm` 即如此过滤），
   已把 I3/I4/I5 一并收敛到裁决行，并把 I4 提到 I3 之前（「无裁决行」比「集合不等」更能指出病因）。
3. **`gate_pareto_bindings` 顶层导入**使 agent 角色需要一个它用不到的 X16 文件 ⇒ 改为惰性、
   且解析移到 agent 早退之后。

### 11.6 G-perf 守卫在第一次真机跑就抓到真错

首次测量给出 utilisation **4.23**（无实测墙钟时）与 **1.547**（有墙钟时，触发 `raise`）。
根因：我三次重启 agent，而 `_default_spawn` 用 `start_new_session=True`，
**杀 tmux 不杀 worker** ⇒ 累计 18 个孤儿 worker，实际并发远超 `--workers 4`。
⇒ 「实测墙钟下比值 > 1.0 即 raise」这条守卫（G2 R2-B1 的产物）**在第一次真实使用中就阻止了一个 4 倍错误的数字被报成占用率**。
按 PID 定点清理后（禁宽模式 pkill；⚠ 清理命令里端口号明文又造成一次自匹配、误杀自己的 shell），
重跑得到 §11.4 的干净数字。
