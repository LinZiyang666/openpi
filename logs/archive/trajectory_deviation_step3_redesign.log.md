# Trajectory Deviation Step 3 Plan — Per-cycle Policy 刻舟求剑

> Status: Validated
> Task: 废弃旧 Step 3 的 "GT teleport + prefill + pure-cache rollout" 设计，改为 per-cycle policy selection 实验。按 Step 2 pre-computed 的 deviate flag 驱动每次 inference cycle 选择 cache 或 inference，直接在真实 env 中测纠偏效果。
> 关联文档：
> - [trajectory_deviation_experiment_plan.log.md](trajectory_deviation_experiment_plan.log.md)（Phase A/B/C 原始方案）
> - [trajectory_deviation_corrective_experiment.log.md](trajectory_deviation_corrective_experiment.log.md)（旧 Step 3 方案，本 plan 实施后废弃）

---

## 1. 背景

### 1.1 核心假设

Cache 轨迹失败来自少数 deviate points 的局部偏差引发的级联效应。若能在这些 point 处用 inference 替代 cache，即可恢复整条轨迹的成功。

### 1.2 不采用旧 Step 3 设计的原因

旧 `exp/trajectory_deviation/run_spawn_experiment.py` 的做法：选 top-k deviate points → teleport env 到 GT 某 cycle → prefill trajectory history → 从该 cycle 起跑**纯 cache** 到 episode 结束。这隐含假设「非 deviate point 处 cache 产出轨迹与 GT 重合」，而这恰是本实验要验证的命题 —— 用结论当前提，不成立。

### 1.3 新思路：刻舟求剑

不再试图"让实验轨迹对齐 GT"。直接在真实 env 里跑 episode，按 **cycle index** 查 GT 上的 deviate flag 决定本 cycle 用 cache 还是 inference：

- Deviate flag 是 pre-computed 属性（Step 2 的 `deviate_score[cycle] > τ`）；
- Cache 跑出的 action 与 GT 偏离多少不关心；
- 触发 inference 后跑出的 action 被当作事实接受；
- 实验 cycle 数超过 GT 长度后无参照，全部走 cache（但已开启的 burst 可跨界延续）。

---

## 2. 实验设计

### 2.1 术语约定

本 plan 中的 **step ≡ cycle ≡ 一次 `conn.infer(obs)` 调用**。与 Step 2 `compute_deviate_scores.py::load_gt_episode` 按 `num_cycles` 枚举的语义对齐。每次 inference 内部执行多少 LIBERO env action、env step 如何推进，**本实验一概不管**（由 `examples/libero/main.py` 默认 `replan_steps=5` 管）。下文统一用 "cycle" 指代此单位。

### 2.2 输入

| 来源 | 产物 | 用途 |
|------|------|------|
| Step 1a | `cache_eval_results.json` | 筛 cache 失败的 `(task_id, orig_init_idx)` 子集 |
| Step 1b | `gt/task_{id}/episode_{idx}.h5` | 提供 GT cycle 数 `T_gt_cycles`；init scene 入口对齐 |
| Step 2 | `deviate_score_{cfg}.json` | 计算 deviate flag |

### 2.3 参数

| 参数 | 符号 | 含义 | 扫描范围 |
|------|------|------|---------|
| Deviate threshold | τ | `deviate_score[c] > τ` → `is_deviate[c] = True` | `{3, 5, 7, 10}`（cross-cfg 绝对值） |
| Inference burst length | n | 命中 deviate point 后连续 n 个 cycle（含本 cycle）走 inference | `{1, 2, 3, 5, 10}` |

**τ 扫描档位**（基于 Step 2 全集 272 ep 统计）：

| τ | %cycles_flagged | mean_flags/ep | %eps_with_any_flag |
|---|-----------------|----------------|---------------------|
| 3 | ~24% | ~5 | 99% |
| 5 | ~6.5% | ~1.4 | ~67% |
| 7 | ~3% | ~0.7 | ~50% |
| 10 | ~2% | ~0.5 | ~40% |

### 2.4 Per-cycle Policy 规则

```python
# Pre-compute per-episode per-cycle deviate flag (client side, from Step 2 output):
#   is_deviate[c] = deviate_score[c] > τ, for c in [0, T_gt_cycles - 1]
# T_gt_cycles = len(deviate_score[ep_key])
# Episode runtime loop (one iteration == one conn.infer() call == one cycle):

burst_remaining = 0
cycle = 0
while not env.done and not env.success:
    # 新 burst 仅在「当前不在 burst 内 AND cycle 在 GT 范围内 AND 该 cycle 是 deviate」时触发
    if burst_remaining == 0 and cycle < T_gt_cycles and is_deviate[cycle]:
        burst_remaining = n

    if burst_remaining > 0:
        decision = "skip"                # 走 inference（gate skip → fall-through）
        burst_remaining -= 1             # 允许 burst 跨过 T_gt_cycles 边界继续 inference
    else:
        decision = "search"              # 走 cache

    action_chunk = conn.infer({**obs, "__gate_decision__": decision})
    obs = env.run_action_chunk(action_chunk)
    cycle += 1
    if cycle >= MAX_CYCLES_GUARD:        # 仅作 runaway 兜底
        break
```

**关键规则**：
1. 按 cycle index 查 flag（刻舟求剑），不看 obs 是否真对齐 GT；
2. Burst 内无条件 inference，不查 flag、不重启、不延长；burst 内命中的 deviate cycle 自然被吞噬；
3. `cycle < T_gt_cycles` 非 burst 且非 deviate → cache；
4. `cycle >= T_gt_cycles`：burst 未完 → 继续 inference；burst 已完 → 一律 cache（无 GT 参照，不再开新 burst）；
5. 终止：LIBERO env 的 `done / success` 优先；client 侧加 `MAX_CYCLES_GUARD = ceil(max_env_steps / replan_steps) + safety`（默认 `220 / 5 + 5 = 49`）作 runaway 兜底。

### 2.5 指标

| 指标 | 定义 | 类型 |
|------|------|------|
| Success | LIBERO `_check_success()` 在 episode 结束前达成 | 主指标 |
| Inference ratio | `num_inference_cycles / total_cycles` | 成本指标 |

所有"count / ratio"均以 cycle 为单位，不以 env step 计。

每 episode 产出：`{cfg, task_id, init_idx, τ, n, success, inference_ratio, total_cycles, num_inference_cycles, T_gt_cycles}`。

聚合：`(cfg, τ, n) → {success_rate, mean_inference_ratio}`。

---

## 3. 实验范围与语义

| 维度 | 取值 | 备注 |
|------|------|------|
| Episode 集合 | per-cfg fail set（clip 159 / spatial16 154 / max_pool 150） | 三 cfg 之间不做 intersection；每 cfg 独立实验结果 |
| Threshold 语义 | cross-cfg 绝对值 | 不做 per-cfg 分位 |
| 主要输出 | `env.success` (bool) + cycle-粒度 inference ratio | |
| Online detector | 不做 | 效度边界：实验 cycle c 的 obs 因漂移已不等于 GT 第 c 次 inference 时的 obs；flag 对齐仅在 cycle-index 层面 |
| Trajectory history 语义 | Cache 和 inference cycle 都正常累积 | 由 `orchestrator.py:219-226` gate-skip 路径保证：`record_query_keys` + `broadcast_action` 都执行 |
| 随机 / 周期 baseline | 独立为 Step 4 / Step 5（本轮不实现） | 见 §7 |
| Pareto 分析 | 独立为 Step 6（本轮不实现） | 见 §7 |

---

## 4. 并行运行布局

硬件：**3 个 server × 每 server 1 client × 每 client 5 worker**。每个 server 跑一个 cfg 的**全量** fail set，不切 shard。

| Slot | Server | YAML | Runner 参数 |
|------|--------|------|-------------|
| A | host_a:8001 | `step3_clip_w7_d4.yaml` | `--deviate-score-json .../deviate_score_clip_w7_d4.json --cfg clip_w7_d4 --host host_a --port 8001 --num-workers 5 --out-dir out/step3/clip/` |
| B | host_b:8001 | `step3_spatial16_w8_d4.yaml` | `--deviate-score-json .../deviate_score_spatial16_w8_d4.json --cfg spatial16_w8_d4 --host host_b --port 8001 --num-workers 5 --out-dir out/step3/spatial16/` |
| C | host_c:8001 | `step3_max_pool_w3_d5.yaml` | `--deviate-score-json .../deviate_score_max_pool_w3_d5.json --cfg max_pool_w3_d5 --host host_c --port 8001 --num-workers 5 --out-dir out/step3/max_pool/` |

**规则**：
- Runner CLI 接受 `--num-workers`（默认 1，硬上限 5；MuJoCo EGL 限制，与 `examples/libero/main.py` 一致）。
- 5 worker 共用一个 `--out-dir` / `--state-path` / JSONL append 文件，用进程内锁保证原子写。
- Runner 不做 sharding：直接按 `deviate_score_{cfg}.json` 全量跑。
- 聚合：合并 3 个 cfg 的 JSONL → 跨 cfg CSV（`merge_step3_cfgs.py`）。
- 三 cfg 互不通信，断点续跑粒度 = `(cfg, episode, τ, n)`，state 文件在 `--out-dir` 下独立。

---

## 5. Step 3 实施

### 5.1 新增 / 修改文件

| 文件 | 类型 | 描述 |
|------|------|------|
| `src/openpi/cache/components/gate.py` | 修改 | `GateFunction.__call__` 加第三参数 `request_context: dict \| None = None`；`AlwaysSearchGate` / `AlwaysSkipGate` 加兼容默认参数；新增 `ClientControlledGate` |
| `src/openpi/cache/orchestrator.py` | 修改 | `CacheOrchestrator.check(...)` 加 kwarg-only `request_context` 并透传给 gate；`__init__` 计算 `accepts_client_signal: bool` 属性 |
| `src/openpi/cache/interceptor.py` | 修改 | `infer()` 最外层（`_input_transform` 之前）pop 保留字段 `__gate_decision__` → 构造 `request_context` → 透传给两次 `orchestrator.check`；配合 `accepts_client_signal` 做 config mismatch fail-loud |
| `src/openpi/cache/config.py` | 修改 | `_build_gate` 分支接受 `gate.type: client_controlled`（无额外字段） |
| `configs/cache_runs/deviate_exp/step3_clip_w7_d4.yaml` | 新建 | `gate.type: client_controlled`，其余字段与 `clip_w7_d4.yaml` 一致 |
| `configs/cache_runs/deviate_exp/step3_spatial16_w8_d4.yaml` | 新建 | 同上（spatial16） |
| `configs/cache_runs/deviate_exp/step3_max_pool_w3_d5.yaml` | 新建 | 同上（max_pool） |
| `exp/trajectory_deviation/run_step3_per_cycle_policy.py` | 新建 | Step 3 主 runner |
| `exp/trajectory_deviation/merge_step3_cfgs.py` | 新建 | 聚合 3 cfg 产物到单份 CSV |

**显式不改动**：
- `packages/openpi-client/src/openpi_client/websocket_client_policy.py` — client 侧 `infer(obs)` 天然 JSON-打包整个 dict，保留字段透明传输。
- `src/openpi/serving/websocket_policy_server.py` — handler 对 obs 字段不可知，无需处理保留字段。

### 5.2 Gate 接口扩展 + `ClientControlledGate`

采用方案 A：扩展 `GateFunction` 协议签名。线性数据流：`interceptor.infer` → `orchestrator.check(request_context=...)` → `gate(ck, cached_data, request_context)`。`SearchContext` 字段不变，不承载 client signal（`SearchContext` 仍在 gate **之后**构造）。

**`GateFunction` 新签名**：
```python
# src/openpi/cache/components/gate.py

@runtime_checkable
class GateFunction(Protocol):
    def __call__(
        self,
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
        request_context: dict | None = None,   # NEW
    ) -> bool: ...
```

`AlwaysSearchGate` / `AlwaysSkipGate`：加默认参数并忽略。

**`ClientControlledGate`**：
```python
# src/openpi/cache/components/gate.py

class ClientControlledGate:
    """Gate whose skip/search decision is driven by a per-request client signal.

    Client runner maintains is_deviate[cycle] and burst_remaining, and injects
    {"__gate_decision__": "skip" | "search"} into each obs. InferenceInterceptor
    pops that field BEFORE _input_transform and forwards it as
    request_context={"gate_decision": <value>}.

    Coupling:
      - REQUIRES: request_context with key "gate_decision"
      - FAILS LOUD: raises ValueError on missing / unknown value
      - UNAFFECTED BY: cached_data
      - Skip-path trajectory semantics unchanged (orchestrator.py:219-226
        handles it identically to AlwaysSkipGate).
    """

    def __call__(
        self,
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
        request_context: dict | None = None,
    ) -> bool:
        if request_context is None or "gate_decision" not in request_context:
            raise ValueError(
                "ClientControlledGate requires request_context['gate_decision']. "
                "Ensure obs carries '__gate_decision__' and that "
                "InferenceInterceptor.infer() forwards it to orchestrator.check(). "
                "Verify the cache YAML sets gate.type='client_controlled'."
            )
        decision = request_context["gate_decision"]
        if decision == "skip":
            return False
        if decision == "search":
            return True
        raise ValueError(
            f"ClientControlledGate: unknown gate_decision={decision!r}. "
            "Expected 'skip' or 'search'."
        )

    def on_episode_start(self) -> None:
        """No-op. Signature matches GateFunction protocol."""

    def record_action(self, action_chunk: torch.Tensor) -> None:
        """No-op. Signature matches GateFunction protocol."""
```

**`CacheOrchestrator.check` 签名变化**：
```python
# src/openpi/cache/orchestrator.py

def check(
    self,
    checkpoint_id: CheckpointID,
    *,
    request_context: dict | None = None,   # NEW kwarg-only
    **stage_outputs,
) -> CheckResult:
    ...
    with self._timer.measure(f"{prefix}_gate"):
        should_search = gate(
            checkpoint_id,
            self._key_builder.cached_data,
            request_context,                # NEW passthrough
        )
    ...
```

`__init__` 新增：
```python
self.accepts_client_signal: bool = any(
    isinstance(g, ClientControlledGate) for g in self._gates.values()
)
```

### 5.3 保留字段 `__gate_decision__` 的透传链

`__gate_decision__` 由 `InferenceInterceptor.infer()` 在 `_input_transform` **之前**显式 pop，绝不进入模型输入管道。

```
[client runner]
  obs = {**env.current_obs(), "__gate_decision__": "skip" | "search"}
  conn.infer(obs)   # WebsocketClientPolicy JSON-packs the whole dict as-is

[server, WebsocketPolicyServer._handler]
  # Receives obs dict, forwards to policy.infer(obs). No changes needed.

[server, InferenceInterceptor.infer(obs)]   # NEW pop logic at top of method
  # --- reserved-field pop (BEFORE _input_transform) ---
  client_signal = obs.pop("__gate_decision__", None)

  # Fail-loud on config mismatch:
  accepts = (self._orchestrator is not None
             and self._orchestrator.accepts_client_signal)
  if client_signal is not None and not accepts:
      raise ValueError(
          "obs carries '__gate_decision__' but no ClientControlledGate is "
          "configured at CP1 or CP3. Remove the field from obs, or load a "
          "cache config with gate.type='client_controlled'."
      )

  # obs (without the reserved field) flows into _input_transform as-is
  inputs = jax.tree.map(lambda x: x, obs)
  inputs = self._input_transform(inputs)
  # ... existing infer() body ...

  request_context = (
      {"gate_decision": client_signal} if client_signal is not None else None
  )
  cp1_result = self._orchestrator.check(
      CheckpointID.CP1, request_context=request_context, **cp1_kwargs
  )
  # ... same request_context passed to CP3 check, even if CP3 has no
  #     ClientControlledGate — default gate signature ignores it ...
```

**关键契约**：
1. **Pop 位置固定**：`interceptor.py` 原 L434-436 之前（即 `inputs = jax.tree.map(...)` 之前）。保留字段永不进入 `_input_transform` / `key_builder.collect` / Stage1 / `SearchContext`。
2. **Config 匹配矩阵**：

   | obs 有字段? | `accepts_client_signal` | 行为 |
   |------------|------------------------|------|
   | ✅ | ✅ | 正常透传到 `ClientControlledGate` |
   | ✅ | ❌ | interceptor **raise**（配置错配） |
   | ❌ | ✅ | `ClientControlledGate.__call__` 时 **raise**（错误点靠近需要信号处） |
   | ❌ | ❌ | 正常，`request_context=None`，各 gate 忽略 |

3. **CP3 同传不消费**：同一 `request_context` 传给 CP1 / CP3 两次 check；CP3 默认 gate 吞掉即可。
4. **保留前缀语义**：`__xxx__` 双下划线前缀 = 框架内部保留字段，必须在 `_input_transform` 前 pop；不需要额外的 "`__` 前缀忽略" 约定（字段压根到不了下游）。

### 5.4 Runner 骨架

并发模型参考 `examples/libero/main.py::_run_concurrent`（L556-693）：每 worker 持有**独立** `WebsocketClientPolicy` + **独立** LIBERO env，env 创建 + WS 连接用 `init_lock` 串行化；`num_workers ≤ 5`（MuJoCo EGL 限制）。

```python
# exp/trajectory_deviation/run_step3_per_cycle_policy.py (伪码)

def run_cfg(args):
    scores = load_json(args.deviate_score_json)          # {ep_key: {deviate_score: [...]}}
    # cfg bundle 加载一次（per server 生命周期）
    CacheConfigRPC(args.host, args.port).load_cache_config(
        f"configs/cache_runs/deviate_exp/step3_{args.cfg}.yaml"
    )

    state = BaseRunState(args.state_path)                 # 原子写 + 文件锁
    jsonl = JsonlAppender(args.jsonl_path)                # append + 进程内锁

    unit_queue = queue.Queue()
    for ep_key in scores:
        for tau in args.tau_grid:
            for n in args.n_grid:
                unit = (ep_key, tau, n)
                if not state.is_done(unit):
                    unit_queue.put(unit)

    init_lock = threading.Lock()
    stop = threading.Event()
    threads = [
        threading.Thread(target=worker_loop,
                         args=(i, init_lock, stop, unit_queue, scores, args, state, jsonl))
        for i in range(min(args.num_workers, 5))          # 硬上限 5
    ]
    for t in threads: t.start()
    for t in threads: t.join()

def worker_loop(wid, init_lock, stop, unit_queue, scores, args, state, jsonl):
    # 每 worker 独立 client + 独立 env
    with init_lock:
        conn = WebsocketClientPolicy(args.host, args.port)
    cached_env = None                                     # {task_id: env}

    while not stop.is_set():
        try:
            unit = unit_queue.get_nowait()
        except queue.Empty:
            return
        ep_key, tau, n = unit
        task_id, init_idx = parse_ep_key(ep_key)

        if cached_env is None or cached_env.task_id != task_id:
            with init_lock:
                cached_env = make_libero_env(args.task_suite_name, task_id)
        env = cached_env
        env.reset_to_init(init_idx)                        # 从 Step 1b inits 还原

        record = run_one_unit(env, conn, ep_key, tau, n, scores, args)
        state.record_atomic(unit, record)
        jsonl.append(record)

def run_one_unit(env, conn, ep_key, tau, n, scores, args):
    score_arr = np.array(scores[ep_key]["deviate_score"])
    T_gt_cycles = len(score_arr)
    is_deviate = score_arr > tau

    conn.episode_start(experiment=args.experiment_tag,
                       episode_name=f"{args.cfg}/tau{tau}_n{n}/{ep_key}")

    burst_remaining = 0
    num_inference = 0
    cycle = 0
    obs = env.current_obs()
    success = False
    while cycle < args.max_cycles_guard:
        if burst_remaining == 0 and cycle < T_gt_cycles and is_deviate[cycle]:
            burst_remaining = n
        if burst_remaining > 0:
            decision = "skip"
            burst_remaining -= 1
            num_inference += 1
        else:
            decision = "search"

        action_chunk = conn.infer({**obs, "__gate_decision__": decision})
        obs, success, done = env.run_action_chunk(action_chunk)
        cycle += 1
        if success or done: break

    conn.episode_end()
    return {
        "cfg": args.cfg, "ep_key": ep_key, "tau": tau, "n": n,
        "success": success, "total_cycles": cycle,
        "num_inference_cycles": num_inference,
        "inference_ratio": num_inference / max(cycle, 1),
        "T_gt_cycles": T_gt_cycles,
    }
```

**线程安全**：
- `state.record_atomic` / `jsonl.append` 必须持锁（沿用 `exp/common/_run_state_base.py` 风格）。
- `cached_env` / `conn` 都是 worker-local，不跨线程共享（websocket send/recv 非线程安全）。
- 同 task 的多个 unit 在同 worker 内复用 env（仅 `reset_to_init`），跨 task 才重建。

### 5.5 产物 Schema

**Per-unit JSONL**（每行一条 unit 结果；所有计数均以 cycle 为单位）：
```json
{"cfg":"clip_w7_d4","ep_key":"task_3/episode_7","tau":5,"n":3,
 "success":true,"inference_ratio":0.20,"total_cycles":25,"T_gt_cycles":22,
 "num_inference_cycles":5,"burst_count":2}
```

**聚合 CSV**（`merge_step3_cfgs.py` 产出）：
```
cfg,tau,n,episodes,success_rate,mean_inference_ratio,std_inference_ratio
clip_w7_d4,3,1,159,0.32,0.23,0.08
clip_w7_d4,3,3,159,0.41,0.28,0.09
...
```

### 5.6 验收条件

- 3 cfg 全部跑完并可 merge
- 三 cfg × 4 τ × 5 n = 60 setting 的聚合 CSV 生成
- 对每个 cfg，至少有一组 `(τ, n)` 的 `success_rate` 显著高于 0（Step 1a 的纯 cache 等于 0），即假设得到初步支持
- 不保证达到最终 "≥15pp 提升" 目标（留给 Step 4/5/6 对比）

### 5.7 工作量估计

- 实现：1-2 天（gate + orchestrator + interceptor + YAML + runner + merge 脚本）
- 单 cfg / 单 ep / 单 `(τ, n)` smoke：0.5 天
- 全量跑（3 server 并行）：~10 小时（每 server ~150 ep × 20 setting × ~25 cycle × 0.3s/cycle ÷ 5 worker ≈ 5 小时）
- 总计：~3 天

---

## 6. 旧 Step 3 代码归档（smoke 后置）

归档不作为新 Step 3 实现的前置 commit。新 Step 3 路径：**smoke 优先，归档后置**。

### 6.1 执行顺序

1. 先最小实现新 Step 3（§5），跑通单 cfg / 单 ep / 单 `(τ, n)` smoke。
2. smoke 通过后执行 Commit A：归档 + 目录清理 + 文档 banner（本节 §6.2–§6.4）。
3. 再跑 3 cfg 全量（§5）。
4. 实验完成后执行 Commit B：完整重写 runbook（§6.5）。

### 6.2 资产盘点

**代码 / 测试（归档）**：

| 文件 | 说明 |
|------|------|
| `exp/trajectory_deviation/run_spawn_experiment.py` | 旧 spawn runner |
| `exp/trajectory_deviation/analyze_deviation_results.py` | 旧聚合脚本 |
| `tests/exp/test_run_spawn_experiment.py` | spawn runner 单测 |
| `tests/exp/test_analyze_deviation_results.py` | 聚合单测 |
| `scripts/verify_env_save_restore.py` | 旧 teleport 前提校验 |
| `scripts/verify_restore_obs_equivalence.py` | 旧 teleport obs 等价校验 |

**共享辅助（保留，不动）**：
- `exp/trajectory_deviation/_libero_env.py` — Step 1a/1b/2/新 3 均依赖
- `exp/trajectory_deviation/compute_deviate_scores.py` — Step 2 主脚本
- `exp/common/_unit_key.py` — 通用 unit key 工具

**产物数据（原地改名归档）**：
- `data/deviation_experiment/spawn_dry_run/` → `data/deviation_experiment/_archive_spawn_dry_run/`

**文档引用（更新）**：
- `docs/experiments/trajectory_deviation.md` — Step 3 章节重写
- `docs/experiments/trajectory_deviation.en.md` — 同步
- `docs/reference/openpi.md` L79 — `run_spawn_experiment.py` 条目替换为新 runner 名字

### 6.3 归档策略

**代码归档**：物理 `git mv` 到各模块下 `archive/` 子目录：

| 新位置 | 来源 |
|--------|------|
| `exp/trajectory_deviation/archive/run_spawn_experiment.py` | 移动 |
| `exp/trajectory_deviation/archive/analyze_deviation_results.py` | 移动 |
| `exp/trajectory_deviation/archive/README.md` | 新建（废弃原因 + 替代方案 + 日期 2026-04-16） |
| `tests/exp/archive/test_run_spawn_experiment.py` | 移动 |
| `tests/exp/archive/test_analyze_deviation_results.py` | 移动 |
| `scripts/archive/verify_env_save_restore.py` | 移动 |
| `scripts/archive/verify_restore_obs_equivalence.py` | 移动 |

**保证事项**：
- `archive/` 下的 py 文件不在 pytest collect 路径（`tests/exp/archive/conftest.py` 加 `collect_ignore_glob = ["*.py"]`，或配在根 `pytest.ini`）
- `archive/` 下的代码不 import
- `archive/README.md` 顶部 banner：`> ⚠️ DEPRECATED — 见 logs/trajectory_deviation_step3_redesign.log.md §1.2`

### 6.4 验收 checklist

Commit A 提交前：
- [ ] 旧 runner / 单测 / verify 脚本全部进 `archive/`
- [ ] `archive/README.md` 三份（exp / tests / scripts 各一）都写清废弃原因 + 替代方案 + 日期
- [ ] `rg run_spawn_experiment` 在主路径（非 archive、非 logs）下零命中
- [ ] `rg analyze_deviation_results` 同上
- [ ] `spawn_dry_run/` 改名完成
- [ ] `docs/experiments/trajectory_deviation.md` Step 3 章节替换为占位 banner（完整重写放 Commit B）
- [ ] `uv run pytest tests/` 全绿
- [ ] `python -c "from exp.trajectory_deviation import compute_deviate_scores"` 可 import（Step 2 不挂）

### 6.5 回滚路径

若新 Step 3 在后期被否决：`git revert <commit_a_sha>` 还原 archive。archive README 标注日期便于未来识别归档范围。

---

## 7. Step 4 / 5 / 6 Outline（本轮不实现）

| Step | 类型 | 描述 |
|------|------|------|
| Step 4 | Random baseline | 依赖 Step 3 产出每个 `(cfg, τ, n)` 的实际 inference cycle 数 `r[ep]`；复用 §5 runner 框架，随机选 `r[ep]` 个 cycle 走 inference；`--random-seeds 0 1 2` 取均值 |
| Step 5 | Periodic baseline | 参数 `(k, n)`：每 k 个 cache cycle 强制 n 个 inference cycle，循环至 episode 结束；扫描 `k ∈ {5, 10, 20}`, `n ∈ {1, 3, 5}` |
| Step 6 | Pareto 分析 | 合并 Step 3/4/5 产物：`success_rate vs inference_ratio` Pareto 曲线；`success vs (τ, n)` 热力图；三 cfg 横向对比 |

对应新文件：`run_step4_random_baseline.py` / `run_step5_periodic_baseline.py` / `analyze_step3_to_6.py`。

---

## 8. 实现注意事项（非阻塞）

- **`InferenceInterceptor.infer()` 顶部处理 `self._orchestrator is None`**：pop `__gate_decision__` 后立即判断，若 orchestrator 为 None 且 obs 带字段，按 `accepts_client_signal=False` 语义 fail loud，避免 `AttributeError`。
- **`obs.pop("__gate_decision__", None)` 会原地修改 dict**：WebSocket 路径无问题（obs 是反序列化出的新 dict）；单测 / in-process 复用同一 obs 时，需要显式拷贝或在测试里覆盖该行为。
- **G2 最小测试覆盖**：
  1. `ClientControlledGate` 四种路径 —— `request_context` 缺失 / `gate_decision` 缺失 / 值非法 / `"skip"` / `"search"`
  2. `CacheOrchestrator.check(request_context=...)` 透传正确
  3. `InferenceInterceptor.infer()` 在 `_input_transform` 前剥离保留字段
  4. 非 `client_controlled` config + obs 带字段 → fail loud
- **并发**：每 worker 独立 `WebsocketClientPolicy` + 独立 LIBERO env，env/WS 创建用 `init_lock` 串行化；`num_workers ≤ 5`。
- **State / JSONL 原子写**：沿用 `exp/common/_run_state_base.py` 的锁和原子写风格，不允许多线程裸写同一文件。

---

## 9. G2 代码审查结论（2026-04-16）

**结论：G2 not approved。** 当前代码实现已经覆盖了 G1 批准的 gate / request_context / reserved field 主路径，但 Step 3 runner 仍有两个会影响实验运行或 success 统计可信度的阻塞问题；测试覆盖也未达到 G2 放行要求。

### 9.1 阻塞问题 1：episode key 命名空间错配，runner 会把已有 score 当成缺失

这不是 `data/deviation_experiment/deviate_scores/deviate_score_clip_w7_d4.json` 里"很多没分"。该文件本身有 score entry；问题是当前 runner 用错 key 的命名空间。

当前代码：
- `exp/trajectory_deviation/run_step3_per_cycle_policy.py::_load_failed_eps_for_cfg()` 从 `data/deviation_experiment/cache_eval_results.json` 读取 Step 1a 失败行；
- 然后用 `task_id` + `init_state_idx` 直接拼 `task_X/episode_Y`；
- 但这里的 `init_state_idx` 是 Step 1a 的 **original LIBERO init index**。

而 Step 2 的 `deviate_score_{cfg}.json` key 来自 Step 1b / GT HDF5 的 **subset episode index**。Step 1b 把失败 init 重新压成每 task 的 subset，例如 `.init_map.json` 里：

```text
[1, 2, 4, 5, 7, 8, 11, 12, 13, ...]
```

这里 original init `13` 可能对应 subset episode `8`，所以 Step 2 key 是 `task_0/episode_8`，不是 `task_0/episode_13`。

本地数据复核结果：

```text
clip_w7_d4
  score entries: 159
  failed rows: 163
  overlap if using Step1a init_state_idx as episode: 58
  failed keys missing in score by that interpretation: 105
  score keys not in failed by that interpretation: 101

spatial16_w8_d4
  score entries: 154
  failed rows: 154
  overlap if using Step1a init_state_idx as episode: 53
  failed keys missing in score by that interpretation: 101

max_pool_w3_d5
  score entries: 150
  failed rows: 152
  overlap if using Step1a init_state_idx as episode: 44
  failed keys missing in score by that interpretation: 108
```

因此 `run_step3_per_cycle_policy.py` 现在会在 missing 检查处把大量已有 Step 2 score 误判为缺失并退出。

**修复要求：**

- 推荐：直接以 `deviate_score_{cfg}.json` 的 keys 作为 Step 3 episode 列表。Step 3 的输入本来就是 Step 2 覆盖集，且 Q1b 已定 per-cfg fail set；以 score keys 驱动最简单、最不容易错。
- 若仍要从 Step 1a 失败集重建 episode 列表，则必须读取 Step 1b 的 `*.init_map.json` 或 GT HDF5 attrs，将 original `orig_init_state_idx` 映射到 subset `episode_Y` 后再拼 key。
- 修复后需加一个纯函数测试，固定 original→subset 映射，防止再次把两个 index 空间混用。

### 9.2 阻塞问题 2：success 判定与 `examples/libero/main.py` 不一致

当前 runner 在 `exp/trajectory_deviation/run_step3_per_cycle_policy.py` 的 rollout loop 中只在 `info.get("success", False)` 为真时设置 `success=True`，但 LIBERO 参考 runner `examples/libero/main.py::_run_episode()` 是直接把 `done` 作为 episode success 返回并统计。

风险：
- 如果 env 已经 `done=True`，但 `info` 不稳定提供 `"success"` 字段，当前 Step 3 会停止 rollout，却把该 episode 记录为 `success=False`；
- 同时 `client.episode_end(success=success)` 也会把 server-side lifecycle 标成失败，影响后续审计。

**修复要求：**

- 与 `examples/libero/main.py` 对齐，`done` 应驱动 success；
- 至少改成 `success = success or bool(done) or bool(info.get("success", False))`，并保证 episode 终止时 JSONL 与 `episode_end(success=...)` 使用同一语义；
- 加一个小型 fake-env 单测，覆盖 `done=True` 但 `info` 无 `"success"` 的情况。

### 9.3 测试覆盖缺口

G1 已明确 G2 需要检查最小测试覆盖，但当前代码没有新增/更新对应测试。至少需要覆盖：

- `ClientControlledGate`：`"skip"`、`"search"`、缺失 `request_context`、非法 decision；
- `CacheOrchestrator.check(request_context=...)`：确认 request_context 透传到 gate；
- `InferenceInterceptor.infer()`：确认 `__gate_decision__` 在 `_input_transform` 前被剥离；
- 非 `client_controlled` config 下 obs 带 `__gate_decision__` 时 fail loud；
- Step 3 runner 的 episode key 选择 / original→subset 映射；
- Step 3 runner 的 `done` success 语义。

### 9.4 验证记录

- `python -m py_compile src/openpi/cache/components/gate.py src/openpi/cache/config.py src/openpi/cache/interceptor.py src/openpi/cache/orchestrator.py exp/common/_unit_key.py exp/trajectory_deviation/run_step3_per_cycle_policy.py exp/trajectory_deviation/merge_step3_cfgs.py`：通过。
- `uv run pytest tests/cache/components/test_gate.py tests/cache/test_config.py tests/cache/test_orchestrator.py -q`：未启动。当前 `.venv/bin/pytest` 指向不存在的 Windows/OneDrive Python 路径：`/mnt/c/Users/lzy66/OneDrive - University of Illinois - Urbana/ai-gaming/openpi/.venv/bin/python: not found`。

### 9.5 放行条件

修复 §9.1 和 §9.2，并补齐 §9.3 的最小测试后，再进入下一轮 G2 复审。当前不建议启动全量 Step 3；否则要么 runner 在 episode mismatch 检查处退出，要么 success rate 存在系统性低估风险。

---

## 10. G2 复审结论（2026-04-16）

**结论：G2 approved。** §9 的三个阻塞项已解锁并落地，当前没有继续阻塞 Step 3 smoke / 实跑的代码问题。

### 10.1 §9.1 episode key 命名空间错配

已修复。`exp/trajectory_deviation/run_step3_per_cycle_policy.py` 不再从 `cache_eval_results.json` 重建 episode 列表，也不再混用 Step 1a original init index 和 Step 1b subset episode index。现在 Step 3 直接以 `deviate_score_{cfg}.json` 的 keys 作为权威 episode 集合：

```text
episodes = sorted(deviate_scores.keys())
```

新增/覆盖测试：
- `tests/exp/test_run_step3_per_cycle_policy.py::test_main_uses_deviate_score_keys_as_authoritative_episode_list`
- `tests/exp/test_run_step3_per_cycle_policy.py::test_build_units_cross_product_uses_subset_episode_keys`
- `tests/exp/test_run_step3_per_cycle_policy.py::test_main_rejects_empty_deviate_score_json`

### 10.2 §9.2 success 判定

已修复。`exp/trajectory_deviation/run_step3_per_cycle_policy.py::_run_one_unit()` 不再在 step 循环内部由 `info["success"]` 单独驱动 success，而是在 rollout 结束后按 LIBERO 参考语义统一计算：

```text
success = bool(done) or bool(info.get("success", False))
```

同时预初始化 `info = {}`，覆盖 warmup 阶段已经 `done=True`、主循环不进入的退化路径。

新增/覆盖测试：
- `tests/exp/test_run_step3_per_cycle_policy.py::test_run_one_unit_uses_init_obs_when_no_warmup_and_done_counts_as_success`
- `tests/exp/test_run_step3_per_cycle_policy.py::test_run_one_unit_max_cycles_guard_records_failure_on_timeout`

### 10.3 §9.3 G2 测试覆盖

已满足。runner 新增测试现为 10 个，覆盖 subset-space episode list、空 score JSON、`done` success、`__gate_decision__` per-cycle 注入、burst 不延展、max-cycle timeout、以及 `(tau × n)` 单元构建。

本轮补齐的两个缺口：
- `tests/exp/test_run_step3_per_cycle_policy.py::test_main_rejects_empty_deviate_score_json`
- `tests/exp/test_run_step3_per_cycle_policy.py::test_run_one_unit_injects_one_gate_decision_per_inference_cycle`

cache 侧覆盖已包含：
- `ClientControlledGate` 的 skip / search / 缺失 / 非法值路径；
- `CacheOrchestrator.check(request_context=...)` 透传与 kw-only 约束；
- `InferenceInterceptor.infer()` 在 `_input_transform` 前剥离 `__gate_decision__`；
- 非 `client_controlled` 或无 orchestrator 时带 `__gate_decision__` fail loud；
- client gate 配置但缺信号时 fail loud。

### 10.4 验证记录

通过：

```text
uv run python -m pytest tests/exp/test_run_step3_per_cycle_policy.py -q
10 passed

uv run python -m pytest tests/cache tests/exp -q
511 passed, 13 warnings

python -m py_compile src/openpi/cache/components/gate.py src/openpi/cache/config.py src/openpi/cache/interceptor.py src/openpi/cache/orchestrator.py exp/common/_unit_key.py exp/trajectory_deviation/run_step3_per_cycle_policy.py exp/trajectory_deviation/merge_step3_cfgs.py tests/exp/test_run_step3_per_cycle_policy.py
passed
```

说明：直接 `python -m pytest tests/cache/...` 会因当前 shell Python 缺少 `torch` 失败；项目 venv 路径下使用 `uv run python -m pytest ...` 可正常回归。

### 10.5 非阻塞注意项

- `obs.pop("__gate_decision__", None)` 是有意的原地剥离；WebSocket 路径每次反序列化新 obs，风险可接受。
- `accepts_client_signal` 以实际 `ClientControlledGate` 实例判定；当前 YAML 构造路径满足需求。若未来引入 gate wrapper，再补 wrapper-aware 判定。
- 本结论只放行 Step 3 smoke / 实跑；Step 4/5/6 仍按 §7 另行实现。
