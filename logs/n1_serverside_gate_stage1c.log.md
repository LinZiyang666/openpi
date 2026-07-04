# N1 服务器端门 + G0a Hook 补丁 — 实现计划（Stage 1c）

- **Status**: In Progress（L2，Authority: Execution，G1 APPROVED / §4 Code / G2 R3 APPROVED / §6 Verify scope-green — 2026-07-04）
- **Date**: 2026-07-04
- **Authority**: Execution
- **Level**: L2（新增 src 侧有状态 gate 组件 `ScoreHysteresisGate` + 两条向后兼容的 gate 生命周期加法；单 subsystem=cache，`GateFunction.__call__` 签名不变、interceptor/wire 零改动；见 §2 判定）
- **上游**: `logs/gate_exploration_roadmap.log.md` §5 Stage 1「1c G0a hook 补丁（verdict 回传 + task_key 广播）」；N1 状态机来源 = `logs/n1_live_validation_stage1b.log.md` §4（G2 APPROVED 的 `N1GateState`）
- **Owner 范围裁定（2026-07-04）**: **Scope B = Plumbing + 服务端 N1 门**——两条 hook 落 src 且由新增 `ScoreHysteresisGate` 作为 verdict-回传 hook 的真实运行时消费者（避免 WA §3.1 dead code）；task_key 广播按 roadmap 明示「捎带」一并落地。N2（FollowWinnerGate）仍属 Stage 2，**不在本计划**——只在 verdict 广播 payload 里为其预留 `winner_id/start_t/hit_type` 字段。

---

## 1. 目标与范围

**目标**：把 1a/1b 在 exp 层用「客户端状态机绕经 `__hit_meta__`→`__gate_decision__` 外部环路」验证的 N1 分数滞回门，**服务器化**为一个真正的 src 侧 `GateFunction`。为此补齐 gate 目前拿不到的两条服务端数据通路（G0a）：

1. **verdict 回传→gate**：把每步搜索产出的 verdict（`cp1_score/hit_type/winner_id/start_t`）在 `orchestrator.check()` 内回喂给**本 checkpoint 的 gate**，供其**下一步**决策。
2. **task_key 广播→gate**：episode 起始时把 `_current_task_key` 一并广播给 gate（沿用 judge 已有的 `_safe_call_lifecycle` 按签名过滤机制）。

**范围内**：
- 新增 src 门 `ScoreHysteresisGate`（忠实复刻 1b `N1GateState` 的 decide/observe 语义，θ/j/probe_interval 由 config 注入）。
- orchestrator 两处加法：verdict 回传广播（check() 内）+ task_key 广播（`_broadcast_episode_start` 内）。
- config：`GateConfig` 4 个新字段 + `_GATE_TYPES` + `validate_cache_config` 校验分支 + `_build_gate` 工厂分支。
- 测试 + 文档同步（cache_system.md §5.5 是 §8 注册子系统文档）。

**范围外（明确不做）**：
- N2 FollowWinnerGate / 盲回放（Stage 2a）。
- 任何 interceptor / websocket / wire 协议改动（本计划**零改**这些）。
- exp 层 `n1_gate_client.py` / `run_n1_live.py` 等 1b 产物（**不动**；客户端环路继续可用，与本 src 门并存）。
- 1b live run 本身（本计划只交付服务端**机制**；调好的 θ/j/probe_interval 操作点由 1b live 结果决定后写入各 config YAML，属后续采集配置，不是本 src 改动）。

---

## 2. Level 判定（L2）

**判 L2 依据**：
- 单一 subsystem（cache）内的改动；**不跨 module**（serving/interceptor/client/wire 全不动）。
- `GateFunction.__call__(checkpoint_id, cached_data, request_context)` 协议签名**不变**；新增的是**可选**生命周期加法（`record_verdict` 经 `hasattr` 守卫广播、`on_episode_start(task_key=…)` 经 `inspect.signature` 过滤），**向后兼容**——现有 4 个门（AlwaysSearch/AlwaysSkip/Random/Periodic/ClientControlled）逐字节不受影响。
- 与 roadmap 作者对 1c 的分类（「L2 小改」）一致。

**为何不判 L3**：L3 门槛是「新 checkpoint 层 / 破坏性 interface change / 跨 module 新子系统」。此处无新 checkpoint、无破坏性签名变更、无跨 module。唯一触及 L3 色彩的是**修改 §8 注册文档 `cache_system.md`**——本计划将其视为**强制交付**并按同等严谨度处理（§8），但这不足以把整体升 L3。若 G1 reviewer 认为 gate 生命周期扩展构成 interface change 应升 L3，欢迎在 Review Log 挑战；无论 L2/L3 流程均为 Understand→Plan→G1→Code→G2→Verify（+doc），不涉及跳门。

---

## 3. 已亲验的架构契约（file:line + 行为断言）

**gate（`src/openpi/cache/components/gate.py`）**：
- `gate.py:23-50` `GateFunction` Protocol：`__call__(checkpoint_id, cached_data, request_context=None) -> bool`（True=搜索，False=跳过）。
- `gate.py:53-287` 现有门 `AlwaysSearchGate` / `AlwaysSkipGate` / `RandomGate` / `PeriodicGate` / `ClientControlledGate`，均实现 `on_episode_start(self)`（**无 task_key 参数**）与 `record_action(self, action_chunk)`（多为 no-op）。**无任何门实现 `record_verdict`**。
- `gate.py:242-287` `ClientControlledGate.__call__` 读 `request_context["gate_decision"]`，`"skip"→False/"search"→True`，缺失/非法 raise ValueError。

**orchestrator（`src/openpi/cache/orchestrator.py`）**：
- `orchestrator.py:147` `self._current_task_key: str = ""`；写入点 `on_task_begin(task_key)` `:228-233`、`on_episode_start(task_key,…)` `:235-258`（`if task_key: self._current_task_key = task_key`, `:251-252`）。
- `orchestrator.py:260-305` `_broadcast_episode_start()`：`:286-287` 对每个 gate 调 `self._safe_call_lifecycle(gate, "on_episode_start")` —— **不传任何 kwarg**；对比 `:288-292` 对 judge 传 `extra_metadata=self._current_episode_extra`。
- `orchestrator.py:319-331` `_safe_call_lifecycle(component, method_name, **kwargs)`：用 `inspect.signature` 过滤 kwargs，只传 callee 实际接受的键 → 老签名安全忽略新 kwarg。**这是 task_key 广播的既有承载机制**。
- `orchestrator.py:351-378` `broadcast_action()`：`:360-362` `for gate…: if hasattr(gate,'record_action'): gate.record_action(action_chunk)` —— **只广播 action，无 verdict 通道**。
- `orchestrator.py:384-535` `check(checkpoint_id, *, request_context=None, **stage_outputs) -> CheckResult`：
  - `:416` `gate = self._gates[checkpoint_id]`；`:423-426` `should_search = gate(checkpoint_id, self._key_builder.cached_data, request_context)`（gate **早于** judge）。
  - `:445-455` gate-skip 分支 → `return CheckResult(hit_type=HitType.MISS, query_keys=query_keys, searched=False)`（**skip 步没有 search verdict**）。
  - `:477-490` `judge_result = judge(...)` → `hit_type/winner_id/start_t = judge_result.*`；`top_score = results[0].score if results else None`（**`cp1_score == results[0].score`**，与 `_build_hit_meta` 一致）。
  - `:498-529` FULL_HIT/WARM_START：`:506-523` WARM_START payload 不完整则降级 `return CheckResult(hit_type=HitType.MISS, …, score=results[0].score, …)`（`:519-523`）；否则 `:525-529` `return CheckResult(hit_type=hit_type, …, score=results[0].score, …)`。
  - `:531-535` 判后 MISS：`return CheckResult(hit_type=HitType.MISS, …, score=top_score, …)`。
  - `HitType` 已在 `:40` import（`from openpi.cache.components.judge import HitType, SimilarityJudge`）。
- **check() 的调用方**：`interceptor.py` CP1 `:704-708`、CP3 `:846-850`——只透传 `request_context`。verdict 由 `_build_hit_meta`(interceptor.py:481-513) 从 `CheckResult.{hit_type,start_t,entry_id,score}` 组 `__hit_meta__`（`:734/:889`）发客户端。**本计划的 verdict 回传发生在 orchestrator 内部、不经 interceptor**，故 interceptor 零改。

**config（`src/openpi/cache/config.py`）**：
- `config.py:81-89` `GateConfig`：字段 `type/p_inference/seed/cache_len/inference_len`（默认 `type="always_search"`，其余 None）。
- `config.py:459` `_GATE_TYPES = frozenset({"always_search","always_skip","client_controlled","random","periodic"})`。
- `config.py:1227-1315` `validate_cache_config` 门校验：`:1237-1239` `_gate_random_fields/_gate_periodic_fields/_gate_all_param_fields`；`:1240-1242` `gate_set_fields = {name for name in _gate_all_param_fields if getattr(cp_config.gate,name) is not None}`；`:1251-1279` random 分支（含 `stray = gate_set_fields - _gate_random_fields` 报错）；`:1280-1306` periodic 分支；`:1308-1315` legacy 3 类 `if gate_set_fields: error "cannot set …"`。
- `config.py:1873-1888` `gates[cp_id] = _build_gate(cp_config.gate)`。
- `config.py:2132-2158` `_build_gate(cfg)`：按 `cfg.type` if/elif 实例化，尾部 `raise ValueError(f"Unknown gate.type …")`。

**1b `N1GateState` 语义（来源，`logs/n1_live_validation_stage1b.log.md` §4，将忠实复刻）**：
- 字段 `searching:bool, low_run:int, since_probe:int`；参数 `theta_low, theta_high, j, M`。
- `decide()`：`searching`→"search"；elif `M is not None and since_probe+1 >= M`→"search"（probe）；else "skip"。
- `observe(decision, score)`：`score is None`→按 `-inf`。searching 分支：`score<θ_low`→`low_run+=1`，`low_run>=j`→`searching=False, since_probe=0`，else `low_run=0`。skipping 分支：`decision=="search"`（probe）→`since_probe=0`，`score>=θ_high`→`searching=True, low_run=0`；`decision=="skip"`→`since_probe+=1`。
- `reset()`：`searching=True, low_run=0, since_probe=0`；首步 `decide()` 恒 "search"。

---

## 4. 设计

### 4.1 文件改动表

| 文件 | 动作 | 责任 |
|---|---|---|
| `src/openpi/cache/components/gate.py` | 改 | 新增 `ScoreHysteresisGate`（服务端 N1 门）；`GateFunction` Protocol docstring 增补可选 `record_verdict` / `on_episode_start(task_key=…)` 说明（**签名不加必填参数**）。 |
| `src/openpi/cache/orchestrator.py` | 改 | (a) `_broadcast_episode_start` gate 循环传 `task_key=self._current_task_key`；(b) `check()` 4 个 return 前经新私有 helper `_feed_verdict_to_gate` 回喂 verdict。 |
| `src/openpi/cache/config.py` | 改 | `GateConfig` +4 字段；`_GATE_TYPES` + `"score_hysteresis"`；`validate_cache_config` + score_hysteresis 校验分支（并入 `_gate_all_param_fields`）；`_build_gate` + score_hysteresis 分支。 |
| `docs/architecture/cache_system.md` | 改 | §5.5 GateFunction 增 `ScoreHysteresisGate` + 两条生命周期 hook 描述（§8 注册文档）。 |
| `docs/cache/tutorial.md` | 改 | Gate 组件段增 `score_hysteresis` YAML 配置 + 参数说明。 |
| `docs/README.md` | 改 | cache_system.md 行摘要同步（index sync 红线）。 |
| `logs/README.md` | 改 | 本 plan 条目（已在创建时加，G2 后随状态更新）。 |
| `tests/cache/components/test_gate.py` | 改 | `ScoreHysteresisGate` 单测（构造校验 + decide/record_verdict golden + 边界）。 |
| `tests/cache/test_orchestrator.py` | 改 | task_key 广播 + verdict 回喂 wiring 测 + 向后兼容测。 |
| `tests/cache/test_config.py` | 改 | score_hysteresis 校验 + `_build_gate` 测。 |

### 4.2 `ScoreHysteresisGate`（新增，`gate.py`）

**接口**：
```python
class ScoreHysteresisGate:
    """Server-side N1 score-hysteresis gate (serverizes exp-layer N1GateState).

    decide (in __call__) and observe (in record_verdict) are split across two
    orchestrator calls but together reproduce N1GateState exactly:
      - __call__ is PURE (no state mutation) -> decision only.
      - record_verdict applies the score-driven transition, branching on the
        current searching/skipping state (unchanged since __call__).
    The orchestrator feeds record_verdict on EVERY per-checkpoint check() path
    (searched steps carry the real cp1_score; skip steps carry cp1_score=None),
    so the decide/observe pairing matches the client loop step-for-step.
    """
    def __init__(self, theta_low, theta_high, j, probe_interval): ...
        # construct-time fail-fast:
        #   reject bool for all four params (TypeError; bool is an int subclass
        #   -> mirror RandomGate/PeriodicGate gate.py:143/152/204/209);
        #   ValueError: theta_low/theta_high finite float, theta_high >= theta_low;
        #   j int >= 1; probe_interval None or int >= 1.
        # state: _searching=True, _low_run=0, _since_probe=0, _task_key=""

    def __call__(self, checkpoint_id, cached_data, request_context=None) -> bool:
        # PURE decide (no mutation):
        #   if _searching: search
        #   elif probe_interval is not None and _since_probe + 1 >= probe_interval: search (probe)
        #   else: skip
        # return decision == "search"

    def record_verdict(self, checkpoint_id, *, hit_type, cp1_score, winner_id,
                       start_t, searched) -> None:
        # observe(decision=("search" if searched else "skip"), score=cp1_score)
        #   score = cp1_score if finite else -inf   (None -> -inf: legit empty-search MISS)
        #   NON-finite (contract-violation, shouldn't occur server-side):
        #       fail-open -> _searching=True,_low_run=0,_since_probe=0; logging.warning once.
        #   if _searching: score<θ_low -> _low_run+=1; if _low_run>=j: _searching=False,_since_probe=0; else _low_run=0
        #   else (skipping):
        #       if searched (probe): _since_probe=0; if score>=θ_high: _searching=True,_low_run=0
        #       else (skip):         _since_probe+=1

    def on_episode_start(self, task_key: str = "") -> None:
        # reset state (_searching=True,_low_run=0,_since_probe=0); _task_key=task_key
        # task_key used in this gate's logging.debug context (real consumer of
        # the G0a task_key broadcast; forward-enabler for per-task params).

    def record_action(self, action_chunk) -> None:
        ...  # no-op (protocol completeness; N1 does not use action history)
```

**与 `N1GateState` 的等价性保证**：`__call__`==`decide()`（纯），`record_verdict(searched, cp1_score)`==`observe(decision, score)`（`searched⟺decision=="search"`，分支读 `_searching`，与 client 完全一致）。golden traces（§7）以 1b 同款序列断言逐步等价，确保 live 结论可直接迁移。

**`__call__` 纯性红线**：`__call__` 绝不改状态（否则同步一次 decide/一次 observe 的配对被打破）。单测断言「连续调 `__call__` 两次而不 record_verdict → 返回值相同、状态不变」。

### 4.3 orchestrator 两处加法

**(a) task_key 广播**（`_broadcast_episode_start`，`orchestrator.py:286-287`）：
```python
for gate in self._gates.values():
    self._safe_call_lifecycle(gate, "on_episode_start", task_key=self._current_task_key)
```
`_safe_call_lifecycle` 按签名过滤：现有门 `on_episode_start(self)` → task_key 被剔除、行为不变；`ScoreHysteresisGate.on_episode_start(self, task_key="")` → 收到。与 judge 收 `extra_metadata` 同机制，向后兼容。

**(b) verdict 回喂**（`check()` 内）：新增私有 helper
```python
def _feed_verdict_to_gate(self, checkpoint_id, *, hit_type, cp1_score,
                          winner_id, start_t, searched) -> None:
    gate = self._gates.get(checkpoint_id)
    if gate is not None and hasattr(gate, "record_verdict"):
        gate.record_verdict(checkpoint_id, hit_type=hit_type, cp1_score=cp1_score,
                            winner_id=winner_id, start_t=start_t, searched=searched)
```
在 check() 的 **4 个 return 前**各调一次（per-checkpoint，只喂 `self._gates[checkpoint_id]`；`hasattr` 守卫使无 `record_verdict` 的门不受影响）：

| 位置 | searched | hit_type | cp1_score |
|---|---|---|---|
| gate-skip return（`:455` 前） | `False` | `HitType.MISS` | `None` |
| WARM_START 降级 return（`:519-523` 前） | `True` | `HitType.MISS` | `results[0].score` |
| FULL_HIT/WARM_START return（`:525-529` 前） | `True` | `hit_type`（最终） | `results[0].score` |
| 判后 MISS return（`:531-535` 前） | `True` | `HitType.MISS` | `top_score` |

`winner_id/start_t` 在每处取当前作用域值（skip 处为 None/None）。广播**最终** hit_type（含降级）以便 N2 后续消费。「checkpoint 未配置」早返（`:411-414`）无对应门，不广播。

### 4.4 config 三处加法

- `GateConfig`（`:81-89`）+：`theta_low: float | None = None`、`theta_high: float | None = None`、`j: int | None = None`、`probe_interval: int | None = None`（注释标「Only for type='score_hysteresis'」）。
- `_GATE_TYPES`（`:459`）+ `"score_hysteresis"`。
- `validate_cache_config`（`:1237-1315`）：
  - `_gate_score_hysteresis_fields = {"theta_low","theta_high","j","probe_interval"}`；并入 `_gate_all_param_fields`。
  - 新增 `elif cp_config.gate.type == "score_hysteresis":` 分支：要求 `theta_low/theta_high` 为有限实数且 `theta_high >= theta_low`；`j` int≥1；`probe_interval` 为 None 或 int≥1（**None=停搜期不 probe / 永久停搜**，合法，对应 1b `M=None` 与 roadmap F6 `K=∞` 对照）；`stray = gate_set_fields - _gate_score_hysteresis_fields` → 报错。random/periodic/legacy 设 theta_* 亦被各自 stray 检查/legacy 检查拦截（`_gate_all_param_fields` 已含新字段）。
- `_build_gate`（`:2132-2158`）：新增
  ```python
  elif cfg.type == "score_hysteresis":
      from openpi.cache.components.gate import ScoreHysteresisGate
      return ScoreHysteresisGate(theta_low=cfg.theta_low, theta_high=cfg.theta_high,
                                 j=cfg.j, probe_interval=cfg.probe_interval)
  ```

---

## 5. 集成点与不变量

- **per-checkpoint 隔离**：verdict 只喂 `self._gates[checkpoint_id]`；CP3 若为 AlwaysSearchGate（无 `record_verdict`）→ `hasattr` 守卫跳过，零影响。N1 典型只配 CP1。
- **`accepts_client_signal` 不受影响**：`ScoreHysteresisGate` 不是 `ClientControlledGate` → `orchestrator.py:140-142` 该 flag 仍 False；本门不读 `request_context`（server 自持 verdict），也不要求客户端注入 `__gate_decision__`。
- **skip 步语义不变**：gate-skip 仍返回 `CheckResult(searched=False)`（`:455`），下游 interceptor/collector 的 C5 selection-bias 语义不变；本计划只在返回**前**多喂一次 gate 内部状态。
- **verdict 回喂 = 纯 gate 本地状态**：无 storage 锁、无 wire、per-connection 单线程（每连接独立 orchestrator/gate，见 `build_per_connection_components`）→ 无并发问题。
- **`_current_task_key` 时序**：`on_episode_start` 先 set task_key（`:251-252`）再 `_broadcast_episode_start`（`:258`）→ 广播时 task_key 已就位。

---

## 6. 反事实/正确性口径

- 服务端门与 1b 客户端门**行为等价**（§4.2 保证）→ 1b live 验证结论（SR/skip%/inf_ratio）对服务端门成立，无需重测机制正确性；服务器化只改「决策在哪算」不改「决策是什么」。
- 调好的 θ/j/probe_interval 操作点仍需 1b live 结果确定后写入 YAML —— 本计划交付**机制**，不预设参数值（各测试用合成 θ）。

---

## 7. 测试策略（全部非 manual：无 GPU/server）

**`tests/cache/components/test_gate.py`（`ScoreHysteresisGate` 单测）**：
- **构造校验**：theta 非有限 / `theta_high<theta_low` / `j<1` / `probe_interval=0` → raise ValueError；合法组合 → 成功。
- **bool-as-number 拒绝**：`theta_low/theta_high/j/probe_interval` 传 `True/False` → raise TypeError（对齐 `RandomGate`/`PeriodicGate` 显式拒 `bool` 混入 int，见 `gate.py:143/152/204/209`），防 YAML/CLI 误配被 Python 数值塔静默接受。config validator 与 gate 构造须给**一致**的 bool 拒绝诊断（同一误配不同入口一致）。
- **`__call__` 纯性**：连续两次 `__call__`（不 record_verdict）返回相同、状态字段不变。
- **decide/record_verdict golden traces**（复刻 1b 序列，附测内 mini-reference 重算，**不** import exp）：连续 j 个 `<θ_low` → 进入 skip；停搜期每 probe_interval 步 probe（`__call__` 返 search）；probe `score>=θ_high` → 恢复；滞回带 `θ_low<θ_high` 防抖；`on_episode_start` → reset（首步恒 search）。
- **None-as-MISS**：searched 步 `cp1_score=None`（空搜索合法 MISS）→ 按 `-inf` 计入 low_run / 不恢复，与真低分同。
- **非有限 fail-open**：searched 步 `cp1_score=NaN/inf`（契约违反）→ 强制恢复全搜索（`_searching=True` 等），下一步 `__call__` 返 search，不 raise。
- **task_key**：`on_episode_start("taskX")` → `_task_key=="taskX"`；跨 episode reset 不泄漏。
- **回归**：AlwaysSearch/AlwaysSkip/Random/Periodic/ClientControlled 现有断言不变（本文件既有）。

**`tests/cache/test_orchestrator.py`（wiring）**：
- **task_key 广播**：spy 门实现 `on_episode_start(self, task_key="")` 记录收到值 → `on_episode_start(task_key="T")` / `on_task_begin("T")` 后断言收到 "T"；对照普通 spy 门 `on_episode_start(self)` 不报错（signature 过滤）。
- **verdict 回喂**：spy 门实现 `record_verdict(...)` 记录参数 →
  - 走 searched FULL_HIT/MISS 的 check() → 断言 `record_verdict(searched=True, cp1_score==results[0].score, hit_type 正确, winner_id/start_t 正确)`；
  - spy 门返回 skip 的 check() → 断言 `record_verdict(searched=False, cp1_score is None)`；
  - WARM_START 降级路径 → 断言 `hit_type==MISS` 且 `cp1_score==results[0].score`。
- **向后兼容**：门无 `record_verdict`（如 AlwaysSearchGate）→ check() 不崩、不调用。
- **端到端小回路**：用合成 storage/strategy/judge + `ScoreHysteresisGate`，喂一段 verdict 序列跑多次 check()，断言 skip/search 模式与 N1 预期一致（决策在 server 侧闭环，无需客户端）。

**`tests/cache/test_config.py`（config）**：
- score_hysteresis 合法 config → `validate_cache_config` 通过、`_build_gate` 返回 `ScoreHysteresisGate` 且参数正确。
- 缺 `theta_low/theta_high/j` → 校验报错；`probe_interval` 省略（None）合法。
- stray 字段（score_hysteresis 设 `p_inference`；random/periodic/legacy 设 `theta_low`）→ 报错。

**staged API / 推理路径回归（既有，须全绿）**：`tests/cache/test_interceptor.py` + `tests/cache/test_interceptor_hit_meta.py`（`InferenceInterceptor.infer()` staged `run_stage1/2/3` 路径 + `__hit_meta__` 组装）+ `tests/serving/test_websocket_policy_server.py` + `tests/serving/test_websocket_response_hit_meta.py`（wire 透传）——check() 改动对现有门的 staged 推理路径 + wire 逐字节不变。

**回归**：`uv run pytest`（§6 Verify）全绿；现有 gate/orchestrator/config/interceptor/serving 单测不受影响（新增均为 additive + 守卫）。

---

## 8. 风险登记

| 风险 | 缓解 |
|---|---|
| `__call__` 意外 mutate → decide/observe 配对错位 | 设计红线：`__call__` 纯读；单测断言双调 `__call__` 幂等 |
| check() 在 gate() 与 verdict 广播之间抛异常 → gate 状态漏推进 | 该步 inference 本身已失败（异常上抛，episode 处理）；verdict 广播在各 return 前、search/judge 之后，正常路径必达；低风险，docstring 注明 |
| verdict 广播漏某个 return 分支 → 状态漂移 | §4.3 枚举全 4 个 return + 「checkpoint 未配置」早返（无门）；orchestrator wiring 测覆盖 4 路 |
| 与 client `N1GateState` 行为不等价 → live 结论不可迁移 | golden traces 复刻 1b 同款序列逐步断言；等价性为单测硬门 |
| 现有门被新 kwarg/新广播波及 | `_safe_call_lifecycle` 签名过滤 + `hasattr(record_verdict)` 守卫；显式回归测老门 |
| `probe_interval=None` 永久停搜丢恢复段（roadmap F3） | 合法但危险语义，docstring/tutorial 明确标注；操作点默认配有限 probe_interval（1b A/B 点均含 M） |
| CP3 门误收 verdict | per-checkpoint 只喂 `self._gates[checkpoint_id]`；CP3 门无 record_verdict → 守卫跳过 |
| §8 注册文档 / 索引漏同步 | §4.1 明列 cache_system.md + tutorial + docs/README + logs/README；WA §4 index sync 红线，同 commit |
| 服务器化早于 1b live 结果 | 只服务器化**机制**（θ/j/probe_interval 走 config，由 1b live 结果后填）；Owner 已裁定 Scope B |

---

## 9. 交付与验收

- **Code 交付物**：§4.1 全部 diff + plan-conformance 声明（「完全遵循批准计划」或列偏差 + 用户同意点）+ §4 本地 `uv run pytest`（advisory）输出。
- **文档同步**（同 commit）：cache_system.md §5.5、tutorial.md gate 段、docs/README.md（cache_system 行）、logs/README.md（本 plan 行）。
- **Verify（§6，含 staged API / 推理路径回归）**：本计划修改 `orchestrator.check()`，它位于 `InferenceInterceptor.infer()` 调用链内、**属 inference path**（WA §2.7）；「无 wire/interceptor 改动」只说明 **wire 契约**不变，**不豁免** staged API 验证。Verify 清单（G2 advisory 亦须列 1–3 结果）：
  1. **新增测**：`tests/cache/components/test_gate.py`（`ScoreHysteresisGate`）+ `tests/cache/test_orchestrator.py`（task_key / verdict wiring）+ `tests/cache/test_config.py`（score_hysteresis 校验 + `_build_gate`）。
  2. **staged API / 推理路径回归（既有，须全绿）**：`tests/cache/test_interceptor.py` + `tests/cache/test_interceptor_hit_meta.py`（驱动 `run_stage1/2/3` 的 `InferenceInterceptor.infer()` staged 路径 + `__hit_meta__` 组装）+ `tests/serving/test_websocket_policy_server.py` + `tests/serving/test_websocket_response_hit_meta.py`（websocket infer 透传 + `__hit_meta__` wire）——证明 check() 改动对现有门的 staged 推理路径与 wire **逐字节不变**。
  3. **全量**：`uv run pytest`（全部非 manual）全绿 + `ruff check` clean + 一行 pass/fail 摘要。
  4. **GPU-manual staged 测的边界说明（非豁免）**：本改动不碰任何 tensor 数学 / staged 模型调用（`run_stage*` 未改），只改 gate 决策的 Python 状态；故 `@pytest.mark.manual` 的 pkl/llm-parity/stage-device 测不构成本改动的回归面。若 Verify 期发现实际触及，须补跑对应 manual 测。
- **验收线**：所有新增测 + 全量回归绿；`ruff check` clean；现有 5 个门与 interceptor/serving 逐字节行为不变（回归证明）。

---

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-07-04 11:09 CDT

- [Blocking] [Concern] `record_verdict` 的 tracked wiring 测未覆盖批准计划要求的 WARM_START payload 不完整降级 return path。— reasoning: 计划 §4.3 明确要求 check() 的 4 个 return 前均回喂 verdict，§7 进一步要求 orchestrator wiring 测覆盖「WARM_START 降级路径 → 断言 `hit_type==MISS` 且 `cp1_score==results[0].score`」。当前 `tests/cache/test_orchestrator.py` 的 G0a spy 测只覆盖 FULL_HIT、empty-store post-judge MISS、gate-skip、legacy no-hook；文件中既有 WARM_START downgrade 测只断言 orchestrator 返回/ miss counter，没有让带 `record_verdict` 的 spy gate 观察该路径。Reviewer 独立探针证明实现当前会正确回喂最终 `MISS` verdict，但该行为未进入项目测试，后续回归会漏。请新增 tracked regression（例如 `test_verdict_fed_to_gate_on_warm_start_downgrade`），断言 searched=True、hit_type=MISS、cp1_score=top score、winner_id/start_t 从 judge 结果传入。
- [Verification] Ruff 通过；计划内定向集合 `tests/cache/components/test_gate.py tests/cache/test_orchestrator.py tests/cache/test_config.py tests/cache/test_interceptor.py tests/cache/test_interceptor_hit_meta.py tests/serving/test_websocket_policy_server.py tests/serving/test_websocket_response_hit_meta.py` 为 250 passed；Reviewer 独立探针 2 passed（覆盖 WARM_START downgrade 和额外 bool 边界）。全量 pytest 曾误启动，触发 GPU/JAX/GCS/network/env-dependent 失败并被中断；按本次审查范围和 Owner 指示，该误跑不作为 G2 gating 依据。

### G2 Round 2 — Executor — 2026-07-04

1 项 blocking Accepted，代码已修（ruff clean，定向集 13 passed 含新增测）。

- Accepted（G2R1 WARM_START downgrade verdict 回喂 tracked 测）—— 缺口属实：§4.3 site 2（WARM_START payload 不完整降级 return）虽已由 `_feed_verdict_to_gate` 回喂，但 `tests/cache/test_orchestrator.py` 的 G0a spy 测只覆盖 FULL_HIT / empty-store post-judge MISS / gate-skip / legacy-no-hook，未让带 `record_verdict` 的 spy 观察降级路径。新增 tracked `test_verdict_fed_to_gate_on_warm_start_downgrade`：`AlwaysWarmStartJudge(start_t=0.5)` + 无 intermediates payload 触发降级，断言 spy 收到 `searched=True`、`hit_type==MISS`（最终）、`cp1_score==results[0].score`（≈1.0 top score）、`winner_id==entry.id`、`start_t==0.5`（均从 judge 结果作用域传入，与实现一致，符合 reviewer 期望）。实现未改（site 2 行为本就正确），仅补测。定向 `-k "verdict or task_key or hysteresis or warm_start or record_verdict"` → 13 passed。

重入 G2。

### G2 Round 3 — Reviewer — APPROVED — 2026-07-04 11:25 CDT

- [Resolved] G2R1 blocking 已闭环：`tests/cache/test_orchestrator.py` 新增 tracked `test_verdict_fed_to_gate_on_warm_start_downgrade`，用 `AlwaysWarmStartJudge(start_t=0.5)` + 无 `intermediates` payload 触发 WARM_START→MISS 降级，并断言 gate 收到最终 `MISS`、`searched=True`、`cp1_score≈1.0`、`winner_id==entry.id`、`start_t==0.5`。这覆盖了批准计划 §4.3/§7 要求的第 4 条 return-path verdict 回喂。
- [Reviewed] 本轮工作区另含 Owner 指示的 pytest harness 变更：根 `conftest.py` 增加 `--run-manual`，裸 pytest 默认 skip `manual/env_dependent`；CI workflow 改为 `--run-manual -m "not manual"`，保持原 CI 排除 manual、继续覆盖 env-dependent 的语义；`src/openpi/models/tokenizer_test.py` 被标记 `env_dependent`，避免裸 pytest 构造 tokenizer 时下载 GCS/HuggingFace 资产。该变更不触及 Stage 1c runtime 路径，按测试 harness 范围审查无 blocker。
- [Verification] Python ruff 通过；`.github/workflows/test.yml` YAML parse 通过；`git diff --check` 通过；Stage 1c 定向集合 `tests/cache/components/test_gate.py tests/cache/test_orchestrator.py tests/cache/test_config.py tests/cache/test_interceptor.py tests/cache/test_interceptor_hit_meta.py tests/serving/test_websocket_policy_server.py tests/serving/test_websocket_response_hit_meta.py` 为 251 passed；Reviewer 独立探针 2 passed；`src/openpi/models/tokenizer_test.py` 裸跑为 2 skipped，验证 `env_dependent` 默认 skip；`tests/cache/components/test_gate.py --run-manual` 为 56 passed，验证新 flag 被 pytest 接受且不影响普通测试。按 Owner 指示，本轮未跑全量 pytest、GPU/manual 或会加载模型的测试。
