# Stage 4a — N2 追随赢家门（FollowWinnerGate / lockstep 盲回放）实现计划

- **Status**: DONE（G1/G2 APPROVED 2026-07-06；Phase A F5 GO；§4 Code + §6 Verify green；**Phase C live 完成 2026-07-07**）。Live verdict：N2 6 点中 **1 点 PASS**（libero_10 budget=3：SR +1.2 vs baseline、vs matched periodic c30 +1.2、net@34 +8.2）；spatial 3 点全 FAIL（同 inf periodic SR 90.4 >> N2 84-85）；l10 b5/b8 FAIL（budget↑ 盲回放漂移，SR 退化）。net@34 全 6 点正（延迟稳健价值）。结论：N2 非广谱优、非全败——延迟处处正，SR 仅「低剂量 × inf-matched periodic 恰近-baseline」窄区间竞争（复现 N1 + F12 甜点 + R2 漂移）。报告 → `exp/gate_research/analysis/stage4a_n2_live.md`
- **Authority**: Execution
- **Level**: L3（跨模块：新增 orchestrator 执行原语 + gate→orchestrator 契约扩展 + config + 架构文档更新）
- **Roadmap**: `logs/gate_exploration_roadmap.log.md` §4 N2 条 / §5 Stage 4 表（4a）
- **前置阶段**: Stage 3a（N4 client 原型 + live，`251eddc`）✅ / Stage 3b（N4 服务器化 `ScoreHysteresisGate`+L，`81656e4`）✅
- **Date**: 2026-07-06

---

## 1. 背景与进入条件对账

### 1.1 N2 是什么，与 N1/N4 的根本差异

N1（滞回门）/ N4（N1 ⊕ V2 注入）的 "skip" = **免搜索但仍全推理**（`inf=1.0`）——省的是 MISS 段的 search（省量上限 = MISS% 19–38%）。

N2 追随赢家门的 "盲回放" = **既不搜索也不推理，直接回放锁定的 winner 库 episode 的后续缓存动作**——省的是命中段的 search+judge+推理（60–80% 的步）。这是 V1 省量上限的翻倍，且在 stock（34ms）/大库（70ms）延迟档意义放大。

**机制**（roadmap §4 N2）：连续 j 步 FULL_HIT 且同一 winner 轨迹、`winner_step` 逐步 +1（lockstep 证据）→ 锁定该库 episode → 接下来 M 步不搜索、直接回放 winner 的后续动作；预算 M 用完或解锁 probe 掉带 → 解锁回 always_search。M 是盲回放债务预算（B3 机制的正确用武之地：此段 verdict 不在场）。

### 1.2 进入条件对账（roadmap §5 Stage 4 表 4a 行，逐条如实处理）

| 进入条件 | 状态 | 本计划的处理 |
|---|---|---|
| ① stock/大库延迟为硬约束 **且** N4 落地后 hit 段搜索仍为主要成本 | **owner 部署上下文判断，非代码可验** | owner 启动 4a ⇒ 认定目标部署档为 stock/大库/远程（优化小库档 N2 无 V1 价值，roadmap C9/F6）。本计划据此推进；若目标档确为优化小库，应在 G1 驳回。**如实标注，交 G1/owner 判定** |
| ② 重启前重测 F5 lockstep（注入改变 hit 段结构） | **可纯离线做（0 GPU）** | **Phase A**：在已本地化的 Stage 3a N4 live per_step 数据上离线重算 F5（persistence + Δwinner_step）。作为 Phase B 的 **gating 前置**——F5 若在 N4 注入档下崩，N2 结构前提不成立，停止不建 L3 |
| ③ build/D2H 省取问题一并押后 | **本计划恪守** | N2 盲回放只省 Stage 2+3（search+judge+denoise），**保留 Stage 1 vision + `build()`**（interceptor.py:696 每步先跑 Stage 1；orchestrator.py:466 `build()` 无条件跑保轨迹）。省 build/D2H 是后续更大改动，不在本计划范围 |

### 1.3 为什么 N2 必须一开始就服务器端实现（方法论与 N1/N4 不同）

N1（Stage 1a client 原型 → 1b live → 1c 服务器化）和 N4（Stage 3a client 原型 → 3b 服务器化）都能先在 exp 层用 `ClientControlledGate` 原型化：客户端注入 `__gate_decision__: search/skip`，server 照单执行。

**N2 装不进这条路**（Explore agent 已核实）：`ClientControlledGate` 的 "skip" 必产全推理（orchestrator.py:495 skip → `HitType.MISS` → interceptor.py:758+ 全推理），且 wire 上**没有** client 可控的 "replay winner step+k 动作" 信号。要盲回放，要么加新 wire verdict，要么服务器端实现。因此本计划**直接服务器端落地**（无 exp 层 client 原型步）。

### 1.4 L3 判定依据（架构约束）

现 gate→orchestrator 契约只有二态：`gate.__call__ → bool`（True=搜 / False=跳）。**skip→MISS→全推理；FULL_HIT→回放，但回放焊死在"每步 search+judge"之后**（orchestrator.py:497-544）。系统里**不存在**"不搜也不推、回放指定 winner 后续动作"的执行原语。N2 需要：

1. **gate 契约扩展**（新增第三态表达："锁定回放 winner X 的后续步"）；
2. **orchestrator `check()` 新增盲回放执行分支**（跨模块）；
3. **持久化 per-connection winner-lock 游标**（现 orchestrator 每步 `check()` 独立、不留 winner 句柄）。

跨模块 + 新执行原语 + 接口变更 = **L3**。

---

## 2. Phase A — F5 lockstep 离线复测（gating 前置，L1 离线分析）

### 2.1 目标

回答进入条件②：**N4 注入档下，命中段是否仍是 lockstep 回放**（N2 盲回放的结构前提是否还在）。这是**结构机会**的判据（C8/C11：N2 的 SR 效应离线不可评、必 live；但"lockstep 机会是否被注入摧毁"可离线判）。

### 2.2 数据来源（已本地化，0 GPU）

- Stage 3a N4 live per_step 数据（L=6 赢点 + L=8/L=12），每步含 `winner_id`（`examples/libero/episode_runner.py:73`，格式 `f"{trajectory_id}:{step_idx}"`，orchestrator.py:677）、`hit_type`、`cp1_score`。
- 对照基线：Stage 0 always-search 数据（原 F5 测量所依，roadmap 附录 A：persistence 93–98% / Δ+1 75–97%）。

> **数据可用性校验**（Phase A 第一步）：Stage 3a raw 数据本地化已 sha256 校验（见 `exp/gate_research/analysis/stage3_n4_live.md` Artifact layout）。Phase A 开始前先确认 per_step JSONL 在本地可读、含 `winner_id` 字段非全 None；若缺，则 Phase A 退化为需一次带 instrumentation 的补采（升级为需 GPU，另行请示 owner）。

### 2.3 度量（复用现成逻辑）

复用 `exp/gate_research/gate_structure_analysis.py:168-186` 的 winner persistence 逻辑（相邻 FULL_HIT 对：same-episode% + Δwinner_step 分布），**在 N4 注入档数据上重跑**：

- **within-FH-run persistence**：注入把长命中段切成子段；度量子段内相邻 FH 对的 same-episode 占比。
- **Δwinner_step 分布**：+1 是否仍占主导（majority）。
- 分 L（6/8/12）× 分 suite（spatial / libero_10）报告。

产物：`exp/gate_research/analysis/stage4a_f5_recheck.md`（纯 .md 报告，遵 `feedback_analysis_md_location`）。逐轮工作日志留本 `.log.md`。

### 2.4 Go/No-Go 判据（对齐 per-suite Stage-0 F5 基线）

判据在 Phase A 报告里对每个 suite 对齐其 Stage-0 F5 基线（spatial / libero_10 各自的 persistence 93–98% / Δ+1 75–97% 区间）后判定：

- **GO**（注入未实质侵蚀 lockstep，进入 Phase B）：N4 档 within-run persistence 与该 suite 原 F5 差距 ≤ 10pp，**且** Δ+1 仍占多数。
- **NO-GO / 重议**（结构前提被削弱）：persistence < 85%，**或** Δ+1 失去多数地位 → N2 盲回放漂移风险显著上调，停止不建 L3，回报 owner。

> 阈值（10pp / 85%）为操作判据；Phase A 报告须同时列出实测 run-length 分布与 per-suite 基线，供 owner 复核判定边界情形。

---

## 3. Phase B — 服务器端 FollowWinnerGate + orchestrator 盲回放执行路径（L3 核心）

> **仅在 Phase A 判定 GO 后开工。** Phase A NO-GO ⇒ 本 Phase 不执行。

### 3.1 契约扩展设计（最小侵入，沿用既有 additive-hook 惯例）

**方案 A（采纳）**：给 `GateFunction` Protocol 增一个**可选** hook `replay_target(self) -> str | None`，orchestrator 用 `hasattr` 守卫查询。这与现有 `on_episode_start` / `record_action` / `record_verdict`（G0a 就是这么加的可选 hook）**同构**，`__call__` 仍返回 `bool`，全部 legacy gate 零影响、向后兼容。

- gate 锁定期间：`__call__` 返回 `False`（不搜），**且** `replay_target()` 返回当前锁定游标 `entry_id`（非 None）。
- 非锁定 / 其它 gate：`replay_target()` 返回 `None`（或无此方法）→ orchestrator 走原 skip→MISS→推理路径。

**方案 B（驳回）**：新增 `HitType`/`CheckResult` 变体 + 新 client wire verdict。仅当要做 exp 层 client 原型才需要；但 §1.3 已证 N2 必服务器端，方案 A 严格更优。**驳回，记录在案。**

### 3.2 `src/openpi/cache/components/gate.py` — 新增 `FollowWinnerGate`

decide/observe 分裂（与 `ScoreHysteresisGate` 同范式：`__call__` PURE、`record_verdict` mutate）：

**状态寄存器**（`on_episode_start` 重置）：
- `_locked: bool`、`_cursor_id: str | None`（最后回放/锚定的 entry_id）
- `_budget_left: int`（当前锁的盲回放剩余预算）
- `_fh_streak: int`（相邻 lockstep transition 计数，见下方计数约定）、`_last_winner_id: str | None`

**构造参数**：
- `lock_streak: int`（触发锁定所需的**相邻 lockstep transition 数**——见下方计数约定；roadmap 建议小起步）
- `budget: int`（每次锁定的盲回放步数 M，roadmap §4 N2：M 从 3–5 小起步）
- `tolerate_delta0: bool = True`（重规划密集段 Δ0 占比高 → 锁定条件容忍 Δ∈{0,1}，roadmap N2 失败模式）

**`__call__`（PURE）**：`if self._locked: return False`（不搜，交 `replay_target` 表达回放）；否则 `return True`（always_search 基线；N2 只在命中段锁定，非命中段照常搜）。

**`replay_target(self) -> str | None`**：`return self._cursor_id if self._locked else None`。

**`record_verdict(...searched, hit_type, winner_id...)`（mutate）**：
- 解锁助手 `_unlock()`：`_locked=False`、`_cursor_id=None`、`_budget_left=0`、`_fh_streak=0`（回到搜索态；证据链清零 → 解锁后需重新累计一整段 lockstep 才能再锁，防抖）。
- 分派 `searched` 优先（同 `ScoreHysteresisGate`）：
  - **searched=True 的 FULL_HIT**（真搜命中）：解析 `winner_id` → `(traj, step)`；判断与 `_last_winner_id` 是否同 traj 且 Δstep ∈ {+1}（或 {0,+1} 若 `tolerate_delta0`）→ 是则 `_fh_streak += 1`，否则 `_fh_streak = 0`（非同轨/断链，证据链断）；**未锁** 且 `_fh_streak >= lock_streak` → 置 `_locked=True`、`_cursor_id = winner_id`、`_budget_left = budget`、`_fh_streak = 0`（锁定即消费证据链）。始终 `_last_winner_id = winner_id`。
  - **searched=True 的非 FULL_HIT**（WS/MISS）：`_fh_streak = 0`；锁定态遇真搜（不应发生——锁定期 `__call__` 恒 False，纯防御）→ `_unlock()`。**`_last_winner_id = None`**（清空可比前驱——lockstep transition 要求**前一个真搜 verdict 本身是 FULL_HIT**；否则 WS(t:0)→FULL_HIT(t:1) 会被误计为 +1 transition 并锁定，G2 R1 Blocking①）。
  - **searched=False 的 FULL_HIT**（**盲回放步成功**，orchestrator 回灌 walked entry id）：`_cursor_id = winner_id`（游标推进到刚回放的 id）、`_budget_left -= 1`；`_budget_left <= 0` → `_unlock()`。**不触** `_last_winner_id`/`_fh_streak`（这两者只由**真搜**证据驱动；解锁后靠新的真搜 lockstep run 重新累计——回答 non-blocking①：盲回放步不改再锁基准，解锁后首个真搜 probe 不会因单步即刻再锁）。
  - **searched=False 的 MISS**（**locked-tail / 异常兜底**，orchestrator 在 `walk_next` 空或抛异常时经**原 skip 分支单次**回灌 `winner_id=None`，见 §3.3）：`_unlock()`——盲回放无法继续 → 立即解锁回搜索态。**这是 locked-tail fail-safe 契约（Blocking①）**：消除"锁死"（否则 `__call__` 恒 False → 反复 `replay_target`→`walk_next` 空 → skip 无预算递减 → 死循环）。
- `winner_id` 解析用 `winner_id.rsplit(":", 1)`（trajectory_id 为 uuid4 无冒号，安全；与 gate_structure_analysis.py:174 一致）。真搜路径非法/None winner_id → fail-safe 不锁。

**`lock_streak` 计数约定（off-by-one 显式化，G1 R2 non-blocking）**：`_fh_streak` 在**每个"与上一真搜 winner 同轨且 Δstep∈容忍集"的真搜 FULL_HIT** 上 +1——它计的是**相邻 lockstep transition 数**，不是原始 FULL_HIT step 数。一段 lockstep run 的**第一个** FULL_HIT 只设 `_last_winner_id`（无前驱可比、`_fh_streak` 不增）。故 `lock_streak=N` 需 **N+1 个连续 lockstep FULL_HIT 步**才触发锁定。golden 测试（§3.6 (a)）用**显式步序列 + 测试名**锁死此约定，杜绝 off-by-one 误读，也据此对齐 §4.1 的 `(lock_streak, budget)` 网格。

### 3.3 `src/openpi/cache/orchestrator.py` — `check()` 新增盲回放分支

在现 `if not should_search:` 分支（orchestrator.py:481-495）**之前**插入盲回放判定：

```
if not should_search:
    replay_id = gate.replay_target() if hasattr(gate, "replay_target") else None
    if replay_id is not None:
        # 盲回放分支：不搜、不判、取 locked winner 后续 payload
        try:
            entries = StoragePayloadView(self._storage).walk_next(replay_id, 1)
        except Exception:            # fork / 缺 entry / backend 不支持 → fail-safe
            logger.warning("[step %d] follow_winner walk_next(%s) failed; "
                           "unlock+fallthrough", self._step_counter, replay_id)
            entries = []
        if entries:
            nxt = entries[0]
            # build() 已在上方无条件执行；这里补记 query_keys 保轨迹 gap-free
            if hasattr(strategy, "record_query_keys"):
                strategy.record_query_keys(query_keys)
            self._feed_verdict_to_gate(
                checkpoint_id, hit_type=HitType.FULL_HIT,
                cp1_score=None, winner_id=nxt.id, start_t=None, searched=False,
            )
            if checkpoint_id == CheckpointID.CP1:
                self._step_counter += 1
            return CheckResult(
                hit_type=HitType.FULL_HIT, payload=nxt.payload,
                entry_id=nxt.id, query_keys=query_keys, searched=False,
            )
        # 轨迹耗尽 / 异常 → 不 early-return，落到下方原 skip 分支（唯一一次回灌）：
        #   原分支 _feed_verdict_to_gate(MISS, winner_id=None, searched=False)
        #   → FollowWinnerGate.record_verdict 见 (searched=False, MISS) → _unlock()（§3.2）
        #   → 本步 MISS→全推理；下一步 __call__ 解锁后恒 True 恢复搜索。无双喂、无锁死。
    # ...（原 skip→MISS→推理路径不变：record_query_keys + miss 计数 +
    #      _feed_verdict_to_gate(MISS, winner_id=None, searched=False) + return MISS）
```

- **返回 `FULL_HIT × searched=False`** = 新组合语义："命中（回放缓存动作）但未真搜"。interceptor 见 FULL_HIT → 短路回放 `payload.action_chunk`（interceptor.py:709-742），跳过 Stage 2/3。
- **成功回放步与兜底步的回灌互斥**：成功（`entries` 非空）在盲回放分支内 early-return（喂 `FULL_HIT×searched=False` 一次）；失败（空/异常）**不** early-return、落原 skip 分支喂 `MISS×searched=False` 一次。二者路径互斥 → gate 每步恰收一次 `record_verdict`，无双喂。
- **backend 约束**：`walk_next` 仅 `InMemoryBackend` 支持（`QdrantVectorStore.fetch_entry` 抛 `NotImplementedError`，payload_view.py 经 `fetch_entry`）。N2 部署走 in_memory preload pkl（gate research 既有配置），满足。构造/校验期对 `follow_winner` + 非 in_memory backend **fail loud**（见 §3.4 校验）——这是**首选**防线；上面的 `try/except` 是运行期兜底（防 fork/缺链等非配置类异常打断 episode），二者互补。
- **fork/边界**：`walk_next` 遇 fork 或跨轨迹抛/停（payload_view.py:182-196）；冻结库单链无 fork、正常停在轨迹尾（返回空 list）。空 list 与抛异常**统一走上面的 unlock+fallthrough**（§3.2 `searched=False, MISS → _unlock()`），单条异常轨迹不致命。

### 3.4 `src/openpi/cache/config.py` — 5 处编辑（沿 Stage 3b 同款模式）

1. `_GATE_TYPES` 加 `"follow_winner"`（config.py:470）。
2. `GateConfig`（config.py:83-100）**只加两个 int 字段**：`lock_streak: int | None = None`、`budget: int | None = None`。`tolerate_delta0` **不进 config**——`FollowWinnerGate` 构造器-only、default `True`（与 Stage 3b `include_ws` 同款规避：bool 默认值一旦进 config 会触发 stray-field 检查、让 legacy gate 全回归）。若 owner 日后要 YAML 可调 `tolerate_delta0`，届时另议、单独加。
3. `_build_gate`（config.py:2211）加 `follow_winner` 分支：assert `lock_streak`/`budget` 非 None → `FollowWinnerGate(lock_streak=..., budget=...)`。
4. 校验（config.py:1244-1394）：加 `_gate_follow_winner_fields = {"lock_streak", "budget"}`；并入 `_gate_all_param_fields`；加 required + bounds（int ≥1）分支 + stray-field 处理；**backend 约束**：`follow_winner` gate + `backend.type != "in_memory"` → error（盲回放需 `fetch_entry`/`walk_next`）。
5. `_GATE_TYPES` 引用处的 docstring/注释同步（config.py:1231-1237 附近说明新 gate 语义）。

### 3.5 `GateFunction` Protocol — `replay_target` 仅文档化，**不得**成为 Protocol 必需方法（Blocking②）

`GateFunction`（gate.py:28）是 `@runtime_checkable` Protocol，已有 `isinstance(<gate>, GateFunction)` 运行时兼容测试覆盖全部 legacy gate（AlwaysSearch / AlwaysSkip / ClientControlled / Random / Periodic / ScoreHysteresis）。Python `@runtime_checkable` Protocol **无法表达 optional 成员**——一旦把 `def replay_target(...)` 写进 Protocol 体，`isinstance` 会要求所有 gate 都实现它，只有 `__call__` 的 legacy gate **全部 fail runtime protocol**。

**契约**：`replay_target(self) -> str | None` **只在 Protocol docstring 的 optional-hook 段描述**（gate.py:39-50，与 `on_episode_start` / `record_action` / `record_verdict` 完全同款——它们也都是 docstring-only、非 Protocol 方法体成员），**绝不**加入 Protocol 方法体。orchestrator 一律经 `hasattr(gate, "replay_target")` 守卫发现（§3.3）。legacy gate 源码**零改动**。回归由 §3.6 的 protocol-compat 测试锁死。

> 备选（不采）：单独定义 `ReplayTargetGate` 辅助 Protocol 仅供 `hasattr` 背后的类型标注。因与既有 optional-hook 惯例（docstring-only）不一致、增无谓表面积而不采；除非 reviewer 坚持要显式类型面。

### 3.6 测试（`tests/cache/` + `tests/exp/`）

- `tests/cache/components/test_gate.py`：`FollowWinnerGate` golden——(a) **锁定触发 + 计数约定锁死**：用**显式步序列**验证 `lock_streak=N` 需 **N+1 个连续 lockstep FULL_HIT 步**（`_fh_streak` 计相邻 transition、非 raw step；测试名 + expected 序列显式标注该 off-by-one，防误读）；(b) 盲回放游标推进 + 预算递减 + 到 0 解锁；(c) 真搜 probe 掉带解锁；(d) Δ0 容忍开关；(e) 非法/None winner_id fail-safe 不锁；(f) `__call__` PURE（多次调用不 mutate）；(g) **locked-tail fail-safe（Blocking①）**——锁定态收 `record_verdict(searched=False, hit_type=MISS, winner_id=None)` → `_unlock()`（`__call__` 回 True、`replay_target()` 回 None、`_budget_left`/`_fh_streak` 清零）；(h) **post-budget reacquire（non-blocking①）**——盲回放步不改 `_last_winner_id`/`_fh_streak`，解锁后需一整段新的 `lock_streak` lockstep run 才再锁（单个真搜 probe 不即刻再锁）。
- `tests/cache/test_gate_protocol_compat.py`（新，**Blocking②**）：`isinstance(g, GateFunction)` 对全部 6 个 legacy gate（AlwaysSearch/AlwaysSkip/ClientControlled/Random/Periodic/ScoreHysteresis）+ `FollowWinnerGate` 均 True；**回归锁**——构造一个只实现 `__call__`（无 `replay_target`）的最小 stub，断言 `isinstance(stub, GateFunction)` 仍 True（若有人误把 `replay_target` 写进 Protocol 体，此断言即 fail）。
- `tests/cache/test_orchestrator.py`：盲回放执行分支——gate `replay_target` 非 None 且 `walk_next` 命中 → 返回 `FULL_HIT × searched=False` 携 walked payload、不触 search/judge、`record_query_keys` 被调用；**`walk_next` 空（轨迹尾）→ 落原 skip 分支、gate 收 (searched=False, MISS) 解锁、返回 `CheckResult(MISS, searched=False)`**；**`walk_next` 抛异常 → 同上（try/except fail-safe，Blocking①）**。
- `tests/cache/test_config.py`：`follow_winner` 校验（required/bounds/stray-field/backend-约束）；legacy gate 非回归。
- `tests/cache/`（backend）：`follow_winner` + qdrant backend → `ConfigValidationError`（fail loud）。
- `tests/exp/test_stage4a_yaml.py`（新）：加载 n2_server YAML，断言 `follow_winner` + `lock_streak`/`budget`。

### 3.7 文档（L3 → 架构文档更新，§Documentation 义务）

- `docs/architecture/cache_system.md`：新增 gate 执行三态（search / skip-infer / **blind-replay**）+ FollowWinnerGate + orchestrator 盲回放分支 + `replay_target` hook + backend 约束。
- `docs/cache/tutorial.md`：N2 部署配方（follow_winner，in_memory backend，lock_streak/budget）。
- `docs/README.md` / `logs/README.md`：索引同步（Index Sync 宪法红线，同 commit）。
- `logs/gate_exploration_roadmap.log.md`：Stage 4a 回填（G2 APPROVED 后）。
- `.zh.md` 冻结在 2026-04-03，不更新（沿 Stage 3b 惯例）。

---

## 4. Phase C — live 实验设计（G1 已审设计；实际 run 在 G2/commit 之后，非本计划 Code/G2 范围）

> 沿 Stage 3a 惯例：代码经 G2/commit 后另起 live-run 执行步。此处只定方法论 + 及格线 + 拓扑。

### 4.1 评估网格（RPG-同构）

N2 = periodic「cache k / infer n」的事件驱动智能版（roadmap §4 N2 依据）。参数映射：`(lock_streak j, budget M)` ↔ periodic `(k, n)` 同构网格（`lock_streak` 计数约定见 §3.2，选点时按"transition 数"折算）。低剂量甜点先行（roadmap F12：最低试验剂量即达满增益）。分 suite（spatial / libero_10）各扫 2–3 点。

### 4.2 及格线（roadmap §5 三元坐标，C10/C11）

一律报 (SR, inf_ratio, net@34)。及格 = **SR ≥ 同 inf_ratio 的 matched periodic**（真正对手，F10/F11）**∧ net@34 ≥ 0 ∧ SR ≥ baseline − 1pp**。N2 的独特价值在省 search（inf_ratio 不受损、net 应显著正于 miss-skip），SR 保真是关键考核。

### 4.3 分析口径新增（盲回放步的记账）

盲回放步 = `FULL_HIT × searched=False`：对 inf_ratio 记为 cache-execution（inf=0，同 FULL_HIT）；对 search 成本记为**已省**（N2 的价值面）。`analyze_n1_live.py` 的 inf/net 记账需识别此新组合（Phase C analyzer 改动）。

### 4.4 拓扑（复用 Stage 3a live，`project_twa_*` / `project_n1_stage1b_live_run` 同款）

ziyang10（server，3 replica / 2 batch，H200，须 expose 走 broker）+ timan107（client，48 worker）。spatial 与 libero_10 两 suite yaml **不同时跑**（memory 既有约束）。

---

## 5. 风险登记

| # | 风险 | 缓解 |
|---|---|---|
| R1 | Phase A F5 复测判 NO-GO（注入摧毁 lockstep） | 这是设计内的 gating——NO-GO 即停、不建 L3，回报 owner。省掉昂贵 L3 白做 |
| R2 | 盲回放段 live 轨迹漂移出库轨迹（verdict 不在场） | budget M 小起步（3–5）+ probe 解锁兜底（B3 安全绳）；SR 必 live 验（C8） |
| R3 | `FULL_HIT × searched=False` 新组合污染既有采集/分析 | Phase B 测试覆盖该组合的 orchestrator 语义；Phase C analyzer 显式识别；不改既有 always-search 采集路径 |
| R4 | 进入条件① 部署延迟档实为优化小库 → N2 无 V1 价值 | 已在 §1.2 如实标注，交 G1/owner 判定；非本计划可验 |
| R5 | backend 非 in_memory 时 `walk_next` 抛 NotImplementedError | config 校验期 fail loud（§3.4 #4）；N2 部署恒 in_memory |
| R6 | **locked-tail 锁死**：`walk_next` 空(轨迹尾)/fork/异常 若不解锁 → `__call__` 恒 False → 反复 `replay_target`→`walk_next` 空 → skip 无预算递减 → 死循环 | **明确 fail-safe 契约**（§3.2/§3.3）：orchestrator `try/except` + 空即落**原 skip 分支单次**回灌 `(searched=False, MISS, winner_id=None)` → gate `_unlock()`；测试覆盖空与异常两路（§3.6 (g)+orchestrator） |
| R7 | L3 接口变更破坏 WA §2.5"走 wrapper 不改推理内核" | 采方案 A（additive optional hook + hasattr 守卫），`__call__` 仍 bool，legacy 零影响；orchestrator 分支是新增非改写 |
| R8 | **Protocol 兼容**：把 `replay_target` 写进 `@runtime_checkable GateFunction` 体 → legacy gate `isinstance` 全坏 | docstring-only + `hasattr` 守卫（§3.5）；新增 protocol-compat 回归测试（§3.6）锁死"只有 `__call__` 的最小 stub 仍 `isinstance` True" |

## 6. 测试策略汇总

- 单测：gate golden（锁定/回放/解锁/容忍/fail-safe/PURE/**locked-tail 解锁**/**post-budget reacquire**/**lock_streak 计数约定**）、**gate protocol runtime-compat（legacy `isinstance` 全绿 + 最小 `__call__` stub 回归锁）**、orchestrator 盲回放分支（命中 + **空/异常两路解锁**）、config 校验（required/bounds/stray/backend）、legacy 非回归、yaml 加载。
- §6 Verify 口径（`reference_pytest_manual_skip` 铁律）：裸 `uv run pytest tests/cache tests/exp`（改动 blast-radius）；**禁** repo-wide / `-m "not manual"` / `tests/review_tests`。
- Phase A/Phase C 为 exp 离线分析 / live，不入本计划 Code 的单测门；Phase A 产报告，Phase C live 在 commit 后。

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-07-06 21:36 CDT

#### Verdict

NEEDS REVISION. The implementation largely matches the G1-approved architecture and the scoped tests pass, but two blocking issues prevent G2 approval: the `FollowWinnerGate` lock evidence can cross a non-FULL_HIT verdict, violating the "consecutive FULL_HIT lockstep" contract; and scoped ruff fails on the new/modified files.

#### Checklist

- Consistency with approved plan: NEEDS REVISION. The blind-replay orchestrator branch, docstring-only `replay_target()` hook, config fields, backend guard, Phase A report, YAMLs, docs, and tests follow the approved plan. However, `FollowWinnerGate.record_verdict()` sets `_last_winner_id = winner_id` on searched non-FULL_HIT verdicts. A `WARM_START(t:0) -> FULL_HIT(t:1)` sequence therefore increments `_fh_streak` and can lock with `lock_streak=1`, even though the plan, docs, and tests define `lock_streak` over consecutive searched FULL_HIT transitions.
- Test coverage and passing: NEEDS REVISION. The submitted scoped tests pass (`269 passed`), but they miss the WS/MISS boundary case above. Add a regression test proving that any searched non-FULL_HIT breaks both `_fh_streak` and the comparable predecessor for lock acquisition.
- Docs & indexes updated: APPROVED. `docs/architecture/cache_system.md`, `docs/cache/tutorial.md`, `docs/README.md`, and `logs/README.md` were updated and are consistent with the new blind-replay semantics. The frozen Chinese companion remains untouched, consistent with prior stage practice.
- No regressions: NEEDS REVISION. Protocol compatibility and legacy gate stray-field handling are covered, but ruff currently fails on the touched files, so the code is not ready to ship under project tooling expectations.

#### Blocking items

- [Blocking] [Concern] `FollowWinnerGate` can lock across a searched non-FULL_HIT verdict. — reasoning: after `record_verdict(... hit_type=HitType.WARM_START, winner_id="t:0", searched=True)`, the implementation resets `_fh_streak` but stores `_last_winner_id="t:0"`. The next searched `FULL_HIT` with `winner_id="t:1"` sees delta +1, increments `_fh_streak`, and locks when `lock_streak=1`. This violates the approved "consecutive FULL_HIT lockstep transitions" contract and can enter blind replay after a WS/FH boundary. Fix by clearing the comparable predecessor on searched non-FULL_HIT (for example `_last_winner_id = None`) or otherwise requiring the previous verdict itself to be `FULL_HIT` before counting a transition. Add a targeted regression test for `WARM_START -> FULL_HIT` and, if relevant, searched MISS with any carried `winner_id`.

- [Blocking] [Concern] Scoped ruff fails on new/modified files. — reasoning: `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/openpi/cache/components/gate.py src/openpi/cache/orchestrator.py src/openpi/cache/config.py exp/gate_research/stage4a_f5_recheck.py tests/cache/components/test_gate.py tests/cache/test_gate_protocol_compat.py tests/cache/test_orchestrator.py tests/cache/test_config.py tests/exp/test_stage4a_yaml.py` reports `E741` for variable `l` in `stage4a_f5_recheck.py`, multiple `E702` semicolon-on-one-line violations in `tests/cache/components/test_gate.py`, and `F401` unused `CheckpointID` in `tests/cache/test_gate_protocol_compat.py`. These are mechanical but blocking for G2.

#### Verification performed

- Read the polished plan `logs/gate_stage4a_n2_follow_winner.log.md`.
- Read the code/doc/test diffs and newly added files for Stage 4a: `src/openpi/cache/components/gate.py`, `src/openpi/cache/orchestrator.py`, `src/openpi/cache/config.py`, `tests/cache/components/test_gate.py`, `tests/cache/test_gate_protocol_compat.py`, `tests/cache/test_orchestrator.py`, `tests/cache/test_config.py`, `tests/exp/test_stage4a_yaml.py`, `exp/gate_research/stage4a_f5_recheck.py`, `exp/gate_research/analysis/stage4a_f5_recheck.md`, the two `n2_server` YAMLs, and updated docs/indexes.
- Ran targeted semantic probe confirming the WS/FH lock bug:
  `WARM_START(t:0)` leaves `_last_winner_id=t:0`; then `FULL_HIT(t:1)` locks at `lock_streak=1`.
- Ran `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run pytest tests/cache/components/test_gate.py tests/cache/test_gate_protocol_compat.py tests/cache/test_orchestrator.py tests/cache/test_config.py tests/exp/test_stage4a_yaml.py -q` — `269 passed, 9 warnings`.
- Ran `python -m py_compile src/openpi/cache/components/gate.py src/openpi/cache/orchestrator.py src/openpi/cache/config.py exp/gate_research/stage4a_f5_recheck.py tests/cache/test_gate_protocol_compat.py tests/exp/test_stage4a_yaml.py` — PASS.
- Ran `git diff --check` — PASS.
- Ran scoped `ruff check` command above — FAIL with the listed lint errors.

### G2 Round 2 — Executor — 2026-07-06

- **Accepted (Blocking①, lock across a searched non-FULL_HIT)** — 真 bug。`record_verdict` 的 searched=True 非 FULL_HIT 分支原先 `self._last_winner_id = winner_id`，使 WS(t:0)→FULL_HIT(t:1) 的 Δ 从 WS 的 winner 算出 +1、误计为 lockstep transition，`lock_streak=1` 下直接锁定，违反"连续真搜 FULL_HIT transition"契约。**修复**（`gate.py`）：该分支改为 `self._last_winner_id = None`——清空可比前驱，使 WS/MISS 之后的首个 FULL_HIT 成为新 run 起点（transition 要求前一真搜 verdict 本身是 FULL_HIT）。同步修计划 §3.2 该行的 spec（原文亦写 `= winner_id`，属计划内与"连续 FULL_HIT"契约自相矛盾之处，一并订正为 `= None`）。**回归测试**（`test_gate.py::test_follow_winner_no_lock_across_non_full_hit_boundary`）：对 `WARM_START` 与**携带 winner_id 的 MISS** 两种真搜非 FULL_HIT，断言其后 `_last_winner_id is None` 且随后的 FULL_HIT(t:1) 在 `lock_streak=1` 下 `_locked is False`、`_fh_streak == 0`。

- **Accepted (Blocking②, scoped ruff FAIL)** — 全部 14 处机械修复：`stage4a_f5_recheck.py` E741（`l`→`lab`）；`test_gate.py` 12 处 E702（把 `_searched_fh`/`_replay_fh` 改为 varargs helper，`A; B` 折叠为单次 `A(…, x, y)` 调用，消除所有分号）；`test_gate_protocol_compat.py` F401（删未用 `CheckpointID` import）。`scoped ruff check` 现 **All checks passed**。

本轮改动：`gate.py`（WS/FH 契约修复）+ 计划 §3.2 spec 订正 + 3 个测试文件 lint 修复 + 1 个新回归测试；无功能面其它改动。**复验证据**：scoped ruff 全绿；目标测试 `270 passed`（+1 新回归）；全 blast-radius `uv run pytest tests/cache tests/exp` = **1848 passed / 6 skipped**，无回归。请复审。

### G2 Round 2 — Reviewer — APPROVED — 2026-07-06 21:47 CDT

#### Verdict

APPROVED. Both G2 Round 1 blocking items are closed. The WS/FH cross-boundary lock bug is fixed by clearing `_last_winner_id` on searched non-FULL_HIT verdicts, and the regression test now locks that contract. The scoped ruff failures are resolved.

#### Checklist

- Consistency with approved plan: APPROVED. The implementation now matches the approved "consecutive searched FULL_HIT lockstep transition" contract: searched WS/MISS clears the comparable predecessor, so the following FULL_HIT becomes a fresh run start instead of a lock transition. The blind-replay branch, docstring-only `replay_target()` hook, config/backend validation, Phase A report, YAMLs, and docs remain consistent with G1.
- Test coverage and passing: APPROVED. The new `test_follow_winner_no_lock_across_non_full_hit_boundary` covers WARM_START and searched MISS with winner_id before a following FULL_HIT, closing the previous coverage gap. Scoped tests pass.
- Docs & indexes updated: APPROVED. The prior docs/index updates remain in the staged snapshot; the plan spec was corrected to `_last_winner_id = None` for searched non-FULL_HIT.
- No regressions: APPROVED. Scoped ruff now passes; protocol compatibility, orchestrator blind-replay paths, config validation, YAML loading, and legacy gate regression tests pass.

#### Verification performed

- Read the executor's G2 Round 2 response and working-tree diff against the previous reviewer-staged baseline.
- Ran targeted semantic probe: `WARM_START(t:0)` leaves `_last_winner_id=None`; subsequent `FULL_HIT(t:1)` does **not** lock at `lock_streak=1`.
- Ran `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run pytest tests/cache/components/test_gate.py tests/cache/test_gate_protocol_compat.py tests/cache/test_orchestrator.py tests/cache/test_config.py tests/exp/test_stage4a_yaml.py -q` — `270 passed, 9 warnings`.
- Ran `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/openpi/cache/components/gate.py src/openpi/cache/orchestrator.py src/openpi/cache/config.py exp/gate_research/stage4a_f5_recheck.py tests/cache/components/test_gate.py tests/cache/test_gate_protocol_compat.py tests/cache/test_orchestrator.py tests/cache/test_config.py tests/exp/test_stage4a_yaml.py` — PASS.
- Ran `python -m py_compile src/openpi/cache/components/gate.py src/openpi/cache/orchestrator.py src/openpi/cache/config.py exp/gate_research/stage4a_f5_recheck.py tests/cache/test_gate_protocol_compat.py tests/exp/test_stage4a_yaml.py` — PASS.
- Ran `git diff --check` — PASS.

#### Non-blocking follow-up

- Plan header still says `Status: Plan（...进入 §4 Code）`; update status during Verify/finalization along with the roadmap Stage 4a回填 after G2 approval.
