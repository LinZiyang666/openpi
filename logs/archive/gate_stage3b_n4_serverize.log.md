# Stage 3b — N4 服务器化（Plan）

- **Task**: 把 3a live 胜出的 N4 混合门（V1 滞回 ⊕ V2 连续缓存执行 run-length ≥ L 注入，赢点 **L=6**）从 exp 层客户端状态机搬到服务器侧——**扩展现有 `ScoreHysteresisGate`**（1c serverize 的 N1）加 V2 分支 + 操作点 YAML + 部署配方。
- **Level**: L2（src 改动：`gate.py` 扩展 + `config.py` 装配 + YAML + 测试）。**含 src → 需 G1/G2 双门。**
- **Authority**: Execution
- **Date**: 2026-07-05（创建）
- **Roadmap 锚点**: `logs/gate_exploration_roadmap.log.md` §5 Stage 3 表 3b 行 + Stage 3a 判决回填的**对 3b 净指令**：服务器化用 **L=6**（SR 档 N4 / 延迟档 N1-A 纯 V1 / 上限对照 periodic）。
- **前置**: Stage 3a ✅（N4 胜出 L=6，4/6 pass，报告 `exp/gate_research/analysis/stage3_n4_live.md`，代码 `251eddc`）；1c ✅（`ScoreHysteresisGate` serverize N1，`dc2815e`）。
- **Status**: G1 APPROVED（R5）；§4 Code 完成（gate+config+n4_server yaml+tests+docs；自检 cache+exp+review 1845 passed / ruff clean，无程序效力）；待 G2 外审。

---

## 1. 目标与范围

### 1.1 目标

3a 用 exp 层 `N4GateState`（客户端）驱动 `ClientControlledGate` 验证了 N4 机制。3b 把该机制**固化进 server 侧 gate**，使部署时不依赖客户端状态机——扩展 1c 的 `ScoreHysteresisGate`（已 serverize N1 的 V1 滞回）加 **V2 连续缓存执行注入分支**，参数 **L=6**（3a 赢点）。

**关键实现事实（决定改动面小 = "1c 管道现成"）**：
- server gate 的 `record_verdict`（`gate.py:403`）**已收 `hit_type`**（当前 unused，注释"forward compatibility"）——V2 数连续 FULL_HIT run 所需的 hit_type 已在手，**无需 orchestrator 改动**。
- orchestrator `_feed_verdict_to_gate`（`orchestrator.py:385`）在每条 check() 路径喂 `hit_type/cp1_score/winner_id/start_t/searched`（:407-413），skip 步 `searched=False`（:491-495）——decide/observe 分离已就绪。

### 1.2 及格线（本 L2 = 正确性关键性，非落位）

- **等价性**：`ScoreHysteresisGate(L=6)` 逐步等于 3a 的 `N4GateState(L=6)`（同 (θ,j,probe_interval,L) 对同一 (decision, hit_type, score) 序列决策序列一致）——这是 server≡client 的核心正确性判据（镜像 1c 对 N1 的等价验证）。
- **向后兼容**：`L=None`（或不设）→ V2 永不触发 → **行为等价现有 N1 `ScoreHysteresisGate`**（decide/observe 序列一致）；1c 全部 golden traces + config 测试逐字不变通过（NB1）。
- **零 orchestrator/wire 改动**：hit_type 已在 record_verdict；只改 gate + config。
- 交付 = src 门（扩展）+ 操作点 YAML（SR 档 L=6 spatial/l10）+ 部署配方 + 测试全绿。

### 1.3 范围外

- **live 再验证**：3a 已 live 验 N4 客户端；3b 是纯 serverize，靠等价 golden trace 保证 server≡client，**不需要新 live run**（可选一次 manual smoke 确认 server gate 决策与 client 一致，列可选）。
- **危险步靶向**：2c 否决，永久排除。
- **danger/其他门**：不动其余 5 个 gate（逐字节不变）。
- **操作点 YAML 化的 conductor 热切换**：沿用 1c/3a 已有机制，不新造。

---

## 2. 数据资产与可复用 API（均已亲验 file:line）

### 2.1 现有 server gate（扩展目标）

- `src/openpi/cache/components/gate.py:306` `ScoreHysteresisGate(theta_low, theta_high, j, probe_interval)` — 1c serverize 的 N1。
  - `__init__`（:334）：校验 θ 有限实数 + theta_high≥theta_low + j int≥1 + probe_interval None/int≥1；state `_searching/_low_run/_since_probe/_task_key`（:381-384）。
  - `__call__`（:386）**PURE decide**（无 mutation）：`_searching`→True(search)；skipping+probe-due(`_since_probe+1>=probe_interval`)→True(probe)；else False(skip)。
  - `record_verdict`（:403，`*, hit_type, cp1_score, winner_id, start_t, searched`）**observe**：cp1_score None→−inf（:419-423）、非有限→fail-open 恢复搜索（:424-436）；`_searching` 分支 low_run 逻辑（:440-447）；skipping 分支 probe/recover（:448-455）。**hit_type 当前未用**（:415-417 注释 forward-compat）。
  - `on_episode_start(task_key="")`（:457）reset state。`record_action`（:470）no-op。
- `src/openpi/cache/orchestrator.py:385` `_feed_verdict_to_gate(...)` — `hasattr(record_verdict)` 守卫（:406）后喂 `hit_type=/cp1_score=/winner_id=/start_t=/searched=`（:407-413）；skip 步 `searched=False`（:491-495）。**V2 所需 hit_type + searched 已全喂到。**

### 2.2 config 装配（加 L 字段）

- `src/openpi/cache/config.py:83` `GateConfig`：字段 `theta_low/theta_high/j/probe_interval`（:92-95）。**只加 `L: int | None = None`**（同 None-默认口径，与现有 optional 字段一致）。**不加 `include_ws`**（G1 R1 item2：`gate_set_fields = {getattr(..) is not None 的字段}`，`bool=False` 非 None 会让**所有** legacy gate 携带 include_ws 被判 stray → 全 config 回归）。include_ws 只留 gate 构造器默认 False，不进 config/YAML（方案 A，见 D1）。
- `config.py:465` `_GATE_TYPES`（含 `score_hysteresis`）——**不新增类型**，扩展现有。
- `config.py:1245` `_gate_score_hysteresis_fields = {theta_low, theta_high, j, probe_interval}`——**加 `L`**（否则 :1365 stray 检查会拒 L）。
- `config.py:1316-1368` validator score_hysteresis 分支——**加 L 校验**（None 或 int≥1，拒 bool；镜像 probe_interval 的 :1356-1364）。
- `config.py:2196` `_build_gate`，score_hysteresis 分支（:~2222 `return ScoreHysteresisGate(theta_low=, theta_high=, j=, probe_interval=)`）——**加 `L=cfg.L`**。

### 2.3 3a 参考实现（V2 语义单一真源 + 等价基准）

- `exp/gate_research/n4_gate_client.py:51` `N4GateState(theta_low, theta_high, j, M, L, include_ws=False)`——3a 客户端机器：`decide()`（V1 skip 优先 → fh_run≥L skip[V2] → search）、`observe(decision, hit_type, score)`（搜索步推进 N1 + fh_run；skip 步 fh_run=0 + `_last_v2`?冻结:since_probe++）、D1-D5 设计。**3b server gate 必须逐步等于它（L=6）**。
- `src/openpi/cache/interceptor.py:500` `_build_hit_meta` → `hit_type` = FULL_HIT/WARM_START/MISS（已在 record_verdict）。

### 2.4 3a 操作点（YAML 参数来源）

- spatial：θ 0.968929/0.968929、j 3、probe_interval(M) 3、**L 6**（3a spatial_n4_L6 pass）。
- l10：θ 0.996873/0.996873、j 3、probe_interval 3、**L 6**（3a l10_n4_L6 pass）。
- 现成 client_controlled/n1 YAML：`exp/gate_research/config/{libero_spatial,libero_10}/n1/*.yaml`（3b 的 score_hysteresis YAML 复用其 keybuilder/judge/preload，仅改 gate 段）。

---

## 3. 设计：扩展 ScoreHysteresisGate 加 V2（本 plan 核心）

### 3.1 状态增量

```
ScoreHysteresisGate(theta_low, theta_high, j, probe_interval, L=None, include_ws=False)
  现有: _searching / _low_run / _since_probe / _task_key
  新增: _L (None | int>=1)          # V2 注入阈值，None = 纯 N1（V2 关）；来自 config
        _include_ws (bool=False)     # 构造器默认 False，NOT config/YAML 字段（D1/item2）
        _fh_run (int)                # 连续缓存执行 run-length，on_episode_start 归零
```

### 3.2 `__call__`（PURE decide，加 V2 分支）

```
if self._searching:
    if self._L is not None and self._fh_run >= self._L:
        return False            # V2 注入（skip）——注意：searching 态的 skip 唯一来源
    return True                 # search
if probe-due (probe_interval and _since_probe+1 >= probe_interval):
    return True                 # probe
return False                    # V1 skip（skipping 态）
```

**保持 PURE**（无 mutation）——mutation 全在 record_verdict。V2 判据只读 `_fh_run/_L`。

### 3.3 `record_verdict`（observe，重构 = 移植 N4GateState.observe）

现有 record_verdict 对所有步按 cp1_score 推进滞回（隐含假设 searching⟺searched）。V2 破坏该假设（searching 态可 skip）→ **按 searched 先分派**：

```
if searched:                                  # 真发生了缓存搜索
    # V2 run 计数（新增）——⚠ hit_type 是 server 侧 HitType ENUM，不是字符串
    is_cache_exec = (hit_type == HitType.FULL_HIT) or (self._include_ws and hit_type == HitType.WARM_START)
    self._fh_run = self._fh_run + 1 if is_cache_exec else 0
    # N1 滞回推进（现有逻辑：cp1_score None→−inf、非有限→fail-open+_fh_run=0）
    <existing score → _searching/_low_run/_since_probe transitions>
else:                                          # skip 步（searched=False）= 全新推理
    self._fh_run = 0                           # 打断缓存执行段
    if not self._searching:                    # V1 skip（skipping 态）
        self._since_probe += 1                 # 现有 probe 计数
    # else: V2 注入（searching 态）→ N1 冻结（不动 _searching/_low_run/_since_probe）
```

**⚠ HitType enum 边界（G1 R1 item1）**：`CacheOrchestrator._feed_verdict_to_gate`（`orchestrator.py:521` `hit_type = judge_result.hit_type`）传给 `record_verdict` 的 `hit_type` 是 **`HitType` enum**（`components/judge.py:47`），**不是字符串**。所以比较**必须** `hit_type == HitType.FULL_HIT` / `HitType.WARM_START`——写成 `== "FULL_HIT"` 会恒 False → `_fh_run` 永不增 → N4 退化纯 N1（静默 bug）。`gate.py` 顶部加 `from openpi.cache.components.judge import HitType`（Code 时验无循环 import；judge 不 import gate，否则用方法内 lazy import）。3a `N4GateState`（客户端）读的是 wire 字符串 `hit_type`（`__hit_meta__["hit_type"]` = `.name`）——两层口径不同，等价测试只在 **N4 参考侧**做 enum→字符串翻译（见 §5.4）。

**V1/V2 区分靠 `_searching` 重构，无需客户端的 `_last_v2` 标志**（D3）——PURE `__call__` + record_verdict 分离天然提供：searching 态的 skip 只可能是 V2（__call__ 唯一出口），skipping 态的 skip 是 V1。

**fail-open**（非有限 cp1_score，现有 :424-436）追加 `_fh_run = 0`（镜像 N4GateClient.force_search）。

### 3.4 `on_episode_start`

现有 reset + 追加 `self._fh_run = 0`。

### 3.5 校验（__init__）

现有 θ/j/probe_interval 校验 + 追加：`L` 为 None 或 int≥1（拒 bool，镜像 j/probe_interval）；`include_ws` 为 bool。

### 3.6 设计决策（G1 焦点）

| # | 决策 | 选择 | 理由 |
|---|------|------|------|
| D1 | 缓存执行 run 定义 | 默认 **仅 FULL_HIT**；`include_ws` **只留构造器默认 False、不进 config/YAML**（方案 A） | 与 3a `N4GateState` 默认一致（3a live 固定 include_ws=False）；WARM_START 已注入部分算力破坏纯回放。**不暴露 config**（item2）：避免 `bool=False` 触发 stray-field 让 legacy gate 全回归；未来要 True 只能直接构造 gate（3a live/roadmap 均不需 server 侧 True）。 |
| D2 | V2 注入长度 | **1 步**（fh_run 到 L 注入一次即清零） | 同 3a D2 / roadmap 单数 skip / F12 低剂量。 |
| D3 | V1/V2 skip 区分 | **靠 `_searching` 状态重构，无 `_last_v2` 标志** | server 的 PURE `__call__`+record_verdict 分离天然区分（searching 态 skip=V2 唯一来源）；比 3a 客户端少一个状态位、更简。 |
| D4 | 扩展 vs 新 gate 类型 | **扩展 `ScoreHysteresisGate`**（加 L，None=N1） | roadmap 明写"扩展"；`L=None` **行为等价** N1（V2 永不触发 → decide/observe 序列一致，1c golden traces + config 不动）；单类少一套装配。（NB1：判据是决策/观察等价 + 1c golden 通过，非渲染/源码逐字节。） |
| D5 | fail-open | 非有限 cp1_score → 现有恢复搜索 **且** `_fh_run=0` | 镜像 N4GateClient.force_search，退化为纯 N1，绝不误注入。 |
| D6 | L=None 语义 | V2 关 = 纯 N1（延迟档 N1-A 用之） | 部署配方：延迟档 score_hysteresis L=None（=1c N1）；SR 档 L=6（N4）；同一 gate 类两档。 |
| D7 | hit_type 类型口径（item1） | **server 侧比 `HitType` enum**（`== HitType.FULL_HIT/WARM_START`）；gate.py import HitType | orchestrator 传的是 enum 非字符串；字符串比较恒 False→静默退化纯 N1。等价测试的 enum→str 翻译只在 N4 参考侧做（§5.4）。 |

---

## 4. 涉及文件

### 4.1 修改（src）

| 文件 | 改动 | 范围 |
|------|------|------|
| `src/openpi/cache/components/gate.py` | 顶部 `from openpi.cache.components.judge import HitType`（D7，Code 时验无循环）；`ScoreHysteresisGate` 加 `L`(config)/`include_ws`(构造器默认 False)/`_fh_run`；`__call__` 加 V2 分支；`record_verdict` 按 searched 重构 + fh_run(`== HitType.FULL_HIT/WARM_START`) + fail-open 清零；`on_episode_start` 清 fh_run；`__init__` 加 L(None/int≥1,拒 bool) + include_ws(bool) 校验。**L=None 时行为等价现有 N1。** | +~35 行，其余 5 gate 不动 |
| `src/openpi/cache/config.py` | `GateConfig` **只加 `L: int\|None=None`**（不加 include_ws，item2）；`_gate_score_hysteresis_fields` 加 L；validator 加 L 校验（None 或 int≥1，拒 bool）；`_build_gate` score_hysteresis 分支传 `L=cfg.L`（include_ws 走构造器默认）。 | +~12 行，其余 gate 装配不动 |

### 4.2 新增（操作点 YAML + 测试）

| 文件 | 内容 |
|------|------|
| `exp/gate_research/config/{libero_spatial,libero_10}/n4_server/*.yaml` | SR 档 score_hysteresis + L=6 操作点（复用 n1 yaml 的 keybuilder/judge/preload，gate 段换 score_hysteresis + θ/j/probe_interval/L=6）。 |
| `tests/cache/components/test_gate.py`（扩） | V2 golden：L=None≡现有 N1；L=6 全 FULL_HIT→search×L,skip,…；WS/MISS 清零 run；V1/V2 skip 区分（searching 态 skip=V2 冻结 N1、skipping 态 skip=V1 since_probe++）；fail-open 清 fh_run；**等价：ScoreHysteresisGate(L=6) ≡ N4GateState(L=6)**（在 tests/exp 或 in-test 重现 N4 参考，见 §6）。 |
| `tests/cache/test_config.py`（扩） | L 校验：valid(L=6)、L=None omit、L 非法(0/bool/float)、含 L 但 stray 其他字段拒；**legacy 非回归**：always_search/periodic/random config（不设 L）不触发 stray（证 item2 方案 A 无回归）。 |
| `tests/cache/test_orchestrator.py`（扩） | V2 闭环过 orchestrator（镜像 :875 `test_score_hysteresis_gate_closes_loop_through_orchestrator`，L=6 版），**喂真实 `HitType` enum**（FULL_HIT 序列驱动注入）。 |
| `tests/exp/test_stage3b_yaml.py`（新增，item4） | **加载真实 n4_server YAML**：过 `load_cache_config` + `validate_cache_config` 两份新 yaml，断言 `gate.type=="score_hysteresis"` + θ/j/probe_interval/L=6 正确、**未误留 client_controlled 或错阈值**；`_build_gate` 建出 `ScoreHysteresisGate` 且 `_L==6`。 |

### 4.3 部署配方（docs + index sync，item3）

- **docs 更新**（每个 docs/logs 改动同 commit 同步 index）：`docs/architecture/cache_system.md`（+ `.zh.md` 变体）gate 段加 score_hysteresis 两档（NB2：延迟档 `L` 省略/None=N1-A、SR 档 `L: 6`=N4）；`docs/cache/tutorial.md` gate 示例同步两 profile；**`docs/README.md` 索引同步**（若新增/改 docs 条目）。
- **logs**：roadmap §5 3b 回填判决（延迟档 L=None / SR 档 L=6 / periodic 上限）+ **`logs/README.md` 索引行**（已建，Code 后同步 Status）。

---

## 5. 测试策略（关键 = 向后兼容 + server≡client 等价）

### 5.1 向后兼容（L=None 行为等价 N1，NB1）

- 现有 1c golden traces（`test_gate.py` ScoreHysteresisGate + `test_orchestrator.py:875` + `test_config.py:1389+`）**逐字不改仍全绿**——证 L=None（默认）decide/observe 行为不变。
- 新增显式测：`ScoreHysteresisGate(L=None)` 与 `ScoreHysteresisGate()`（不传 L）decide/record_verdict 序列对同一 trace 完全一致（判据 = 决策/观察等价，非源码/渲染逐字节）。

### 5.2 V2 语义 golden（镜像 3a `test_n4_gate_client.py`）

- L=6 全 FULL_HIT + N1 恒 searching → decide 序列 `search×6, skip, search×6, skip, …`（连续缓存执行被 cap 在 6）。
- FULL_HIT 段插 WARM_START(默认 include_ws=False)/MISS → fh_run 归零、注入延后。
- include_ws=True → WS 计入 → 注入前移。
- fail-open：非有限 cp1_score → 恢复搜索 + fh_run=0。

### 5.3 V1/V2 区分（D3 核心）

- searching 态 fh_run≥L 的 skip（V2 注入）：record_verdict(searched=False) → fh_run=0 且 **N1 态冻结**（_searching/_low_run/_since_probe 不变）。
- skipping 态的 skip（V1）：record_verdict(searched=False) → _since_probe++（现有）。
- 断言两者 N1 相位演化正确（V2 注入不污染）。

### 5.4 server≡client 等价（本 plan 最强判据）

- 驱动 `ScoreHysteresisGate(θ,j,pi,L=6)`（server：`__call__`→bool，`record_verdict(hit_type=<HitType enum>,cp1_score,searched=call结果)`）与 `N4GateState(θ,j,M=pi,L=6)`（client：`decide()`→str，`observe(decision,hit_type=<str>,score)`）在同一 (score, hit) 序列 lockstep，断言**决策序列逐步一致**。
- **跨层依赖 + enum 口径处理（定案 (a)，item1/R5）**：等价测试放 **`tests/exp/test_stage3b_serverize.py`**（测试层可同时 import `openpi.cache.components.gate.ScoreHysteresisGate` + `exp.gate_research.n4_gate_client.N4GateState`，非 src→exp 依赖）。**enum→str 翻译只在 N4 参考侧做**：server gate 喂真实 `HitType` enum，喂 client `N4GateState` 时用 `hit.name`（字符串）。这样 server 的 enum 路径被真实覆盖（避免"两边都用字符串"漏掉 D7 的 enum bug）。

### 5.5 config 校验（镜像 `test_config.py:1389+` score_hysteresis）

- L=6 valid pass；L omit(None) pass；L=0/L=True/L=1.5 reject；含 L 但 stray 其他字段 reject。
- **legacy 非回归（item2 核心）**：`always_search`/`periodic`/`random` config（均不设 L）过 validator **无 stray L 报错**——证只加 `L`（None 默认）+ 不加 include_ws 后，旧 config 逐字节仍合法。

**Blast radius**：src(gate/config) 改动 → 全量 `uv run pytest -m "not manual"`（CI 等价）是 §6 Verify 正确范围（含 tests/cache + tests/exp + review_tests）；改动触碰 cache gate 装配，需跑 cache/orchestrator/config 全套确认零回归。

---

## 6. 风险登记

| # | 风险 | 影响 | 缓解 |
|---|------|------|------|
| R1 | L=None 未行为等价 N1 | 回归 1c，破坏延迟档 | §5.1 现有 golden 不改仍绿 + 显式 L=None≡默认测；record_verdict 重构后 L=None 路径 searching⟺searched 恒成立 |
| R2 | V1/V2 靠 _searching 重构错判 | 注入污染 N1 相位 / 漏注入 | D3 论证 + §5.3/§5.4 等价测；searching 态 skip 唯一来源是 V2（__call__ 出口穷举） |
| R3 | record_verdict 重构改了 N1 fail-open/None 语义 | 微妙回归 | fail-open/None→−inf 逻辑保留在 searched 分支；skip 分支不读 score；§5.1 覆盖 |
| R4 | config stray 校验漏 L → 拒合法 config | L=6 yaml 被 validator 拒 | :1245 集合加 L + §5.5 stray 测 |
| R5 | 跨层等价测试引入 src→exp 依赖 | 架构污染 | §5.4 定案 (a) 放 tests/exp（测试层跨 import 可接受，非 src 依赖 exp） |
| R6 | 其余 5 gate/wire 被动改动 | 面外回归 | 只改 score_hysteresis 分支 + GateConfig 加 L（其余 gate 装配读不到 L）；全量 pytest 兜底 |
| **R7** | **hit_type enum/str 类型边界（item1）** | **字符串比较恒 False → fh_run 不增 → 静默退化纯 N1** | D7：`== HitType.FULL_HIT/WARM_START` + gate.py import HitType；§5.2/§5.4/orchestrator 测**喂真实 HitType enum**（不喂字符串），等价测只在 N4 参考侧翻译 |
| **R8** | **config bool 默认值陷阱（item2）** | **`include_ws: bool=False` 非 None → legacy gate 全被判 stray → 所有旧 config 回归** | 方案 A：**不把 include_ws 进 GateConfig**（只留构造器默认）；§5.5 legacy 非回归测（always_search/periodic/random 无 stray）兜底 |

---

## 7. 及格线与交付

- **代码**（G1→Code→G2→Verify）：§4 gate/config 改 + n4_server YAML + 测试；全量 `pytest -m "not manual"` 绿（含 1c golden 不回归 + V2 新测 + 等价测 + YAML load 测 + legacy 非回归）。
- **docs/index sync（item3）**：docs 改动（cache_system.md + .zh.md / tutorial.md）同 commit 同步 `docs/README.md`；logs 改动（roadmap / plan）同步 `logs/README.md`——§6 Verify 前用 `git status` 核对每个 docs/logs 改动都配了对应 index 行。
- **判决**：server≡client 等价（L=6）+ 向后兼容（L=None=N1）双证 → N4 服务器化完成；roadmap §5 3b 回填 + 部署配方（延迟档 L=None=N1-A / SR 档 L=6=N4 / periodic 上限）。

---

## 8. 范围外 / 押后

- **live 再验证 server gate**：可选 manual smoke（1 config few-ep，确认 server score_hysteresis+L=6 决策与 3a 客户端一致），不入 3b 必做。
- **操作点 YAML 全套化 + conductor 部署自动化**：沿用现有，不新造。
- **危险步靶向 / 其余门**：永久/暂不。

---

## Review Log

（G2 外审 append。Executor 逐条回应 Accepted/Rejected，见 execution_authority §10。）

### G2 Round 1 — Reviewer — APPROVED — 2026-07-06 10:21 CDT

#### Verdict

APPROVED。实现与 G1-approved plan 一致：N4 V2 分支落在 `ScoreHysteresisGate`，server 侧使用真实 `HitType` enum 计数，`GateConfig` 只新增 `L` 且 `include_ws` 不进 config，`L=None` 保持 N1 行为等价；n4_server YAML、docs/index、测试覆盖均到位。未发现重大问题。

#### Checklist

- Consistency with approved plan: APPROVED。`gate.py` 扩展 `ScoreHysteresisGate(L=None|int, include_ws=False)`，`__call__` PURE V2 注入，`record_verdict` 按 `searched` 分派并用 `HitType.FULL_HIT/WARM_START` 计数；`config.py` 只加 `L` validator/build path；无 orchestrator/wire 改动。
- Test coverage and passing: APPROVED。覆盖 L=None 兼容、V2 run cap、WS/MISS reset、include_ws constructor path、V1/V2 phase separation、fail-open、真实 HitType enum、orchestrator loop、server≡client、legacy config、n4_server YAML load/validate。
- Docs & indexes updated: APPROVED。`docs/architecture/cache_system.md`、`docs/cache/tutorial.md`、`docs/README.md`、`logs/README.md` 已同步 Stage 3b 设计；`logs/README.md` 行与 current plan 口径一致。
- No regressions: APPROVED。1c `ScoreHysteresisGate` review regression 和 Stage 3a N4 client tests 通过；legacy config 非回归测试覆盖 `always_search/always_skip/client_controlled/random/periodic`。

#### Verification

- `python -m py_compile src/openpi/cache/components/gate.py src/openpi/cache/config.py tests/exp/test_stage3b_serverize.py tests/exp/test_stage3b_yaml.py` — PASS
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run pytest tests/cache/components/test_gate.py tests/cache/test_config.py tests/cache/test_orchestrator.py tests/exp/test_stage3b_serverize.py tests/exp/test_stage3b_yaml.py -q` — PASS, 250 passed
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run pytest tests/review_tests/test_n1_serverside_stage1c_g2.py tests/exp/test_n4_gate_client.py -q` — PASS, 33 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/openpi/cache/components/gate.py src/openpi/cache/config.py tests/cache/components/test_gate.py tests/cache/test_config.py tests/cache/test_orchestrator.py tests/exp/test_stage3b_serverize.py tests/exp/test_stage3b_yaml.py` — PASS
- `git diff --check` — PASS

#### Non-blocking follow-up

- Roadmap `logs/gate_exploration_roadmap.log.md` still shows Stage 3b as pending. That is acceptable at G2 because Verify has not completed; update roadmap + `logs/README.md` during Verify/finalization once this code is actually verified/landed.
- G1 plan mentioned the Chinese cache-system companion as a variant, but `docs/README.md` marks it frozen at 2026-04-03. The English architecture doc and index carry the current authoritative update; no G2 blocker.
