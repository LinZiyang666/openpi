# Stage 4a Phase C 使能 — server-side gate 的 per-step `searched` 记录 + N2 run/analyze 支持

- **Status**: In Progress（G1 APPROVED + G2 APPROVED 2026-07-06；§6 Verify green；N4 byte-regression `cmp -s`=0；使能代码就绪，Phase C 扫点 live 未跑）
- **Authority**: Execution
- **Level**: L2（src wire-schema 加字段 + exp harness/analyzer 扩展；含推理响应路径改动 → 需 staged API 测试）
- **Roadmap / 上游**: `logs/gate_exploration_roadmap.log.md` §5 Stage 4a；N2 gate 已服务器化并 commit `eaa4263`（`gate_stage4a_n2_follow_winner.log.md`，G1/G2 APPROVED）
- **Date**: 2026-07-06

---

## 1. 背景与问题

Stage 4a 的 `FollowWinnerGate` 已服务器化（`eaa4263`）。**Phase C live** 要在真机上验 N2 的 SR 保真 + 延迟净收益（C8/C11：SR 离线不可评，必 live）。但上机前有一个架构缺口挡路。

### 1.1 缺口：server-side gate 的 per-step `searched` 当前无处记录

分析器 pass-line 需要每步 `(hit_type, searched)`：
- `hit_type`：走常开的 `__hit_meta__`（`interceptor.py:482` `_build_hit_meta`，无 C5 guard），**已可得**。
- `searched`：**当前只经 `__collect_meta__`**（`interceptor.py:516` `_build_collect_meta` 读 `cp1_result.searched`）——但 `export_collect_meta` 有 **C5 硬校验：CP1 gate 必须 `always_search`**（`config.py:1106`，防采集选择偏差），`follow_winner` 被直接拒。

N1/N4（`client_controlled`）绕过此限是靠**客户端自 stamp**：`n4_gate_client` 知道自己发的 skip/search，把 `searched` 写进 `result["__collect_meta__"]`。**N2 是服务器端事件驱动 gate，无客户端决策**，客户端无从得知 searched；periodic 服务器端 gate 当初靠分析器**闭式重建**（`analyze_n1_live.py:134` `reconstruct_searched`）绕过，但 **N2 事件驱动无法重建**（盲回放 FULL_HIT 与真搜 FULL_HIT 从 hit_type 上不可区分）。

**净结论**：`inf_ratio`（`inf_value(hit_type)`）与 `SR`（journal）**不需要 searched** → 3 个及格条件里 2 个（periodic_pass + SR≥baseline−1）可算；但 **net@34 需要 skip%（=searched=False 比例）**，roadmap 及格线含 net@34 → 必须记录 server-side `searched`。

### 1.2 干净解法：把 `searched` 挂到常开的 `__hit_meta__`

给 `__hit_meta__`（无 C5 guard、每步必产）加 `searched` 字段（additive，源自 `CheckResult.searched`，`orchestrator.py:83`，默认 True，仅 gate-skip / 盲回放置 False）。runner 从 `__hit_meta__` 读 searched。这让**任何** server-side gate 从此都能记录 searched，且不碰 C5 采集语义。

---

## 2. 设计

### 2.1 src wire-schema：`searched` 进 `__hit_meta__`（L2 核心）

**`src/openpi/cache/interceptor.py` `_build_hit_meta`（482-513）**：返回 dict 加 `"searched": cp1_result.searched`；None-placeholder（cache-off / orchestrator 未执行，492-498）加 `"searched": True`（cache-off 每步全推理、searched 概念为真，与 `CheckResult.searched` 默认一致）。
- 无需改调用点：`_build_hit_meta` 已在 FULL_HIT 短路（`interceptor.py:734`，盲回放传入 `cp1_result.searched=False`）与 MISS/WS 路径（`interceptor.py:889`）被调用，两路都带上正确 searched。
- **零推理行为改动**（纯响应元数据 additive）；wire schema 加一个 bool 字段。

**`examples/libero/episode_runner.py`**：
- `_hit_row`（58-74）加 `"searched": hit.get("searched")`（从 `__hit_meta__` 取；旧数据无此键 → None，向后兼容）。
- `infer_recorder`（163-182）：`collect_meta` present 时**仍以 collect_meta 的 searched 为准**（保 N1/N4 客户端 stamp + 采集 provenance `searched_all` 逻辑不变）；`collect_meta` 缺失（N2：无 client stamp + 服务器 export 关）时，`row["searched"]` 由 `_hit_row` 从 hit_meta 填。

**schema 测试同步**：grep 断言 `__hit_meta__` 精确键集的测试（若有）+ 加 searched 覆盖。

### 2.2 exp harness：`run_n1_live.py` 支持 follow_winner（L1 面）

`follow_winner` 是**服务器端 gate**，与 periodic 同类（`DEFAULT_WORKER_MODULE`，无 client 信号）：
- `gate_info`（69-79）：也读 `lock_streak` / `budget`。
- `_resolve_worker_and_env`（114-160）：加 `follow_winner` 分支 → 校验 `lock_streak`/`budget` present → 返回 `DEFAULT_WORKER_MODULE`（不发 client 信号，同 periodic）。
- `build_manifest`（82-111）：记 `lock_streak` / `budget`；`gate_family` 取 `"n2"`（与 n1/n4/periodic 区分，供分析器分派）。
- **注意**：N2 run 的 per_step searched 来自服务器 `__hit_meta__`（§2.1），**不需要** `export_collect_meta`（C5 guard 不触）。

### 2.3 exp analyzer：`analyze_n1_live.py`

**(a) 修 inf 记账潜伏 bug（`run_metrics` 166-179）**：现 `searched=False` 硬编码 `inf_sum += 1.0`。改为 `inf_sum += inf_value(hit_type, start_t)`（FULL_HIT→0 / MISS→1 / WS→warm）。**向后兼容**：N1/N4/periodic 的 `searched=False` 恒 hit_type=MISS → `inf_value=1.0` = 原值（数值不变）；N2 盲回放步 `FULL_HIT×searched=False` → 0（修正）。回归验证：在已本地化的 Stage 3a N4 数据上重跑分析器，断言结果与既有报告**逐值一致**。

**(b) `episode_searched`（141-156）**：加 `follow_winner` → 用**权威记录 searched**（同 `client_controlled` 路径，缺 bool 即 data-integrity error；**不**像 periodic 重建）。

**(c) `analyze()`（361-429）+ 配对/overall**：`follow_winner` 的 pass-line 与 N4 **完全一致**（inf-matched periodic + 3 条件：`SR≥inf-matched periodic` ∧ `SR≥baseline−1` ∧ `net@34≥0`）。**具体方案（已定）**：`analyze()` 里把 inf-matched 分派条件从 `gate_family == "n4"` **broaden** 到 `gate_family in ("n4", "n2")`，两者走**同一** `match_periodic_n4`（298）+ `n4_overall`（344），**不改这两个函数的实现、不改其返回 dict 的键名**（`periodic_pass_n4` / `n4_sr` 等对 N2 略 misnomer 但功能正确，为保 N4 result-manifest 逐字节不变而不重命名）。仅 `render_md`/`_periodic_cell`/`_n4_components_table` 的**标题/标签**按 `has_n4 || has_n2` 泛化措辞（见 (d)）。**N4 不回归的硬保证**：N4-only 报告与 result-manifest 逐字节不变（回归锁测试，§3）。

**(d) 渲染（`render_md` 470 / `_periodic_cell` 437 / `_n4_components_table` 448）**：让标题/分量表/caption 覆盖 follow_winner（N2）；N4-only 与 N1-only 报告逐字节不变。

---

## 3. 测试（覆盖 reviewer G2 核验清单）

- **`tests/cache/test_interceptor_hit_meta.py`**（既有，同步）：`_build_hit_meta` 含 `searched`——FULL_HIT 真搜 True、盲回放 `searched=False`、gate-skip False、None-placeholder True；更新任何精确-键-集断言。
- **`tests/serving/test_websocket_response_hit_meta.py`**（既有，同步）：wire 响应的 `__hit_meta__` 含 `searched`（**单文件跑**，不跑整 `tests/serving`——真-websocket-server 用例会挂，见 §5）。
- **`tests/libero/test_episode_runner_collect.py`**（既有，同步/扩展）：`infer_recorder` 在 `collect_meta=None` 时从 hit_meta 填 `row["searched"]`；`collect_meta` present 时仍以其为准（N1/N4 provenance / `searched_all` 非回归）。
- **standalone `examples/libero/main.py` per-step writer**：`_rec` 的 `**hit_meta` 展开（main.py:627/882）会自然落盘新增 `searched`；确认/保持 `collect_meta` 覆盖 hit_meta 的优先级不变（若触及该路径则加断言）。
- **`tests/exp/test_analyze_n1_live.py`**（既有，扩展）：
  - inf 记账**向后兼容**（N1/N4/periodic searched=False 仍 inf=1.0）；
  - N2 盲回放 `FULL_HIT×searched=False` → inf=0（合成行）；
  - `episode_searched(follow_winner)` 用**记录** searched，**缺失/非 bool → fail-fast**（不静默当 skip/search）；
  - `analyze()` follow_winner inf-matched 配对 + 3 条件 overall（合成 N2 + periodic run）；
  - **N4 逐值回归锁**：在 Stage 3a N4 数据上分析器 render + result-manifest 与既有**逐字节一致**。
- **`tests/exp/test_run_n1_live_n4.py`** 同目录新增 follow_winner 用例：dispatch → `DEFAULT_WORKER_MODULE` + manifest 记 lock_streak/budget/gate_family=n2；缺 lock_streak/budget fail-fast。
- **`tests/exp/test_n1_gate_client.py`** / `test_n4_gate_client.py`：确认 N1/N4 客户端 searched-stamp 路径不受影响（非回归）。

---

## 4. 风险登记

| # | 风险 | 缓解 |
|---|---|---|
| R1 | `__hit_meta__` 加字段破坏既有 wire schema 断言 / 下游 | additive bool；grep 精确-键断言同步；N1/N4 客户端仍以 collect_meta 为准，行为不变 |
| R2 | analyzer inf-fix 改动既有 N1/N4/periodic 结果 | 数学上恒等（searched=False 恒 MISS→1.0）；加 Stage 3a N4 数据逐值回归锁 |
| R3 | N4 逻辑泛化引入回归 | 泛化保持 N4 输出逐字节不变，G1 审泛化方案 + N4 非回归测试 |
| R4 | N2 run 忘开 searched 记录 → 分析器 fail | searched 走常开 `__hit_meta__`（无需任何 flag）；`episode_searched` follow_winner 缺 bool 即 fail-fast（不静默） |
| R5 | 进入条件①（延迟档）仍是 owner 部署判断 | live 无论如何验 SR 保真；net@34 只在延迟受限档有意义，如实标注（承 4a plan §1.2 R4） |

## 5. 测试策略汇总

- 单测：interceptor hit_meta searched、episode_runner searched 双源、analyzer inf 记账 + follow_winner 配对/overall + **N4 逐值回归锁**、run_n1_live follow_winner dispatch。
- §6 Verify（`reference_pytest_manual_skip` 铁律）：裸 `uv run pytest tests/cache tests/exp tests/libero`（本次 blast-radius 含 `examples/libero/episode_runner` 改动）**外加单文件** `tests/serving/test_websocket_response_hit_meta.py`（interceptor wire schema 改动波及）。**禁**整目录 `tests/serving`（真-websocket-server 用例无 timeout 兜底会挂）/ repo-wide / `-m "not manual"` / `tests/review_tests`。
- 推理路径改动（§2.1）→ 附 staged API 测试口径（WA §2.5）。
- **本计划不含 live run**：本 plan 只交付"N2 可被 run + analyze"的使能代码；实际 Phase C 扫点 live 在此 commit 后、按 experiment-lifecycle skill 另起（(lock_streak,budget) 网格见 4a plan §4，owner 定）。

---

## Review Log

<!-- reviewer entries go here. -->

### G1 Round 1 — Reviewer — APPROVED — 2026-07-06 23:08 CDT

结论：APPROVED。该计划可进入实现；未发现需要退回的架构/接口阻塞项。

审查要点：

- 架构一致性：问题定义成立。N2 `follow_winner` 是 server-side event-driven gate，无法复用 N1/N4 的客户端 stamp，也无法像 periodic 闭式重建 searched；把 `searched` 加到常开的 `__hit_meta__` 是最小且正确的数据通路，不削弱 `export_collect_meta` 的 C5 always-search guard。
- 接口兼容：`__hit_meta__.searched` 是 additive bool 字段；`collect_meta` 存在时继续以 `collect_meta.searched` 为准，缺失时才从 hit_meta 回填，能保持 N1/N4 provenance 不变。`cp1_result is None` placeholder 设 `searched=True` 合理：cache-off/no-orchestrator 语义是每步全推理，不能被统计成 skip。
- 执行 harness：`run_n1_live.py` 对 `follow_winner` 应按 server-side gate 处理，走 `DEFAULT_WORKER_MODULE`、不发 client signal；manifest 必须记录 `lock_streak`、`budget`、`gate_family="n2"`，并对缺参 fail-fast。
- Analyzer：`searched=False` 时用 `inf_value(hit_type,start_t)` 是 N2 正确性要求，不只是清理。当前盲回放会产生 `FULL_HIT × searched=False`，继续硬编码 `+1.0` 会把不推理的 replay 算成 full inference。N1/N4/periodic 的既有 `searched=False` 数据仍是 MISS，占比应保持数值等价。
- N4 泛化：复用/泛化 `match_periodic_n4` + `n4_overall` 路线可接受；G2 必须证明 N4 已有报告/结果不回归，尤其是 inf-matched pairing、三条件 overall、render caption/table。

G2 核验要求：

- 测试保持 scoped，不跑 repo-wide/GPU。覆盖面至少包括：`tests/cache/test_interceptor_hit_meta.py`、`tests/serving/test_websocket_response_hit_meta.py`、`tests/libero/test_episode_runner_collect.py`、`tests/exp/test_analyze_n1_live.py`、`tests/exp/test_n1_gate_client.py`、`tests/exp/test_run_n1_live_n4.py`，以及新增的 follow_winner run/analyze 测试。
- 对 standalone `examples/libero/main.py` 的 per-step writer 需确认：当前 `**hit_meta` 展开会自然落盘新增 `searched`；若实现改动该路径，必须保持 `collect_meta` 覆盖 hit_meta 的优先级。
- `episode_searched(..., follow_winner, ...)` 必须对缺失/非 bool `searched` fail-fast；不能把旧数据或缺字段静默当 skip/search。
- 本 G1 不要求 live run；live 扫点应在使能代码 G2 通过并提交后另起。

### G2 Round 1 — Reviewer — APPROVED — 2026-07-06 23:47 CDT

结论：APPROVED。实现符合 G1 设计，可进入提交/后续 live 使能流程。

审查覆盖：

- `src/openpi/cache/interceptor.py`：`__hit_meta__` 增加 `searched`，`cp1_result is None` placeholder 为 `searched=True`，真实 `CheckResult.searched` 原样透传；未改变推理分支。
- `examples/libero/episode_runner.py`：`_hit_row` 从 hit_meta 记录 searched；`collect_meta` 存在时继续覆盖该字段，保持 N1/N4/collection provenance。
- `exp/gate_research/run_n1_live.py`：`follow_winner` 走 server-side/default worker 路径，无 client signal；manifest 派生 `gate_family="n2"` 并记录 `lock_streak`/`budget`。
- `exp/gate_research/analyze_n1_live.py`：inf 改为按 verdict 计算，N2 `FULL_HIT × searched=False` 正确计 inf=0；`follow_winner` 走与 N4 相同的 inf-matched periodic + 三条件 overall 路径；N1/N4 路径保持兼容。
- 测试覆盖到 hit_meta wire、runner 双源优先级、N2 analyzer/run dispatch、N4 回归。

验证结果：

- `PYTHONPATH=. uv run pytest tests/cache/test_interceptor_hit_meta.py tests/serving/test_websocket_response_hit_meta.py tests/libero/test_episode_runner_collect.py tests/exp/test_analyze_n1_live.py tests/exp/test_n1_gate_client.py tests/exp/test_run_n1_live_n4.py -q` → **113 passed, 3 warnings**。
- `.venv/bin/ruff check src/openpi/cache/interceptor.py examples/libero/episode_runner.py exp/gate_research/analyze_n1_live.py exp/gate_research/run_n1_live.py tests/cache/test_interceptor_hit_meta.py tests/serving/test_websocket_response_hit_meta.py tests/libero/test_episode_runner_collect.py tests/exp/test_analyze_n1_live.py tests/exp/test_n1_gate_client.py tests/exp/test_run_n1_live_n4.py` → **All checks passed**。
- Stage3 N4 analyzer byte-regression：复跑 11 个 Stage3 manifest 到 `/tmp/stage3_n4_live.g2.md` / `/tmp/stage3_result.g2.json`，与 `exp/gate_research/data/n1_live/stage3_n4_live.md` / `stage3_result.json` `cmp -s` 均为 **0**。

非阻塞注记：

- 本轮未跑 repo-wide、整目录 `tests/serving` 或 GPU/live 测试；与本阶段 scope 和既有人工要求一致。
- 当前 plan/index 的状态文字仍偏计划态；提交前建议执行者按项目惯例把 `logs/README.md` 状态从“待 G1”更新为 G2 approved / code done，但不影响本 G2 代码放行结论。
