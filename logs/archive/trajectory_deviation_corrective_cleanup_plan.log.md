# Trajectory-Deviation Corrective Experiment — Cleanup Plan

- Status: `G1 Approved — Ready for Wave A`
- Level: **L2**（多文件结构重构，不改功能语义；需过 Understand → Plan → G1 → Code → G2 → Verify）
- Scope: post-hoc 清理 `Ziyang` 分支相对 `main` 的实验代码，目标是把"频繁妥协"的地方改清爽，不引入新能力
- G1: 经 Round 1 REQUEST CHANGES + Round 2 APPROVE（2026-04-14, reviewer: Codex）
- 关联 log：
  - 上游实验计划：[`trajectory_deviation_corrective_experiment.log.md`](trajectory_deviation_corrective_experiment.log.md)
  - 实现 + G2 + Verify 记录：[`trajectory_deviation_corrective_implementation_review.log.md`](trajectory_deviation_corrective_implementation_review.log.md)（Verify APPROVE 见 §28.9）
- 起草日期：2026-04-14

---

## 0. TL;DR

Verify 阶段 APPROVE 后，分支上沉积了三类可见妥协：**大函数**（`main()` 303 行 / `_execute_spawn_unit()` 208 行）、**跨文件样板复制**（LIBERO env 构造 3 处、conda subprocess 2 处、unit-key 格式 3 套）、**info-only 化后遗留的死参数**（`--state-atol`）。本 plan 把清理拆为 **P0 结构瘦身 → P1 抽公共 helper → P2 小修**，每步独立通过测试，累计不改变实验行为。

4 条 sub-agent 报告的"功能 bug"经人肉核实**全是假阳性**，不在本 plan 范围内（见 §3）。

---

## 1. 分支现状（核实后的事实）

| 指标 | 数值 | 文件 |
|------|------|------|
| `main()` 行数 | 约 303 (L303–L606) | `exp/run_cache_experiments.py` |
| `_execute_spawn_unit()` 行数 | 约 208 (L418–L626) | `exp/run_spawn_experiment.py` |
| `run_spawn_experiment.py` 总行数 | 799 | — |
| `run_cache_experiments.py` 总行数 | 859 | — |
| `compute_deviate_scores.py` 总行数 | 546 | — |
| LIBERO env 构造重复 | 3 处 | `scripts/verify_env_save_restore.py`, `scripts/verify_restore_obs_equivalence.py`, `exp/run_spawn_experiment.py::_SpawnCommon.make_env` |
| Conda subprocess 构造重复 | 2 处 | `exp/run_cache_experiments.py`, `exp/run_step1b_gt.py` |
| Unit-key 格式自定义 | 3 套 | `compute_deviate_scores.py`（手撕 `split`）、`run_step1b_gt.py`（有 helper）、`run_spawn_experiment.py`（五元组） |
| `state_atol` 死参数 | L100 + L202 | `scripts/verify_restore_obs_equivalence.py` |

**不在本 cleanup 范围的 follow-up**：
- F1：spawn env 当前未调用 `env.seed(...)`（`exp/run_spawn_experiment.py:256-280`），GT HDF5 里有 seed attr。是否该用 GT seed 是独立 bugfix，需独立 G1 + Verify。
- F2：Phase1/2 runner 每单元创建 client、调用 `episode_end(success=True)` 但从不 `close()`；infer 抛异常时 success 判定错误。独立 bugfix。
- F3：`run_cache_experiments.py` 的 resume 与 `--task-ids` 子集不一致时行为不定（不会补新 task，也可能跑旧 pending），独立决策。

---

## 2. 问题清单（按优先级）

### 🔥 P0 结构石山

**P0-1  `run_cache_experiments.py::main()` 过长 + 嵌套过深**
- Anchor: `exp/run_cache_experiments.py:303-604`
- 症状：resume / retry / execute 三条路径混在一个函数里，4 层嵌套；`task_progress` 累加逻辑在"resume completed"分支（L455–485）和"execute remaining"分支（L487–559）里几乎复制。
- 改法：拆成
  - `_init_and_validate_resume(state, runs) -> RunPlan`
  - `_execute_one_run(state, run_plan) -> RunResult`
  - `_finalize_run(state, result) -> None`
  - `main()` 只做 CLI → state load → 循环派发 → save
- 验收：`main` ≤ 60 行；每个子函数 ≤ 80 行；`tests/exp/test_run_cache_experiments.py` 全绿；resume、retry、干净运行三条路径各有 1 条回归 case；额外新增"resume × `--task-ids` 不一致"行为锁定测试（锁定既有行为，不修 F3——若需修必须走独立 bugfix）。

**P0-2  `run_spawn_experiment.py::_execute_spawn_unit()` 怪兽函数**
- Anchor: `exp/run_spawn_experiment.py:418-626`
- 症状：HDF5 读 → env init → prefill → rollout → finally 三层 try-except 全挤在一个函数；cleanup 顺序硬编码进函数体，单测需要 mock 过多。
- 改法：
  - `_load_gt_and_spawn_inputs(gt_path, s, n, k_idx) -> SpawnInputs`（纯读）
  - `_prefill_cache(client, spawn_inputs) -> None`（只管 prefill RPC）
  - `_run_rollout(env, client, spawn_inputs, budget) -> RolloutResult`（env loop）
  - `_execute_spawn_unit` 变成 `try: load; prefill; rollout` + `finally: cleanup_in_order(...)`
- 验收：`_execute_spawn_unit` ≤ 60 行；`tests/exp/test_run_spawn_experiment.py` 全绿；新增单测覆盖
  - `_run_rollout` 的 budget-exceeded 与 success 两个分支；
  - cleanup 边界：`make_client()` 抛异常（env 已建）、`prefill_trajectory()` 抛异常、`env.close()` 本身抛异常，三种情形下均不吞原始异常、不跳过必要 close。

**P0-3  LIBERO env 构造三处复制（V-3 根因）**
- Anchors:
  - `exp/run_spawn_experiment.py:256-280` (`_SpawnCommon.make_env`)
  - `scripts/verify_env_save_restore.py:55-81`
  - `scripts/verify_restore_obs_equivalence.py:125-138`
- 症状：三处都手撕 `get_libero_path("bddl_files") / task.problem_folder / task.bddl_file` + `OffScreenRenderEnv(...)`；V-3 漏修就是因为只有 examples 层做对了、spawn 层没同步。
- 改法：新建 `exp/_libero_env.py`：
  ```python
  def build_libero_env(
      task_suite: str,
      task_id: int,
      *,
      resolution: int = 256,
      seed: int | None = None,  # 仅 seed is not None 时调用 env.seed()
  ) -> Any: ...
  def resolve_bddl_path(task_suite: str, task_id: int) -> Path: ...
  ```
  docstring 明确：
  - `seed=None` = 不 seed（等同当前三个调用点行为）；
  - `seed` 作为形参暴露是为了与 `examples/libero/main.py::_get_libero_env` 的官方参考实现保持能力对等（后者固定调用 `env.seed(seed)`），避免 helper "把关键语义藏起来"；
  - **本 cleanup 范围内三个调用点均传 `seed=None`**，是否改 spawn 走 GT seed 属 F1 follow-up。
- 验收：
  - 三处 import 收敛到单一 helper；
  - `tests/exp/test_libero_env_helper.py` 覆盖：`seed=None` 时 `env.seed` 不被调用；`seed=7` 时 `env.seed` 被调用且仅一次；`resolution` 通过；`resolve_bddl_path` 对 libero_10 task_id=1 返回预期路径；
  - `tests/exp/` + `tests/scripts/` + manual smoke of `verify_env_save_restore.py` 通过；
  - V-3 类 regression：构造 task_1 env 不报错。

**P0-4  Conda subprocess 构造重复**
- Anchors:
  - `exp/run_cache_experiments.py:148-161`
  - `exp/run_step1b_gt.py:229-248`
- 症状：两处都手剥 `VIRTUAL_ENV / PYTHONPATH / PYTHONHOME`、重组 `PATH`、设 `MUJOCO_GL=egl`，细节略有漂移。
- 改法：新建 `exp/_subprocess.py`：
  ```python
  def build_subprocess_cmd(
      main_args: list[str],
      *,
      conda_env: str | None = None,       # None → uv run；非 None → conda run -n <env>
      extra_env: dict[str, str] | None = None,  # 在剥离后的 env 上 update（在 MUJOCO_GL 之后，允许 caller 覆盖）
  ) -> tuple[list[str], dict[str, str] | None]: ...
  ```
  注意：**`conda_env` 默认 `None`**，保持当前两个 runner 的"未显式传 --conda-env 则走 uv run"的默认路径不变。
  两个 runner 都改成 `cmd, env = build_subprocess_cmd([...], conda_env=args.conda_env)` → `subprocess.run(cmd, env=env, ...)`。
- 验收：两处调用点 ≤ 3 行 subprocess 样板；新增 `tests/exp/test_subprocess_helpers.py` 至少 5 case：
  - `conda_env=None` 返回 `(["uv", "run", *main_args], None)`；
  - `conda_env="libero_sim"` 剥 `VIRTUAL_ENV / PYTHONPATH / PYTHONHOME`；
  - venv bin 从 PATH 中移除；
  - 注入 `MUJOCO_GL=egl`；
  - `extra_env={"MUJOCO_GL": "osmesa"}` 能覆盖默认 egl；`extra_env` 未提供的 key 保持 defaults。

**P0-5  Unit-key 格式三套**
- Anchors:
  - `exp/compute_deviate_scores.py:316-317`（手撕 `split(":", maxsplit=2)`）
  - `exp/run_step1b_gt.py` 有 `build_unit_key` / `parse_unit_key`
  - `exp/run_spawn_experiment.py:85-103`（五元组 `build_unit_key` / `parse_unit_key`）
- 症状：每个 step 自造 key schema，JSON state 文件 key 漂移风险高；合规性不可测。
- 改法：新建 `exp/_unit_key.py`，**不做单一 UnitKey**（避免字段语义失真：Step1b 的首字段是 `task_id` 不是 cache config）。改成三个独立的 frozen dataclass，各自 encode/decode：
  ```python
  @dataclass(frozen=True)
  class Step1bKey:
      task_id: int
      init_idx: int
      def encode(self) -> str: ...
      @classmethod
      def decode(cls, key: str) -> "Step1bKey": ...

  @dataclass(frozen=True)
  class DeviateKey:           # Phase1 / Phase2 共用
      cfg: str
      ep: str
      sample_idx: int | None = None   # Phase1 有；Phase2 为 None
      def encode(self) -> str: ...
      @classmethod
      def decode(cls, key: str) -> "DeviateKey": ...

  @dataclass(frozen=True)
  class SpawnKey:
      cfg: str
      ep: str
      s: int
      n: int
      k_idx: int
      def encode(self) -> str: ...
      @classmethod
      def decode(cls, key: str) -> "SpawnKey": ...
  ```
  对应 runner：
  - `exp/run_step1b_gt.py` 现有 `build_unit_key` / `parse_unit_key` 替换为 `Step1bKey.encode/decode`；
  - `exp/compute_deviate_scores.py` 的 Phase1/Phase2 `split(":", maxsplit=2)` 替换为 `DeviateKey.decode`；
  - `exp/run_spawn_experiment.py:85-103` 的 `build_unit_key` / `parse_unit_key` 替换为 `SpawnKey.encode/decode`。
- 验收：
  - 三个 helper 各 ≥ 4 case：encode/decode round-trip、invalid 报错、边界（如 `Phase2 DeviateKey(sample_idx=None)` 和 `Phase1 DeviateKey(sample_idx=0)` 互不冲突）；
  - 迁移后 **新写出的** state JSON key 字符串与旧格式逐字节相等（通过固定输入 dump 比对 golden）；
  - **不做 legacy decoder**（D2）：迁移前将 `data/deviation_experiment/` 下现存 dry-run state 备份到 `data/deviation_experiment/_pre_cleanup/<timestamp>/`，新 schema 不回读旧备份；Step 1a 真 run 的 state（若存在）走同样归档。

### ⚠️ P1 清爽化

**P1-1  `BaseRunState.run()` vs `parallel_run()` 80% 重复**
- Anchor: `exp/_run_state_base.py:160-202` vs `204-255`
- 改法：抽 `_run_batch_impl(units, *, executor=None)`，`run()` 传 `None`（同步循环），`parallel_run()` 传 `ThreadPoolExecutor`。retry loop、filter、失败处理只写一遍。
- 验收（**不以行数为标**）：
  - 现有测试全绿；
  - 新增行为测试：锁定 serial 模式下"每个 unit 改 pending 后立即 save"，parallel 模式下"批量改 pending 后只 save 一次，再由 `_execute_one()` 逐个写 running/done"——两种时序在重构前后必须一致；
  - `unit_filter` 作用域、`retry_count` 单调递增、失败 unit 最终状态（`failed` vs `done(success=False)`）、parallel retry 的 save/load 可恢复性均需 case 覆盖。

**P1-2  Phase1Runner / Phase2Runner 结构重复**
- Anchor: `exp/compute_deviate_scores.py:293-390`
- 改法：合并成 `_PhaseRunner(phase_type)`，差异点（M/config_prefix/filename prefix）做构造参数。若某种差异难以参数化（如 build_units 的目标列表），保留 `build_units` 由子类 override，但 `execute_unit` 应共享。
- **Scope 声明**：本轮**只合并结构**，**不修 client lifecycle**。现存行为（每单元建 client、`episode_end(success=True)`、不 `close()`、`infer` 抛异常时 success 判定错误）在合并后逐字节保持。lifecycle 修复登记为 **F2 follow-up**，独立 G1+commit。
- 验收：两个 runner 合计代码从约 100 行压到 ≤ 60 行；`tests/exp/test_compute_deviate_scores.py` 全绿；新增一条 regression 锁定 "合并前后 `execute_unit` 在 infer 成功路径下的 client / episode_end 调用序列一致"。

**P1-3  `--state-atol` 对齐 V-2 决议**
- Anchor: `scripts/verify_restore_obs_equivalence.py:100, 202`
- 背景：V-2 决议（`logs/trajectory_deviation_corrective_implementation_review.log.md`）明确"保留 CLI 接口以向后兼容，运行期不参与断言"。删除 flag 等于推翻既有决议。
- 改法（对齐 V-2）：**保留** `state_atol` 形参与 CLI 定义；`--help` 追加标注 `(deprecated; info-only, no assertion)`；确保代码路径上不再有任何 `assert`/`raise` 依赖该阈值（当前已 info-only，仅补文档）。
- 验收：`--help` 仍显示 `--state-atol`，描述含 "deprecated" / "info-only"；`tests/scripts/test_verify_smoke_scripts.py` 全绿；grep 确认无 `state_atol` 出现在 `assert` / 条件分支（只在 log / info 输出）。

**P1-4  `aggregate_spawn_results` 魔法字符串（独立 commit，非 misc）**
- Anchor: `exp/run_spawn_experiment.py:654`（`strategy = name.split("_")[2] if name.count("_") >= 2 else "unknown"`）
- 约束：**不得**在 state JSON 顶层写 `__meta__`——`BaseRunState.load()` (`exp/_run_state_base.py:104-108`) 会对顶层每个 key 调 `UnitState(**v)`，注入非 unit key 会直接 `TypeError`；且 `BaseRunState.save()` 只从 `self.units` 重写 JSON，会丢 meta。
- 改法：走 **sidecar 文件** `spawn_state_<config>.meta.json`：
  - spawn runner 启动时一次性写入 `{"strategy": ..., "config": ...}`；若文件已存在且内容不符则报 mismatch；
  - `aggregate_spawn_results(state_path)` 先找同目录 sidecar，存在则读；不存在则 fallback 到文件名解析 `split("_")` + 一次 `logger.warning("no sidecar, falling back to filename parse")`；
  - `BaseRunState` 保持不动（本 cleanup 不扩展 base class）。
- 独立性：P1-4 **单独一个 commit**（`cleanup/09`），不与 P1-5 / P2-* 混装——触碰 state 持久化语义。
- 验收：新增 `tests/exp/test_spawn_metadata_sidecar.py` ≥ 4 case：写入幂等、mismatch 报错、aggregate 优先读 sidecar、sidecar 缺失时 fallback 且产生 warning；`BaseRunState.load()` 在存在 sidecar 的目录下读 state 不受影响（sidecar 是独立文件，不是 state key）。

**P1-5  `_BaseSpawnRunner._iter_targets()` 缺 `@abstractmethod`**
- Anchor: `exp/run_spawn_experiment.py:305-312`
- 改法：加 `@abstractmethod` + `from abc import ABC, abstractmethod`；`_BaseSpawnRunner(BaseRunState, ABC)`。
- 验收：直接实例化 `_BaseSpawnRunner` 立刻抛 `TypeError`；子类 `SpawnRunner` / `BaselineRunner` 无需改动。

### 🔎 P2 小修（一次性，不拆 PR）

**P2-1  prefill_trajectory 错误 ack 风格不一致（独立 commit = cleanup/10）**
- Anchor: `src/openpi/serving/websocket_policy_server.py:234-238`、`packages/openpi-client/src/openpi_client/websocket_client_policy.py:113-116`
- 症状：该分支 send 裸字符串，其他 error 路径用 `{"__ack__": "error", "msg": ...}`。客户端目前能识别两种，但风格不统一，且收到 JSON error ack 后目前**不 raise**，可能把错误当成功。
- 改法（server + client + 双方测试，同一 commit）：
  - server: 错误路径统一发 `msgpack.packb({"__ack__": "error", "msg": str(exc)})`；
  - client: 收到字节流 ack 时 `unpackb`；若是 dict 且 `__ack__ == "error"`，**raise `RuntimeError(msg)`**；
  - 旧 server 兼容：若收到裸字符串（非 msgpack 可解码），`logger.warning("legacy string error ack from server: %s", s)` 后同样 raise。
- 独立性：**单独一个 commit**（`cleanup/10`），不与 P1-3 / P1-4 / P1-5 混装——触碰 wire protocol。
- 验收：
  - `tests/serving/test_websocket_policy_server.py` 补 case 断言错误为 JSON 且 `__ack__ == "error"`；
  - `packages/openpi-client/.../websocket_client_policy_test.py` 补 case：client 收到 JSON error ack 时 raise；收到旧裸字符串 error ack 时 warning + raise；
  - 正常 ack 路径不受影响。

**P2-2  `dump_step1a_failed_inits.py` 失败定义不显式**
- Anchor: `scripts/dump_step1a_failed_inits.py:65-66`
- 改法：抽局部变量 `is_failed = not r.get("success", False)` 让意图直白。
- 验收：手眼 review pass；无行为变化。

**P2-3  `pathlib.Path` vs `Path` 风格统一**
- Anchor: `scripts/verify_env_save_restore.py:62`
- 改法：`from pathlib import Path` 后去掉 `pathlib.` 前缀。
- 验收：脚本行为不变。

---

## 3. 已排除的假阳性（sub-agent 误报，留存归档）

这 4 条在 post-hoc review 中由 sub-agent 报告为 P0 功能 bug，经人肉核实**均为误读**，**不列入本 plan**。保留此记录是为了避免后续被再次拉起来讨论。

| 报告项 | Anchor | 核实结论 |
|--------|--------|----------|
| `examples/libero/main.py::_run_episode` 返回值 3→5 元组未传播 | L443、L629 两处调用 | 两处调用全是 5 元组解包（`_run_episode\(` grep 全仓只有这两处），sub-agent 把其他函数的调用点记错。✅ 无 bug |
| `collection_policy.on_episode_start` forward 异常会吞 `_collecting=True` | `src/openpi/collect/collection_policy.py:115-144` | 实际顺序是：`_prompt_captured=False` → `_collector.on_episode_start` → `_collecting=True` → forward 到 inner policy。`_collecting=True` 在 forward **之前**就设上了，forward 抛异常不影响此 flag。✅ 无 bug |
| `websocket_policy_server.prefill_trajectory` 错误路径双 send | `src/openpi/serving/websocket_policy_server.py:233-251` | L239 有显式 `continue`，错误路径只 send 一次字符串、跳过后续 ack。✅ 无 bug（仅风格不一致，收在 P2-1） |
| `cache/interceptor.prefill_trajectory` finally 吞异常 | `src/openpi/cache/interceptor.py:377-386` | finally 只 call `exit_prefill_mode()`，无 `return` / `raise` 覆盖原异常；Python 语义下 try 块的异常会继续冒泡。✅ 无 bug |

---

## 4. Gate 策略（Working Agreement §2+§4）

本 cleanup 是 **L2** 结构重构：
- **G1（Plan 审阅）**：本文档即 Plan 输出。建议 spawn 一个独立 review sub-agent 质询本 plan 的拆分是否最小、是否漏掉真正的功能风险、验收标准是否可验证。若 reviewer 对任一 P0 项提出"应拆得更细 / 应合并 / 不该动"的实质反馈，必须在进入 Code 之前回本文档迭代。
- **G2（Code 审阅）**：每个 P0 项单独 commit + 独立 review sub-agent 做 diff review，重点在"是否引入行为差异"。P1/P2 可合并成 1-2 个 commit 一起过 G2。
- **Verify**：
  1. 全仓跑 `.venv/bin/python -m pytest tests/ -q`（含新增测试）。
  2. 针对 `run_cache_experiments.py` 跑一次 dry-run：`clip_w7_d4` 单 config、2 episodes、`--runs 0-1`，断言 state JSON 和 cleanup 前一致。
  3. 针对 `run_spawn_experiment.py` 复跑 Step G dry-run（规模同 §28.7），断言 aggregate CSV 18/18 done 与清理前一致。

---

## 5. 执行顺序建议（每步独立 commit，单独过 G2）

```
Wave A（helpers 抽取）
cleanup/01   extract exp/_libero_env.py                  (P0-3)   low risk, unblocks tests
cleanup/02   extract exp/_subprocess.py                  (P0-4)   low risk
cleanup/03a  add exp/_unit_key.py helpers + tests only   (P0-5)   no callsite change
cleanup/03b  migrate callsites to *Key.encode/decode     (P0-5)   medium, touches 3 runners

Wave B（大函数拆分）
cleanup/04   split run_cache_experiments::main           (P0-1)   medium, behavioral-invariant
cleanup/05   split _execute_spawn_unit                   (P0-2)   medium

Wave C（清爽化 + 独立语义改动）
cleanup/06   _run_state_base dedupe                      (P1-1)   low
cleanup/07   PhaseRunner merge（structure only, no lifecycle fix）  (P1-2)  low
cleanup/08   state_atol 对齐 V-2 + abstractmethod + misc  (P1-3 / P1-5 / P2-2 / P2-3)  small
cleanup/09   spawn metadata sidecar                      (P1-4)   独立 wire/state 语义，独立 commit
cleanup/10   prefill error ack protocol (server+client)  (P2-1)   独立 wire protocol，独立 commit
```

每个分支结束前必须：
- `.venv/bin/python -m pytest <affected dirs> -q` 全绿；
- `git diff` 逐行自检"是否偷偷改了行为"；
- 若触及 resume / retry 路径，手跑一次真 rollout dry-run 对比 state JSON。

---

## 6. 回归风险清单

| 风险 | 影响 | 缓解 |
|------|------|------|
| `*Key` 迁移破坏旧 state JSON 向后兼容 | Phase1/Spawn resume 断链 | D2 策略：旧 dry-run state 归档到 `_pre_cleanup/`，不回读；cleanup/03b 逐调用点迁移 + 每次 dump 对比新 key 字符串 byte-equal；不写 legacy decoder |
| `_libero_env` 分辨率 / seed 默认值变 | smoke 脚本 obs 尺寸偏移 / object position 漂移 | helper 签名 `resolution: int = 256, seed: int | None = None`；三个调用点保持原默认（verify_env_save_restore 默认 128、verify_restore 默认 LIBERO_ENV_RESOLUTION、spawn 默认 256），seed 全部传 `None`（等同当前行为）；F1 跟进 |
| `_subprocess` 抽取丢 env 变量或改默认路径 | step1b 子进程找不到 libero / uv run 被切成 conda | helper 单测覆盖 `conda_env=None` 返回 uv；VIRTUAL_ENV/PYTHONPATH/PYTHONHOME 剥离；先给 step1b 切，跑一次 Step 1b GT smoke |
| `main()` 拆分漏传 state 或改变 resume × --task-ids 行为 | resume 行为变化 | 每个子函数签名显式、无全局状态；`tests/exp/test_run_cache_experiments.py` 加 end-to-end resume 测试 + resume × task-ids 不一致行为锁定 |
| P1-2 合并 PhaseRunner 改了 output 文件名 / 触碰 client lifecycle | Step F 结果读不出 / 掩盖 F2 bug | 保留 filename 前缀作构造参数；scope 声明"只合并结构不修 lifecycle"；regression 锁定 `execute_unit` 调用序列不变 |
| P1-4 sidecar 与旧 state JSON 共存 | aggregate 读到不一致 meta | sidecar mismatch 时报错；aggregate 缺 sidecar 时 fallback 到 filename + warning；tests 覆盖两种路径 |
| P2-1 wire protocol 改动使旧 client 读不到 error | silent success 误判 | client 侧同一 commit 更新 JSON error raise + 裸字符串兼容 raise；test 覆盖双路径 |

---

## 7. 关键决策

- **D1**  不把 `examples/libero/main.py::_get_libero_env` 并入 `exp/_libero_env.py`：`examples/` 保持自包含。`exp/_libero_env.py` 通过形参 `seed: int | None = None` 暴露 seed 能力，但三个 cleanup 范围内调用点本轮一律传 `seed=None`（行为守恒）；spawn 是否该用 GT seed = **F1 follow-up**。
- **D2**  三个 `*Key` helper **不写 legacy decoder**。cleanup/03b 落地前，把 `data/deviation_experiment/` 下现存 `spawn_state_*.json` / `phase*_state_*.json` / `step1b_*_state_*.json` 归档到 `data/deviation_experiment/_pre_cleanup/<timestamp>/`，新 schema 不回读。验收锁定"新写出 key 字符串 byte-equal 现有格式"。
- **D3**  P2-1 独立 commit（`cleanup/10`）。server send、client parse、双方测试在同一个 commit 内，避免 server/client 中间态版本错配。
- **D4**  十个 commit 分三波：Wave A = cleanup/01/02/03a/03b；Wave B = cleanup/04/05；Wave C = cleanup/06/07/08 + cleanup/09（sidecar，独立 commit）+ cleanup/10（wire protocol，独立 commit）。每波之间等 G2；Wave A 完成后做 §28.7 dry-run 快照作 Wave B/C 基线。
- **D5**  §3 四条假阳性就地归档，不做补丁。

---

## 8. G2 Watchpoints（每个 commit 的 G2 必须覆盖）

1. `tests/exp/test_libero_env_helper.py` **不得**依赖真实 LIBERO 安装——通过 monkeypatch/stub 锁定 `get_benchmark_dict`、`get_libero_path`、`OffScreenRenderEnv`，在普通 uv venv 即可跑。
2. cleanup/03b 的"key 字符串 byte-equal"必须用**固定输入 golden 测试**，不能只做 encode/decode round-trip（round-trip 只证自洽）。
3. cleanup/05 拆 `_execute_spawn_unit` 时，`env = common.make_env(...)` 的 try/finally 边界逐行审：保留"env 创建失败无 cleanup / client 创建失败关 env / prefill/rollout 失败 episode_end + client.close + env.close"三段可观察行为。
4. cleanup/09 sidecar 命名必须同时覆盖 top-k 与 baseline state：`spawn_state_<cfg>.json` 与 `spawn_state_<strategy>_<cfg>.json` 都能找到各自 meta；不得把 baseline config 解析成包含 strategy 的字符串。
5. cleanup/10 client 兼容旧裸字符串错误时，不能把所有 `str` response 都判为 legacy error；若当前协议仅错误路径返回 str，在注释里写明该不变量；若未来有正常 str，需要更窄判定。

---

## Appendix D — G2 Review Round 1（2026-04-14）

### D.0 Verdict

**REQUEST CHANGES**。

本轮代码整体方向与 G1 plan 对齐，定向测试也通过；但有两处实现与 plan 中已经写死的验收条件不一致，属于 G2 阻塞项。修完后再进入下一轮 G2。

已执行验证：

```bash
uv run python -m pytest tests/exp/test_unit_key.py tests/exp/test_libero_env_helper.py tests/exp/test_subprocess_helpers.py tests/exp/test_run_state_base.py tests/exp/test_run_spawn_experiment.py tests/exp/test_compute_deviate_scores.py tests/serving/test_websocket_policy_server.py packages/openpi-client/src/openpi_client/websocket_client_policy_test.py -q
uv run python -m pytest tests/scripts -q
git diff --check
```

结果：
- 113 passed；
- 12 passed；
- `git diff --check` 无输出。

### D.1 阻塞问题

#### D.1.1 cleanup/06 改变了 serial retry 的持久化节奏

- Anchor: `exp/_run_state_base.py:227-240`
- Plan anchor: §2 P1-1 验收明确要求锁定 serial 模式下“每个 unit 改 pending 后立即 save”，parallel 模式下“批量改 pending 后只 save 一次”。这是行为锁，不是可自由优化项。
- 当前实现：`_run_impl()` 不区分 serial / parallel，所有 retry unit 都先改成 `pending`，然后统一 `self.save()` 一次，再 dispatch。
- 影响：
  - serial `run()` 的 crash-resume 中间态变了。cleanup 前如果在第一个 failed unit 改 pending 后崩溃，state 只暴露已重排的那一个；现在会一次性把所有 failed unit 都写成 pending。
  - plan 的“行为不变”验收被实现注释反向覆盖了：`exp/_run_state_base.py:207-212` 写成“per-unit flip+save was collapsed”，但 G1 没批准这个行为变更。
- 建议：
  - `_run_impl()` 在 `executor is None` 时保留旧 serial retry loop：逐个 `retry_count = attempt`、`status = "pending"`、`save()`、`_execute_one(u)`。
  - `executor is not None` 时保留当前 batch save。
  - 补测试：用 save spy 锁定 serial retry 每个 failed unit pending 后都会单独 save；parallel retry 只在批量 pending 后 save 一次。

#### D.1.2 cleanup/09 sidecar 写入没有 mismatch 保护

- Anchor: `exp/run_spawn_experiment.py:289-293`
- Plan anchor: §2 P1-4 写明“若文件已存在且内容不符则报 mismatch”；§6 风险清单也写明“sidecar mismatch 时报错”。
- 当前实现：`_write_meta_sidecar()` 无条件 `write_text()`，会直接覆盖已有 sidecar。
- 影响：
  - 如果同一个 `spawn_state_*.json` 被错误地用不同 `strategy` / `config` 重新构造 runner，旧 sidecar 会被静默改写，`aggregate_spawn_results()` 之后会把历史 state 归到新的 meta 下。
  - 这正是 P1-4 试图避免的“state 与 meta 不一致”风险。
- 建议：
  - 写入前若 `meta_path.exists()`，先读 JSON；内容等于 payload 则 no-op/可重写，内容不同则 `raise RuntimeError` 或 `ValueError`，错误信息带 `meta_path`、existing、expected。
  - 补测试：预置冲突 sidecar 后构造 `SpawnRunner` / `BaselineRunner` 必须 raise；同内容 sidecar 构造必须通过。

### D.2 非阻塞建议

1. `exp/_subprocess.py:42-47` 对 `conda_env=None + extra_env` 选择 raise，超出 G1 plan 中“extra_env 在剥离后的 env 上 update”的描述。当前无调用点触发，不阻塞；建议在 docstring 或 plan 中明确这是有意限制，避免后续 caller 以为 uv-run path 也能注入 env。
2. `exp/compute_deviate_scores.py:344-349` 用 `assert` 校验 key shape；若未来用 `python -O` 跑，assert 会被移除。当前 key 都由 runner 自己生成，不阻塞；若要让 decoder 失败更稳定，建议改成显式 `ValueError`。
3. `tests/exp/test_run_state_base.py` 新增了 parallel/retry 覆盖，但还没覆盖 §2 P1-1 最关键的 save cadence。建议把 D.1.1 的 spy 测试作为 cleanup/06 的主验收。

### D.3 已通过的 G2 Watchpoints

- cleanup/01：`tests/exp/test_libero_env_helper.py` 使用 `sys.modules` stub，不依赖真实 LIBERO；`seed=None` / `seed=7` / resolution / bddl path 均覆盖。
- cleanup/02：`build_subprocess_cmd()` 保持 `conda_env=None -> uv run`，conda env stripping、`MUJOCO_GL`、`extra_env` 覆盖均有测试。
- cleanup/03a/03b：`Step1bKey` / `DeviateKey` / `SpawnKey` 拆分合理，新增 golden 测试锁定新写出 key 字符串与旧格式 byte-equal。
- cleanup/05：`_execute_spawn_unit()` 拆分后资源边界基本清楚，新增测试覆盖 client 创建失败关 env、rollout 异常不被 env.close 异常掩盖、budget exhausted 返回失败结果。
- cleanup/07：`Phase1Runner` / `Phase2Runner` 合并保持 success path 调用序列，未顺手修 lifecycle，符合 G1 scope。
- cleanup/08：`--state-atol` 保留并标注 deprecated/info-only；`_BaseSpawnRunner` 抽象化测试覆盖；misc 改动无行为风险。
- cleanup/10：server 统一 msgpack error ack，client 对 JSON error ack 和 legacy bare string 都 raise；双方测试覆盖。

### D.4 下一步裁决

请实现方先修 D.1.1 和 D.1.2，并补对应测试。修复后 G2 Round 2 重点只需复查：

1. `exp/_run_state_base.py` retry save cadence 是否与 §2 P1-1 一致；
2. `exp/run_spawn_experiment.py` sidecar mismatch 是否会阻止静默覆盖；
3. 新增测试是否能在普通 `uv` venv 下稳定运行。

---

## Appendix E — G2 Review Round 2（2026-04-14）

### E.0 Verdict

**APPROVE — G2 通过**。

Round 1 的两个阻塞项均已修复，新增测试能覆盖对应不变量。当前代码可以进入后续 Verify / commit 整理阶段。

已执行验证：

```bash
uv run python -m pytest tests/exp/test_unit_key.py tests/exp/test_libero_env_helper.py tests/exp/test_subprocess_helpers.py tests/exp/test_run_state_base.py tests/exp/test_run_spawn_experiment.py tests/exp/test_compute_deviate_scores.py tests/serving/test_websocket_policy_server.py packages/openpi-client/src/openpi_client/websocket_client_policy_test.py tests/scripts -q
uv run python -m pytest tests/ packages/openpi-client/src/ -q
git diff --check
```

结果：
- 131 passed；
- 518 passed, 1 skipped；
- `git diff --check` 无输出。

### E.1 Round 1 阻塞项复核

#### E.1.1 D.1.1 serial retry save cadence

- Anchor: `exp/_run_state_base.py:227-247`
- 裁决：**ACCEPT**。
- 复核：
  - `executor is None` 时已恢复 serial 旧语义：每个 failed unit 逐个 `retry_count = attempt`、`status = "pending"`、`save()`、`_execute_one(u)`。
  - `executor is not None` 时仍保持 parallel batch 语义：先批量 flip，再 `save()` 一次，再 `_dispatch()`。
  - docstring 已改成明确说明 serial/parallel save cadence 非对称，避免后续重构再次误合并。
- 测试：
  - `tests/exp/test_run_state_base.py::test_serial_retry_saves_per_unit_flip` 覆盖 serial 的 per-unit pending save。
  - `tests/exp/test_run_state_base.py::test_parallel_retry_batch_flips_before_single_save` 覆盖 parallel 的 batch pending save。

#### E.1.2 D.1.2 sidecar mismatch guard

- Anchor: `exp/run_spawn_experiment.py:289-315`
- 裁决：**ACCEPT**。
- 复核：
  - sidecar 不存在时写入 `{"strategy": ..., "config": ...}`。
  - sidecar 已存在且内容相同时 no-op，支持 resume / runner 重建。
  - sidecar 已存在但内容不同时抛 `ValueError`，阻止静默覆盖。
  - sidecar 已存在但 JSON 不可读时抛 `RuntimeError`，阻止在无法验证一致性时覆盖。
- 测试：
  - idempotent 同内容；
  - strategy mismatch；
  - config mismatch；
  - corrupt sidecar；
  - 原有 aggregate 优先 sidecar / 缺失 fallback / corrupt fallback 测试仍通过。

### E.2 非阻塞项复核

- D.2.1：`exp/_subprocess.py` 已在 docstring 说明 `extra_env` 只支持 conda path；现有调用点无风险。**不阻塞**。
- D.2.2：`exp/compute_deviate_scores.py::_PhaseRunner.execute_unit` 已把 `assert` 改为显式 `ValueError`，`python -O` 下仍会 fail loud。**ACCEPT**。
- D.2.3：save cadence spy 测试已随 D.1.1 补齐。**ACCEPT**。

### E.3 仍建议后续清理

1. `tests/exp/test_run_state_base.py` 中 `fail_once` 相关注释写成“primary + retry-1 失败，retry-2 成功”，但 `_Runner.execute_unit()` 实际是 primary fail 后 retry-1 成功。断言本身正确，不阻塞；建议改注释以免误导后续 reviewer。
2. `exp/run_spawn_experiment.py::_read_spawn_meta()` 对 valid JSON 但非 dict 的 sidecar 会走到 `meta["strategy"]` 并抛 `TypeError`，不会 fallback warning。当前新增写入保护已覆盖 corrupt-on-write 场景，不阻塞；若希望 aggregate 对所有 corrupt sidecar 都温和 fallback，可把 `_read_spawn_meta()` 的异常集合扩到 `TypeError` 或先校验 `isinstance(meta, dict)`。

### E.4 Final G2 Conclusion

本轮 G2 未发现新的阻塞问题。G1 的核心验收点已由实现和测试覆盖：

- helper 抽取保持默认行为；
- key string byte-equal 有 golden 覆盖；
- spawn cleanup 边界有异常路径覆盖；
- BaseRunState retry save cadence 已恢复并锁定；
- sidecar metadata 不再静默覆盖；
- prefill error ack protocol server/client 双侧测试通过。

批准进入 Verify：全仓 pytest 已通过；仍需按 §4 Verify 跑 `run_cache_experiments.py` dry-run 与 `run_spawn_experiment.py` Step G dry-run，并确认 state / aggregate 与 cleanup 前基线一致。
