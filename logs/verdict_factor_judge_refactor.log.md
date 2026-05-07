# Verdict Factor Judge 整体重构 — 调研与重构边界

| 字段 | 值 |
|---|---|
| Status | `In Progress`（阶段 0：依赖调研 + 硬契约清单完成；阶段 1：重构方案待用户确认） |
| Level | L3（跨模块结构性变化，必走 §2 Plan + §3 G1） |
| Authority | Execution |
| 重构域 | `src/openpi/cache/components/judge.py` + `src/openpi/cache/components/factors/**` + `src/openpi/cache/components/payload_view.py` |
| 历史 logs | 已加 `old_` 前缀归档，见本文 §7 |

> 本文是"重构前调研笔记"，记录运动学 judge（CompositeJudge + 5 因子 + Composer/Normalizer + PayloadView/HistoryView）当前实现对外的所有引用边界、必须保留的硬契约，以及重构域 / 不动域划分。**不是完整 Plan**——文件级改动方案 / 测试策略 / 风险登记将在 §6 用户确认重构方向后补全，再提 G1。

---

## 1. 重构动机

当前运动学 judge 代码（B0 + B1 + B2 + dedicated_runner + phase2 redesign 5 个 commit 累计）历经多次修补，存在系统性的"石山"问题：

**P1 — 架构 / 设计层面**：
- `_DESCRIPTOR_ORIENTATIONS` 表的事实之源放在 `source_window.py`，`runtime_continuity.py` 不得不 lazy import 绕循环依赖（`runtime_continuity.py:111-114`）。
- F1a-T 用 `NotImplementedError` 当运行时 boundary 信号（fork chain），与"开发者尚未实现"的异常语义混淆，被同一个 `except` 吞掉（`runtime_continuity.py:159-164`）。
- `judge.py` 单文件 708 行：5 个 Judge 类 + 2 个 Protocol + 多 helper 全堆一起。
- `JudgeResult.factor_outputs: Optional[dict]` schema 用 docstring 描述、不能静态校验。
- `ForkPolicy` 5 个值有 4 个直接 raise NotImplementedError —— 提前抽象，YAGNI。

**P2 — 代码组织 / 重复**：
- `_VERDICT_DEBUG = os.environ.get(...)` + `_verdict_logger = logging.getLogger(...)` 在 `judge.py / runtime_continuity.py / consensus.py / composers/__init__.py` 4 处复制粘贴。
- `_apply_direction` 与 `_passes_threshold` 各自 parse `range:[lo,hi]` 字符串、每次 verdict 重 split。
- `_normalize_windows` 在 `__init__` / `describe` / validator 三处重复调用，dict↔tuple 兼容层泄漏。
- CompositeJudge `__init__` 70 行：fallback 校验 + `min_required_top_k` + collect orientations + 冲突检测 + bind 全堆 `__init__`。
- `_emit_vd` 在 `__call__` 两个分支各调一次。
- F1a `_library_eps()` 硬编码 0.01，F1b `active_eps` 是参数 —— 量纲可能错配。

**P3 — Hack 痕迹 / 小不一致**：
- `key_initial` vs `source` 表面解耦实际绑死，注释长篇辩护。
- F1a / F1b 的 `_select_library_arrays` 签名不一致（self vs 显式参数）。
- `consensus.py` 的 `_F2_CANDIDATE_LOCAL_EPS` 硬编码与 F1b `active_eps` 参数化不对称。
- DumpingJudge 每 verdict open + close JSONL（240 yaml × 100 ep × ~21 step ≈ 50w 次 IO）。
- DumpingJudge 静默吞写 JSONL 异常（`except Exception: pass`）。
- DumpingJudge `record_action` 假设 inner 4 个 judge 共享 `(self, action_chunk)` 签名 —— 文档化的隐式契约。
- F1b OnlineExtractor 路径是 no-op（仅 `factors.get(key, NaN)`），却仍走 `view.get(...)` 取整 payload。

**架构骨架不烂**（CompositeJudge pipeline / capability flags / `describe` classmethod / key contract assertion / Protocol 拆分 OnlineExtractor + OfflineWriter 都合理），**但中层全是修补痕迹**。重构目标是 keep architectural intent + clean up middle layer + remove all hack patches。

---

## 2. 重构范围

### 2.1 重构域（可改）
| 路径 | 说明 |
|---|---|
| `src/openpi/cache/components/judge.py` | 拆分 / 重组（4 层 Composite / DumpingJudge / Legacy judges） |
| `src/openpi/cache/components/factors/__init__.py` | 包级 docstring 与 re-export 清理 |
| `src/openpi/cache/components/factors/base.py` | 4 层 Protocol（Normalization / Factor / Calibration / Composer）、HistoryView |
| `src/openpi/cache/components/factors/registry.py` | 17 因子注册（新 `<desc>_<source>_<channel>` 命名） |
| `src/openpi/cache/components/factors/_descriptor_kernel.py` | 4 个共享描述子（jerk / direction / dispersion / path_length） |
| `src/openpi/cache/components/factors/runtime_continuity.py` | B1-B6 完全保留旧实现不动；B7 cut-over 改为 import-only deprecation stub（旧类构造抛 NotImplementedError 指向新类）。新 8 个 online 因子住 `factors/online.py`（B2 新建） |
| `src/openpi/cache/components/factors/source_window.py` | B1-B6 完全保留旧实现不动；B7 cut-over 改 stub。新 8 个 offline 因子住 `factors/offline.py`（B2 新建） |
| `src/openpi/cache/components/factors/consensus.py` | B1-B6 完全保留；B7 cut-over 改 stub。新 `topk_action_variance`（公式不变）住 `factors/topk.py`（B2 新建） |
| `src/openpi/cache/components/factors/online.py` / `offline.py` / `topk.py` | **新增**（B2）：17 因子新实现 |
| `src/openpi/cache/components/factors/normalization/` | **新增**（B1）：第 1 层 Normalization 子模块（Protocol + ZScoreNormalization） |
| `src/openpi/cache/components/factors/calibrations/` | **新增**（B3）：第 3 层 Calibration（PercentileRollingCalibration，无 cold-start）。**与旧 `normalizers/` 在 B1-B6 期间并存**（不是 rename）；B7 cut-over 时旧 `normalizers/__init__.py` 改 stub |
| `src/openpi/cache/components/factors/composers/__init__.py` | 第 4 层 Composer：WeightedSum / AndGate / OrGate + WarmFallback 变种（承接旧 all_nan_fallback 语义） |
| `src/openpi/cache/components/payload_view.py` | PayloadView / StoragePayloadView；ForkPolicy 砍至只剩 TRAJECTORY |
| `src/openpi/cache/__init__.py` | 必要时调整 re-export |
| `src/openpi/cache/config.py` | yaml schema 大改（4 层块 + source 必备字段）；validator 加 calibration data fail-fast 校验；`_build_judge` / `_build_inner_judge` / `_build_composer` / `_build_normalizer` 全部重写 |
| `exp/verdict_factor_judge/phase1_spec.py` / `phase2_spec.py` / `phase2_layer2_spec.py` / `generate_yamls.py` / `run_phase.py` / `per_step_log_writer.py` | **重构**：17 因子新名笛卡尔 + 新 4 层 yaml schema；详见 §6.13 |
| `exp/common/factor_postprocess.py` | **重构**：`enrich_artifact_with_factors` 用 17 因子新 registry |
| `exp/common/build_clip_cache_artifact.py` / `build_in_memory_cache_artifact.py` / `build_llm_layer_matrix.py` | `--factors-yaml` CLI 路径适配新 schema |

### 2.2 不动域（修订 2026-05-07：缩小，原 exp/ 划入 §6.13 重构域）
| 路径 | 原因 |
|---|---|
| `src/openpi/cache/orchestrator.py` | 主流程已稳，verdict 调用 / view+history 注入 / OfflineWriter merge / `_safe_call_lifecycle` 全在此；`record_action` 仍要保留（处理 strategy/gate，仅 judge 层不再消费） |
| `src/openpi/cache/interceptor.py` | `__hit_meta__["factor_outputs"]` wire 透传形状不变（含字段 rename + schema_version=2，但传输机制不变） |
| `src/openpi/cache/storage_types.py` | `CachePayload.factors: dict[str, float]` 字段名 + 类型不变；只是装的 key 模板变 |
| `src/openpi/cache/cache_storage.py` | `fetch_entry` / `library_stats` facade |
| `src/openpi/cache/backends/in_memory_backend.py` | `library_stats` load + fallback；`fetch_entry` 公开方法 |
| `src/openpi/cache/warmup_pool.py` + `src/openpi/serving/websocket_policy_server.py` | warmup → eval preload 协议已稳，新架构强化（§6.3 必备）但接口不动 |
| 所有**已落盘**旧 yaml + 旧 jsonl + 旧 pkl factors keys | 不删旧实验产物；新生成的 yaml / 新 jsonl / 新 pkl 走新命名空间，与旧并存（§6.11 决策 #11） |

### 2.3 测试侧
| 路径 | 处理 |
|---|---|
| `tests/cache/components/factors/**`（7 个文件） | 必须全部跑通；签名变化时同步 import / 构造路径 |
| `tests/cache/components/test_judge.py` / `test_dumping_judge.py` | 同上 |
| `tests/cache/test_config_factor.py` / `test_dump_config.py` / `test_factor_postprocess.py` | 同上 |
| `tests/cache/test_orchestrator_offline_writers.py` / `test_orchestrator_history.py` / `test_episode_extra_propagation.py` | 同上 |
| `tests/cache/test_artifact_roundtrip.py` / `test_cache_storage_factor_facade.py` / `test_payload_view.py` | 同上 |
| `tests/cache/test_interceptor_hit_meta.py` 等 | 验 `__hit_meta__` schema 不变 |
| `tests/serving/test_serve_policy_yaml_id_propagation.py` | 验 warmup pool 接入不变 |
| `tests/exp/verdict_factor_judge/test_phase1_spec_warmup_yamls.py` | 不动 |
| **`tests/review_tests/`** | **§1 sealed reviewer space — 重构域内代码 / 测试不读不改** |

---

## 3. 依赖矩阵

### 3.1 调用图（重构域 → 上游）

```
                   wire-level (跨进程，已落盘)
   client side  ◄──┐
   examples/libero/main.py: result["__hit_meta__"]["factor_outputs"]
   exp/verdict_factor_judge/per_step_log_writer.py: jsonl schema
                   │
                   ▼
   src/openpi/cache/interceptor.py:451-453
     读 cp1_result.factor_outputs → meta["factor_outputs"]
                   │
                   ▼
   src/openpi/cache/orchestrator.py
     line 38  : import HistoryView, LibraryStats, OfflineWriter
     line 40  : import HitType, SimilarityJudge
     line 42  : import StoragePayloadView
     line 113 : __init__ 收 library_stats / offline_writers
     line 154 : _action_history / _state_history (B1)
     line 277 : _safe_call_lifecycle 给 judge 传 extra_metadata
     line 372 : append _action_history
     line 437 : append _state_history (anchor cp policy)
     line 461 : 构 PayloadView+HistoryView 注入 judge.__call__
     line 479 : 读 judge_result.factor_outputs
     line 646 : episode-end OfflineWriter merge → entry.payload.factors
                   │
                   ▼
   src/openpi/cache/config.py (yaml → 实例化总入口)
     line 344 : _JUDGE_TYPES = {…, "composite"}
     line 565 : registry import (validator)
     line 602 : capability flag → backend 兼容性检查
     line 758 : all_nan_fallback 校验
     line 973 : _validate_composite_judge_static
     line 1369: _preload_normalizer_from_warmup_pool
     line 1462: _validate_composite_judge_state_library
     line 1464: _build_judge(cfg, library_stats=)
     line 1469: 抠 judges[cp].min_required_top_k
     line 1474: 喂 search strategy `min_top_k_hint`
     line 1725: _build_judge / _build_inner / _composer / _normalizer
                / _dump_extractors / _wrap_with_dumping_judge
                   │
                   ▼
        ┌────────────────────────┐
        │  本次重构域             │
        │  judge.py + factors/** │
        │  + payload_view.py     │
        └────────────────────────┘
                   ▲
                   │ 反向被读
   src/openpi/cache/storage_types.py       CachePayload.factors
   src/openpi/cache/cache_storage.py       fetch_entry / library_stats facade
   src/openpi/cache/backends/in_memory.py  load_artifact 加载/fallback library_stats
   src/openpi/cache/warmup_pool.py         WarmupPool.preload
   exp/common/factor_postprocess.py        enrich_artifact_with_factors
   exp/common/build_*.py (3)               --factors-yaml CLI
   exp/verdict_factor_judge/**             yaml 生成 / phase 编排
```

### 3.2 反向 import 列表（grep 结果）

#### `from openpi.cache.components.judge` import
- 公开 facade `from openpi.cache import {HitType, JudgeResult, SimilarityJudge, ThresholdJudge, AlwaysHitJudge, AlwaysWarmStartJudge}`：通过 `cache/__init__.py:58-65`。
- 直接 import `CompositeJudge` / `DumpingJudge` / `_build_factor_outputs`：tests 8+ 处、`config.py:1769, 1831`。
- 直接 import `ThresholdJudge`：tests/cache/conftest.py + 12 处测试。
- 直接 import `HitType`：interceptor.py、orchestrator.py、tests 多处。

#### `from openpi.cache.components.factors.*` import
- `base`：HistoryView / LibraryStats / OfflineWriter — orchestrator、tests、in_memory_backend lazy、factor_postprocess。
- `registry`：config validator + factor_postprocess + `_build_judge` / `_build_dump_extractors`。
- `runtime_continuity` / `source_window` / `consensus`：直接构造类的测试（test_config_factor / test_runtime_continuity / test_source_window / test_consensus）。
- `composers`：WeightedSumComposer / AndGateComposer / OrGateComposer — tests + config.py。
- `normalizers`：PercentileRollingNormalizer — tests + config.py + serving。
- `_descriptor_kernel`：tests/cache/components/factors/test_descriptor_kernel.py。

#### `from openpi.cache.components.payload_view` import
- `StoragePayloadView`：orchestrator.py:42 + tests/cache/test_artifact_roundtrip.py + tests/cache/test_payload_view.py。
- `PayloadView` / `ForkPolicy`：tests 直接构造、type-hint。

---

## 4. 必须保留的硬契约

按"违反就线上崩"程度分级。

### ❶ 跨进程 / 已落盘的契约（**绝对不动**）

| 契约 | 谁锁死它 |
|---|---|
| `JudgeResult.factor_outputs = {"raw": dict[str, float\|None], "norm": dict[str, float\|None], "score": float\|None, "sentinel": str\|None}` | 已写进 `exp/verdict_factor_judge/data/phase2_layer1/**/per_step/*.jsonl`；client `__hit_meta__` |
| `result["__hit_meta__"]["factor_outputs"]` 透传 | examples/libero/main.py:230-237 |
| `CachePayload.factors: Optional[dict[str, float]]` 字段名与类型 | 已 pickle 进 `libero_spatial/*.pkl` 等 6 份 artifact |
| 因子 registry 名：`f1a_a / f1a_t / f1b_a / f1b_t / f2` | 所有 yaml `factors[].type` 引用，含 24 + 240 phase2 yaml |
| Descriptor key 命名模板：`f1a_<a\|t>_<desc>` / `f1b_<a\|t>_<desc>__p<p>_f<f>` / `f2_var` | warmup → eval normalizer preload 通过 key 串联（`PercentileRollingNormalizer.preload_buffer` 失败 raise KeyError） |
| `LibraryStats` dataclass 字段名：`action_sigma / action_active_mask / state_sigma / state_active_mask` | pkl artifact 字段 + ProcessPool IPC `_detach_entries` |
| yaml schema：`FactorConfig{type, params}` / `ComposerConfig{type, weights, per_factor_thresholds, tier_thresholds, directions, warm_start_t}` / `NormalizerConfig{type, window_size, cold_start_strategy}` / `AllNanFallbackConfig{type, start_t}` / `DumpConfig{path, config_id, factors, deferred}` / `JudgeConfig{type, threshold, factors, composer, normalizer, all_nan_fallback, export_factor_outputs, dump}` | config.py + 全部 yaml |
| `_JUDGE_TYPES` 包含 `"composite"` | yaml load-time validator |
| `composer.type ∈ {weighted_sum, and, or}` / `normalizer.type ∈ {percentile_rolling}` / `all_nan_fallback.type ∈ {miss, warm_start}` | yaml 文本 |
| 4 个描述子名 + orientation 表：`jerk: risky / dir: safe / curv_radius: non_monotonic / cum_disp: non_monotonic` | yaml `directions` 配置 + composer flip 语义 |

### ❷ 模块边界（**可重组但 import path / 名字要稳**）

| 契约 | 引用方 |
|---|---|
| `from openpi.cache import HitType, JudgeResult, SimilarityJudge, ThresholdJudge, AlwaysHitJudge, AlwaysWarmStartJudge` | `cache/__init__.py:58-65` + tests 30+ 处 |
| `from openpi.cache.components.judge import CompositeJudge, DumpingJudge` | tests + config.py |
| `from openpi.cache.components.factors.{base, registry, consensus, runtime_continuity, source_window, _descriptor_kernel}` | tests + config.py + factor_postprocess |
| `from openpi.cache.components.factors.composers import WeightedSumComposer, AndGateComposer, OrGateComposer` | config.py + tests |
| `from openpi.cache.components.factors.normalizers import PercentileRollingNormalizer` | config.py + warmup_pool 链 + tests |
| `from openpi.cache.components.payload_view import StoragePayloadView, ForkPolicy, PayloadView` | orchestrator.py + tests |

> 拆 `judge.py` 时若新增子模块（如 `composite_judge.py` / `dumping_judge.py`），必须在 `judge.py` 保留 facade re-export，否则 30+ test import 全坏。

### ❸ 类与方法签名（**可 rename 但要全局同步**）

| 契约 | 引用方 |
|---|---|
| `SimilarityJudge.__call__(results, checkpoint_id, cached_data, *, view=None, history=None)` | Orchestrator.check 注入 |
| `Judge.min_required_top_k` 属性 | config.py:1469 |
| `Judge.on_episode_start(extra_metadata=None)` / `Judge.record_action(action_chunk)` | Orchestrator `_safe_call_lifecycle` + broadcast_action |
| OnlineExtractor 类属性 `requires_library_stats / requires_chain_walk / required_top_k / descriptor_orientations` + classmethod `describe(params)` | config.py validator + `_build_judge` + `_build_dump_extractors` |
| `OnlineExtractor.extract(results, view, history, cached_data)` | CompositeJudge + DumpingJudge |
| `OfflineWriter.required_payload_fields()` / `compute_for_episode(entries, library_stats)` | Orchestrator line 651 + factor_postprocess |
| `Composer.bind_orientations(dict)` / `compose(norm, *, winner_id)` | CompositeJudge `__init__` + `__call__` |
| `Normalizer.bind_keys(list)` / `__call__(raw)` / `on_episode_start()` / `preload_buffer(values)` | CompositeJudge + warmup pool |
| `PayloadView.{get, get_entry, get_many, walk_prev, walk_next}` | F1a-T 用 walk_next + get_entry，F1b 用 get，F2 用 get_many |
| `LibraryStats.compute_from_entries(entries, active_eps_action, active_eps_state)` classmethod | factor_postprocess + in_memory_backend fallback + 三个 build_*.py |
| `CacheStorage.fetch_entry(id)` / `CacheStorage.library_stats` facade | StoragePayloadView + config.py |
| `judge_result.factor_outputs` 字段 | Orchestrator → CheckResult.factor_outputs → Interceptor → `__hit_meta__` |
| `CheckResult.factor_outputs` 字段 | Interceptor |

### ❹ 可以放心改（**只在重构域内**）

- `_descriptor_kernel.py` 内部实现（kernel 函数体、NaN 路径细节）。
- `_RuntimeContinuityBase` / `_SourceWindowSmoothnessBase` 私有继承结构。
- `_DESCRIPTOR_ORIENTATIONS` 表的位置（**注**：表内容是 ❶ 契约的一部分，但表的住址不是；可移到 `_descriptor_kernel.py` 或 `base.py`，去 lazy import）。
- `_VERDICT_DEBUG` env-var 复制粘贴（抽 `factors/_debug.py`）。
- F1a `_library_eps()` / F1b `active_eps` 内部不一致（统一处理）。
- CompositeJudge `__init__` 拆 helper（验证 / collect / bind 分函数）。
- DumpingJudge file handle 持有 / 异常 log。
- `_apply_direction` / `_passes_threshold` 内部抽象（提前 parse direction）。
- F1a / F1b 之间 `_select_library_arrays` 签名统一。
- ForkPolicy 砍占位值（**注**：先确认 yaml/test 没引用 `FIRST/STOP/ALL_BRANCHES/SCORE` 再砍；`TRAJECTORY` 是 `walk_*` 默认值，必须保留）。
- judge.py 拆为 `composite_judge.py` / `dumping_judge.py` / `legacy_judge.py`（保留 `cache.components.judge` 作为 facade re-export）。

---

## 5. 影响排查中的两个潜在坑

1. **`tests/review_tests/`**：`tests/` 目录下存在但 `protocols/execution_authority.md` §1 sealed reviewer space — 重构域内代码 / 测试不读不改。重构提交后 G2 reviewer 会用其中独立测试探针验证；这意味着 ❶❷❸ 的硬契约严守，否则 G2 reviewer 的测试会失败。

2. **跨 commit 的 artifact**：
   - `exp/warm_start/data/`（500 ep × 3 cfg 的 baseline）。
   - `libero_spatial/*.pkl`（带 168 keys 的 F1b factors，phase2 layer1/layer2 复用）。
   - Phase 2 Layer 1 的 `per_step/*.jsonl`（已含 `factor_outputs` 字段）。
   - Phase 2 Layer 2 的 240 yaml + 240 sibling warmup yaml。

   重构不能改 key 命名 / `payload.factors` schema / `__hit_meta__` schema —— 否则需要重跑全部数据。

---

## 6. 重构后目标架构（用户决策 2026-05-07）

> 用户已选定动机 (b)「借机引入新设计能力」。本节是用户口述需求的逐字落档，作为后续 Plan §8-§12 展开的基础。新架构会破坏原 §4 ❶ 列出的若干已落盘契约 — 影响在 §7 单独评估。

### 6.1 因子集合扁平化为 17 个独立因子

不再有"先选因子家族（F1a/F1b）再选 descriptor"的两层结构。每个 descriptor 直接就是一个因子身份。

**4 个核心运动学 descriptor**（数学公式与现在 `_descriptor_kernel.py` 完全相同）：
- `jerk` (risky)
- `direction` (safe，原 `dir`)
- `dispersion` (non_monotonic，原 `curv_radius`)
- `path_length` (non_monotonic，原 `cum_disp`)

**4 个变体维度**（笛卡尔乘起来 = 4 × 4 = 16 个运动学因子）：
| 变体维度 | 取值 | 含义 |
|---|---|---|
| 数据来源 | `online` / `offline` | online = verdict 时算（拼 history + winner / chain walk）；offline = artifact build / episode-end 算（写 `payload.factors`） |
| 数据通道 | `action` / `robot_state` | 因子计算用 action 序列还是 state 序列 |

**第 17 个因子**：`Top-K action variance`（即原 F2），公式与现在 `consensus.py` 完全相同（candidate-local active mask + per-DOF variance + 平均）。

**因子总数**：4 desc × 2 source × 2 channel + 1 = **17 个独立因子**。

**命名约定**（待 Plan 阶段定稿；下面是占位提案，最终在 Plan §9 锁定）：
- `<desc>_<source>_<channel>`，例如 `jerk_online_action` / `direction_offline_state` / `dispersion_online_action` / `path_length_offline_state`。
- Top-K variance：单独命名，如 `topk_action_variance` 或保留 `f2_var`（待定）。

### 6.2 Judge 4 层正交架构

```
┌───────────────────────────────────────────────────────┐
│ 第 1 层：Normalization                                 │
│   输入：原始 action / robot_state（per-step）          │
│   动作：z-score per-DOF（除以 σ，限 active subspace） │
│   输出：normalized action / state（沿用现 F1a/F1b 的    │
│         z-score 公式，不变）                            │
│   父类：Normalization Protocol                         │
│   子类示例：ZScoreNormalization（默认）                │
│   校准数据：σ + active_mask（来自 §6.3 两条路其一）    │
└────────────────────────┬──────────────────────────────┘
                         │ 归一化后的 action/state
                         ▼
┌───────────────────────────────────────────────────────┐
│ 第 2 层：因子层                                         │
│   输入：第 1 层归一化后的 action/state                  │
│   动作：选启用的因子（17 个中的子集），每个因子可配     │
│         多个窗口；按因子公式（与现行 _descriptor_kernel │
│         一致）算原始因子值                              │
│   输出：raw factor dict {key -> float}                  │
│   父类：Factor Protocol（17 个子类）                   │
│   注意：本层因子不再做 z-score —— z-score 已经由第 1 层 │
│         统一完成。本层只做形状构建（splice/window/walk）│
│         + descriptor 公式                              │
└────────────────────────┬──────────────────────────────┘
                         │ raw factor dict
                         ▼
┌───────────────────────────────────────────────────────┐
│ 第 3 层：Calibration（每因子独立校准）                  │
│   输入：第 2 层 raw factor dict                         │
│   动作：对**每个因子的每个 key 独立**做 percentile 校准 │
│         （公式与现 PercentileRollingNormalizer 一致）   │
│   输出：calibrated factor dict {key -> float in [0,1]}  │
│         （NaN 在 cold-start / force_miss 下保留传播）   │
│   父类：Calibration Protocol                           │
│   子类示例：PercentileRollingCalibration（默认）        │
│   校准数据：per-key rolling buffer（来自 §6.3 两条路其一）│
└────────────────────────┬──────────────────────────────┘
                         │ calibrated factor dict
                         ▼
┌───────────────────────────────────────────────────────┐
│ 第 4 层：Composer                                       │
│   输入：第 3 层 calibrated factor dict                  │
│   动作：按子类逻辑聚合 + 阈值/规则判定 → 输出 hit type  │
│   输出：JudgeResult(FULL_HIT / WARM_START(t) / MISS)    │
│   父类：Composer Protocol                              │
│   子类示例：WeightedSumComposer / AndGateComposer /     │
│             OrGateComposer（聚合 + fallback 逻辑各子类  │
│             自决，先不固定）                            │
│   配置时校验：composer 子类声明依赖的因子 key 集合，    │
│                 与第 2 层启用因子的 union key 集合做     │
│                 静态校验 — 缺失即报错                   │
└───────────────────────────────────────────────────────┘
```

### 6.3 校准数据来源 — **二选一强制启动前就绪（no cold-start）**（用户决策 2026-05-07）

> 用户决策：第 1 层和第 3 层启动时**必须**有完整校准数据，**不允许任何 cold-start 状态存在**。两条来源每层各 pick 一个，缺失则 yaml 加载失败（fail-fast），verdict 路径压根不启动。

每层独立 pick：

| 层 | offline 来源 | warmup 来源 |
|---|---|---|
| 第 1 层 Normalization | `LibraryStats.{action_sigma, action_active_mask, state_sigma, state_active_mask}` 从 backend 的 `load_artifact` 读（pkl artifact 字段已有，不变） | **不支持**（G1 R1 Item 1：当前 WarmupPool / DumpingJudge / preload ctrl 三条通道仅承载 factor raw 样本，没有原始 action/state 数据通道；扩展通道会大幅增加协议复杂度。第 1 层强制 offline；warmup σ 留作未来扩展） |
| 第 3 层 Calibration | 预先计算好的 per-key 校准样本，yaml 显式配 `path` + `format ∈ {jsonl, pkl}`（默认 jsonl） | warmup yaml 的 `factor_raw` JSONL → 聚合 per-key list → 灌满每个 rolling window（同现行机制） |

**强制语义（启动 validator）**：
- yaml 中第 1 层 `normalization.stats_source.type ∈ {offline}` — **当前唯一可选 offline**（warmup 留作未来扩展）
- yaml 中第 3 层 `calibration.samples_source.type ∈ {offline, warmup}`，**必填**
- 加载时 validator 拉数据：
  - 第 1 层 backend 没暴露 library_stats → reject
  - 第 3 层 `samples_source: offline` 但 path 不可读 / format 解析失败 / 任一 key 样本数 < window_size → reject
  - 第 3 层 `samples_source: warmup` 但对应 `WarmupPool[eval_yaml_id]` entry 不存在 / 任一 bound key 样本数 < window_size → reject
- buffer 没"半满状态" → 启动即满，第一个 verdict 起就是 production 状态

**buffer 跨 episode 持续**（用户决策 #3 改名版）：rolling window 启动即满后，跨 episode 滑动持续 — eval 阶段新样本进 buffer 滑动替换 warmup 灌入的初始样本。寿命 = server connection 寿命。

### 6.3.1 cold-start 概念彻底废除

旧 `PercentileRollingNormalizer` 的三个 cold-start 策略（`force_miss / passthrough / lenient`）+ `_LENIENT_MIN_SAMPLES = 10` + `cold_start_strategy` 参数 + `__call__` 内部 if-elif 分支 — **全部删除**。

NaN 唯一合法来源在新架构里收紧成：
- 因子层物理边界（`history` 不够 P 步、`walk_next` 走完 < F 步、winner 数据缺失 / fork detected）
- 因子公式自身的退化（zero-norm velocity → `direction` NaN、单点窗口 → 多 desc NaN）

**绝不来自**：calibration buffer 不满（不存在）、Normalization σ 缺失（启动失败而不是出 NaN）。

Composer 子类处理的 NaN **永远是"这一步因子在物理上没法算"**，不是"系统还没暖好"。这让 composer 子类的 NaN 处理逻辑可以严格按"物理意义不可用"来设计，而不是兼顾"早期不稳定状态"。

### 6.4 Composer 配置时静态校验

Composer 子类需要声明它消费哪些因子 key（或 key 模板）。yaml load-time validator 把 composer 的依赖 key 集合与第 2 层启用因子的 union key 集合（通过 `Factor.describe(params)` classmethod 推导）做差集对比 — 缺失即抛 ConfigError，不让进 `_build_judge`。

> 依赖声明的具体形式（key 列表 / key 模板正则 / 类属性 vs 实例方法）待 Plan §9 确定。

### 6.5 设计原则

1. **每层 Protocol 互不知道彼此实现**：层之间只通过约定的输入输出 dict（normalized action/state、raw factor dict、calibrated factor dict、JudgeResult）通讯，绝不持有跨层对象引用。
2. **可插拔 + yaml 选子类**：每层都有 Protocol 父类，yaml 用 `type` 字段选子类，参数走 `params`。新加因子 / Composer / Calibration 只加子类不改框架。
3. **没有框架级 fallback / cold-start 短路 / 全 NaN 短路**（§6.11 用户决策 #2 #3）：
   - 因子 NaN（仅来自物理边界：history 不够、walk_next 走完、数据缺失、退化退化分母）**直接**带着 NaN 进 Calibration。
   - Calibration **不会**产 cold-start NaN（§6.3 启动时已 fail-fast 校验 buffer 满），因子 NaN 输入 → percentile_rank 输入 NaN → 输出 NaN，**直接**进 Composer。
   - Composer 接到 NaN 后由**子类**决定怎么处理（哪些子类把 NaN 视为 risky、哪些视为 skip、哪些直接 fallback 到 WARM_START 等），框架不替它做决定。
   - 后果：旧 `all_nan_fallback: warm_start@0.7` yaml 字段废除；旧 `JudgeResult.factor_outputs.sentinel` 字段废除；旧 `cold_start_strategy: force_miss/passthrough/lenient` 配置废除。
7. **No cold-start**（§6.3 用户决策 2026-05-07）：第 1 / 第 3 层启动时校准数据必备，不允许任何"系统还在暖机"中间态。yaml 加载阶段 fail-fast 校验校准数据来源可达。
4. **第 2 层不做 z-score**：z-score 提到第 1 层；第 2 层因子只做"形状装配 + descriptor 公式"。
5. **第 3 层只看因子，不看原始数据**：第 3 层 Calibration 是 per-key 的，不依赖 action/state 本身。
6. **历史由 Orchestrator 维护，4 层无 `record_action` 接口**（§6.11 决策 D）：Orchestrator 持 `_action_history` / `_state_history`，每次 verdict 把当前 history snapshot 注入第 2 层因子。4 层 Protocol 不暴露 `record_action`。

### 6.6 Online 因子的统一窗口形状（用户决策 #1）

所有 4 个 online 因子（jerk_online_action / direction_online_action / dispersion_online_action / path_length_online_action 等 8 个 online 变体）共用同一个 splice 形状：

```
splice = [history[-P:],   winner,   walk_next(F)]
              ^               ^         ^
              past            anchor    future
              P 步实执行     winner     F 步沿 chain 下游
```

`P` 和 `F` 在 yaml 中独立选，每因子可多窗口配置 `windows: [{past: P, future: F}, ...]`。

**state 通道**（online_state 因子）：
- `history[-P:]` = `_state_history[-P:]`（Orchestrator 维护的 P 个 state quote）
- `winner` = `winner.query_keys["robot_state"]`
- `walk_next(F)` = `view.walk_next(winner_id, F)` 取 F 个下游 entry，每个取 `e.query_keys["robot_state"]`
- splice 长度 = P + 1 + F

**action 通道**（online_action 因子，用户决策 #1 子问题选 (i)）：
- `history[-P:]` = `_action_history[-P:]`（Orchestrator 维护的 P 个执行 action）
- `winner` = `winner.payload.action_chunk[0]`
- `walk_next(F)` = `view.walk_next(winner_id, F)` 取 F 个下游 entry，每个取 `e.payload.action_chunk[0]`
- splice 长度 = P + 1 + F
- 与 state 完全对称（取下游 entry 的 action_chunk[0]，不取 winner 自己 chunk 的后 F 步）

**统一含义**：splice 第 t 步的 state 与第 t 步的 action 来自**同一次推理**，即 chain entry t 的 query_keys + payload。这保证 4 个 online 变体（动作 vs 状态、jerk vs direction vs dispersion vs path_length）的物理坐标系一致。

**P / F 边界**：
- `len(history) < P` → 整个 factor 出 NaN（per §6.5 原则 3）
- `walk_next` 返回 < F 个 entry（chain 走完 / fork） → 整个 factor 出 NaN
- `winner.query_keys["robot_state"]` 缺失（state 因子）/ winner 没 action_chunk（不可能，每 entry 都有）→ NaN

### 6.7 PayloadView 与 chain walk（保留 + 微调）

`PayloadView` facade 保留：第 2 层 online 因子（含 action 和 state）通过 `view.walk_next(winner_id, F)` 拿下游 entry。

- `walk_prev` / `walk_next` 接口不变。
- `ForkPolicy` enum 简化：只保留 `TRAJECTORY`（默认），其他 4 个占位值删除（§4 ❹ 中已列为可删）。fork 检测仍走 NotImplementedError → factor 出 NaN（per §6.5 原则 3）。
- backend 兼容性：`requires_chain_walk=True` 的因子（**所有 8 个 online 因子**，因 splice 都需要 walk_next）必须搭配实现 `fetch_entry` 的 backend（当前唯一是 InMemoryBackend）— validator 静态校验保留（§6.11 决策 C）。

### 6.8 Capability flags（修订）

旧因子的三个 capability flag 在新架构下重新分布：

| Flag | 旧 | 新 |
|---|---|---|
| `requires_library_stats` | F1a / F1b 都 True（自己内嵌 z-score） | **废除** — z-score 移到第 1 层 Normalization；因子层不持 library_stats |
| `requires_chain_walk` | 仅 F1a-T True | **8 个 online 因子全 True**（4 desc × 2 channel；统一 walk_next splice） |
| `required_top_k` | F2 = K，其他 = 0 | 仅 topk_action_variance = K；其他 = 0 |
| `descriptor_orientations` (实例属性) | 各因子声明自己产的 key → 取向 | 保留同语义，但 key 模板变 |

`required_top_k` 仍由第 2 层因子层向上汇总（取所有启用因子的 max），传回 search strategy 的 `min_top_k_hint`（§6.11 决策 B）。

### 6.9 DumpingJudge 在新架构（用户决策 #4 选 (b)）

DumpingJudge 仍是**透明旁路 dumper**，包整个 4 层 judge 之外。verdict 行为字节不变。

```
                ┌────── Inner Judge (4 层) ──────┐
                │  Norm → Factor → Calib → Comp  │ ──► JudgeResult
   ┌──────────┐ └────────────────────────────────┘
   │ verdict  │                  ▲
   │  request │──────────────────┤ 同请求并联送给
   └──────────┘                  ▼
              ┌─── DumpingJudge 自持子树 ────┐
              │  Norm' (副本) → Factor' (独立 │
              │  dump 因子集合，第 2 层子集)  │
              │  ※ 不持 Calibration / Composer│
              └──────────────────────────────┘
                              │
                              ▼
                       JSONL row 写入
```

**关键设计**：
- DumpingJudge 持自己的**第 1 层 Normalization 副本**（与 inner 同配置），不读 inner 的内部状态。z-score 廉价，重复一次比让 inner 暴露 `last_normalized` 内部钩子干净得多。
- DumpingJudge 持自己的**独立 dump 因子列表**（第 2 层子集），可以是 17 因子的任意子集 — 通常 warmup yaml 里 over-collect 全 17 个，eval yaml 各取所需。
- DumpingJudge **不持** Calibration 和 Composer — dump JSONL 只写 raw 值，calibration 是 per-yaml 的状态、composer 是 per-yaml 的策略，dump 出来意义不大。
- verdict 时：(1) inner 跑完整 4 层 → JudgeResult；(2) DumpingJudge 用同一份请求并联跑自己的 Norm + Factor → raw dict；(3) 写 JSONL；(4) 返回 inner 的 JudgeResult。

JSONL row schema（沿用旧字段名 + 新增字段，详见 §6.10）：
```
{
  "config_id": str,
  "task_id": str | None,
  "orig_init_state_idx": int | None,
  "step_idx": int,
  "winner_id": str | None,        # inner 的 hit winner
  "cp1_score": float | None,      # search top-1 cosine score
  "factor_raw": {key: float},     # DumpingJudge 自持因子的 raw 值
  "factor_nan": {key: bool},      # raw 是否 NaN
  // ↓ 新增：inner 4 层主路的诊断字段（per-verdict snapshot）
  "inner_hit_type": "FULL_HIT" | "WARM_START" | "MISS",
  "inner_start_t": float | None,
  "inner_factor_raw":        {key: float | None},
  "inner_factor_calibrated": {key: float | None},
  "inner_composer_score":    float | None
}
```

### 6.10 诊断记录字段（用户新增需求 2026-05-07）

> 用户要求：每个因子的 raw 值、percentile（calibrated）值、composer 值、threshold 决定都要可记录。

**主路（inner judge）的 `JudgeResult.factor_outputs` schema 调整**：

| 旧字段 | 新字段 | 说明 |
|---|---|---|
| `raw: {key: float\|None}` | `raw: {key: float\|None}` | 第 2 层因子层输出（同义） |
| `norm: {key: float\|None}` | `calibrated: {key: float\|None}` | 第 3 层 Calibration 输出（rename 反映新语义） |
| `score: float\|None` | `composer_score: float\|None` | 第 4 层内部聚合分数（rename） |
| `sentinel: str\|None` | **废除** | 框架不再做 cold-start 短路（§6.5 原则 3） |

`hit_type` / `winner_id` / `start_t`（即"threshold 决定"）仍在 `JudgeResult` 顶层（不进 factor_outputs）。

`JudgeResult.factor_outputs` 通过 `CheckResult.factor_outputs` → `Interceptor.__hit_meta__["factor_outputs"]` → client per-step jsonl 的 wire 透传链路保留。

**Wire schema 形状**：
```
__hit_meta__ = {
  "hit_type":   "FULL_HIT" | "WARM_START" | "MISS",
  "start_t":    float | None,
  "winner_id":  str | None,
  "cp1_score":  float | None,
  "factor_outputs": {
    "raw":             {key: float | None},
    "calibrated":      {key: float | None},
    "composer_score":  float | None
  }
}
```

**待细化（§9 第 6 项）**：是否需要更细粒度的 per-factor 诊断字段，如：
- `composer_per_factor_contribution`：WeightedSum 下每 key 的 `w_i * v_i_oriented`
- `composer_per_factor_passed`：AndGate / OrGate 下每 key 的 threshold 比较结果（bool）

这些"composer 内部决策痕迹"对实验调参很有用，但 schema 复杂度高。先按上面 4 字段最小集落地，第二轮再加。

### 6.11 用户已确认的决策清单（2026-05-07）

| 编号 | 决策点 | 用户选择 |
|---|---|---|
| #1 | online 窗口语义 | 统一 `[history[-P:], winner, walk_next(F)]`，P/F 独立配；详见 §6.6 |
| #1 子 | online_action 的 walk_next 取啥 | (i) `walk_next(F).action_chunk[0]`，与 state 完全对称 |
| #2 | 框架层 cold-start 短路 / all_nan_fallback | **取消**。所有因子结果（含 NaN）都进 composer，由 composer 子类决定 |
| #3 | 框架层全 NaN 短路 | **取消**，同 #2 |
| #4 | DumpingJudge 实现 | **(b)** 包外 + 自持独立 dump 因子列表 + 第 1 层 Normalization 副本（详见 §6.9） |
| #5 | offline 因子的 online 阶段读取 | **每个因子自己实现**（不是 framework 自动读 `payload.factors`） |
| #6 | 第 1 层 Normalization 输出怎么用 | 第 1 层往后传 normalized data；第 2 层因子自己决定怎么用 |
| #7 | 17 因子 registry 命名 | `<desc>_<source>_<channel>`（jerk_online_action 等 16 个）+ `topk_action_variance` |
| #8 | descriptor 改名 | `jerk` 不变 / `dir → direction` / `curv_radius → dispersion` / `cum_disp → path_length` |
| #9 | Calibration 跨 episode | **持续**（rolling window 跨 episode 滑动，不重置） |
| #10 | cold-start 概念 | **彻底废除**（§6.3.1）；启动时校准数据必备，二选一 fail-fast 校验 |
| #11 | 旧实验数据处理 | **不迁移**；老旧 yaml stem / 老旧 jsonl key 名永远归旧脚本读，新分析脚本只认新 stem / 新 key |
| #12 | wire schema_version | 默认 **加 schema_version: 2**（factor_outputs 内字段；client 解析时区分新旧） |
| #13 | 诊断粒度 v1 落地 | 默认 **最小集** `{raw, calibrated, composer_score}` + JudgeResult 顶层 `{hit_type, winner_id, start_t}`；v2 (per-factor contribution / passed) 后续再加 |
| #14 | 老代码处理 | **(i) 仅靠 git history**（commit `d9bf877` / `68a1d74` / `508e21a` / `0d32fa0` / `87eff6a` 永久可追）；旧文件直接删除替换，不留 `_legacy_*.py`，不留大段注释；设计/决策已落 `logs/old_*.log.md` 8 份 |
| A | PayloadView walk_next | **保留**（§6.7） |
| B | `min_required_top_k` 反向喂回 SearchStrategy | **保留**（§6.8） |
| C | `requires_chain_walk` capability flag → backend 校验 | **保留**（§6.7） |
| D | `record_action` lifecycle | **不保留**。Orchestrator 维护 history（§6.5 原则 6） |
| E | 实验脚本（exp/verdict_factor_judge/** + exp/common/factor_postprocess.py） | **纳入重构范围**；详见 §6.13 |

### 6.12 新旧功能等价性核对

| 旧功能 | 新对应 | 状态 |
|---|---|---|
| 4 descriptor 公式 (jerk/dir/curv_radius/cum_disp) | 4 descriptor (jerk/direction/dispersion/path_length)，公式不变 | ✅ |
| F1a-A.\<desc\> | \<desc\>_online_action（统一 splice，公式同） | ✅ + 多窗口能力扩展 |
| F1a-T.\<desc\> | \<desc\>_online_state（统一 splice，公式同） | ✅ + 多窗口能力扩展 |
| F1b-A.\<desc\>__p_f | \<desc\>_offline_action__p_f | ✅ |
| F1b-T.\<desc\>__p_f | \<desc\>_offline_state__p_f | ✅ |
| F2 (TopKActionConsensus) | topk_action_variance | ✅ |
| F1a/F1b 内嵌 z-score | 第 1 层 Normalization 统一 | ✅ 提到上层 |
| PercentileRollingNormalizer | 第 3 层 Calibration | ✅ rename |
| WeightedSum / AndGate / OrGate Composer | 第 4 层 Composer 三子类 | ✅ |
| F1b 多窗口 (p, f) | 因子层多窗口 | ✅ + 扩展到所有因子 |
| LibraryStats σ + active_mask offline 读 | 第 1 层 offline 校准来源 | ✅ |
| `<eval>__warmup.yaml` + WarmupPool + preload_buffer | **仅**第 3 层 Calibration 的 warmup 来源（第 1 层 Normalization 的 warmup 通道未实现，强制 offline；G1 R1 Item 1） | ✅ + 部分强制化 |
| `payload.factors: dict[str, float]` | 同 | ✅ schema 不变 |
| `JudgeResult.factor_outputs.{raw, norm, score}` | `.{raw, calibrated, composer_score}` | ✅ rename |
| `JudgeResult.factor_outputs.sentinel` | **废除** | ❌ 不再有框架级 fallback 概念 |
| `all_nan_fallback: warm_start@0.7` yaml 字段 | **废除** | ❌ 移到 composer 子类内部 |
| `requires_library_stats` capability flag | **废除** | ❌ 因子不再持 library_stats |
| `record_action(action_chunk)` lifecycle | **废除** | ❌ Orchestrator 持 history |
| `cold_start_strategy: force_miss/passthrough/lenient` | **废除** | ❌ no cold-start，启动 fail-fast |
| `_LENIENT_MIN_SAMPLES = 10` 常量 | **废除** | ❌ |
| Calibration 启动时 buffer 不满 → 输出 NaN | **不存在** | ❌ 启动即满 |
| Normalization σ 缺失 → factor 内部 fail-soft NaN | **不存在** | ❌ 启动 fail-fast |
| DumpingJudge 包装 + 旁路 dump | 同模式 (b)，复用第 1 层 Norm 配置 | ✅ |
| `requires_chain_walk` flag | 保留，且 8 个 online 因子全 True | ✅ + 扩展 |
| `min_required_top_k` 反向喂 strategy | 保留 | ✅ |
| Top-K 候选 search 通过 `min_top_k_hint` 升 top_k | 保留 | ✅ |
| 4 个 lifecycle hooks (`on_episode_start`/`on_task_begin`/`on_task_end`) | 保留（4 层 + DumpingJudge 仍要支持 `on_episode_start(extra_metadata)`） | ✅ |

**结论**：核心功能等价 ✅。废除的 7 项（sentinel / all_nan_fallback / requires_library_stats / record_action / cold_start_strategy / `_LENIENT_MIN_SAMPLES` / cold-start 全部 NaN 状态）都是设计简化，**不丢能力**：
- sentinel + all_nan_fallback 的等价能力由"composer 子类自带 fallback 逻辑"承接（如 `WarmFallbackWeightedSumComposer` 子类）
- requires_library_stats 由第 1 层 Normalization 必带 σ 替代
- record_action 由 Orchestrator 维护 history 替代
- cold_start_strategy 三选一由"启动时校准数据必备 fail-fast"替代 — 反而更严格也更清晰

### 6.13 实验侧脚本重构（用户决策 E，2026-05-07）

实验侧脚本依赖旧 yaml schema、旧 17→5 因子映射、旧 descriptor key 模板，必须跟着 src 层一起改。重构范围：

| 文件 | 重构内容 |
|---|---|
| `exp/verdict_factor_judge/phase1_spec.py` | 改用 17 因子新名笛卡尔生成 yaml；适配新 4 层 schema（`normalization` / `calibration` 块） |
| `exp/verdict_factor_judge/phase2_spec.py` | 同上 |
| `exp/verdict_factor_judge/phase2_layer2_spec.py` | 同上；`_r_*` recipe builder 全部按新因子名重写；`build_eval_yaml` / `build_warmup_yaml` 改新 schema |
| `exp/verdict_factor_judge/generate_yamls.py` | yaml writer 通用工具；新 schema 校验 |
| `exp/verdict_factor_judge/run_phase.py` | orchestration（subprocess.run main.py + ws ctrl）；可能不需大改，主要是适配新 yaml stem 命名 |
| `exp/verdict_factor_judge/per_step_log_writer.py` | per-step jsonl schema mirror — 改 `factor_outputs.{raw, calibrated, composer_score, schema_version}` |
| `exp/verdict_factor_judge/phase1_debug_analyze.py` | 旧分析脚本；按 §6.11 决策 #11 **不动**，仍读旧 jsonl 旧 key — 但新分析脚本必须独立写 |
| `exp/verdict_factor_judge/analysis/*.py` (新 / 旧) | 旧脚本读旧 jsonl 不动；新脚本认新 jsonl key + schema_version=2 |
| `exp/common/factor_postprocess.py` | `enrich_artifact_with_factors` 用 17 因子新 registry；写入 entries[i].payload.factors 的 key 用新模板 |
| `exp/common/build_clip_cache_artifact.py` / `build_in_memory_cache_artifact.py` / `build_llm_layer_matrix.py` 的 `--factors-yaml` CLI 路径 | 适配新 factors-yaml schema（17 因子声明） |

**实验数据迁移影响**（同 §8 但补充实验侧）：
- 旧 240 phase2 layer2 yaml + 24 phase2 layer1 yaml + 6 phase0/1 yaml + 24 sibling warmup yaml — **全部废**，新 spec 重新生成
- 旧 phase2 jsonl `factor_outputs.{raw, norm, score, sentinel}` 字段名失效 — 旧分析脚本读旧 yaml 名空间下的 jsonl 不变
- 旧 6 份 libero pkl 中 168 个 F1b factor keys — `payload.factors` 字段保留，但 key 失效；需重跑 OfflineWriter 写入 17 因子新 key（`library_stats` σ + active_mask **可保留**直接 reuse）

**重构粒度**（§9 待澄清）：是仅"接 schema/registry 变化最小改动"，还是借机重新设计实验阶段（phase 编排 / yaml 笛卡尔结构）？默认建议：**仅 minimal 改动**，实验设计本身和这次重构正交。

### 6.14 老代码处理 — git history（用户决策 #14，2026-05-07）

| 项 | 处理 |
|---|---|
| 旧 src 文件 (`runtime_continuity.py` / `source_window.py` / `consensus.py` / `composers/__init__.py` / `normalizers/__init__.py` / `judge.py` / `payload_view.py`) | B1-B6 完全保留旧实现；B7 cut-over 时按 §16 B7 改造（旧因子 / 旧 normalizer 类改 import-only deprecation stub；judge.py 拆为 facade re-export 新 `composite_judge.py / dumping_judge.py / legacy_judge.py`）。WA §3.1 No dead code 通过 stub 自动满足（旧实现内容被替换为单行 raise） |
| 旧 exp 文件（同 §6.13） | 同上 |
| 旧设计 / 决策记录 | 已落 `logs/old_verdict_factor_*.log.md` 8 份归档 |
| 旧 commit | `d9bf877` (B0) / `68a1d74` (B1+B2) / `508e21a` (DumpingJudge) / `0d32fa0` (dedicated_runner) / `87eff6a` (phase2_layer2 redesign) — git 永久可追，`git show <sha>:<path>` 拿任意旧版本 |

> 即如果将来需要看"老 F1a-T splice 怎么写的"，直接 `git show 68a1d74:src/openpi/cache/components/factors/runtime_continuity.py` 即可。代码区里不留任何 deprecated 注释。

### 6.14.1 §4 Code 实施期 deviation：stub 文件全删（2026-05-07）

实施 §16 B7 时用户追加指令"老代码删干净" + "B1 到 B4 的老代码删除你准备多就做"，明确要求**完全删除**旧文件而非保留 import-only deprecation stub。这覆盖了：

- §14.4 / §16 B7 / §17 R1 中 "legacy class import path 保留 不抛 ImportError" 的承诺
- "保留旧文件外壳作 deprecation stub" 的实施细节

**实际落地**：

| 旧文件 | 处理 |
|---|---|
| `factors/runtime_continuity.py` | **整文件删除**（不保留 stub） |
| `factors/source_window.py` | **整文件删除** |
| `factors/consensus.py` | **整文件删除** |
| `factors/normalizers/__init__.py` + 整个 `normalizers/` 目录 | **整目录删除** |
| `factors/base.py` 中 `OnlineExtractor` Protocol | **整 Protocol 删除** |
| `judge.py` 中旧 `CompositeJudge` + 旧 `DumpingJudge` 类（420 行） | **整类删除**，`judge.py` 末尾仅留 facade re-export `from openpi.cache.components.composite_judge import CompositeJudge` 等 |
| `JudgeConfig.normalizer` / `JudgeConfig.all_nan_fallback` 字段 | **字段删除**（旧 yaml 含此字段时 dataclass parser 报"unknown field"或被 §13.3 规则 1 reject 并提示"see logs/verdict_factor_judge_refactor.log.md §13"） |
| `NormalizerConfig` / `AllNanFallbackConfig` dataclass | **整类删除** |
| 旧 phase yaml 生成器 `phase1_spec.py / phase2_spec.py / phase2_layer2_spec.py` | **整文件删除**（yaml 生成器需按新 17 因子重新写；不阻塞 src 重构） |
| `tests/cache/`、`tests/serving/`、`tests/exp/` 中 19 个 plan §14.2 标"重写"的旧测试文件 | **整文件删除**（相关新测试已在 B1-B4 / B5 各批落地：`test_normalization_zscore.py` / `test_online_factors.py` / `test_offline_factors.py` / `test_topk_variance.py` / `test_calibration_percentile_rolling.py` / `test_composer_warm_fallback.py` / `test_composite_judge_v2.py` / `test_build_enrich_existing_pkl.py` / `test_registry.py` 已重写） |

**对 G2 reviewer 的影响**：

`tests/review_tests/` 是 sealed reviewer space（执行端不可见）。如果其中存在
```python
from openpi.cache.components.factors.runtime_continuity import RuntimeContinuityAction
```
这类直接引用旧 import path 的代码，本次 deviation 会让该 import 直接抛 `ModuleNotFoundError` —— pytest collection 阶段失败，影响整个 `tests/review_tests/` 子树的测试运行。

如果 G2 reviewer 把这种断当作 blocking：执行端的回退方案是按 §16 B7 原方案恢复 4 个 stub 文件（`runtime_continuity.py / source_window.py / consensus.py / normalizers/__init__.py`），每个文件保留旧类名作 import-only stub（构造旧类立即 `raise NotImplementedError`，避免 hidden TypeError）。回退由 G1 复审决定。

**为什么用户偏向激进删除**：从 5 月 7 日对话可见，用户多次提示当前实现"看起来像加新代码而不是重构"，希望旧代码彻底从 working tree 消失（让 IDE 中只看到新结构）。

`git show <sha>:<path>` 仍是恢复旧实现的唯一路径（plan §6.14 git history 兜底承诺保住）。

---

## 7. 新架构对原硬契约（§4）的冲击

按 §4 的 ❶❷❸❹ 四档逐项重新评估。

### ❶ 跨进程 / 已落盘的契约 — **大部分被打破**

| 原契约 | 新架构下 | 落盘数据影响 |
|---|---|---|
| 因子 registry 名 `f1a_a / f1a_t / f1b_a / f1b_t / f2` | **全部废弃**，换成 17 个新名（如 `jerk_online_action`...） | 全部 yaml `factors[].type` 字段失效 |
| Descriptor key 命名模板 `f1a_<a\|t>_<desc>` / `f1b_<a\|t>_<desc>__p<p>_f<f>` / `f2_var` | **全部废弃**，新模板待定（提案：`<desc>_<source>_<channel>__p<p>_f<f>`） | phase2 layer1 jsonl 中 `factor_outputs.raw / norm` 的 key 名失效；libero_*.pkl 中 168 个 F1b factors keys 失效 |
| `payload.factors: Optional[dict[str, float]]` 字段类型 | **保留**（仍是 dict[str, float]），但 key 命名约定换 | 字段本身可读，但内容 key 不再被新代码识别 |
| `LibraryStats` dataclass 字段名 `{action_sigma, action_active_mask, state_sigma, state_active_mask}` | **可保留**（第 1 层 ZScoreNormalization 仍需要），但语义从「F1a/F1b 内嵌 z-score」变成「Normalization 层独立 z-score」 | pkl 中 library_stats 字段仍可读，可直接喂给新第 1 层 |
| yaml schema：`JudgeConfig{factors[], composer, normalizer, all_nan_fallback, ...}` | **大改**：变成 4 层 schema `JudgeConfig{normalization, factors[], calibration, composer}`；`all_nan_fallback` 移到 composer 子类参数 | 全部 240 phase2 layer2 + 24 phase2 layer1 + 6 phase0/1 yaml 失效 |
| `_JUDGE_TYPES` 包含 `"composite"` | **保留**（`composite` 仍是 type token，但内部结构改） | yaml judge.type 还是 composite |
| 4 个 descriptor 名 + orientation 表 `jerk: risky / dir: safe / curv_radius: non_monotonic / cum_disp: non_monotonic` | **descriptor 内部公式不变**，但名字会跟随提案改 (`dir → direction`、`curv_radius → dispersion`、`cum_disp → path_length`)；orientation 表保留 | yaml `directions` 字段中的 key 名要跟着改 |
| `JudgeResult.factor_outputs = {raw, norm, score, sentinel}` schema | **schema 形状保留**（仍是 raw/norm/score/sentinel），但 raw/norm 内的 key 命名变 → wire 协议算字符串级 break | phase2 layer1 jsonl 仍可读，但旧分析脚本认不出新 key |

**结论**：动机 (b) 引入新能力意味着已落盘 240 + 24 + 6 个 yaml、3 套 phase2 jsonl、6 份 libero pkl 的 168 keys 全部需要重建或迁移。具体迁移策略待 Plan §11 风险登记。

### ❷ 模块边界 — **新增 + 重组**

| 原 import path | 新架构下 |
|---|---|
| `from openpi.cache.components.factors.composers import WeightedSumComposer, AndGateComposer, OrGateComposer` | 保留为 facade（旧名仍可 import），但内部实现重写为第 4 层 Composer 子类（含新 `WeightedSumWithWarmFallbackComposer` 兄弟类） |
| `from openpi.cache.components.factors.normalizers import PercentileRollingNormalizer` | **B1-B6 完全保留旧实现不动**（G1 R3 Item 2）；B7 cut-over 改为 import-only deprecation stub —— `__init__` 立即 raise NotImplementedError 指向新 `factors.calibrations.PercentileRollingCalibration`（不写无脑 alias，避免 `cold_start_strategy=` 旧参数被静默吞）。新第 3 层 Calibration 住在 `factors/calibrations/`（独立新建模块，与 `normalizers/` B1-B6 期间并存） |
| `from openpi.cache.components.factors.{base, registry, _descriptor_kernel}` | 保留；`base.py` 加 `HistoryView / CalibrationSamples / FactorContext` dataclass + 新 4 层 Protocol |
| `from openpi.cache.components.factors.{runtime_continuity, source_window, consensus}` | **B1-B6 完全保留旧实现不动**（G1 R3 Item 2）；B7 cut-over 改为 import-only deprecation stub（旧 `RuntimeContinuityAction / RuntimeContinuityState / SourceWindowSmoothness* / TopKActionConsensus` 类的 `__init__` 立即 raise NotImplementedError 指向新 17 因子注册名 + §6.11 #7 命名表）；新 17 因子住在新建 `factors/online.py / offline.py / topk.py` |
| 新增 `from openpi.cache.components.factors.normalization import ...` 第 1 层模块 | **新增**模块 |

### ❸ 类与方法签名 — **全面重写**（G1 R3 Item 1：与 §11 / §12 最终 Protocol 对齐）

| 原签名 | 新架构下 |
|---|---|
| `OnlineExtractor.extract(results, view, history, cached_data)` | **新签名**：`Factor.extract(ctx: FactorContext)`，单一 dataclass 入参，`FactorContext = {results, view, history, normalization}`（详见 §11.1 / §11.3） |
| `OfflineWriter.compute_for_episode(entries, library_stats)` | **新签名**：`compute_for_episode(entries, library_stats: LibraryStats)`（与现行 `orchestrator._build_entry_chain` 调用完全兼容，§2.2 orchestrator 不动；详见 §11.3） |
| Composer 接口 `bind_orientations(dict) / compose(norm, *, winner_id)` | **重写**：`Composer.declared_dependencies` 是**实例属性**（构造时算出，G1 R1 Item 4）；`compose(calibrated, *, winner_id: str)` — `winner_id` 由 CompositeJudge 保证非 None（空 results 早返 MISS，G1 R1 Item 5）；`bind_orientations(orientations)` 不变（详见 §11.5） |
| Normalizer 接口 `bind_keys(list) / __call__(raw) / preload_buffer(values)` | **拆为两个 Protocol**：第 1 层 Normalization（`__init__(library_stats: LibraryStats)`，per-step input → normalized action/state，**仅 offline 来源**，无 preload_buffer，G1 R1 Item 1）；第 3 层 Calibration（`__init__(samples: CalibrationSamples)`，per-factor raw → calibrated dict，offline / warmup 都支持，`bind_keys(keys)` 内部 fail-fast 校验灌入样本数 ≥ window_size，G1 R1 Item 2；详见 §11.2 / §11.4） |
| 因子类属性 `requires_library_stats / requires_chain_walk / required_top_k / descriptor_orientations` + classmethod `describe(params)` | **重写**：因子第 2 层不再持 library_stats（移到第 1 层 Normalization，因子通过 `ctx.normalization` 调），`requires_library_stats` flag **废弃**；保留 `requires_chain_walk`（8 online 因子 True，§6.8）/ `required_top_k`（topk = K，其他 = 0）/ `descriptor_orientations`（实例属性）/ `describe(params)` classmethod（详见 §11.3 / §12.1） |

### ❹ 可改 — **保留**（仍是只在重构域内自由改的部分）

不变。

### 7.1 影响摘要

- ❶ 80% 破：registry 名、key 命名、yaml schema 全废，落盘数据需迁移。
- ❷ 重组 + 新增：增第 1 层 normalization 模块、Calibration 层 rename、因子按 desc 分文件。
- ❸ 全面重写：4 层各自的 Protocol 都新写。
- ❹ 不变。

---

## 8. 落盘数据 / 实验 artifact 迁移影响

### 8.1 不可读 / 需要重建
- **240 phase2 layer2 yaml**（spatial16 only）：全部失效，需用新 schema 重生成。
- **24 phase2 layer1 yaml**（3 cfg × 8 desc）：全部失效。
- **6 phase0 / phase1 yaml + 24 sibling warmup yaml**：全部失效。
- **6 libero_*.pkl artifact 中的 168 个 F1b factors keys**：失效，需要重新跑 OfflineWriter 写入新 key（**LibraryStats σ 字段可保留**，省一遍重算）。

### 8.2 可读但语义需要新分析脚本
- **phase2 layer1 / layer2 已有 per_step jsonl 中的 factor_outputs.raw/norm**：key 名失效但 schema 形状不变；旧分析脚本（`exp/verdict_factor_judge/analysis/*.py`）需改为认旧 key 模板，否则数据失访问。
- **`__hit_meta__["factor_outputs"]` wire 协议**：schema 形状不变，但 key 命名变 → 客户端 jsonl 认旧 key 的逻辑需更新或加 schema_version 字段。

### 8.3 可重用的资产
- `exp/warm_start/data/`：500 ep × 3 cfg baseline，**完全不依赖** verdict factor，保留。
- `exp/random_periodic_gate/`：78 个 gate 散点，**完全不依赖**，保留。
- LibraryStats σ + active_mask（在 6 份 libero pkl 中）：第 1 层 ZScoreNormalization 直接复用。
- 4 个 descriptor 数学公式（`_descriptor_kernel.py` 内部）：保留，只搬位置。
- Top-K variance 公式（`consensus.py` 内部）：保留，只搬位置。

---

## 9. 待用户进一步确认 — **全部 RESOLVED**（2026-05-07）

| 编号 | 决策点 | 用户选择 |
|---|---|---|
| #15 | 第 3 层 Calibration offline 文件格式 | **(c) + (a)** — yaml 字段配 `calibration.source.offline.{path, format}`；`format` 取值 `{jsonl, pkl}`；**默认 `jsonl`** 复用 warmup JSONL 结构（per-key list of historical raw values） |
| #16 | 实验脚本重构粒度 | **(i) minimal** — phase 编排 / 笛卡尔结构 / 6-server 拓扑全保留；只改 17 因子新名 + 4 层 yaml schema + per-step jsonl schema_version=2 |

→ §6.11 14 项 + A-E + §9 16 项 = **全部决策完成**，进 §11-§17 详细 Plan。

---

## 11. 4 层 Protocol 最终签名

> 所有 dataclass 入参（不传 dict）；所有 Protocol `runtime_checkable`；放在 `src/openpi/cache/components/factors/base.py`（第 2 层 Factor + OfflineWriter + 共享 dataclass）+ `factors/normalization/base.py`（第 1 层）+ `factors/calibrations/base.py`（第 3 层）+ `factors/composers/base.py`（第 4 层）。

### 11.1 共享 dataclass

> G1 R1 Item 3 修订：第 1 层不再引入 `NormalizationCalibration`，直接复用现有 `LibraryStats`（字段名已锁死 §4 ❶，跨进程 pkl/IPC 兼容）。OfflineWriter.compute_for_episode 保持 `library_stats: LibraryStats` 入参，与 `orchestrator._build_entry_chain` 现行调用 `writer.compute_for_episode(entries, self._library_stats)` 完全兼容（满足 §2.2 orchestrator 不动）。

```python
# factors/base.py

@dataclass
class HistoryView:
    """Orchestrator 持有的 per-episode action / state 历史快照。每 verdict 注入。
    newest-last：actions[-1] 是最新执行的 action；states[-1] 是最新观测 state。"""
    actions: list[torch.Tensor]    # each [A]
    states:  list[torch.Tensor]    # each [S]

# LibraryStats 已存在（无需新建）；第 1 层 Normalization + OfflineWriter 都接此 dataclass
# from openpi.cache.components.factors.base import LibraryStats   ← 旧 import path 保留

@dataclass
class CalibrationSamples:
    """第 3 层 Calibration 启动时灌入的 per-key 历史样本。
    G1 R1 Item 3 改名（NormalizationCalibration → CalibrationSamples）— 与第 1 层语义不同，
    分名避免混淆。"""
    samples: dict[str, list[float]]    # key -> list of historical raw factor values

@dataclass
class FactorContext:
    """第 2 层因子接收的 verdict-time 上下文。"""
    results:       list["SearchResultLite"]
    view:          "PayloadView"          # walk_next / get / get_entry
    history:       HistoryView
    normalization: "Normalization"        # 第 1 层注入；factor 自己决定要不要调
```

### 11.2 第 1 层 — Normalization Protocol

```python
# factors/normalization/base.py

@runtime_checkable
class Normalization(Protocol):
    def __init__(self, library_stats: LibraryStats, **params) -> None: ...   # G1 R1 Item 3
    def normalize_action(self, raw: torch.Tensor) -> torch.Tensor:
        """raw [..., A] → normalized [..., A_active]（z-score + active mask 应用）"""
        ...
    def normalize_state(self, raw: torch.Tensor) -> torch.Tensor:
        """raw [..., S] → normalized [..., S_active]"""
        ...

# factors/normalization/zscore.py
class ZScoreNormalization:
    """default 子类：raw / σ.clamp_min(eps)，应用 active mask。eps=0.01."""
```

### 11.3 第 2 层 — Factor Protocol + OfflineWriter Protocol

```python
# factors/base.py

@runtime_checkable
class Factor(Protocol):
    """第 2 层因子。所有 17 个因子都实现这个；offline 因子还实现 OfflineWriter。"""

    # 类属性（validator 不实例化即可读）
    requires_chain_walk: bool       # 8 个 online 因子 = True
    required_top_k:      int        # topk_action_variance = K, 其他 = 0

    # 实例属性（__init__ 设置；CompositeJudge 装配时读）
    descriptor_orientations: dict[str, str]   # key -> "safe"|"risky"|"non_monotonic"

    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        """从 params 推 key -> orientation 映射；纯函数，不依赖 calibration data。
        Validator 调用此方法在 yaml load 时推 union key 集合并校验
        `composer.declared_dependencies` 实例属性（G1 R1 Item 4：实例属性，不是方法调用）。"""
        ...

    def extract(self, ctx: FactorContext) -> dict[str, float]:
        """运行时算 raw factor dict。返回 dict 的 key 集合 MUST == descriptor_orientations.keys()
        （CompositeJudge key contract assertion 强制）。值可为 NaN。"""
        ...


@runtime_checkable
class OfflineWriter(Protocol):
    """offline 因子额外实现这个，artifact build / Orchestrator episode-end 用。"""
    def required_payload_fields(self) -> set[str]: ...
    def compute_for_episode(
        self,
        entries: list["CacheEntry"],
        library_stats: LibraryStats,             # G1 R1 Item 3：保持 orchestrator 兼容
    ) -> list[dict[str, float]]:
        """返回 per-entry 因子 dict，调用方 merge 进 entries[i].payload.factors。
        签名与现行 `orchestrator._build_entry_chain` 调 `writer.compute_for_episode(entries,
        self._library_stats)` 完全兼容（§2.2 orchestrator 不动）。"""
        ...
```

### 11.4 第 3 层 — Calibration Protocol

```python
# factors/calibrations/base.py

@runtime_checkable
class Calibration(Protocol):
    def __init__(self, samples: CalibrationSamples, **params) -> None: ...
    def bind_keys(self, keys: list[str]) -> None:
        """CompositeJudge 装配时调一次，传入第 2 层启用因子的 union key 集合。
        Calibration 子类用此预分配 per-key state，**并且在此处做"每 key 样本数 ≥ window_size"
        的 fail-fast 校验**（G1 R1 Item 2：union key 在 CompositeJudge.__init__ 收集后才已知，
        构造期校验拿不到 key 集，必须延后到 bind_keys）。"""
        ...
    def __call__(self, raw: dict[str, float]) -> dict[str, float]:
        """raw factor dict → calibrated dict（典型在 [0, 1]）。
        NaN 输入 → NaN 输出（NaN 在新架构中只来自因子物理边界）。
        启动时 buffer 已满，无 cold-start 分支。"""
        ...
    def on_episode_start(self) -> None:
        """default no-op；rolling window 跨 episode 持续（用户决策 #9）。"""
        ...

# factors/calibrations/percentile_rolling.py
class PercentileRollingCalibration:
    """rolling window 跨 episode 持续。
    __init__: 仅存 samples 引用 + window_size，不校验。
    bind_keys(keys):
        1. 对 keys 中每个 k 拿 samples.samples[k]（缺 key → raise KeyError）
        2. 校验 len(samples.samples[k]) >= window_size，否则 raise ValueError
        3. 创建 deque(maxlen=window_size) 并灌入历史样本（NaN 跳过）
    __call__: 直接 percentile_rank；NaN 不 enqueue 也不 normalize。
    无 cold_start_strategy / force_miss / passthrough / lenient（彻底删除）。"""
    def __init__(self, samples: CalibrationSamples, *, window_size: int = 200) -> None:
        self._samples = samples
        self._window_size = window_size
        self._buffers: dict[str, deque] = {}

    def bind_keys(self, keys: list[str]) -> None:
        for k in keys:
            history = self._samples.samples.get(k)
            if history is None:
                raise KeyError(f"Calibration source missing key {k!r}")
            non_nan = [v for v in history if not math.isnan(v)]
            if len(non_nan) < self._window_size:
                raise ValueError(
                    f"key {k!r}: only {len(non_nan)} non-NaN samples, "
                    f"need >= window_size={self._window_size}"
                )
            buf = deque(maxlen=self._window_size)
            for v in non_nan[-self._window_size:]:
                buf.append(v)
            self._buffers[k] = buf
```

### 11.5 第 4 层 — Composer Protocol

```python
# factors/composers/base.py

@runtime_checkable
class Composer(Protocol):
    # G1 R1 Item 4：实例属性，构造时计算填好；CompositeJudge 直接读 instance.declared_dependencies
    declared_dependencies: set[str]

    def bind_orientations(self, orientations: dict[str, str]) -> None:
        """CompositeJudge 装配时调一次，注入 union orientation 表。
        子类：non_monotonic key 必须有 directions 配置，否则 raise。"""
        ...
    def compose(self, calibrated: dict[str, float], *, winner_id: str) -> JudgeResult:
        """返回 JudgeResult。winner_id 由 CompositeJudge 保证非 None（空 results 已被早返）。
        子类自带 fallback：calibrated 全 NaN 时是 MISS / WARM_START / 别的，由子类决定。"""
        ...

# 子类示例：__init__ 时算 declared_dependencies 填实例属性
class WeightedSumComposer:
    def __init__(self, *, weights: dict[str, float], full_hit_threshold: float, ...):
        self._weights = dict(weights)
        # G1 R1 Item 4：依赖是 weight 非零的 key 集合
        self.declared_dependencies = {k for k, w in weights.items() if w != 0.0}
        ...

class WeightedSumWithWarmFallbackComposer(WeightedSumComposer):
    """承接旧 all_nan_fallback 语义。params 加 warm_fallback_start_t；compose 内：
    所有非零权重 key 都 NaN → JudgeResult(WARM_START, winner_id, start_t=warm_fallback_start_t)"""
    ...

class AndGateComposer:
    def __init__(self, *, per_factor_thresholds: dict[str, float], ...):
        self._thresholds = dict(per_factor_thresholds)
        self.declared_dependencies = set(per_factor_thresholds.keys())
        ...

class OrGateComposer: ...   # 同 AndGate 结构
```

### 11.6 顶层 CompositeJudge（重写）

```python
# components/composite_judge.py（新文件，从 judge.py 拆出来）

class CompositeJudge:
    """4 层装配壳。SimilarityJudge 接口不变（Orchestrator 不改）。"""

    def __init__(self,
                 normalization: Normalization,
                 factors: list[Factor],
                 calibration: Calibration,
                 composer: Composer,
                 *,
                 export_factor_outputs: bool = False) -> None:
        # 1. 收集 union orientation + key contract preflight
        all_orientations: dict[str, str] = {}
        for f in factors:
            for k, ori in f.descriptor_orientations.items():
                if all_orientations.setdefault(k, ori) != ori:
                    raise ValueError(...)
        all_keys = list(all_orientations.keys())

        # 2. 静态校验 composer 依赖（G1 R1 Item 4：用实例属性而非 classmethod）
        missing = composer.declared_dependencies - set(all_keys)
        if missing:
            raise ValueError(f"Composer needs factors not in factor layer: {missing}")

        # 3. 注入（G1 R1 Item 2：bind_keys 内部 fail-fast 校验 calibration samples ≥ window_size）
        calibration.bind_keys(all_keys)
        composer.bind_orientations(all_orientations)

        # 4. 反向 hint
        self.min_required_top_k = max(getattr(f, 'required_top_k', 0) for f in factors)

        # 状态
        self._normalization = normalization
        self._factors = list(factors)
        self._calibration = calibration
        self._composer = composer
        self._export_factor_outputs = export_factor_outputs

    def __call__(self, results, checkpoint_id, cached_data, *, view=None, history=None) -> JudgeResult:
        # G1 R1 Item 5：空 results 早返 MISS，与现行 SimilarityJudge 子类合约一致；
        # 不把无 winner_id 的决定权丢给 composer 子类（避免 WarmFallback 子类返 WARM_START
        # 但 winner_id=None，Orchestrator 降成 MISS 丢失语义）。
        if not results:
            return JudgeResult(HitType.MISS)

        ctx = FactorContext(
            results=results, view=view, history=history,
            normalization=self._normalization,
        )
        raw: dict[str, float] = {}
        for f in self._factors:
            out = f.extract(ctx)
            # key contract assertion（同旧）
            assert set(out.keys()) == set(f.descriptor_orientations.keys())
            raw.update(out)

        calibrated = self._calibration(raw)
        result = self._composer.compose(calibrated, winner_id=results[0].id)

        if self._export_factor_outputs:
            result.factor_outputs = {
                "schema_version": 2,
                "raw":            _nan_to_none(raw),
                "calibrated":     _nan_to_none(calibrated),
                "composer_score": result.composer_score,
            }
        return result

    def on_episode_start(self, *, extra_metadata: dict | None = None) -> None:
        self._calibration.on_episode_start()
    def record_action(self, action_chunk) -> None:
        return None    # history 由 Orchestrator 持
```

### 11.7 DumpingJudge（重写，§6.9 (b) 模式）

```python
# components/dumping_judge.py（新文件）

class DumpingJudge:
    """透明包装 inner judge；旁路自持的第 1 层 + 独立 dump 因子集合写 JSONL。"""

    def __init__(self,
                 inner: SimilarityJudge,
                 dump_normalization: Normalization,   # 配置同 inner 但持副本
                 dump_factors: list[Factor],
                 dump_path: str,
                 config_id: str) -> None:
        self._inner = inner
        self._dump_normalization = dump_normalization
        self._dump_factors = list(dump_factors)
        self._dump_path = dump_path
        self._config_id = config_id
        # min_required_top_k 取 max(inner, dump factors)
        self.min_required_top_k = max(
            int(getattr(inner, 'min_required_top_k', 0)),
            max((int(getattr(f, 'required_top_k', 0)) for f in dump_factors), default=0),
        )
        # JSONL 文件 handle 持有（线程不安全 — 每连接一个 DumpingJudge 实例）
        self._fh = None    # lazy open in first __call__

    def __call__(self, results, checkpoint_id, cached_data, *, view=None, history=None):
        # 1) inner 跑完整 4 层
        judge_result = self._inner(results, checkpoint_id, cached_data, view=view, history=history)

        # 2) 旁路跑 dump 因子层
        try:
            ctx = FactorContext(
                results=results, view=view, history=history,
                normalization=self._dump_normalization,
            )
            factor_raw, factor_nan = {}, {}
            for f in self._dump_factors:
                out = f.extract(ctx)
                for k, v in out.items():
                    factor_raw[k] = float(v)
                    factor_nan[k] = math.isnan(float(v))
            row = {
                "config_id": self._config_id,
                "task_id": self._current_extra.get("task_id"),
                "orig_init_state_idx": self._current_extra.get("orig_init_state_idx"),
                "step_idx": self._step_idx,
                "winner_id": results[0].id if results else None,
                "cp1_score": float(results[0].score) if results else None,
                "factor_raw": factor_raw,
                "factor_nan": factor_nan,
                # inner 主路诊断快照（§6.10）
                "inner_hit_type":   judge_result.hit_type.name,
                "inner_start_t":    judge_result.start_t,
                "inner_factor_outputs": getattr(judge_result, 'factor_outputs', None),
            }
            self._write_row(row)    # 持 file handle，不再 open/close per verdict
        except Exception:
            logger.warning(...)     # 不静默吞，至少 log
        finally:
            self._step_idx += 1
        return judge_result

    def __getattr__(self, name):
        return getattr(self._inner, name)
    def on_episode_start(self, *, extra_metadata=None):
        self._current_extra = dict(extra_metadata or {})
        self._step_idx = 0
        # 转发给 inner，filtered dispatch
        ...
```

---

## 12. 17 因子最终实现矩阵

### 12.1 因子总表

| 因子注册名 | 实现类位置 | requires_chain_walk | required_top_k | 多窗口 | descriptor_orientations |
|---|---|:-:|:-:|:-:|---|
| `jerk_online_action`        | factors/online.py    | True  | 0 | ✓ | `{f"jerk_online_action__p{P}_f{F}":   "risky"}` |
| `jerk_online_state`         | factors/online.py    | True  | 0 | ✓ | `{f"jerk_online_state__p{P}_f{F}":    "risky"}` |
| `direction_online_action`   | factors/online.py    | True  | 0 | ✓ | `{f"direction_online_action__p{P}_f{F}":   "safe"}` |
| `direction_online_state`    | factors/online.py    | True  | 0 | ✓ | `{f"direction_online_state__p{P}_f{F}":    "safe"}` |
| `dispersion_online_action`  | factors/online.py    | True  | 0 | ✓ | `{f"dispersion_online_action__p{P}_f{F}":  "non_monotonic"}` |
| `dispersion_online_state`   | factors/online.py    | True  | 0 | ✓ | `{f"dispersion_online_state__p{P}_f{F}":   "non_monotonic"}` |
| `path_length_online_action` | factors/online.py    | True  | 0 | ✓ | `{f"path_length_online_action__p{P}_f{F}": "non_monotonic"}` |
| `path_length_online_state`  | factors/online.py    | True  | 0 | ✓ | `{f"path_length_online_state__p{P}_f{F}":  "non_monotonic"}` |
| `jerk_offline_action`       | factors/offline.py   | False | 0 | ✓ | `{f"jerk_offline_action__p{P}_f{F}":   "risky"}` |
| `jerk_offline_state`        | factors/offline.py   | False | 0 | ✓ | `{f"jerk_offline_state__p{P}_f{F}":    "risky"}` |
| `direction_offline_action`  | factors/offline.py   | False | 0 | ✓ | `{f"direction_offline_action__p{P}_f{F}":   "safe"}` |
| `direction_offline_state`   | factors/offline.py   | False | 0 | ✓ | `{f"direction_offline_state__p{P}_f{F}":    "safe"}` |
| `dispersion_offline_action` | factors/offline.py   | False | 0 | ✓ | `{f"dispersion_offline_action__p{P}_f{F}":  "non_monotonic"}` |
| `dispersion_offline_state`  | factors/offline.py   | False | 0 | ✓ | `{f"dispersion_offline_state__p{P}_f{F}":   "non_monotonic"}` |
| `path_length_offline_action`| factors/offline.py   | False | 0 | ✓ | `{f"path_length_offline_action__p{P}_f{F}": "non_monotonic"}` |
| `path_length_offline_state` | factors/offline.py   | False | 0 | ✓ | `{f"path_length_offline_state__p{P}_f{F}":  "non_monotonic"}` |
| `topk_action_variance`      | factors/topk.py      | False | K | ✗ | `{"topk_action_variance": "risky"}` |

### 12.2 文件组织

```
src/openpi/cache/components/factors/
├── __init__.py
├── base.py                         # Protocols + dataclass (HistoryView / CalibrationSamples / FactorContext / Factor / OfflineWriter); 同时 re-export 旧 LibraryStats（无新增 NormalizationCalibration）
├── registry.py                     # @register(name) + get_class + build + 强制 import 全 17 因子
├── _descriptor_kernel.py           # 4 个 descriptor 公式 (jerk / direction / dispersion / path_length) + _DESCRIPTOR_ORIENTATIONS 表（迁到这里）+ all_nan_for / is_all_nan helper
├── _debug.py                       # 集中 _VERDICT_DEBUG + _verdict_logger（消除 4 处复制粘贴）
├── online.py                       # _OnlineFactorBase + 8 thin subclass
├── offline.py                      # _OfflineFactorBase + 8 thin subclass + OfflineWriter mixin
├── topk.py                         # TopkActionVariance
├── normalization/
│   ├── __init__.py                 # facade re-export Normalization + ZScoreNormalization
│   ├── base.py                     # Normalization Protocol
│   └── zscore.py                   # ZScoreNormalization
├── calibrations/                   # （旧 normalizers/ rename）
│   ├── __init__.py
│   ├── base.py                     # Calibration Protocol
│   └── percentile_rolling.py       # PercentileRollingCalibration
└── composers/
    ├── __init__.py
    ├── base.py                     # Composer Protocol + 共享 helper (_apply_direction parsed 一次)
    ├── weighted_sum.py             # WeightedSumComposer + WeightedSumWithWarmFallbackComposer
    ├── and_gate.py
    └── or_gate.py
```

### 12.3 OnlineFactorBase 共享逻辑

```python
class _OnlineFactorBase(Factor):
    """8 个 online 因子共享：splice 装配 + walk_next 边界 + descriptor 公式。"""
    # 子类设
    descriptor:  str   # "jerk" | "direction" | "dispersion" | "path_length"
    channel:     str   # "action" | "state"
    requires_chain_walk = True
    required_top_k = 0

    def __init__(self, *, windows: list[dict | tuple]) -> None:
        self._windows = _normalize_windows(windows)   # [(P, F), ...]
        self.descriptor_orientations = self.__class__.describe({"windows": self._windows})

    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        prefix = f"{cls.descriptor}_online_{cls.channel}"
        ori = _DESCRIPTOR_ORIENTATIONS[cls.descriptor]
        return {f"{prefix}__p{p}_f{f}": ori for (p, f) in _normalize_windows(params["windows"])}

    def extract(self, ctx: FactorContext) -> dict[str, float]:
        results, view, history, norm = ctx.results, ctx.view, ctx.history, ctx.normalization
        keys = list(self.descriptor_orientations.keys())
        if not results:
            return all_nan_for(keys)

        out: dict[str, float] = {}
        for (P, F) in self._windows:
            seq = self._build_splice(results[0].id, view, history, P, F)   # [P+1+F, D] or None
            if seq is None:
                # boundary：history 不足 / walk_next 走完 / fork
                for d_key in self._keys_for_window(P, F):
                    out[d_key] = float('nan')
                continue
            # 第 1 层 normalize（factor 自决调哪条）
            normed = (norm.normalize_action(seq) if self.channel == "action"
                      else norm.normalize_state(seq))
            v = normed[1:] - normed[:-1]
            j = normed[2:] - 2 * normed[1:-1] + normed[:-2]
            val = compute_descriptors([self.descriptor], normed, v, j)[self.descriptor]
            out[self._key(self.descriptor, P, F)] = float(val)
        return out

    def _build_splice(self, winner_id, view, history, P, F):
        """返回 [P+1+F, D] tensor 或 None (boundary)。统一 splice：
           [history[-P:], winner, walk_next(F)]
        子类（_OnlineActionBase / _OnlineStateBase）覆盖具体取数。"""
        ...

class _OnlineActionBase(_OnlineFactorBase):
    channel = "action"
    def _build_splice(self, winner_id, view, history, P, F):
        if len(history.actions) < P: return None
        winner = view.get(winner_id)
        cand = torch.as_tensor(winner.action_chunk[0], dtype=torch.float32)
        try:
            forward_entries = view.walk_next(winner_id, k=F)   # raise NotImplementedError on fork
        except NotImplementedError:
            return None
        if len(forward_entries) < F: return None
        forward = [torch.as_tensor(e.payload.action_chunk[0], dtype=torch.float32)
                   for e in forward_entries]
        hist_tail = [torch.as_tensor(a, dtype=torch.float32) for a in history.actions[-P:]]
        return torch.stack(hist_tail + [cand] + forward, dim=0)   # [P+1+F, A]

class _OnlineStateBase(_OnlineFactorBase):
    channel = "state"
    # 类似但走 query_keys["robot_state"]；missing field → return None

@register("jerk_online_action")
class JerkOnlineAction(_OnlineActionBase):
    descriptor = "jerk"

@register("jerk_online_state")
class JerkOnlineState(_OnlineStateBase):
    descriptor = "jerk"

# ... 6 个其他 thin subclass 类似
```

### 12.4 OfflineFactorBase 共享逻辑

```python
class _OfflineFactorBase(Factor, OfflineWriter):
    """8 offline 因子。online 阶段直接读 winner.payload.factors；offline 阶段算并写。"""
    descriptor: str
    channel:    str
    requires_chain_walk = False
    required_top_k = 0

    def extract(self, ctx: FactorContext) -> dict[str, float]:
        keys = list(self.descriptor_orientations.keys())
        if not ctx.results: return all_nan_for(keys)
        winner = ctx.view.get(ctx.results[0].id)
        if winner.factors is None: return all_nan_for(keys)
        return {k: float(winner.factors.get(k, float('nan'))) for k in keys}

    def required_payload_fields(self) -> set[str]:
        return set()    # offline 因子读 payload.action_chunk / query_keys，schema 已有

    def compute_for_episode(
        self, entries: list["CacheEntry"],
        library_stats: LibraryStats,            # G1 R2 Item 1 sweep：与 §11.3 OfflineWriter Protocol 一致
    ) -> list[dict[str, float]]:
        """每 entry 在 (P, F) 滑窗算 descriptor。
        library_stats 直接用 σ + active_mask 做 z-score，与 online 相同公式。"""
        ...
```

### 12.5 TopkActionVariance（公式不变）

```python
@register("topk_action_variance")
class TopkActionVariance(Factor):
    requires_chain_walk = False
    descriptor_orientations = {"topk_action_variance": "risky"}

    def __init__(self, *, K: int) -> None:
        if K < 2: raise ValueError(...)
        self.K = K
        self.required_top_k = K

    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        return {"topk_action_variance": "risky"}

    def extract(self, ctx: FactorContext) -> dict[str, float]:
        # 同旧 consensus.py，candidate-local active mask（var > 1e-8），不依赖第 1 层 Normalization
        ...
```

---

## 13. yaml 新 schema 完整定义

### 13.1 schema 树

```yaml
checkpoints:
  cp1:
    enabled: true
    gate: { type: always_search }

    judge:
      type: composite

      # ── 第 1 层 ──
      normalization:
        type: zscore                                  # registered Normalization 子类
        params: {}                                    # ZScoreNormalization 当前无 param
        stats_source:                                 # G1 R1 Non-1 改名（旧叫 source；与 factor type 后缀的 online/offline 重名混淆）
          type: offline                               # 当前唯一可选值（G1 R1 Item 1：warmup 通道未实现）
          # offline 模式：σ + active_mask 自动从 backend.load_artifact 的 library_stats 字段读
          #            （pkl artifact 已带，不需要额外路径配置）

      # ── 第 2 层 ──
      factors:
        - type: jerk_online_state
          params:
            windows:
              - { past: 5, future: 5 }
              - { past: 7, future: 7 }
        - type: dispersion_offline_state
          params:
            windows:
              - { past: 5, future: 5 }
        - type: topk_action_variance
          params:
            K: 5

      # ── 第 3 层 ──
      calibration:
        type: percentile_rolling                       # registered Calibration 子类
        params:
          window_size: 50
        samples_source:                                # G1 R1 Non-1 改名（旧叫 source）
          type: offline                                # | warmup
          offline:                                     # 仅 samples_source.type == offline 时读
            path: data/calibration/spatial16_w50_v2.jsonl
            format: jsonl                              # | pkl  (默认 jsonl, §9 #15)
          # warmup 模式：从 WarmupPool[eval_yaml_id] 读 raw factor JSONL
          #            → 聚合 per-key list → 灌入每个 deque

      # ── 第 4 层 ──
      composer:
        type: weighted_sum_with_warm_fallback          # | weighted_sum / and_gate / or_gate
        params:
          weights:
            jerk_online_state__p5_f5:        1.0
            jerk_online_state__p7_f7:        1.0
            dispersion_offline_state__p5_f5: 1.0
            topk_action_variance:            0.5
          full_hit_threshold: 0.30
          warm_start_threshold: 0.10
          warm_start_t: 0.7
          warm_fallback_start_t: 0.7                   # 子类内部 fallback；旧 all_nan_fallback 等价
          directions:                                  # non_monotonic key 必填
            dispersion_offline_state__p5_f5: "range:[0.3, 0.7]"

      export_factor_outputs: true                       # 默认 false；true 时写 schema_version=2

      dump:                                            # 可选；存在则包 DumpingJudge
        path: /tmp/dump.jsonl
        config_id: spatial16_xxx_yaml_id
        normalization:                                 # DumpingJudge 持自己的第 1 层副本，stats_source 必须与 inner 一致
          type: zscore
          params: {}
          stats_source: { type: offline }              # 当前必须 offline；与 inner.normalization.stats_source 同
        factors:                                       # over-collect 17 因子超集
          - type: jerk_online_action
            params: { windows: [{past: 5, future: 5}, ...] }
          - type: jerk_online_state
            params: { ... }
          # ... 16 + 1
```

### 13.2 dataclass

```python
# config.py 修订（旧 dataclass 删除替换）

@dataclass
class SamplesSourceOfflineConfig:                # G1 R1 Non-1: 旧 CalibrationSourceOfflineConfig
    path: str
    format: str = "jsonl"     # jsonl | pkl

@dataclass
class SamplesSourceConfig:                       # G1 R1 Non-1: 旧 CalibrationSourceConfig
    type: str                 # "offline" | "warmup"
    offline: Optional[SamplesSourceOfflineConfig] = None

@dataclass
class StatsSourceConfig:                         # G1 R1 Non-1: 旧 NormalizationSourceConfig
    type: str                 # 当前唯一 "offline"（G1 R1 Item 1：warmup 通道未实现）
    # offline 走 backend.library_stats（pkl artifact 已带）

@dataclass
class NormalizationConfig:
    type: str                 # "zscore"
    params: dict = field(default_factory=dict)
    stats_source: StatsSourceConfig = ...        # G1 R1 Non-1: 旧 source

@dataclass
class FactorConfig:
    type: str                 # 17 注册名之一
    params: dict = field(default_factory=dict)

@dataclass
class CalibrationConfig:
    type: str                 # "percentile_rolling"
    params: dict = field(default_factory=dict)
    samples_source: SamplesSourceConfig = ...    # G1 R1 Non-1: 旧 source

@dataclass
class ComposerConfig:
    type: str                 # 4 子类
    params: dict = field(default_factory=dict)

@dataclass
class DumpConfig:
    path: str
    config_id: str
    normalization: NormalizationConfig = ...
    factors: list[FactorConfig] = field(default_factory=list)

@dataclass
class JudgeConfig:
    type: str                                        # "threshold" | "always_hit" | "always_warm_start" | "composite"
    threshold: float = 0.98                          # legacy 字段（threshold judge）
    cp3_threshold: float = 0.95
    warm_tiers: list[dict] = field(default_factory=list)
    start_t: Optional[float] = None
    # ── composite 字段 ──
    normalization: Optional[NormalizationConfig] = None
    factors:       Optional[list[FactorConfig]] = None
    calibration:   Optional[CalibrationConfig] = None
    composer:      Optional[ComposerConfig] = None
    export_factor_outputs: bool = False
    dump: Optional[DumpConfig] = None
    # ── 删除字段 ──
    # all_nan_fallback: 移到 composer.params.warm_fallback_start_t
    # normalizer:       rename + 改语义到 calibration
```

### 13.3 Validator 规则（12 静态 + 3 运行时 = 15 项）

`_validate_composite_judge_static(judge: JudgeConfig, ...)` 在 yaml load 时执行（不依赖运行时数据）：

1. `judge.factors` 至少 1 个
2. `judge.normalization / calibration / composer` 三个必有 — **G1 R1 Item 7：缺失任一即明确 reject 并提示"this looks like a legacy schema; rewrite to 4-layer schema"**（旧 composite yaml 仅有 normalizer/all_nan_fallback 字段，加载时被此规则 reject，不再有歧义；不依赖 stem namespace 判断）
3. 每 factor.type ∈ 17 注册名（`registry.get_class` 查得到）— factor.type 含旧名 `f1a_a / f1a_t / f1b_a / f1b_t / f2` 即 reject（明确属于旧 schema）
4. 通过 `Factor.describe(params)` 收集 union key 集合 → composer 实例 `composer.declared_dependencies` 属性必须 ⊆ union（G1 R1 Item 4：实例属性而非 classmethod）
5. 含 `requires_chain_walk=True` 因子时，backend.type 必须 == "in_memory"
6. 若有 non_monotonic key 进 composer，composer.params.directions 必须覆盖
7. `normalization.stats_source.type` 当前唯一 `"offline"`（G1 R1 Item 1：warmup 通道未实现）；`calibration.samples_source.type ∈ {offline, warmup}`
8. `calibration.samples_source.offline.format ∈ {jsonl, pkl}`（若 samples_source.type == offline）
9. `topk_action_variance.params.K >= 2`
10. 各因子 windows 字段：所有 (P, F) 都 P >= 0 且 F >= 0；不允许 P == 0 且 F == 0（splice 长度 1，descriptor 无意义）
11. `dump.normalization.stats_source.type` 必须与 `judge.normalization.stats_source.type` 一致（避免 dump 路与 inner 路 z-score 量纲不一致）
12. **G1 R1 Item 7 新增**：旧字段显式 reject — `judge.normalizer` / `judge.all_nan_fallback` / `judge.cold_start_strategy` 在 yaml 中出现即 reject 并提示"this looks like a legacy schema; remove and rewrite to 4-layer schema"

`_validate_composite_judge_runtime(...)`（在 `_build_judge` 装配时执行，可访问 backend / WarmupPool）：

13. `normalization.stats_source == offline` → backend.library_stats 必须可达
14. `calibration.samples_source == offline` → file path 可读 + format 解析成功 + per-key list 长度 >= window_size
15. `calibration.samples_source == warmup` → WarmupPool[yaml_id] 必须有 raw factor JSONL + 每 bound key 的 list 长度 >= window_size

任一违反 → raise + reject yaml load。

---

## 14. 测试策略

### 14.1 必须保留 import path（兼容 `tests/review_tests/` 不可见测试）

旧 facade 入口 import path 全部保留为 re-export（避免 review_tests 引用旧名导致 G2 测试断）：

```python
# components/judge.py 保留为 facade
from openpi.cache.components.composite_judge import CompositeJudge
from openpi.cache.components.dumping_judge   import DumpingJudge
from openpi.cache.components.legacy_judge    import (
    SimilarityJudge, AlwaysHitJudge, AlwaysWarmStartJudge, ThresholdJudge,
    HitType, JudgeResult,
)
# 旧 _build_factor_outputs / _nan_to_none helper 保留 facade
```

```python
# components/factors/composers/__init__.py 保留 facade
from openpi.cache.components.factors.composers.weighted_sum import WeightedSumComposer
from openpi.cache.components.factors.composers.and_gate import AndGateComposer
from openpi.cache.components.factors.composers.or_gate import OrGateComposer
```

```python
# components/factors/normalizers/__init__.py 的处理 — G1 R3 Item 2 阶段性策略
# B1-B6 期间：本文件 + 旧 PercentileRollingNormalizer 实现 100% 不动
#             （旧 yaml 走旧 _build_judge → 旧 PercentileRollingNormalizer，行为字节不变）
# B7 cut-over：本文件改为 import-only deprecation stub
#              旧 PercentileRollingNormalizer 类的 __init__ 第一行立即
#              raise NotImplementedError("PercentileRollingNormalizer has been replaced
#              by PercentileRollingCalibration in factors.calibrations; cold_start_strategy
#              is removed (no cold-start), see logs/verdict_factor_judge_refactor.log.md
#              §6.3.1")
# 这避免了 B3 时把旧 normalizer 改 alias → 旧 yaml 在 B5 之前就崩 / 行为漂移的 corner case
```

**阶段性策略**：legacy `PercentileRollingNormalizer` 实现在 B1-B6 期间**完全保留不动**；B5 装配层落地时新 schema validator 会 reject 旧 yaml（§13.3 规则 12），意味着 B5 之后没人再调旧 PercentileRollingNormalizer；B7 cut-over 时把旧文件改为 deprecation stub。这保证：
- B1-B6 旧 yaml 加载行为字节不变（旧 _build_judge → 旧 normalizer 实例化 → 旧 cold_start_strategy 路径全部走老代码）
- B5 之后旧 yaml load-time reject，旧 normalizer 自然成 dead code
- B7 把旧 normalizer 改 stub 是安全的（已无 caller）

### 14.2 测试文件清单

| 现有测试文件（`tests/cache/`） | 处理 |
|---|---|
| `components/factors/test_descriptor_kernel.py` | 不变 |
| `components/factors/test_runtime_continuity.py` | **重写**为 `test_online_factors.py`，覆盖 8 个 online 因子 |
| `components/factors/test_source_window.py` | **重写**为 `test_offline_factors.py` |
| `components/factors/test_consensus.py` | **重写**为 `test_topk_variance.py` |
| `components/factors/test_composers_protocol.py` | 重写：`Composer.declared_dependencies` 校验 |
| `components/factors/test_composers_algorithm.py` | 重写：4 子类（含 WarmFallback）算法 |
| `components/factors/test_normalizer_protocol.py` | 重写为 `test_calibration_protocol.py` |
| `components/factors/test_normalizer_algorithm.py` | 重写为 `test_calibration_algorithm.py`（无 cold-start 分支） |
| `components/factors/test_normalizers_preload.py` | 重写为 `test_calibration_preload.py` |
| `components/factors/test_composite_judge.py` | 重写：4 层装配 + key contract + composer dep check |
| `components/factors/test_base.py` | 重写：新 dataclass + Protocol |
| `components/factors/test_registry.py` | 重写：17 注册名 |
| `components/test_judge.py` | 不变（legacy judge 不变） |
| `components/test_dumping_judge.py` | **重写**：(b) 模式 + 持副本 normalization |
| `test_config.py` | 大改：旧 schema 字段删，新 schema 字段加 |
| `test_config_factor.py` | 重写：4 层 yaml 端到端 |
| `test_dump_config.py` | 重写：新 dump schema |
| `test_artifact_roundtrip.py` | 重写：新 17 因子 key 模板 |
| `test_factor_postprocess.py` | 重写：17 因子 enrich |
| `test_orchestrator_offline_writers.py` | 适配：新 OfflineWriter Protocol |
| `test_payload_view.py` | 微调：ForkPolicy 砍后 |
| `test_orchestrator_history.py` | 不变 |
| `test_orchestrator.py` | 不变 |
| `test_interceptor.py` / `test_interceptor_hit_meta.py` | 微调：`factor_outputs.schema_version=2` 字段 |
| `test_episode_extra_propagation.py` | 不变 |
| `test_search_session_lifecycle.py` | 不变 |
| `test_cache_storage_factor_facade.py` | 不变 |
| `test_llm_layer_extract_parity.py` | 不变 |
| `test_episode_write.py` | 不变 |
| `tests/serving/test_serve_policy_yaml_id_propagation.py` | 微调：CompositeJudge / DumpingJudge 装配 |
| `tests/exp/verdict_factor_judge/test_phase1_spec_warmup_yamls.py` | **重写**：新 17 因子 yaml |

### 14.3 新增测试文件

```
tests/cache/components/factors/
├── test_normalization_zscore.py            # 第 1 层 ZScoreNormalization
├── test_normalization_source_offline.py    # 从 LibraryStats 构造（第 1 层唯一来源）
├── test_calibration_source_offline.py      # offline jsonl/pkl 加载 + fail-fast
├── test_calibration_source_warmup.py       # warmup pool 加载 + fail-fast
├── test_factor_validator_static.py         # 12 项静态校验
├── test_factor_validator_runtime.py        # 3 项运行时校验
└── test_composer_warm_fallback.py          # WeightedSumWithWarmFallback NaN 行为
```

### 14.4 review_tests 兼容性策略

- ❷ **模块边界 import-only 兼容**（G1 R2 Item 2 收紧）：facade re-export **仅承诺 import 不抛 ImportError**，旧类名（`RuntimeContinuityAction / SourceWindowSmoothness* / TopKActionConsensus / PercentileRollingNormalizer`）在旧文件中保留作 deprecation stub — 任何尝试**构造**旧类立即 raise `NotImplementedError` 指向新类与 §6.11 #7 命名表（详见 §16 B7）。这不保证旧 test 的构造行为正确，但保证：
  - `from openpi.cache.components.factors.runtime_continuity import RuntimeContinuityAction` 不抛 `ImportError`，pytest collection 不断
  - 任何旧测试一旦构造旧类立即收到明确错误信息（不是 hidden TypeError / 隐式 attribute miss）
- ❸ **核心 SimilarityJudge 接口签名全保留**：`SimilarityJudge.__call__(results, checkpoint_id, cached_data, *, view, history)` / `JudgeResult.{hit_type, winner_id, start_t, factor_outputs, composer_score}` / `min_required_top_k` 属性 / `on_episode_start(extra_metadata=)` lifecycle / `record_action(action_chunk)` lifecycle / `HitType` enum / `CompositeJudge` / `DumpingJudge` / `AlwaysHitJudge` / `AlwaysWarmStartJudge` / `ThresholdJudge` 类名全部不变 — 新 4 层架构只重写**实现**，外壳合约稳。
- 已落盘字段不动：`payload.factors: dict[str, float]` 字段名 + `LibraryStats` dataclass 字段名 + `__hit_meta__["factor_outputs"]` schema 形状（含 `schema_version` 新字段，向后兼容 `dict.get`）

如果 G2 reviewer 的 review_tests **构造**旧 17 之前的因子类 / 用旧 yaml schema 加载 → 这些测试会按 deprecation stub 显式失败（不是 hidden bug），是正面的设计变更。G2 reviewer session 必须接受新 schema；如果反对，回到 §3 G1 重谈。

---

## 15. 实验数据处理

按 §6.11 决策 #11 **不迁移**。物理分隔策略：

### 15.1 yaml stem 命名空间

新生成的 yaml 加 `v2_` 前缀或放新目录：

```
exp/verdict_factor_judge/config/
├── {clip,max_pool,spatial16}/                       # 旧（不动）
│   ├── phase0/
│   ├── phase1/
│   ├── phase2_layer1_{a,b}/
│   └── phase2_layer2_b{1..6}/
└── v2_spatial16/                                    # 新
    ├── phase2_layer2_b{1..6}/                       # minimal 重写：240 cell 同结构，新 schema
    └── ...
```

旧分析脚本（`phase1_debug_analyze.py` / `analysis/plot_pareto.py` 等）继续读旧目录；新分析脚本独立写。

### 15.2 jsonl schema_version 区分

旧 jsonl `factor_outputs` 无 `schema_version` 字段（隐含 v1）。
新 jsonl `factor_outputs.schema_version: 2`，含 `raw / calibrated / composer_score`。

新分析脚本 `if schema_version != 2: raise UnsupportedSchemaError`。

### 15.3 pkl artifact 处理

6 份 `libero_*.pkl`：
- `library_stats` 字段保留（直接 reuse 给新第 1 层）
- 旧 168 keys（`f1b_*`）保留在 `payload.factors` 里 — 新代码不读，旧分析脚本仍读
- 重新跑新 OfflineWriter 时把 17 因子新 keys 也写入 `payload.factors`（dict 增量更新，新旧并存）

具体重跑 5 步：
1. 从 `exp/common/factor_postprocess.py` 入口
2. `--factors-yaml` 文件指定 17 因子集合（新 schema）
3. 跑 `enrich_artifact_with_factors` 写新 keys
4. 输出 `libero_*_v2.pkl`（新文件，不覆盖旧 pkl）
5. 新 yaml `backend.in_memory.preload_path` 指向 `_v2.pkl`

### 15.4 WarmupPool 复用

`WarmupPool` 接口不变；新 yaml 跑 warmup 时按新 17 因子 over-collect → 新 jsonl → preload 给新 calibration。
旧 warmup 数据按需保留（不做迁移）。

---

## 16. 改动分批 B0-B7

每批结束都跑 `uv run pytest tests/cache/ tests/serving/ tests/exp/`（不读 `tests/review_tests/`），全 PASS 才进下批。

### B0 准备
- 不删任何旧代码
- freeze 当前 baseline：`uv run pytest tests/cache/ tests/serving/ tests/exp/ -v` 全 PASS（记录 commit `d9a5d00` 状态）
- 创建空目录 + `__init__.py`：`factors/normalization/` `factors/calibrations/` 占位（旧 `normalizers/` 仍在，并存）
- 写 `factors/_debug.py` 集中 `_VERDICT_DEBUG`
- ❌ 还不动旧文件

### B1 第 1 层 Normalization
- `factors/base.py` 加 `HistoryView` / `CalibrationSamples` / `FactorContext` dataclass（**不引入** `NormalizationCalibration`，直接复用 `LibraryStats`，G1 R1 Item 3）
- `factors/normalization/base.py` Protocol（`__init__(library_stats: LibraryStats, ...)`）
- `factors/normalization/zscore.py` ZScoreNormalization
- 测试：3 个新 test_normalization_*.py（zscore + source_offline + 协议 contract；**无** source_warmup 测试，G1 R1 Item 1）
- ❌ B1 期间不动旧 _build_judge / 旧 factor 文件 / 旧 yaml schema validator —— 旧 yaml load 路径仍走旧 `_build_inner_judge` 分支不变化（新 schema validator 在 B5 装配层落地后才生效，届时旧 yaml 一律 load-time reject）

### B2 第 2 层 17 因子
- `factors/_descriptor_kernel.py` 把 `_DESCRIPTOR_ORIENTATIONS` 表迁入；旧 desc 公式名 `_dir / _curv_radius / _cum_disp` 实现保留作 helper 不变，新 desc 名 `direction / dispersion / path_length` 通过 dispatch 表路由到同一组 helper（公式不变，分发名换）
- `factors/online.py` `_OnlineFactorBase` + `_OnlineActionBase` + `_OnlineStateBase` + 8 thin subclass + `@register`
- `factors/offline.py` `_OfflineFactorBase` + 8 thin subclass + OfflineWriter mixin（`compute_for_episode(entries, library_stats: LibraryStats)`）
- `factors/topk.py` TopkActionVariance
- `factors/registry.py` 强制 import 17 个新 factor 模块（新名）+ 旧 5 名（`f1a_a/f1a_t/f1b_a/f1b_t/f2`）暂时保留指向旧 class（B7 时旧 5 名移除注册，旧 yaml 走 §13.3 validator 规则 3 显式 reject）
- 测试：3 个 重写 test_online_factors.py / test_offline_factors.py / test_topk_variance.py
- 旧 `runtime_continuity.py / source_window.py / consensus.py` **此批不动**；旧 yaml 走旧 `_build_inner_judge` 分支仍可加载（新 schema validator 在 B5 落地）

### B3 第 3 层 Calibration
- `factors/calibrations/base.py` Protocol
- `factors/calibrations/percentile_rolling.py` PercentileRollingCalibration（无 cold-start，`bind_keys` 内部 fail-fast 校验）
- `factors/calibrations/__init__.py` re-export
- **旧 `factors/normalizers/__init__.py` 此批不动**（G1 R3 Item 2）：保留旧 PercentileRollingNormalizer 完整实现，包括 cold_start_strategy 三策略；旧 yaml 走旧 `_build_judge` 仍可加载。alias / deprecation 改造延迟到 B7（B5 之后旧 yaml 已被新 schema validator reject，旧 normalizer 自然成 dead code）
- 测试：3 个 calibration tests + 2 个 source loading tests

### B4 第 4 层 Composer
- `factors/composers/base.py` Protocol（含 `declared_dependencies`）
- `factors/composers/{weighted_sum, and_gate, or_gate}.py` 4 子类 + WarmFallback 变种
- `factors/composers/__init__.py` re-export
- 测试：重写 test_composers_*.py + 新 test_composer_warm_fallback.py

### B5 装配层
- `components/composite_judge.py` 新 CompositeJudge（4 层装配）
- `components/dumping_judge.py` 新 DumpingJudge（(b) 模式 + 文件 handle 复用 + 异常 log）
- `components/legacy_judge.py` 把 AlwaysHit / AlwaysWarm / Threshold / SimilarityJudge / HitType / JudgeResult 搬过来
- `components/judge.py` 改为 facade re-export
- `cache/__init__.py` 必要时更新 re-export
- `cache/config.py` 新 dataclass + 15 项 validator + builder（_build_judge / _build_normalization / _build_calibration / _build_composer / _build_dump_extractors）
- 测试：重写 test_composite_judge / test_dumping_judge / test_config / test_config_factor / test_dump_config / test_factor_validator_*

### B6 实验侧适配（minimal）
- `exp/verdict_factor_judge/phase1_spec.py` 改 17 因子新名 + 4 层 yaml schema
- `exp/verdict_factor_judge/phase2_spec.py` 同
- `exp/verdict_factor_judge/phase2_layer2_spec.py` 同（`_r_*` recipe builder 重写）
- `exp/verdict_factor_judge/generate_yamls.py` writer 适配
- `exp/verdict_factor_judge/per_step_log_writer.py` schema_version=2
- `exp/common/factor_postprocess.py` 17 因子 enrich
- `exp/common/build_*.py` `--factors-yaml` 适配（按现行 CLI 直接接 `--factors-yaml`，不动其余参数）
- **G1 R2 Item 3 新增**：`exp/common/build_in_memory_cache_artifact.py` 加 `enrich-existing-pkl` 子命令（或 `--enrich-from <existing.pkl>` flag），实现"读已有 pkl entries + library_stats → 跑新 17 因子 OfflineWriter → 写新 pkl，原 entries 不重新生成"路径。CLI 设计：
  ```
  uv run python -m exp.common.build_in_memory_cache_artifact enrich-existing-pkl \
      --input  exp/warm_start/data/spatial16/cp1_spatial_pool_16.pkl \
      --factors-yaml exp/verdict_factor_judge/config/v2_spatial16/factors_full_17.yaml \
      --output /tmp/cp1_spatial_pool_16_v2.pkl
  ```
- **G1 R3 Item 4 新增**：`exp/common/factor_postprocess.py` `enrich_artifact_with_factors` 签名扩展为：
  ```python
  def enrich_artifact_with_factors(
      entries: list[CacheEntry],
      offline_writers: list[OfflineWriter],
      *,
      library_stats: LibraryStats | None = None,
      ...,
  ) -> LibraryStats:
      """If `library_stats` is None, compute via `LibraryStats.compute_from_entries`
      (current behavior, used by from-scratch HDF5 build). If `library_stats` is
      passed (e.g. read from input pkl during enrich-existing-pkl), reuse directly
      without recomputing — the smoke command MUST pass `library_stats` to avoid
      a 30+ min recompute on every smoke run."""
  ```
  - 旧调用方（HDF5 build pipeline）继续不传 `library_stats`，行为字节不变（向后兼容）
  - `enrich-existing-pkl` 子命令显式传 `library_stats=input_pkl["library_stats"]`，确保不重算
  - 内部 entries 字段（query_keys / payload.action_chunk / 旧 payload.factors keys）保留；新 17 因子 keys 增量 merge 进 `payload.factors`
  - 单测：`tests/exp/common/test_build_enrich_existing_pkl.py`（新增），覆盖 happy path（传 library_stats，不重算）+ 缺失 library_stats fallback（None → compute_from_entries）+ factors yaml 含未注册 type 时 reject 三个分支
- 测试：tests/exp/verdict_factor_judge/test_phase1_spec_warmup_yamls.py 重写

### B6.5 单 cfg pkl 重建 smoke + 回滚（G1 R1 Non-2）
> 在 B6 完成后、B7 删旧代码前插入。验证新 OfflineWriter + 新 17 因子在真实 artifact 上跑得通，再扩展到 6 份。

```bash
# 1. 单 cfg smoke：spatial16 唯一一份；用 B6 新加的 enrich-existing-pkl 子命令（G1 R2 Item 3）
uv run python -m exp.common.build_in_memory_cache_artifact enrich-existing-pkl \
    --input  exp/warm_start/data/spatial16/cp1_spatial_pool_16.pkl \
    --factors-yaml exp/verdict_factor_judge/config/v2_spatial16/factors_full_17.yaml \
    --output /tmp/cp1_spatial_pool_16_v2_smoke.pkl

# 2. 字段检查（acceptance criteria）
uv run python -c "
import pickle
with open('/tmp/cp1_spatial_pool_16_v2_smoke.pkl', 'rb') as f:
    art = pickle.load(f)
entries = art['entries']
print(f'entries: {len(entries)}')
# 旧 168 keys 仍在
sample = entries[len(entries)//2]
old_keys = [k for k in (sample.payload.factors or {}) if k.startswith(('f1a_', 'f1b_', 'f2_'))]
new_keys = [k for k in (sample.payload.factors or {}) if k.startswith(('jerk_', 'direction_', 'dispersion_', 'path_length_', 'topk_'))]
print(f'old keys (preserved): {len(old_keys)}')
print(f'new keys (written):   {len(new_keys)}')
# library_stats 可读
ls = art['library_stats']
print(f'lib stats: action_sigma {tuple(ls.action_sigma.shape)}, state_sigma {tuple(ls.state_sigma.shape)}')
"

# 3. 通过 acceptance：new keys >= 100（4 desc × 8 variants × 多窗口典型 100+）；old keys 全保留
# 4. 失败回滚：直接删 /tmp/*_v2_smoke.pkl，旧 pkl 完整不动；下一次 build 修 spec 后重跑

# 5. smoke pass 后扩展：六份并行
for cfg in clip max_pool spatial16; do
    for kb in clip_vit_b_32 cp1_max_pool cp1_spatial_pool_16; do
        # ... 同步骤 1-3，输出 *_v2.pkl
    done
done
```

回滚条件：smoke 步骤 2 中 `new keys < 100` 或 step 3 失败 → reject B6.5、回 B6 修 spec、重跑 smoke；不删旧 pkl，不进 B7。

### B7 旧代码清理 + 验证
> G1 R2 Item 2 修订：旧 facade 采用 **import-only deprecation stub** 策略（不写 alias，不做参数 wrapper）。
>
> 理由：旧类 `RuntimeContinuityAction` 等接受 `descriptors: list[str] / window_k: int`；新类 `JerkOnlineAction` 接受 `windows: list[(P, F)]`。直接 alias `RuntimeContinuityAction = JerkOnlineAction` 只能避免 `ImportError`，但旧 test 一旦构造 `RuntimeContinuityAction(descriptors=["jerk", "dir"], window_k=5, library_stats=ls)` 会因为参数名不匹配抛 hidden TypeError，调试痛苦。写真实兼容 wrapper 又违背"清理石山"动机。
>
> 选 (a) **import-only 兼容 + 构造时显式 raise**：旧类名作为可被 import 的 deprecation stub，构造时立即抛 `NotImplementedError` 指向新类与迁移文档。这保证：
> 1. `from openpi.cache.components.factors.runtime_continuity import RuntimeContinuityAction` 不抛 ImportError（review_tests collection 不断）
> 2. 任何尝试构造旧类的测试 → 显式失败，不是 hidden bug；错误消息指向新类与 §6.11 #7 命名表
>
> §14.4 + §17 R1 中"类与方法签名全保留"措辞同步收紧为"**legacy class import path 保留；构造行为是 deprecation raise**"。

- 旧 factor 文件改为 import-only deprecation stub：
  - `factors/runtime_continuity.py`：保留旧类名（`RuntimeContinuityAction / RuntimeContinuityState / _RuntimeContinuityBase`），每个旧类的 `__init__` 第一行 `raise NotImplementedError("RuntimeContinuityAction has been replaced by Jerk/Direction/Dispersion/PathLength variants in factors.online; see logs/verdict_factor_judge_refactor.log.md §6.11 #7 for the mapping table")`；旧类不持有任何状态、不实现 `extract`
  - `factors/source_window.py`：同上策略，旧 `SourceWindowSmoothnessAction / SourceWindowSmoothnessState / _SourceWindowSmoothnessBase` 全部 stub
  - `factors/consensus.py`：旧 `TopKActionConsensus` stub（指向 `TopkActionVariance`，区别仅类名）
  - `factors/normalizers/__init__.py`（G1 R3 Item 2 阶段策略）：B1-B6 期间保留完整旧实现；B7 才改为 deprecation stub —— 即在此 B7 步骤把 `PercentileRollingNormalizer.__init__` 第一行替换为 `raise NotImplementedError("PercentileRollingNormalizer has been replaced by PercentileRollingCalibration in factors.calibrations; cold_start_strategy is removed (no cold-start), see logs/verdict_factor_judge_refactor.log.md §6.3.1 #7")`；保留旧类名作 import-only stub。前提是 B5 装配层已让新 schema validator reject 所有旧 yaml（§13.3 规则 12），因此 B7 替换时已无 runtime caller，不会触发 NotImplementedError
  - `OnlineExtractor / OfflineWriter` Protocol 旧名（如有用）：从 `factors.base` 直接 re-export 到旧 path 即可（接口字面相同，仅 `library_stats` 入参名维持），这部分不需要 deprecation stub
- 删除旧 yaml schema dataclass：
  - `JudgeConfig.normalizer` 字段
  - `JudgeConfig.all_nan_fallback` 字段
  - 旧 `FactorConfig` 仅当字段语义变 → 直接修改不删
- `git log --all -- src/openpi/cache/components/factors/runtime_continuity.py` 确认 commit `d9bf877` ... `87eff6a` 5 次旧版本可追（git history 是老代码唯一保留通道，§6.14）
- 文档同步（G1 R1 Item 8 — Working Agreement §8 赋予下列文档架构约束地位，必须在 B7 一并更新，与 `docs/README.md` index 一同 commit）：
  - `docs/cache/verdict_factor_judge.md` — 旧 5 因子 → 新 17 因子 + 4 层架构 + 4 desc 改名 + no cold-start
  - `docs/architecture/cache_system.md` §5.12（旧 Verdict Factor System）→ 重写为 4 层架构 + 17 因子 + capability flag 修订
  - `docs/architecture/cache_system.md` §5.13（旧 wire-level observability + warmup preload）→ 修订 wire schema_version=2 + factor_outputs 字段名
  - `docs/cache/tutorial.md` §6（旧 Judge / Normalizer 配置示例）→ 改为新 4 层 yaml 示例
  - `docs/README.md` index 同步上述 3 份 docs 描述行
- 跑全测试：`uv run pytest tests/cache/ tests/serving/ tests/exp/`（不读 `tests/review_tests/`）
- manual verify：跑一个新 yaml 端到端（spatial16 v2 + 17 因子超集 warmup yaml → eval yaml；client per-step jsonl 应含 `factor_outputs.schema_version=2`）

---

## 17. 风险登记

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| R1 | `tests/review_tests/` 不可见 | 旧测试可能引用旧 17 之前的 factor 注册名 / 旧 yaml schema → G2 reviewer 那边 PASS 不了 | §14.4 / §16 B7：legacy class **import path 保留** 不抛 `ImportError`（pytest collection 不断），但**构造旧类立即 raise NotImplementedError** 指向新类与 §6.11 #7 命名表（不写 alias，旧测试若仅 import 行为不变；旧测试若构造 → 显式失败，不是 hidden TypeError）。如 G2 reviewer 仍断 → 接受这是设计变更，回 §3 G1 复审 |
| R2 | 6 份 libero pkl artifact 重生成 | 一份约 30-60 min，6 份 cumul 3-6h | B7 阶段单 cfg pkl 先验通过；并行跑余 5 cfg；旧 keys 不删可回滚 |
| R3 | min_required_top_k 注册顺序 | 17 因子 union 含 topk_action_variance K 时，search strategy `min_top_k_hint` 升 → 召回行为变 | CompositeJudge.__init__ 打印 hint；测试覆盖 |
| R4 | DumpingJudge 第 1 层副本配置不一致 | dump 路 z-score 与 inner 路不一致 → dump factor raw 与 inner factor raw 在 wire 上量纲不可比 | §13.3 validator 规则 11：`dump.normalization.stats_source` 与 `judge.normalization.stats_source` 强制一致（Non-1 改名 + R2 sweep） |
| R5 | wire schema_version=2 旧客户端兼容 | 旧 client 读不到 schema_version 直接崩 | client 用 `dict.get("schema_version", 1)` 取；server 不主动发未配置 export_factor_outputs 的字段 |
| R6 | 旧 phase2 jsonl 分析脚本 lib_stats schema 变化 | `LibraryStats` 字段名改了的话旧脚本断 | §4 ❶ 已确认 LibraryStats 字段名不变；新增字段不影响旧读取 |
| R7 | 17 因子注册一次性引入 → 性能回归 | 17 因子全启用时 verdict latency 17× 单因子 | 测试覆盖：单因子 baseline / 17 因子全开的 wall-clock 比较；典型 yaml 只启 3-5 因子 |
| R8 | calibration source=offline path 安全 | yaml 配 path traversal / symlink escape | `Path(...).resolve()` + 限制根目录（沿用 dedicated_runner_plan §B2.5 警告） |
| R9 | 新 yaml schema 把 `judge.normalizer / all_nan_fallback / cold_start_strategy` 字段废了 | 旧 phase0/1/2 yaml load-time reject | **G1 R1 Item 7 修订**：放弃 stem namespace 判断（config.py 看不到 stem 语义）；改为 §13.3 validator 规则 2 + 12 显式 reject — 旧字段出现即 raise 并提示 "looks like legacy schema, rewrite to 4-layer"。旧 yaml 不再 server-loadable，仅供旧分析脚本读旧 jsonl |
| R10 | B1-B4 期间新旧代码并存 + 注册名 17+5 = 22 个，旧 yaml 仍能加载到旧 _build_judge 路径 | 期间任何对旧 5 因子的实验跑结果与新 17 因子混淆 | B1-B4 仅做"新代码增加，旧代码不动"，旧 yaml 行为字节不变；B5 装配层落地后旧 yaml load-time reject（§13.3 规则 12），cut-over 一次性发生 |
| R11 | 旧 5 因子 registry 名（f1a_a/f1a_t/f1b_a/f1b_t/f2）在 B7 移除注册时，所有引用旧名的 yaml 立即变 `Unknown factor name` | 用户旧实验脚本错误信息晦涩 | §13.3 规则 3 明确：factor.type 含旧 5 名时 reject 并提示"legacy factor name; replaced by `<new_name_template>`, see logs/verdict_factor_judge_refactor.log.md §6.11 #7" |

---



### 10.1 已完成
- 8 个旧 verdict_factor logs 加 `old_` 前缀归档。
- `logs/README.md` 同步更新。
- §1-§5 现状调研：动机 / 范围 / 依赖矩阵 / 硬契约 / 潜在坑。
- §6 重构后目标架构（14 节，含 4 层架构 / online splice / DumpingJudge / 诊断字段 / 等价性核对 / 实验侧 / 老代码）。
- §6.11 用户决策清单 14 项 + A-E 全收尾。
- §7 新架构对原硬契约冲击评估。
- §8 落盘数据迁移影响。
- §9 剩 2 项决策（#15 #16）已 RESOLVED。
- §11 4 层 Protocol 最终签名（含 dataclass + Protocol 定义 + CompositeJudge / DumpingJudge 重写代码）。
- §12 17 因子最终实现矩阵 + 文件组织 + 共享 base class 代码骨架。
- §13 yaml 新 schema 完整定义 + 11 项静态 + 4 项运行时 validator。
- §14 测试策略（review_tests 兼容 / 重写清单 / 新增清单）。
- §15 实验数据处理（不迁移 + 物理分隔策略）。
- §16 改动分批 B0-B7。
- §17 风险登记 11 项。

### 10.2 待办
- 用户审本文 §11-§17 详细 Plan
- 用户确认后：
  1. 提交本 plan log（`git add` 等用户显式指示）
  2. 状态从 `In Progress` 推进到 `Plan` 或 `Plan (G1 pending)`
  3. 进 §3 G1：state to user verbatim "G1 gate reached. Please initiate a separate Review Authority session to audit `logs/verdict_factor_judge_refactor.log.md`"
  4. halt code modifications 等 G1 verdict

---

<!-- G1 Review Log deleted per execution_authority.md §3.1 Post-G1 polish. G1 verdict: APPROVED at Round 4 (2026-05-07 13:34 CDT). -->

## Review Log

### G2 Round 1 — Reviewer — REJECTED — 2026-05-07 15:09 CDT

- [Blocking] [Concern] `weighted_sum_with_warm_fallback` is planned, documented, tested as a concrete Composer subclass, and generated by `exp/verdict_factor_judge/v2_spec.py`, but the config schema/build path cannot instantiate it. `ComposerConfig` has only flattened `weights/tier_thresholds/per_factor_thresholds/warm_start_t/directions` fields and no `params` or `warm_fallback_start_t`; `_build_composer()` only accepts `weighted_sum`, `and`, and `or`. A direct smoke check raises `ConfigValidationError: Unknown composer.type 'weighted_sum_with_warm_fallback'`. — reasoning: all generated v2 eval recipes use this composer, so the eval YAMLs cannot reach runtime even though unit tests for the class itself pass.
- [Blocking] [Concern] The v2 YAML generator emits invalid search-strategy configs for the current validator. `CFG_SPECS` sets `trajectory_depth` to 4 or 5 but never emits `trajectory_weights`, while `validate_cache_config()` requires `trajectory_weights` whenever `trajectory_depth > 1`. A generated eval YAML fails `load_cache_config()` with `checkpoints.cp1.search_strategy: trajectory_weights required when trajectory_depth=4`; it also emits an ignored `search_strategy.weights` key. — reasoning: G2 requires generated experiment YAMLs to pass the project config validator, not merely have the expected dictionary shape.
- [Blocking] [Concern] The warmup YAML generator does not actually over-collect a descriptor-key superset for its sibling eval YAMLs. `build_warmup_yaml()` dumps only `(1,1)`, `(3,3)`, and `(5,5)` windows, but generated eval recipes require `(7,7)`, `(0,3)`, and `(0,5)` keys. A registry-based key-coverage smoke check reports missing `jerk_online_state__p7_f7`, `jerk_online_state__p0_f3`, `jerk_online_state__p0_f5`, and `jerk_offline_state__p7_f7` across the generated recipes. — reasoning: Layer 3 calibration is no-cold-start and fail-fast per plan; missing warmup keys will make eval-side `PercentileRollingCalibration.bind_keys()` reject the config.
- [Blocking] [Concern] Composite judge `dump` validation is internally inconsistent with `DumpConfig`. The approved plan includes `DumpConfig.normalization`, and `_validate_composite_judge_static()` dereferences `judge.dump.normalization`, but the implemented `DumpConfig` dataclass has no such field. A minimal composite+dump validator smoke check raises `AttributeError: 'DumpConfig' object has no attribute 'normalization'`. — reasoning: composite judges with dump enabled crash in validation instead of producing a controlled `ConfigValidationError`, and the planned dump-side normalization consistency rule is not implementable as written.
- [Blocking] [Concern] Documentation and indexes are not synchronized with the landed implementation. `docs/cache/verdict_factor_judge.en.md` still describes the pre-refactor 5-factor / `PercentileRollingNormalizer` / cold-start architecture and links the old plan, `logs/README.md` still marks `verdict_factor_judge_refactor.log.md` as "待用户审 → G1", and `docs/cache/README.md` does not list the verdict-factor guide. — reasoning: G2 requires docs and indexes updated; the primary Chinese guide is updated, but published companion/index paths still point readers at obsolete contracts.
- [Blocking] [Concern] Test coverage misses the failing integration seams above. The reviewer ran `PYTHONPATH=. uv run pytest tests/cache tests/exp/verdict_factor_judge tests/exp/common tests/serving -q` and got `800 passed, 4 skipped`, but independent smoke checks still found generated v2 YAMLs that fail `load_cache_config()`, an unbuildable generated composer type, missing warmup calibration keys, and a composite+dump validator crash. — reasoning: passing unit tests are not sufficient for G2 approval when the generated configuration and validator/build integration paths are not covered.

### G2 Round 2 — Executor — 2026-05-07

- Accepted (Item 1, Blocking) — `weighted_sum_with_warm_fallback` is now buildable end-to-end. `ComposerConfig` gained the `warm_fallback_start_t: Optional[float]` field; `_build_composer` accepts `weighted_sum_with_warm_fallback` (sharing the `weighted_sum` weights / tier_thresholds plumbing) and constructs `WeightedSumWithWarmFallbackComposer` with the fallback `start_t`. `_validate_composite_judge_static` rules 4 + 5 + 5d are widened to treat `weighted_sum_with_warm_fallback` as a member of the weighted-sum family (composer-keys = non-zero weights, full_hit / warm_start tier check, `warm_fallback_start_t` must be a canonical denoise timestep). New integration test `tests/exp/verdict_factor_judge/test_v2_spec_integration.py::test_generated_eval_yaml_passes_validator` runs every `v2_spec.GENERATION` cell through `load_cache_config` so the `weighted_sum_with_warm_fallback` build path is exercised on real yamls.
- Accepted (Item 2, Blocking) — `v2_spec.CFG_SPECS` no longer emits the spurious `search_strategy.weights` field (SearchStrategyConfig has no such attribute; fusion weights come from the yaml top-level `keys.<name>.weight`). Each cfg now sets `trajectory_weights = _exp_decay_weights(trajectory_depth)` (newest-first exponential decay 0.7) so the `trajectory_depth > 1` validator branch is satisfied. The above integration test confirms each generated eval yaml passes `load_cache_config` end-to-end.
- Accepted (Item 3, Blocking) — `v2_spec.build_warmup_yaml` now takes an optional `eval_factors` argument and constructs the dump-side superset via the new `_build_dump_factor_superset` helper: it groups the eval recipe's `(descriptor, source, channel) → list[windows]`, unions each group with `_W_UNION_DEFAULT` (covers `(0,3) (0,5) (1,1) (3,0) (3,3) (5,5) (7,7)`), and emits one dump factor per (descriptor × source × channel) carrying the merged window list. `main()` now passes the eval recipe's factors block into `build_warmup_yaml` so generated warmup yamls actually cover the demand keys. New integration test `test_warmup_dump_covers_eval_factor_keys` walks every cell in `v2_spec.GENERATION` and asserts `eval_keys ⊆ dump_keys` (registry-based key derivation), so the failure mode the reviewer surfaced is now in CI.
- Accepted (Item 4, Blocking) — `DumpConfig` gained the `normalization: Optional[NormalizationConfig] = None` field documented in the plan. `_validate_composite_judge_static` rule 11 now matches the dataclass shape (it dereferences the field only when set). New integration tests `tests/cache/test_composite_dump_validator_integration.py::{test_composite_with_dump_no_normalization_passes,test_composite_with_dump_matching_normalization_passes}` cover both the unset-default and the matching-stats_source paths through `load_cache_config`, so the validator no longer raises `AttributeError` and the dump-side normalization consistency rule is implementable.
- Accepted (Item 5, Blocking) — Documentation and indexes synchronized:
  - `docs/cache/verdict_factor_judge.en.md` rewritten as a stub pointing at the refactored Chinese guide + the plan log; legacy 5-factor narrative removed
  - `docs/cache/README.md` adds the `verdict_factor_judge.md` row (with EN companion link) under the cache user guides
  - `logs/README.md` updates the refactor log entry: status flipped from "In Progress (Plan 完成，待用户审 → G1)" to "Implemented (G1 APPROVED Round 4 / B0-B7 已 land / G2 进行中)" and the description summarises the cut-over (deviation, 17-factor flat layout, 4-layer architecture, no-cold-start, schema_version=2, v2_spec generator)
- Accepted (Item 6, Blocking) — Integration test seams added covering exactly the failing paths the reviewer hit:
  - `tests/exp/verdict_factor_judge/test_v2_spec_integration.py` (3 parametrised tests × 5 generated cells = 15 assertions): every generated eval yaml + every sibling warmup yaml passes `load_cache_config`; the warmup dump factor list always covers the eval factor keys via the registry-derived `describe()` map
  - `tests/cache/test_composite_dump_validator_integration.py` (2 tests): composite + dump with unset / matching `dump.normalization` both round-trip through `load_cache_config`
  - Net delta: `pytest tests/cache/ tests/serving/ tests/exp/ -q --ignore=tests/exp/test_generate_batches.py --ignore=tests/exp/test_run_gate_sweep.py` now reports `1018 passed, 4 skipped` (1001 from G1 polish + 17 new integration assertions). The two pre-existing exp env failures excluded by the ignore list are unrelated to the refactor.

→ Plan body has not changed in this round (G1 plan stays APPROVED at Round 4); only `config.py` schema / validator / builder, `v2_spec.py` generator, the doc indexes, and the new integration tests changed. Please re-audit §11.5 (Composer Protocol — `weighted_sum_with_warm_fallback` build path), §13 (DumpConfig + validator rule 11), §15 (v2 yaml namespace), §6.13 (factor_postprocess + enrich-existing-pkl smoke), and the new integration tests.

### G2 Round 2 — Reviewer — REJECTED — 2026-05-07 15:26 CDT

- [Blocking] [Concern] The primary Chinese user guide still shows a `composer.params` YAML shape for `weighted_sum_with_warm_fallback`, but the implemented `ComposerConfig` schema is flat (`weights`, `tier_thresholds`, `warm_start_t`, `warm_fallback_start_t`, `directions`) and has no `params` field. A doc-style smoke YAML logs `Unknown config key 'params' in ComposerConfig, ignoring.` and then fails validation because `tier_thresholds.full_hit` is missing. — reasoning: G2 docs/index sync is still failing for the main user-facing example; a reader following `docs/cache/verdict_factor_judge.md` will produce a rejected config even though the v2 generator emits the flat shape.
- [Blocking] [Concern] `weighted_sum_with_warm_fallback` introduces a second WARM_START emission path, but `_validate_composite_judge_static()` only applies the CP1-only guard to `composer.warm_start_t`, not to `composer.warm_fallback_start_t`. A direct validator smoke check accepts a CP3 composite judge with `type='weighted_sum_with_warm_fallback'` and `warm_fallback_start_t=0.7` (`cp3_fallback_accepted`). — reasoning: the validator comment and approved plan rule say warm-start emission is CP1-only because CP3 has no intermediates payload to resume from; the all-NaN fallback path returns `JudgeResult(WARM_START, start_t=warm_fallback_start_t)`, so it must be rejected on CP3 as well.
- [Blocking] [Concern] The new Round 2 tests cover generated YAML validator shape and dump-key coverage, but they still miss the two integration seams above: no test asserts the documentation-style composer schema is either supported or removed from docs, and no test asserts `warm_fallback_start_t` is rejected on CP3. The reviewer ran the executor's broad command successfully (`1018 passed, 4 skipped`) and additionally verified generated eval YAMLs can reach `build_per_connection_components()`, but these independent smoke checks still found uncovered failures. — reasoning: G2 test coverage is not sufficient while a published example remains invalid and a plan-mandated CP1-only rule is missing for a new WARM_START path.

### G2 Round 3 — Executor — 2026-05-07

- Accepted (Item 1, Blocking) — `docs/cache/verdict_factor_judge.md` §3 / §7.4 now publish the **flat** ComposerConfig shape that matches the implemented dataclass. The §3 yaml example places `weights` / `tier_thresholds` / `warm_start_t` / `warm_fallback_start_t` / `directions` directly under `composer:` (no `params:` wrapper); a callout above the example states this explicitly. §7.4 gains a "schema 形状" callout and a per-row reminder that fields are flat, plus a CP3 limitation note on `warm_start_t` / `warm_fallback_start_t`. New regression test `tests/cache/test_warm_fallback_cp_guard.py::test_doc_style_flat_composer_shape_loads` round-trips the documented yaml shape through `load_cache_config` field-by-field; companion test `test_legacy_params_wrapper_is_ignored_so_doc_must_not_use_it` locks the failure mode (composer.params wrapper → reject) so a future doc edit can't silently regress.
- Accepted (Item 2, Blocking) — `_validate_composite_judge_static` now applies the CP1-only guard to `composer.warm_fallback_start_t` in addition to `composer.warm_start_t`. Both are WARM_START emission paths and CP3 has no `intermediates` payload to resume from regardless of which path emits the WARM_START (plan §3.6 / §13.3 rule 5c). The check is colocated with the existing `warm_fallback_start_t` validator block (canonical-timestep + required-when-fallback) and emits a clear "use composer.type='weighted_sum' on CP3" hint. New regression tests `test_warm_fallback_start_t_accepted_on_cp1` and `test_warm_fallback_start_t_rejected_on_cp3` lock the rule end-to-end through `load_cache_config`.
- Accepted (Item 3, Blocking) — `tests/cache/test_warm_fallback_cp_guard.py` covers exactly the two integration seams the reviewer surfaced:
  - 4 tests on the CP1/CP3 × `warm_start_t` / `warm_fallback_start_t` matrix (the new fallback-path guard plus the existing `warm_start_t` rule, locking the regression baseline)
  - 2 tests on the documentation-shape regression (the doc-published flat shape loads; the legacy `composer.params` wrapper is now explicitly rejected so the docs cannot silently drift back)
  - Net delta: `pytest tests/cache/ tests/serving/ tests/exp/ -q --ignore=tests/exp/test_generate_batches.py --ignore=tests/exp/test_run_gate_sweep.py` now reports `1024 passed, 4 skipped` (1018 from Round 2 + 6 new regression assertions). The two pre-existing exp env failures excluded by the ignore list remain unrelated to the refactor.

→ Plan body unchanged in this round (G1 plan still APPROVED at Round 4); only `docs/cache/verdict_factor_judge.md` §3 / §7.4, `config.py` validator, and the new regression test file changed. Please re-audit those four locations.

### G2 Round 3 — Reviewer — APPROVED — 2026-05-07 15:35 CDT

- [Pass] Documentation/schema alignment is restored for the primary Chinese guide. `docs/cache/verdict_factor_judge.md` now publishes the implemented flat `ComposerConfig` shape (`weights`, `tier_thresholds`, `warm_start_t`, `warm_fallback_start_t`, `directions`, `per_factor_thresholds` directly under `composer:`) and explicitly warns that the legacy `composer.params` wrapper is invalid for composer fields. This resolves the Round 2 doc-style YAML blocker.
- [Pass] The CP1-only warm-start invariant now covers both WARM_START emission paths. `_validate_composite_judge_static()` applies the CP1 guard to `warm_fallback_start_t` as well as `warm_start_t`, while retaining canonical timestep validation and required-field validation for `weighted_sum_with_warm_fallback`. A CP3 composite fallback config is now rejected instead of being accepted.
- [Pass] Coverage now locks the two Round 2 seams. `tests/cache/test_warm_fallback_cp_guard.py` covers CP1/CP3 acceptance/rejection for `warm_start_t` and `warm_fallback_start_t`, plus the documented flat composer schema and the legacy `params` failure mode.
- [Verification] `PYTHONPATH=. uv run pytest tests/cache/test_warm_fallback_cp_guard.py -q` → `6 passed, 1 warning`.
- [Verification] `PYTHONPATH=. uv run pytest tests/cache/test_config_factor.py tests/cache/test_composite_dump_validator_integration.py tests/exp/verdict_factor_judge/test_v2_spec.py tests/exp/verdict_factor_judge/test_v2_spec_integration.py tests/cache/components/factors/test_composer_warm_fallback.py tests/cache/test_warm_fallback_cp_guard.py -q` → `56 passed, 1 warning`.
- [Verification] `PYTHONPATH=. uv run pytest tests/cache tests/serving tests/exp -q --ignore=tests/exp/test_generate_batches.py --ignore=tests/exp/test_run_gate_sweep.py` → `1024 passed, 4 skipped, 14 warnings`.
- [Verification] `git diff --check` and `git diff --cached --check` both clean.

Final G2 verdict: APPROVED.
