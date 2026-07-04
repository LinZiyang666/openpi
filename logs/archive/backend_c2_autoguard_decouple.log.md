# Backend C2 Freeze Guard — Interface-Side Auto-Guard (decouple from concrete backends)

**Status**: `In Progress` (L3 — G1 APPROVED / G2 APPROVED 2026-05-25; owner-instructed qdrant `np` F821 fix folded in; §6 Verify done; awaiting commit).
**Authority**: written under Execution Authority. The G1 Review Log was deleted in Post-G1 polish (execution_authority §3.1); a fresh `## Review Log` is opened by the independent G2 Review Authority session (WORKING_AGREEMENT §2.6 / execution_authority §5, §10).

> 本文件对没有对话历史的 reviewer 自洽。变更基线 = 当前工作树（含已落地、尚未 commit 的 `write_policy` fail-fast 改动），目标 commit 为本 plan 的实现。

---

## 1. 目标与背景

### 1.1 问题
C2（runtime write-frozen）契约目前**不是接口侧强制**，而是**每个具体 backend 逐个 opt-in**：
- 基类 `VectorStoreBackend`（`backend_base.py`）只**提供** `freeze()` / `is_frozen` / `_check_frozen()`，但不自动守卫任何 mutation。
- 实际生效靠每个具体 backend 在自己每个写方法开头**手抄** `self._check_frozen(...)`：
  - `in_memory_backend.py`：`insert` / `batch_insert`(override 仅为加守卫) / `delete` / `load_artifact`
  - `qdrant_backend.py`：`insert` / `batch_insert` / `delete`

**后果**：DB backend 是框架的可插拔模块（用户要自由试验不同 backend）。当前设计下，插入一个新 backend 若忘记在写方法里调 `_check_frozen`，`freeze()` 照样设标志位但写操作穿透，**C2 静默失效**。这是 C2 实现与具体 backend 的耦合，违反"DB 修改只应在接口侧"的架构约束。

### 1.2 目标
把 C2 守卫上提到 ABC，使**任意 `VectorStoreBackend` 子类（含本仓看不见的 `tests/review_tests/` 下 reviewer fake）自动获得 C2，零改动、不改方法名**；并删除两个内置 backend 里手抄的 `_check_frozen`。

### 1.3 路线（已与 owner 确认）
`__init_subclass__` **自动包装**（owner 在 2 选项中选定，否决 template-method + 抽象 `_xxx_impl`）。否决理由：抽象 `_xxx_impl` 会让所有只实现了 `insert` 的旧子类实例化失败，而 `tests/review_tests/` 是执行法 §1 封闭区，本 Authority 不能读/搜/迁移其中的 fake 子类 → template-method 路线很可能破坏 G2 reviewer 探针。`__init_subclass__` 不改子类接口契约，对未见子类**by construction** 安全。

---

## 2. 设计

### 2.1 机制（`backend_base.py`）
模块级守卫工厂 + 类级 `__init_subclass__` 钩子：

```python
import functools

def _make_frozen_guarded(op_name, fn):
    """Wrap a backend mutation method so it checks the C2 frozen flag first."""
    @functools.wraps(fn)
    def _guarded(self, *args, **kwargs):
        self._check_frozen(op_name)
        return fn(self, *args, **kwargs)
    _guarded.__c2_guarded__ = True   # idempotency marker
    return _guarded


class VectorStoreBackend(ABC):
    # Mutation entry names auto-guarded on every subclass. A subclass with an
    # extra mutation method can extend this tuple to have it guarded too.
    _MUTATION_METHODS = ("insert", "batch_insert", "delete", "upsert", "load_artifact")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name in cls._MUTATION_METHODS:
            fn = cls.__dict__.get(name)          # only methods defined ON this class
            if fn is None or not callable(fn):
                continue
            if getattr(fn, "__c2_guarded__", False):
                continue                         # already wrapped (idempotent)
            setattr(cls, name, _make_frozen_guarded(name, fn))
```

### 2.2 base 自身的默认 `batch_insert`
`__init_subclass__` 只对**子类**触发，不包装基类自己定义的 `batch_insert` 默认实现（loops `self.insert`）。为让"未 override batch_insert 的子类"也能在 batch 入口 fail-fast，在 base 默认 `batch_insert` 头部保留**一行**显式 `self._check_frozen("batch_insert")`。

> 说明：这一行在**接口层（ABC 自己的默认实现）**，不是具体可插拔 backend。用户关切的是"具体 backend 耦合"；ABC 自守卫其默认实现是合理且必要的，不构成耦合。子类若 override `batch_insert`（如 qdrant 的原生 bulk 实现）则由 `__init_subclass__` 自动包装。

### 2.3 正确性论证
- **幂等 / 继承**：`cls.__dict__.get(name)` 只取本类定义的方法；继承来的（已在父类创建时包装）不在 `__dict__` 中，不会被二次包装。`__c2_guarded__` 标记是第二道幂等保险。
- **super() 链**：子类 wrapped `insert` 内调 `super().insert()`（父类 wrapped）→ 双重 `_check_frozen`，幂等无害（未冻结=两次 no-op；已冻结=第一道即 raise）。
- **顺序保持**：包装在原方法**之前**插 `_check_frozen`。in_memory `insert`/`delete`/`load_artifact` 原本就是"先 `_check_frozen` 再 SearchSessionActive 检查再写"，包装后仍是"`_check_frozen` → (原方法体: session 检查 → 写)"，顺序不变。
- **introspection**：`functools.wraps` 保留 `__name__` / `__doc__` / `__wrapped__` / 签名。
- **非 mutation 同名方法**：`_MUTATION_METHODS` 全是 mutation 语义名；只包装 `callable`，`load_artifact`/`upsert` 不存在则 `__dict__.get` 返回 None 跳过。

---

## 3. 改动文件与职责

| 文件 | 改动 |
|---|---|
| `src/openpi/cache/backend_base.py` | 加 `_make_frozen_guarded` + `_MUTATION_METHODS` 类属性 + `__init_subclass__`；base 默认 `batch_insert` 头部保留单行 `_check_frozen`；更新 `freeze()` / `_check_frozen()` docstring（"concrete backends are expected to call ..." → "subclass mutation methods are auto-guarded via `__init_subclass__`；`_check_frozen` 由包装层内部调用"）；保留 `freeze`/`is_frozen`/`_check_frozen`/`BackendFrozenError` 不变 |
| `src/openpi/cache/backends/in_memory_backend.py` | 删 `insert` / `delete` / `load_artifact` 三处手抄 `self._check_frozen(...)`；**整体删除** `batch_insert` override（其唯一作用是加守卫，删后继承 base 默认 = 行为等价）；随之删除现已 unused 的两个导入 `BatchInsertResult`（storage_types）与 `BackendFrozenError`（backend_base）以避免 F401；`_is_frozen=False` 初始化保留。注：`load_artifact` docstring 中 "Also raises BackendFrozenError post-freeze" 文本保持准确（auto-guard 照样抛该异常） |
| `src/openpi/cache/backends/qdrant_backend.py` | 删 `insert` / `batch_insert` / `delete` 三处手抄 `self._check_frozen(...)`；方法体不变 |
| `docs/architecture/cache_system.md` | 更新 C2 段（约 1129-1132 行）：把"backends are frozen 且 mutation 抛 BackendFrozenError"补充为**接口侧自动守卫**（`__init_subclass__` 透明注入，子类零改动即合规）。L3 强制架构文档同步 |
| `tests/cache/test_backend_frozen_autoguard.py`（新） | 自动守卫机制专项测试（见 §6） |
| `logs/README.md` | 索引同步：Active Logs › Server Infrastructure 加本 plan 行（WA §4） |

**不改**：`config.py` / `backend_pool.py` / `cache_storage.py` / `interceptor.py` / `serve_policy.py`——公共方法名（`insert`/`batch_insert`/`delete`/`load_artifact`）全部保留，所有调用方零感知。`docs/cache/migration.md` 无 C2/`_check_frozen` 措辞，免改。

---

## 4. 接口变化与向后兼容

- **公共 API**：mutation 方法名与签名**完全不变** → `CacheStorage` 及一切调用方不受影响。
- **子类实现契约**：**不变**。子类照常定义 `insert`/`delete`/`batch_insert`/`load_artifact`，无需改名、无需调 `_check_frozen`。现有 + 未见子类（review_tests fake）继续工作，且**额外免费获得 C2**。
- **严格更解耦**：无任何 break。新 backend 作者从此不必知道 C2 的存在即自动合规。
- **可扩展**：子类如有额外 mutation 方法，可在类体里扩展 `_MUTATION_METHODS` 让其被守卫。

---

## 5. 集成点

- `BackendPool`：load 后 `backend.freeze()` 不变；守卫现自动生效。
- `CacheStorage.is_frozen`：`getattr(self._backend, "is_frozen", False)` 不变。
- 离线工具（`exp/common/factor_postprocess.py` 等）在 `freeze()` **之前**建库写入：守卫 pre-freeze 为 no-op，不受影响。
- runtime（任一 server 模式）：backend 经 `build_shared_storage` → pool → frozen，守卫自动拦截写。

---

## 6. 测试策略

新建 `tests/cache/test_backend_frozen_autoguard.py`（无 GPU，纯 Python）：
- **T1 裸子类自动守卫**：定义 `_BareBackend(VectorStoreBackend)`，只实现抽象方法 + `insert`/`delete`，**不写任何 `_check_frozen`**；pre-freeze 写成功，`freeze()` 后 `insert`/`delete` 抛 `BackendFrozenError`。（这是核心验收：证明零 opt-in 即获 C2。）
- **T2 base 默认 batch_insert 入口 fail-fast**：子类不 override batch_insert，frozen 后调 batch_insert 立即抛（不进入逐条循环）。
- **T3 introspection 保真**：`_BareBackend.insert.__name__ == "insert"`、docstring 保留。
- **T4 继承链幂等**：`class B(_BareBackend)` 不 override / override `insert` 两种情形，均恰好守卫一次（无异常、无双重副作用）；frozen 后正确 raise。
- **T5 原生 override 被守卫**：子类提供原生 `batch_insert`（镜像 qdrant），frozen 后该 override 也抛。
- **T6 `_MUTATION_METHODS` 扩展**：子类追加自定义 mutation 名并扩展元组，frozen 后该方法被守卫。

回归锚点（**保持全绿、不改**）：`tests/cache/test_serving_optimization.py` 的 M4 段 `test_freeze_lifecycle_and_idempotency` / `test_frozen_blocks_all_mutation_entries`（验 in_memory 全 mutation entry post-freeze raise）/ `test_frozen_allows_reads_and_session_lifecycle` / `test_cache_storage_propagates_frozen_error` + pool 冻结测试（258+）。

Verify（§6 执行法）：`uv run pytest`，关注 `tests/cache/` 全绿；`ruff --select F` 干净（特别是删 `BatchInsertResult` 后 in_memory 无 F401）。13 个预存 JAX/下载/数据集环境失败与本改动无关（已知基线）。

---

## 7. 风险登记

| ID | 风险 | 缓解 |
|---|---|---|
| R1 | metaprogramming 包装错方法类型（staticmethod/property/classmethod） | 只包装 `cls.__dict__` 中匹配已知 mutation 名且 `callable` 的实例方法；T1-T6 覆盖；文档注明 instance-method 假设 |
| R2 | super() 链双重 `_check_frozen` | 幂等无害（未冻结两次 no-op；冻结第一道即 raise）；T4 覆盖 |
| R3 | 删 in_memory `batch_insert` override 改变返回/行为 | base 默认 `batch_insert` loops `self.insert`，与原 `super().batch_insert()` 路径逐位等价；回归测试 `test_frozen_blocks_all_mutation_entries` + batch 相关测试守住 |
| R4 | 未见的 `tests/review_tests/` fake 子类受影响 | 本路线**不改子类接口契约**，fake 照常定义 `insert` 即工作；若其探针从不 `freeze()`，守卫永远 no-op，行为零变化。封闭区不可读，靠设计不变性而非迁移保证安全 |
| R5 | 删 `BatchInsertResult` 导入引入 F401 / 漏删 | Verify 跑 `ruff --select F`；已确认该符号仅 import 行 + 待删 override 用到 |
| R6 | 文档/docstring 漂移（freeze() docstring 旧措辞变假） | §3 明列 docstring + cache_system.md C2 段同步 |

---

## 8. 解耦声明（WORKING_AGREEMENT §2.5）

本改动是**接口侧增强**：C2 守卫从"具体 backend 手抄"上移到 ABC 的 `__init_subclass__` 透明注入。不改任何推理内部、不改公共方法签名、不改 pool/storage/interceptor。具体 backend 从"必须 opt-in"变为"自动合规"，是更彻底的 wrapper/hook 模式。

## 9. 非目标 / 范围外

- 不引入 backend 注册表（backend 历来是 `config.py` 字符串 `cfg.type` 派发，非本 plan 范畴）。
- 不动 `write_policy` fail-fast 逻辑（已单独落地于工作树）。
- 不实现 `upsert`（全仓无此方法，仅在 `_MUTATION_METHODS` 中预留，未来 backend 若添加则自动被守卫）。
- 不改 C1（non-concurrent）路径。

---

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-05-25 11:04 CDT

- [Blocking] [Concern] `VectorStoreBackend.__init_subclass__` trusts a copyable `__c2_guarded__` marker and can skip wrapping a newly-defined subclass mutation method, letting frozen backends mutate silently — reasoning: `_make_frozen_guarded` sets `__c2_guarded__` on the wrapper (`src/openpi/cache/backend_base.py:71`), and `__init_subclass__` skips any subclass-defined callable carrying that attribute (`src/openpi/cache/backend_base.py:102`). Standard `functools.wraps` copies a wrapped function's `__dict__`, so a subclass override like `@functools.wraps(Parent.insert) def insert(...): ...` inherits `__c2_guarded__ = True` even though the new function is not guarded. An independent probe with such a subclass allowed `insert()` after `freeze()` without raising `BackendFrozenError`. This violates the approved plan's core guarantee that arbitrary `VectorStoreBackend` subclasses get C2 automatically. Recommended fix: do not skip wrapping based on a function attribute copied by `functools.wraps`; since `cls.__dict__.get(name)` already avoids inherited methods, always wrapping subclass-defined mutation callables is simple and safe (double `_check_frozen` remains idempotent), or use an idempotency mechanism that cannot be copied onto an unguarded replacement. Add a regression test for the `functools.wraps(parent.insert)` override case.

- [Non-blocking] [Concern] Full `ruff --select F` over the touched backend files still reports `F821 Undefined name np` in `src/openpi/cache/backends/qdrant_backend.py:315` — reasoning: this is present in `HEAD` and was not introduced by the auto-guard diff, while the new/changed auto-guard files excluding that pre-existing qdrant annotation issue pass `ruff --select F`. Keep it visible for final Verify planning, but it is not a blocker for this G2 item.

### G2 Round 2 — Executor — 2026-05-25

Responses to every G2 Round 1 item (one per item; §10.2).

- [Blocking] copyable `__c2_guarded__` marker skips wrapping a `functools.wraps`-decorated override — **Accepted**. The bug is real and confirmed: `functools.wraps` copies the wrapped function's `__dict__`, so `@functools.wraps(Parent.insert) def insert(...)` inherits `__c2_guarded__=True` and `__init_subclass__` skipped it, leaving the override unguarded after `freeze()`. Fix per the reviewer's first recommended option: **removed the marker entirely** (`_make_frozen_guarded` no longer sets `__c2_guarded__`) **and dropped the skip check** in `__init_subclass__` — it now *always* wraps any mutation method present in `cls.__dict__`. Safe because `cls.__dict__.get` never returns inherited (already-wrapped) methods, so each method is wrapped exactly once per defining class; a `super()`-chaining override merely re-checks the idempotent frozen flag (no correctness impact, no infinite recursion). Docstrings on both `_make_frozen_guarded` and `__init_subclass__` now explain why no copyable marker is used. Regression test added: `test_wraps_parent_override_is_still_guarded` (T7) defines `class _WrapsOverride(_BareBackend)` with `@functools.wraps(_BareBackend.insert)`-decorated `insert` and asserts it raises `BackendFrozenError` post-freeze (fails under the old marker-skip code, passes now). Existing `test_guard_preserves_introspection` (T3) updated to drop the now-removed marker assertion (still asserts `__name__` + `__wrapped__`).
- [Non-blocking] pre-existing `F821 Undefined name np` in `qdrant_backend.py:315` — **Accepted (observation)**. Confirmed accurate and pre-existing at `HEAD` (independently verified via `git show HEAD:...` + `ruff --select F821`); it is a method-local-`import numpy as np` used in a type annotation, untouched by this diff. Per WA §3.1 (no drive-by cleanups) I am not fixing it inside this C2 auto-guard change. For §6 Verify the `ruff --select F` clean claim is scoped to the auto-guard files (`backend_base.py`, `in_memory_backend.py`, `tests/cache/test_backend_frozen_autoguard.py`) and the touched lines of `qdrant_backend.py`; this single pre-existing qdrant annotation F821 is flagged as out-of-scope and left for a separate cleanup.

Re-entering G2 with the updated code + this response round. Changes are in the working tree only (not staged) — per owner standing instruction the executor does not `git add`; will stage on APPROVED.

### G2 Round 2 — Reviewer — APPROVED — 2026-05-25 11:10 CDT

- [Non-blocking] [Concern] The pre-existing `F821 Undefined name np` in `src/openpi/cache/backends/qdrant_backend.py:315` remains visible when running `uv run ruff check --select F src/openpi/cache/backends/qdrant_backend.py` — reasoning: it is present at `HEAD`, was not introduced by the C2 auto-guard change, and does not affect the reviewed runtime frozen-guard behavior. Keep it as a separate cleanup item or account for it in final Verify reporting.

### G2 Round 3 — Executor (post-approval, owner-instructed) — 2026-05-25

G2 was APPROVED in Round 2. Per explicit owner instruction ("顺便修复那个无关的 np 错误，让以后的人不用在这上面费精力") the pre-existing non-blocking `F821 Undefined name np` in `qdrant_backend.py:315` is now **fixed** rather than deferred: added `import numpy as np` under the existing `if TYPE_CHECKING:` block (the annotation `torch.Tensor | np.ndarray` now resolves; zero runtime cost since the file has `from __future__ import annotations`, and the method-local `import numpy as np` at the two runtime call sites are unchanged). `ruff --select F` is now clean on ALL touched files including `qdrant_backend.py` — no separate cleanup item remains. This is an owner-directed L0 lint fix riding along with the approved change; it touches no runtime behavior and is covered by the §6 Verify run.
