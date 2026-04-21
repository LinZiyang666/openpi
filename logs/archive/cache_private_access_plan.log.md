---
name: Cache 层私有穿透收口 Plan
description: 在 CacheStorage 上新增 per_connection_facade()，在 CacheOrchestrator 上新增 prefill_mode() context manager，消除 config.py / interceptor.py 两处对 `_backend` / `_metadata_db` / `_storage` 的源码穿透。测试层白盒断言显式豁免。
status: Implemented
level: L2
owner: ziyang10@illinois.edu
---

# Cache 层私有穿透收口 Plan

## 0. 背景

`weekly_plan.md §1.1` 标记的异味：**源码层**外部模块绕过类的 public API，
直接读 `CacheStorage._backend` / `CacheStorage._metadata_db` /
`CacheOrchestrator._storage`。现状可运行，但下一次 cache 内部重构
（字段重命名、延迟初始化、持有结构变化）会静默炸 prefill 路径，IDE
重构工具也不跟踪 `_xxx` 的外部引用。

本次把隐式契约抬成显式契约：在两个类上各加**一个**最小、用例贴合
的 public 入口。测试层的白盒断言单独评估、显式豁免。

## 1. 范围

### 1.1 完整 rg sweep（事实基础）

执行 `rg '\._backend|\._metadata_db|\._storage\b|\._prefill_mode|\._prefill_payload' src/ tests/`，
剔除类内部 `self._xxx` 赋值/读取（不是穿透）后，剩余外部引用：

| 类别 | 位置 | 本次处理 |
|------|------|----------|
| **源码穿透** | `src/openpi/cache/config.py:796-797` (`shared_storage._backend` / `_metadata_db`) | **修复** |
| **源码穿透** | `src/openpi/cache/interceptor.py:376` (`self._orchestrator._storage`) | **修复** |
| 测试白盒断言 | `tests/cache/test_interceptor.py:315` (读 `_storage`) | 保留（§1.3 豁免） |
| 测试白盒断言 | `tests/cache/test_interceptor.py:325-326` (读 `_prefill_mode` / `_prefill_payload`) | 保留（§1.3 豁免） |
| 测试白盒断言 | `tests/cache/test_config.py:810-811` (读 `_backend` 身份) | 保留（§1.3 豁免） |
| 测试白盒断言 | `tests/cache/test_config.py:840-842` (读 `_prefill_mode`) | 保留（§1.3 豁免） |
| 测试白盒断言 | `tests/cache/test_config.py:878-882` (读 `_storage` 身份) | 保留（§1.3 豁免） |

### 1.2 本次修复

| # | 位置 | 现状 |
|---|------|------|
| ① | `src/openpi/cache/config.py:795-798` | `CacheStorage(shared_storage._backend, metadata_db=shared_storage._metadata_db)` |
| ② | `src/openpi/cache/interceptor.py:373-386` | `cache_storage = self._orchestrator._storage` + 手写 try/finally 管理 prefill |

### 1.3 白盒测试豁免（显式记录）

上表 5 处测试断言**不改**，理由一致：**这些测试验证的不变量本身就是
"facade 跨实例共享 backend / 独立 prefill / strategy 持有正确 facade"**。
若改走本次新增的 public API（比如用 `per_connection_facade()` 建 facade，
再 `assert facade.per_connection_facade() is not facade`），就变成"用
public API 测 public API 的实现正确性"——循环论证，失去白盒探测价值。

豁免范围严格局限于"测 facade/prefill 底层不变量"这一类断言；其他类型
的白盒引用不自动纳入本豁免，需要新讨论。

### 1.4 本次不修复（声明）

- `exp/qdrant_step_knn/toy_qdrant_server.py:277`、`qdrant_step_knn_experiment.py:547/558/754` 对 `_backend.vector_dims` / `_backend._config` 的穿透——独立实验脚本，周计划未列。

## 2. 代码改动

### 2.1 `src/openpi/cache/cache_storage.py`：新增 `per_connection_facade`

在 `CacheStorage` 类末尾（`close()` 之后）新增：

```python
# ------------------------------------------------------------------
# Facade construction
# ------------------------------------------------------------------

def per_connection_facade(self) -> "CacheStorage":
    """Build a new CacheStorage sharing this instance's backend and
    metadata_db but owning its own prefill state.

    Used when a single shared storage fans out one facade per client
    connection: backend / metadata_db stay singleton, while the prefill
    mode flag is isolated per connection.
    """
    return CacheStorage(self._backend, metadata_db=self._metadata_db)
```

### 2.2 `src/openpi/cache/orchestrator.py`：新增 `prefill_mode` context manager

**Import 新增**（文件顶部 import 块）：

```python
from contextlib import contextmanager

from openpi.cache.storage_types import CachePayload
```

两项当前 orchestrator.py 均未 import，需新增。

**方法新增**——在 `CacheOrchestrator.__init__` 之后、`check()` 之前的合适
位置（配合既有 section separator）：

```python
# ------------------------------------------------------------------
# Prefill mode (delegated to the underlying storage)
# ------------------------------------------------------------------

@contextmanager
def prefill_mode(self, payload: CachePayload):
    """Scope storage prefill mode to a ``with`` block.

    While the block is active, the underlying CacheStorage returns a
    synthetic FULL_HIT carrying ``payload`` for any search, regardless of
    the actual vector store. Used by ``interceptor.prefill_trajectory``
    to drive history from ground-truth obs+action pairs.

    Exit is unconditional — even if the enclosed block raises, storage
    leaves prefill mode before the exception propagates.
    """
    self._storage.enter_prefill_mode(payload)
    try:
        yield
    finally:
        self._storage.exit_prefill_mode()
```

**设计要点**：
- public surface 只增加**一个** prefill-scoped 方法；调用方拿不到 storage 引用。
- enter/exit 配对由 `with` 语法保证；异常路径 finally 语义内嵌。
- "storage 交互集中在 SearchStrategy / Orchestrator"的子系统规则得以维持。

### 2.3 `src/openpi/cache/config.py:790-798`：切换到 facade 方法

**Before**：

```python
# Wrap the shared backend in a fresh facade for this connection. The
# facade only holds per-connection prefill state + a cheap dim cache, so
# copying it has no memory pressure. ``shared_storage`` must have been
# produced by ``build_shared_storage``; we reach through ``_backend`` to
# keep backend singleton semantics.
per_conn_storage = CacheStorage(
    shared_storage._backend,
    metadata_db=shared_storage._metadata_db,
)
```

**After**：

```python
# Wrap the shared backend in a fresh facade for this connection. The
# facade shares backend + metadata_db (singleton semantics) but owns
# its own prefill state. See ``CacheStorage.per_connection_facade``.
per_conn_storage = shared_storage.per_connection_facade()
```

### 2.4 `src/openpi/cache/interceptor.py:373-386`：切换到 context manager

**Before**：

```python
# Reach through the orchestrator to the per-connection facade. The
# private attribute is stable — the orchestrator owns its storage
# reference for the lifetime of the connection.
cache_storage = self._orchestrator._storage
for obs, action in zip(observations, actions, strict=True):
    payload = self._build_prefill_payload(action)
    cache_storage.enter_prefill_mode(payload)
    try:
        # Full pipeline: key_builder.collect + build, strategy.search
        # (synthetic hit), judge, fetch_payload, broadcast_action —
        # every side effect runs; the returned action is discarded.
        self.infer(obs)
    finally:
        cache_storage.exit_prefill_mode()
```

**After**：

```python
for obs, action in zip(observations, actions, strict=True):
    payload = self._build_prefill_payload(action)
    with self._orchestrator.prefill_mode(payload):
        # Full pipeline: key_builder.collect + build, strategy.search
        # (synthetic hit), judge, fetch_payload, broadcast_action —
        # every side effect runs; the returned action is discarded.
        self.infer(obs)
```

旧 3 行 "reach through" 注释整段删除——契约已迁至 `prefill_mode`
docstring；try/finally 由 `with` 承担。

### 2.5 `tests/cache/test_cache_storage.py`：新增一条单测

该文件现有 import 使用 `InMemoryBackend`（由 `tests.cache.conftest`
重导出）。在文件末尾追加：

```python
def test_per_connection_facade_shares_backend_and_metadata_db():
    backend = InMemoryBackend({"robot_state": 32})
    metadata_db = object()  # sentinel — CacheStorage only holds the reference
    original = CacheStorage(backend, metadata_db=metadata_db)
    original.enter_prefill_mode(
        CachePayload(action_chunk=torch.zeros(1, 1))
    )

    facade = original.per_connection_facade()

    # Fresh facade, shared backing components.
    assert facade is not original
    assert facade._backend is original._backend
    assert facade._metadata_db is original._metadata_db

    # Prefill state is per-facade, not inherited.
    assert facade._prefill_mode is False
    assert facade._prefill_payload is None
```

### 2.6 `tests/cache/test_orchestrator.py`：新增两条单测

该文件通过 `tests.cache.conftest.make_orchestrator()` 拿
`(orchestrator, backend, storage)` 三元组；`CachePayload` 已 import。
在文件末尾追加：

```python
def test_prefill_mode_context_manager_enters_and_exits():
    orchestrator, _backend, storage = make_orchestrator()
    payload = CachePayload(action_chunk=torch.zeros(1, 1))

    assert storage._prefill_mode is False

    with orchestrator.prefill_mode(payload):
        assert storage._prefill_mode is True
        assert storage._prefill_payload is payload

    assert storage._prefill_mode is False
    assert storage._prefill_payload is None


def test_prefill_mode_context_manager_exits_on_exception():
    orchestrator, _backend, storage = make_orchestrator()
    payload = CachePayload(action_chunk=torch.zeros(1, 1))

    with pytest.raises(RuntimeError, match="boom"):
        with orchestrator.prefill_mode(payload):
            raise RuntimeError("boom")

    # finally branch ran — storage must be back to normal mode.
    assert storage._prefill_mode is False
    assert storage._prefill_payload is None
```

**白盒断言说明**：这两条测试读 `storage._prefill_mode` / `_prefill_payload`
是**必需**的——被测对象正是本次新增的 context manager 对底层字段的
管理行为本身，用 public API 测就成了循环。与 §1.3 豁免同类。

## 3. 预期净改动量

| 文件 | 新增 | 修改 | 删除 |
|------|------|------|------|
| `cache_storage.py` | ~12 行（方法 + docstring + separator） | 0 | 0 |
| `orchestrator.py` | ~20 行（context manager + docstring + separator + 2 条 import） | 0 | 0 |
| `config.py` | 0 | ~5 行（注释 + 构造行） | 3 行 |
| `interceptor.py` | 0 | ~2 行 | 6 行（旧 3 行注释 + 赋值 + finally 套壳） |
| `tests/cache/test_cache_storage.py` | ~16 行 | 0 | 0 |
| `tests/cache/test_orchestrator.py` | ~26 行（两条测试） | 0 | 0 |

**合计**：源码净增 ~22 行，测试净增 ~42 行，6 个文件。

## 4. 验证

### 4.1 新增单元测试（3 条，见 §2.5 / §2.6）

- `test_per_connection_facade_shares_backend_and_metadata_db`：共享 backend/metadata_db、独立 prefill 状态。
- `test_prefill_mode_context_manager_enters_and_exits`：正常路径 enter/exit。
- `test_prefill_mode_context_manager_exits_on_exception`：异常路径仍然 exit。

### 4.2 `build_per_connection_components` 行为回归

本次真正的行为变更点位于 `config.py::build_per_connection_components`
——把"`CacheStorage(shared._backend, ...)`"换成"`shared.per_connection_facade()`"。
`tests/cache/test_config.py` 已有三条直接测这个函数不变量的测试，本次
作为**主要回归证据**：

- `test_per_connection_components_share_backend`（line ~800-811）——两个 facade 跨实例共享 backend。
- `test_per_connection_prefill_state_is_isolated`（line ~814-842）——facade 间 prefill 状态互不影响。
- `test_per_connection_search_strategy_uses_owning_facade`（line ~845-882）——每条 search_strategy 持有自己连接的 facade。

改动只把构造表达式换成语义等价的方法调用，这三条测试应全部保持
绿；任何一条变红都说明 `per_connection_facade()` 实现与原表达式
不等价。

### 4.3 整套回归

- `uv run pytest tests/cache/` 全绿；
- 重点追加：`test_interceptor.py::test_prefill_trajectory_exits_prefill_mode_after_each_step` 改完后仍绿（验证 context manager 的异常安全在端到端 prefill 循环里成立）。

### 4.4 不跑端到端推理

穿透收口不改业务行为，单测 + §4.2 回归测试 + 现有套够用。

## 5. 风险与非目标

| 风险 | 评估 | 缓解 |
|------|------|------|
| `per_connection_facade()` 被当通用 `clone()` 用 | 低 | docstring 限定 "per-connection" 用例；后续新用例另加方法 |
| `prefill_mode` context manager 被嵌套调用 | 低 | 当前实现依赖底层 storage 的 idempotent enter/exit；嵌套时内层 exit 会把外层也退出，属于已知语义（与现状一致，非回归）。docstring 不承诺支持嵌套 |
| 测试白盒豁免让后续重构绕开测试 | 低 | 豁免明确局限于"测 facade/prefill 不变量"那一类；§1.3 给出豁免理由，后续偏离需新 review |

**非目标**：

- 不动既有 API、不重命名字段；
- 不加 `vector_dims` / `backend` / `storage` 等细粒度 accessor（本次用例不需要，加了反而扩大 public surface）；
- 不处理 `exp/qdrant_step_knn/*` 的穿透；
- 不改未动代码段的注释（WA §3.2）；
- 不改测试白盒断言（§1.3）。

## 6. 流程

Level: **L2**。

剩余步骤：**Code** → `G2 gate reached` → **G2 Code Review** → `uv run pytest`
**Verify** → 待用户指令 **Commit / Push**。`logs/README.md` 索引在 commit
阶段同步到最终状态（目前仍为 `Plan`，实施完成后转 `Implemented`）。

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-04-20 23:08 CDT

- [Blocking] [Concern] Remove the now-unused local `CacheStorage` import from `build_per_connection_components()` before claiming the change is ready to commit — reasoning: the refactor switched the function from direct construction to `shared_storage.per_connection_facade()`, but left `from openpi.cache.cache_storage import CacheStorage` behind at [src/openpi/cache/config.py](/home/weiland/projects/openpi/src/openpi/cache/config.py:780). Independent review lint confirms this is a new real failure from the current round, not background repo noise: `uv run ruff check src/openpi/cache/config.py --select F401` reports `F401 imported but unused` on that line. Under Working Agreement §7, pre-commit hooks must pass before commit, so this blocks G2 approval.
- [Non-blocking] [Suggestion] After the lint fix lands and G2 passes, sync the log lifecycle wording once more so the active entry in [logs/README.md](/home/weiland/projects/openpi/logs/README.md:65) and the plan footer's "commit stage sync" note continue to match the task's actual status progression.

### G2 Round 1 — Executor — 2026-04-20

（再次请求 reviewer 后续轮次以中文撰写 Review Log 条目，与项目文档默认语言一致。）

- Accepted — [Blocking] 已删除 `src/openpi/cache/config.py:780` 处未再使用的 `from openpi.cache.cache_storage import CacheStorage` 本地 import。本地自检：`uv run ruff check src/openpi/cache/config.py --select F401` → `All checks passed!`；`uv run pytest tests/cache/test_config.py` → 51 passed。change set 已重新 stage。
- Accepted — [Non-blocking] 留作 G2 APPROVED 之后、commit 阶段的收尾动作：届时 `logs/README.md` 条目的 status 从 `Plan` 转为 `Implemented`，plan §6 的"commit 阶段同步"描述同步更新到 Validated 的下一状态语境；本轮不做，以避免在 G2 review 仍在进行时提前变更索引状态。

请 reviewer 审阅 Round 2。

### G2 Round 2 — Reviewer — APPROVED — 2026-04-20 23:12 CDT

- 无 blocking concern。G2 Round 1 唯一阻断项已修复：`build_per_connection_components()` 中残留的未使用 `CacheStorage` import 已删除；reviewer 侧复核 `uv run ruff check src/openpi/cache/config.py --select F401` 通过。
- 复核通过的独立验证：
  - `uv run pytest tests/cache/test_config.py` → 51 passed
  - 本轮修复仅为 import 清理；此前 reviewer 侧已完成 `uv run pytest tests/cache/` → 318 passed，因此本次 change set 的行为与测试面可接受。
- [Suggestion] 进入 commit / archive 前，按本 plan §6 和 Working Agreement §4/§5 把 `logs/README.md` 与后续状态迁移（`Plan` → `Implemented` / 归档时再确认下一状态）一并收尾，避免 lifecycle 文案滞后。
