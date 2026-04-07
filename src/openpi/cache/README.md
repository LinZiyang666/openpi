# openpi/cache — 模块状态说明

> 最后更新：2026-04-04

## 整体状态：⚠️ 高危 · 暂时完成

Step 3（存储层）+ Step 4（Orchestrator 骨架 + Config 配置系统）代码已落地。
- Step 4 单元测试通过（`tests/cache/`）
- Step 3 存储层仍无测试覆盖
- 未经端到端集成验证（需要真实模型 + Qdrant 实例）

---

## 文件清单

### Step 3: 存储层

| 文件 | 说明 | 状态 |
|------|------|------|
| `types.py` | 字段名常量（`VISION_0` 等）和 `CheckpointID` 枚举 | ✅ 稳定（Step 4 测试覆盖） |
| `storage_types.py` | 核心数据类型：`CachePayload`、`CacheEntry`、`QuerySpec`、`QueryFilter`、`SearchResult*`、`BatchInsertResult` | ⚠️ 高危 |
| `backend_base.py` | `VectorStoreBackend` 抽象基类（ABC） | ⚠️ 高危 |
| `cache_storage.py` | `CacheStorage` 门面：线程安全、维度校验、filter 校验、两阶段搜索 | 🟡 部分验证 |
| `timing.py` | `SystemTimer` / `GpuTimerBackend` 等计时组件（Step 2） | ✅ 稳定 |
| `backends/qdrant_backend.py` | Qdrant 后端：命名向量、分块写入/查询、RRF fusion、两阶段搜索 | ⚠️ 高危 |

### Step 4: Orchestrator 骨架

| 文件 | 说明 | 状态 |
|------|------|------|
| `components/key_builder.py` | `QueryKeyBuilder` Protocol + `PlaceholderKeyBuilder`（state-only）+ `FullOriginalKeyBuilder`（vision+prompt+state 原始 flatten） | 🟡 单元测试通过，未集成验证 |
| `components/gate.py` | `GateFunction` Protocol + `AlwaysSearchGate` | ✅ 稳定（极简） |
| `components/judge.py` | `HitType` enum + `SimilarityJudge` Protocol + `ThresholdJudge` | 🟡 单元测试通过，阈值未校准 |
| `components/search_strategy.py` | `SearchStrategy` Protocol + `SearchContext` + `QdrantWeightedRrfKnnStrategy` | 🟡 单元测试通过，未集成验证 |
| `orchestrator.py` | `CacheOrchestrator`：分检查点 dict 编排 + step counter + CP3 stubs | 🟡 单元测试通过，未集成验证 |
| `interceptor.py` | `InferenceInterceptor`：CP1 check/write + CP3 consume/check + on_task_begin 转发 | 🟡 FakeModel 测试通过，未真实模型验证 |

### Step 4: Config 配置系统

| 文件 | 说明 | 状态 |
|------|------|------|
| `config.py` | `CacheConfig` dataclass 树 + YAML 加载 + 校验 + 组件工厂 | 🟡 单元测试通过，未真实模型验证 |
| `backends/in_memory_backend.py` | `InMemoryBackend` 正式实现（从 conftest 提升） | 🟡 间接测试覆盖 |
| `cache.yaml`（项目根目录） | 默认 cache 配置文件 | ✅ 稳定（纯数据文件） |

---

## 不稳定部件详述

### 🔴 Step 3 高危（无测试覆盖）

#### `storage_types.py`
- `torch` 改为 `TYPE_CHECKING` 懒导入，运行时不再 import torch；注解依赖 `from __future__ import annotations`
- `CacheEntry.validate()` 逻辑未完整测试
- `CachePayload.validate_for_checkpoint()` 各 CP 分支未独立测试

#### `backend_base.py`
- `batch_insert()` 默认实现的失败分支未测试
- `close()` 调用 `flush()` 但具体后端的 flush 行为未验证

#### `cache_storage.py`
- `_check_entry_dims` 已修复为交集校验（Step 4 测试覆盖 ✅）
- `metadata_db` 接口预留但未实现
- `_check_filters` 路径未测试

#### `backends/qdrant_backend.py`
- chunked numpy 路径未测试
- `_tensor_to_b64` / `_b64_to_tensor` 懒导入 torch，uv 环境未回归
- RRF fusion weight 分配逻辑未独立验证
- `fetch_payload()` 静默跳过非预期 Qdrant payload 格式

### 🟡 Step 4 部分验证（单元测试通过，待集成）

#### `components/key_builder.py`（PlaceholderKeyBuilder）
- 只使用 `ROBOT_STATE` 一个字段（32维），未利用 vision/prompt embedding
- `prefix_embs` 分离为独立 vision/prompt 需要 Stage API 扩展
- GPU→CPU 转换路径在 FakeModel 测试中实际跑的是 CPU→CPU

#### `components/judge.py`（ThresholdJudge）
- cp1_threshold=0.98 / cp3_threshold=0.95 为占位值，未在真实数据上校准
- 阈值基于 cosine similarity [-1, 1]，如果后端切换到 RRF 分数需要重新校准

#### `orchestrator.py`
- 单元测试使用 InMemoryBackend，与 Qdrant 行为可能有差异
- episode write path (buffer_for_write + on_episode_end) 产生带 prev_ids/next_ids 的链式 entry

#### `interceptor.py`
- CP1 hit 早返回路径的 output 构建与正常路径重复（两处 `jax.tree.map + output_transform`）
- `torch.compile` 路径未测试（测试用 eager mode）

---

## 待完成（高危解除条件）

### Step 3
- [ ] `storage_types` 单元测试：validate_for_checkpoint 各分支
- [ ] `qdrant_backend` 集成测试（本地 Qdrant 实例）
- [ ] `cache_storage._check_filters` 路径测试

### Step 4
- [ ] 真实模型端到端集成测试（serve_policy --cache）
- [ ] ThresholdJudge 阈值校准（真实 state 数据的 cosine 分布）
- [ ] GPU 环境下 interceptor + orchestrator 集成验证
- [ ] CP3 真实实现（Step 6: DeferredWriter）

### 跨步骤
- [ ] 在 uv 环境运行全量 `uv run pytest` 回归，确认不影响已有测试
