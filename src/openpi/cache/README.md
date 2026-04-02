# openpi/cache — 模块状态说明

> 最后更新：2026-04-02

## 整体状态：⚠️ 高危 · 暂时完成

本目录下所有文件均为 **2026-04-02** 新建或重构，当前处于 **高危状态**：
- 接口已按设计文档落地，基本功能可用
- **尚未经过完整测试**（无单元测试、无集成测试覆盖）
- 部分路径（`fetch_payload`、`batch_insert` 失败分支、filter 边界）未经端到端验证
- torch 懒导入改动（2026-04-02 末）尚未在 uv 环境回归

---

## 文件清单

| 文件 | 说明 | 状态 |
|------|------|------|
| `types.py` | 字段名常量（`VISION_0` 等）和 `CheckpointID` 枚举 | ⚠️ 高危 |
| `storage_types.py` | 核心数据类型：`CachePayload`、`CacheEntry`、`QuerySpec`、`QueryFilter`、`SearchResult*`、`BatchInsertResult` | ⚠️ 高危 |
| `backend_base.py` | `VectorStoreBackend` 抽象基类（ABC） | ⚠️ 高危 |
| `cache_storage.py` | `CacheStorage` 门面：线程安全、维度校验、filter 校验、两阶段搜索 | ⚠️ 高危 |
| `timing.py` | `SystemTimer` / `GpuTimerBackend` 等计时组件（Step 2 遗留，本次未改） | 稳定 |
| `interceptor.py` | 推理拦截器（Step 2 遗留，本次未改） | 稳定 |
| `backends/qdrant_backend.py` | Qdrant 后端：命名向量、分块写入/查询、RRF fusion、两阶段搜索 | ⚠️ 高危 |

---

## 高危原因详述

### `storage_types.py`
- `torch` 改为 `TYPE_CHECKING` 懒导入，运行时不再 import torch；注解依赖 `from __future__ import annotations` 保证，若调用方在 Python < 3.10 非注解上下文中使用可能出错
- `CacheEntry.validate()` 逻辑未完整测试

### `cache_storage.py`
- `_check_entry_dims` 要求 backend 声明的所有字段都必须出现在 entry 中，过于严格——目前 exp 脚本的 named mode 只用部分字段，若走 `insert` 路径会报错（当前 exp 只用 `search`，暂时不暴露）
- `metadata_db` 接口预留但未实现，`close()` 调用 `backend.close()` 而 ABC 未定义此方法

### `backends/qdrant_backend.py`
- `_field_to_chunks` 现在同时接受 `torch.Tensor` 和 `np.ndarray`，分支逻辑新增，未测试 chunked numpy 路径
- `_tensor_to_b64` / `_b64_to_tensor` 改为函数内懒导入 torch，在 uv 环境的回归未验证
- chunked vector 的 RRF fusion weight 分配逻辑（`_build_fusion_weights_for_chunks`）计算正确性未独立验证
- `search()` 的 `checkpoint_id` payload 解析若 Qdrant 返回非预期格式会静默跳过条目

---

## 待完成（高危解除条件）

- [ ] 补充 `storage_types` 和 `cache_storage` 的单元测试
- [ ] `qdrant_backend` 集成测试（本地 Qdrant 实例）
- [ ] `_check_entry_dims` 改为只校验 query_keys 与 backend 声明的交集，而非要求全集
- [ ] `backend_base.py` 补充 `close()` 抽象方法
- [ ] 在 uv 环境运行 `uv run pytest src/openpi/cache/` 回归
