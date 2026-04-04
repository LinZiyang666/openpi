# Weekly Plan — 2026-04-03 ~ 04-09

## 1. Cache 真实推理集成 + Config 配置系统（2026-04-04）

**目标**：在真实模型上运行 cache-aware 推理服务，验证 Step 4 全链路。

### 1.1 构建 Cache Config 配置系统
- 新增 cache config dataclass，支持从 CLI / serve_policy.py 传入
- 配置项：backend 类型、vector_dims、threshold、timer 开关等
- 集成到 `serve_policy.py --cache` 启动路径

### 1.2 AlwaysSearch + AlwaysPass 模式真实推理
- Gate: AlwaysSearchGate（每次都查）
- Judge: 阈值设为 0.0（任何结果都命中），验证 write → check → hit 全链路
- 用 InMemoryBackend 先跑通，再切 Qdrant

### 1.3 端到端验证
- 启动 serve_policy.py --cache，用 simple_client 发请求
- 确认：timing 表格输出正常、CP1 write/check 路径触发、cache hit 后 stage2+3 跳过
- 对比 cache 开/关时的 action 输出一致性

### 预期产出
- `src/openpi/cache/config.py` — cache 配置系统
- `serve_policy.py` 集成修改
- 运行日志确认端到端可用
