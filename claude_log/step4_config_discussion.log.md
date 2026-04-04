# Step 5 Config 配置系统 — 讨论记录

> 开始日期：2026-04-04
> 状态：讨论中

---

## 讨论 1：serve_policy.py 现有命令行参数梳理

### 参数全景

```
serve_policy.py
├── --env            → EnvMode 枚举（aloha/aloha_sim/droid/libero），选环境，决定默认 checkpoint
├── --port           → int，WebSocket 服务端口，默认 8000
├── --default_prompt → str|None，后备 prompt（客户端未提供 prompt 时注入）
├── --record         → bool，启用 PolicyRecorder，逐步保存 obs+action 到 .npy（调试用）
├── --collect        → bool，启用 CollectionPolicy，forward hook 收集 embedding 写 HDF5
│   └── --collect_dir → str，收集数据根目录，默认 ./data
├── --cache          → bool，启用 InferenceInterceptor，走 CP1 check/write 路径
│   └── --timing_csv_dir → str|None，timing CSV 输出目录（None 则只打印终端）
└── policy（tyro union branch）
    ├── Default()        → 使用 env 对应的预设 checkpoint
    └── Checkpoint()     → 自定义加载
        ├── --policy.config → 训练配置名称
        └── --policy.dir    → checkpoint 目录路径
```

### 决议

**Config 系统必须覆盖以上所有参数**，不能遗漏任何一个。

---

## 讨论 2：Cache 系统可配置项梳理

通过阅读 cache 子系统全部源码，提取出所有硬编码或构造函数参数中的可配置项。

### InferenceInterceptor（interceptor.py）

配置很少，是组装入口。核心配置分散在下层组件。

| 可配置项 | 当前值/来源 | 说明 |
|----------|-------------|------|
| `timer` | 构造时传入或默认 `SystemTimer()` | 计时器实例 |
| `orchestrator` | 构造时传入或 `None` | None = 纯 staged 推理，非 None = 启用 cache |

### SystemTimer（timing.py）

| 可配置项 | 当前值 | 说明 |
|----------|--------|------|
| `enabled` | `True` | False = measure() 零开销 no-op |
| `buffer_size` | `10_000` | 环形缓冲区大小 |
| `output_csv_dir` | `None`（已暴露到 CLI） | task 结束时写 CSV 的目录 |

### ThresholdJudge（components/judge.py）

| 可配置项 | 当前值 | 说明 |
|----------|--------|------|
| `cp1_threshold` | `0.98` | CP1 cosine 阈值，⚠️ 占位值未校准 |
| `cp3_threshold` | `0.95` | CP3 cosine 阈值，⚠️ 占位值未校准 |

### PlaceholderKeyBuilder（components/key_builder.py）

当前无可配置项。仅用 `ROBOT_STATE` 32 维。未来可配置使用哪些 query fields。

### AlwaysSearchGate（components/gate.py）

当前无可配置项（永远返回 True）。未来可选 gate 策略类型。

### QdrantBackendConfig（backends/qdrant_backend.py）

| 可配置项 | 当前值 | 说明 |
|----------|--------|------|
| `url` | `http://localhost:6333` | Qdrant 服务地址 |
| `collection_name` | `openpi_cache` | 集合名称 |
| `vector_dims` | `{vision_0: 1024, prompt_emb: 2048, robot_state: 14}` | 命名向量维度 |
| `prefer_grpc` | `False` | 是否优先 gRPC |
| `grpc_port` | `6334` | gRPC 端口 |
| `request_timeout` | `30` | 超时秒数 |
| `rrf_k` | `60` | RRF fusion 参数 k |
| `candidate_multiplier` | `5` | prefetch limit = top_k × multiplier |
| `fusion_weights` | `None`（等权） | 各字段 fusion 权重 |

### InMemoryBackend（tests/cache/conftest.py，仅测试用）

仅 `vector_dims` 一个配置。

---

## 讨论 3：SearchStrategy 组件抽象 + 搜索参数归属

### 背景

搜索逻辑散落在两层：
- `top_k` 硬编码在 Orchestrator（`orchestrator.py:143`，写死 `top_k=1`）
- `rrf_k`、`fusion_weights`、`candidate_multiplier` 在 QdrantBackendConfig
- `step_filter` / `step_window` 只在 exp/ 实验代码中，正式系统未集成

搜索策略会频繁变更（top_k 机制、fusion 方式、自定义 re-ranking 等），每次改动要同时动 Orchestrator + Backend。

### 决议：新增 SearchStrategy 可插拔组件

SearchStrategy 成为 Orchestrator 的第五个可插拔组件（与 Gate、Judge 同级），是数据库搜索的唯一出口。**所有搜索参数归 SearchStrategy 管**。

**职责划分**：

| 组件 | 职责 | 不管什么 |
|------|------|----------|
| **Orchestrator** | 纯编排（collect → gate → search → judge） | 不管 top_k、filter、fusion 等 |
| **SearchStrategy** | 搜索策略 + 数据库沟通唯一出口 | 不管 hit/miss 判定 |
| **Backend** | 纯 KNN 执行器 | 不做策略决策 |

**SearchStrategy 负责**：
- 持有所有搜索参数：top_k、step_filter、step_window、rrf_k、fusion_weights、candidate_multiplier
- 接收 query_keys + checkpoint_id → 构造完整 QuerySpec（含 filters + fusion 参数）
- 调用 CacheStorage.search()
- 未来可扩展：多轮搜索、re-ranking、自定义 fusion

**参数传递路径**：

```
SearchStrategy（持有搜索参数）
  → 构造 QuerySpec（含 top_k, filters, fusion 参数）
  → CacheStorage.search(spec)（校验、加锁、转发）
    → Backend.search(spec)（用 spec 中的参数执行搜索）
```

Backend config 退化为纯连接配置（url、collection_name 等），不再持有搜索策略参数。

**Protocol 接口**：

```python
class SearchStrategy(Protocol):
    def search(
        self,
        query_keys: dict[str, torch.Tensor],
        checkpoint_id: CheckpointID,
    ) -> list[SearchResultLite]:
        ...
```

**Orchestrator 调用链变更**：

```python
# 之前：Orchestrator 自己组装 QuerySpec
query_keys = key_builder.build(cp)
spec = QuerySpec(query_keys=query_keys, top_k=1, checkpoint_id=cp)
results = storage.search(spec)

# 之后：全权委托 SearchStrategy
query_keys = key_builder.build(cp)
results = search_strategy.search(query_keys, checkpoint_id=cp)
```

**storage 访问分工**：
- SearchStrategy 持有 CacheStorage 引用，负责搜索路径（search）
- Orchestrator 也持有 CacheStorage 引用，负责写入路径（insert/fetch_payload）
- 两者共享同一个 CacheStorage 实例

---

## 讨论 4：分检查点配置 + keys 统一配置

### 决议 1：CP1/CP3 独立配置

CP1 和 CP3 可以使用不同的 Gate、Judge、SearchStrategy。

Orchestrator 构造时用 dict 映射：

```python
CacheOrchestrator(
    storage=storage,
    key_builder=key_builder,
    gates={CP1: always_gate, CP3: always_gate},
    judges={CP1: threshold_judge_098, CP3: threshold_judge_095},
    search_strategies={CP1: knn_strategy, CP3: knn_strategy_window},
)
```

### 决议 2：keys 统一配置，分发到两层

```json
"keys": {
    "vision_0":    {"enabled": true,  "weight": 1.0},
    "robot_state": {"enabled": true,  "weight": 10.0},
    "prompt_emb":  {"enabled": false, "weight": 1.0}
}
```

config 加载时分发：
- `enabled=true` 的字段列表 → 传给 KeyBuilder（决定提取什么向量）
- 对应的 `weight` → 传给 SearchStrategy 的 fusion_weights

### 当前最终 Config 树

```
CacheConfig
├── enabled: bool
├── timer
│   ├── enabled: bool
│   ├── buffer_size: int
│   └── output_csv_dir: str | None
├── keys                                      # 统一配置，分发到 key_builder + search_strategy
│   ├── vision_0:    {enabled: bool, weight: float}
│   ├── vision_1:    {enabled: bool, weight: float}
│   ├── vision_2:    {enabled: bool, weight: float}
│   ├── prompt_emb:  {enabled: bool, weight: float}
│   └── robot_state: {enabled: bool, weight: float}
├── key_builder
│   └── type: "placeholder" | ...
├── checkpoints                                # 分检查点配置
│   ├── cp1
│   │   ├── gate: {type: "always_search"}
│   │   ├── judge: {type: "threshold", threshold: 0.98}
│   │   └── search_strategy:
│   │       ├── type: "simple_knn"
│   │       ├── top_k: int
│   │       ├── step_filter: "all" | "exact" | "window"
│   │       ├── step_window: int
│   │       ├── rrf_k: int
│   │       ├── candidate_multiplier: int
│   │       └── fusion_weights: dict[str, float] | None  # 或从 keys.weight 自动生成
│   └── cp3
│       ├── gate: {type: "always_search"}
│       ├── judge: {type: "threshold", threshold: 0.95}
│       └── search_strategy: (同上结构，可不同参数)
└── backend                                    # 退化为纯连接配置
    ├── type: "in_memory" | "qdrant"
    ├── vector_dims: dict[str, int]            # 校验用
    └── qdrant (仅 type=qdrant)
        ├── url: str
        ├── collection_name: str
        ├── prefer_grpc: bool
        ├── grpc_port: int
        └── request_timeout: int
```

---

## 设计原则：组件与 Config 的关系

**SearchStrategy（以及 Gate、Judge、KeyBuilder）是独立的可插拔组件，不是 config 系统的一部分。**

- 组件自己不读 config，不依赖 config 的数据结构
- 组件通过构造函数接收参数（普通 Python 值），不知道参数从哪来
- Config 系统是"组装工厂"：读配置文件 → 实例化组件 → 注入 Orchestrator
- 测试时可以直接构造组件，完全绕过 config 系统

```
Config 系统（工厂）          组件（独立）
──────────────             ──────────
读 YAML/dataclass          SearchStrategy(top_k=1, rrf_k=60, ...)
  → 实例化组件              Gate(...)
  → 注入 Orchestrator       Judge(threshold=0.98)
                            KeyBuilder(...)
```

这意味着：
- 组件的 Protocol 接口设计不受 config 格式影响
- 换 config 格式（YAML→TOML、CLI→文件）不改任何组件代码
- 组件可以在没有 config 系统的情况下独立使用和测试

---

## 讨论 5：Config 树覆盖盲区排查

### 5.1 顶层 Config 结构

CacheConfig 只是子配置。完整的 ServePolicyConfig 还需覆盖 serve_policy.py 的所有参数（env、port、default_prompt、record、collect、policy 等）。待后续讨论。

### 5.2 QuerySpec 扩展

SearchStrategy 是新增组件，它的引入必然带来已实现内容（QuerySpec、Backend.search 接口等）的修改。这部分修改在预期内，但详细计划在正式拟定 plan 时确定，不在 config 讨论阶段展开。

### 5.3 KeyBuilder 与 key 内容

**可用的原始数据**：

| 字段 | 维度 | 可用时机 |
|------|------|----------|
| vision_0/1/2 | 1024 each | Stage 1 后 |
| prompt_emb | 2048 | Stage 1 后 |
| robot_state | 32 | Stage 1 后 |
| action_chunk | [50, 32] | Stage 3 后 |

**决议**：
- KeyBuilder 应允许按检查点添加不同信息——CP1 时 action 为空（还没推理），CP3 时可加入 action
- 具体 key 构建方案需要重新设计，不再局限于 PlaceholderKeyBuilder 的纯 state 模式

**跨 cycle 历史记忆的维护**：
- 决议：各组件自己维护自己需要的历史状态（如 Gate 自己存上一步 state，Judge 自己存上一步 action）
- 理由：最灵活，不引入新的共享组件，不降低编码自由度
- 各组件通过 Protocol 接口暴露 clear() 等生命周期方法即可

### 5.4 InMemoryBackend 提升

InMemoryBackend 目前仅在 `tests/cache/conftest.py` 中。属于效率优化范畴，当前不处理，后续需要时再提升到正式代码。

### 5.5 vector_dims 一致性校验

三处声明需要一致：keys config（enabled 哪些字段）、KeyBuilder（build 输出维度）、Backend（vector_dims）。

**决议**：config 加载时提前校验，其他位置做好报错信息。
- 主校验：config 工厂实例化组件时交叉校验 keys/key_builder/backend 的字段和维度一致性，启动时就报错
- 兜底：CacheStorage 运行时维度校验保留，报错信息要清晰说明是哪个字段、期望多少维、实际多少维，方便定位是 config 配错还是 KeyBuilder 输出异常

---

## 讨论 6：顶层结构 + 配置格式

### 决议 1：顶层 ServePolicyConfig 分组嵌套

```
ServePolicyConfig
├── server: {env, port, default_prompt}
├── policy: {config, dir} | Default
├── debug: {record}
├── collect: {enabled, dir}
└── cache: CacheConfig
```

理由：cache 子系统配置复杂度高，扁平化后 CLI 会有几十个 flag，不可维护。

### 决议 2：YAML 文件格式

选用 YAML 作为配置文件格式。

理由：
- 支持注释，方便记录参数含义
- 深嵌套可读性好（缩进自然）
- YAML anchor（`&defaults` / `*defaults`）可在文件内共享默认值，CP1/CP3 共用参数时减少重复
- 环境变量替换（`${QDRANT_URL:-default}`）用 Python 侧几行代码实现，不需要额外依赖

变量能力：
- 文件内变量引用：用 YAML anchor 原生语法
- 环境变量：Python 侧加载时做 `${VAR:-default}` 替换
- 不引入 OmegaConf 等第三方依赖

---

### 决议 3：不支持 CLI override，一切以 YAML 为准

- 所有配置参数只从 YAML 文件读取，不支持命令行覆盖单个参数
- 命令行只用来选择 config 文件路径：`--config path/to/config.yaml`
- 有一个默认 config 文件，不指定 `--config` 时自动加载

```bash
# 使用默认配置
uv run scripts/serve_policy.py

# 指定配置文件
uv run scripts/serve_policy.py --config configs/experiment_rrf.yaml
```

理由：
- 配置来源唯一（只有 YAML），不存在"最终值从哪来"的歧义
- 实验可复现：一个 YAML 文件完整描述一次运行的所有参数
- 简化实现：不需要 YAML + CLI 合并逻辑

### 决议 4：默认文件与 CLI 简化

- 默认配置文件：项目根目录 `server.yaml`
- 不指定 `--config` 时自动加载 `server.yaml`
- `serve_policy.py` 的 CLI 参数简化为只保留 `--config`，其余全部迁入 YAML
- YAML 文件中每个参数都要有合理的 default 值和注释说明

```bash
# 使用默认 server.yaml
uv run scripts/serve_policy.py

# 指定其他配置
uv run scripts/serve_policy.py --config configs/experiment.yaml
```

serve_policy.py 的 Args dataclass 变为：

```python
@dataclasses.dataclass
class Args:
    config: str = "server.yaml"
```

---

### 决议 5：代码注释规范

所有新增代码必须遵循项目已有的注释规范（参考 Step 4 各组件），包括：

- **模块级 docstring**：Overview、Coupling map（DEPENDS ON / CONSUMED BY / IF CHANGED）、Data flow
- **类级 docstring**：职责说明、Data flow、Coupling 关系
- **方法级 docstring**：参数含义、返回值、副作用
- **行内注释**：关键决策点标注 why，不解释 what

示例（参考 `orchestrator.py` 风格）：
```
Coupling map:
  DEPENDS ON:  ...
  CONSUMED BY: ...
  IF CHANGED:  ...
```

---

## 讨论 7：待讨论

待定话题：
- 是否还有遗漏的可配置项
- 是否可以开始拟定正式 plan
