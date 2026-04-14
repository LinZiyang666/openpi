# Pi0.5 Inference Cache System - Architecture Specification (中文版)

> **⚠️ 本中文版未同步更新。** 最新架构规格请参阅英文版 [cache_system.md](cache_system.md)（2026-04-10 更新）。
> 以下内容冻结在 2026-04-03 版本，仅供历史参考。主要差异：
> - 英文版 §5 已更新为当前实现状态（KeyBuilder/SearchStrategy/WritePolicy/Trajectory）
> - 英文版 §10-13（Configuration/File Structure/Roadmap）已删除，改为指向 tutorial 和 logs
>
> 原始版本信息：
> Version: 0.3 (Step 1 已验证, Step 2 已验证, Step 3 ⚠️ 代码落地·高危)
> Status: 实施阶段 — Step 0-2 已验证，Step 3 ⚠️ 不稳定（无测试覆盖，接口将频繁变动），CP2 搁置
> Scope: PyTorch inference pipeline only (JAX path disabled)
> Last updated: 2026-04-03 (frozen)

---

## 1. System Goals

在 Pi0.5 的推理管线中引入多级缓存系统，通过复用历史计算结果来减少冗余推理，降低端到端延迟。系统设计遵循以下原则：

1. **与推理管线解耦**：Cache 系统作为外挂组件，通过 hook/interceptor 模式接入推理管线，不修改现有 inference 代码的内部逻辑。
2. **多级渐进式命中**：在推理管线的三个关键位置设置检查点，越早命中节省越多计算。
3. **存储层后端无关**：存储层通过 `VectorStoreBackend` ABC 将上层逻辑与具体向量数据库解耦，更换 backend（Qdrant、FAISS、TorchGPU 等）无需修改 orchestrator 或业务代码。*（搁置：GPU/CPU 混合数据分配与异步传输——见 Section 7）*
4. **精确计时**：每个组件（检索、判定、数据传输）独立计时，支撑后续性能优化决策。
5. **递进式实现**：从单机单任务重复场景起步，逐步扩展到多任务/多机器人/分布式。

---

## 2. 推理管线三阶段模型

基于 Pi0.5 的 PyTorch 推理路径（`src/openpi/models_pytorch/pi0_pytorch.py`），将 inference 划分为三个阶段：

```
Stage 1: Token Preparation        Stage 2: LLM Backbone         Stage 3: Action Expert
+---------------------------+     +------------------------+     +---------------------------+
| SigLIP vision encoder     |     | Gemma 2B (PaliGemma)   |     | Gemma 300M + adaRMSNorm   |
| Prompt tokenization       |     | Prefix-LM attention    |     | Flow matching (10 steps)  |
| State discretization      |     | Fill prefix KV cache   |     | Euler ODE: x1 -> x0      |
| -> prefix tokens + KV     |     | (no autoregressive gen)|     | -> action chunk [50, 32]  |
+---------------------------+     +------------------------+     +---------------------------+
            |                                |                                |
         [CP1]                            [CP2]                           [CP3]
      Cache Check 1                   Cache Check 2                   Cache Check 3
```

**各阶段计算量估算（单次推理）**：

| Stage | 主要计算 | 参数量 | 特点 |
|-------|---------|--------|------|
| 1. Token Prep | SigLIP forward + tokenize | ~400M | 单次前向，可并行 |
| 2. LLM Backbone | Gemma 2B prefix forward (KV 填充) | ~2B | 单次前向，填充 KV cache（PyTorch 路径无自回归生成） |
| 3. Action Expert | 10x Gemma 300M forward | ~300M x10 | 迭代式，可部分跳过 |

---

## 3. 三个检查点的语义定义

### CP1: Vision 之后

- **触发时机**：Stage 1 完成，prefix tokens 和 KV cache 已生成。
- **可用信息**：vision embedding, prompt embedding, state embedding。
- **命中行为**：跳过 Stage 2 + Stage 3，直接输出缓存的 action chunk。
- **节省量**：最大（跳过 LLM 解码 + 全部 flow matching）。
- **风险**：最高——跳过了 subtask 预测，如果场景发生了微妙变化（如物体被移走），缓存的 subtask 可能不再正确。
- **适用场景**：高度重复的操作（如流水线上的同一动作）。

### CP2: LLM Backbone 之后 — ⚠️ 暂时搁置

> **搁置原因**：架构设计时假设 Pi0.5 在 Stage 2 会进行自回归子任务文本生成（产出 command tokens + command embedding），使 CP2 具有 "same command → same action" 的语义基础。但 Step 1 代码分析发现，**PyTorch 实现中 Pi0.5 的 Stage 2 只做 prefix KV cache 填充，没有自回归文本生成**（JAX 路径已禁用）。Stage 2 完成后新增的信息只有 `past_key_values`（HuggingFace DynamicCache，opaque 对象），无法直接用作检索键。CP2 的语义前提不成立，暂时搁置，待未来有合适的 Stage 2 表征提取方案（如从 KV cache 最后一层 hidden state 提取 embedding）后再启用。

- **触发时机**：Stage 2 完成，~~low-level command（subtask text tokens）已生成~~ KV cache 已填充。
- **可用信息**：CP1 的全部信息 + `past_key_values`（opaque KV cache，不能直接用作检索键）。
- **命中行为（两种模式）**（*设计保留，暂不实现*）：
  - **Full hit**：跳过 Stage 3 全部，直接输出缓存的 action chunk。
  - **Partial hit (warm start)**：用缓存的中间状态 `x_t`（t < 1.0）作为 flow matching 起点，跳过部分去噪步骤。
- **节省量**：中等（跳过全部或部分 flow matching）。
- **风险**：中等——subtask 已由当前推理计算，缓存的 action 与当前场景的一致性更高。
- **适用场景**：相同 subtask 在相似场景下的复用。
- **当前状态**：**搁置** — 无可用检索键，需要新的表征提取方案。

### CP3: Action Expert 之后

- **触发时机**：Stage 3 完成，当前推理周期的 action chunk 已生成。
- **可用信息**：全部信息（vision + prompt + state + action chunk）。
- **命中行为**：不影响当前周期的输出。判定**下一个推理周期**是否可以跳过，直接执行 cache 中的后续 action chunk。
- **节省量**：最大（跳过完整的下一次推理）。
- **风险**：中等——依赖对未来状态的预测准确性。
- **适用场景**：连续动作序列具有时间局部性的场景（如长程物体搬运的中间阶段）。

### 检查点关系图

```
                    Inference Cycle N                          Cycle N+1
            ┌─────────┬─────────┬──────────┐          ┌──────────────────┐
            │ Stage 1 │ Stage 2 │ Stage 3  │          │ Stage 1,2,3      │
            │ Vision  │  LLM    │ FlowMatch│          │ (may be skipped) │
            └────┬────┴────┬────┴─────┬────┘          └────────┬─────────┘
                 │         │          │                         │
              [CP1]  [CP2:搁置]    [CP3]─── predict ──────> skip?
                 │                    │
          hit: skip              hit: schedule
          S2+S3                  next cycle's
                                 action from cache
```

---

## 4. 系统架构

### 4.1 顶层组件图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        CacheOrchestrator                                 │
│  (controls all cache workflow, decoupled from inference)                  │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │ CP1 Handler  │  │ CP2 Handler  │  │ CP3 Handler  │                   │
│  │              │  │              │  │              │   CheckpointHandler│
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                   │
│         │                 │                 │                             │
│  ┌──────┴─────────────────┴─────────────────┴───────┐                   │
│  │              QueryKeyBuilder (pluggable)          │                   │
│  │  Converts stage outputs -> fixed-dim query vector │                   │
│  └──────────────────────┬───────────────────────────┘                   │
│                         │                                                │
│  ┌──────────────────────┴───────────────────────────┐                   │
│  │              GateFunction (pluggable)             │                   │
│  │  Decides: should we even search cache?            │                   │
│  │  (heuristic / lightweight model / always-on)      │                   │
│  └──────────────────────┬───────────────────────────┘                   │
│                         │                                                │
│  ┌──────────────────────┴───────────────────────────┐                   │
│  │              SimilarityJudge (pluggable)          │                   │
│  │  Given search results, decide: hit or miss?       │                   │
│  │  (threshold / learned / composite)                │                   │
│  └──────────────────────────────────────────────────┘                   │
└──────────────────────────┬───────────────────────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          │    CacheStorage（门面）           │
          │  ┌───────────────┐ ┌──────────┐ │
          │  │VectorStore-   │ │MetadataDB│ │
          │  │Backend (ABC)  │ │（预留，   │ │
          │  │ ⚠️ 不稳定     │ │ 未实现） │ │
          │  │               │─│          │ │
          │  │┄Qdrant（当前）┄│ └──────────┘ │
          │  │┄FAISS（未来） │               │
          │  │┄TorchGPU(未来)│               │
          │  └───────────────┘               │
          └─────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                     InferencePipeline (existing, unmodified)              │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐               │
│  │  Stage 1    │───>│   Stage 2    │───>│   Stage 3     │               │
│  │  Vision     │    │   LLM        │    │   FlowMatch   │               │
│  └─────────────┘    └──────────────┘    └───────────────┘               │
└──────────────────────────────────────────────────────────────────────────┘
          ^                                         |
          |          Hook / Interceptor              |
          └─────── CacheOrchestrator ───────────────┘
```

### 4.2 与推理管线的接入方式

Cache 系统通过 **Interceptor 模式** 接入，不修改 `PI0Pytorch` 的内部代码。

> **Step 1 实现说明**：`InferenceInterceptor` 已在 Step 1（commit `a6c9f43`）中实现，作为 `BasePolicy` 子类。它通过 `self._model` 直接访问 `PI0Pytorch` 实例（从被包装的 `Policy` 借用引用），自行管理 transform 管线。它**不**调用 `self.policy.run_stage*()` —— 而是直接调用 `self._model.run_stage1/2/3()`。

```python
class InferenceInterceptor(BasePolicy):
    """包装 Policy，将推理路由到 staged API。

    实现 BasePolicy 接口，对 WebsocketPolicyServer 透明替换。
    通过 serve_policy.py 的 --cache 参数激活。
    """

    def __init__(self, policy: Policy):
        # 借用引用（非拷贝）
        self._model = policy._model          # PI0Pytorch 实例
        self._input_transform = policy._input_transform
        self._output_transform = policy._output_transform
        self._timer = SystemTimer()          # Step 2: CUDA event 计时

    def infer(self, observation: dict, *, noise=None) -> dict:
        """带缓存的推理。替代直接的 policy.infer() 调用。"""

        # --- Stage 1: Token Preparation ---
        with self._timer.measure("stage1_vision"):
            stage1_output = self._model.run_stage1(observation)

        # --- CP1: Cache Check (TODO: Step 4+) ---
        # cp1_result = self.orchestrator.check(CP1, stage1_output)
        # if cp1_result.hit: return cp1_result.cached_action

        # --- Stage 2: LLM Backbone ---
        with self._timer.measure("stage2_llm"):
            stage2_output = self._model.run_stage2(stage1_output)

        # --- CP2: 搁置 ---
        # CP2 搁置，因为 Stage 2 只产出 opaque 的 past_key_values（DynamicCache），
        # 没有 command embedding 可用作检索键。详见 Section 3。

        # --- Stage 3: Action Expert (full flow matching) ---
        with self._timer.measure("stage3_flow"):
            stage3_output = self._model.run_stage3(stage2_output, noise=noise)

        # --- CP3: Predictive Cache Check (TODO: Step 4+) ---
        # cp3_result = self.orchestrator.check(CP3, ...)
        # if cp3_result.hit:
        #     self.orchestrator.schedule_next_action(cp3_result.cached_next_action)

        return stage3_output.action_chunk
```

关键设计点：
- `PI0Pytorch.run_stage1/2/3` 是在已有私有方法 `_stage1_token_prep`、`_stage2_llm_backbone`、`_stage3_action_expert` 之上添加的**公共类型化包装**。原有的 `sample_actions()` 未做任何修改。
- `InferenceInterceptor` 实现 `BasePolicy` 接口，对 WebsocketPolicyServer 和客户端透明替换，无需任何改动。
- `return_intermediates=True` 让 Stage 3 返回 flow matching 过程中选定时间步的 `x_t`，用于未来的 warm start cache。
- CP2 检查被有意省略（搁置）—— 原因见 Section 3。

---

## 5. 核心组件详细设计

### 5.1 CacheOrchestrator

> **状态**：已实现（Step 4）——不稳定，接口可能在 Step 5/6 变更。
> ⚠️ 使用 Step 3 存储层类型（`QuerySpec`、`SearchResultLite`、`CacheEntry` 等），这些接口不稳定。`CacheContext` 未采用，改为组件级 Protocol；`CacheResult` 改为 `CheckResult`。

总控组件。管理所有检查点的生命周期、协调 gate/search/judge 流程、处理异步写回。

> **注**：CP2 搁置后，orchestrator 当前只处理 CP1 和 CP3。`config.cp2_enabled` 默认为 `False`。

```python
class CacheOrchestrator:
    def __init__(
        self,
        storage: CacheStorage,
        key_builder: QueryKeyBuilder,
        gate: GateFunction,
        judge: SimilarityJudge,
        config: CacheConfig,
        timer: SystemTimer,
    ):
        ...
        self._next_action_scheduled: Optional[torch.Tensor] = None  # [50, 32]
        # 写入队列实现 TBD in Step 4

    def should_skip_inference(self) -> Optional[torch.Tensor]:
        """在推理开始前调用。如果上一周期 CP3 预调度了 cached action，直接返回并跳过整次推理。"""
        if self._next_action_scheduled is not None:
            action = self._next_action_scheduled
            self._next_action_scheduled = None
            return action
        return None

    def check(self, checkpoint: CheckpointID, context: CacheContext) -> CacheResult:
        """给定 checkpoint 的核心 cache 检查逻辑。

        CacheContext, CacheResult：Step 4 编排层类型（尚未定义）。
        与存储层的交互使用 Step 3 类型（⚠️ 不稳定）。
        """

        # Step 1: Gate — 是否需要搜索？
        with self.timer.measure(f"{checkpoint.name}_gate"):
            if not self.gate.should_search(checkpoint, context):
                return CacheResult.miss()

        # Step 2: 构建多命名查询向量
        with self.timer.measure(f"{checkpoint.name}_key_build"):
            keys = self.key_builder.build(checkpoint, context)  # dict[str, Tensor]

        # Step 3: 搜索向量 DB — ⚠️ 使用 Step 3 存储层类型
        with self.timer.measure(f"{checkpoint.name}_search"):
            spec = QuerySpec(
                query_keys=keys,
                top_k=self.config.top_k,
                checkpoint_id=checkpoint,
            )
            candidates: list[SearchResultLite] = self.storage.search(spec)

        # Step 4: Judge — 最佳候选是否足够好？
        with self.timer.measure(f"{checkpoint.name}_judge"):
            result = self.judge.evaluate(checkpoint, context, candidates)

        return result

    def write_async(self, entry: CacheEntry):
        """非阻塞 cache 写入。在后台线程执行。
        参数为 CacheEntry（⚠️ Step 3 类型，不稳定）。"""
        ...

    def schedule_next_action(self, action: torch.Tensor):
        """CP3 预调度下一周期的 action chunk。"""
        self._next_action_scheduled = action
```

### 5.2 CacheStorage — ⚠️ Step 3 已实现（不稳定）

> **实现**：`src/openpi/cache/cache_storage.py` | **设计日志**：`logs/archive/step3_cache.log`
> **状态**：⚠️ 代码已落地，无测试覆盖，接口将频繁变动。

存储层门面。`CacheOrchestrator` 是唯一消费方。内含 `VectorStoreBackend`（ABC）和可选的 `MetadataDB`（预留，尚未实现）。

**为什么做 ABC 抽象？** 我们尚不确定最终使用哪个向量数据库。当前用 Qdrant 做实验，但**一定会更换**。`VectorStoreBackend` ABC 将上层逻辑与具体 DB 解耦，换 backend 时 `CacheOrchestrator` 和其他业务代码无需任何修改。

```python
class CacheStorage:
    """存储层门面。CacheOrchestrator 唯一的存储入口。

    职责：
    - 线程安全（RLock 保护所有 backend 调用）
    - 维度校验（insert/search 时检查 query_key.shape）
    - Filter 能力检查（fail-fast，通过 supported_filters() 判断，不静默忽略）
    - CacheEntry 校验（调用 entry.validate()）
    - 两段式搜索（search 返回 SearchResultLite，fetch_payload 按需取完整 payload）
    - MetadataDB 预留（vector 先写，metadata 后写，vector 为 source of truth）
    """

    def __init__(self, backend: VectorStoreBackend, metadata_db=None):
        self._backend = backend
        self._metadata_db = metadata_db   # 预留，暂未使用
        self._dims = backend.vector_dims   # dict[str, int] — 各字段维度
        self._lock = threading.RLock()

    def search(self, spec: QuerySpec) -> list[SearchResultLite]:
        """向量搜索，返回轻量结果（无 payload）。"""
        ...

    def fetch_payload(self, id: str) -> CachePayload:
        """按 id 取完整 payload，仅对命中候选调用。"""
        ...

    def search_and_fetch(self, spec: QuerySpec) -> list[SearchResult]:
        """便利方法：搜索后对所有结果逐一 fetch payload。"""
        ...

    def insert(self, entry: CacheEntry) -> None:
        """校验 entry，然后委托 backend.insert()。"""
        ...

    def batch_insert(self, entries: list[CacheEntry]) -> BatchInsertResult: ...
    def delete(self, ids: list[str]) -> None: ...
    def count(self) -> int: ...
    def close(self) -> None: ...
```

关键设计点：
- **两段式搜索**：`search()` 返回 `SearchResultLite`（仅含 score，无 payload tensor）。只有命中候选调用 `fetch_payload()` 获取完整 `CachePayload`，避免无用数据传输。
- **Filter fail-fast**：`CacheStorage` 在调用 `search()` 前检查 `spec.filters` 是否被 `backend.supported_filters()` 支持。不支持的 filter 抛出 `UnsupportedFilterError`，而非静默忽略。
- **维度校验**：每次 `insert()` 和 `search()` 调用均逐字段校验 `query_keys` 的各维度是否与 `backend.vector_dims` 一致。

### 5.3 VectorStoreBackend — ⚠️ Step 3 已实现（不稳定）

> **实现**：`src/openpi/cache/backend_base.py` + `src/openpi/cache/backends/qdrant_backend.py`
> **设计日志**：`logs/archive/step3_cache.log`
> **状态**：⚠️ 代码已落地，无测试覆盖，接口将频繁变动。

**向量数据库选型未定。** 当前用 Qdrant 做实验，但未来**一定会更换**（候选：FAISS、自研 TorchGPU store 或其他）。为将上层逻辑与选型决策隔离，Step 3 引入 `VectorStoreBackend` ABC 作为最小公约数接口。

```python
class VectorStoreBackend(ABC):
    """向量存储后端的最小公约数接口。

    契约：
    - insert() 幂等：相同 id 重复写入以最新为准，不报错
    - search() 返回 SearchResultLite（无 payload），score 为归一化
      cosine similarity ∈ [-1, 1]，所有 backend 必须统一转换
    - search() 返回至多 top_k 条；client-side filter 后可能少于 k
    - fetch_payload() 按 id 取完整 CachePayload
    - delete() 容忍不存在的 id
    - 实现不保证线程安全，CacheStorage 负责加锁
    """

    @property
    @abstractmethod
    def vector_dims(self) -> dict[str, int]:
        """字段名 → 向量维度。键为 CACHE_QUERY_FIELDS 子集。"""
        ...

    @abstractmethod
    def supported_filters(self) -> frozenset[str]: ...

    @abstractmethod
    def insert(self, entry: CacheEntry) -> None: ...

    @abstractmethod
    def search(self, spec: QuerySpec) -> list[SearchResultLite]: ...

    @abstractmethod
    def fetch_payload(self, id: str) -> CachePayload: ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None: ...

    @abstractmethod
    def count(self) -> int: ...

    # --- 可选（有默认实现）---
    def batch_insert(self, entries: list[CacheEntry]) -> BatchInsertResult: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
```

**当前 backend — QdrantVectorStore**（`src/openpi/cache/backends/qdrant_backend.py`）：
- 通过 HTTP/gRPC 连接远程 Qdrant 服务器
- Collection 由运维人员预先创建，backend 不做自动创建
- 序列化：tensor → `torch.save` bytes → base64 字符串 → Qdrant JSON payload
- 支持的 filter：`checkpoint_id`、`task_key`、`step_range`
- Score：Qdrant Cosine distance 已在 [-1, 1] 范围内，直接使用

**为什么 ABC 接口故意很小**：payload filter 语法、gRPC 选项、融合策略（RRF 权重等）——这些全是 backend 特有细节，配在各 backend 的 `__init__` config 里，永远不暴露到 ABC 边界之上。上层代码只问一件事："给我一组命名查询向量（`dict[str, Tensor]`），返回最相似的 top-k 条目"。多字段如何融合（如 RRF、加权平均）是 backend 内部决策。

#### 5.3.1 搁置设计：GPU/CPU 混合 VectorStore

> **状态**：搁置——未实现。以下设计保留用于未来需要本地高性能向量存储时开发（如 FAISS GPU index + CPU 回退）。当前所有存储经由远程 Qdrant backend。

```python
class VectorStore:
    """Hybrid GPU/CPU vector store for cache lookup.

    Design rationale:
    - Hot data (frequently accessed, recent) lives on GPU for fast search.
    - Cold data lives on CPU, searched when GPU partition misses.
    - Transfers between CPU/GPU use pinned memory + CUDA streams
      to avoid blocking the main inference CUDA stream.
    """

    def __init__(self, config: VectorStoreConfig):
        self.dim = config.embedding_dim
        self.gpu_capacity = config.gpu_capacity
        self.cpu_capacity = config.cpu_capacity
        self.gpu_vectors: torch.Tensor  # [gpu_capacity, dim] on cuda
        self.cpu_index: faiss.Index     # CPU-resident index
        self._transfer_stream = torch.cuda.Stream()

    def search(self, query, top_k):
        """Search GPU first, then CPU if needed. Both can run concurrently."""
        ...

    def promote_to_gpu(self, cpu_ids): ...
    def demote_to_cpu(self, gpu_ids): ...
```

**GPU 显存预算管理**：VectorStore 通过 `gpu_capacity` 硬上限控制显存占用，实际占用 = `gpu_capacity * dim * sizeof(float16)` 字节。例如 10k entries x 1024 dim x 2 bytes = **20MB**，对 inference 的显存影响可忽略。

### 5.4 QueryKeyBuilder（可插拔）

> **状态**：已实现（Step 4）——不稳定。两种实现：`PlaceholderKeyBuilder`（仅 L2 归一化 state）和 `FullOriginalKeyBuilder`（多模态分割 + flatten，无 pooling/归一化）。
> ⚠️ 返回类型 `dict[str, torch.Tensor]` 与 `QuerySpec.query_keys` / `CacheEntry.query_keys`（Step 3 存储层类型，不稳定）对齐。`CacheContext` 未采用；Gate/Judge 直接从 KeyBuilder 读取 `cached_data`。

```python
class QueryKeyBuilder(Protocol):
    """将 stage 输出转换为命名查询向量。

    返回 dict，键为 CACHE_QUERY_FIELDS 子集，值为 L2 归一化 tensor。
    Backend 只存储/查询其 vector_dims 声明的字段；多余字段静默忽略。
    """

    def build(self, checkpoint: CheckpointID, context: CacheContext) -> dict[str, torch.Tensor]:
        """返回命名查询向量 {field: [dim] tensor, L2 归一化}。"""
        ...


class MeanPoolKeyBuilder(QueryKeyBuilder):
    """Baseline：对各 embedding 源投影到固定维度，返回命名向量。

    注意：'command' projection 已移除 — Stage 2 只产出 opaque 的
    KV cache，无可提取的 command embedding（CP2 搁置，详见 Section 3）。
    """

    def __init__(self, output_dim: int = 1024):
        self.projections = nn.ModuleDict({
            "vision_0": nn.Linear(..., output_dim),
            "prompt_emb": nn.Linear(..., output_dim),
            "robot_state": nn.Linear(..., output_dim),
            # "command": 已移除 — 无 command embedding（CP2 搁置）
        })

    def build(self, checkpoint, context):
        keys = {}
        if context.stage1:
            keys["vision_0"] = F.normalize(
                self.projections["vision_0"](context.stage1.vision_emb.mean(dim=1)), dim=-1
            )
            keys["robot_state"] = F.normalize(
                self.projections["robot_state"](context.stage1.state_emb), dim=-1
            )
            keys["prompt_emb"] = F.normalize(
                self.projections["prompt_emb"](context.stage1.prompt_emb), dim=-1
            )
        return keys


class PlaceholderKeyBuilder(QueryKeyBuilder):
    """早期开发用：仅使用 raw state 向量作为唯一查询字段。"""

    def build(self, checkpoint, context):
        return {"robot_state": F.normalize(context.stage1.raw_state.float(), dim=-1)}
```

设计为 Protocol，后续可替换为学习型 encoder 或其他方案，不影响系统其他部分。

### 5.5 GateFunction（可插拔）

> **状态**：已实现（Step 4）——不稳定。当前实现：`AlwaysSearchGate`（始终返回 True）。
> `CacheContext` 未采用；Gate 直接接收 `checkpoint_id` 和 `cached_data` dict。接口可能随 state-change gate 的引入而演进。

决定是否在某个 checkpoint 启动检索。避免每次都搜索的开销。

```python
class GateFunction(Protocol):
    def should_search(self, checkpoint: CheckpointID, context: CacheContext) -> bool:
        ...

class AlwaysSearchGate(GateFunction):
    """Baseline: always search. For benchmarking overhead."""
    def should_search(self, checkpoint, context):
        return True

class IntervalGate(GateFunction):
    """Only search every N inference cycles."""
    def __init__(self, interval: int = 3):
        self.interval = interval
        self.counter = 0

    def should_search(self, checkpoint, context):
        self.counter += 1
        return self.counter % self.interval == 0

class StateChangeGate(GateFunction):
    """Search only when state change exceeds threshold.
    If robot barely moved since last check, cache result likely same -> skip search."""
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self.last_state: Optional[torch.Tensor] = None

    def should_search(self, checkpoint, context):
        current = context.stage1.raw_state
        if self.last_state is None:
            self.last_state = current
            return True
        delta = (current - self.last_state).norm().item()
        if delta > self.threshold:
            self.last_state = current
            return True
        return False
```

### 5.6 SimilarityJudge（可插拔）

> **状态**：已实现（Step 4）——不稳定。当前实现：`ThresholdJudge`，按 checkpoint 设阈值（cp1=0.98, cp3=0.95，未校准）。
> ⚠️ 使用 Step 3 存储层的 `SearchResultLite`（不稳定）。`CacheContext` 未采用；Judge 接收 results + `checkpoint_id` + `cached_data`。阈值尚未在真实数据上校准。

判定检索结果是否构成有效 hit。

```python
class SimilarityJudge(Protocol):
    def evaluate(
        self, checkpoint: CheckpointID, context: CacheContext,
        candidates: list[SearchResultLite],  # ⚠️ Step 3 类型
    ) -> CacheResult:
        ...

class ThresholdJudge(SimilarityJudge):
    """Simple threshold-based judge with per-checkpoint thresholds."""

    def __init__(self, storage: CacheStorage, thresholds: dict[CheckpointID, float]):
        self.storage = storage  # 命中时需要 fetch_payload
        self.thresholds = thresholds
        # CP1 阈值更严格（要求更高相似度），因为跳过更多计算风险更大。
        # 默认：CP1=0.98, CP3=0.90（CP2 搁置）

    def evaluate(self, checkpoint, context, candidates):
        if not candidates:
            return CacheResult.miss()

        best: SearchResultLite = candidates[0]
        threshold = self.thresholds[checkpoint]

        if best.score >= threshold:
            return self._make_hit(checkpoint, best)
        return CacheResult.miss()

    def _make_hit(self, checkpoint, candidate: SearchResultLite):
        # 两段式：仅对命中候选 fetch 完整 payload
        payload: CachePayload = self.storage.fetch_payload(candidate.id)

        # CP2 warm start 分支 — 搁置（无 command embedding 可用）。
        # 设计保留，待 CP2 重新启用时使用。
        # if checkpoint == CheckpointID.CP2 and payload.intermediates:
        #     if candidate.score >= self.thresholds[CheckpointID.CP2_FULL]:
        #         return CacheResult.full_hit(payload.action_chunk)
        #     else:
        #         t = max(payload.intermediates.keys())
        #         return CacheResult.warm_start(
        #             cached_noisy_action=payload.intermediates[t],
        #             cached_timestep=t,
        #             num_steps=payload.denoising_num_steps,
        #         )
        return CacheResult.full_hit(payload.action_chunk)
```

### 5.7 Cache 数据模型 — ⚠️ Step 3 已实现（不稳定）

> **实现**：`src/openpi/cache/storage_types.py` + `src/openpi/cache/types.py`
> **设计日志**：`logs/archive/step3_cache.log`
> **状态**：⚠️ 代码已落地，无测试覆盖，接口将频繁变动。

Step 3 用更丰富的数据模型替换了原来的扁平 `CacheEntry`。与原设计的主要差异：`CachePayload` 是独立的嵌套 dataclass（非 `CacheEntry` 上的扁平字段），新增 `QueryFilter` 用于结构化过滤，搜索结果分为 `SearchResultLite`（轻量，用于阈值筛选）和 `SearchResult`（完整，含 payload）。

```python
# src/openpi/cache/types.py
class CheckpointID(Enum):
    CP1 = auto()
    CP2 = auto()
    CP3 = auto()

# src/openpi/cache/storage_types.py

@dataclass
class CachePayload:
    """缓存条目的有效载荷。强类型，CP1/CP2/CP3 共用。

    Tensor 契约：所有 tensor 必须是 CPU contiguous float32。
    调用方在构造前负责转换；search() 返回同样保证 CPU float32。

    CP1：action_chunk 必填，其余为 None。
    CP2 warm start：action_chunk + intermediates + denoising_num_steps 必填。
    CP3：action_chunk + next_action_chunk 必填。
    """
    action_chunk: torch.Tensor                          # [50, 32]
    intermediates: Optional[dict[float, torch.Tensor]]  # {t: x_t}
    denoising_num_steps: Optional[int]                  # warm start 时必填
    next_action_chunk: Optional[torch.Tensor]           # [50, 32]，CP3 必填
    task_key: str = ""                                  # 规范化 task 标识符

    def validate_for_checkpoint(self, checkpoint_id: CheckpointID) -> None: ...


@dataclass
class CacheEntry:
    """写入存储的完整单元。

    id 语义：stable_hash(checkpoint_id.name + ":" + sorted_concat_of_query_key_bytes)
    同一 context + 同一 CP 只保留一条记录（语义去重键）。
    checkpoint_id 参与 hash，确保同一 observation 在 CP1/CP2/CP3 不互相覆盖。
    """
    id: str
    checkpoint_id: CheckpointID
    query_keys: dict[str, torch.Tensor]  # {field: [dim] CPU float32, L2 归一化}
    payload: CachePayload
    timestamp: float = field(default_factory=time.time)

    def validate(self) -> None: ...


@dataclass
class QueryFilter:
    """每次查询不同的约束条件。不支持某个字段的 backend 必须在
    supported_filters() 中不列出它，CacheStorage 会 fail-fast。"""
    task_key: Optional[str] = None
    step_range: Optional[tuple[int, int]] = None


@dataclass
class QuerySpec:
    query_keys: dict[str, torch.Tensor]  # {field: [dim] CPU float32}
    top_k: int = 10
    checkpoint_id: Optional[CheckpointID] = None
    filters: Optional[QueryFilter] = None


@dataclass
class SearchResultLite:
    """轻量搜索结果（无 payload）。search() 返回此类型。
    score：越高越相似。范围取决于 backend/mode：
      单字段 cosine：∈ [-1, 1]；多字段 RRF 融合：小正数。
    阈值需按 backend/mode 校准。"""
    id: str
    score: float
    checkpoint_id: CheckpointID


@dataclass
class SearchResult:
    """完整搜索结果（含 payload）。由 fetch_payload() 填充。"""
    id: str
    score: float
    payload: CachePayload
    checkpoint_id: CheckpointID
```

**Warm start 的中间状态选择**（设计保留，尚未填充）：不缓存所有 10 步的中间状态，只缓存 2-3 个关键时间点（如 t=0.7, 0.5, 0.3）。`denoising_num_steps` 字段告知 `run_stage3_from()` 从哪里恢复。此机制依赖 CP2 warm start，目前搁置。

---

## 6. 数据流与时序

> **注意**：以下数据流图中涉及的 cache search/write 操作依赖存储层（Section 5.2/5.3）。存储层当前 ⚠️ 不稳定——接口和 backend 实现将频繁变动。时序结构（stage 划分、checkpoint 位置）是稳定的，存储交互细节不是。

### 6.1 完整推理周期（无 cache hit）

> **注意**：下图中 `[GPU Transfer Stream]` 和 `[CPU Thread Pool]` 为 GPU/CPU 混合 VectorStore 的**目标设计**（搁置——见 Section 7）。当前使用远程 Qdrant backend，cache search 在主线程同步执行，无 GPU 向量分区和专用 transfer stream。

```
Time ──────────────────────────────────────────────────────────────>

[GPU Main Stream]
│ Stage1 ││ Stage2 ││ Stage3 (10 denoise steps)          ││
│ Vision ││ LLM    ││ step1 step2 ... step10             ││
│        ││        ││                                     ││

[GPU Transfer Stream]（搁置 — GPU/CPU 混合方案，见 Section 7）
         ││  CP1   ││       CP2        ││            CP3  ││  write-back
         ││ search ││      search      ││           check ││  (async)

[CPU Thread Pool]（搁置 — GPU/CPU 混合方案，见 Section 7）
         ││ CP1 CPU││   CP2 CPU search ││ CP3 CPU search  ││ metadata write
         ││ search ││  (if GPU miss)   ││                 ││
```

### 6.2 CP1 Hit 的时序

> **注意**：`[GPU Transfer Stream]` 为搁置设计（见 Section 7）。当前实际：CP1 search 为远程 Qdrant 调用，cached action 通过网络获取而非 GPU 内存。

```
Time ──────────────────────────>

[GPU Main Stream]
│ Stage1 ││ (idle - stages 2,3 skipped)
│ Vision ││

[GPU Transfer Stream]（搁置 — 当前为远程 Qdrant fetch）
         ││ CP1 search ──> HIT!
         ││ load cached action

Total: Stage1 + CP1 latency only
```

### 6.3 CP2 Warm Start 的时序 — ⚠️ 搁置（设计保留）

```
Time ──────────────────────────────────────────────>

[GPU Main Stream]
│ Stage1 ││ Stage2 ││ Partial Stage3 (3 steps)    ││
│ Vision ││ LLM    ││ from cached x_0.3           ││

[GPU Transfer Stream]
         ││ CP1    ││ CP2 search ──> WARM START HIT
         ││ miss   ││ load cached x_0.3

Total: Stage1 + Stage2 + CP2 latency + 3 denoise steps (instead of 10)
```

### 6.4 CP3 Predictive Hit 的时序

```
Cycle N:                                             Cycle N+1:
│ Full inference │ CP3: match found ──> schedule │    │ Skip inference, use cached action │
                                                      │ (only run Stage1 for state update) │
```

---

## 7. 硬件资源分配策略 — 搁置

> **状态**：搁置——未实现。以下设计保留用于未来本地 GPU/CPU 混合向量存储替代当前远程 Qdrant backend 时开发。当前所有缓存存储为远程调用，无 GPU 向量分区、无 pinned memory pool、无 cache 专用 CUDA stream。

### 7.1 GPU 显存布局

```
GPU VRAM (e.g., 24GB)
├── Model weights (fixed)          ~5 GB  (PaliGemma 2B + Action Expert 300M, bf16)
├── KV Cache (per inference)       ~1 GB  (varies with sequence length)
├── Activations (transient)        ~2 GB  (peak during forward pass)
├── VectorStore GPU partition      ~20 MB (10k entries x 1024 dim x fp16)
├── Transfer buffers (pinned)      ~10 MB
└── Free                           ~16 GB
```

VectorStore 的 GPU 占用极小，不会成为瓶颈。

### 7.2 CUDA Stream 隔离

```python
class CacheHardwareManager:
    """Manages CUDA resources for cache operations."""

    def __init__(self):
        # Separate stream for cache operations, does NOT block inference
        self.cache_stream = torch.cuda.Stream(priority=-1)  # low priority
        # Pinned memory pool for CPU<->GPU transfers
        self.pinned_pool = PinnedMemoryPool(size_mb=32)

    @contextmanager
    def cache_context(self):
        """Execute cache operations on dedicated stream."""
        with torch.cuda.stream(self.cache_stream):
            yield

    def async_to_gpu(self, tensor_cpu: torch.Tensor) -> torch.Tensor:
        """Non-blocking CPU->GPU transfer via pinned memory."""
        pinned = self.pinned_pool.allocate(tensor_cpu.shape, tensor_cpu.dtype)
        pinned.copy_(tensor_cpu)
        gpu_tensor = torch.empty_like(pinned, device="cuda")
        with torch.cuda.stream(self.cache_stream):
            gpu_tensor.copy_(pinned, non_blocking=True)
        return gpu_tensor
```

### 7.3 CPU 线程分配

```
Thread 0 (main):       Inference orchestration
Thread 1:              CPU-side vector search (FAISS)
Thread 2:              Cache write-back (vector DB insert + metadata)
Thread 3 (optional):   Cache maintenance (eviction, compaction)
```

---

## 8. Cache 管理与动态优化 — 搁置

> **状态**：搁置——未实现。写入策略、淘汰策略、GPU/CPU 数据迁移为未来 GPU/CPU 混合存储（Section 5.3.1 / Section 7）设计。当前远程 Qdrant backend 的淘汰由 Qdrant 服务端或运维手动处理。以下设计保留用于未来开发。

### 8.1 写入策略

- **在线写入**：每次正常推理完成后，异步写入 cache。不阻塞下一次推理。
- **离线预填充**：从训练数据或离线 rollout 中批量导入。
- **选择性写入**：不是所有推理结果都值得缓存。通过 `WriteFilter` 判断：
  - 如果当前状态与已有 cache 条目过于相似（< 某阈值），不写入（避免冗余）。
  - 如果动作置信度低（flow matching 收敛不好），不写入。

### 8.2 淘汰策略

```python
class EvictionPolicy(Protocol):
    def select_evictions(self, store: VectorStore, count: int) -> list[int]:
        ...

class CompositeEviction(EvictionPolicy):
    """Combine multiple signals for eviction."""

    def select_evictions(self, store, count):
        scores = []
        for entry in store.entries():
            score = (
                0.4 * recency_score(entry.timestamp)    # LRU component
                + 0.3 * frequency_score(entry.hit_count) # LFU component
                + 0.3 * entry.quality_score               # quality component
            )
            scores.append((entry.id, score))
        scores.sort(key=lambda x: x[1])
        return [id for id, _ in scores[:count]]
```

### 8.3 GPU/CPU 数据迁移策略

- **Promotion (CPU -> GPU)**：当某条 CPU 条目被命中超过 N 次，提升到 GPU partition。
- **Demotion (GPU -> CPU)**：当 GPU partition 满且有新的高频条目需要入驻时，将最低频的 GPU 条目降级到 CPU。
- **迁移在 `cache_stream` 上异步进行，不阻塞推理。**

---

## 9. 计时系统 — ✅ 已实现（Step 2）

> 实现文件：`src/openpi/cache/timing.py` | 设计日志：`logs/archive/step2.log`

计时系统采用 **基于 probe 的架构**：每个流水线组件在启动时注册一个命名 probe 并指定后端类型，`SystemTimer` 在热路径上提供零开销的 `measure()` 上下文管理器。

### 9.1 Backend 协议

```python
class TimingBackend(Protocol):
    def start(self) -> Any: ...          # 返回不透明的计时句柄
    def stop(self, handle: Any) -> float: ...  # 返回耗时 ms

class CudaEventBackend:
    """GPU 计时，使用 torch.cuda.Event。
    end_event.synchronize() 仅等待该 event 完成，不阻断整个 CUDA 流水线。
    CUDA 不可用时自动降级为 PerfCounterBackend。"""
    def __init__(self, stream: torch.cuda.Stream | None = None): ...

class PerfCounterBackend:
    """CPU 计时，使用 time.perf_counter_ns，亚微秒精度。"""
```

关键设计决策：
- **CudaEventBackend handle 设计**：返回 `("cuda", start_evt, end_evt)` 元组，含类型标记——支持并发/嵌套调用，不在实例上保存 per-call 状态。
- **自动降级**：`CudaEventBackend` 内部持有 `PerfCounterBackend`；无 GPU 时返回 `("cpu", ns)` handle，CI 环境也能跑。

### 9.2 SystemTimer

```python
class SystemTimer:
    def __init__(self, enabled=True, buffer_size=10000, output_csv_dir=None): ...
    def register_probe(self, name: str, backend: str = "cuda", stream=None): ...

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        # enabled=False → yield + return（零开销）
        # 未注册的 probe → 默认 CudaEventBackend（宽松模式，带警告）

    # TaskLifecycle（见 9.3）
    def on_task_begin(self) -> None: ...   # 记录当前 ring buffer 位置
    def on_task_end(self) -> None: ...     # 打印摘要 + 可选写 CSV + task_id++

    def summary(self, task_only=True) -> dict[str, TimingStats]: ...
    def export_csv(self, path: str): ...   # 内存构建全部行，一次性写盘

    # 预留接口（Step 2 为 stub）
    def add_resource_monitor(self, monitor: ResourceMonitor): ...
    def record_resource_snapshot(self, name: str): ...
```

环形缓冲区使用**单调追加计数器**（`_total_appended`）追踪任务边界。`on_task_begin()` 保存该值；`on_task_end()` 从 deque 尾部切片。Ring buffer wrap 时给出警告。

**total (sum) 行**的正确计算方式：将三个 stage 的 record 按顺序对齐、逐次相加得到每次推理总耗时列表，再求 p50/p95/p99（而非简单累加均值）。

`_DISPLAY_ORDER` 常量控制摘要表行顺序；后续新增 probe 只需加入该列表。

### 9.3 TaskLifecycle 协议

```python
@runtime_checkable
class TaskLifecycle(Protocol):
    def on_task_begin(self) -> None: ...
    def on_task_end(self) -> None: ...
```

`InferenceInterceptor` 实现该协议，转发给内部 `SystemTimer`。Server 侧用 `hasattr(policy, 'on_task_begin')` 判断（避免强依赖）。

**Server 侧集成**（`websocket_policy_server.py`）：
- 连接建立 → `policy.on_task_begin()`
- 连接关闭（正常或异常）→ `policy.on_task_end()` → 打印摘要 / 写 CSV
- 已移除：旧的 `stage_timing_records` 收集、`action.pop("stage_timing")`、手工均值 print（约 -15 行）

### 9.4 已注册 Probes

| Probe | 后端 | 状态 | 说明 |
|-------|------|------|------|
| `stage1_vision` | cuda | ✅ 已注册 | Vision encoder + tokenization |
| `stage2_llm` | cuda | ✅ 已注册 | LLM backbone prefix KV 填充 |
| `stage3_flow` | cuda | ✅ 已注册 | 完整 flow matching（10 次 denoise step） |
| `total_inference` | cpu | ✅ 已注册 | Wall-clock 总耗时（外层 `measure()` 包裹三个 stage） |
| `cp1_*`, `cp3_*` | cpu | ✅ 已注册（Step 4）——不稳定 | Cache 子步骤 probe（gate, build, search, judge, write） |
| `cp2_*` | — | 搁置 | CP2 搁置（见 Section 3） |
| `write_vectordb`, `write_metadata` | cpu | 计划中 | 异步写回 |
| `gpu_to_cpu`, `cpu_to_gpu` | cuda | 计划中 | `transfer_stream` 上的数据迁移 |

### 9.5 摘要输出示例

```
=== Inference Timing Summary (task #0, 4 inferences) ===
  Probe                          N     mean      p50      p95      p99  ms
  ------------------------------------------------------------------------
  stage1_vision                  4      2.1      2.1      2.3      2.3
  stage2_llm                     4      6.1      6.1      6.1      6.1
  stage3_flow                    4      3.1      3.1      3.1      3.1
  total_inference                4     11.3     11.3     11.5     11.5
  ------------------------------------------------------------------------
  total (sum)                    4     11.3     11.2     11.4     11.5
```

---

## 10. 配置系统

> **注意**：`CacheConfig` 尚未作为独立文件实现。以下配置项反映完整设计意图。标注 **搁置** 的项依赖 GPU/CPU 混合存储（Section 5.3.1 / Section 7），不适用于当前 Qdrant backend。标注 **⚠️** 的项与 Step 3 存储层相关，将随接口演化而变。

```python
@dataclass
class CacheConfig:
    """Top-level cache system configuration."""

    # ── 通用配置（稳定）─────────────────────────────────────────
    enabled: bool = True

    # Per-checkpoint enable/disable
    cp1_enabled: bool = True
    cp2_enabled: bool = False              # 搁置 — 无 command embedding 可用（见 Section 3）
    cp3_enabled: bool = True

    # Retrieval
    top_k: int = 5                         # candidates per search

    # Similarity thresholds（越高越严格）
    # 注意：阈值尺度取决于 backend/mode — 见 SearchResultLite.score 文档
    cp1_threshold: float = 0.98            # CP1 最严格：跳过最多计算
    cp2_full_threshold: float = 0.96       # CP2 full hit（搁置）
    cp2_warm_threshold: float = 0.90       # CP2 warm start（搁置）
    cp3_threshold: float = 0.92            # CP3 predictive

    # Flow matching warm start
    intermediate_timesteps: list[float] = field(
        default_factory=lambda: [0.7, 0.5, 0.3]
    )  # which timesteps to cache x_t for

    # Write policy
    write_similarity_threshold: float = 0.99  # 与已有条目太相似时不写入
    write_async: bool = True

    # Gate
    gate_type: str = "always"              # "always", "interval", "state_change"
    gate_interval: int = 1
    gate_state_threshold: float = 0.01

    # Timing（✅ Step 2 已实现）
    timing_enabled: bool = True
    timing_buffer_size: int = 10000

    # ── 存储配置 — ⚠️ 不稳定，将随 backend 演化而变 ──────────────
    vector_db: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    metadata_db: Optional[MetadataStoreConfig] = None

    # ── 硬件配置 — 搁置（需要 GPU/CPU 混合存储，Section 5.3.1 / Section 7）──
    gpu_capacity: int = 10000              # 搁置
    cpu_capacity: int = 100000             # 搁置
    pinned_memory_mb: int = 32             # 搁置
```

---

## 11. 对现有代码的改动边界

### 实际改动（Step 1 — 仅添加，零修改已有代码）

> **关键发现**：`pi0_pytorch.py` 中已有三个私有 stage 方法（`_stage1_token_prep`、`_stage2_llm_backbone`、`_stage3_action_expert`），`sample_actions()` 内部已串联调用它们。Step 1 **没有修改** `sample_actions()` —— 而是在已有私有方法之上添加了公共类型化包装。

| 文件 | 改动 | 类型 |
|------|------|------|
| `models_pytorch/pi0_pytorch.py` | 添加 3 个 dataclass（`Stage1Output`、`Stage2Output`、`Stage3Output`）+ 5 个方法（`run_stage1/2/3`、`run_stage3_from`、`_stage3_with_intermediates`） | **仅添加**（+256 行，0 行修改） |
| `src/openpi/cache/__init__.py` | 新建模块标记 | **新建文件** |
| `src/openpi/cache/interceptor.py` | `InferenceInterceptor(BasePolicy)` — 将推理路由到 staged API | **新建文件**（+137 行） |
| `scripts/serve_policy.py` | 添加 `--cache` CLI 参数和 `InferenceInterceptor` 包装逻辑 | **仅添加**（+13 行） |

### 实际改动（Step 3 — ⚠️ 不稳定，将频繁变动）

> **状态**：⚠️ 代码已落地，无测试覆盖，接口将频繁变动。
> Step 3 包含两个子部分：数据收集（稳定的 observer 模式）和 cache 存储层（不稳定）。

**Step 3a：数据收集**（`logs/archive/step3_data_collection.log`）

| 文件 | 改动 | 类型 |
|------|------|------|
| `src/openpi/collect/__init__.py` | 空模块标记 | **新建文件** |
| `src/openpi/collect/data_collector.py` | `InferenceEmbeddings` dataclass + `EpisodeDataCollector`（缓冲 + HDF5 写入） | **新建文件** |
| `src/openpi/collect/collection_policy.py` | `CollectionPolicy` wrapper（最外层，4 个 forward hook） | **新建文件** |
| `scripts/serve_policy.py` | 添加 `--collect` / `--collect_dir` 参数，最外层 `CollectionPolicy` 包装 | **仅添加** |
| `serving/websocket_policy_server.py` | 带内控制消息识别（`__ctrl__: episode_start/end`） | **修改** |
| `openpi-client/websocket_client_policy.py` | 添加 `episode_start()` / `episode_end()` 方法 | **仅添加** |
| `examples/libero/main.py` | 在 episode 循环前后插入 `client.episode_start/end()` 调用 | **修改** |

**Step 3b：Cache 存储层**（`logs/archive/step3_cache.log`）— ⚠️

| 文件 | 改动 | 类型 |
|------|------|------|
| `src/openpi/cache/types.py` | `CheckpointID` 枚举 | **新建文件** |
| `src/openpi/cache/storage_types.py` | `CachePayload`、`CacheEntry`、`QueryFilter`、`QuerySpec`、`SearchResultLite`、`SearchResult`、`BatchInsertResult` | **新建文件** |
| `src/openpi/cache/backend_base.py` | `VectorStoreBackend` ABC | **新建文件** |
| `src/openpi/cache/cache_storage.py` | `CacheStorage` 门面（线程安全、校验、两段式搜索） | **新建文件** |
| `src/openpi/cache/backends/__init__.py` | 空模块标记 | **新建文件** |
| `src/openpi/cache/backends/qdrant_backend.py` | `QdrantVectorStore`（`VectorStoreBackend` 的 Qdrant 实现） | **新建文件** |
| `src/openpi/cache/__init__.py` | 更新导出新符号 | **修改** |

### 不修改的部分

| 文件 | 说明 |
|------|------|
| `models_pytorch/pi0_pytorch.py` — 已有代码 | `sample_actions()`、`_stage1/2/3_*`、`denoise_step()` — 零修改 |
| `policies/policy.py` | Policy 类不变，InferenceInterceptor 在外层包装 |
| `models/pi0.py` | JAX 路径已关闭，不动 |
| `serving/websocket_policy_server.py` | ✅ Step 2：移除旧 `stage_timing_records` 聚合，添加 `TaskLifecycle` 回调；✅ Step 3a：添加控制消息处理 |
| `training/` | 训练代码完全不动 |
| `transforms.py` | 数据变换不变 |

---

## 12. 文件结构

### 实际（截至 Step 3）

```
src/openpi/cache/                           # Cache 模块
├── __init__.py                             # ✅ Step 1，Step 3 更新
├── interceptor.py                          # ✅ Step 1 — InferenceInterceptor (wraps Policy)
├── timing.py                               # ✅ Step 2 — SystemTimer, TimingRecord, TimingStats
├── types.py                                # ⚠️ Step 3 — CheckpointID 枚举
├── storage_types.py                        # ⚠️ Step 3 — CachePayload, CacheEntry, QuerySpec 等
├── backend_base.py                         # ⚠️ Step 3 — VectorStoreBackend ABC
├── cache_storage.py                        # ⚠️ Step 3 — CacheStorage 门面
├── backends/
│   ├── __init__.py                         # ⚠️ Step 3
│   └── qdrant_backend.py                   # ⚠️ Step 3 — QdrantVectorStore
└── README.md                               # ⚠️ Step 3 — 模块级文档

src/openpi/collect/                         # 数据收集模块（Step 3a）
├── __init__.py
├── data_collector.py                       # InferenceEmbeddings + EpisodeDataCollector
└── collection_policy.py                    # CollectionPolicy wrapper（最外层）
```

### 计划（未来步骤，尚未创建）

```
src/openpi/cache/
├── config.py                               # CacheConfig, VectorStoreConfig 等
├── orchestrator.py                         # CacheOrchestrator 主控
├── components/
│   ├── key_builder.py                      # QueryKeyBuilder protocol + 实现
│   ├── gate.py                             # GateFunction protocol + 实现
│   └── judge.py                            # SimilarityJudge protocol + 实现
├── hardware/                               # 搁置 — GPU/CPU 混合存储
│   ├── cuda_manager.py                     # CacheHardwareManager，stream 管理
│   └── memory_pool.py                      # PinnedMemoryPool
├── maintenance/                            # 搁置 — 淘汰、迁移
│   ├── eviction.py
│   ├── promotion.py
│   └── writer.py
└── backends/
    ├── faiss_backend.py                    # 计划 — FAISS 本地 backend
    └── torch_gpu_backend.py                # 计划 — TorchGPU 进程内 backend
```

---

## 13. 开发路线图

> 核心原则：**先搭骨架跑通，再在可运行的系统上做实验和优化。**
> 不要在没有端到端 pipeline 之前研究 similarity metric；不要在没有计时数据之前做性能优化。

---

### Step 0: 认识现有推理管线 — ✅ 已融入 Step 1

> Step 0 没有作为独立步骤执行。代码分析和基线理解在 Step 1 的规划阶段完成。详见 `logs/archive/step1.log` 第一节，包含张量形状、阶段间数据流、Pi0 vs Pi0.5 架构对比等完整分析结果。

---

### Step 1: Staged Public API + Interceptor 骨架 — ✅ 已完成

> **状态**：已验证 | **Commit**：`a6c9f43` on branch `Ziyang` | **日期**：2026-03-29
> **日志**：`logs/archive/step1.log`

**目标**：在已有的私有 stage 方法之上添加公共类型化包装，并创建 `InferenceInterceptor` 骨架将推理路由到 staged API。

**代码分析关键发现**：`pi0_pytorch.py` 中已有三个私有方法（`_stage1_token_prep`、`_stage2_llm_backbone`、`_stage3_action_expert`），`sample_actions()` 内部已串联调用。不需要"拆分"——只需添加公共类型化包装。

**关键发现 — CP2 前提失效**：架构设计时假设 Pi0.5 的 Stage 2 会进行自回归子任务文本生成（产出 command tokens + command embedding）。但 **PyTorch 路径没有自回归生成** —— Stage 2 只做 prefix KV cache 填充。唯一输出是 opaque 的 `past_key_values`（DynamicCache），无法作为检索键。**CP2 因此搁置**，待找到合适的表征提取方案后再启用。

**实际实现内容**：

1.1. 在 `pi0_pytorch.py` 中添加 3 个 dataclass（`PI0Pytorch` 类之前）：

```python
@dataclass
class Stage1Output:
    state: torch.Tensor               # [B, action_dim] — 原始状态，用于 cache key
    prefix_embs: torch.Tensor         # [B, prefix_len, emb_dim] (bfloat16)
    prefix_pad_masks: torch.Tensor    # [B, prefix_len] (bool)
    prefix_att_2d_masks_4d: torch.Tensor  # [B, 1, prefix_len, prefix_len]
    prefix_position_ids: torch.Tensor # [B, prefix_len] (int64)

@dataclass
class Stage2Output:
    stage1: Stage1Output
    past_key_values: Any              # HuggingFace DynamicCache — 原样传递，禁止 clone

@dataclass
class Stage3Output:
    action_chunk: torch.Tensor        # [B, action_horizon, action_dim] (float32)
    intermediates: Optional[dict[float, torch.Tensor]] = None  # warm start 用
```

> **注意**：`Stage2Output` 中没有 `command_tokens` 或 `command_embedding` —— PyTorch 路径中不存在这些。这是 CP2 搁置的根本原因。

1.2. 在 `PI0Pytorch` 类中添加 5 个方法（全部为新增，零修改已有代码）：
  - `run_stage1(observation) -> Stage1Output` — 包装 `_stage1_token_prep`
  - `run_stage2(stage1) -> Stage2Output` — 包装 `_stage2_llm_backbone`
  - `run_stage3(stage2, *, noise, num_steps, return_intermediates, save_timesteps) -> Stage3Output`
  - `run_stage3_from(stage2, start_x, start_t, *, num_steps) -> Stage3Output` — warm start 入口
  - `_stage3_with_intermediates(...)` — 中间状态捕获的内部辅助方法

1.3. 创建 `src/openpi/cache/interceptor.py`（+137 行）：
  - `InferenceInterceptor(BasePolicy)` — 从被包装的 Policy 借用 `_model`、`_input_transform`、`_output_transform`
  - `infer()` 调用 `run_stage1 → run_stage2 → run_stage3`，每阶段计时
  - `TODO(Step 4)` 标记 CP1/CP3 缓存检查的插入点

1.4. 修改 `scripts/serve_policy.py`（+13 行）：
  - 添加 `--cache` CLI 参数
  - `if args.cache: policy = InferenceInterceptor(policy)`

**验证结果**：

| 验证项 | 结果 |
|--------|------|
| AST 语法检查（所有新增文件） | ✅ 通过 |
| server 以 `--cache` 启动 | ✅ `INFO: Cache mode enabled` + `listening on 0.0.0.0:8001` |
| 已有计时系统（`stage_timing` 字段） | ✅ interceptor 输出相同字段 |
| 外部接口兼容性 | ✅ WebsocketPolicyServer 和客户端无需任何修改 |

**产出**：`pi0_pytorch.py` 新增公共 staged API + `interceptor.py` + `--cache` 集成。

---

### Step 2: 计时系统 — ✅ 已完成

**目标**：实现 `SystemTimer`，为所有后续的性能量化提供基础设施。

**实际实现**（完整细节见 `logs/archive/step2.log`）：

2.1. `src/openpi/cache/timing.py`：
  - 基于 probe 的 `SystemTimer`，支持 `register_probe(name, backend="cuda"/"cpu", stream=None)`
  - `TimingBackend` 协议，`CudaEventBackend`（无 GPU 时自动降级为 CPU）和 `PerfCounterBackend`
  - `enabled=False` 零开销关闭；未注册 probe 以宽松模式处理并给出警告
  - 环形缓冲区 + 单调计数器追踪任务边界
  - `summary()` 输出逐 probe 的 mean/p50/p95/p99 + 正确的 total (sum) 行
  - `export_csv()` 导出原始记录；通过 `output_csv_dir` 在任务结束时自动写 CSV

2.2. `InferenceInterceptor` 注册 4 个 probe：`stage1_vision`/`stage2_llm`/`stage3_flow`（cuda）+ `total_inference`（cpu），替换 Step 0 的手动计时。

2.3. `TaskLifecycle` 协议：`on_task_begin()`/`on_task_end()` 实现任务级聚合。

2.4. `websocket_policy_server.py`：移除旧 `stage_timing_records` 聚合（约 -15 行），在连接建立/关闭时添加 `TaskLifecycle` 回调。

2.5. 验证：`scripts/verify_step2.py` 全部 12 项测试通过。

**产出**：`timing.py` + 集成到 interceptor 的计时 + server 侧生命周期回调。

---

### Step 3: 数据收集 + Cache 存储层 — ⚠️ 已落地（高危）

> **状态**：⚠️ 代码已落地，无测试覆盖，接口将频繁变动。
> **日志**：`logs/archive/step3_data_collection.log`（3a）、`logs/archive/step3_cache.log`（3b）
> **日期**：2026-04-02

**目标**：两个子部分——（3a）构建纯 observer 数据收集系统，将推理嵌入写入 HDF5；（3b）实现 cache 存储层抽象，使上层逻辑与具体向量数据库解耦。

**为什么做存储抽象？** 我们尚不确定最终使用哪个向量数据库。当前用 Qdrant 做实验，但**一定会更换**（候选：FAISS、自研 TorchGPU store 等）。`VectorStoreBackend` ABC 确保换 backend 时 orchestrator 和业务代码无需任何修改。

**实际实现内容**：

3a. **数据收集**（稳定的 observer 模式）：
  - `src/openpi/collect/collection_policy.py`：`CollectionPolicy` wrapper（wrapper 链最外层），每次推理注册 4 个临时 forward hook，将 `InferenceEmbeddings` 写入 `EpisodeDataCollector`
  - `src/openpi/collect/data_collector.py`：`InferenceEmbeddings` dataclass + `EpisodeDataCollector`（缓冲 + HDF5 原子写入）
  - 4 个 forward hook：`_vision_hook`（multi_modal_projector）、`_lang_hook`（embed_tokens）、`_action_in_hook`（action_in_proj）、`_action_out_hook`（action_out_proj）
  - HDF5 schema：per-step 分组，含 vision_0/1/2、prompt_emb、robot_state、noise_action_1..N-1、clean_action
  - 带内控制消息（`__ctrl__: episode_start/end`）实现客户端-服务器 episode 生命周期
  - `scripts/serve_policy.py`：`--collect` / `--collect_dir` 参数

3b. **Cache 存储层**（⚠️ 不稳定）：
  - `src/openpi/cache/types.py`：`CheckpointID` 枚举（CP1/CP2/CP3）
  - `src/openpi/cache/storage_types.py`：`CachePayload`、`CacheEntry`、`QueryFilter`、`QuerySpec`、`SearchResultLite`、`SearchResult`、`BatchInsertResult` — 全部强类型，含 CP 级校验
  - `src/openpi/cache/backend_base.py`：`VectorStoreBackend` ABC — 最小接口（insert/search/fetch_payload/delete/count + supported_filters）
  - `src/openpi/cache/cache_storage.py`：`CacheStorage` 门面 — 线程安全（RLock）、维度校验、filter fail-fast、两段式搜索
  - `src/openpi/cache/backends/qdrant_backend.py`：`QdrantVectorStore` — Qdrant 实现，tensor 序列化（torch.save → base64），支持 filter：checkpoint_id/task_key/step_range

**关键设计决策**（详见 `logs/archive/step3_cache.log`）：
- `CachePayload` 是独立的嵌套 dataclass（非扁平字段），含 `validate_for_checkpoint()` 做 CP 级不变量校验
- 两段式搜索：`search()` 返回 `SearchResultLite`（无 payload），`fetch_payload()` 按需取 — 避免传输无用 tensor 数据
- `QueryFilter` + `supported_filters()` fail-fast — 不支持的 filter 抛出 `UnsupportedFilterError`，绝不静默忽略
- ABC 接口故意很小：named vector、multivector、gRPC 选项全在 backend 内部 config，不暴露到 ABC 边界之上
- Cache 系统向量（融合后 `[1024]` query key）与 HDF5 实验向量（原始 embedding）使用不同 Qdrant collection — 禁止混用

**⚠️ 已知风险**：
- 无测试覆盖 — 所有接口都可能变动
- `torch` 懒导入改动未在 uv 环境回归测试
- Qdrant backend 将被替换；ABC 接口可能随新 backend 需求演化
- `CacheConfig` 尚未作为独立文件实现

**产出**：数据收集系统（3a）+ 存储层抽象及 Qdrant backend（3b）。

---

### Step 4: Orchestrator 骨架（CP1 + CP3）

**目标**：将 cache 检查逻辑与推理管线连接起来，实现端到端的 cache 工作流。此阶段使用最简单的组件实现（PlaceholderKeyBuilder + AlwaysSearchGate + ThresholdJudge），**开启 CP1 和 CP3**（CP2 搁置 — 见 Section 3）。

> **注**：`InferenceInterceptor` 已在 Step 1 创建。此步骤添加 `CacheOrchestrator`，将 CP1/CP3 检查插入 interceptor 中已有的 `TODO(Step 4)` 槽位。

**为什么是 CP1 + CP3（而非 CP2）**：
- CP2 的原始理由（"same command → same action"）依赖 command embedding，而 PyTorch 路径中不存在。CP2 搁置。
- CP1 在 vision 之后：使用 `raw_state` 和/或 vision embedding 作为键。语义为 "相同场景+相同状态→相同动作"。最严格阈值（0.98）缓解跳过 subtask 预测的风险。
- CP3 在 action expert 之后：预测性缓存，用于跳过下一推理周期。当连续动作具有时间局部性时，这是节省量最大的路径。

**工作内容**：

4.1. 实现 `src/openpi/cache/components/key_builder.py`：
  - `QueryKeyBuilder` Protocol
  - `PlaceholderKeyBuilder`：使用 `stage1_output.state`（原始状态向量 `[B, 32]`）做 L2 normalize 作为最简键。CP3 额外拼接 state + action chunk。

4.2. 实现 `src/openpi/cache/components/gate.py`：
  - `GateFunction` Protocol
  - `AlwaysSearchGate`：永远返回 True

4.3. 实现 `src/openpi/cache/components/judge.py`：
  - `SimilarityJudge` Protocol
  - `ThresholdJudge`：cosine similarity > threshold → hit

4.4. 实现 `src/openpi/cache/orchestrator.py`：
  - `CacheOrchestrator`：组合 key_builder + gate + judge + storage
  - `check()` 方法：gate → build key → search → judge
  - `write_async()` 方法：先用同步写入（async 在 Step 8 优化）

4.5. 集成到 `src/openpi/cache/interceptor.py`：
  - 将 CP1 检查插入 Stage 1 之后的 `TODO(Step 4)` 槽位
  - 将 CP3 检查插入 Stage 3 之后的 `TODO(Step 4)` 槽位
  - CP2 槽位保持注释状态

4.6. **端到端测试**：
  - 加载模型，跑 10 次相同输入 → 第 1 次 miss，第 2-10 次应该 CP1 hit（输入完全相同）
  - 跑 10 次不同输入 → 全部 miss
  - 验证 CP1 hit 时返回的 action 与正常推理结果的 L2 距离（应该 = 0）

**产出**：可运行的端到端 cache 系统（CP1 + CP3），通过上述测试。 ✅ 已完成（不稳定）——CP3 检查基础设施就位，但 CP3 写入/调度/跳过逻辑为 stub（延迟到 Step 6）。

---

### Step 5: 基础实验——Cache 可行性验证

**目标**：回答核心问题——"对于相似但不完全相同的输入，cache hit 返回的 action 质量如何？" 这决定了整个 cache 系统是否有意义。

> **这是整个项目的第一个关键实验节点。** 如果实验结果表明相似输入的 action 差异过大，整个 cache 思路需要重新评估。不要在这个实验之前投入更多开发工作。

**实验设计**：

5.1. **数据准备**：收集一组推理 episode（100-500 步），记录每一步的：
  - 输入 observation（images, state, prompt）
  - Stage 1 输出（vision embedding, state）
  - 最终 action chunk
  - 将所有上述数据保存到磁盘（HDF5 或 pickle）
  - *（注：command embedding 不可用 — CP2 搁置）*

5.2. **实验 A：State 空间中的 action 连续性**
  - 对记录的 episode，计算所有 step 两两之间的 state cosine similarity
  - 计算对应的 action L2 距离
  - 画 scatter plot：x=state_similarity, y=action_distance
  - **期望看到**：state similarity 高时 action distance 低（正相关）
  - **如果看不到这个趋势**：cache 思路存在根本问题

5.3. **实验 B：Cache hit 的 action 质量**
  - 用 Step 4 的系统，逐步降低 CP1 threshold（从 0.99 到 0.80）
  - 记录每个 threshold 下的：hit rate, action L2 error (vs 正常推理), 延迟节省
  - 画三条曲线：threshold vs hit_rate, threshold vs action_error, threshold vs latency_saving
  - **寻找 sweet spot**：action error 可接受（< 某个值）的前提下 hit rate 最大化

5.4. **实验 C：不同 query key 的区分度**
  - 对比几种 key 构建方式的检索质量：
    - (a) raw state vector only
    - (b) vision embedding mean pool
    - (c) state + vision embedding concatenation
    - (d) state + action chunk concatenation（用于 CP3）
  - *（command embedding 已移除 — PyTorch 路径中不可用）*
  - 指标：precision@k（top-k 检索结果中，action 真正相近的比例）
  - **这个实验指导后续 QueryKeyBuilder 的设计**

**产出**：实验报告，包含上述图表和结论。决定是否继续，以及初步确定 threshold 范围和 key builder 方向。

---

### Step 6: CP1/CP3 细化 + CP3 延迟写入器

**前置条件**：Step 5 实验结果正面（cache 可行性得到验证）。

> **注**：CP1 和 CP3 的基本集成已在 Step 4 完成。此步骤专注于 CP3 的延迟写入机制，以及基于 Step 5 实验结果的细化调优。

**工作内容**：

6.1. **CP1 细化**：
  - 基于 Step 5 实验 C 结果调优 key builder（哪些信息源效果最好）
  - CP1 使用更严格的 threshold（默认 0.98）
  - 测试：相同场景相同 prompt 应该 hit，更换物体或 prompt 应该 miss

6.2. **CP3 延迟写入器**：
  - `schedule_next_action()` 机制：在 orchestrator 中维护一个 `_next_action_scheduled` 槽位
  - `should_skip_inference()`：在每个 cycle 开始前检查是否有预调度的 action
  - CP3 的 key 需要包含 action chunk 信息（因为是预测"下一步"）
  - 需要维护 **连续 action 序列的对应关系**——entry 中增加 `next_entry_id` 字段，指向时间上紧接的下一个 entry

6.3. **CP3 特殊问题**：CP3 的 cache entry 需要记录 "当前 action → 下一步 action" 的对应关系。这意味着：
  - 写入 cache 时，需要等到**下一个** cycle 的 action 产出后，才能补全当前 entry 的 `next_action_chunk` 字段
  - 实现一个 `DeferredWriter`：在 cycle N 写入 entry（不含 next），在 cycle N+1 回填 next_action_chunk

6.4. **实验**：
  - 在 episode 上统计 CP1/CP3 各自的 hit rate（CP2 搁置）
  - 量化各检查点命中时的延迟节省
  - CP3 的 predictive accuracy：预调度的 action 与实际推理出的 action 的 L2 距离

**产出**：细化后的 CP1 + CP3（含延迟写入器）+ 命中率和延迟报告。

---

### Step 7: Flow Matching Warm Start

**前置条件**：Step 6 完成，CP1/CP3 已验证。*（注：warm start 最初为 CP2 设计。CP2 搁置后，warm start 可能用于 CP1 的部分跳过，或推迟到 CP2 重新启用时。）*

**工作内容**：

7.1. 修改 `run_stage3()`：
  - `return_intermediates=True` 时，在 flow matching 循环中保存选定时间步的 `x_t`
  - 默认保存 t=0.7, 0.5, 0.3 三个点（可配置）
  - 保存的张量 shape = `[B, action_horizon, action_dim]`，与 noise 相同

7.2. 新增 `run_stage3_from(stage2_output, start_x, start_t)`：
  - 从 `start_x` 和 `start_t` 开始执行剩余的 Euler steps
  - 例如 `start_t=0.3` 时只跑 3 步（0.3 → 0.2 → 0.1 → 0.0），而非 10 步

7.3. CP2 judge 增加 warm start 判定逻辑：
  - similarity > `cp2_full_threshold` → FULL hit
  - `cp2_warm_threshold` < similarity < `cp2_full_threshold` → WARM_START hit
  - similarity < `cp2_warm_threshold` → miss

7.4. **关键实验：Warm Start 精度 vs 速度 tradeoff**

  这是第二个关键实验节点。

  - 对同一组输入，分别跑：
    - (a) 完整 10 步 flow matching（baseline）
    - (b) 从自身的 cached x_0.7 warm start（3 步跳过）
    - (c) 从自身的 cached x_0.5 warm start（5 步跳过）
    - (d) 从自身的 cached x_0.3 warm start（7 步跳过）
  - 测量 action L2 error vs baseline
  - 这个实验的 "自身 cached" 意味着用**完全相同输入**的中间状态，隔离 warm start 本身的误差（不涉及 state 相似度问题）

  - 然后用**相似但不同**输入的 cached x_t 做 warm start：
    - 从相似 state 的 episode 中取 cached x_0.5
    - 用当前 observation 的 velocity field 继续 denoise
    - 测量 action L2 error
  - **期望**：error 在可接受范围内，且比 "直接用 cache action 不做 flow matching" 更小

  - 画 trade-off 图：x=跳过的步数, y=action_error, 多条线表示不同 state similarity 级别

**产出**：warm start 实现 + trade-off 实验数据，确定默认的 warm start 时间点和 threshold。

---

### Step 8: 系统效率优化——异步与硬件

**前置条件**：Step 7 完成，功能正确性已验证。到这一步才做性能优化，因为优化之前需要精确的 timing 数据来指导投入方向。

**优化决策流程**：先用 Step 2 的 timer 生成完整的延迟分解报告，识别瓶颈所在，然后有针对性地优化。不要凭猜测优化。

**可能的优化方向**（按预期收益排序）：

8.1. **异步 cache 写入**（几乎肯定需要）：
  - 当前 Step 4 的写入是同步的，会阻塞推理
  - 实现 `AsyncWriteWorker`：后台线程从队列中消费写入请求
  - 使用 `threading.Thread` + `queue.Queue`（不需要 multiprocessing，因为写入是 I/O bound）
  - 验证：写入延迟从推理关键路径中消失

8.2. **GPU VectorStore**（如果 CPU 搜索是瓶颈）：
  - 查看 timer 报告中 `cp*_search` 的延迟
  - 如果 CPU FAISS 搜索延迟 > 1ms 且 cache 条目 > 10k，考虑 GPU partition
  - 实现 `torch.mm` cosine similarity search on dedicated CUDA stream
  - 使用独立 stream 避免阻塞主推理 stream
  - 验证：search 延迟下降，主 stream 推理延迟不受影响

8.3. **CUDA Stream 隔离**（如果 cache 操作阻塞了推理）：
  - 实现 `CacheHardwareManager`
  - Cache 的所有 GPU 操作（search, key 构建中的 projection）在 `cache_stream` 上执行
  - Pinned memory pool 用于 CPU↔GPU 数据传输

8.4. **Gate 优化**（如果 cache check 本身成为瓶颈）：
  - 实现 `StateChangeGate`：state 变化小于阈值时跳过搜索
  - 实现 `IntervalGate`：每 N 次推理只搜索一次
  - 这可以大幅减少搜索频率，在 hit rate 低的场景下尤其有用

8.5. **淘汰策略**（如果 cache 增长导致搜索变慢）：
  - 实现 `CompositeEviction`（LRU + LFU + quality）
  - 设置 capacity 上限，定期淘汰
  - 淘汰在后台线程执行

**产出**：优化后的系统 + 优化前后的对比延迟报告。

---

### Step 9: Query Key 研究（实验密集型）

**前置条件**：Step 8 完成，系统性能在可接受范围。此时有一个稳定运行的 cache 系统，可以作为实验平台。

**为什么放在这里而非更早**：Query key 的研究需要在真实运行的系统上做，需要真实的 hit/miss 数据、真实的延迟数字。在系统跑通之前研究 key 是空中楼阁。

**实验方向**：

9.1. **Key 信息源消融实验**：
  - 在已有的 episode 数据上，对比不同信息组合作为 key 的检索质量：

  | Key 组合 | 维度 | Precision@5 | Recall@5 | 计算开销 |
  |---------|------|-------------|----------|---------|
  | raw_state | 32 | ? | ? | 极低 |
  | state + prompt_hash | 32+64 | ? | ? | 低 |
  | vision_emb (mean pool) | 2048 | ? | ? | 中 |
  | state + vision_emb | 32+2048 | ? | ? | 中 |
  | state + action_chunk (CP3) | 32+1600 | ? | ? | 中 |
  | learned projection | 128/256/512 | ? | ? | 需训练 |

  > *（command_emb 行已移除 — PyTorch 路径中不可用，CP2 搁置）*

  - 其中 "Precision@5" 定义为：top-5 检索到的 entry 中，其 action 与当前推理 action 的 L2 距离 < epsilon 的比例

9.2. **Learned Key Builder**（如果简单方法效果不够）：
  - 训练一个小 projection head（2-3 层 MLP），输入为 stage output 的 concatenation，输出为低维 key
  - 训练目标：contrastive loss——相似 state 的 key 距离近，不同 state 的 key 距离远
  - 训练数据来自离线 episode 收集
  - 约束：projection head 的推理延迟 < 0.5ms（否则不如不用 cache）

9.3. **不同检查点的最优 key 可能不同**：
  - CP1 的 key 有 vision + state 信息，实验不同权重组合
  - CP3 的 key 需要 action 信息来预测后续
  - *（CP2 搁置 — 重新启用前无需设计 key）*
  - 每个活跃 checkpoint 独立调参

**产出**：每个 checkpoint 的最优 key builder 方案 + 实验数据支撑。

---

### Step 10: 离线预填充管道

**前置条件**：Step 9 完成，key builder 方案确定。

**工作内容**：

10.1. 实现 `scripts/prefill_cache.py`：
  - 从训练数据或离线 rollout 中加载 episodes
  - 对每个 timestep 跑模型推理（或直接从保存的 episode 数据中读取）
  - 构建 key，写入 VectorStore
  - 支持增量预填充（检查已有 entry，避免重复）

10.2. 序列化/反序列化：
  - VectorStore 支持 `save(path)` 和 `load(path)`
  - 预填充后保存到磁盘，推理时直接加载，避免每次冷启动

10.3. **预填充质量验证**：
  - 用预填充的 cache 跑 inference episode
  - 对比有 cache vs 无 cache 的 action 轨迹差异
  - 量化预填充 cache 的 hit rate（期望远高于空 cache 在线积累）

**产出**：离线预填充脚本 + 序列化支持。

---

### Step 11: 集成测试与鲁棒性

**工作内容**：

11.1. **长时间运行稳定性测试**：
  - 运行 1000+ 步的 episode，监控：
    - 内存是否泄漏（cache 增长是否受控）
    - 延迟是否稳定（是否有逐渐变慢的趋势）
    - hit rate 是否合理变化

11.2. **异常场景测试**：
  - cache 为空时的行为（全 miss，正常推理）
  - cache 满时的淘汰行为
  - 输入异常（全黑图像、空 prompt）时不 crash
  - GPU OOM 时 graceful fallback 到纯 CPU

11.3. **A/B 对比框架**：
  - 实现 `--cache_enabled` flag，一键开关 cache
  - 对比相同 episode 下 cache on/off 的：
    - 端到端延迟（mean, p95）
    - Action 质量（与 no-cache baseline 的 L2 距离）
    - GPU 利用率

**产出**：稳定性测试报告 + A/B 对比数据。

---

### Step 12: 进阶功能（按需）

以下各项相互独立，按需求优先级选择实现：

12.1. **Metadata DB（MongoDB/SQLite）**：
  - 当需要存储 vector DB 之外的丰富信息时引入（task name, episode id, success/failure label 等）
  - 用于 cache 分析和离线质量评估
  - 不在关键路径上，不影响推理延迟

12.2. **GPU/CPU 动态迁移**：
  - Promotion/Demotion 策略
  - 基于访问频率的自动数据分层

12.3. **学习型 Gate Function**：
  - 训练一个二分类器预测 "当前 state 是否可能 cache hit"
  - 输入：state delta (vs 上次)、task prompt hash、cache 统计信息
  - 减少无意义的搜索开销

12.4. **分布式 Cache**：
  - 多机器人共享 vector DB
  - 需要考虑网络延迟和一致性

12.5. **Cache-aware 训练**：
  - 在训练时引入 cache-hit 模拟，让模型适应偶尔跳步的场景
  - 长期方向，需要大量实验

---

### 开发依赖关系总图

```
Step 0: 认识推理管线 ─── ✅ 已融入 Step 1
  │
  ▼
Step 1: Staged Public API + Interceptor ─── ✅ 已完成 (a6c9f43)
  │  关键发现：无自回归生成 → CP2 搁置
  ▼
Step 2: 计时系统 ─── ✅ 已完成
  │
  ├──────────────────┐
  ▼                  ▼
Step 3: 数据结构    (并行开发)
  │
  ▼
Step 4: Orchestrator (CP1 + CP3) ──── ✅ 已完成（不稳定） ──── CP2 搁置，不在关键路径上
  │
  ▼
Step 5: ★ 可行性实验 ★  ── 如果失败 ──> 重新评估整体方案
  │ (通过)
  ▼
Step 6: CP1/CP3 细化 + CP3 延迟写入器
  │
  ▼
Step 7: Warm Start（可能推迟 — 原为 CP2 设计）
  │
  ▼
Step 8: 系统效率优化 (async, GPU, stream)
  │
  ▼
Step 9: ★ Query Key 研究 ★ (实验密集)
  │
  ▼
Step 10: 离线预填充
  │
  ▼
Step 11: 集成测试
  │
  ▼
Step 12: 进阶功能 (按需)
         ├── CP2 重新启用（待表征提取方案就绪）
```

**★ 标记的步骤是关键实验节点**，其结论直接决定后续工作的方向甚至是否继续。

---

### 各步骤预估工作量

| Step | 状态 | 工作类型 | 主要产出 |
|------|------|---------|---------|
| 0 | ✅ 已融入 Step 1 | 阅读+分析 | （包含在 Step 1 日志中） |
| 1 | ✅ 已完成 | Staged API + Interceptor | pi0_pytorch.py 公共 API + interceptor.py |
| 2 | ✅ 已完成 | 基础设施开发 | timing.py + SystemTimer |
| 3 | ⚠️ 代码落地，无测试 | 基础设施开发 | 数据结构 + VectorStore backend |
| 4 | 待开始 | 核心开发 | Orchestrator（CP1 + CP3） |
| 5 | 待开始 | **实验** | 可行性报告 |
| 6 | 待开始 | 核心开发 | CP1/CP3 细化 + CP3 延迟写入器 |
| 7 | 待开始（可能推迟） | 开发+**实验** | Warm start（原为 CP2 设计） |
| 8 | 待开始 | 性能优化 | 异步/GPU/Stream |
| 9 | 待开始 | **实验** | Key builder 研究 |
| 10 | 待开始 | 工具开发 | 预填充脚本 |
| 11 | 待开始 | 测试 | 稳定性+A/B 报告 |
| 12 | 待开始 | 进阶 | 按需（含 CP2 重新启用） |

> **注**：Step 2 和 Step 3 之间无依赖，可以并行开发。Step 7（warm start）原为 CP2 设计，CP2 搁置后可能推迟或转用。

---

## 14. 关键设计决策与 Tradeoff 记录

| 决策 | 选择 | 替代方案 | 理由 |
|------|------|---------|------|
| 接入方式 | Interceptor 外包装 | 修改 PI0Pytorch 内部 | 解耦优先，便于回退和 A/B 测试 |
| GPU vector search | torch.mm 在独立 CUDA stream | FAISS GPU index | 更可控，避免 FAISS GPU 的 CUDA 上下文竞争 |
| 中间状态缓存 | 选择性时间点 (2-3个) | 全部 10 步 | 存储效率，10步全存 action_dim=32 也只多几 KB，但检索判定更复杂 |
| Query key | Protocol 接口，初期 raw state | 固定方案 | 目前信息不足以确定最优方案，保持灵活 |
| CP1 阈值最严 | 0.98 | 统一阈值 | CP1 跳过最多计算，错误代价最高 |
| 写入去重 | 相似度检查 | 全写 | 避免 cache 膨胀，保持检索效率 |
| CP2 搁置 | 暂不实现 CP2 | 从 KV cache 提取 hidden state 做 key | PyTorch 路径 Stage 2 无自回归生成，无 command embedding 可用；强行提取 KV cache 特征复杂度高且效果未知 |
| 首轮验证用 CP1+CP3 | 跳过 CP2，先做 CP1+CP3 | 原计划先做 CP2 | CP2 前提失效；CP1 语义简单（state 相似→action 相似），CP3 节省量最大 |
