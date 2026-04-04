# Step 4 Test Plan: Orchestrator Skeleton Verification

> Status: Plan
> Scope: Step 4 所有新增/修改代码的自动化测试
> Dependencies: pytest, torch (CPU-only sufficient)

---

## Test Strategy

### Principles

1. **CPU-only**: 所有测试在 CPU 上运行，不依赖 GPU。Stage outputs 用 mock tensor 模拟。
2. **No model dependency**: 不加载任何模型权重。使用 mock/fake 对象替代 Policy、Stage1Output 等。
3. **Backend mock**: 用内存字典实现 `VectorStoreBackend`，替代 Qdrant。
4. **Isolated**: 每个 test function 独立，无状态共享。
5. **Test file位置**: 与被测代码同目录，`_test.py` 后缀（项目惯例）。

### Test Files

| Test File | Tested Module | Priority |
|-----------|--------------|----------|
| `components/key_builder_test.py` | PlaceholderKeyBuilder | P0 |
| `components/gate_test.py` | AlwaysSearchGate + Protocol conformance | P1 |
| `components/judge_test.py` | ThresholdJudge + HitType | P0 |
| `orchestrator_test.py` | CacheOrchestrator check/write/clear | P0 |
| `cache_storage_test.py` | _check_entry_dims intersection fix | P0 |
| `interceptor_test.py` | InferenceInterceptor cache integration | P2 (需 mock 较多) |

---

## Test File 1: `components/key_builder_test.py`

### Fixtures / Helpers

```python
def make_stage1_output(state_tensor):
    """Create a mock Stage1Output with given state tensor."""
    # SimpleNamespace with .state attribute
    return SimpleNamespace(state=state_tensor)

def make_stage3_output(action_chunk_tensor):
    """Create a mock Stage3Output with given action_chunk tensor."""
    return SimpleNamespace(action_chunk=action_chunk_tensor)
```

### Test Cases

#### T1.1: `test_collect_stores_gpu_refs`
- **Purpose**: collect() 后 cached_data 包含正确的 key
- **Input**: stage1 with state [1, 32]
- **Assert**: `kb.cached_data["state"]` is the same tensor object (零拷贝验证)
- **Assert**: `kb.cached_data["state"].shape == (1, 32)`

#### T1.2: `test_collect_stage3_stores_action_chunk`
- **Input**: stage1 (state) + stage3 (action_chunk [1, 50, 32])
- **Assert**: cached_data 包含 "state" 和 "action_chunk" 两个 key

#### T1.3: `test_collect_clears_previous`
- **Purpose**: 连续两次 collect()，第二次覆盖第一次
- **Input**: 第一次传 stage1_a，第二次传 stage1_b
- **Assert**: cached_data["state"] is stage1_b.state

#### T1.4: `test_build_cp1_returns_normalized_cpu_float32`
- **Purpose**: build(CP1) 输出满足 tensor contract
- **Input**: state = random [1, 32] (any device)
- **Assert**:
  - 返回 dict 包含 key `"robot_state"`
  - tensor.device == cpu
  - tensor.dtype == float32
  - tensor.is_contiguous() == True
  - tensor.shape == (32,)
  - L2 norm ~= 1.0 (torch.linalg.norm)

#### T1.5: `test_build_cp3_same_as_cp1`
- **Purpose**: CP3 目前与 CP1 使用相同 key（Step 6 前）
- **Input**: 同 T1.4
- **Assert**: build(CP3) 输出与 build(CP1) 值相同

#### T1.6: `test_build_unsupported_checkpoint_raises`
- **Input**: collect + build(CP2)
- **Assert**: raises ValueError

#### T1.7: `test_build_deterministic`
- **Purpose**: 相同输入产生相同输出
- **Input**: 同一 state tensor，调用 build(CP1) 两次
- **Assert**: torch.allclose(result1["robot_state"], result2["robot_state"])

#### T1.8: `test_clear_empties_cache`
- **Input**: collect → clear
- **Assert**: cached_data == {}

#### T1.9: `test_build_without_collect_raises`
- **Input**: 直接 build(CP1)，不 collect
- **Assert**: raises KeyError

---

## Test File 2: `components/gate_test.py`

### Test Cases

#### T2.1: `test_always_search_gate_returns_true`
- **Input**: AlwaysSearchGate()(CP1, {})
- **Assert**: returns True

#### T2.2: `test_always_search_gate_cp3_returns_true`
- **Input**: AlwaysSearchGate()(CP3, {"state": torch.randn(32)})
- **Assert**: returns True

#### T2.3: `test_always_search_gate_conforms_to_protocol`
- **Assert**: isinstance(AlwaysSearchGate(), GateFunction)

---

## Test File 3: `components/judge_test.py`

### Fixtures / Helpers

```python
def make_result(id, score, cp=CheckpointID.CP1):
    return SearchResultLite(id=id, score=score, checkpoint_id=cp)
```

### Test Cases

#### T3.1: `test_threshold_judge_full_hit`
- **Input**: results=[SearchResultLite(score=0.99)], CP1, threshold=0.98
- **Assert**: (HitType.FULL_HIT, "entry_id")

#### T3.2: `test_threshold_judge_miss_below_threshold`
- **Input**: results=[SearchResultLite(score=0.95)], CP1, threshold=0.98
- **Assert**: (HitType.MISS, None)

#### T3.3: `test_threshold_judge_miss_empty_results`
- **Input**: results=[], CP1
- **Assert**: (HitType.MISS, None)

#### T3.4: `test_threshold_judge_exact_threshold_is_hit`
- **Input**: results=[SearchResultLite(score=0.98)], CP1, threshold=0.98
- **Assert**: (HitType.FULL_HIT, ...) — `>=` 语义

#### T3.5: `test_threshold_judge_cp3_uses_cp3_threshold`
- **Input**: results=[SearchResultLite(score=0.96)], CP3, cp3_threshold=0.95
- **Assert**: FULL_HIT (0.96 >= 0.95)

#### T3.6: `test_threshold_judge_cp3_miss`
- **Input**: results=[SearchResultLite(score=0.93)], CP3, cp3_threshold=0.95
- **Assert**: MISS

#### T3.7: `test_threshold_judge_unknown_cp_uses_default`
- **Input**: results=[SearchResultLite(score=0.99)], CP2
- **Assert**: uses default 0.98, returns FULL_HIT

#### T3.8: `test_threshold_judge_conforms_to_protocol`
- **Assert**: isinstance(ThresholdJudge(), SimilarityJudge)

#### T3.9: `test_threshold_judge_custom_thresholds`
- **Input**: ThresholdJudge(cp1_threshold=0.5, cp3_threshold=0.3)
- **Assert**: score=0.6 @ CP1 => FULL_HIT; score=0.4 @ CP3 => FULL_HIT

---

## Test File 4: `orchestrator_test.py`

### Fixtures / Helpers

需要一个 **InMemoryBackend** 实现 VectorStoreBackend，用于测试：

```python
class InMemoryBackend(VectorStoreBackend):
    """Minimal in-memory backend for unit tests.
    
    Stores entries in a dict. Search computes cosine similarity
    against stored vectors for the first matching field.
    """
    def __init__(self, vector_dims: dict[str, int]):
        self._dims = vector_dims
        self._entries: dict[str, CacheEntry] = {}
    
    @property
    def vector_dims(self) -> dict[str, int]:
        return self._dims
    
    def supported_filters(self) -> frozenset[str]:
        return frozenset({"checkpoint_id"})
    
    def insert(self, entry: CacheEntry) -> None:
        self._entries[entry.id] = entry
    
    def search(self, spec: QuerySpec) -> list[SearchResultLite]:
        if not self._entries:
            return []
        # Compute cosine similarity for each stored entry
        results = []
        for eid, entry in self._entries.items():
            # Filter by checkpoint_id if specified
            if spec.checkpoint_id and entry.checkpoint_id != spec.checkpoint_id:
                continue
            # Compute cosine similarity on first matching field
            score = 0.0
            for field in spec.query_keys:
                if field in entry.query_keys:
                    q = spec.query_keys[field].float()
                    e = entry.query_keys[field].float()
                    score = float(F.cosine_similarity(q.unsqueeze(0), e.unsqueeze(0)))
                    break
            results.append(SearchResultLite(id=eid, score=score, checkpoint_id=entry.checkpoint_id))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:spec.top_k]
    
    def fetch_payload(self, id: str) -> CachePayload:
        if id not in self._entries:
            raise KeyError(id)
        return self._entries[id].payload
    
    def delete(self, ids: list[str]) -> None:
        for i in ids:
            self._entries.pop(i, None)
    
    def count(self) -> int:
        return len(self._entries)
```

组装 helper:

```python
def make_orchestrator(vector_dims=None):
    """Create orchestrator with InMemoryBackend + default components."""
    dims = vector_dims or {"robot_state": 32}
    backend = InMemoryBackend(dims)
    storage = CacheStorage(backend)
    kb = PlaceholderKeyBuilder()
    gate = AlwaysSearchGate()
    judge = ThresholdJudge(cp1_threshold=0.98, cp3_threshold=0.95)
    timer = SystemTimer(enabled=False)
    orch = CacheOrchestrator(storage, kb, gate, judge, timer)
    return orch, backend, storage
```

### Test Cases

#### T4.1: `test_check_miss_on_empty_store`
- **Purpose**: 空 store 上 check 返回 MISS
- **Flow**: orchestrator.check(CP1, stage1=mock_stage1)
- **Assert**: result.hit_type == HitType.MISS, result.payload is None

#### T4.2: `test_write_then_check_exact_match_hits`
- **Purpose**: CP1 端到端闭环 — write 后用相同 state check，应命中
- **Flow**:
  1. state = torch.randn(1, 32)
  2. orchestrator.write(CP1, payload, stage1=mock_stage1(state))
  3. orchestrator.clear()
  4. result = orchestrator.check(CP1, stage1=mock_stage1(state))
- **Assert**:
  - result.hit_type == HitType.FULL_HIT
  - result.payload is not None
  - result.payload.action_chunk 内容与写入一致
  - result.score ~= 1.0 (cosine of identical vector)

#### T4.3: `test_write_then_check_different_state_misses`
- **Purpose**: 不同 state 不命中
- **Flow**:
  1. write with state_a
  2. check with state_b
- **Input requirement**: 使用确定性、显式构造的向量，不用纯 random。
  例如 `state_a = e1`, `state_b = -e1`，保证 cosine = -1。
- **Assert**: result.hit_type == HitType.MISS

#### T4.4: `test_write_then_check_similar_state_near_threshold`
- **Purpose**: 验证 threshold boundary
- **Flow**:
  1. write with normalized base vector
  2. construct another normalized vector with cosine 明确大于 0.98（例如 0.99）
- **Input requirement**: 使用确定性构造，不用 random + small noise，避免 flaky。
- **Assert**: FULL_HIT

#### T4.5: `test_judge_miss_does_not_fetch_payload`
- **Purpose**: 验证两阶段搜索语义。即使 search 返回 candidate，只要 Judge 判定 MISS，就不能调用 `fetch_payload()`
- **Setup**:
  - 预写入一条 entry
  - 使用高阈值 Judge 或构造低分 query，使 `search()` 有结果但 `judge()` 返回 MISS
  - backend / storage 记录 `fetch_payload()` 调用次数
- **Assert**:
  - result.hit_type == HitType.MISS
  - result.payload is None
  - `fetch_payload()` 调用次数 == 0

#### T4.6: `test_gate_false_skips_search`
- **Purpose**: Gate 返回 False 时直接 MISS，不调用 storage
- **Setup**: 用自定义 NeverSearchGate
- **Assert**: result.hit_type == HitType.MISS
- **Assert**: backend.search() 未被调用（用显式计数器或 mock 验证）

#### T4.7: `test_write_idempotent_upsert`
- **Purpose**: 相同 state 写两次，store 只有 1 条记录（stable_hash 相同）
- **Flow**: write 同一 state 两次
- **Assert**: backend.count() == 1

#### T4.8: `test_write_different_states_creates_separate_entries`
- **Flow**: write state_a, write state_b
- **Assert**: backend.count() == 2

#### T4.9: `test_cp3_check_always_misses_in_step4`
- **Purpose**: CP3 没有写入，check 必定 MISS
- **Flow**: write CP1 entry, then check CP3
- **Assert**: MISS (because checkpoint_id filter or no CP3 entries)

#### T4.10: `test_should_skip_inference_returns_none`
- **Purpose**: Step 4 stub 始终返回 None
- **Assert**: orchestrator.should_skip_inference() is None

#### T4.11: `test_schedule_next_action_is_noop`
- **Purpose**: Step 4 stub 不报错
- **Flow**: orchestrator.schedule_next_action(torch.randn(50, 32))
- **Assert**: no exception, should_skip_inference() still None

#### T4.12: `test_clear_resets_key_builder`
- **Flow**: orchestrator.check(CP1, ...) then orchestrator.clear()
- **Assert**: key_builder.cached_data == {}

#### T4.13: `test_check_returns_score_and_entry_id_on_hit`
- **Flow**: write + check (exact match)
- **Assert**: result.score is not None, result.entry_id is not None

#### T4.14: `test_stable_hash_deterministic`
- **Purpose**: 相同 checkpoint + query_keys 产生相同 hash
- **Import**: `from openpi.cache.orchestrator import _stable_hash`
- **Assert**: hash1 == hash2

#### T4.15: `test_stable_hash_different_checkpoints_differ`
- **Purpose**: CP1 和 CP3 的 hash 不同（即使 query_keys 相同）
- **Assert**: hash_cp1 != hash_cp3

---

## Test File 5: `cache_storage_test.py`

### Test Cases (focused on _check_entry_dims fix)

#### T5.1: `test_check_entry_dims_subset_accepted`
- **Purpose**: entry 只有 backend fields 的子集时不报错
- **Setup**: backend.vector_dims = {"robot_state": 32, "vision_0": 1024}
- **Input**: entry.query_keys = {"robot_state": tensor(32)}
- **Assert**: insert 成功，无 ValueError

#### T5.2: `test_check_entry_dims_no_overlap_raises`
- **Purpose**: entry 字段与 backend 完全不重叠时报错
- **Setup**: backend.vector_dims = {"vision_0": 1024}
- **Input**: entry.query_keys = {"robot_state": tensor(32)}
- **Assert**: raises ValueError

#### T5.3: `test_check_entry_dims_wrong_dim_raises`
- **Purpose**: 字段重叠但维度不匹配
- **Setup**: backend.vector_dims = {"robot_state": 32}
- **Input**: entry.query_keys = {"robot_state": tensor(64)}
- **Assert**: raises ValueError

#### T5.4: `test_check_entry_dims_extra_fields_ignored`
- **Purpose**: entry 有 backend 未声明的字段，被静默忽略
- **Setup**: backend.vector_dims = {"robot_state": 32}
- **Input**: entry.query_keys = {"robot_state": tensor(32), "unknown_field": tensor(10)}
- **Assert**: insert 成功

---

## Test File 6: `interceptor_test.py` (P2, optional)

### Rationale

InferenceInterceptor 与 Policy/model 耦合度高，需要 mock 大量内部接口。
可在服务器上做集成测试替代，此处只列出关键 mock 点。

### Mock Structure

```python
class FakeModel:
    """Minimal model with staged API."""
    config = SimpleNamespace(pytorch_compile_mode=None)  # eager mode
    
    def run_stage1(self, observation):
        return SimpleNamespace(
            state=torch.randn(1, 32),
            prefix_embs=torch.randn(1, 100, 2048),
        )
    
    def run_stage2(self, stage1):
        return SimpleNamespace(kv_cache=None)
    
    def run_stage3(self, stage2, noise=None):
        return SimpleNamespace(
            action_chunk=torch.randn(1, 50, 32),
        )

class FakePolicy:
    """Minimal Policy mock."""
    _is_pytorch_model = True
    _model = FakeModel()
    _input_transform = lambda self, x: x
    _output_transform = lambda self, x: x
    _pytorch_device = "cpu"
    metadata = {}
```

### Test Cases (if implemented)

#### T6.1: `test_infer_without_orchestrator_returns_actions`
- 验证 orchestrator=None 时正常推理路径

#### T6.2: `test_infer_with_orchestrator_cp1_miss_full_pipeline`
- 验证 CP1 MISS 时走完 stage1 -> stage2 -> stage3 + cp1_write + cp3_check

#### T6.3: `test_infer_with_orchestrator_cp1_hit_skips_stage2_3`
- 预写入 CP1 entry，验证命中时跳过 stage2 + stage3
- **Assert**:
  - `run_stage2()` / `run_stage3()` 调用次数为 0
  - `orchestrator.write()` 未被调用
  - CP3 check 路径未被调用

#### T6.4: `test_infer_cp3_consume_stub_does_not_skip`
- 验证 should_skip_inference() 返回 None 时不跳过

#### T6.5: `test_infer_cp3_consume_with_scheduled_action_skips_all_stages`
- **Purpose**: 覆盖 CP3 consume 的非空返回分支，验证 shape 处理和早返回逻辑
- **Setup**:
  - fake orchestrator 的 `should_skip_inference()` 返回 `[50, 32]` action tensor
  - fake policy/model 记录 `run_stage1/2/3()` 调用次数
- **Assert**:
  - `infer()` 直接返回，不进入 stage1 / stage2 / stage3
  - `outputs["actions"].shape == (50, 32)`
  - `outputs["state"]` 仍为去 batch 后的 state
  - `orchestrator.clear()` 被调用

---

## Shared Test Infrastructure

### `conftest.py` (for `src/openpi/cache/`)

```python
"""Cache test fixtures and helpers."""
import pytest
import torch
from types import SimpleNamespace

from openpi.cache.types import CheckpointID, ROBOT_STATE
from openpi.cache.storage_types import CachePayload, CacheEntry

@pytest.fixture
def random_state():
    """Random [1, 32] state tensor (CPU)."""
    return torch.randn(1, 32)

@pytest.fixture
def mock_stage1(random_state):
    """Mock Stage1Output with random state."""
    return SimpleNamespace(state=random_state)

@pytest.fixture
def sample_payload():
    """Valid CP1 CachePayload."""
    return CachePayload(action_chunk=torch.randn(50, 32))
```

---

## Execution

```bash
# 运行全部 cache 测试
uv run pytest src/openpi/cache/ -v

# 运行单个文件
uv run pytest src/openpi/cache/orchestrator_test.py -v

# 运行单个测试
uv run pytest src/openpi/cache/orchestrator_test.py::test_write_then_check_exact_match_hits -v
```

---

## Coverage Goals

| Module | Target Coverage | Notes |
|--------|----------------|-------|
| key_builder.py | 90%+ | 所有 public method + edge cases |
| gate.py | 100% | 极简，2-3 个测试即可 |
| judge.py | 95%+ | threshold boundary + empty results |
| orchestrator.py | 85%+ | check/write 主路径 + stubs |
| cache_storage.py | +增量 | 只测 _check_entry_dims 修复 |
| interceptor.py | 50%+ | P2, 依赖 mock 复杂度 |

---

## Implementation Order

1. **InMemoryBackend** helper（在 conftest.py 或 orchestrator_test.py 中）
2. **key_builder_test.py** — 最基础，无外部依赖
3. **judge_test.py** — 纯逻辑，只需 SearchResultLite
4. **gate_test.py** — 极简
5. **cache_storage_test.py** — _check_entry_dims 修复验证
6. **orchestrator_test.py** — 集成以上所有组件
7. **interceptor_test.py** — P2, 可延后

---

## Risk Notes

- **InMemoryBackend 的 cosine similarity 必须与 Qdrant 语义一致**：都返回 [-1, 1] 范围。ThresholdJudge 阈值基于此语义。
- **torch.compile 不在测试范围**：eager mode 足够验证逻辑正确性。
- **GPU tensor path**: 测试全部用 CPU tensor，但 build() 中的 `.cpu()` 调用仍有效（CPU→CPU 是 no-op）。
- **_stable_hash 用 numpy**：确保测试中 tensor 在 CPU 上，否则 `.numpy()` 会报错。
- **Threshold 边界测试避免 random**：`near-threshold` 场景必须用确定性构造的归一化向量，避免 flaky。
