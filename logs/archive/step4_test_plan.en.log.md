# Step 4 Test Plan: Orchestrator Skeleton Verification

> Status: Plan
> Scope: Automated tests for all new/modified code in Step 4
> Dependencies: pytest, torch (CPU-only sufficient)

---

## Test Strategy

### Principles

1. **CPU-only**: All tests run on CPU with no GPU dependency. Stage outputs are simulated with mock tensors.
2. **No model dependency**: No model weights are loaded. Mock/fake objects replace Policy, Stage1Output, etc.
3. **Backend mock**: An in-memory dict-based implementation of `VectorStoreBackend` replaces Qdrant.
4. **Isolated**: Each test function is independent with no shared state.
5. **Test file location**: Same directory as the tested code, with `_test.py` suffix (project convention).

### Test Files

| Test File | Tested Module | Priority |
|-----------|--------------|----------|
| `components/key_builder_test.py` | PlaceholderKeyBuilder | P0 |
| `components/gate_test.py` | AlwaysSearchGate + Protocol conformance | P1 |
| `components/judge_test.py` | ThresholdJudge + HitType | P0 |
| `orchestrator_test.py` | CacheOrchestrator check/write/clear | P0 |
| `cache_storage_test.py` | _check_entry_dims intersection fix | P0 |
| `interceptor_test.py` | InferenceInterceptor cache integration | P2 (requires extensive mocking) |

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
- **Purpose**: After collect(), cached_data contains the correct key
- **Input**: stage1 with state [1, 32]
- **Assert**: `kb.cached_data["state"]` is the same tensor object (zero-copy verification)
- **Assert**: `kb.cached_data["state"].shape == (1, 32)`

#### T1.2: `test_collect_stage3_stores_action_chunk`
- **Input**: stage1 (state) + stage3 (action_chunk [1, 50, 32])
- **Assert**: cached_data contains both "state" and "action_chunk" keys

#### T1.3: `test_collect_clears_previous`
- **Purpose**: Two consecutive collect() calls; the second overwrites the first
- **Input**: First call with stage1_a, second call with stage1_b
- **Assert**: cached_data["state"] is stage1_b.state

#### T1.4: `test_build_cp1_returns_normalized_cpu_float32`
- **Purpose**: build(CP1) output satisfies the tensor contract
- **Input**: state = random [1, 32] (any device)
- **Assert**:
  - Returned dict contains key `"robot_state"`
  - tensor.device == cpu
  - tensor.dtype == float32
  - tensor.is_contiguous() == True
  - tensor.shape == (32,)
  - L2 norm ~= 1.0 (torch.linalg.norm)

#### T1.5: `test_build_cp3_same_as_cp1`
- **Purpose**: CP3 currently uses the same key as CP1 (before Step 6)
- **Input**: Same as T1.4
- **Assert**: build(CP3) output has the same value as build(CP1)

#### T1.6: `test_build_unsupported_checkpoint_raises`
- **Input**: collect + build(CP2)
- **Assert**: raises ValueError

#### T1.7: `test_build_deterministic`
- **Purpose**: Same input produces the same output
- **Input**: Same state tensor, call build(CP1) twice
- **Assert**: torch.allclose(result1["robot_state"], result2["robot_state"])

#### T1.8: `test_clear_empties_cache`
- **Input**: collect → clear
- **Assert**: cached_data == {}

#### T1.9: `test_build_without_collect_raises`
- **Input**: Call build(CP1) directly without collect
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
- **Assert**: (HitType.FULL_HIT, ...) -- `>=` semantics

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

An **InMemoryBackend** implementing VectorStoreBackend is needed for testing:

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

Assembly helper:

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
- **Purpose**: check returns MISS on an empty store
- **Flow**: orchestrator.check(CP1, stage1=mock_stage1)
- **Assert**: result.hit_type == HitType.MISS, result.payload is None

#### T4.2: `test_write_then_check_exact_match_hits`
- **Purpose**: CP1 end-to-end round trip -- write then check with the same state should hit
- **Flow**:
  1. state = torch.randn(1, 32)
  2. orchestrator.write(CP1, payload, stage1=mock_stage1(state))
  3. orchestrator.clear()
  4. result = orchestrator.check(CP1, stage1=mock_stage1(state))
- **Assert**:
  - result.hit_type == HitType.FULL_HIT
  - result.payload is not None
  - result.payload.action_chunk content matches what was written
  - result.score ~= 1.0 (cosine of identical vector)

#### T4.3: `test_write_then_check_different_state_misses`
- **Purpose**: Different state does not hit
- **Flow**:
  1. write with state_a
  2. check with state_b
- **Input requirement**: Use deterministic, explicitly constructed vectors, not pure random.
  For example, `state_a = e1`, `state_b = -e1`, guaranteeing cosine = -1.
- **Assert**: result.hit_type == HitType.MISS

#### T4.4: `test_write_then_check_similar_state_near_threshold`
- **Purpose**: Verify threshold boundary
- **Flow**:
  1. write with normalized base vector
  2. construct another normalized vector with cosine explicitly greater than 0.98 (e.g., 0.99)
- **Input requirement**: Use deterministic construction, not random + small noise, to avoid flaky tests.
- **Assert**: FULL_HIT

#### T4.5: `test_judge_miss_does_not_fetch_payload`
- **Purpose**: Verify two-phase search semantics. Even if search returns a candidate, as long as Judge determines MISS, `fetch_payload()` must not be called
- **Setup**:
  - Pre-write one entry
  - Use a high-threshold Judge or construct a low-score query so that `search()` has results but `judge()` returns MISS
  - backend / storage records `fetch_payload()` call count
- **Assert**:
  - result.hit_type == HitType.MISS
  - result.payload is None
  - `fetch_payload()` call count == 0

#### T4.6: `test_gate_false_skips_search`
- **Purpose**: When Gate returns False, directly return MISS without calling storage
- **Setup**: Use a custom NeverSearchGate
- **Assert**: result.hit_type == HitType.MISS
- **Assert**: backend.search() was not called (verify with explicit counter or mock)

#### T4.7: `test_write_idempotent_upsert`
- **Purpose**: Writing the same state twice results in only 1 record in the store (same stable_hash)
- **Flow**: write the same state twice
- **Assert**: backend.count() == 1

#### T4.8: `test_write_different_states_creates_separate_entries`
- **Flow**: write state_a, write state_b
- **Assert**: backend.count() == 2

#### T4.9: `test_cp3_check_always_misses_in_step4`
- **Purpose**: CP3 has no writes, so check must always return MISS
- **Flow**: write CP1 entry, then check CP3
- **Assert**: MISS (because checkpoint_id filter or no CP3 entries)

#### T4.10: `test_should_skip_inference_returns_none`
- **Purpose**: Step 4 stub always returns None
- **Assert**: orchestrator.should_skip_inference() is None

#### T4.11: `test_schedule_next_action_is_noop`
- **Purpose**: Step 4 stub does not raise errors
- **Flow**: orchestrator.schedule_next_action(torch.randn(50, 32))
- **Assert**: no exception, should_skip_inference() still None

#### T4.12: `test_clear_resets_key_builder`
- **Flow**: orchestrator.check(CP1, ...) then orchestrator.clear()
- **Assert**: key_builder.cached_data == {}

#### T4.13: `test_check_returns_score_and_entry_id_on_hit`
- **Flow**: write + check (exact match)
- **Assert**: result.score is not None, result.entry_id is not None

#### T4.14: `test_stable_hash_deterministic`
- **Purpose**: Same checkpoint + query_keys produces the same hash
- **Import**: `from openpi.cache.orchestrator import _stable_hash`
- **Assert**: hash1 == hash2

#### T4.15: `test_stable_hash_different_checkpoints_differ`
- **Purpose**: CP1 and CP3 hashes differ (even with identical query_keys)
- **Assert**: hash_cp1 != hash_cp3

---

## Test File 5: `cache_storage_test.py`

### Test Cases (focused on _check_entry_dims fix)

#### T5.1: `test_check_entry_dims_subset_accepted`
- **Purpose**: No error when entry has only a subset of backend fields
- **Setup**: backend.vector_dims = {"robot_state": 32, "vision_0": 1024}
- **Input**: entry.query_keys = {"robot_state": tensor(32)}
- **Assert**: insert succeeds, no ValueError

#### T5.2: `test_check_entry_dims_no_overlap_raises`
- **Purpose**: Error when entry fields have no overlap with backend
- **Setup**: backend.vector_dims = {"vision_0": 1024}
- **Input**: entry.query_keys = {"robot_state": tensor(32)}
- **Assert**: raises ValueError

#### T5.3: `test_check_entry_dims_wrong_dim_raises`
- **Purpose**: Fields overlap but dimension mismatch
- **Setup**: backend.vector_dims = {"robot_state": 32}
- **Input**: entry.query_keys = {"robot_state": tensor(64)}
- **Assert**: raises ValueError

#### T5.4: `test_check_entry_dims_extra_fields_ignored`
- **Purpose**: Entry fields not declared by the backend are silently ignored
- **Setup**: backend.vector_dims = {"robot_state": 32}
- **Input**: entry.query_keys = {"robot_state": tensor(32), "unknown_field": tensor(10)}
- **Assert**: insert succeeds

---

## Test File 6: `interceptor_test.py` (P2, optional)

### Rationale

InferenceInterceptor has high coupling with Policy/model, requiring extensive mocking of internal interfaces.
Integration tests on the server can serve as an alternative; only key mock points are listed here.

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
- Verify that the normal inference path works when orchestrator=None

#### T6.2: `test_infer_with_orchestrator_cp1_miss_full_pipeline`
- Verify that on CP1 MISS, the full pipeline runs: stage1 -> stage2 -> stage3 + cp1_write + cp3_check

#### T6.3: `test_infer_with_orchestrator_cp1_hit_skips_stage2_3`
- Pre-write a CP1 entry, verify that on hit, stage2 + stage3 are skipped
- **Assert**:
  - `run_stage2()` / `run_stage3()` call count is 0
  - `orchestrator.write()` was not called
  - CP3 check path was not invoked

#### T6.4: `test_infer_cp3_consume_stub_does_not_skip`
- Verify that when should_skip_inference() returns None, inference is not skipped

#### T6.5: `test_infer_cp3_consume_with_scheduled_action_skips_all_stages`
- **Purpose**: Cover the non-empty return branch of CP3 consume, verifying shape handling and early return logic
- **Setup**:
  - fake orchestrator's `should_skip_inference()` returns a `[50, 32]` action tensor
  - fake policy/model records `run_stage1/2/3()` call counts
- **Assert**:
  - `infer()` returns directly without entering stage1 / stage2 / stage3
  - `outputs["actions"].shape == (50, 32)`
  - `outputs["state"]` is still the unbatched state
  - `orchestrator.clear()` was called

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
# Run all cache tests
uv run pytest src/openpi/cache/ -v

# Run a single file
uv run pytest src/openpi/cache/orchestrator_test.py -v

# Run a single test
uv run pytest src/openpi/cache/orchestrator_test.py::test_write_then_check_exact_match_hits -v
```

---

## Coverage Goals

| Module | Target Coverage | Notes |
|--------|----------------|-------|
| key_builder.py | 90%+ | All public methods + edge cases |
| gate.py | 100% | Minimal; 2-3 tests suffice |
| judge.py | 95%+ | Threshold boundary + empty results |
| orchestrator.py | 85%+ | check/write main paths + stubs |
| cache_storage.py | +incremental | Only test _check_entry_dims fix |
| interceptor.py | 50%+ | P2, depends on mock complexity |

---

## Implementation Order

1. **InMemoryBackend** helper (in conftest.py or orchestrator_test.py)
2. **key_builder_test.py** -- Most basic, no external dependencies
3. **judge_test.py** -- Pure logic, only needs SearchResultLite
4. **gate_test.py** -- Minimal
5. **cache_storage_test.py** -- _check_entry_dims fix verification
6. **orchestrator_test.py** -- Integrates all above components
7. **interceptor_test.py** -- P2, can be deferred

---

## Risk Notes

- **InMemoryBackend cosine similarity must be semantically consistent with Qdrant**: Both return values in the [-1, 1] range. ThresholdJudge thresholds are based on this semantics.
- **torch.compile is not in test scope**: Eager mode is sufficient to verify logical correctness.
- **GPU tensor path**: All tests use CPU tensors, but the `.cpu()` call in build() remains effective (CPU->CPU is a no-op).
- **_stable_hash uses numpy**: Ensure tensors are on CPU during testing; otherwise `.numpy()` will raise an error.
- **Threshold boundary tests must avoid randomness**: Near-threshold scenarios must use deterministic, normalized vectors to avoid flaky tests.
