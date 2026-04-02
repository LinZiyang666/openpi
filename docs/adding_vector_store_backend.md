# Adding a New Vector Store Backend

> Applies to: Step 3 storage layer and later.
> Design reference: `claude_log/step3_cache.log`

This guide walks through adding a new vector database (e.g. FAISS, Milvus, a
custom GPU store) to the cache system.  The process requires creating one new
file and zero modifications to existing code.

---

## How the layers fit together

```
CacheOrchestrator
      ↓
CacheStorage          ← your only caller; handles locking, validation, filtering
      ↓
VectorStoreBackend    ← the ABC you must implement
      ↓
Your new backend
```

`CacheStorage` takes care of thread safety, dimension validation, and filter
capability checks before every backend call.  You do not need to repeat any of
that logic inside your backend.

---

## Step 1 — Understand the interface

Open `src/openpi/cache/backend_base.py`.  The methods you **must** implement:

```python
class VectorStoreBackend(ABC):

    @property
    @abstractmethod
    def vector_dim(self) -> int: ...          # embedding dimension this backend expects

    @abstractmethod
    def supported_filters(self) -> frozenset[str]: ...  # which QueryFilter fields you handle

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
```

The following methods have working defaults and are **optional** to override:

| Method | Default behaviour | Why override |
|--------|-------------------|--------------|
| `batch_insert()` | calls `insert()` in a loop; partial failures are tolerated | use a native bulk API for higher throughput |
| `flush()` | no-op | in-memory backends persist data here |
| `close()` | calls `flush()` | release connections or file handles |

---

## Step 2 — Understand the data types

All types live in `src/openpi/cache/storage_types.py`.

### `CacheEntry` — what you receive on `insert()`

```python
@dataclass
class CacheEntry:
    id: str                  # stable hash — same context + same CP always maps to the same id
    checkpoint_id: CheckpointID
    query_key: torch.Tensor  # [dim] CPU float32, L2-normalised — the vector you store
    payload: CachePayload
    timestamp: float
```

### `CachePayload` — the data you must serialise alongside the vector

```python
@dataclass
class CachePayload:
    action_chunk: torch.Tensor                          # [50, 32] CPU float32, always present
    intermediates: Optional[dict[float, torch.Tensor]]  # CP2 warm-start snapshots
    denoising_num_steps: Optional[int]                  # required when intermediates is set
    next_action_chunk: Optional[torch.Tensor]           # [50, 32] CPU float32, CP3 only
    task_key: str                                       # canonical task identifier
```

Serialising tensors into your database's storage format is your responsibility.
See the Qdrant backend for a reference implementation (`torch.save` → base64 string).

### `SearchResultLite` — what `search()` must return

```python
@dataclass
class SearchResultLite:
    id: str
    score: float          # normalised cosine similarity ∈ [-1, 1]; higher = more similar
    checkpoint_id: CheckpointID
```

**The `score` contract is mandatory.**  If your database returns a distance
(lower = closer), convert it before returning.  `SimilarityJudge` thresholds
are calibrated against this scale and must work across all backends.

| What your DB returns | Conversion |
|----------------------|------------|
| Cosine similarity (already normalised) | use directly — Qdrant does this |
| Cosine distance (= 1 − similarity) | `score = 1.0 - distance` |
| L2 distance (vectors are L2-normalised) | `score = 1.0 - distance² / 2` |
| Inner product (vectors are L2-normalised) | equivalent to cosine similarity, use directly |

### `QuerySpec` — what you receive on `search()`

```python
@dataclass
class QuerySpec:
    query_key: torch.Tensor
    top_k: int = 10
    checkpoint_id: Optional[CheckpointID] = None
    filters: Optional[QueryFilter] = None

@dataclass
class QueryFilter:
    task_key: Optional[str] = None
    step_range: Optional[tuple[int, int]] = None
```

You only need to apply the filter fields you declared in `supported_filters()`.
Fields you did not declare will never reach your backend — `CacheStorage` raises
`UnsupportedFilterError` before calling you.

---

## Step 3 — Create the backend file

Create a new file under `src/openpi/cache/backends/`, e.g. `faiss_backend.py`.

### Minimal implementation skeleton

```python
# src/openpi/cache/backends/faiss_backend.py
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from openpi.cache.backend_base import VectorStoreBackend
from openpi.cache.storage_types import (
    BatchInsertResult, CacheEntry, CachePayload, QuerySpec, SearchResultLite,
)
from openpi.cache.types import CheckpointID

logger = logging.getLogger(__name__)


@dataclass
class FaissBackendConfig:
    index_path: str
    vector_dim: int = 1024
    # add any backend-specific parameters here


class FaissVectorStore(VectorStoreBackend):

    def __init__(self, config: FaissBackendConfig) -> None:
        self._config = config
        # initialise your DB connection / load index here
        raise NotImplementedError("TODO: initialise FAISS index")

    # --- metadata ---

    @property
    def vector_dim(self) -> int:
        return self._config.vector_dim

    def supported_filters(self) -> frozenset[str]:
        return frozenset()   # declare what you actually support; see Step 4.4

    # --- CRUD ---

    def insert(self, entry: CacheEntry) -> None:
        raise NotImplementedError

    def search(self, spec: QuerySpec) -> list[SearchResultLite]:
        raise NotImplementedError

    def fetch_payload(self, id: str) -> CachePayload:
        raise NotImplementedError

    def delete(self, ids: list[str]) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError
```

---

## Step 4 — Implement each method

### 4.1 `insert()`

```python
def insert(self, entry: CacheEntry) -> None:
    vec = entry.query_key.float().cpu().numpy()   # [dim] numpy float32

    # 1. add the vector to the index (use entry.id mapped to an integer if needed)
    int_id = self._str_to_int(entry.id)
    self._index.add_with_ids(vec.reshape(1, -1), [int_id])

    # 2. store the serialised payload alongside (dict / SQLite / file, etc.)
    self._payloads[entry.id] = self._serialize(entry)
```

`entry.id` is a string hash.  If your database requires integer IDs, maintain a
stable `str → int` mapping: `int_id = abs(hash(str_id)) % (2**63)` works when
the hash domain is small, or derive one with `hashlib.sha256`.

### 4.2 `search()`

```python
def search(self, spec: QuerySpec) -> list[SearchResultLite]:
    q = spec.query_key.float().cpu().numpy().reshape(1, -1)
    distances, int_ids = self._index.search(q, spec.top_k)

    results = []
    for dist, int_id in zip(distances[0], int_ids[0]):
        if int_id == -1:          # FAISS uses -1 when fewer than top_k results exist
            continue
        str_id = self._int_to_str[int_id]
        score = self._dist_to_score(dist)    # apply your conversion from the table above
        cp = self._id_to_checkpoint[str_id]
        results.append(SearchResultLite(id=str_id, score=score, checkpoint_id=cp))
    return results
```

**Do not return payload tensors here.**  `search()` only returns `SearchResultLite`.
The full payload is fetched separately via `fetch_payload()` for the winner only.

**Result count contract:** return *at most* `spec.top_k` results.  Returning
fewer (e.g. after client-side filtering) is acceptable — callers handle it.

### 4.3 `fetch_payload()`

```python
def fetch_payload(self, id: str) -> CachePayload:
    if id not in self._payloads:
        raise KeyError(f"id {id!r} not found")
    return self._deserialize(self._payloads[id])
```

This is called only for the winning candidate after the judge's score threshold
check, so it will not be called for every search result.

### 4.4 `supported_filters()`

Declare only the filter fields your backend actually applies.  A field you list
here but silently ignore would change cache semantics (searching the full DB
instead of a subset, causing incorrect cache hits).

```python
def supported_filters(self) -> frozenset[str]:
    return frozenset({"checkpoint_id"})   # list what you genuinely handle
```

If you implement client-side filtering (over-fetch then filter), you may still
list the field — but you must filter correctly in `search()`:

```python
def search(self, spec: QuerySpec) -> list[SearchResultLite]:
    # over-fetch to compensate for post-filter drop
    raw = self._raw_search(spec.query_key, spec.top_k * 5)
    filtered = [r for r in raw if self._passes_filter(r, spec.filters)]
    return filtered[:spec.top_k]
```

---

## Step 5 — Write the config dataclass

```python
@dataclass
class FaissBackendConfig:
    index_path: str
    vector_dim: int = 1024
    use_gpu: bool = False
    nprobe: int = 16          # any backend-specific tuning parameters
```

All backend-specific parameters belong in the config.  They must not appear in
`QuerySpec` or `VectorStoreBackend`; doing so would expose implementation
details to the orchestration layer.

---

## Step 6 — Wire it into CacheStorage

```python
from openpi.cache.backends.faiss_backend import FaissVectorStore, FaissBackendConfig
from openpi.cache import CacheStorage

backend = FaissVectorStore(FaissBackendConfig(index_path="cache.faiss", vector_dim=1024))
storage = CacheStorage(backend)

# pass storage to CacheOrchestrator — no other changes needed
orchestrator = CacheOrchestrator(storage=storage, ...)
```

`CacheOrchestrator` and `SimilarityJudge` require no modification.

---

## Step 7 — Write tests

Create `tests/cache/test_faiss_backend.py` (or equivalent).  Minimum coverage:

```python
def test_insert_and_search():
    """Insert one entry; searching with the same query_key should return it as top-1."""

def test_fetch_payload():
    """After insert, fetch_payload returns the original tensors unchanged."""

def test_idempotent_insert():
    """Inserting the same id twice must not increase count()."""

def test_delete():
    """After delete, count() decreases and fetch_payload raises KeyError."""

def test_score_range():
    """All scores returned by search() are in [-1, 1]."""

def test_self_query_score():
    """Querying with an entry's own query_key should yield score ≈ 1.0."""

def test_unsupported_filter_raises():
    """Passing an unsupported filter field raises UnsupportedFilterError at the
    CacheStorage level before the backend is called."""
```

---

## FAQ

**My database has no `get_by_id` — how do I implement `fetch_payload()`?**

Maintain a local `dict[str, bytes]` alongside the vector index.  Serialise the
payload to bytes on `insert()` and deserialise on `fetch_payload()`.

**My database does not support vector deletion.**

Mark deletions locally (`deleted_ids: set[str]`), filter them out in `search()`,
and subtract the count in `count()`.

**My database only accepts integer IDs but `CacheEntry.id` is a string.**

Keep a bidirectional mapping.  On insert: `int_id = stable_hash(str_id)`,
store `int_id → str_id`.  On search: look up the int_id returned by the DB to
recover the original string id.

**How do I verify my score conversion is correct?**

Insert one entry, then search using its own `query_key`.  The top-1 score must
be close to `1.0`.  If it is not, the conversion formula is wrong.

**When should I split the backend into multiple files?**

When a single file exceeds ~300 lines, consider extracting `_serialisation.py`
and `_filter.py` as siblings.  For most backends a single file is sufficient.
