"""Real-pkl artifact parity for trajectory rewrite (plan §7.6).

Marked `@pytest.mark.manual`: requires the pre-built artifacts under
`exp/common/data/cache_artifacts/{libero_10,libero_spatial}` which are
not part of the test suite and not produced by CI. Run locally before
shipping the cache rewrite:

    uv run pytest tests/cache/test_in_memory_backend_pkl_parity.py -m manual

For each enabled artifact, the test:
  1. loads it into an `InMemoryBackend`,
  2. constructs a depth-3 trajectory query from the loaded entries,
  3. runs `search()` through both the new single-chain main path AND
     `force_legacy_path()`,
  4. asserts ids/score parity per plan §10 #2 (atol=1e-6, ties order
     not strictly preserved).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.storage_types import QuerySpec
from openpi.cache.types import CheckpointID


# ---------------------------------------------------------------------------
# Artifact discovery — limited to the two datasets named in plan §10 #8.
# ---------------------------------------------------------------------------


_ARTIFACT_ROOT = Path("exp/common/data/cache_artifacts")
_DATASETS = ("libero_spatial", "libero_10")
_PKL_NAMES = ("cp1_mean_pool.pkl",)  # one stable artifact per dataset is enough


def _candidates() -> list[tuple[str, Path]]:
    out = []
    for d in _DATASETS:
        for f in _PKL_NAMES:
            p = _ARTIFACT_ROOT / d / f
            if p.is_file():
                out.append((f"{d}/{f}", p))
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _peek_dims(path: Path) -> dict[str, int]:
    import pickle

    with open(path, "rb") as f:
        data = pickle.load(f)
    return dict(data.get("vector_dims") or {})


def _build_history(entries, field: str, *, depth: int = 3) -> tuple[list[dict[str, torch.Tensor]], list[float]]:
    sample = entries[0]
    vec = sample.query_keys[field].float()
    history = [{field: vec} for _ in range(depth)]
    weights = [0.5, 0.3, 0.2][:depth]
    return history, weights


# ---------------------------------------------------------------------------
# Parametrised parity test — one row per discovered artifact
# ---------------------------------------------------------------------------


@pytest.mark.manual
@pytest.mark.parametrize("label, path", _candidates() or [pytest.param(
    "no-artifacts", None,
    marks=pytest.mark.skip(reason="cache artifacts not staged on this machine"),
)])
def test_pkl_parity_new_vs_legacy(label, path):
    """Real artifact: new path (cache off via fresh sid) vs legacy path."""
    if path is None:
        pytest.skip(f"no artifact at {label}")

    dims = _peek_dims(path)
    backend = InMemoryBackend(dims)
    backend.load_artifact(str(path))

    entries = list(backend._entries.values())
    if len(entries) < 5:
        pytest.skip(f"{label}: too few entries ({len(entries)}) for trajectory parity")

    # Pick the most-populated field so trajectory history aligns with
    # the dominant signal in the artifact.
    field_counts: dict[str, int] = {}
    for e in entries:
        for k in e.query_keys:
            field_counts[k] = field_counts.get(k, 0) + 1
    field = max(field_counts, key=field_counts.get)
    query_vec = entries[0].query_keys[field].float()

    history, weights = _build_history(entries, field, depth=3)

    spec_new = QuerySpec(
        query_keys={field: query_vec},
        top_k=10,
        checkpoint_id=CheckpointID.CP1,
        trajectory_history=history,
        trajectory_weights=weights,
        # search_session_id None → same code path, just no cache.
    )
    new_results = backend.search(spec_new)

    # Legacy path through the test-only context manager.
    with backend.force_legacy_path():
        legacy_results = backend.search(spec_new)

    new_ids = {r.id for r in new_results}
    legacy_ids = {r.id for r in legacy_results}
    assert new_ids == legacy_ids, (
        f"{label}: id sets differ. "
        f"new={sorted(new_ids - legacy_ids)[:5]} "
        f"legacy={sorted(legacy_ids - new_ids)[:5]}"
    )

    # Per-id score equivalence (atol=1e-6).
    legacy_score = {r.id: r.score for r in legacy_results}
    for r in new_results:
        assert abs(r.score - legacy_score[r.id]) < 1e-6, (
            f"{label}: score drift for {r.id}: "
            f"new={r.score} legacy={legacy_score[r.id]}"
        )

    # Top scorer must agree unless there is a tie at rank 0.
    if len(new_results) >= 2 and abs(new_results[0].score - new_results[1].score) > 1e-9:
        assert new_results[0].id == legacy_results[0].id, (
            f"{label}: top scorer disagreement (no tie). "
            f"new={new_results[0]} legacy={legacy_results[0]}"
        )
