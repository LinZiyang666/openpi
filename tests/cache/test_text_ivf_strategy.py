"""TextIvfKnnStrategy tests.

Covers: QuerySpec shape (hint / no task_key / score_sum fusion), the two
non-regression anchors (single-task library ON == OFF; in-library query ==
task_key-filtered brute force), trajectory-path parity inside the bucket,
score-memo compatibility, and out-of-library routing to the nearest bucket.
"""

from __future__ import annotations

import pytest
import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.search_strategy import (
    SearchContext,
    TextIvfKnnStrategy,
    WeightedScoreSumKnnStrategy,
)
from openpi.cache.config import TextIvfIndexConfig
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID

DIMS = {"prompt_emb": 4, "robot_state": 2}
PROMPTS = {
    "taskA": [1.0, 0.2, 0.0, 0.0],
    "taskB": [0.0, 0.2, 1.0, 0.0],
}

_SCORE_NORM = {
    "type": "per_field",
    "fields": {
        "robot_state": {"method": "exp_l2", "params": {"tau": 1.0}},
        "prompt_emb": {"method": "affine_clip", "params": {"lo": 0.0, "hi": 1.0}},
    },
}
_FIELD_SIM = {"robot_state": {"type": "l2"}, "prompt_emb": {"type": "cosine"}}
_WEIGHTS = {"robot_state": 1.0, "prompt_emb": 0.5}


def _populate(backend: InMemoryBackend, tasks: list[str], steps: int = 4) -> None:
    for task in tasks:
        prev_id = None
        for i in range(steps):
            eid = f"{task}:{i}"
            entry = CacheEntry(
                id=eid,
                checkpoint_id=CheckpointID.CP1,
                query_keys={
                    "prompt_emb": torch.tensor(PROMPTS[task], dtype=torch.float32),
                    "robot_state": torch.tensor([0.1 * i, float(len(task))]),
                },
                payload=CachePayload(action_chunk=torch.zeros(5, 3), task_key=task),
                step_idx=i,
                trajectory_id=task,
            )
            if prev_id is not None:
                entry.prev_ids = [prev_id]
            backend.insert(entry)
            if prev_id is not None:
                backend._entries[prev_id].next_ids = [eid]
            prev_id = eid


def _make(tasks: list[str], *, with_ivf: bool = True, **strategy_kwargs):
    backend = InMemoryBackend(
        DIMS, text_ivf=TextIvfIndexConfig(max_buckets=8) if with_ivf else None
    )
    _populate(backend, tasks)
    storage = CacheStorage(backend)
    common = dict(
        top_k=3,
        fusion_weights=_WEIGHTS,
        field_similarity=_FIELD_SIM,
        score_normalization=_SCORE_NORM,
        **strategy_kwargs,
    )
    ivf = TextIvfKnnStrategy(storage, **common) if with_ivf else None
    wss = WeightedScoreSumKnnStrategy(storage, **common)
    return backend, storage, ivf, wss


def _ctx(task: str, step: int = 1, task_key=None) -> SearchContext:
    return SearchContext(
        query_keys={
            "prompt_emb": torch.tensor(PROMPTS[task], dtype=torch.float32),
            "robot_state": torch.tensor([0.1 * step + 0.02, float(len(task))]),
        },
        checkpoint_id=CheckpointID.CP1,
        current_step=step,
        task_key=task_key,
    )


def test_requires_prompt_emb_in_query_keys():
    _, storage, ivf, _ = _make(["taskA"])
    ctx = SearchContext(
        query_keys={"robot_state": torch.tensor([0.1, 5.0])},
        checkpoint_id=CheckpointID.CP1,
    )
    with pytest.raises(ValueError, match="prompt_emb"):
        ivf.search(ctx)


def test_never_emits_task_key_filter():
    backend, storage, ivf, _ = _make(["taskA", "taskB"])
    captured = {}
    orig = storage.search

    def spy(spec):
        captured["spec"] = spec
        return orig(spec)

    storage.search = spy
    ivf.search(_ctx("taskA", task_key="taskA"))
    spec = captured["spec"]
    assert spec.backend_hints == {"text_ivf": True}
    assert spec.fusion_method == "weighted_score_sum"
    assert spec.filters is None  # step_filter=all + NO task_key even when ctx has one


def test_anchor_single_task_library_on_equals_off():
    """Anchor 1: one task = one bucket = whole library; ON == OFF value-for-value."""
    _, _, ivf, wss = _make(["taskA"])
    r_on = ivf.search(_ctx("taskA"))
    r_off = wss.search(_ctx("taskA"))
    assert [(r.id, pytest.approx(r.score)) for r in r_on] == [
        (r.id, pytest.approx(r.score)) for r in r_off
    ]


def test_anchor_in_library_query_equals_task_key_filtered_search():
    """Anchor 2: in-library instruction == task_key-filtered brute force."""
    _, _, ivf, wss = _make(["taskA", "taskB"])
    r_ivf = ivf.search(_ctx("taskA"))
    r_task = wss.search(_ctx("taskA", task_key="taskA"))
    assert [(r.id, pytest.approx(r.score)) for r in r_ivf] == [
        (r.id, pytest.approx(r.score)) for r in r_task
    ]


def test_anchor_trajectory_parity_inside_bucket():
    """Anchor 2, trajectory flavour: depth>1 chain scoring matches inside bucket."""
    traj = dict(trajectory_depth=2, trajectory_weights=[0.7, 0.3])
    _, _, ivf, wss = _make(["taskA", "taskB"], **traj)
    for strat in (ivf, wss):
        strat.on_episode_start()
    # Step 0 fills history; step 1 runs a depth-2 trajectory search.
    ivf.search(_ctx("taskA", step=0))
    wss.search(_ctx("taskA", step=0, task_key="taskA"))
    r_ivf = ivf.search(_ctx("taskA", step=1))
    r_task = wss.search(_ctx("taskA", step=1, task_key="taskA"))
    assert [(r.id, pytest.approx(r.score)) for r in r_ivf] == [
        (r.id, pytest.approx(r.score)) for r in r_task
    ]


def test_score_memo_session_compatible():
    """Two identical episodes under active sessions give identical results,
    and the trajectory search actually engages the cross-step score memo."""
    backend, storage, ivf, _ = _make(["taskA", "taskB"],
                                     trajectory_depth=2,
                                     trajectory_weights=[0.7, 0.3])

    def run_episode():
        ivf.on_episode_start()
        sid = ivf.get_search_session_id()
        backend.open_search_session(sid)
        try:
            ivf.search(_ctx("taskA", step=0))
            result = ivf.search(_ctx("taskA", step=1))
            memo_engaged = bool(backend._score_memo.get(sid))
            return result, memo_engaged
        finally:
            backend.close_search_session(sid)

    r1, memo1 = run_episode()
    r2, memo2 = run_episode()
    assert memo1 and memo2
    assert [(r.id, r.score) for r in r1] == [(r.id, r.score) for r in r2]


def test_out_of_library_query_routes_to_nearest_bucket():
    _, _, ivf, _ = _make(["taskA", "taskB"])
    ctx = SearchContext(
        query_keys={
            # Unseen instruction, closer to taskA's prompt.
            "prompt_emb": torch.tensor([0.9, 0.2, 0.1, 0.0]),
            "robot_state": torch.tensor([0.1, 5.0]),
        },
        checkpoint_id=CheckpointID.CP1,
        current_step=0,
    )
    res = ivf.search(ctx)
    assert res and all(r.id.startswith("taskA:") for r in res)


def test_step_filter_window_applies():
    _, _, ivf, _ = _make(["taskA"], step_filter="window", step_window=0)
    res = ivf.search(_ctx("taskA", step=2))
    assert {r.id for r in res} == {"taskA:2"}
