"""X15 U1/U2 — the offline scoring gate and the dump schema it checks against.

The offline pipeline exists to rebuild the retrieval scores that the live judge
saw. If the rebuild silently disagrees, every downstream feature and label is
wrong in a way no later test can detect — so the pipeline gates itself on a
parity threshold and refuses to proceed rather than warn.

The dump side is tested with synthetic fixtures here; the numeric parity on real
P0-b data is an execution-time gate, not something a unit test can stand in for.

Key dependencies: ``exp.rl_router.analysis.offline_scores``,
``openpi.cache.components.dumping_judge``.
"""

from __future__ import annotations

import json

import pytest
import torch

from exp.rl_router.analysis.offline_scores import (
    MAX_FUSED_MAE,
    MIN_TOP1_AGREEMENT,
    ParityError,
    RssBudgetExceeded,
    assert_parity,
    check_rss,
    compare_parity,
    iter_dump_rows,
    summarise_parity,
)
from openpi.cache.storage_types import StepRetrievalFeatures

# ------------------------------------------------------------------
# Parity gate
# ------------------------------------------------------------------


def _online(ids=("a", "b", "c"), scores=(0.9, 0.5, 0.2)) -> dict:
    return {"fused_topk": [[i, s] for i, s in zip(ids, scores)]}


def test_identical_scores_pass_the_gate() -> None:
    c = compare_parity(_online(), _online())
    assert c["fused_mae"] == pytest.approx(0.0)
    assert c["top1_same"] and c["top5_overlap"] == pytest.approx(1.0)


def test_the_recomputed_side_comes_from_the_library_not_the_dump(tmp_path) -> None:
    """The gate must be falsifiable.

    An earlier revision compared the dump against a field the dump carried, so
    a row that agreed with itself reported perfect parity. Here the offline side
    is produced by replaying the query through the frozen library, which is what
    makes disagreement possible at all.
    """

    scorer = _scorer(tmp_path)
    keys = {"robot_state": torch.zeros(4)}
    keys["robot_state"][0] = 1.0
    recomputed = scorer.score(keys)

    # It found the library entry that actually matches, not whatever a row said.
    assert recomputed["n_results"] > 0
    assert recomputed["fused_topk"][0][0] == "e0"

    # And a dump claiming a different winner is caught.
    lying = {"fused_topk": [["e9", 0.99], ["e8", 0.5]]}
    assert compare_parity(lying, recomputed)["top1_same"] is False


def test_a_flipped_top1_fails_the_gate() -> None:
    """Score drift that reorders the neighbours changes which chunk gets
    replayed — the one disagreement that cannot be tolerated."""
    report = summarise_parity([
        compare_parity(_online(), _online(ids=("b", "a", "c")))
        for _ in range(10)
    ])
    assert report["top1_agreement"] == 0.0
    with pytest.raises(ParityError, match="top-1 agreement"):
        assert_parity(report)


def test_small_numeric_drift_is_tolerated() -> None:
    drifted = _online(scores=(0.9 + 1e-5, 0.5 - 1e-5, 0.2))
    report = summarise_parity([compare_parity(_online(), drifted)] * 20)
    assert report["fused_mae"] < MAX_FUSED_MAE
    assert_parity(report)


def test_large_numeric_drift_is_refused() -> None:
    drifted = _online(scores=(0.9, 0.5, 0.9))
    report = summarise_parity([compare_parity(_online(), drifted)] * 20)
    with pytest.raises(ParityError, match="fused MAE"):
        assert_parity(report)


def test_no_comparable_decision_is_a_failure_not_a_pass() -> None:
    """An empty comparison set means parity is unproven; treating that as a
    pass is how a broken pipeline ships."""
    with pytest.raises(ParityError, match="parity is unproven"):
        assert_parity(summarise_parity([]))


def test_gate_thresholds_match_the_frozen_plan() -> None:
    assert MAX_FUSED_MAE == 1e-3
    assert MIN_TOP1_AGREEMENT == 0.995


# ------------------------------------------------------------------
# Memory budget
# ------------------------------------------------------------------


def test_rss_budget_raises_before_the_cgroup_killer_would() -> None:
    """ziyang10's OOM killer takes the whole pod, tether agent included, so the
    loop has to stop itself first."""
    with pytest.raises(RssBudgetExceeded, match="cgroup OOM"):
        check_rss(budget_gb=0.0, where="unit test")


def test_generous_budget_passes() -> None:
    check_rss(budget_gb=1024.0)


# ------------------------------------------------------------------
# Dump streaming and schema
# ------------------------------------------------------------------


def test_dump_rows_stream_and_skip_the_manifest(tmp_path) -> None:
    (tmp_path / "manifest.jsonl").write_text('{"not": "a decision"}\n', encoding="utf-8")
    (tmp_path / "ep1.jsonl").write_text(
        '{"decision_idx": 0}\n{"decision_idx": 1}\n', encoding="utf-8"
    )
    rows = list(iter_dump_rows(tmp_path))
    assert [r["decision_idx"] for r in rows] == [0, 1]


def test_dump_row_carries_step_features_round_trip(tmp_path) -> None:
    """U2: without this the P0-b collection has no parity baseline at all."""
    from openpi.cache.components.dumping_judge import _features_to_row

    features = StepRetrievalFeatures(
        fused_topk=(("a", 0.9), ("b", 0.5)),
        winner_per_field={"vision_0": 0.8},
        field_own_margin={"vision_0": 0.3},
        fused_margin=0.4,
        n_results=2,
    )
    row = _features_to_row(features)
    # Must survive strict JSON, which is what the offline reader uses.
    restored = json.loads(json.dumps(row))

    assert restored["fused_topk"] == [["a", 0.9], ["b", 0.5]]
    assert restored["winner_per_field"]["vision_0"] == pytest.approx(0.8)
    assert restored["n_results"] == 2
    assert compare_parity(restored, restored)["top1_same"]


def test_absent_features_stay_absent_in_a_legacy_dump() -> None:
    """A legacy judge's dump gains the key as null, not as fabricated content."""
    from openpi.cache.components.dumping_judge import _features_to_row

    assert _features_to_row(None) is None


# ------------------------------------------------------------------
# Real recomputation (U1)
# ------------------------------------------------------------------


def _library(tmp_path, n: int = 6) -> str:
    """A tiny frozen library in the on-disk artifact shape."""
    import pickle

    from openpi.cache.storage_types import CacheEntry, CachePayload
    from openpi.cache.types import CheckpointID

    entries = []
    for i in range(n):
        # Distinct vectors with a strict similarity order against the query
        # [1,0,0,0]: e0 aligns exactly, later entries tilt further away, so the
        # expected winner is unambiguous (equal cosines would make topk's
        # tie-break, not the scoring, decide the test).
        vec = torch.zeros(4)
        vec[0] = 1.0 - 0.1 * i
        vec[1 + (i % 3)] = 0.1 * i
        entries.append(CacheEntry(
            id=f"e{i}",
            checkpoint_id=CheckpointID.CP1,
            query_keys={"robot_state": vec},
            payload=CachePayload(action_chunk=torch.zeros(50, 32)),
            step_idx=i,
        ))
    path = tmp_path / "lib.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"entries": entries, "vector_dims": {"robot_state": 4}}, fh)
    return str(path)


def _arm_yaml(tmp_path) -> str:
    import yaml as _yaml

    cfg = {
        "keys": {"robot_state": {"enabled": True, "weight": 1.0}},
        "checkpoints": {"cp1": {"search_strategy": {
            "top_k": 5,
            "field_similarity": {"robot_state": {"type": "cosine"}},
        }}},
    }
    path = tmp_path / "arm.yaml"
    path.write_text(_yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def _scorer(tmp_path):
    from exp.rl_router.analysis.offline_scores import OfflineScorer

    return OfflineScorer(_library(tmp_path), _arm_yaml(tmp_path))


def test_scorer_loads_the_library_it_was_given(tmp_path) -> None:
    """`--library` must actually be opened; the earlier revision ignored it."""
    assert _scorer(tmp_path).n_entries == 6


def test_a_missing_library_fails_immediately(tmp_path) -> None:
    from exp.rl_router.analysis.offline_scores import OfflineScorer

    with pytest.raises((FileNotFoundError, OSError)):
        OfflineScorer(str(tmp_path / "nope.pkl"), _arm_yaml(tmp_path))


def test_output_is_not_published_when_the_gate_fails(tmp_path) -> None:
    """A failed run must leave no file that downstream code could mistake for
    validated training data."""
    from exp.rl_router.analysis.offline_scores import ParityError, run

    dump = tmp_path / "dump"
    dump.mkdir()
    # Query matches e0, but the row claims a different winner -> gate fails.
    (dump / "ep.jsonl").write_text(json.dumps({
        "decision_idx": 0,
        "query_keys": {"robot_state": [1.0, 0.0, 0.0, 0.0]},
        "step_features": {"fused_topk": [["e3", 0.99], ["e2", 0.4]]},
    }) + "\n", encoding="utf-8")

    out = tmp_path / "feats.jsonl"
    with pytest.raises(ParityError):
        run(str(dump), _scorer(tmp_path), str(out))

    assert not out.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_output_is_published_when_the_gate_passes(tmp_path) -> None:
    from exp.rl_router.analysis.offline_scores import run

    scorer = _scorer(tmp_path)
    keys = {"robot_state": torch.tensor([1.0, 0.0, 0.0, 0.0])}
    truth = scorer.score(keys)

    dump = tmp_path / "dump"
    dump.mkdir()
    (dump / "ep.jsonl").write_text(json.dumps({
        "decision_idx": 0,
        "query_keys": {"robot_state": [1.0, 0.0, 0.0, 0.0]},
        "step_features": truth,
    }) + "\n", encoding="utf-8")

    out = tmp_path / "feats.jsonl"
    report = run(str(dump), scorer, str(out))

    assert out.exists() and report["n_compared"] == 1
    row = json.loads(out.read_text().strip())
    assert row["offline_features"]["fused_topk"][0][0] == "e0"


# ------------------------------------------------------------------
# Legacy dump schema (U2)
# ------------------------------------------------------------------


def test_legacy_dump_row_does_not_gain_a_null_column() -> None:
    """A legacy judge's dump must keep its exact key set: adding
    ``step_features: null`` to every historical row changes the schema for
    configs that have nothing to do with X15."""
    from openpi.cache.components.dumping_judge import DumpingJudge

    class _Legacy:
        def __call__(self, results, checkpoint_id, cached_data, *,
                     view=None, history=None, retrieval_signals=None):
            from openpi.cache.components.judge import HitType, JudgeResult
            return JudgeResult(hit_type=HitType.MISS)

    written: list[dict] = []
    wrapper = DumpingJudge(
        inner=_Legacy(), dump_normalization=None, dump_factors=[],
        dump_path="", config_id="legacy_test",
    )
    object.__setattr__(wrapper, "_append_jsonl", written.append)

    from openpi.cache.types import CheckpointID
    wrapper([], CheckpointID.CP1, {})

    assert written, "the wrapper must still dump"
    assert "step_features" not in written[0]


def test_x15_dump_row_carries_the_diagnostics() -> None:
    from openpi.cache.components.dumping_judge import DumpingJudge

    class _Inner:
        def __call__(self, results, checkpoint_id, cached_data, *,
                     view=None, history=None, retrieval_signals=None,
                     step_features=None):
            from openpi.cache.components.judge import HitType, JudgeResult
            return JudgeResult(hit_type=HitType.MISS)

    written: list[dict] = []
    wrapper = DumpingJudge(
        inner=_Inner(), dump_normalization=None, dump_factors=[],
        dump_path="", config_id="x15_test",
    )
    object.__setattr__(wrapper, "_append_jsonl", written.append)

    from openpi.cache.types import CheckpointID
    wrapper([], CheckpointID.CP1, {},
            step_features=StepRetrievalFeatures(n_results=3, fused_margin=0.2))

    assert written[0]["step_features"]["n_results"] == 3
    # Strict-JSON round trip, which is what the offline reader does.
    assert json.loads(json.dumps(written[0]))["step_features"]["fused_margin"] == 0.2


# ------------------------------------------------------------------
# Production round trip: real writer -> real artifact -> real replay
# ------------------------------------------------------------------


def _numpy_artifact(tmp_path, n: int = 6) -> str:
    """A library in the shape the REAL builder writes: NumPy, not tensors.

    ``_detach_entries()`` stores query keys and action chunks as NumPy arrays;
    only ``load_artifact()`` converts them back. Fixtures that store tensors
    directly cannot catch a loader that skips that conversion.
    """
    import pickle

    import numpy as np

    from openpi.cache.storage_types import CacheEntry, CachePayload
    from openpi.cache.types import CheckpointID

    entries = []
    for i in range(n):
        vec = np.zeros(4, dtype=np.float32)
        vec[0] = 1.0 - 0.1 * i
        vec[1 + (i % 3)] = 0.1 * i
        entries.append(CacheEntry(
            id=f"e{i}",
            checkpoint_id=CheckpointID.CP1,
            query_keys={"robot_state": vec},                  # NumPy on purpose
            payload=CachePayload(action_chunk=np.zeros((50, 32), dtype=np.float32)),
            step_idx=i,
        ))
    path = tmp_path / "lib_numpy.pkl"
    with open(path, "wb") as fh:
        pickle.dump({
            "key_builder_type": "placeholder",
            "checkpoint_id": "CP1",
            "vector_dims": {"robot_state": 4},
            "entries": entries,
        }, fh)
    return str(path)


def test_scorer_loads_a_real_numpy_artifact(tmp_path) -> None:
    """The production artifact shape must work, not just tensor fixtures."""
    from exp.rl_router.analysis.offline_scores import OfflineScorer

    scorer = OfflineScorer(_numpy_artifact(tmp_path), _arm_yaml(tmp_path))
    assert scorer.n_entries == 6

    # And it can actually score — the NumPy->Tensor conversion happened.
    out = scorer.score({"robot_state": torch.tensor([1.0, 0.0, 0.0, 0.0])})
    assert out["n_results"] > 0
    assert out["fused_topk"][0][0] == "e0"


def test_dump_written_by_the_real_wrapper_replays_end_to_end(tmp_path) -> None:
    """The closed loop the parity gate depends on.

    Earlier tests hand-built the JSONL, which is exactly how a writer that
    never persists ``query_keys`` goes unnoticed: the reader was fed rows the
    writer could not actually produce. Here the row comes from ``DumpingJudge``
    itself and is replayed by ``OfflineScorer``.
    """
    from openpi.cache.components.dumping_judge import DumpingJudge
    from openpi.cache.components.judge import HitType, JudgeResult
    from openpi.cache.types import CheckpointID
    from exp.rl_router.analysis.offline_scores import OfflineScorer, run

    scorer = OfflineScorer(_numpy_artifact(tmp_path), _arm_yaml(tmp_path))
    query = {"robot_state": torch.tensor([1.0, 0.0, 0.0, 0.0])}
    truth = scorer.score(query)

    class _Inner:
        def __call__(self, results, checkpoint_id, cached_data, *,
                     view=None, history=None, retrieval_signals=None,
                     step_features=None):
            return JudgeResult(hit_type=HitType.MISS)

    dump_dir = tmp_path / "dump"
    dump_dir.mkdir()
    wrapper = DumpingJudge(
        inner=_Inner(), dump_normalization=None, dump_factors=[],
        dump_path=str(dump_dir / "ep.jsonl"), config_id="x15_roundtrip",
    )
    wrapper(
        [], CheckpointID.CP1, {},
        step_features=StepRetrievalFeatures(
            fused_topk=tuple((i, s) for i, s in truth["fused_topk"]),
            winner_per_field=truth["winner_per_field"],
            field_own_margin=truth["field_own_margin"],
            fused_margin=truth["fused_margin"],
            n_results=truth["n_results"],
        ),
        query_keys=query,
    )

    # The writer must have persisted what the reader needs.
    written = json.loads((dump_dir / "ep.jsonl").read_text().strip())
    assert "query_keys" in written, "the real writer must persist query_keys"

    out = tmp_path / "feats.jsonl"
    report = run(str(dump_dir), scorer, str(out))
    assert report["n_compared"] == 1
    assert out.exists()


def test_legacy_wrapper_row_persists_no_query_keys(tmp_path) -> None:
    """Only X15 rows gain the field; a legacy dump's key set is unchanged."""
    from openpi.cache.components.dumping_judge import DumpingJudge
    from openpi.cache.components.judge import HitType, JudgeResult
    from openpi.cache.types import CheckpointID

    class _Legacy:
        def __call__(self, results, checkpoint_id, cached_data, *,
                     view=None, history=None, retrieval_signals=None):
            return JudgeResult(hit_type=HitType.MISS)

    path = tmp_path / "legacy.jsonl"
    wrapper = DumpingJudge(
        inner=_Legacy(), dump_normalization=None, dump_factors=[],
        dump_path=str(path), config_id="legacy",
    )
    wrapper([], CheckpointID.CP1, {}, query_keys={"robot_state": torch.zeros(4)})

    row = json.loads(path.read_text().strip())
    assert "query_keys" not in row and "step_features" not in row
