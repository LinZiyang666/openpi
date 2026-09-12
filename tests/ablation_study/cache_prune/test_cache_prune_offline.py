"""Independent boundary probes for pruning, grid selection and physical exports.

Synthetic sources use real CacheEntry objects, HDF5 provenance, production
retrieval and backend serialization. Tiny exhaustive references verify the
greedy rule and rate matching independently of their implementations.
"""

from __future__ import annotations

import copy
from itertools import combinations
import json
from pathlib import Path
import pickle
import shutil

import h5py
import numpy as np
import pytest
import torch

from openpi.cache.components.factors.base import LibraryStats
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID

from exp.ablation_study.cache_prune.common import (
    file_identity,
    seal,
    template,
    validate_config,
)
from exp.ablation_study.cache_prune.prune_library import (
    audit_source,
    compute_task_scores,
    export_pruned,
    load_raw,
    load_score_blocks,
    select_retained,
    source_rows,
)
from exp.ablation_study.cache_prune.select_prune_grid import (
    candidate_thresholds,
    grid_spec,
    match_targets,
    select_grid,
)
from exp.ablation_study.cache_prune.verify_prune import value_digest, verify_pruned


@pytest.fixture
def tiny_source(tmp_path):
    """Build two tasks with successful multi-step trajectories and real provenance."""
    torch.set_num_threads(1)
    entries, records = [], []
    rng = np.random.default_rng(11)
    for task in range(2):
        for trajectory in range(3):
            tid = f"task{task}_traj{trajectory}"
            h5_path = tmp_path / f"{tid}.h5"
            with h5py.File(h5_path, "w") as handle:
                handle.attrs.update(success=True, task=f"task {task}", num_steps=4)
            records.append(
                {
                    "trajectory_id": tid,
                    "h5_path": str(h5_path),
                    "suite": "libero_spatial",
                    "prompt": f"task {task}",
                    "entry_count": 4,
                }
            )
            for step in range(4):
                vectors = {
                    "vision_0": rng.normal(size=3).astype("float32"),
                    "vision_1": rng.normal(size=3).astype("float32"),
                    "robot_state": rng.normal(size=2).astype("float32"),
                    "prompt_emb": np.array([123.0], dtype=np.float32),
                }
                payload = CachePayload(
                    rng.normal(size=(5, 2)).astype("float32"), task_key=f"task {task}"
                )
                entry = CacheEntry(
                    f"{tid}:{step}",
                    CheckpointID.CP1,
                    vectors,
                    payload,
                    step_idx=step,
                    trajectory_id=tid,
                    prev_ids=[] if step == 0 else [f"{tid}:{step - 1}"],
                    next_ids=[] if step == 3 else [f"{tid}:{step + 1}"],
                )
                # Legacy pickles may lack these attributes entirely.
                del entry.__dict__["outcome"]
                del payload.__dict__["schedule_id"]
                entries.append(entry)
    cfg = template("libero_spatial")
    cfg["backend"]["vector_dims"] = {
        "vision_0": 3,
        "vision_1": 3,
        "robot_state": 2,
        "prompt_emb": 1,
    }
    for field in ("vision_0", "vision_1", "robot_state"):
        cfg["checkpoints"]["cp1"]["search_strategy"]["score_normalization"]["fields"][
            field
        ]["params"].update(mu=0.0, sigma=1.0)
    path = tmp_path / "source.pkl"
    artifact = {
        "entries": entries,
        "vector_dims": cfg["backend"]["vector_dims"],
        "key_builder_type": "cp1_spatial_pool_16",
        "checkpoint_id": "cp1",
        "library_stats": LibraryStats.compute_from_entries(entries),
        "unknown_future_field": {
            "values": np.array([1, 2], dtype=np.int16),
            "labels": ("a", "b"),
        },
    }
    with path.open("wb") as handle:
        pickle.dump(artifact, handle)
    init_map = tmp_path / "init_map.json"
    init_map.write_text(json.dumps(records))
    expected = {
        "suite": "libero_spatial",
        "regime": "rit50",
        "entries": 24,
        "trajectories": 6,
        "tasks": {"0": "task 0", "1": "task 1"},
        "kind": "hdf5",
        "repository": str(tmp_path),
        "init_map": str(init_map),
    }
    return path, expected, cfg, entries


def _brute_prune(rows, blocks, threshold):
    keep, witness = set(), {}
    for task, (ordinals, scores) in blocks.items():
        positions = {ordinal: i for i, ordinal in enumerate(ordinals)}
        for i in sorted(ordinals, key=lambda i: (rows[i]["remaining"], i)):
            eligible = []
            for j in ordinals:
                if (
                    j in keep
                    and rows[j]["trajectory_id"] != rows[i]["trajectory_id"]
                    and rows[j]["remaining"] < rows[i]["remaining"]
                    and scores[positions[i], positions[j]] >= threshold
                ):
                    eligible.append(j)
            if not eligible:
                keep.add(i)
            else:
                witness[i] = sorted(
                    eligible,
                    key=lambda j: (
                        -scores[positions[i], positions[j]],
                        rows[j]["remaining"],
                        j,
                    ),
                )[0]
    return [rows[i]["id"] for i in sorted(keep)], {
        rows[i]["id"]: rows[j]["id"] for i, j in witness.items()
    }


def test_empty_task_grid_failure_preserves_candidate_report(
    tiny_source, tmp_path, monkeypatch
):
    """Even an otherwise unreachable empty-task condition yields a completed failure report."""
    from exp.ablation_study.cache_prune import select_prune_grid as grid_module

    path, expected, cfg, entries = tiny_source
    source = audit_source(path, expected)
    scores = compute_task_scores(
        entries, cfg, tmp_path / "scores", source_manifest=source
    )
    monkeypatch.setattr(
        grid_module, "match_targets", lambda *args: [{"task_counts": {"task 0": 0}}]
    )
    report = select_grid(source, scores, grid_spec())
    assert report["status"] == "grid_not_representable" and report["complete"]
    assert report["candidates"] and report["seconds"] >= 0
    assert report["reason"] == "empty task after pruning"


@pytest.mark.parametrize(
    "mutation", [None, "task", "duplicate", "missing_h5", "failed_h5"]
)
def test_compact_map_proves_success_without_inventing_init_identity(
    tiny_source, mutation
):
    """Accept exact H5 provenance but reject missing, failed or inconsistent inputs."""
    path, expected, _, _ = tiny_source
    root = Path(expected["repository"])
    directory = root / "exp/common/data/db/libero_cache/libero_spatial"
    directory.mkdir(parents=True)
    records = json.loads(Path(expected["init_map"]).read_text())
    compact = []
    for ordinal, record in enumerate(records):
        target = directory / Path(record["h5_path"]).name
        shutil.copyfile(record["h5_path"], target)
        with h5py.File(target, "r+") as handle:
            handle.attrs["experiment_name"] = expected["suite"]
            # Repeated IDs deliberately cannot identify an init.
            handle.attrs["episode_id"] = 0
        task, index = divmod(ordinal, 3)
        compact.append(
            {
                "task_id": task,
                "task_name": f"task_{task}",
                "prompt": f"task {task}",
                "subset_idx": index,
                "orig_init_state_idx": 10 + index,
            }
        )
    if mutation == "task":
        compact[0]["prompt"] = "wrong task"
    elif mutation == "duplicate":
        compact[0]["subset_idx"] = 1
    elif mutation == "missing_h5":
        target.unlink()
    elif mutation == "failed_h5":
        with h5py.File(target, "r+") as handle:
            handle.attrs["success"] = False
    Path(expected["init_map"]).write_text(json.dumps(compact))
    if mutation:
        with pytest.raises(ValueError):
            audit_source(path, expected)
    else:
        result = audit_source(path, expected)
        assert result["trajectories"] == 6
        assert len(result["success_provenance"]) == 7
        assert all("orig_init_state_idx" not in row for row in result["rows"])


def test_prune_matches_exhaustive_direct_witness_oracle(tiny_source):
    """Check prune matches exhaustive direct witness oracle."""
    _, _, _, entries = tiny_source
    rows = source_rows(entries)
    rng = np.random.default_rng(6)
    for _ in range(20):
        blocks = {
            f"task {task}": (
                list(range(task * 12, (task + 1) * 12)),
                rng.integers(0, 4, (12, 12)).astype(np.float32) / 3,
            )
            for task in range(2)
        }
        for threshold in (0.0, float(np.float32(1 / 3)), 1.0, 1.1):
            selection = select_retained(rows, blocks, threshold)
            expected, witnesses = _brute_prune(rows, blocks, np.float32(threshold))
            assert selection.kept_ids == expected
            assert {
                w["removed_id"]: w["retained_id"] for w in selection.witnesses
            } == witnesses
            assert all(r["id"] in expected for r in rows if r["remaining"] == 0)
    assert select_retained(rows, {}, None).kept_ids == [r["id"] for r in rows]


def test_equal_remaining_same_trajectory_and_deleted_witness(tiny_source):
    """Check equal remaining same trajectory and deleted witness."""
    _, _, _, entries = tiny_source
    rows = source_rows(entries)
    blocks = {
        f"task {task}": (
            list(range(task * 12, (task + 1) * 12)),
            np.zeros((12, 12), dtype=np.float32),
        )
        for task in range(2)
    }
    matrix = blocks["task 0"][1]
    matrix[2, 3] = 1  # Same trajectory is ineligible.
    matrix[2, 6] = 1  # Equal remaining is ineligible.
    matrix[6, 3] = 1  # Delete B2 via the retained A3 terminal.
    matrix[9, 6] = 1  # Deleted B2 must not cover C1 transitively.
    result = select_retained(rows, blocks, 1.0)
    assert rows[2]["id"] in result.kept_ids
    assert rows[6]["id"] not in result.kept_ids
    assert rows[9]["id"] in result.kept_ids


@pytest.mark.parametrize(
    "mutation",
    [
        "failure",
        "missing_provenance",
        "failed_h5",
        "broken_chain",
        "duplicate_id",
        "task_mix",
    ],
)
def test_source_audit_rejects_invalid_provenance(tiny_source, mutation):
    """Check source audit rejects invalid provenance."""
    path, expected, _, _ = tiny_source
    raw = load_raw(path)
    if mutation == "failure":
        raw["entries"][-1].outcome = -1
    elif mutation == "missing_provenance":
        records = json.loads(Path(expected["init_map"]).read_text())
        Path(expected["init_map"]).write_text(json.dumps(records[1:]))
    elif mutation == "failed_h5":
        with h5py.File(path.parent / "task0_traj0.h5", "a") as handle:
            handle.attrs["success"] = False
    elif mutation == "broken_chain":
        raw["entries"][1].next_ids = []
    elif mutation == "duplicate_id":
        raw["entries"][0].id = raw["entries"][1].id
    else:
        raw["entries"][0].payload.task_key = "task 1"
    with path.open("wb") as handle:
        pickle.dump(raw, handle)
    with pytest.raises(ValueError):
        audit_source(path, expected)


def test_real_backend_export_preserves_raw_arrays_missing_attrs_and_unknown_keys(
    tiny_source, tmp_path
):
    """Check real backend export preserves raw arrays missing attrs and unknown keys."""
    path, expected, cfg, entries = tiny_source
    source = audit_source(path, expected)
    before = file_identity(path)
    scores = compute_task_scores(
        entries, cfg, tmp_path / "scores", source_manifest=source
    )
    blocks = load_score_blocks(scores, source["rows"])
    selected = select_retained(source["rows"], blocks, 0.0)
    output = export_pruned(path, selected, tmp_path / "pruned", source_manifest=source)
    report = verify_pruned(source, output, cfg, score_manifest=scores)
    assert report["passed"] and report["queries"] == 24
    assert report["retained_entries"] == 6
    assert file_identity(path) == before
    raw = load_raw(output["file"]["path"])
    assert "outcome" not in vars(raw["entries"][0])
    assert "schedule_id" not in vars(raw["entries"][0].payload)
    assert isinstance(raw["entries"][0].payload.action_chunk, np.ndarray)
    assert all(e.prev_ids == [] and e.next_ids == [] for e in raw["entries"])
    baseline = export_pruned(
        path,
        select_retained(source["rows"], blocks, None),
        tmp_path / "p00",
        source_manifest=source,
    )
    assert baseline["file"] == before
    assert verify_pruned(source, baseline, cfg, score_manifest=scores)["passed"]
    with pytest.raises(ValueError, match="already exists"):
        export_pruned(path, selected, tmp_path / "pruned", source_manifest=source)
    assert not (tmp_path / "pruned.lock").exists()


def test_export_and_verifier_reject_changed_parent_and_payload(tiny_source, tmp_path):
    """Check export and verifier reject changed parent and payload."""
    path, expected, cfg, entries = tiny_source
    source = audit_source(path, expected)
    scores = compute_task_scores(
        entries, cfg, tmp_path / "scores", source_manifest=source
    )
    blocks = load_score_blocks(scores, source["rows"])
    result = export_pruned(
        path,
        select_retained(source["rows"], blocks, 0.0),
        tmp_path / "export",
        source_manifest=source,
    )
    raw = load_raw(result["file"]["path"])
    raw["unknown_future_field"]["values"][0] = 99
    with Path(result["file"]["path"]).open("wb") as handle:
        pickle.dump(raw, handle)
    result["file"] = file_identity(result["file"]["path"])
    result = seal(result)
    with pytest.raises(ValueError, match="top-level value changed"):
        verify_pruned(source, result, cfg, score_manifest=scores)
    with path.open("ab") as handle:
        handle.write(b"changed framing")
    with pytest.raises(ValueError, match="parent"):
        export_pruned(
            path,
            select_retained(source["rows"], blocks, None),
            tmp_path / "wrong",
            source_manifest=source,
        )


def test_rate_matching_exact_dp_against_exhaustive():
    """Check rate matching exact dp against exhaustive."""
    candidates = [
        {"threshold": (19 - i) / 20, "removed_count": count, "keep_digest": f"m{i}"}
        for i, count in enumerate([5, 8, 10, 19, 22, 31, 40, 50])
    ]
    targets = [0.05, 0.2, 0.4]
    selected = match_targets(candidates, 100, targets, 0.05, 0.075)
    feasible = []
    for items in combinations(candidates, 3):
        rates = [c["removed_count"] / 100 for c in items]
        if rates[0] <= 0.075 and all(
            abs(rate - t) <= 0.05 for rate, t in zip(rates, targets)
        ):
            feasible.append(
                (
                    sum(abs(rate - t) for rate, t in zip(rates, targets)),
                    tuple(rates),
                    tuple(-c["threshold"] for c in items),
                    items,
                )
            )
    assert selected == list(min(feasible)[-1])
    ties = [
        {"threshold": 0.7, "removed_count": 10, "keep_digest": "x"},
        {"threshold": 0.8, "removed_count": 10, "keep_digest": "x"},
        {"threshold": 0.9, "removed_count": 10, "keep_digest": "y"},
    ]
    assert match_targets(ties, 100, [0.1], 0.05, 0.15)[0]["threshold"] == 0.9
    for bad in ([], ties):
        with pytest.raises(ValueError, match="grid_not_representable"):
            match_targets(bad, 100, [0.05, 0.1], 0.05, 0.075)


def test_grid_failure_and_candidate_generation_are_source_only(tiny_source, tmp_path):
    """Check grid failure and candidate generation are source only."""
    path, expected, cfg, entries = tiny_source
    source = audit_source(path, expected)
    scores = compute_task_scores(
        entries, cfg, tmp_path / "scores", source_manifest=source
    )
    blocks = load_score_blocks(scores, source["rows"])
    thresholds = candidate_thresholds(source["rows"], blocks, grid_spec())
    assert thresholds == sorted(set(thresholds)) and len(thresholds) <= 329
    outcome_file = tmp_path / "eval_outcome.json"
    outcome_file.write_text('{"success_rate": 0}')
    first = select_grid(source, scores, grid_spec())
    outcome_file.write_text('{"success_rate": 1, "best_theta": 0}')
    second = select_grid(source, scores, grid_spec())
    assert first["status"] == second["status"] == "frozen"
    assert [(c["threshold"], c["keep_digest"]) for c in first["candidates"]] == [
        (c["threshold"], c["keep_digest"]) for c in second["candidates"]
    ]
    assert all(c["complete"] and c["seconds"] >= 0 for c in first["candidates"])
    # A uniform score matrix has only P00 and one nonzero deletion level.
    for block in scores["blocks"]:
        block_path = Path(scores["directory"]) / block["file"]
        np.save(block_path, np.ones(block["shape"], dtype=np.float32))
        block["sha256"] = file_identity(block_path)["sha256"]
    failed = select_grid(source, seal(scores), grid_spec())
    assert failed["status"] == "grid_not_representable" and failed["complete"]


def test_cache_templates_only_allow_preload_changes():
    """Check cache templates only allow preload changes."""
    for suite in ("libero_spatial", "libero_10"):
        cfg = template(suite)
        cfg["backend"]["in_memory"]["preload_path"] = "/tmp/new.pkl"
        validate_config(cfg, suite)
        for key, value in (
            ("top_k", 2),
            ("task_scoped", False),
            ("trajectory_depth", 2),
        ):
            wrong = copy.deepcopy(cfg)
            wrong["checkpoints"]["cp1"]["search_strategy"][key] = value
            with pytest.raises(ValueError):
                validate_config(wrong, suite)
        cfg["keys"]["vision_0"]["weight"] += 0.001
        with pytest.raises(ValueError):
            validate_config(cfg, suite)


def test_value_digest_distinguishes_missing_attributes_and_array_dtype():
    """Check value digest distinguishes missing attributes and array dtype."""
    assert value_digest(np.array([1], dtype=np.int32)) != value_digest(
        np.array([1], dtype=np.int64)
    )
    assert value_digest({}) != value_digest({"outcome": None})


def test_success_list_provenance_checks_complete_ids_and_outcome_census(
    tiny_source, tmp_path
):
    """Require the frozen large-source success list, including exact task identities."""
    path, expected, _, _ = tiny_source
    raw = load_raw(path)
    mapping = {}
    for entry in raw["entries"]:
        task, trajectory = entry.trajectory_id.replace("task", "").split("_traj")
        mapping[entry.id] = f"task_{task}/episode_{trajectory}:{entry.step_idx}"
    for entry in raw["entries"]:
        entry.id = mapping[entry.id]
        entry.trajectory_id = entry.id.rsplit(":", 1)[0]
        entry.prev_ids = [mapping[x] for x in entry.prev_ids]
        entry.next_ids = [mapping[x] for x in entry.next_ids]
    with path.open("wb") as handle:
        pickle.dump(raw, handle)
    success_list = tmp_path / "success.txt"
    success_list.write_text(
        "\n".join(sorted({e.trajectory_id + ".h5" for e in raw["entries"]})) + "\n"
    )
    census_record = tmp_path / "success.json"
    census = {
        "suite": expected["suite"],
        "outcome_filter": "success",
        "tiers": [{"tier": "S6", "episodes": 6, "entries": 24}],
    }
    census_record.write_text(json.dumps(census))
    expected.update(
        kind="success_list",
        regime="cs500_success",
        success_list=str(success_list),
        census_record=str(census_record),
    )
    assert audit_source(path, expected)["trajectories"] == 6
    original = success_list.read_text()
    success_list.write_text("\n".join(original.splitlines()[1:]))
    with pytest.raises(ValueError, match="success membership"):
        audit_source(path, expected)
    success_list.write_text(original)
    census["outcome_filter"] = "all"
    census_record.write_text(json.dumps(census))
    with pytest.raises(ValueError, match="success census"):
        audit_source(path, expected)


def test_no_cross_trajectory_pairs_produce_completed_nonrepresentable_report(
    tiny_source, tmp_path
):
    """Even a no-candidate source produces an explicit failed-grid report."""
    path, expected, cfg, entries = tiny_source
    raw = load_raw(path)
    raw["entries"] = [e for e in entries if e.trajectory_id.endswith("traj0")]
    with path.open("wb") as handle:
        pickle.dump(raw, handle)
    expected.update(entries=8, trajectories=2)
    source = audit_source(path, expected)
    scores = compute_task_scores(
        raw["entries"], cfg, tmp_path / "scores", source_manifest=source
    )
    result = select_grid(source, scores, grid_spec())
    assert result["status"] == "grid_not_representable" and result["complete"]
    assert result["candidates"] == []


def test_target_error_tie_prefers_lexicographically_smaller_rate():
    """Exact rational ties follow the preregistered smaller-rate rule."""
    candidates = [
        {"removed_count": 6, "threshold": 0.9, "keep_digest": "six"},
        {"removed_count": 4, "threshold": 0.7, "keep_digest": "four"},
    ]
    assert match_targets(candidates, 100, [0.05], 0.05, 0.075)[0]["removed_count"] == 4


@pytest.mark.parametrize(
    "filename,prompt",
    [
        (
            "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
            "put both the alphabet soup and the tomato sauce in the basket",
        ),
        (
            "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
            "put both moka pots on the stove",
        ),
        ("STUDY_SCENE1_pick_up_the_book", "pick up the book"),
        ("pick_up_the_black_bowl", "pick up the black bowl"),
    ],
)
def test_libero_scene_prefix_is_not_part_of_the_task_language(filename, prompt):
    """Match the real LIBERO-10 artifact/HDF5 prompt rather than scene-prefixed file names."""
    from exp.ablation_study.cache_prune.common import task_prompt

    assert task_prompt(filename) == prompt
