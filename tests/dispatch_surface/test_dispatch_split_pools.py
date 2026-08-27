"""Tests for split_init_pools: content-level census, quotas, disjointness."""

from __future__ import annotations

import json
import pathlib

import h5py
import pytest
import torch

from exp.dispatch_surface.split_init_pools import (
    CAL_PER_TASK,
    DLIB_PER_TASK,
    FIT_PER_TASK,
    TEST_PER_TASK,
    census_dlib_inits,
    materialize_pool,
    split_remaining,
    verify_dlib_content,
)
from exp.dispatch_surface.rebuild_dispatch_library import validate_split_manifest_binding
from exp.dispatch_surface.fit_surface import validate_dlib_chain

GOOD = {t: [t, 10 + t, 20 + t, 30 + t, 40 + t] for t in range(10)}


def _library(tmp_path):
    """Real H5 files + official pools backing the init map rows."""
    h5_dir = tmp_path / "lib_h5"
    h5_dir.mkdir(exist_ok=True)
    pool_dir = tmp_path / "official"
    pool_dir.mkdir(exist_ok=True)
    collection_dir = tmp_path / "collection"
    collection_dir.mkdir(exist_ok=True)
    for t in GOOD:
        states = torch.arange(150, dtype=torch.float32).reshape(50, 3) + t * 1000
        torch.save(states, pool_dir / f"task_{t}.init")
    return h5_dir, pool_dir, collection_dir


def _init_map_rows(tmp_path, per_task_indices, *, h5_content="hdf5"):
    h5_dir, pool_dir, collection_dir = _library(tmp_path)
    rows = []
    for tid, idxs in per_task_indices.items():
        official = torch.load(pool_dir / f"task_{tid}.init", weights_only=False)
        torch.save(official[idxs], collection_dir / f"task_{tid}.init")
        for i, idx in enumerate(idxs):
            h5_path = h5_dir / f"ep_{tid}_{i}.h5"
            if h5_content == "hdf5":
                with h5py.File(h5_path, "w") as f:
                    f.attrs["task"] = f"task_{tid}"
                    f.create_group("step_0000")
            else:
                h5_path.write_bytes(b"not-hdf5")
            rows.append({
                "task_id": tid, "task_name": f"task_{tid}",
                "orig_init_state_idx": idx,
                "subset_init_state_idx": i,
                "init_path": str(collection_dir / f"task_{tid}.init"),
                "full_init_path": str(pool_dir / f"task_{tid}.init"),
                "h5_path": str(h5_path),
                "trajectory_id": h5_path.stem,
                "prompt": f"task_{tid}",
                "entry_count": 1,
                "attrs": {"task": f"task_{tid}"} if h5_content == "hdf5" else {},
            })
    return rows, h5_dir


def _official_dir(rows):
    return pathlib.Path(rows[0]["full_init_path"]).parent


def _write_map(tmp_path, rows):
    p = tmp_path / "init_map.json"
    p.write_text(json.dumps(rows))
    return p


def test_census_accepts_authoritative_map(tmp_path):
    rows, h5_dir = _init_map_rows(tmp_path, GOOD)
    per_task, digests = census_dlib_inits(
        _write_map(tmp_path, rows), h5_dir=h5_dir, official_pool_dir=_official_dir(rows),
    )
    assert len(per_task) == 10
    assert per_task[0]["indices"] == sorted(GOOD[0])
    assert len(digests["h5"]) == 50 and len(digests["official_pools"]) == 10
    assert "init_map" in digests


@pytest.mark.parametrize("corrupt", [
    "missing_file", "wrong_count", "missing_field", "wrong_per_task",
])
def test_census_aborts_on_provenance_anomaly(tmp_path, corrupt):
    if corrupt == "missing_file":
        h5_dir, pool_dir, _ = _library(tmp_path)
        with pytest.raises(SystemExit):
            census_dlib_inits(
                tmp_path / "nope.json", h5_dir=h5_dir, official_pool_dir=pool_dir,
            )
        return
    rows, h5_dir = _init_map_rows(tmp_path, GOOD)
    official_dir = _official_dir(rows)
    if corrupt == "wrong_count":
        rows = rows[:-1]
    elif corrupt == "missing_field":
        rows[0] = {k: v for k, v in rows[0].items() if k != "full_init_path"}
    else:
        rows[1]["orig_init_state_idx"] = rows[0]["orig_init_state_idx"]
    with pytest.raises(SystemExit):
        census_dlib_inits(
            _write_map(tmp_path, rows), h5_dir=h5_dir, official_pool_dir=official_dir,
        )


def test_census_rejects_fake_h5_content(tmp_path):
    """G2R3-B1 adversarial reproduction: 50 files whose bytes are not HDF5
    used to pass on basename existence alone. Content-level opening refuses."""
    rows, h5_dir = _init_map_rows(tmp_path, GOOD, h5_content="fake")
    with pytest.raises(SystemExit):
        census_dlib_inits(
            _write_map(tmp_path, rows), h5_dir=h5_dir, official_pool_dir=_official_dir(rows),
        )


def test_census_rejects_missing_official_pool(tmp_path):
    rows, h5_dir = _init_map_rows(tmp_path, GOOD)
    rows[0]["full_init_path"] = str(tmp_path / "does_not_exist.init")
    with pytest.raises(SystemExit):
        census_dlib_inits(
            _write_map(tmp_path, rows), h5_dir=h5_dir, official_pool_dir=_official_dir(rows),
        )


def test_census_rejects_attrs_drift_and_escape(tmp_path):
    rows, h5_dir = _init_map_rows(tmp_path, GOOD)
    rows[0]["attrs"] = {"task": "a_different_task"}
    with pytest.raises(SystemExit):
        census_dlib_inits(
            _write_map(tmp_path, rows), h5_dir=h5_dir, official_pool_dir=_official_dir(rows),
        )
    (tmp_path / "b").mkdir()
    rows, h5_dir = _init_map_rows(tmp_path / "b", GOOD)
    outside = tmp_path / "b" / "outside.h5"
    with h5py.File(outside, "w") as f:
        f.attrs["task"] = "task_0"
    rows[0]["h5_path"] = str(outside)
    rows[0]["attrs"] = {"task": "task_0"}
    with pytest.raises(SystemExit):
        census_dlib_inits(
            _write_map(tmp_path / "b", rows), h5_dir=h5_dir,
            official_pool_dir=_official_dir(rows),
        )


def test_verify_rejects_out_of_range_official_index(tmp_path):
    rows, h5_dir = _init_map_rows(tmp_path, GOOD)
    rows[0]["orig_init_state_idx"] = 99
    with pytest.raises(SystemExit):
        verify_dlib_content(rows, h5_dir, _official_dir(rows))


def test_census_rejects_swapped_official_init_identities(tmp_path):
    """An in-range permutation of orig indices must not alter D_lib membership."""
    rows, h5_dir = _init_map_rows(tmp_path, GOOD)
    rows[0]["orig_init_state_idx"], rows[1]["orig_init_state_idx"] = (
        rows[1]["orig_init_state_idx"], rows[0]["orig_init_state_idx"],
    )
    with pytest.raises(SystemExit):
        census_dlib_inits(
            _write_map(tmp_path, rows), h5_dir=h5_dir,
            official_pool_dir=_official_dir(rows),
        )


def test_census_requires_a_bijection_over_the_library_h5_tree(tmp_path):
    rows, h5_dir = _init_map_rows(tmp_path, GOOD)
    rows[1]["h5_path"] = rows[0]["h5_path"]
    rows[1]["trajectory_id"] = rows[0]["trajectory_id"]
    rows[1]["attrs"] = rows[0]["attrs"]
    rows[1]["entry_count"] = rows[0]["entry_count"]
    with pytest.raises(SystemExit):
        census_dlib_inits(
            _write_map(tmp_path, rows), h5_dir=h5_dir,
            official_pool_dir=_official_dir(rows),
        )


def test_dlib_digests_are_bound_through_rebuild_and_fit(tmp_path):
    rows, h5_dir = _init_map_rows(tmp_path, GOOD)
    init_map = _write_map(tmp_path, rows)
    _, digests = census_dlib_inits(
        init_map, h5_dir=h5_dir, official_pool_dir=_official_dir(rows),
    )
    split = tmp_path / "split_manifest.json"
    split.write_text(json.dumps({
        "init_map": str(init_map),
        "apool_dir": str(_official_dir(rows)),
        "dlib_content_digests": digests,
    }, sort_keys=True))
    _, actual = validate_split_manifest_binding(split, h5_dir)
    assert actual == digests

    import hashlib

    rebuild = tmp_path / "rebuild_record.json"
    rebuild.write_text(json.dumps({
        "split_manifest_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
        "dlib_content_digests": digests,
    }))
    assert validate_dlib_chain(split, rebuild)["dlib_content_digests"] == digests

    # Post-split init-map drift is rejected by the rebuild re-census, even if
    # all paths and H5 attrs still look valid.
    rows[0]["orig_init_state_idx"], rows[1]["orig_init_state_idx"] = (
        rows[1]["orig_init_state_idx"], rows[0]["orig_init_state_idx"],
    )
    init_map.write_text(json.dumps(rows))
    with pytest.raises(SystemExit):
        validate_split_manifest_binding(split, h5_dir)


def test_split_quotas_and_disjointness():
    dlib = GOOD[3]
    splits = split_remaining(dlib, seed=1, task_id=3)
    assert len(splits["fit"]) == FIT_PER_TASK
    assert len(splits["cal"]) == CAL_PER_TASK
    assert len(splits["test"]) == TEST_PER_TASK
    all_idx = set(dlib) | set(splits["fit"]) | set(splits["cal"]) | set(splits["test"])
    assert len(all_idx) == 50  # partition of the official pool, no overlap
    assert DLIB_PER_TASK + FIT_PER_TASK + CAL_PER_TASK + TEST_PER_TASK == 50


def test_split_is_deterministic():
    a = split_remaining(GOOD[0], seed=7, task_id=0)
    b = split_remaining(GOOD[0], seed=7, task_id=0)
    c = split_remaining(GOOD[0], seed=8, task_id=0)
    assert a == b and a != c


def test_materialize_pool_subsets_official_states(tmp_path):
    apool = tmp_path / "apool"
    apool.mkdir()
    states = torch.arange(50 * 3, dtype=torch.float32).reshape(50, 3)
    torch.save(states, apool / "task_0.init")
    assignment = {0: {"task_name": "task_0", "test": [2, 5, 7]}}
    digests = materialize_pool(apool, tmp_path / "out", assignment, ["test"])
    saved = torch.load(tmp_path / "out" / "task_0.init", weights_only=False)
    assert saved.shape == (3, 3)
    assert torch.equal(saved, states[[2, 5, 7]])
    assert digests["task_0"]["count"] == 3
