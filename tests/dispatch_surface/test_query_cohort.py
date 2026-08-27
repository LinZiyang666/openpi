"""Integration tests for the query-cohort identity chain (G2R2-B1):
plan -> client cohort adapter -> collector-written H5 -> SUCCESSFUL verify ->
cohort manifest -> downstream split lookup."""

from __future__ import annotations

import json
import pathlib

import h5py
import numpy as np
import pytest

from examples.libero.cohort_plan import (
    cohort_extra_metadata,
    episode_filter_pairs,
    load_cohort_map,
)
from exp.dispatch_surface.build_dispatch_table import _build_split_lookup
from exp.dispatch_surface.collect_query_cohort import cmd_plan, cmd_verify


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _split_manifest(tmp_path, n_tasks=10):
    assignment = {}
    for t in range(n_tasks):
        remaining = [i for i in range(50) if i % 10 != t % 10][:45]
        assignment[str(t)] = {
            "task_name": f"task_{t}",
            "dlib": [i for i in range(50) if i % 10 == t % 10][:5],
            "fit": remaining[:5],
            "cal": remaining[5:15],
            "test": remaining[15:45],
        }
    p = tmp_path / "split_manifest.json"
    p.write_text(json.dumps({"assignment": assignment}))
    return p


def _plan(tmp_path):
    manifest = _split_manifest(tmp_path)
    out = tmp_path / "plan.json"
    cmd_plan(_Args(split_manifest=manifest, pool_dir=tmp_path / "pool", out=out))
    return json.loads(out.read_text()), out


def _tiny_embs():
    from openpi.collect.data_collector import InferenceEmbeddings

    return InferenceEmbeddings(
        vision_embs=[np.zeros((2, 4), dtype=np.float16)],
        prompt_emb=np.zeros((2, 4), dtype=np.float16),
        robot_state=np.zeros(3, dtype=np.float32),
        noise_action_steps=[np.zeros((2, 3), dtype=np.float32)],
        clean_action=np.zeros((2, 3), dtype=np.float32),
    )


def _write_h5_direct(path, md, success=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in md.items():
            f.attrs[k] = v
        f.attrs["success"] = success


def test_full_chain_plan_client_collector_verify_build(tmp_path):
    """The executable end-to-end chain, ending in a SUCCESSFUL verify."""
    from openpi.collect.data_collector import EpisodeDataCollector

    plan, plan_path = _plan(tmp_path)
    cohort_map = load_cohort_map(str(plan_path))
    filter_pairs, orig_map = episode_filter_pairs(cohort_map)
    assert len(filter_pairs) == 150 and len(orig_map) == 150

    h5_dir = tmp_path / "cohort_h5"
    collector = EpisodeDataCollector(base_dir=str(h5_dir))
    for i, e in enumerate(plan["episodes"]):
        # The client-side metadata assembly every episode goes through.
        md = cohort_extra_metadata(
            cohort_map, e["task_id"], e["subset_init_state_idx"], e["orig_init_state_idx"],
        )
        if i < 2:
            # Real collector chain: episode_start metadata -> H5 attrs.
            collector.on_episode_start(
                "cohort", f"task_{e['task_id']}", i,
                episode_name=f"ep_{i:03d}", extra_metadata=md,
            )
            collector.record_inference(_tiny_embs())
            collector.on_episode_end(success=True)
        else:
            _write_h5_direct(h5_dir / "cohort" / f"ep_{i:03d}.h5", md)

    out = tmp_path / "cohort_manifest.json"
    cmd_verify(_Args(plan=plan_path, h5_dir=h5_dir, out=out))  # must SUCCEED
    manifest = json.loads(out.read_text())
    assert manifest["counts"] == {"fit": 50, "cal": 100}
    assert len(manifest["files"]) == 150
    for f in manifest["files"]:
        assert {"task_id", "init_idx", "subset_init_state_idx", "split", "sha256"} <= set(f)

    lookup = _build_split_lookup(str(_split_manifest(tmp_path)))
    for e in plan["episodes"]:
        assert lookup[(e["task_id"], e["orig_init_state_idx"])] == e["split"]


def test_cohort_metadata_helper_rejections(tmp_path):
    plan, plan_path = _plan(tmp_path)
    cohort_map = load_cohort_map(str(plan_path))
    e = plan["episodes"][0]
    with pytest.raises(ValueError):
        cohort_extra_metadata(cohort_map, 9, 999, 0)  # not in the plan
    with pytest.raises(ValueError):
        cohort_extra_metadata(  # official index disagrees with the plan
            cohort_map, e["task_id"], e["subset_init_state_idx"],
            e["orig_init_state_idx"] + 1,
        )


@pytest.mark.parametrize("corrupt", ["wrong_subset", "wrong_split", "missing_attr", "alien"])
def test_verify_rejects_identity_breaks(tmp_path, corrupt):
    plan, plan_path = _plan(tmp_path)
    h5_dir = tmp_path / "cohort"
    h5_dir.mkdir()
    e = plan["episodes"][0]
    md = {"task_id": e["task_id"], "orig_init_state_idx": e["orig_init_state_idx"],
          "subset_init_state_idx": e["subset_init_state_idx"], "split": e["split"]}
    if corrupt == "wrong_subset":
        md["subset_init_state_idx"] += 1
    elif corrupt == "wrong_split":
        md["split"] = "cal" if e["split"] == "fit" else "fit"
    elif corrupt == "missing_attr":
        md.pop("orig_init_state_idx")
    else:
        md.update(task_id=9, orig_init_state_idx=999)
    _write_h5_direct(h5_dir / "x.h5", md)
    with pytest.raises(SystemExit):
        cmd_verify(_Args(plan=plan_path, h5_dir=h5_dir, out=tmp_path / "o.json"))


def test_collector_persists_allowlisted_metadata(tmp_path):
    from openpi.collect.data_collector import EpisodeDataCollector

    collector = EpisodeDataCollector(base_dir=str(tmp_path))
    collector.on_episode_start(
        "exp", "task", 0, episode_name="probe",
        extra_metadata={"task_id": 3, "orig_init_state_idx": 17,
                        "subset_init_state_idx": 2, "split": "fit",
                        "free_form_junk": "must-not-leak"},
    )
    collector.record_inference(_tiny_embs())
    collector.on_episode_end(success=True)
    h5_path = next(pathlib.Path(tmp_path).rglob("*.h5"))
    with h5py.File(h5_path, "r") as f:
        assert f.attrs["task_id"] == 3 and f.attrs["orig_init_state_idx"] == 17
        assert f.attrs["subset_init_state_idx"] == 2 and f.attrs["split"] == "fit"
        assert "free_form_junk" not in f.attrs
