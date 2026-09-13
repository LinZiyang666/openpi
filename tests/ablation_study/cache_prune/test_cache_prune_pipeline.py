"""Exercise four independent sources through real exports, forty arms and analysis.

Only the approved production census/dimensions are scaled down in this test.
Provenance files, init hashes, pruning, score computation, serialization, emitter
validation, journal parsing and final figures all execute their real code paths.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import pickle
import shutil

import h5py
import numpy as np
import pytest
import torch
import yaml

from openpi.cache.components.factors.base import LibraryStats
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID

from exp.ablation_study.cache_size.verify_apool import rollup_digest
from exp.ablation_study.cache_prune import common, emit_prune_arms
from exp.ablation_study.cache_prune.common import file_identity
from exp.ablation_study.cache_prune.analysis.analyze_prune import analyze_families
from exp.ablation_study.cache_prune.analysis.plot_prune import plot_prune
from exp.ablation_study.cache_prune.analysis import benchmark_prune as microbench
from exp.ablation_study.cache_prune.prepare_membership import (
    prepare_membership,
    validate_membership,
)
from exp.ablation_study.cache_prune.prune_library import (
    audit_source,
    compute_task_scores,
)
from exp.ablation_study.cache_prune.select_prune_grid import grid_spec, select_grid


def _source(root, suite, regime, config, seed):
    root.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    entries, records = [], []
    for task in range(10):
        for index in range(3):
            tid = (
                f"task_{task}/episode_{index}"
                if regime == "cs500_success"
                else f"small_{task}_{index}"
            )
            path = root / f"{tid}.h5"
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, "w") as handle:
                handle.attrs.update(success=True, task=f"task {task}", num_steps=4)
                handle.create_group("step_0000")
            records.append(
                {
                    "trajectory_id": tid,
                    "h5_path": str(path),
                    "suite": suite,
                    "task_id": task,
                    "task_name": f"task_{task}",
                    "prompt": f"task {task}",
                    "entry_count": 4,
                    "subset_init_state_idx": index,
                    "orig_init_state_idx": index,
                    "init_path": str(root.parent / "small_pool" / f"task_{task}.init"),
                    "full_init_path": str(root.parent / "apool" / f"task_{task}.init"),
                }
            )
            for step in range(4):
                keys = {
                    "vision_0": rng.normal(size=3).astype(np.float32),
                    "vision_1": rng.normal(size=3).astype(np.float32),
                    "robot_state": rng.normal(size=2).astype(np.float32),
                    "prompt_emb": np.ones(1, dtype=np.float32),
                }
                entry = CacheEntry(
                    f"{tid}:{step}",
                    CheckpointID.CP1,
                    keys,
                    CachePayload(
                        rng.normal(size=(5, 2)).astype(np.float32),
                        task_key=f"task {task}",
                    ),
                    step_idx=step,
                    trajectory_id=tid,
                    prev_ids=[] if step == 0 else [f"{tid}:{step - 1}"],
                    next_ids=[] if step == 3 else [f"{tid}:{step + 1}"],
                )
                entries.append(entry)
    path = root / "source.pkl"
    with path.open("wb") as handle:
        pickle.dump(
            {
                "entries": entries,
                "vector_dims": config["backend"]["vector_dims"],
                "key_builder_type": "cp1_spatial_pool_16",
                "checkpoint_id": "cp1",
                "library_stats": LibraryStats.compute_from_entries(entries),
            },
            handle,
        )
    init_map = root / "init_map.json"
    init_map.write_text(json.dumps(records))
    expected = {
        "suite": suite,
        "regime": regime,
        "trajectories": 30,
        "entries": 120,
        "tasks": {str(t): f"task {t}" for t in range(10)},
        "kind": "hdf5",
        "repository": str(root),
        "init_map": str(init_map),
    }
    source = audit_source(path, expected)
    scores = compute_task_scores(
        entries, config, root / "scores", source_manifest=source
    )
    grid = select_grid(source, scores, grid_spec())
    assert grid["status"] == "frozen"
    return source, scores, grid


def _membership(root, suite, sources):
    suite_dir = root / suite
    for dirname in ("apool", "bpool", "small_pool"):
        (suite_dir / dirname).mkdir()
    for task in range(10):
        states = np.array([[task, i] for i in range(50)], dtype=np.float64)
        torch.save(states, suite_dir / "apool" / f"task_{task}.init")
        torch.save(states + 10000, suite_dir / "bpool" / f"task_{task}.init")
        torch.save(states[:3], suite_dir / "small_pool" / f"task_{task}.init")
    roster = root / f"exp/rit_pareto/config/task_order_{suite}.json"
    roster.parent.mkdir(parents=True, exist_ok=True)
    roster.write_text(
        json.dumps(
            {
                "suite": suite,
                "assignment": {str(t): {"task_name": f"task_{t}"} for t in range(10)},
            }
        )
    )
    all_list = (
        root / f"exp/ablation_study/cache_size/config/lists_all/episodes_{suite}_S6.txt"
    )
    all_list.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"task_{task}/episode_{index}.h5" for task in range(10) for index in range(45)
    ]
    all_list.write_text("\n".join(lines) + "\n")
    collection = suite_dir / "cs500_success"
    for task in range(10):
        for index in range(3, 45):
            with h5py.File(
                collection / f"task_{task}/episode_{index}.h5", "w"
            ) as handle:
                handle.attrs.update(success=False, task=f"task {task}")
                handle.create_group("step_0000")
    digests = {
        f"task_{t}": file_identity(suite_dir / "apool" / f"task_{t}.init")["sha256"]
        for t in range(10)
    }
    apool = {
        "suite": suite,
        "apool_dir": str(suite_dir / "apool"),
        "total_inits": 500,
        "per_task_digests": digests,
        "rollup_sha256": rollup_digest(digests),
    }
    record = suite_dir / "apool.yaml"
    record.write_text(yaml.safe_dump(apool))
    result = prepare_membership(
        suite,
        record,
        sources,
        repository=root,
        large_collection=collection,
        difference_pool=suite_dir / "bpool",
    )
    assert result["common_unseen_count"] == 470
    validate_membership(result)
    init_map = Path(
        next(source for source in sources if source["regime"] == "rit50")[
            "expected_census"
        ]["init_map"]
    )
    original = init_map.read_text()
    records = json.loads(original)
    for item in records:
        h5_target = (
            root / f"exp/common/data/db/libero_cache/{suite}/{item['trajectory_id']}.h5"
        )
        h5_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item["h5_path"], h5_target)
        with h5py.File(h5_target, "r+") as handle:
            handle.attrs["experiment_name"] = suite
        for kind, source_path in (
            ("libero_cache", item["init_path"]),
            ("libero", item["full_init_path"]),
        ):
            target = (
                root
                / f"exp/common/data/db_init/{kind}/{suite}/{item['task_name']}.init"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target)
    init_map.write_text(
        json.dumps(
            [
                {
                    "task_id": record["task_id"],
                    "subset_idx": record["subset_init_state_idx"],
                    "orig_init_state_idx": record["orig_init_state_idx"],
                    "task_name": record["task_name"],
                    "prompt": record["prompt"],
                }
                for record in records
            ]
        )
    )
    try:
        compact_result = prepare_membership(
            suite,
            record,
            sources,
            repository=root,
            large_collection=collection,
            difference_pool=suite_dir / "bpool",
        )
        assert compact_result["rows"] == result["rows"]
        assert all(
            "trajectory_id" not in item
            and item["identity_basis"] == "declared_collection_init_set"
            for item in compact_result["original_collections"]["rit50"]
        )
        validate_membership(compact_result)
        invalid = json.loads(init_map.read_text())
        invalid[0]["orig_init_state_idx"] = 49
        init_map.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match="actual state bytes"):
            prepare_membership(
                suite,
                record,
                sources,
                repository=root,
                large_collection=collection,
                difference_pool=suite_dir / "bpool",
            )
    finally:
        init_map.write_text(original)
    return result


def test_complete_four_source_pipeline(tmp_path, monkeypatch):
    """Export, emit, evaluate (synthetic journals), analyse and plot four families."""
    torch.set_num_threads(1)
    configs = {suite: common.template(suite) for suite in common.SUITES}
    for config in configs.values():
        config["backend"]["vector_dims"] = {
            "vision_0": 3,
            "vision_1": 3,
            "robot_state": 2,
            "prompt_emb": 1,
        }
        for field in common.FIELDS:
            config["checkpoints"]["cp1"]["search_strategy"]["score_normalization"][
                "fields"
            ][field]["params"].update(mu=0.0, sigma=1.0)
    monkeypatch.setattr(common, "template", lambda suite: copy.deepcopy(configs[suite]))
    monkeypatch.setattr(
        emit_prune_arms, "template", lambda suite: copy.deepcopy(configs[suite])
    )
    for pair in common.CENSUS:
        monkeypatch.setitem(common.CENSUS, pair, (30, 120))
    triples = [
        _source(
            tmp_path / suite / regime,
            suite,
            regime,
            configs[suite],
            seed=100 + 3 * s + r,
        )
        for s, suite in enumerate(common.SUITES)
        for r, regime in enumerate(common.REGIMES)
    ]
    memberships = {
        suite: _membership(
            tmp_path, suite, [s for s, _, _ in triples if s["suite"] == suite]
        )
        for suite in common.SUITES
    }
    artifacts = tmp_path / "artifacts"
    grids = {(g["suite"], g["regime"]): g for _, _, g in triples}
    # Export in two disjoint halves, the way a parallel run splits the points.
    for source, scores, grid in triples:
        emit_prune_arms.export_points(source, scores, grid, artifacts, {"P01", "P02", "P03", "P04"})
        emit_prune_arms.export_points(
            source, scores, grid, artifacts, {"P05", "P06", "P07", "P08", "P09"}
        )
        emit_prune_arms.export_points(source, scores, grid, artifacts, {"P01"})  # present
    config_dir = tmp_path / "config"
    missing = emit_prune_arms.emit_yamls([s for s, _, _ in triples], grids, artifacts, config_dir)
    assert missing == {}
    matrices = {
        (s, r): yaml.safe_load((config_dir / f"matrix_{s}_{r}.yaml").read_text())
        for s in common.SUITES
        for r in common.REGIMES
    }
    assert all(len(m["arms"]) == 10 for m in matrices.values())
    for (suite, regime), matrix in matrices.items():
        for row, point in zip(matrix["arms"], grids[suite, regime]["points"]):
            config = yaml.safe_load(Path(row["yaml"]).read_text())
            library = config["backend"]["in_memory"]["preload_path"]
            expected = (
                next(s for s, _, _ in triples if (s["suite"], s["regime"]) == (suite, regime))["file"]["path"]
                if point["point"] == "P00"
                else str(artifacts / row["arm"] / "library.pkl")
            )
            assert library == expected
            assert config["checkpoints"]["cp1"]["judge"]["type"] == "always_hit"
    # Re-emitting is idempotent; a hand-edited YAML is refused, not overwritten.
    emit_prune_arms.emit_yamls([s for s, _, _ in triples], grids, artifacts, config_dir)
    edited = Path(matrices["libero_spatial", "rit50"]["arms"][3]["yaml"])
    edited.write_text(edited.read_text().replace("top_k: 1", "top_k: 2"))
    with pytest.raises(ValueError, match="different content"):
        emit_prune_arms.emit_yamls([s for s, _, _ in triples], grids, artifacts, config_dir)
    edited.write_text(edited.read_text().replace("top_k: 2", "top_k: 1"))

    monkeypatch.setattr(microbench, "template", lambda suite: copy.deepcopy(configs[suite]))
    source = next(s for s, _, _ in triples if (s["suite"], s["regime"]) == ("libero_spatial", "rit50"))
    latency = microbench.benchmark_prune(source, matrices["libero_spatial", "rit50"])
    assert len(latency["samples"]) == 7000 and len(latency["summary"]) == 10
    assert all(row["samples"] == 700 for row in latency["summary"])

    # Synthetic conductor output: one family per directory, every episode a
    # FULL_HIT success except one arm that loses a task, so the paired
    # statistics have something to say.
    direct = tmp_path / "direct"
    for (suite, regime), matrix in matrices.items():
        family = direct / "L" / f"{suite}_{regime}"
        family.mkdir(parents=True)
        jrows, prows = [], []
        for row in matrix["arms"]:
            name = row["arm"]
            for task in range(10):
                for index in range(50):
                    success = not (name.endswith("P09") and task == 3)
                    base = {
                        "task_uid": f"{name}:eval:{task}:{index}",
                        "yaml_id": name,
                        "run_id": "producer",
                        "attempt": 1,
                        "accepted": True,
                    }
                    jrows.append({**base, "phase": "eval", "status": "done" if success else "failed",
                                  "success": success, "duration_s": 0.1, "ts": 1.0})
                    prows.append({**base, "step_idx": 0, "searched": True, "hit_type": "FULL_HIT"})
                    prows.append({**base, "_kind": "client_timing", "steps": 20, "infers": 2, "infer_ms": 4.0})
        (family / "journal.jsonl").write_text("".join(json.dumps(r) + "\n" for r in jrows))
        (family / "per_step.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prows))
    rows = {s: memberships[s]["rows"] for s in common.SUITES}
    sources = {(s["suite"], s["regime"]): s for s, _, _ in triples}
    analysis = analyze_families(direct, rows, artifacts, {k: (grids[k], sources[k]) for k in grids})
    assert len(analysis["curves"]) == 4 and all(c["complete"] for c in analysis["curves"])
    curve = analysis["curves"][0]
    assert [lib["entries"] for lib in curve["libraries"]][0] == 120
    assert curve["arms"][0]["sr"] == 1.0 and curve["arms"][9]["sr"] == 0.9
    assert curve["arms"][9]["full_hit"] == 500 and curve["arms"][9]["ms_per_call"] == 2.0
    common_unseen = curve["subsets"]["common_unseen"]["points"]
    assert common_unseen[0]["n"] == 470 and common_unseen[9]["delta_sr"] < 0
    assert common_unseen[9]["lost_successes"] > 0 and common_unseen[9]["new_successes"] == 0
    figures = plot_prune(analysis, tmp_path / "figures", latency=[latency])
    assert len(figures) == 8 and all(Path(path).stat().st_size > 1000 for path in figures)

    # An incomplete family is reported per arm without intervals.
    short = direct / "L" / "libero_10_rit50" / "journal.jsonl"
    lines = short.read_text().splitlines()
    short.write_text("\n".join(lines[:-1]) + "\n")
    partial = analyze_families(direct, rows, artifacts, {k: (grids[k], sources[k]) for k in grids})
    incomplete = [c for c in partial["curves"] if not c["complete"]]
    assert len(incomplete) == 1 and "subsets" not in incomplete[0]
    assert incomplete[0]["arms"][9]["n"] == 499
    membership = memberships["libero_spatial"]
    override = Path(membership["apool"]["apool_dir"]) / "task_0.pruned_init"
    override.write_bytes(b"unapproved higher-priority file")
    with pytest.raises(ValueError, match="pruned_init"):
        validate_membership(membership)
