"""CPU tests for exp/dp_nfe/analysis/merge_x0_hosts.py: two host trees generated from the same matrix (different yaml
bytes, disjoint trained cells) merge into one aggregator input whose yaml bytes come from the owning host; frozen-field
disagreement, double ownership, stray results and non-empty targets are refused."""

import json

import pytest
import yaml

from exp.dp_nfe.analysis import aggregate_x0 as A
from exp.dp_nfe.analysis import merge_x0_hosts as M
from exp.dp_nfe.x0_identity import sha256_file

from tests.dp_nfe.test_x0_aggregate import GOOD, SEEDS, _cells_dir, _matrix, _rec, _write


def _host(tmp_path, name, tasks, image=True, explore=True, skip_task="kitchen"):
    """A host tree whose cell yamls carry a host-specific field (different bytes) and results for ``tasks`` only."""
    root = tmp_path / name; root.mkdir()
    cells = _cells_dir(root, image=image, explore=explore, skip_task=skip_task)
    for p in cells.glob("*.yaml"):
        c = yaml.safe_load(p.read_text()); c["src"] = f"/data/{name}/subset"; p.write_text(yaml.safe_dump(c))
    results = root / "results_trailing"; results.mkdir()
    if tasks:
        _matrix(results, GOOD, tasks=tasks)
    return root


def test_merge_takes_yaml_bytes_from_the_owning_host_and_aggregates(tmp_path):
    wls = _host(tmp_path, "wls", ["pusht"], image=False)      # wls lists no image cells (like the real split)
    p = wls / "cells" / "cells_manifest.json"; man = json.loads(p.read_text())
    man["skipped"].append({"task": "pusht", "arm": "image", "reason": "skipped: no --image-src given"}); p.write_text(json.dumps(man))
    h100 = _host(tmp_path, "h100", ["blockpush"])
    official = {"task_name": "pusht", "modality": "image", "variant": "official", "head": "epsilon", "train_seed": 0, "budget_id": "official"}
    _write(wls / "results_trailing", _rec(official, "ddim", 1, "test", [1.0] * 50), name="test_ddim_1")
    hosts = [M.load_host("wls", wls), M.load_host("h100", h100)]
    assert hosts[0]["official"] == {A.cell_key(official): 1} and A.cell_key(official) not in hosts[0]["owned"]
    report = M.merge(hosts, tmp_path / "merged")
    assert report["official"] == {A.cell_key(official): {"host": "wls", "summaries": 1}}
    assert (tmp_path / "merged" / "results_trailing" / A.cell_key(official) / "test_ddim_1" / "summary.json").is_file()
    merged_dir = tmp_path / "merged"
    man = json.loads((merged_dir / "cells" / "cells_manifest.json").read_text())
    assert man["train_seeds"] == SEEDS and len(man["by_arm"]["image"]) == 4 and len(man["by_arm"]["core"]) == 24
    assert man["skipped"] == [{"task": "kitchen", "arm": "core", "reason": "one label"}]   # the wls-local image skip is dropped
    assert report["host_local_skips"] == [{"task": "pusht", "arm": "image", "reason": "skipped: no --image-src given"}]
    assert sorted(man["merged_from"]) == ["h100", "wls"]
    pusht_cid = "pusht_lowdim_U_epsilon_s42_B50k"; block_cid = "blockpush_lowdim_U_epsilon_s42_B50k"
    assert report["cells"][pusht_cid]["host"] == "wls" and report["cells"][block_cid]["host"] == "h100"
    assert sha256_file(merged_dir / "cells" / f"{pusht_cid}.yaml") == sha256_file(wls / "cells" / f"{pusht_cid}.yaml")
    assert sha256_file(merged_dir / "cells" / f"{block_cid}.yaml") == sha256_file(h100 / "cells" / f"{block_cid}.yaml")
    assert sha256_file(merged_dir / "cells" / f"{pusht_cid}.yaml") != sha256_file(h100 / "cells" / f"{pusht_cid}.yaml")
    # untrained explore/image cells fall back to the first host listing them, and are reported
    assert set(report["untrained"]) == set(man["by_arm"]["explore"]) | set(man["by_arm"]["image"])
    assert report["cells"]["pusht_image_U_epsilon_s42_B20k"]["host"] == "h100"
    assert report["counts"] == {"cells": 30, "trained": 24, "untrained": 6, "core": 24, "explore": 2, "image": 4}
    # the aggregator accepts the merged tree: both core tasks get formal verdicts, nothing is invalid
    out = A.aggregate(merged_dir / "cells", merged_dir / "results_trailing", boot=200, seed=1)
    status = {t["task"]: t["status"] for t in out["tasks"]}
    assert status["kitchen"] == "skipped" and all(t["verdict"] for t in out["tasks"] if t["task"] in ("pusht", "blockpush"))
    assert out["records"]["invalid"] == [] and out["records"]["unexpected"] == []
    assert out["completeness"]["core"]["pending"] == [] and list(out["descriptive"]["official"]) == [A.cell_key(official)]


def test_double_ownership_stray_results_and_frozen_mismatch_are_refused(tmp_path):
    wls = _host(tmp_path, "wls", ["pusht"]); h100 = _host(tmp_path, "h100", ["pusht"])
    with pytest.raises(M.MergeError, match="more than one host"):
        M.merge([M.load_host("wls", wls), M.load_host("h100", h100)], tmp_path / "m1")
    stray = _host(tmp_path, "stray", ["blockpush"])
    ident = {"task_name": "lift_mh", "modality": "lowdim", "variant": "U", "head": "epsilon", "train_seed": 42, "budget_id": "B50k", "subset_sha256": "x"}
    _write(stray / "results_trailing", _rec(ident, "ddim", 100, "test", [1.0] * 100))
    with pytest.raises(M.MergeError, match="absent from every manifest"):
        M.merge([M.load_host("wls", wls), M.load_host("stray", stray)], tmp_path / "m2")
    official = {"task_name": "can_mh", "modality": "image", "variant": "official", "head": "epsilon", "train_seed": 0, "budget_id": "official"}
    twice = _host(tmp_path, "twice", ["blockpush"])
    for r in (wls, twice):
        _write(r / "results_trailing", _rec(official, "ddim", 1, "test", [1.0] * 50), name="test_ddim_1")
    with pytest.raises(M.MergeError, match="official ladder .* more than one host"):
        M.merge([M.load_host("wls", wls), M.load_host("twice", twice)], tmp_path / "m2b")
    other = _host(tmp_path, "other", ["blockpush"])
    p = other / "cells" / "cells_manifest.json"; man = json.loads(p.read_text()); man["budgets"]["lowdim"] = 100000; p.write_text(json.dumps(man))
    with pytest.raises(M.MergeError, match="budgets differs"):
        M.merge([M.load_host("wls", wls), M.load_host("other", other)], tmp_path / "m3")
    target = tmp_path / "m4"; target.mkdir(); (target / "x").write_text("")
    with pytest.raises(M.MergeError, match="not empty"):
        M.merge([M.load_host("wls", wls)], target)
