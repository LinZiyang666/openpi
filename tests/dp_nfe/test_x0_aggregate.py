"""CPU tests for exp/dp_nfe/analysis/aggregate_x0.py (G2 R1-B8 / R1-B9) and the dispersion core: a synthetic frozen cell
matrix + result records drive the validator (protocol, grid, episode ids, completeness, identity, checkpoint hash,
budget), the pairing, the Bonferroni family, the decision rules and their counterexamples, the descriptive routing of
explore / image / official records, and the completeness ledger against the manifest."""

import copy
import json
import math
import pathlib

import numpy as np
import pytest
import yaml

from exp.dp_nfe.analysis import aggregate_x0 as A
from exp.dp_nfe.analysis import dispersion_index as D
from exp.dp_nfe.dp_sampler import make_timesteps
from exp.dp_nfe.x0_identity import expected_identity, sha256_file

SEEDS = [42, 43, 44]
TASKS = ["pusht", "blockpush"]


def _cells_dir(tmp_path, image=True, explore=True, skip_task=None):
    cells = tmp_path / "cells"; cells.mkdir(exist_ok=True)
    man = {"cells": [], "by_arm": {"core": [], "explore": [], "image": []}, "skipped": [],
           "budgets": {"lowdim": 50000, "image": 20000}, "train_seeds": SEEDS}
    def emit(cid, arm, ident, extra=None):
        c = {"cell_id": cid, "head": ident["head"], "train_seed": ident["train_seed"], "budget_steps": 20000 if ident["modality"] == "image" else 50000,
             "normalizer_sha256": "N" * 64, "identity": ident, **(extra or {})}
        (cells / f"{cid}.yaml").write_text(yaml.safe_dump(c)); man["cells"].append(cid); man["by_arm"][arm].append(cid)
    for t in TASKS:
        for v in ("U", "M"):
            for h in ("epsilon", "sample"):
                for s in SEEDS:
                    emit(f"{t}_lowdim_{v}_{h}_s{s}_B50k", "core", {"task_name": t, "modality": "lowdim", "variant": v, "head": h,
                                                                 "train_seed": s, "budget_id": "B50k", "subset_sha256": f"S{t}{v}"})
    if skip_task:
        man["skipped"].append({"task": skip_task, "arm": "core", "reason": "one label"})
    if explore:
        for h in ("epsilon", "sample"):
            emit(f"can_mh_lowdim_full_{h}_s42_B50k", "explore", {"task_name": "can_mh", "modality": "lowdim", "variant": "full",
                                                              "head": h, "train_seed": 42, "budget_id": "B50k"}, {"normalizer_sha256": None})
    if image:
        for v in ("U", "M"):
            for h in ("epsilon", "sample"):
                emit(f"pusht_image_{v}_{h}_s42_B20k", "image", {"task_name": "pusht", "modality": "image", "variant": v, "head": h,
                                                            "train_seed": 42, "budget_id": "B20k", "subset_sha256": f"Ipusht{v}"})
    (cells / "cells_manifest.json").write_text(json.dumps(man))
    return cells


def _rec(ident, sampler, k, split, scores, n=None, sha="C", **over):
    n = len(scores)
    start = A.SPLITS[split]["start_seed"]
    ids = [str(start + i) for i in range(n)]
    ts = list(range(99, -1, -1)) if sampler == "ddpm" else make_timesteps(100, k)
    r = {"cell": {**expected_identity({"cell_id": A.cell_key(ident), "head": ident["head"], "train_seed": ident["train_seed"], "budget_steps": 20000 if ident["modality"] == "image" else 50000, "identity": ident}), "normalizer_sha256": "N" * 64}, "sampler": {"sampler": sampler, "k": k, "T": 100, "head": ident["head"],
         "timesteps": ts, "nfe_per_call": len(ts), "eps_mode": "recompute", "protocol_id": A.PROTOCOL_ID, "clip_sample": True, "sampling_seed": 0},
         "split": split, "start_seed": start, "n_test": n, "episode_ids_expected": ids,
         "episodes": {i: float(v) for i, v in zip(ids, scores)}, "complete": True, "error": None,
         "manifest": {"checkpoint_sha256": sha + ident["task_name"] + ident["variant"] + ident["head"] + str(ident["train_seed"]) + ident["modality"]}}
    r.update(over)
    return r


def _write(root, r, name=None):
    cid = A.cell_key(r["cell"]); d = root / cid / (name or f"{r['split']}_{r['sampler']['sampler']}_{r['sampler']['k']}")
    cell_file = root.parent / "cells" / f"{cid}.yaml"
    if cell_file.is_file():
        r["cell"].setdefault("cell_yaml_sha256", sha256_file(cell_file))
    d.mkdir(parents=True, exist_ok=True); (d / "summary.json").write_text(json.dumps(r))


def _matrix(root, q, n_test=100, seeds=SEEDS, tasks=TASKS, rng_seed=0, arms=("ddim", 100, "ddim", 1)):
    """q[(task, variant, head)] = (Q at ddim_100, Q at ddim_1): episode scores drawn from Bernoulli, paired by id."""
    rng = np.random.default_rng(rng_seed)
    for t in tasks:
        for v in ("U", "M"):
            for h in ("epsilon", "sample"):
                q100, q1 = q[(t, v, h)]
                for s in seeds:
                    ident = {"task_name": t, "modality": "lowdim", "variant": v, "head": h, "train_seed": s, "budget_id": "B50k", "subset_sha256": f"S{t}{v}"}
                    s100 = (rng.random(n_test) < q100).astype(float)
                    # a lossless arm (q1 == q100) reproduces the anchor's episodes exactly (paired by id)
                    s1 = s100.copy() if q1 == q100 else (rng.random(n_test) < q1).astype(float)
                    for sk in A.LADDER:
                        sampler, k = sk.split("_")
                        if sk not in (A.ANCHOR, A.ONE_STEP):
                            _write(root, _rec(ident, sampler, int(k), "test", s100))
                    _write(root, _rec(ident, "ddim", 100, "test", s100))
                    _write(root, _rec(ident, "ddim", 1, "test", s1))
                    _write(root, _rec(ident, "ddim", 100, "screen", (rng.random(32) < q100).astype(float)))


GOOD = {("pusht", "U", "epsilon"): (0.9, 0.1), ("pusht", "M", "epsilon"): (0.9, 0.1),
        ("pusht", "U", "sample"): (0.9, 0.9), ("pusht", "M", "sample"): (0.9, 0.3),
        ("blockpush", "U", "epsilon"): (0.9, 0.1), ("blockpush", "M", "epsilon"): (0.9, 0.1),
        ("blockpush", "U", "sample"): (0.9, 0.9), ("blockpush", "M", "sample"): (0.9, 0.3)}


def test_wilson_basic():
    lo, hi = A.wilson(50, 100)
    assert 0.40 < lo < 0.5 < hi < 0.60
    with pytest.raises(A.AggregationError):
        A.wilson(0, 0)


def test_valid_matrix_gives_verdicts_and_full_ledger(tmp_path):
    cells = _cells_dir(tmp_path); root = tmp_path / "res"; _matrix(root, GOOD)
    out = A.aggregate(cells, root, boot=400, seed=1)
    assert out["records"]["valid"] == 2 * 4 * 3 * 7 and out["records"]["invalid"] == [] and out["records"]["unexpected"] == []
    by = {t["task"]: t for t in out["tasks"]}
    for t in TASKS:
        assert by[t]["status"] == "complete" and by[t]["screening_passed"]
        v = by[t]["verdict"]
        assert v["H1_data"] == "supported" and v["H2_head"] == "supported" and v["I"] == "supported"
        assert by[t]["stats"]["S_x0"]["level"] == pytest.approx(1 - 0.05 / 16)
        assert by[t]["ladders"]["sample_U"]["ddim_1"] > 0.8 and "ddpm_100" in by[t]["ladders"]["sample_U"]
    comp = out["completeness"]
    assert comp["core"]["expected_cells"] == 24 and comp["core"]["valid_test_records"] == 144 and len(comp["core"]["pending"]) == 0
    assert comp["image"]["valid_test_records"] == 0 and comp["explore"]["valid_test_records"] == 0
    assert out["expected_cells"] == 24 + 2 + 4


def test_interval_level_is_bonferroni_over_16():
    assert A.FAMILY_SIZE == 16 and math.isclose(A.ALPHA / A.FAMILY_SIZE, 0.003125)
    mats = {("U", "epsilon", A.ANCHOR): np.ones(50), ("U", "epsilon", A.ONE_STEP): np.zeros(50),
            ("M", "epsilon", A.ANCHOR): np.ones(50), ("M", "epsilon", A.ONE_STEP): np.zeros(50),
            ("U", "sample", A.ANCHOR): np.ones(50), ("U", "sample", A.ONE_STEP): np.ones(50),
            ("M", "sample", A.ANCHOR): np.ones(50), ("M", "sample", A.ONE_STEP): np.zeros(50)}
    st = A.stats_from_mats(mats, A.bootstrap_indices(50, 100, 0))
    assert st["S_x0"]["level"] == 1 - 0.05 / 16 and st["I"]["point"] == pytest.approx(1.0) and st["S_x0"]["point"] == 1.0


@pytest.mark.parametrize("mutate,reason_part", [
    (lambda r: r.update(complete=False), "not complete"),
    (lambda r: r.pop("complete"), "not complete"),
    (lambda r: r.update(error="RuntimeError: boom"), "error recorded"),
    (lambda r: r["sampler"].update(protocol_id="leading_v0"), "protocol"),
    (lambda r: r["sampler"].update(timesteps=list(range(0, 100, 10))[::-1]), "grid"),
    (lambda r: r["sampler"].update(nfe_per_call=7), "nfe_per_call"),
    (lambda r: r["sampler"].update(eps_mode="raw"), "eps_mode"),
    (lambda r: r["cell"].update(subset_sha256="tampered"), "subset_sha256"),
    (lambda r: r["cell"].update(normalizer_sha256="x" * 64), "normalizer_sha256"),
    (lambda r: r["manifest"].pop("checkpoint_sha256"), "checkpoint_sha256"),
    (lambda r: r["episodes"].__setitem__("100001", float("nan")), "non-finite"),
    (lambda r: r.update(n_test=1, episode_ids_expected=["100000"], episodes={"100000": 1.0}), "n_test"),
    (lambda r: r.update(start_seed=90000), "start_seed"),
])
def test_invalid_record_blocks_the_verdict_of_its_task_only(tmp_path, mutate, reason_part):
    cells = _cells_dir(tmp_path); root = tmp_path / "res"; _matrix(root, GOOD)
    ident = {"task_name": "pusht", "modality": "lowdim", "variant": "U", "head": "sample", "train_seed": 43, "budget_id": "B50k", "subset_sha256": "SpushtU"}
    r = _rec(ident, "ddim", 1, "test", np.ones(100))
    mutate(r)
    _write(root, r, name="test_ddim_1")
    out = A.aggregate(cells, root, boot=200, seed=1)
    inv = out["records"]["invalid"]
    assert len(inv) == 1 and reason_part in inv[0]["reason"], inv
    by = {t["task"]: t for t in out["tasks"]}
    assert by["pusht"]["verdict"] is None and by["pusht"]["status"] == "incomplete" and by["pusht"]["missing"] == ["pusht_lowdim_U_sample_s43_B50k/test/ddim_1"]
    assert by["blockpush"]["verdict"] is not None


def test_other_budget_is_unexpected_and_leaves_the_arm_missing(tmp_path):
    cells = _cells_dir(tmp_path); root = tmp_path / "res"; _matrix(root, GOOD)
    ident = {"task_name": "pusht", "modality": "lowdim", "variant": "U", "head": "sample", "train_seed": 43, "budget_id": "B100k", "subset_sha256": "SpushtU"}
    d = root / "pusht_lowdim_U_sample_s43_B50k" / "test_ddim_1"   # overwrite the B50k arm's directory with a B100k result
    (d / "summary.json").write_text(json.dumps(_rec(ident, "ddim", 1, "test", np.ones(100))))
    out = A.aggregate(cells, root, boot=100, seed=1)
    assert len(out["records"]["unexpected"]) == 1 and out["records"]["unexpected"][0]["cell"] == "pusht_lowdim_U_sample_s43_B100k"
    by = {t["task"]: t for t in out["tasks"]}
    assert by["pusht"]["verdict"] is None and by["pusht"]["missing"] == ["pusht_lowdim_U_sample_s43_B50k/test/ddim_1"]


def test_screen_and_test_ids_must_be_disjoint_and_exact(tmp_path):
    cells = _cells_dir(tmp_path); root = tmp_path / "res"; _matrix(root, GOOD)
    ident = {"task_name": "pusht", "modality": "lowdim", "variant": "U", "head": "sample", "train_seed": 42, "budget_id": "B50k", "subset_sha256": "SpushtU"}
    # a screening record built from test ids (wrong seed range) is invalid
    r = _rec(ident, "ddim", 100, "screen", np.ones(32)); r.update(start_seed=100000, episode_ids_expected=[str(100000 + i) for i in range(32)],
                                                             episodes={str(100000 + i): 1.0 for i in range(32)})
    _write(root, r)
    out = A.aggregate(cells, root, boot=100, seed=1)
    assert any("start_seed" in i["reason"] for i in out["records"]["invalid"])
    # ids of another split's size on the test split
    r2 = _rec(ident, "ddim", 100, "test", np.ones(32)); _write(root, r2)
    out = A.aggregate(cells, root, boot=100, seed=1)
    assert any("n_test" in i["reason"] for i in out["records"]["invalid"])


def test_checkpoint_hash_must_agree_within_a_cell_and_duplicates_are_rejected(tmp_path):
    cells = _cells_dir(tmp_path); root = tmp_path / "res"; _matrix(root, GOOD)
    ident = {"task_name": "blockpush", "modality": "lowdim", "variant": "M", "head": "epsilon", "train_seed": 44, "budget_id": "B50k", "subset_sha256": "SblockpushM"}
    _write(root, _rec(ident, "ddim", 10, "test", np.ones(100), sha="OTHER"))   # different checkpoint for the same cell
    _write(root, _rec(ident, "ddim", 100, "test", np.ones(100)), name="dup_dir")  # duplicate arm in another directory
    out = A.aggregate(cells, root, boot=100, seed=1)
    reasons = [i["reason"] for i in out["records"]["invalid"]]
    assert any("checkpoint_sha256 differs" in x for x in reasons) and any("duplicate" in x for x in reasons)


def test_missing_cells_and_manifest_skips_are_reported_not_hidden(tmp_path):
    cells = _cells_dir(tmp_path, skip_task="kitchen"); root = tmp_path / "res"
    _matrix(root, GOOD, seeds=[42, 43])   # seed 44 never trained
    out = A.aggregate(cells, root, boot=100, seed=1)
    by = {t["task"]: t for t in out["tasks"]}
    for t in TASKS:
        assert by[t]["status"] == "incomplete" and by[t]["verdict"] is None and len(by[t]["missing"]) == 4 * 7
    assert by["kitchen"]["status"] == "skipped" and by["kitchen"]["reason"] == "one label" and by["kitchen"]["verdict"] is None
    assert out["skipped"] == [{"task": "kitchen", "arm": "core", "reason": "one label"}]


def test_explore_image_and_official_are_descriptive_only(tmp_path):
    cells = _cells_dir(tmp_path); root = tmp_path / "res"; _matrix(root, GOOD)
    rng = np.random.default_rng(3)
    for h in ("epsilon", "sample"):
        ident = {"task_name": "can_mh", "modality": "lowdim", "variant": "full", "head": h, "train_seed": 42, "budget_id": "B50k"}
        for s, k in (("ddim", 100), ("ddim", 1), ("ddpm", 100)):
            _write(root, _rec(ident, s, k, "test", (rng.random(100) < 0.7).astype(float)))
    for v in ("U", "M"):
        for h in ("epsilon", "sample"):
            ident = {"task_name": "pusht", "modality": "image", "variant": v, "head": h, "train_seed": 42, "budget_id": "B20k", "subset_sha256": f"Ipusht{v}"}
            for s, k in (("ddim", 100), ("ddim", 1)):
                _write(root, _rec(ident, s, k, "test", (rng.random(50) < 0.6).astype(float)))
            _write(root, _rec(ident, "ddim", 100, "screen", np.ones(32)))
    off = {"task_name": "square_mh", "modality": "image", "variant": "official", "head": "epsilon", "train_seed": 42, "budget_id": "official"}
    _write(root, _rec(off, "ddim", 100, "test", (rng.random(50) < 0.8).astype(float)))
    out = A.aggregate(cells, root, boot=200, seed=1)
    assert out["records"]["invalid"] == [] and out["records"]["unexpected"] == []
    assert [t["task"] for t in out["tasks"]] == sorted(TASKS)  # formal family: core lowdim only
    d = out["descriptive"]
    assert set(d["explore"]) == {"can_mh_lowdim_full_epsilon_s42_B50k", "can_mh_lowdim_full_sample_s42_B50k"}
    assert set(d["explore"]["can_mh_lowdim_full_sample_s42_B50k"]["test"]) == {"ddim_100", "ddim_1", "ddpm_100"}
    assert d["image_pairs"]["pusht"]["complete"] and "stats" in d["image_pairs"]["pusht"] and "descriptive" in d["image_pairs"]["pusht"]["note"]
    assert "verdict" not in d["image_pairs"]["pusht"]
    assert d["official"]["square_mh_image_official_epsilon_s42_official"]["test"]["ddim_100"]["n"] == 50
    assert out["completeness"]["image"]["valid_test_records"] == 8 and out["completeness"]["explore"]["valid_test_records"] == 6
    # a record whose cell is not in the manifest is 'unexpected', never used
    stray = {"task_name": "pusht", "modality": "lowdim", "variant": "U", "head": "sample", "train_seed": 45, "budget_id": "B50k", "subset_sha256": "SpushtU"}
    _write(root, _rec(stray, "ddim", 100, "test", np.ones(100)))
    out2 = A.aggregate(cells, root, boot=100, seed=1)
    assert len(out2["records"]["unexpected"]) == 1 and out2["records"]["valid"] == out["records"]["valid"]


def test_counterexample_I_positive_with_zero_S_x0_yields_H1_not_supported(tmp_path):
    q = copy.deepcopy(GOOD)
    for t in TASKS:  # x0 loses nothing on either dataset; eps gains from M (so I = S_x0 - S_eps > 0 while S_x0 ~ 0)
        q[(t, "U", "sample")] = (0.9, 0.9); q[(t, "M", "sample")] = (0.9, 0.9)
        q[(t, "U", "epsilon")] = (0.9, 0.1); q[(t, "M", "epsilon")] = (0.9, 0.8)
    cells = _cells_dir(tmp_path); root = tmp_path / "res"; _matrix(root, q)
    out = A.aggregate(cells, root, boot=400, seed=2)
    for t in out["tasks"]:
        assert t["verdict"]["H1_data"] == "not_supported" and t["verdict"]["S_x0_equivalent_within_delta"] == "yes"
        assert t["verdict"]["I"] == "supported" and t["verdict"]["H2_head"] == "supported"  # I alone is not evidence for H1


def test_wide_interval_is_inconclusive(tmp_path):
    q = copy.deepcopy(GOOD)
    for t in TASKS:
        q[(t, "M", "sample")] = (0.9, 0.78)  # S_x0 ~ 0.10 with n=100 -> the 99.7% interval straddles 0.05
    cells = _cells_dir(tmp_path); root = tmp_path / "res"; _matrix(root, q, rng_seed=5)
    out = A.aggregate(cells, root, boot=400, seed=3)
    assert {t["verdict"]["H1_data"] for t in out["tasks"]} <= {"inconclusive", "supported"}
    assert any(t["verdict"]["H1_data"] == "inconclusive" for t in out["tasks"])


def test_screening_gate_failure_blocks_verdict_for_whole_task(tmp_path):
    cells = _cells_dir(tmp_path); root = tmp_path / "res"; _matrix(root, GOOD)
    ident = {"task_name": "pusht", "modality": "lowdim", "variant": "M", "head": "epsilon", "train_seed": 44, "budget_id": "B50k", "subset_sha256": "SpushtM"}
    _write(root, _rec(ident, "ddim", 100, "screen", np.zeros(32)))  # overwrite: Q = 0 < 0.5
    out = A.aggregate(cells, root, boot=100, seed=1)
    by = {t["task"]: t for t in out["tasks"]}
    assert by["pusht"]["verdict"] is None and not by["pusht"]["screening_passed"] and "stats" in by["pusht"]
    assert by["blockpush"]["verdict"] is not None


# ---------------------------------------------------------------- dispersion core
def test_dispersion_and_mixture_controls():
    c = D.check_controls(0)
    assert c["bimodal_separated"]["delta_bic_2_vs_1"] > c["unimodal_high_var"]["delta_bic_2_vs_1"]
    assert c["bimodal_separated"]["silhouette_pc1_split"] > 0.6 > c["unimodal_high_var"]["silhouette_pc1_split"]
    assert c["unimodal_high_var"]["mean_pairwise_l2"] > c["bimodal_separated"]["mean_pairwise_l2"] * 0.5
    assert c["bimodal_off_pc"]["delta_bic_2_vs_1"] < c["bimodal_separated"]["delta_bic_2_vs_1"]  # the documented blind spot


def test_dispersion_scaling_zero_variance_clip_and_nearest_demo():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(32, 8, 3)); s[..., 2] = 0.0
    std = np.array([1.0, 2.0, 0.0])
    d = D.dispersion(s, std)
    assert d["n_dims_used"] == 2 and d["n_dims_dropped"] == 1
    d2 = D.dispersion(s * np.array([1, 2, 1]), np.array([1.0, 4.0, 0.0]))
    assert math.isclose(d["mean_pairwise_l2"], d2["mean_pairwise_l2"], rel_tol=1e-9)
    with pytest.raises(ValueError):
        D.dispersion(s[:1], std)
    clipped = np.clip(s, -1, 1)
    c = D.clip_diagnostics(clipped, s)
    assert 0 < c["frac_coords_out_of_range_unclipped"] < 1 and c["mean_abs_diff_clip_vs_unclipped"] > 0
    assert D.clip_diagnostics(clipped, clipped)["frac_coords_out_of_range_unclipped"] == 0.0
    pool = np.concatenate([s[:4], rng.normal(size=(20, 8, 3)) + 5])
    nd = D.nearest_demo_distance(s, pool, np.array([1.0, 1.0, 1.0]))
    assert nd["nearest_demo_median"] > 0 and nd["nearest_demo_max"] >= nd["nearest_demo_mean"] >= 0
    assert D.nearest_demo_distance(s[:4], pool, np.ones(3))["nearest_demo_max"] == pytest.approx(0.0, abs=1e-5)
