"""K=3 export / runner gate / aggregation: rules share one deployment form."""

from __future__ import annotations

import json
import math
import pathlib
import types

import numpy as np
import pytest
import yaml

from exp.gate_threshold_pareto.run_gtp import validate_arms
from exp.rit_pareto import aggregate_rit, export_k3, rit_k
from exp.rit_pareto.rit_k import tier_cost
from openpi.cache.config import load_cache_config


def _table(path: pathlib.Path, n: int = 500, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.6, 1.0, size=n)
    base = 12.0 * (1.0 - s) + 1.0
    with path.open("w") as f:
        for i in range(n):
            f.write(json.dumps({
                "episode_id": f"ep{i % 150}", "task_id": i % 10, "init_idx": i % 50, "step_idx": i // 150,
                "split": "fit", "s": float(s[i]), "v": 1.0, "k_eff": 5, "winner_id": "w",
                "y_tau7": float(0.7 * base[i] + rng.gamma(2.0, 0.5)),
                "y_tau10": float(base[i] + rng.gamma(2.0, 0.6)),
                "y_tau5": float(0.5 * base[i] + rng.gamma(2.0, 0.4)),
                "ref_mode": "tau1", "episode_success": True,
            }) + "\n")


def _args(tmp_path, table, **over):
    base = dict(suite="libero_spatial", table=str(table), library_pkl="/srv/lib.pkl", library_pkl_local="",
                ref_mode="tau1", alpha=0.05, target_ir="20,40,60,80,95", gst_step=20, gst_max_sum=80,
                out_dir=str(tmp_path / "out"))
    base.update(over)
    return types.SimpleNamespace(**base)


def test_gst_grid_is_the_34_cell_step20_simplex():
    cells = export_k3.gst_cells()
    assert len(cells) == 34 and (0, 0, 0) not in cells
    assert all(sum(c) <= 80 and all(x % 20 == 0 for x in c) for c in cells)
    assert (80, 0, 0) in cells and (0, 0, 80) in cells and (20, 20, 40) in cells


def test_gst_cuts_follow_the_descending_quantile_convention():
    scores = np.linspace(0.5, 1.0, 100)
    cuts = export_k3.gst_cuts(scores, (0.2, 0.2, 0.2))
    desc = np.sort(scores)[::-1]
    assert cuts == [float(desc[19]), float(desc[39]), float(desc[59])]
    cuts = export_k3.gst_cuts(scores, (0.2, 0.0, 0.2))
    assert math.isinf(cuts[1]) and cuts[2] == float(desc[39])


def test_export_writes_both_rules_with_valid_threshold_judges(tmp_path):
    table = tmp_path / "table.jsonl"
    _table(table)
    rec = export_k3.export(_args(tmp_path, table))
    assert rec["layers"]["rit"]["n_arms"] == 5
    assert rec["layers"]["gst"]["n_arms"] + len(rec["gst"]["skipped"]) == 34
    assert rec["gst"]["skipped"] == {}  # continuous synthetic scores never coincide
    for rule in ("rit", "gst"):
        matrix = yaml.safe_load(pathlib.Path(rec["layers"][rule]["matrix"]).read_text())
        assert len(matrix["arms"]) == rec["layers"][rule]["n_arms"]
        for row in matrix["arms"]:
            cfg = load_cache_config(row["yaml"])
            cp1 = cfg.checkpoints["cp1"]
            assert cp1.judge.type == "threshold" and cp1.gate.type == "always_search"
            tiers = cp1.judge.warm_tiers or []
            cuts = [cp1.judge.threshold] + [t["threshold"] for t in tiers]
            assert all(b < a for a, b in zip(cuts, cuts[1:]))
            assert [t["start_t"] for t in tiers] == [x for x in (0.3, 0.5) if x in [t["start_t"] for t in tiers]]
            assert cfg.backend.in_memory.preload_path == "/srv/lib.pkl"
        # runner gate: the declared ladder passes, the GTP default rejects the tiers
        assert validate_arms(matrix["arms"], phase="eval", eval_gate="always_search", warm_tiers=(0.3, 0.5))
        with pytest.raises(SystemExit, match="warm tier present"):
            validate_arms(matrix["arms"], phase="eval", eval_gate="always_search")
    # RIT arms: single delta per arm, nested cuts, gap within tolerance
    for arm, r in rec["rit"]["arms"].items():
        th = [r["thetas"][t.name] for t in rit_k.K3_TIERS]
        assert th[0] >= th[1] >= th[2] and abs(r["ir_gap"]) <= rit_k.IR_MAX_GAP
    # GST cell (80,0,0): FULL-only arm
    full_only = rec["gst"]["arms"]["k3_sp_gst_f80w00v00"]
    assert full_only["deployed_tiers"] == ["full"]
    warm_only = rec["gst"]["arms"]["k3_sp_gst_f00w00v80"]
    assert warm_only["deployed_tiers"] == ["warm05"]
    cfg = load_cache_config(warm_only["yaml"])
    assert cfg.checkpoints["cp1"].judge.threshold == 2.0  # FULL never fires
    with pytest.raises(SystemExit, match="must be empty"):
        export_k3.export(_args(tmp_path, table))


def test_export_refuses_a_table_without_the_extra_tier(tmp_path):
    table = tmp_path / "t.jsonl"
    _table(table)
    rows = [json.loads(line) for line in table.open()]
    for r in rows:
        del r["y_tau5"]
    table.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit, match="y_tau5"):
        export_k3.export(_args(tmp_path, table))


def test_runner_gate_rejects_wrong_tier_order_and_non_decreasing_cuts(tmp_path):
    base = yaml.safe_load((export_k3.libs.REPO_ROOT / export_k3.libs.TEMPLATE["libero_spatial"]).read_text())

    def arm(name, judge):
        doc = json.loads(json.dumps(base))
        doc["checkpoints"]["cp1"]["judge"] = judge
        doc["checkpoints"]["cp1"]["gate"] = {"type": "always_search"}
        p = tmp_path / f"{name}.yaml"
        p.write_text(yaml.safe_dump(doc, sort_keys=False))
        return [{"arm": name, "yaml": str(p)}]

    ok = arm("ok", {"type": "threshold", "threshold": 0.99,
                    "warm_tiers": [{"threshold": 0.97, "start_t": 0.3}, {"threshold": 0.95, "start_t": 0.5}]})
    assert validate_arms(ok, phase="eval", eval_gate="always_search", warm_tiers=(0.3, 0.5))
    bad_order = arm("bo", {"type": "threshold", "threshold": 0.99,
                           "warm_tiers": [{"threshold": 0.97, "start_t": 0.5}, {"threshold": 0.95, "start_t": 0.3}]})
    with pytest.raises(SystemExit, match="ordered subset"):
        validate_arms(bad_order, phase="eval", eval_gate="always_search", warm_tiers=(0.3, 0.5))
    # non-decreasing cuts are already refused by the config loader, before the runner gate
    from openpi.cache.config import ConfigValidationError

    bad_cuts = arm("bc", {"type": "threshold", "threshold": 0.95,
                          "warm_tiers": [{"threshold": 0.97, "start_t": 0.3}]})
    with pytest.raises(ConfigValidationError, match="strictly decreasing"):
        validate_arms(bad_cuts, phase="eval", eval_gate="always_search", warm_tiers=(0.3, 0.5))
    foreign = arm("fo", {"type": "threshold", "threshold": 0.99,
                         "warm_tiers": [{"threshold": 0.97, "start_t": 0.7}]})
    with pytest.raises(SystemExit, match="ordered subset"):
        validate_arms(foreign, phase="eval", eval_gate="always_search", warm_tiers=(0.3, 0.5))


def test_aggregate_prices_the_05_tier_and_labels_gst_cells(tmp_path):
    journal = [{"yaml_id": "k3_sp_gst_f20w20v20", "task_uid": "a", "attempt": 1, "status": "done", "accepted": True}]
    steps = [
        {"yaml_id": "k3_sp_gst_f20w20v20", "task_uid": "a", "attempt": 1, "hit_type": "FULL_HIT", "start_t": None},
        {"yaml_id": "k3_sp_gst_f20w20v20", "task_uid": "a", "attempt": 1, "hit_type": "WARM_START", "start_t": 0.3},
        {"yaml_id": "k3_sp_gst_f20w20v20", "task_uid": "a", "attempt": 1, "hit_type": "WARM_START", "start_t": 0.5},
        {"yaml_id": "k3_sp_gst_f20w20v20", "task_uid": "a", "attempt": 1, "hit_type": "MISS", "start_t": None},
    ]
    (tmp_path / "journal.jsonl").write_text("".join(json.dumps(r) + "\n" for r in journal))
    (tmp_path / "per_step.jsonl").write_text("".join(json.dumps(r) + "\n" for r in steps))
    res = aggregate_rit.aggregate(tmp_path)["k3_sp_gst_f20w20v20"]
    exp_cost = tier_cost("FULL_HIT", None) + tier_cost("WARM_START", 0.3) + tier_cost("WARM_START", 0.5) + tier_cost("MISS", None)
    assert res["ir_percent"] == pytest.approx(100.0 * exp_cost / (4 * tier_cost("MISS", None)))
    assert res["counts"]["WARM_START"] == 2 and res["counts"]["WARM_START@0.5"] == 1
    assert res["gst_cell"] == (20, 20, 20) and res["label"] == "20/20/20" and res["target_ir"] is None


def test_plot_k3_renders_with_reference(tmp_path):
    rit = {f"k3_sp_rit_ir{t}": {"ir_percent": float(t) + 2, "success_rate": 0.6 + t / 300, "target_ir": float(t),
                                "label": f"IR={t}"} for t in (20, 40, 60)}
    gst = {f"k3_sp_gst_f{f:02d}w00v00": {"ir_percent": 100 - f, "success_rate": 0.9 - f / 200, "target_ir": None,
                                         "label": f"{f}/0/0"} for f in (20, 40, 60)}
    ref = {f"rit_sp_ng_ir{t}": {"ir_percent": float(t) + 5, "success_rate": 0.55 + t / 300, "target_ir": float(t)} for t in (20, 60)}
    paths = aggregate_rit.plot_suite("libero_spatial", {"RIT-K3": rit, "GST-K3": gst}, tmp_path / "fig", None,
                                     reference={"K2": ref}, stem="pareto_k3", series_prefix="")
    assert all(p.is_file() and p.stat().st_size > 0 for p in paths)
