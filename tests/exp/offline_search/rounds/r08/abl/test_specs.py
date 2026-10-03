import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from exp.offline_search.rounds.r08.abl.make_arms import CELLS, artifact, make_specs, references
from exp.offline_search.rounds.r08.abl.validate import assert_source_fields, state_differences


def test_reference_convention_is_paper_ab():
    path = Path("exp/offline_search/rounds/r06/ops/paper_ab.py")
    fn, = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "arms"]
    scope = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), scope)
    for model, suite, scale in CELLS:
        a, b = scope["arms"](model, f"{'sp' if suite == 'spatial' else suite}_{scale}")
        assert references(model, suite, scale) == dict(A=a, B=b)


def test_coverage_and_verbatim_non_guard_fields(tmp_path):
    sources = {}
    for cell in CELLS:
        model, suite, scale = cell
        sources[cell] = dict(source_row=dict(name="original", model=model, suite=suite, mode="plugin",
            method="original:Judge", kwargs=dict(base_kwargs=dict(lib="current" if scale == 50 else "big", kref=5 if scale == 50 else 8)),
            plugin_args=["--os-blind", "--os-fit-artifact", "source.pkl", "--os-policy-tail"],
            full_model=True, cost_ledger=True, client_overrides=dict(replan_steps=5),
            future_field=dict(preserve=[1, 2, 3])))
    before = copy.deepcopy(sources)
    rows, provenance = make_specs(sources, tmp_path)
    assert sources == before and len(rows) == 24
    assert len({r["name"] for r in rows}) == 24
    assert sum("onlynp" in r["name"] for r in rows) == 8
    assert sum("onlynp" not in r["name"] for r in rows) == 16
    for row, prov in zip(rows, provenance):
        src = prov["source_row"]
        restored = copy.deepcopy(row)
        restored["name"], restored["method"] = src["name"], src["method"]
        disabled = restored["kwargs"].pop("disabled_guards")
        if "onlynp" in row["name"]:
            assert disabled == ["stuck", "terminal", "overtime"]
        else:
            assert row["name"].endswith("_500") and len(disabled) == 1
        args = restored["plugin_args"]
        args[args.index("--os-fit-artifact") + 1] = artifact(src)
        assert json.dumps(restored) == json.dumps(src)
        assert artifact(row) == str(tmp_path / "fits" / (row["name"] + ".pkl"))


@pytest.mark.parametrize("mutation", ["kref", "arg_order", "field_presence"])
def test_non_guard_mutations_are_rejected(mutation):
    src = dict(name="B", method="base:B", kwargs=dict(base_kwargs=dict(kref=8)),
               plugin_args=["--os-blind", "--os-fit-artifact", "B.pkl"], cost_ledger=True)
    row = copy.deepcopy(src)
    row.update(name="R8", method="r8:B")
    row["kwargs"]["disabled_guards"] = ["stuck"]
    row["plugin_args"][2] = "R8.pkl"
    prov = dict(source_row=src, disabled_guards=["stuck"])
    assert assert_source_fields(row, prov)["non_guard_differences"] == []
    if mutation == "kref":
        row["kwargs"]["base_kwargs"]["kref"] = 5
    elif mutation == "arg_order":
        row["plugin_args"] = ["--os-fit-artifact", "R8.pkl", "--os-blind"]
    else:
        row.pop("cost_ledger")
    with pytest.raises(AssertionError):
        assert_source_fields(row, prov)


def test_fitted_array_comparison_is_byte_exact():
    # Equal numeric values with different dtypes or different signed zeros fail.
    assert state_differences(np.array([0., np.nan]), np.array([0., np.nan])) == []
    assert state_differences(np.array([0.]), np.array([-0.])) == ["method"]
    assert state_differences(np.array([1.], np.float32), np.array([1.], np.float64)) == ["method"]
    assert state_differences({"threshold": 2}, {"threshold": 3}) == ["method/threshold"]
