"""Compact final evidence/inventory check; CPU and local files only."""
import json
from pathlib import Path

from .common import HERE, RUN, fit_path, sha, write_json


def main():
    rows = json.loads((RUN / "arms.json").read_text())
    records = json.loads((RUN / "prefit_report.json").read_text())
    assert len(rows) == len(records) == 12
    for record in records:
        assert sha(record["artifact"]) == record["sha256"]
    replay = json.loads((HERE / "evidence/replay.json").read_text())
    validation = json.loads((HERE / "evidence/validation.json").read_text())
    assert replay["status"] == validation["status"] == "PASS" and len(replay["inits"]) == 30
    totals = {k: sum(c[k] for c in replay["checks"]) for k in (
        "episodes", "decisions", "branch_exact_decisions", "logged_action_checked", "logged_action_exact",
        "calls", "policy_tails")}
    assert totals["logged_action_checked"] == totals["logged_action_exact"]
    assert max(c["logged_action_max_abs"] for c in replay["checks"]) == 0
    manifest = json.loads((RUN / "manifests/discovery300.json").read_text())
    assert manifest == [[t, i] for t in range(10) for i in range(30)]
    assert len(list((RUN / "config").glob("r9p1_*.yaml"))) == 12
    assert len(list((RUN / "config").glob("matrix_*.yaml"))) == 12
    assert len(list((RUN / "fits").glob("*.pkl"))) == 6
    pytest = (HERE / "evidence/pytest.txt").read_text().strip()
    assert "22 passed" in pytest and "failed" not in pytest
    idempotence = (HERE / "evidence/idempotence.txt").read_text()
    assert "P1_IDEMPOTENCE_OK" in idempotence
    pred = json.loads((HERE / "predictions.json").read_text())["arms"]
    assert len(pred) == 12
    paired = json.loads((HERE / "evidence/paired_simulator.json").read_text())
    passed_bars = sum(c["proposal_point_bar_vs_CU"] is True for cell in paired["cells"].values()
                      for c in cell["comparisons"] if c["reference"] == "CU")
    artifacts = {str(p.relative_to(HERE)): sha(p) for p in HERE.iterdir() if p.is_file() and p.suffix in (".py", ".json")}
    write_json(HERE / "evidence/final_audit.json", dict(status="PASS", replay=totals, code_artifact_sha256=artifacts,
        config_yamls=12, matrices=12, wrapper_fits=6, pairs=300, simulator_CU_bars_passed=passed_bars))
    print("P1_INVENTORY_OK arms=12 config_yamls=12 matrices=12 wrappers=6 pairs=300 fit_SHA=12")
    print("P1_REPLAY_TOTALS " + json.dumps(totals, sort_keys=True) + " logged_max_abs=0.0")
    print("P1_SIMULATOR_BAR_CHECK top3_treatments_passing_vs_CU=" + str(passed_bars) + "/6")
    print("P1_REMOTE_CODE_SHA init=" + sha(HERE / "__init__.py") + " methods=" + sha(HERE / "methods.py"))
    print(pytest.splitlines()[-1])


if __name__ == "__main__":
    main()
