"""Retain a CPU fixture pipeline sample; never a model/benchmark quality claim."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        p.error("set CUDA_VISIBLE_DEVICES='' for this CPU-only generator")
    a.output.mkdir(parents=True, exist_ok=False)
    from exp.r17_integration.tests.debug.fixtures import binding, captured
    from exp.r17_integration.debug.capture_bridge import export_h5
    from exp.r17_integration.debug.reader_profiles import CaptureReader
    from exp.r17_integration.debug.analysis_profiles import decision_profile, physical_profile, write_report
    from exp.r17_integration.search.build_library import build_library, save_stock_artifact
    from exp.r17_integration.search.fit import fit_recipe, save_fitted_recipe, load_fitted_recipe
    from exp.r17_integration.search.analyze import summarize
    from exp.r17_integration.debug.validate import file_sha
    from exp.r17_integration.common.contracts import RequestContext
    from exp.r17_integration.common.query_view import QueryView
    b = binding()
    path, episodes = captured(a.output, b, episodes=6, decisions=4)
    h5 = export_h5(path, a.output / "captured.h5")
    store = build_library([h5], a.output / "library", b, episodes, key_fields=b["model"]["feature_spec"]["fields"])
    save_stock_artifact(store, a.output / "library.pkl")
    recipe, receipt = fit_recipe(store, task_names=["reach-v3"], corrector=True, early=False,
                                 guard=False, commit_horizon=5, k=4, kref=3)
    save_fitted_recipe(recipe, receipt, a.output / "fit")
    restored = load_fitted_recipe(a.output / "fit", expected_binding=b)
    q = QueryView(RequestContext("fixture", b["benchmark_protocol"], "reach-v3", 0, 1),
                  {"vision_0": store.array("key_vision_0")[0]}, store.array("state")[0],
                  hist_has_vision=(True,), hist_hit=(0,))
    expected, loaded = recipe.query(q), restored.query(q)
    assert expected.action.tobytes() == loaded.action.tobytes()
    decisions = []
    for row, arrays, keys in CaptureReader(path).decisions():
        row.pop("cost_ms", None)  # Synthetic fixture timings are not measured costs.
        decisions.append(row)
    journal = [dict(ep, accepted=True, status="done", error=None) for ep in episodes]
    report = summarize([ep["task_uid"] for ep in episodes], journal, decisions,
                       task_groups={"reach-v3": "fixture_only"}, binding=b)
    report["quality_claim"] = "synthetic fixture outcomes; no policy SR or GPU pricing"
    _, arrays, _ = next(CaptureReader(path).decisions())
    write_report(report, a.output / "analysis", decision_cards=[decision_profile(b, arrays)],
                 physical_cards=[physical_profile({"capabilities": {}}, {})])
    proof = dict(schema="r17.cpu_pipeline_sample.v1", cpu_only=True, synthetic=True,
                 decisions=len(decisions), library_episodes=len(episodes), actual_gpu_executed=False,
                 exact_fit_reload=True, corrector_loeo_rows=len(recipe.fit_evidence["loeo"]),
                 hashes={"h5": file_sha(h5), "stock": file_sha(a.output / "library.pkl")})
    (a.output / "SAMPLE_RECEIPT.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
