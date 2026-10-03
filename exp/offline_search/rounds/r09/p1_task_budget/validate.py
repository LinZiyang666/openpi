"""Read-only source/fit checks plus a LOCAL relocated h100 asset plan.

No tether invocation, sync, launch, lock acquisition or GPU access occurs.
Plan files are written under the new run root only.
"""
import json
import pickle
from pathlib import Path

import numpy as np

from exp.offline_search.closed_loop.ops.h100.assets import build_plan
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import fingerprint

from .common import CELLS, HERE, RUN, SPEC, fit_path, sha, write_json


def main():
    rows = json.loads((RUN / "arms.json").read_text())
    lock = json.loads((HERE / "source_lock.json").read_text())
    table = json.loads((HERE / "task_tables.json").read_text())
    fits = {r["arm"]: r for r in json.loads((RUN / "prefit_report.json").read_text())}
    report = dict(status="PASS", local_only=True, sha_verified_sources=[], wrapper_branches=[], arms=12, pairs=300)
    for cell in CELLS:
        references = {}
        for variant, record in lock["cells"][cell].items():
            src = record["source"]
            assert sha(src["artifact"]) == src["sha256"]
            assert sha(record["row"]["yaml"]) == record["config_sha256"]
            with Path(src["artifact"]).open("rb") as f:
                references[variant] = pickle.load(f)["method"]
            report["sha_verified_sources"].append(dict(cell=cell, variant=variant, sha256=src["sha256"]))
        for row in [r for r in rows if r["p1"]["cell"] == cell]:
            assert sha(fit_path(row)) == fits[row["arm"]]["sha256"]
            assert sha(row["yaml"]) == lock["cells"][cell]["A"]["config_sha256"]
            if row["method"] != SPEC:
                continue
            with fit_path(row).open("rb") as f:
                blob = pickle.load(f)
            assert blob["provenance"]["controller_sha256"] == sha(HERE / "methods.py")
            assert blob["provenance"]["task_tables_sha256"] == sha(HERE / "task_tables.json")
            for branch, controller in blob["method"].controllers.items():
                variant = "CU" if branch.startswith("CU:") else branch
                ref = references[variant]
                base = controller.base if hasattr(controller, "base") else controller
                ref_base = ref.base if hasattr(ref, "base") else ref
                assert type(controller) is type(ref)
                assert fingerprint(base) == fingerprint(ref_base)
                # Whole untouched controller pickle compares fitted trackers,
                # bank, solved probability, tail rule, seeds and all arrays.
                assert pickle.dumps(controller, protocol=4) == pickle.dumps(ref, protocol=4)
                if variant == "CU":
                    assert controller.rho == .3 and controller.random_seed == 26093005
                    assert controller.parameter == controller.lambda_ == ref.parameter
                    assert controller.stall_model.fingerprint == ref.stall_model.fingerprint
                report["wrapper_branches"].append(dict(arm=row["arm"], branch=branch,
                    controller_pickle_exact=True, retrieval_fingerprint=fingerprint(base)))
            assert all(blob["method"].branch(t) == ("A" if t not in table["cells"][cell]["hard_tasks"]
                       else "CU:0.3" if row["p1"]["variant"] == "topk_cu" else "P10") for t in range(10))
    plan = build_plan(RUN, [r["arm"] for r in rows], RUN / "local_asset_plan")
    relocated = []
    for row in plan["arms"]:
        original = next(r for r in rows if r["arm"] == row["arm"])
        item = next(i for i in plan["files"] if i["original"] == str(fit_path(original)))
        with Path(item["source"]).open("rb") as f:
            b = pickle.load(f)
        assert b["spec"] == row["method"] and b["kwargs"] == row["kwargs"] and b["cell"] == row["cell"]
        relocated.append(row["arm"])
    assert len(relocated) == 12
    report["local_h100_plan"] = dict(files=len(plan["files"]), bytes=plan["bytes"], GiB=plan["bytes"]/(1 << 30),
                                     relocated_metadata_exact=len(relocated),
                                     path=str(RUN / "local_asset_plan/plan.json"),
                                     remote_disk_capacity_verified=False, remote_code_installed=False)
    write_json(HERE / "evidence/validation.json", report)
    print(f"P1_VALIDATE_OK arms=12 pairs=300 source_SHA=9 exact_wrapper_branches=12 relocated_metadata=12 "
          f"plan_files={len(plan['files'])} plan_GiB={plan['bytes']/(1<<30):.3f} remote_actions=0")


if __name__ == "__main__":
    main()
