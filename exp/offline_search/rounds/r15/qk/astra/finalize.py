"""CPU-only consistency audit and final, content-addressed exploration freeze.

Does not train, evaluate a policy, launch a process or initialize CUDA.
Writes only inside this script's directory. Source inputs are read-only.
"""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from exp.offline_search.rounds.r15.qk.astra.prepare import HERE, DERIVED, REPO, sha


DOCS = ["DATA.md", "PROTOTYPE.md", "COST.md", "SPEC.md", "GATES.md",
        "PREDICTION.md", "REQUESTS.md", "HANDBACK.md", "OFFLINE_GATES.md"]
METHOD = "symmetric_cnn_action_cpu"


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON scalar in {path}: {value}")
    return json.loads(path.read_text(), parse_constant=reject)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def audit():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
    allowed = set(range(10, 22)) | set(range(54, 66))
    affinity = os.sched_getaffinity(0)
    assert affinity and affinity <= allowed
    torch.set_num_threads(4)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "cpu_affinity": sorted(affinity), "gpu_used": False,
              "syntax": [], "documents": DOCS, "cells": {}}
    for path in sorted(HERE.glob("*.py")):
        ast.parse(path.read_text(), filename=str(path))
        report["syntax"].append(path.name)
    for name in DOCS:
        assert (HERE / name).stat().st_size > 100
    for path in sorted(HERE.rglob("*.json")):
        if "cuda_cache" not in path.parts:
            read_json(path)
    verification = read_json(HERE / "verification.json")
    assert verification["passed"]
    analysis = read_json(HERE / "analysis.json")
    gates = read_json(HERE / "gate_report.json")
    assert not gates["offline_pass"] and not gates["online_authorized"]
    csv_rows = list(csv.DictReader((HERE / "metrics.csv").open()))
    expected_rows = sum(len(c["methods"]) for c in analysis.values())
    assert len(csv_rows) == expected_rows
    for row in csv_rows:
        m = analysis[row["cell"]]["methods"][row["method"]]
        assert math.isclose(float(row["balanced_err10"]), m["episode_task_balanced_err10"])
        assert math.isclose(float(row["err10"]), m["err10"])

    for suite, size, root in (("l10", 50, "r12_devlooks_50L"),
                              ("l10", 200, "r12_devlooks_50L"),
                              ("spatial", 50, "r12_devlooks_50S_local")):
        cell = f"pi05_{suite}_{size}"
        path = HERE / cell
        tr_path = DERIVED / cell / "train.npz"
        assert sha(tr_path) == verification["cells"][cell]["train_npz_sha256"]
        tr = np.load(tr_path)
        meta = read_json(DERIVED / cell / "train_meta.json")
        for kind in ("subset", "artifact"):
            assert sha(Path(meta[kind])) == meta[kind + "_sha256"]
        subset = read_json(Path(meta["subset"]))
        assert set(tr["episode"].tolist()) == set(sum(subset["episode_ids_by_task"].values(), []))
        assert len(np.unique(tr["episode"])) == size
        dp = DERIVED / f"dev_pi05_{suite}_{root}"
        with np.load(dp / "dev.npz") as full:
            keep = np.load(dp / f"unseen_{size}.npy")
            dev = {k: full[k][keep] for k in ("uid", "task_id", "init", "label_valid")}
        library = Path(meta["fit_info"]["parent"])
        episodes = read_json(library / "episodes.json")
        ep_ids = np.load(library / "episode.npy", mmap_mode="r")
        selected = set(tr["episode"].tolist())
        own_inits = {(e["task_id"], e["orig_init_state_idx"]) for e in episodes
                     if int(ep_ids[e["start"]]) in selected}
        assert not any((int(t), int(i)) in own_inits for t, i in zip(dev["task_id"], dev["init"]))
        assert len(set(dev["uid"])) == analysis[cell]["episodes"]
        assert int(dev["label_valid"].sum()) == analysis[cell]["labelled_looks"]
        lo = read_json(path / "loeo_results.json")
        assert lo["own_episode_donors"] == 0
        assert lo["rows"] == len(tr["rows"]) and lo["episodes"] == size
        for method in lo["methods"]:
            with np.load(path / f"loeo_{method['name']}.npz") as a:
                top = a["top"]
                assert not np.any(tr["episode"][top] == tr["episode"][:, None])
                assert np.all(tr["task_id"][top] == tr["task_id"][:, None])
                assert np.isfinite(a["action"]).all() and np.isfinite(a["err10"]).all()
        for name, method in analysis[cell]["methods"].items():
            filename = "eval_reference.npz" if name == "deployed_key_geometry" else f"eval_{name}.npz"
            with np.load(path / filename) as a:
                assert len(a["top"]) == len(dev["uid"])
                assert np.all(tr["task_id"][a["top"]] == dev["task_id"][:, None])
                assert np.isfinite(a["action"]).all()
                assert np.isfinite(a["err10"][dev["label_valid"]]).all()
                assert np.isclose(a["err10"][dev["label_valid"]].mean(), method["err10"])
        detail = {"train_rows": len(tr["rows"]), "train_episodes": size,
                  "unseen_episodes": len(set(dev["uid"])), "unseen_looks": len(dev["uid"]),
                  "methods_checked": len(analysis[cell]["methods"]),
                  "loeo_own_episode_donors": 0, "library_only_provenance": True,
                  "kernel_reference": meta["fit_info"]["kref"]}
        assert detail["kernel_reference"] == (5 if size == 50 else 8)
        if suite == "l10":
            train = read_json(path / "cnn_action_cpu_train.json")
            assert train["guard"]["device"] == "cpu"
            assert [x["epoch"] for x in train["trace"]] == list(range(1, 81))
            checkpoint = torch.load(path / "cnn_action_cpu.pt", map_location="cpu", weights_only=True)
            assert checkpoint["epochs"] == 80 and checkpoint["parameters"] == 856299
            assert all(torch.isfinite(v).all().item() for v in checkpoint["state_dict"].values())
            lat = read_json(HERE / f"latency_cpu_extra_{size}.json")
            assert lat["device"] == "cpu" and not lat["gpu_used"]
            primary = next(r for r in lat["results"] if r["name"] == METHOD and
                           r["mode"] == "torchscript" and r["cpu_threads"] == 1)
            assert primary["end_to_end"]["n"] == 400
            assert primary["max_abs_export_parity"] < 1e-4
            g = gates["cells"][cell]
            assert not g["b_dev"]["pass"] and not g["offline_pass"]
            assert g["local_cpu_cost"]["pass"]
            detail.update(cpu_epochs=80, cpu_fit_seconds=train["fit_seconds"],
                          local_median_ms=primary["end_to_end"]["median_ms"],
                          balanced_b_dev_ratio=g["b_dev"]["ratio"])
        report["cells"][cell] = detail

    cost = read_json(HERE / "cost_counts.json")
    assert len(cost["arms"]) == 52 and all(a["ledger_reconciled"] for a in cost["arms"])
    for source, digest in cost["input_sha256"].items():
        assert sha(Path(source)) == digest, source
    for arm in cost["arms"]:
        p = arm["at_gate_epsilon"]
        cv = arm["owner_cv"]
        assert math.isclose(p["new_ir"], (cv / 10 * p["V"] + p["M"]) / p["N"])
    report.update(cost_arms_reconciled=52, cost_source_hashes_checked=len(cost["input_sha256"]),
                  metrics_csv_rows=len(csv_rows), passed=True,
                  limitations=["Key-only prototype; full five-layer plugin is a sol implementation request.",
                               "CPU export parity measured on four tasks; all-task runtime export remains pending.",
                               "Local CPU cost is not an H100 host measurement.",
                               "B-dev accuracy fails; online work is not authorized."])
    write_json(HERE / "final_verification.json", report)
    print(json.dumps({k: report[k] for k in ("passed", "cells", "cost_arms_reconciled", "metrics_csv_rows")}))


def freeze():
    audit_report = read_json(HERE / "final_verification.json")
    assert audit_report["passed"]
    assert (HERE / "HANDBACK_INTERIM.md").exists()
    assert (HERE / "HANDBACK.md").read_text().startswith("# Final handback")
    files = {}
    for root in (HERE, DERIVED):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "cuda_cache" in path.parts:
                continue
            if path.name in ("FINAL_MANIFEST.json", "FINAL_MANIFEST.sha256"):
                continue
            files[str(path)] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    contracts = [REPO / "exp/offline_search/rounds/r15/BRIEF.md",
                 HERE.parent / "OWNER_ADDENDUM.md",
                 Path("/home/weiland/.claude/jobs/a607dd74/tmp/r15/astra_qk.md")]
    manifest = {"frozen_utc": datetime.now(timezone.utc).isoformat(),
                "frozen_chicago": datetime.now(ZoneInfo("America/Chicago")).isoformat(),
                "scope": "Exploration, prototype evidence and design; before any new-method closed-loop outcome.",
                "online_authorized": False, "offline_pass": False,
                "write_roots": [str(HERE), str(DERIVED)],
                "excluded": ["cuda_cache runtime cache", "FINAL_MANIFEST.json and its checksum (self-reference)"],
                "contract_sha256": {str(p): sha(p) for p in contracts}, "files": files}
    path = HERE / "FINAL_MANIFEST.json"
    write_json(path, manifest)
    digest = sha(path)
    (HERE / "FINAL_MANIFEST.sha256").write_text(f"{digest}  FINAL_MANIFEST.json\n")
    print(json.dumps({"frozen_files": len(files), "manifest_sha256": digest,
                      "frozen_chicago": manifest["frozen_chicago"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        audit()
