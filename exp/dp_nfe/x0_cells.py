"""Generate the frozen cell manifest (one yaml per training cell) of the x0-head experiment from
``exp/dp_nfe/config/x0_multimodal/tasks.yaml``, the subset manifests written by ``mode_filter_datasets.py`` and the frozen
training-pool normalisers written by ``x0_normalizer.py``::

    python -m exp.dp_nfe.x0_cells --tasks exp/dp_nfe/config/x0_multimodal/tasks.yaml --subsets <data_root>/subsets \
        --budget-lowdim 50000 --budget-image 20000 --out exp/dp_nfe/config/x0_multimodal/cells [--explore] [--image]

cell_id = <task>_<modality>_<variant>_<head>_s<seed>_<budget_id>. U/M cells point at the exported subsets, share the
task's common held-out subset (``heldout_path``) and the task's frozen normaliser (``normalizer_path`` +
``normalizer_sha256``; a task whose normaliser file is missing is listed under ``skipped`` with that reason). Image cells
use the subset manifest's ``image_exports`` (square: hdf5 exported from the image source by demo id; pusht: the zarr
subsets that already carry ``data/img``) and never the lowdim files. The identity block of every cell carries the subset
/ held-out / manifest / normaliser hashes so the trainer can bind them into the checkpoint.

``${RAW_ROOT}`` in tasks.yaml is replaced by ``--raw-root`` (default ``$DP_DATA/data``), so the same table serves both hosts.
``cells_manifest.json`` enumerates every *expected* cell of the matrix: ``cells`` (emitted, grouped by arm) and
``skipped`` (task, arm, reason), so completeness is judged against the plan and not against the files that happen to
exist.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, Optional

import yaml

HEADS = ("epsilon", "sample")
EXT = {"zarr": ".zarr", "npy": "", "mjl": "", "hdf5": ".hdf5"}


def _subset_paths(subsets: pathlib.Path, task: str, native: str, modality: str) -> Optional[Dict[str, object]]:
    """Paths + hashes of one task's U/M/held-out exports for ``modality`` (``lowdim`` uses ``exports``, ``image``
    uses ``image_exports``); ``None`` without a manifest; ``{"usable": False, "reason"}`` when no U/M pair exists."""
    ext = EXT[native]
    man = subsets / f"{task}_subset_manifest.json"
    if not man.exists():
        return None
    m = json.loads(man.read_text())
    if not m["selection"]["usable"]:
        return {"usable": False, "reason": m["selection"]["reason"]}
    if modality == "lowdim":
        paths = {n: str(subsets / f"{task}_{n}{ext}") for n in ("U", "M", "heldout")}
        hashes = m["exports"]
    else:
        ie = m.get("image_exports") or {}
        if not all(n in ie for n in ("U", "M", "heldout")):
            return {"usable": False, "reason": m.get("image_note", "no image exports in the subset manifest")}
        if task == "pusht":
            paths = {n: str(subsets / f"{task}_{n}{ext}") for n in ("U", "M", "heldout")}
        else:
            paths = {n: str(subsets / f"{task}_image_{n}.hdf5") for n in ("U", "M", "heldout")}
        hashes = ie
    return {"usable": True, **paths, "hashes": {n: hashes[n] for n in ("U", "M", "heldout", "trainpool")},
            "manifest_path": str(man), "manifest_sha256": _sha(man), "n": m["selection"]["n"], "u_label": m["selection"]["u_label"],
            "kitchen_set": (m.get("kitchen") or {}).get("set")}


def _sha(path: pathlib.Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalizer(subsets: pathlib.Path, task: str, modality: str, pool_sha: str) -> Optional[Dict[str, str]]:
    """``{"path", "sha256"}`` of the frozen training-pool normaliser of (task, modality), None when missing."""
    p = subsets / f"{task}_{modality}_normalizer.pt"
    side = pathlib.Path(str(p) + ".json")
    if not (p.is_file() and side.is_file()):
        return None
    metadata = json.loads(side.read_text())
    if metadata.get("source", {}).get("dataset_sha256") != pool_sha:
        return None
    return {"path": str(p), "sha256": metadata["sha256"]}


def resolve_raw_root(tasks: dict, raw_root: str) -> dict:
    """Return a copy of the task table with every ``${RAW_ROOT}`` replaced by ``raw_root`` (the host's raw data dir)."""
    text = json.dumps(tasks).replace("${RAW_ROOT}", raw_root.rstrip("/"))
    return json.loads(text)


def make_cells(tasks: dict, subsets: pathlib.Path, out: pathlib.Path, budget_lowdim: int, budget_image: int,
               explore: bool, image: bool) -> dict:
    """Write one yaml per cell into ``out`` and ``out/cells_manifest.json``; returns the manifest dict
    ``{"cells": [...], "by_arm": {"core": [...], "explore": [...], "image": [...]}, "skipped": [...], "budgets": {...},
    "train_seeds": [...]}``. Core cells: every core task x U/M x head x ``tasks.train_seeds``; explore: full official
    datasets x head x first seed (descriptive only); image: U/M x head x first seed."""
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"cells": [], "by_arm": {"core": [], "explore": [], "image": []}, "skipped": [],
                "budgets": {"lowdim": budget_lowdim, "image": budget_image}, "train_seeds": list(tasks["train_seeds"]),
                "budgets_by_task": tasks.get("frozen_budgets", {})}
    defaults = {"lowdim": budget_lowdim, "image": budget_image}

    def budget_for(name, modality):
        budget = int(manifest["budgets_by_task"].get(modality, {}).get(name, defaults[modality]))
        if budget <= 0 or budget % 1000:
            raise ValueError("frozen budgets must be positive multiples of 1000")
        return budget, f"B{budget // 1000}k"

    def emit(cell, arm):
        (out / f"{cell['cell_id']}.yaml").write_text(yaml.safe_dump(cell, sort_keys=False))
        manifest["cells"].append(cell["cell_id"]); manifest["by_arm"][arm].append(cell["cell_id"])

    for name, t in tasks["core"].items():
        budget_lowdim, bid_l = budget_for(name, "lowdim")
        sp = _subset_paths(subsets, name, t["native"], "lowdim")
        if sp is None or not sp["usable"]:
            manifest["skipped"].append({"task": name, "arm": "core", "reason": (sp or {}).get("reason", "no subset manifest")}); continue
        nz = _normalizer(subsets, name, "lowdim", sp["hashes"]["trainpool"])
        if nz is None:
            manifest["skipped"].append({"task": name, "arm": "core", "reason": f"frozen normalizer {name}_lowdim_normalizer.pt missing or fitted on a different pool"}); continue
        for variant in ("U", "M"):
            for head in HEADS:
                for seed in tasks["train_seeds"]:
                    cid = f"{name}_lowdim_{variant}_{head}_s{seed}_{bid_l}"
                    emit({"cell_id": cid, "workspace_config": tasks["workspace_lowdim"], "task": t["task"], "head": head,
                          "train_seed": seed, "budget_steps": budget_lowdim, "batch_size": 256, "window_seed": 1000 + seed,
                          "val_every": 1000, "save_every": 2000, "val_batch_size": 256, "num_workers": 8,
                          "overrides": [f"{t['path_key']}={sp[variant]}"] + list(t.get("overrides", [])),
                          "heldout_path_key": t["path_key"].split(".")[-1], "heldout_path": sp["heldout"],
                          "normalizer_path": nz["path"], "normalizer_sha256": nz["sha256"],
                          "identity": {"task_name": name, "modality": "lowdim", "variant": variant, "budget_id": bid_l,
                                       "arm": "core", "subset_n": sp["n"], "u_label": sp["u_label"],
                                       "subset_path": sp[variant], "subset_sha256": sp["hashes"][variant],
                                       "heldout_path": sp["heldout"], "subset_manifest_path": sp["manifest_path"],
                                       "heldout_sha256": sp["hashes"]["heldout"], "subset_manifest_sha256": sp["manifest_sha256"],
                                       "kitchen_set": sp["kitchen_set"]},
                          "runner_overrides": ([f"{t['runner_key']}={t.get('runner_src', t['src'])}"] if t.get("runner_key") else [])}, "core")
    if explore:
        for name, t in tasks["explore"].items():
            budget_lowdim, bid_l = budget_for(name, "lowdim")
            for head in HEADS:
                seed = tasks["train_seeds"][0]
                cid = f"{name}_lowdim_full_{head}_s{seed}_{bid_l}"
                emit({"cell_id": cid, "workspace_config": tasks["workspace_lowdim"], "task": t["task"], "head": head,
                      "train_seed": seed, "budget_steps": budget_lowdim, "batch_size": 256, "window_seed": 1000 + seed,
                      "val_every": 1000, "save_every": 2000, "val_batch_size": 256, "num_workers": 8,
                      "overrides": [f"{t['path_key']}={t['src']}"] + list(t.get("overrides", [])),
                      "identity": {"task_name": name, "modality": "lowdim", "variant": "full", "budget_id": bid_l,
                                   "arm": "explore", "subset_path": t["src"]},
                      "runner_overrides": ([f"{t['runner_key']}={t.get('runner_src', t['src'])}"] if t.get("runner_key") else [])}, "explore")
    if image:
        for name, t in tasks["image"].items():
            budget_image, bid_i = budget_for(name, "image")
            sp = _subset_paths(subsets, name, "zarr" if name == "pusht" else "hdf5", "image")
            if sp is None or not sp["usable"]:
                manifest["skipped"].append({"task": name, "arm": "image", "reason": (sp or {}).get("reason", "no subset manifest")}); continue
            nz = _normalizer(subsets, name, "image", sp["hashes"]["trainpool"])
            if nz is None:
                manifest["skipped"].append({"task": name, "arm": "image", "reason": f"frozen normalizer {name}_image_normalizer.pt missing or fitted on a different pool"}); continue
            for variant in ("U", "M"):
                for head in HEADS:
                    seed = tasks["train_seeds"][0]
                    cid = f"{name}_image_{variant}_{head}_s{seed}_{bid_i}"
                    emit({"cell_id": cid, "workspace_config": tasks["workspace_image"], "task": t["task"], "head": head,
                          "train_seed": seed, "budget_steps": budget_image, "batch_size": 64, "window_seed": 1000 + seed,
                          "val_every": 1000, "save_every": 2000, "val_batch_size": 64, "num_workers": 8,
                          "overrides": [f"{t['path_key']}={sp[variant]}"] + list(t.get("overrides", [])),
                          "heldout_path_key": t["path_key"].split(".")[-1], "heldout_path": sp["heldout"],
                          "normalizer_path": nz["path"], "normalizer_sha256": nz["sha256"],
                          "identity": {"task_name": name, "modality": "image", "variant": variant, "budget_id": bid_i,
                                       "arm": "image", "subset_n": sp["n"], "u_label": sp["u_label"],
                                       "subset_path": sp[variant], "subset_sha256": sp["hashes"][variant],
                                       "heldout_path": sp["heldout"], "subset_manifest_path": sp["manifest_path"],
                                       "heldout_sha256": sp["hashes"]["heldout"], "subset_manifest_sha256": sp["manifest_sha256"]},
                          "runner_overrides": ([f"{t['runner_key']}={t.get('runner_src', t['src'])}"] if t.get("runner_key") else [])}, "image")
    (out / "cells_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main() -> None:
    """CLI entry: generate the cell yamls + ``cells_manifest.json`` and print the counts and skips."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True); ap.add_argument("--subsets", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--budget-lowdim", type=int, required=True); ap.add_argument("--budget-image", type=int, default=20000)
    ap.add_argument("--explore", action="store_true"); ap.add_argument("--image", action="store_true")
    ap.add_argument("--raw-root", default=None, help="host raw data dir substituted for ${RAW_ROOT} (default $DP_DATA/data)")
    a = ap.parse_args()
    import os
    raw_root = a.raw_root or (os.environ.get("DP_DATA", "/data/dp") + "/data")
    tasks = resolve_raw_root(yaml.safe_load(open(a.tasks)), raw_root)
    m = make_cells(tasks, pathlib.Path(a.subsets), pathlib.Path(a.out), a.budget_lowdim, a.budget_image, a.explore, a.image)
    print(f"cells={len(m['cells'])} core={len(m['by_arm']['core'])} explore={len(m['by_arm']['explore'])} "
          f"image={len(m['by_arm']['image'])} skipped={m['skipped']}")


if __name__ == "__main__":
    main()
