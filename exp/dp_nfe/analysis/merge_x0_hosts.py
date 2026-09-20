"""Merge the per-host frozen matrices and results of the x0 experiment into one aggregator input.

Every host generates its own ``cells/`` from the same ``tasks.yaml`` (same cell ids, budgets and seeds) but the cell yaml
*bytes* differ per host (host-resolved data paths, non-deterministic zarr bytes), and every result record is bound to the
yaml of the host that trained it (``cell.cell_yaml_sha256``). ``aggregate_x0`` judges completeness against a single
``cells_manifest.json``, so the two host trees are merged with these rules:

* the manifests must agree on ``train_seeds``, ``budgets`` and ``budgets_by_task``; ``cells`` / ``by_arm`` are the
  ordered union (a cell keeps its arm); ``skipped`` is the union keyed by (task, arm) and must not disagree on the reason,
  except that a *host-local* skip (a (task, arm) another host does list cells for, e.g. the square image cells that only
  exist on h100) is dropped and reported under ``host_local_skips``;
* a cell is *owned* by the host whose ``results_trailing/<cell>/`` holds at least one ``summary.json``; owning a cell on
  two hosts is an error (the plan keeps the four cells of a task on one machine), and the owner's yaml bytes, results and
  (when present) ``runs/<cell>/`` are copied verbatim; an un-trained cell takes the yaml of the first host listing it and
  is reported under ``untrained`` so the aggregator still counts it as missing;
* a result directory whose records all carry ``cell.variant == "official"`` (the P0 official-checkpoint ladders) is not a
  matrix cell: it is copied through for the aggregator's descriptive ``official`` summary and may live on one host only.

usage: python -m exp.dp_nfe.analysis.merge_x0_hosts --host wls=<dir> --host h100=<dir> --out <merged dir>
       (each <dir> holds cells/ [+ results_trailing/ + runs/]; <merged dir> must not exist yet)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
from typing import Dict, List, Tuple

import yaml

from exp.dp_nfe.x0_identity import sha256_file

FROZEN_KEYS = ("train_seeds", "budgets", "budgets_by_task")
ARMS = ("core", "explore", "image")


class MergeError(ValueError):
    """Raised when the host trees cannot be merged into one consistent matrix."""
    pass


def _ordered_union(lists: List[List[str]]) -> List[str]:
    out: List[str] = []
    for lst in lists:
        for x in lst:
            if x not in out:
                out.append(x)
    return out


def _is_official(result_dir: pathlib.Path) -> bool:
    """True when every readable record of the directory names the official-checkpoint variant."""
    flags = []
    for f in result_dir.glob("*/summary.json"):
        try:
            flags.append((json.loads(f.read_text()).get("cell") or {}).get("variant") == "official")
        except Exception:  # noqa: BLE001
            flags.append(False)
    return bool(flags) and all(flags)


def load_host(name: str, root: pathlib.Path) -> dict:
    """``{"name", "root", "manifest", "manifest_sha256", "owned": {cell_id: n_summaries}, "official": {dir: n}}`` for one
    host tree."""
    man = root / "cells" / "cells_manifest.json"
    if not man.is_file():
        raise MergeError(f"{name}: {man} missing")
    manifest = json.loads(man.read_text())
    for cid in manifest["cells"]:
        if not (root / "cells" / f"{cid}.yaml").is_file():
            raise MergeError(f"{name}: cell yaml {cid}.yaml missing")
    owned, official = {}, {}
    results = root / "results_trailing"
    if results.is_dir():
        for d in sorted(p for p in results.iterdir() if p.is_dir()):
            n = len(list(d.glob("*/summary.json")))
            if n:
                (official if _is_official(d) else owned)[d.name] = n
    return {"name": name, "root": root, "manifest": manifest, "manifest_sha256": sha256_file(man), "owned": owned,
            "official": official}


def merge_manifests(hosts: List[dict]) -> dict:
    """Union manifest of the hosts (``MergeError`` on any frozen-field or skip-reason disagreement)."""
    first = hosts[0]["manifest"]
    for h in hosts[1:]:
        for k in FROZEN_KEYS:
            if h["manifest"].get(k) != first.get(k):
                raise MergeError(f"{k} differs: {hosts[0]['name']}={first.get(k)!r} {h['name']}={h['manifest'].get(k)!r}")
    merged = {"cells": _ordered_union([h["manifest"]["cells"] for h in hosts]),
              "by_arm": {arm: _ordered_union([h["manifest"]["by_arm"][arm] for h in hosts]) for arm in ARMS},
              "skipped": [], **{k: first[k] for k in FROZEN_KEYS if k in first}}
    listed = [c for arm in ARMS for c in merged["by_arm"][arm]]
    if len(listed) != len(set(listed)) or sorted(listed) != sorted(merged["cells"]):
        raise MergeError("merged arms do not partition the merged cell list (a cell changes arm between hosts)")
    seen: Dict[Tuple[str, str], str] = {}
    for h in hosts:
        for sk in h["manifest"].get("skipped", []):
            key = (sk["task"], sk["arm"])
            if key in seen and seen[key] != sk["reason"]:
                raise MergeError(f"skip reason for {key} differs between hosts: {seen[key]!r} vs {sk['reason']!r}")
            if key not in seen:
                seen[key] = sk["reason"]; merged["skipped"].append(dict(sk))
    merged["merged_from"] = {h["name"]: {"cells_manifest_sha256": h["manifest_sha256"], "cells": len(h["manifest"]["cells"]),
                                         "owned_cells": len(h["owned"])} for h in hosts}
    return merged


def assign_owners(hosts: List[dict], cells: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """``({cell_id: owner host}, untrained cell ids)``; a cell with results on two hosts is a ``MergeError``."""
    owners: Dict[str, str] = {}
    untrained: List[str] = []
    for cid in cells:
        have = [h["name"] for h in hosts if cid in h["owned"]]
        if len(have) > 1:
            raise MergeError(f"{cid} has results on more than one host: {have}")
        if have:
            owners[cid] = have[0]
        else:
            untrained.append(cid)
            owners[cid] = next(h["name"] for h in hosts if cid in h["manifest"]["cells"])
    stray = {h["name"]: sorted(set(h["owned"]) - set(cells)) for h in hosts}
    stray = {k: v for k, v in stray.items() if v}
    if stray:
        raise MergeError(f"results for cells absent from every manifest: {stray}")
    return owners, untrained


def merge(hosts: List[dict], out: pathlib.Path) -> dict:
    """Write ``out/cells``, ``out/results_trailing`` (+ ``out/runs``) and ``out/merge_report.json``; returns the report."""
    if out.exists() and any(out.iterdir()):
        raise MergeError(f"{out} exists and is not empty")
    merged = merge_manifests(hosts)
    owners, untrained = assign_owners(hosts, merged["cells"])
    by_name = {h["name"]: h for h in hosts}
    present = set()
    for arm in ARMS:
        for cid in merged["by_arm"][arm]:
            cell = yaml.safe_load((by_name[owners[cid]]["root"] / "cells" / f"{cid}.yaml").read_text())
            present.add((cell["identity"]["task_name"], arm))
    host_local = [sk for sk in merged["skipped"] if (sk["task"], sk["arm"]) in present]
    merged["skipped"] = [sk for sk in merged["skipped"] if (sk["task"], sk["arm"]) not in present]
    (out / "cells").mkdir(parents=True, exist_ok=True)
    (out / "results_trailing").mkdir(exist_ok=True)
    per_cell = {}
    for cid in merged["cells"]:
        h = by_name[owners[cid]]
        src_yaml = h["root"] / "cells" / f"{cid}.yaml"
        shutil.copyfile(src_yaml, out / "cells" / f"{cid}.yaml")
        entry = {"host": h["name"], "cell_yaml_sha256": sha256_file(src_yaml), "summaries": h["owned"].get(cid, 0), "runs": False}
        if cid in h["owned"]:
            shutil.copytree(h["root"] / "results_trailing" / cid, out / "results_trailing" / cid)
            run = h["root"] / "runs" / cid
            if run.is_dir():
                shutil.copytree(run, out / "runs" / cid); entry["runs"] = True
        per_cell[cid] = entry
    official = {}
    for h in hosts:
        for name, n in h["official"].items():
            if name in official:
                raise MergeError(f"official ladder {name} exists on more than one host: {official[name]['host']}, {h['name']}")
            shutil.copytree(h["root"] / "results_trailing" / name, out / "results_trailing" / name)
            official[name] = {"host": h["name"], "summaries": n}
    (out / "cells" / "cells_manifest.json").write_text(json.dumps(merged, indent=1))
    report = {"hosts": {h["name"]: {"root": str(h["root"]), "cells_manifest_sha256": h["manifest_sha256"],
                                    "owned": sorted(h["owned"]), "official": sorted(h["official"])} for h in hosts},
              "cells": per_cell, "untrained": untrained, "host_local_skips": host_local, "official": official,
              "counts": {"cells": len(merged["cells"]), "trained": len(merged["cells"]) - len(untrained), "untrained": len(untrained),
                         **{arm: len(merged["by_arm"][arm]) for arm in ARMS}}}
    (out / "merge_report.json").write_text(json.dumps(report, indent=1))
    return report


def main() -> None:
    """CLI entry: merge the named host trees and print the ownership counts."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", action="append", required=True, metavar="NAME=DIR", help="host tree (repeatable, order = yaml precedence for untrained cells)")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args()
    hosts = []
    for spec in a.host:
        name, _, d = spec.partition("=")
        if not name or not d:
            raise SystemExit(f"--host expects NAME=DIR, got {spec!r}")
        hosts.append(load_host(name, pathlib.Path(d)))
    report = merge(hosts, a.out)
    c = report["counts"]
    print(f"MERGED cells={c['cells']} trained={c['trained']} untrained={c['untrained']} core={c['core']} explore={c['explore']} "
          f"image={c['image']} -> {a.out}")
    for name, h in report["hosts"].items():
        print(f"  {name}: owns {len(h['owned'])} cells + {len(h['official'])} official ladders (manifest {h['cells_manifest_sha256'][:12]})")


if __name__ == "__main__":
    main()
