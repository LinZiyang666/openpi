"""Tabulate the plan §4.3 conditional-distribution diagnostics written by ``dispersion_index sample``
(``<cell>_<sampler>_<k>.json``) into one JSON + one Markdown table: per cell and sampler the mean conditional sampling
dispersion, the mean GMM(2)-vs-GMM(1) delta-BIC, the fraction of unclipped coordinates outside the normalised range and
the nearest-demo distance, then the seed-averaged rows per (task, modality, variant, head). Files whose ``cell`` block is
not a matrix identity (pre-experiment diagnostics) are listed under ``ignored``.

usage: python -m exp.dp_nfe.analysis.diagnostics_table --dir <diag dir> [--dir ...] --out <table.json> [--md <table.md>]
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict
from typing import Dict, List

import numpy as np

FIELDS = ("mean_dispersion", "mean_delta_bic", "mean_frac_out_of_range_unclipped", "mean_nearest_demo")
KEY = ("task_name", "modality", "variant", "head")


def load(dirs: List[pathlib.Path]) -> Dict[str, list]:
    """``{"rows": [...], "ignored": [...]}``; one row per diagnostics file with the cell identity and the four means."""
    rows, ignored = [], []
    for d in dirs:
        for f in sorted(d.glob("*.json")):
            try:
                j = json.loads(f.read_text())
            except Exception as e:  # noqa: BLE001
                ignored.append({"path": str(f), "reason": f"unreadable: {e}"}); continue
            c = j.get("cell") or {}
            if not all(k in c for k in KEY + ("train_seed",)) or not isinstance(j.get("sampler"), dict):
                ignored.append({"path": str(f), "reason": "no matrix cell identity"}); continue
            s = j["sampler"]
            rows.append({"path": str(f), "cell_id": c.get("cell_id"), **{k: c[k] for k in KEY}, "train_seed": int(c["train_seed"]),
                         "sampler": f"{s.get('sampler')}_{s.get('k')}", "n_histories": j.get("n_histories"), "n_samples": j.get("n_samples"),
                         "std_source": j.get("std_source"), **{k: j.get(k) for k in FIELDS}})
    return {"rows": rows, "ignored": ignored}


def summarise(rows: List[dict]) -> List[dict]:
    """Seed-averaged rows per (task, modality, variant, head, sampler): mean and min/max over seeds of every field."""
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in KEY) + (r["sampler"],)].append(r)
    out = []
    for key, rs in sorted(groups.items()):
        rec = dict(zip(KEY + ("sampler",), key)); rec["n_seeds"] = len(rs); rec["seeds"] = sorted(r["train_seed"] for r in rs)
        for f in FIELDS:
            vals = [r[f] for r in rs if r.get(f) is not None]
            rec[f] = None if not vals else {"mean": float(np.mean(vals)), "min": float(min(vals)), "max": float(max(vals))}
        out.append(rec)
    return out


def markdown(summary: List[dict]) -> str:
    """One Markdown table: task | modality | variant | head | sampler | dispersion | delta-BIC | frac out of range | nearest demo."""
    lines = ["| task | mod | data | head | sampler | n | dispersion | ΔBIC(2v1) | out-of-range (unclipped) | nearest demo |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    def fmt(v, p=2):
        return "–" if v is None else f"{v['mean']:.{p}f} [{v['min']:.{p}f}, {v['max']:.{p}f}]"
    for r in summary:
        lines.append(f"| {r['task_name']} | {r['modality']} | {r['variant']} | {r['head']} | {r['sampler']} | {r['n_seeds']} | "
                     f"{fmt(r['mean_dispersion'])} | {fmt(r['mean_delta_bic'], 0)} | {fmt(r['mean_frac_out_of_range_unclipped'], 3)} | {fmt(r['mean_nearest_demo'])} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """CLI entry: read every diagnostics dir, write the JSON table (+ Markdown) and print the row counts."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", action="append", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path); ap.add_argument("--md", type=pathlib.Path)
    a = ap.parse_args()
    loaded = load(a.dir)
    summary = summarise(loaded["rows"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"fields": FIELDS, "rows": loaded["rows"], "summary": summary, "ignored": loaded["ignored"]}, indent=1))
    if a.md:
        a.md.write_text(markdown(summary))
    print(f"DIAG_TABLE rows={len(loaded['rows'])} groups={len(summary)} ignored={len(loaded['ignored'])} -> {a.out}")


if __name__ == "__main__":
    main()
