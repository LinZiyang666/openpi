"""Plot Phase-2 weight-search results: success rate per weight config.

Reads a conductor results JSON (a list of ``{"yaml_id": .., "success_rate": ..}``
or a dict keyed by yaml_id) and renders a bar chart of success rate per weight
config, grouped by keybuilder stem (the part before ``__``). Mirrors the style of
``exp/common/analysis/phase1/libero_spatial``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load_rates(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {k: float(v["success_rate"] if isinstance(v, dict) else v) for k, v in data.items()}
    rates = {}
    for row in data:
        rates[row["yaml_id"]] = float(row["success_rate"])
    return rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Phase-2 results JSON")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "phase2_results.png"))
    args = ap.parse_args()

    rates = _load_rates(Path(args.results))
    if not rates:
        print("no results to plot")
        return
    by_stem: dict[str, dict[str, float]] = defaultdict(dict)
    for yaml_id, sr in rates.items():
        stem, _, cfg = yaml_id.partition("__")
        by_stem[stem][cfg or yaml_id] = sr

    stems = sorted(by_stem)
    fig, axes = plt.subplots(len(stems), 1, figsize=(10, 3 * len(stems)), squeeze=False)
    best_overall = max(rates.values()) if rates else 0.0
    for ax, stem in zip(axes[:, 0], stems):
        cfgs = sorted(by_stem[stem])
        vals = [by_stem[stem][c] for c in cfgs]
        ax.bar(range(len(cfgs)), vals)
        ax.axhline(best_overall, color="r", ls="--", lw=1, label=f"best={best_overall:.2f}")
        ax.set_xticks(range(len(cfgs)))
        ax.set_xticklabels(cfgs, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("success rate")
        ax.set_title(stem)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
