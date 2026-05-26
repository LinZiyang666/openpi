"""Plot Phase-1 calibration diagnostics: per (keybuilder, field) best-method J.

Reads ``calibration_normalizers.json`` and renders a grouped bar chart of the
top-1 candidate's ``mag_sep`` (magnitude separation) per field, grouped by
keybuilder, plus a printed table of the full shortlist. No CLI args beyond the
JSON path; output PNG sits next to this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", required=True)
    ap.add_argument(
        "--out", default=str(Path(__file__).resolve().parent / "phase1_calibration.png")
    )
    args = ap.parse_args()

    calib = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    stems = sorted(calib)
    fields = sorted({f for s in calib.values() for f in s["fields"]})

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(stems)), 5))
    width = 0.8 / max(1, len(fields))
    for fi, field in enumerate(fields):
        xs, ys = [], []
        for si, stem in enumerate(stems):
            entry = calib[stem]["fields"].get(field)
            if entry is None:
                continue
            best = entry["shortlist"][0]
            xs.append(si + fi * width)
            ys.append(best["mag_sep"])
            print(f"{stem:28s} {field:12s} best={best['method']:18s} "
                  f"J={best['J']:.4f} mag_sep={best['mag_sep']:.4f} sat={best['sat']:.3f}")
        ax.bar(xs, ys, width=width, label=field)

    ax.set_xticks([i + 0.4 for i in range(len(stems))])
    ax.set_xticklabels(stems, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mag_sep (same - cross, normalized space)")
    ax.set_title("Phase-1 calibration: top-1 normalizer magnitude separation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
