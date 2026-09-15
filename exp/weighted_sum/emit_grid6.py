"""Emit the uniform 1/6 closed-simplex fusion-weight grid (28 cells) for every suite and teacher.

Each cell is the suite's historical pure-cache recipe with only the three
fusion weights replaced (same templates as emit_fusion_ablation.py), so all
four suites share one lattice: (v0, v1, rs) = (a, b, 6-a-b)/6, a+b <= 6, edges
and corners included. Writes one yaml per cell plus a run_size_eval arm matrix
per suite (GR00T matrices point at the paths the serving node sees).

Usage:
  uv run exp/weighted_sum/emit_grid6.py --groot-template-dir <dir with <suite>_r1_template.yaml> \
      --out exp/weighted_sum/config/grid6
"""

from __future__ import annotations

import argparse
import json
import pathlib

import yaml

from exp.weighted_sum.emit_fusion_ablation import PI05_LIBRARY, PI05_TEMPLATE, SUITES, set_weights

STEP = 6
GROOT_SERVED_DIR = "/data/libero_cache/search/grid6/{suite}/cells"


def cells():
    for a in range(STEP + 1):
        for b in range(STEP + 1 - a):
            yield a, b, STEP - a - b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--groot-template-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    manifest = {}
    for teacher in ("pi05", "groot"):
        for suite in SUITES:
            if teacher == "pi05":
                template = yaml.safe_load(pathlib.Path(PI05_TEMPLATE.format(suite=suite)).read_text())
                template["backend"]["in_memory"]["preload_path"] = PI05_LIBRARY.format(suite=suite)
            else:
                template = yaml.safe_load((args.groot_template_dir / f"{suite}_r1_template.yaml").read_text())
            out_dir = args.out / teacher / suite
            out_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for a, b, c in cells():
                name = f"g6_{teacher}_{suite}_v0@{a}_v1@{b}_rs@{c}"
                w = [a / STEP, b / STEP, c / STEP]
                doc = set_weights(yaml.safe_load(yaml.safe_dump(template, sort_keys=False)), w)
                path = out_dir / f"{name}.yaml"
                path.write_text(yaml.safe_dump(doc, sort_keys=False))
                served = str(path) if teacher == "pi05" else f"{GROOT_SERVED_DIR.format(suite=suite)}/{name}.yaml"
                rows.append({"arm": name, "yaml": served, "sidecar": None})
                manifest[name] = {"teacher": teacher, "suite": suite, "raw": [a, b, c], "w": w}
            (out_dir / f"matrix_{suite}.yaml").write_text(yaml.safe_dump({"suite": suite, "arms": rows}, sort_keys=False))
            print(f"{teacher}/{suite}: {len(rows)} cells -> {out_dir}")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
