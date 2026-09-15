"""Emit the fusion-weight ablation arms: uniform / action-error grid / phase LDA, per suite and teacher.

Each arm is the suite's historical pure-cache recipe (``always_search`` +
``always_hit`` + ``top_k=1`` + the grid's own normalizers + ``write_policy:
never``) with only the three fusion weights replaced, so the closed-loop reading
differs from the grid cells by the weights alone. Pi0.5 arms start from the
cache_prune search template (normalizers byte-identical to the phase-2 grid);
GR00T arms start from the r1 coarse-grid cell yaml (original S3 library and its
calibration), which is what the 28-cell grid was served with.

Weights come from the offline studies, read off the JSON they wrote, never
retyped: LDA from ``lcw_fit_weights.json`` (``lda@0.05``), the action-error
argmin from the codex replay arrays on the served library, uniform is 1/3.

Usage:
  uv run exp/weighted_sum/emit_fusion_ablation.py \
      --groot-template-dir <dir with <suite>_r1_template.yaml> \
      --out exp/weighted_sum/config/fusion_ablation
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import yaml

SUITES = ("libero_spatial", "libero_10")
FIELDS = ("vision_0", "vision_1", "robot_state")
ZC = pathlib.Path("exp/weighted_sum/data/low_cost_weights/zero_cost_priors")
CODEX = pathlib.Path("exp/weighted_sum/data/low_cost_weights/20260913")
PI05_TEMPLATE = "exp/ablation_study/cache_prune/config/search_{suite}.yaml"
PI05_LIBRARY = "/home/weiland/openpi/exp/common/data/cache_artifacts/{suite}/cp1_spatial_pool_16.pkl"


def lda_weights(tag: str) -> list[float]:
    return json.loads((ZC / tag / "lcw_fit_weights.json").read_text())["fits"]["lda@0.05"]["w"]


def action_error_weights(teacher: str, suite: str) -> list[float]:
    """Argmin of the 7-dim executed-action error over the 1/48 simplex grid, on the served library."""
    if teacher == "pi05":
        z = np.load(CODEX / suite / "replay.npz")
        loss = z["loss_valid7_l2"]
    else:
        z = np.load(CODEX / "groot_results" / f"groot_{suite}_original_replay.npz")
        loss = z["loss_valid7"]
    hist = int(z["hist_start"])
    grid_w, grid_loss = z["weights"][:hist], loss[:hist]
    i = int(np.argmin(grid_loss))
    w = grid_w[i]
    if abs(float(w.sum()) - 1.0) > 1e-6:
        raise SystemExit(f"{teacher}/{suite}: argmin weight does not lie on the simplex: {w}")
    return [float(x) for x in w]


# Historical grid leaders, copied from the winning cells' own yaml files (pi0.5:
# weighted_sum round_1 / phase2 configs; GR00T: r2 / r1 cell yamls), so the
# re-run serves exactly the weights the grid served. Ties: pi0.5 LIBERO-10 has
# three n=100 cells at 0.52; the raw argmax cell is used.
GRID_LEADERS = {
    ("pi05", "libero_spatial"): [0.0625, 0.4375, 0.5],
    ("pi05", "libero_10"): [0.125, 0.875, 0.0],
    ("groot", "libero_spatial"): [1 / 6, 7 / 12, 0.25],
    ("groot", "libero_10"): [2 / 3, 1 / 6, 1 / 6],
}


def set_weights(doc: dict, w: list[float]) -> dict:
    for f, x in zip(FIELDS, w):
        doc["keys"][f]["enabled"] = True
        doc["keys"][f]["weight"] = float(x)
    cp1 = doc["checkpoints"]["cp1"]
    if cp1["judge"]["type"] != "always_hit" or cp1["gate"]["type"] != "always_search":
        raise SystemExit("template is not the pure-cache recipe")
    if doc["write_policy"]["type"] != "never":
        raise SystemExit("template must be write-frozen")
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--groot-template-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    manifest = {}
    for teacher in ("pi05", "groot"):
        for suite in SUITES:
            tag = f"{teacher}_{suite}" + ("_origS3" if teacher == "groot" else "")
            arms = {
                "uniform": [1 / 3, 1 / 3, 1 / 3],
                "acterr": action_error_weights(teacher, suite),
                "lda": lda_weights(tag),
                "leader": GRID_LEADERS[(teacher, suite)],
            }
            if teacher == "pi05":
                template = yaml.safe_load(pathlib.Path(PI05_TEMPLATE.format(suite=suite)).read_text())
                template["backend"]["in_memory"]["preload_path"] = PI05_LIBRARY.format(suite=suite)
            else:
                template = yaml.safe_load((args.groot_template_dir / f"{suite}_r1_template.yaml").read_text())
            out_dir = args.out / teacher / suite
            out_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for arm, w in arms.items():
                name = f"fw_{teacher}_{suite}_{arm}"
                doc = set_weights(yaml.safe_load(yaml.safe_dump(template, sort_keys=False)), w)
                path = out_dir / f"{name}.yaml"
                path.write_text(yaml.safe_dump(doc, sort_keys=False))
                rows.append({"arm": name, "yaml": str(path), "sidecar": None})
                manifest[name] = {"teacher": teacher, "suite": suite, "arm": arm, "w": w,
                                  "library": doc["backend"]["in_memory"]["preload_path"]}
                print(f"{name:32s} w=" + "/".join(f"{x:.4f}" for x in w))
            if teacher == "pi05":
                (out_dir / f"matrix_{suite}.yaml").write_text(yaml.safe_dump(
                    {"suite": suite, "arms": rows}, sort_keys=False))
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
