"""Refit the per-field normalizer on the population the served score actually sees.

The shipped params were fitted on the library's own internal pairwise
similarities. At serve time the quantity being normalized is a QUERY-vs-library
similarity. When those two populations sit at different centres the z-score
saturates and the fused score stops resolving anything -- every threshold then
cuts in dead space.

This reads the raw query-vs-winner similarities collected by
``probe_raw_sim.py``, reports how much of that population the SHIPPED
normalizer squashes flat, and refits each field through the library's own
``fit_from_scores`` so the fix uses the same estimator the calibration does.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from openpi.cache.components.score_normalizers import SCORE_NORMALIZER_REGISTRY

SIM_TYPE = {"vision_0": "cosine", "vision_1": "cosine", "vision_2": "cosine",
            "prompt_emb": "cosine", "robot_state": "l2"}


def saturation(vals: np.ndarray, mu: float, sigma: float, edge: float = 0.01) -> float:
    """Fraction of the population the squash pins to an end of [0, 1].

    ``ZScoreNormalizer`` maps z to ``0.5*(tanh(z)+1)``, so "saturated" means the
    output sits within ``edge`` of 0 or 1: those inputs are all mapped to the
    same value and the score can no longer tell them apart.
    """
    y = 0.5 * (np.tanh((vals - mu) / max(sigma, 1e-12)) + 1.0)
    return float(((y < edge) | (y > 1 - edge)).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-jsonl", required=True)
    ap.add_argument("--shipped", required=True, help="calibration_normalizers json")
    ap.add_argument("--library-key", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in pathlib.Path(args.raw_jsonl).read_text().splitlines() if l.strip()]
    shipped = json.loads(pathlib.Path(args.shipped).read_text())[args.library_key]["fields"]

    per_field: dict[str, list[float]] = {}
    for r in rows:
        for f, v in r["raw"].items():
            per_field.setdefault(f, []).append(float(v))

    out = {}
    print("%-12s %7s | %-34s | %-34s" % ("field", "n", "SHIPPED (library-internal)", "REFIT (query-vs-library)"))
    for f, vals in sorted(per_field.items()):
        x = np.asarray(vals, dtype=np.float64)
        sp = shipped.get(f, {}).get("selected", {})
        method = sp.get("method", "zscore")
        p = sp.get("params", {})
        old_mu, old_sd = float(p.get("mu", 0.0)), float(p.get("sigma", 1.0))
        cls = SCORE_NORMALIZER_REGISTRY[method]
        new = cls.fit_from_scores(x, SIM_TYPE.get(f, "cosine"))
        np_ = new.params_to_dict()
        new_mu = float(np_.get("mu", np.mean(x)))
        new_sd = float(np_.get("sigma", np.std(x)))
        out[f] = {"method": method, "params": np_}
        print("%-12s %7d | mu %9.5f sd %8.5f sat %5.1f%% | mu %9.5f sd %8.5f sat %5.1f%%"
              % (f, len(x), old_mu, old_sd, 100 * saturation(x, old_mu, old_sd),
                 new_mu, new_sd, 100 * saturation(x, new_mu, new_sd)))
        print("             query population: mean %.5f sd %.5f  p5 %.5f p50 %.5f p95 %.5f"
              % (x.mean(), x.std(), np.percentile(x, 5), np.percentile(x, 50), np.percentile(x, 95)))
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
