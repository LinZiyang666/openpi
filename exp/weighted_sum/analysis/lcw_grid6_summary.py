"""Summarise the uniform 1/6 fusion-weight grid (28 cells x 500 A-pool episodes) per suite and teacher.

Reads the conductor journals, prints each landscape as a triangle (v0 rows,
v1 columns, rs implied), the top cells, the uniform cell, and locates the
zero-cost candidates (LDA / action-error argmin from the fusion-ablation
manifest) by their nearest grid cell. Also reports the kernel-smoothed value
at the candidate itself, so the reading does not depend on where the lattice
happens to fall.

Usage:
  python exp/weighted_sum/analysis/lcw_grid6_summary.py --root exp/weighted_sum/data/grid6 \
      --manifest exp/weighted_sum/config/fusion_ablation/manifest.json --out <json>
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import re

PAT = re.compile(r"v0@(\d)_v1@(\d)_rs@(\d)")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load(journal: pathlib.Path):
    acc = collections.defaultdict(lambda: [0, 0])
    eps = collections.defaultdict(dict)
    for line in journal.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        m = PAT.search(r["yaml_id"])
        cell = tuple(int(g) for g in m.groups())
        acc[cell][0] += int(r["success"])
        acc[cell][1] += 1
        arm, _, t, e = r["task_uid"].rsplit(":", 3)
        eps[cell][(int(t), int(e))] = bool(r["success"])
    return {c: (s / n, n) for c, (s, n) in acc.items()}, eps


def sign_test(a, b):
    s = set(a) & set(b)
    wa = sum(a[k] and not b[k] for k in s)
    wb = sum(b[k] and not a[k] for k in s)
    n = wa + wb
    k = min(wa, wb)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return wa, wb, p


def l1(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def smoothed(w, cells, bw=0.15):
    num = den = 0.0
    for c, (sr, n) in cells.items():
        k = math.exp(-((l1(w, [x / 6 for x in c]) / bw) ** 2)) * n
        num += k * sr
        den += k
    return num / den


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=pathlib.Path, required=True)
    ap.add_argument("--manifest", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    man = json.loads(args.manifest.read_text())
    report = {}
    for teacher in ("pi05", "groot"):
        for suite in ("libero_spatial", "libero_10"):
            cells, eps = load(args.root / teacher / suite / "journal.jsonl")
            assert len(cells) == 28 and all(n == 500 for _, n in cells.values()), (teacher, suite)
            order = sorted(cells.items(), key=lambda kv: -kv[1][0])
            best, (best_sr, _) = order[0]
            uni = cells[(2, 2, 2)][0]
            print(f"\n== {teacher} {suite}: max {best_sr:.3f} at {best}  uniform {uni:.3f}  min {order[-1][1][0]:.3f} at {order[-1][0]}")
            print("   triangle (rows v0=0..6, cols v1=0..6; rs = 6-v0-v1):")
            for a in range(7):
                row = "   v0={} ".format(a) + " ".join(f"{cells[(a, b, 6 - a - b)][0]:.3f}" if a + b <= 6 else "     " for b in range(7))
                print(row)
            plateau = [c for c, (sr, _) in cells.items() if sr >= best_sr - 0.02]
            print(f"   cells within 2 pp of max: {len(plateau)}; within 4 pp: {sum(1 for c,(sr,_) in cells.items() if sr >= best_sr-0.04)}")
            entry = {"max": {"cell": best, "sr": best_sr, "wilson95": wilson(round(best_sr * 500), 500)},
                     "uniform": uni, "min": {"cell": order[-1][0], "sr": order[-1][1][0]},
                     "plateau_2pp": len(plateau), "cells": {"/".join(map(str, c)): sr for c, (sr, _) in cells.items()},
                     "candidates": {}}
            for arm in ("lda", "acterr", "leader"):
                w = man[f"fw_{teacher}_{suite}_{arm}"]["w"]
                near = min(cells, key=lambda c: l1(w, [x / 6 for x in c]))
                sm = smoothed(w, cells)
                wa, wb, p = sign_test(eps[near], eps[best])
                entry["candidates"][arm] = {"w": w, "nearest": near, "nearest_sr": cells[near][0], "smoothed": sm,
                                            "gap_to_max_pp": 100 * (best_sr - cells[near][0]), "sign_p_vs_max": p}
                print(f"   {arm:7s} w={'/'.join(f'{x:.2f}' for x in w)} nearest {near} sr={cells[near][0]:.3f} smoothed={sm:.3f} gap={100*(best_sr-cells[near][0]):+.1f} pp (paired p vs max {p:.2f})")
            report[f"{teacher}_{suite}"] = entry
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
