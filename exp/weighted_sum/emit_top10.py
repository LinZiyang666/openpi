"""Pick the global top-10 configs from all_results.csv and stage their YAMLs.

Replaces the hand-curated ``config/top10/`` of the libero_spatial run with a
deterministic, reproducible command (plan §5, G1 R1 Item4 / R2 Item2-A). The
top-10 set feeds ``emit_trajectory_yamls.select_base_configs`` (Stage 2 base
union), so it MUST be reproducible.

Rule (locked in the plan):
  - aggregate mean ``success_rate`` per ``yaml_id`` across all rows of the CSV
    (a config re-run in multiple refine rounds has several rows);
  - sort by mean SR descending, breaking ties by ``yaml_id`` ascending;
  - take EXACTLY the first 10 (no "tie-include": keeping len==10 keeps the
    Stage-2 base count stable). Within-1-SE neighbours at the #10/#11 boundary
    are recorded in the manifest for transparency but do not change the cut.

Outputs:
  - ``<out-dir>/<yaml_id>.yaml`` copied from the first ``--src-config-dirs`` that
    contains it (so Stage 2 can glob the dir);
  - ``<out-dir>/top10_manifest.json`` — the 10 chosen entries plus the runner-up
    (#11) and whether it sits within 1 SE of #10.

Usage:
    uv run exp/weighted_sum/emit_top10.py \
        --results-csv exp/weighted_sum/data/phase2/libero_10/all_results.csv \
        --src-config-dirs exp/weighted_sum/config/phase2/libero_10 \
                          exp/weighted_sum/config/round_1/libero_10 \
                          exp/weighted_sum/config/round_2/libero_10 \
                          exp/weighted_sum/config/round_3/libero_10 \
        --out-dir exp/weighted_sum/config/top10/libero_10
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import shutil
import statistics
from pathlib import Path

TOP_N = 10


def aggregate_mean_sr(results_csv: Path) -> dict[str, dict]:
    """Aggregate per-yaml_id mean success_rate + total n across all CSV rows.

    Mirrors ``select_base_configs``'s unweighted ``statistics.mean`` over a
    config's per-stage success_rate values; ``n`` is summed for the SE estimate.
    """
    by_yid: dict[str, list[tuple[float, int]]] = collections.defaultdict(list)
    for r in csv.DictReader(results_csv.open()):
        by_yid[r["yaml_id"]].append((float(r["success_rate"]), int(r.get("n", 0) or 0)))
    agg: dict[str, dict] = {}
    for yid, rows in by_yid.items():
        srs = [sr for sr, _ in rows]
        agg[yid] = {
            "yaml_id": yid,
            "mean_sr": statistics.mean(srs),
            "n_total": sum(n for _, n in rows),
            "n_rows": len(rows),
        }
    return agg


def _wilson_se(p: float, n: int) -> float:
    """Binomial standard error sqrt(p(1-p)/n); 0 when n<=0 (no estimate)."""
    if n <= 0:
        return 0.0
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def pick_top10(agg: dict[str, dict]) -> tuple[list[dict], dict | None]:
    """Return (top10, runner_up). Deterministic: sort by (-mean_sr, yaml_id)."""
    ordered = sorted(agg.values(), key=lambda d: (-d["mean_sr"], d["yaml_id"]))
    top10 = ordered[:TOP_N]
    runner_up = ordered[TOP_N] if len(ordered) > TOP_N else None
    return top10, runner_up


def _find_yaml(yaml_id: str, src_dirs: list[Path]) -> Path | None:
    for d in src_dirs:
        cand = d / f"{yaml_id}.yaml"
        if cand.is_file():
            return cand
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Pick global top-10 configs + stage their YAMLs")
    ap.add_argument("--results-csv", required=True, type=Path)
    ap.add_argument(
        "--src-config-dirs",
        required=True,
        nargs="+",
        type=Path,
        help="dirs to search for each chosen <yaml_id>.yaml (first match wins)",
    )
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    agg = aggregate_mean_sr(args.results_csv)
    if len(agg) < TOP_N:
        raise SystemExit(f"only {len(agg)} distinct yaml_ids in {args.results_csv}; need >= {TOP_N}")

    top10, runner_up = pick_top10(agg)

    # #10/#11 within-1-SE note (transparency only — the cut stays at exactly 10).
    tenth = top10[-1]
    se10 = _wilson_se(tenth["mean_sr"], tenth["n_total"])
    runner_up_note = None
    if runner_up is not None:
        gap = tenth["mean_sr"] - runner_up["mean_sr"]
        runner_up_note = {
            "yaml_id": runner_up["yaml_id"],
            "mean_sr": runner_up["mean_sr"],
            "gap_to_rank10": gap,
            "within_1_se_of_rank10": gap < se10,
        }

    # Resolve every source yaml FIRST so a missing one fails before any copy
    # (no orphan partial copies left in the staging dir).
    resolved: list[tuple[int, dict, Path]] = []
    missing: list[str] = []
    for rank, e in enumerate(top10, start=1):
        src = _find_yaml(e["yaml_id"], args.src_config_dirs)
        if src is None:
            missing.append(e["yaml_id"])
        else:
            resolved.append((rank, e, src))
    if missing:
        raise SystemExit(f"top-10 yaml(s) not found under {args.src_config_dirs}: {missing}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Idempotent (G2 R1 Item2): clear any stale *.yaml first so a re-run, a
    # changed results CSV, or pre-existing hand-placed files leave EXACTLY the
    # 10 chosen configs. Stage 2 globs this dir, so a stale file would silently
    # change N_base.
    for stale in args.out_dir.glob("*.yaml"):
        stale.unlink()
    manifest_entries = []
    for rank, e, src in resolved:
        shutil.copy2(src, args.out_dir / f"{e['yaml_id']}.yaml")
        manifest_entries.append(
            {
                "rank": rank,
                "yaml_id": e["yaml_id"],
                "mean_sr": e["mean_sr"],
                "n_total": e["n_total"],
                "n_rows": e["n_rows"],
                "src": str(src),
            }
        )

    manifest = {
        "results_csv": str(args.results_csv),
        "top_n": TOP_N,
        "rank10_se": se10,
        "entries": manifest_entries,
        "excluded_runner_up": runner_up_note,
    }
    (args.out_dir / "top10_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[top10] staged {len(manifest_entries)} yamls -> {args.out_dir}")
    print(f"[top10] rank10 SR={tenth['mean_sr']:.4f} (SE={se10:.4f})")
    if runner_up_note is not None:
        flag = "WITHIN 1 SE" if runner_up_note["within_1_se_of_rank10"] else "clear"
        print(
            f"[top10] runner-up #{TOP_N + 1} {runner_up_note['yaml_id']} "
            f"SR={runner_up_note['mean_sr']:.4f} gap={runner_up_note['gap_to_rank10']:.4f} ({flag})"
        )


if __name__ == "__main__":
    main()
