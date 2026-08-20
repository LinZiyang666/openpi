"""Collect every plotted data point into one versioned file (plan §8.5).

The figure used to read the analyzer JSON directly, which made supplementary
experiments awkward: a new measurement (say, a latency re-run on different
hardware) had no place to land except ad-hoc CLI arguments. Now the pipeline is

    analyze_size.py --> emit_plot_data.py --> plot_data.json --> plot_size.py

and ``plot_data.json`` is the single file the figure reads. Each (suite,
outcome_filter) family is one block; re-collecting a family replaces exactly
that block and leaves the others untouched, so families can be added or
refreshed independently. Latency measurements attach to existing points under a
caller-chosen label (``--latency-label``), so any number of hosts/backends can
coexist without touching the SR data.

Provenance is kept, not weakened, by the indirection: every family records the
path and sha256 of the analyzer JSON it came from, and the SR/CI/teacher/verdict
fields are copied verbatim -- this collector computes nothing statistical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

# Internal tier ids of the pre-registered matrix. They appear in the data file
# ONLY inside `source_arm` provenance strings (real artifact names); every
# descriptive field is self-explanatory instead ("trajectories_per_task", ...),
# so a reader needs no decoder ring.
TIERS = ("S1", "S2", "S3", "S4", "S5", "S6")
SCHEMA = 2


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family_key(suite: str, outcome_filter: str | None) -> str:
    return f"{suite}/{outcome_filter or 'single'}"


def load_data(path: pathlib.Path) -> dict:
    if not path.exists():
        return {"schema": SCHEMA, "families": {}}
    data = json.loads(path.read_text())
    if data.get("schema") != SCHEMA:
        raise SystemExit(
            f"{path}: schema {data.get('schema')!r} != {SCHEMA}; refusing to merge "
            "across schema versions -- migrate the file first"
        )
    return data


def collect_family(
    result: dict,
    *,
    result_path: pathlib.Path,
    grid: dict | None,
    entries: dict[str, int] | None,
    bucket_sizes: dict | None,
) -> dict:
    """One family block from an analyzer JSON, copied verbatim -- no recomputation."""
    for key in ("tier_sr", "tier_ci", "teacher_sr"):
        if key not in result:
            raise SystemExit(
                f"{result_path}: analysis JSON lacks {key!r}; regenerate it with the "
                "current analyze_size.py rather than supplying numbers by hand"
            )
    tier_sr, tier_ci = result["tier_sr"], result["tier_ci"]
    missing = [t for t in TIERS if t not in tier_sr or t not in tier_ci]
    if missing:
        raise SystemExit(f"{result_path}: analysis JSON is missing tiers {missing}")

    suite = result.get("suite") or _suite_from_name(result_path)
    filt = result.get("outcome_filter")
    mean_realized = (grid or {}).get("mean_realized", {})
    nominal = dict(zip(TIERS, (1, 2, 5, 10, 20, 45)))
    n_tasks = 10  # both LIBERO suites

    points = []
    for t in TIERS:
        traj = float(mean_realized.get(t, nominal[t]))
        arm = f"cache_size_{suite}_{filt}_{t}" if filt else f"cache_size_{suite}_{t}"
        pt: dict = {
            "trajectories_per_task": traj,
            "trajectories_per_task_is_realized": t in mean_realized,
            "episodes_in_library": round(traj * n_tasks),
            "success_rate": tier_sr[t],
            "success_rate_ci95": list(tier_ci[t]),
            "source_arm": arm,
        }
        if entries and t in entries:
            pt["library_entries_total"] = int(entries[t])
        if bucket_sizes:
            tag = f"{suite}_{filt}_{t}" if filt else f"{suite}_{t}"
            b = bucket_sizes.get(tag)
            if b:
                sizes = list(b["per_task"].values())
                pt["entries_scanned_per_call"] = {
                    "min": min(sizes),
                    "mean": sum(sizes) / len(sizes),
                    "max": max(sizes),
                }
        points.append(pt)

    filter_meaning = {
        "all": "library keeps every collected trajectory, failed ones included",
        "success": "library keeps successful trajectories only",
    }.get(filt, "single-family layout (no outcome filter)")

    return {
        "suite": suite,
        "outcome_filter": filt,
        "library_filter_meaning": filter_meaning,
        "x_axis": "trajectories_per_task",
        "family_role": result.get("family_role", "primary"),
        "teacher_success_rate": float(result["teacher_sr"]),
        "verdict": result.get("verdict", {}),
        "figures": [],
        "source": {
            "result_json": str(result_path),
            "sha256": _sha256(result_path),
        },
        "points": points,
    }


def _suite_from_name(path: pathlib.Path) -> str:
    # size_<suite>_<filter>.json fallback; the analyzer JSON should carry suite.
    stem = path.stem
    for cand in ("libero_spatial", "libero_10"):
        if cand in stem:
            return cand
    raise SystemExit(f"{path}: cannot infer suite; analyzer JSON carries none")


def attach_latency(
    data: dict, *, key: str, label: str, latency: dict[str, dict | float]
) -> None:
    """Attach per-tier latency under ``label`` to an existing family's points.

    Keyed by ``source_arm`` (the real artifact id of the measured arm). Refuses
    to invent points: latency for an arm the family does not have is an error,
    because it would create an SR-less point the plot cannot anchor.
    """
    fam = data["families"].get(key)
    if fam is None:
        raise SystemExit(
            f"family {key!r} not in the data file; collect its SR results first "
            "(latency attaches to existing points, it does not create them)"
        )
    by_arm = {p["source_arm"]: p for p in fam["points"]}
    unknown = sorted(set(latency) - set(by_arm))
    if unknown:
        raise SystemExit(f"latency carries unknown arms {unknown} for family {key!r}")
    for arm, v in latency.items():
        by_arm[arm].setdefault("retrieval_latency_ms", {})[label] = v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="plot_data.json to create or merge into")
    ap.add_argument("--result-json", default=None,
                    help="analyze_size.py output; collects/replaces that family's block")
    ap.add_argument("--suite", default=None,
                    help="required with --result-json when the JSON lacks 'suite'")
    ap.add_argument("--grid", default=None, help="size grid yaml, for realized x values")
    ap.add_argument("--entries", default=None,
                    help='entries json ({"tiers": [...]} or {"S1": n, ...})')
    ap.add_argument("--bucket-sizes", default=None,
                    help="bucket_sizes json keyed by <suite>_<filter>_<tier>")
    ap.add_argument("--attach-latency", default=None,
                    help='json keyed by source_arm: {"cache_size_<suite>_<filt>_<tier>": '
                         '{...stats...}, ...}')
    ap.add_argument("--latency-label", default=None,
                    help="label for --attach-latency, e.g. wsl_optimized_r4")
    ap.add_argument("--family", default=None,
                    help="family key (<suite>/<filter>) for --attach-latency")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    data = load_data(out)

    if args.result_json:
        rp = pathlib.Path(args.result_json)
        result = json.loads(rp.read_text())
        if args.suite:
            result.setdefault("suite", args.suite)
        grid = None
        if args.grid:
            import yaml

            grid = yaml.safe_load(pathlib.Path(args.grid).read_text())
        entries = None
        if args.entries:
            raw = json.loads(pathlib.Path(args.entries).read_text())
            entries = ({t["tier"]: t["entries"] for t in raw["tiers"]}
                       if isinstance(raw, dict) and "tiers" in raw else raw)
        buckets = (json.loads(pathlib.Path(args.bucket_sizes).read_text())
                   if args.bucket_sizes else None)
        fam = collect_family(result, result_path=rp, grid=grid,
                             entries=entries, bucket_sizes=buckets)
        data["families"][family_key(fam["suite"], fam["outcome_filter"])] = fam

    if args.attach_latency:
        if not (args.latency_label and args.family):
            raise SystemExit("--attach-latency requires --latency-label and --family")
        attach_latency(data, key=args.family, label=args.latency_label,
                       latency=json.loads(pathlib.Path(args.attach_latency).read_text()))

    if not args.result_json and not args.attach_latency:
        raise SystemExit("nothing to do: pass --result-json and/or --attach-latency")

    out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"{out}: {len(data['families'])} families")


if __name__ == "__main__":
    main()
