"""E1 secondary read-outs: the phase-oracle contrast and the exploratory grid.

``e1_loeo_residual.run_family`` emits only the four registered primary cells.
The plan also pre-registers **E1-O** (section 5.1) -- group B restricted to
candidates whose normalised progress is within ``eps`` of the query's -- and
files the rest of the 192-cell grid as exploratory. Every one of those rows is
already in the parquet that ``run_family`` wrote, so this module re-reads them
instead of re-running LOEO.

Two things it deliberately does not do:

  * it does not re-run the family verdict -- ``family_analysis`` remains the
    sole path that may emit one;
  * it does not Holm-adjust the oracle cells. They are a pre-registered
    *secondary* read-out with a fixed decision rule (relative delta >= 5% and a
    CI lower bound above zero), not members of the primary family, and pooling
    them into that family after the fact would change the registered alpha
    allocation.

Public interface: :func:`load_rows`, :func:`oracle_cells`,
:func:`exploratory_cells`, :func:`main`.

Key dependencies: :mod:`e1_loeo_residual` (``aggregate``), :mod:`_stats`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any, Mapping, Optional

from exp.markov_sufficiency import e1_loeo_residual as e1

#: Plan section 5.1: H-B is supported only if the oracle-aligned B beats this.
ORACLE_EFFECT_FLOOR = 0.05

_ORACLE_A = re.compile(r"^O-A-e(?P<eps>[\d.]+)$")
_ORACLE_B = re.compile(r"^O-B-d(?P<depth>\d+)-e(?P<eps>[\d.]+)$")


# ------------------------------------------------------------------
# Row loading
# ------------------------------------------------------------------


def load_rows(path: str | pathlib.Path) -> list[dict[str, Any]]:
    """Read the residual rows written by ``run_family`` (parquet or JSONL)."""
    path = pathlib.Path(path)
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict("records")
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ------------------------------------------------------------------
# E1-O: the phase-oracle contrast
# ------------------------------------------------------------------


def _oracle_pairs(rows: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Discover the ``(eps, depth)`` cells actually present in the rows."""
    eps_seen, depth_seen = set(), set()
    for r in rows:
        m = _ORACLE_A.match(r["group"])
        if m:
            eps_seen.add(m.group("eps"))
            continue
        m = _ORACLE_B.match(r["group"])
        if m:
            eps_seen.add(m.group("eps"))
            depth_seen.add(int(m.group("depth")))
    return [(e, d) for e in sorted(eps_seen) for d in sorted(depth_seen)]


def _relabel(rows: list[dict[str, Any]], eps: str, depth: int) -> list[dict[str, Any]]:
    """Recast one oracle cell as the (A, B) shape ``aggregate`` expects.

    The baseline must be the oracle-aligned A, not the global A: comparing an
    aligned B against an unaligned A would fold the candidate-pool restriction
    into the contrast and stop measuring history at all.
    """
    a_name, b_name = f"O-A-e{eps}", f"O-B-d{depth}-e{eps}"
    out = []
    for r in rows:
        if r["group"] == a_name:
            out.append({**r, "group": "A"})
        elif r["group"] == b_name:
            out.append({**r, "group": "B"})
    return out


def _oracle_verdict(cell: dict[str, Any]) -> str:
    delta, lo = cell["relative_delta"], cell["hl_ci"][0]
    if delta != delta:  # NaN
        return "no_data"
    if lo > 0 and delta >= ORACLE_EFFECT_FLOOR:
        return "h_b_supported"
    if lo > 0:
        return "positive_but_below_floor"
    return "not_supported"


def oracle_cells(rows: list[dict[str, Any]], k: int = e1.PRIMARY_K) -> list[dict[str, Any]]:
    """E1-O per ``(suite, eps, depth)``, with the plan's fixed decision rule."""
    out = []
    for suite in sorted({r["suite"] for r in rows}):
        srows = [r for r in rows if r["suite"] == suite]
        for eps, depth in _oracle_pairs(srows):
            paired = _relabel(srows, eps, depth)
            if not paired:
                continue
            cell = e1.aggregate(paired, "B", k)
            cell.pop("paired_diffs", None)
            cell.update({"suite": suite, "eps": float(eps), "depth": depth, "k": k})
            cell["verdict"] = _oracle_verdict(cell)
            out.append(cell)
    return out


# ------------------------------------------------------------------
# Degeneracy guard for the oracle contrast
# ------------------------------------------------------------------


def oracle_pool_sizes(
    library_path: str | pathlib.Path,
    eps_values: tuple[float, ...] = (0.05, 0.10),
) -> list[dict[str, Any]]:
    """Size of the progress-aligned candidate pool, per ``eps``.

    A zero E1-O contrast is only informative if the aligned pool still offers a
    real choice. If alignment collapsed it to one entry, groups A and B would
    return the identical ``k=1`` neighbour and the contrast would be zero by
    construction -- the finding would be an artifact of the restriction rather
    than a statement about history. This reports the distribution so the claim
    can be checked instead of assumed.
    """
    import numpy as np

    from exp.markov_sufficiency import _library
    from exp.markov_sufficiency import e1_loeo_residual as e1_mod

    lib = _library.load_library(library_path)
    out = []
    for eps in eps_values:
        aligned, unaligned = [], []
        for held_out in sorted(lib.by_traj):
            pool = [e for e in lib.entries if e.trajectory_id != held_out]
            for entry in lib.by_traj[held_out]:
                cands = [e for e in pool if e.payload.task_key == entry.payload.task_key]
                if not cands:
                    continue
                progress = e1_mod._progress(lib, entry)
                keep = [c for c in cands if abs(e1_mod._progress(lib, c) - progress) <= eps]
                aligned.append(len(keep))
                unaligned.append(len(cands))
        arr = np.asarray(aligned, dtype=float)
        out.append(
            {
                "eps": eps,
                "n_queries": int(arr.size),
                "aligned_median": float(np.median(arr)) if arr.size else float("nan"),
                "aligned_p10": float(np.percentile(arr, 10)) if arr.size else float("nan"),
                "aligned_min": float(arr.min()) if arr.size else float("nan"),
                "share_degenerate": float(np.mean(arr <= 1)) if arr.size else float("nan"),
                "unaligned_median": float(np.median(unaligned)) if unaligned else float("nan"),
                # A contrast built on pools this small is not interpretable.
                "degenerate": bool(arr.size and float(np.mean(arr <= 1)) > 0.05),
            }
        )
    return out


# ------------------------------------------------------------------
# Exploratory grid
# ------------------------------------------------------------------


def exploratory_cells(
    rows: list[dict[str, Any]],
    ks: tuple[int, ...] = (1, 5),
    include_padding: bool = False,
) -> list[dict[str, Any]]:
    """Every non-oracle (suite, group, k) cell, reported without a verdict.

    These are the cells outside the registered family; they carry effect sizes
    and intervals so the report can show the full grid, but no cell here may be
    read as a decision.
    """
    out = []
    for suite in sorted({r["suite"] for r in rows}):
        srows = [r for r in rows if r["suite"] == suite]
        groups = sorted(
            g for g in {r["group"] for r in srows}
            if g != "A" and not _ORACLE_A.match(g) and not _ORACLE_B.match(g)
        )
        for group in groups:
            for k in ks:
                cell = e1.aggregate(srows, group, k, include_padding=include_padding)
                cell.pop("paired_diffs", None)
                cell.update({"suite": suite, "group": group, "k": k, "exploratory": True})
                out.append(cell)
    return out


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def analyse(
    paths: list[str],
    include_padding: bool = False,
    libraries: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Merge the per-suite artifacts and produce both secondary read-outs."""
    rows: list[dict[str, Any]] = []
    for p in paths:
        rows.extend(load_rows(p))
    if not rows:
        raise SystemExit(f"no residual rows found in {paths}")
    return {
        "n_rows": len(rows),
        "sources": list(paths),
        "oracle_effect_floor": ORACLE_EFFECT_FLOOR,
        "multiplicity": "not adjusted -- pre-registered secondary read-out, outside the primary family",
        "oracle": oracle_cells(rows),
        "oracle_pool": {s: oracle_pool_sizes(p) for s, p in (libraries or {}).items()},
        "exploratory": exploratory_cells(rows, include_padding=include_padding),
        "padding_included": include_padding,
    }


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="E1 secondary read-outs (E1-O + exploratory grid)")
    ap.add_argument("--rows", action="append", required=True, help="repeatable: residual parquet/JSONL")
    ap.add_argument("--include-padding", action="store_true", help="appendix variant: keep the first two frames")
    ap.add_argument(
        "--pool-diagnostic", action="append", default=[], metavar="SUITE=LIBRARY",
        help="repeatable: report the aligned candidate-pool sizes, so a zero E1-O contrast "
        "can be distinguished from a pool that alignment collapsed to one entry",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    libraries = {}
    for raw in args.pool_diagnostic:
        suite, _, path = raw.partition("=")
        if not (suite and path):
            ap.error(f"--pool-diagnostic must be SUITE=LIBRARY, got {raw!r}")
        libraries[suite] = path

    result = analyse(args.rows, include_padding=args.include_padding, libraries=libraries)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result["oracle"], indent=2))


if __name__ == "__main__":
    main()
