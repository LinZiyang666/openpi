"""Fit the dispatch surface: bimonotone quantile fit, OOF delta selection,
episode-level conformal correction, boundary export, artifact emission.

Implements plan section 4.5 mechanically (zero execution-time freedom):

  1. Cohort audit (G2-B8): the calibration-table rows must match the verified
     cohort manifest exactly — 50 fit / 100 cal episodes, 5/10 per task,
     unique (task, official init), fit/cal disjoint.
  2. Equal-frequency (s, v) binning with a mechanical 2D sparse-cell rule:
     grid descends (12,6) -> (8,4) -> (6,3), and EVERY Cartesian cell
     (including empty cells) must hold at least 8 samples; exhaustion is
     stop-loss A.
     (``--s-only`` uses (12,1) -> (8,1) -> (6,1).)
  3. Joint bimonotone pinball LP for tau in {7, 10} (nonincreasing in s,
     nondecreasing in v, layer-nested).
  4. Delta selection on the fit split only (SKIPPED under ``--s-only``, which
     REQUIRES ``--frozen-record`` and inherits delta* — G2-B2): task-stratified
     mod-5 folds; each fold refits its OWN edges + surface; one n=50 OOF
     safety offset (order statistic; not a conformal certificate); per
     candidate delta each fold exports its boundaries and every held-out row
     is judged by the SHARED ``surface_verdict`` — the deployed decision rule,
     not a proxy (G2-B1).
  5. Formal split conformal on the calibration split (|E| >= 19), boundary
     export at delta* (+ neighbours for the SV run), NPZ artifact emission
     with the full retrieval contract (action_dim derived from W).

Exit codes: 0 = artifacts written; 3 = pre-registered stop-loss A triggered.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
import pathlib

import numpy as np

from openpi.cache.components.surface_judge import (
    PINNED_START_T_WS,
    SURFACE_ARTIFACT_SCHEMA_VERSION,
    SurfaceArtifact,
    save_surface_artifact,
    surface_verdict,
)

TAUS = (7, 10)
N_STEPS = 10
STOP_LOSS_EXIT = 3

HITSHARE_TARGET = 0.40
ACCURACY_SLACK = 0.05
MIN_BIN_SAMPLES = 8
N_FOLDS = 5
GRID_LADDER_SV = ((12, 6), (8, 4), (6, 3))
GRID_LADDER_S_ONLY = ((12, 1), (8, 1), (6, 1))

EXPECTED_FIT = {"episodes": 50, "per_task": 5}
EXPECTED_CAL = {"episodes": 100, "per_task": 10}
N_TASKS = 10


# ------------------------------------------------------------------
# Data loading, cohort audit and binning
# ------------------------------------------------------------------


@dataclasses.dataclass
class Table:
    s: np.ndarray
    v: np.ndarray
    y7: np.ndarray
    y10: np.ndarray
    episode: np.ndarray
    task: np.ndarray
    init_idx: np.ndarray     # OFFICIAL init index (orig_init_state_idx)
    split: np.ndarray


def load_table(path: str, *, ref_mode: str = "fresh") -> Table:
    rows = [json.loads(line) for line in open(path)]
    rows = [r for r in rows if r["ref_mode"] == ref_mode and r["v"] is not None]
    if not rows:
        raise SystemExit(f"no usable rows with ref_mode={ref_mode} in {path}")
    return Table(
        s=np.array([r["s"] for r in rows], dtype=np.float64),
        v=np.array([r["v"] for r in rows], dtype=np.float64),
        y7=np.array([r["y_tau7"] for r in rows], dtype=np.float64),
        y10=np.array([r["y_tau10"] for r in rows], dtype=np.float64),
        episode=np.array([r["episode_id"] for r in rows]),
        task=np.array([r["task_id"] for r in rows], dtype=np.int64),
        init_idx=np.array([r["init_idx"] for r in rows], dtype=np.int64),
        split=np.array([r["split"] for r in rows]),
    )


def _file_sha256(path: pathlib.Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def validate_dlib_chain(
    split_manifest_path: pathlib.Path, rebuild_record_path: pathlib.Path,
) -> dict:
    """Require split-time D_lib authority to survive into the rebuilt library."""
    if not split_manifest_path.is_file():
        raise SystemExit(f"split manifest missing: {split_manifest_path}")
    if not rebuild_record_path.is_file():
        raise SystemExit(f"rebuild record missing: {rebuild_record_path}")
    split = json.loads(split_manifest_path.read_text())
    rebuild = json.loads(rebuild_record_path.read_text())
    split_digest = _file_sha256(split_manifest_path)
    if rebuild.get("split_manifest_sha256") != split_digest:
        raise SystemExit(
            "rebuild record was not produced from this split manifest — D_lib authority broken"
        )
    expected = split.get("dlib_content_digests")
    if not isinstance(expected, dict) or rebuild.get("dlib_content_digests") != expected:
        raise SystemExit(
            "rebuild record D_lib content digests differ from the split manifest"
        )
    return rebuild


def audit_cohort(table: Table, manifest_path: str, *, verify_files: bool = True) -> None:
    """Exact-quota / completeness / bijection audit against the verified
    cohort manifest (G2-B8 / G2R2-B2). Any mismatch aborts before fitting.

    Beyond membership, the table must COVER the manifest exactly: every one
    of the 150 (task, official init) identities present, identity <->
    episode_id a bijection (a table repeating 20 identities under fresh
    episode names must fail), and — when the manifest paths exist — every
    cohort file's sha256 recomputed against the manifest's claim.
    """
    manifest = json.loads(pathlib.Path(manifest_path).read_text())
    if verify_files:
        for f in manifest["files"]:
            p = pathlib.Path(f["path"])
            if not p.is_file():
                raise SystemExit(f"cohort file missing on disk: {p}")
            if _file_sha256(p) != f.get("sha256"):
                raise SystemExit(f"cohort file content drifted: {p}")
    expected: dict[tuple[int, int], str] = {}
    for f in manifest["files"]:
        key = (int(f["task_id"]), int(f["init_idx"]))
        if key in expected:
            raise SystemExit(f"cohort manifest duplicates {key}")
        expected[key] = f["split"]
    quota = collections.Counter((k[0], split) for k, split in expected.items())
    for t in range(N_TASKS):
        if quota[(t, "fit")] != EXPECTED_FIT["per_task"]:
            raise SystemExit(f"task {t}: manifest fit quota {quota[(t, 'fit')]} != 5")
        if quota[(t, "cal")] != EXPECTED_CAL["per_task"]:
            raise SystemExit(f"task {t}: manifest cal quota {quota[(t, 'cal')]} != 10")

    seen: dict[tuple[int, int], str] = {}
    id_to_episode: dict[tuple[int, int], str] = {}
    episode_to_id: dict[str, tuple[int, int]] = {}
    for i in range(len(table.s)):
        key = (int(table.task[i]), int(table.init_idx[i]))
        split = str(table.split[i])
        episode = str(table.episode[i])
        if key not in expected:
            raise SystemExit(f"table row {key} not in cohort manifest")
        if expected[key] != split:
            raise SystemExit(
                f"table row {key} labelled '{split}' but manifest says '{expected[key]}'"
            )
        prev = seen.setdefault(key, split)
        if prev != split:
            raise SystemExit(f"table row {key} carries conflicting split labels")
        # identity <-> episode bijection (multi-step episodes allowed).
        if id_to_episode.setdefault(key, episode) != episode:
            raise SystemExit(f"identity {key} maps to multiple episode_ids")
        if episode_to_id.setdefault(episode, key) != key:
            raise SystemExit(f"episode {episode!r} maps to multiple identities")
    missing = set(expected) - set(seen)
    if missing:
        raise SystemExit(
            f"table covers only {len(seen)}/{len(expected)} manifest identities "
            f"(missing e.g. {sorted(missing)[:3]}) — refusing partial cohorts"
        )
    for split_name, spec in (("fit", EXPECTED_FIT), ("cal", EXPECTED_CAL)):
        eps = {
            str(table.episode[i]) for i in range(len(table.s))
            if table.split[i] == split_name
        }
        if len(eps) != spec["episodes"]:
            raise SystemExit(
                f"{split_name} split has {len(eps)} episodes in the table, "
                f"expected exactly {spec['episodes']}"
            )


def equal_freq_edges(values: np.ndarray, n_bins: int, min_samples: int) -> np.ndarray:
    """Equal-frequency edges with 1-D small-bin merging."""
    if n_bins <= 1:
        return np.array([-np.inf, np.inf])
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(values, qs))
    if len(edges) < 3:
        return edges
    while len(edges) > 2:
        counts, _ = np.histogram(values, bins=edges)
        small = np.where(counts < min_samples)[0]
        if small.size == 0:
            break
        i = int(small[0])
        drop = i + 1 if i + 1 < len(edges) - 1 else i
        edges = np.delete(edges, drop)
    return edges


def bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)


def min_cell_occupancy(sb: np.ndarray, vb: np.ndarray, n_s: int, n_v: int) -> int:
    """Minimum occupancy over ALL cartesian (s, v) cells, empty ones included.

    Empty joint cells matter: the LP constrains them only through
    monotonicity, so an online query landing there would be judged by a
    boundary no local data ever supported (G2R2-B3).
    """
    counts = np.zeros((n_s, n_v))
    np.add.at(counts, (sb, vb), 1)
    return int(counts.min())


def choose_grid(s: np.ndarray, v: np.ndarray, ladder) -> tuple[np.ndarray, np.ndarray] | None:
    """Mechanical grid-descent: first ladder rung where EVERY cartesian cell
    (including empty ones) holds >= MIN_BIN_SAMPLES samples; exhaustion of
    the ladder is stop-loss A. Zero tolerance constants, zero freedom."""
    for b_s, b_v in ladder:
        s_edges = equal_freq_edges(s, b_s, MIN_BIN_SAMPLES)
        v_edges = equal_freq_edges(v, b_v, MIN_BIN_SAMPLES)
        sb, vb = bin_index(s, s_edges), bin_index(v, v_edges)
        if min_cell_occupancy(sb, vb, len(s_edges) - 1, len(v_edges) - 1) >= MIN_BIN_SAMPLES:
            return s_edges, v_edges
    return None


# ------------------------------------------------------------------
# Joint bimonotone pinball LP
# ------------------------------------------------------------------


def fit_bimonotone_quantile(
    s_bin: np.ndarray, v_bin: np.ndarray, y7: np.ndarray, y10: np.ndarray,
    n_s: int, n_v: int, alpha: float,
) -> np.ndarray:
    """Fit q[layer, s_bin, v_bin] at level 1 - alpha under lattice monotonicity."""
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix

    n_grid = 2 * n_s * n_v

    def gid(layer: int, i: int, j: int) -> int:
        return layer * n_s * n_v + i * n_v + j

    ys = [y7, y10]
    n_samples = len(s_bin)
    n_slack = 2 * 2 * n_samples
    n_var = n_grid + n_slack
    c = np.zeros(n_var)
    for layer in range(2):
        base = n_grid + layer * 2 * n_samples
        c[base:base + n_samples] = 1.0 - alpha
        c[base + n_samples:base + 2 * n_samples] = alpha

    rows: list[tuple[dict[int, float], float]] = []
    for layer in range(2):
        base = n_grid + layer * 2 * n_samples
        for idx in range(n_samples):
            g = gid(layer, int(s_bin[idx]), int(v_bin[idx]))
            y = ys[layer][idx]
            rows.append(({g: -1.0, base + idx: -1.0}, -y))
            rows.append(({g: 1.0, base + n_samples + idx: -1.0}, y))
    for layer in range(2):
        for i in range(n_s - 1):
            for j in range(n_v):
                rows.append(({gid(layer, i + 1, j): 1.0, gid(layer, i, j): -1.0}, 0.0))
        for i in range(n_s):
            for j in range(n_v - 1):
                rows.append(({gid(layer, i, j): 1.0, gid(layer, i, j + 1): -1.0}, 0.0))
    for i in range(n_s):
        for j in range(n_v):
            rows.append(({gid(0, i, j): 1.0, gid(1, i, j): -1.0}, 0.0))

    a_ub = lil_matrix((len(rows), n_var))
    b_ub = np.zeros(len(rows))
    for r, (coefs, b) in enumerate(rows):
        for col, val in coefs.items():
            a_ub[r, col] = val
        b_ub[r] = b
    bounds = [(None, None)] * n_grid + [(0, None)] * n_slack
    res = linprog(c, A_ub=a_ub.tocsr(), b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"bimonotone LP failed: {res.message}")
    return res.x[:n_grid].reshape(2, n_s, n_v)


# ------------------------------------------------------------------
# OOF fold models and delta selection (deployed-verdict semantics)
# ------------------------------------------------------------------


@dataclasses.dataclass
class FoldModel:
    q: np.ndarray          # [2, n_s, n_v]
    s_edges: np.ndarray
    v_edges: np.ndarray
    heldout_local: np.ndarray  # bool over fit rows


def assign_folds(table: Table, fit_mask: np.ndarray) -> np.ndarray:
    folds = np.full(len(table.s), -1, dtype=np.int64)
    for task in np.unique(table.task[fit_mask]):
        m = fit_mask & (table.task == task)
        inits = np.unique(table.init_idx[m])
        rank = {int(x): r for r, x in enumerate(sorted(inits))}
        for init_val, r in rank.items():
            folds[m & (table.init_idx == init_val)] = r % N_FOLDS
    return folds


def fit_fold_models(
    table: Table, fit_mask: np.ndarray, folds: np.ndarray,
    ladder, alpha: float,
) -> list[FoldModel] | None:
    """Per-fold: OWN edges from fold-train data, own LP surface (G2-B1)."""
    fit_idx = np.where(fit_mask)[0]
    models: list[FoldModel] = []
    for f in range(N_FOLDS):
        train_local = folds[fit_idx] != f
        s_tr = table.s[fit_idx][train_local]
        v_tr = table.v[fit_idx][train_local]
        grid = choose_grid(s_tr, v_tr, ladder)
        if grid is None:
            return None
        s_edges, v_edges = grid
        sb = bin_index(s_tr, s_edges)
        vb = bin_index(v_tr, v_edges)
        q = fit_bimonotone_quantile(
            sb, vb, table.y7[fit_idx][train_local], table.y10[fit_idx][train_local],
            len(s_edges) - 1, len(v_edges) - 1, alpha,
        )
        models.append(FoldModel(q=q, s_edges=s_edges, v_edges=v_edges,
                                heldout_local=~train_local))
    return models


def oof_predictions(models: list[FoldModel], table: Table,
                    fit_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fit_idx = np.where(fit_mask)[0]
    q7 = np.full(len(fit_idx), np.nan)
    q10 = np.full(len(fit_idx), np.nan)
    for m in models:
        held = m.heldout_local
        sb = bin_index(table.s[fit_idx][held], m.s_edges)
        vb = bin_index(table.v[fit_idx][held], m.v_edges)
        q7[held] = m.q[0, sb, vb]
        q10[held] = m.q[1, sb, vb]
    if np.isnan(q7).any():
        raise SystemExit("OOF coverage incomplete — fold assignment bug")
    return q7, q10


def episode_max_residual(q7, q10, y7, y10, episodes) -> dict[str, float]:
    res = np.maximum(y7 - q7, y10 - q10)
    return {str(ep): float(res[episodes == ep].max()) for ep in np.unique(episodes)}


def order_statistic_offset(residuals: list[float], alpha: float) -> float:
    n = len(residuals)
    k = math.ceil((1 - alpha) * (n + 1))
    if k > n:
        return float("inf")
    return float(np.sort(residuals)[k - 1])


def export_boundaries(q_grid: np.ndarray, s_edges: np.ndarray,
                      delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Conservative per-v-bin s thresholds for (full=tau10, warm=tau7)."""
    n_v = q_grid.shape[2]
    warm = np.full(n_v, np.inf)
    full = np.full(n_v, np.inf)
    for j in range(n_v):
        for layer, target in ((0, warm), (1, full)):
            ok = np.where(q_grid[layer, :, j] <= delta)[0]
            if ok.size:
                target[j] = s_edges[int(ok.min()) + 1]
    if (warm > full).any():
        raise RuntimeError("boundary nesting violated after export")
    return full, warm


def evaluate_candidate_deployed(
    delta: float, models: list[FoldModel], table: Table, fit_mask: np.ndarray,
    offset: float, *, uses_disagreement: bool,
) -> tuple[float, float]:
    """(hitshare, accepted_step_accuracy) by EXECUTING the deployed verdict.

    For every fold: export the fold surface's boundaries at this delta (with
    the OOF safety offset applied to the quantile grid) and run the shared
    ``surface_verdict`` on each held-out row — s participates exactly as it
    does online, including boundary rounding and v support-domain semantics.
    """
    fit_idx = np.where(fit_mask)[0]
    verdicts = np.empty(len(fit_idx), dtype=object)
    for m in models:
        full, warm = export_boundaries(m.q + offset, m.s_edges, delta)
        v_edges = m.v_edges if uses_disagreement else np.array([-np.inf, np.inf])
        for local_i in np.where(m.heldout_local)[0]:
            row = fit_idx[local_i]
            verdicts[local_i] = surface_verdict(
                float(table.s[row]), float(table.v[row]),
                v_edges, full, warm, uses_disagreement=uses_disagreement,
            )
    accepted = verdicts != "miss"
    hitshare = float(accepted.mean())
    if not accepted.any():
        return hitshare, 1.0
    y_eff = np.where(
        verdicts[accepted] == "full",
        table.y10[fit_mask][accepted], table.y7[fit_mask][accepted],
    )
    return hitshare, float((y_eff <= delta).mean())


def select_delta(candidates: np.ndarray, metrics: dict, alpha: float):
    acc_gate = 1.0 - alpha - ACCURACY_SLACK
    passing = [d for d in candidates if metrics[d][1] >= acc_gate]
    if not passing:
        return None, "stop_loss_accuracy_gate"
    qualified = [d for d in passing if metrics[d][0] >= HITSHARE_TARGET]
    pool = qualified if qualified else passing
    best_share = max(metrics[d][0] for d in pool)
    if best_share <= 0.0:
        return None, "stop_loss_zero_hitshare"
    winners = [d for d in pool if metrics[d][0] == best_share]
    return float(min(winners)), ("qualified" if qualified else "fallback_accuracy_only")


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------


def _stop(record: dict, out_dir: pathlib.Path, reason: str, s_only: bool) -> None:
    record["stop_loss"] = reason
    name = "fit_record_s_only.json" if s_only else "fit_record.json"
    (out_dir / name).write_text(json.dumps(record, indent=2))
    raise SystemExit(STOP_LOSS_EXIT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", required=True)
    ap.add_argument("--cohort-manifest", required=True,
                    help="collect_query_cohort verify output (identity audit)")
    ap.add_argument("--cache-yaml", required=True)
    ap.add_argument("--rebuild-record", required=True)
    ap.add_argument("--split-manifest", required=True,
                    help="split_init_pools manifest bound into the rebuild record")
    ap.add_argument("--weights-npz", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--h-exec", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--s-only", action="store_true")
    ap.add_argument("--frozen-record", default=None,
                    help="REQUIRED with --s-only: the SV fit_record.json whose "
                         "delta_star this run inherits (G2-B2; no re-selection)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rebuild_record_path = pathlib.Path(args.rebuild_record)
    rebuild = validate_dlib_chain(pathlib.Path(args.split_manifest), rebuild_record_path)

    if args.s_only and not args.frozen_record:
        raise SystemExit("--s-only requires --frozen-record (delta* is frozen by the SV fit)")
    if not args.s_only and args.frozen_record:
        raise SystemExit("--frozen-record is only meaningful with --s-only")

    table = load_table(args.table)
    audit_cohort(table, args.cohort_manifest)
    fit_mask = table.split == "fit"
    cal_mask = table.split == "cal"
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Canonical content digests of EVERY fitting input. The SV record freezes
    # them; the s-only run must fit on byte-identical inputs (G2R2-B6).
    input_digests = {
        "table": _file_sha256(pathlib.Path(args.table)),
        "cohort_manifest": _file_sha256(pathlib.Path(args.cohort_manifest)),
        "weights_npz": _file_sha256(pathlib.Path(args.weights_npz)),
        "rebuild_record": _file_sha256(pathlib.Path(args.rebuild_record)),
        "split_manifest": _file_sha256(pathlib.Path(args.split_manifest)),
        "cache_yaml": _file_sha256(pathlib.Path(args.cache_yaml)),
    }
    record: dict = {"s_only": args.s_only, "alpha": args.alpha,
                    "cohort_manifest": str(args.cohort_manifest),
                    "input_digests": input_digests}
    ladder = GRID_LADDER_S_ONLY if args.s_only else GRID_LADDER_SV
    uses_v = not args.s_only

    # -- final binning grid (fit split, mechanical ladder) ----------
    grid = choose_grid(table.s[fit_mask], table.v[fit_mask], ladder)
    if grid is None:
        _stop(record, out_dir, "stop_loss_sparse_cells", args.s_only)
    s_edges, v_edges = grid
    record["n_s_bins"], record["n_v_bins"] = len(s_edges) - 1, len(v_edges) - 1

    # -- delta: select (SV) or inherit (s-only) --------------------
    if args.s_only:
        frozen = json.loads(pathlib.Path(args.frozen_record).read_text())
        if "delta_star" not in frozen:
            raise SystemExit("frozen record carries no delta_star")
        frozen_digests = frozen.get("input_digests") or {}
        for name, digest in input_digests.items():
            if frozen_digests.get(name) != digest:
                raise SystemExit(
                    f"s-only input {name!r} digest {digest[:12]}… differs from the "
                    f"SV frozen record ({str(frozen_digests.get(name))[:12]}…) — the "
                    "nested ablation must fit on byte-identical inputs (G2R2-B6)"
                )
        delta_star = float(frozen["delta_star"])
        neighbours = {"minus": None, "plus": None}
        record["delta_star"] = delta_star
        record["frozen_from"] = str(args.frozen_record)
        record["delta_selection_reason"] = "frozen_from_sv"
    else:
        folds = assign_folds(table, fit_mask)
        models = fit_fold_models(table, fit_mask, folds, ladder, args.alpha)
        if models is None:
            _stop(record, out_dir, "stop_loss_sparse_cells_oof", args.s_only)
        q_oof7, q_oof10 = oof_predictions(models, table, fit_mask)
        fit_idx = np.where(fit_mask)[0]
        oof_res = episode_max_residual(
            q_oof7, q_oof10, table.y7[fit_idx], table.y10[fit_idx], table.episode[fit_idx],
        )
        if len(oof_res) < 19:
            _stop(record, out_dir, "fit_episode_count_below_19", args.s_only)
        oof_offset = order_statistic_offset(list(oof_res.values()), args.alpha)
        record["oof_safety_offset"] = oof_offset
        record["n_fit_episodes"] = len(oof_res)

        grid_d = np.unique(np.percentile(table.y10[fit_mask], np.arange(10, 100, 10)))
        record["delta_grid"] = grid_d.tolist()
        if len(grid_d) < 2:
            _stop(record, out_dir, "degenerate_delta_grid", args.s_only)
        metrics = {
            float(d): evaluate_candidate_deployed(
                float(d), models, table, fit_mask, oof_offset, uses_disagreement=True,
            )
            for d in grid_d
        }
        record["delta_metrics"] = {
            str(d): {"hitshare": m[0], "accepted_step_accuracy": m[1]}
            for d, m in metrics.items()
        }
        delta_star, reason = select_delta(grid_d, metrics, args.alpha)
        record["delta_selection_reason"] = reason
        if delta_star is None:
            _stop(record, out_dir, reason, args.s_only)
        record["delta_star"] = delta_star
        pos = int(np.where(grid_d == delta_star)[0][0])
        neighbours = {
            "minus": float(grid_d[pos - 1]) if pos > 0 else None,
            "plus": float(grid_d[pos + 1]) if pos + 1 < len(grid_d) else None,
        }
        record["delta_neighbours"] = neighbours

    # -- formal fit + split conformal ------------------------------
    sb_fit = bin_index(table.s[fit_mask], s_edges)
    vb_fit = bin_index(table.v[fit_mask], v_edges)
    q_hat = fit_bimonotone_quantile(
        sb_fit, vb_fit, table.y7[fit_mask], table.y10[fit_mask],
        len(s_edges) - 1, len(v_edges) - 1, args.alpha,
    )
    sb_cal = bin_index(table.s[cal_mask], s_edges)
    vb_cal = bin_index(table.v[cal_mask], v_edges)
    cal_res = episode_max_residual(
        q_hat[0, sb_cal, vb_cal], q_hat[1, sb_cal, vb_cal],
        table.y7[cal_mask], table.y10[cal_mask], table.episode[cal_mask],
    )
    record["n_calibration_episodes"] = len(cal_res)
    if len(cal_res) < 19:
        _stop(record, out_dir, "calibration_episode_count_below_19", args.s_only)
    c = order_statistic_offset(list(cal_res.values()), args.alpha)
    record["conformal_c"] = c
    q_tilde = q_hat + c

    # -- diagnostics (SV run only; A1 on unconstrained bin quantiles)
    if not args.s_only:
        n_s, n_v = len(s_edges) - 1, len(v_edges) - 1
        emp_q = np.full((2, n_s, n_v), np.nan)
        for layer, y in ((0, table.y7[fit_mask]), (1, table.y10[fit_mask])):
            for i in range(n_s):
                for j in range(n_v):
                    m = (sb_fit == i) & (vb_fit == j)
                    if m.sum() >= MIN_BIN_SAMPLES:
                        emp_q[layer, i, j] = np.quantile(y[m], 1 - args.alpha)
        viol = total = 0
        for layer in range(2):
            for j in range(n_v):
                col = emp_q[layer, :, j]
                for i in range(n_s - 1):
                    if np.isfinite(col[i]) and np.isfinite(col[i + 1]):
                        total += 1
                        viol += int(col[i + 1] > col[i])
            for i in range(n_s):
                rr = emp_q[layer, i, :]
                for j in range(n_v - 1):
                    if np.isfinite(rr[j]) and np.isfinite(rr[j + 1]):
                        total += 1
                        viol += int(rr[j + 1] < rr[j])
        record["a1_violation_rate"] = (viol / total) if total else None
        z = (table.y10[fit_mask] <= delta_star).astype(float)
        zb = bin_index(table.s[fit_mask], np.unique(np.quantile(
            table.s[fit_mask], np.linspace(0, 1, 11))))
        rates = [float(z[zb == b].mean()) for b in range(zb.max() + 1) if (zb == b).any()]
        record["mlr_rate_by_s_decile"] = rates
        record["mlr_monotone_violations"] = int(
            sum(rates[i + 1] < rates[i] for i in range(len(rates) - 1))
        )
        if record["a1_violation_rate"] is not None and record["a1_violation_rate"] > 0.20:
            _stop(record, out_dir, "a1_violation_rate_above_20pct", args.s_only)

    # -- contract + artifact emission ------------------------------
    from openpi.cache.config import compute_surface_retrieval_contract, load_cache_config
    from openpi.serving.policy_identity import compute_policy_fingerprint, resolve_checkpoint_root

    weights = np.load(args.weights_npz)
    action_dim = int(np.asarray(weights["w"]).shape[0])
    config = load_cache_config(args.cache_yaml)
    contract = compute_surface_retrieval_contract(config)
    ckpt_root = resolve_checkpoint_root(rebuild["checkpoint_dir"])
    contract.update({
        "library_sha256": rebuild["library_sha256"],
        "library_entry_count": rebuild["entry_count"],
        "action_dim": action_dim,
        "num_steps": N_STEPS,
        "h_exec": args.h_exec,
        "policy_fingerprint": compute_policy_fingerprint(
            str(ckpt_root), rebuild["config_name"],
        ),
    })
    if args.s_only:
        contract["top_k"] = args.top_k  # width is a retrieval fact, not a v need

    tag = "s_only" if args.s_only else "sv"
    emitted = {}
    for name, delta in (
        ("primary", delta_star),
        ("minus", neighbours["minus"]),
        ("plus", neighbours["plus"]),
    ):
        if delta is None:
            continue
        full, warm = export_boundaries(q_tilde, s_edges, delta)
        artifact = SurfaceArtifact(
            schema_version=SURFACE_ARTIFACT_SCHEMA_VERSION,
            k=args.top_k,
            h_exec=args.h_exec,
            w=weights["w"],
            active_mask=weights["active_mask"],
            start_t_ws=PINNED_START_T_WS,
            delta=float(delta),
            alpha=args.alpha,
            uses_disagreement=uses_v,
            v_bin_edges=v_edges,
            s_min_full=full,
            s_min_warm=warm,
            conformal_c=c,
            n_calibration_episodes=len(cal_res),
            retrieval_contract=contract,
            meta={
                "ref_mode": "fresh",
                "delta_name": name,
                "delta_selection_reason": record["delta_selection_reason"],
                "oof_safety_offset": record.get("oof_safety_offset"),
                "frozen_from": record.get("frozen_from"),
                "table": str(args.table),
                "input_digests": input_digests,
                "deviation_metric": "mean over h_exec steps of weighted L2 (active dims)",
            },
        )
        path = out_dir / f"surface_{tag}_{name}.npz"
        save_surface_artifact(artifact, str(path))
        emitted[name] = str(path)
    record["artifacts"] = emitted
    (out_dir / ("fit_record_s_only.json" if args.s_only else "fit_record.json")).write_text(
        json.dumps(record, indent=2)
    )
    print(json.dumps({"delta_star": delta_star,
                      "reason": record["delta_selection_reason"],
                      "artifacts": emitted}, indent=2))


if __name__ == "__main__":
    main()
