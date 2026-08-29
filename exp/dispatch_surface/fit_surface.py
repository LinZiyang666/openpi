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
    CERTIFICATION_CONFORMAL,
    CERTIFICATION_EMPIRICAL,
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
# Rev 1 development set: the whole verified cohort, 15 episodes per task.
EXPECTED_DEV_EPISODES = 150
EXPECTED_TASKS = 10
# Rev 1 formal parameters. The empirical mode is the protocol's only formal
# path, so these are contract, not defaults: an artifact fitted at some other
# alpha/h_exec/width would still be accepted by emitter and runner, and would
# not be the surface the protocol froze.
FORMAL_ALPHA = 0.05
FORMAL_H_EXEC = 5
FORMAL_SV_TOP_K = 5
FORMAL_S0_TOP_K = 1
# The delta each suite's D_dev mechanically yields, frozen by the Rev 1 ruling.
# The fit recomputes it from the table and must land here; the constant exists
# to catch a changed input, not to substitute for the computation.
FROZEN_DELTA_STAR = {
    "libero_spatial": 6.1298201,
    "libero_10": 5.9096355,
}
FROZEN_DELTA_TOL = 1e-6
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


def validate_d0_record(
    d0_path: pathlib.Path,
    *,
    table_path: pathlib.Path,
    weights_path: pathlib.Path,
    cache_yaml_path: pathlib.Path,
    split_manifest_path: pathlib.Path,
    rebuild: dict,
) -> dict:
    """Authenticate the D0 replay and bind it to this exact formal fit.

    D0 is not a free-standing narrative result: it is the executable bridge
    between the current warm-start implementation and the y7/y10 bytes being
    fitted. Recompute its complete input attestation here so copying a PASS JSON
    next to different inputs cannot authorize an artifact.
    """
    from exp.dispatch_surface.d0_check import (
        CONTROL_ROWS_PER_TASK,
        D0_PROTOCOL,
        EXPECTED_TASKS,
        FORMAL_H_EXEC as D0_FORMAL_H_EXEC,
        validate_input_attestation,
    )
    from openpi.serving.policy_identity import compute_policy_fingerprint, resolve_checkpoint_root

    if not d0_path.is_file():
        raise SystemExit(f"D0 record missing: {d0_path}")
    d0 = json.loads(d0_path.read_text())
    split = json.loads(split_manifest_path.read_text())
    if d0.get("protocol") != D0_PROTOCOL or d0.get("D0") != "PASS":
        raise SystemExit("fit requires a PASS record from the frozen Rev 1 D0 protocol")
    if d0.get("suite") != split.get("suite"):
        raise SystemExit("D0 suite does not match the split manifest suite")
    if d0.get("h_exec") != D0_FORMAL_H_EXEC:
        raise SystemExit(f"D0 h_exec must be {D0_FORMAL_H_EXEC}")

    census = d0.get("census") or {}
    check1 = d0.get("check1_self_resume_parity") or {}
    check2 = d0.get("check2_payload_sidecar_identity") or {}
    check3 = d0.get("check3_path_decomposition") or {}
    if census.get("passed") is not True or census.get("problems") != []:
        raise SystemExit("D0 census is missing, failed, or carries unresolved problems")
    for label, check in (("self-resume", check1), ("payload/sidecar", check2)):
        if check.get("passed") is not True or check.get("failures") != 0 or not check.get("n"):
            raise SystemExit(f"D0 {label} check is incomplete or failed")
    if (check3.get("complete") is not True
            or check3.get("table_semantics_passed") is not True
            or not check3.get("n")):
        raise SystemExit("D0 sampled table-semantics replay is incomplete or failed")
    sample = d0.get("sample") or {}
    if sample.get("control_rows") != CONTROL_ROWS_PER_TASK * EXPECTED_TASKS:
        raise SystemExit("D0 does not contain the frozen two controls per task")
    if sample.get("tasks_covered") != list(range(EXPECTED_TASKS)):
        raise SystemExit("D0 replay does not cover all ten formal tasks")
    if not isinstance(sample.get("rows_sha256"), str) or len(sample["rows_sha256"]) != 64:
        raise SystemExit("D0 sample identity digest is missing")

    attestation = d0.get("inputs")
    if not isinstance(attestation, dict):
        raise SystemExit("D0 input attestation is missing")
    try:
        validate_input_attestation(attestation)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"D0 input attestation is invalid: {exc}") from exc
    files = attestation["files"]
    current = {
        "table": _file_sha256(table_path),
        "weights_npz": _file_sha256(weights_path),
        "cache_yaml": _file_sha256(cache_yaml_path),
    }
    for name, digest in current.items():
        if files[name].get("sha256") != digest:
            raise SystemExit(f"D0 {name} digest does not match this fit input")
    if files["library_pkl"].get("sha256") != rebuild.get("library_sha256"):
        raise SystemExit("D0 library digest does not match the rebuild authority")
    if files["noise_sidecar"].get("sha256") != rebuild.get("noise_sidecar_sha256"):
        raise SystemExit("D0 noise sidecar digest does not match the rebuild authority")
    expected_policy = compute_policy_fingerprint(
        str(resolve_checkpoint_root(rebuild["checkpoint_dir"])), rebuild["config_name"],
    )
    if attestation["policy"].get("policy_fingerprint") != expected_policy:
        raise SystemExit("D0 policy fingerprint does not match the rebuild authority")
    return {
        "record_sha256": _file_sha256(d0_path),
        "input_rollup_sha256": attestation.get("rollup_sha256"),
        "sample_rows_sha256": sample["rows_sha256"],
        "suite": d0["suite"],
    }


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


@dataclasses.dataclass
class FinalFit:
    """The formal (all-development-rows) surface fit and its binning grid."""

    q_hat: np.ndarray
    s_edges: np.ndarray
    v_edges: np.ndarray
    sb: np.ndarray
    vb: np.ndarray


def final_fit(
    table: Table, dev_mask: np.ndarray, *, alpha: float,
    edges: tuple[np.ndarray, np.ndarray] | None = None,
    ladder=None,
) -> FinalFit | None:
    """Pure final fit on ``dev_mask``: bin on the grid, fit the bimonotone quantile.

    Exactly the computation ``main`` performs for the deployed q-grid, exposed
    so the Rev 2 exploratory exporter can recompute it from the frozen inputs
    and compare digests. ``edges`` supplies a pre-chosen grid (as ``main``
    does); otherwise the mechanical ladder picks it and ``None`` signals the
    sparse-cell stop-loss.
    """
    if edges is None:
        if ladder is None:
            raise ValueError("final_fit needs either edges or a grid ladder")
        grid = choose_grid(table.s[dev_mask], table.v[dev_mask], ladder)
        if grid is None:
            return None
        s_edges, v_edges = grid
    else:
        s_edges, v_edges = edges
    sb = bin_index(table.s[dev_mask], s_edges)
    vb = bin_index(table.v[dev_mask], v_edges)
    q_hat = fit_bimonotone_quantile(
        sb, vb, table.y7[dev_mask], table.y10[dev_mask],
        len(s_edges) - 1, len(v_edges) - 1, alpha,
    )
    return FinalFit(q_hat=q_hat, s_edges=s_edges, v_edges=v_edges, sb=sb, vb=vb)


def assign_folds(table: Table, dev_mask: np.ndarray) -> np.ndarray:
    folds = np.full(len(table.s), -1, dtype=np.int64)
    for task in np.unique(table.task[dev_mask]):
        m = dev_mask & (table.task == task)
        inits = np.unique(table.init_idx[m])
        rank = {int(x): r for r, x in enumerate(sorted(inits))}
        for init_val, r in rank.items():
            folds[m & (table.init_idx == init_val)] = r % N_FOLDS
    return folds


def fit_fold_models(
    table: Table, dev_mask: np.ndarray, folds: np.ndarray,
    ladder, alpha: float,
) -> list[FoldModel] | None:
    """Per-fold: OWN edges from fold-train data, own LP surface (G2-B1)."""
    dev_idx = np.where(dev_mask)[0]
    models: list[FoldModel] = []
    for f in range(N_FOLDS):
        train_local = folds[dev_idx] != f
        s_tr = table.s[dev_idx][train_local]
        v_tr = table.v[dev_idx][train_local]
        grid = choose_grid(s_tr, v_tr, ladder)
        if grid is None:
            return None
        s_edges, v_edges = grid
        sb = bin_index(s_tr, s_edges)
        vb = bin_index(v_tr, v_edges)
        q = fit_bimonotone_quantile(
            sb, vb, table.y7[dev_idx][train_local], table.y10[dev_idx][train_local],
            len(s_edges) - 1, len(v_edges) - 1, alpha,
        )
        models.append(FoldModel(q=q, s_edges=s_edges, v_edges=v_edges,
                                heldout_local=~train_local))
    return models


def oof_predictions(models: list[FoldModel], table: Table,
                    dev_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dev_idx = np.where(dev_mask)[0]
    q7 = np.full(len(dev_idx), np.nan)
    q10 = np.full(len(dev_idx), np.nan)
    for m in models:
        held = m.heldout_local
        sb = bin_index(table.s[dev_idx][held], m.s_edges)
        vb = bin_index(table.v[dev_idx][held], m.v_edges)
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
    delta: float, models: list[FoldModel], table: Table, dev_mask: np.ndarray,
    offset: float, *, uses_disagreement: bool,
) -> tuple[float, float]:
    """(hitshare, accepted_step_accuracy) by EXECUTING the deployed verdict.

    For every fold: export the fold surface's boundaries at this delta (with
    the OOF safety offset applied to the quantile grid) and run the shared
    ``surface_verdict`` on each held-out row — s participates exactly as it
    does online, including boundary rounding and v support-domain semantics.
    """
    dev_idx = np.where(dev_mask)[0]
    verdicts = np.empty(len(dev_idx), dtype=object)
    for m in models:
        full, warm = export_boundaries(m.q + offset, m.s_edges, delta)
        v_edges = m.v_edges if uses_disagreement else np.array([-np.inf, np.inf])
        for local_i in np.where(m.heldout_local)[0]:
            row = dev_idx[local_i]
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
        table.y10[dev_mask][accepted], table.y7[dev_mask][accepted],
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


def _validate_d0_record(args) -> dict:
    """Re-check the D0 report against the inputs THIS fit is about to use.

    D0 was a side script: nothing forced a fit to have run it, and a report
    naming its table by path could vouch for a file that had since changed.
    Both holes close here. The attestation is re-attested from disk by D0's own
    validator -- hashing the JSON would only authenticate a statement about
    paths, not the bytes that were replayed -- and the digests it records must
    equal this run's files.
    """
    from exp.dispatch_surface.d0_check import D0_PROTOCOL, validate_input_attestation

    record = json.loads(pathlib.Path(args.d0_record).read_text())
    if record.get("D0") != "PASS":
        raise SystemExit(
            f"D0 record says {record.get('D0')!r}; the fit may not run on inputs "
            "whose data semantics have not been cleared"
        )
    if record.get("protocol") != D0_PROTOCOL:
        raise SystemExit(
            f"D0 record protocol {record.get('protocol')!r} != {D0_PROTOCOL!r}; "
            "re-run D0 under the current checks"
        )
    attestation = record.get("inputs")
    if not isinstance(attestation, dict):
        raise SystemExit("D0 record carries no input attestation")
    try:
        validate_input_attestation(attestation)
    except ValueError as exc:
        raise SystemExit(f"D0 input attestation no longer re-attests: {exc}") from exc

    files = attestation["files"]
    for name, path in (("table", args.table),
                       ("weights_npz", args.weights_npz),
                       ("cache_yaml", args.cache_yaml)):
        actual = _file_sha256(pathlib.Path(path))
        recorded = files.get(name, {}).get("sha256")
        if recorded != actual:
            raise SystemExit(
                f"D0 cleared {name} {str(recorded)[:12]}… but this fit was handed "
                f"{path} ({actual[:12]}…); they are not the same file"
            )

    # The library and its noise sidecar are not fit_surface arguments, so the
    # comparison above cannot reach them: a D0 record that cleared library A
    # could be handed to a fit whose rebuild record describes library B at a
    # different path, and D0's own validator -- which re-attests its OWN
    # recorded paths -- would still pass. Bind them through the rebuild record,
    # whose library_sha256/noise_sidecar_sha256 are those files' digests.
    rebuild = json.loads(pathlib.Path(args.rebuild_record).read_text())
    for d0_name, rebuild_key in (("library_pkl", "library_sha256"),
                                 ("noise_sidecar", "noise_sidecar_sha256")):
        cleared = files.get(d0_name, {}).get("sha256")
        declared = rebuild.get(rebuild_key)
        if cleared != declared:
            raise SystemExit(
                f"D0 cleared {d0_name} {str(cleared)[:12]}… but the rebuild record "
                f"declares {rebuild_key}={str(declared)[:12]}…; the audited library "
                "and the fitted library are not the same artifact"
            )

    # Cheap and explicit: a record from the other suite would already fail the
    # table digest, but saying so directly beats relying on that side effect.
    split_suite = json.loads(pathlib.Path(args.split_manifest).read_text()).get("suite")
    if record.get("suite") != split_suite:
        raise SystemExit(
            f"D0 record is for suite {record.get('suite')!r} but this fit's split "
            f"manifest is {split_suite!r}"
        )
    return record


def _digest_obj(obj) -> str:
    """Canonical sha256 of a JSON-serialisable audit payload."""
    import hashlib

    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _development_audit(
    table: Table, dev_mask: np.ndarray, folds: np.ndarray | None,
) -> tuple[list, list]:
    """Return canonical episode membership and fold assignment.

    Every row in an episode must agree on task/init/fold. This is stronger than
    selecting an arbitrary first row and makes the digests meaningful audit
    objects rather than hashes of whichever duplicate happened to come first.
    """
    membership = []
    fold_map = []
    for episode in sorted(str(ep) for ep in np.unique(table.episode[dev_mask])):
        mask = dev_mask & (table.episode.astype(str) == episode)
        tasks = np.unique(table.task[mask])
        inits = np.unique(table.init_idx[mask])
        if len(tasks) != 1 or len(inits) != 1:
            raise SystemExit(f"episode {episode!r} has inconsistent task/init identity")
        membership.append((episode, int(tasks[0]), int(inits[0])))
        if folds is not None:
            assigned = np.unique(folds[mask])
            if len(assigned) != 1 or int(assigned[0]) not in range(N_FOLDS):
                raise SystemExit(f"episode {episode!r} has invalid/inconsistent fold")
            fold_map.append((episode, int(assigned[0])))
    return membership, fold_map


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
    ap.add_argument(
        "--d0-record", required=True,
        help="PASS output from d0_check for this exact table/library/model",
    )
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--h-exec", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--s-only", action="store_true")
    ap.add_argument("--frozen-record", default=None,
                    help="REQUIRED with --s-only: the SV fit_record.json whose "
                         "delta_star this run inherits (G2-B2; no re-selection)")
    ap.add_argument(
        "--certification-mode", required=True,
        choices=[CERTIFICATION_EMPIRICAL, CERTIFICATION_CONFORMAL],
        help="Rev 1 emits only the empirical mode; the conformal branch is kept "
             "explicit so a certified artifact can never be produced by accident",
    )
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    d0_record = _validate_d0_record(args)
    rebuild_record_path = pathlib.Path(args.rebuild_record)
    rebuild = validate_dlib_chain(pathlib.Path(args.split_manifest), rebuild_record_path)
    d0_binding = validate_d0_record(
        pathlib.Path(args.d0_record),
        table_path=pathlib.Path(args.table),
        weights_path=pathlib.Path(args.weights_npz),
        cache_yaml_path=pathlib.Path(args.cache_yaml),
        split_manifest_path=pathlib.Path(args.split_manifest),
        rebuild=rebuild,
    )

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
        "d0_record": d0_binding["record_sha256"],
    }
    empirical = args.certification_mode == CERTIFICATION_EMPIRICAL
    if empirical:
        frozen = {
            "alpha": (args.alpha, FORMAL_ALPHA),
            "h_exec": (args.h_exec, FORMAL_H_EXEC),
            "top_k": (args.top_k, FORMAL_S0_TOP_K if args.s_only else FORMAL_SV_TOP_K),
        }
        drift = {k: v for k, v in frozen.items() if v[0] != v[1]}
        if drift:
            raise SystemExit(
                "formal empirical fit runs at frozen parameters; got "
                + ", ".join(f"{k}={got!r} (frozen {want!r})" for k, (got, want) in drift.items())
            )
    # Rev 1 development set: the whole verified cohort is exploratory data, so
    # there is no held-out calibration split left to certify with. The conformal
    # branch keeps the old fit/cal separation.
    dev_mask = (fit_mask | cal_mask) if empirical else fit_mask
    record: dict = {"s_only": args.s_only, "quantile_alpha": args.alpha,
                    "certification_mode": args.certification_mode,
                    "d0_record_sha256": _file_sha256(pathlib.Path(args.d0_record)),
                    "d0_suite": d0_record.get("suite"),
                    "d0_sample_rows_sha256": d0_record.get("sample", {}).get("rows_sha256"),
                    "d0_inputs_rollup_sha256": (d0_record.get("inputs") or {}).get("rollup_sha256"),
                    "cohort_manifest": str(args.cohort_manifest),
                    "input_digests": input_digests,
                    "d0_binding": d0_binding}
    folds: np.ndarray | None = None
    dev_eps = np.unique(table.episode[dev_mask])
    record["n_dev_episodes"] = len(dev_eps)
    if empirical:
        if len(dev_eps) != EXPECTED_DEV_EPISODES:
            _stop(record, out_dir, "dev_episode_count_mismatch", args.s_only)
        per_task = collections.Counter(
            int(table.task[table.episode == ep][0]) for ep in dev_eps
        )
        if sorted(per_task.values()) != [EXPECTED_DEV_EPISODES // EXPECTED_TASKS] * EXPECTED_TASKS:
            _stop(record, out_dir, "dev_episodes_not_evenly_task_stratified", args.s_only)
    ladder = GRID_LADDER_S_ONLY if args.s_only else GRID_LADDER_SV
    uses_v = not args.s_only

    # -- final binning grid (fit split, mechanical ladder) ----------
    grid = choose_grid(table.s[dev_mask], table.v[dev_mask], ladder)
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
        if empirical:
            suite = json.loads(pathlib.Path(args.split_manifest).read_text()).get("suite")
            want = FROZEN_DELTA_STAR.get(suite)
            if want is None:
                raise SystemExit(
                    f"split manifest suite {suite!r} has no frozen delta; the formal "
                    f"fit is defined only for {sorted(FROZEN_DELTA_STAR)}"
                )
            if abs(delta_star - want) > FROZEN_DELTA_TOL:
                raise SystemExit(
                    f"recomputed delta* {delta_star!r} differs from the frozen "
                    f"{suite} value {want!r} by more than {FROZEN_DELTA_TOL}; the "
                    "inputs are not the ones the protocol froze"
                )
            record["frozen_delta_star"] = want
            record["frozen_delta_suite"] = suite
        record["frozen_from"] = str(args.frozen_record)
        record["delta_selection_reason"] = "frozen_from_sv"
    else:
        folds = assign_folds(table, dev_mask)
        record["fold_sizes"] = [
            int(len(np.unique(table.episode[dev_mask & (folds == f)]))) for f in range(N_FOLDS)
        ]
        models = fit_fold_models(table, dev_mask, folds, ladder, args.alpha)
        if models is None:
            _stop(record, out_dir, "stop_loss_sparse_cells_oof", args.s_only)
        q_oof7, q_oof10 = oof_predictions(models, table, dev_mask)
        dev_idx = np.where(dev_mask)[0]
        oof_res = episode_max_residual(
            q_oof7, q_oof10, table.y7[dev_idx], table.y10[dev_idx], table.episode[dev_idx],
        )
        if len(oof_res) < 19:
            _stop(record, out_dir, "fit_episode_count_below_19", args.s_only)
        # Rev 1: the episode-max OOF offset is NOT applied. It was a second
        # extreme-value tax on top of the calibration correction, and either
        # layer alone empties the acceptance region on this deviation scale.
        oof_offset = 0.0 if empirical else order_statistic_offset(
            list(oof_res.values()), args.alpha
        )
        record["oof_safety_offset"] = oof_offset
        record["oof_episode_max_residual_p95"] = order_statistic_offset(
            list(oof_res.values()), args.alpha
        )
        record["n_fit_episodes"] = len(oof_res)

        grid_d = np.unique(np.percentile(table.y10[dev_mask], np.arange(10, 100, 10)))
        record["delta_grid"] = grid_d.tolist()
        if len(grid_d) < 2:
            _stop(record, out_dir, "degenerate_delta_grid", args.s_only)
        metrics = {
            float(d): evaluate_candidate_deployed(
                float(d), models, table, dev_mask, oof_offset, uses_disagreement=True,
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

        if empirical:
            suite = json.loads(pathlib.Path(args.split_manifest).read_text()).get("suite")
            want = FROZEN_DELTA_STAR.get(suite)
            if want is None:
                raise SystemExit(
                    f"split manifest suite {suite!r} has no frozen delta; the formal "
                    f"fit is defined only for {sorted(FROZEN_DELTA_STAR)}"
                )
            if abs(delta_star - want) > FROZEN_DELTA_TOL:
                raise SystemExit(
                    f"recomputed delta* {delta_star!r} differs from the frozen "
                    f"{suite} value {want!r} by more than {FROZEN_DELTA_TOL}; "
                    "refusing to label a changed fit as the frozen protocol"
                )
            record["frozen_delta_star"] = want
            record["frozen_delta_suite"] = suite

    # -- formal fit + split conformal ------------------------------
    ff = final_fit(table, dev_mask, alpha=args.alpha, edges=(s_edges, v_edges))
    sb_fit, vb_fit, q_hat = ff.sb, ff.vb, ff.q_hat
    if empirical:
        # No certification stage: delta was chosen by cross-fitted deployed
        # verdict and the boundaries are exported from q_hat as fitted. Nothing
        # here is a coverage statement, so nothing may be recorded as one.
        n_cal_episodes = 0
        c = 0.0
        q_tilde = q_hat
        record["n_calibration_episodes"] = 0
        record["conformal_c"] = 0.0
    else:
        sb_cal = bin_index(table.s[cal_mask], s_edges)
        vb_cal = bin_index(table.v[cal_mask], v_edges)
        cal_res = episode_max_residual(
            q_hat[0, sb_cal, vb_cal], q_hat[1, sb_cal, vb_cal],
            table.y7[cal_mask], table.y10[cal_mask], table.episode[cal_mask],
        )
        n_cal_episodes = len(cal_res)
        record["n_calibration_episodes"] = n_cal_episodes
        if n_cal_episodes < 19:
            _stop(record, out_dir, "calibration_episode_count_below_19", args.s_only)
        c = order_statistic_offset(list(cal_res.values()), args.alpha)
        record["conformal_c"] = c
        q_tilde = q_hat + c

    # -- diagnostics (SV run only; A1 on unconstrained bin quantiles)
    if not args.s_only:
        n_s, n_v = len(s_edges) - 1, len(v_edges) - 1
        emp_q = np.full((2, n_s, n_v), np.nan)
        for layer, y in ((0, table.y7[dev_mask]), (1, table.y10[dev_mask])):
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
        z = (table.y10[dev_mask] <= delta_star).astype(float)
        zb = bin_index(table.s[dev_mask], np.unique(np.quantile(
            table.s[dev_mask], np.linspace(0, 1, 11))))
        rates = [float(z[zb == b].mean()) for b in range(zb.max() + 1) if (zb == b).any()]
        record["mlr_rate_by_s_decile"] = rates
        record["mlr_monotone_violations"] = int(
            sum(rates[i + 1] < rates[i] for i in range(len(rates) - 1))
        )
        # Rev 1 records (A1) but does not gate on it. Section 4.1 lets ONLY
        # cell occupancy drive the ladder and section 4.2 freezes only the
        # accuracy/hitshare stops; the old A1 stop belonged to the certified
        # construction Rev 1 dropped. It is also strongly grid-dependent -- the
        # rate is computed from unconstrained per-cell 0.95 quantiles, and at
        # (12,6) a cell holds ~47 rows, where that statistic is essentially the
        # third largest value. Re-introducing it as a gate, or letting it drive
        # the ladder, would change the frozen fitter and the frozen delta.
        record["a1_gate_applied"] = not empirical
        if not empirical and record["a1_violation_rate"] is not None \
                and record["a1_violation_rate"] > 0.20:
            _stop(record, out_dir, "a1_violation_rate_above_20pct", args.s_only)

    # Freeze the full development/fold/refit audit BEFORE artifact emission so
    # the deployable object can carry the same immutable proof. S0 inherits the
    # SV fold map even though it does not repeat delta selection: that is the
    # only honest representation of a nested ablation on the same split.
    dev_rows = np.where(dev_mask)[0]
    membership, computed_fold_map = _development_audit(table, dev_mask, folds)
    if args.s_only:
        frozen_membership = frozen.get("dev_membership")
        frozen_fold_map = frozen.get("fold_map")
        if frozen_membership != [list(x) for x in membership]:
            raise SystemExit("S0 development membership differs from the frozen SV fit")
        if not isinstance(frozen_fold_map, list) or not frozen_fold_map:
            raise SystemExit("SV frozen record carries no auditable fold map")
        fold_map = [tuple(x) for x in frozen_fold_map]
    else:
        fold_map = computed_fold_map
    record["dev_membership"] = membership
    record["dev_membership_sha256"] = _digest_obj(membership)
    record["fold_map"] = fold_map
    record["fold_map_sha256"] = _digest_obj(fold_map)
    record["final_fit_digests"] = {
        "s_edges": _digest_obj(np.asarray(s_edges).tolist()),
        "v_edges": _digest_obj(np.asarray(v_edges).tolist()),
        "q_deploy": _digest_obj(np.asarray(q_tilde).tolist()),
        "n_dev_rows": int(len(dev_rows)),
    }
    if args.s_only:
        if record["dev_membership_sha256"] != frozen.get("dev_membership_sha256"):
            raise SystemExit("S0 development membership digest differs from SV")
        if record["fold_map_sha256"] != frozen.get("fold_map_sha256"):
            raise SystemExit("S0 inherited fold map digest differs from SV")

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
    # The artifact's effective retrieval width, not the yaml's configured one.
    # search_digest stays bound to the on-disk yaml (top_k=1) so the arms match
    # the table; this field is what SurfaceArtifact.validate() checks against k,
    # and it must therefore be the width the surface was actually fitted at.
    # S0 needs no extra candidates, so it declares k=1.
    artifact_k = args.top_k if uses_v else 1
    contract["top_k"] = artifact_k

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
            k=artifact_k,
            h_exec=args.h_exec,
            w=weights["w"],
            active_mask=weights["active_mask"],
            start_t_ws=PINNED_START_T_WS,
            delta=float(delta),
            quantile_alpha=args.alpha,
            certification_mode=args.certification_mode,
            uses_disagreement=uses_v,
            v_bin_edges=v_edges,
            s_min_full=full,
            s_min_warm=warm,
            conformal_c=c,
            n_calibration_episodes=n_cal_episodes,
            retrieval_contract=contract,
            meta={
                "ref_mode": "fresh",
                "delta_name": name,
                "delta_selection_reason": record["delta_selection_reason"],
                "oof_safety_offset": record.get("oof_safety_offset"),
                "certification_mode": args.certification_mode,
                "n_dev_episodes": record.get("n_dev_episodes"),
                "d0_record_sha256": record.get("d0_record_sha256"),
                "frozen_from": record.get("frozen_from"),
                "table": str(args.table),
                "input_digests": input_digests,
                "d0_binding": d0_binding,
                "dev_membership_sha256": record["dev_membership_sha256"],
                "fold_map_sha256": record["fold_map_sha256"],
                "final_fit_digests": record["final_fit_digests"],
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
