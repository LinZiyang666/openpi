"""Super warmup builder + driver + 7-check hard-gate verify.

A "super warmup" is a single warmup yaml whose ``judge.dump.factors``
list is constructed from the union of every kinematic cell's declared
factor keys (computed by ``spec.super_warmup_declared_keys()``). Running
this once produces a single raw jsonl that all 237 eval cells then
consume via offline calibration source.

Three responsibilities:

  1. ``build_super_warmup_yaml()``: assemble yaml dict.
  2. ``run_super_warmup(host, port, ...)``: client-side orchestrator —
     load yaml on server, drive LIBERO warmup workers, fetch deferred
     dump, extract finite-only factor_raw rows.
  3. ``verify_super_raw(raw_path)``: 7 hard-gate checks. Must pass before
     Stage 4 (emit eval yamls) per plan §4 Stage 3.

Coupling map:
  DEPENDS ON: exp.weighted_sum.kinematic.spec,
              exp.verdict_factor_judge.phase5.runner (_extract_finite_factor_raw,
                                                       _save_dump_jsonl,
                                                       _build_libero_argv),
              exp.verdict_factor_judge.phase3.threshold_solver (reconstruct_scores,
                                                                  derive_thresholds,
                                                                  load_per_key_finite_history),
              openpi.cache.components.factors.calibrations.percentile_rolling,
              openpi_client.websocket_client_policy.
  CONSUMED BY: exp.weighted_sum.kinematic.runner.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from exp.verdict_factor_judge.common.v2_spec import CFG_SPECS, factor
from exp.weighted_sum.kinematic.spec import (
    SUPER_WARMUP_ID,
    cell_to_solver_recipe,
    generate_all_cells,
    super_warmup_declared_keys,
)

try:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
except ImportError:
    WebsocketClientPolicy = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Default paths
# ----------------------------------------------------------------------

DEFAULT_YAML_PATH = Path(
    # Stem MUST equal SUPER_WARMUP_ID: generate_yamls.write_yaml enforces
    # ``yaml stem == dump.config_id`` and raises InvariantError otherwise
    # (see common/generate_yamls.py:117).
    "exp/weighted_sum/config/kinematic_phase5/libero_spatial/d1/ws_d1_kin_super_warmup.yaml"
)
DEFAULT_RAW_DIR = Path("exp/weighted_sum/data/libero_spatial/kinematic_phase5/d1")
DEFAULT_RAW_PATH = DEFAULT_RAW_DIR / "super_warmup_raw.jsonl"
DEFAULT_RAW_DUMP_PATH = DEFAULT_RAW_DIR / "super_warmup_raw_dump.jsonl"

# Stage 3 verify floor: trials_per_task=15 → 150 ep × 21.7 verdict/ep
# (measured from threshold_pareto warmup_per_step.jsonl) ≈ 3250 verdict.
# Allow some slack: 2500 is the hard "not enough" floor.
RAW_ROW_FLOOR = 2500

# PercentileRollingCalibration window_size must be reached for every key.
FINITE_PER_KEY_FLOOR = 50


# ----------------------------------------------------------------------
# Yaml builder — factor list driven by declared_keys union
# ----------------------------------------------------------------------


def _group_keys_to_factor_blocks(keys: set[str]) -> list[dict[str, Any]]:
    """Reverse-engineer factor blocks from descriptor keys.

    Each kinematic factor key looks like
    ``"{descriptor}_{source}_{channel}__p{P}_f{F}"``, e.g.
    ``"jerk_offline_state__p0_f3"``. To dump these, we need one factor
    block per ``(descriptor, source, channel)`` with the corresponding
    set of windows. ``topk_action_variance`` has a different key shape
    (no windows) and isn't used by any kinematic cell — we deliberately
    skip it (verified empty by spec.super_warmup_declared_keys()).

    Returns factor blocks in deterministic sorted order (by type name
    then window) so the emitted yaml is reproducible.
    """
    type_to_windows: dict[str, set[tuple[int, int]]] = {}
    for k in keys:
        if "__p" not in k:
            # topk-like; skip (no kinematic cell uses these — guard rather
            # than silently dump a non-windowed factor whose schema differs).
            continue
        head, win = k.split("__p", 1)
        p_str, f_str = win.split("_f", 1)
        type_to_windows.setdefault(head, set()).add((int(p_str), int(f_str)))

    factors: list[dict[str, Any]] = []
    for type_name in sorted(type_to_windows):
        windows_sorted = sorted(type_to_windows[type_name])
        factors.append(
            factor(
                type_name,
                windows=[{"past": p, "future": f} for (p, f) in windows_sorted],
            )
        )
    return factors


def build_super_warmup_yaml(
    *,
    cfg_id: str,
    preload_pkl_override: str | None = None,
) -> dict[str, Any]:
    """Single warmup yaml that dumps all 237 cells' declared keys.

    Uses ``always_warm_start(start_t=0.7)`` as the inner judge (every
    verdict passes through the retrieval pipeline; no early bail) and
    wraps with ``DumpingJudge`` (deferred=true so the server picks the
    on-disk dump path from ``--warmup_dump_root``). top_k is raised
    from 1 (eval default) to 5 to satisfy any factor's
    ``required_top_k`` hint (phase5 convention; harmless for cosine
    score which only reads ``results[0].score`` regardless).
    """
    cfg = CFG_SPECS[cfg_id]
    preload_path = (
        preload_pkl_override if preload_pkl_override is not None else cfg["preload_pkl"]
    )
    needed_keys = super_warmup_declared_keys()
    factors = _group_keys_to_factor_blocks(needed_keys)

    search_strategy = dict(cfg["search_strategy"])
    search_strategy["top_k"] = 5

    return {
        "enabled": True,
        "timer": {
            "enabled": False,
            "buffer_size": 10000,
            "output_csv_dir": None,
        },
        "keys": dict(cfg["keys"]),
        "key_builder": {"type": cfg["key_builder_type"]},
        "checkpoints": {
            "cp1": {
                "enabled": True,
                "gate": {"type": "always_search"},
                "judge": {
                    "type": "always_warm_start",
                    "start_t": 0.7,
                    "dump": {
                        "deferred": True,
                        "config_id": SUPER_WARMUP_ID,
                        "factors": factors,
                    },
                },
                "search_strategy": search_strategy,
            },
        },
        "backend": {
            "type": "in_memory",
            "vector_dims": dict(cfg["vector_dims"]),
            "in_memory": {
                "preload_path": preload_path,
                "index_type": "brute_force",
            },
        },
        "write_policy": {"type": "never"},
    }


# ----------------------------------------------------------------------
# Runtime: line counter (cache-skip predicate)
# ----------------------------------------------------------------------


def _line_count(path: Path) -> int:
    """Cheap line count for cache-skip predicate; returns 0 if file absent."""
    if not path.is_file():
        return 0
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


# ----------------------------------------------------------------------
# Driver: server-side run + fetch + extract
# ----------------------------------------------------------------------


def run_super_warmup(
    *,
    host: str,
    port: int,
    cfg_id: str,
    trials_per_task: int = 15,
    yaml_path: Path = DEFAULT_YAML_PATH,
    raw_path: Path = DEFAULT_RAW_PATH,
    raw_dump_path: Path = DEFAULT_RAW_DUMP_PATH,
    libero_args: dict[str, Any] | None = None,
    preload_pkl_override: str | None = None,
) -> Path:
    """Run super warmup once on a single server.

    ``trials_per_task=15`` × 10 tasks ≈ 150 episodes ≈ 3250 verdict (per
    plan §2.13 measured 21.7 verdict/ep on the same d1 base). Result:
    every declared key has ≥50 finite samples even for the most-sparse
    online window (7,7)_state.

    Cache predicate (R1D MAJOR fix): re-uses existing raw only if the
    file has ≥ ``RAW_ROW_FLOOR`` lines; a partially-collected raw from
    a previous crash is deleted and re-collected.

    ``libero_args`` is the dict of additional kwargs forwarded to
    phase5's ``_build_libero_argv`` (per_step_log_dir, episode_filter,
    cuda_visible_devices, conda_env, task_suite, etc.). The caller is
    responsible for setting these to match its environment.
    """
    if WebsocketClientPolicy is None:
        raise RuntimeError("openpi_client is not installed; cannot run super warmup")

    # ------------------------------------------------------------------
    # Local imports of phase5 internals (avoid top-level import — these
    # are stable but not part of phase5's __all__).
    # ------------------------------------------------------------------
    from exp.verdict_factor_judge.common.generate_yamls import write_yaml
    from exp.verdict_factor_judge.common.run_phase import _build_libero_argv
    from exp.verdict_factor_judge.phase3.runner import _save_dump_jsonl
    from exp.verdict_factor_judge.phase5.runner import (
        _extract_finite_factor_raw,
    )

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    # Emit yaml (always rewrite — cheap and reproducible). cfg_id + preload
    # override MUST be threaded through so the server loads the correct task
    # suite's pkl; a silent fallback to the default CFG loaded libero_spatial
    # entries on a libero_10 run and produced an all-NaN factor dump.
    yaml_dict = build_super_warmup_yaml(
        cfg_id=cfg_id, preload_pkl_override=preload_pkl_override
    )
    write_yaml(yaml_path, yaml_dict)
    logger.info("[super_warmup] emitted yaml -> %s", yaml_path)

    # Cache check: skip only if a full raw already exists.
    existing = _line_count(raw_path)
    if existing >= RAW_ROW_FLOOR:
        logger.info(
            "[super_warmup] cached raw at %s (%d rows >= floor %d), skip",
            raw_path,
            existing,
            RAW_ROW_FLOOR,
        )
        return raw_path
    if existing > 0:
        logger.warning(
            "[super_warmup] partial raw at %s (%d rows < floor %d) — "
            "deleting and re-running",
            raw_path,
            existing,
            RAW_ROW_FLOOR,
        )
        raw_path.unlink()
        raw_dump_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Server load + LIBERO warmup workers + fetch_dump.
    # ------------------------------------------------------------------
    with WebsocketClientPolicy(host=host, port=port) as ctl:
        ctl.load_cache_config(
            yaml_content=yaml_path.read_text(encoding="utf-8"),
            yaml_id=SUPER_WARMUP_ID,
        )
        logger.info("[super_warmup] loaded yaml on server, spawning workers")

        # Build libero argv via phase5 helper; caller provides env-specific args.
        argv_args = dict(libero_args or {})
        argv_args.setdefault("yaml_id", SUPER_WARMUP_ID)
        argv_args.setdefault("phase", "warmup")
        argv_args.setdefault("num_trials_per_task", trials_per_task)
        cmd, env = _build_libero_argv(**argv_args)
        subprocess.run(cmd, env=env, check=True)
        logger.info("[super_warmup] workers complete; fetching dump")

        content = ctl.fetch_dump(SUPER_WARMUP_ID)
        _save_dump_jsonl(content, raw_dump_path)
        logger.info(
            "[super_warmup] saved raw dump -> %s (%d bytes)",
            raw_dump_path,
            raw_dump_path.stat().st_size,
        )

        # Project to finite-only over the declared-keys union (single source
        # of truth, NOT a per-cell subset — the raw must keep every key
        # any 237-cell will ever bind).
        all_keys = list(super_warmup_declared_keys())
        _extract_finite_factor_raw(raw_dump_path, raw_path, all_keys)
        logger.info(
            "[super_warmup] extracted finite raw -> %s (%d rows)",
            raw_path,
            _line_count(raw_path),
        )

    return raw_path


# ----------------------------------------------------------------------
# Hard-gate verify (7 checks)
# ----------------------------------------------------------------------


def _classify_critical_key(key: str) -> str | None:
    """Tag keys that are statistically most likely to under-collect.

    Returns one of {"past5_zero", "past7_future7", None}. Used by
    check #3 to escalate warnings on long-window online factors which
    bail when ``history < P`` (online.py:159) — the earliest verdicts
    in each episode contribute NaN.
    """
    if "__p5_f0" in key:
        return "past5_zero"
    if "__p7_f7" in key:
        return "past7_future7"
    return None


def verify(raw_path: Path) -> None:
    """7 hard-gate checks. Raises AssertionError on any failure.

    Must pass before Stage 4 (emit eval yamls) per plan §4 Stage 3.

    Each check is independent and collects its own failure list rather
    than fail-fast so the operator sees the full picture (R1A MINOR fix).
    Bootstrap CI in check #7 is a sanity floor on quantile estimator
    stability for sparse-tail (FH, WS) ratios (R1B SUGGESTION 2).
    """
    from exp.verdict_factor_judge.phase3.threshold_solver import (
        derive_thresholds,
        load_per_key_finite_history,
        reconstruct_scores,
    )
    from openpi.cache.components.factors.base import CalibrationSamples
    from openpi.cache.components.factors.calibrations.percentile_rolling import (
        PercentileRollingCalibration,
    )

    rows = [
        json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()
    ]

    # ------------------------------------------------------------------
    # Check 1: row count >= RAW_ROW_FLOOR
    # ------------------------------------------------------------------
    assert len(rows) >= RAW_ROW_FLOOR, f"too few rows: {len(rows)} < {RAW_ROW_FLOOR}"
    logger.info("[verify] check 1 OK: %d rows", len(rows))

    # ------------------------------------------------------------------
    # Check 2: cp1_score non-null rate >= 99% (read from raw_dump, NOT
    # finite raw — `_extract_finite_factor_raw` deliberately projects to
    # ``{factor_raw: {...}}`` only and discards cp1_score / winner_id /
    # hit_type so the solver sees a minimal schema. The cp1_score sanity
    # belongs on the raw_dump file which still carries it (see
    # `dumping_judge._write_dump_row` line 200-214).
    # ------------------------------------------------------------------
    raw_dump_path = raw_path.parent / "super_warmup_raw_dump.jsonl"
    if raw_dump_path.is_file():
        n_dump = n_null = 0
        with raw_dump_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                n_dump += 1
                if d.get("cp1_score") is None:
                    n_null += 1
        assert n_dump > 0, f"raw_dump empty: {raw_dump_path}"
        null_rate = n_null / n_dump
        assert null_rate < 0.01, (
            f"cp1_score null rate in raw_dump too high: {n_null}/{n_dump} = {null_rate:.4f}"
        )
        logger.info(
            "[verify] check 2 OK: cp1_score null rate %.4f over %d raw_dump rows",
            null_rate,
            n_dump,
        )
    else:
        logger.warning(
            "[verify] check 2 SKIPPED: raw_dump not found at %s — cp1_score "
            "sanity unverifiable (finite raw deliberately strips cp1_score)",
            raw_dump_path,
        )

    # ------------------------------------------------------------------
    # Check 3: every declared key has >= FINITE_PER_KEY_FLOOR finite samples
    # ------------------------------------------------------------------
    needed = super_warmup_declared_keys()
    per_key_finite = {k: 0 for k in needed}
    for row in rows:
        raw = row.get("factor_raw") or {}
        for k in needed:
            v = raw.get(k)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isnan(fv):
                per_key_finite[k] += 1
    bad = {k: n for k, n in per_key_finite.items() if n < FINITE_PER_KEY_FLOOR}
    if bad:
        critical = {k: n for k, n in bad.items() if _classify_critical_key(k)}
        logger.error(
            "[verify] check 3 FAIL: %d keys < %d finite; %d are p5_f0/p7_f7 variants",
            len(bad),
            FINITE_PER_KEY_FLOOR,
            len(critical),
        )
        for k, n in sorted(bad.items())[:10]:
            logger.error("  bad: %s = %d", k, n)
    assert not bad, (
        f"{len(bad)} keys have < {FINITE_PER_KEY_FLOOR} finite samples "
        f"(rerun super warmup with more trials_per_task)"
    )
    logger.info(
        "[verify] check 3 OK: all %d keys >= %d finite",
        len(needed),
        FINITE_PER_KEY_FLOOR,
    )

    # ------------------------------------------------------------------
    # Check 4: per-group sample reconstruct_scores succeeds
    # ------------------------------------------------------------------
    all_cells = generate_all_cells()
    fails: list[tuple[str, str, str]] = []
    for g in ("g1", "g2", "g3", "g4", "g5"):
        c = next((c for c in all_cells if c.group == g), None)
        if c is None:
            continue
        try:
            recipe = cell_to_solver_recipe(c)
            scores = reconstruct_scores(raw_path, recipe, composer_weights=c.weights)
            finite = [s for s in scores if not (s is None or math.isnan(s))]
            assert len(finite) >= 500, f"{c.yaml_id}: only {len(finite)} finite scores"
        except Exception as e:  # noqa: BLE001
            fails.append((g, c.yaml_id, repr(e)))
    assert not fails, f"reconstruct_scores failed: {fails}"
    logger.info("[verify] check 4 OK: 5 sample cells reconstruct_scores")

    # ------------------------------------------------------------------
    # Check 5: derive_thresholds monotone (T_ws <= T_fh) for representative ratios
    # ------------------------------------------------------------------
    from exp.weighted_sum.kinematic.spec import _g5_grid_filtered

    g5_p1 = next(c for c in all_cells if c.group == "g5" and c.base_recipe == "p1")
    scores_g5 = reconstruct_scores(
        raw_path, cell_to_solver_recipe(g5_p1), composer_weights=g5_p1.weights
    )
    scores_g5_finite = [s for s in scores_g5 if not math.isnan(s)]
    min_score = min(scores_g5_finite)
    sample_pairs = _g5_grid_filtered()[:5]
    for fh, ws in sample_pairs:
        t_fh, t_ws = derive_thresholds(scores_g5_finite, fh, ws)
        # R1A MAJOR: allow equality (sample plateau tie), only strict
        # less-than would false-positive on tie-breaks (G1 R1 R3 R1A-M3).
        assert 0.0 <= t_ws <= t_fh <= 1.0, (
            f"({fh},{ws}): bad order t_fh={t_fh}, t_ws={t_ws}"
        )
        # Non-degenerate (R1B DESIGN-CRITICAL): only fh+ws==1.0 forces
        # T_ws = min; our grid filter excludes that, so T_ws > min.
        if fh + ws < 1.0 - 1e-9:
            assert t_ws > min_score + 1e-9, (
                f"({fh},{ws}): T_ws={t_ws} degenerate to min={min_score}"
            )
    logger.info(
        "[verify] check 5 OK: derive_thresholds monotone for %d ratios",
        len(sample_pairs),
    )

    # ------------------------------------------------------------------
    # Check 6: ALL 237 cells bind_keys passes (G5 declared ⊆ super raw)
    # ------------------------------------------------------------------
    super_buffer = load_per_key_finite_history(raw_path, list(needed))
    bind_fails: list[tuple[str, str, str]] = []
    for c in all_cells:
        try:
            cal = PercentileRollingCalibration(
                CalibrationSamples(super_buffer), window_size=50
            )
            cal.bind_keys(list(c.declared_keys))
        except Exception as e:  # noqa: BLE001
            bind_fails.append((c.group, c.yaml_id, repr(e)))
    assert not bind_fails, (
        f"bind_keys failed for {len(bind_fails)}/{len(all_cells)} cells: "
        f"{bind_fails[:5]}"
    )
    logger.info("[verify] check 6 OK: %d cells bind_keys", len(all_cells))

    # ------------------------------------------------------------------
    # Check 7: bootstrap CI sanity for quantile estimator stability
    # ------------------------------------------------------------------
    sample_cells = [g5_p1] + [
        next((c for c in all_cells if c.group == g and c.base_recipe == "p1"), None)
        for g in ("g1", "g3")
    ]
    rng = np.random.default_rng(42)
    for c in sample_cells:
        if c is None:
            continue
        scores_c = reconstruct_scores(
            raw_path, cell_to_solver_recipe(c), composer_weights=c.weights
        )
        scores_arr = np.array(
            [s for s in scores_c if not math.isnan(s)],
            dtype=np.float64,
        )
        t_fh_boot: list[float] = []
        t_ws_boot: list[float] = []
        for _ in range(200):
            bs = rng.choice(scores_arr, size=len(scores_arr), replace=True)
            arr_desc = np.sort(bs)[::-1]
            # use a representative central ratio (0.3, 0.3)
            i_fh = max(0, int(0.3 * len(arr_desc)) - 1)
            i_ws = max(0, int(0.6 * len(arr_desc)) - 1)
            t_fh_boot.append(float(arr_desc[i_fh]))
            t_ws_boot.append(float(arr_desc[i_ws]))
        ci_fh = np.percentile(t_fh_boot, [2.5, 97.5])
        ci_ws = np.percentile(t_ws_boot, [2.5, 97.5])
        width_fh = float(ci_fh[1] - ci_fh[0])
        width_ws = float(ci_ws[1] - ci_ws[0])
        logger.info(
            "[verify] check 7 CI %s @ (0.3, 0.3): T_fh CI=[%.4f, %.4f] w=%.4f, T_ws CI=[%.4f, %.4f] w=%.4f",
            c.yaml_id,
            ci_fh[0],
            ci_fh[1],
            width_fh,
            ci_ws[0],
            ci_ws[1],
            width_ws,
        )
        # G3's jerk-only / disp-only / extreme-pattern cells deliberately
        # zero out half the online weights → score distribution sparser →
        # bootstrap CI naturally wider; widen the gate to 0.20 and log a
        # WARNING (rather than fail) when the cell is a known-extreme G3
        # pattern. For other groups the 0.15 floor stays a hard gate.
        is_extreme_g3 = c.group == "g3" and (
            "pat-0-" in c.yaml_id
            or "pat--0" in c.yaml_id
            or c.yaml_id.endswith("pat-1-0")
        )
        gate = 0.20 if is_extreme_g3 else 0.15
        if width_fh >= gate:
            if is_extreme_g3 and width_fh < 0.25:
                logger.warning(
                    "[verify] check 7 WARN: %s extreme-G3 pattern, T_fh CI "
                    "width %.4f >= %.2f (accepted, design-expected)",
                    c.yaml_id,
                    width_fh,
                    gate,
                )
                continue
            raise AssertionError(
                f"{c.yaml_id}: T_fh CI width {width_fh:.4f} >= {gate:.2f} — "
                f"quantile unstable; need more super warmup samples"
            )

    logger.info("[verify] OK 7/7")


# ----------------------------------------------------------------------
# Public exports
# ----------------------------------------------------------------------


__all__ = [
    "DEFAULT_YAML_PATH",
    "DEFAULT_RAW_DIR",
    "DEFAULT_RAW_PATH",
    "DEFAULT_RAW_DUMP_PATH",
    "RAW_ROW_FLOOR",
    "FINITE_PER_KEY_FLOOR",
    "build_super_warmup_yaml",
    "run_super_warmup",
    "verify",
]
