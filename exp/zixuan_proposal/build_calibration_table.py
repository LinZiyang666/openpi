"""Pass 2 offline replay: per-episode h5 -> threshold-independent signal table (Phase 5).

TRACER Phase 5 calibration pipeline, stage 2 (see
``logs/tracer_phase5_calibration.log.md`` §1). For every calibration episode
(one per-step embedding HDF5 from a Pass 1 ``serve_policy --collect`` rollout on
the ``I_cal`` split) this script faithfully reconstructs the ONLINE chain-aware
failure-aware signals — ``s_pos`` / ``s_neg`` / ``margin`` / ``delta_pos`` and the
Phase 5 kinematic ``u_t`` — by feeding the episode's steps, in trajectory order
and WITHOUT resetting the strategy's history, through the production
``DualRetrievalKnnStrategy`` + ``FailureAwareGateJudge``. The timing mirrors
``CacheOrchestrator.check()`` exactly (state appended before the search; the
executed action appended after, so the current step's action only enters the
NEXT step's window).

Leakage control (plan §1, LOEO — leave-one-episode-out): the calibration
episodes are the same rollouts that built the D+/D- libraries, so a naive replay
would let a query self-match its own stored entries. For each episode we rebuild
a temporary read-only search pool that EXCLUDES that episode's own entries
(identity = composite ``(task_key, trajectory_id)`` — the builder uses the h5
stem as ``trajectory_id`` and stems collide across task dirs, so we filter by
BOTH), while PRESERVING the merged artifact's D+-only ``library_stats`` on the
temp pool (recomputing over D+∪D- would fold D- into the u_t normalizer and
break Phase 4's contamination guard). A fail-loud assert guarantees zero
self-entries survive in the pool.

The gate (``FailureAwareGateJudge``) is reused across episodes from
``build_cache_components`` on the FULL merged artifact — it only needs the
D+-only ``library_stats`` (unchanged by the LOEO exclusion) to build its u_t
normalizer. When the config's judge carries no ``u_t_factor`` (b2 == 0), the
gate's ``_compute_u_t`` returns None and every row's ``u_t`` is None; that is the
correct margin-only behaviour and the solver drops the term.

Output: one JSONL row per step, ready for
``solve_calibration.py`` / ``MarginGateCalibrator.calibrate``.

Usage:
  PYTHONPATH=. uv run python exp/zixuan_proposal/build_calibration_table.py \
      --config       exp/zixuan_proposal/config/dual_retrieval_active.yaml \
      --merged-pkl   exp/common/data/cache_artifacts/libero_spatial/cp1_mean_pool_dual.pkl \
      --h5-dir       exp/zixuan_proposal/data/pass1_collect/libero_spatial \
      --episodes-json exp/zixuan_proposal/data/pass1_episode_results_spatial.json \
      --suite        libero_spatial \
      --out-jsonl    exp/zixuan_proposal/data/phase5_calib_rows_libero_spatial.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Injecting the repo root mirrors how the codebase runs exp/ scripts
# (PYTHONPATH=. uv run ...) and lets the exp.common sibling import resolve.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openpi.cache.backends.in_memory_backend import InMemoryBackend  # noqa: E402
from openpi.cache.cache_storage import CacheStorage  # noqa: E402
from openpi.cache.components.factors.base import HistoryView  # noqa: E402
from openpi.cache.components.margin_gate_calibrator import _WARM_START_T  # noqa: E402
from openpi.cache.components.payload_view import StoragePayloadView  # noqa: E402
from openpi.cache.components.search_strategy import SearchContext  # noqa: E402
from openpi.cache.config import (  # noqa: E402
    _build_search_strategy,
    build_cache_components,
    load_cache_config,
    validate_cache_config,
)
from openpi.cache.types import CheckpointID  # noqa: E402

# Production KeyBuilder chain (h5 step group -> query_keys). Reuses the SAME
# factory the merged artifact was built with so replay keys land in the exact
# vector space of the stored entries.
from exp.common.build_in_memory_cache_artifact import (  # noqa: E402
    _build_fake_stage1,
    _create_builder,
)


# ---------------------------------------------------------------------------
# Identity + id helpers
# ---------------------------------------------------------------------------


# ``trajectory_id`` / h5 stem looks like ``episode_<global-id>_<YYYYMMDD>_...``;
# the ``<global-id>`` is init-stable across re-collections while the timestamp is
# not, so the LOEO identity keys on the global id, NOT the full stem (G2 R1
# finding 2 — a Phase 5 re-collect of the same init gets a new timestamp/stem).
_EPISODE_ID_RE = re.compile(r"episode_0*(\d+)")


def _parse_global_episode_id(name: Optional[str]) -> Optional[int]:
    """Extract the timestamp-independent global episode id from an
    ``episode_<id>[_<timestamp>...]`` trajectory_id / stem (None if absent)."""
    if not name:
        return None
    m = _EPISODE_ID_RE.search(str(name))
    return int(m.group(1)) if m else None


def _entry_identity(entry) -> tuple[str, Optional[int]]:
    """Timestamp-independent LOEO identity of a stored entry:
    ``(task_key, global_episode_id)``.

    ``global_episode_id`` is parsed from the entry's ``trajectory_id`` prefix so a
    Phase-5 re-collection of the same init (new timestamp -> new stem but the SAME
    global id) still matches and is excluded. ``task_key`` disambiguates the same
    global id reused across task dirs.
    """
    task_key = getattr(entry.payload, "task_key", "") or ""
    return (task_key, _parse_global_episode_id(getattr(entry, "trajectory_id", None)))


def _episode_identity(h5_file: h5py.File, h5_path: Path, ids: dict) -> tuple[str, int]:
    """Timestamp-independent LOEO identity of the calibration episode:
    ``(task attr, global_episode_id)``.

    The global id comes from the h5 ``episode_id`` attr (preferred) else the
    ``episode_<id>_...`` stem. Fail-loud when unresolvable: LOEO correctness
    depends on a stable id — silently falling back to the timestamped stem would
    re-open the leak (G2 R1 finding 2).
    """
    task_key = str(h5_file.attrs.get("task", ""))
    gid = ids.get("episode_id")
    if gid is None:
        gid = _parse_global_episode_id(h5_path.stem)
    if gid is None:
        raise SystemExit(
            f"cannot resolve a stable global episode id for {h5_path.name} "
            "(need h5 attr 'episode_id' or an 'episode_<id>_...' stem); LOEO "
            "exclusion requires a timestamp-independent identity"
        )
    return (task_key, gid)


def _derive_episode_ids(h5_file: h5py.File, h5_path: Path) -> dict[str, Optional[int]]:
    """Derive ``(task_id, init_state_idx, episode_id)`` for the success join.

    Prefers explicit HDF5 attrs (the ``examples/libero/main.py`` collect schema
    writes ``task_id`` / ``init_state_idx`` / ``episode_id``); falls back to the
    ``task_<T>/episode_<S>.h5`` path layout; last resort derives task/init from
    ``episode_id`` via the 50-init-per-task convention (episode_id = task_id*50 +
    init_state_idx), matching the provenance id convention (plan §1).
    """
    def _attr_int(name: str) -> Optional[int]:
        if name in h5_file.attrs:
            try:
                return int(h5_file.attrs[name])
            except (TypeError, ValueError):
                return None
        return None

    task_id = _attr_int("task_id")
    init_state_idx = _attr_int("init_state_idx")
    episode_id = _attr_int("episode_id")

    # Path fallback: ``.../task_<T>/episode_<S>.h5``.
    if task_id is None:
        parent = h5_path.parent.name
        if parent.startswith("task_") and parent[len("task_"):].isdigit():
            task_id = int(parent[len("task_"):])
    if init_state_idx is None:
        stem = h5_path.stem
        if stem.startswith("episode_") and stem[len("episode_"):].isdigit():
            init_state_idx = int(stem[len("episode_"):])

    # Last-resort decomposition of a global episode_id (50 init states/task).
    if episode_id is not None:
        if task_id is None:
            task_id = episode_id // 50
        if init_state_idx is None:
            init_state_idx = episode_id % 50

    return {"task_id": task_id, "init_state_idx": init_state_idx, "episode_id": episode_id}


def _load_episode_success_map(path: str) -> dict:
    """Read ``main.py --save-episode-results`` JSON into keyed success lookups.

    Returns a dict with two keys: ``by_full`` keyed by
    ``(task_id, init_state_idx, episode_id)`` and ``by_task_init`` keyed by
    ``(task_id, init_state_idx)`` (fallback when the h5 lacks a global
    ``episode_id``). success stays ``bool | None`` (None = incomplete episode).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("episodes", [])
    by_full: dict[tuple, Optional[bool]] = {}
    by_task_init: dict[tuple, Optional[bool]] = {}
    for r in rows:
        t = r.get("task_id")
        s = r.get("init_state_idx")
        e = r.get("episode_id")
        succ = r.get("success")
        succ_b = None if succ is None else bool(succ)
        if t is not None and s is not None:
            by_task_init[(int(t), int(s))] = succ_b
            if e is not None:
                by_full[(int(t), int(s), int(e))] = succ_b
    return {"by_full": by_full, "by_task_init": by_task_init}


def _lookup_success(success_map: dict, ids: dict[str, Optional[int]]) -> Optional[bool]:
    """Join episode success by ``(task_id, init_state_idx, episode_id)``, then
    fall back to ``(task_id, init_state_idx)``. None when unresolved."""
    t, s, e = ids["task_id"], ids["init_state_idx"], ids["episode_id"]
    if t is not None and s is not None and e is not None:
        hit = success_map["by_full"].get((t, s, e))
        if hit is not None:
            return hit
    if t is not None and s is not None:
        return success_map["by_task_init"].get((t, s))
    return None


# ---------------------------------------------------------------------------
# Entry tensor materialization (numpy artifact -> torch, mirrors load_artifact)
# ---------------------------------------------------------------------------


def _entries_to_torch(entries: list) -> None:
    """Convert every entry's numpy query_keys / payload tensors to torch in place.

    Mirrors ``InMemoryBackend.load_artifact`` (numpy -> torch float32 + trajectory
    / outcome field backfill) so a temp backend populated by direct ``insert``
    matches the production load path byte-for-byte. Idempotent (already-torch
    tensors pass through untouched), so it is safe to run once and share the
    entries across every per-episode temp pool.
    """
    for e in entries:
        if not hasattr(e, "prev_ids"):
            e.prev_ids = []
        if not hasattr(e, "next_ids"):
            e.next_ids = []
        if not hasattr(e, "trajectory_id"):
            e.trajectory_id = None
        if not hasattr(e, "outcome"):
            e.outcome = None
        if e.query_keys:
            e.query_keys = {
                k: (torch.from_numpy(v).float() if isinstance(v, np.ndarray) else v)
                for k, v in e.query_keys.items()
            }
        p = e.payload
        if p.action_chunk is not None and isinstance(p.action_chunk, np.ndarray):
            p.action_chunk = torch.from_numpy(p.action_chunk).float()
        if p.intermediates:
            p.intermediates = {
                k: (torch.from_numpy(v).float() if isinstance(v, np.ndarray) else v)
                for k, v in p.intermediates.items()
            }


# ---------------------------------------------------------------------------
# LOEO temp storage
# ---------------------------------------------------------------------------


def _assert_no_self_entries(entries: list, identity: tuple) -> None:
    """Fail-loud: raise if ANY entry carries the current episode's identity.

    Run over the POST-exclusion pool this is the LOEO leakage guard (plan §1):
    a surviving self-entry means the query could top-1-match its own stored step.
    """
    leaked = [e for e in entries if _entry_identity(e) == identity]
    if leaked:
        raise AssertionError(
            f"LOEO leakage: {len(leaked)} entr(y/ies) with identity {identity!r} "
            f"survived exclusion (first id={leaked[0].id!r}); the query would "
            "self-match. Check the (task_key, trajectory_id) identity derivation."
        )


def _temp_storage_excluding(
    all_entries: list,
    *,
    vector_dims: dict[str, int],
    library_stats,
    identity: tuple,
) -> CacheStorage:
    """Build a read-only ``CacheStorage`` whose pool excludes ``identity``'s entries.

    The merged artifact's D+-only ``library_stats`` is preserved verbatim on the
    temp backend (NOT recomputed from the filtered D+∪D- pool — that would fold
    D- into the u_t normalizer). Fail-loud asserts zero self-entries survive.
    """
    keep = [e for e in all_entries if _entry_identity(e) != identity]
    _assert_no_self_entries(keep, identity)
    backend = InMemoryBackend(vector_dims)
    for e in keep:
        # backend.insert is the public write path; no session is active so the
        # mutation guard is a no-op. Entries are already torch (see
        # _entries_to_torch) and read-only during search.
        backend.insert(e)
    # Preserve D+-only stats (public attribute read by the CacheStorage facade).
    backend.library_stats = library_stats
    return CacheStorage(backend)


# ---------------------------------------------------------------------------
# Fusion-weight helper (mirrors config.build_per_connection_components)
# ---------------------------------------------------------------------------


def _fusion_weights_from_config(config) -> dict[str, float]:
    """Per-field fusion weights for enabled keys (same as the production build)."""
    keys = config.keys
    pairs = [
        ("vision_0", keys.vision_0),
        ("vision_1", keys.vision_1),
        ("vision_2", keys.vision_2),
        ("prompt_emb", keys.prompt_emb),
        ("robot_state", keys.robot_state),
    ]
    return {name: kf.weight for name, kf in pairs if kf.enabled}


# ---------------------------------------------------------------------------
# Winner-payload inspection
# ---------------------------------------------------------------------------


def _winner_has_warm_snapshot(view: StoragePayloadView, winner_id: str) -> bool:
    """True iff the winner d+ carries a warm-start intermediate at start_t=0.5.

    The offline artifact stores all 9 denoise intermediates keyed by float
    timestep; a WARM_START at ``_WARM_START_T`` needs that key present. Absent
    intermediates (success-only artifacts built without noise snapshots) -> False.
    """
    try:
        payload = view.get(winner_id)
    except Exception:  # noqa: BLE001 - missing winner payload => no warm snapshot
        return False
    inter = getattr(payload, "intermediates", None)
    if not inter:
        return False
    return any(abs(float(k) - _WARM_START_T) < 1e-6 for k in inter.keys())


# ---------------------------------------------------------------------------
# Per-episode replay
# ---------------------------------------------------------------------------


def _step_sort_key(name: str) -> tuple[bool, int, str]:
    """Sort ``step_<NNNN>`` groups by numeric suffix (mirrors the builder)."""
    suffix = name.split("_", 1)[1] if "_" in name else ""
    if suffix.isdigit():
        return (False, int(suffix), name)
    return (True, 0, name)


def build_rows_for_episode(
    h5_path: Path,
    *,
    builder,
    all_entries: list,
    vector_dims: dict[str, int],
    library_stats,
    strategy_cfg,
    fusion_weights: dict[str, float],
    gate,
    suite: str,
    episode_success: Optional[bool],
    ids: dict[str, Optional[int]],
    min_top_k_hint: int = 0,
) -> list[dict]:
    """Replay one calibration episode -> list of per-step signal rows.

    Reconstructs the online chain-aware signals faithfully:
      * a LOEO temp pool excludes THIS episode's own entries;
      * a fresh production ``DualRetrievalKnnStrategy`` is rebuilt over that pool
        (history NOT reset between steps -> TrajectoryMixin accumulates depth-5
        chain context);
      * per step, state is appended to the history BEFORE the search and the
        executed action AFTER (mirrors ``orchestrator.check`` / ``broadcast_action``);
      * the reused gate computes u_t over ``[history[-P:], winner, walk_next(F)]``.
    """
    rows: list[dict] = []
    with h5py.File(h5_path, "r") as f:
        identity = _episode_identity(f, h5_path, ids)
        task_key = identity[0]

        temp_storage = _temp_storage_excluding(
            all_entries, vector_dims=vector_dims,
            library_stats=library_stats, identity=identity,
        )
        strat = _build_search_strategy(
            strategy_cfg, temp_storage, fusion_weights, min_top_k_hint=min_top_k_hint,
        )

        # Episode-start lifecycle: mint the strategy search session and register
        # it with the backend (exactly what the orchestrator does before search).
        strat.on_episode_start()
        sid = strat.get_search_session_id() if hasattr(strat, "get_search_session_id") else None
        if sid is not None:
            temp_storage.open_search_session(sid)
        if hasattr(builder, "on_episode_start"):
            builder.on_episode_start()

        # Newest-last history buffers (start empty; NOT reset across steps).
        state_hist: list[torch.Tensor] = []
        action_hist: list[torch.Tensor] = []

        step_names = sorted(
            (k for k in f.keys() if k.startswith("step_")), key=_step_sort_key,
        )
        try:
            for step_name in step_names:
                group = f[step_name]

                # (1) Production KeyBuilder chain: h5 step group -> query_keys.
                builder.collect(CheckpointID.CP1, stage1=_build_fake_stage1(group))
                query_keys = builder.build(CheckpointID.CP1)
                builder.clear()

                suffix = step_name.split("_", 1)[1] if "_" in step_name else ""
                step_idx = int(suffix) if suffix.isdigit() else len(rows)

                # (2) BEFORE search: append this step's robot_state to state
                # history (mirrors orchestrator appending at check() head, anchor
                # = CP1). Guard the field like the orchestrator does.
                if "robot_state" in query_keys:
                    state_hist.append(query_keys["robot_state"].detach().cpu())

                # (3) Chain-aware search over the LOEO pool.
                ctx = SearchContext(
                    query_keys=query_keys,
                    checkpoint_id=CheckpointID.CP1,
                    current_step=step_idx,
                    task_key=task_key or None,
                )
                results = strat.search(ctx)

                # (4) Failure-aware retrieval signals for this query.
                sig = strat.last_retrieval_signals()

                # (5) u_t (kinematic quality of winner d+); None when the gate has
                # no u_t_factor or the verdict context is insufficient. Build the
                # view/history exactly as the orchestrator does.
                view = StoragePayloadView(temp_storage)
                history = HistoryView(actions=list(action_hist), states=list(state_hist))
                u_t = gate._compute_u_t(results, view, history) if results else None

                # (6) Winner + warm-snapshot availability.
                winner_id = results[0].id if results else None
                warm = (
                    _winner_has_warm_snapshot(view, winner_id)
                    if winner_id is not None else False
                )

                # (7) AFTER search: append this step's executed action (mirrors
                # broadcast_action appending action_chunk[0] once check returns,
                # so it only enters the NEXT step's window).
                action = torch.from_numpy(np.array(group["clean_action"])).float()
                if action.dim() == 1:
                    action = action.unsqueeze(0)
                action_hist.append(action[0].detach().cpu())

                # (8) Emit one row. NaN / None u_t serialize to None.
                u_t_out = (
                    None
                    if (u_t is None or (isinstance(u_t, float) and math.isnan(u_t)))
                    else float(u_t)
                )
                rows.append({
                    "suite": suite,
                    "task_id": ids["task_id"],
                    "init_state_idx": ids["init_state_idx"],
                    "episode_id": ids["episode_id"],
                    "step_idx": step_idx,
                    "s_pos": float(sig.s_pos) if sig is not None else 0.0,
                    "s_neg": float(sig.s_neg) if sig is not None else 0.0,
                    "delta_pos": float(sig.delta_pos) if sig is not None else 0.0,
                    "u_t": u_t_out,
                    "winner_id": winner_id,
                    "winner_has_warm_snapshot": bool(warm),
                    "episode_success": bool(episode_success),
                })
        finally:
            if sid is not None:
                temp_storage.close_search_session(sid)
    return rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_table(
    config_path: str,
    merged_pkl: str,
    h5_dir: str,
    episodes_json: str,
    suite: str,
    out_jsonl: str,
    *,
    u_t_factor: dict,
) -> list[dict]:
    """Full Pass 2: iterate calibration episodes -> write the signal-table JSONL.

    ``u_t_factor`` (required) is INJECTED into the gate before the table is built
    so ``u_t`` is actually computed and dumped — the reference
    ``dual_retrieval_active{,_l10}.yaml`` carry no ``u_t_factor`` (b2=0, Phase 3
    shape), which would leave every row's ``u_t`` None and make the calibrated b2
    a no-op (G2 R1 finding 1). The b2 the calibrator later fits is orthogonal to
    this structural factor choice.
    """
    config = load_cache_config(config_path)

    # Bind artifact identity: the config-preloaded artifact (the pool the gate's
    # library_stats and the LOEO temp pools derive from) MUST be the same file as
    # --merged-pkl, else a passing config + a different tagged artifact could
    # fabricate an inconsistent table (mirrors validate_dual_artifact's guard).
    cfg_preload = config.backend.in_memory.preload_path
    if not cfg_preload or os.path.realpath(cfg_preload) != os.path.realpath(merged_pkl):
        raise SystemExit(
            f"artifact identity mismatch: config preload_path={cfg_preload!r} vs "
            f"--merged-pkl={merged_pkl!r} (must resolve to the same file)"
        )

    # Inject the u_t factor so the gate computes u_t (b2 stays whatever the config
    # holds — the gate builds the factor whenever u_t_factor is set, regardless of
    # b2, and _compute_u_t returns the value we dump). Re-validate so a malformed
    # factor / non-in_memory backend fails loud here rather than at verdict time.
    config.checkpoints["cp1"].judge.u_t_factor = dict(u_t_factor)
    validate_cache_config(config)

    # Reuse the production gate (needs only the D+-only library_stats, unchanged
    # by LOEO). build_cache_components loads the full merged artifact via the
    # backend pool and wires the FailureAwareGateJudge with its u_t normalizer.
    comps = build_cache_components(config)
    gate = comps["judges"][CheckpointID.CP1]
    library_stats = comps["library_stats"]
    if getattr(gate, "_u_t_factor", None) is None:
        raise SystemExit(
            "gate has no active u_t factor after injection; calibration needs a "
            "computable u_t (check --u-t-descriptor/channel/past/future and that "
            "the judge is failure_aware_gate on an in_memory backend)"
        )

    # Entries + vector_dims come straight from the merged pkl (materialized to
    # torch once, shared read-only across every per-episode temp pool).
    with open(merged_pkl, "rb") as fh:
        art = pickle.load(fh)
    all_entries = art["entries"]
    vector_dims = art["vector_dims"]
    _entries_to_torch(all_entries)

    builder = _create_builder(
        config.key_builder.type,
        inner_type=config.key_builder.projection.inner_type,
        projection_weights_path=config.key_builder.projection.weights_path,
    )
    fusion_weights = _fusion_weights_from_config(config)
    strategy_cfg = config.checkpoints["cp1"].search_strategy
    min_top_k_hint = getattr(gate, "min_required_top_k", 0)

    success_map = _load_episode_success_map(episodes_json)

    h5_paths = sorted(Path(h5_dir).rglob("*.h5"))
    if not h5_paths:
        raise SystemExit(f"no .h5 files under --h5-dir {h5_dir}")

    all_rows: list[dict] = []
    n_episodes = n_skipped = 0
    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as f:
            ids = _derive_episode_ids(f, h5_path)
        # I_cal guard (G2 R1 finding 2/3): calibration MUST run on the even-index
        # split only; an odd (I_val) init here would silently calibrate on the
        # validation set. Fail loud rather than depend on manual --h5-dir choice.
        init_idx = ids["init_state_idx"]
        if init_idx is None or init_idx % 2 != 0:
            raise SystemExit(
                f"{h5_path.name}: init_state_idx={init_idx} is not an even I_cal "
                "index — Pass 1 calibration collection must cover only I_cal "
                "(init_idx % 2 == 0); check the --h5-dir split"
            )
        success = _lookup_success(success_map, ids)
        if success is None:
            # No usable success label -> cannot serve the Eq 24 objective
            # (episode_success is a required bool); skip loudly.
            n_skipped += 1
            logger.warning(
                "skipping %s: no success label for ids=%s in --episodes-json",
                h5_path.name, ids,
            )
            continue
        all_rows.extend(build_rows_for_episode(
            h5_path,
            builder=builder,
            all_entries=all_entries,
            vector_dims=vector_dims,
            library_stats=library_stats,
            strategy_cfg=strategy_cfg,
            fusion_weights=fusion_weights,
            gate=gate,
            suite=suite,
            episode_success=success,
            ids=ids,
            min_top_k_hint=min_top_k_hint,
        ))
        n_episodes += 1

    # u_t coverage fail-loud (G2 R1 finding 1): with u_t injected, at least some
    # interior steps MUST yield a computable (non-None) u_t; zero coverage means
    # the kinematic term is dead and the calibration cannot activate b2.
    n_u_t = sum(1 for r in all_rows if r["u_t"] is not None)
    if all_rows and n_u_t == 0:
        raise SystemExit(
            f"u_t non-NaN coverage is 0/{len(all_rows)} rows despite an injected "
            "u_t factor — every step degraded to margin-only (short chains / all "
            "forks?). Calibration would be b2-agnostic; aborting."
        )

    out = Path(out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row) + "\n")

    cov = (n_u_t / len(all_rows)) if all_rows else 0.0
    logger.info(
        "wrote %d rows from %d episodes (%d skipped, no label) -> %s",
        len(all_rows), n_episodes, n_skipped, out,
    )
    logger.info("u_t non-NaN coverage: %d/%d rows (%.1f%%)", n_u_t, len(all_rows), 100.0 * cov)
    return all_rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(
        description="Pass 2 replay: build the threshold-independent Phase 5 signal table"
    )
    ap.add_argument("--config", required=True, help="dual_retrieval_active(_l10).yaml")
    ap.add_argument("--merged-pkl", required=True, help="Merged D+/D- .pkl (== config preload_path)")
    ap.add_argument("--h5-dir", required=True, help="Pass 1 per-episode collect HDF5 dir (I_cal)")
    ap.add_argument("--episodes-json", required=True, help="main.py --save-episode-results JSON")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out-jsonl", required=True)
    # u_t factor to INJECT so u_t is computed (default = plan D-P7:
    # direction_online_state, window (past=2, future=1)). b2 is fit later.
    ap.add_argument("--u-t-descriptor", default="direction",
                    choices=["jerk", "direction", "dispersion", "path_length"])
    ap.add_argument("--u-t-channel", default="state", choices=["action", "state"])
    ap.add_argument("--u-t-past", type=int, default=2)
    ap.add_argument("--u-t-future", type=int, default=1)
    args = ap.parse_args()

    u_t_factor = {
        "descriptor": args.u_t_descriptor, "channel": args.u_t_channel,
        "past": args.u_t_past, "future": args.u_t_future,
    }
    build_table(
        args.config, args.merged_pkl, args.h5_dir, args.episodes_json,
        args.suite, args.out_jsonl, u_t_factor=u_t_factor,
    )


if __name__ == "__main__":
    main()
