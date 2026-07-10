"""Merge a success library D+ and a failure library D- into one tagged artifact.

TRACER Phase 4 (M2 failure-aware retrieval). Produces a single in-memory cache
artifact where every ``CacheEntry.outcome`` is set to ``+1`` (success pool D+)
or ``-1`` (failure pool D-), which is exactly what ``DualRetrievalKnnStrategy``
with ``enable_dual=True`` filters on (``pos_outcome=1`` / ``extra_filter_outcome=-1``).

Pipeline (plan §4.2 / D5):
  1. Load an existing D+ artifact (built success-only) and tag every entry ``+1``.
  2. Build D- from failure-rollout HDF5 via ``build_artifact(..., outcome_filter="failure")``
     and tag every entry ``-1``.
  3. Assert the two pools share ``vector_dims`` / ``key_builder_type`` and that
     all entry ids are globally unique.
  4. Concatenate entries; reuse the D+ artifact's ``library_stats`` unchanged
     (kinematic normalization stays over the success distribution only, so the
     failure states never contaminate the Phase-5 u_t normalizer).
  5. Pickle the merged artifact.

The D- HDF5 must have been collected from the SAME builder family as D+ (so the
query-key space matches); the script fails fast otherwise.

Usage:
  PYTHONPATH=. uv run python exp/zixuan_proposal/build_dual_artifact.py \
      --pos-artifact exp/common/data/cache_artifacts/libero_spatial/cp1_mean_pool.pkl \
      --neg-h5-dir   exp/common/data/db/failure_lib/libero_spatial \
      --builder-type cp1_mean_pool \
      --output       exp/common/data/cache_artifacts/libero_spatial/cp1_mean_pool_dual.pkl
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Import build_artifact from the sibling exp/common builder. Injecting the repo
# root mirrors how the codebase runs exp/ scripts (PYTHONPATH=. uv run ...).
# ------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from exp.common.build_in_memory_cache_artifact import build_artifact  # noqa: E402


def _tag_outcome(entries: list, outcome: int) -> None:
    """Set ``CacheEntry.outcome`` in place for every entry."""
    for entry in entries:
        entry.outcome = outcome


# Per-step datasets every complete collected episode must carry (see
# data_collector.py / guide.md). Missing any of these means a corrupt / partial
# episode that must not enter D-.
_REQUIRED_STEP_DATASETS = frozenset({"vision_0", "vision_1", "prompt_emb", "robot_state", "clean_action"})


def check_failure_h5_completeness(h5_dir: str, min_steps: int = 2) -> list:
    """Fail-loud gate on failure-rollout HDF5 before building D- (plan §5 / risk #10).

    Rejects (a) step holes (non-contiguous ``step_0000..N``), (b) stub episodes
    below ``min_steps`` (mid-episode disconnect flushes success=False at
    collection_policy.py:162), and (c) ANY step (not just step 0) missing a
    required embedding dataset. Returns the validated h5 path list; raises
    SystemExit on any violation so a corrupt D- is never built.
    """
    import glob

    import h5py

    paths = sorted(glob.glob(str(Path(h5_dir) / "*.h5")))
    if not paths:
        raise SystemExit(f"no .h5 files in failure dir {h5_dir}")
    problems: list[tuple[str, str]] = []
    for p in paths:
        with h5py.File(p, "r") as f:
            steps = sorted(k for k in f.keys() if k.startswith("step_"))
            idxs = [int(s.split("_")[1]) for s in steps]
            if idxs != list(range(len(idxs))):
                problems.append((p, f"step holes (indices {idxs[:6]}...)"))
                continue
            if len(steps) < min_steps:
                problems.append((p, f"{len(steps)} steps < min_steps={min_steps} (likely disconnect flush)"))
                continue
            bad = None
            for s in steps:  # every retained step must carry all embeddings
                missing = _REQUIRED_STEP_DATASETS - set(f[s].keys())
                if missing:
                    bad = (s, sorted(missing))
                    break
            if bad is not None:
                problems.append((p, f"{bad[0]} missing datasets {bad[1]}"))
    if problems:
        detail = "\n".join(f"  {Path(p).name}: {why}" for p, why in problems[:20])
        raise SystemExit(
            f"D- HDF5 completeness check FAILED ({len(problems)}/{len(paths)} bad episodes):\n{detail}"
        )
    logger.info("D- completeness OK: %d failure episodes, all contiguous + complete", len(paths))
    return paths


def build_dual_artifact(
    pos_artifact_path: str,
    neg_h5_dir: str,
    builder_type: str,
    output_path: str,
    checkpoint_id: str = "CP1",
    workers: int = 0,
    min_steps: int = 2,
    **builder_kwargs,
) -> dict:
    """Build and write the merged D+/D- artifact; return the artifact dict."""
    # ---- 1. D+ : load existing success library, tag +1 ----
    pos_path = Path(pos_artifact_path)
    if not pos_path.is_file():
        raise SystemExit(f"--pos-artifact not found: {pos_path}")
    with open(pos_path, "rb") as fh:
        pos = pickle.load(fh)
    if "entries" not in pos:
        raise SystemExit(f"D+ artifact {pos_path} missing 'entries'")
    pos_entries = pos["entries"]
    _tag_outcome(pos_entries, +1)
    logger.info("D+ : %d entries from %s (tagged outcome=+1)", len(pos_entries), pos_path.name)

    # ---- 2. D- : completeness gate, then build failure pool from HDF5, tag -1 ----
    check_failure_h5_completeness(neg_h5_dir, min_steps=min_steps)
    neg = build_artifact(
        neg_h5_dir, builder_type, checkpoint_id,
        workers=workers, outcome_filter="failure", **builder_kwargs,
    )
    neg_entries = neg["entries"]
    if not neg_entries:
        raise SystemExit(
            f"D- build produced 0 entries from {neg_h5_dir} -- no failure episodes "
            "found (check the collection actually captured success=False rollouts)."
        )
    _tag_outcome(neg_entries, -1)
    logger.info("D- : %d entries from %s (tagged outcome=-1)", len(neg_entries), neg_h5_dir)

    # ---- 3. consistency guards ----
    if pos["vector_dims"] != neg["vector_dims"]:
        raise SystemExit(
            f"vector_dims mismatch: D+={pos['vector_dims']} D-={neg['vector_dims']}. "
            "D- must be built from the same builder family as D+."
        )
    if pos.get("key_builder_type") != neg.get("key_builder_type"):
        raise SystemExit(
            f"key_builder_type mismatch: D+={pos.get('key_builder_type')} "
            f"D-={neg.get('key_builder_type')}"
        )
    all_ids = [e.id for e in pos_entries] + [e.id for e in neg_entries]
    if len(all_ids) != len(set(all_ids)):
        dupes = len(all_ids) - len(set(all_ids))
        raise SystemExit(f"entry id collision between D+ and D-: {dupes} duplicate id(s)")

    # ---- 4. merge (library_stats stays D+-only) ----
    # Carry D+ library_stats forward. If the D+ source artifact lacks the key
    # (e.g. libero_10/cp1_mean_pool.pkl was built without the enrich pass), we
    # MUST compute it from the D+ entries ONLY and never leave it None: the
    # in_memory backend (in_memory_backend.py ~L272) treats a None library_stats
    # as "legacy artifact" and fallback-recomputes it from ALL loaded entries
    # (D+ ∪ D-), which would fold the D- failure steps into Phase 5 u_t
    # normalization — exactly the D+-only contract this merge exists to protect
    # (plan D5 / roadmap Risk #6).
    pos_stats = pos.get("library_stats")
    if pos_stats is None:
        from openpi.cache.components.factors.base import LibraryStats

        pos_stats = LibraryStats.compute_from_entries(pos_entries)
        logger.info(
            "D+ source lacked library_stats -> computed D+-only from %d D+ entries "
            "(prevents backend fallback recompute over D+ ∪ D-)",
            len(pos_entries),
        )
    merged = {
        "key_builder_type": pos["key_builder_type"],
        "checkpoint_id": pos.get("checkpoint_id", checkpoint_id),
        "vector_dims": pos["vector_dims"],
        "entries": pos_entries + neg_entries,
        "library_stats": pos_stats,
    }
    # Carry the D+ provenance blocks forward if present (reducer/projection).
    for k in ("reducer_params", "projection_params"):
        if k in pos:
            merged[k] = pos[k]

    n_pos = sum(1 for e in merged["entries"] if e.outcome == 1)
    n_neg = sum(1 for e in merged["entries"] if e.outcome == -1)
    n_none = sum(1 for e in merged["entries"] if e.outcome is None)
    logger.info(
        "merged artifact: %d entries (D+=%d, D-=%d, untagged=%d)",
        len(merged["entries"]), n_pos, n_neg, n_none,
    )
    if n_none:
        raise SystemExit(f"{n_none} merged entries left outcome=None -- would vanish under enable_dual")

    # ---- 5. write ----
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(merged, fh)
    logger.info("wrote %d entries -> %s", len(merged["entries"]), out)
    return merged


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Merge D+/D- into one outcome-tagged artifact")
    parser.add_argument("--pos-artifact", required=True, help="Existing success-only D+ .pkl")
    parser.add_argument("--neg-h5-dir", required=True, help="Failure-rollout HDF5 dir (D- source)")
    parser.add_argument("--builder-type", required=True, help="Must match the D+ builder family")
    parser.add_argument("--output", required=True, help="Merged .pkl output path")
    parser.add_argument("--checkpoint-id", default="CP1", choices=["CP1"])
    parser.add_argument("--workers", type=int, default=0, help="D- build workers (0=all CPUs, -1=serial)")
    parser.add_argument("--min-steps", type=int, default=2, help="Reject failure h5 below this many steps (disconnect-flush guard)")
    # Pass-through builder knobs (only those a pool builder needs).
    parser.add_argument("--reducer-type", default="mean_pool")
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--inner-type", default="cp1_mean_pool")
    parser.add_argument("--projection-weights", default=None)
    args = parser.parse_args()

    build_dual_artifact(
        args.pos_artifact, args.neg_h5_dir, args.builder_type, args.output,
        checkpoint_id=args.checkpoint_id, workers=args.workers, min_steps=args.min_steps,
        reducer_type=args.reducer_type, output_tokens=args.output_tokens,
        inner_type=args.inner_type, projection_weights_path=args.projection_weights,
    )


if __name__ == "__main__":
    main()
