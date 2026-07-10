"""Exit-gate validation for the TRACER Phase 4 merged D+/D- artifact.

Confirms the roadmap Phase 4 exit condition: "D- artifact loads AND dual
retrieval produces a NON-TRIVIAL margin distribution on the real library."

What it checks (all must pass):
  1. Coverage   : every entry outcome in {+1,-1}, both pools non-empty, zero None
                  (else entries vanish under enable_dual's outcome filter).
  2. Non-trivial: s_neg is not identically zero across queries (D- actually
                  participates) and the margin distribution has spread.
  3. Discrimination: failure-region queries (D- entries) score a higher mean
                  s_neg than success-region queries (D+ entries). Necessary but
                  WEAK -- D- entries self-match inside D-, so this mainly confirms
                  the pool holds real failure states the mechanism retrieves;
                  check #2 carries the load-bearing non-degeneracy signal.

It builds the real DualRetrievalKnnStrategy(enable_dual=True) via the config
loader (dual_retrieval_active.yaml) so the path under test is the production
one, then samples entries as queries and reads strategy.last_retrieval_signals().

Note (caveat, recorded in the report): a D- entry query self-matches inside D-,
so its s_neg is inflated toward 1.0; D+ entry queries are NOT in D-, so their
s_neg is a genuine cross-pool similarity. The discrimination check leans on the
D+/D- contrast; the honest signal is the D+ queries' non-zero s_neg spread.

Usage:
  PYTHONPATH=. uv run python exp/zixuan_proposal/validate_dual_artifact.py \
      --config     exp/zixuan_proposal/config/dual_retrieval_active.yaml \
      --merged-pkl exp/common/data/cache_artifacts/libero_spatial/cp1_mean_pool_dual.pkl \
      --suite      libero_spatial
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openpi.cache.components.search_strategy import SearchContext  # noqa: E402
from openpi.cache.config import build_cache_components, load_cache_config  # noqa: E402
from openpi.cache.types import CheckpointID  # noqa: E402


def _to_torch(query_keys: dict) -> dict:
    out = {}
    for k, v in query_keys.items():
        if isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(v).float()
        elif torch.is_tensor(v):
            out[k] = v.float()
        else:
            out[k] = v
    return out


def _evenly(seq: list, k: int) -> list:
    """Deterministic even-spaced sample of up to k items (no RNG)."""
    if len(seq) <= k:
        return seq
    step = len(seq) / k
    return [seq[int(i * step)] for i in range(k)]


def _summ(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": round(statistics.fmean(xs), 4),
        "median": round(statistics.median(xs), 4),
        "min": round(min(xs), 4),
        "max": round(max(xs), 4),
        "std": round(statistics.pstdev(xs), 4),
    }


def validate(config_path: str, merged_pkl: str, suite: str, sample_k: int, out_md: str) -> bool:
    # ---- build the production dual strategy + storage from config ----
    config = load_cache_config(config_path)
    # G2 R1: bind artifact identity. The backend loads
    # config.backend.in_memory.preload_path while coverage/report read
    # --merged-pkl; they MUST resolve to the same file, else a valid config
    # artifact + a different tagged artifact could fabricate a passing report.
    cfg_preload = config.backend.in_memory.preload_path
    if not cfg_preload or os.path.realpath(cfg_preload) != os.path.realpath(merged_pkl):
        raise SystemExit(
            f"artifact identity mismatch: config preload_path={cfg_preload!r} vs "
            f"--merged-pkl={merged_pkl!r} (must resolve to the same file)"
        )
    comps = build_cache_components(config)
    strat = comps["search_strategies"][CheckpointID.CP1]
    if not hasattr(strat, "last_retrieval_signals"):
        raise SystemExit(f"strategy {type(strat).__name__} has no last_retrieval_signals (not dual_retrieval_knn?)")

    # ---- coverage from the merged artifact ----
    with open(merged_pkl, "rb") as fh:
        art = pickle.load(fh)
    entries = art["entries"]
    pos = [e for e in entries if getattr(e, "outcome", None) == 1]
    neg = [e for e in entries if getattr(e, "outcome", None) == -1]
    none = [e for e in entries if getattr(e, "outcome", None) is None]
    coverage_ok = len(pos) > 0 and len(neg) > 0 and len(none) == 0

    # ---- run dual retrieval on sampled queries from each pool ----
    def run(qs: list) -> list[dict]:
        rows = []
        for e in qs:
            # Reset trajectory history so each entry is an INDEPENDENT single-state
            # query -- otherwise sequential searches accumulate a chain-aware
            # history and cross-pollute the margin (same-pool runs would inflate
            # discrimination artificially).
            if hasattr(strat, "on_episode_start"):
                strat.on_episode_start()
            ctx = SearchContext(
                query_keys=_to_torch(e.query_keys),
                checkpoint_id=CheckpointID.CP1,
                current_step=0,
            )
            strat.search(ctx)
            sig = strat.last_retrieval_signals()
            rows.append({"s_pos": sig.s_pos, "s_neg": sig.s_neg, "margin": sig.margin, "delta_pos": sig.delta_pos})
        return rows

    pos_rows = run(_evenly(pos, sample_k))
    neg_rows = run(_evenly(neg, sample_k))

    pos_sneg = [r["s_neg"] for r in pos_rows]
    neg_sneg = [r["s_neg"] for r in neg_rows]
    pos_margin = [r["margin"] for r in pos_rows]
    all_sneg = pos_sneg + neg_sneg

    # ---- gate checks (all three, per plan §4.3) ----
    # #2 Non-trivial (load-bearing): D+ queries are NOT in D-, so a non-zero,
    # varying s_neg proves D- is actively searched and pulls their margin below
    # s_pos (vs enable_dual=false where s_neg=0 => margin=s_pos).
    pos_sneg_active = len(pos_sneg) > 0 and max(pos_sneg) > 0.0 and statistics.pstdev(pos_sneg) > 1e-6
    pos_margin_below_spos = sum(1 for r in pos_rows if r["margin"] < r["s_pos"] - 1e-9)
    non_trivial = pos_sneg_active and pos_margin_below_spos > 0

    # #3 Discrimination (necessary but weak): failure-region queries score a
    # higher mean s_neg than success-region queries. D- entries self-match inside
    # D-, so this mainly confirms the pool holds real failure states; #2 carries
    # the honest non-degeneracy signal.
    mean_pos_sneg = statistics.fmean(pos_sneg) if pos_sneg else 0.0
    mean_neg_sneg = statistics.fmean(neg_sneg) if neg_sneg else 0.0
    discriminates = mean_neg_sneg > mean_pos_sneg

    passed = coverage_ok and non_trivial and discriminates

    # ---- report ----
    lines = []
    lines.append(f"# TRACER Phase 4 — dual-retrieval margin report ({suite})\n")
    lines.append(f"- **Verdict**: {'PASS ✅' if passed else 'FAIL ❌'}")
    lines.append(f"- config: `{config_path}`")
    lines.append(f"- artifact: `{merged_pkl}`")
    lines.append(f"- entries: {len(entries)} (D+ = {len(pos)}, D- = {len(neg)}, untagged = {len(none)})")
    lines.append(f"- sampled per pool: {len(pos_rows)} D+ / {len(neg_rows)} D-\n")
    lines.append("## Gate checks — all 3 must pass (D+ queries = success states, genuinely NOT in D-; history reset per query)")
    lines.append(f"1. **Coverage** (both pools non-empty, zero None): {'PASS' if coverage_ok else 'FAIL'}")
    lines.append(f"2. **Non-trivial margin** (load-bearing: D+ queries' s_neg non-zero + varies, and margin < s_pos for {pos_margin_below_spos}/{len(pos_rows)} of them): {'PASS' if non_trivial else 'FAIL'}")
    lines.append(f"3. **Discrimination** (necessary-but-weak: mean s_neg D- queries {round(mean_neg_sneg,4)} > D+ queries {round(mean_pos_sneg,4)}; D- self-match inflates it): {'PASS' if discriminates else 'FAIL'}\n")
    lines.append("## Signal distributions")
    lines.append(f"- D+ queries (success states, NOT in D- -> genuine cross-pool s_neg):")
    lines.append(f"    - s_pos  : {_summ([r['s_pos'] for r in pos_rows])}")
    lines.append(f"    - s_neg  : {_summ(pos_sneg)}")
    lines.append(f"    - margin : {_summ(pos_margin)}")
    lines.append(f"- D- queries (failure states, self-match in D- inflates s_neg toward 1):")
    lines.append(f"    - s_pos  : {_summ([r['s_pos'] for r in neg_rows])}")
    lines.append(f"    - s_neg  : {_summ(neg_sneg)}")
    lines.append(f"    - margin : {_summ([r['margin'] for r in neg_rows])}\n")
    lines.append("## Caveat")
    lines.append(
        "A D- entry used as a query self-matches inside D- (s_neg -> ~1.0), so the D- "
        "row's s_neg is inflated. The honest non-degeneracy signal is the D+ queries' "
        "s_neg spread (they are not in D-). The margin = s_pos - lambda*s_neg is now "
        "data-driven, no longer the enable_dual=false degenerate margin = s_pos.\n"
    )

    out = Path(out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    logger.info("wrote report -> %s", out)

    print("=" * 60)
    print(f"coverage_ok={coverage_ok}  non_trivial={non_trivial}  discriminates={discriminates}  (D+ margin<s_pos {pos_margin_below_spos}/{len(pos_rows)})")
    print(f"D+ query s_neg mean={round(mean_pos_sneg,4)} (genuine)  D- query s_neg mean={round(mean_neg_sneg,4)} (self-match inflated)")
    print(f"VERDICT: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)
    return passed


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="Validate merged D+/D- artifact produces non-trivial margins")
    p.add_argument("--config", required=True, help="dual_retrieval_active.yaml (enable_dual:true)")
    p.add_argument("--merged-pkl", required=True, help="Merged D+/D- .pkl (matches config preload_path)")
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--sample-k", type=int, default=60, help="Queries sampled per pool")
    p.add_argument("--out-md", default="exp/zixuan_proposal/analysis/phase4_dual_margin_report.md")
    args = p.parse_args()
    ok = validate(args.config, args.merged_pkl, args.suite, args.sample_k, args.out_md)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
