"""Pass 3 analyzer: paired illustrative-vs-calibrated rollout -> BHR/FFR/IR/SR.

Computes the TRACER exit-gate metrics (Eq 26-28) with explicit numerators /
denominators from the per-step gate JSONL of the illustrative and calibrated
cache-ON runs, plus ``SR_base(I_val)`` from the cache-OFF baseline's
episode-results JSON. The ``safe_reuse(t) := episode_success == True`` proxy is
identical to the offline solver's (``MarginGateCalibrator``). Incomplete
episodes (``success is None``) are excluded from BHR/FFR/SR numerator +
denominator and reported as ``n_incomplete``.

Exit gate: BHR_calibrated < BHR_illustrative, SR_calibrated >= SR_base - eps,
IR_calibrated <= IR_illustrative.

Usage:
  PYTHONPATH=. uv run python exp/zixuan_proposal/analyze_phase5_rollout.py \
      --baseline-episodes  .../baseline_episode_results.json \
      --illustrative-jsonl .../illustrative.jsonl \
      --calibrated-jsonl   .../calibrated.jsonl \
      --illustrative-episodes .../illustrative_episode_results.json \
      --calibrated-episodes   .../calibrated_episode_results.json \
      --suite libero_spatial --c-warm 0.75 --eps 0.02 \
      --out-md exp/zixuan_proposal/analysis/phase5_calibration_report_spatial.md
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_VERDICTS = {"FULL_HIT", "WARM_START", "MISS"}


def _load_jsonl(path: str) -> list[dict]:
    return [
        json.loads(ln)
        for ln in Path(path).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


def compute_metrics(rows: list[dict], *, c_warm: float) -> dict:
    """BHR/FFR/IR with explicit numerators/denominators from per-step verdict rows.

    ``rows`` are per-step gate records carrying ``phase``, ``hit_type`` and
    ``success`` (bool | None). Only eval-phase verdict rows count. success==None
    rows are excluded from BHR/FFR (they lack a safety label) but still counted
    in IR's denominator (the verdict happened) and reported as ``n_incomplete``.
    """
    n = n_fh = n_ws = n_miss = 0
    fh_labeled = fh_bad = safe = miss_safe = n_incomplete = 0
    for r in rows:
        if r.get("phase") != "eval":
            continue
        ht = (r.get("hit_type") or "").upper()
        if ht not in _VERDICTS:
            continue
        n += 1
        success = r.get("success")
        if ht == "FULL_HIT":
            n_fh += 1
        elif ht == "WARM_START":
            n_ws += 1
        else:
            n_miss += 1
        if success is None:
            n_incomplete += 1
            continue
        if success is True:
            safe += 1
            if ht == "MISS":
                miss_safe += 1
        if ht == "FULL_HIT":
            fh_labeled += 1
            if success is False:
                fh_bad += 1
    bhr = (fh_bad / fh_labeled) if fh_labeled else 0.0          # Eq 27
    ffr = (miss_safe / safe) if safe else 0.0                   # Eq 28
    ir = ((n_miss + c_warm * n_ws) / n) if n else 0.0           # Eq 26
    return {
        "n_verdicts": n, "n_full_hit": n_fh, "n_warm_start": n_ws, "n_miss": n_miss,
        "n_incomplete": n_incomplete, "BHR": bhr, "BHR_num": fh_bad, "BHR_den": fh_labeled,
        "FFR": ffr, "FFR_num": miss_safe, "FFR_den": safe, "IR": ir,
        "zero_full_hit": fh_labeled == 0,
    }


def sr_from_episodes(path: str) -> dict:
    """Task success rate over main.py --save-episode-results JSON (success!=None)."""
    return _sr_from_rows(_load_episode_rows(path))


def _load_episode_rows(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("episodes", [])


def _sr_from_rows(rows: list[dict]) -> dict:
    labeled = [r for r in rows if r.get("success") is not None]
    n_succ = sum(1 for r in labeled if r["success"])
    return {"SR": (n_succ / len(labeled)) if labeled else 0.0,
            "n_success": n_succ, "n_episodes": len(labeled)}


def _init_key(row: dict) -> tuple:
    """Canonical ``(task_id, init_idx)`` — episode-results write ``init_state_idx``
    while ``examples/libero/main.py``'s per-step recorder writes
    ``subset_init_state_idx`` (the SAME subset index under a different name, G2 R2
    finding 1). Accept either so real production JSONL is analyzable."""
    s = row.get("init_state_idx")
    if s is None:
        s = row.get("subset_init_state_idx")
    return row.get("task_id"), s


def _provenance_keys(
    rows: list[dict], *, what: str, require_seed: bool, require_unique: bool = False,
) -> tuple[set, set]:
    """``((task_id, init_idx) set, seed set)`` from episode / per-step rows.

    Fail-loud on rows missing the provenance ids. ``require_seed`` (episode-results
    only — per-step rows carry no seed) additionally fails loud on any row lacking
    a seed, so a same-seed claim is never defaulted-true on unverifiable data
    (G2 R2 finding 3). ``require_unique`` (episode-results only — per-step logs
    legitimately have one row per step) fails loud on a repeated
    ``(task_id, init_idx)``: set-ification would otherwise swallow a duplicate /
    contradictory record while ``_sr_from_rows`` still counts it, biasing the
    paired SR (G2 R3 finding)."""
    keys, seeds = set(), set()
    for r in rows:
        t, s = _init_key(r)
        if t is None or s is None:
            raise SystemExit(
                f"{what}: a row is missing task_id / init_state_idx(|subset_init_state_idx) "
                "— cannot verify paired provenance across the three rollouts"
            )
        k = (int(t), int(s))
        if require_unique and k in keys:
            raise SystemExit(
                f"{what}: duplicate episode identity {k} — episode-results must carry "
                "exactly one row per (task_id, init_idx); a repeated or contradictory "
                "record would bias the paired SR (non 1-to-1 samples)"
            )
        keys.add(k)
        if require_seed:
            sd = r.get("seed")
            if sd is None:
                raise SystemExit(
                    f"{what}: a row is missing 'seed' — same-seed pairing cannot be "
                    "verified (fail loud rather than assume it holds)"
                )
            seeds.add(int(sd))
    return keys, seeds


def check_paired_provenance(
    baseline: list[dict], illustrative: list[dict], calibrated: list[dict],
    *, illus_steps: list[dict], calib_steps: list[dict],
) -> set:
    """Fail-loud unless the three runs cover the SAME non-empty I_val episode set
    at a single shared seed, and each cache-ON per-step log covers exactly that set.

    Enforces (1) identical non-empty ``(task_id, init_idx)`` sets across baseline /
    illustrative / calibrated episode-results, each with UNIQUE per-run identities
    (one row per episode — a duplicate would bias SR), (2) I_val = odd ``init_idx`` only,
    (3) every episode row carries a seed, each run has exactly one seed, and all
    three seeds are equal, and (4) each cache-ON run's per-step rows cover the
    FULL shared episode set (equality, not a subset — a truncated log would
    otherwise bias the paired metrics). Returns the shared key set.
    """
    b_keys, b_seeds = _provenance_keys(
        baseline, what="baseline episodes", require_seed=True, require_unique=True)
    i_keys, i_seeds = _provenance_keys(
        illustrative, what="illustrative episodes", require_seed=True, require_unique=True)
    c_keys, c_seeds = _provenance_keys(
        calibrated, what="calibrated episodes", require_seed=True, require_unique=True)
    if not b_keys:
        raise SystemExit("baseline episode-results are empty — nothing to pair")
    if not (b_keys == i_keys == c_keys):
        raise SystemExit(
            "paired-provenance mismatch: the three rollouts cover different "
            f"(task_id, init_idx) sets (|baseline|={len(b_keys)} "
            f"|illustrative|={len(i_keys)} |calibrated|={len(c_keys)}; "
            f"b△i={len(b_keys ^ i_keys)} i△c={len(i_keys ^ c_keys)})"
        )
    even = sorted({s for (_, s) in b_keys if s % 2 == 0})
    if even:
        raise SystemExit(
            f"validation set contains even (non-I_val) init_idx {even[:10]} "
            "— Pass 3 must run on I_val (init_idx % 2 == 1) only"
        )
    # Every run has exactly one seed (require_seed guarantees each is non-empty)
    # and all three agree — no defaulting-true when seeds are absent.
    if not (len(b_seeds) == len(i_seeds) == len(c_seeds) == 1 and b_seeds == i_seeds == c_seeds):
        raise SystemExit(
            f"seed mismatch (single shared seed required): baseline={sorted(b_seeds)} "
            f"illustrative={sorted(i_seeds)} calibrated={sorted(c_seeds)}"
        )
    # Each cache-ON per-step log must cover the FULL shared episode set (a strict
    # subset = a truncated log biasing the paired comparison).
    for name, steps in (("illustrative", illus_steps), ("calibrated", calib_steps)):
        s_keys, _ = _provenance_keys(steps, what=f"{name} per-step rows", require_seed=False)
        if s_keys != b_keys:
            raise SystemExit(
                f"{name} per-step rows do not cover the full paired I_val set "
                f"(missing {len(b_keys - s_keys)}, stray {len(s_keys - b_keys)}) — "
                "a truncated per-step log would bias BHR/FFR/IR"
            )
    return b_keys


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Phase 5 paired rollout analysis (BHR/FFR/IR/SR)")
    ap.add_argument("--baseline-episodes", required=True, help="cache-OFF I_val episode-results (SR_base)")
    ap.add_argument("--illustrative-jsonl", required=True)
    ap.add_argument("--calibrated-jsonl", required=True)
    ap.add_argument("--illustrative-episodes", required=True)
    ap.add_argument("--calibrated-episodes", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--c-warm", type=float, default=0.75)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    base_eps = _load_episode_rows(args.baseline_episodes)
    illus_eps = _load_episode_rows(args.illustrative_episodes)
    calib_eps = _load_episode_rows(args.calibrated_episodes)
    illus_steps = _load_jsonl(args.illustrative_jsonl)
    calib_steps = _load_jsonl(args.calibrated_jsonl)

    # Fail loud unless the three runs are genuinely paired on I_val at one seed.
    check_paired_provenance(
        base_eps, illus_eps, calib_eps,
        illus_steps=illus_steps, calib_steps=calib_steps,
    )

    illus = compute_metrics(illus_steps, c_warm=args.c_warm)
    calib = compute_metrics(calib_steps, c_warm=args.c_warm)
    sr_base = _sr_from_rows(base_eps)["SR"]
    sr_illus = _sr_from_rows(illus_eps)["SR"]
    sr_calib = _sr_from_rows(calib_eps)["SR"]

    bhr_down = calib["BHR"] < illus["BHR"]
    sr_ok = sr_calib >= sr_base - args.eps
    ir_ok = calib["IR"] <= illus["IR"]
    passed = bhr_down and sr_ok and ir_ok

    lines = [
        f"# TRACER Phase 5 — calibration report ({args.suite})\n",
        f"- **Verdict**: {'PASS ✅' if passed else 'FAIL ❌'}",
        f"- exit gate: BHR↓ ({bhr_down}) AND SR_calibrated ≥ SR_base − ε ({sr_ok}) AND IR↓/= ({ir_ok})",
        f"- SR_base(I_val)={sr_base:.4f}  SR_illustrative={sr_illus:.4f}  SR_calibrated={sr_calib:.4f}  (ε={args.eps})\n",
        "## Metrics (paired, same I_val + seed)",
        f"- BadHitRate: illustrative {illus['BHR']:.4f} ({illus['BHR_num']}/{illus['BHR_den']}) "
        f"-> calibrated {calib['BHR']:.4f} ({calib['BHR_num']}/{calib['BHR_den']})",
        f"- FFR: illustrative {illus['FFR']:.4f} ({illus['FFR_num']}/{illus['FFR_den']}) "
        f"-> calibrated {calib['FFR']:.4f} ({calib['FFR_num']}/{calib['FFR_den']})",
        f"- IR: illustrative {illus['IR']:.4f} -> calibrated {calib['IR']:.4f} (c_warm={args.c_warm})",
        f"- verdicts illustrative={illus['n_verdicts']} (incomplete {illus['n_incomplete']}) / "
        f"calibrated={calib['n_verdicts']} (incomplete {calib['n_incomplete']})",
    ]
    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("VERDICT=%s  BHR %.4f->%.4f  SR_base=%.4f SR_cal=%.4f  IR %.4f->%.4f -> %s",
                "PASS" if passed else "FAIL", illus["BHR"], calib["BHR"], sr_base, sr_calib,
                illus["IR"], calib["IR"], out)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
