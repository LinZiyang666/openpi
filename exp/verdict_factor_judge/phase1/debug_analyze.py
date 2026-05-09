"""Phase 1 dead-loop diagnostic — joins server-side `[vd ...]` log with
client-side `cache_eval_results.json` to produce a 4-step verdict on
which layer is the bottleneck.

Designed to be invoked locally after:
  - Server (remote): runs with `OPENPI_CACHE_VERDICT_DEBUG=1` and
    `|& tee /tmp/vd_<yaml>.log`, then user ships the log to local via
    tar / scp.
  - Client (local): runs `exp.common.run_cache_experiments.py --runs N`
    against the remote server, outputs land in --log-dir + cache_eval
    json under --yaml-dir.

Usage:
  uv run python -m exp.verdict_factor_judge.phase1.debug_analyze \
      --server-log /path/to/vd_idx4.log \
      --client-aggregate /path/to/cache_eval_results.json
      [--client-dir /path/to/log-dir]    # optional, for episode_results sanity

The analyzer never modifies inputs. Output is plain text to stdout — copy
back into the runbook / a follow-up letter as needed.

Coupling map:
  DEPENDS ON: nothing in openpi codebase — pure parser over text logs +
              JSON aggregates. Safe to run standalone with only stdlib.
  CONSUMED BY: human reviewer interpreting Phase 1 yaml outcomes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


# --------------------------------------------------------------------
# Regex patterns matching the four `[vd ...]` log prefixes from
# src/openpi/cache/components/{judge.py, factors/{composers/__init__.py,
# runtime_continuity.py, consensus.py}}
# --------------------------------------------------------------------

_RE_VD_STEP = re.compile(
    r"\[vd step=(?P<step>\d+) cp=(?P<cp>\w+)\] "
    r"raw_nan: (?P<raw_nan>.*?) \| "
    r"norm valid=(?P<valid>\d+) nan=(?P<nan>\d+) cold=(?P<cold>\d+)"
    r"(?: sentinel=(?P<sentinel>\w+))? \| "
    r"hit=(?P<hit>\w+) start_t=(?P<start_t>\S+) winner=(?P<winner>\S+)"
)
_RE_VD_COMPOSER = re.compile(
    r"\[vd composer\] (?P<body>.*)"
)
_RE_COMPOSER_SCORE = re.compile(r"score=(?P<score>[0-9.]+)")
_RE_COMPOSER_DECISION = re.compile(r"-> (?P<decision>FULL_HIT|WARM_START|MISS)")
_RE_VD_F1A_T_BAIL = re.compile(r"\[vd f1a_t bail\] (?P<body>.*)")
_RE_VD_F2_BAIL = re.compile(r"\[vd f2 bail\] (?P<body>.*)")


# --------------------------------------------------------------------
# Step 1: hit_type distribution from the per-verdict summary line
# --------------------------------------------------------------------


def parse_server_log(path: Path) -> dict:
    """Single pass over the server stdout log; bucket every `[vd ...]` line.

    Returns a dict with: hits (Counter), sentinels (Counter),
    composer_decisions (Counter), composer_scores (list[float]),
    f1a_t_bail (Counter), f2_bail (Counter), per_ext_nan (dict[str, list[
    tuple[int, int]]]).
    """
    hits = Counter()
    sentinels = Counter()
    composer_scores: list[float] = []
    composer_decisions = Counter()
    f1a_t_bail = Counter()
    f2_bail = Counter()
    # `raw_nan: F1aA=0/4 F1aT=4/4 F1bA=8/20 F1bT=11/20 F2=0/1` — track
    # mean nan ratio per family for the Step 3 readout.
    per_ext_nan: dict[str, list[tuple[int, int]]] = defaultdict(list)
    n_vd_lines = 0

    with path.open() as fh:
        for line in fh:
            if "[vd " not in line:
                continue
            m = _RE_VD_STEP.search(line)
            if m:
                n_vd_lines += 1
                hits[m.group("hit")] += 1
                sentinel = m.group("sentinel")
                if sentinel:
                    sentinels[sentinel] += 1
                # raw_nan body: "RuntimeContinuityAction=0/4 RuntimeContinuityState=4/4 ..."
                for tally in m.group("raw_nan").split():
                    if "=" not in tally or "/" not in tally:
                        continue
                    name, ratio = tally.split("=", 1)
                    nan_s, total_s = ratio.split("/", 1)
                    try:
                        per_ext_nan[name].append((int(nan_s), int(total_s)))
                    except ValueError:
                        pass
                continue
            m = _RE_VD_COMPOSER.search(line)
            if m:
                body = m.group("body")
                score_m = _RE_COMPOSER_SCORE.search(body)
                dec_m = _RE_COMPOSER_DECISION.search(body)
                if score_m:
                    try:
                        composer_scores.append(float(score_m.group("score")))
                    except ValueError:
                        pass
                if dec_m:
                    composer_decisions[dec_m.group("decision")] += 1
                continue
            m = _RE_VD_F1A_T_BAIL.search(line)
            if m:
                # Normalize the bail message to a short bucket key.
                body = m.group("body")
                if "history.states" in body:
                    f1a_t_bail["boundary (history.states < K)"] += 1
                elif "walk_next returned" in body:
                    f1a_t_bail["walk_next short (winner near tail)"] += 1
                elif "winner.robot_state missing" in body:
                    f1a_t_bail["winner has no robot_state"] += 1
                elif "missing robot_state" in body:
                    f1a_t_bail["forward entry has no robot_state"] += 1
                else:
                    f1a_t_bail[body[:60]] += 1
                continue
            m = _RE_VD_F2_BAIL.search(line)
            if m:
                body = m.group("body")
                if "K_eff=" in body:
                    f2_bail["K_eff < 2 (search returned too few)"] += 1
                elif "candidate pool agreement" in body:
                    f2_bail["candidate pool agreement (variance below eps)"] += 1
                else:
                    f2_bail[body[:60]] += 1
    return {
        "n_vd_lines": n_vd_lines,
        "hits": hits,
        "sentinels": sentinels,
        "composer_scores": composer_scores,
        "composer_decisions": composer_decisions,
        "f1a_t_bail": f1a_t_bail,
        "f2_bail": f2_bail,
        "per_ext_nan": per_ext_nan,
    }


def load_client_aggregate(path: Path) -> dict:
    """Pull (n_total, n_success) from the runner's
    `cache_eval_results.json` — supports both list-of-rows and
    {episodes: rows} shapes since the runner schema has shifted across
    versions."""
    if not path.exists():
        return {"n_total": 0, "n_success": 0}
    d = json.loads(path.read_text())
    rows = d.get("episodes", d) if isinstance(d, dict) else d
    n_total = len(rows) if isinstance(rows, list) else 0
    n_success = sum(1 for r in rows if r.get("success"))
    return {"n_total": n_total, "n_success": n_success}


# --------------------------------------------------------------------
# Step 4 helper — percentile summary on the composer score sample.
# --------------------------------------------------------------------


def summarize_scores(scores: list[float]) -> str:
    if not scores:
        return "  (no composer score samples — composer was never invoked)"
    s = sorted(scores)
    n = len(s)
    return (
        f"  n={n} | min={s[0]:.4f} P25={s[n//4]:.4f} P50={s[n//2]:.4f} "
        f"P75={s[(n*3)//4]:.4f} max={s[-1]:.4f} | "
        f">=0.5: {sum(1 for x in s if x >= 0.5)} ({sum(1 for x in s if x >= 0.5)/n*100:.1f}%) | "
        f">=0.3: {sum(1 for x in s if x >= 0.3)} ({sum(1 for x in s if x >= 0.3)/n*100:.1f}%)"
    )


def fmt_counter(c: Counter, total: int) -> str:
    if not c:
        return "  (none)"
    out = []
    for k, v in c.most_common():
        pct = v / total * 100 if total else 0
        out.append(f"  {k:50s} {v:>6d} ({pct:>5.1f}%)")
    return "\n".join(out)


def diagnose(parsed: dict, client: dict) -> None:
    n_vd = parsed["n_vd_lines"]
    print()
    print("=" * 78)
    print("PHASE 1 DEAD-LOOP DIAGNOSIS")
    print("=" * 78)
    print()
    print(f"server log: {n_vd} per-verdict summary lines parsed")
    print(f"client outcome: {client['n_success']}/{client['n_total']} success "
          f"({client['n_success']/client['n_total']*100:.1f}% if not 0 else n/a)" if client['n_total'] else "client outcome: (no cache_eval_results.json found)")

    # ----- Step 1: hit_type distribution -----
    print()
    print("─" * 78)
    print("STEP 1 — hit_type distribution (per verdict)")
    print("─" * 78)
    print(fmt_counter(parsed["hits"], n_vd))

    # ----- Step 2: sentinel firing -----
    print()
    print("─" * 78)
    print("STEP 2 — cold-start sentinel triggers")
    print("─" * 78)
    sent = parsed["sentinels"]
    if sent:
        print(fmt_counter(sent, n_vd))
        if sent.get("all_nan_warm_fallback", 0) > 0:
            print("  → P0 fallback IS firing (good — it's the path that prevents pure-MISS lock-in)")
        if sent.get("all_nan_miss", 0) > 0:
            print("  ⚠ all_nan_miss firing — fallback config did NOT propagate to this CompositeJudge."
                  " Check yaml `judge.all_nan_fallback` block.")
    else:
        print("  (no sentinel triggered — composer ran on every verdict, see Step 4)")

    # ----- Step 3: extractor NaN attribution -----
    print()
    print("─" * 78)
    print("STEP 3 — extractor NaN attribution (which factor family is the dead-loop driver)")
    print("─" * 78)
    print()
    print("  Per-extractor mean raw NaN ratio across all verdicts:")
    for ext_name, samples in sorted(parsed["per_ext_nan"].items()):
        if not samples:
            continue
        ratios = [n / t for (n, t) in samples if t > 0]
        if not ratios:
            continue
        mean_pct = statistics.mean(ratios) * 100
        nan100 = sum(1 for (n, t) in samples if t > 0 and n == t)
        nan100_pct = nan100 / len(samples) * 100
        print(f"    {ext_name:35s} mean_nan={mean_pct:>5.1f}%  100%-NaN-verdicts: "
              f"{nan100}/{len(samples)} ({nan100_pct:.1f}%)")

    print()
    print("  F1a-T bail-out reasons (only when F1a-T was 100% NaN):")
    print(fmt_counter(parsed["f1a_t_bail"], sum(parsed["f1a_t_bail"].values())))
    print()
    print("  F2 bail-out reasons (only when F2 was NaN):")
    print(fmt_counter(parsed["f2_bail"], sum(parsed["f2_bail"].values())))

    # ----- Step 4: composer score distribution -----
    print()
    print("─" * 78)
    print("STEP 4 — composer score distribution (only verdicts where composer ran)")
    print("─" * 78)
    print(summarize_scores(parsed["composer_scores"]))
    print()
    print("  Composer-internal decision tally:")
    print(fmt_counter(parsed["composer_decisions"], sum(parsed["composer_decisions"].values())))

    # ----- Verdict -----
    print()
    print("=" * 78)
    print("BOTTLENECK INTERPRETATION")
    print("=" * 78)
    n_hits_warm = parsed["hits"].get("WARM_START", 0)
    n_hits_full = parsed["hits"].get("FULL_HIT", 0)
    n_miss = parsed["hits"].get("MISS", 0)
    n_sentinel_warm = parsed["sentinels"].get("all_nan_warm_fallback", 0)
    n_sentinel_miss = parsed["sentinels"].get("all_nan_miss", 0)
    n_composer_hits = sum(parsed["composer_decisions"].get(k, 0) for k in ("FULL_HIT", "WARM_START"))

    if n_vd == 0:
        print("  ⚠ No `[vd step=` lines in server log — was OPENPI_CACHE_VERDICT_DEBUG=1 set?")
    elif n_sentinel_miss > 0.5 * n_vd:
        print("  → all_nan_miss dominates: P0 fallback NOT plumbed through to this yaml's "
              "CompositeJudge. Open the yaml and confirm the `judge.all_nan_fallback` "
              "block; regenerate via `uv run -m exp.verdict_factor_judge.phase1_spec`.")
    elif n_sentinel_warm > 0.5 * n_vd:
        print("  → all_nan_warm_fallback dominates: factor pipeline is dead-locked "
              "(every factor returns NaN). Step 3 attributes which extractor (F1a-T "
              "boundary vs walk_next vs F2 candidate-pool collapse vs F1b boundary). "
              "If `walk_next short` + `candidate pool agreement` together cover >80% "
              "of bails, the dead-loop hypothesis from "
              "logs/phase1_dead_loop_diagnosis.letter.md is empirically confirmed.")
    elif n_composer_hits == 0 and len(parsed["composer_scores"]) > 0:
        scores = sorted(parsed["composer_scores"])
        p50 = scores[len(scores) // 2]
        print(f"  → composer ran ({len(scores)} times) but score P50={p50:.3f} is "
              f"below every threshold. Threshold tuning is plan-level; lower "
              f"`tier_thresholds.warm_start` from 0.3 in the affected yamls.")
    elif n_hits_warm + n_hits_full > 0.3 * n_vd:
        print(f"  ✓ pipeline is producing hits "
              f"({(n_hits_full + n_hits_warm)/n_vd*100:.1f}% hit rate). "
              f"User's 'all MISS' impression was a misread of success-rate noise; "
              f"compare client success rate vs Phase 0 Ceiling-W (0.98) — they "
              f"should align.")
    else:
        print("  (mixed outcome — see Steps 1-4 individually; no single dominant pattern)")
    print()


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server-log", required=True, type=Path,
                   help="Path to server stdout capture (the |& tee target).")
    p.add_argument("--client-aggregate", type=Path, default=None,
                   help="Optional path to client-side cache_eval_results.json "
                        "for a success-rate cross-check.")
    p.add_argument("--client-dir", type=Path, default=None,
                   help="Optional --log-dir from the client run; not "
                        "consumed yet but reserved for future per-batch joins.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if not args.server_log.exists():
        raise SystemExit(f"server log not found: {args.server_log}")
    parsed = parse_server_log(args.server_log)
    client = (
        load_client_aggregate(args.client_aggregate)
        if args.client_aggregate is not None else {"n_total": 0, "n_success": 0}
    )
    diagnose(parsed, client)


if __name__ == "__main__":
    main()
