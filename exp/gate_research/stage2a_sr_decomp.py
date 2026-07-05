"""Stage 2a -- SR-gain decomposition + H1/H2/H3 adjudication (offline, 0 GPU).

For each (suite, point) group of {Stage-0 always_search baseline, N1, matched
periodic}, computes the evidence that decides where the V2 SR gain comes from:

- **H1 (dose / truncation)**: continuous cache-execution run-length distribution
  x episode success x condition. Periodic caps runs at ``<= cache_len``; H1 asks
  whether SR gain tracks how hard the runs are truncated.
- **H2 (on-manifold feedback)**: searched-step FULL_HIT rate per condition (does
  uniform injection make later searches hit better?).
- **H3 (WS execution poisoning)**: per-episode WARM_START-execution count split by
  success (is SR loss concentrated in partial-denoise replay?).
- **Delta-inf decomposition**: d_inf(cond - baseline) = skip-conversion +
  verdict-mix-migration + ep-length residual.
- **Statistics**: reused ``analyze_n1_live.mcnemar`` (b/c/continuity-corrected
  chi2) plus ``stage2_common.mcnemar_exact_p`` (exact binomial p), for
  N1-vs-baseline and periodic-vs-baseline, with a per-task Delta-SR slice.

Reads only recorded run manifests + Stage-0 gate_rows; writes one markdown
report. No ``src`` / inference-path dependency.

Usage:
    python -m exp.gate_research.stage2a_sr_decomp <manifest.json> [...] --out <report.md>
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

from exp.gate_research.analyze_n1_live import inf_value, mcnemar
from exp.gate_research.stage2_common import (
    WARM_START,
    action_source_seq,
    cache_run_lengths,
    load_run_episodes,
    load_stage0_episodes,
    mcnemar_exact_p,
)


# ------------------------------------------------------------------
# Per-condition primitives (pure, unit-tested)
# ------------------------------------------------------------------
def _ep_sources(ep):
    return action_source_seq(ep.hit_type_seq, ep.searched_seq)


def run_length_stats(eps, include_ws: bool) -> dict:
    """Cache-execution run-length distribution over episodes, split by success."""
    by_succ = {True: [], False: []}
    for e in eps:
        by_succ[e.success].extend(cache_run_lengths(_ep_sources(e), include_ws))
    allruns = by_succ[True] + by_succ[False]

    def _summ(runs):
        if not runs:
            return {"n": 0, "mean": 0.0, "median": 0.0, "max": 0}
        return {"n": len(runs), "mean": statistics.mean(runs),
                "median": statistics.median(runs), "max": max(runs)}

    return {"all": _summ(allruns), "success": _summ(by_succ[True]),
            "fail": _summ(by_succ[False])}


def searched_fh_rate(eps) -> dict:
    """FULL_HIT fraction among searched steps (H2 on-manifold-feedback signal)."""
    n_searched = n_fh = 0
    for e in eps:
        for hit, searched in zip(e.hit_type_seq, e.searched_seq):
            if searched:
                n_searched += 1
                n_fh += hit == "FULL_HIT"
    return {"n_searched": n_searched, "fh_rate": (n_fh / n_searched) if n_searched else 0.0}


def ws_exec_by_success(eps) -> dict:
    """Mean per-episode WARM_START-execution count in success vs fail episodes (H3)."""
    ws_succ, ws_fail = [], []
    for e in eps:
        n_ws = sum(1 for s in _ep_sources(e) if s == WARM_START)
        (ws_succ if e.success else ws_fail).append(n_ws)
    return {"success_mean": (statistics.mean(ws_succ) if ws_succ else 0.0),
            "fail_mean": (statistics.mean(ws_fail) if ws_fail else 0.0),
            "n_success": len(ws_succ), "n_fail": len(ws_fail)}


def _pooled_inf(eps) -> tuple[float, float, float]:
    """Return (pooled_inf, skip_frac, searched_inf_mean) over all steps of ``eps``.

    A skipped step contributes inf=1.0 (forced full inference, C10); a searched
    step contributes ``inf_value(hit_type, start_t)``.
    """
    n_steps = n_skip = 0
    inf_sum = searched_inf_sum = 0.0
    for e in eps:
        for hit, st, searched in zip(e.hit_type_seq, e.start_t_seq, e.searched_seq):
            n_steps += 1
            if not searched:
                n_skip += 1
                inf_sum += 1.0
            else:
                v = inf_value(hit, st)
                inf_sum += v
                searched_inf_sum += v
    if n_steps == 0:
        return 0.0, 0.0, 0.0
    n_searched = n_steps - n_skip
    return (inf_sum / n_steps, n_skip / n_steps,
            (searched_inf_sum / n_searched) if n_searched else 0.0)


def _ep_avg_inf(eps) -> float:
    """Mean over episodes of the per-episode mean inference fraction (for the
    ep-length composition residual vs the step-pooled mean)."""
    vals = []
    for e in eps:
        n = len(e.searched_seq)
        if n == 0:
            continue
        s = 0.0
        for hit, st, searched in zip(e.hit_type_seq, e.start_t_seq, e.searched_seq):
            s += 1.0 if not searched else inf_value(hit, st)
        vals.append(s / n)
    return statistics.mean(vals) if vals else 0.0


def decompose_delta_inf(baseline_eps, cond_eps) -> dict:
    """d_inf(cond - baseline) = skip_conversion + verdict_mix + ep_length_residual.

    - skip_conversion = skip_frac * (1 - searched_inf_baseline): steps converted
      to full inference instead of the baseline searched average.
    - verdict_mix = (1 - skip_frac) * (searched_inf_cond - searched_inf_baseline):
      the shift in verdict mix on searched steps.
    - ep_length_residual = balances the step-pooled d_inf against the
      episode-averaged d_inf (unequal episode lengths reweight the mean).
    """
    base_pooled, _base_skip, base_searched = _pooled_inf(baseline_eps)
    cond_pooled, skip_frac, cond_searched = _pooled_inf(cond_eps)
    d_inf_pooled = cond_pooled - base_pooled
    skip_conversion = skip_frac * (1.0 - base_searched)
    verdict_mix = (1.0 - skip_frac) * (cond_searched - base_searched)
    d_inf_epavg = _ep_avg_inf(cond_eps) - _ep_avg_inf(baseline_eps)
    ep_length_residual = d_inf_epavg - (skip_conversion + verdict_mix)
    return {
        "d_inf_pooled": d_inf_pooled, "d_inf_epavg": d_inf_epavg,
        "skip_conversion": skip_conversion, "verdict_mix": verdict_mix,
        "ep_length_residual": ep_length_residual,
        "skip_frac": skip_frac,
        "searched_inf_baseline": base_searched, "searched_inf_cond": cond_searched,
    }


def paired_sr(cond_eps, baseline_eps) -> dict:
    """Paired SR delta of ``cond`` vs ``baseline`` over the FULL shared unit set:
    reused continuity-corrected chi2 + exact binomial p. ``require_equal=True``
    fails fast on any unit-set mismatch (a partial run must surface as a
    data-integrity error, never silently shrink to an intersection)."""
    cond_map = {e.unit: e.success for e in cond_eps}
    base_map = {e.unit: e.success for e in baseline_eps}
    m = mcnemar(cond_map, base_map, require_equal=True)
    m["exact_p"] = mcnemar_exact_p(m["b"], m["c"])
    return m


def per_task_delta_sr(cond_eps, baseline_eps) -> dict:
    """Per-task-id Delta-SR (pp) + exact p, to flag single-task anomalies. Full
    pairing is enforced globally (``paired_sr``) and per task (``require_equal=
    True``): once the global sets match, each task's units match, so a per-task
    mismatch is a genuine integrity error, not silently intersected. ``n`` is
    reported so any surviving imbalance is visible."""
    base_by_task = collections.defaultdict(dict)
    for e in baseline_eps:
        base_by_task[e.task_id][e.unit] = e.success
    cond_by_task = collections.defaultdict(dict)
    for e in cond_eps:
        cond_by_task[e.task_id][e.unit] = e.success
    if set(base_by_task) != set(cond_by_task):
        d = set(base_by_task) ^ set(cond_by_task)
        raise ValueError(f"per-task: baseline/cond task-id sets differ ({sorted(d)})")
    out = {}
    for tid in sorted(cond_by_task):
        m = mcnemar(cond_by_task[tid], base_by_task[tid], require_equal=True)
        out[tid] = {"sr_delta_pp": m["sr_delta_pp"], "n": m["n_paired"],
                    "b": m["b"], "c": m["c"], "exact_p": mcnemar_exact_p(m["b"], m["c"])}
    return out


# ------------------------------------------------------------------
# Grouping + report
# ------------------------------------------------------------------
def _load_manifests(paths):
    return [json.loads(Path(p).read_text()) for p in paths]


def build_groups(manifests):
    """Pair each N1 (client_controlled) run with its matched periodic run and its
    Stage-0 baseline (from the N1 manifest's ``baseline_*`` fields)."""
    n1s = [m for m in manifests if m["gate_type"] == "client_controlled"]
    periodics = {m["run_id"]: m for m in manifests if m["gate_type"] == "periodic"}
    by_matched = collections.defaultdict(list)
    for m in periodics.values():
        by_matched[m["matched_to"]].append(m)
    groups = []
    for n1 in n1s:
        matched = by_matched.get(n1["run_id"], [])
        groups.append({"n1": n1, "periodic": matched[0] if matched else None})
    return groups


def analyze_group(g, include_ws: bool = False) -> dict:
    n1_m = g["n1"]
    n1_eps = load_run_episodes(n1_m)
    base_eps = load_stage0_episodes(
        n1_m["baseline_gate_rows_path"], n1_m["baseline_yaml_id"],
        replan_steps=n1_m["replan_steps"], journal_path=n1_m.get("baseline_journal_path"))
    out = {
        "run_id": n1_m["run_id"], "suite": n1_m["suite"], "config": n1_m["config"],
        "point": n1_m["point"],
        "baseline": {"runlen": run_length_stats(base_eps, include_ws),
                     "fh": searched_fh_rate(base_eps), "ws": ws_exec_by_success(base_eps)},
        "n1": {"runlen": run_length_stats(n1_eps, include_ws),
               "fh": searched_fh_rate(n1_eps), "ws": ws_exec_by_success(n1_eps),
               "d_inf": decompose_delta_inf(base_eps, n1_eps),
               "paired": paired_sr(n1_eps, base_eps),
               "per_task": per_task_delta_sr(n1_eps, base_eps)},
    }
    if g["periodic"] is not None:
        per_eps = load_run_episodes(g["periodic"])
        out["periodic"] = {
            "run_id": g["periodic"]["run_id"], "cache_len": g["periodic"]["cache_len"],
            "inference_len": g["periodic"]["inference_len"],
            "runlen": run_length_stats(per_eps, include_ws),
            "fh": searched_fh_rate(per_eps), "ws": ws_exec_by_success(per_eps),
            "d_inf": decompose_delta_inf(base_eps, per_eps),
            "paired": paired_sr(per_eps, base_eps),
            "per_task": per_task_delta_sr(per_eps, base_eps)}
    return out


def _summ_line(tag, s):
    r = s["runlen"]["all"]
    return (f"| {tag} | {r['n']} | {r['mean']:.2f} | {r['median']:.1f} | {r['max']} | "
            f"{100 * s['fh']['fh_rate']:.1f} | {s['ws']['success_mean']:.2f}/{s['ws']['fail_mean']:.2f} |")


def render_md(results, include_ws: bool) -> str:
    L = ["# Stage 2a — SR 增益分解 + H1/H2/H3 裁决", "",
         f"cache-run include_ws={include_ws}；口径见 plan §3.2。net/SR 单位 pp。", ""]
    for r in results:
        L += [f"## {r['run_id']} ({r['suite']} / {r['point']})", ""]
        L += ["**H1 run-length / H2 FH率 / H3 WS执行**", "",
              "| cond | n_runs | mean | median | max | searched FH% | WS succ/fail |",
              "|---|---|---|---|---|---|---|",
              _summ_line("baseline", r["baseline"]),
              _summ_line("N1", r["n1"])]
        if "periodic" in r:
            L.append(_summ_line(f"periodic(k{r['periodic']['cache_len']}/n{r['periodic']['inference_len']})",
                                r["periodic"]))
        L += ["", "**Δinf 分解 + 配对统计（vs baseline）**", "",
              "| cond | d_inf(pool) | skip_conv | verdict_mix | ep_len_res | ΔSR pp | chi2 | exact_p | b/c |",
              "|---|---|---|---|---|---|---|---|---|"]
        for cond in ("n1", "periodic"):
            if cond not in r:
                continue
            d, p = r[cond]["d_inf"], r[cond]["paired"]
            L.append(f"| {cond} | {d['d_inf_pooled']:+.3f} | {d['skip_conversion']:+.3f} | "
                     f"{d['verdict_mix']:+.3f} | {d['ep_length_residual']:+.3f} | "
                     f"{p['sr_delta_pp']:+.1f} | {p['mcnemar_chi2']:.2f} | {p['exact_p']:.4f} | "
                     f"{p['b']}/{p['c']} |")
        # per-task anomaly slice (N1)
        pt = r["n1"]["per_task"]
        worst = sorted(pt.items(), key=lambda kv: kv[1]["sr_delta_pp"])[:3]
        L += ["", "N1 per-task 最低 ΔSR（排单任务异常）: " +
              ", ".join(f"t{tid} {v['sr_delta_pp']:+.0f}pp(n{v['n']})" for tid, v in worst), ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Stage 2a SR-gain decomposition")
    ap.add_argument("manifests", nargs="+", help="N1 + matched-periodic run manifest paths")
    ap.add_argument("--out", required=True, help="output markdown report path")
    ap.add_argument("--include-ws", action="store_true",
                    help="fold WARM_START into cache-execution runs (R8 sensitivity)")
    a = ap.parse_args()

    manifests = _load_manifests(a.manifests)
    groups = build_groups(manifests)
    results = [analyze_group(g, include_ws=a.include_ws) for g in groups]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_md(results, a.include_ws))
    print(f"[stage2a] wrote {out} ({len(results)} groups)")


if __name__ == "__main__":
    main()
