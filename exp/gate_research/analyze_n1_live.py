"""Deterministic analyzer for Stage 1b N1 live-validation runs.

Consumes one or more run manifests (written by ``run_n1_live.py``) and produces
the pass-line metrics: actual SR, skip%, actual inference_ratio, searched-step
verdict mix, the C9 three-tier latency net, the paired SR delta vs the Stage-0
always-search baseline (McNemar), and the direct N1-vs-matched-periodic verdict.

口径 (all verified against the plan):
- **skip is authoritative from the recorded ``searched`` field** for
  client_controlled runs (a missing / non-bool ``searched`` is a data-integrity
  error, never silently a skip). Periodic runs carry no ``searched`` field, so it
  is reconstructed from the PeriodicGate closed form over the per-episode
  decision ordinal. ``cp1_score is None`` never implies skip.
- **episode-global max-attempt dedup** is applied uniformly to live per-step
  rows, the Stage-0 baseline gate_rows, and the offline-replay input: per
  ``task_uid`` keep ONLY the global max ``attempt``'s rows (never mix attempts).
  Duplicate terminal ``task_uid`` in a journal is a fail-fast error.
- **cross-run pairing** is by ``(task_id, subset_init_state_idx)`` and REQUIRES
  equal unit sets (a short/partial run fails fast rather than degrading the
  paired N silently). Run episode counts are checked against the manifest.
- **matched-budget periodic**: each N1 run is paired to its periodic run via the
  periodic manifest's ``matched_to``; the analyzer verifies ``|Δskip|<=2pp`` and
  computes ``N1 SR >= periodic SR`` on the shared canonical units.

Usage:
    python analyze_n1_live.py <manifest.json> [<manifest2.json> ...] --out <report.md>
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

# Latency ledger (mirrors n1_offline_scan.py / roadmap C9).
SEARCH_MS = {"opt_2.6k": 4.0, "stock_2.6k": 34.0, "opt_50k": 70.0}
INFER_MS = 300.0
SKIP_MATCH_TOL_PP = 2.0
# N4 pairs its matched periodic on inf_ratio (the C10 pass-line axis), not skip%:
# N4's V2 skips land on HIT steps so skip% and inf_ratio decouple (roadmap C10).
# 0.03 ~= half a C9 latency tier.
INF_MATCH_TOL = 0.03


def warm_cost(start_t) -> float:
    t = 0.5 if start_t is None else float(start_t)
    return 1.0 - 0.5 * (1.0 - t)


def inf_value(hit_type, start_t) -> float:
    """Per-step inference fraction: FULL_HIT=0, WARM_START=warm_cost, MISS=1."""
    if hit_type == "FULL_HIT":
        return 0.0
    if hit_type == "WARM_START":
        return warm_cost(start_t)
    return 1.0


def load_jsonl(path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def parse_task_uid(task_uid: str) -> tuple[int, int]:
    """``{yaml_id}:{phase}:{task_id}:{subset_init_state_idx}`` -> (task_id, idx)."""
    parts = task_uid.rsplit(":", 3)
    if len(parts) != 4:
        raise ValueError(f"unexpected task_uid format: {task_uid!r}")
    return int(parts[2]), int(parts[3])


def dedup_episodes(rows: list[dict], yaml_id: str | None = None) -> dict[str, list[dict]]:
    """Return {task_uid: [step rows sorted by step_idx]} using the episode-global
    max attempt per task_uid (keep ONLY that attempt's rows, then dedup by
    step_idx). Excludes episode_summary rows; optionally filters to one yaml_id."""
    by_uid = collections.defaultdict(list)
    for r in rows:
        if r.get("_kind") == "episode_summary":
            continue
        if yaml_id is not None and r.get("yaml_id") != yaml_id:
            continue
        by_uid[r["task_uid"]].append(r)
    out: dict[str, list[dict]] = {}
    for uid, rs in by_uid.items():
        max_att = max(r.get("attempt", 1) for r in rs)
        kept = [r for r in rs if r.get("attempt", 1) == max_att]
        seen: dict[int, dict] = {}
        for r in kept:
            seen.setdefault(r["step_idx"], r)
        out[uid] = [seen[k] for k in sorted(seen)]
    return out


def unit_set(episodes: dict[str, list[dict]]) -> set[tuple[int, int]]:
    """Canonical (task_id, subset_init_state_idx) set from deduped episodes."""
    return {parse_task_uid(uid) for uid in episodes}


def check_complete_decisions(episodes: dict[str, list[dict]], spacing: int) -> None:
    """Every episode's decision ``step_idx`` must equal ``[0, spacing, 2*spacing,
    ...]`` using the TRUSTED replan-step ``spacing`` (from the run manifest, NOT
    inferred from the audited data -- inferring the spacing from the data cannot
    detect a UNIFORM drop across all episodes, e.g. every episode losing every
    other decision so 0,5,10 becomes 0,10,20). A dropped decision breaks the
    progression and would misalign periodic ordinal reconstruction -> fail-fast."""
    if isinstance(spacing, bool) or not isinstance(spacing, int) or spacing < 1:
        raise ValueError(f"trusted replan spacing must be an int >= 1, got {spacing!r}")
    for uid, ep in episodes.items():
        steps = [r["step_idx"] for r in ep]
        if steps != [k * spacing for k in range(len(ep))]:
            raise ValueError(
                f"episode {uid!r} decision step_idx {steps[:8]} != expected multiples of "
                f"trusted replan spacing {spacing}; a dropped decision misaligns ordinal "
                "reconstruction -- refusing to analyze")


def journal_success(rows: list[dict], yaml_id: str | None = None) -> dict[str, bool]:
    """{task_uid: success} for terminal episodes. Fail-fast on a duplicate
    terminal task_uid (the journal carries no attempt to disambiguate)."""
    out: dict[str, bool] = {}
    for r in rows:
        if yaml_id is not None and r.get("yaml_id") != yaml_id:
            continue
        uid = r["task_uid"]
        if uid in out:
            raise ValueError(
                f"duplicate terminal task_uid in journal: {uid!r}; cannot pick "
                "accepted attempt (journal carries no attempt). Inspect manually.")
        out[uid] = bool(r["success"])
    return out


def reconstruct_searched(n_steps: int, cache_len: int, inference_len: int) -> list[bool]:
    """PeriodicGate closed form: decision ordinal k searches iff
    ``k % (cache_len+inference_len) < cache_len`` (counter +1 per gate call)."""
    period = cache_len + inference_len
    return [(k % period) < cache_len for k in range(n_steps)]


def episode_searched(ep: list[dict], gate_type: str, manifest: dict) -> list[bool]:
    """Per-step ``searched`` flags for one episode. client_controlled uses the
    authoritative recorded field (missing / non-bool -> data-integrity error);
    periodic reconstructs from the closed form."""
    if gate_type == "periodic":
        return reconstruct_searched(len(ep), manifest["cache_len"], manifest["inference_len"])
    flags = []
    for r in ep:
        s = r.get("searched")
        if not isinstance(s, bool):
            raise ValueError(
                f"client_controlled per-step row lacks a bool 'searched' "
                f"(task_uid={r.get('task_uid')!r} step={r.get('step_idx')}); "
                "the N1 wrapper must stamp it -- refusing to guess skip.")
        flags.append(s)
    return flags


def run_metrics(manifest: dict) -> dict:
    """Compute observable metrics for one run from its per-step rows + journal."""
    episodes = dedup_episodes(load_jsonl(manifest["per_step_out_path"]),
                              yaml_id=manifest["yaml_id"])
    gate_type = manifest["gate_type"]
    check_complete_decisions(episodes, manifest["replan_steps"])  # trusted spacing

    n_steps = n_skip = 0
    inf_sum = 0.0
    verdict = collections.Counter()  # over searched steps only
    for ep in episodes.values():
        for r, searched in zip(ep, episode_searched(ep, gate_type, manifest)):
            n_steps += 1
            if not searched:
                n_skip += 1
                inf_sum += 1.0
            else:
                inf_sum += inf_value(r.get("hit_type"), r.get("start_t"))
                verdict[r.get("hit_type")] += 1
    skip_pct = n_skip / n_steps if n_steps else 0.0
    live_inf = inf_sum / n_steps if n_steps else 0.0

    jr = journal_success(load_jsonl(manifest["journal_path"]), yaml_id=manifest["yaml_id"])
    sr_by_unit = {parse_task_uid(uid): succ for uid, succ in jr.items()}
    # per-step episode set must equal the journal episode set (no orphan rows,
    # no journal episode missing its per-step data).
    ps_units = unit_set(episodes)
    if ps_units != set(sr_by_unit):
        d = ps_units ^ set(sr_by_unit)
        raise ValueError(
            f"run {manifest['run_id']!r}: per-step episode set != journal episode set "
            f"({len(d)} differing units, e.g. {sorted(d)[:5]})")
    exp_n = manifest.get("n_episodes")
    if exp_n is not None and len(sr_by_unit) != exp_n:
        raise ValueError(
            f"run {manifest['run_id']!r}: journal has {len(sr_by_unit)} episodes for "
            f"yaml_id={manifest['yaml_id']!r} but manifest expects {exp_n}")
    sr = sum(sr_by_unit.values()) / len(sr_by_unit) if sr_by_unit else 0.0
    return {"n_steps": n_steps, "skip_pct": skip_pct, "live_inf_ratio": live_inf,
            "verdict_mix": dict(verdict), "sr": sr, "sr_by_unit": sr_by_unit,
            "n_episodes": len(sr_by_unit)}


def baseline_inf_ratio(gate_rows_path, yaml_id: str, replan_steps: int) -> float:
    """Mean inf_value over the Stage-0 always-search rows of ``yaml_id`` after
    episode-global max-attempt dedup. Every kept row MUST have ``searched is
    True`` (baseline is always_search); fail-fast otherwise."""
    episodes = dedup_episodes(load_jsonl(gate_rows_path), yaml_id=yaml_id)
    if not episodes:
        raise ValueError(f"baseline gate_rows has no rows for yaml_id={yaml_id!r}")
    check_complete_decisions(episodes, replan_steps)
    rows = [r for ep in episodes.values() for r in ep]
    for r in rows:
        if r.get("searched") is not True:
            raise ValueError(
                f"baseline gate_rows for {yaml_id!r} has searched={r.get('searched')!r} "
                "(baseline must be always_search with searched=True on every row)")
    return sum(inf_value(r.get("hit_type"), r.get("start_t")) for r in rows) / len(rows)


def replay_offline(gate_rows_path, yaml_id: str, theta_low, theta_high, j, M,
                   replan_steps: int) -> dict:
    """Diagnostic (non-pass-gate): replay N1 over the Stage-0 always-search rows
    of ``yaml_id`` (episode-global max-attempt deduped) using the SHARED
    ``N1GateState`` to get the offline-predicted skip% and verdict mix."""
    from exp.gate_research.n1_gate_client import N1GateState

    episodes = dedup_episodes(load_jsonl(gate_rows_path), yaml_id=yaml_id)
    check_complete_decisions(episodes, replan_steps)
    n_steps = n_skip = 0
    verdict = collections.Counter()
    for ep in episodes.values():
        st = N1GateState(theta_low, theta_high, j, M)
        for r in ep:
            d = st.decide()
            n_steps += 1
            if d == "search":
                st.observe("search", r.get("cp1_score"))
                verdict[r.get("hit_type")] += 1
            else:
                n_skip += 1
                st.observe("skip", None)
    return {"skip_pct": n_skip / n_steps if n_steps else 0.0, "verdict_mix": dict(verdict)}


def mcnemar(n1: dict[tuple, bool], base: dict[tuple, bool], require_equal: bool = True) -> dict:
    """Paired SR comparison over the common (task_id, idx) units. When
    ``require_equal`` the two unit sets MUST be identical (a partial run fails
    fast rather than silently shrinking the paired N)."""
    if require_equal and set(n1) != set(base):
        diff = set(n1) ^ set(base)
        raise ValueError(
            f"pairing unit sets differ by {len(diff)} units (e.g. {sorted(diff)[:5]}); "
            "refusing to pair on a partial intersection")
    common = set(n1) & set(base)
    b = sum(1 for u in common if not n1[u] and base[u])  # N1 fail, baseline succeed
    c = sum(1 for u in common if n1[u] and not base[u])  # N1 succeed, baseline fail
    n = len(common)
    n1_sr = sum(n1[u] for u in common) / n if n else 0.0
    base_sr = sum(base[u] for u in common) / n if n else 0.0
    chi2 = ((abs(b - c) - 1) ** 2) / (b + c) if (b + c) > 0 else 0.0  # continuity-corrected
    return {"n_paired": n, "n1_sr": n1_sr, "base_sr": base_sr,
            "sr_delta_pp": 100 * (n1_sr - base_sr), "b": b, "c": c, "mcnemar_chi2": chi2}


def net_row(skip_pct: float, d_inf: float) -> dict:
    return {tag: skip_pct * ms - d_inf * INFER_MS for tag, ms in SEARCH_MS.items()}


def match_periodic(n1_run: dict, periodic_runs: list[dict]) -> dict | None:
    """Pair an N1 run to its UNIQUE matched periodic run (periodic manifest
    ``matched_to == n1 run_id``, same suite/config); verify the budget match and
    compute the direct N1-vs-periodic verdict. ``periodic_pass`` requires BOTH a
    matched budget (``|Δskip|<=2pp``) AND ``N1 SR >= periodic SR`` -- a budget
    out-of-bounds is a FAIL, not a pass."""
    matches = [pr for pr in periodic_runs if pr["matched_to"] == n1_run["run_id"]]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(
            f"N1 run {n1_run['run_id']!r} has {len(matches)} matched periodic runs "
            f"({[pr['run_id'] for pr in matches]}); expected exactly one")
    pr = matches[0]
    if (pr["suite"], pr["config"]) != (n1_run["suite"], n1_run["config"]):
        raise ValueError(
            f"periodic {pr['run_id']!r} suite/config {(pr['suite'], pr['config'])} "
            f"!= N1 {n1_run['run_id']!r} {(n1_run['suite'], n1_run['config'])}")
    skip_delta = abs(n1_run["skip_pct"] - pr["skip_pct"]) * 100
    skip_ok = skip_delta <= SKIP_MATCH_TOL_PP
    pair = mcnemar(n1_run["sr_by_unit"], pr["sr_by_unit"], require_equal=True)
    n1_ge = pair["n1_sr"] >= pair["base_sr"]
    return {
        "periodic_run_id": pr["run_id"], "skip_delta_pp": skip_delta,
        "skip_match_ok": skip_ok, "n1_sr": pair["n1_sr"], "periodic_sr": pair["base_sr"],
        "sr_delta_pp": pair["sr_delta_pp"], "n1_ge_periodic": n1_ge,
        "periodic_pass": skip_ok and n1_ge,
    }


def match_periodic_n4(n4_run: dict, periodic_runs: list[dict]) -> dict | None:
    """Pair an N4 run to a matched periodic by NEAREST live inf_ratio.

    Non-mutating: candidates are the same-``(suite, config)`` periodic runs; their
    manifest ``matched_to`` (which points at old N1 runs) is read-only provenance
    and is NEVER rewritten -- unlike ``match_periodic`` which pairs N1 via
    ``matched_to``. Returns None when the nearest candidate is out of
    ``INF_MATCH_TOL`` (-> overall pending). Raises on a true inf_ratio tie among
    in-tolerance candidates (refuse a silent pick, mirroring ``match_periodic``'s
    raise-on->1). ``periodic_pass_n4 = (|Δinf| <= INF_MATCH_TOL) and (N4 SR >=
    periodic SR)``; since we only return inside tolerance, it reduces to the SR
    comparison."""
    cands = [pr for pr in periodic_runs
             if (pr["suite"], pr["config"]) == (n4_run["suite"], n4_run["config"])]
    if not cands:
        return None

    def dinf(pr: dict) -> float:
        return abs(n4_run["live_inf_ratio"] - pr["live_inf_ratio"])

    cands_sorted = sorted(cands, key=dinf)
    best = cands_sorted[0]
    inf_delta = dinf(best)
    if inf_delta > INF_MATCH_TOL:
        # Nearest candidate out of tolerance: no valid matched periodic -> pending
        # (a tie out here is moot since no pick is made).
        return None
    if len(cands_sorted) >= 2 and abs(dinf(cands_sorted[1]) - inf_delta) < 1e-9:
        raise ValueError(
            f"N4 run {n4_run['run_id']!r} has a true inf_ratio tie between periodic "
            f"{best['run_id']!r} and {cands_sorted[1]['run_id']!r} "
            f"(|Δinf|={inf_delta:.6f}); disambiguate the candidate set")
    pair = mcnemar(n4_run["sr_by_unit"], best["sr_by_unit"], require_equal=True)
    n4_ge = pair["n1_sr"] >= pair["base_sr"]
    skip_delta = abs(n4_run["skip_pct"] - best["skip_pct"]) * 100
    return {
        "periodic_run_id": best["run_id"],
        # provenance only, copied verbatim from the candidate manifest (unchanged):
        "periodic_matched_to": best.get("matched_to"),
        "inf_delta": inf_delta, "inf_match_ok": True, "skip_delta_pp": skip_delta,
        "n4_sr": pair["n1_sr"], "periodic_sr": pair["base_sr"],
        "sr_delta_pp": pair["sr_delta_pp"], "n4_ge_periodic": n4_ge,
        "periodic_pass_n4": n4_ge,
    }


def n4_overall(row: dict, pv: dict | None) -> tuple[str, dict | None]:
    """Compute the N4 overall verdict and its three pass components.

    N4 pass requires ALL of: ``periodic_pass_n4`` (inf-matched SR >= periodic),
    SR preserved vs baseline (``sr_preservation_ok``), and ``net@34 >= 0`` -- the
    net gate is what N1's ``overall`` omits, so it is spelled out here. No matched
    periodic (pv is None) -> ``pending`` (not pass, not fail)."""
    if pv is None:
        return "pending", None
    comps = {
        "sr_ok": row["vs_baseline"]["sr_preservation_ok"],
        "periodic_pass_n4": pv["periodic_pass_n4"],
        "net34_ok": row["net"]["stock_2.6k"] >= 0,
    }
    return ("pass" if all(comps.values()) else "fail"), comps


def analyze(manifests: list[dict]) -> list[dict]:
    """Compute the full result row for each run, including the N1-vs-periodic
    verdict for client_controlled runs that have a matched periodic run."""
    runs = []
    for m in manifests:
        met = run_metrics(m)
        live_units = set(met["sr_by_unit"])
        base_eps = dedup_episodes(load_jsonl(m["baseline_gate_rows_path"]),
                                  yaml_id=m["baseline_yaml_id"])
        check_complete_decisions(base_eps, m["replan_steps"])
        base_gr_units = unit_set(base_eps)
        base_sr = {parse_task_uid(uid): succ for uid, succ in
                   journal_success(load_jsonl(m["baseline_journal_path"]),
                                   yaml_id=m["baseline_yaml_id"]).items()}
        # Completeness: the live run, its SR baseline journal, and the C9 baseline
        # gate_rows must cover exactly the same canonical unit set.
        if not (live_units == set(base_sr) == base_gr_units):
            raise ValueError(
                f"run {m['run_id']!r}: canonical unit sets differ across live "
                f"({len(live_units)}) / baseline-journal ({len(base_sr)}) / "
                f"baseline-gate-rows ({len(base_gr_units)})")
        base_inf = baseline_inf_ratio(m["baseline_gate_rows_path"], m["baseline_yaml_id"],
                                      m["replan_steps"])
        pair = mcnemar(met["sr_by_unit"], base_sr, require_equal=True)
        pair["sr_preservation_ok"] = pair["sr_delta_pp"] >= -1.0
        d_inf = met["live_inf_ratio"] - base_inf
        row = {
            "run_id": m["run_id"], "suite": m["suite"], "config": m["config"],
            "point": m["point"], "gate_type": m["gate_type"],
            "gate_family": m.get("gate_family", "n1"), "L": m.get("L"),
            "matched_to": m.get("matched_to"),
            "skip_pct": met["skip_pct"], "sr_by_unit": met["sr_by_unit"],
            "live_inf_ratio": met["live_inf_ratio"], "baseline_inf_ratio": base_inf,
            "d_inf": d_inf, "verdict_mix": met["verdict_mix"],
            "n_episodes": met["n_episodes"], "net": net_row(met["skip_pct"], d_inf),
            "vs_baseline": pair,
        }
        # replay_offline reconstructs offline skip% with N1GateState -- valid only
        # for N1 runs. N4 (gate_family="n4") is a different machine, so bypass it
        # (its live skip% is the authoritative signal; no offline N1 replay).
        if (m["gate_type"] == "client_controlled" and m.get("theta_low") is not None
                and m.get("gate_family", "n1") != "n4"):
            row["offline"] = replay_offline(
                m["baseline_gate_rows_path"], m["baseline_yaml_id"],
                m["theta_low"], m["theta_high"], m["j"], m["M"], m["replan_steps"])
        runs.append(row)

    periodic_runs = [r for r in runs if r["gate_type"] == "periodic"]
    for r in runs:
        if r["gate_type"] != "client_controlled":
            continue
        if r.get("gate_family", "n1") == "n4":
            # N4 pairs its periodic by inf_ratio and gates overall on 3 conditions.
            pv = match_periodic_n4(r, periodic_runs)
            r["periodic_verdict"] = pv
            r["overall"], comps = n4_overall(r, pv)
            if comps is not None:  # surface components so any failure is localizable
                r["overall_components"] = comps
            continue
        pv = match_periodic(r, periodic_runs)
        r["periodic_verdict"] = pv
        # Overall pass = SR preserved AND periodic present & passing. No periodic
        # yet -> pending (NOT a pass).
        if pv is None:
            r["overall"] = "pending"
        else:
            r["overall"] = ("pass" if r["vs_baseline"]["sr_preservation_ok"] and pv["periodic_pass"]
                            else "fail")
    return runs


def _strip_units(runs: list[dict]) -> list[dict]:
    """Drop the non-serializable (tuple-keyed) sr_by_unit before JSON output."""
    return [{k: v for k, v in r.items() if k != "sr_by_unit"} for r in runs]


def _periodic_cell(r: dict) -> str:
    pv = r.get("periodic_verdict")
    if pv is None:
        return "pending" if r["gate_type"] == "client_controlled" else "—"
    if "periodic_pass_n4" in pv:  # N4 verdict: matched by inf_ratio, not skip%
        tag = "PASS" if pv["periodic_pass_n4"] else "FAIL"
        return f"ΔSR {pv['sr_delta_pp']:+.1f}pp |Δinf|={pv['inf_delta']:.3f} {tag}"
    tag = "PASS" if pv["periodic_pass"] else ("FAIL(OOB)" if not pv["skip_match_ok"] else "FAIL")
    return f"ΔSR {pv['sr_delta_pp']:+.1f}pp |Δskip|={pv['skip_delta_pp']:.1f}pp {tag}"


def _n4_components_table(runs: list[dict]) -> list[str]:
    """N4-only verdict-components table: one row per N4 run, showing the three
    overall pass components so a fail is localizable on the same page as overall
    (e.g. net34 fails while periodic/SR pass). Empty for reports with no N4 run."""
    n4_runs = [r for r in runs if r.get("gate_family") == "n4"]
    if not n4_runs:
        return []
    out = ["", "N4 overall 分量（三条件同真 = pass；任一 False = fail；无 inf 内对照 = pending）：",
           "| N4 run | L | sr_ok | periodic_pass_n4 | net34_ok | overall |",
           "|---|---|---|---|---|---|"]
    for r in n4_runs:
        c = r.get("overall_components")
        if c is None:  # pending: no matched periodic within tolerance
            cells = "— | — | —"
        else:
            cells = (f"{'ok' if c['sr_ok'] else 'FAIL'} | "
                     f"{'PASS' if c['periodic_pass_n4'] else 'FAIL'} | "
                     f"{'ok' if c['net34_ok'] else 'FAIL'}")
        out.append(f"| {r['run_id']} | {r.get('L', '—')} | {cells} | {r.get('overall', '—')} |")
    return out


def render_md(runs: list[dict]) -> str:
    has_n4 = any(r.get("gate_family") == "n4" for r in runs)
    # N1-only reports keep the Stage-1b title/caption byte-for-byte; N4 reports get
    # a correct title, a components table, and the N4 three-condition caption.
    title = "# Gate Live 验证结果（N4 混合门 / Stage 3a）" if has_n4 else "# N1 Live 验证结果（Stage 1b）"
    lines = [title, ""]
    lines.append("| run | gate | skip% | SR(run/base) | ΔSR pp | SR-ok | vs-periodic | overall |"
                 if has_n4 else
                 "| run | gate | skip% | SR(N1/base) | ΔSR pp | SR-ok | vs-periodic | overall |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in runs:
        p = r["vs_baseline"]
        sr_ok = "ok" if p.get("sr_preservation_ok") else "FAIL"
        lines.append(
            f"| {r['run_id']} | {r['gate_type']} | {100*r['skip_pct']:.1f} | "
            f"{100*p['n1_sr']:.1f}/{100*p['base_sr']:.1f} | {p['sr_delta_pp']:+.1f} | "
            f"{sr_ok} | {_periodic_cell(r)} | {r.get('overall', '—')} |")
    lines += ["", "| run | inf(live/base) | Δinf | net@4/34/70 | b/c |", "|---|---|---|---|---|"]
    for r in runs:
        p = r["vs_baseline"]
        nets = "/".join(f"{r['net'][t]:+.1f}" for t in SEARCH_MS)
        lines.append(
            f"| {r['run_id']} | {r['live_inf_ratio']:.3f}/{r['baseline_inf_ratio']:.3f} | "
            f"{r['d_inf']:+.3f} | {nets} | {p['b']}/{p['c']} |")
    lines += _n4_components_table(runs)
    caption = ("口径：skip% 由权威 `searched`（periodic 按 ordinal 重建）；SR/periodic 配对按 "
               "`(task_id, subset_init_state_idx)`，要求 live/baseline-journal/baseline-gate-rows 三方 unit 等集；"
               "baseline_inf_ratio 来自 Stage-0 gate_rows（episode-global max-attempt 去重）。"
               "**overall pass** 需 SR 保真（ΔSR ≥ −1pp）**且** 同预算 N1 SR ≥ periodic（|Δskip| ≤ 2pp，越界=FAIL）；"
               "periodic 未跑 = pending（非 pass）。")
    if has_n4:
        caption += ("　**N4 overall pass** 需三条件同真：matched periodic 按 **|Δinf| ≤ 0.03** 配对"
                    "（inf_ratio 轴，非 skip%）下 **N4_SR ≥ periodic_SR**（periodic_pass_n4）、"
                    "**SR ≥ baseline − 1pp**（sr_ok）、**net@stock_2.6k ≥ 0**（net34_ok）；"
                    "inf 容差内无候选 = pending，inf 并列 = raise。")
    lines += ["", caption, ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="analyze Stage 1b N1 live runs")
    ap.add_argument("manifests", nargs="+", help="run manifest json paths (N1 + matched periodic)")
    ap.add_argument("--out", required=True, help="output markdown report path")
    ap.add_argument("--result-manifest", default="", help="optional result-manifest json path")
    a = ap.parse_args()

    manifests = [json.loads(Path(p).read_text()) for p in a.manifests]
    runs = analyze(manifests)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_md(runs))
    print(f"[analyze_n1_live] wrote {out}")
    if a.result_manifest:
        Path(a.result_manifest).write_text(json.dumps(_strip_units(runs), indent=2))
        print(f"[analyze_n1_live] wrote {a.result_manifest}")


if __name__ == "__main__":
    main()
