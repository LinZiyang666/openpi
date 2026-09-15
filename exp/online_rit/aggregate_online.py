"""Aggregate online RIT rollouts: success, priced inference ratio, calibration, dynamics.

Episode acceptance follows ``exp.libero_groot.aggregate_rit_lg`` and adds the
per-step identity checks the online line needs:

* only journal rows that are terminal **and** ``accepted`` count; a row whose
  ``error`` is non-empty is an infrastructure failure and is reported apart
  from environment failures (``success=false, error=None``) -- it never enters
  the success rate, and an arm that has one is marked ``incomplete``;
* per-step rows must match the accepted attempt, carry ``accepted is True``,
  agree with the journal's ``run_id`` and ``yaml_id``, and be unique on
  ``(yaml_id, task_uid, attempt, step_idx)``; a second row with the same key
  and different content is a conflict and aborts the aggregate;
* an arm whose online stream was marked ``flow_invalid`` (any decision) is
  reported ``incomplete`` -- its numbers are printed for diagnosis only.

Beyond success and pricing (side-step batches included, and without them) it
reports, per arm: the executed violation rate ``V_a = P(d > delta | executed)``
with a Wilson upper bound (descriptive: decisions are correlated), the
pre-decision tail rate ``E = P(d > q_pre(s))`` per tier x source x frozen score
band x support kind (``None`` below 30 decisions / 10 episodes) with the
pinball loss, tier shares, shadowed tiers, consecutive-warm run lengths,
rejected counts and the revision trajectory. ``paired_bootstrap`` gives the
task-stratified whole-episode bootstrap intervals for success and executed
risk differences, pairing by suite, parent pool, task and original init index
(``--pair A,B``). Risk uses resampled counts and denominators.

With ``--pool-manifest`` the per-step ``orig_init_state_idx`` of a subset run
is checked against the manifest (``--pool-key adapt|terminal``) so terminal /
adaptation pools are provably disjoint.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

from exp.online_rit.common import TIER_TS, CostLedger, load_ledger, tier_index_of, write_json

VERDICTS = ("FULL_HIT", "WARM_START", "MISS")
MIN_CELL_ROWS = 30
MIN_CELL_EPISODES = 10


def accepted_episodes(journal: pathlib.Path) -> tuple[dict[str, int], dict[str, dict[str, dict]], dict[str, str], dict[str, int]]:
    """accepted attempt per uid, outcome per (arm, uid), run_id per uid, infra failures per arm."""
    attempt: dict[str, int] = {}
    episodes: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    run_ids: dict[str, str] = {}
    infra: dict[str, int] = collections.Counter()
    with journal.open(encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row.get("status") not in ("done", "failed") or not row.get("accepted"):
                continue
            uid = row["task_uid"]
            arm = row["yaml_id"]
            if row.get("error"):
                infra[arm] += 1
                continue  # infrastructure failure: not an environment outcome
            attempt[uid] = row["attempt"]
            rec = {"success": row["status"] == "done", "yaml_id": arm,
                   "attempt": row["attempt"], "run_id": row.get("run_id"), "violation": {}}
            if uid in episodes[arm] and episodes[arm][uid] != rec:
                raise SystemExit(f"conflicting accepted journal rows for {uid}")
            episodes[arm][uid] = rec
            if row.get("run_id") is not None:
                run_ids[uid] = str(row["run_id"])
    return attempt, episodes, run_ids, dict(infra)


def _wilson_upper(k: int, n: int, z: float = 1.96) -> float | None:
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((centre + half) / denom)


def _band(s: float | None, edges: list[float] | None) -> str:
    if s is None or edges is None:
        return "q?"
    return f"q{sum(float(s) > e for e in edges)}"


def aggregate(
    data_dir: pathlib.Path,
    ledger: CostLedger | None,
    *,
    score_bands: list[float] | None = None,
    pool_manifest: dict | None = None,
    pool_key: str | None = None,
) -> dict:
    attempt, episodes, run_ids, infra = accepted_episodes(data_dir / "journal.jsonl")
    journal_arm = {uid: arm for arm, eps in episodes.items() for uid in eps}
    tier_idx = {tier_index_of(t): t for t in TIER_TS}
    per_arm: dict[str, dict] = {}
    seen_rows: dict[tuple, str] = {}
    pool_index: dict[int, set[int]] | None = None
    if pool_manifest is not None:
        if not pool_key or pool_key not in pool_manifest:
            raise SystemExit(f"pool manifest has no key {pool_key!r}")
        pool_index = {int(t): {int(i) for i in v} for t, v in pool_manifest[pool_key].items()}

    def arm(yaml_id: str) -> dict:
        if yaml_id not in per_arm:
            per_arm[yaml_id] = {
                "counts": collections.Counter(), "spend": 0.0, "spend_no_fb": 0.0,
                "fb_batches": collections.Counter(), "viol": {t: [0, 0] for t in tier_idx},
                "tail": collections.defaultdict(lambda: {"exceed": 0, "n": 0, "pinball": 0.0, "episodes": set()}),
                "rejected": 0, "unavailable": collections.Counter(), "shadowed": collections.Counter(),
                "runs": [], "n_updates_trace": [], "dynamics": [], "delta": None, "learned": 0, "flow_invalid": False,
                "invalid_reasons": [], "pool_violations": 0,
            }
        return per_arm[yaml_id]

    per_step = data_dir / "per_step.jsonl"
    current_run: dict[tuple, int] = {}
    if per_step.exists():
        with per_step.open(encoding="utf-8") as fh:
            for raw in fh:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                hit = row.get("hit_type")
                if hit is None:
                    continue
                uid = row["task_uid"]
                if attempt.get(uid) != row.get("attempt"):
                    continue
                if row.get("accepted") is not True:
                    continue
                if row.get("yaml_id") != journal_arm[uid]:
                    raise SystemExit(f"per_step yaml_id for {uid} differs from its journal arm")
                if row.get("run_id") != episodes[journal_arm[uid]][uid].get("run_id"):
                    raise SystemExit(f"per_step row for {uid} carries run_id {row['run_id']} but the journal accepted {run_ids.get(uid)}")
                key = (row["yaml_id"], uid, row.get("attempt"), row.get("step_idx"))
                digest = json.dumps(row, sort_keys=True)
                if key in seen_rows:
                    if seen_rows[key] != digest:
                        raise SystemExit(f"conflicting per_step rows for {key}")
                    continue
                seen_rows[key] = digest
                if hit not in VERDICTS:
                    raise SystemExit(f"unpriceable hit_type {hit!r} in {data_dir}")
                a = arm(row["yaml_id"])
                if pool_index is not None:
                    allowed = pool_index.get(int(row.get("task_id", -1)), set())
                    if int(row.get("orig_init_state_idx", -1)) not in allowed:
                        a["pool_violations"] += 1
                ep = episodes[row["yaml_id"]][uid]
                for field in ("suite", "parent_pool_sha256", "task_id", "orig_init_state_idx"):
                    value = row.get(field)
                    if pool_manifest is not None and field in ("suite", "parent_pool_sha256"):
                        expected_value = pool_manifest.get(field)
                        if expected_value is not None and value is not None and value != expected_value:
                            raise SystemExit(f"episode {field} differs from the declared pool")
                        value = value or expected_value
                    if field in ep and ep[field] != value:
                        raise SystemExit(f"conflicting episode {field} for {uid}")
                    ep[field] = value
                start_t = row.get("start_t")
                key_hit = hit if hit != "WARM_START" else f"WARM_START@{float(start_t):g}"
                a["counts"][key_hit] += 1
                diag = row.get("online_rit") or {}
                fb_batch = int(diag.get("fb_batch_size") or 0)
                a["fb_batches"][str(fb_batch)] += 1
                if ledger is not None:
                    pricing = {"online": bool(diag), "update_enabled": diag.get("update_enabled", True),
                               "has_candidate": diag.get("candidate", True)}
                    a["spend_no_fb"] += ledger.decision_ms(hit, start_t, 0, include_feedback=False, **pricing)
                    a["spend"] += ledger.decision_ms(hit, start_t, fb_batch, **pricing)
                    ep["online"] = bool(diag)
                    ep["update_enabled"] = pricing["update_enabled"]
                ep_key = (row["yaml_id"], uid)
                if hit == "WARM_START":
                    current_run[ep_key] = current_run.get(ep_key, 0) + 1
                else:
                    if current_run.get(ep_key):
                        a["runs"].append(current_run[ep_key])
                    current_run[ep_key] = 0
                if not diag:
                    continue
                a["dynamics"].append({
                    "task_uid": uid, "step_idx": row.get("step_idx"),
                    **{k: diag.get(k) for k in ("server_instance_id", "update_seq", "decision_revision", "update_revision_after", "cuts", "support_kind", "rejected")},
                    "supported_fraction": sum(v == "supported" for v in (diag.get("support_kind") or {}).values()) / len(tier_idx),
                    "unavailable_fraction": sum(v == "unavailable" for v in (diag.get("support_kind") or {}).values()) / len(tier_idx),
                })
                a["delta"] = diag.get("delta", a["delta"])
                a["rejected"] += int(diag.get("rejected") or 0)
                a["learned"] += int(bool(diag.get("learned")))
                if diag.get("flow_invalid"):
                    a["flow_invalid"] = True
                    a["invalid_reasons"].extend(diag.get("invalid_reasons") or [])
                if diag.get("update_revision_after") is not None:
                    a["n_updates_trace"].append(int(diag["update_revision_after"]))
                for t in diag.get("shadowed") or []:
                    a["shadowed"][str(t)] += 1
                for t, kind in (diag.get("support_kind") or {}).items():
                    if kind == "unavailable":
                        a["unavailable"][str(t)] += 1
                exec_tier = tier_index_of(float(start_t)) if hit == "WARM_START" else None
                q_pre = diag.get("q_pre") or {}
                kinds = diag.get("support_kind") or {}
                band = _band(row.get("cp1_score", row.get("score")), score_bands)
                for fb in diag.get("fb") or []:
                    t = int(fb["tier"])
                    d = float(fb["d"])
                    src = fb.get("source")
                    if src == "executed" and t == exec_tier and a["delta"] is not None:
                        a["viol"][t][0] += d > float(a["delta"])
                        a["viol"][t][1] += 1
                        v = ep["violation"].setdefault(str(t), {"k": 0, "n": 0})
                        v["k"] += int(d > float(a["delta"]))
                        v["n"] += 1
                    q = q_pre.get(str(t))
                    if q is not None:
                        cell = a["tail"][f"{t}|{src}|{band}|{kinds.get(str(t), '?')}"]
                        cell["exceed"] += d > float(q)
                        cell["n"] += 1
                        resid = d - float(q)
                        cell["pinball"] += 0.95 * max(resid, 0) + 0.05 * max(-resid, 0)
                        cell["episodes"].add(uid)
    for ep_key, run in current_run.items():
        if run:
            per_arm[ep_key[0]]["runs"].append(run)

    out = {}
    miss_ms = ledger.miss_ms if ledger is not None else None
    for yaml_id in sorted(set(episodes) | set(per_arm) | set(infra)):
        eps = episodes.get(yaml_id, {})
        a = per_arm.get(yaml_id) or arm(yaml_id)
        if ledger is not None:
            a["spend"] += sum(ledger.episode_end_ms(online=e.get("online", False), update_enabled=e.get("update_enabled", True)) for e in eps.values())
        n_dec = sum(a["counts"].values())
        warm_keys = sorted(k for k in a["counts"] if k.startswith("WARM_START@"))
        runs = np.array(a["runs"]) if a["runs"] else np.array([0])
        n_infra = infra.get(yaml_id, 0)
        incomplete_reasons = []
        if n_infra:
            incomplete_reasons.append(f"{n_infra} infrastructure-failed episode(s)")
        if a["flow_invalid"]:
            incomplete_reasons.append("online stream marked flow_invalid")
        if a["pool_violations"]:
            incomplete_reasons.append(f"{a['pool_violations']} decision(s) outside the declared init pool")
        tail = {}
        for cell, v in a["tail"].items():
            ok = v["n"] >= MIN_CELL_ROWS and len(v["episodes"]) >= MIN_CELL_EPISODES
            tail[cell] = {"n": v["n"], "n_episodes": len(v["episodes"]), "E": (v["exceed"] / v["n"] if ok else None), "pinball_mean": (v["pinball"] / v["n"] if ok else None)}
        out[yaml_id] = {
            "n_ep": len(eps),
            "n_infra_failed": n_infra,
            "incomplete": bool(incomplete_reasons),
            "incomplete_reasons": incomplete_reasons,
            "success_rate": sum(e["success"] for e in eps.values()) / len(eps) if eps else None,
            "decisions": n_dec,
            "counts": {"FULL_HIT": a["counts"].get("FULL_HIT", 0), "WARM_START": sum(a["counts"][k] for k in warm_keys), "MISS": a["counts"].get("MISS", 0), **{k: a["counts"][k] for k in warm_keys}},
            "ir_percent": 100.0 * a["spend"] / (n_dec * miss_ms) if ledger is not None and n_dec else None,
            "ir_percent_no_fb": 100.0 * a["spend_no_fb"] / (n_dec * miss_ms) if ledger is not None and n_dec else None,
            "fb_batches": dict(a["fb_batches"]),
            "delta": a["delta"],
            "violation": {f"{t}": {"n": n, "rate": (k / n if n else None), "wilson_upper_descriptive": _wilson_upper(k, n)} for t, (k, n) in a["viol"].items()},
            "tail_rate": tail,
            "unavailable": dict(a["unavailable"]),
            "shadowed": dict(a["shadowed"]),
            "rejected": a["rejected"],
            "invalid_reasons": a["invalid_reasons"][:20],
            "learned_batches": a["learned"],
            "warm_run_len": {"median": float(np.median(runs)), "max": int(runs.max()), "share_ge_6": float(np.mean(runs >= 6))},
            "n_updates_final": max(a["n_updates_trace"]) if a["n_updates_trace"] else None,
            "episodes": dict(sorted(eps.items())),
            "dynamics": sorted(a["dynamics"], key=lambda r: (r.get("server_instance_id") or "", r.get("update_seq") or 0)),
            "first_finite_cut": next((r for r in a["dynamics"] if any(v is not None for v in (r.get("cuts") or {}).values())), None),
        }
    return out


def paired_bootstrap(a: dict[str, dict], b: dict[str, dict], *, n_boot: int = 1000, seed: int = 0) -> dict:
    """Bootstrap whole paired episodes within tasks, keyed by verified original init identity.

    Describes these realised trajectories, not uncertainty in the shared learner
    or arrival order. Risk rates are ratios of resampled counts, not means of rates.
    """
    fields = ("suite", "parent_pool_sha256", "task_id", "orig_init_state_idx")

    def index(episodes):
        result = {}
        for uid, rec in episodes.items():
            if not isinstance(rec, dict) or any(rec.get(f) is None for f in fields):
                raise SystemExit(f"episode {uid} lacks original init identity for pairing")
            key = tuple(rec[f] for f in fields)
            if key in result:
                raise SystemExit(f"duplicate original init in pairing: {key}")
            result[key] = rec
        return result

    ka, kb = index(a), index(b)
    if ka and kb and {k[:2] for k in ka} != {k[:2] for k in kb}:
        raise SystemExit("paired arms use different suites or parent pools")
    shared = sorted(set(ka) & set(kb))
    if not shared:
        return {"n_shared": 0, "diff": None, "ci95": None, "risk": {}}
    by_task = collections.defaultdict(list)
    for key in shared:
        by_task[key[:3]].append(key)
    tiers = sorted({t for k in shared for rec in (ka[k], kb[k]) for t in rec.get("violation", {})})

    def risk(records, tier):
        counts = [v for rec in records for t, v in rec.get("violation", {}).items() if tier == "all" or t == tier]
        n = sum(v["n"] for v in counts)
        return sum(v["k"] for v in counts) / n if n else None

    def stats(keys):
        ra, rb = [ka[k] for k in keys], [kb[k] for k in keys]
        result = {"sr": float(np.mean([r["success"] for r in rb]) - np.mean([r["success"] for r in ra]))}
        for tier in ["all", *tiers]:
            x, y = risk(ra, tier), risk(rb, tier)
            result[tier] = y - x if x is not None and y is not None else None
        return result

    point = stats(shared)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        keys = [group[int(i)] for group in by_task.values() for i in rng.integers(len(group), size=len(group))]
        boots.append(stats(keys))

    def interval(name):
        vals = [x[name] for x in boots if x[name] is not None]
        return {"diff": point[name], "ci95": [float(x) for x in np.quantile(vals, [0.025, 0.975])] if vals else None,
                "n_boot_valid": len(vals)}

    return {"n_shared": len(shared), "n_tasks": len(by_task), **interval("sr"),
            "risk": {tier: interval(tier) for tier in ["all", *tiers]}, "n_boot": n_boot,
            "note": "whole-episode init-sampling only; shared learning and arrival-order uncertainty excluded"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ledger", default="", help="cost json with fb_batch_ms; omit for null IR")
    ap.add_argument("--knots", default="", help="knots.json carrying the frozen score_bands")
    ap.add_argument("--pool-manifest", default="", help="init_pools_manifest.json of a subset run")
    ap.add_argument("--pool-key", default="", help="adapt | terminal")
    ap.add_argument("--pair", default="", help="A,B : paired bootstrap of SR(B)-SR(A)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ledger = load_ledger(args.ledger) if args.ledger else None
    bands = json.loads(pathlib.Path(args.knots).read_text(encoding="utf-8")).get("score_bands") if args.knots else None
    manifest = json.loads(pathlib.Path(args.pool_manifest).read_text(encoding="utf-8")) if args.pool_manifest else None
    result = aggregate(pathlib.Path(args.data_dir), ledger, score_bands=bands, pool_manifest=manifest, pool_key=args.pool_key or None)
    if args.pair:
        a_id, b_id = args.pair.split(",")
        result["_paired"] = {f"{a_id}->{b_id}": paired_bootstrap(result[a_id]["episodes"], result[b_id]["episodes"])}
    write_json(args.out, result)
    for yaml_id, r in sorted(result.items()):
        if yaml_id.startswith("_"):
            continue
        ir = "   -" if r["ir_percent"] is None else f"{r['ir_percent']:5.1f}"
        sr = "  -  " if r["success_rate"] is None else f"{r['success_rate']:.3f}"
        flag = " INCOMPLETE" if r["incomplete"] else ""
        print(f"  {yaml_id:32s} n={r['n_ep']:4d} sr={sr} ir={ir} WS={r['counts']['WARM_START']:6d} MISS={r['counts']['MISS']:6d} learned={r['learned_batches']}{flag}")


if __name__ == "__main__":
    main()
