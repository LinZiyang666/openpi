"""Read-only, CPU-only incremental analysis of CACHE_STUDY.md sections 5 and 9.

Explicit lanes and their resume ancestors contribute accepted terminal outcomes,
regardless of receipt status. Incomplete cells and cost evidence are reported.
No model, campaign, simulator, or GPU imports/actions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
SIZES = (50, 100, 200, 300, 400, 500)
SUITES = ("libero_spatial", "libero_10")
GROUPS = ("c", "cr", "ch", "chr", "cg", "crg", "chg", "chrg")
ALIASES = {"ch": "c", "chr": "cr", "chg": "cg", "chrg": "crg"}
MODELS = ("DP", "Fast-WAM", "Cosmos")
MODEL_IDS = {"dp": "DP", "fastwam": "Fast-WAM", "cosmos": "Cosmos", "pi05": "π0.5", "groot": "GR00T"}
FAMILY, BOOT, SEED = 40, 10_000, 20261010
CONTROL_NAMES = {
    "DP": "hs5dp_ctl_142812", "Fast-WAM": "hs6fw_res_152931",
    "Cosmos": "formal_cosmos_libero_official2000_20261010_130639_a8ca2043",
}


class Sources:
    def __init__(self):
        self.files = set()
        self.missing = set()
        self.incomplete_lines = Counter()

    def obj(self, path):
        path = Path(path)
        if not path.exists():
            self.missing.add(str(path))
            return {}
        self.files.add(str(path))
        return json.loads(path.read_bytes())

    def rows(self, path):
        path = Path(path)
        if not path.exists():
            self.missing.add(str(path))
            return
        self.files.add(str(path))
        # Bound each read to the current file length while a lane is appending.
        with path.open("rb") as f:
            remaining = path.stat().st_size
            while remaining:
                line = f.readline(remaining)
                if not line:
                    break
                remaining -= len(line)
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    if remaining or line.endswith(b"\n"):
                        raise
                    self.incomplete_lines[str(path)] += 1


def receipt_snapshot(run):
    out = dict(lane=str(run), observed_utc=datetime.now(timezone.utc).isoformat(),
               receipt=None, statuses={})
    for name in ("RECEIPT.json", "RECOVERED_RECEIPT.json"):
        p = run / name
        if not p.exists():
            continue
        try:
            r = json.loads(p.read_bytes())
        except json.JSONDecodeError:
            out["statuses"][name] = {"status": "being_written"}
            continue
        out["statuses"][name] = dict(status=r.get("status"), closed=r.get("all_owned_closed"))
        out["receipt"] = name
    out["status"] = out["statuses"].get(out["receipt"], {}).get("status") or "NO_RECEIPT"
    return out


def load_exclusions():
    with (HERE / "CONTAMINATION_DISTANCES.csv").open(newline="") as f:
        return {(r["suite"], r["task"], int(r["a_init"]))
                for r in csv.DictReader(f) if r["flag"] == "True"}


def physical_groups(model):
    return tuple(g for g in GROUPS if model != "Cosmos" or g not in ALIASES)


def canonical(task):
    suite = "mt50" if task["experiment"] == "metaworld_mt50" else task["experiment"]
    return suite, task["extra"]["task_name"], task["orig_init_state_idx"]


def resume_segments(run, sources):
    """Oldest first; resume selection/pins do not filter accepted outcomes."""
    segments, seen = [], set()
    while run not in seen:
        seen.add(run)
        campaign = sources.obj(run / "campaign.json")
        segments.append((run, campaign))
        selection = campaign.get("cache_resume_selection") or {}
        parent = selection.get("source") or campaign.get("resume_from")
        if not parent and (run / "CACHE_RESUME.json").exists():
            parent = sources.obj(run / "CACHE_RESUME.json").get("source")
        if not parent:
            break
        parent = Path(parent)
        run = (parent if parent.is_absolute() else run / parent).resolve()
    return list(reversed(segments))


def evidence_rows(segments, name, sources):
    for run, _ in segments:
        for row in sources.rows(run / "driver" / name):
            row["_segment"] = str(run)
            yield row


def timestamp(row):
    return row.get("ts", row.get("time", 0)) or 0


def infrastructure_error(row):
    error = row.get('error') or ''
    return 'FatalEpisodeError' in error and 'retry budget exhausted' in error


def accepted_terminal(rows, diagnostics=None, infra_errors=None):
    out = {}
    for r in rows:
        if r.get("accepted") is not True:
            continue
        uid = r["task_uid"]
        if infrastructure_error(r):
            if diagnostics is not None:diagnostics['infrastructure_errors'] += 1
            if infra_errors is not None:infra_errors.add(uid)
            continue
        if uid in out:
            if diagnostics is not None:
                diagnostics["duplicates"] += 1
                diagnostics["conflicting_outcomes"] += out[uid].get("success") != r.get("success")
            # Merged control journals copy ancestor rows with the same ts;
            # preserve their original segment so its step evidence is usable.
            if timestamp(r) <= timestamp(out[uid]):
                continue
        out[uid] = r
    return out


def finish_episode(d):
    decisions, acks = d["decisions"], d["acks"]
    totals = Counter(); scopes = Counter()
    for seq in sorted(decisions):
        r, ack = decisions[seq], acks[seq]
        if r.get("actual_nfe") is None or ack.get("executed_count") is None:
            return
        totals[r["source"]] += 1; totals["requests"] += 1
        totals["nfe"] += r["actual_nfe"]; totals["controls"] += ack["executed_count"]
        for field, value in (("host_ms", r.get("server_timing", {}).get("infer_ms")),
                             ("service_ms", r.get("service_ms")), ("queue_ms", r.get("queue_wait_ms"))):
            if value is not None:
                totals[field] += value; totals[field + "_present"] += 1
        totals["pricing_true"] += r.get("pricing_eligible") is True
        scopes[str(r.get("service_timing_scope"))] += 1
    d["cost"] = dict(totals); d["service_scopes"] = dict(scopes)
    del d["decisions"], d["acks"], d["timing"]


def maybe_finish(d):
    if "cost" in d:
        return
    n = d["timing"].get("infers") if d["timing"] else None
    if n is None:
        n = next((seq + 1 for seq, ack in d["acks"].items() if ack.get("terminal")), None)
    if n and set(d["decisions"]) == set(d["acks"]) == set(range(n)):
        finish_episode(d)


def load_lane(run, snapshot, sources, sizes, suites=SUITES):
    segments = resume_segments(run, sources)
    campaign = segments[-1][1]
    model = MODEL_IDS[campaign["model"]]
    groups, tasks = {}, {}
    for segment, _ in segments:
        manifest_path = segment / "assets/CACHE_STUDY.json"
        manifest = sources.obj(manifest_path) if manifest_path.exists() else {}
        for g in manifest.get("groups", []):
            groups[g.get("library_size", 500), g["suite"], g["group"]] = g
        plan_path = segment / "FULL_EPISODE_PLAN.json"
        if not plan_path.exists():
            plan_path = segment / "EPISODE_PLAN.json"
        tasks.update((t["task_uid"], t) for t in sources.obj(plan_path).get("tasks", []))
    cells, task_cells = defaultdict(dict), {}
    for uid, t in tasks.items():
        if canonical(t)[0] not in suites:
            continue
        key = canonical(t); ex = t["extra"]
        g = ex.get("cache_group", "control")
        n = ex.get("library_size", 500) if g != "control" else 0
        if g != "control" and n not in sizes:
            continue
        if model == "Cosmos":
            g = ALIASES.get(g, g)
        cell = model, n, g, key[0]
        cells[cell]
        task_cells[uid] = cell, key
    diagnostic = {name: Counter() for name in ("journal", "attempts")}
    infra_errors = set()
    journal = accepted_terminal(evidence_rows(segments, "journal.jsonl", sources), diagnostic["journal"], infra_errors)
    attempts = accepted_terminal(evidence_rows(segments, "attempts.jsonl", sources), diagnostic["attempts"])
    by_uid = {}
    for uid, r in journal.items():
        if uid not in task_cells:
            continue
        cell, key = task_cells[uid]
        old = cells[cell].get(key)
        if old and timestamp(r) < old["ts"]:
            continue
        attempt = attempts.get(uid, {})
        d = dict(key=key, success=r["success"], attempt=r.get("attempt"), ts=timestamp(r),
                 segment=r["_segment"], worker=attempt.get("worker_id", ""),
                 duration_s=attempt.get("duration_s"), decisions={}, acks={}, timing=None)
        cells[cell][key] = d
        by_uid[uid] = d
    step_duplicates = 0
    for row in evidence_rows(segments, "per_step.jsonl", sources):
        uid = row.get("task_uid")
        if row.get("accepted") is not True or uid not in by_uid:
            continue
        d = by_uid[uid]
        if row.get("attempt") != d["attempt"] or row["_segment"] != d["segment"]:
            continue
        if "cost" in d:
            step_duplicates += 1
            continue
        kind = row.get("_kind") or "decision"
        if kind == "client_timing":
            d["timing"] = {"infers": row["infers"]}
        elif kind in ("decision", "execution_ack"):
            bucket = d["decisions"] if kind == "decision" else d["acks"]
            seq = row["decision_seq"]
            if seq in bucket:
                step_duplicates += 1
                if timestamp(row) < timestamp(bucket[seq]):
                    continue
            fields = ("source", "actual_nfe", "server_timing", "service_ms", "queue_wait_ms",
                      "pricing_eligible", "service_timing_scope", "ts", "time") if kind == "decision" else (
                      "executed_count", "terminal", "ts", "time")
            bucket[seq] = {k: row[k] for k in fields if k in row}
        maybe_finish(d)
    for d in by_uid.values():
        for field in ("decisions", "acks", "timing"):
            d.pop(field, None)
    snapshot["segments"] = [receipt_snapshot(p) for p, _ in segments]
    snapshot["accepted_episodes"] = len(by_uid)
    snapshot["cost_episodes"] = sum("cost" in d for d in by_uid.values())
    snapshot["missing_cost_episodes"] = snapshot["accepted_episodes"] - snapshot["cost_episodes"]
    snapshot["terminal_duplicates"] = {k: dict(v) for k, v in diagnostic.items()}
    snapshot["infrastructure_error_task_uids"] = sorted(infra_errors & task_cells.keys())
    snapshot["missing_infrastructure_error_task_uids"] = sorted((infra_errors & task_cells.keys()) - journal.keys())
    snapshot["duplicate_step_rows"] = step_duplicates
    result = {}
    for cell, data in cells.items():
        bundle = dict(groups.get((cell[1], cell[3], cell[2]), {}))
        # Calibration metadata is descriptive; no view/fit or pin validation.
        if bundle.get("stall_rel"):
            stall_dir = run / "assets/cache_inputs" / bundle["stall_rel"]
            metadata = sources.obj(stall_dir / "READY.json") if (stall_dir / "READY.json").exists() else {}
            calibration = sources.obj(stall_dir / "stall.json") if (stall_dir / "stall.json").exists() else {}
            bundle["analysis_stall_unavailable_tasks"] = metadata.get("unavailable_tasks", [])
            bundle["analysis_stall_task_calibration"] = calibration.get("tasks", {})
        result[cell] = dict(data=data, lane=str(run), receipt=snapshot["receipt"],
                            receipt_status=snapshot["status"],
                            treatment={k: campaign.get(k) for k in ("server_node", "device", "replicas", "mps",
                                "dp_batching", "fastwam_batching", "worker_nodes", "workers")}, bundle=bundle)
    return result


def cp_lower(g, loss, n, family=FAMILY):
    alpha = .05 / (2 * family)
    return float((beta.ppf(alpha, g, n - g + 1) if g else 0)
                 - (beta.ppf(1 - alpha, loss + 1, n - loss) if loss < n else 1))


def paired(keys, ref, arm):
    keys = sorted(set(keys) & ref.keys() & arm.keys()); n = len(keys)
    if not n:
        return dict(n=0, status="PENDING_PAIRS")
    g = sum(arm[k]["success"] and not ref[k]["success"] for k in keys)
    loss = sum(ref[k]["success"] and not arm[k]["success"] for k in keys)
    rng = np.random.default_rng(SEED); draws = np.zeros(BOOT, dtype=np.int64)
    for task in sorted({k[1] for k in keys}):
        ks = [k for k in keys if k[1] == task]
        delta = np.array([int(arm[k]["success"]) - int(ref[k]["success"]) for k in ks])
        draws += delta[rng.integers(len(ks), size=(BOOT, len(ks)))].sum(axis=1)
    r = dict(n=n, gains=g, losses=loss, diff=(g-loss)/n,
             cp_lb_family=cp_lower(g, loss, n), cp_lb_pointwise=cp_lower(g, loss, n, 1),
             bootstrap_lb_family=float(np.quantile(draws/n, .05/FAMILY)),
             bootstrap_lb_pointwise=float(np.quantile(draws/n, .05)))
    r["ni_family"] = min(r["cp_lb_family"], r["bootstrap_lb_family"]) > -.02
    r["ni_pointwise"] = min(r["cp_lb_pointwise"], r["bootstrap_lb_pointwise"]) > -.02
    return r


def cost(keys, data):
    available = [k for k in keys if "cost" in data[k]]
    total = Counter()
    for k in available:
        total.update(data[k]["cost"])
    t, hit, miss = total["controls"], total["library"], total["policy"]
    fresh = hit + miss
    def ratio(a, b):
        return a/b if b else None
    r = dict(total, episodes=len(available), missing_episodes=len(keys)-len(available),
             fresh_decisions=fresh, fresh_hit_rate=ratio(hit, fresh),
             miss_per_control=ratio(miss, t), nfe_per_control=ratio(total["nfe"], t),
             calls_per_episode=ratio(miss, len(available)), nfe_per_fresh_decision=ratio(total["nfe"], fresh),
             requests_per_episode=ratio(total["requests"], len(available)),
             nfe_per_request=ratio(total["nfe"], total["requests"]), nfe_per_episode=ratio(total["nfe"], len(available)))
    complete = bool(available) and total["service_ms_present"] == total["queue_ms_present"] == total["pricing_true"] == total["requests"]
    r["service_complete"] = complete
    for name in ("service_ms", "queue_ms", "host_ms"):
        r[name + "_per_control"] = ratio(total[name], t) if available and total[name+"_present"] == total["requests"] else None
    # An available host timer is not an eligible isolated GPU service timer.
    if not complete:
        r["service_ms_per_control"] = None
    return r


def sr(keys, data):
    n = len(keys); s = sum(data[k]["success"] for k in keys)
    if not n:
        return dict(n=0, successes=0, sr=None, cp_lo=None, cp_hi=None)
    return dict(n=n, successes=s, sr=s/n, cp_lo=float(beta.ppf(.025, s, n-s+1)) if s else 0.,
                cp_hi=float(beta.ppf(.975, s+1, n-s)) if s < n else 1.)


def analyze(cells, flags):
    rows = []
    for (model, n, group, suite), cell in sorted(cells.items()):
        if group == "control":
            continue
        for cohort in ("full", "sensitivity"):
            data = cell["data"]; keys = sorted(k for k in data if cohort == "full" or k not in flags)
            out = dict(model=model, size=n, group=group, suite=suite, cohort=cohort, lane=cell["lane"],
                       status="COMPLETE" if len(data) >= 500 else "INTERIM", receipt_status=cell["receipt_status"],
                       sr=sr(keys, data), cost=cost(keys, data), comparisons={},
                       stall_unavailable_tasks=cell["bundle"].get("analysis_stall_unavailable_tasks", []),
                       stall_task_calibration=cell["bundle"].get("analysis_stall_task_calibration", {}))
            references = [("control", (model, 0, "control", suite), "F500" if n == 500 else None),
                          ("N500", (model, 500, group, suite), "Fsize" if n == 50 else None)]
            for knob, letter in (("corrector", "r"), ("long", "h"), ("gate", "g")):
                if letter in group:
                    references.append((knob+"_ablation", (model, n, group.replace(letter, ""), suite), None))
            for name, target, family in references:
                if name == "N500" and n == 500:
                    continue
                if target not in cells:
                    out["comparisons"][name] = dict(status="PENDING_REFERENCE", family=family)
                    continue
                ref = cells[target]
                pair_keys = sorted(set(keys) & ref["data"].keys())
                stats = paired(pair_keys, ref["data"], data)
                stats.update(family=family, primary=bool(family and cohort == "full"))
                if not pair_keys:
                    out["comparisons"][name] = stats
                    continue
                expected_pairs = 500 - (sum(k[0] == suite for k in flags) if cohort == "sensitivity" else 0)
                stats["status"] = "PAIRED" if len(data) >= 500 and len(ref["data"]) >= 500 and len(pair_keys) == expected_pairs else "INTERIM"
                cost_keys = [k for k in pair_keys if "cost" in data[k] and "cost" in ref["data"][k]]
                ac, rcost = cost(cost_keys, data), cost(cost_keys, ref["data"])
                stats["cost_pairs"] = len(cost_keys)
                stats["nfe_per_control_reduction"] = 1-ac["nfe_per_control"]/rcost["nfe_per_control"] if rcost["nfe_per_control"] else None
                stats["miss_per_control_reduction"] = 1-ac["miss_per_control"]/rcost["miss_per_control"] if rcost["miss_per_control"] else None
                stats["treatment_differences"] = {k: [ref["treatment"].get(k), cell["treatment"].get(k)]
                    for k in set(ref["treatment"]) | set(cell["treatment"]) if ref["treatment"].get(k) != cell["treatment"].get(k)}
                scopes, ref_scopes = set(), set()
                for k in cost_keys:
                    scopes.update(data[k]["service_scopes"])
                    ref_scopes.update(ref["data"][k]["service_scopes"])
                stats["service_comparable"] = (not stats["treatment_differences"] and ac["service_complete"]
                                                 and rcost["service_complete"] and scopes == ref_scopes
                                                 and "None" not in scopes)
                stats["service_per_control_reduction"] = (1-ac["service_ms_per_control"]/rcost["service_ms_per_control"]
                    if stats["service_comparable"] and rcost["service_ms_per_control"] else None)
                if model == "Cosmos" and name == "control":
                    stats["renderer_subgroups"] = {}
                    for node in ("timan107", "timan108"):
                        sub = [k for k in pair_keys if f"_{node}_" in ref["data"][k]["worker"]]
                        if sub:
                            stats["renderer_subgroups"][node] = dict(sr=sr(sub, data), control_sr=sr(sub, ref["data"]),
                                                                    paired=paired(sub, ref["data"], data))
                out["comparisons"][name] = stats
            rows.append(out)
    return rows


def claim_table(rows):
    out = []
    for family, size, ref in (("F500", 500, "control"), ("Fsize", 50, "N500")):
        for model in MODELS:
            for group in physical_groups(model):
                selected = [r for r in rows if (r["model"], r["size"], r["group"], r["cohort"])
                            == (model, size, group, "full")]
                comparisons = [r["comparisons"].get(ref, {}) for r in selected]
                complete = len(selected) == 2 and all(r.get("status") == "PAIRED" for r in comparisons)
                ni = complete and all(r["ni_family"] for r in comparisons)
                cost_pass = complete and all((r.get("service_per_control_reduction") or 0) > 0 for r in comparisons)
                pending = "INTERIM" if selected and any(r.get("n", 0) for r in comparisons) else "PENDING"
                out.append(dict(family=family, model=model, group=group, both_suites=pending if not complete else
                                ("NI_PASS" if ni else "NI_NOT_ESTABLISHED"),
                                adoption=pending if not complete else
                                ("NI_AND_MEASURED_SERVICE_SAVING" if ni and cost_pass and family == "F500" else
                                 "SIZE_NI_ONLY" if ni and family == "Fsize" else
                                 "NI_PASS_SERVICE_COST_PENDING" if ni else "NOT_ESTABLISHED")))
    return out


def pct(value):
    return "—" if value is None else f"{value*100:.2f}"


def number(value, precision=5):
    return "—" if value is None else f"{value:.{precision}f}"


def write_report(outdir, report):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "CACHE_RESULTS.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    lines = ["# R17 cache study: accepted episode snapshot", "", f"UTC: {report['as_of_utc']}. Selected sizes: {report['allowed_sizes']}.",
             "", "F500 与 Fsize 各固定40比较/5%；不随已到达 lane 缩小族，不声称合并族5%。中间尺寸与敏感性均为描述性。",
             "Cosmos ch/chr/chg/chrg 分别复用 c/cr/cg/crg。CI为逐点双侧95% CP；差值LB为单侧。",
             "Pairs use canonical task/init keys; only common keys contribute. The fixed 32-init sensitivity exclusion leaves spatial489/10:479 when complete.",
             "Receipt status is information only. Cells with fewer than 500 accepted episodes are INTERIM; missing outcomes are never imputed.", "",
             "## Lane receipts and evidence", "", "| Lane | Receipt status | Receipts | accepted | cost episodes | missing cost |", "|---|---|---|---:|---:|---:|"]
    for s in report["lanes"]:
        receipts = "; ".join(f"{k}:{v['status']}" for k, v in s.get("statuses", {}).items()) or "未出现"
        lines.append(f"| {s['lane']} | {s['status']} | {receipts} | {s.get('accepted_episodes', 0)} | {s.get('cost_episodes', 0)} | {s.get('missing_cost_episodes', 0)} |")
    available = sum(r['cohort'] == 'full' and r['sr']['n'] > 0 for r in report['rows'])
    complete = sum(r['cohort'] == 'full' and r['status'] == 'COMPLETE' for r in report['rows'])
    lines += ["", f"{available}/240 cells have accepted outcomes; {complete}/240 have all 500 episodes.",
              "", "## N500 vs inference control (full cohort)", "",
              "| model | group | suite | n | SR% | pairs | Δ pp | g/l | F500 CP LB pp | status |",
              "|---|---|---|---:|---:|---:|---:|---|---:|---|"]
    for r in report['rows']:
        if r['size'] != 500 or r['cohort'] != 'full':
            continue
        c = r['comparisons']['control']
        lines.append(f"| {r['model']} | {r['group']} | {r['suite']} | {r['sr']['n']} | {pct(r['sr']['sr'])} | {c.get('n', 0)} | {pct(c.get('diff'))} | {c.get('gains', '—')}/{c.get('losses', '—')} | {pct(c.get('cp_lb_family'))} | {r['status']} |")
    lines += ["", "## Sizes 50–400 so far (full cohort)", "",
              "| model | group | suite | N50 n / SR% | N100 n / SR% | N200 n / SR% | N300 n / SR% | N400 n / SR% |",
              "|---|---|---|---|---|---|---|---|"]
    index = {(r['model'], r['size'], r['group'], r['suite']): r for r in report['rows'] if r['cohort'] == 'full'}
    for model in MODELS:
        for group in physical_groups(model):
            for suite in SUITES:
                if not any((model, n, group, suite) in index for n in SIZES[:-1]):
                    continue
                values = []
                for n in SIZES[:-1]:
                    r = index.get((model, n, group, suite))
                    values.append(f"{r['sr']['n']} / {pct(r['sr']['sr'])}" if r else "0 / —")
                lines.append(f"| {model} | {group} | {suite} | " + " | ".join(values) + " |")
    lines += ["", "## SR and costs", "", "| model | N | group | suite | cohort | status | n | SR% [CP95%] | cost n | fresh hit% | MISS/control | NFE/control | calls/ep |",
              "|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for r in report["rows"]:
        s, c = r["sr"], r["cost"]
        lines.append(f"| {r['model']} | {r['size']} | {r['group']} | {r['suite']} | {r['cohort']} | {r['status']} | {s['n']} | {pct(s['sr'])} [{pct(s['cp_lo'])},{pct(s['cp_hi'])}] | {c['episodes']} | {pct(c['fresh_hit_rate'])} | {number(c['miss_per_control'])} | {number(c['nfe_per_control'])} | {number(c['calls_per_episode'], 2)} |")
    lines += ["", "## Paired comparisons (Δ/LB in pp)", "", "| model/N/group/suite | cohort | reference | family | status | pairs | cost pairs | Δ | g/l | CP LB family | boot LB family | CP/boot LB pointwise | NI |",
              "|---|---|---|---|---|---:|---:|---:|---|---:|---:|---|---|"]
    flat = []
    for r in report["rows"]:
        for ref, c in r["comparisons"].items():
            label = f"{r['model']}/{r['size']}/{r['group']}/{r['suite']}"
            primary = c.get("primary", False)
            if c.get("n", 0):
                ni = "INTERIM" if c["status"] == "INTERIM" else (("PASS" if c["ni_family"] else "未证明") if primary else "描述性")
                lines.append(f"| {label} | {r['cohort']} | {ref} | {c['family'] or '—'} | {c['status']} | {c['n']} | {c['cost_pairs']} | {pct(c['diff'])} | {c['gains']}/{c['losses']} | {pct(c['cp_lb_family']) if c['family'] else '—'} | {pct(c['bootstrap_lb_family']) if c['family'] else '—'} | {pct(c['cp_lb_pointwise'])}/{pct(c['bootstrap_lb_pointwise'])} | {ni} |")
            else:
                lines.append(f"| {label} | {r['cohort']} | {ref} | {c.get('family') or '—'} | {c['status']} | 0 | 0 | — | — | — | — | — | PENDING |")
            record = {k: r[k] for k in ("model", "size", "group", "suite", "cohort", "lane")}
            record.update(r["sr"]); record.update({"cost_"+k: v for k, v in r["cost"].items()})
            record.update(cell_status=r['status'], receipt_status=r['receipt_status'], reference=ref,
                          **{('pair_n' if k == 'n' else k): v for k, v in c.items() if not isinstance(v, (dict, list))})
            flat.append(record)
    lines += ["", "## 两套件主判定", "", "| family | model | group | NI（两套件） | 成本/采用 |", "|---|---|---|---|---|"]
    lines += [f"| {c['family']} | {c['model']} | {c['group']} | {c['both_suites']} | {c['adoption']} |" for c in report["claims"]]
    lines += ["", "## 限制与待到达", "", "NFE是动作头计算量，不含视觉/检索/修正器；零NFE不等于零耗时。service无完整可比证据时不宣称墙钟加速。",
              "Cosmos旧控制含timan108与timan107 renderer；JSON按旧控制节点分层，不能把跨节点差异解释为纯缓存因果效应。",
              "Q3实际校准来自各尺寸B成功episode LOEO，不是独立B验证录制；门不可用任务在JSON逐组列出。",
              "", "Missing cells, resume segments, duplicate counts and missing evidence are listed in CACHE_RESULTS.json; CSV includes paired counts and raw cost totals. Curves leave missing outcomes blank.",
              "This CPU analysis performs no GPU/model/campaign operations. Input files are read without hashes or seals."]
    (outdir / "CACHE_RESULTS.md").write_text("\n".join(lines) + "\n")
    fields = sorted({k for row in flat for k in row}) or ["model", "size", "group", "suite", "cohort", "status"]
    with (outdir / "CACHE_RESULTS.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(flat)
    plot(report, outdir)


def plot(report, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for cohort in ("full", "sensitivity"):
        for metric in ("sr", "fresh_hit_rate", "miss_per_control", "nfe_per_control"):
            fig, axes = plt.subplots(3, 2, figsize=(13, 12), squeeze=False)
            for i, model in enumerate(MODELS):
                for j, suite in enumerate(SUITES):
                    ax = axes[i, j]; ax.set_title(f"{model} / {suite}")
                    for group in physical_groups(model):
                        if metric in ("miss_per_control", "nfe_per_control") and "g" not in group:
                            continue
                        selected = {r["size"]: r for r in report["rows"] if (r["model"], r["suite"], r["group"], r["cohort"])
                                    == (model, suite, group, cohort) and r["sr"]["n"]}
                        y = [(selected[n]["sr"]["sr"] if metric == "sr" else selected[n]["cost"][metric])
                             if n in selected else np.nan for n in SIZES]
                        y = [np.nan if value is None else value for value in y]
                        line, = ax.plot(SIZES, y, "o-", label=group)
                        if metric == "sr" and selected:
                            xs = sorted(selected); mean = np.array([selected[n]["sr"]["sr"] for n in xs])
                            lo = np.array([selected[n]["sr"]["cp_lo"] for n in xs]); hi = np.array([selected[n]["sr"]["cp_hi"] for n in xs])
                            ax.errorbar(xs, mean, yerr=[mean-lo, hi-mean], fmt="none", color=line.get_color(), alpha=.5)
                    ctl = report["controls"].get(f"{model}/{suite}/{cohort}")
                    if metric == "sr" and ctl:
                        ax.axhline(ctl["sr"], color="black", ls="--", label="inference control")
                    ax.set_xticks(SIZES); ax.set_xlabel("B library episodes N"); ax.set_ylabel(metric)
                    if metric in ("sr", "fresh_hit_rate"):
                        ax.set_ylim(0, 1.02)
                    ax.grid(alpha=.2); ax.legend(fontsize=7)
            fig.suptitle(f"R17 cache: {metric}, {cohort}; accepted outcomes, including interim")
            fig.text(.5, .005, "Cosmos aliases: ch=c, chr=cr, chg=cg, chrg=crg", ha="center", fontsize=8)
            fig.tight_layout(); fig.savefig(outdir / f"CACHE_{metric.upper()}_{cohort}.svg"); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lane", type=Path, action="append", required=True, help="Explicit lane; repeat for sizes/resumes")
    p.add_argument("--sizes", type=int, nargs="+", default=[500], choices=SIZES)
    p.add_argument("--eval-root", type=Path, default=Path("/data/mw_finetune/eval"), help="Location of default controls")
    p.add_argument("--out-dir", type=Path, default=HERE)
    args = p.parse_args(argv)
    # Do not allow an accidental output path in the data/campaign tree.
    if args.out_dir.resolve().is_relative_to(Path("/data")):
        p.error("Analysis output must stay outside /data")
    sources = Sources(); flags = load_exclusions()
    cells, snapshots = {}, []
    runs = list(dict.fromkeys(x.resolve() for x in args.lane))
    # Pure-inference lanes can also be supplied directly with --lane.
    for model in MODELS:
        if not any(MODEL_IDS[sources.obj(run / "campaign.json")["model"]] == model
                   and not (run / "assets/CACHE_STUDY.json").exists() for run in runs):
            control = args.eval_root / CONTROL_NAMES[model]
            if control.exists():
                runs.append(control.resolve())
    for run in runs:
        snap = receipt_snapshot(run); snapshots.append(snap)
        print(f"Reading {run.name}", flush=True)
        loaded = load_lane(run, snap, sources, args.sizes)
        print(f"  accepted={snap['accepted_episodes']} cost={snap['cost_episodes']} receipt={snap['status']}", flush=True)
        for cell, incoming in loaded.items():
            if cell not in cells:
                cells[cell] = incoming
                continue
            current = cells[cell]
            for key, row in incoming["data"].items():
                if key not in current["data"] or row["ts"] >= current["data"][key]["ts"]:
                    current["data"][key] = row
            current["lane"] = "; ".join(dict.fromkeys([current["lane"], incoming["lane"]]))
    rows = analyze(cells, flags)
    controls = {}
    for (model, n, group, suite), cell in cells.items():
        if group == "control":
            for cohort in ("full", "sensitivity"):
                ks = [k for k in cell["data"] if cohort == "full" or k not in flags]
                controls[f"{model}/{suite}/{cohort}"] = sr(ks, cell["data"])
    expected = [(m, n, g, s) for m in MODELS for n in SIZES for g in physical_groups(m) for s in SUITES]
    report = dict(schema="r17.cache_size_analysis.v1", as_of_utc=datetime.now(timezone.utc).isoformat(),
                  allowed_sizes=args.sizes, families={"F500": FAMILY, "Fsize": FAMILY}, bootstrap=BOOT, seed=SEED,
                  lanes=snapshots, rows=rows, controls=controls, claims=claim_table(rows),
                  pending_cells=[dict(model=m, size=n, group=g, suite=s) for m, n, g, s in expected
                                 if not cells.get((m,n,g,s), {}).get("data")],
                  input_files=sorted(sources.files), missing_evidence_files=sorted(sources.missing),
                  incomplete_jsonl_lines=dict(sources.incomplete_lines), exclusions=[list(k) for k in sorted(flags)],
                  cosmos_aliases=ALIASES)
    write_report(args.out_dir, report)
    print(json.dumps(dict(available_cells=sum(r["cohort"] == "full" and r["sr"]["n"] > 0 for r in rows),
                          complete_cells=sum(r["cohort"] == "full" and r["status"] == "COMPLETE" for r in rows),
                          pending_cells=len(report["pending_cells"]),
                          output=str(args.out_dir / "CACHE_RESULTS.md"), sizes=args.sizes)))


if __name__ == "__main__":
    main()
