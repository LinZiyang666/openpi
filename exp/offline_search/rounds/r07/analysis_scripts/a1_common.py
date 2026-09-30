"""R7 A1 read-only loaders, actual-work accounting, and paired statistics.

No serving/model imports. Run with the A1 CPU/environment prefix in ANALYSIS_BRIEF.
R6 identities and exact CP/McNemar definitions are reused; the R7 seed is explicit.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from exp.offline_search.rounds.r06.analysis_scripts import common as R6

HERE = Path(__file__).resolve().parent
R7 = HERE.parent
RUNS = Path(R6.ROOT)
FRONTIER = R7.parent / "r06/frontier_final"
SEED, DRAWS = 20260930, 10_000
EXPECTED = R6.EXPECTED
CELLS = [R6.c_cell(m, c) for m, c in R6.CELLS]
SPARSE = [c for c in CELLS if c.endswith("_50")]
PI05 = [c for c in CELLS if c.startswith("pi05_")]
PRICES = {"pi05": (.152, .848), "groot": (.148, .852)}


def read_json(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    with Path(path).open() as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"malformed JSON: {path}:{line_no}") from exc


def stamp(path, digest=False):
    path = Path(path)
    stat = path.stat()
    row = dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if digest:
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return row


def frozen_arms():
    """SELECTION Freeze 2, not the unpruned 32-spec arms.json."""
    rows = []
    for cell in CELLS:
        for variant in ("SF1", "UF1"):
            rows.append(dict(cell=cell, variant=variant, arm=f"r7_{cell}_{variant}"))
        if cell in PI05:
            short = cell.removeprefix("pi05_").replace("spatial_", "sp_")
            rows.append(dict(cell=cell, variant="SW", arm=f"r7_sw_pi05_{short}"))
        if cell in SPARSE:
            for variant in ("CU30", "CT30"):
                rows.append(dict(cell=cell, variant=variant, arm=f"r7_{cell}_{variant}"))
    assert len(rows) == 28
    return rows


def completion(root, arm):
    root = Path(root)
    markers = sorted((root / "state").glob(f"{arm}.manifest_*.DONE"))
    plain = root / "state" / f"{arm}.DONE"
    if plain.is_file():
        markers.append(plain)
    summary = root / "runs" / arm / "summary.json"
    return dict(status="complete" if markers and summary.is_file() else "missing",
                done_markers=[stamp(p) for p in markers], summary_present=summary.is_file())


def pair(uid):
    return tuple(int(s) for s in uid.split(":eval:", 1)[1].split(":")[:2])


def attempt_key(row):
    return row.get("task_uid"), row.get("run_id"), int(row.get("attempt", 1) or 1)


def accepted_journal(path):
    records, duplicate_rows, ignored = {}, 0, 0
    for row in jsonl(path):
        uid = row.get("task_uid", "")
        if (":eval:" not in uid or row.get("status") not in ("done", "failed")
                or row.get("accepted") is False or row.get("error") not in (None, "")):
            ignored += 1
            continue
        if row.get("success") not in (True, False, 0, 1):
            raise ValueError(f"invalid outcome: {uid}")
        key = pair(uid)
        if key in records:
            if records[key] != row:
                raise ValueError(f"multiple accepted terminal records: {uid}")
            duplicate_rows += 1
        records[key] = row
    return records, dict(accepted_pairs=len(records), exact_duplicate_rows=duplicate_rows,
                         ignored_nonaccepted_rows=ignored)


def exception_audit(client, records):
    """Same (uid, run_id, attempt) join as purge_exc.py; never execute the purge."""
    client = Path(client)
    reasons, timing, exceptions = defaultdict(set), {}, set()
    path = client / "per_step.jsonl"
    malformed_timing = 0
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if '"client_timing"' not in line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    malformed_timing += 1
                    continue
                if row.get("_kind") != "client_timing":
                    continue
                key = attempt_key(row)
                reason = row.get("termination_reason")
                if reason is not None:
                    reasons[key].add(reason)
                if reason == "exception":
                    exceptions.add(key)
                if key in timing and timing[key].get("infers") != row.get("infers"):
                    raise ValueError(f"conflicting client timing counts: {key}")
                timing[key] = row
    accepted_keys = {attempt_key(r) for r in records.values()}
    residual = accepted_keys & exceptions
    residual |= {attempt_key(r) for r in records.values() if r.get("termination_reason") == "exception"}
    uncovered = accepted_keys - reasons.keys()
    conflicting = [k for k in accepted_keys if len(reasons[k]) > 1]
    purged_path = client / "purged_exceptions.jsonl"
    purged = list(jsonl(purged_path)) if purged_path.exists() else []
    audit = dict(status="ok" if not (residual or uncovered or conflicting or malformed_timing) else "invalid",
                 per_step_present=path.exists(), accepted=len(accepted_keys),
                 covered=len(accepted_keys) - len(uncovered), residual_exceptions=len(residual),
                 residual_keys=sorted(residual), uncovered_keys=sorted(uncovered),
                 conflicting_keys=sorted(conflicting), malformed_timing_rows=malformed_timing,
                 accepted_termination_counts=dict(Counter(next(iter(reasons[k])) for k in accepted_keys
                                                        if len(reasons[k]) == 1)),
                 exception_attempts_all=len(exceptions), purged_records=len(purged),
                 purged_all_failed=all(r.get("success") is False for r in purged),
                 purged_all_match_exception=all(attempt_key(r) in exceptions for r in purged),
                 sources=[stamp(p) for p in (path, purged_path) if p.exists()])
    return audit, timing


def chain_audit(root):
    path = Path(root) / "runs/chain.log"
    if not path.exists():
        return dict(status="missing", path=str(path), events=[], per_arm={})
    events, counts = [], Counter()
    # One finite snapshot; never follow a growing chain log.
    for line in path.read_text().splitlines():
        if "EXC_PURGED" in line:
            match = re.search(r"EXC_PURGED arm=(\S+) n=(\d+)", line)
            if match:
                arm, n = match.group(1), int(match.group(2))
                events.append(dict(arm=arm, n=n, logged_line=line))
                counts[arm] += n
            else:
                events.append(dict(unparsed_line=line))
    return dict(status="ok", source=stamp(path), events=events, per_arm=dict(counts))


def decision_cost(row, model, sw=False):
    if row.get("ok") is False or row.get("error"):
        raise ValueError("failed decision in accepted stream")
    vision, miss = bool(row["vision"]), not bool(row["hit"])
    if miss and not vision:
        raise ValueError("policy call without vision")
    mode = row.get("camera_mode", row.get("stage1_mode", "full")) if vision else "blind"
    if mode not in ("full", "wrist_only", "blind") or (vision and mode == "blind"):
        raise ValueError(f"unpriced camera mode: {mode}")
    if mode == "wrist_only" and model != "pi05":
        raise ValueError("wrist has no GR00T price")
    if sw and not all(k in row for k in ("camera_mode", "camera_stage1_calls", "camera_completion_calls", "owner_cost")):
        raise ValueError("SW actual-work telemetry missing")
    completions = row.get("camera_completion_calls", 0)
    if completions != int(completions) or completions != int(mode == "wrist_only" and miss):
        raise ValueError("invalid actual missing-camera completion count")
    if "camera_stage1_calls" in row and row["camera_stage1_calls"] != int(vision):
        raise ValueError("camera stage1 calls disagree with vision")
    if "camera_completion" in row and bool(row["camera_completion"]) != bool(completions):
        raise ValueError("camera completion flag/count mismatch")
    cv, cm = PRICES[model]
    parts = dict(full_looks=cv * (mode == "full"), wrist_looks=.055198 * (mode == "wrist_only"),
                 completion=.049890 * completions, calls=cm * miss, blind=0.)
    total = sum(parts.values())
    if "owner_cost" in row and not math.isclose(row["owner_cost"], total, rel_tol=0, abs_tol=1e-8):
        raise ValueError("logged owner_cost disagrees with camera/call counters")
    return total, parts, mode


def accepted_decisions(directory, records, timing, model, sw=False):
    """Select latest step-zero incarnation before the accepted journal timestamp.

    Preserve an exact full-record digest to reject conflicting duplicate steps;
    retain only cost fields in memory. Client request counts attest completeness.
    """
    by_uid = {r["task_uid"]: r for r in records.values()}
    groups = defaultdict(list)
    sources = []
    for path in sorted(Path(directory).glob("server_*/decisions_*.jsonl")):
        sources.append(stamp(path))
        for row in jsonl(path):
            j = by_uid.get(row.get("uid"))
            if (row.get("ev") != "dec" or j is None
                    or int(row.get("attempt", 1) or 1) != int(j.get("attempt", 1) or 1)
                    or row.get("ts", 0) > j.get("ts", math.inf)):
                continue
            slim = {k: row[k] for k in ("step", "ts", "vision", "hit", "ok", "error", "task_id", "init",
                    "camera_mode", "stage1_mode", "camera_completion", "camera_completion_calls",
                    "camera_stage1_calls", "owner_cost", "stage1_calls") if k in row}
            slim["digest"] = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            groups[row["uid"]].append(slim)
    episodes, modes, totals = [], Counter(), Counter()
    discarded, duplicates, counter_rows, completion_calls = 0, 0, 0, 0
    for key, j in sorted(records.items()):
        uid = j["task_uid"]
        rows = groups[uid]
        starts = [r["ts"] for r in rows if r["step"] == 0]
        if not starts:
            raise ValueError(f"missing accepted step zero: {uid}")
        start = max(starts)
        unique = {}
        for row in rows:
            if row["ts"] < start:
                discarded += 1
                continue
            step = int(row["step"])
            if step in unique:
                if unique[step]["digest"] != row["digest"]:
                    raise ValueError(f"conflicting accepted decision: {uid}:{step}")
                duplicates += 1
            unique[step] = row
        if sorted(unique) != list(range(len(unique))):
            raise ValueError(f"gapped accepted stream: {uid}")
        ct = timing.get(attempt_key(j), {})
        if ct.get("infers") is None or ct["infers"] != len(unique):
            raise ValueError(f"client request count mismatch: {uid} client={ct.get('infers')} server={len(unique)}")
        ep_parts, vision, misses = Counter(), 0, 0
        decision_costs = []
        previous_stage1, previous_step = None, None
        for step, row in sorted(unique.items()):
            if (row.get("task_id"), row.get("init")) != key:
                raise ValueError(f"decision pair differs from journal: {uid}:{step}")
            counted_cost, parts, mode = decision_cost(row, model, sw)
            # SW's primary price is the logged per-decision owner_cost, checked
            # independently against the actual camera/call counters above.
            decision_costs.append(float(row["owner_cost"]) if sw else counted_cost)
            completion_calls += row.get("camera_completion_calls", 0)
            vision += bool(row["vision"])
            misses += not bool(row["hit"])
            if "stage1_calls" in row:
                counter_rows += 1
                # This counter can persist across episodes on a connection.
                # Audit within-episode increments, not an assumed reset to zero.
                counter = row["stage1_calls"]
                if counter < int(bool(row["vision"])) or (previous_stage1 is not None
                        and previous_step == step - 1 and counter - previous_stage1 != int(bool(row["vision"]))):
                    raise ValueError(f"cumulative stage1 count mismatch: {uid}:{step}")
                previous_stage1, previous_step = counter, step
            ep_parts.update(parts)
            modes[mode] += 1
        totals.update(ep_parts)
        n = len(unique)
        cost = math.fsum(decision_costs)
        episodes.append(dict(task=key[0], init=key[1], Y=int(j["success"]), N=n, V=vision, M=misses,
                             owner_cost=cost, IR=cost / n,
                             components=dict(ep_parts)))
    return episodes, dict(status="ok", discarded_prior_prefix=discarded, exact_duplicate_decisions=duplicates,
                         stage1_counter_rows=counter_rows, camera_modes=dict(modes),
                         camera_completion_calls=completion_calls,
                         total_components=dict(totals), sources=sources)


def load_arm(root, arm, cell, variant, *, expected=EXPECTED, raw=True, historical=False):
    root = Path(root)
    directory = root / "runs" / arm
    result = dict(id=f"{root.name}/{arm}", arm=arm, cell=cell, variant=variant, status="invalid")
    try:
        summary = read_json(directory / "summary.json")
        records, journal_audit = accepted_journal(directory / "client/journal.jsonl")
        outcomes = {k: int(r["success"]) for k, r in records.items()}
        result.update(journal_audit=journal_audit,
                      sources=[stamp(directory / "summary.json", True), stamp(directory / "client/journal.jsonl", True)])
        if set(outcomes) != set(expected):
            raise ValueError(f"pair-set mismatch: missing={len(set(expected)-outcomes.keys())}, extra={len(outcomes.keys()-set(expected))}")
        if summary.get("success") != sum(outcomes.values()) or summary.get("complete") != len(outcomes):
            raise ValueError("summary/journal outcome count mismatch")
        exc, timing = exception_audit(directory / "client", records)
        result["exception_audit"] = exc
        if exc["status"] != "ok":
            raise ValueError("exception audit failed (residual, uncovered, or conflicting timing)")
        model = cell.split("_")[0]
        if summary.get("model") != model:
            raise ValueError("summary model mismatch")
        ledger = summary["cost_ledger"]
        n, v, m = (int(ledger[k]) for k in ("decisions", "vision_decisions", "misses"))
        length = float(ledger["l_per_request"]["mean"])
        if n <= 0 or not 0 <= m <= v <= n or not math.isfinite(length) or length <= 0:
            raise ValueError("invalid ledger counts/request length")
        if not historical and length != 5:
            raise ValueError("R7 requires five-control requests")
        cv, cm = PRICES[model]
        # Match build_frontier's arithmetic order as well as its formula.
        full_ir = (cv * (v / n) + (1 - cv) * (m / n)) * 5 / length
        if variant in ("L10", "L5"):
            if not n == v == m:
                raise ValueError("pure reference has non-policy requests")
            full_ir = 5 / length
        result.update(SR=float(np.mean(list(outcomes.values()))), n=len(outcomes), success=sum(outcomes.values()),
                      N=n, V=v, M=m, L=length, owner_IR=full_ir, owner_IR_full_camera_counterfactual=full_ir,
                      stock_ledger_IR=ledger.get("ir_per_five_controls"),
                      stock_ledger_cost_source=ledger.get("cost_source"),
                      client_decisions=summary.get("client_decisions"),
                      ledger_minus_client_decisions=n - summary.get("client_decisions", n),
                      denominator="nominal requested five-control blocks; not measured applied controls",
                      IR_active_controls=None, _outcomes=outcomes)
        if raw:
            episodes, audit = accepted_decisions(directory, records, timing, model, variant == "SW")
            result["decision_audit"] = audit
            for field in ("N", "V", "M"):
                if sum(e[field] for e in episodes) != result[field]:
                    raise ValueError(f"accepted-stream/summary {field} mismatch")
            if summary.get("client_decisions") != n:
                raise ValueError("R7 client/ledger request-count mismatch")
            result.update(episodes=episodes, owner_IR=math.fsum(e["owner_cost"] for e in episodes) / n * 5 / length,
                          owner_IR_equal_episode_mean=float(np.mean([e["IR"] * 5 / length for e in episodes])),
                          owner_IR_components={k: x / n * 5 / length for k, x in audit["total_components"].items()})
            if variant != "SW" and not math.isclose(result["owner_IR"], full_ir, rel_tol=0, abs_tol=1e-10):
                raise ValueError("non-SW owner cost differs from full-camera formula")
            if variant == "SW":
                modes = audit["camera_modes"]
                counted_ir = (.152 * modes.get("full", 0) + .055198 * modes.get("wrist_only", 0)
                    + .049890 * audit["camera_completion_calls"] + .848 * m) / n * 5 / length
                result["owner_IR_from_camera_counts"] = counted_ir
                result["owner_IR_logged_minus_counted"] = result["owner_IR"] - counted_ir
                result["owner_IR_source"] = "sum of accepted per-decision owner_cost / nominal five-control blocks"
                if not math.isclose(result["owner_IR"], counted_ir, rel_tol=0, abs_tol=1e-10):
                    raise ValueError("SW aggregate logged/counted owner cost mismatch")
            else:
                # Preserve historical count-pricing arithmetic for strict ties.
                result["owner_IR"] = full_ir
        elif variant == "SW":
            raise ValueError("SW cannot use only the summary ledger")
        if variant == "SW":
            result["cost_assumption"] = "R4 proportional-latency assumption: wrist .055198, completion .049890"
        if historical:
            flags = read_json(FRONTIER / "ledger_tolerance_flags.json")
            result["r6_ledger_tolerance_flag"] = flags.get(arm)
            if result["ledger_minus_client_decisions"] and arm not in flags:
                raise ValueError("unregistered historical ledger/client mismatch")
        result["status"] = "complete"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["error"] = str(exc)
    return result


def average_records(records, cell, label):
    bad = [r["id"] for r in records if r["status"] != "complete"]
    if bad:
        return dict(id=f"{cell}/{label}", cell=cell, variant=label, status="missing", missing=bad)
    keys = set(records[0]["_outcomes"])
    if any(set(r["_outcomes"]) != keys for r in records):
        raise ValueError("replicates have different pair sets")
    outcomes = {k: float(np.mean([r["_outcomes"][k] for r in records])) for k in keys}
    return dict(id=f"{cell}/{label}", cell=cell, variant=label, status="complete", n=len(keys),
                replicate_ids=[r["id"] for r in records], replicate_SR=[r["SR"] for r in records],
                replicate_owner_IR=[r["owner_IR"] for r in records],
                SR=float(np.mean(list(outcomes.values()))),
                owner_IR=float(np.mean([r["owner_IR"] for r in records])),
                IR_aggregation="arithmetic mean of replicate ledger ratios (R6 ANALYSIS convention)",
                _outcomes=outcomes, _replicates=records)


def bootstrap_differences(pairs, *, seed=SEED, draws=DRAWS):
    """Equal-cell pooling; independent cell resampling as in R6 c_validation.

    A single RNG is shared across cells WITHIN a contrast, then reset for the
    next contrast. Resetting it per cell would induce spurious cell covariance.
    Replicates are already averaged within task/init before this function.
    """
    rng = np.random.default_rng(seed)
    simulations = np.zeros(draws)
    estimates = []
    for x, y in pairs:
        if not x or set(x) != set(y):
            raise ValueError("paired bootstrap requires identical nonempty pair sets")
        keys = sorted(x)
        estimates.append(float(np.mean([x[k] - y[k] for k in keys])))
        cell_sims = np.zeros(draws)
        for task in sorted({k[0] for k in keys}):
            vals = np.asarray([x[k] - y[k] for k in keys if k[0] == task], dtype=float)
            cell_sims += vals[rng.integers(len(vals), size=(draws, len(vals)))].sum(axis=1)
        simulations += cell_sims / len(keys) / len(pairs)
    return dict(estimate=float(np.mean(estimates)), lo=float(np.quantile(simulations, .025)),
                hi=float(np.quantile(simulations, .975)), draws=draws, seed=seed,
                cells=len(pairs), n_pairs_per_cell=[len(x) for x, _ in pairs]), simulations


def exact_pair(x, y, alpha=.05):
    if not x or set(x) != set(y) or any(v not in (0, 1) for v in [*x.values(), *y.values()]):
        raise ValueError("exact paired tests require identical binary pair sets")
    lower, wins, losses, n = R6.ni_lower(x, y, alpha=alpha)
    return dict(wins=wins, losses=losses, n=n, delta_SR=(wins-losses)/n,
                mcnemar_two_sided_exact_p=R6.mcnemar(wins, losses), cp_lower95=lower,
                alpha=alpha, ni_margin=.02, nominal_NI=lower > -.02,
                interpretation="nominal CP bound only; no certified or simultaneous NI claim")


def contrast(x, y):
    if x["status"] != "complete" or y["status"] != "complete":
        return dict(status="missing", candidate=x["id"], reference=y["id"])
    boot, _ = bootstrap_differences([(x["_outcomes"], y["_outcomes"])])
    row = dict(status="complete", candidate=x["id"], reference=y["id"], delta_SR=boot,
               delta_owner_IR=x["owner_IR"] - y["owner_IR"])
    # A/B means remain fractional even if a particular sample happens to be binary.
    if not x.get("_replicates") and not y.get("_replicates"):
        row["exact"] = exact_pair(x["_outcomes"], y["_outcomes"])
    else:
        row["exact"] = None
        row["exact_note"] = "McNemar/CP not applied to fractional replicate-mean outcomes"
        row["replicate_exact"] = [dict(candidate=a["id"], reference=b["id"], **exact_pair(a["_outcomes"], b["_outcomes"]))
                                  for a in x.get("_replicates", [x]) for b in y.get("_replicates", [y])]
        xs, ys = x.get("_replicates", [x]), y.get("_replicates", [y])
        alpha = .05 / (len(xs) * len(ys))
        lower = float(np.mean([exact_pair(a["_outcomes"], b["_outcomes"], alpha=alpha)["cp_lower95"]
                               for a in xs for b in ys]))
        row["replicate_mean_CP"] = dict(lower95=lower, constituent_alpha=alpha, ni_margin=.02,
            nominal_NI=lower > -.02,
            method="Q2: Bonferroni alpha over binary constituent comparisons, then average their CP bounds; no certification")
    if x.get("episodes") and y.get("episodes"):
        xe = {(e["task"], e["init"]): e["IR"] for e in x["episodes"]}
        ye = {(e["task"], e["init"]): e["IR"] for e in y["episodes"]}
        row["delta_equal_episode_IR"], _ = bootstrap_differences([(xe, ye)])
        row["delta_aggregate_owner_IR_CI"] = aggregate_cost_interval(x, y)
    else:
        row["delta_aggregate_owner_IR_CI"] = None
        row["IR_interval_note"] = "Historical IR uses summary ledgers; no per-pair cost interval is inferred from aggregate counts. SR intervals are paired."
    return row


def aggregate_cost_interval(x, y):
    """Paired task/init bootstrap of ratios of totals, separate from episode IR."""
    xe = {(e["task"], e["init"]): e for e in x["episodes"]}
    ye = {(e["task"], e["init"]): e for e in y["episodes"]}
    if set(xe) != set(ye) or not xe:
        raise ValueError("aggregate cost bootstrap requires identical pair sets")
    rng = np.random.default_rng(SEED)
    totals = np.zeros((DRAWS, 4))
    for task in sorted({k[0] for k in xe}):
        keys = sorted(k for k in xe if k[0] == task)
        vals = np.asarray([[xe[k]["owner_cost"], xe[k]["N"] * x.get("L", 5) / 5,
                            ye[k]["owner_cost"], ye[k]["N"] * y.get("L", 5) / 5] for k in keys])
        totals += vals[rng.integers(len(keys), size=(DRAWS, len(keys)))].sum(axis=1)
    sims = totals[:, 0] / totals[:, 1] - totals[:, 2] / totals[:, 3]
    return dict(estimate=x["owner_IR"] - y["owner_IR"], lo=float(np.quantile(sims, .025)),
                hi=float(np.quantile(sims, .975)), draws=DRAWS, seed=SEED,
                estimand="difference of aggregate owner cost / nominal five-control blocks")


def dominates(a, b):
    return (a["owner_IR"] <= b["owner_IR"] and a["SR"] >= b["SR"]
            and (a["owner_IR"] < b["owner_IR"] or a["SR"] > b["SR"]))


def pareto(rows):
    """Exact R6 ops/frontier_refresh -> build_frontier.pareto rule; ties retained."""
    return [r for r in rows if not any(dominates(s, r) for s in rows if s is not r)]


def public(value):
    if isinstance(value, dict):
        return {str(k): public(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [public(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def write_json(path, value):
    path = Path(path).resolve()
    allowed = (R7 / "analysis_r7").resolve()
    if allowed not in path.parents and Path("/tmp") not in path.parents:
        raise ValueError("A1 outputs must be under analysis_r7 or /tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(public(value), indent=2, allow_nan=False) + "\n")
