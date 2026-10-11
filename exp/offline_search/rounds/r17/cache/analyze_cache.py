"""Read-only, CPU-only incremental analysis of CACHE_STUDY.md section 9.

Only explicitly supplied, closed PASS lanes are read for outcomes. The default
size allowlist is [500]; even a mixed-size ledger is skipped unless every size
is explicitly allowed. No model, campaign, simulator, or GPU imports/actions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta
import yaml

HERE = Path(__file__).resolve().parent
SIZES = (50, 100, 200, 300, 400, 500)
SUITES = ("libero_spatial", "libero_10")
GROUPS = ("c", "cr", "ch", "chr", "cg", "crg", "chg", "chrg")
ALIASES = {"ch": "c", "chr": "cr", "chg": "cg", "chrg": "crg"}
MODELS = ("DP", "Fast-WAM", "Cosmos")
MODEL_IDS = {"dp": "DP", "fastwam": "Fast-WAM", "cosmos": "Cosmos"}
K = {"DP": 100, "Fast-WAM": 10, "Cosmos": 5}
FAMILY, BOOT, SEED = 40, 10_000, 20261010
AUDIT_PINS = {
    "CONTAMINATION_DISTANCES.csv": "35bc907a6da58891e206ac19d12b6fcf42eee4cef77e62f9ddf1df09924337c8",
    "CONTAMINATION_CHECK.json": "e695537620daad6665ee282c1c5808f9b59639f9183f2a2e76dacfc8b3bfa30a",
}
CONTROL_NAMES = {
    "DP": "hs5dp_ctl_142812", "Fast-WAM": "hs6fw_res_152931",
    "Cosmos": "formal_cosmos_libero_official2000_20261010_130639_a8ca2043",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sealed(obj, field="digest"):
    require(obj[field] == digest({k: v for k, v in obj.items() if k != field}),
            f"Bad {field} seal")


class Sources:
    def __init__(self):
        self.pins = {}

    def pin(self, path, value):
        name = str(Path(path).resolve())
        require(name not in self.pins or self.pins[name] == value, f"Input changed: {name}")
        self.pins[name] = value
        return value

    def read(self, path):
        value = Path(path).read_bytes()
        self.pin(path, hashlib.sha256(value).hexdigest())
        return value

    def obj(self, path):
        return json.loads(self.read(path))

    def sha(self, path):
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(block)
        return self.pin(path, h.hexdigest())

    def rows(self, path):
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for line in f:
                h.update(line)
                if line.strip():
                    yield json.loads(line)
        self.pin(path, h.hexdigest())

    def verify(self):
        for path in list(self.pins):
            self.sha(path)


def receipt_snapshot(run):
    """Never open SUMMARY/journal/per_step here, including for pending counts."""
    out = dict(lane=str(run), observed_utc=datetime.now(timezone.utc).isoformat(),
               receipt=None, statuses={})
    for name in ("RECEIPT.json", "RECOVERED_RECEIPT.json"):
        p = run / name
        if not p.exists():
            continue
        raw = p.read_bytes()
        try:
            r = json.loads(raw)
        except json.JSONDecodeError:
            out["statuses"][name] = {"status": "being_written"}
            continue
        out["statuses"][name] = dict(status=r.get("status"), closed=r.get("all_owned_closed"),
                                     sha256=hashlib.sha256(raw).hexdigest())
        if (r.get("status") == "PASS" and r.get("all_owned_closed") is True
                and all(x.get("closure_confirmed") is True for x in r.get("cleanup", []))):
            out["receipt"] = name
    out["status"] = "SEALED_PASS" if out["receipt"] else "PENDING_SEAL"
    return out


def load_exclusions(sources):
    raw = {name: sources.read(HERE / name) for name in AUDIT_PINS}
    for name, value in raw.items():
        require(hashlib.sha256(value).hexdigest() == AUDIT_PINS[name], "Frozen audit changed")
    audit = json.loads(raw["CONTAMINATION_CHECK.json"])
    dims = {t["task"]: (t["dtype"], t["dim"]) for t in audit["tasks"]}
    flags, identities = set(), {}
    for row in csv.DictReader(io.StringIO(raw["CONTAMINATION_DISTANCES.csv"].decode())):
        key = row["suite"], row["task"], int(row["a_init"])
        dtype, dim = dims[key[1]]
        identities[key] = "sha256:" + digest(dict(dtype=dtype, shape=[dim],
                                                 state_sha256=row["a_sha256"], task_name=key[1]))
        if row["flag"] == "True":
            flags.add(key)
    require(len(identities) == 1000 and Counter(k[0] for k in flags)
            == {"libero_spatial": 11, "libero_10": 21}, "Wrong frozen cohort")
    return flags, identities


def physical_groups(model):
    return tuple(g for g in GROUPS if model != "Cosmos" or g not in ALIASES)


def expected_c(model, group):
    return 16 if model == "Cosmos" else (10 if "h" in group else 5)


def canonical(task):
    return task["experiment"], task["extra"]["task_name"], task["orig_init_state_idx"]


def complete_keys(keys, suite, identities):
    require(set(keys) == {k for k in identities if k[0] == suite},
            f"Not the original 500 official A identities: {suite}")


def check_fit_selection(view, parent, fit, stall, n, c):
    r = fit["receipt"]
    require(view["size"] == n == len(set(view["episode_ids"]))
            and len(view["episode_ids"]) == n, "Wrong N / duplicate selected episode")
    require(view["parent_digest"] == parent["content_digest"], "Wrong parent")
    require(r["store_digest"] == r["selection_digest"] == view["content_digest"]
            and r["library_size"] == n and r["parent_digest"] == parent["content_digest"],
            "Fit uses a different size/view")
    require(r["config"]["commit_horizon"] == c and r["config"]["k"] == 16
            and r["config"]["kref"] == 8, "Fit C/k/kref changed")
    require(fit["fit_evidence"]["partition"] == "library"
            and fit["fit_evidence"]["source_digest"] == view["content_digest"], "Fit source is not this B view")
    require(stall["store_digest"] == view["content_digest"] and stall["fitted_digest"] == fit["digest"]
            and stall["calibration_partition"] == "B_library_successful_LOEO"
            and stall.get("library_size", n) == n, "Stall uses a different B fit/view")


def validate_bundles(run, seal, sources):
    """Metadata/array integrity checks; does not unpickle a model or fit."""
    sealed(seal)
    require(seal["eval_pool"] == "A" and seal["debug_capture"] is False, "Non-A/debug lane")
    root = run / "assets/cache_inputs"
    inputs = sources.obj(root / "manifest.json")
    sealed(inputs)
    require(inputs["digest"] == seal["inputs_digest"], "Wrong staged inputs")
    cache, groups, selections = {}, {}, {}
    for g in seal["groups"]:
        n, suite, group = g.get("library_size", 500), g["suite"], g["group"]
        require(n in SIZES and suite in SUITES and group in GROUPS, "Unknown study cell")
        key = n, suite, group
        require(key not in groups, "Duplicate sealed group")
        groups[key] = g
        view_dir, fit_dir, stall_dir = (root / g[f] for f in ("view_rel", "fit_rel", "stall_rel"))
        yp = run / "assets" / g["file"]
        require(sources.sha(yp) == g["sha256"], "YAML checksum mismatch")
        config = yaml.safe_load(sources.read(yp)); recipe = config["recipe"]
        cache_key = str(view_dir), str(fit_dir), str(stall_dir)
        if cache_key not in cache:
            view = sources.obj(view_dir / "VIEW.json"); sealed(view, "content_digest")
            parent = sources.obj(view_dir / view["parent_path"] / "READY.json"); sealed(parent, "content_digest")
            fit = sources.obj(fit_dir / "READY.json"); sealed(fit)
            stall = sources.obj(stall_dir / "READY.json"); sealed(stall)
            check_fit_selection(view, parent, fit, stall, n, g["settings"]["commit"])
            require(sources.sha(view_dir / "row_indices.npy") == view["row_indices_sha256"], "Bad row indices")
            indices = np.load(view_dir / "row_indices.npy", allow_pickle=False)
            selected = set(view["episode_ids"])
            expected = [i for i, r in enumerate(parent["rows"]) if r["library_episode_id"] in selected]
            require(np.array_equal(indices, expected) and len(indices) == view["rows"], "View row selection mismatch")
            eps = {e["library_episode_id"]: e for e in parent["metadata"]["episodes"]}
            require(selected <= set(eps), "Unknown B episode")
            counts = Counter(eps[i]["task_name"] for i in selected)
            require(len(counts) == 10 and set(counts.values()) == {n // 10}, "Unbalanced per-task B selection")
            expected_episodes = set()
            for local_task in range(10):
                expected_episodes.update(np.random.default_rng(20261002 + local_task).permutation(
                    np.arange(local_task*50, (local_task+1)*50))[:n//10].tolist())
            require(selected == expected_episodes, "Not the frozen R10 nested B selection")
            require(all(eps[i]["init_pool"] == "B" and eps[i]["partition"] == "library" for i in selected), "Non-B episode")
            require(all(parent["rows"][i]["init_pool"] == "B" and parent["rows"][i]["split"] == "library"
                        for i in indices), "Non-B row")
            require(sources.sha(fit_dir / "fit.npz") == fit["arrays_sha256"], "Bad frozen fit arrays")
            with np.load(fit_dir / "fit.npz", allow_pickle=False) as arrays:
                node = dict(fit["tree"]["mapping"])["episode"]
                fitted_eps = np.asarray(arrays[node["array"]]).reshape(-1)
                require(np.array_equal(fitted_eps, [parent["rows"][i]["library_episode_id"] for i in indices]),
                        "Frozen fit arrays use different B rows")
            for name, pin in stall["files"].items():
                require(sources.sha(stall_dir / name) == pin, "Bad stall input")
            calibration = sources.obj(stall_dir / "stall.json")
            cache[cache_key] = view, fit, stall, counts, calibration["tasks"]
            selection = dict(parent=parent["content_digest"], episodes=selected)
            require((n, suite) not in selections or selections[n, suite] == selection, "Group-specific B selection")
            selections[n, suite] = selection
        view, fit, stall, counts, calibration_tasks = cache[cache_key]
        require(recipe["store_digest"] == view["content_digest"]
                and recipe["fitted_digest"] == fit["digest"] == g["fit_digest"], "Recipe fit/view mismatch")
        settings = g["settings"]
        require(recipe["enabled"] is True and recipe["corrector"] == settings["corrector"]
                and recipe["miss_mode"] == ("stall_only" if settings["gated"] else "never")
                and recipe["guard"] is False and recipe["periodic_gap"] is None
                and recipe["target_ir"] is None and recipe["ir_costs"] is None
                and recipe["k"] == 16 and recipe["kref"] == 8
                and not config.get("checkpoints") and config["write_policy"]["type"] == "never"
                and config["adapter_binding"]["commit_horizon"] == settings["commit"], "Recipe semantics changed")
        if settings["gated"]:
            require(recipe["stall_digest"] == stall["digest"] == g["stall_digest"], "Wrong gated calibration")
        g["analysis_stall_unavailable_tasks"] = stall["unavailable_tasks"]
        g["analysis_independent_B_validation"] = stall["independent_B_validation_recording"]
        g["analysis_B_episodes_per_task"] = dict(counts)
        g["analysis_stall_task_calibration"] = calibration_tasks
    return groups, selections


def nested(selections):
    for model in MODELS:
        for suite in SUITES:
            vals = sorted((n, v) for (m, n, s), v in selections.items() if (m, s) == (model, suite))
            for (_, a), (_, b) in zip(vals, vals[1:]):
                require(a["parent"] == b["parent"] and a["episodes"] <= b["episodes"],
                        f"Library views not nested: {model}/{suite}")


def evidence_rows(run, name, sources, campaign, seen=None):
    """Reproduce read-only cache resume inheritance without importing ops."""
    seen = set() if seen is None else seen
    require(run not in seen, "Resume cycle")
    seen = seen | {run}
    selection = campaign.get("cache_resume_selection")
    if selection:
        for p, pin in selection["source_pins"].items():
            require(not Path(p).exists() if pin is None else sources.sha(p) == pin, "Resume source changed")
        old = Path(selection["source"]).resolve()
        receipt = sources.obj(old / "RECEIPT.json")
        require(receipt.get("all_owned_closed") is True, "Resume parent not closed")
        keep = set(selection["retained_task_uids"])
        for row in evidence_rows(old, name, sources, sources.obj(old / "campaign.json"), seen):
            if row.get("task_uid") in keep:
                yield row
    yield from sources.rows(run / "driver" / name)


def accepted_terminal(rows):
    out = {}
    for r in rows:
        if r.get("accepted") is not True:
            continue
        uid = r["task_uid"]
        require(uid not in out or out[uid] == r, "Conflicting accepted terminal/retry; no outcome selection")
        out[uid] = r
    return out


def add_unique(bucket, key, value):
    require(key not in bucket or bucket[key] == value, "Conflicting duplicate step evidence")
    bucket[key] = value


def finish_episode(d, task, *, is_cache, model):
    ex = task["extra"]; decisions, acks = d.pop("decisions"), d.pop("acks")
    require(decisions and set(decisions) == set(acks) == set(range(len(decisions))), "Incomplete decision/ACK sequence")
    c = ex["commit_horizon"] if is_cache else (16 if model == "Cosmos" else 5)
    allowed = {"library", "library_tail", "policy", "policy_tail"} if is_cache else {"policy", "policy_tail"}
    totals = Counter(); scopes = Counter()
    for seq in sorted(decisions):
        r, a = decisions[seq], acks[seq]
        source = r["source"]; final = seq == len(decisions) - 1
        require(source in allowed and a["source"] == source, "Unexpected source")
        require(r["commit_horizon"] == c and r["num_steps"] == K[model], "C/K mismatch")
        require(r["actual_nfe"] == (K[model] if source == "policy" else 0), "Actual NFE mismatch")
        if source == "policy":
            require(r["stage_calls"] == [1, 1, K[model]], "Bad model-call accounting")
            if is_cache:
                require(ex["miss_mode"] == "stall_only" and r.get("stall_call") is True, "Forbidden ordinary MISS")
        else:
            require(r["stage_calls"] in ([1, 0, 0], [0, 0, 0]), "Tail/cache action-head computation")
        if is_cache:
            require(all(r[k] == ex[k] for k in ("bundle_identity", "recipe_identity", "yaml_identity")), "Bundle changed mid-episode")
            if source == "library":
                require(r["corrector_enabled"] == ex["corrector"] and r["miss_mode"] == ex["miss_mode"], "Wrong corrector/gate")
            if not ex["corrector"]:
                require(not r.get("corrector_applied"), "Corrector used in off group")
        require(r["capture_enabled"] is False, "Capture enabled")
        available = min(c - r["cursor"], 16 if model == "Cosmos" else 5)
        require(available > 0 and 0 <= a["executed_count"] <= available
                and (final or a["executed_count"] == available)
                and a["cursor_end"] == r["cursor"] + a["executed_count"]
                and a["terminal"] == final, "Invalid execution ACK")
        if final:
            server = a.get("server_terminal_ack")
            if is_cache or server is not None:
                require(isinstance(server, dict) and server.get("terminal") is True
                        and all(server.get(k) == a[k] for k in ("decision_seq", "executed_count", "cursor_end")),
                        "Missing terminal server ACK")
            else:
                # The three named, already sealed pure-inference controls were
                # recorded before nested server ACK preservation was introduced.
                require(a.get("ack_protocol") == "r17.execution.v1", "Missing legacy execution ACK")
        totals[source] += 1; totals["requests"] += 1
        totals["nfe"] += r["actual_nfe"]; totals["controls"] += a["executed_count"]
        host = r.get("server_timing", {}).get("infer_ms")
        for field, value in (("host_ms", host), ("service_ms", r.get("service_ms")), ("queue_ms", r.get("queue_wait_ms"))):
            if value is not None:
                require(math.isfinite(value) and value >= 0, "Invalid time")
                totals[field] += value; totals[field + "_present"] += 1
        totals["pricing_true"] += r.get("pricing_eligible") is True
        scopes[str(r.get("service_timing_scope"))] += 1
    require(totals["controls"] == d["controls"], "Attempt/ACK control count mismatch")
    timing = d.pop("timing")
    require(timing is not None and timing["infers"] == totals["requests"]
            and timing["steps"] == d["controls"] + 10, "Client timing / settle count mismatch")
    d["cost"] = dict(totals); d["service_scopes"] = dict(scopes)


def maybe_finish(d, task, *, is_cache, model):
    """Release ledger rows as soon as this episode's timing/ACKs are complete.

    The driver writes episode batches (decisions, client timing, then ACKs).
    Only a small set of per-row digests remains, so 120k episodes need not keep
    giant raw per_step dictionaries resident in memory.
    """
    if "cost" in d or d["timing"] is None:
        return
    n = d["timing"]["infers"]
    if len(d["decisions"]) == len(d["acks"]) == n:
        finish_episode(d, task, is_cache=is_cache, model=model)


def load_lane(run, snapshot, sources, identities, sizes, *, control_model=None):
    require(snapshot["receipt"], "Attempt to read an unsealed lane")
    receipt = sources.obj(run / snapshot["receipt"])
    require(receipt.get("status") == "PASS" and receipt.get("all_owned_closed") is True, "PASS changed")
    campaign = sources.obj(run / "campaign.json")
    is_cache = control_model is None
    model = MODEL_IDS[campaign["model"]]
    require(control_model is None or control_model == model, "Wrong control model")
    require(campaign.get("official_50") is True and not campaign.get("smoke"), "Not official A50")
    groups, selections = {}, {}
    if is_cache:
        seal = sources.obj(run / "assets/CACHE_STUDY.json")
        declared = {g.get("library_size", 500) for g in seal["groups"]}
        require(declared <= set(sizes), "Mixed-size lane includes outcomes outside explicit allowlist")
        groups, selections = validate_bundles(run, seal, sources)
    plan_path = run / ("FULL_EPISODE_PLAN.json" if campaign.get("cache_resume_selection") else "EPISODE_PLAN.json")
    plan = sources.obj(plan_path); tasks = {t["task_uid"]: t for t in plan["tasks"]}
    require(len(tasks) == len(plan["tasks"]), "Duplicate plan UID")
    cells = defaultdict(dict)
    for uid, t in tasks.items():
        if t["experiment"] not in SUITES:
            require(not is_cache, "Unexpected cache suite")
            continue
        key = canonical(t); ex = t["extra"]
        # Historical pure-inference plans predate explicit pool labels. Their
        # official_50 flag, exact 500-key set and runtime byte hashes attest A.
        require(key in identities and ex.get("init_pool", None if is_cache else "A") == "A"
                and ex.get("partition", None if is_cache else "test") == "test", "Noncanonical A init")
        if is_cache:
            n, g = ex.get("library_size", 500), ex["cache_group"]
            require(g in physical_groups(model), "Duplicate Cosmos logical arm")
            row = groups[n, key[0], g]
            require(row["settings"] == dict(alias=None, commit=expected_c(model, g), corrector="r" in g, gated="g" in g),
                    "Unregistered group settings")
            require(all(ex[k] == row[k] for k in ("bundle_identity", "recipe_identity"))
                    and ex["yaml_identity"] == row["sha256"] and ex["commit_horizon"] == expected_c(model, g)
                    and ex["corrector"] == ("r" in g) and ex["miss_mode"] == ("stall_only" if "g" in g else "never"),
                    "Plan/manifest identity mismatch")
        else:
            n, g = 0, "control"
        cell = model, n, g, key[0]
        require(key not in cells[cell], "Duplicate canonical planned cell")
        cells[cell][key] = uid
    for cell, keys in cells.items():
        complete_keys(keys, cell[3], identities)
    # Outcome boundary: only after receipt, size, provenance and complete plan admission.
    summary = sources.obj(run / "SUMMARY.json")
    require(summary["status"] == "PASS" and summary["completed_episodes"] == len(tasks)
            == summary["expected_episodes"] and summary.get("terminal_errors", 0) == 0
            and summary["full_arm_execution_verified"] is True, "Incomplete sealed SUMMARY")
    if receipt.get("recovery", {}).get("summary_sha256"):
        require(receipt["recovery"]["summary_sha256"] == sources.sha(run / "SUMMARY.json"), "Recovered SUMMARY changed")
    if snapshot["receipt"] == "RECOVERED_RECEIPT.json":
        rp = run / "RECOVERED_RESULT_SEAL.json"
        require(rp.exists(), "Recovered PASS lacks result seal")
        recovery = sources.obj(rp)
        # Preserve the explicit seal; its input hashes are checked when present.
        for field in ("source_sha256", "files", "pins"):
            pins = recovery.get(field)
            if isinstance(pins, dict):
                for name, pin in pins.items():
                    if isinstance(pin, str) and len(pin) == 64:
                        p = Path(name) if Path(name).is_absolute() else run / name
                        require(sources.sha(p) == pin, "Recovered result pin changed")
    journal = accepted_terminal(evidence_rows(run, "journal.jsonl", sources, campaign))
    attempts = accepted_terminal(evidence_rows(run, "attempts.jsonl", sources, campaign))
    require(set(journal) == set(tasks) == set(attempts), "Missing/extra accepted episodes")
    by_uid = {}
    for cell, keys in cells.items():
        for key, uid in keys.items():
            r, a = journal[uid], attempts[uid]
            require(type(r["success"]) is bool and r["status"] == ("done" if r["success"] else "failed")
                    and r.get("error") is None and a.get("error") is None
                    and a["success"] == r["success"] and a["attempt"] == r["attempt"], "Invalid accepted attempt")
            require(not is_cache or "_timan107_" in a["worker_id"], "Cache renderer is not timan107")
            by_uid[uid] = dict(key=key, success=r["success"], attempt=r["attempt"], controls=a["n_steps"],
                               worker=a["worker_id"], duration_s=a["duration_s"], init_hash=identities[key],
                               decisions={}, acks={}, timing=None, row_digests={})
    for row in evidence_rows(run, "per_step.jsonl", sources, campaign):
        if row.get("accepted") is not True or row.get("task_uid") not in by_uid:
            continue
        d = by_uid[row["task_uid"]]
        if row["attempt"] != d["attempt"]:
            continue
        kind = row.get("_kind") or "decision"
        token = kind, row.get("decision_seq") if kind != "client_timing" else None
        pin = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).digest()
        if token in d["row_digests"]:
            require(d["row_digests"][token] == pin, "Conflicting duplicate step evidence")
            continue
        require("cost" not in d, "Unexpected extra row after complete episode")
        d["row_digests"][token] = pin
        if kind == "client_timing":
            d["timing"] = row
            maybe_finish(d, tasks[row["task_uid"]], is_cache=is_cache, model=model)
            continue
        require(kind in ("decision", "execution_ack"), "Unknown ledger kind")
        require(row["task_name"] == d["key"][1] and row["original_init_idx"] == d["key"][2]
                and row["task_data_hash"] == d["init_hash"] and row["success"] == d["success"], "Init identity/outcome mismatch")
        if is_cache:
            require(row["init_pool"] == "A" and row["partition"] == "test", "Non-A execution")
        add_unique(d["decisions"] if kind == "decision" else d["acks"], row["decision_seq"], row)
        maybe_finish(d, tasks[row["task_uid"]], is_cache=is_cache, model=model)
    for uid, d in by_uid.items():
        require("cost" in d, "Incomplete accepted episode timing/ACK evidence")
        del d["row_digests"]
    result = {}
    for cell, keys in cells.items():
        data = {key: by_uid[uid] for key, uid in keys.items()}
        if is_cache:
            prefix = f"n{cell[1]}/" if len({t['extra'].get('library_size', 500) for t in tasks.values()}) > 1 else ""
            s = summary["groups"][prefix + cell[2] + "/" + cell[3]]
        else:
            s = summary["suites"][cell[3]]
        require(s["completed"] == 500 and s["successes"] == sum(d["success"] for d in data.values()), "SUMMARY SR mismatch")
        result[cell] = dict(data=data, lane=str(run), receipt=snapshot["receipt"],
                            treatment={k: campaign.get(k) for k in ("server_node", "device", "replicas", "mps",
                                "dp_batching", "fastwam_batching", "worker_nodes", "workers")},
                            bundle=groups.get((cell[1], cell[3], cell[2])))
    return result, {(model, n, s): v for (n, s), v in selections.items()}


def cp_lower(g, loss, n, family=FAMILY):
    alpha = .05 / (2 * family)
    return float((beta.ppf(alpha, g, n - g + 1) if g else 0)
                 - (beta.ppf(1 - alpha, loss + 1, n - loss) if loss < n else 1))


def paired(keys, ref, arm):
    keys = sorted(keys); n = len(keys)
    require(all(ref[k]["init_hash"] == arm[k]["init_hash"] for k in keys), "Paired init hashes differ")
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
    total = Counter()
    for k in keys:
        total.update(data[k]["cost"])
    t, hit, miss = total["controls"], total["library"], total["policy"]
    fresh = hit + miss
    require(t > 0 and fresh > 0, "No executed controls/fresh decisions")
    r = dict(total, episodes=len(keys), fresh_decisions=fresh, fresh_hit_rate=hit/fresh,
             miss_per_control=miss/t, nfe_per_control=total["nfe"]/t,
             calls_per_episode=miss/len(keys), nfe_per_fresh_decision=total["nfe"]/fresh,
             requests_per_episode=total["requests"]/len(keys), nfe_per_request=total["nfe"]/total["requests"],
             nfe_per_episode=total["nfe"]/len(keys))
    complete = total["service_ms_present"] == total["queue_ms_present"] == total["pricing_true"] == total["requests"]
    r["service_complete"] = complete
    for name in ("service_ms", "queue_ms", "host_ms"):
        r[name + "_per_control"] = total[name]/t if total[name+"_present"] == total["requests"] else None
    # An available host timer is not an eligible isolated GPU service timer.
    if not complete:
        r["service_ms_per_control"] = None
    return r


def sr(keys, data):
    n = len(keys); s = sum(data[k]["success"] for k in keys)
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
                       sr=sr(keys, data), cost=cost(keys, data), comparisons={},
                       stall_unavailable_tasks=cell["bundle"].get("analysis_stall_unavailable_tasks", []),
                       B_episodes_per_task=cell["bundle"].get("analysis_B_episodes_per_task", {}),
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
                require(set(data) == set(ref["data"]), "Incomplete paired cohort")
                stats = paired(keys, ref["data"], data)
                stats.update(status="PAIRED", family=family, primary=bool(family and cohort == "full"))
                rcost = cost(keys, ref["data"])
                stats["nfe_per_control_reduction"] = 1-out["cost"]["nfe_per_control"]/rcost["nfe_per_control"] if rcost["nfe_per_control"] else None
                stats["miss_per_control_reduction"] = 1-out["cost"]["miss_per_control"]/rcost["miss_per_control"] if rcost["miss_per_control"] else None
                stats["treatment_differences"] = {k: [ref["treatment"].get(k), cell["treatment"].get(k)]
                    for k in set(ref["treatment"]) | set(cell["treatment"]) if ref["treatment"].get(k) != cell["treatment"].get(k)}
                scopes, ref_scopes = set(), set()
                for k in keys:
                    scopes.update(data[k]["service_scopes"])
                    ref_scopes.update(ref["data"][k]["service_scopes"])
                stats["service_comparable"] = (not stats["treatment_differences"] and out["cost"]["service_complete"]
                                                 and rcost["service_complete"] and scopes == ref_scopes
                                                 and "None" not in scopes)
                stats["service_per_control_reduction"] = (1-out["cost"]["service_ms_per_control"]/rcost["service_ms_per_control"]
                    if stats["service_comparable"] and rcost["service_ms_per_control"] else None)
                if model == "Cosmos" and name == "control":
                    stats["renderer_subgroups"] = {}
                    for node in ("timan107", "timan108"):
                        sub = [k for k in keys if f"_{node}_" in ref["data"][k]["worker"]]
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
                out.append(dict(family=family, model=model, group=group, both_suites="PENDING" if not complete else
                                ("NI_PASS" if ni else "NI_NOT_ESTABLISHED"),
                                adoption="PENDING" if not complete else
                                ("NI_AND_MEASURED_SERVICE_SAVING" if ni and cost_pass and family == "F500" else
                                 "SIZE_NI_ONLY" if ni and family == "Fsize" else
                                 "NI_PASS_SERVICE_COST_PENDING" if ni else "NOT_ESTABLISHED")))
    return out


def pct(value):
    return "—" if value is None else f"{value*100:.2f}"


def write_report(outdir, report):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "CACHE_RESULTS.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    lines = ["# R17 六尺寸缓存研究：封存 lane 增量快照", "", f"UTC：{report['as_of_utc']}。仅分析显式指定尺寸 {report['allowed_sizes']}。",
             "", "F500 与 Fsize 各固定40比较/5%；不随已到达 lane 缩小族，不声称合并族5%。中间尺寸与敏感性均为描述性。",
             "Cosmos ch/chr/chg/chrg 分别复用 c/cr/cg/crg。CI为逐点双侧95% CP；差值LB为单侧。",
             "配对按原任务/初态字节；近邻固定剔除32例（spatial剩489，10剩479）。没有PASS的lane不读取outcome。", "",
             "## Lane 封存状态", "", "| Lane | 状态 | 收据 |", "|---|---|---|"]
    for s in report["lanes"]:
        receipts = "; ".join(f"{k}:{v['status']}" for k, v in s.get("statuses", {}).items()) or "未出现"
        lines.append(f"| {s['lane']} | {s['status']} | {receipts} |")
    lines += ["", f"当前 {len([r for r in report['rows'] if r['cohort']=='full'])}/240 个不同组×尺寸×suite 可分析；未到齐不宣布研究完成。",
              "", "## SR 与成本", "", "| model | N | group | suite | cohort | n | SR% [CP95%] | fresh hit% | MISS/control | NFE/control | calls/ep |",
              "|---|---:|---|---|---|---:|---|---:|---:|---:|---:|"]
    for r in report["rows"]:
        s, c = r["sr"], r["cost"]
        lines.append(f"| {r['model']} | {r['size']} | {r['group']} | {r['suite']} | {r['cohort']} | {s['n']} | {pct(s['sr'])} [{pct(s['cp_lo'])},{pct(s['cp_hi'])}] | {pct(c['fresh_hit_rate'])} | {c['miss_per_control']:.5f} | {c['nfe_per_control']:.5f} | {c['calls_per_episode']:.2f} |")
    lines += ["", "## 配对比较（差值/LB单位pp）", "", "| model/N/group/suite | cohort | reference | family | Δ | g/l | CP LB family | boot LB family | CP/boot LB未校正 | NI |",
              "|---|---|---|---|---:|---|---:|---:|---|---|"]
    flat = []
    for r in report["rows"]:
        for ref, c in r["comparisons"].items():
            label = f"{r['model']}/{r['size']}/{r['group']}/{r['suite']}"
            primary = c.get("primary", False)
            if c["status"] == "PAIRED":
                ni = ("PASS" if c["ni_family"] else "未证明") if primary else "描述性"
                lines.append(f"| {label} | {r['cohort']} | {ref} | {c['family'] or '—'} | {pct(c['diff'])} | {c['gains']}/{c['losses']} | {pct(c['cp_lb_family']) if c['family'] else '—'} | {pct(c['bootstrap_lb_family']) if c['family'] else '—'} | {pct(c['cp_lb_pointwise'])}/{pct(c['bootstrap_lb_pointwise'])} | {ni} |")
            else:
                lines.append(f"| {label} | {r['cohort']} | {ref} | {c.get('family') or '—'} | — | — | — | — | — | PENDING |")
            record = {k: r[k] for k in ("model", "size", "group", "suite", "cohort", "lane")}
            record.update(r["sr"]); record.update({"cost_"+k: v for k, v in r["cost"].items()})
            record.update(reference=ref, **{k: v for k, v in c.items() if not isinstance(v, (dict, list))})
            flat.append(record)
    lines += ["", "## 两套件主判定", "", "| family | model | group | NI（两套件） | 成本/采用 |", "|---|---|---|---|---|"]
    lines += [f"| {c['family']} | {c['model']} | {c['group']} | {c['both_suites']} | {c['adoption']} |" for c in report["claims"]]
    lines += ["", "## 限制与待到达", "", "NFE是动作头计算量，不含视觉/检索/修正器；零NFE不等于零耗时。service无完整可比证据时不宣称墙钟加速。",
              "Cosmos旧控制含timan108与timan107 renderer；JSON按旧控制节点分层，不能把跨节点差异解释为纯缓存因果效应。",
              "Q3实际校准来自各尺寸B成功episode LOEO，不是独立B验证录制；门不可用任务在JSON逐组列出。",
              "", "未到达 cell 清单与输入 SHA256 见 CACHE_RESULTS.json；CSV含逐比较界、成本原始计数；曲线未封存点留空。",
              "本次运行没有GPU/模型/campaign操作，没有读取未选尺寸outcome。"]
    (outdir / "CACHE_RESULTS.md").write_text("\n".join(lines) + "\n")
    fields = sorted({k for row in flat for k in row}) or ["model", "size", "group", "suite", "cohort", "status"]
    with (outdir / "CACHE_RESULTS.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(flat)
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
                                    == (model, suite, group, cohort)}
                        y = [(selected[n]["sr"]["sr"] if metric == "sr" else selected[n]["cost"][metric])
                             if n in selected else np.nan for n in SIZES]
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
            fig.suptitle(f"R17 cache: {metric}, {cohort}; sealed points only")
            fig.tight_layout(); fig.savefig(outdir / f"CACHE_{metric.upper()}_{cohort}.svg"); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lane", type=Path, action="append", required=True, help="Explicit lane; repeat for sizes/resumes")
    p.add_argument("--sizes", type=int, nargs="+", default=[500], choices=SIZES)
    p.add_argument("--eval-root", type=Path, default=Path("/data/mw_finetune/eval"), help="Location of frozen controls")
    p.add_argument("--out-dir", type=Path, default=HERE)
    args = p.parse_args(argv)
    # Do not allow an accidental output path in the data/campaign tree.
    require(not args.out_dir.resolve().is_relative_to(Path("/data")), "Analysis output must stay outside /data")
    sources = Sources(); flags, identities = load_exclusions(sources)
    for name in ("CACHE_STUDY.md", "OWNER_RULINGS.md", "analyze_cache.py"):
        sources.sha(HERE / name)
    cells, selections, snapshots = {}, {}, []
    for run in dict.fromkeys(x.resolve() for x in args.lane):
        snap = receipt_snapshot(run); snapshots.append(snap)
        if not snap["receipt"]:
            continue
        # Size metadata must precede *any* reads of a shared outcome ledger.
        seal = sources.obj(run / "assets/CACHE_STUDY.json")
        declared = {g.get("library_size", 500) for g in seal["groups"]}
        if not declared <= set(args.sizes):
            snap["status"] = "SKIPPED_UNSELECTED_SIZE"; continue
        loaded, views = load_lane(run, snap, sources, identities, args.sizes)
        require(not set(loaded) & set(cells), "Duplicate sealed cell; specify one justified final/resume lane")
        cells.update(loaded)
        for key, value in views.items():
            require(key not in selections or selections[key] == value, "Different views for the same N")
            selections[key] = value
    nested(selections)
    for model in MODELS:
        if not any(k[0] == model for k in cells):
            continue
        run = args.eval_root / CONTROL_NAMES[model]
        snap = receipt_snapshot(run); snapshots.append(snap)
        if snap["receipt"]:
            loaded, _ = load_lane(run, snap, sources, identities, args.sizes, control_model=model)
            cells.update(loaded)
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
                  pending_cells=[dict(model=m, size=n, group=g, suite=s) for m, n, g, s in expected if (m,n,g,s) not in cells],
                  source_sha256=sources.pins, exclusions=[list(k) for k in sorted(flags)],
                  selections={"/".join(map(str,k)): dict(parent=v["parent"], episodes=sorted(v["episodes"])) for k,v in selections.items()})
    sources.verify()
    write_report(args.out_dir, report)
    print(json.dumps(dict(sealed_cells=len(rows)//2, pending_cells=len(report["pending_cells"]),
                          output=str(args.out_dir / "CACHE_RESULTS.md"), sizes=args.sizes)))


if __name__ == "__main__":
    main()
