"""Read-only CPU analysis of the R17 DP/Fast-WAM horizon and step arms.

No campaign/model/ops imports, remote calls, CUDA, or writes below the data root.
Only closed PASS (including explicitly recovered PASS) runs are admitted.
Pending runs are inventoried without analyzing their outcomes. Re-run this
script to include newly sealed arms; the comparison family always stays 24.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from scipy.stats import beta, binomtest
import yaml

HERE = Path(__file__).resolve().parent
SUITES = ("libero_spatial", "libero_10")
FAMILY, BOOT, SEED = 24, 10000, 20261010
CONTROLS = {"DP": "hs5dp_ctl_142812", "Fast-WAM": "hs6fw_res_152931"}


@dataclass(frozen=True)
class Arm:
    model: str
    label: str
    prefix: str
    c: int
    k: int
    kind: str = "horizon"


ARMS = [
    Arm("DP", "C10", "hs5dp_c10", 10, 100),
    Arm("DP", "C15", "hs5dp_c15", 15, 100),
    Arm("DP", "C16", "hs5dp_c16", 16, 100),
    Arm("DP", "DDIM10", "hs5dp_ddim10", 5, 10, "step"),
    Arm("DP", "DDIM5", "hs5dp_ddim5", 5, 5, "step"),
    Arm("Fast-WAM", "C10", "hs6fw_c10", 10, 10),
    Arm("Fast-WAM", "C16", "hs6fw_c16", 16, 10),
    Arm("Fast-WAM", "C20", "hs6fw_c20", 20, 10),
    Arm("Fast-WAM", "C32", "hs6fw_c32", 32, 10),
    Arm("Fast-WAM", "K2", "hs6fw_k2", 5, 2, "step"),
    Arm("Fast-WAM", "K1", "hs6fw_k1", 5, 1, "step"),
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Sources:
    """Hash every admitted input while reading; verify again before publishing."""
    def __init__(self):
        self.sha = {}

    def read(self, path):
        path = Path(path)
        value = path.read_bytes()
        digest = hashlib.sha256(value).hexdigest()
        require(str(path) not in self.sha or self.sha[str(path)] == digest,
                f"Input changed between reads: {path}")
        self.sha[str(path)] = digest
        return value

    def obj(self, path):
        return json.loads(self.read(path))

    def rows(self, path):
        path = Path(path)
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for line in stream:
                h.update(line)
                if line.strip():
                    yield json.loads(line)
        digest = h.hexdigest()
        require(str(path) not in self.sha or self.sha[str(path)] == digest,
                f"Input changed between reads: {path}")
        self.sha[str(path)] = digest

    def verify(self):
        for name, expected in self.sha.items():
            h = hashlib.sha256()
            with Path(name).open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    h.update(block)
            require(h.hexdigest() == expected, f"Input changed during analysis: {name}")


def receipt_snapshot(run):
    # Pending snapshots are timestamped, not claimed immutable. Read no per-step
    # outcomes until an explicit PASS receipt exists.
    statuses, eligible = {}, None
    for name in ("RECEIPT.json", "RECOVERED_RECEIPT.json"):
        p = run / name
        if p.exists():
            raw = p.read_bytes()
            try:
                d = json.loads(raw)
                statuses[name] = dict(status=d.get("status"),
                                     recovered=bool(d.get("recovered_evidence")),
                                     sha256=hashlib.sha256(raw).hexdigest())
                if d.get("status") == "PASS":
                    eligible = name
            except json.JSONDecodeError:
                statuses[name] = {"status": "being written"}
    accepted = None
    p = run / "driver/journal.jsonl"
    if p.exists():
        ids = set()
        with p.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # Pending count only; admitted load is strict.
                if row.get("accepted"):
                    ids.add(row["task_uid"])
        accepted = len(ids)
    return dict(run=str(run), receipt=eligible, statuses=statuses,
                accepted_cells=accepted, observed_utc=datetime.now(timezone.utc).isoformat())


def discover(root, arm):
    pattern = re.compile(re.escape(arm.prefix) + r"(?:s(\d+))?$")
    candidates = []
    for p in root.glob(arm.prefix + "*"):
        m = pattern.fullmatch(p.name)
        if p.is_dir() and m:
            candidates.append((int(m[1]) if m[1] is not None else -1, p))
    snapshots = [receipt_snapshot(p) for _, p in sorted(candidates, reverse=True)]
    # Prefer latest re-prepared sealed generation, never higher SR. Incomplete
    # newer preparations do not replace an already valid completed arm.
    chosen = next((s for s in snapshots if s["receipt"]), None)
    return chosen, snapshots


def canonical(task):
    return task["experiment"], task["extra"]["task_name"], task["orig_init_state_idx"]


def last_accepted(rows):
    result = {}
    for r in rows:
        if r.get("accepted"):
            result[r["task_uid"]] = r
    return result


def validate_execution(decisions, acks, c, k):
    require(set(decisions) == set(acks) == set(range(len(decisions))) and decisions,
            "Missing/extra/non-contiguous decisions or execution ACKs")
    slots = math.ceil(c / 5)
    for seq in range(len(decisions)):
        d, a = decisions[seq], acks[seq]
        cursor = (seq % slots) * 5
        source = "policy" if cursor == 0 else "policy_tail"
        require(d["cursor"] == cursor and d["source"] == a["source"] == source,
                "Wrong policy/tail source or cursor")
        require(d["nfe"] == (k if cursor == 0 else 0), "Wrong actual NFE")
        available = min(5, c - cursor)
        final = seq == len(decisions) - 1
        require(0 <= a["count"] <= available and (final or a["count"] == available),
                "Wrong executed_count / partial tail")
        require(a["end"] == cursor + a["count"] and a["terminal"] == final,
                "Wrong cursor_end / terminal ACK")


def load_run(run, sources, *, c, k, expected, receipt_name):
    receipt = sources.obj(run / receipt_name)
    require(receipt.get("status") == "PASS", "Receipt no longer PASS")
    require(receipt.get("all_owned_closed") is True, "PASS lacks closed-run evidence")
    plan = sources.obj(run / "EPISODE_PLAN.json")
    campaign = sources.obj(run / "campaign.json")
    summary = sources.obj(run / "SUMMARY.json")
    require(summary["status"] == "PASS" and summary["completed_episodes"] == expected
            and summary["expected_episodes"] == expected, "Incomplete/invalid SUMMARY")
    require(summary.get("terminal_errors", 0) == 0 and not summary["cache_enabled"]
            and summary["full_arm_execution_verified"], "Full-arm audit missing")
    require(campaign["official_50"] and not campaign["smoke"], "Not official-50")
    require((campaign.get("commit_horizon") or 5) == c, "Campaign C mismatch")
    assets = Path(campaign["local_assets"])
    model = sources.obj(assets / "model_spec.json")
    manifest = sources.obj(assets / "model_manifest.json")
    benchmark = sources.obj(assets / "benchmark.json")
    config = yaml.safe_load(sources.read(assets / "full.yaml"))
    binding = config["adapter_binding"]
    require(binding["num_steps"] == k and binding["commit_horizon"] == c,
            "Sealed config K/C mismatch")
    require(not config["enabled"] and not config["recipe"]["enabled"], "Cache-enabled arm")
    require(benchmark["seed"] == 7 and benchmark["settle_steps"] == 10
            and benchmark["replan_steps"] == 5, "Benchmark seed/settle/R drift")
    attestations = sources.obj(run / "SERVER_ATTESTATIONS.json")
    batch = campaign.get("dp_batching", campaign.get("fastwam_batching"))
    require(len(attestations) == campaign["replicas"], "Replica attestation count")
    for att in attestations:
        meta = att["metadata"]
        require(meta["execution_commit_horizon"] == c and meta["execution_sampler"]["num_steps"] == k,
                "Server attestation K/C mismatch")
        require(meta["r17_batching"] == batch, "Server batch config mismatch")
    sampler_path = assets / "execution_sampler.json"
    sampler = sources.obj(sampler_path) if sampler_path.exists() else model["default_sampler"]
    require(sampler["num_steps"] == k, "Sampler K mismatch")
    if campaign["model"] == "dp":
        require(sampler["family"] == ("ddpm" if k == 100 else "ddim"), "Wrong DP sampler family")
        if k != 100:
            require(sampler["config"]["timestep_spacing"] == "trailing"
                    and sampler["config"]["eta"] == 0.0, "Not trailing-DDIM eta0")
    else:
        require(sampler["family"] == "shifted_flow", "Wrong Fast-WAM sampler")
    if receipt.get("recovery", {}).get("summary_sha256"):
        require(receipt["recovery"]["summary_sha256"] == sources.sha[str(run / "SUMMARY.json")],
                "Recovered SUMMARY hash mismatch")
    merge = None
    if (run / "RESUME_MERGE.json").exists():
        merge = sources.obj(run / "RESUME_MERGE.json")
        require(merge["retained_episodes"] + merge["dispatched_episodes"] == expected,
                "Resume count mismatch")
    tasks = {t["task_uid"]: t for t in plan["tasks"]}
    require(len(tasks) == len(plan["tasks"]) == plan["expected_episodes"] == expected,
            "Plan identity count mismatch")
    journal_rows = list(sources.rows(run / "driver/journal.jsonl"))
    journal = last_accepted(journal_rows)
    attempts = last_accepted(sources.rows(run / "driver/attempts.jsonl"))
    require(set(journal) == set(attempts) == set(tasks), "Incomplete/extra accepted cell set")
    data, by_uid = {}, {}
    for uid, row in journal.items():
        task, attempt = tasks[uid], attempts[uid]
        key = canonical(task)
        require(type(row["success"]) is bool and row["status"] == ("done" if row["success"] else "failed"),
                "Invalid terminal outcome")
        require(row["attempt"] == attempt["attempt"] and attempt["error"] is None
                and row["success"] == attempt["success"], "Accepted attempt mismatch/error")
        require(uid.rsplit(":", 2)[-2:] == [str(task["task_id"]), str(key[2])], "UID suffix mismatch")
        require(key not in data and "_timan107_" in attempt["worker_id"], "Duplicate cell or renderer node drift")
        d = dict(key=key, uid=uid, task_id=task["task_id"], success=row["success"],
                 attempt=row["attempt"], worker=attempt["worker_id"], controls=attempt["n_steps"],
                 duration_s=attempt["duration_s"], decisions={}, acks={}, timing=None, hashes=set())
        data[key] = by_uid[uid] = d
    duplicate_rows, stale_rows = 0, 0
    for row in sources.rows(run / "driver/per_step.jsonl"):
        if not row.get("accepted"):
            continue
        require(row["task_uid"] in by_uid, "Step outside accepted plan")
        d = by_uid[row["task_uid"]]
        if row["attempt"] != d["attempt"]:
            stale_rows += 1
            continue
        kind = row.get("_kind", "decision")
        if kind == "client_timing":
            duplicate_rows += d["timing"] is not None
            d["timing"] = row
            continue
        require(kind in ("decision", "execution_ack"), "Unknown step ledger kind")
        require(row["task_name"] == d["key"][1] and row["original_init_idx"] == d["key"][2]
                and row["success"] == d["success"], "Step identity/outcome mismatch")
        require(row["commit_horizon"] == c and row["num_steps"] == k and row["seed"] == 7,
                "Step C/K/seed mismatch")
        for name in ("model_identity", "norm_identity", "layout_identity", "benchmark_protocol",
                     "task_table_identity", "protocol_id"):
            require(row[name] == tasks[d["uid"]]["extra"][name], f"Step/plan {name} mismatch")
        require(isinstance(row["task_data_hash"], str) and row["task_data_hash"].startswith("sha256:"),
                "Missing init byte identity")
        d["hashes"].add(row["task_data_hash"])
        seq = row["decision_seq"]
        bucket = d["decisions"] if kind == "decision" else d["acks"]
        duplicate_rows += seq in bucket
        if kind == "decision":
            require(row["sampler_identity"] == binding["sampler_identity"], "Sampler identity mismatch")
            require(row["source"] in ("policy", "policy_tail"), "Non-policy source")
            policy = row["source"] == "policy"
            require(row["stage_calls"] == ([1, 1, k] if policy else [0, 0, 0])
                    and row["capture_enabled"] is False, "Stage accounting/capture mismatch")
            host = row["server_timing"]["infer_ms"]
            require(math.isfinite(host) and host >= 0, "Invalid host time")
            service, queue = row.get("service_ms"), row.get("queue_wait_ms")
            for val in (service, queue):
                require(val is None or (math.isfinite(val) and val >= 0), "Invalid service/queue time")
            bucket[seq] = dict(cursor=row["cursor"], source=row["source"], nfe=row["actual_nfe"],
                               host=host, service=service, queue=queue,
                               pricing=row.get("pricing_eligible") is True,
                               timing_scope=row.get("service_timing_scope"))
        else:
            bucket[seq] = dict(source=row["source"], count=row["executed_count"],
                               end=row["cursor_end"], terminal=row["terminal"])
    for d in data.values():
        require(len(d["hashes"]) == 1, "Inconsistent episode init hash")
        d["init_hash"] = d.pop("hashes").pop()
        dec, ack = d.pop("decisions"), d.pop("acks")
        validate_execution(dec, ack, c, k)
        require(sum(a["count"] for a in ack.values()) == d["controls"], "Executed controls/attempt mismatch")
        timing = d.pop("timing")
        require(timing is not None and timing["infers"] == len(dec)
                and timing["steps"] == d["controls"] + 10, "Client count / settle mismatch")
        ds = list(dec.values())
        fresh = [v for v in ds if v["source"] == "policy"]
        d.update(requests=len(ds), calls=len(fresh), tails=len(ds)-len(fresh),
                 nfe=sum(v["nfe"] for v in ds), host_ms=sum(v["host"] for v in ds),
                 policy_host_ms=sum(v["host"] for v in fresh),
                 service_ms=sum(v["service"] or 0 for v in ds),
                 queue_ms=sum(v["queue"] or 0 for v in ds),
                 service_present=sum(v["service"] is not None for v in ds),
                 queue_present=sum(v["queue"] is not None for v in ds),
                 pricing_true=sum(v["pricing"] for v in ds),
                 client_ms=timing["infer_ms"],
                 service_scopes=dict(Counter(str(v["timing_scope"]) for v in ds)))
    require(sum(d["requests"] for d in data.values()) == summary["decisions"], "SUMMARY decision count mismatch")
    for suite, s in summary["suites"].items():
        group = [d for key, d in data.items() if key[0] == suite]
        require(len(group) == s["completed"] and sum(d["success"] for d in group) == s["successes"],
                f"SUMMARY SR/count mismatch: {suite}")
    for name, key in (("policy_calls", "calls"), ("policy_tail_requests", "tails")):
        if name in summary:
            require(sum(d[key] for d in data.values()) == summary[name], f"SUMMARY {name} mismatch")
    primary = {key: d for key, d in data.items() if key[0] in SUITES}
    for suite in SUITES:
        ks = [key for key in primary if key[0] == suite]
        require(len(ks) == 500 and len({key[1] for key in ks}) == 10, "Not 500/10 tasks per suite")
        for task in {key[1] for key in ks}:
            require({key[2] for key in ks if key[1] == task} == set(range(50)), "Missing official init identity")
    if expected == 1000:
        require(len(primary) == len(data), "Nonprimary suite in candidate")
    treatment_keys = ("server_node", "device", "replicas", "worker_nodes", "workers", "worker_memory_mib",
                      "worker_vram_mib", "worker_egl_devices", "worker_egl_device", "worker_max_episodes",
                      "worker_restart_limit", "driver_mode", "mps", "dp_batching", "fastwam_batching")
    return dict(run=str(run), receipt=receipt_name, recovered=bool(receipt.get("recovered_evidence")),
                data=primary, treatment={key: campaign.get(key) for key in treatment_keys},
                model=model, manifest=manifest, benchmark=benchmark, sampler=sampler,
                attestation_modes=[{key: a["metadata"].get(key) for key in
                                    ("cuda_graph", "resident_execution", "gpu_family")} for a in attestations],
                summary=dict(episodes=expected, decisions=summary["decisions"],
                             driver_seconds=summary["driver_seconds"],
                             error_attempts=summary.get("error_attempts"),
                             retry_attempts=summary.get("retry_attempts")),
                duplicate_step_rows=duplicate_rows, ignored_prior_attempt_rows=stale_rows,
                merged_counts=({key: merge.get(key) for key in ("retained_episodes", "dispatched_episodes")}
                               if merge else None))


def cp_lower(g, loss, n, family=FAMILY):
    alpha = .05 / (2 * family)
    lower = 0. if g == 0 else beta.ppf(alpha, g, n-g+1)
    upper = 1. if loss == n else beta.ppf(1-alpha, loss+1, n-loss)
    return float(lower - upper)


def paired_stats(keys, control, arm):
    keys = sorted(keys)
    n = len(keys)
    gains = sum(not control[k]["success"] and arm[k]["success"] for k in keys)
    losses = sum(control[k]["success"] and not arm[k]["success"] for k in keys)
    rng = np.random.default_rng(SEED)
    boot = np.zeros(BOOT, dtype=np.int64)
    # Reset identical stream for every arm: identical suite/task/init order gives
    # identical resampled paired control indices across arms sharing a control.
    for task in sorted({key[1] for key in keys}):
        group = [key for key in keys if key[1] == task]
        delta = np.array([int(arm[k]["success"])-int(control[k]["success"]) for k in group])
        idx = rng.integers(len(group), size=(BOOT, len(group)))
        boot += delta[idx].sum(axis=1)
    samples = boot / n
    out = dict(n=n, control_success=sum(control[k]["success"] for k in keys),
               arm_success=sum(arm[k]["success"] for k in keys), gains=gains, losses=losses,
               diff=(gains-losses)/n, cp_lb_family=cp_lower(gains, losses, n),
               bootstrap_lb_family=float(np.quantile(samples, .05/FAMILY, method="linear")),
               cp_lb_uncorrected=cp_lower(gains, losses, n, 1),
               bootstrap_lb_uncorrected=float(np.quantile(samples, .05, method="linear")),
               mcnemar_p=float(binomtest(gains, gains+losses, .5).pvalue) if gains+losses else 1.)
    out["ni_pass"] = out["cp_lb_family"] > -.02 and out["bootstrap_lb_family"] > -.02
    out["uncorrected_ni_pass"] = out["cp_lb_uncorrected"] > -.02 and out["bootstrap_lb_uncorrected"] > -.02
    out["margin_sensitivity"] = {str(m): out["cp_lb_family"] > -m and out["bootstrap_lb_family"] > -m
                                 for m in (0., .02, .05)}
    return out


def cost(keys, data):
    ds = [data[key] for key in keys]
    names = ("calls", "tails", "requests", "nfe", "controls", "host_ms", "policy_host_ms",
             "service_ms", "queue_ms", "service_present", "queue_present", "pricing_true", "client_ms", "duration_s")
    out = {name: sum(d[name] for d in ds) for name in names}
    n, m, t, r = len(ds), out["calls"], out["controls"], out["requests"]
    out.update(episodes=n, calls_per_episode=m/n, requests_per_episode=r/n,
               nfe_per_policy_call=out["nfe"]/m, nfe_per_decision=out["nfe"]/r,
               calls_per_control=m/t, nfe_per_control=out["nfe"]/t,
               controls_per_episode=t/n, nfe_per_episode=out["nfe"]/n,
               policy_host_ms_per_call=out["policy_host_ms"]/m,
               host_ms_per_control=out["host_ms"]/t, host_ms_per_episode=out["host_ms"]/n,
               client_ms_per_request=out["client_ms"]/r, episode_seconds=out["duration_s"]/n,
               service_complete=out["service_present"] == r == out["queue_present"] == out["pricing_true"])
    out["service_ms_per_control"] = out["service_ms"]/t if out["service_complete"] else None
    out["service_ms_per_episode"] = out["service_ms"]/n if out["service_complete"] else None
    return out


def compare(control, candidate, arm):
    require(control["treatment"] == candidate["treatment"], "Control/arm GPU/batch/worker/MPS treatment differs")
    require(control["benchmark"] == candidate["benchmark"], "Benchmark profile differs")
    require(control["manifest"]["content_digests"] == candidate["manifest"]["content_digests"],
            "Native model/feature/source content changed; needs explicit audit")
    for field in ("checkpoint_identity", "norm_identity", "layout", "feature_spec"):
        require(control["model"][field] == candidate["model"][field], f"Model {field} differs")
    require(control["attestation_modes"] == candidate["attestation_modes"], "Resident execution/GPU/graph mode differs")
    a, b = control["data"], candidate["data"]
    require(set(a) == set(b), "Unpaired primary cells")
    require(all(a[key]["init_hash"] == b[key]["init_hash"] and a[key]["task_id"] == b[key]["task_id"]
                for key in a), "Init hash/task identity mismatch")
    suites, task_rows = {}, []
    for suite in SUITES:
        keys = sorted(key for key in a if key[0] == suite)
        stats = paired_stats(keys, a, b)
        ca, cb = cost(keys, a), cost(keys, b)
        ratios = {name: 1-cb[name]/ca[name] for name in
                  ("calls_per_episode", "calls_per_control", "nfe_per_control", "host_ms_per_control", "policy_host_ms_per_call")}
        work_key = "calls_per_control" if arm.kind == "horizon" else "nfe_per_control"
        # Queue-free service alone cannot certify equal external GPU contention.
        # With the current null service fields this necessarily stays unavailable.
        service_pass = (cb["service_ms_per_control"] < ca["service_ms_per_control"]
                        if ca["service_complete"] and cb["service_complete"] else None)
        both_success = [key for key in keys if a[key]["success"] and b[key]["success"]]
        suites[suite] = dict(stats=stats, control_cost=ca, arm_cost=cb, reductions=ratios,
                             work_reduced=cb[work_key] < ca[work_key],
                             service_lower_observed=service_pass,
                             service_cost_gate="unverified: no isolated/equal-contention cost certificate",
                             common_success=dict(n=len(both_success),
                                control_cost=cost(both_success, a) if both_success else None,
                                arm_cost=cost(both_success, b) if both_success else None))
        for task in sorted({key[1] for key in keys}):
            ks = [key for key in keys if key[1] == task]
            g = sum(not a[key]["success"] and b[key]["success"] for key in ks)
            l = sum(a[key]["success"] and not b[key]["success"] for key in ks)
            task_rows.append(dict(suite=suite, task=task, task_id=a[ks[0]]["task_id"], n=len(ks),
                                  control_success=sum(a[key]["success"] for key in ks),
                                  arm_success=sum(b[key]["success"] for key in ks), gains=g, losses=l,
                                  diff=(g-l)/len(ks), loss_inits=[key[2] for key in ks if a[key]["success"] and not b[key]["success"]],
                                  gain_inits=[key[2] for key in ks if not a[key]["success"] and b[key]["success"]]))
    ni = all(v["stats"]["ni_pass"] for v in suites.values())
    return dict(model=arm.model, arm=arm.label, kind=arm.kind, c=arm.c, k=arm.k,
                run=candidate["run"], control=control["run"], suites=suites, tasks=task_rows,
                sr_pass_both=ni, adopted=False,
                verdict="SR通过；服务成本未定" if ni else "未通过：SR证据不足；服务成本未定",
                paired_cells=[dict(suite=key[0], task=key[1], init=key[2], init_hash=a[key]["init_hash"],
                                   control_uid=a[key]["uid"], arm_uid=b[key]["uid"],
                                   control_success=a[key]["success"], arm_success=b[key]["success"])
                              for key in sorted(a)],
                audit={k: candidate[k] for k in ("receipt", "recovered", "summary", "treatment",
                                                "duplicate_step_rows", "ignored_prior_attempt_rows")})


def percent(value):
    return f"{100*value:.2f}"


def link(path, label=None):
    return f"[{label or Path(path).name}]({path}:1)"


def render(result):
    final = not result["pending"]
    horizon_status = "最终" if final else "暂定"
    passed = [f"{a['model']}/{a['arm']}" for a in result["arms"] if a["sr_pass_both"]]
    uncorrected = [f"{a['model']}/{a['arm']}" for a in result["arms"]
                   if all(v["stats"]["uncorrected_ni_pass"] for v in a["suites"].values())]
    lines = ["# DP / Fast-WAM：R17 horizon / 纯减步预注册分析", "",
             f"快照 UTC：{result['as_of_utc']}。仅分析有明确 PASS 收据且通过本脚本完整性审计的臂。",
             f"**当前已验收{len(result['arms'])}臂、pending {len(result['pending'])}臂；两suite同时通过24项校正SR门："
             f"{', '.join(passed) or '无'}；未校正下同时通过：{', '.join(uncorrected) or '无'}。** "
             f"服务成本门尚未认证；cache的{horizon_status}L与fallback理由见末表。",
             "原始结果、STUDY和运行队列只读；未调用GPU、远程工具或campaign代码。", "",
             "## 方法与入组", "",
             f"复跑：`.venv/bin/python {Path(__file__).relative_to(HERE.parents[4])}`（在仓库根执行）；"
             "仅重写本目录分析Markdown/JSON。`--self-test`运行CPU合成审计。", "",
             "配对键为(suite, canonical task, official init index)，再逐局核验init字节hash、任务ID、seed7、"
             "同benchmark profile；每suite10任务×50初态。control先审计全部2000局，再只分析spatial/10。"
             "按最后accepted terminal及对应attempt取数据；decision/execution_ack/client_timing分别计数，"
             "重复uid/step取最后记录；不把retry/error、尾请求或ACK当作新模型调用。", "",
             "保持STUDY的24比较family（含Cosmos，未完成时也不缩小比较族），a=.05/(2×24)："
             "`CP_low(g,500,a)-CP_high(l,500,a)`；这是保守的配对gains/losses CP差界，"
             "不是声称另一个精确风险差检验。bootstrap为task内成对重采样10000次，PCG64 seed20261010，"
             "canonical task/init排序、所有共享控制臂使用相同索引，取.05/24下分位。"
             "两种下界都严格大于−2pp且两个suite都过才通过SR门。未校正单侧95%界仅作敏感性，"
             "不替代主判定；未通过表示证据不足，不等于已证明损失超过2pp。", "",
             "原horizon首次失败即停的采用规则与cache研究“最长通过C，否则10”分别报告；"
             "不修改STUDY、不据未校正结果选C。step成本门用实际NFE/控制步，horizon用模型调用/控制步，"
             "二者还都需要可比服务时间/控制步下降。", "",
             "来源："+link(HERE/"STUDY.md", "预注册")+"；"+link(HERE/"COSMOS_K2.md", "同方法Cosmos报告")+
             "；"+link(HERE.parent/"cache/CACHE_STUDY.md", "cache选择规则")+"。", "",
             "| 控制 | 验收/规模 | 实际配置 |", "|---|---|---|"]
    for model, c in result["controls"].items():
        batch = c["treatment"].get("dp_batching") or c["treatment"].get("fastwam_batching")
        t = c["treatment"]
        lines.append(f"| {model} {link(Path(c['run'])/'RECEIPT.json',Path(c['run']).name)} | "
                     f"PASS；2000审计、1000主集；recovered={c['recovered']} | {t['server_node']}, "
                     f"R{t['replicas']}/B{batch['max_batch']}/{batch['batch_wait_ms']}ms/MPS；"
                     f"timan107×{t['workers']}；{t['driver_mode']} |")
    lines += ["", "Fast控制使用merged resume完整2000局，不把首段与续段重复累计。"
              "当前控制/入组臂逐项匹配GPU节点、replicas、batch设置、MPS、worker资源、EGL、"
              "benchmark、原生content_digests、模型/归一化/feature/layout与resident执行模式。"
              "与STUDY首版的旧replica/worker配置有偏离，本次按owner指定的hs5/hs6匹配控制，"
              "不追改旧登记；相同batch cap不保证各次实际batch组成、外部负载完全相同。", "",
              "## 已验收的配对结果", "",
              "SR单位%；Δ及下界单位pp；未校正LB列依次为CP / bootstrap。表内判决只针对该suite的SR门。", "",
              "| 模型 | 臂 | suite | n | control SR | arm SR | Δ | CP LB family | bootstrap LB family | 未校正LB CP / boot | 判决 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for arm in result["arms"]:
        for suite, v in arm["suites"].items():
            s = v["stats"]
            lines.append(f"| {arm['model']} | {arm['arm']} | {suite} | {s['n']} | "
                         f"{percent(s['control_success']/s['n'])} | {percent(s['arm_success']/s['n'])} | "
                         f"{100*s['diff']:+.2f} | {percent(s['cp_lb_family'])} | {percent(s['bootstrap_lb_family'])} | "
                         f"{percent(s['cp_lb_uncorrected'])} / {percent(s['bootstrap_lb_uncorrected'])} | "
                         f"{'通过' if s['ni_pass'] else '证据不足'} |")
    lines += ["", "| 模型/臂 | suite | gains / losses | McNemar双侧p（仅描述） | 未校正2pp门 |", "|---|---|---:|---:|---|"]
    for arm in result["arms"]:
        for suite, v in arm["suites"].items():
            s = v["stats"]
            lines.append(f"| {arm['model']}/{arm['arm']} | {suite} | {s['gains']} / {s['losses']} | "
                         f"{s['mcnemar_p']:.6g} | {'通过' if s['uncorrected_ni_pass'] else '未通过'} |")
    lines += ["", "**500对的精度限制：** 对接近持平、存在不一致结局的臂，24重校正让通过2pp很困难，"
              "但不是数学上不可能。以下固定n=500示例说明，不能把所有未通过都归于SR下降。", "",
              "| 假设g / l | family CP LB，pp | 未校正CP LB，pp |", "|---|---:|---:|"]
    for g, l in ((0, 0), (0, 1), (0, 2), (5, 5), (10, 10)):
        lines.append(f"| {g} / {l} | {percent(cp_lower(g,l,500))} | {percent(cp_lower(g,l,500,1))} |")
    lines += ["", "bootstrap的.05/24分位约落在10000次中的第21个尾部样本，具有Monte Carlo离散误差；"
              "保持已登记draw数和seed，不为让某臂通过而增加样本或换随机种子。"
              "尤其不能将点估计已下降超过2pp的suite也解释成只有检验保守；点差与区间应一起判断。", "",
              "## 成本：调用、NFE与观测时间分开", "",
              "M=真正policy调用，Q=含policy_tail的全部请求，T=ACK确认的实际控制步（不含10步settle）。"
              "NFE/decision使用Q作分母，另给NFE/policy-call，避免把零NFE尾请求说成缩短了sampler。"
              "host总时包括尾请求；policy host/call只平均真正前向调用。", "",
              "| 模型/臂 | suite | M/局 control→arm | NFE/policy-call | NFE/decision control→arm | M/T减少 | NFE/T减少 | host ms/T control→arm | host观测减少 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in result["arms"]:
        for suite, v in arm["suites"].items():
            a,b,r = v["control_cost"], v["arm_cost"], v["reductions"]
            lines.append(f"| {arm['model']}/{arm['arm']} | {suite} | {a['calls_per_episode']:.3f}→{b['calls_per_episode']:.3f} | "
                         f"{a['nfe_per_policy_call']:.0f}→{b['nfe_per_policy_call']:.0f} | "
                         f"{a['nfe_per_decision']:.3f}→{b['nfe_per_decision']:.3f} | "
                         f"{percent(r['calls_per_control'])}% | {percent(r['nfe_per_control'])}% | "
                         f"{a['host_ms_per_control']:.3f}→{b['host_ms_per_control']:.3f} | {percent(r['host_ms_per_control'])}% |")
    lines += ["", "| 模型/臂 | suite | Q/局 control→arm | T/局 control→arm | policy host ms/call control→arm | service/queue/pricing有效行 control→arm（各） |",
              "|---|---|---:|---:|---:|---|"]
    for arm in result["arms"]:
        for suite,v in arm["suites"].items():
            a,b = v["control_cost"],v["arm_cost"]
            line = lambda d: f"{d['service_present']}/{d['queue_present']}/{d['pricing_true']} of {d['requests']}"
            lines.append(f"| {arm['model']}/{arm['arm']} | {suite} | {a['requests_per_episode']:.3f}→{b['requests_per_episode']:.3f} | "
                         f"{a['controls_per_episode']:.3f}→{b['controls_per_episode']:.3f} | "
                         f"{a['policy_host_ms_per_call']:.2f}→{b['policy_host_ms_per_call']:.2f} | {line(a)} → {line(b)} |")
    lines += ["", "**服务成本门尚未认证。** 当前计时不能以queue-free service证据证明相同负载或无外部竞争；"
              "host wall差包含排队、batch组成、CPU/GPU竞争。上表实测host降幅是描述，不是净服务时间或CUDA kernel节省；"
              "只有调用/NFE减少可据逐行receipt确认。JSON另存共同成功pairs、每局时间与全量成本。"
              "控制driver耗时覆盖四suite且Fast含resume，候选仅两suite；不能直接相除宣称端到端加速。", "",
              "DP减步应称DDPM100 vs 同ε模型trailing-DDIM10/5（eta0）；不是截短DDPM。"
              "Fast K2/K1仍为原生shifted-flow scheduler；脚本逐行核对K、sampler identity和实际NFE。", "",
              f"## 每臂结论、{horizon_status}长段与验收状态", "",
              "| 模型/臂 | 两suite SR门 | 成本/总体采用结论 | 数据证据 |", "|---|---|---|---|"]
    for arm in result["arms"]:
        p = Path(arm["run"])
        lines.append(f"| {arm['model']}/{arm['arm']} | {'通过' if arm['sr_pass_both'] else '未通过'} | {arm['verdict']}；未标可采用 | "
                     f"{link(p/arm['audit']['receipt'],'收据')}、{link(p/'SUMMARY.json','SUMMARY')}、"
                     f"{link(p/'EPISODE_PLAN.json','plan')}、{link(p/'driver/per_step.jsonl','step ledger')} |")
    lines += ["", f"| 模型 | cache研究{horizon_status}L | 来源 | 原horizon采用顺序 |", "|---|---:|---|---|"]
    for model, v in result["provisional_long_horizon"].items():
        lines.append(f"| {model} | {v['L']} | {v['basis']} | {v['sequential_note']} |")
    lines += ["", f"{horizon_status}L只使用已验收臂的24项校正SR门；未校正界、partial结果和减步成绩都不参与。"
              "fallback10是owner规定的历史选择，**不是C10已获非劣认证**。" +
              ("全部预注册DP/Fast臂已验收，无pending。" if final else "其余臂落地后重新运行本脚本更新。")]
    if not final:
        lines += ["", "| 模型/臂 | 状态 | 目录/验收现状 |", "|---|---|---|"]
    for p in result["pending"]:
        detail = "; ".join(f"{Path(s['run']).name}: " + ", ".join(
            f"{k}={v['status']}" for k,v in s["statuses"].items()) + f", accepted={s['accepted_cells']}"
            for s in p["candidates"]) or "尚无目录"
        lines.append(f"| {p['model']}/{p['arm']} | {p['status']} | {detail}; {p['reason']} |")
    lines += ["", "1000 accepted或SUMMARY PASS不能替代PASS/明确恢复验收；未验收臂的SR和成本不进入本报告。"
              "s<n>按已封存重准备序号选择，绝不按SR挑retry；旧r<n>调试目录不在owner指定arm族内。", "",
              "完整逐task计数、g/l初态、1000对身份、0/5pp敏感性、共同成功成本和输入SHA256见 "
              +link(Path(result['output_dir'])/'RESULTS_DP_FASTWAM.json','分析JSON')+"。"
              f"本次复核{len(result['source_sha256'])}个输入文件在分析期间SHA未改变。"]
    return "\n".join(lines) + "\n"


def self_test():
    # Statistics must exactly match the previously delivered Cosmos machinery.
    from analyze_cosmos_k2 import stats as cosmos_stats
    keys = [("suite", f"t{t:02}", i) for t in range(10) for i in range(50)]
    a = {k: {"success": k[2] >= 3} for k in keys}
    b = {k: {"success": k[2] >= 2 and k[2] != 7} for k in keys}
    s, old = paired_stats(keys,a,b), cosmos_stats(keys,a,b)
    for new, prior in (("cp_lb_family","cp_lb_fwer95"), ("bootstrap_lb_family","bootstrap_lb_fwer95"),
                       ("cp_lb_uncorrected","cp_lb_pointwise95"), ("bootstrap_lb_uncorrected","bootstrap_lb_pointwise95")):
        require(s[new] == old[prior], f"Cosmos method mismatch: {new}")
    require(abs(cp_lower(0,0,500) - ((.05/(2*24))**(1/500)-1)) < 1e-14, "CP zero boundary")
    require(cp_lower(0,0,500) > -.02 and cp_lower(0,2,500) < -.02, "CP threshold fixtures")
    for c in (5,10,15,16,20,32):
        ds,acks={},{}
        slots=math.ceil(c/5)
        for i in range(slots+1):
            cursor=(i%slots)*5; src="policy" if cursor==0 else "policy_tail"
            ds[i]=dict(cursor=cursor,source=src,nfe=100 if cursor==0 else 0)
            count=min(5,c-cursor) if i<slots else 1
            acks[i]=dict(source=src,count=count,end=cursor+count,terminal=i==slots)
        validate_execution(ds,acks,c,100)
        acks[0]["end"]+=1
        try: validate_execution(ds,acks,c,100)
        except ValueError: pass
        else: raise AssertionError("Corrupt execution ACK accepted")
    require(last_accepted([dict(task_uid="x",accepted=True,attempt=1),
                           dict(task_uid="x",accepted=False,attempt=2),
                           dict(task_uid="x",accepted=True,attempt=3)])["x"]["attempt"]==3,
            "Accepted-attempt selection")
    print("CPU self-test PASS: Cosmos interval parity, CP boundaries, accepted attempts, C5/10/15/16/20/32 ACK/tails")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/data/mw_finetune/eval"))
    p.add_argument("--output-dir", type=Path, default=HERE)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test(); return
    require(not args.output_dir.resolve().is_relative_to(args.root.resolve()), "Cannot write below campaign data root")
    sources = Sources()
    sources.read(Path(__file__).resolve())
    sources.read(HERE / "STUDY.md")
    sources.read(HERE.parent / "cache/CACHE_STUDY.md")
    as_of = datetime.now(timezone.utc).isoformat()
    catalogue = [(arm, *discover(args.root,arm)) for arm in ARMS]
    controls={}
    for model,name in CONTROLS.items():
        run=args.root/name; snap=receipt_snapshot(run)
        require(snap["receipt"], f"Control lacks PASS: {run}")
        controls[model]=load_run(run,sources,c=5,k=100 if model=="DP" else 10,expected=2000,
                                 receipt_name=snap["receipt"])
        print(f"Audited control {name}: 2000 accepted / 1000 primary",flush=True)
    arms,pending=[],[]
    for arm,chosen,candidates in catalogue:
        if not chosen:
            pending.append(dict(model=arm.model,arm=arm.label,status="pending",candidates=candidates,
                                reason="无PASS收据；不分析partial结果"))
            continue
        candidate=load_run(Path(chosen["run"]),sources,c=arm.c,k=arm.k,expected=1000,
                           receipt_name=chosen["receipt"])
        arms.append(compare(controls[arm.model],candidate,arm))
        print(f"Analyzed {arm.model} {arm.label}: {Path(chosen['run']).name}",flush=True)
    choices={}
    for model in CONTROLS:
        passed=[a["c"] for a in arms if a["model"]==model and a["kind"]=="horizon" and a["sr_pass_both"]]
        first=next((a for a in arms if a["model"]==model and a["arm"]=="C10"),None)
        choices[model]=dict(L=max(passed) if passed else 10,
                           basis="最长已验收且两suite通过family SR门" if passed else "无已认证长C；按cache规则fallback10",
                           sequential_note=("C10 SR未通过，保留C5" if first and not first["sr_pass_both"]
                                            else "完整采用还需服务成本门及前序臂"))
    sources.verify()
    result=dict(as_of_utc=as_of, output_dir=str(args.output_dir.resolve()),
                method=dict(family=FAMILY,bootstrap_draws=BOOT,seed=SEED,margin=.02),
                controls={m:{k:v for k,v in c.items() if k not in ("data","manifest","model","benchmark","sampler")}
                          for m,c in controls.items()}, arms=arms,pending=pending,
                complete=not pending, provisional_long_horizon=choices,source_sha256=sources.sha,
                pending_inventory=catalogue and [dict(model=a.model,arm=a.label,candidates=ss) for a,_,ss in catalogue])
    markdown=render(result)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    # Output-only replacement; no result/campaign mutation.
    for name,body in (("RESULTS_DP_FASTWAM.json",json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n"),
                      ("RESULTS_DP_FASTWAM.md",markdown)):
        target=args.output_dir/name; tmp=target.with_suffix(target.suffix+".tmp")
        tmp.write_text(body);tmp.replace(target)
    print(json.dumps(dict(analyzed=len(arms),pending=len(pending),L=choices,
                          report=str(args.output_dir/"RESULTS_DP_FASTWAM.md")),ensure_ascii=False))


if __name__ == "__main__":
    main()
