#!/usr/bin/env python3
"""Read-only, standard-library campaign telemetry; never imports a GPU runtime.

One batched H100 exec per sample. A persistent read-only timan107 exec emits
30-second node samples; the second exec slot probes timan108 (or reconnects
107). Offline 108 is retried every five minutes. No automatic tether retries.
Only --output-root is written. Campaign files and remote nodes are read-only.
Use --no-h100 --no-timan108 during an exclusive H100 sweep: the only remote
invocation is then the persistent timan107 collector (or a later reconnect).
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

SCHEMA = "r17.perf.v1"
GPU_FIELDS = "index,uuid,name,utilization.gpu,clocks.sm,power.draw,power.limit,memory.used,memory.total"
WORKER_MODULES = {"exp.r17_integration.conductor.worker_entry", "exp.r17_integration.ops.worker_lifetime"}
MAX_READ = 32 * 1024 * 1024


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def command(argv, timeout=10):
    started = time.monotonic()
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {"returncode": p.returncode, "stdout": p.stdout,
                "stderr": p.stderr[-1000:], "seconds": time.monotonic() - started}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc)[:1000],
                "seconds": time.monotonic() - started}


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def gpu_snapshot():
    result = command(["nvidia-smi", "--query-gpu=" + GPU_FIELDS, "--format=csv,noheader,nounits"], 6)
    names = ("index", "uuid", "name", "util_pct", "sm_mhz", "power_w", "power_limit_w", "memory_mib", "total_mib")
    gpus = []
    for line in result["stdout"].splitlines():
        values = [v.strip() for v in line.split(",")]
        if len(values) == len(names):
            gpus.append({k: (v if k in ("uuid", "name") else number(v)) for k, v in zip(names, values)})
    procs = command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                     "--format=csv,noheader,nounits"], 6)
    processes = []
    for line in procs["stdout"].splitlines():
        values = [v.strip() for v in line.split(",")]
        if len(values) == 4:
            processes.append(dict(uuid=values[0], pid=number(values[1]), name=values[2], memory_mib=number(values[3])))
    return dict(gpus=gpus, processes=processes, error=result["stderr"] if result["returncode"] else None,
                process_error=procs["stderr"] if procs["returncode"] else None)


def process_snapshot():
    """Read only module identity and worker id; do not record unrelated argv."""
    output = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            argv = (p / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if "-m" not in argv:
                continue
            module = argv[argv.index("-m") + 1]
            if module not in WORKER_MODULES and module not in (
                    "exp.r17_integration.ops.formal_server", "exp.r17_integration.ops.formal_driver"):
                continue
            fields = (p / "stat").read_text().rsplit(") ", 1)[1].split()
            if fields[0] == "Z":
                continue
            status = (p / "status").read_text().splitlines()
            rss = next((int(x.split()[1]) for x in status if x.startswith("VmRSS:")), 0)
            worker_id = argv[argv.index("--worker-id") + 1] if "--worker-id" in argv else None
            directory = argv[argv.index("--directory") + 1] if "--directory" in argv else None
            entry = dict(pid=int(p.name), module=module, worker_id=worker_id, directory=directory,
                               state=fields[0], ticks=int(fields[11]) + int(fields[12]),
                               start_ticks=int(fields[19]), rss_kib=rss)
            if module == "exp.r17_integration.ops.formal_driver":
                waits = Counter()
                for task in (p / "task").iterdir():
                    try:
                        waits[(task / "wchan").read_text().strip()] += 1
                    except OSError:
                        pass
                entry["thread_waits"] = dict(waits)
                entry["io"] = {line.split(":")[0]: int(line.split()[1]) for line in (p/"io").read_text().splitlines()}
            output.append(entry)
        except (OSError, ValueError, IndexError):
            continue
    return output


def node_snapshot(previous=None):
    counters = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
    cpu = None
    if previous:
        delta = [b - a for a, b in zip(previous, counters)]
        if sum(delta) > 0:
            cpu = 100 * (sum(delta) - delta[3] - delta[4]) / sum(delta)
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, rest = line.split(":", 1)
        if key in ("MemAvailable", "MemTotal", "SwapFree", "SwapTotal"):
            mem[key + "_kib"] = int(rest.split()[0])
    psi = {}
    for kind in ("cpu", "memory", "io"):
        try:
            psi[kind] = Path("/proc/pressure", kind).read_text().strip()
        except OSError:
            pass
    processes = process_snapshot()
    fleets = {}
    for path in Path("/scratch/zixuans8/mw_r17/formal/runs").glob("*/receipt.json"):
        receipt = read_json(path, {})
        if receipt.get("status") in ("RUNNING", "STARTING"):
            fleets[path.parent.name] = dict(status=receipt["status"], failures=len(receipt.get("worker_failures", [])),
                recycles=len(receipt.get("worker_recycles", [])), reason=receipt.get("reason"))
    return dict(time=time.time(), cpu_pct=cpu, cpu_counters=counters, cpu_count=os.cpu_count(),
                load=[float(v) for v in Path("/proc/loadavg").read_text().split()[:3]],
                memory=mem, pressure=psi, gpu=gpu_snapshot(), processes=processes,
                worker_count=sum(p["worker_id"] is not None for p in processes), fleets=fleets,
                clock_ticks=os.sysconf("SC_CLK_TCK"))


def quantiles(values):
    values = sorted(x for x in values if x is not None and math.isfinite(x))
    if not values:
        return dict(n=0, mean=None, p50=None, p90=None, p99=None)
    def q(p):
        index = (len(values) - 1) * p
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    return dict(n=len(values), mean=sum(values)/len(values), p50=q(.5), p90=q(.9), p99=q(.99))


def empty_metrics():
    return dict(counts=Counter(), values={}, batches={}, client=dict(infers=0, infer_ms=0., env_ms=0., img_ms=0.,
                gap_ms=0., duration_ms=0., episodes=0, duration_episodes=0), errors=Counter())


def add_value(metrics, key, value):
    value = number(value)
    if value is not None:
        metrics["values"].setdefault(key, []).append(value)


def add_row(m, row, kind, cap):
    c = m["counts"]
    if kind == "attempts":
        c["attempts"] += 1
        if row.get("accepted") is True and not row.get("error"):
            c["accepted"] += 1
            add_value(m, "episode_duration_s", row.get("duration_s"))
            if number(row.get("duration_s")) is not None:
                m["client"]["duration_ms"] += row["duration_s"] * 1000
                m["client"]["duration_episodes"] += 1
        if row.get("error"):
            c["errors"] += 1
            m["errors"][str(row["error"])[:180]] += 1
        if (number(row.get("attempt")) or 1) > 1:
            c["retry_attempts"] += 1
        if row.get("accepted") is False:
            c["rejected_results"] += 1
    elif kind == "steps":
        if row.get("accepted") is not True:
            return
        if row.get("_kind") == "client_timing":
            cl = m["client"]
            cl["episodes"] += 1
            for key in ("infers", "infer_ms", "env_ms", "img_ms"):
                cl[key] += number(row.get(key)) or 0
            cl["gap_ms"] += number(row.get("gap_ms_sum")) or 0
            if row.get("infers"):
                add_value(m, "client_call_episode_mean_ms", row.get("infer_ms", 0) / row["infers"])
        elif not row.get("_kind"):
            c["accepted_policy_calls"] += 1
            add_value(m, "policy_host_ms", row.get("cost_ms"))
            add_value(m, "queue_ms", row.get("queue_wait_ms"))
            add_value(m, "service_ms", row.get("service_ms"))
            add_value(m, "server_ws_infer_ms", row.get("server_timing", {}).get("infer_ms"))
            for key, value in (row.get("stage_ms") or {}).items():
                add_value(m, "stage_" + key + "_ms", value)
            for key, value in (row.get("stage_queue_wait_ms") or {}).items():
                add_value(m, "queue_" + key + "_ms", value)
            c["queue_unsupported"] += row.get("queue_wait_status") == "unsupported"
    elif kind == "serving":
        c["server_calls"] += 1
        add_value(m, "server_elapsed_ms", (number(row.get("elapsed_seconds")) or 0) * 1000)
        add_value(m, "queue_ms", None if row.get("queue_seconds") is None else row["queue_seconds"] * 1000)
        # Batched service_seconds explicitly includes waiting, so never call it compute.
        if "including_stage_queue_wait" not in row.get("timing_scope", ""):
            add_value(m, "service_ms", None if row.get("service_seconds") is None else row["service_seconds"] * 1000)
    elif kind == "batching" and ".stage_batch." in row.get("schema", ""):
        stage = str(row.get("stage"))
        sizes = m["batches"].setdefault(stage, Counter())
        size = int(row["size"])
        sizes[size] += 1
        c["stage_batches"] += 1
        c["under_cap_batches"] += size < cap
        # Ledger does not timestamp enqueue/dispatch or record a flush reason.
        # Under-cap flush is only an expiry proxy (stage3 may split compatibility buckets).
        c["expiry_proxy_batches"] += size < cap
        reason = row.get("flush_reason", row.get("reason"))
        c["explicit_window_expiries"] += reason in ("timeout", "wait_window", "window_expired")
    elif kind == "tether":
        c["tether_calls_logged"] += 1


def read_increment(path, cursor):
    """Bound reads, keep partial lines for the next sample, handle truncation."""
    try:
        stat = path.stat()
    except OSError:
        return [], cursor, dict(missing=True)
    initial = cursor is None or cursor.get("inode") != stat.st_ino or cursor.get("offset", 0) > stat.st_size
    offset = max(0, stat.st_size - MAX_READ) if initial else cursor["offset"]
    skipped = offset if initial else 0
    if stat.st_size - offset > MAX_READ:
        skipped += stat.st_size - offset - MAX_READ
        offset = stat.st_size - MAX_READ
    rows, malformed = [], 0
    with path.open("rb") as f:
        f.seek(offset)
        if offset and (initial or skipped):
            f.readline()  # Bootstrap/tail starts inside a record.
        while f.tell() < stat.st_size:
            start = f.tell()
            line = f.readline()
            if not line.endswith(b"\n"):
                f.seek(start)
                break
            try:
                rows.append(json.loads(line))
            except (ValueError, UnicodeDecodeError):
                malformed += 1
        end = f.tell()
    return rows, dict(inode=stat.st_ino, offset=end), dict(initial=initial, skipped_bytes=skipped, malformed=malformed)


def collect_files(specs, cursors, cutoff):
    campaigns, next_cursors, coverage = {}, dict(cursors), {}
    for spec in specs:
        name, cap = spec["name"], spec["cap"]
        m = empty_metrics()
        base = Path(spec["path"])
        if spec["scope"] == "driver":
            files = [(base / "driver" / file, kind) for file, kind in
                     (("attempts.jsonl", "attempts"), ("per_step.jsonl", "steps"), ("../TETHER_CALLS.jsonl", "tether"))]
            # Count accepted episodes exactly on first discovery (small journal).
            accepted, attempts = set(), 0
            try:
                for line in (base / "driver/attempts.jsonl").open():
                    row = json.loads(line)
                    attempts += 1
                    if row.get("accepted") is True and not row.get("error"):
                        accepted.add(row.get("task_uid"))
                m["accepted_total"] = len(accepted)
                m["attempts_total"] = attempts
            except (OSError, ValueError):
                m["accepted_total"] = None
        else:
            files = [(p, p.stem) for kind in ("serving", "batching") for p in base.glob("replica_*/" + kind + ".jsonl")]
        for path, kind in files:
            key = str(path)
            rows, next_cursors[key], cov = read_increment(path, cursors.get(key))
            coverage[key] = cov
            for row in rows:
                stamp = number(row.get("time", row.get("ts")))
                if cov.get("initial") and (stamp is None or stamp < cutoff):
                    continue  # Untimestamped per_step rows bootstrap at EOF, not historical rate.
                add_row(m, row, kind, cap)
        if name in campaigns:
            previous = campaigns[name]
            merged = merge_metrics([previous, m])
            for key in ("accepted_total", "attempts_total"):
                if key in previous or key in m:
                    merged[key] = m.get(key, previous.get(key))
            campaigns[name] = merged
        else:
            campaigns[name] = m
    return dict(campaigns=campaigns, cursors=next_cursors, coverage=coverage)


def probe(request):
    output = dict(time=time.time(), node=node_snapshot(), files=collect_files(
        request.get("specs", []), request.get("cursors", {}), request.get("cutoff", time.time())))
    print(json.dumps(output, allow_nan=False), flush=True)


def worker_stream(interval):
    previous = None
    while True:
        start = time.monotonic()
        sample = node_snapshot(previous)
        previous = sample["cpu_counters"]
        print(json.dumps(sample, allow_nan=False), flush=True)
        time.sleep(max(0, interval - (time.monotonic() - start)))


def discover(root):
    output = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        config = read_json(path / "campaign.json", {})
        if not config:
            continue
        # Include original baselines with recovery resumes and future formal campaigns.
        if not (path.name.startswith("hs3_") or config.get("resume_from") or
                (path / "recovery").is_dir() or (path / "driver/.eval.lock").exists()):
            continue
        receipt = read_json(path / "RECEIPT.json", {})
        terminal = receipt.get("all_owned_closed") is True or receipt.get("status") in ("FAIL", "PASS")
        # Initial PARTIAL is written while running; a final PARTIAL has cleanup.
        terminal |= bool(receipt.get("cleanup")) and receipt.get("all_owned_closed") is not None
        if terminal:
            continue
        batching = config.get("fastwam_batching") or config.get("dp_batching") or {}
        output.append(dict(name=path.name, path=str(path), server_path=config.get("server_run", str(path / "servers")),
                           server_node=config.get("server_node", "local"), model=config.get("model"),
                           domain=config.get("domain"), workers=config.get("workers"), replicas=config.get("replicas"),
                           worker_nodes=config.get("worker_nodes", {}), cap=batching.get("max_batch", 1),
                           window_ms=batching.get("batch_wait_ms", 0), resume_from=config.get("resume_from")))
    return output


def merge_metrics(items):
    result = empty_metrics()
    for m in items:
        result["counts"].update(m.get("counts", {}))
        result["errors"].update(m.get("errors", {}))
        for key, values in m.get("values", {}).items():
            result["values"].setdefault(key, []).extend(values)
        for stage, histogram in m.get("batches", {}).items():
            result["batches"].setdefault(stage, Counter()).update({int(k): v for k, v in histogram.items()})
        for key, value in m.get("client", {}).items():
            result["client"][key] += value
    return result


def metrics_report(m, seconds, cap):
    values = {k: quantiles(v) for k, v in m["values"].items()}
    batches = {}
    for stage, histogram in m["batches"].items():
        sizes = [size for size, count in histogram.items() for _ in range(count)]
        batches[stage] = dict(histogram=dict(histogram), cap=cap, **quantiles(sizes),
                             under_cap_pct=100 * sum(v for k, v in histogram.items() if k < cap) / max(1, len(sizes)))
    cl = dict(m["client"])
    timed = cl["infer_ms"] + cl["env_ms"] + cl["img_ms"]
    cl["active_split_pct"] = {key: 100 * cl[key] / timed if timed else None for key in ("infer_ms", "env_ms", "img_ms")}
    # duration and client timing must describe the same number of completed episodes.
    cl["residual_idle_setup_ms"] = (max(0, cl["duration_ms"] - timed)
                                    if cl["episodes"] == cl["duration_episodes"] and cl["episodes"] else None)
    cl["client_call_mean_ms"] = cl["infer_ms"] / cl["infers"] if cl["infers"] else None
    server = values.get("server_elapsed_ms", {}).get("mean")
    cl["transport_plus_protocol_residual_ms"] = (max(0, cl["client_call_mean_ms"] - server)
                                                if cl["client_call_mean_ms"] is not None and server is not None else None)
    return dict(counts=dict(m["counts"]), values=values, batches=batches, client=cl, errors=dict(m["errors"]),
                completed_episode_worker_equivalents=cl["duration_ms"]/1000/max(seconds, 1),
                episodes_per_min=60*m["counts"]["accepted"]/max(seconds, 1),
                policy_calls_per_s=m["counts"]["server_calls"]/max(seconds, 1),
                accepted_policy_calls_per_s=m["counts"]["accepted_policy_calls"]/max(seconds, 1),
                queue_timing_status="measured" if values.get("queue_ms", {}).get("n") else "unsupported",
                window_expiry_status="under-cap proxy only; enqueue/flush reason absent; stage3 compatibility splits possible",
                transport_status="unpaired client-minus-server mean; includes serialization, proxy, ack; not pure tunnel RTT")


def bottleneck(report, gpu, workers, driver=None, expected_workers=None):
    if not report["counts"].get("server_calls") and not report["counts"].get("accepted"):
        return "No live calls/completions; prepared or awaiting resume (verify worker census)."
    infer = report["client"]["active_split_pct"].get("infer_ms") or 0
    util = gpu.get("util_pct") or 0
    node_cpu = workers.get("cpu_pct") or 0
    available = workers.get("memory", {}).get("MemAvailable_kib")
    total = workers.get("memory", {}).get("MemTotal_kib")
    busy = report["completed_episode_worker_equivalents"] / max(1, expected_workers or 1)
    waits = (driver or {}).get("thread_waits", {})
    if busy < .7 and any("jbd2" in name or "folio_wait" in name for name in waits):
        return "Driver durability/dispatch stalls: fsync waits, many workers outside episodes, underfilled arrivals."
    if available and total and available/total < .1:
        return "Worker RAM pressure; inspect memory PSI and episode setup/recycles."
    if node_cpu > 85:
        return "Worker CPU pressure; inference/renderer split requires correlation."
    if util >= 90 and infer > 65:
        return "Server busy, workers mainly wait for inference; stage3 wall includes queue + compute."
    if infer < 50:
        return "Worker simulation/render/setup dominates; inspect render GPU and CPU pressure."
    stage = report["batches"].get("stage3", {})
    if stage.get("p50") is not None and stage["p50"] < .5*stage["cap"]:
        return "Underfilled server batches; concurrency/arrival rate or stage pipeline limits."
    return "Inference/protocol dominated; queue/compute timing unsupported."


def fmt(value, digits=1):
    return "N/A" if value is None else f"{value:.{digits}f}"


def summary_text(samples):
    if not samples:
        return "No samples yet.\n"
    latest, first = samples[-1], samples[0]
    seconds = latest["time"] - first["time"]
    intervals = [(a, b) for a, b in zip(samples, samples[1:]) if a["pid"] == b["pid"]]
    configs = {c["name"]: c for s in samples for c in s["configs"]}
    details = []
    lines = ["# Continuous campaign performance", "", f"UTC: {latest['utc']}; recorder PID: {latest['pid']}; "
             f"window {seconds/60:.1f} min, {len(samples)} samples. Rates exclude each recorder's bootstrap sample "
             "and use each campaign's observed interval.", "",
             "| Campaign | Accepted total | ep/min | calls/s | GPU util % / W | stage3 fill p50/p90 : cap | queue p50 ms | latency ms | Bottleneck |",
             "|---|---:|---:|---:|---|---|---|---|---|"]
    for name, config in configs.items():
        relevant = [b["campaigns"][name] for a, b in intervals if name in b["campaigns"]]
        campaign_seconds = sum(b["time"]-a["time"] for a, b in intervals if name in b["campaigns"])
        merged = merge_metrics(relevant)
        report = metrics_report(merged, campaign_seconds, config["cap"])
        observed = [s for s in samples if name in s["campaigns"]]
        campaign_last = observed[-1]
        node = config["server_node"]
        gpus = [g for s in observed for g in s.get("nodes", {}).get(node, {}).get("gpu", {}).get("gpus", [])]
        gpu = {k: quantiles([g.get(k) for g in gpus])["mean"] for k in ("util_pct", "power_w")}
        worker = campaign_last["nodes"].get("timan107", {})
        driver = next((p for p in campaign_last["nodes"].get("local", {}).get("processes", [])
                       if p.get("directory") == config["path"] + "/driver"), {})
        batch = report["batches"].get("stage3", {})
        q = report["values"].get("queue_ms", {})
        server = report["values"].get("server_elapsed_ms", {})
        accepted = next((s["campaigns"][name].get("accepted_total") for s in reversed(samples)
                         if name in s["campaigns"] and s["campaigns"][name].get("accepted_total") is not None), None)
        latency = "server p50 " + fmt(server.get("p50")) + "; client mean " + fmt(report["client"]["client_call_mean_ms"])
        ended = "Ended; " if name not in latest["campaigns"] else ""
        lines.append(f"| {name} | {accepted if accepted is not None else 'N/A'} | {report['episodes_per_min']:.2f} | "
                     f"{report['policy_calls_per_s']:.2f} | {fmt(gpu['util_pct'])} / {fmt(gpu['power_w'])} | "
                     f"{fmt(batch.get('p50'))}/{fmt(batch.get('p90'))} : {config['cap']} | {fmt(q.get('p50'))} | "
                     f"{latency} | {ended}{bottleneck(report, gpu, worker, driver, config['workers'])} |")
        details.extend(["", f"**{name}**: inference/sim/image active split {report['client']['active_split_pct']}; "
                      f"batch distributions {json.dumps(report['batches'], sort_keys=True)}; retries/errors "
                      f"{report['counts'].get('retry_attempts', 0)}/{report['counts'].get('errors', 0)}; "
                      f"completed-episode worker equivalents {report['completed_episode_worker_equivalents']:.1f} "
                      f"of {config['workers']}; observed {campaign_seconds/60:.1f} min; "
                      f"driver thread waits {driver.get('thread_waits')}.", ""])
    lines.extend(details)
    lines.extend(["", "GPU utilization/power are node-wide; campaigns sharing H100 cannot be attributed individually.",
                  "Stage wall times include queue wait and synchronization. Missing queue/compute values stay N/A.",
                  "Under-cap batches are a wait-window expiry proxy, not an explicit expiry counter.",
                  "Client-minus-server means include transport, serialization, proxy and ack overhead; samples are unpaired.",
                  "Worker timing is from completed episodes, so ongoing long episodes enter later.", "", "## Worker nodes", ""])
    for name in ("timan107", "timan108"):
        node = latest["nodes"].get(name, {})
        lines.append(f"- {name}: age {fmt(latest['time'] - node['time']) if node.get('time') else 'N/A'} s; "
                     f"CPU {fmt(node.get('cpu_pct'))}%; load {node.get('load')}; workers {node.get('worker_count')}; "
                     f"memory {node.get('memory')}; error {node.get('error')}; GPU {node.get('gpu', {}).get('gpus', [])}.")
    lines.extend(["", f"Tether invocations this sample: {latest['tether_calls']} (limit 2); "
                  f"sample collection {latest['collection_seconds']:.2f} s.", ""])
    return "\n".join(lines)


def analyze_samples(path):
    """Offline evidence, including a matched completed-episode latency cohort."""
    samples = [json.loads(line) for line in Path(path).open() if line.strip()]
    if not samples:
        return dict(samples=0)
    # Ignore each process's first sample: it bootstraps tails instead of a delta.
    intervals = [(a, b) for a, b in zip(samples, samples[1:]) if a["pid"] == b["pid"]]
    valid = [b for a, b in intervals]
    seconds = sum(b["time"]-a["time"] for a, b in intervals)
    first, last = samples[0]["time"], samples[-1]["time"]
    configs = {c["name"]: c for s in samples for c in s["configs"]}
    output = dict(schema="r17.perf.analysis.v1", samples=len(samples), start_utc=samples[0]["utc"],
                  end_utc=samples[-1]["utc"], span_minutes=(last-first)/60, measured_seconds=seconds,
                  max_tether_calls=max(s["tether_calls"] for s in samples), campaigns={}, nodes={})
    for name in ("local", "h100", "timan107", "timan108"):
        nodes = [s["nodes"][name] for s in samples if name in s["nodes"]]
        if not nodes:
            continue
        node = dict(cpu_pct=quantiles([n.get("cpu_pct") for n in nodes]),
                    worker_count=quantiles([n.get("worker_count") for n in nodes]),
                    available_gib=quantiles([n.get("memory", {}).get("MemAvailable_kib", 0)/2**20 for n in nodes
                                            if n.get("memory")]),
                    last=nodes[-1], gpus={})
        for index in sorted({g["index"] for n in nodes for g in n.get("gpu", {}).get("gpus", [])}):
            rows = [g for n in nodes for g in n.get("gpu", {}).get("gpus", []) if g["index"] == index]
            node["gpus"][str(index)] = {key: quantiles([g.get(key) for g in rows])
                for key in ("util_pct", "power_w", "memory_mib", "sm_mhz")}
        output["nodes"][name] = node
    for name, config in configs.items():
        metrics = merge_metrics([s["campaigns"][name] for s in valid if name in s["campaigns"]])
        if not metrics["counts"].get("server_calls") and not metrics["counts"].get("accepted"):
            continue
        campaign_seconds = sum(b["time"]-a["time"] for a, b in intervals if name in b["campaigns"])
        report = metrics_report(metrics, campaign_seconds, config["cap"])
        report["measured_seconds"] = campaign_seconds
        report["config"] = config
        observed = [s for s in samples if name in s["campaigns"]]
        gpu_rows = [g for s in observed for g in s["nodes"].get(config.get("server_node", "local"), {}).get("gpu", {}).get("gpus", [])]
        report["gpu"] = {key: quantiles([g.get(key) for g in gpu_rows])
                         for key in ("util_pct", "power_w", "memory_mib", "sm_mhz")}
        report["batch_request_weighted"] = {stage: quantiles([int(size) for size, count in batch["histogram"].items()
            for _ in range(int(size)*count)]) for stage, batch in report["batches"].items()}
        report["accepted_first"] = observed[0]["campaigns"][name].get("accepted_total")
        report["accepted_last"] = observed[-1]["campaigns"][name].get("accepted_total")
        report["receipt_status"] = read_json(Path(config["path"])/"RECEIPT.json", {}).get("status")
        report["driver_fsync_wait_samples"] = sum(any("jbd2" in w or "folio_wait" in w
            for w in s["campaigns"][name].get("driver", {}).get("thread_waits", {})) for s in observed)
        report["driver_observed_samples"] = sum(bool(s["campaigns"][name].get("driver")) for s in observed)
        report["workers"] = {node: {key: quantiles([s["campaigns"][name].get("workers", {}).get(node, {}).get(key)
            for s in observed]) for key in ("count", "rss_kib", "cpu_pct_one_core", "running")}
            for node in ("timan107", "timan108")}
        accepted = {}
        try:
            for line in (Path(config["path"])/"driver/attempts.jsonl").open():
                row = json.loads(line)
                if first < row.get("time", 0) <= last and row.get("accepted") is True and not row.get("error"):
                    accepted[(row["task_uid"], row.get("attempt", 1))] = row
        except (OSError, ValueError):
            pass
        assignments = {}
        by_worker = {}
        try:
            for line in (Path(config["path"])/"driver/worker_dispatch.jsonl").open():
                row = json.loads(line)
                assignments[(row["task_uid"], row.get("attempt", 1))] = row
                by_worker.setdefault(row.get("worker_id"), []).append(row)
        except (OSError, ValueError):
            pass
        delays, gaps = [], []
        for key, row in accepted.items():
            assignment = assignments.get(key)
            if assignment and row.get("duration_s") is not None:
                delays.append(row["time"]-assignment["time"]-row["duration_s"])
        for rows in by_worker.values():
            for a, b in zip(rows, rows[1:]):
                ending = accepted.get((a["task_uid"], a.get("attempt", 1)))
                if ending and b["time"] <= last:
                    gaps.append(max(0, b["time"]-ending["time"]))
        report["report_durability_delay_s"] = quantiles(delays)
        report["between_episode_dispatch_gap_s"] = quantiles(gaps)
        resume = read_json(Path(config["path"])/"RESUME.json", {})
        report["retained_episodes"] = resume.get("retained_episodes", 0)
        cohort = empty_metrics()
        client_keys = set()
        try:
            for line in (Path(config["path"])/"driver/per_step.jsonl").open():
                row = json.loads(line)
                key = row.get("task_uid"), row.get("attempt", 1)
                if key not in accepted:
                    continue
                add_row(cohort, row, "steps", config["cap"])
                if row.get("_kind") == "client_timing" and row.get("accepted") is True:
                    client_keys.add(key)
        except (OSError, ValueError):
            pass
        for key in client_keys:
            add_row(cohort, accepted[key], "attempts", config["cap"])
        cohort_report = metrics_report(cohort, last-first, config["cap"])
        client = cohort_report["client"]
        ws = cohort_report["values"].get("server_ws_infer_ms", {}).get("mean")
        client["transport_plus_protocol_residual_ms"] = (client["client_call_mean_ms"]-ws
            if client["client_call_mean_ms"] is not None and ws is not None else None)
        cohort_report["matched_episodes"] = len(client_keys)
        cohort_report["transport_status"] = "matched completed-episode client infer mean minus server WebSocket infer mean; includes protocol/ack/serialization"
        duration = client["duration_ms"]
        client["episode_wall_split_pct"] = {key: 100*client[key]/duration if duration else None
                                           for key in ("infer_ms", "env_ms", "img_ms")}
        client["episode_wall_split_pct"]["setup_other"] = (100*client["residual_idle_setup_ms"]/duration
            if duration and client["residual_idle_setup_ms"] is not None else None)
        report["completed_cohort"] = cohort_report
        report["episode_busy_worker_equivalents"] = duration/1000/(last-first) if last > first else None
        output["campaigns"][name] = report
    return output


class Recorder:
    def __init__(self, args):
        self.args = args
        self.source = Path(__file__).read_text()
        self.cursors = {}
        self.worker = None
        self.worker_latest = {}
        self.worker_lock = threading.Lock()
        self.nodes = {}
        self.next108 = 0
        self.stopped = threading.Event()
        self.history = deque()
        self.last_summary = 0
        self.remote_error = None
        self.previous_workers = {}

    def start_worker(self):
        argv = ["tether", "exec", "--safe", "--timeout", "24h", "timan107", "--", "python3", "-u", "-c",
                self.source, "--worker-stream", "--interval", str(self.args.interval)]
        self.worker = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                       stdin=subprocess.DEVNULL)
        def reader(proc):
            for line in proc.stdout:
                try:
                    node = json.loads(line)
                    with self.worker_lock:
                        self.worker_latest = node
                except ValueError:
                    continue
        threading.Thread(target=reader, args=(self.worker,), daemon=True).start()

    def remote(self, node, specs):
        request = dict(specs=specs, cursors=self.cursors.get(node, {}), cutoff=time.time() - self.args.interval)
        result = command(["tether", "exec", "--safe", "--timeout", "20s", node, "--", "python3", "-c",
                          self.source, "--probe", json.dumps(request)], 25)
        try:
            parsed = json.loads(result["stdout"])
            parsed["exec_seconds"] = result["seconds"]
            return parsed
        except ValueError:
            return dict(error=result["stderr"], returncode=result["returncode"], exec_seconds=result["seconds"])

    def sample(self):
        started = time.monotonic()
        now = time.time()
        configs = discover(self.args.eval_root)
        driver_specs = [dict(name=c["name"], path=c["path"], cap=c["cap"], scope="driver") for c in configs]
        local_specs = [dict(name=c["name"], path=c["server_path"], cap=c["cap"], scope="server")
                       for c in configs if c["server_node"] == "local"]
        remote_specs = [dict(name=c["name"], path=c["server_path"], cap=c["cap"], scope="server")
                        for c in configs if c["server_node"] == "h100"]
        calls = 0 if getattr(self.args, "no_h100", False) else 1
        restart = self.worker is None or self.worker.poll() is not None
        if restart:
            self.start_worker()
            calls += 1
        with ThreadPoolExecutor(max_workers=3) as pool:
            h100 = None if getattr(self.args, "no_h100", False) else pool.submit(self.remote, "h100", remote_specs)
            local = pool.submit(node_snapshot, self.nodes.get("local", {}).get("cpu_counters"))
            secondary = None
            if not getattr(self.args, "no_timan108", False) and not restart and now >= self.next108:
                secondary = pool.submit(self.remote, "timan108", [])
                calls += 1
            files = collect_files(driver_specs + local_specs, self.cursors.get("local", {}), now-self.args.interval)
            self.cursors["local"] = files["cursors"]
            self.nodes["local"] = local.result()
            remote = h100.result() if h100 else {}
            if not h100:
                self.nodes.pop("h100", None)
            elif remote.get("files"):
                self.cursors["h100"] = remote["files"]["cursors"]
                self.nodes["h100"] = remote["node"]
            else:
                self.nodes["h100"] = dict(time=now, error=remote.get("error"))
            if secondary:
                result = secondary.result()
                previous = self.nodes.get("timan108", {}).get("cpu_counters")
                self.nodes["timan108"] = result.get("node", dict(time=now, error=result.get("error")))
                if previous and result.get("node"):
                    delta = [b-a for a, b in zip(previous, result["node"]["cpu_counters"])]
                    if sum(delta) > 0:
                        self.nodes["timan108"]["cpu_pct"] = 100*(sum(delta)-delta[3]-delta[4])/sum(delta)
                self.next108 = now + (self.args.interval if result.get("node") else 300)
        with self.worker_lock:
            self.nodes["timan107"] = dict(self.worker_latest)
        campaigns = {}
        server_files = remote.get("files", {}).get("campaigns", {})
        for config in configs:
            name = config["name"]
            driver = files["campaigns"].get(name, {})
            parts = [driver]
            if config["server_node"] == "h100":
                parts.append(server_files.get(name, {}))
            merged = merge_metrics(parts)
            merged.update(accepted_total=driver.get("accepted_total"), attempts_total=driver.get("attempts_total"))
            census = {}
            for worker_node in ("timan107", "timan108"):
                current = self.nodes.get(worker_node, {})
                previous = self.previous_workers.get(worker_node, {})
                processes = [p for p in current.get("processes", [])
                             if (p.get("worker_id") or "").startswith(name + "_" + worker_node + "_")]
                old = {(p["pid"], p["start_ticks"]): p for p in previous.get("processes", [])}
                elapsed = (current.get("time", 0) - previous.get("time", 0))
                cpu_ticks = sum(max(0, p["ticks"] - old[(p["pid"], p["start_ticks"])]["ticks"])
                                for p in processes if (p["pid"], p["start_ticks"]) in old)
                census[worker_node] = dict(count=len(processes), running=sum(p["state"] == "R" for p in processes),
                    rss_kib=sum(p["rss_kib"] for p in processes), sample_time=current.get("time"),
                    cpu_pct_one_core=100*cpu_ticks/current.get("clock_ticks", 100)/elapsed if elapsed > 0 else None)
            merged["workers"] = census
            merged["driver"] = next((p for p in self.nodes.get("local", {}).get("processes", [])
                if p.get("directory") == config["path"] + "/driver"), {})
            merged["report"] = metrics_report(merged, self.args.interval, config["cap"])
            campaigns[name] = merged
        self.previous_workers = {name: self.nodes.get(name, {}) for name in ("timan107", "timan108")}
        return dict(schema=SCHEMA, time=now, utc=datetime.fromtimestamp(now, timezone.utc).isoformat(), pid=os.getpid(),
                    configs=configs, campaigns=campaigns, nodes=self.nodes.copy(), tether_calls=calls,
                    remote_exec_seconds=remote.get("exec_seconds"), collection_seconds=time.monotonic()-started,
                    coverage={"local": files["coverage"], "h100": remote.get("files", {}).get("coverage", {})})

    def run(self):
        self.args.output_root.mkdir(parents=True, exist_ok=True)
        # One recorder per output root. No campaign locks are acquired.
        lock = (self.args.output_root / ".recorder.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Preserve the rolling window across recorder-only restarts, bounded by MAX_READ.
        existing = self.args.output_root / datetime.now(timezone.utc).strftime("%Y-%m-%d") / "perf.jsonl"
        if existing.exists():
            rows, _, _ = read_increment(existing, None)
            self.history.extend(row for row in rows if row.get("time", 0) >= time.time()-self.args.summary_window)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: self.stopped.set())
        while not self.stopped.is_set():
            started = time.monotonic()
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            directory = self.args.output_root / date
            directory.mkdir(parents=True, exist_ok=True)
            try:
                sample = self.sample()
                with (directory / "perf.jsonl").open("a") as f:
                    f.write(json.dumps(sample, allow_nan=False) + "\n")
                    f.flush()
                self.history.append(sample)
                while self.history and self.history[0]["time"] < sample["time"]-self.args.summary_window:
                    self.history.popleft()
                if time.monotonic()-self.last_summary >= self.args.summary_every or self.args.once:
                    temporary = directory / "summary.md.tmp"
                    temporary.write_text(summary_text(list(self.history)))
                    temporary.replace(directory / "summary.md")
                    self.last_summary = time.monotonic()
                print(f"{sample['utc']} campaigns={len(sample['campaigns'])} tether={sample['tether_calls']} "
                      f"collection_s={sample['collection_seconds']:.2f}", flush=True)
            except Exception as exc:
                with (directory / "recorder_errors.jsonl").open("a") as f:
                    f.write(json.dumps(dict(time=time.time(), error=repr(exc))) + "\n")
                print(repr(exc), file=sys.stderr, flush=True)
            if self.args.once:
                break
            self.stopped.wait(max(0, self.args.interval-(time.monotonic()-started)))
        if self.worker is not None and self.worker.poll() is None:
            # Only our telemetry transport child; never a campaign PID.
            self.worker.terminate()
            try:
                self.worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.worker.kill()
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=Path("/data/mw_finetune/eval"))
    parser.add_argument("--output-root", type=Path, default=Path("/data/mw_finetune/perf"))
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--summary-every", type=float, default=300)
    parser.add_argument("--summary-window", type=float, default=1800)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-h100", action="store_true", help="Do not contact or probe H100")
    parser.add_argument("--no-timan108", action="store_true", help="Do not contact timan108")
    parser.add_argument("--probe")
    parser.add_argument("--worker-stream", action="store_true")
    parser.add_argument("--analyze", type=Path, help="Analyze existing perf.jsonl without any remote/GPU probes")
    parser.add_argument("--analysis-output", type=Path)
    args = parser.parse_args()
    if args.analyze:
        report = json.dumps(analyze_samples(args.analyze), indent=2, allow_nan=False)
        if args.analysis_output:
            if not args.analysis_output.resolve().is_relative_to(args.output_root.resolve()):
                parser.error("analysis output must be inside --output-root")
            args.analysis_output.write_text(report + "\n")
        else:
            print(report)
    elif args.probe:
        probe(json.loads(args.probe))
    elif args.worker_stream:
        worker_stream(args.interval)
    else:
        if args.interval < 30:
            parser.error("interval must be >=30 seconds (two tether invocations per 30 seconds)")
        Recorder(args).run()


if __name__ == "__main__":
    main()
