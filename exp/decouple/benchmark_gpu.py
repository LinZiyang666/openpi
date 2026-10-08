"""Coordinator-only paired GPU benchmark. Never invoked by CPU validation.

Config contains arms with server_command, worker_command and summary_json.
Commands are argv lists; {port}, {out} are expanded without a shell. The
unchanged worker/conductor writes summary_json with episodes_completed and
decision_latencies_ms. Startup is excluded from timing. Each arm uses exactly
the declared model count under the same GPU memory budget. Include a one-copy
control and the largest current fleet that fits that budget.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from exp.decouple.package_deadline import PackageDeadlineReached, stop_owned_groups


def expand_tokens(value, **tokens):
    # JSON method kwargs contain literal braces; str.format would interpret
    # those as format fields. Only these explicit tokens are substitutions.
    result = str(value)
    for key, item in tokens.items():
        result = result.replace('{' + key + '}', str(item))
    return result


def accepted_cohort(records, measured_uids=None):
    from openpi.conductor.journal import Journal
    return {r['task_uid']:r for r in records if Journal.record_counts_as_done(r)
            and not r.get('error') and r.get('attempt', 1) > 0
            and (measured_uids is None or r['task_uid'] in measured_uids)}


def run_arm(arm, root, gpu_index, budget_mb):
    import numpy as np
    import psutil
    import pynvml as nvml
    from openpi_client import msgpack_numpy
    from websockets.sync.client import connect
    nvml.nvmlInit()
    gpu = nvml.nvmlDeviceGetHandleByIndex(gpu_index)
    name = nvml.nvmlDeviceGetName(gpu)
    if isinstance(name, bytes):
        name = name.decode()
    if "4090" not in name:
        raise ValueError(f"benchmark requires the requested single RTX 4090, found {name}")
    out = root / arm["name"]
    out.mkdir(parents=True, exist_ok=False)
    if arm.get('case'):
        from exp.decouple.package_config import materialize, write_json
        materialize(arm['case'], out)
        write_json(out/'arm.json', {**arm, 'gpu_memory_budget_mb':budget_mb})
    measured = arm.get("measure_latency_proxy", True)
    backend_port = arm.get("backend_port", arm["port"] + 100) if measured else arm["port"]
    def expand(argv, port=None):
        return [expand_tokens(arg, port=port or arm['port'], backend_port=backend_port, out=out) for arg in argv]
    samples, stop = [], threading.Event()
    sampling_errors = []
    startup_peak = nvml.nvmlDeviceGetMemoryInfo(gpu).used / (1 << 20)
    server = worker = relay = thread = None
    with (out / "server.log").open("w") as server_log, (out / "worker.log").open("w") as worker_log, (out/'resource_samples.jsonl').open('w') as resource_log:
        def persist_startup(copies=None):
            if arm.get('case'):
                write_json(out/'startup.json',dict(gpu_name=name, gpu_memory_peak_mb=startup_peak,
                                                  model_processes=copies or arm.get('model_processes')))
        def check_package_deadline():
            if (time.monotonic() >= arm.get('package_deadline', float('inf')) or
                    time.time() >= arm.get('package_wall_deadline', float('inf'))):
                raise PackageDeadlineReached('arm reached package shutdown cutoff')
        try:
            check_package_deadline()
            persist_startup()
            server = subprocess.Popen(expand(arm["server_command"], backend_port), stdout=server_log, stderr=subprocess.STDOUT,
                                      env=dict(os.environ, **{k:expand_tokens(v, out=out) for k,v in arm.get('server_env', {}).items()}), start_new_session=True)
            deadline = min(time.monotonic() + arm.get('startup_timeout_s', 900), arm.get('package_deadline', float('inf')))
            while True:
                check_package_deadline()
                if time.monotonic() >= deadline:
                    raise TimeoutError('server startup timed out')
                startup_peak = max(startup_peak, nvml.nvmlDeviceGetMemoryInfo(gpu).used / (1 << 20))
                persist_startup()
                if startup_peak > budget_mb:
                    raise RuntimeError('startup exceeded common GPU memory budget')
                if server.poll() is not None:
                    raise RuntimeError(f"server exited during startup ({server.returncode})")
                try:
                    with connect(f"ws://127.0.0.1:{backend_port}", open_timeout=5, compression=None) as client:
                        metadata = msgpack_numpy.unpackb(client.recv(timeout=5))
                    break
                except (OSError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise TimeoutError("server startup timed out")
                    time.sleep(0.5)
            # Equal budget, explicit weight-copy count. A split arm must attest
            # its backend model count; current fleets may spend the budget on
            # additional copies and should be included as a stronger baseline.
            copies = arm.get("model_processes")
            if copies == 'auto':
                copies = metadata.get('model_processes')
                if not metadata.get('benchmark_current'):
                    raise ValueError('auto current fleet did not attest its model count')
            if not isinstance(copies, int) or copies < 1:
                raise ValueError("every arm must declare a positive model_processes count")
            if arm.get("split") and not metadata.get("split"):
                raise ValueError("split arm did not attest a split frontend")
            if arm.get("split") and metadata.get("model_processes") != copies:
                raise ValueError("split model-process count differs from the declared arm")
            if metadata.get('benchmark_current') and metadata.get('model_processes') != copies:
                raise ValueError('current model-process count differs from the declared arm')
            persist_startup(copies)
            if measured:
                relay = subprocess.Popen([sys.executable, "-m", "exp.decouple.latency_proxy",
                                          "--port", str(arm["port"]), "--backend-port", str(backend_port),
                                          "--out", str(out / "latencies.jsonl")],
                                         stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
                while True:
                    check_package_deadline()
                    if time.monotonic() >= deadline:
                        raise TimeoutError('latency relay startup timed out')
                    if relay.poll() is not None or server.poll() is not None:
                        raise RuntimeError("server or latency relay exited during startup")
                    try:
                        with connect(f"ws://127.0.0.1:{arm['port']}", open_timeout=5, compression=None) as client:
                            client.recv(timeout=5)
                        break
                    except (OSError, TimeoutError):
                        if time.monotonic() > deadline:
                            raise TimeoutError("latency relay startup timed out")
                        time.sleep(0.25)
            processes = {}
            def sample():
                while not stop.wait(0.5):
                    try:
                        parent = psutil.Process(server.pid)
                        family = [parent, *parent.children(recursive=True)]
                        if relay is not None:
                            family.append(psutil.Process(relay.pid))
                        cpu = rss = 0.
                        for process in family:
                            if process.pid not in processes:
                                processes[process.pid] = process
                                process.cpu_percent(None)
                            cpu += processes[process.pid].cpu_percent(None)
                            rss += process.memory_info().rss / (1 << 20)
                        utilization = nvml.nvmlDeviceGetUtilizationRates(gpu)
                        memory = nvml.nvmlDeviceGetMemoryInfo(gpu)
                        samples.append(dict(t=time.monotonic(), gpu_util_pct=utilization.gpu,
                                            gpu_memory_mb=memory.used / (1 << 20), cpu_proc_pct=cpu,
                                            host_rss_mb=rss, system_cpu_pct=psutil.cpu_percent(None)))
                        resource_log.write(json.dumps(samples[-1])+'\n')
                        resource_log.flush()
                        if memory.used / (1 << 20) > budget_mb:
                            sampling_errors.append('common GPU memory budget exceeded')
                    except psutil.Error:
                        pass
                    except Exception as error:
                        sampling_errors.append(f'GPU/CPU sampler failed: {error}')
                        break
            thread = threading.Thread(target=sample)
            thread.start()
            start = time.monotonic()
            worker = subprocess.Popen(expand(arm["worker_command"]), stdout=worker_log, stderr=subprocess.STDOUT,
                                      env=dict(os.environ, **arm.get('worker_env', {})), start_new_session=True)
            worker_deadline = min(time.monotonic() + arm.get('worker_timeout_s', 1800), arm.get('package_deadline', float('inf')))
            while worker.poll() is None:
                check_package_deadline()
                if sampling_errors:
                    raise RuntimeError(sampling_errors[0])
                if server.poll() is not None or relay is not None and relay.poll() is not None:
                    raise RuntimeError('owned model fleet or latency relay exited')
                if time.monotonic() > worker_deadline:
                    raise TimeoutError('arm worker/package wall budget exceeded')
                time.sleep(.25)
            code = worker.returncode
            elapsed = time.monotonic() - start
            stop.set()
            thread.join(2)
            if code:
                raise RuntimeError(f"worker exited {code}; inspect {out / 'worker.log'}")
            if sampling_errors:
                raise RuntimeError(sampling_errors[0])
            if arm.get('case'):
                # Runtime and offline re-analysis share the exact collector.
                # All inputs are already durable if collection raises.
                from exp.decouple.collect_gpu import collect
                result = collect(out)
                write_json(out/'metrics.json',result)
                return result
            measured_uids = None
            window = None
            if arm.get('measurement_json'):
                window = json.loads(Path(expand_tokens(arm['measurement_json'], out=out)).read_text())
                if not window['start'] or not window['end'] or window['end'] <= window['start']:
                    raise ValueError('incomplete warmup-excluded measurement window')
                if window['manifest_sha256'] != arm['manifest_sha256']:
                    raise ValueError('measurement manifest identity differs')
                elapsed = window['end'] - window['start']
                measured_uids = set(window['measured_uids'])
            if arm.get("journal"):
                records = [json.loads(line) for line in Path(expand_tokens(arm['journal'], out=out)).read_text().splitlines()]
                accepted = accepted_cohort(records, measured_uids)
                summary = {'episodes_completed':len(accepted),
                           'episode_failures':sum(not r.get('success', False) for r in accepted.values()),
                           'episode_error_records':sum(bool(r.get('error')) and r.get('accepted') is not False
                                                       and (measured_uids is None or r['task_uid'] in measured_uids) for r in records)}
                if measured_uids is not None and set(accepted) != measured_uids:
                    raise ValueError('accepted episodes differ from the complete sealed B manifest')
            else:
                summary = json.loads(Path(expand_tokens(arm['summary_json'], out=out)).read_text())
                accepted = None
            if arm.get('expected_episodes') and summary['episodes_completed'] != arm['expected_episodes']:
                raise ValueError('accepted episode target differs')
            if measured:
                rows = [json.loads(line) for line in (out / "latencies.jsonl").read_text().splitlines()]
                lat = np.asarray([r["latency_ms"] for r in rows if r["ok"] and
                                  (accepted is None or r["uid"] in accepted and
                                   r["attempt"] == accepted[r["uid"]].get("attempt"))], np.float64)
            else:
                lat = np.asarray(summary["decision_latencies_ms"], np.float64)
            measured_samples = [s for s in samples if window is None or window['start'] <= s['t'] <= window['end']]
            if not len(lat) or not measured_samples or not summary["episodes_completed"]:
                raise ValueError("benchmark has no complete episodes/latencies/utilization samples")
            peak = max(startup_peak, max(s['gpu_memory_mb'] for s in samples))
            result = dict(name=arm["name"], model_processes=copies, elapsed_s=elapsed,
                          logical_arm=arm.get('logical_arm', arm['name']), model=arm.get('model'),
                          split=arm.get('split', False), fronts=arm.get('fronts'),
                          manifest_sha256=arm.get('manifest_sha256'), warmup_excluded=window is not None,
                          measurement_window=window, episode_failures=summary.get('episode_failures', 0),
                          episode_error_records=summary.get('episode_error_records', 0),
                          episodes_completed=summary["episodes_completed"],
                          episodes_per_min=60 * summary["episodes_completed"] / elapsed,
                          latency_p50_ms=float(np.percentile(lat, 50)), latency_p99_ms=float(np.percentile(lat, 99)),
                          latency_scope=("benchmark relay ingress-to-response-send; worker network excluded" if measured else
                                         summary.get("latency_scope", "worker supplied")),
                          gpu_util_mean_pct=float(np.mean([s["gpu_util_pct"] for s in measured_samples])),
                          cpu_proc_mean_pct=float(np.mean([s["cpu_proc_pct"] for s in measured_samples])),
                          system_cpu_mean_pct=float(np.mean([s["system_cpu_pct"] for s in measured_samples])),
                          gpu_memory_peak_mb=peak, gpu_memory_budget_mb=budget_mb, within_budget=peak <= budget_mb,
                          gpu_name=name, metadata=metadata, samples=samples)
            (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
            if peak > budget_mb:
                raise ValueError(f"{arm['name']} exceeded common GPU memory budget ({peak:.0f}>{budget_mb} MiB)")
            return {k: v for k, v in result.items() if k not in ("samples", "metadata")}
        finally:
            stop.set()
            try:
                stop_owned_groups([('server', server), ('relay', relay), ('worker', worker)],
                                  hard_deadline=arm.get('package_shutdown_deadline', float('inf')),
                                  wall_deadline=arm.get('package_wall_shutdown_deadline', float('inf')))
            finally:
                if thread is not None:
                    thread.join(2)
                nvml.nvmlShutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-memory-budget-mb", type=float, default=23000)
    args = parser.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        parser.error("GPU benchmark is coordinator-only; CUDA_VISIBLE_DEVICES must select the 4090")
    config = json.loads(Path(args.config).read_text())
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for arm in config['arms']:
        try:
            results.append(run_arm(arm, root, args.gpu_index, args.gpu_memory_budget_mb))
        except BaseException:
            print(f'PKG_ARM_FAIL {arm.get("logical_arm", arm["name"])}', flush=True)
            raise
    (root / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
