"""Unprofiled CPU method/decision/log/serialization scaling on recorded B rows.

Model/transform costs are reported separately by profile_cpu. The zero-wait
inline stage stub makes this a CPU service-demand benchmark, not a GPU forecast.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import time

_barrier = None


def initialize(barrier):
    global _barrier
    _barrier = barrier


def worker(model, directory, decisions, index):
    import numpy as np
    from openpi_client import msgpack_numpy
    from exp.decouple.recorded import load
    from exp.decouple.replay import frozen_runtime, make_connections, tape
    from openpi.serving.batching_coordinator import Pi05StageBatcher
    from tests.serving.decouple.stubs import PiModel
    class Inline:
        def __init__(self):
            self.batcher = Pi05StageBatcher(PiModel(), "cpu")
        def submit_to_stage(self, stage, bundle, payload):
            if stage == 1:
                return self.batcher.run_stage1_batch([payload])[0]
            if stage == 2:
                return self.batcher.run_stage2_batch([payload], capture=False)[0]
            return self.batcher.run_stage3_miss([payload], num_steps=payload.num_steps,
                                               save_timesteps_per_request=[None])[0]
    rt, fit = frozen_runtime(model, Path(directory) / f"front_{index}")
    from openpi.serving.split_state import bind_runtime
    bind_runtime(rt, Path(directory) / "state", identity=dict(model=model))
    data = load(model, limit=2)
    connections = make_connections(rt, data, Inline(), count=2)
    frames = list(tape(data, steps=min(decisions // 2, 32)))
    # Warm all lazily built indexes/method state, then start fresh episodes.
    for ci, start, obs in frames[:2]:
        connections[ci][0].on_episode_start(**start)
        connections[ci][0].infer(dict(obs))
        connections[ci][0].on_episode_end(success=False)
    latencies = []
    if _barrier is not None:
        _barrier.wait(timeout=120)
    begin = time.monotonic()
    cpu_begin = time.process_time()
    for i in range(decisions):
        ci, start, obs = frames[i % len(frames)]
        conn, session, _ = connections[ci]
        if i % len(frames) < 2 or session.pending:
            conn.on_episode_start(**start)
        clock = time.perf_counter_ns()
        wire = msgpack_numpy.Packer().pack(obs)
        output = conn.infer(msgpack_numpy.unpackb(wire))
        msgpack_numpy.Packer().pack(output)
        latencies.append((time.perf_counter_ns() - clock) / 1e6)
    cpu = time.process_time() - cpu_begin
    end = time.monotonic()
    for conn, _, _ in connections:
        conn.on_episode_end(success=False)
    return dict(begin=begin, end=end, cpu_s=cpu, decisions=decisions, latencies_ms=latencies,
                fit=str(fit), source=str(data.path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("pi05", "groot"), required=True)
    parser.add_argument("--fronts", default="1,2,4,8")
    parser.add_argument("--decisions-per-front", type=int, default=128)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    counts = [int(n) for n in args.fronts.split(",")]
    if not counts or min(counts) < 1 or args.decisions_per_front < 2:
        parser.error("positive fronts and at least two decisions per front required")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for n in counts:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n, mp_context=context,
                                 initializer=initialize, initargs=(context.Barrier(n),)) as executor:
            futures = [executor.submit(worker, args.model, out / f"n{n}", args.decisions_per_front, i)
                       for i in range(n)]
            rows = [f.result() for f in futures]
        import numpy as np
        elapsed = max(r["end"] for r in rows) - min(r["begin"] for r in rows)
        lat = sum((r["latencies_ms"] for r in rows), [])
        report.append(dict(fronts=n, decisions=sum(r["decisions"] for r in rows), elapsed_s=elapsed,
                           decisions_per_s=sum(r["decisions"] for r in rows) / elapsed,
                           cpu_s=sum(r["cpu_s"] for r in rows),
                           cpu_ms_per_decision=1000 * sum(r["cpu_s"] for r in rows) / len(lat),
                           latency_p50_ms=float(np.percentile(lat, 50)), latency_p99_ms=float(np.percentile(lat, 99))))
    result = dict(model=args.model, scope="method/corrector/plugin/log/serialization with inline stage stubs; transforms excluded",
                  rows=report)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
