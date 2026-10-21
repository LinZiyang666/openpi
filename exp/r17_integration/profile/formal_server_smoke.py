"""Smoke formal_server.serve's real connection factory without a socket.

The WebSocket server class is replaced by a driver that opens N connections through the
factory serve() built and calls infer from N threads (each pausing --think-ms between
requests). Pure campaign (full.yaml): every request runs prepare/S1/S2/S3/to_wire.

  python -m exp.r17_integration.profile.formal_server_smoke --assets DIR --directory OUT \
      [--threads 20 --seconds 25 --think-ms 50] [--max-batch 8 --batch-wait-ms 5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import threading
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--assets', required=True)
    p.add_argument('--directory', required=True)
    p.add_argument('--threads', type=int, default=20)
    p.add_argument('--seconds', type=float, default=25)
    p.add_argument('--warmup', type=float, default=10)
    p.add_argument('--think-ms', type=float, default=50)
    p.add_argument('--max-batch', type=int, default=1)
    p.add_argument('--batch-wait-ms', type=float, default=5)
    a = p.parse_args()
    from exp.r17_integration.common import websocket_policy_server as wps
    from exp.r17_integration.ops import formal_server
    from exp.r17_integration.profile.groot_server_load import observation
    assets = Path(a.assets)
    bench = json.loads((assets / 'benchmark.json').read_text())
    tasks = bench['tasks']
    out = {}

    class Driver:
        def __init__(self, base, *, connection_policy_factory, ready_callback=None, **kwargs):
            self.factory, self.ready = connection_policy_factory, ready_callback

        def request_stop(self):
            pass

        def serve_forever(self, stop_on_request=True):
            if self.ready:
                self.ready()
            stop, measuring, rows, errors, lock = threading.Event(), threading.Event(), [], [], threading.Lock()
            threads_seen = set()

            def worker(i):
                task_id = i % len(tasks)
                name, prompt = tasks[task_id][0], tasks[task_id][1]
                pool = [observation(i, j, prompt) for j in range(8)]
                conn = self.factory(None)
                conn.on_task_begin()
                conn.on_episode_start(task_id=task_id, init_state_id=i, episode_id=i, task=name,
                                      extra_metadata=dict(campaign='smoke', task_name=name, original_init_idx=i,
                                                          task_id=task_id, seed=7, explicit_seed=2000 + i))
                seq = 0
                while not stop.is_set():
                    begin = time.perf_counter()
                    try:
                        result = conn.infer(pool[seq % 8])
                        assert result['actions'].shape[1] == 4
                    except Exception as exc:
                        errors.append(repr(exc)[:300]); time.sleep(0.05); continue
                    if measuring.is_set():
                        with lock:
                            rows.append((time.perf_counter() - begin) * 1000)
                            threads_seen.add(i)
                    seq += 1
                    time.sleep(a.think_ms / 1000)

            workers = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(a.threads)]
            for t in workers:
                t.start()
            time.sleep(a.warmup); measuring.set(); t0 = time.perf_counter()
            time.sleep(a.seconds); measuring.clear(); span = time.perf_counter() - t0; stop.set()
            for t in workers:
                t.join(timeout=30)
            out.update(requests=len(rows), req_per_s=len(rows) / span, wall_ms_mean=statistics.mean(rows) if rows else None,
                       errors=len(errors), first_errors=errors[:3], connections=len(threads_seen))

    wps.WebsocketPolicyServer = Driver
    campaign = dict(model='groot', domain='metaworld', assets=str(assets), device='cuda:0', replicas=1,
                    memory_budget_gib=20, server_node='h100', run_name='smoke')
    if a.max_batch > 1:
        campaign['stage_batching'] = dict(max_batch=a.max_batch, batch_wait_ms=a.batch_wait_ms)
    formal_server.serve(campaign, 0, a.directory)
    serving = [json.loads(l) for l in (Path(a.directory) / 'serving.jsonl').read_text().splitlines()]
    out.update(serving_rows=len(serving), service_ms_mean=1000 * statistics.mean(r['service_seconds'] for r in serving),
               max_batch=a.max_batch, think_ms=a.think_ms, threads=a.threads)
    print('SMOKE ' + json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
