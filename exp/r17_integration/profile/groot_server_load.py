"""In-process load on one GR00T MetaWorld resident: N connection threads, B1 (resident lock) or stage batching.

Mirrors formal_server.serve without the WebSocket: the same model_runtime.load, connection factory,
resident lock (B1) or BatchRuntime (B>1). ``--mode hit`` runs a cache-hit decision (prepare, stage1,
retrieval fields, state, to_wire); ``--mode miss`` runs ConnectionPolicy.infer on the formal full.yaml
(prepare, stage1, stage2, stage3, to_wire). Synthetic 480x480 MetaWorld frames, real task prompts.

  python -m exp.r17_integration.profile.groot_server_load --assets DIR --threads 20 --seconds 60 \
      [--max-batch 8 --batch-wait-ms 5] [--mode hit|miss] [--actions-out x.npz] --out result.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import statistics
import threading
import time

import numpy as np


def observation(thread, seq, prompt):
    rng = np.random.default_rng(thread * 100003 + seq)
    image = rng.integers(0, 256, (480, 480, 3), dtype=np.uint8)
    state = rng.uniform(-0.5, 0.5, 4).astype(np.float32)
    return {'observation/image': image, 'observation/state': state, 'prompt': prompt}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--assets', required=True)
    p.add_argument('--threads', type=int, default=20)
    p.add_argument('--seconds', type=float, default=60)
    p.add_argument('--warmup', type=float, default=15)
    p.add_argument('--max-batch', type=int, default=1)
    p.add_argument('--batch-wait-ms', type=float, default=5)
    p.add_argument('--mode', choices=('hit', 'miss'), default='hit')
    p.add_argument('--no-gate', action='store_true', help='batched: fully concurrent connections (no resident lock)')
    p.add_argument('--executor', action='store_true', help='B1: one dedicated thread runs every request (connections wait)')
    p.add_argument('--think-ms', type=float, default=0, help='per-connection pause between requests (env stepping)')
    p.add_argument('--actions-out')
    p.add_argument('--out', required=True)
    a = p.parse_args()

    import torch
    from exp.r17_integration.qualification.model_runtime import load
    from exp.r17_integration.common.config import load_cache_config, build_shared_storage
    from exp.r17_integration.common.serving import make_connection_factory
    from exp.r17_integration.common.contracts import RequestContext
    assets = Path(a.assets)
    bench = json.loads((assets / 'benchmark.json').read_text())
    adapter, provider = load(dict(model='groot', domain='metaworld', assets=str(assets),
                                  benchmark_spec=str(assets / 'benchmark.json')), 'cuda:0')
    config = load_cache_config(assets / 'full.yaml')
    storage = build_shared_storage(config)
    runtime, kwargs = None, {}
    if a.max_batch > 1:
        from exp.r17_integration.common.batch_runtime import BatchedAdapter, create_runtime
        runtime = create_runtime(adapter, a.max_batch, a.batch_wait_ms, device='cuda:0')
        kwargs['adapter_factory'] = lambda: BatchedAdapter(provider.load_adapter(), runtime)
    factory = make_connection_factory(provider, config, storage, **kwargs)
    if runtime is not None:
        factory = runtime.wrap_connection_factory(factory)
    resident_lock = threading.RLock()
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=1) if a.executor else None
    if runtime is not None and not a.no_gate:
        runtime.gate = resident_lock
    tasks = bench['tasks']
    protocol = config.adapter_binding['benchmark_protocol']
    stop = threading.Event()
    measuring = threading.Event()
    rows, actions, errors, per_thread = [], {}, [], {}
    sink = threading.Lock()

    # Frames are generated up front: generating them inside the load threads holds the GIL and
    # would slow the lock holder (the live servers' waiting connections are blocked, not runnable).
    pools = [[observation(i, j, tasks[i % len(tasks)][1]) for j in range(8)] for i in range(a.threads)]

    def worker(i):
        task_id = i % len(tasks)
        name, prompt = tasks[task_id][0], tasks[task_id][1]
        conn = factory()
        conn.on_task_begin()
        conn.on_episode_start(task_id=task_id, init_state_id=i, episode_id=i, task=name,
                              extra_metadata=dict(campaign='sprof', task_name=name, original_init_idx=i,
                                                  task_id=task_id, seed=7, explicit_seed=1000 + i))
        adapter_i = conn.adapter
        seq = 0
        while not stop.is_set():
            obs = pools[i][seq % len(pools[i])]
            begin = time.perf_counter()
            try:
                def body():
                  with resident_lock if runtime is None or not a.no_gate else contextlib.nullcontext():
                    inside = time.perf_counter()
                    if a.mode == 'miss':
                        out = conn.infer(obs)
                        stage = dict(out['__hit_meta__']['stage_ms'])
                        sizes = {k: v['size'] for k, v in ((out['__hit_meta__'].get('batching') or {}).get('stages') or {}).items()}
                        act = out['actions']
                    else:
                        stage = {}
                        req = RequestContext('sprof', protocol, name, i, seq, task_id=task_id, explicit_seed=1000 + i)
                        t = time.perf_counter(); state = adapter_i.extract_state_only(obs); stage['state'] = time.perf_counter() - t
                        t = time.perf_counter(); prepared = adapter_i.prepare(obs, req); stage['prepare'] = time.perf_counter() - t
                        t = time.perf_counter(); s1 = adapter_i.run_stage1(prepared); named = adapter_i.retrieval_fields(s1)
                        stage['stage1'] = time.perf_counter() - t
                        sizes = getattr(adapter_i, 'last_batch_sizes', {})
                        del named, state
                        chunk = np.zeros((16, 32), np.float32)
                        t = time.perf_counter(); act = adapter_i.to_wire(chunk, prepared).actions; stage['to_wire'] = time.perf_counter() - t
                        stage = {k: v * 1000 for k, v in stage.items()}
                    return inside, stage, sizes, act
                inside, stage, sizes, act = executor.submit(body).result() if executor else body()
                stage.update({'batch_' + k: float(v) for k, v in sizes.items()})
                end = time.perf_counter()
            except Exception as exc:  # recorded, the load continues
                errors.append(repr(exc)[:300])
                time.sleep(0.05)
                continue
            if measuring.is_set():
                with sink:
                    per_thread[i] = per_thread.get(i, 0) + 1
                    rows.append(dict(t=end, wall_ms=(end - begin) * 1000, service_ms=(end - inside) * 1000, **stage))
                    if a.actions_out and seq < 4:
                        actions[f'{i}_{seq}'] = np.asarray(act, np.float32)
            seq += 1
            if a.think_ms:
                time.sleep(a.think_ms / 1000)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(a.threads)]
    for t in threads:
        t.start()
    time.sleep(a.warmup)
    measuring.set()
    t0 = time.perf_counter()
    time.sleep(a.seconds)
    measuring.clear()
    span = time.perf_counter() - t0
    stop.set()
    for t in threads:
        t.join(timeout=30)
    if runtime is not None:
        runtime.close()
    keys = sorted({k for r in rows for k in r} - {'t'})
    result = dict(mode=a.mode, threads=a.threads, max_batch=a.max_batch, batch_wait_ms=a.batch_wait_ms,
                  torch_threads=torch.get_num_threads(), seconds=span, requests=len(rows),
                  no_gate=a.no_gate, executor=a.executor, req_per_s=len(rows) / span, errors=len(errors), first_errors=errors[:3], think_ms=a.think_ms,
                  active_threads=len(per_thread), per_thread_min=min(per_thread.values(), default=0),
                  per_thread_max=max(per_thread.values(), default=0),
                  mean_ms={k: statistics.mean(r[k] for r in rows if k in r) for k in keys} if rows else {},
                  p50_ms={k: statistics.median(r[k] for r in rows if k in r) for k in keys} if rows else {})
    Path(a.out).write_text(json.dumps(result, indent=1))
    if a.actions_out:
        np.savez(a.actions_out, **actions)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
