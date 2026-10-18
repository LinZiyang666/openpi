"""Real R17 socket clients, deterministic logical requests, concurrent scheduling.

Recorded frames are replayed as a declared stress sequence, NOT a reconstructed
physical trajectory. This exercises protocol/clock/cache state but cannot claim
trajectory coverage, independent query counts or closed-loop SR.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import threading
import time
import traceback
import numpy as np
from artifacts import save_tree, write_json
from capture import read_inputs


def decision_seed(benchmark, task, init, seq, seed):
    if benchmark.get('seed_mode') == 'fixed':
        return benchmark['seed']
    prefix = 'libero' if benchmark.get('benchmark') == 'libero' else 'mt50'
    return int.from_bytes(hashlib.sha256(f'{prefix}:{seed}:{task}:{init}:{seq}'.encode()).digest()[:4], 'little')


def ack(seq, actions, replan):
    a = np.asarray(actions)
    if a.ndim != 2 or not len(a) or not np.isfinite(a).all():
        raise ValueError('invalid action chunk')
    return dict(decision_seq=seq, executed_count=min(replan, len(a)), terminal=False)


def drive(client_factory, rows, benchmark, *, clients, deliveries, round_index,
          bundle_id, output, timeout=600):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    barrier = threading.Barrier(clients, timeout=timeout)
    stop = threading.Event()
    failure_lock = threading.Lock()
    def episode(i):
        client = client_factory()
        try:
            selected = client.select_bundle(bundle_id)
            if selected.get('__ack__') == 'error':
                raise ValueError(selected)
            base = rows[i % len(rows)]['request']
            context = {**base, 'campaign': 'xgpu_concurrent', 'arm': bundle_id,
                'bundle_generation': bundle_id, 'attempt': 1,
                'benchmark_protocol': base['benchmark_protocol'],
                'ack_protocol': 'r17.execution.v1', 'partial_tail_protocol': 'r17.partial_tail.v1'}
            # Different clients have different noise, stable across GPU/round.
            context['seed'] = benchmark.get('seed', 7) if benchmark.get('seed_mode') == 'fixed' else 7000 + i
            started = client.episode_start(experiment='xgpu_concurrent', task=base['task_name'],
                episode_id=i, episode_name=f'client_{i:04d}', extra_metadata=context)
            if started.get('__ack__') == 'error':
                raise ValueError(started)
            # Protocol declarations are episode metadata, never RequestContext.
            context.pop('ack_protocol')
            context.pop('partial_tail_protocol')
            barrier.wait()
            rng = random.Random(round_index * 100000 + i)
            previous, records = None, []
            for seq in range(deliveries):
                if stop.is_set():
                    raise RuntimeError('another client failed')
                time.sleep(rng.uniform(0, .010))
                row = rows[(i + seq) % len(rows)]
                # Do not mix another task's images into this episode.
                if row['request']['task_name'] != base['task_name']:
                    raise ValueError('this stress sequence requires one recorded task')
                req = {**context, 'decision_seq': seq,
                    'explicit_seed': decision_seed(benchmark, base['task_name'], base['original_init_idx'], seq, context['seed'])}
                wire = dict(row['observation'], __request__=req)
                if previous is not None:
                    wire['__execution__'] = previous
                t = time.monotonic()
                response = client.infer(wire)
                if response.get('__ack__') == 'error':
                    raise ValueError(response)
                previous = ack(seq, response['actions'], benchmark['replan_steps'])
                name = f'client_{i:04d}_delivery_{seq:04d}'
                seal = save_tree(output / name, dict(request=req, input_index=(i+seq) % len(rows),
                    response=response, execution=previous, latency_seconds=time.monotonic()-t))
                records.append(dict(key=[bundle_id, i, seq], file=name, sha256=seal))
            ended = client.episode_end(success=False, execution=dict(previous, terminal=True))
            if ended.get('__ack__') == 'error':
                raise ValueError(ended)
            return records
        except BaseException as error:
            with failure_lock:
                if not stop.is_set():
                    write_json(output/'FIRST_ERROR.json',dict(client=i,error=repr(error),traceback=traceback.format_exc()))
                stop.set()
            barrier.abort()
            raise
        finally:
            client.close()
    with ThreadPoolExecutor(max_workers=clients) as pool:
        futures = [pool.submit(episode, i) for i in range(clients)]
        result = [row for future in futures for row in future.result(timeout=timeout)]
    write_json(output / 'ROWS.json', dict(rows=result, round=round_index, bundle_id=bundle_id,
        clients=clients, deliveries=deliveries, scheduling_seed=round_index,
        physical_trajectory=False, independent_query_count=None))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('inputs', 'benchmark', 'output', 'bundle-id'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--clients', type=int, required=True)
    p.add_argument('--deliveries', type=int, default=24)
    p.add_argument('--round', type=int, default=0)
    a = p.parse_args()
    manifest, rows = read_inputs(a.inputs)
    from exp.r17_integration.common.websocket_policy_server import WebsocketClientPolicy
    benchmark = json.loads(Path(a.benchmark).read_text())
    records = drive(lambda: WebsocketClientPolicy('127.0.0.1', a.port), rows, benchmark,
        clients=a.clients, deliveries=a.deliveries, round_index=a.round,
        bundle_id=a.bundle_id, output=a.output)
    print(json.dumps(dict(rows=len(records), input_identity=manifest['identity'])))


if __name__ == '__main__':
    main()
