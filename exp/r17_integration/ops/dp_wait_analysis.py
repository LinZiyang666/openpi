#!/usr/bin/env python3
"""Offline DP wall-time accounting; never contacts a node or GPU."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

from exp.r17_integration.ops.perf_recorder import quantiles


def rows(path):
    try:
        with Path(path).open() as stream:
            for line in stream:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue
    except OSError:
        return


def union(intervals, start, end):
    intervals = sorted((max(start, a), min(end, b)) for a, b in intervals if b > start and a < end)
    total = 0
    finish = start
    gaps = []
    for a, b in intervals:
        if a > finish:
            gaps.append(a-finish)
        total += max(0, b-max(a, finish))
        finish = max(finish, b)
    if finish < end:
        gaps.append(end-finish)
    return total, gaps


def analyze(campaign, root, start, end):
    attempts = list(rows(campaign/'driver/attempts.jsonl'))
    dispatches = list(rows(campaign/'driver/worker_dispatch.jsonl'))
    accepted = {(r['task_uid'], r.get('attempt', 1)): r for r in attempts if r.get('accepted') and not r.get('error')}
    dispatch = {(r['task_uid'], r.get('attempt', 1)): r for r in dispatches}
    serving = list(rows(campaign/'servers/replica_0/serving.jsonl'))
    cohort = {key for key, r in accepted.items() if start <= r['time'] <= end}
    calls = []
    clients = {}
    by_decision = {}
    for r in rows(campaign/'driver/per_step.jsonl'):
        if not r.get('accepted'):
            continue
        key = r.get('task_uid'), r.get('attempt', 1)
        if r.get('_kind') == 'client_timing':
            clients[key] = r
        elif not r.get('_kind'):
            by_decision[(r['task_uid'], r['decision_seq'])] = r
            if key in cohort:
                calls.append(r)
    matched = cohort & clients.keys()
    busy = sum(accepted[k]['duration_s'] for k in matched)
    sums = {key: sum(clients[k].get(key, 0) for k in matched) for key in ('infer_ms', 'env_ms', 'img_ms', 'infers')}
    ws = sum(r.get('server_timing', {}).get('infer_ms', 0) for r in calls)
    runner_other = busy*1000-sums['infer_ms']-sums['env_ms']-sums['img_ms']
    delay = [max(0, accepted[k]['time']-dispatch[k]['time']-accepted[k]['duration_s']) for k in matched if k in dispatch]
    sequences = defaultdict(list)
    for r in dispatches:
        sequences[r['worker_id']].append(r)
    cycles = []
    gaps = []
    for seq in sequences.values():
        for previous, current in zip(seq, seq[1:]):
            key = previous['task_uid'], previous.get('attempt', 1)
            if start <= previous['time'] and current['time'] <= end and key in accepted and key in clients:
                duration = accepted[key]['duration_s']
                cycle = current['time']-previous['time']
                if cycle >= duration:
                    cycles.append(dict(key=key, seconds=cycle, runner=duration, client=clients[key]))
                    gaps.append(max(0, current['time']-accepted[key]['time']))
    cycle_seconds = sum(c['seconds'] for c in cycles)
    cycle_runner = sum(c['runner'] for c in cycles)
    cycle_ms = {key: sum(c['client'].get(key, 0) for c in cycles) for key in ('infer_ms', 'env_ms', 'img_ms')}
    cycle_accounting = dict(n=len(cycles), total_worker_seconds=cycle_seconds,
        runner_pct=100*cycle_runner/cycle_seconds if cycle_seconds else None,
        policy_infer_pct=100*cycle_ms['infer_ms']/1000/cycle_seconds if cycle_seconds else None,
        simulation_image_pct=100*(cycle_ms['env_ms']+cycle_ms['img_ms'])/1000/cycle_seconds if cycle_seconds else None,
        episode_setup_controls_other_pct=100*(cycle_runner-sum(cycle_ms.values())/1000)/cycle_seconds if cycle_seconds else None,
        between_episode_driver_and_recycle_pct=100*(cycle_seconds-cycle_runner)/cycle_seconds if cycle_seconds else None)
    worker_samples = [r for r in rows(root/'dp_workers_probe.jsonl') if r.get('kind') == 'sample' and start <= r['time'] <= end]
    worker_counts = Counter()
    kernel = Counter()
    cpu_deltas = []
    last = {}
    policy_queues, driver_queues = [], []
    for sample in worker_samples:
        n = len(sample['processes'])
        worker_counts['missing_recycle_or_census_lag'] += max(0, 32-n)
        for p in sample['processes']:
            kernel[str(p['main_wchan'])] += 1
            tcp = [s for s in p['sockets'] if s.get('state') == '01']
            driver_tcp = [s for s in tcp if s['remote'].endswith(':5A95')]
            policy_tcp = [s for s in tcp if s['remote'].endswith(':5A8C')]
            for ss, output in ((driver_tcp, driver_queues), (policy_tcp, policy_queues)):
                output.extend(s['tx_bytes']+s['rx_bytes'] for s in ss)
            wchan = str(p['main_wchan'])
            if not driver_tcp:
                kind = 'startup_attestation'
            elif wchan == 'wait_woken':
                kind = 'driver_reply_wait'
            elif 'futex' in wchan and policy_tcp:
                kind = 'policy_response_or_control_wait'
            else:
                kind = 'simulation_render_episode_setup_other'
            worker_counts[kind] += 1
            ident = p['pid'], p['start_ticks']
            if ident in last:
                old_time, old_ticks = last[ident]
                dt = sample['time']-old_time
                if dt > 0:
                    cpu_deltas.append(100*(p['ticks']-old_ticks)/100/dt)
            last[ident] = sample['time'], p['ticks']
    denominator = len(worker_samples)*32
    worker_pct = {k: 100*v/denominator for k, v in worker_counts.items()} if denominator else {}
    local_samples = [r for r in rows(root/'dp_local_probe.jsonl') if r.get('kind') == 'sample' and start <= r['time'] <= end]
    gpu_util, gpu_watts = [], []
    fsync = queued = idle_fsync = idle = 0
    local_cost = []
    for sample in local_samples:
        util = None
        if sample.get('gpu', {}).get('stdout'):
            values = sample['gpu']['stdout'].splitlines()[0].split(',')
            util = float(values[1]); gpu_util.append(util); gpu_watts.append(float(values[2]))
        driver = next((p for p in sample['processes'] if p['pid'] == 2022909), {})
        waiting_fsync = any('jbd2' in str(t['wchan']) for t in driver.get('threads', []))
        fsync += waiting_fsync
        queued += any(s.get('rx_bytes', 0)>0 for s in driver.get('sockets', []))
        if util == 0:
            idle += 1; idle_fsync += waiting_fsync
        local_cost.append(sample['probe_seconds'])
    # Recover the generic stage-loop dispatch from per-request host spans.
    # Ledger is stamped after adapter execution, before reply publication.
    # Thus these are estimates, not CUDA event measurements. OFF monitor has
    # no true enqueue/dispatch timestamps. Use single-stage, same-K batches.
    stages = defaultdict(dict)
    for r in serving:
        step = by_decision.get((r['task_uid'], r['decision_seq']))
        for stage, receipt in r.get('batching', {}).get('stages', {}).items():
            b = stages[stage].setdefault(receipt['batch_id'], dict(time=receipt['time'], size=receipt['size'], enqueues=[]))
            ms = step.get('stage_ms', {}).get(stage) if step else None
            if ms is not None:
                b['enqueues'].append(receipt['time']-ms/1000)
    stage_analysis = {}
    queue_estimate = []
    for stage, mapping in stages.items():
        batch_rows = sorted(mapping.values(), key=lambda r:r['time'])
        intervals = []; waits = []; sizes = []; sizes_window=[]; spans=[]
        previous_end = 0
        complete = incomplete = 0
        for b in batch_rows:
            t = b['time']
            if start <= t <= end:
                sizes_window.append(b['size'])
            if len(b['enqueues']) != b['size']:
                if start <= t <= end: incomplete += 1
                previous_end=t
                continue
            first = min(b['enqueues'])
            window = .005 if b['size'] < 16 else 0
            dispatch_t = min(t, max(first, previous_end)+window)
            if start <= t <= end:
                complete += 1
                intervals.append((dispatch_t, t))
                sizes.append(b['size']); spans.append((t-dispatch_t)*1000)
                waits.extend(max(0, dispatch_t-e)*1000 for e in b['enqueues'])
            previous_end=t
        inflight, stage_gaps = union(intervals, start, end)
        stage_analysis[stage] = dict(batch_sizes=quantiles(sizes_window), under_cap_pct=100*sum(s<16 for s in sizes_window)/len(sizes_window) if sizes_window else None,
            reconstructed_batches=complete, unreconstructed_batches=incomplete,
            batch_host_execution_pct_estimate=100*inflight/(end-start), batch_host_ms_estimate=quantiles(spans),
            queue_wait_ms_estimate=quantiles(waits), idle_gaps_seconds_estimate=quantiles(stage_gaps))
    # End-to-end server request residency, including all queues, is directly
    # measured and should not be labelled GPU/batch execution.
    server_inflight, server_gaps = union([(r['time']-r['elapsed_seconds'], r['time']) for r in serving], start, end)
    in_window = [r for r in serving if start <= r['time'] <= end]
    report = dict(campaign=campaign.name, start=start, end=end, seconds=end-start,
        accepted_episodes=len(cohort), episodes_per_min=len(cohort)*60/(end-start), server_policy_calls=len(in_window),
        policy_calls_per_s=len(in_window)/(end-start), matched_completed_episodes=len(matched),
        completed_episode_wall=dict(runner_seconds=busy, **sums, residual_setup_controls_other_ms=runner_other),
        mean_call_ms=dict(client=sums['infer_ms']/sums['infers'] if sums['infers'] else None,
            server_websocket=ws/len(calls) if calls else None,
            transport_serialization_and_protocol=sums['infer_ms']/sums['infers']-ws/len(calls) if sums['infers'] and calls else None,
            stage_host_wall_ms={key: quantiles([r.get('stage_ms',{}).get(key) for r in calls]) for key in ('prepare','stage1','stage2','stage3','to_wire')}),
        result_durability_delay_s=quantiles(delay), accepted_result_to_next_dispatch_s=quantiles(gaps), closed_worker_cycle_accounting=cycle_accounting,
        worker_probe=dict(samples=len(worker_samples), seconds=worker_samples[-1]['time']-worker_samples[0]['time'] if worker_samples else None,
            wall_pct=worker_pct, kernel_counts=dict(kernel), cpu_one_core_pct=quantiles(cpu_deltas), policy_socket_queue_bytes=quantiles(policy_queues), driver_socket_queue_bytes=quantiles(driver_queues),
            max_probe_seconds=max((r['probe_seconds'] for r in worker_samples), default=None)),
        local_probe=dict(samples=len(local_samples), gpu_util=quantiles(gpu_util), gpu_watts=quantiles(gpu_watts),
            driver_fsync_wall_pct=100*fsync/len(local_samples) if local_samples else None,
            driver_unread_result_queue_wall_pct=100*queued/len(local_samples) if local_samples else None,
            gpu_zero_util_samples=idle, gpu_zero_util_with_driver_fsync_samples=idle_fsync, cost_seconds=quantiles(local_cost)),
        server=dict(any_request_inflight_pct=100*server_inflight/(end-start), no_request_gap_s=quantiles(server_gaps), stages=stage_analysis),
        errors=sum(bool(r.get('error')) for r in attempts if start<=r['time']<=end), retries=sum(r.get('attempt',1)>1 for r in attempts if start<=r['time']<=end))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('/data/mw_finetune/perf/2026-10-10'))
    parser.add_argument('--start', type=float, required=True)
    parser.add_argument('--end', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(args.root.resolve()):
        parser.error('output must be inside --root')
    args.output.write_text(json.dumps(analyze(args.campaign, args.root, args.start, args.end), indent=2)+'\n')
