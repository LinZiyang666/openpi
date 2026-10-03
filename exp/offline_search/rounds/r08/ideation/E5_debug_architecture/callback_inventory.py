"""Campaign metadata, capacity and telemetry audit; all outputs are scratch."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np

from exp.offline_search.debug import capacity, reader

ROOT = Path('/home/weiland/trace_runs/os_closed_loop/r08_main')
OUT = Path('/tmp/r8cb_E5_debug_architecture')


def quant(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    return dict(n=len(a), mean=float(a.mean()), p50=float(np.percentile(a, 50)),
                p95=float(np.percentile(a, 95)), p99=float(np.percentile(a, 99)),
                max=float(a.max()), min=float(a.min()), total=float(a.sum())) if len(a) else dict(n=0)


def put_count(dest, field, value):
    dest[field][str(value)] += 1


def scan(spec):
    name = spec['arm']
    arm = reader.open_arm(ROOT, name)
    arm.cache_enabled = False
    journal = arm.journal()
    accepted = journal.loc[journal.accepted.eq(True)].to_dict('records')
    accepted_keys = {r['episode_key'] for r in accepted}
    pairs = {(int(r['task_id']), int(r['init'])) for r in accepted}
    expected_pairs = {(t, i) for t in range(10) for i in range(50)}
    result = dict(arm=name, model=spec['model'], suite=spec['suite_short'],
        variant=spec['r8']['variant'], library_size=spec['r8'].get('library_size'),
        accepted=len(accepted), unaccepted=len(journal)-len(accepted),
        pair_set_exact=pairs == expected_pairs, successes=sum(bool(r['success']) for r in accepted),
        aug_done_at_start=(ROOT/'state'/(name+'.AUG_DONE')).exists(),
        published_validation=reader.read_json(arm.debug_dir/'validation.json'),
        published_verified=reader.read_json(arm.debug_dir/'verified.json'))
    status = defaultdict(Counter)
    episode_stats, sender_stats = defaultdict(list), defaultdict(list)
    summary = Counter()
    episodes = []
    snapshots = 0
    for outcome in accepted:
        ek = outcome['episode_key']
        directory = arm.debug_dir/'client'/ek
        ep = reader.read_json(directory/'episode.json')
        for key in ('env_seed', 'server_join', 'termination_reason', 'dispatch_gen'):
            put_count(status, 'episode.'+key, ep.get(key))
        for key, value in ep.get('capabilities', {}).items():
            put_count(status, 'capability.'+key, value.get('status'))
            if value.get('status') != 'available':
                put_count(status, 'capability_reason.'+key, value.get('reason'))
        for key in ('n_controls', 'n_decisions', 'settle_controls', 'control_dt'):
            if isinstance(ep.get(key), (int, float)):
                episode_stats[key].append(ep[key])
        episode_stats['wall_seconds'].append(ep['t_end']-ep['t_start'])
        summary['episode_capture_errors'] += len(ep.get('capture_errors', []))
        summary['episode_error'] += bool(ep.get('error'))
        snapshots += len(list(directory.glob('snap_*.npz')))
        complete = reader.read_json(arm.debug_dir/'receipts'/ek/'complete.json')
        ss = complete.get('sender_stats', {})
        summary['sender_stats_present'] += bool(ss)
        summary['spilled_attempts'] += bool(ss.get('spill'))
        for key, value in ss.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                sender_stats[key].append(value)
        if ss.get('spill_reason'):
            put_count(status, 'spill_reason', ss['spill_reason'])
        episodes.append(dict(episode_key=ek, task_id=ep['task_id'], init=ep['init'],
            decisions=ep['n_decisions'], controls=ep['n_controls'], settle=ep.get('settle_controls'),
            wall_seconds=ep['t_end']-ep['t_start'], reset_sha=ep.get('reset_state_sha256'),
            initial_sha=ep.get('initial_state_sha256'), env_seed=ep.get('env_seed')))
    catalog = arm.catalog()
    catalog_shas = set(catalog.lib_sha.astype(str)) if catalog is not None else set()
    summary['catalog_rows'] = len(catalog) if catalog is not None else 0
    times = defaultdict(list)
    diagnostic_examples = {}
    server_seq_by_file = {}
    decision_id_set = set()
    for directory in arm.server_dirs:
        for path in sorted(directory.glob('decisions*.jsonl')):
            seen_seq = []
            for rec, line_bytes in reader.iter_jsonl(path, arm.read_issues):
                seen_seq.append(rec.get('server_seq'))
                if rec.get('episode_key') not in accepted_keys:
                    summary['unaccepted_server_records'] += 1
                    continue
                summary['decisions'] += 1
                summary['accepted_server_json_bytes'] += line_bytes
                did = rec['decision_id']
                summary['duplicate_decision_ids'] += did in decision_id_set
                decision_id_set.add(did)
                summary['live_looks'] += int(rec.get('stage1_calls', 0))
                summary['live_policy_calls'] += int(rec.get('policy_calls', 0))
                summary['live_stage23_calls'] += int(rec.get('stage23_calls', 0))
                summary['camera_completions'] += int(rec.get('camera_completions', 0))
                summary['rawkeys_selected'] += bool(rec.get('rawkeys_sampled'))
                summary['owner_cost'] += float(rec.get('owner_cost', 0))
                for field in ('src', 'camera_mode', 'status', 'lib', 'lib_sha'):
                    put_count(status, field, rec.get(field))
                for field, value in rec.items():
                    if field.endswith('_status') and isinstance(value, dict):
                        if 'status' in value:
                            put_count(status, field, value['status'])
                            if value['status'] not in ('available', 'not_applicable', 'not_sampled'):
                                put_count(status, field+'.reason', value.get('reason'))
                        else:
                            for sub, val in value.items():
                                if isinstance(val, dict) and 'status' in val:
                                    put_count(status, field+'.'+sub, val['status'])
                src = str(rec.get('src'))
                for field in ('observer_ms', 'queue_wait_ms', 'infer_ms', 's1_ms', 's23_ms', 'method_ms', 'pre_ms'):
                    value = rec.get(field)
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        times[field].append(value)
                        times[src+'.'+field].append(value)
                rows, weights, scores = rec.get('rows') or [], rec.get('weights') or [], rec.get('scores') or []
                if rows:
                    summary['rows_present'] += 1
                    summary['rows_weights_length_mismatch'] += len(rows) != len(weights)
                    summary['rows_scores_length_mismatch'] += len(rows) != len(scores)
                    if len(rows) != len(scores):
                        diagnostic_examples.setdefault('rows_scores', dict(id=did, file=str(path), rows=len(rows), scores=len(scores), src=src))
                    summary['weight_sum_bad_1e_5'] += not np.isclose(sum(weights), 1., atol=1e-5, rtol=0.)
                    summary['negative_weights'] += any(w < 0 for w in weights)
                    summary['lib_sha_not_catalog_sha'] += rec.get('lib_sha') not in catalog_shas
                    if rec.get('lib_sha') not in catalog_shas:
                        diagnostic_examples.setdefault('lib_sha', dict(id=did, captured=rec.get('lib_sha'), catalog=sorted(catalog_shas), library=rec.get('lib')))
                if rec.get('policy_chunk_status', {}).get('status') != 'available' and rec.get('array_status', {}).get('policy_chunk', {}).get('status') == 'available':
                    summary['policy_semantic_vs_array_status'] += 1
                for key in ('stage_pre', 'pre_stage', 'p_effective', 'eligible', 'coin'):
                    summary['field_present.'+key] += rec.get(key) is not None
            server_seq_by_file[str(path.relative_to(arm.debug_dir))] = dict(rows=len(seen_seq),
                first=seen_seq[0] if seen_seq else None, last=seen_seq[-1] if seen_seq else None,
                contiguous=all(b == a+1 for a,b in zip(seen_seq, seen_seq[1:])) if all(isinstance(v,int) for v in seen_seq) else False)
    result.update(status_counts={k:dict(v) for k,v in status.items()}, counts=dict(summary),
        episode_stats={k:quant(v) for k,v in episode_stats.items()}, sender_stats={k:quant(v) for k,v in sender_stats.items()},
        timings={k:quant(v) for k,v in times.items()}, episodes=episodes, snapshots=snapshots,
        examples=diagnostic_examples, server_seq=server_seq_by_file, read_issues=list(arm.read_issues))
    return result


def main():
    specs = json.loads((ROOT/'arms.json').read_text())
    for sub in ('inventory', 'capacity'):
        (OUT/sub).mkdir(parents=True, exist_ok=True)
    snapshot = OUT/'augmentation_at_start.json'
    if not snapshot.exists():
        snapshot.write_text(json.dumps(dict(time=time.time(), arms=[s['arm'] for s in specs if (ROOT/'state'/(s['arm']+'.AUG_DONE')).exists()]), indent=2)+'\n')
    for spec in specs:
        name = spec['arm']
        start = time.time()
        print(json.dumps(dict(ev='start', arm=name)), flush=True)
        for mode in ('inventory', 'capacity'):
            path = OUT/mode/(name+'.json')
            if path.exists():
                continue
            try:
                if mode == 'inventory':
                    report = scan(spec)
                else:
                    arm = reader.open_arm(ROOT, name)
                    arm.cache_enabled = False
                    report = capacity.capacity(arm)
                report.update(audit_time=time.time())
            except Exception:
                report = dict(arm=name, error=traceback.format_exc())
            path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            print(json.dumps(dict(ev=mode, arm=name, error=report.get('error'), seconds=time.time()-start)), flush=True)


if __name__ == '__main__':
    main()
