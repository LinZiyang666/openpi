"""Extra campaign contract checks; inspect captures, never run a producer."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from exp.offline_search.debug import reader
from exp.offline_search.debug.fixtures import _sample
from exp.offline_search.debug.server.replay_tape import compare_outputs
from exp.offline_search.debug.tools.decision.common import field, clean
from exp.offline_search.debug.tools.physical.common import Episode
from exp.offline_search.debug.tools.physical.selfcheck import check

ROOT = Path('/home/weiland/trace_runs/os_closed_loop/r08_main')
OUT = Path('/tmp/r8cb_E5_debug_architecture')


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=2, allow_nan=False)+'\n')


def audit(spec, aug_names):
    name = spec['arm']
    arm = reader.open_arm(ROOT, name)
    arm.cache_enabled = False
    inventory = json.loads((OUT/'inventory'/(name+'.json')).read_text())
    accepted = {x['episode_key'] for x in inventory['episodes']}
    catalog = arm.catalog()
    mapping = {int(r.row):int(r.task_id) for r in catalog.itertuples()}
    lib = spec['r8'].get('library') or 'current'
    cat_dir = ROOT/'catalog'/f"{spec['model']}_{spec['suite_short']}_{lib}"
    cat_meta = reader.read_json(cat_dir/'rows.json')
    attestation = []
    for directory in arm.server_dirs:
        for path in directory.glob('meta*.json'):
            meta = reader.read_json(path)
            compared, missing, mismatch = [], [], []
            for key, item in cat_meta['store_arrays'].items():
                captured = meta.get('artifacts', {}).get('library_'+lib+'_'+key+'.npy', {})
                if not captured.get('sha256'):
                    missing.append(key)
                elif captured['sha256'] != item['sha256']:
                    mismatch.append(key)
                else:
                    compared.append(key)
            attestation.append(dict(file=str(path), compared=compared, missing=missing, mismatch=mismatch,
                catalog_parquet_hash_ok=hashlib.sha256((cat_dir/'rows.parquet').read_bytes()).hexdigest()==cat_meta['sha256']))
    counts = Counter()
    reasons = defaultdict(Counter)
    all_ids, expected_draws = set(), set()
    samples = {}
    server_sequences = {}
    for directory in arm.server_dirs:
        for path in directory.glob('decisions*.jsonl'):
            seqs = []
            for row, _ in reader.iter_jsonl(path):
                seqs.append(row['server_seq'])
                if row.get('episode_key') not in accepted:
                    continue
                counts['decisions'] += 1
                all_ids.add(row['decision_id'])
                if name in aug_names and _sample(arm.manifest.get('campaign', ROOT.name), row['task_uid'], row['decision_seq'], 32, 2):
                    expected_draws.add(row['decision_id'])
                for key in ('stage_pre', 'pre_stage', 'p_effective', 'p_nominal', 'coin', 'eligible', 'override', 'drawn_e', 'support', 'propensities'):
                    value = field(row, key)
                    counts['present.'+key] += value is not None
                for key in ('retrieval_status', 'cache_chunk_status', 'policy_chunk_status', 'rawkeys_status'):
                    reasons[key][row.get(key, {}).get('status', 'absent')] += 1
                rows = row.get('rows') or []
                if rows:
                    counts['retrieval_records'] += 1
                    counts['members'] += len(rows)
                    counts['unknown_rows'] += sum(int(r) not in mapping for r in rows)
                    counts['wrong_task_rows'] += sum(mapping.get(int(r), row['task_id']) != row['task_id'] for r in rows)
                    counts['score_lengths.'+str(len(row.get('scores') or []))] += 1
                if row.get('retrieval_status', {}).get('status') == 'available':
                    counts['live_retrieval'] += 1
                    counts['live_score_length_mismatch'] += len(rows) != len(row.get('scores') or [])
                if row.get('vision') and row.get('src') not in ('cache_tail', 'policy_tail', 'follow'):
                    counts['fresh_anchors'] += 1
                    eligible = field(row, 'eligible', 'os_c_eligible', 'os_c_fresh')
                    p = field(row, 'p_effective', 'actual_propensity', 'os_c_p')
                    coin = field(row, 'coin', 'os_c_coin', 'os_c_u')
                    z = field(row, 'treatment', 'executed_policy', 'os_c_call')
                    if eligible and p is not None and coin is not None and z is not None and 0 < p < 1:
                        counts['auditable_coin'] += 1
                        counts['coin_treatment_mismatch'] += bool(coin < p) != bool(z)
                if len(samples)<3:
                    samples[row['decision_id']] = dict(src=row.get('src'), diag=row.get('diag'), lib=row.get('lib'), lib_sha=row.get('lib_sha'))
            server_sequences[str(path)] = dict(rows=len(seqs), duplicate_sequences=len(seqs)-len(set(seqs)),
                set_contiguous=sorted(seqs)==list(range(len(seqs))),
                adjacent_inversions=sum(a>b for a,b in zip(seqs,seqs[1:])))
    physical, numeric, snapshot_stats = [], defaultdict(Counter), Counter()
    skipped = Counter()
    for ep in inventory['episodes']:
        if ep['init'] != 0:
            continue
        ek = ep['episode_key']
        meta = reader.read_json(arm.debug_dir/'client'/ek/'episode.json')
        controls = arm.controls(ek)
        for key, a in controls.items():
            if a.dtype.kind == 'f':
                numeric[key]['values'] += a.size
                numeric[key]['nonfinite'] += int((~np.isfinite(a)).sum())
        episode = Episode(meta=meta, controls=controls, manifest=arm.manifest, reader_arm=arm)
        physical.extend(check(episode))
        for key, snap in arm.snapshots(ek).items():
            snapshot_stats['snapshots'] += 1
            snapshot_stats['certified'] += bool(snap.get('restore_certified', False))
            snapshot_stats['selection_p.'+str(float(snap.get('selection_p', np.nan)))] += 1
            snapshot_stats['server_state_present'] += any(k.startswith('server') or k.startswith('policy') or k.startswith('method') for k in snap)
            for item in json.loads(str(snap.get('controller_skipped_json', '[]'))):
                skipped[item] += 1
    augmentation = {}
    if name in aug_names:
        for kind in spec['r8']['augmentation']:
            ids, statuses, numeric_finite = [], Counter(), Counter()
            for path in sorted((arm.debug_dir/'aug'/kind).glob('part_*.npz')):
                with np.load(path, allow_pickle=False) as z:
                    ids.extend(z['decision_id'].astype(str).tolist())
                    for key in z.files:
                        if key in ('chunk', 'chunks', 'cache_chunk'):
                            a=z[key]
                            numeric_finite[key+'.values'] += a.size
                            numeric_finite[key+'.nonfinite'] += int((~np.isfinite(a)).sum())
                        if 'status' in key:
                            statuses.update(str(x) for x in z[key].reshape(-1))
            expected = expected_draws if kind == 'policy_draws' else all_ids
            augmentation[kind] = dict(rows=len(ids), duplicate_ids=len(ids)-len(set(ids)), expected=len(expected),
                missing=len(expected-set(ids)), unexpected=len(set(ids)-expected), statuses=dict(statuses), finite=dict(numeric_finite))
    return dict(arm=name, time=time.time(), counts=dict(counts), attestation=attestation,
                status_counts={k:dict(v) for k,v in reasons.items()}, examples=samples,
                physical_sample_pairs=[[t,0] for t in range(10)], physical_checks=physical,
                physical_numeric={k:dict(v) for k,v in numeric.items()}, snapshots=dict(snapshot_stats),
                controller_skipped=dict(skipped), augmentation=augmentation,
                augmentation_job=reader.read_json(arm.debug_dir/'aug/job_stats.json') if name in aug_names else None,
                server_sequences=server_sequences,
                read_issues=arm.read_issues)


def parity():
    base = Path('/tmp/r8_S1/gpu_parity2')
    reports = {}
    for name in ('groot_A','groot_CU','groot_Policy','pi05_A','pi05_CU'):
        path = base/name
        report = compare_outputs(path/'off/responses', path/'on/responses')
        report['timings'] = {}
        for state in ('off','on'):
            values=defaultdict(list)
            for row,_ in reader.iter_jsonl(path/state/'legacy/decisions_parity.jsonl'):
                if 'infer_ms' not in row:
                    continue
                for key in ('infer_ms','pre_ms','method_ms','s1_ms','s23_ms'):
                    value=row.get(key)
                    if isinstance(value,(int,float)) and np.isfinite(value):
                        values[key].append(value)
            report['timings'][state]={k:dict(n=len(v),mean=float(np.mean(v)),p50=float(np.median(v)),p95=float(np.percentile(v,95))) for k,v in values.items()}
        reports[name]=report
    save(OUT/'parity.json', reports)


def main():
    specs = json.loads((ROOT/'arms.json').read_text())
    aug_names = set(json.loads((OUT/'augmentation_at_start.json').read_text())['arms'])
    parity()
    for spec in specs:
        path = OUT/'quality'/(spec['arm']+'.json')
        if path.exists():
            continue
        start=time.time()
        print(json.dumps(dict(ev='start',arm=spec['arm'])),flush=True)
        report=audit(spec,aug_names)
        save(path,report)
        print(json.dumps(dict(ev='done',arm=spec['arm'],seconds=time.time()-start)),flush=True)


if __name__ == '__main__':
    main()
