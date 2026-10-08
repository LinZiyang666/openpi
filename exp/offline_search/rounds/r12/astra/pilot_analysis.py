"""CPU-only post-mortem of explicitly allowed, accepted B-pilot look shards.

Run from repository root; never imports fitting, rollout, or server modules.
"""
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PILOTS = HERE.parent / 'lookhalf' / 'pilots'
CELLS = ('pi05_l10_50', 'pi05_spatial_50', 'pi05_spatial_200',
         'groot_spatial_50', 'groot_l10_50')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_arm(cell, method):
    run = PILOTS / f'r12_lookhalf_fix1_{cell}' / 'runs' / f'r12_lh_{cell}_{method}'
    assert 'opus' not in str(run) and method in ('astra_consensus2', 'astra_strict', 'off')
    summary = json.loads((run / 'summary.json').read_text())
    assert summary['init_pool'] == 'B'
    accepted = {}
    for line in (run / 'client/journal.jsonl').read_text().splitlines():
        a = json.loads(line)
        if a.get('accepted') and a.get('status') in ('done', 'failed'):
            assert a['init_pool'] == 'B'
            accepted[a['task_uid']] = a
    assert len(accepted) == summary['complete'] == 100
    assert sum(a['success'] for a in accepted.values()) == summary['success']
    rows, skipped, manifests, seen = [], 0, [], set()
    for manifest in sorted(run.glob('server_*/looks/manifest_*.json')):
        manifests.append(dict(path=str(manifest), sha256=sha(manifest)))
        m = json.loads(manifest.read_text())
        assert m['dev'] is True and m['init_pool'] == 'B'
        for shard in m['shards']:
            a = accepted.get(shard['uid'])
            if a is None or a.get('attempt', 1) != shard['attempt']:
                skipped += 1
                continue
            f = manifest.parent / shard['file']
            assert f.parent == manifest.parent and f.name == shard['file']
            assert sha(f) == shard['sha256']
            with np.load(f, allow_pickle=False) as z:
                meta = json.loads(str(z['meta']))
                assert meta['dev'] is True and meta['init_pool'] == 'B'
                assert meta['arm'] == run.name
                pair = (meta['task_id'], meta['init'])
                assert list(pair) in m['contract']['pairs']
                assert pair[1] not in m['contract']['subset']['excluded_inits_by_task'][str(pair[0])]
                key = (meta['uid'], int(z['step']))
                assert key not in seen
                seen.add(key)
                d = json.loads(str(z['lookhalf']))
                cost = json.loads(str(z['lookhalf_cost']))
                guard = json.loads(str(z['guard_state']))
                rows.append(dict(uid=key[0], step=key[1], task=pair[0], init=pair[1],
                                 path=str(f), hit=bool(z['hit']), d=d, cost=cost,
                                 span=guard.get('method', {}).get('1', {}).get('_noprog_span'),
                                 availability=z['camera_availability'].tolist()))
    rows.sort(key=lambda r: (r['uid'], r['step']))
    ledger = summary['lookhalf_cost_ledger']
    assert len(rows) == ledger['vision_decisions']
    assert sum(not r['hit'] for r in rows) == ledger['misses']
    assert sum(r['cost']['camera_encoded_count'] == 1 for r in rows) == ledger['one_camera_looks']
    return rows, summary, dict(manifests=manifests, skipped_attempt_shards=skipped,
                              journal_sha256=sha(run / 'client/journal.jsonl'),
                              summary_sha256=sha(run / 'summary.json'))


def analyze(cell, method):
    rows, summary, provenance = read_arm(cell, method)
    buckets, pre, witness, combinations, exclusive, per_witness = [Counter() for _ in range(6)]
    lifecycle, variants = Counter(), {}
    previous = {}
    for r in rows:
        d, c = r['d'], r['cost']
        if c['camera_encoded_count'] == 1:
            buckets['one_camera'] += 1
        elif d['plan'] == 'wrist_first':
            buckets['complete_' + d['completion_reason']] += 1
        else:
            buckets['full_' + d['reason']] += 1
        if d['reason'] == 'lifecycle_or_due':
            last = previous.get(r['uid'])
            # Descriptive, disjoint attribution; logs merge the remaining causes.
            cause = ('warmup_step_lt4' if r['step'] < 4 else
                     'after_accepted_half' if last and last['d'].get('accepted_half') else
                     'previous_positive_span' if last and last['span'] and last['span'] > 0 else
                     'other_lifecycle_or_memory')
            lifecycle[cause] += 1
        previous[r['uid']] = r
        if d['plan'] == 'full_planned' and d['reason'] == 'atlas_or_state':
            for k, fail in [('support', d['support'] < .99), ('safe_mass', d['safe_mass'] < .75),
                            ('state_error', d['state_error'] > .2)]:
                if fail:
                    pre[k] += 1
        if 'witness_rms' not in d:
            continue
        failures = dict(motion=max(d['witness_rms']) > d['tau'],
                        gripper=any(d['witness_grip']), progress=max(d['witness_progress']) > .5,
                        advance=d['advance_h'] <= .5)
        for j, name in enumerate(('wrist', 'imputed_full')):
            for k, fail in [('motion', d['witness_rms'][j] > d['tau']),
                            ('gripper', d['witness_grip'][j]), ('progress', d['witness_progress'][j] > .5)]:
                if fail:
                    per_witness[name + '_' + k] += 1
        keys = [k for k, fail in failures.items() if fail]
        combinations['+'.join(keys) or 'pass'] += 1
        for k in keys:
            witness[k] += 1
        if len(keys) == 1:
            exclusive[keys[0]] += 1
        assert bool(keys) != d['accepted_half']
    for tau in (.05, .1, .15, .2, .3, float('inf')):
        eligible = [r for r in rows if 'witness_rms' in r['d']
                    and max(r['d']['witness_rms']) <= tau
                    and not any(r['d']['witness_grip'])
                    and max(r['d']['witness_progress']) <= .5 and r['d']['advance_h'] > .5]
        n = sum(r['hit'] for r in eligible)
        # Frozen recorded memory and guards; adjacent newly accepted looks would
        # require a refresh. These counts are optimistic local ceilings, not SR.
        variants[str(tau)] = dict(passes=len(eligible), noncall_passes=n,
                                  share=n / len(rows),
                                  fixed_history_saving=(.076 if cell.startswith('pi05') else .074)
                                  * n / summary['client_decisions'])
    attempts = [r for r in rows if 'witness_rms' in r['d']]
    masks = {
        'drop_witnesses': lambda d: True,
        'motion_only_0.1': lambda d: max(d['witness_rms']) <= .1,
        'motion_grip_0.1': lambda d: max(d['witness_rms']) <= .1 and not any(d['witness_grip']),
        'grip_progress_advance_only': lambda d: not any(d['witness_grip']) and max(d['witness_progress']) <= .5 and d['advance_h'] > .5,
        'wrist_only_0.1': lambda d: d['witness_rms'][0] <= .1 and not d['witness_grip'][0] and d['witness_progress'][0] <= .5 and d['advance_h'] > .5,
        'imputed_only_0.1': lambda d: d['witness_rms'][1] <= .1 and not d['witness_grip'][1] and d['witness_progress'][1] <= .5 and d['advance_h'] > .5,
    }
    ablations = {k: sum(r['hit'] and f(r['d']) for r in attempts) for k, f in masks.items()}
    out = dict(cell=cell, method=method, looks=len(rows), episodes=summary['complete'],
               decisions=summary['client_decisions'], sr=summary['sr'],
               ledger={k: v for k, v in summary['lookhalf_cost_ledger'].items() if k != 'per_episode' and k != 'first_guard_call'},
               buckets=dict(buckets), prelook_failures=dict(pre), lifecycle=dict(lifecycle),
               witness_attempts=len(attempts), witness_failures=dict(witness),
               exclusive_witness_failures=dict(exclusive), per_witness=dict(per_witness),
               combinations=dict(combinations), threshold_local_ceiling=variants,
               ablation_noncall_local_ceiling=ablations, provenance=provenance)
    (HERE / f'pilot_rows_{cell}_{method}.json').write_text(json.dumps(rows, separators=(',', ':')) + '\n')
    return out


if __name__ == '__main__':
    results = []
    for cell in CELLS:
        for method in ('astra_consensus2', 'astra_strict') if cell in ('pi05_l10_50', 'pi05_spatial_50') else ('astra_consensus2',):
            out = analyze(cell, method)
            results.append(out)
            print(json.dumps({k: v for k, v in out.items() if k not in ('provenance', 'ledger')}), flush=True)
    (HERE / 'pilot_analysis.json').write_text(json.dumps(dict(
        cpu_affinity=sorted(os.sched_getaffinity(0)), cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
        scope='Accepted B dev episodes only; local threshold counts freeze actual history, not SR predictions',
        results=results), indent=2) + '\n')
