"""Accepted-decision R12 work ledger and numerical diagnostic aggregation."""
from collections import Counter
import numpy as np
from .lookhalf_cameras import ledger


def summarize(model, decisions):
    counts = Counter()
    work, legacy = 0., 0.
    reasons = Counter()
    first_guard, by_episode = {}, {}
    for d in decisions:
        c = d.get('lookhalf_cost')
        if not isinstance(c, dict):
            raise ValueError('R12 accepted decision lacks camera cost ledger')
        vision = bool(d.get('vision', True))
        miss = not bool(d['hit'])
        check = ledger(model, vision=vision, wrist_first=c['camera_initial_count'] == 1,
                       completions=c['camera_completion_count'], miss=miss)
        for key in check:
            if not np.isclose(c[key], check[key], rtol=0, atol=1e-12):
                raise ValueError(f'inconsistent R12 camera work {key}')
        counts['decisions'] += 1
        counts['vision_decisions'] += vision
        counts['misses'] += miss
        counts['blind_decisions'] += not vision
        counts['full_looks'] += vision and c['camera_encoded_count'] == 2
        counts['one_camera_looks'] += vision and c['camera_encoded_count'] == 1
        counts['wrist_first_looks'] += vision and c['camera_initial_count'] == 1
        counts['completions'] += c['camera_completion_count']
        counts['encoded_cameras'] += c['camera_encoded_count']
        counts['half_to_policy_completions'] += vision and miss and c['camera_initial_count'] == 1
        work += c['compute_share_work']; legacy += c['legacy_work']
        if c['camera_completion_count']:
            reasons[d.get('completion_reason') or d.get('lookhalf', {}).get('completion_reason', 'call')] += 1
        uid = d.get('uid', '')
        stats = by_episode.setdefault(uid, dict(decisions=0, looks=0, calls=0, first_guard_call=None))
        stats['decisions'] += 1; stats['looks'] += vision; stats['calls'] += miss
        if miss and d.get('extras', {}).get('os_reason') == 4:
            first_guard[uid] = min(first_guard.get(uid, d['step']), d['step'])
            stats['first_guard_call'] = first_guard[uid]
    n = counts['decisions']
    return dict(schema='offline_search.r12_compute_share.v1', **counts,
                work=work, ir_per_five_controls=work/n if n else None,
                legacy_work=legacy, legacy_ir_per_five_controls=legacy/n if n else None,
                completion_reasons=dict(reasons), first_guard_call=first_guard,
                per_episode=by_episode, cost_basis='camera compute share; policy remainder full price',
                diagnostics_excluded=True)


def numerical_report(looks):
    import json
    rows = []
    for look in looks:
        if 'groot_numerical_path' in look:
            info = json.loads(str(look['groot_numerical_path']))
            if not info['exact_input'] or not info['diagnostic_only']:
                raise ValueError('non-exact numerical diagnostic')
            rows.append(dict(uid=look['meta']['uid'], step=int(look['step']), **info))
    stats = {}
    for label, values in [('policy_max_abs', [r['policy_delta']['max_abs'] for r in rows]),
                          ('policy_rms', [r['policy_delta']['rms'] for r in rows]),
                          ('key_max_abs', [d['max_abs'] for r in rows for d in r['key_deltas']]),
                          ('key_relative_l2', [d['relative_l2'] for r in rows for d in r['key_deltas']])]:
        stats[label] = dict(mean=float(np.mean(values)), max=float(np.max(values))) if values else None
    return dict(samples=len(rows), statistics=stats, samples_detail=rows,
                status='recorded exact pixels with matched noise' if rows else 'pending coordinator B pilot')
