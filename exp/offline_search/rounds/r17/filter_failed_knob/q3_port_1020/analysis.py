"""Off-vs-pure decisions for the Q3 batches, with the existing R15/R16 loaders.

The loaders required --os-judge guard_only (the removed gate); that requirement is removed here
(the 'Changed judge' check of r16.analyze.load and the bundled 'Changed client/policy protocol'
check of r16.conductor_analysis.load_entry). Everything else is the formal-A rule: a cell is tuned
only when its complete 500-episode knob-off cohort has fewer successes than the same-step pure A.
"""
from __future__ import annotations

from pathlib import Path

from exp.offline_search.rounds.r16 import analyze as a, conductor_analysis as ca

from .fit import read

REMOVED = ('Changed judge', 'Changed client/policy protocol')
_require = a.require


def _require_q3(ok, message):
    if message in REMOVED:
        return
    _require(ok, message)


a.require = _require_q3
_jsonl = a.Reader.jsonl


def _jsonl_segment_off(self, path):
    """Long segment OFF serves without --os-blind: every decision runs the vision stage and the plugin records
    neither `vision` nor a policy call's `miss_k` (only the blind path does). Fill them: vision True, miss_k the
    shard's startup miss_steps."""
    k = None
    for d in _jsonl(self, path):
        if d.get('ev') == 'startup':
            k = d.get('miss_steps', k)
        elif d.get('ev') == 'dec' and 'vision' not in d:
            d['vision'] = True
            if d.get('hit') is False and 'miss_k' not in d:
                d['miss_k'] = k
        yield d


a.Reader.jsonl = _jsonl_segment_off


def load_job(job):
    from ..formal_a_analysis import legacy_data
    result, root = {}, Path(job['run'])
    if job['pipeline'] == 'R15':
        record = read(root/'ROOT.json')
        for row in read(root/'arms.json'):
            if not (root/'runs'/row['arm']/'summary.json').exists():
                continue
            ep = a.load(record, row['arm'], [(t, i) for t in range(10) for i in range(50)], 'A',
                        10 if job['model'] == 'pi05' else 8, a.Reader())
            result.setdefault((row['suite'], row['r10_size']), {})[row['r14_descent_level']] = legacy_data(ep, row['suite'])
    elif (root/'DONE.json').exists():
        group = read(root/'group.json')
        for entry in group['entries']:
            view = dict(entry, pairs=[(t, i) for t in range(10) for i in range(50)])
            ep = ca.load_entry(group, view, a.Reader()); row = entry['row']
            result.setdefault((row['suite'], row['r10_size']), {}).setdefault(entry['formal_grid_level'], {}).update(
                legacy_data(ep, row['suite']))
    return result


def evaluate(model, mode, arm):
    from .. import formal_a_queue as q
    from .build import INDEX
    unit = next(u for u in read(INDEX)['units'] if (u['model'], u['mode'], u['arm'], u['stage']) == (model, mode, arm, 'off'))
    data = {}
    for job in unit['jobs']:
        for cell, levels in load_job(job).items():
            data.setdefault(cell, {}).update(levels.get('off', {}))
    pure = q.references(model, mode)
    rows = [dict(suite=s, size=n, **q.classify(data.get((s, n), {}), pure.get(s, {}))) for s, n in q.cells(model)]
    return dict(model=model, mode=mode, arm=arm, unit=unit['name'], rows=rows, reference='same-step same-GPU pure A',
                rule='tune iff complete off successes < pure successes', dev_outcomes_used=False)
