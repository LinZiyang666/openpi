"""Measure-only paired R10 A / new A+C / R10 G / R10 GC_dist tables.

Run after all 24 arms finish. Preparation tests this script on synthetic files only.
IR is the standard stage-cost ledger's ir_per_five_controls, as in R10.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scipy.stats import binomtest

from .build import HERE, EXPECTED
from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import uid_pair
from exp.offline_search.rounds.r10.data import CELLS, SIZES, sha


def paired(new, old):
    if len(new) != len(old) or not new:
        raise ValueError('pair vectors must have equal nonzero lengths')
    gains = sum(a and not b for a, b in zip(new, old))
    losses = sum(b and not a for a, b in zip(new, old))
    p = float(binomtest(gains, gains + losses, .5, alternative='two-sided').pvalue) if gains + losses else 1.
    return dict(n=len(new), gains=gains, losses=losses, p_two_sided_exact=p,
                delta_sr=(sum(new) - sum(old)) / len(new))


def read_arm(root, name, *, manifest_sha256, summary_sha256=None):
    root = Path(root)
    rows = json.loads((root / 'arms.json').read_text())
    row, = [r for r in rows if r['arm'] == name]
    if row.get('init_pool', 'A') != 'A' or row.get('dev', False) or row.get('recording', False):
        raise ValueError('requires recording-off official A arms')
    if row.get('replan_steps', 5) != 5 or row.get('rollout_replicate_seed', 7) != 7:
        raise ValueError('wrong rollout controls or seed')
    template = Path(row['manifest'])
    run = root / 'runs' / name
    manifest = run / 'manifest.json'
    if any(json.loads(p.read_text()) != EXPECTED for p in (template, manifest)):
        raise ValueError('changed R10 A identities/template')
    summary_path = run / 'summary.json'
    summary = json.loads(summary_path.read_text())
    if summary['arm'] != name or summary.get('init_pool', 'A') != 'A' or summary['complete'] != 500:
        raise ValueError('incomplete or wrong A summary')
    journal = run / 'client/journal.jsonl'
    accepted = {}
    for line in journal.read_text().splitlines():
        record = json.loads(line)
        if record.get('init_pool', 'A') != 'A':
            raise ValueError('non-A journal row')
        if record.get('accepted') and record.get('status') in ('done', 'failed') and not record.get('error'):
            uid = record['task_uid']
            if not uid.startswith(name + ':') or uid_pair(uid) not in map(tuple, EXPECTED):
                raise ValueError('unexpected accepted episode identity')
            if type(record.get('success')) is not bool:
                raise ValueError('nonbinary episode outcome')
            # Match the standard collector: last accepted terminal row per UID.
            accepted[uid] = record
    outcomes = {uid_pair(uid): record['success'] for uid, record in accepted.items()}
    if len(accepted) != len(outcomes) or set(outcomes) != set(map(tuple, EXPECTED)):
        raise ValueError('missing, extra or duplicate paired episodes')
    y = [outcomes[tuple(pair)] for pair in EXPECTED]
    if summary['success'] != sum(y) or not math.isclose(summary['sr'], sum(y) / 500, abs_tol=1e-8):
        raise ValueError('summary/journal outcome mismatch')
    ledger = summary['cost_ledger']
    ir = float(ledger['ir_per_five_controls'])
    weight = float(ledger['controls']) / 5
    if weight <= 0 or not math.isfinite(ir) or ir < 0:
        raise ValueError('invalid IR ledger')
    return dict(y=y, sr=sum(y) / 500, ir=ir, ir_weight=weight,
                summary_sha256=sha(summary_path), journal_sha256=sha(journal),
                retried=summary.get('retried', 0))


def analyze(audit_path=HERE / 'audit.json'):
    audit = json.loads(Path(audit_path).read_text())
    records = audit['records']
    if not audit['PASS'] or len(records) != 24:
        raise ValueError('requires the complete 24-row frozen audit')
    index = {(r['model'], r['suite'], r['size']): r for r in records}
    if set(index) != {(m, s, n) for m, s in CELLS for n in SIZES}:
        raise ValueError('wrong cell/size design')
    results = []
    for model, suite in CELLS:
        cells = []
        for size in SIZES:
            record = index[model, suite, size]
            manifest_sha = record['references']['A']['files']['manifest']['sha256']
            arms = {}
            for v in ('A', 'G', 'GC_dist'):
                ref = record['references'][v]
                arms[v] = read_arm(ref['root'], ref['arm'], manifest_sha256=manifest_sha,
                                   summary_sha256=ref['files']['summary']['sha256'])
            arms['A+C'] = read_arm(record['root'], record['arm'], manifest_sha256=manifest_sha)
            cells.append(dict(model=model, suite=suite, size=size, arms=arms,
                              pairing=paired(arms['A+C']['y'], arms['A']['y'])))
        pooled_arms = {v: dict(sr=sum(sum(c['arms'][v]['y']) for c in cells) / 3000,
                              ir=sum(c['arms'][v]['ir'] * c['arms'][v]['ir_weight'] for c in cells) /
                                 sum(c['arms'][v]['ir_weight'] for c in cells))
                       for v in ('A', 'A+C', 'G', 'GC_dist')}
        pooled = paired([y for c in cells for y in c['arms']['A+C']['y']],
                        [y for c in cells for y in c['arms']['A']['y']])
        results.append(dict(model=model, suite=suite, cells=cells, pooled_pairing=pooled, pooled_arms=pooled_arms))
    return results


def render(results):
    lines = ['Official test A: measure only; frozen artifacts; no selection or refitting.',
             'Each entry is SR @ IR (standard stage-cost IR per five controls).',
             'Pooled pairing treats (size, task, init) as identity, following R10; sizes reuse inits.', '']
    for group in results:
        lines += [f"{group['model']} {group['suite']}", '',
                  '| size | R10 A | A+C | R10 G | R10 GC_dist | gains/losses vs A | two-sided exact p |',
                  '|---|---|---|---|---|---|---|']
        for c in group['cells']:
            entries = [f"{c['arms'][v]['sr']:.3f} @ {c['arms'][v]['ir']:.3f}" for v in ('A', 'A+C', 'G', 'GC_dist')]
            p = c['pairing']
            lines.append(f"| {c['size']} | " + ' | '.join(entries) +
                         f" | +{p['gains']}/-{p['losses']} | {p['p_two_sided_exact']:.5g} |")
        p = group['pooled_pairing']
        entries = [f"{group['pooled_arms'][v]['sr']:.3f} @ {group['pooled_arms'][v]['ir']:.3f}" for v in ('A', 'A+C', 'G', 'GC_dist')]
        lines += ['| pooled (3000) | ' + ' | '.join(entries) +
                  f" | +{p['gains']}/-{p['losses']} | {p['p_two_sided_exact']:.5g} |", '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=HERE / 'audit.json')
    args = parser.parse_args()
    print(render(analyze(args.audit)))


if __name__ == '__main__':
    main()
