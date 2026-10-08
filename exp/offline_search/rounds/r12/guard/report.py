"""Report accepted B pilot guard targeting, SR, camera share and ledger IR."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from exp.offline_search.closed_loop.look_recording import read_looks, validate_root
from exp.offline_search.closed_loop.lookhalf_ledger import numerical_report


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def report(root):
    """Summarize only complete accepted attempts, never refitting thresholds."""
    root = Path(root)
    validate_root(root)
    out = {}
    for row in json.loads((root/'arms.json').read_text()):
        run = root/'runs'/row['arm']
        journal = {r['task_uid']: r for r in _jsonl(run/'client/journal.jsonl')
                   if r.get('accepted') and r.get('status') in ('done', 'failed') and not r.get('error')}
        looks = []
        for directory in sorted(run.glob('server_*/looks')):
            for lk in read_looks(directory):
                accepted = journal.get(lk['meta']['uid'])
                if accepted is not None and int(lk['meta']['attempt']) == int(accepted.get('attempt', 1) or 1):
                    looks.append(lk)
        by = defaultdict(list)
        for lk in looks:
            by[lk['meta']['uid']].append(lk)
        flags, failed_calls, first, missed = 0, 0, [], 0
        half, risks, failures, latency = 0, [], 0, []
        total_calls = sum(not bool(lk['hit']) for lk in looks)
        for uid, ls in by.items():
            failed = not bool(journal[uid]['success'])
            failures += failed
            called = []
            for lk in sorted(ls, key=lambda r: int(r['step'])):
                ex = lk['guard_state'].get('extras') or {}
                guard = not bool(lk['hit']) and ex.get('os_reason') == 4
                flags += guard
                failed_calls += failed and guard
                if guard:
                    called.append(int(lk['step']))
                diag = json.loads(str(lk.get('lookhalf', '{}')))
                risk = diag.get('h2', {}).get('scalars', {}).get('risk')
                if risk is not None:
                    if len(diag['h2']['features']) != 7:
                        raise ValueError('H2 look is missing feature audit')
                    risks.append(risk)
                if diag.get('plan_us') is not None:
                    latency.append(diag['plan_us'])
                cost = json.loads(str(lk['lookhalf_cost']))
                half += cost['camera_encoded_count'] == 1
            if failed:
                if called:
                    first.append(called[0])
                else:
                    missed += 1
        summary_path = run/'summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        n = len(looks)
        out[row['kwargs']['method']] = dict(arm=row['arm'], status='complete' if len(journal) == 100 else 'pending',
            accepted_episodes=len(journal), recorded_accepted_looks=n,
            SR=sum(bool(r['success']) for r in journal.values())/len(journal) if journal else None,
            total_call_rate=total_calls/n if n else None, guard_call_rate=flags/n if n else None,
            calls_in_failing_episodes=failed_calls/flags if flags else None,
            failing_episodes=failures, first_guard_call_failed_mean=float(np.mean(first)) if first else None,
            failed_episodes_without_guard_call=missed, one_camera_share=half/n if n else None,
            guard_risk_samples=len(risks), plan_us_mean=float(np.mean(latency)) if latency else None,
            cost_ledger=summary.get('lookhalf_cost_ledger'),
            IR_compute_share=summary.get('lookhalf_cost_ledger', {}).get('ir_per_five_controls'),
            numerical_paths=numerical_report(looks))
    base = out['off']
    expected = .22 if 'periodic' in root.name else {('pi05', 'l10'): .35, ('pi05', 'spatial'): .35,
                                                     ('groot', 'l10'): .23, ('groot', 'spatial'): .37}[
        (row['model'], row['suite_short'])]
    for method in ('h2', 'm2'):
        r = out[method]
        checks = {}
        if base['total_call_rate'] is not None and r['total_call_rate'] is not None:
            delta = r['total_call_rate']-base['total_call_rate']
            checks.update(call_rate_delta=delta, call_rate_within_03=abs(delta) <= .03)
        if base['first_guard_call_failed_mean'] is not None and r['first_guard_call_failed_mean'] is not None:
            delta = r['first_guard_call_failed_mean']-base['first_guard_call_failed_mean']
            checks.update(first_call_failed_mean_delta=delta, stop_for_late_calls=delta >= 3)
        if method == 'm2' and r['one_camera_share'] is not None:
            checks.update(expected_half_share=expected, half_share_within_10=abs(r['one_camera_share']-expected) <= .10)
        if base['SR'] is not None and r['SR'] is not None:
            checks['SR_delta'] = r['SR']-base['SR']
        r['pilot_checks'] = checks
    result = dict(root=str(root), arms=out, thresholds='frozen B-dev constants; no pilot refit',
                  first_call_comparison='means in each arm failing episodes; no-call failures reported separately')
    (root/'guard_readout.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: {f: v[f] for f in ('status', 'accepted_episodes', 'SR', 'IR_compute_share', 'pilot_checks') if f in v}
                      for k, v in out.items()}, indent=2))
    return result


def main():
    """Write the pilot read-out after coordinator collection."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path)
    report(ap.parse_args().root)


if __name__ == '__main__':
    main()
