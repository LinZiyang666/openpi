#!/usr/bin/env python3
"""Re-run Q2's frontier audit (ideation_Q2/frontier/build_frontier.py, same ledger and paired-test definitions) into a
fresh output directory, after the exception-episode repair and with the R6 configuration-C / Bmech families added.

Usage: frontier_refresh.py <out-dir>   (the directory must not exist)
"""
import datetime
import json
import sys
from pathlib import Path

from exp.offline_search.rounds.r06.ideation_Q2.frontier import build_frontier as f

f.OUT = Path(sys.argv[1]).resolve()
f.OUT.mkdir(parents=True, exist_ok=False)
base_family = f.family


def family(s, spec, run, arm, pure, length):
    method = s.get('method', '')
    kw = s.get('kwargs', {}) or {}
    if 'CalibratedRescue' in method:
        tag = 'C' if kw.get('stall_model_path') else kw.get('placement', 'R')
        return f'configC_{tag}'
    if 'Bmech' in method:
        return 'ablation_Bmech_np_miss_off'
    if 'ExtraDose' in method:
        return 'B_plus_random_anchor_dose'
    if 'RiskLottery' in method:
        return 'episode_lottery_' + kw.get('allocation', 'risk')
    if 'CacheDose' in method:
        return 'A_plus_random_anchor_dose'
    if 'p2_ablations' in method:
        return 'ablation_' + method.split('.')[-1]
    return base_family(s, spec, run, arm, pure, length)


f.family = family
TOL = 0.02  # repaired arms: server-log ledger vs client decision count may differ by <= 2 % (flagged, see notes)
FLAGGED = {}
_orig_ledger = f.ledger_counts


def ledger_counts(s, spec, cached, jp):
    try:
        return _orig_ledger(s, spec, cached, jp)
    except ValueError as e:
        ledger = s.get('cost_ledger') or {}
        if str(e) != 'ledger/client mismatch' or not ledger:
            raise
        N, V, M = (int(ledger[k]) for k in ['decisions', 'vision_decisions', 'misses'])
        cd = int(s['client_decisions'])
        if abs(N - cd) > TOL * cd or not 0 <= M <= V <= N:
            raise
        FLAGGED[s.get('arm', str(jp))] = dict(ledger_decisions=N, client_decisions=cd, rel_diff=(N - cd) / cd)
        return N, V, M, float(ledger['l_per_request']['mean']), 'summary.cost_ledger(tolerance)', True


f.ledger_counts = ledger_counts
f.dump('inventory_started.json', {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                  'scope': 'post-repair refresh; one snapshot of finished summaries'})
points, outcomes, evidence, skipped = f.collect()
f.analyze(points, outcomes, evidence, skipped)
f.dump('ledger_tolerance_flags.json', FLAGGED)
print(json.dumps({'points': len(points), 'eligible': sum(p['eligible'] for p in points), 'output': str(f.OUT)}, indent=2))
