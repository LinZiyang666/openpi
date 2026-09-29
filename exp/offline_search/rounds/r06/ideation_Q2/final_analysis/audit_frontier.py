"""Post-hoc snapshot, using the existing Q2 ledger and paired-test definitions."""
from pathlib import Path
import datetime
import json
from exp.offline_search.rounds.r06.ideation_Q2.frontier import build_frontier as f

HERE = Path(__file__).resolve().parent
f.OUT = HERE / 'frontier'
f.OUT.mkdir(exist_ok=False)
base_family = f.family

def family(s, spec, run, arm, pure, length):
    method = s.get('method', '')
    if 'ExtraDose' in method:
        return 'B_plus_random_anchor_dose'
    if 'RiskLottery' in method:
        return 'episode_lottery_' + s['kwargs'].get('allocation', 'risk')
    if 'CacheDose' in method:
        return 'A_plus_random_anchor_dose'
    if 'p2_ablations' in method:
        return 'ablation_' + method.split('.')[-1]
    return base_family(s, spec, run, arm, pure, length)

f.family = family
f.dump('inventory_started.json', {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope': 'post-hoc; one snapshot of finished summaries; no polling'})
points, outcomes, evidence, skipped = f.collect()
f.analyze(points, outcomes, evidence, skipped)
print(json.dumps({'points': len(points), 'eligible': sum(p['eligible'] for p in points),
    'frontier_completed': [p['id'] for p in points if p['run'] == 'r06_frontier'],
    'output': str(f.OUT)}, indent=2))
