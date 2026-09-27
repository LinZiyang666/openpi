"""Validate completed diagnostic coverage and arithmetic; no original-asset writes."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent
allowed = set(range(12, 24)) | set(range(56, 68))
assert set(os.sched_getaffinity(0)) <= allowed
for path in OUT.glob('*.py'):
    ast.parse(path.read_text(), filename=str(path))

paths = [p for p in OUT.glob('retrieval_*.json') if p.name != 'retrieval_validation.json']
rows = [r for p in paths for r in json.loads(p.read_text())]
assert len(paths) == 8 and len(rows) == 80
assert len({(r['model'], r['suite'], r['library'], r['arm'], r['variant']) for r in rows}) == 80
assert all(r['query_episodes'] == 50 for r in rows)
assert all(r['fit_source'] == 'same candidate library only' for r in rows)
for r in rows:
    assert math.isfinite(r['metrics']['all']['err'])
    if r['variant'] == 'full':
        assert r['metrics']['all']['overlap16'] == 1
        assert r['metrics']['all']['delta_err_full'] == 0

validation = json.loads((OUT / 'retrieval_validation.json').read_text())
assert len(validation) == 16 and sum(r['n'] for r in validation) == 1974
assert min(r['top1_agreement'] for r in validation) > .99
assert min(r['overlap16'] for r in validation) > .998

replays = json.loads((OUT / 'schedule_replay.json').read_text())
assert len(replays) == 6 and all(r['episodes'] == 500 for r in replays)
for r in replays:
    if r['arm'] in ['r3mx_p_l10_g', 'r3mx_p_l10_g500', 'r3mx_p_sp_g']:
        assert r['variants']['guard3']['misses'] == r['observed_miss']
costs = json.loads((OUT / 'cost_tables.json').read_text())
for r in costs['stacks']:
    m = r['m']
    assert math.isclose(r['full_K2'], .152 + m * (.410 + .438 * .2))
    assert math.isclose(r['duplicate_r05_K2'] - r['reusable_r05_K2'], m * .076)
    assert math.isclose(r['full_K2'] - r['reusable_r05_K2'], (1 - m) * .076)
semantics = json.loads((OUT / 'semantic_audit.json').read_text())
assert all(c[k] for c in semantics['mask_checks'] for k in
           ['positions_equal', 'prefix_mask_equal', 'suffix_position_offset_equal'])
assert json.loads((OUT / 'vision_gpu.json').read_text())['status'] == 'blocked'
assert json.loads((OUT / 'vision_cpu.json').read_text())['status'] == 'ok'

manifest = {p.name: {'bytes': p.stat().st_size,
                     'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(OUT.iterdir()) if p.is_file() and p.suffix in ['.py', '.json', '.md']
            and p.name != 'final_audit.json'}
assert max(x['bytes'] for x in manifest.values()) < 50_000_000
result = {'status': 'passed', 'retrieval_cells': len(rows),
          'validation_queries': sum(r['n'] for r in validation),
          'accepted_replay_episodes': sum(r['episodes'] for r in replays),
          'gpu_measurements_completed': False, 'files': manifest}
(OUT / 'final_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'files'}))
