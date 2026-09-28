"""Read-only final source/artifact checks; optional manifest verification."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

B = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--verify-manifest', action='store_true')
args = parser.parse_args()
s = json.loads((B / 'results/final_summary.json').read_text())
checks = []

for row in s['installed']:
    p = Path(row['path'])
    assert hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256'], p
    assert p.read_bytes() == (B / 'dev' / p.name).read_bytes(), p
checks.append('five installed sources equal tested candidates and installation hashes')

sources = sorted(B.rglob('*.py')) + [Path(r['path']) for r in s['installed']]
for p in sources:
    ast.parse(p.read_text(), filename=str(p))
checks.append(f'{len(sources)} Python source files parse')

for row in s['arms']['fits']:
    p = Path(row['path'])
    assert p.stat().st_size == row['bytes'], p
    assert hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256'], p
assert len(s['arms']['fits']) == 4
checks.append('four deployed prefits retain validated sizes and SHA256')

expected = {'k6_concurrency': (12, 2196), 'k10_concurrency': (9, 1647),
            'new_concurrency': (8, 1464), 'new_selftests': (8, 384),
            'k10_selftests': (9, 432), 'new_edges': (8, 420), 'k7_plugins': (5, 240)}
for key, (count, decisions) in expected.items():
    rows = s[key]
    assert len(rows) == count and sum(r['decisions'] for r in rows) == decisions, key
    assert all(r['PASS'] for r in rows), key
assert len(s['existing']['rows']) == 35
assert sum(r['decisions'] for r in s['existing']['rows']) == 1599
assert s['totals']['k7_scalar'] == 15272 and s['totals']['k7_fullcell'] == 189904
assert s['totals']['actual_queue_controls_repeated'] == 51200
assert len(s['parity_off']) == 6 and len(s['parity_pi05_tail']) == 3
assert all(r['byte_identical'] for key in ('parity_off', 'parity_pi05_tail') for r in s[key])
assert all(r['returncode'] == 0 for r in s['final_commands'])
assert 'PASS K10 nine plugin selftests' in (B / 'results/installed_regression.log').read_text()
checks.append('all final matrices, exact byte parity, subprocess results and completion marker verified')

if args.verify_manifest:
    for name in ('deliverable_hashes.json', 'file_inventory.json'):
        entries = json.loads((B / 'results' / name).read_text())
        for row in entries:
            p = Path(row['path'])
            assert p.stat().st_size == row['bytes'], p
            with p.open('rb') as f:
                assert hashlib.file_digest(f, 'sha256').hexdigest() == row['sha256'], p
        checks.append(f'{name}: {len(entries)} unchanged files')
    assert (B / 'HANDBACK.md').is_file()

report = dict(PASS=True, verified_utc=datetime.now(timezone.utc).isoformat(), checks=checks)
if not args.verify_manifest:
    (B / 'results/final_audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
