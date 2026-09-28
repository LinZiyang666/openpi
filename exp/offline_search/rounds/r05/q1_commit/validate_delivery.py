"""Validate final fits, emit_arms full-model/L5/K10 contract and exact commands."""
import ast
import hashlib
import json
from pathlib import Path
import pickle

from exp.offline_search.closed_loop import plugin
from exp.offline_search.closed_loop.ops.emit_arms import main as emit
from exp.offline_search.harness import store
from exp.offline_search.rounds.r05.q1_commit.judge import CommitJudge, GraspCheckJudge
from openpi.cache.config import load_cache_config
from openpi.cache.types import PI05_V1
from prepare import ROOT

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)
arms = json.loads((HERE / 'arms_q1.json').read_text())
out = Path('/tmp/q1_emitted')
resolved = json.loads(json.dumps(arms).replace('<RUN>', str(out)))
spec = OUT / 'arms_resolved.json'
spec.write_text(json.dumps(resolved, indent=2))
emit(['--run-root', str(out), '--spec', str(spec)])
emitted = {a['arm']: a for a in json.loads((out / 'arms.json').read_text())}
reports = []
for arm in arms:
    row = emitted[arm['name']]
    cfg = load_cache_config(row['yaml'])
    assert row['full_model'] and row['cost_ledger'] and row['client_overrides']['replan_steps'] == 5
    assert cfg.miss is None and PI05_V1.num_steps == 10
    opts, rest = plugin.parse_cli(['--os-method', arm['method'], '--os-cell', row['cell'],
                                  '--os-log-dir', str(out / 'logs')] + arm['plugin_args'])
    assert not rest and opts.os_blind and opts.os_no_shadow_native and opts.os_judge == 'guard_only'
    c10 = 'c10' in arm['name']
    assert opts.os_policy_tail == c10
    path = Path('/tmp/q1_fits') / (arm['name'] + '.pkl')
    with path.open('rb') as f:
        payload = pickle.load(f)
    assert payload['spec'] == arm['method'] and payload['kwargs'] == arm['kwargs'] and payload['cell'] == row['cell']
    method = payload['method']
    assert type(method) is (CommitJudge if c10 else GraspCheckJudge)
    assert (method.base.serving, method.base.budget, method.base.gates) == ('anchor_tail', 1, 'budget_only')
    assert method.progress_guard == 'noprog_span' and method.stuck_guard == 'vision_confirmed'
    assert not method.memo_reset_after_miss
    if c10:
        assert method.policy_tail_gate == 'lifecycle' and method.monitor == 'off'
    lib = store.LibraryView(ROOT, 'pi05_' + arm['suite'], method.base.cand_name)
    d = dict(arm=arm['name'], path=str(path), bytes=path.stat().st_size,
             sha256=hashlib.file_digest(path.open('rb'), 'sha256').hexdigest(), library=method.base.cand_name,
             library_rows=lib.L, library_disk_bytes=sum(p.stat().st_size for p in lib.dir.rglob('*') if p.is_file()),
             bytes_per_entry=method.bytes_per_entry(), owner_deployed_pkl_MB=1103 if arm['suite']=='l10' else 431)
    if not c10:
        d.update(aperture_lo=float(method.contact['lo']), aperture_hi=float(method.contact['hi']),
                 contact_array_bytes=sum(method.contact[k].nbytes for k in ('width', 'pattern', 'next')) + 8)
    reports.append(d)
for path in HERE.glob('*.py'):
    ast.parse(path.read_bytes())
result = dict(PASS=True, arms=6, full_model=True, cost_ledger=True, L=5, miss_steps=10, fits=reports)
(OUT / 'delivery.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
