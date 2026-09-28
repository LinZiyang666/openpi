"""Final evidence audit; fail rather than report an unrun matrix as passed."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent;R=B/'regression/results/installed';F=B/'final_regression/results/installed'
def read(p):return json.loads(Path(p).read_text())
def report(p):
    x=read(p);assert x.get('PASS',True),p;return x
s={}
s['hardware']=(B/'results/hardware.txt').read_text().strip()
s['software']=read(B/'results/software.json')
for path,expected in read(B/'results/frozen_sources.json').items():
    assert hashlib.sha256((B/path).read_bytes()).hexdigest()==expected,path
s['installed']=read(B/'results/install.json')
assert hashlib.sha256(Path(s['installed']['path']).read_bytes()).hexdigest()==s['installed']['sha256']
assert Path(s['installed']['path']).read_bytes()==(B/'dev/plugin.py').read_bytes()
assert hashlib.sha256((B/'dev/gpu_retrieval.py').read_bytes()).hexdigest()==s['installed']['helper_sha256']
s['existing']=report(F/'existing_summary.json')
s['k5_overlay']=report(F/'overlay_installed.json');s['k5_estimator']=report(F/'estimator_validation.json')
s['k5_audit']=report(F/'final_audit.json')
s['k6_concurrency']=read('/tmp/q5_installed_concurrency/summary.json')
s['k6_edges']=report('/tmp/q5_installed_k6_edges/edge.json')
s['k10_concurrency']=read('/tmp/q5_installed_tail_concurrency/summary.json')
s['k10_edges']=report('/tmp/q5_installed_edges/report.json')
s['k10_method']=report(R/'method_test.json');s['k10_selftests']=read(R/'tail_matrix.json')
s['parity_off']=read(F/'parity.json');s['parity_pi05_tail']=read(B/'q2checks/results/tail_parity_installed.json')
s['q2_contracts']=report(B/'q2checks/results/contracts.json')
s['q2_edges']=[report(p) for p in sorted(Path('/tmp/q5_q2_edges').glob('*/report.json'))]
s['q2_selftests']=[report(p) for p in sorted(Path('/tmp/q5_q2_selftests').glob('*/selftest_report.json'))]
s['q2_concurrency']=read('/tmp/q5_q2_concurrency/summary.json')
s['arms']=report(B/'results/arms_validation.json')
s['k7_commands']=read(R/'k7_commands.json');assert all(x['returncode']==0 for x in s['k7_commands'])
s['k7_parity']=[report(p) for p in sorted((R/'k7_unit/results').glob('parity_*.json'))]
s['k7_rates']=[read(p) for p in sorted((R/'k7_unit/results').glob('rates_*.json'))]
assert all(r['stock']==r['k7'] for rows in s['k7_rates'] for r in rows if r['budget']==0)
s['k7_edges']=report(R/'k7_unit/results/edges.json')
s['k7_plugins']=[report(p) for p in sorted(Path('/tmp/q5_installed/k7').glob('*/selftest_report.json'))]
assert len(s['q2_edges'])==len(s['q2_selftests'])==len(s['q2_concurrency'])==8
assert len(s['k7_parity'])==len(s['k7_rates'])==4 and len(s['k7_plugins'])==5
assert len(s['k6_concurrency'])==12 and len(s['k10_concurrency'])==len(s['k10_selftests'])==9
assert len(s['existing']['rows'])==35
assert len(s['parity_off'])==6 and len(s['parity_pi05_tail'])==3
for key in ('k6_concurrency','k10_concurrency','q2_concurrency','q2_selftests','k10_selftests'):
    assert all(r['PASS'] for r in s[key]),key
for key in ('parity_off','parity_pi05_tail'):assert all(r['byte_identical'] for r in s[key])
s['q2_commands']=read(B/'results/q2_commands.json');assert all(r.get('returncode')==0 for r in s['q2_commands'])
s['totals']={key:dict(configurations=len(s[key]),decisions=sum(r['decisions'] for r in s[key])) for key in ('k6_concurrency','k10_concurrency','q2_concurrency','q2_selftests','k10_selftests','q2_edges','k7_plugins')}
s['totals']['k7_scalar']=sum(r['decisions'] for p in s['k7_parity'] for r in p['rows'])
s['totals']['k7_fullcell']=sum(r['decisions'] for p in s['k7_rates'] for r in p if r['budget']==0)
s['gpu']=[read(p) for p in sorted((B/'results').glob('gpu_*_r*.json'))];assert len(s['gpu'])==8
for r in s['gpu']:
    assert r['agreement']['n']==2200 and r['agreement']['top1']==2200
    assert r['precision']=='float64' and r['tf32_unchanged'] and r['concurrency_exact']
    assert r['capture_during_replay'] and r['concurrency_connections']==8
    assert r['gpu']['samples'][0]['free_mib']>=14336 and max(x['own_mib'] for x in r['gpu']['samples'])<=6144
    assert r['source_sha256']['dev/gpu_retrieval.py']==s['installed']['helper_sha256']
    assert all(v['max']<=1e-4 for k,v in r['chunk_by_regime'].items() if k!='0')
    if r['config']['family']=='MixedJudge':
        assert r['agreement']['guard_flags']==r['agreement']['guard_reason']==2200
s['stage1']=[report(p) for p in sorted((B/'results').glob('stage1_*/report.json'))];assert len(s['stage1'])==4
for r in s['stage1']:
    assert len(r['checks'])==6 and all(q['pool_exact'] for q in r['checks'])
    assert r['source_sha256']==s['installed']['helper_sha256']
    assert r['gpu']['samples'][0]['free_mib']>=14336 and max(x['own_mib'] for x in r['gpu']['samples'])<=6144
s['stack']=[report(p) for p in sorted(Path('/tmp').glob('q5_stack_dev64_*/report.json'))];assert len(s['stack'])==8
assert all(r['per_connection_lock_audited'] and r['failure_requires_reset'] for r in s['stack'])
s['postinstall']=read(B/'results/postinstall.json')
assert s['postinstall']['PASS'] and len(s['postinstall']['stack'])==8
assert all(r['PASS'] and r['per_connection_lock_audited'] and r['failure_requires_reset']
           for r in s['postinstall']['stack'])
s['verified_utc']=datetime.now(timezone.utc).isoformat()
(B/'results/final_summary.json').write_text(json.dumps(s,indent=2)+'\n')
print(json.dumps(s['totals'],indent=2))
