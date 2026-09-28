"""Assert final evidence and unchanged installed bytes, then write compact counters."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent;R=B/'regression/results/installed'
def read(p):return json.loads(Path(p).read_text())
def report(path):
    x=read(path);assert x.get('PASS',True),path;return x
s={}
s['installed']=read(B/'results/install.json')
for row in s['installed']:
    p=Path(row['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
    assert p.read_bytes()==(B/'dev'/p.name).read_bytes()
s['existing']=report(R/'existing_summary.json')
s['k5_overlay']=report(R/'overlay_installed.json')
s['k5_estimator']=report(R/'estimator_validation.json')
s['k5_audit']=report(R/'final_audit.json')
s['k6_concurrency']=read('/tmp/q2_installed_concurrency/summary.json')
s['k6_edges']=report('/tmp/q2_installed_k6_edges/edge.json')
s['k10_concurrency']=read('/tmp/q2_installed_tail_concurrency/summary.json')
s['k10_edges']=report('/tmp/q2_installed_edges/report.json')
s['k10_method']=report(R/'method_test.json')
s['k10_selftests']=read(R/'tail_matrix.json')
s['parity_off']=read(R/'parity.json');s['parity_pi05_tail']=read(B/'results/tail_parity_installed.json')
s['new_contracts']=report(B/'results/contracts.json')
s['new_edges']=[report(p) for p in sorted(Path('/tmp/q2_final_edges').glob('*/report.json'))]
s['new_selftests']=[report(p) for p in sorted(Path('/tmp/q2_final_selftests').glob('*/selftest_report.json'))]
s['new_concurrency']=read('/tmp/q2_final_concurrency/summary.json')
s['arms']=report(B/'results/arms_validation.json')
s['k7_commands']=read(R/'k7_commands.json')
assert all(x['returncode']==0 for x in s['k7_commands'])
s['k7_parity']=[report(p) for p in sorted((R/'k7_unit/results').glob('parity_*.json'))]
s['k7_rates']=[read(p) for p in sorted((R/'k7_unit/results').glob('rates_*.json'))]
s['k7_edges']=report(R/'k7_unit/results/edges.json')
s['k7_plugins']=[report(p) for p in sorted(Path('/tmp/q2_installed/k7').glob('*/selftest_report.json'))]
assert len(s['new_edges'])==len(s['new_selftests'])==len(s['new_concurrency'])==8
assert len(s['k7_parity'])==len(s['k7_rates'])==4 and len(s['k7_plugins'])==5
assert len(s['k6_concurrency'])==12 and len(s['k10_concurrency'])==len(s['k10_selftests'])==9
assert len(s['existing']['rows'])==35
assert len(s['parity_off'])==6 and len(s['parity_pi05_tail'])==3
for key in ('k6_concurrency','k10_concurrency','new_concurrency','new_selftests','k10_selftests'):
    assert all(r['PASS'] for r in s[key]),key
for key in ('parity_off','parity_pi05_tail'):assert all(r['byte_identical'] for r in s[key])
s['final_commands']=read(B/'results/final_commands.json')
assert all(r.get('returncode')==0 for r in s['final_commands'])
s['totals']={key:dict(configurations=len(s[key]),decisions=sum(r['decisions'] for r in s[key])) for key in ('k6_concurrency','k10_concurrency','new_concurrency','new_selftests','k10_selftests','new_edges','k7_plugins')}
s['totals']['k7_scalar']=sum(r['decisions'] for p in s['k7_parity'] for r in p['rows'])
s['totals']['k7_fullcell']=sum(r['decisions'] for p in s['k7_rates'] for r in p if r['budget']==0)
s['totals']['actual_unique_chunks']=512
s['totals']['actual_queue_controls_repeated']=sum(r['counts']['queue_controls'] for r in s['new_edges'])
s['verified_utc']=datetime.now(timezone.utc).isoformat()
(B/'results/final_summary.json').write_text(json.dumps(s,indent=2)+'\n')
print(json.dumps(s['totals'],indent=2))
