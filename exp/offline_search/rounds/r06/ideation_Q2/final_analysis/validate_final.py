"""Independent product checks, without reading additional running experiments."""
from pathlib import Path
import csv
import hashlib
import json
import re
import math
from scipy.stats import binomtest

HERE=Path(__file__).resolve().parent
Q2=HERE.parent
def read(path):return list(csv.DictReader(path.open()))
pilot=json.loads((HERE/'preregistered/estimates.json').read_text())
audit=json.loads((HERE/'preregistered/audit_q2.json').read_text())
front=json.loads((HERE/'frontier/frontier_data.json').read_text())
outcomes=json.loads((HERE/'frontier/outcomes.json').read_text())
checks={}
checks['pilot_full_support']=len(pilot['completeness'])==8 and all(r['complete'] for r in pilot['completeness'])
checks['frozen_selection']=all(r['recommendation']=='P10_reference_inconclusive' and not r['eligible'] and not r['uncertainty_available'] for r in pilot['recommendations'])
checks['validation_variance_unavailable']=all(r['SR_simul_lower'] is None for r in pilot['contrasts'] if r['split']=='validation' and r['task']=='ALL')
checks['family_296']=audit['alpha_per_test']==.05/296
points=front['points']
checks['frontier_owner_price']=all(abs(r['owner_IR']-((.152 if r['model']=='pi05' else .148)*r['v']+(1-(.152 if r['model']=='pi05' else .148))*r['m']))<1e-12 for r in points if not r['pure'])
checks['eligible_common_init_set']=all(len(outcomes[r['id']])==500 and set(outcomes[r['id']])=={f'{t}:{i}' for t in range(10) for i in range(50)} for r in points if r['eligible'])
mc=read(HERE/'frontier/paired_mcnemar.csv')
mc_ok=True
for r in mc:
    a,b=outcomes[r['candidate']],outcomes[r['reference']]
    w=sum(a[k]==1 and b[k]==0 for k in a)
    l=sum(a[k]==0 and b[k]==1 for k in a)
    p=binomtest(w,w+l,.5).pvalue if w+l else 1.
    mc_ok &= w==int(r['wins']) and l==int(r['losses']) and abs(p-float(r['mcnemar_two_sided_exact_p']))<1e-12
checks['all_plain_paired_tests_recomputed']=mc_ok
checks['minima_valid']=all(g['minimum_NI'].startswith('REFERENCE/') or g['minimum_NI_lower']>-.02 for g in front['gaps'])
new=json.loads((HERE/'derived/new_frontier_native_tests.json').read_text())
checks['new_frontier_no_NI']=len(new)==18 and all(not r['L5_NI_pass'] and not r['L10_NI_pass'] for r in new)
checks['measured_precision_design']=all(200<=float(r['full_validation_neff'])<=400 and int(r['full_validation_episodes'])==400 and int(r['full_validation_init_clusters'])==200 for r in read(HERE/'derived/measured_precision.csv') if r.get('full_validation_neff'))
report=(Q2/'FINAL.md').read_text()
links=re.findall(r'\]\(([^)]+)\)',report)
checks['report_artifact_links_exist']=all((Q2/x).exists() for x in links if not x.startswith(('http','mailto')))
checks['report_cost_transcription']=all(s in report for s in ['.240171','.227529','.205440'])
checks['report_no_old_CPU_command']='taskset -c 18-21' not in report
assert all(checks.values()),checks
output=dict(checks={k: bool(v) for k,v in checks.items()},paired_tests_recomputed=len(mc),frontier_points=len(points),
            report_sha256=hashlib.sha256((Q2/'FINAL.md').read_bytes()).hexdigest())
(HERE/'validation.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
