"""Sanity checks for the second-pass evidence; no simulator or test suite invocation."""
import ast,json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
files=list(OUT.glob('p2_*.json'));parsed={f.name:json.loads(f.read_text()) for f in files}
for f in OUT.glob('p2_*.py'):ast.parse(f.read_text())
r=json.loads((OUT/'p2_recoverability_summary.json').read_text());assert len(r)==14 and sum(x['episodes'] for x in r)==7000
expected={'g':.740,'g500':.864,'awm_h70':.816,'awm500_h70':.872}
for s,sr in expected.items():
 d=parsed[f'p2_recovery_r3mx_p_l10_{s}.json'];assert d['sr']==sr
 assert abs(d['ir']-(.152+.848*d['n_miss']/d['n_dec']))<1e-12
 assert len(d['per_episode'])==500
for m in ['pi05','groot']:
 for s in ['spatial','l10']:
  d=parsed[f'p2_tokens_{m}_{s}.json'];assert d['episodes']==50
  for n in [50,500]:
   a=parsed[f'p2_projection_{m}_{s}_{n}.json'];assert a['n']==100 and a['additional_entry_bytes']==16384
   k=parsed[f'p2_kernel_{m}_{s}_{n}.json'];assert len(k['per_episode'])<=500
for m in ['pi05','groot']:
 for n in [50,500]:
  d=parsed[f'p2_cross_task_{m}_{n}.json'];assert set(x['regime'] for x in d['rows'])=={'inf','cache'}
new=list(OUT.glob('p2_*'))+[OUT/'REPORT_2.md'];assert max(f.stat().st_size for f in new if f.is_file())<50_000_000
pilot=parsed['p2_causal_pilot.json'];assert np.prod([len(x) for x in pilot['contexts'].values()])*2==pilot['bins_including_landmark_class']==48
out={'parsed_json_files':len(files),'parsed_python_files':len(list(OUT.glob('p2_*.py'))),'mixed_arms':14,'accepted_mixed_episodes':7000,
     'token_episodes':200,'pure_cache_awm_episodes':4000,'report_bytes':(OUT/'REPORT_2.md').stat().st_size,'largest_new_file_bytes':max(f.stat().st_size for f in new if f.is_file()),
     'first_report_bytes':(OUT/'REPORT.md').stat().st_size,'first_report_expected_bytes':29374,'checks_passed':True}
assert out['first_report_bytes']==out['first_report_expected_bytes']
(OUT/'p2_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
for s in expected:
 d=parsed[f'p2_recovery_r3mx_p_l10_{s}.json'];print(s,'IR',format(d['ir'],'.6f'))
