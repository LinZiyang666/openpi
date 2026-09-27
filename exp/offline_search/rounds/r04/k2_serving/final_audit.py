import json
import hashlib
from pathlib import Path
base=Path(__file__).resolve().parent
s=json.loads((base/'results'/'validation_summary.json').read_text())
assert s['total_checks'] == 21 and s['total_decisions'] == 963
for name,digest in s['installed_sha256'].items():
    assert hashlib.sha256((base.parents[2]/'closed_loop'/name).read_bytes()).hexdigest() == digest, name
for name,d in s['checks'].items():
    assert d['PASS'], name
    if 'vision' in d:
        assert d['stage1_calls']==d['vision']
        assert d['broadcasts']==d['decisions']
        verify=json.loads((base/'results'/name/'verify_blind.json').read_text())
        assert verify['PASS'] and all(v==d['decisions'] for v in verify['equal'].values()),name
hist=json.loads((base/'results'/'final_hist'/'verify.json').read_text())
assert all(v==1 for v in hist['offline_equal'].values())
for model in ('pi05','groot'):
    d=s['gpu'][model]
    assert d['final_replay']['offline_wire']=={'checked':24,'bit_equal':24,'max_abs':0.0}
    assert d['normalized_replay']['PASS']
assert (base/'HANDBACK.md').is_file()
result={'PASS':True,'cpu_checks':21,'cpu_decisions':963,'gpu_wire_checked':48,'gpu_wire_bit_equal':48,
        'legacy_pure_bytes':63330,'legacy_mixed_bytes':91754,'installed_sha_verified':len(s['installed_sha256'])}
(base/'results'/'final_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
