"""Reuse exact K7 verification artifacts, never refit another owner's paths."""
from common import *
import os,shutil
arms=json.loads((B.parents[1]/'r04/k7_guard/arms_k7.json').read_text());audit=[]
for a in arms:
    if a['name'] not in ('r4k7_p_l10_50_ph2g','r4k7_p_sp_500_ph2g'):continue
    src=pathlib.Path('/home/weiland/trace_runs/os_closed_loop/r04_k7/fits')/(a['name']+'.pkl')
    dst=pathlib.Path('/tmp/q5_regression_fits')/src.name
    with src.open('rb') as f:blob=Loader(f).load()
    wanted=dict(spec=a['method'],kwargs=a['kwargs'],cell=f"pi05_{a['suite']}_cache")
    assert {k:blob[k] for k in wanted}==wanted
    if not dst.exists():
        tmp=dst.with_suffix('.q5tmp');shutil.copyfile(src,tmp);os.replace(tmp,dst)
    assert hashlib.sha256(dst.read_bytes()).hexdigest()==hashlib.sha256(src.read_bytes()).hexdigest()
    audit.append(dict(source=str(src),destination=str(dst),sha256=hashlib.sha256(src.read_bytes()).hexdigest(),metadata=wanted))
dump(B/'results/k7_fit_reuse.json',audit);print(json.dumps(audit,indent=2))
