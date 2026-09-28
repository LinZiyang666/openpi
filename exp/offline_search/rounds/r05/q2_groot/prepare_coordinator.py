"""Resolve the six arms and copy verified Q2 fits into RUN; never launch anything."""
import argparse,hashlib,json,pickle,shutil
from pathlib import Path
from exp.offline_search.closed_loop.ops.emit_arms import main as emit
from exp.offline_search.rounds.r05.q2_groot.judge import CycleTail
B=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--run-root',required=True,type=Path);a=p.parse_args()
a.run_root.mkdir(parents=True,exist_ok=True)
spec=json.loads((B/'arms_q2.json').read_text().replace('<RUN>',str(a.run_root.resolve())))
for r in spec:
    if r.get('pure_inference'):continue
    src=Path('/tmp/q2_fits')/(r['name']+'.pkl')
    with src.open('rb') as f:blob=pickle.load(f)
    assert {k:blob[k] for k in ('spec','kwargs','cell')}==dict(spec=r['method'],kwargs=r['kwargs'],cell=f"groot_{r['suite']}_cache")
    dst=a.run_root/'fits'/src.name;dst.parent.mkdir(exist_ok=True)
    if dst.exists():assert hashlib.sha256(dst.read_bytes()).digest()==hashlib.sha256(src.read_bytes()).digest()
    else:
        temp=dst.with_suffix('.q2.tmp');shutil.copyfile(src,temp);temp.replace(dst)
path=a.run_root/'arms_q2_resolved.json';path.write_text(json.dumps(spec,indent=2)+'\n')
emit(['--run-root',str(a.run_root),'--spec',str(path)])
