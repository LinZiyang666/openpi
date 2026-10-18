"""CPU: add completed 4090 concurrent results to a narrow read-only rsync module."""
import json
from pathlib import Path
from concurrent_prepare import include_list


if __name__=='__main__':
    work=Path(__file__).resolve().parent
    root=work/'results';files=[];runs=[]
    for run in root.glob('4090_*_concurrent_*'):
        receipt=run/'RUN.json'
        if not receipt.exists():continue
        r=json.loads(receipt.read_text())
        if r.get('status')!='COMPLETE' or r.get('schema')!='xgpu.concurrent.run.v1':continue
        runs.append(run.name)
        files.extend(p for p in run.rglob('*') if p.is_file() and not p.is_symlink() and
            (p.suffix in ('.json','.npz') or p.name=='batching.jsonl'))
    inc=work/'concurrent/complete_4090_results.include'
    inc.write_text(include_list(files,root) if files else '/__NO_COMPLETE_RUNS__\n')
    config=Path('/home/weiland/.claude/jobs/35f2869b/tmp/xgpu_rsyncd/rsyncd.conf')
    text=config.read_text()
    if '149.165.152.232' not in text or '149.165.153.233' not in text:raise ValueError('authorized host ACL missing')
    if '[xc_local_results]' not in text:
        text+=f'\n[xc_local_results]\npath = {root}\nread only = yes\ninclude from = {inc}\nexclude = *\n'
        config.write_text(text)
    print(json.dumps(dict(runs=runs,files=len(files),bytes=sum(p.stat().st_size for p in files))))
