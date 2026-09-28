"""Exact JSONL parity for K10 pi05 with --os-policy-tail alone."""
import json,os,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent
P=['taskset','-c','30-33,74-77','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=.:src',sys.executable]
source=sys.argv[1];reports=[]
for judge in ('guard_only','threshold:inf','periodic:5'):
    tag=judge.replace(':','_');out=Path(f'/tmp/q2_tail_parity_{source}')/tag;out.mkdir(parents=True,exist_ok=True)
    kw=dict(base_kwargs=dict(lib='current',kref=5,serving='anchor_tail',budget=1,gates='budget_only'),guards=judge!='threshold:inf',ncal=64)
    args=['--blind','--policy-tail','--cell','pi05_l10_cache','--yaml','exp/trace_dual/config/tr_pi05_l10_cache.yaml','--root','/home/weiland/trace_runs/offline_search_store','--method','exp.offline_search.rounds.r04.k10_policy_tail.judge:PolicyTailJudge','--kwargs',json.dumps(kw),'--judge',judge,'--out',str(out)]
    for variant in ('before',source):
        for f in out.glob('decisions_*.jsonl'):f.unlink()
        cmd=P+[str(B/'regression/launch_test.py'),variant,'parity']
        with (B/'results'/f'tail_parity_{source}_{tag}_{variant}.log').open('w') as f:
            subprocess.run(cmd,env={**os.environ,'K10_TEST_ARGS':json.dumps(args)},stdout=f,stderr=subprocess.STDOUT,check=True)
        data=next(out.glob('decisions_*.jsonl')).read_bytes()
        (B/'results'/f'tail_parity_{source}_{tag}_{variant}.jsonl').write_bytes(data)
        if variant=='before':old=data
        else:assert data==old,(tag,len(old),len(data))
    reports.append(dict(judge=judge,bytes=len(data),rows=len(data.splitlines()),byte_identical=True))
(B/'results'/f'tail_parity_{source}.json').write_text(json.dumps(reports,indent=2));print(reports)
