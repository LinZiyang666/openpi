"""Run the unchanged harness smoke across both policies, suites, regimes and deployed scales."""
import concurrent.futures, json, pathlib, subprocess, sys
OUT=pathlib.Path(__file__).resolve().parent/'results'
PFX=['taskset','-c','18-21,62-65','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
     'CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1',sys.executable]
PREFIX='exp.offline_search.rounds.r04.k1_blind.'


def job(item):
    tag,method,kw,cell=item
    cmd=PFX+['-m','exp.offline_search.harness.smoke','--method',method,'--kwargs',json.dumps(kw),
             '--cell',cell,'--episodes','2','--root','/home/weiland/trace_runs/offline_search_store',
             '--no-ref','--out',str(OUT/'smokes'/tag)]
    with (OUT/f'smoke_{tag}.log').open('w') as f:
        r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    report=dict(tag=tag,returncode=r.returncode,command=cmd)
    print(json.dumps(report),flush=True)
    return report

jobs=[]
for scale in (50,500):
    kw=dict(lib='current' if scale==50 else 'big',kref=5 if scale==50 else 8)
    for model in ('pi05','groot'):
        for suite in ('l10','spatial'):
            for arm in ('inf','cache'):
                cell=f'{model}_{suite}_{arm}'
                jobs.append((f'blind_{cell}_{scale}',PREFIX+'blind_awm:BlindAWM',kw,cell))
                for variant in ('G','GS'):
                    jobs.append((f'csl{variant}_{cell}_{scale}',PREFIX+'control_step:ControlStepLibrary',
                                 dict(kw,ablation=variant),cell))
    for arm in ('inf','cache'):
        cell=f'pi05_l10_{arm}'
        for mode in ('noprog_span','noprog_n'):
            jobs.append((f'mixed_{mode}_{cell}_{scale}',PREFIX+'judge:BlindMixedJudge',
                         dict(base_kwargs=kw,progress_guard=mode,events='none'),cell))
        jobs.append((f'memo_{cell}_{scale}',PREFIX+'judge:MemoResetMixedJudge',dict(base_kwargs=kw),cell))
prior = []
if '--only-control' in sys.argv:
    jobs = [job for job in jobs if job[0].startswith('csl')]
    prior = [r for r in json.loads((OUT/'smokes.json').read_text()) if not r['tag'].startswith('csl')]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    result=list(pool.map(job,jobs))
result = prior + result
(OUT/'smokes.json').write_text(json.dumps(result,indent=2))
assert all(r['returncode']==0 for r in result), [r['tag'] for r in result if r['returncode']]
