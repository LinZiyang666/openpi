"""Run the original CPU plugin/orchestrator selftest in a fresh process per arm."""
import os
from pathlib import Path
import subprocess

from .audit_counts import HERE, REPO, read
from .build import write


def main():
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH='.:src')
    reports=[]
    for job in read(HERE/'CAMPAIGNS.json')['jobs']:
        if job['domain']!='r10_libero':continue
        for arm in job['arms']:
            log=Path(job['run'])/'validation'/f'{arm}.log';log.parent.mkdir(exist_ok=True)
            argv=[str(REPO/'.venv/bin/python'),'-m','exp.offline_search.rounds.r10.selftest','--run',job['run'],'--arm',arm]
            with log.open('w') as stream:subprocess.run(argv,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            report=read(Path(job['run'])/'selftest'/arm/'selftest_report.json')
            assert report['PASS'] and report['miss']==0 and report['policy_tail']==0
            reports.append(report);print('CPU plugin PASS '+arm,flush=True)
    write(HERE/'R10_SELFTESTS.json',dict(status='PASS',arms=len(reports),decisions=sum(r['decisions'] for r in reports),reports=reports))


if __name__=='__main__':main()
