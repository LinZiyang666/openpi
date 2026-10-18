"""Stage server inputs and run frozen CPU startup identity checks, never serve."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit_counts import HERE, read
from .build import write


def check(job, *, stage_assets=True):
    from exp.r17_integration.ops import formal_eval as launch
    from .stage import stage_server_inputs
    root = Path(job['run'])
    if any((root/name).exists() for name in ('LAUNCH.json','RECEIPT.json')):
        raise ValueError('CPU staging checker requires an unlaunched/reset run')
    c = read(root/'campaign.json')
    selected = read(HERE/'SERVER_ENVIRONMENTS.json')['jobs'][job['name']]
    c['server_python'] = selected['python']
    write(root/'campaign.json', c)
    if stage_assets:
        with launch.retry_scope(root):
            stage_server_inputs(launch, c, root)
    cpath = c.get('server_campaign') if c['server_node']=='h100' else str(root/'campaign.json')
    body = (HERE/'cpu_startup.py').read_text()
    code = 'import os; os.chdir('+repr(c['runtime'])+');\n'+body+'\nprint(json.dumps(probe('+repr(cpath)+')))'
    env = dict(launch.server_env(c, c['runtime']), CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu')
    argv = ['env', *[k+'='+v for k,v in env.items()], c['server_python'], '-c', code]
    output = launch.remote(c['server_node'], argv, seconds=600, replay_safe=True)
    result = json.loads(output.splitlines()[-1])
    result.update(source=job['source'], runtime=c['runtime'], server_node=c['server_node'])
    ready = read(HERE/'SOURCE_READY_ACTUAL.json')[job['source']]
    source_ids = sorted({r['spec']['checkpoint_identity'] if r.get('spec') else r.get('model_identity') for r in ready})
    result['source_actual_identities'] = source_ids
    result['source_ready_paths'] = [r['path'] for r in ready]
    result['equals_source_actual'] = (result['status']=='PASS' and bool(source_ids)
                                     and source_ids == [result['actual_spec']['checkpoint_identity']])
    directory = HERE/'startup_checks'
    directory.mkdir(exist_ok=True)
    write(directory/(job['name']+'.json'), result)
    print(json.dumps(dict(job=job['name'], passes=result['status']=='PASS', equals_source=result['equals_source_actual'],
                          error=result.get('error'))), flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', action='append')
    p.add_argument('--no-stage', action='store_true')
    args=p.parse_args()
    all_jobs=[j for j in read(HERE/'CAMPAIGNS.json')['jobs'] if j['domain']!='r10_libero']
    jobs=[j for j in all_jobs if j['name'] in args.job] if args.job else [j for j in all_jobs if j.get('arm','filtered')=='filtered']
    for job in jobs:
        check(job, stage_assets=not args.no_stage)
        rows=[read(HERE/'startup_checks'/(j['name']+'.json')) for j in all_jobs
              if j.get('arm','filtered')=='filtered' and (HERE/'startup_checks'/(j['name']+'.json')).exists()]
        write(HERE/'STARTUP_CHECKS.json', dict(cpu_only=True, status='PASS' if len(rows)==19 and all(r['status']=='PASS' for r in rows) else 'INCOMPLETE', jobs=rows))


if __name__=='__main__':main()
