"""Read-only inventory for the CPU prepare tests; never invoke remote tools."""
import argparse
import json
from pathlib import Path
from .defaults import physical_maximum
from .node import atomic

ROOT = Path('/data/mw_finetune/eval')
HERE = Path(__file__).parent
LONGSEG = HERE.parent/'longseg_1017'


def active_lists():
    return {lane:(LONGSEG/file).read_text().splitlines() for lane,file in
            (('ffk-h100ls','H100_NAMES.txt'),('ffk-4090ls','4090_NAMES.txt'))}


def unit(name):
    if name.startswith('fa_ls_'):
        queued=json.loads((LONGSEG/'FORMAL_QUEUE.json').read_text())['units']
        found=next((u for u in queued if u['name']==name),None)
        if found:return found
        job=next(j for j in json.loads((LONGSEG/'CAMPAIGNS.json').read_text())['jobs'] if j['name']==name)
        return dict(name=name,model=job['model'],mode=job.get('mode'),stage=job.get('stage'),
                    workers=job['workers'],status='BUILT',pipeline='R17',jobs=[job])
    from exp.offline_search.rounds.r17.filter_failed_knob.formal_a_queue import INDEX
    index=json.loads(INDEX.read_text())
    found=next((u for u in index['units'] if u['name']==name),None)
    if found:return found
    # fa_co_f_f_oc1 is dispatched through ff's continuation command, not QUEUE.
    c=json.loads((ROOT/name/'campaign.json').read_text())
    return dict(name=name,model=c['model'],mode='full',arm='filtered',stage='off_complete',status='BUILT',
        workers=c.get('execution_worker_nodes',c['worker_nodes']),jobs=[dict(name=name,run=str(ROOT/name),pipeline='R17')])


def inventory():
    rows=[]
    lists=active_lists()
    for name in dict.fromkeys(n for names in lists.values() for n in names):
        u=unit(name)
        pipeline=('R17' if name.startswith('fa_ls_') or u['model']=='cosmos' else 'R16' if u['mode']=='reduced' else 'R15')
        rows.append(dict(name=name,pipeline=pipeline,queue_status=u['status'],jobs=len(u['jobs']),
            queues=[lane for lane,names in lists.items() if name in names],
            disposition='kept_fixed' if name=='fa_co_f_f_oc1' else 'dynamic_at_prepare',
            reason='source driver/control/worker pins retained' if name=='fa_co_f_f_oc1' else 'shared prepare default',
            markers={j['name']:[n for n in ('PREPARED.json','LAUNCH.json','RECEIPT.json','launch.json','DONE.json') if (Path(j['run'])/n).exists()] for j in u['jobs']}))
    longseg=[]
    # The new queues explicitly include vNN controls; use sol's authoritative
    # built catalog rather than the earlier name-prefix exclusion.
    for job in json.loads((LONGSEG/'CAMPAIGNS.json').read_text())['jobs']:
        p=Path(job['run'])
        c=json.loads((p/'campaign.json').read_text())
        marks=[n for n in ('PREPARED.json','LAUNCH.json','RECEIPT.json','DONE.json','launch.json') if (p/n).exists()]
        mode=(c.get('autoscale') or {}).get('mode','fixed')
        disposition=('already_launched' if any(n!='PREPARED.json' for n in marks) else
                     'already_dynamic' if marks and mode=='dynamic' else
                     'reprepare_required' if marks else 'dynamic_at_prepare')
        longseg.append(dict(name=p.name,domain=c.get('domain'),markers=marks,prepared_mode=mode if marks else None,
            disposition=disposition,
            reprepare_command=('PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m exp.offline_search.rounds.r17.autoscale_1017.formal_trial prepare --run '+str(p)+' --reprepare') if disposition=='reprepare_required' else None))
    return dict(scope='local metadata only; prepare tests in tests/test_new_chains.py stub external effects',
        active_lists=lists,queued=rows,longseg=longseg,
        defaults={d:{n:physical_maximum(n,d) for n in ('timan1','timan107','local')} for d in ('libero','metaworld')})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'PREPARE_DRY_RUN.json')
    a=parser.parse_args()
    atomic(a.output,inventory())
    print(a.output)
