"""Build real next-stage definitions in a private CPU-fixture directory only."""
import argparse
import copy
from pathlib import Path
import shutil
from unittest.mock import patch

import yaml

from .common import HERE, EVAL, LIBERO, read, dump


def r17():
    from . import build as b
    from .workflow import TEST_SOURCE, LABELS
    from exp.r17_integration.ops.cache_study import validate_manifest
    target=HERE/'cpu_fixtures/r17';target.mkdir(parents=True,exist_ok=True)
    results=[]
    for model in LIBERO:
        for mode in ('reduced','full'):
            source=EVAL/TEST_SOURCE[model][mode=='full'];ladder=EVAL/LIBERO[model][mode=='full']
            row=next(r for r in read(ladder/'assets/CACHE_STUDY.json')['groups'] if r['suite']=='libero_spatial' and r['library_size']==50)
            doc=yaml.safe_load((ladder/'assets'/row['file']).read_text());names=[n for n in doc['recipe']['ordered_tasks'] if n.startswith('libero_spatial/')]
            table={LABELS[model]:{'libero_spatial':{'50':{names[0]:'k10',names[1]:'pure'}}}}
            name=f'ffk1015_cpu_{model}_{mode}_A';root=target/name
            if not root.exists():
                with patch.object(b,'EVAL',target):b.build(source,name,table=table,ladder_source=ladder)
            validate_manifest(root/'STUDY.json')
            plan=read(root/'EPISODE_PLAN.json')
            assert plan['expected_episodes']==100
            assert {t['extra']['init_pool'] for t in plan['tasks']}=={'A'}
            assert {t['extra']['cache_group'].rsplit('_',1)[-1] for t in plan['tasks']}=={'pure','k10'}
            results.append(dict(model=model,mode=mode,episodes=100,status='PASS',launchable_campaign=False))
    dump(HERE/'CPU_FUTURE_R17.json',dict(cases=results,cpu_only=True,server_started=False,worker_started=False))
    print(results)


def vla():
    from . import vla as v
    target=HERE/'cpu_fixtures/vla';target.mkdir(parents=True,exist_ok=True)
    actual=v.ROOT;state=read(actual/'inventory/inventory.json')
    for relative in ('inventory/inventory.json','inventory/SEAL.json','CODE_LOCK.json','CELLS.json'):
        dest=target/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(actual/relative,dest)
    v.ROOT=target;v.w.ROOT=target
    requests=[dict(cell=f'{m}_{s}_50_k1',task=0,level='ir45') for m in ('pi05','groot') for s in ('spatial','l10')]
    path=target/'runs/dev01/plan.json'
    if not path.exists():v.emit('dev01','dev',requests,state)
    plan=read(path)
    for group in plan['groups']:v.w.checked_group(group)
    assert plan['episodes']==120
    dump(HERE/'CPU_FUTURE_R16.json',dict(status='PASS',episodes=120,groups=plan['groups'],cpu_only=True,
                                        server_started=False,worker_started=False,launchable_campaign=False))
    print('R16 next-dev-wave definitions: PASS (120 private fixture episodes, never simulated)')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--vla',action='store_true')
    vla() if p.parse_args().vla else r17()
