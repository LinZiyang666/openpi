"""CPU-only future-level plans, preserving old scientific task namespaces."""
from dataclasses import asdict
from pathlib import Path
import copy
import hashlib
import time

from .audit import HERE
from .io_budget import filesystem_budget,dump
from .recalibrate import recipe_inputs,scientific_hash
from .workflow import name
from exp.offline_search.rounds.r17.filter_failed_knob.common import read,sha
from exp.offline_search.rounds.r17.filter_failed_knob.build import cpu_plan,publish_study


def main():
    units=read(HERE/'FORMAL_QUEUE.json')['units'];proofs=[]
    with filesystem_budget():
        for u in units:
            if u['stage']!='levels':continue
            off=next(v for v in units if v['name']==name(u['model'],u['mode'],u['arm'],'off'))
            base=Path(off['jobs'][0]['run']);src=Path(u['source'])
            c=read(base/'campaign.json');c['local_assets']=str(src/'assets')
            seal=read(src/'assets/CACHE_STUDY.json')
            seal['groups']=[g for g in seal['groups'] if not g['group'].endswith(('_koff','_pure'))]
            root=HERE/'recalibration_templates'/u['name']
            if root.exists():root.rename(root.with_name(root.name+'.previous_'+str(time.time_ns())))
            (root/'assets/cache_inputs').mkdir(parents=True)
            for file in ('load.json','model_spec.json','model_manifest.json','benchmark.json','full.yaml'):
                (root/'assets'/file).write_bytes((src/'assets'/file).read_bytes())
            old=copy.deepcopy(c)
            # Match the full old level-grid subset with the unchanged native
            # planner. No outcome or old OFF decision is read.
            from unittest.mock import patch
            from exp.r17_integration.ops import cache_study
            selected=cache_study.selected_groups
            def subset(campaign):
                s,g=selected(campaign)
                return s,[r for r in g if not r['group'].endswith(('_koff','_pure'))]
            with patch.object(cache_study,'selected_groups',subset):_,_,before,_=cpu_plan(old)
            manifest=read(src/'assets/cache_inputs/manifest.json')
            dump(root/'assets/cache_inputs/manifest.json',manifest)
            for g in seal['groups']:
                p=root/'assets'/g['file'];p.parent.mkdir(parents=True,exist_ok=True)
                if not p.exists():p.write_bytes((src/'assets'/g['file']).read_bytes())
            c.update(local_assets=str(root/'assets'),assets=str(root/'assets'),cache_study=str(root/'STUDY.json'))
            recipe_inputs(root,c,seal,aliases=False)
            publish_study(root,src,c,seal)
            _,_,after,_=cpu_plan(c)
            assert len(before)==len(after)==30000
            assert [t.task_uid for t in before]==[t.task_uid for t in after]
            assert len({t.task_uid for t in after})==len(after)
            assert scientific_hash(before)==scientific_hash(after)
            prior=read(HERE/'cpu'/f'{base.name}.json')
            pipe=prior['mps_pipe'].replace(base.name,u['name']);assert len(pipe.encode())<=85
            evidence=dict(name=u['name'],status='PASS',potential_episodes=len(after),
                unique_task_uids=len(after),UIDs_inits_seeds_equal=True,
                task_uid_sha256=hashlib.sha256('\n'.join(t.task_uid for t in after).encode()).hexdigest(),
                descriptor_bytes_equal={n:sha(root/'assets'/n)==sha(base/'assets'/n)
                    for n in ('load.json','model_spec.json','model_manifest.json','benchmark.json','full.yaml')},
                startup_proof=str(HERE/'cpu'/f'{base.name}.json'),
                runtime_interpreter_equal=True,mps_pipe_bytes=len(pipe.encode()),
                calibrated_level_groups=len(seal['groups']),Q3_decision_clock=True,
                no_campaign_built_or_staged=True)
            assert all(evidence['descriptor_bytes_equal'].values())
            proofs.append(evidence);print(u['name'],'PASS',len(after),'potential UIDs',flush=True)
    dump(HERE/'RECALIBRATION_DEFERRED_CPU_PROOF.json',dict(status='PASS',rows=proofs))


if __name__=='__main__':main()
