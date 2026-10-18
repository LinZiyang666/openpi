"""Plan all potential test-A levels without building/dispatching skipped cells."""
import copy
from pathlib import Path
from unittest.mock import patch
from .audit import HERE
from .io_budget import dump
from .workflow import name
from exp.offline_search.rounds.r17.filter_failed_knob.common import read,sha
from exp.offline_search.rounds.r17.filter_failed_knob.build import cpu_plan
from exp.r17_integration.ops import cache_study as study


def main():
    rows=[];units=read(HERE/'FORMAL_QUEUE.json')['units']
    for u in units:
        if u['stage']!='levels':continue
        off=next(v for v in units if v['name']==name(u['model'],u['mode'],u['arm'],'off'))
        j=off['jobs'][0];c=read(Path(j['run'])/'campaign.json');src=Path(u['source'])
        native=copy.deepcopy(c);native['local_assets']=str(src/'assets')
        _,_,all_tasks,_=cpu_plan(native)
        original=study.selected_groups
        def levels(c):
            seal,groups=original(c)
            return seal,[g for g in groups if not g['group'].endswith(('_koff','_pure'))]
        with patch.object(study,'selected_groups',levels):_,_,tasks,_=cpu_plan(native)
        ids={t.task_uid for t in tasks}
        assert len(ids)==len(tasks)==30000 and ids<={t.task_uid for t in all_tasks}
        descriptors=('load.json','model_spec.json','model_manifest.json','full.yaml','benchmark.json')
        equality={n:sha(src/'assets'/n)==sha(Path(j['run'])/'assets'/n) for n in descriptors}
        assert all(equality.values())
        proof=read(HERE/'cpu'/f'{j["name"]}.json');assert proof['status']=='PASS'
        c2=read(src/'campaign.json')
        assert c2['runtime']==c['runtime'] and c2['server_python']==c['server_python']
        pipe=proof['mps_pipe'].replace(j['name'],u['name']);assert len(pipe.encode())<=85
        rows.append(dict(name=u['name'],status='PASS',potential_episodes=len(tasks),unique_task_uids=len(ids),
            source_uid_subset=True,descriptors_equal=equality,startup_proof=str(HERE/'cpu'/f'{j["name"]}.json'),
            runtime_interpreter_equal_to_probed_OFF=True,mps_pipe=pipe,mps_pipe_bytes=len(pipe.encode()),
            pending='Actual tuned cells depend only on new OFF + same-step same-GPU pure, not withdrawn outcomes'))
    dump(HERE/'DEFERRED_CPU_PROOF.json',dict(status='PASS',rows=rows))
if __name__=='__main__':main()
