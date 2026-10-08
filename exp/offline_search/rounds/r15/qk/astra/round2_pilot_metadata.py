"""Read-only identity/cap/baseline audit. Does not read success outcomes."""
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from exp.offline_search.rounds.r15.qk.astra.prepare import HERE,REPO,STORE,RUNS,sha,dump


def main():
    source=REPO/'exp/offline_search/rounds/r15/descent/selection.json'
    selection=json.loads(source.read_text());pairs=selection['model_suites']['pi05_l10']['pairs']
    assert len(pairs)==len(set(map(tuple,pairs)))==300
    assert all(sum(t==task for t,i in pairs)==30 for task in range(10))
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'selection':str(source),
            'selection_sha256':sha(source),'pairs':pairs,'episodes':300,'per_task':30,
            'rollout_replicate_seed':selection['seed'],'independent_validation':False,
            'success_outcomes_read':False,'cells':{}}
    lib=STORE/'library/pi05_l10/bpool_cs';eps=json.loads((lib/'episodes.json').read_text());ep_ids=np.load(lib/'episode.npy')
    for size in (50,200):
        subpath=REPO/f'exp/offline_search/rounds/r10/subsets/pi05_l10_{size}.json'
        subset=json.loads(subpath.read_text())['episode_ids_by_task'];selected=set(sum(subset.values(),[]))
        own={(e['task_id'],e['orig_init_state_idx']) for e in eps if int(ep_ids[e['start']]) in selected}
        assert not own & set(map(tuple,pairs))
        f=REPO/f'exp/offline_search/rounds/r11/knob/artifacts/r11_pi05_l10_{size}_periodic_ir25.pkl'
        method=pickle.load(f.open('rb'))['method'];assert method.episode_subset==subset
        c={'size':size,'disjoint_from_library':True,'subset_sha256':sha(subpath),'ir25_cap':method.knob_settings['setting'],
           'ir25_artifact':str(f),'ir25_artifact_sha256':sha(f),'baseline_arms':[]}
        root=RUNS/f'r15_descent_pi05_l10_{size}'
        if root.exists():
            manifest=json.loads((root/'dev_selection.json').read_text())
            assert manifest['init_pool']=='B' and manifest['dev'] and manifest['pairs']==pairs
            assert manifest['selection_sha256']==sha(source)
            rows=json.loads((root/'arms.json').read_text())
            for row in rows:
                if row['kwargs'].get('method')!='off' and row['kwargs'].get('target')!=.25:continue
                assert row['dev'] and row['init_pool']=='B' and row['episodes']==300
                assert row['rollout_replicate_seed']==selection['seed']
                assert row['kwargs']['episode_subset']==subset
                run=root/'runs'/row['arm']
                c['baseline_arms'].append({'arm':row['arm'],'root':str(root),'run':str(run),
                    'level':row['r14_descent_level'],'intended_host':row['intended_host'],
                    'artifact':row['plugin_args'][row['plugin_args'].index('--os-fit-artifact')+1],
                    'replan_steps':row['replan_steps'],'full_model':row['full_model'],'recording':row['recording'],
                    'summary_exists_at_audit':(run/'summary.json').is_file(),
                    'run_manifest_exists_at_audit':(run/'manifest.json').is_file(),
                    'reuse':'eligible by planned identity; require 300 accepted paired completions and host/runtime/cap/fit equivalence before reuse'})
            c['root_manifest_sha256']=sha(root/'dev_selection.json');c['arms_sha256']=sha(root/'arms.json')
        else:
            c['reuse']='no size-50 R15 descent root; collect its current-system off/ir25 baselines on this exact list; size100/200 cannot substitute'
        report['cells'][f'pi05_l10_{size}']=c
    dump(HERE/'round2/pilot_identity_baseline.json',report)
    print(json.dumps({k:{'cap':v['ir25_cap'],'r15_baseline_arms':len(v['baseline_arms'])} for k,v in report['cells'].items()}))


if __name__=='__main__':main()
