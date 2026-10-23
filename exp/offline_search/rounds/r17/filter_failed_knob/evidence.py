"""Let the native CPU evidence audit recognize the owner's empty-cache MISS.

Workers and their evidence schema stay unchanged. Eligibility is reconstructed
from the sealed selection, original episode labels and the planned LOEO init.
Every other native audit condition remains in force.
"""
from pathlib import Path
import json


def empty_uids(directory):
    directory=Path(directory)
    read=lambda p:json.loads(Path(p).read_text())
    campaign=read(directory/'campaign.json')
    meta=campaign.get('filter_failed_knob',{})
    if not meta.get('selection'):return set()
    groups={(r['suite'],r['library_size'],r['group']):r
            for r in read(directory/'assets/CACHE_STUDY.json')['groups']}
    labels={};cohorts={};allowed=set()
    name='FULL_EPISODE_PLAN.json' if campaign.get('cache_resume_selection') else 'EPISODE_PLAN.json'
    for task in read(directory/name)['tasks']:
        ex=task['extra'];suite='mt50' if campaign['domain']=='metaworld' else task['experiment']
        group=groups[suite,ex['library_size'],ex['cache_group']]
        kept=meta['selection'].get(group['fit_digest'])
        if kept is None or group.get('periodic_gap')==0:continue
        key=suite,ex['library_size'],group['fit_digest']
        if key not in cohorts:
            # Seeded libraries (seeds_1023) carry their own labels on the group row.
            path=(group.get('labels') or meta['cells'][f'{suite}/{ex["library_size"]}']['labels']).split('#')[0]
            if path not in labels:labels[path]=read(path)['metadata']['episodes']
            ids=set(kept);by_task={}
            for episode in labels[path]:
                if episode['library_episode_id'] in ids:
                    by_task.setdefault(episode['task_name'],[]).append(episode['orig_init_idx'])
            cohorts[key]=by_task
        inits=cohorts[key].get(ex['task_name'],[])
        if ex.get('serve_loeo'):
            inits=[i for i in inits if i!=task['orig_init_state_idx']]
        if not inits:allowed.add(task['task_uid'])
    return allowed


def install():
    import inspect
    import textwrap
    from exp.r17_integration.ops import cache_study as study
    source=textwrap.dedent(inspect.getsource(study.aggregate))
    marker='    composite=None\n'
    condition="if source=='policy' and (ex['miss_mode']!='stall_only' or not ("
    if marker not in source or condition not in source:
        raise ValueError('Inspect the frozen empty-MISS evidence branch')
    source=source.replace(marker,'    ffk_empty=_ffk_empty_uids(directory)\n'+marker,1)
    source=source.replace(condition,"if source=='policy' and task['task_uid'] not in ffk_empty and (ex['miss_mode']!='stall_only' or not (",1)
    namespace=dict(vars(study),_ffk_empty_uids=empty_uids)
    exec(compile(source,inspect.getfile(study),'exec'),namespace)
    study.aggregate=namespace['aggregate']
