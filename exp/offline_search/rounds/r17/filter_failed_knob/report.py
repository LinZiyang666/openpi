"""Refresh the CPU audit/provenance and generated handback tables."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import pickle

from .common import HERE, cells, read, dump, summarize, sha


def main():
    rows=[c for c in cells().values() if c['source']=='R17']
    parents={}
    for c in rows:
        path=c['parent']
        if path not in parents:parents[path]=read(Path(path)/'READY.json')['metadata']['episodes']
        actual=summarize(parents[path],read(c['selection'])['episode_ids'],id_key='library_episode_id',task_key='task_name')
        assert (actual['collected'],actual['kept'],actual['failed'],set(actual['kept_episode_ids']))==(c['collected'],c['kept'],c['failed'],set(c['kept_episode_ids']))
    vla=read(HERE/'vla/CELLS.json')['cells']
    inventory=read(HERE/'vla/inventory/inventory.json');provenance=[]
    for c in vla:
        for level in ('c10','ir45','ir40','ir32','ir25','off'):
            source_cell=c['cell']
            if level=='c10' and f'{source_cell}/0/{level}' not in inventory['dev_sources']:
                source_cell=source_cell.replace('_200_k1','_100_k1').replace('_500_k1','_100_k1')
            entry=inventory['dev_sources'][f'{source_cell}/0/{level}'];args=entry['row']['plugin_args'];p=Path(args[args.index('--os-fit-artifact')+1])
            provenance.append(dict(cell=c['cell'],source_cell=source_cell,level=level,fit=str(p),fit_sha256=sha(p),library=c['parent'],labels=c['labels']))
        blob=pickle.loads(Path(c['original_fit']).read_bytes())
        assert Path(blob['method'].inner.base.act.path)==Path(c['parent'])/'action.npy'
        actual=summarize(read(c['labels']),c['kept_episode_ids']+[e for es in entry['row']['kwargs']['episode_subset'].values() for e in es if e not in c['kept_episode_ids']],id_key='episode_id',task_key='task_id')
        assert (actual['collected'],actual['kept'],actual['failed'])==(c['collected'],c['kept'],c['failed'])
    dump(HERE/'LIBRARY_PROVENANCE.json',dict(r17_count_audits=len(rows),r16_cells=len(vla),r16_dev_fits=provenance))
    all_rows=rows+vla
    grouped=defaultdict(list)
    totals=defaultdict(lambda:dict(collected_view_memberships=0,kept_view_memberships=0,collected_parent_episodes=0,kept_parent_episodes=0))
    for c in all_rows:
        domain='MT50' if c['suite']=='mt50' else 'LIBERO'
        grouped[c['model'],domain,c['suite']].append(c)
        t=totals[c['model']];t['collected_view_memberships']+=c['collected'];t['kept_view_memberships']+=c['kept']
        if c['size']==(2500 if domain=='MT50' else 500):t['collected_parent_episodes']+=c['collected'];t['kept_parent_episodes']+=c['kept']
    dump(HERE/'COUNTS.json',dict(cells=[{k:c[k] for k in ('model','suite','size','collected','kept','failed','empty_tasks')} for c in all_rows],totals=dict(totals)))
    lines=['### Initial campaign definitions','', '| Campaign (all start `ffk1015_`) | Episodes | GPU | R / B | Workers per node |','|---|---:|---|---|---|']
    jobs=read(HERE/'ALL_CAMPAIGNS.json')['jobs']
    for j in jobs:
        if j.get('run'):
            c=read(Path(j['run'])/'campaign.json')
            batch=max((c.get(k) or {}).get('max_batch',1) for k in ('dp_batching','fastwam_batching','stage_batching'))
        else:batch=1
        workers=', '.join(f'{n}×{v}' for n,v in j['workers'].items())
        lines.append(f'| `{j["name"].removeprefix("ffk1015_")}` | {j["episodes"]:,} | {j["gpu"]} | {j["replicas"]} / {batch} | {workers} |')
    lines+=['',f'Initial total: **{sum(j["episodes"] for j in jobs):,} episodes across {len(jobs)} definitions**. These totals exclude outcome-dependent A/raise stages and pending full-step VLA alternatives.','',
            '### Failures removed (kept episodes in parentheses)','','The two R17 step modes share the same episode sets. R16 counts apply to reduced k1; full VLA remains awaiting a decision. MT50 columns are 250/500/1000/1500/2000/2500; R17 LIBERO columns are 50/100/200/300/400/500; R16 columns are 50/100/200/500.','',
            '| Model | Benchmark / suite | Failed (kept), in size order |','|---|---|---|']
    for (model,domain,suite),cs in grouped.items():
        values=' / '.join(f'{c["failed"]} ({c["kept"]})' for c in sorted(cs,key=lambda c:c['size']))
        lines.append(f'| {model} | {domain} / {suite} | {values} |')
    lines+=['','### Episode totals per model','','Parent episodes count each maximum-size collected library once. View memberships count overlaps across nested sizes once per view; neither total is doubled for the two step modes.','',
            '| Model | Parent collected / kept | View memberships collected / kept |','|---|---:|---:|']
    for model,t in totals.items():lines.append(f'| {model} | {t["collected_parent_episodes"]:,} / {t["kept_parent_episodes"]:,} | {t["collected_view_memberships"]:,} / {t["kept_view_memberships"]:,} |')
    lines+=['','### Frozen startup matrix','','| Campaign | CPU startup | Actual source identity | Interpreter |','|---|---|---|---|']
    for j in jobs:
        p=HERE/'startup_checks'/(j['name']+'.json')
        if p.exists():
            r=read(p);actual='yes' if r.get('equals_source_actual') else 'N/A: source off not launched' if not r.get('source_actual_identities') else 'NO'
            lines.append(f'| `{j["name"]}` | {r["status"]} | {actual} | `{r["python"]}` |')
        elif j.get('pipeline')=='R16 k1ref':lines.append(f'| `{j["name"]}` | PASS: sealed snapshot + actual CPU registry | sealed R16 inputs | original R16 interpreter choices |')
        else:lines.append(f'| `{j["name"]}` | pending | pending | `{j["python"]}` |')
    path=HERE/'HANDBACK.md';text=path.read_text().split('<!-- GENERATED_TABLES -->')[0]
    path.write_text(text+'<!-- GENERATED_TABLES -->\n\n'+'\n'.join(lines)+'\n')
    print(f'{len(all_rows)} library-size audits; {len(provenance)} exact R16 source fits; {len(jobs)} campaigns')


if __name__=='__main__':main()
