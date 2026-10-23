"""Success-filtered R16 k1 cohorts, using the original workflow and conductor.

Run through vla.sh so the R16 sealed snapshot supplies every serving import.
Only the inventory's non-pure measurements are cleared; pure references remain.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import pickle
import shutil

import numpy as np

from .common import HERE, LEGACY, read, dump, sha, summarize
from exp.offline_search.rounds.r17.filter_failed.row_filter import _slice, R10_CANDIDATES
from exp.offline_search.rounds.r16.k1ref import workflow as w

ROOT = HERE/'vla'
SOURCE = Path('/home/weiland/projects/openpi/exp/offline_search/rounds/r16/k1ref')


def candidates(blob, kept, empty=()):
    """Keep only the kept episodes' candidate rows. Empty tasks are refused: TaskDescentKnob's c10 queries
    Q3StallInner/base before forcing a MISS, and --os-policy-tail may serve the next decision without a call.
    An unfiltered placeholder is therefore not a no-retrieval fallback, even when explicitly declared empty."""
    result = dict(blob)
    method = copy.copy(blob['method']); result['method'] = method
    method.inner = copy.copy(method.inner)
    base = copy.copy(method.inner.base); method.inner.base = base
    selected = np.isin(base.lib_ep, list(kept)); base.tasks = {}
    missing = sorted(int(t) for t, table in blob['method'].inner.base.tasks.items() if not selected[table.rows].any())
    if missing != sorted(map(int, empty)):
        raise ValueError(f'tasks without a kept episode {missing} differ from declared {sorted(empty)}')
    if missing:
        raise ValueError(f'empty tasks {missing}: c10 still queries the unfiltered table and permits policy tails; '
                         'a no-retrieval fallback must be defined before serving this cell')
    for task, table in blob['method'].inner.base.tasks.items():
        mask = selected[table.rows]
        base.tasks[task] = table if mask.all() else _slice(table,mask,R10_CANDIDATES)
    if hasattr(method,'_full_tasks'):
        method._full_tasks = dict(base.tasks)
    return result


def audit(state):
    rows=[]
    for cell in w.CELLS:
        model, short, size, _=w.parts(cell)
        row=state['dev_sources'][w.key(cell,0,'off')]['row']
        parent=LEGACY/f'{model}_{short}'/row['kwargs']['library']
        ids=sorted(i for es in row['kwargs']['episode_subset'].values() for i in es)
        blob=pickle.loads(w.rt.fit_path(row).read_bytes()); method=blob['method']
        ep=np.load(parent/'episode.npy',mmap_mode='r')
        assert np.array_equal(method.inner.base.lib_ep,ep[method.row_subset]), 'R16 parent row/episode provenance differs'
        assert set(np.unique(method.inner.base.lib_ep))==set(ids)
        summary=summarize(read(parent/'episodes.json'),ids,id_key='episode_id',task_key='task_id')
        rows.append(dict(cell=cell,model=model,suite=row['suite'],size=size,parent=str(parent),
                         ignored_legacy_row_library=row['library'],original_fit=str(w.rt.fit_path(row)),
                         labels=str(parent/'episodes.json'),**summary))
    dump(ROOT/'CELLS.json',dict(cells=rows))
    return {r['cell']:r for r in rows}


def configure():
    w.ROOT=ROOT
    return w


def collect():
    state=w.collect()
    # Previously measured pure C10 cohorts are library-independent, just like
    # the shared A/B pure references. No old cache/off measurement is reused.
    source=read(SOURCE/'inventory/inventory.json')
    for key,value in source['test'].items():
        if key.endswith('/c10') and key not in state['test']:state['test'][key]=value
    state['reused_pure_c10_source']=str(SOURCE/'inventory/inventory.json')
    return state


def initialize():
    ROOT.mkdir(exist_ok=False)
    state=read(SOURCE/'inventory/inventory.json')
    state.update(dev={},test={},filter_failed_knob=True)
    state['references'] = copy.deepcopy(state['references'])
    dump(ROOT/'inventory/inventory.json',state)
    w.seal(ROOT/'inventory',['inventory.json'])
    shutil.copyfile(SOURCE/'CODE_LOCK.json',ROOT/'CODE_LOCK.json')
    audit(state)
    return state


def emit(name,phase,requests,state,*,previous=None):
    """R16 sparse group format, with private filtered artifacts and source UIDs."""
    out=ROOT/'runs'/name;w.fresh(out);out.mkdir(parents=True)
    reader=w.a.Reader();design=w.a.verify(reader); lookup={r['cell']:r for r in read(ROOT/'CELLS.json')['cells']}
    groups=[]
    for model in ('pi05','groot'):
        chosen=[r for r in requests if w.parts(r['cell'])[0]==model]
        if not chosen:continue
        group=copy.deepcopy(w.c.read_group(state['original_groups'][model],reader))
        group_root=out/f'ffk1015_{model}_libero_reduced_{name}';group_root.mkdir()
        entries=[];buckets={}
        for r in chosen:buckets.setdefault((r['cell'],r['level'] if phase=='dev' else ''),[]).append(r)
        for (cell,fixed),rows in sorted(buckets.items()):
            tasks=sorted(r['task'] for r in rows); arm=f'ffk1015_{cell}_{name}'+('_'+fixed if fixed else '')
            if phase=='dev':
                entry=copy.deepcopy(state['dev_sources'][w.key(cell,0,fixed)])
                entry.update(original_arm=entry['row']['arm'],template_pairs=entry['pairs'],
                             levels={str(t):fixed for t in range(10)})
                entry['pairs']=[p for p in entry['template_pairs'] if p[0] in tasks]
                entry['dispatch_pairs']=entry['pairs']; entry.pop('reuse',None)
            else:
                table=state['serving_levels'][cell]
                entry=w.rt.build_entry(state['originals'][cell],group_root,arm,table,tasks,design,reader)
                entry['template_pairs']=w.rt.PAIRS;entry['levels']=table
            row=entry['row']; original=Path(entry['source_root'])
            definitions=group_root/'filtered'/arm;definitions.mkdir(parents=True)
            blob=pickle.loads(reader.read(w.rt.fit_path(row)))
            blob=candidates(blob,lookup[cell]['kept_episode_ids'])
            fit=definitions/'fit.pkl';fit.write_bytes(pickle.dumps(blob,protocol=4))
            row['plugin_args'][row['plugin_args'].index('--os-fit-artifact')+1]=str(fit)
            for field,filename in (('yaml','config.yaml'),('manifest','manifest.json')):
                dest=definitions/filename;dest.write_bytes(reader.read(row[field]));row[field]=str(dest)
            row['arm']=arm;row['episodes']=len(entry['dispatch_pairs'])
            dump(definitions/'arms.json',[row])
            dump(definitions/'ROOT.json',dict(definitions_sha256={p.name:sha(p) for p in definitions.iterdir() if p.is_file()},
                                            fit_sha256={str(fit):sha(fit)},recording=False,fleet_binding=False))
            entry.update(source_root=str(definitions),source_seal_sha256=sha(definitions/'ROOT.json'),
                         fit_sha256=sha(fit),yaml_sha256=sha(row['yaml']))
            if phase=='dev':
                entry['pool_contract']=dict(entry['pool_contract'],fit_sha256=sha(fit))
            entries.append(entry)
        # An unchanged dormant fit supplies the other suite checkpoint if needed.
        for suite in ('libero_10','libero_spatial'):
            if not any(e['row']['suite']==suite for e in entries):
                e=copy.deepcopy(next(e for e in group['entries'] if e['row']['suite']==suite))
                e.update(original_arm=e['row']['arm'],template_pairs=e['pairs'],pairs=[],dispatch_pairs=[],levels={})
                entries.append(e)
        group.update(root=str(group_root),phase=phase,entries=entries,references=None,k1ref=True,
                     k1ref_plan=str(out/'plan.json'),selected_at_utc=w.now(),original_episode_uids=True,
                     yaml_id_is_original_arm=False)
        from exp.offline_search.closed_loop.devset import pool_record
        group['pool_digests']={s:pool_record(s,'B' if phase=='dev' else 'A')[1]['per_task_digests'] for s in ('libero_10','libero_spatial')}
        group['definitions_sha256']={str(p):sha(p) for p in sorted(group_root.rglob('*')) if p.is_file()}
        for e in entries:
            for p in (e['row']['yaml'],e['row']['manifest'],str(w.rt.fit_path(e['row']))):group['definitions_sha256'][p]=sha(p)
        group.pop('group_sha256',None);group['group_sha256']=w.a.sha(w.c.canonical(group))
        dump(group_root/'group.json',group);groups.append(str(group_root/'group.json'))
    plan=dict(schema='r16.k1ref.plan.v1',name=name,phase=phase,requests=requests,groups=groups,
              group_sha256={p:w.c.read_group(p)['group_sha256'] for p in groups},
              episodes=sum(len(e['dispatch_pairs']) for p in groups for e in w.c.read_group(p)['entries']),
              inventory_sha256=sha(ROOT/'inventory/inventory.json'),code_lock_sha256=sha(ROOT/'CODE_LOCK.json'),
              previous_analysis=str(previous) if previous else None,previous_sha256=sha(previous) if previous else None,inputs=reader.inputs)
    dump(out/'plan.json',plan);(out/'queue.txt').write_text(''.join(f'@local {p}\n' for p in groups));w.seal(out,['plan.json','queue.txt'])
    for p in groups:w.checked_group(p)
    print(json.dumps(dict(name=name,episodes=plan['episodes'],queue=str(out/'queue.txt'))))
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['initial','off','dev','freeze','test','collect','verify','cpu'])
    p.add_argument('--generation',type=int,default=1);p.add_argument('--previous',type=Path)
    p.add_argument('--group',type=Path)
    args=p.parse_args();configure()
    if args.action=='verify':
        from exp.offline_search.rounds.r16.k1ref.snapshot import verify
        verify(w.SERVER_REPO)
        for p in sorted(ROOT.glob('runs/*/plan.json')):
            for group in read(p)['groups']:w.checked_group(group)
        print('FFK_R16_DEFINITIONS_OK');return
    if args.action=='cpu':
        # Registry installation parses the actual staged plugin/fit/YAML inputs
        # and shares their CPU arrays. It never constructs a policy or server.
        from exp.offline_search.rounds.r16.resident import install
        group=w.checked_group(args.group)
        log_root=ROOT/'cpu'/Path(group['root']).name;log_root.mkdir(parents=True,exist_ok=True)
        test_group=dict(group,root=str(log_root))
        registry=install(test_group,group['model'],'cpu')
        import torch
        assert not torch.cuda.is_initialized()
        result=dict(group=str(args.group),group_sha256=group['group_sha256'],model=group['model'],
                    entries=len(group['entries']),status='PASS',cuda_initialized=False,model_loaded=False,
                    cpu_registry_installed=True,worker_started=False,server_started=False)
        dump(log_root/'PROOF.json',result);print(json.dumps(result));return
    if args.action=='initial':
        state=initialize()
        state['serving_levels']={cell:{str(t):'off' for t in range(10)} for cell in w.CELLS}
        emit('off','test',[dict(cell=cell,task=t,level='off') for cell in w.CELLS for t in range(10)],state)
        # First-wave definition is conditional on the off-A rule; emit it after
        # that outcome is known so qualifying cells never enter descent.
        return
    state=collect()
    if args.action=='collect':dump(ROOT/'COLLECTED.json',state);return
    if args.action=='off':
        _,missing=w.off_decisions(state);state['serving_levels']={cell:{str(t):'off' for t in range(10)} for cell in w.CELLS}
        emit('off','test',missing,state)
    elif args.action=='dev':
        selection=w.operational_dev_selection(state)
        if selection['missing']:emit(f'dev{args.generation:02d}','dev',selection['missing'],state)
        else:print('DEV_COMPLETE')
    elif args.action=='freeze':
        selection=w.operational_dev_selection(state);w.a.require(selection['fully_resolved'],'Missing dev cohorts')
        out=ROOT/'freeze';w.fresh(out);out.mkdir()
        dump(out/'levels.json',selection['levels']);dump(out/'dev.json',selection)
        dump(out/'TABLE_FREEZE.json',dict(frozen_at_utc=w.now(),inventory_sha256=sha(ROOT/'inventory/inventory.json'),
             code_lock_sha256=sha(ROOT/'CODE_LOCK.json'),supplement_inputs=state['supplement_inputs']))
        w.seal(out,['levels.json','dev.json','TABLE_FREEZE.json'])
    elif args.action=='test':
        initial=read(args.previous)['selection']['levels'] if args.previous else w.frozen()
        selection=w.resolve_test(state,initial,allow_raise=args.previous is not None)
        out=ROOT/'analyses'/f'g{args.generation:02d}';out.mkdir(parents=True,exist_ok=False)
        dump(out/'analysis.json',dict(selection=selection,post_test_selected=True));state['serving_levels']=selection['levels']
        if selection['missing']:emit(f'test{args.generation:02d}','test',selection['missing'],state,previous=out/'analysis.json')
        else:dump(ROOT/'FINAL.json',dict(selection=selection,episodes={cell:w.c.pack(w.matrix(state,cell,selection['levels'][cell]),{}) for cell in w.CELLS}))


if __name__=='__main__':main()
