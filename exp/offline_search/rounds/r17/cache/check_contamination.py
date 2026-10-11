"""CPU/read-only source audit. Writes only sibling JSON/CSV analysis artifacts.

No simulator construction, GPU, remote calls, fitting, or campaign mutations.
Live ledgers are read to their initial byte length; complete-line prefix is pinned.
"""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.distance import cdist
import torch
import yaml

from exp.r17_integration.ops.library_alignment import array_identity, task_data_identity
from exp.r17_integration.debug.validate import digest
from exp.offline_search.rounds.r06.ideation_Q3.stall.stall import _digest as stall_digest, StallModel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
EVAL = Path('/data/mw_finetune/eval')
A_ROOT = Path('/home/weiland/projects/openpi_ext/envs/libero_sim/lib/python3.8/site-packages/libero/libero/init_files')
SUITES = ('libero_spatial', 'libero_10')
MODELS = (('DP', 'dp_libero', 'csl10d', 'hs5dp_ctl_142812'),
          ('Fast-WAM', 'fastwam_libero', 'csl10f', 'hs6fw_res_152931'),
          ('Cosmos', 'cosmos_libero', 'csl10c', 'formal_cosmos_libero_official2000_20261010_130639_a8ca2043'))
PINS = {}


def sha(path):
    p = Path(path); h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    value = h.hexdigest()
    assert str(p) not in PINS or PINS[str(p)] == value, ('changed static input', p)
    PINS[str(p)] = value
    return value


def obj(path):
    sha(path)
    return json.loads(Path(path).read_text())


def sealed(value, field='digest'):
    assert digest({k:v for k,v in value.items() if k != field}) == value[field]


def load_states(path):
    sha(path)
    values = torch.load(path, map_location='cpu', weights_only=False)
    assert len(values) == 50
    return [v.detach().cpu().numpy() if hasattr(v, 'detach') else np.asarray(v) for v in values]


def decode(node, arrays):
    if 'array' in node: return arrays[node['array']]
    if 'namespace' in node: return SimpleNamespace(**decode(node['namespace'], arrays))
    if 'mapping' in node: return {k:decode(v, arrays) for k,v in node['mapping']}
    if 'sequence' in node: return [decode(v, arrays) for v in node['sequence']]
    return node['value']


def ledger(path, identities, *, required_pool_labels=True, plan_extras=None):
    if not path.exists(): return dict(present=False, identities=0)
    start = datetime.now(timezone.utc).isoformat()
    size = path.stat().st_size; used = 0; h = hashlib.sha256(); count = 0
    bad = Counter(); seen = set(); labels = Counter(); plans = set()
    with path.open('rb') as f:
        while used < size:
            line = f.readline(size-used)
            if not line.endswith(b'\n'): break
            used += len(line); h.update(line)
            row = json.loads(line)
            name = row.get('task_name')
            if not name or name.split('/')[0] not in SUITES: continue
            idx = row.get('orig_init_state_idx', row.get('original_init_idx'))
            key = name, idx
            if key not in identities:
                bad['unknown_identity'] += 1; continue
            # client_timing rows may carry the same identity; all identified rows
            # must attest the same state, irrespective of eventual task outcome.
            count += 1; seen.add(key); plans.add(row.get('task_uid'))
            labels[str((row.get('init_pool'),row.get('partition')))] += 1
            if required_pool_labels:
                if row.get('init_pool') != 'A' or row.get('partition') != 'test': bad['pool_partition'] += 1
            elif row.get('init_pool') not in (None,'A') or row.get('partition') not in (None,'test'):
                bad['pool_partition'] += 1
            if row.get('task_data_hash') != identities[key]['task_data_hash']: bad['state_hash'] += 1
            if plan_extras is not None:
                extra=plan_extras[row['task_uid']]
                for field in ('bundle_identity','recipe_identity','yaml_identity'):
                    if row.get(field) != extra[field]: bad[field] += 1
    assert not bad, (path, bad)
    return dict(present=True, snapshot_utc=start, initial_bytes=size, complete_prefix_bytes=used,
                prefix_sha256=h.hexdigest(), identified_records=count, identities=len(seen),
                task_uids=len(plans), labels=dict(labels), mismatches=dict(bad),
                per_suite={s:sum(name.startswith(s+'/') for name,i in seen) for s in SUITES})


def main():
    result = dict(as_of_utc=datetime.now(timezone.utc).isoformat(), source_sha256=PINS,
                  tasks=[], init_distances=[], libraries=[], fits=[], lanes=[], controls=[])
    for p in [Path(__file__), HERE/'CACHE_STUDY.md', HERE/'OWNER_RULINGS.md',
              ROOT/'exp/r17_integration/ops/library_alignment.py',
              ROOT/'examples/libero/worker_entry.py', ROOT/'examples/libero/main.py',
              ROOT/'exp/r17_integration/benchmarks/libero_bridge.py',
              ROOT/'exp/r17_integration/search/fit.py', ROOT/'exp/r17_integration/ops/cache_study.py',
              ROOT/'exp/r17_integration/common/layers/recipe.py',
              ROOT/'exp/offline_search/rounds/r06/ideation_Q3/stall/stall.py',
              A_ROOT.parent/'benchmark/__init__.py']:
        sha(p)
    a_ids, b_ids, a_hashes, b_arrays = {}, {}, set(), {}
    for suite in SUITES:
        alignment = obj(ROOT/'exp/r17_integration/ops/samples/libero_library_alignment'/f'{suite}.json')
        sealed(alignment)
        witnesses = {(r['task_id'],r['orig_init_state_idx']):r for r in alignment['rows']}
        pool = yaml.safe_load(Path(alignment['pool_record']).read_text())
        assert sha(alignment['pool_record']) == alignment['pool_record_sha256']
        assert pool['init_pool'] == 'B'
        for local,f in sorted(alignment['task_files'].items(), key=lambda pair:int(pair[0])):
            local = int(local); name = suite+'/'+f['name']
            ap = A_ROOT/suite/(f['name']+'.pruned_init'); bp = Path(f['file'])
            a,b = load_states(ap),load_states(bp)
            full_path=A_ROOT/suite/(f['name']+'.init');sha(full_path)
            full=torch.load(full_path,map_location='cpu',weights_only=False)
            full_hash={array_identity(v)['state_sha256'] for v in full}
            ah={array_identity(v)['state_sha256'] for v in a};bh={array_identity(v)['state_sha256'] for v in b}
            assert len(full)==len(full_hash)==100 and ah<=full_hash and bh==full_hash-ah
            assert sha(bp) == f['sha256'] == pool['per_task_digests'][f['name']]
            for i in range(50):
                ai,bi = array_identity(a[i]),array_identity(b[i])
                assert bi == witnesses[local,i]['init_array']
                a_ids[name,i] = dict(**ai,task_data_hash=task_data_identity(a[i],name))
                b_ids[name,i] = dict(**bi,task_data_hash=task_data_identity(b[i],name))
                a_hashes.add(ai['state_sha256'])
            aa,bb = np.stack(a).astype(np.float64),np.stack(b).astype(np.float64)
            assert aa.shape == bb.shape and np.isfinite(aa).all() and np.isfinite(bb).all()
            ab=cdist(aa,bb); within_a=cdist(aa,aa); within_b=cdist(bb,bb)
            np.fill_diagonal(within_a,np.inf); np.fill_diagonal(within_b,np.inf)
            dab,daa,dbb=ab.min(1),within_a.min(1),within_b.min(1)
            q=float(np.quantile(dbb,.01,method='linear'))
            task=dict(suite=suite,task=name,task_id=local+(30 if suite=='libero_10' else 0),
                local_task_id=local,dim=aa.shape[1],dtype=str(a[0].dtype),
                a_path=str(ap),b_path=str(bp),a_count=50,b_count=50,
                a_unique=len({array_identity(x)['state_sha256'] for x in a}),
                b_unique=len({array_identity(x)['state_sha256'] for x in b}),
                exact=sum(a_ids[name,i]['state_sha256'] in {b_ids[name,j]['state_sha256'] for j in range(50)} for i in range(50)),
                median_aa=float(np.median(daa)),median_ab=float(np.median(dab)),median_bb=float(np.median(dbb)),
                bb_q01=q,bb_min_nn=float(dbb.min()),min_ab=float(dab.min()),flags=int((dab<q).sum()),
                full_path=str(full_path),full_count=100,B_exact_full_minus_A=True)
            result['tasks'].append(task);b_arrays[name]=b
            for i in range(50):
                j=int(ab[i].argmin());delta=aa[i]-bb[j]
                result['init_distances'].append(dict(suite=suite,task=name,task_id=task['task_id'],
                    a_init=i,b_nearest_init=j,a_sha256=a_ids[name,i]['state_sha256'],b_sha256=b_ids[name,j]['state_sha256'],
                    ab_nn=float(dab[i]),aa_nn=float(daa[i]),nearest_b_bb_nn=float(dbb[j]),bb_q01=q,
                    ratio_ab_to_bb_q01=float(dab[i]/q) if q else None,flag=bool(dab[i]<q),
                    max_abs_component=float(np.max(np.abs(delta))),changed_components=int(np.count_nonzero(delta))))
    assert len(a_ids)==1000 and len(a_hashes)==1000
    result['a_unique_total']=len(a_hashes)
    for model,bundle,lane_name,control in MODELS:
        lane=EVAL/lane_name;inputs=lane/'study_inputs';meta=obj(inputs/'manifest.json');sealed(meta)
        assert meta['rulings_sha256']==sha(HERE/'OWNER_RULINGS.md')
        for name,pin in meta['files'].items():
            assert sha(inputs/name)==pin, ('transported artifact checksum',inputs/name)
            local=lane/'assets/cache_inputs'/name
            if local.exists(): assert sha(local)==pin
        plan=obj(lane/'EPISODE_PLAN.json');spec=obj(lane/'assets/benchmark.json')
        cache_seal=obj(lane/'assets/CACHE_STUDY.json');sealed(cache_seal)
        assert cache_seal['eval_pool']=='A' and cache_seal['inputs_digest']==meta['digest']
        catalog={(g['suite'],g['group']):g for g in cache_seal['groups']}
        for g in catalog.values(): assert sha(lane/'assets'/g['file'])==g['sha256']
        ack_path=lane/'STUDY_BUNDLE_ACKS.json';acked=0
        if ack_path.exists():
            ack_doc=obj(ack_path);assert ack_doc['seal_digest']==cache_seal['digest']
            for row in ack_doc['rows']:
                g=catalog[row['suite'],row['group']]
                assert row['ack']['__ack__']=='load_cache_config'
                for key in ('bundle_id','bundle_identity'): assert row[key]==row['ack'][key]==g[key]
                acked+=1
        assert spec['formal_indices']==list(range(50))
        groups={}
        for t in plan['tasks']:
            e=t['extra'];key=e['task_name'],t['orig_init_state_idx']
            assert key in a_ids and e['init_pool']=='A' and e['partition']=='test'
            assert spec['tasks'][t['task_id']][0]==key[0]
            g=catalog[t['experiment'],e['cache_group']]
            assert e['yaml_identity']==g['sha256'] and e['recipe_identity']==g['recipe_identity']
            assert e['bundle_identity']==g['bundle_identity'] and t['bundle_id']==g['bundle_id']
            groups.setdefault((e['cache_group'],t['experiment']),[]).append(key)
        assert all(len(v)==len(set(v))==500 for v in groups.values())
        lane_info=dict(model=model,lane=str(lane),planned=len(plan['tasks']),groups=len(groups),
                       planned_unique_inits=len(set(k for v in groups.values() for k in v)),
                       live=ledger(lane/'driver/per_step.jsonl',a_ids,
                                   plan_extras={t['task_uid']:t['extra'] for t in plan['tasks']}),
                       yaml_count=0,server_bundle_acks=acked,
                       inputs_digest=meta['digest'],prereg_matches=meta['prereg_sha256']==sha(HERE/'CACHE_STUDY.md'))
        result['controls'].append(dict(model=model,run=control,ledger=ledger(EVAL/control/'driver/per_step.jsonl',a_ids,required_pool_labels=False)))
        assert result['controls'][-1]['ledger']['identities']==1000
        for suite in SUITES:
            parent_path=inputs/suite/'parent/READY.json';parent=obj(parent_path);sealed(parent,'content_digest')
            source=Path('/data/r17_library')/(bundle+'_'+suite)/'bpool'
            assert sha(source/'READY.json')==sha(parent_path)
            for path,pin in parent['sources'].items(): assert sha(path)==pin
            episodes=parent['metadata']['episodes'];rows=parent['rows'];eps={e['library_episode_id']:e for e in episodes}
            assert len(episodes)==len(eps)==500 and set(eps)==set(range(500))
            assert parent['metadata']['init_pool']=='B'
            bad_rows=0
            for e in episodes:
                name=e['task_name'];i=e['orig_init_idx'];ident=b_ids[name,i]
                assert name.startswith(suite+'/') and e['init_pool']=='B' and e['partition']==e['split']=='library'
                assert e['init_state_sha256']==ident['state_sha256'] and e['task_data_hash']==ident['task_data_hash']
                assert e['library_episode_id']==e['suite_task_id']*50+i
            for row in rows:
                ep=eps[row['library_episode_id']]
                assert row['init_pool']=='B' and row['split']=='library'
                for key in ('task_uid','task_name','orig_init_idx','init_state_sha256','task_data_hash','attempt'):
                    assert row[key]==ep[key],('row episode identity mismatch',model,suite,key)
                assert row.get('partition','library')=='library'
            view_path=parent_path.parent/'subsets/size_500';view=obj(view_path/'VIEW.json');sealed(view,'content_digest')
            assert view['parent_digest']==parent['content_digest'] and view['binding']==parent['binding']
            assert sha(view_path/'row_indices.npy')==view['row_indices_sha256']
            indices=np.load(view_path/'row_indices.npy',allow_pickle=False)
            assert np.array_equal(indices,np.arange(len(rows))) and view['size']==500 and set(view['episode_ids'])==set(eps)
            assert sha(source/'subsets/size_500/VIEW.json')==sha(view_path/'VIEW.json')
            base_arrays={}
            for field,entry in parent['arrays'].items():
                if field.startswith('key_') or field in ('state','normalized_actions'):
                    assert sha(source/entry['file'])==entry['sha256']
                    base_arrays[field]=np.load(source/entry['file'],mmap_mode='r',allow_pickle=False)
                    assert len(base_arrays[field])==len(rows)
            item=dict(model=model,suite=suite,parent=str(parent_path),source=str(source),rows=len(rows),episodes=500,
                      exact_global=sum(e['init_state_sha256'] in a_hashes for e in episodes),
                      exact_same_task=sum(e['init_state_sha256'] in {a_ids[e['task_name'],i]['state_sha256'] for i in range(50)} for e in episodes),
                      b_verified=500,b_only_rows=len(rows),capture_manifests_verified=len(parent['sources']),
                      parent_digest=parent['content_digest'],selection_digest=view['content_digest'],
                      flags=sum(t['flags'] for t in result['tasks'] if t['suite']==suite))
            result['libraries'].append(item)
            for fit_dir in sorted((inputs/suite).glob('C*/fit')):
                fit=obj(fit_dir/'READY.json');sealed(fit);receipt=fit['receipt'];sealed(receipt)
                assert receipt['store_digest']==view['content_digest']
                assert fit['fit_evidence']['partition']=='library' and fit['fit_evidence']['source_digest']==view['content_digest']
                assert receipt['binding']==parent['binding']
                assert sha(fit_dir/'fit.npz')==fit['arrays_sha256']
                with np.load(fit_dir/'fit.npz',allow_pickle=False) as arrays:
                    frozen=decode(fit['tree'],arrays)
                for field,src in [('state','state'),('action','normalized_actions')]:
                    assert np.array_equal(frozen[field],np.asarray(base_arrays[src],dtype=np.float32))
                assert np.array_equal(frozen['episode'],[r['library_episode_id'] for r in rows])
                assert np.array_equal(frozen['step'],[r['decision_seq'] for r in rows])
                task_names=receipt['config']['task_names']
                assert np.array_equal(frozen['task'],[task_names.index(r['task_name']) for r in rows])
                assert len(fit['fit_evidence']['loeo'])==len(rows)
                assert {r['row'] for r in fit['fit_evidence']['loeo']}==set(range(len(rows)))
                for witness in fit['fit_evidence']['loeo']:
                    q=witness['query_episode'];donors=witness['donor_episodes']
                    assert q==int(frozen['episode'][witness['row']]) and q in eps and q not in donors
                    assert set(donors)<=set(eps)
                    assert all(eps[x]['task_name']==eps[q]['task_name'] for x in donors)
                folds=fit['fit_evidence']['distance_folds']
                assert len(folds)==5
                for fold in folds:
                    tr,te=set(fold['train_episodes']),set(fold['heldout_episodes'])
                    assert not tr&te and tr|te==set(eps)
                assert Counter(e for fold in folds for e in fold['heldout_episodes'])==Counter(eps.keys())
                stall_dir=fit_dir.parent/'stall';sr=obj(stall_dir/'READY.json');sealed(sr)
                assert sr['store_digest']==view['content_digest'] and sr['fitted_digest']==fit['digest']
                for name,pin in sr['files'].items(): assert sha(stall_dir/name)==pin
                assert sr['calibration_partition']=='B_library_successful_LOEO' and sr['independent_B_validation_recording'] is False
                sm=obj(stall_dir/'stall.json');R=parent['binding']['model']['layout']['replan_steps']
                labels=dict(task=frozen['task'],episode=frozen['episode'],step=frozen['step'],controls=frozen['step']*R,
                            success=np.array([r['success'] for r in rows],bool))
                assert sm['provenance']['bank_labels_sha256']==stall_digest(labels)
                assert sm['provenance']['metric_provenance']['source_digest']==view['content_digest']
                trained=StallModel.load(stall_dir)
                assert stall_digest(trained.metric)==sm['provenance']['metric_sha256']
                for key,mt in trained.metric['tasks'].items():
                    table=frozen['tasks'][int(key)]
                    assert np.array_equal(mt['rows'],table.rows) and np.array_equal(mt['codes']['main'],table.Z)
                for task,part in trained.tasks.items():
                    for section in ('templates','references','calibration'):
                        for sample in part[section]:
                            ep=eps[sample['episode']]
                            assert ep['success'] is True and task_names[int(task)]==ep['task_name']
                csv_rows=list(csv.DictReader(io.StringIO((stall_dir/'stall.loeo.csv').read_text())))
                for row in csv_rows:
                    ep=eps[int(row['episode'])]
                    assert ep['success'] is True and task_names[int(row['task_key'])]==ep['task_name']
                assert len(csv_rows)==sum(len(v['calibration']) for v in trained.tasks.values())
                result['fits'].append(dict(model=model,suite=suite,C=int(fit_dir.parent.name[1:]),fit=str(fit_dir),
                    fit_digest=fit['digest'],stall_digest=sr['digest'],loeo_rows=len(rows),folds=5,episodes=500,
                    stall_windows=len(csv_rows),stall_success_episodes=len({int(r['episode']) for r in csv_rows}),
                    only_B=True,independent_validation=False))
                del frozen,trained,base_arrays  # release potentially large fitted banks
                base_arrays={k:np.load(source/v['file'],mmap_mode='r',allow_pickle=False) for k,v in parent['arrays'].items()
                             if k.startswith('key_') or k in ('state','normalized_actions')}
            print('audited',model,suite,'exact',item['exact_global'],'rows',len(rows),'flags',item['flags'],flush=True)
        for path in sorted((lane/'assets/cache_yamls').glob('*.yaml')):
            sha(path);config=yaml.safe_load(path.read_text());recipe=config['recipe']
            relative=Path(recipe['fitted_path']).relative_to(lane/'assets/cache_inputs')
            assert recipe['fitted_digest']==obj(inputs/relative/'READY.json')['digest']
            view_rel=Path(recipe['store_path']).relative_to(lane/'assets/cache_inputs')
            assert recipe['store_digest']==obj(inputs/view_rel/'VIEW.json')['content_digest']
            if recipe.get('stall_path'):
                rel=Path(recipe['stall_path']).relative_to(lane/'assets/cache_inputs')
                assert recipe['stall_digest']==obj(inputs/rel/'READY.json')['digest']
            assert recipe.get('periodic_gap') is None
            lane_info['yaml_count']+=1
        result['lanes'].append(lane_info)
    for path,pin in list(PINS.items()): assert sha(path)==pin
    result['checks']='PASS: raw state bytes, row identities, sealed B-only fit arrays/LOEO/folds/stall labels and templates, YAML bindings'
    result['finished_utc']=datetime.now(timezone.utc).isoformat()
    (HERE/'CONTAMINATION_CHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (HERE/'CONTAMINATION_DISTANCES.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(result['init_distances'][0]));writer.writeheader();writer.writerows(result['init_distances'])
    print(json.dumps(dict(libraries=result['libraries'],lanes=result['lanes'],fits=len(result['fits']),pins=len(PINS)),ensure_ascii=False))


if __name__ == '__main__':
    main()
