"""Selection, leakage, worker source, resume and scoring boundary tests."""
from contextlib import nullcontext
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import pytest
import yaml

from exp.offline_search.closed_loop import devset, worker_pool
from exp.offline_search.closed_loop.ops.h100 import assets, control
from exp.offline_search.closed_loop.ops.remote import count
from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import load_manifest


@pytest.fixture
def fixture(tmp_path):
    store = tmp_path / 'store'
    d = store / 'library/pi05_l10/small'
    d.mkdir(parents=True)
    eps = [dict(task_id=t, orig_init_state_idx=i, success=(i % 2 == 0), stem=f'{t}/{i}') for t in range(10) for i in range(3)]
    (d / 'episodes.json').write_text(json.dumps(eps))
    parent = d.with_name('bpool_cs');parent.mkdir()
    (parent / 'episodes.json').write_text(json.dumps([dict(task_id=t, orig_init_state_idx=i, success=(i % 2 == 0), stem=f'{t}/{i}') for t in range(10) for i in range(50)]))
    root = tmp_path / 'run';root.mkdir()
    row = dict(arm='sample', dev=True, init_pool='B', model='pi05', suite='libero_10',
               mode='plugin', cell='pi05_l10_cache', method='fixture:Recipe', kwargs=dict(library='small'),
               plugin_args=['--os-fit-artifact',str(root / 'fit.pkl')])
    (root / 'arms.json').write_text(json.dumps([row]))
    subset = devset.subset_record('pi05','l10','small',store=store)
    method=SimpleNamespace(library='small',fit_info=dict(library_only=True,episode_ids_by_task=subset['episode_ids_by_task']))
    blob=dict(spec=row['method'],kwargs=row['kwargs'],cell=row['cell'],method=method)
    (root / 'fit.pkl').write_bytes(pickle.dumps(blob))
    manifest=devset.build_manifest('pi05','l10',library='small',per_task=5,seed=9,store=store)
    path=root/'manifest.json';path.write_text(json.dumps(manifest));row['manifest']=str(path)
    (root/'arms.json').write_text(json.dumps([row]))
    return root,store,row,load_manifest(path),blob


def test_builder_determinism_and_exact_complement(fixture):
    root,store,row,m,_=fixture
    again=devset.build_manifest('pi05','libero_10',library='small',per_task=5,seed=9,store=store)
    assert again==m['data']
    assert len(m['selected'])==50
    assert all(i>=3 for t,i in m['selected'])
    assert all(sum(t==task for t,i in m['selected'])==5 for task in range(10))
    assert devset.build_manifest('pi05','l10',library='small',per_task=5,seed=10,store=store)!=again


@pytest.mark.parametrize('k,seed',[(0,0),(48,0),(1,-1)])
def test_builder_refuses_empty_or_insufficient_complement(fixture,k,seed):
    _,store,_,_,_=fixture
    with pytest.raises(ValueError):
        devset.build_manifest('pi05','l10',library='small',per_task=k,seed=seed,store=store)


def test_unknown_init_never_guessed(fixture):
    _,store,_,_,_=fixture
    p=store/'library/pi05_l10/small/episodes.json';eps=json.loads(p.read_text());eps[0]['orig_init_state_idx']=None;p.write_text(json.dumps(eps))
    with pytest.raises(ValueError,match='UNKNOWN_LIBRARY_INIT'):
        devset.subset_record('pi05','l10','small',store=store)


def test_subset_ids_are_metadata_positions_not_episode_attributes(fixture):
    _,store,_,_,_=fixture
    ids={str(t):[t*3+1] for t in range(10)}
    out=devset.subset_record('pi05','l10','small',ids,store)
    assert out['excluded_inits_by_task']=={str(t):[1] for t in range(10)}
    ids['0']=[4]
    with pytest.raises(ValueError,match='invalid parent episode'):
        devset.subset_record('pi05','l10','small',ids,store)


@pytest.mark.parametrize('dev,pool',[(True,'A'),(False,'B'),('true','B')])
def test_root_pool_never_crosses(dev,pool):
    with pytest.raises(ValueError):devset.pool_for(dict(dev=dev,init_pool=pool))


def test_environment_cannot_override_root(monkeypatch):
    monkeypatch.setenv('OSCL_INIT_POOL','B')
    with pytest.raises(ValueError,match='POOL_MISMATCH'):devset.pool_for({})


def test_mixed_root_refused(fixture):
    root,_,row,m,_=fixture
    (root/'arms.json').write_text(json.dumps([row,dict(arm='a')]))
    with pytest.raises(ValueError,match='POOL_MISMATCH'):devset.root_pool(root)


def test_actual_prefit_selection_guard(fixture):
    root,store,row,m,blob=fixture
    contract=devset.arm_contract(root,row,m,store)
    assert contract['init_pool']=='B' and contract['subset']['excluded_inits_by_task']['0']==[0,1,2]
    m['selected'][0,0]={}
    with pytest.raises(ValueError,match='DEV_LIBRARY_OVERLAP'):devset.arm_contract(root,row,m,store)


def test_plan_refuses_overlap_before_any_transfer(fixture,tmp_path):
    root,store,row,m,_=fixture
    data=m['data'];data['pairs'].append([0,0]);Path(row['manifest']).write_text(json.dumps(data))
    with pytest.raises(ValueError,match='DEV_LIBRARY_OVERLAP'):
        assets.build_plan(root,['sample'],tmp_path/'work',store=store)


def test_chain_refuses_pool_mismatch_before_any_remote_call(fixture,monkeypatch):
    root,store,row,m,_=fixture
    contract=devset.arm_contract(root,row,m,store)
    (root/'h100_sync').mkdir()
    (root/'h100_sync/synced.json').write_text(json.dumps(dict(worker_host='timan108',arms_sha256=devset.sha(root/'arms.json'),arms=[row],dev_contracts={'sample':contract})))
    data=m['data'];data['init_pool']='A';Path(row['manifest']).write_text(json.dumps(data))
    monkeypatch.setenv('PORTS','23220');monkeypatch.setenv('WPS','1');monkeypatch.setenv('WORKER_HOST','timan108')
    monkeypatch.setattr(control,'fleet_lock',lambda *a:nullcontext())
    monkeypatch.setattr(control.signal,'signal',lambda *a:None)
    monkeypatch.setattr(control,'rpc',lambda *a,**k:pytest.fail('remote action before pool guard'))
    with pytest.raises(ValueError,match='POOL_MISMATCH'):control.chain(root,['sample'])


@pytest.mark.parametrize('change',['ids','library','metadata','library_only'])
def test_prefit_lies_refused(fixture,change):
    root,store,row,m,blob=fixture
    if change=='ids':blob['method'].fit_info['episode_ids_by_task']['0']=[0]
    if change=='library':blob['method'].library='other'
    if change=='metadata':blob['kwargs']={}
    if change=='library_only':blob['method'].fit_info['library_only']=False
    (root/'fit.pkl').write_bytes(pickle.dumps(blob))
    with pytest.raises(ValueError):devset.arm_contract(root,row,m,store)


@pytest.mark.parametrize('pool',['A','B'])
def test_manifest_pool_tags_are_required(fixture,pool):
    _,_,_,m,_=fixture
    if pool=='A':
        with pytest.raises(ValueError,match='POOL_MISMATCH'):devset.check_manifest_pool(m,'A')
    else:
        m['data']={}
        with pytest.raises(ValueError,match='POOL_MISMATCH'):devset.check_manifest_pool(m,'B')


def test_reference_includes_failed_episodes_and_exact_pairs(fixture):
    _,store,_,m,_=fixture
    result=devset.pure_reference(m,store)
    assert result['complete']==50
    assert result['success']==sum(i%2==0 for t,i in m['selected'])
    assert {(r['task_id'],r['orig_init_state_idx']) for r in result['reference']}==set(m['selected'])


def test_reference_detects_changed_library(fixture):
    _,store,_,m,_=fixture
    p=store/'library/pi05_l10/small/episodes.json';p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError,match='provenance changed'):devset.pure_reference(m,store)


def test_journal_never_mixes_and_legacy_only_A():
    devset.validate_journal_pool([{}],'A')
    for pool,rows in [('B',[{}]),('B',[{'init_pool':'A'}]),('A',[{'init_pool':'B'}])]:
        with pytest.raises(ValueError,match='POOL_MISMATCH'):devset.validate_journal_pool(rows,pool)


def test_journal_stamps_actual_pool(tmp_path,monkeypatch):
    from openpi.conductor import driver
    old=driver.Journal
    monkeypatch.setattr(driver,'Journal',old)
    cls=worker_pool.install_journal('B');path=tmp_path/'journal.jsonl';j=cls(path)
    j.record(task_uid='a:eval:0:7',yaml_id='a',phase='eval',status='failed',success=False,accepted=True,attempt=1,duration_s=1.23456)
    value=json.loads(path.read_text());assert value['init_pool']=='B' and value['duration_s']==1.235
    assert j.replay_done_uids()=={'a:eval:0:7'}


def test_count_refuses_wrong_pool(fixture,tmp_path):
    root,_,_,m,_=fixture
    journal=tmp_path/'journal.jsonl';journal.write_text(json.dumps(dict(accepted=True,status='done',task_uid='sample:eval:0:7',success=True))+'\n')
    with pytest.raises(ValueError,match='POOL_MISMATCH'):count.main([str(journal),'--manifest',str(root/'manifest.json')])


def worker_args(tmp_path,pool):
    repo=tmp_path/'repo';suite='libero_10';directory=repo/'exp/common/data/db_init/libero'/(suite if pool=='B' else suite+'_apool');directory.mkdir(parents=True)
    config=repo/('exp/offline_search/closed_loop/ops/h100' if pool=='B' else 'exp/ablation_study/cache_size/config')/(('bpool_' if pool=='B' else 'apool_')+suite+'.yaml')
    config.parent.mkdir(parents=True)
    record=dict(suite=suite,apool_dir=str(directory),per_task_digests={'task':'sha'},rollup_sha256='rollup',total_inits=500)
    if pool=='B':record.update(dev=True,init_pool='B')
    config.write_text(yaml.safe_dump(record));matrix=repo/'matrix.json';matrix.write_text(json.dumps(dict(arms=[dict(arm='sample')])))
    return repo,directory,['--task-suite',suite,'--apool-record',str(config),'--apool-dir',str(directory),'--journal',str(repo/'journal.jsonl'),'--arm-matrix',str(matrix)]


def test_worker_B_contract_and_A_path_refusal(fixture,tmp_path,monkeypatch):
    root,store,row,m,_=fixture
    repo,directory,args=worker_args(tmp_path,'B')
    contract=devset.arm_contract(root,row,m,store);p=repo/'contract.json';p.write_text(json.dumps(contract))
    monkeypatch.setenv('OSCL_INIT_POOL','B');monkeypatch.setenv('OSCL_POOL_CONTRACT',str(p));monkeypatch.setenv('OSCL_POOL_CONTRACT_SHA256',devset.sha(p))
    assert worker_pool.validate(args,m,repo)[0]=='B'
    args[args.index('--apool-dir')+1]=str(directory.with_name('libero_10_apool'))
    with pytest.raises(ValueError,match='POOL_MISMATCH'):worker_pool.validate(args,m,repo)


@pytest.mark.parametrize('bad',['path','shadow','duplicate','init-map','manifest','journal'])
def test_worker_A_guards(tmp_path,monkeypatch,bad):
    monkeypatch.delenv('OSCL_INIT_POOL',raising=False)
    repo,directory,args=worker_args(tmp_path,'A');manifest=None
    if bad=='path':args[args.index('--apool-dir')+1]=str(directory.with_name('libero_10'))
    if bad=='shadow':(directory/'task.pruned_init').write_bytes(b'not hashed')
    if bad=='duplicate':args+=['--apool-dir',str(directory)]
    if bad=='init-map':args+=['--init-map','somewhere']
    if bad=='manifest':manifest=dict(data=dict(dev=True,init_pool='B'))
    if bad=='journal':(repo/'journal.jsonl').write_text('{"init_pool":"B"}\n')
    with pytest.raises(ValueError):worker_pool.validate(args,manifest,repo)


def test_paired_scoring_cross_pool_refused(fixture,tmp_path):
    from exp.offline_search.closed_loop.ops.kpi import paired
    root,_,_,_,_=fixture
    a=tmp_path/'A';a.mkdir();(a/'arms.json').write_text('[{"arm":"a"}]')
    with pytest.raises(ValueError,match='POOL_MISMATCH'):
        paired({'info':{'run_root':str(root)}},{'info':{'run_root':str(a)}},100,0)


def test_summary_and_kpi_reject_foreign_journals(fixture):
    from exp.offline_search.closed_loop.ops.collect import summarize
    from exp.offline_search.closed_loop.ops.kpi import load_journal
    root,_,row,m,_=fixture
    client=root/'runs/sample/client';client.mkdir(parents=True)
    journal=client/'journal.jsonl'
    task,init=min(m['selected'])
    journal.write_text(json.dumps(dict(task_uid=f'sample:eval:{task}:{init}',accepted=True,status='done',success=True,init_pool='B'))+'\n')
    result=summarize(root,'sample',write=False)
    assert result['init_pool']=='B' and result['complete']==1
    journal.write_text(journal.read_text().replace('"B"','"A"'))
    with pytest.raises(ValueError,match='POOL_MISMATCH'):summarize(root,'sample',write=False)
    with pytest.raises(ValueError,match='POOL_MISMATCH'):load_journal(root,'sample')


@pytest.mark.parametrize('model',['pi05','groot'])
@pytest.mark.parametrize('suite',['l10','spatial'])
def test_real_r10_and_current_complements(model,suite):
    for n in (50,100,200,300,400):
        data=devset.build_manifest(model,suite,r10_size=n,per_task=5,seed=20261003)
        m=dict(data=data,selected={tuple(p):{} for p in data['pairs']})
        devset.require_disjoint(m,data['subset'])
    with pytest.raises(ValueError,match='only 0 held-out'):
        devset.build_manifest(model,suite,r10_size=500,per_task=1,seed=0)
    data=devset.build_manifest(model,suite,library='current',per_task=5,seed=20261003)
    assert len(data['pairs'])==50
