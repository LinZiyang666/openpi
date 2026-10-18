from types import SimpleNamespace as NS

import numpy as np

from .row_filter import r17_candidates, r10_candidates
from .analyze import score, means
from .prove_retrieval import compare


def table(rows):
    n=len(rows)
    return NS(rows=np.array(rows),Z=np.arange(n*3).reshape(n,3),z2=np.arange(n),
              HD=np.arange(n*2).reshape(n,2),h2=np.arange(n),Wf=np.eye(3),shift=np.ones(3),s_c=3.,
              early=NS(n20=np.arange(n),A0=np.eye(3),W0f=np.eye(3),c0=np.ones(3)))


def test_r17_selects_candidates_and_preserves_every_fitted_parameter():
    original=NS(episode=np.array([0,0,1,1,2,2]),tasks={0:table([0,1,2,3]),1:table([4,5])},
                pca={'v':np.eye(3)},sigma=np.array([2.]),heads={'head':np.ones(3)},action=np.ones((6,5,4)))
    all_rows=r17_candidates(original,[0,1,2])
    assert all(r['equal'] for r in compare(vars(original),vars(all_rows)))
    selected=r17_candidates(original,[0])
    assert set(selected.tasks)=={0}
    out=selected.tasks[0];before=original.tasks[0]
    assert out.rows.tolist()==[0,1]
    np.testing.assert_array_equal(out.Z,before.Z[:2])
    np.testing.assert_array_equal(out.early.n20,before.early.n20[:2])
    for key in ('Wf','shift','s_c'):assert getattr(out,key) is getattr(before,key)
    for key in ('A0','W0f','c0'):assert getattr(out.early,key) is getattr(before.early,key)
    for key in ('pca','sigma','heads','action'):assert getattr(original,key) is getattr(selected,key)
    assert before.rows.tolist()==[0,1,2,3]


def test_r10_preserves_parent_row_ids_and_blind_continuations():
    t=table([0,1,2,3]);t.early=None
    for key in ('V0','V1','RS'):setattr(t,key,np.arange(12).reshape(4,3))
    for key in ('rs2','n20'):setattr(t,key,np.arange(4))
    t.Z0=None;t.A0=np.eye(3)
    base=NS(lib_ep=np.array([10,10,11,11]),tasks={0:t},blind_next=np.array([1,-1,3,-1]),
            state_scale_by_task={0:np.array([.5])},n_cand={0:4})
    method=NS(variant='A',inner=base,row_subset=np.array([100,101,200,201]))
    original=dict(method=method,spec='original',kwargs={'size':50})
    selected=r10_candidates(original,[11])
    after=selected['method']
    assert after.row_subset is method.row_subset
    assert after.inner.blind_next is base.blind_next
    assert after.inner.state_scale_by_task is base.state_scale_by_task
    assert after.inner.tasks[0].A0 is t.A0
    assert after.row_subset[after.inner.tasks[0].rows].tolist()==[200,201]
    np.testing.assert_array_equal(after.inner.tasks[0].RS,t.RS[2:])
    assert t.rows.tolist()==[0,1,2,3]


def fixture():
    cell=dict(model='test',suite='libero_spatial',size=50,collected=50,kept=40,failed=10,empty_tasks=['0'])
    baseline=[dict(task=t,init=i,success=True,source='unfiltered') for t in ('0','1') for i in range(50)]
    observed={('1',i):True for i in range(50)}
    return cell,baseline,observed


def test_primary_scores_empty_tasks_and_secondary_pairs_the_same_tasks():
    row,pairs=score(*fixture())
    assert row['status']=='COMPLETE'
    assert row['primary']['filtered']['task_mean']==.5
    assert row['primary']['difference']['task_mean']==-.5
    assert row['primary']['losses']==50
    assert row['secondary']['filtered']['task_mean']==1
    assert row['secondary']['difference']['task_mean']==0
    assert sum(p['automatic_failure'] for p in pairs)==50


def test_missing_nonempty_outcomes_are_pending_not_failures():
    cell,baseline,observed=fixture();observed.pop(('1',3))
    row,pairs=score(cell,baseline,observed)
    assert row['status']=='PENDING' and row['primary']['filtered'] is None
    assert next(p for p in pairs if p['task']=='1' and p['init']==3)['filtered'] is None


def test_rerun_control_replaces_primary_and_retains_historical_baseline():
    cell,baseline,observed=fixture()
    required={(str(r['task']),r['init']) for r in baseline}
    rerun={k:False for k in required}
    row,pairs=score(cell,baseline,observed,rerun=rerun,rerun_required=required,
                    control_sources={'unfiltered':'new_control'})
    assert row['primary']['unfiltered']['task_mean']==0
    assert row['primary']['historical_unfiltered']['task_mean']==1
    assert row['primary']['difference']['task_mean']==.5
    assert row['secondary']['difference']['task_mean']==1
    assert all(p['unfiltered_source']=='new_control' and p['historical_unfiltered'] for p in pairs)
    assert sum(p['automatic_failure'] for p in pairs)==50


def test_missing_rerun_control_never_falls_back_to_old_outcomes():
    cell,baseline,observed=fixture()
    required={(str(r['task']),r['init']) for r in baseline}
    row,pairs=score(cell,baseline,observed,rerun_required=required)
    assert row['status']=='PENDING'
    assert row['primary']['unfiltered'] is None
    assert row['primary']['historical_unfiltered']['task_mean']==1
    assert row['primary']['filtered']['task_mean']==.5
    assert row['primary']['difference'] is None
    assert all(p['unfiltered'] is None and p['difference'] is None for p in pairs)


def test_mt50_equal_group_mean_differs_from_equal_task_mean():
    from exp.metaworld.tasks import DIFFICULTY
    first=next(iter(DIFFICULTY))
    same=next(t for t in DIFFICULTY if t!=first and DIFFICULTY[t]==DIFFICULTY[first])
    other=next(t for t in DIFFICULTY if DIFFICULTY[t]!=DIFFICULTY[first])
    result=means([(first,True),(same,True),(other,False)],True)
    assert result['task_mean']==2/3 and result['group_mean']==.5


def test_legacy_transfers_never_execute_tether_push_or_pull():
    from .legacy import rsync_transport
    calls=[];transfers=[]
    run=rsync_transport(lambda argv,**kw:calls.append(argv),transfers.append)
    run(['tether','push','--force','/local/file','h100:/tmp/file'])
    run(['tether','pull','--force','h100:/tmp/file','/local/file'])
    run(['tether','exec','h100','--','true'])
    assert transfers==[['rsync','push','/local/file','h100:/tmp/file'],['rsync','pull','h100:/tmp/file','/local/file']]
    assert calls==[['tether','exec','h100','--','true']]


def test_dp_capacity_bridge_only_changes_the_admission_count():
    from .server_cli import bridge
    seen=[]
    module=NS(cache_study_admission=lambda c,*args:seen.append((c,args)))
    original=dict(replicas=4,dp_batching={'max_batch':8},checkpoint='original')
    bridge(module);module.cache_study_admission(original,'cache','batch','provider')
    assert original['replicas']==4
    assert seen==[(dict(original,replicas=2),('cache','batch','provider'))]


def test_server_bridge_preserves_the_outer_mps_launcher():
    from .launch import server_command
    prefix=['python','-m','exp.r17_integration.models.fastwam.ops.mps','client','--ready','ready.json','--','python','-u']
    argv=[*prefix,'-m','exp.r17_integration.ops.formal_server','--campaign','campaign.json']
    transformed=server_command(argv,True)
    assert transformed[:len(prefix)]==prefix
    assert transformed[len(prefix)]=='-c'
    assert transformed[-2:]==['--campaign','campaign.json']
    assert server_command(argv,False) is argv


def test_frozen_server_cli_calls_original_serve_with_unchanged_arguments(tmp_path):
    import json
    from .server_cli import run
    campaign=dict(replicas=4,checkpoint='original',cache_study='study.json')
    path=tmp_path/'campaign.json';path.write_text(json.dumps(campaign))
    admissions=[];calls=[]
    module=NS(cache_study_admission=lambda c,*args:admissions.append(c))
    def serve(c,port,directory,graph):
        calls.append((c,port,directory,graph));module.cache_study_admission(c)
    run(['--campaign',str(path),'--port','23151','--directory','replica_0','--cuda-graph'],module,serve)
    assert calls==[(campaign,23151,'replica_0',True)]
    assert admissions==[dict(campaign,replicas=2)]
