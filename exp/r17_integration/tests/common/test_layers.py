from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from exp.r17_integration.common.contracts import ActionStateLayout, RequestContext
from exp.r17_integration.common.layers import LayoutRecipe, LibraryData
from exp.r17_integration.common.layers.descent import TaskDescent, passes_gate
from exp.r17_integration.common.layers.guard import NoProgressGuard
from exp.r17_integration.common.layers.math import _kernel_w, fit_head, fit_metric, pca_fit, predict
from exp.r17_integration.common.layers.periodic import periodic
from exp.r17_integration.common.query_view import QueryView


def library(h=16,d=4,views=1,state_width=None,width=None):
    rng=np.random.default_rng(99)
    n=24
    state_width=state_width or d
    width=width or d
    layout=ActionStateLayout(h,min(5,h),d,tuple(range(state_width)),tuple(range(d-1)),d-1,model_action_width=width)
    lib=LibraryData({f"vision_{i}":rng.normal(size=(n,4)).astype(np.float32) for i in range(views)},
        rng.normal(size=(n,state_width)).astype(np.float32),rng.normal(size=(n,h,width)).astype(np.float32),
        np.zeros(n,np.int64),np.repeat(np.arange(6),4),np.tile(np.arange(4),6))
    return layout,lib


def q_for(lib,step=1,idx=2,task_count=50):
    return QueryView(RequestContext("c","mt50.v1","reach",0,step,task_id=0),
                     {k:v[idx] for k,v in lib.features.items()},lib.state[idx],
                     [True]*step,[1]*step)


@pytest.mark.parametrize("h,d,v",[(5,4,1),(16,4,1),(32,7,2)])
def test_dynamic_library_loeo_corrector_no_gripper_change(h,d,v):
    l,lib=library(h,d,v)
    raw=LayoutRecipe(l,50,corrector=False).fit(lib)
    corrected=LayoutRecipe(l,50).fit(lib)
    q=q_for(lib)
    result=corrected.query(q)
    plain=raw.query(q)
    assert result.action.shape==(h,d) and np.isfinite(result.action).all()
    np.testing.assert_array_equal(result.action[:,l.gripper_channel],plain.action[:,l.gripper_channel])
    for item in corrected.fit_evidence["loeo"]:
        assert item["query_episode"] not in item["donor_episodes"]
    for fold in corrected.fit_evidence["distance_folds"]:
        assert not set(fold["train_episodes"]) & set(fold["heldout_episodes"])
    assert corrected.heads[0]["w"].shape[1]==384
    assert corrected._features(q,*corrected._project(q),plain.action).shape[1]==(4*v+d+min(10,h)*d+1+50)
    fork=corrected.fork()
    assert fork.action is corrected.action and fork.guard is not corrected.guard


def test_padding_does_not_change_metric_or_active_output():
    l,lib=library(width=32)
    changed=replace(lib,action=lib.action.copy())
    changed.action[:,:,4:]+=1e5
    a=LayoutRecipe(l,50,corrector=False).fit(lib)
    b=LayoutRecipe(l,50,corrector=False).fit(changed)
    r1,r2=a.query(q_for(lib)),b.query(q_for(lib))
    np.testing.assert_array_equal(r1.rows,r2.rows)
    np.testing.assert_array_equal(r1.weights,r2.weights)
    np.testing.assert_array_equal(r1.action[:,:4],r2.action[:,:4])


def test_fit_rejects_query_and_incomplete_and_small_corrector_library():
    l,lib=library()
    with pytest.raises(ValueError,match="commit"):
        LayoutRecipe(l,50,commit_horizon=0)
    with pytest.raises(ValueError,match="partitions"):
        LayoutRecipe(l,50).fit(replace(lib,partition="test"))
    with pytest.raises(ValueError,match="complete"):
        LayoutRecipe(l,50).fit(replace(lib,step=lib.step+1))
    small=LibraryData({k:v[:8] for k,v in lib.features.items()},lib.state[:8],lib.action[:8],lib.task_id[:8],lib.episode[:8],lib.step[:8])
    with pytest.raises(ValueError,match=">=3"):
        LayoutRecipe(l,50).fit(small)


def test_numerical_helpers_match_frozen_sources_bitwise():
    from exp.offline_search.rounds.r02.g1_awm import awm
    from exp.offline_search.rounds.r10.train import fit_head as old_fit_head
    from exp.offline_search.rounds.r10.recipe.recipe import predict as old_predict
    rng=np.random.default_rng(42)
    X=rng.normal(size=(20,8)).astype(np.float32);Y=rng.normal(size=(20,5)).astype(np.float32)
    for a,b in zip(pca_fit(X,keep=4),awm.pca_fit(X,keep=4)):
        np.testing.assert_array_equal(a,b)
    for a,b in zip(fit_metric(X.astype(float),Y.astype(float),np.repeat(np.arange(5),4)),
                    awm.fit_metric(X.astype(float),Y.astype(float),np.repeat(np.arange(5),4))):
        np.testing.assert_array_equal(a,b)
    h=fit_head(X,Y,np.ones(20));old=old_fit_head(X,Y,np.ones(20))
    for name in h:np.testing.assert_array_equal(h[name],old[name])
    np.testing.assert_array_equal(predict(h,X),old_predict(old,X))
    np.testing.assert_array_equal(_kernel_w(np.arange(8.),8),awm._kernel_w(np.arange(8.),8))


def test_legacy_corrector_features_apply_and_five_fold_scale_match():
    from exp.offline_search.rounds.r10.recipe.fitting import distance_scale
    from exp.offline_search.rounds.r10.recipe.recipe import _CorrectedCache
    l,lib=library(d=7,views=2,state_width=8,width=32)
    recipe=LayoutRecipe(l,10,corrector=True).fit(lib)
    q=q_for(lib)
    plain,xv,rs=recipe._retrieve(q)
    old=_CorrectedCache.__new__(_CorrectedCache)
    old.sig_head=recipe.sigma;old.chans=6;old.blend=.5
    np.testing.assert_array_equal(recipe._features(q,xv,rs,plain.action),old._features(q,xv,rs,plain.action))
    ratio=plain.extras["raw_anchor_distance"]/recipe.distance_scale
    factor=float(np.clip((2.-ratio)/1.25,0.,1.))
    corr=predict(recipe.heads[0],old._features(q,xv,rs,plain.action))[0].reshape(10,6)*recipe.sigma[:6]
    corr=corr*np.float32(factor)
    np.testing.assert_array_equal(recipe.query(q).action,old._apply(plain.action,corr))
    old_lib=SimpleNamespace(chosen={"0":list(range(6))},rs=lib.state,action=lib.action,
                            episode=lib.episode,step=lib.step,task_id=lib.task_id)
    old_scale,_=distance_scale(old_lib,{"v0":recipe.projected[:,:4],"v1":recipe.projected[:,4:]},recipe.kref)
    assert recipe.distance_scale==old_scale


@pytest.mark.parametrize("step,prev_hit",[(0,None),(1,True),(1,False)])
def test_legacy_double_view_distances_topk_kernel_chunk_match(step,prev_hit):
    from exp.offline_search.rounds.r02.g1_awm.awm import AWM
    l,lib=library(d=7,views=2,state_width=8,width=32)
    new=LayoutRecipe(l,10,corrector=False).fit(lib)
    old=AWM(early=True,lib="current")
    old.model="pi05";old.sig=new.sigma;old.act=new.action
    old.B0T,old.muB0=new.pca["vision_0"];old.B1T,old.muB1=new.pca["vision_1"]
    old.tasks={}
    for task,T in new.tasks.items():
        ot=SimpleNamespace(**vars(T));ot.Z0=None;ot.As0=None
        for name in ("A0","W0f","c0","n20"):setattr(ot,name,getattr(T.early,name))
        old.tasks[task]=ot
    q=q_for(lib,step=step);q.prev_hit=prev_hit;q.previous_action=lib.action[0]
    oq=SimpleNamespace(step=step,task_id=0,key_v0=q.features["vision_0"],key_v1=q.features["vision_1"],
                       rs=q.state,prev_hit=prev_hit,prev_a_exec=q.previous_action)
    dt=old._dist(oq)[-1]
    pos=np.lexsort((np.arange(len(dt)),dt))[:old.k]
    dk=dt[pos].astype(np.float64)
    wn,_,chunk=old._mix(pos,_kernel_w(dk-dk[0],old.kref))
    result=new.query(q)
    np.testing.assert_array_equal(result.rows,pos)
    np.testing.assert_array_equal(result.distances,dk)
    np.testing.assert_array_equal(result.weights,wn)
    np.testing.assert_array_equal(result.action,chunk)


def test_only_no_progress_matches_legacy_span_with_blind_gaps_and_miss_reset():
    from exp.offline_search.rounds.r04.k1_blind.judge import BlindMixedJudge
    old=SimpleNamespace(_vision_progress=[],memo_reset_after_miss=True,prog_eps=.5,
                         C=SimpleNamespace(prog=np.array([.1,.11,.4]),ep_len=np.array([11,11,11])))
    guard=NoProgressGuard()
    for step,row,ph in [(0,0,None),(2,1,True),(3,1,True),(4,2,True),(5,2,False)]:
        span=BlindMixedJudge._progress(old,SimpleNamespace(step=step,prev_hit=ph),row)
        force=guard.observe(step,old.C.prog[row],old.C.ep_len[row],prev_hit=ph)
        assert guard.span==span and force==(span>=2)


def test_periodic_force_and_guard_priority_match_frozen_r11():
    from exp.offline_search.rounds.r11.knob.recipe import uniform as old_uniform
    for step in range(5):
        r=RequestContext("c","b","task",4,step,task_id=7)
        histories=[True]*step;hits=[1]*step
        force,ex=periodic(r,histories,hits,1.7)
        u=old_uniform("R11-gap-v1",0,7,4,-1,"cap")
        assert ex["coin"]==u
        assert force==(step>=1+int(u<.7))
        assert periodic(r,histories,hits,1.7,guard=True)[0] is False


def test_descent_stale_error_and_first_failure_barrier():
    d=TaskDescent(("a","b","c"),frozenset(map(str,range(20))),20)
    assert not d.accept("0",False,accepted=False)
    assert not d.accept("0",False,error=True)
    for i in range(20):d.accept(str(i),i!=0)
    assert d.selected=="a" and not d.stopped
    for i in range(20):d.accept(str(i),i>1)
    assert d.stopped and d.selected=="a"
    assert passes_gate(19,20,20) and not passes_gate(18,20,20)
