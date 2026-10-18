import copy
import json
from types import SimpleNamespace

import pytest

from exp.offline_search.rounds.r17.autoscale_1017.fairness import allocate, pending_gpu
from exp.offline_search.rounds.r17.autoscale_1017.node import LocalGovernor, atomic
from exp.offline_search.rounds.r17.autoscale_1017.policy import GIB


class World:
    def __init__(self,tmp_path,monkeypatch,*,gpu=False):
        self.root=tmp_path;self.time=0.;self.sizes={};self.gpu_sizes={};self.next=900000;self.lanes=[];self.gpu=gpu
        self.total=(220 if gpu else 120)*GIB
        monkeypatch.setattr('exp.offline_search.rounds.r17.autoscale_1017.node.time.time',self.wall)
    def wall(self):return 1700000000+self.time
    def memory(self):return self.total-sum(v[0] for v in self.sizes.values()),{'full':{'avg10':0}}
    def lane(self,name,*,calibrated=True,peak=2*GIB,vram=200):
        g=LocalGovernor(dict(run_name=name,worker_node='timan107',node_workers=100,worker_vram_mib=vram,
            autoscale=dict(peak_margin=1,hold_s=2)),self.root/name,ledger=self.root/'ledger',
            clock=lambda:self.time,reader=self.memory,process_reader=lambda p,t:self.sizes.get(p))
        g.simulated_peak=peak
        if calibrated:atomic(g.history_path,[dict(peak=peak,complete=True)])
        self.lanes.append(g);self.target(g,100)
        if self.gpu:g.slots=SimpleNamespace(view=lambda:dict(time=self.time,cards=[dict(device=0,uuid='GPU',
            total_mib=12000,used_mib=sum(self.gpu_sizes.values()),usable=True,
            processes=[dict(pid=p,used_mib=self.gpu_sizes[p]) for p in self.sizes])]))
        g.tick();return g
    def target(self,g,n,done=False):
        atomic(g.control_path,dict(campaign=g.campaign['run_name'],time=self.wall(),targets=dict(timan107=n),done=done))
    def sample(self):
        self.time+=2
        for g in self.lanes:
            self.target(g,g.target)
            g.tick()
    def spawn(self,g):
        with g.permit() as ok:
            if not ok:return False
            p=self.next;self.next+=1
            self.sizes[p]=(g.simulated_peak,g.simulated_peak)
            self.gpu_sizes[p]=g.campaign['worker_vram_mib']
            g.register(dict(pid=p,start_ticks='1',worker_id=f'{g.run.name}-{p}',born=self.time,
                            egl_device=0,gpu_uuid='GPU'))
            return True
    def boundary(self,g,workers=None):
        for w in list(g.retired if workers is None else workers):
            p=g.entries[w]['pid']
            with g.lock():g.finished(w,normal=True)
            self.sizes.pop(p);self.gpu_sizes.pop(p)
    def ramp(self,iterations=120):
        for _ in range(iterations):
            self.sample()
            for g in self.lanes:self.boundary(g);self.spawn(g)
        self.sample()


@pytest.mark.parametrize('gpu,expected',[(False,25),(True,27)])
def test_two_campaigns_ramp_share_memory_and_gpu_then_return_borrowing(tmp_path,monkeypatch,gpu,expected):
    w=World(tmp_path,monkeypatch,gpu=gpu);a=w.lane('H100');b=w.lane('4090')
    w.ramp()
    assert [len(a.entries),len(b.entries)]==[expected,expected]
    w.target(a,8);a.tick();w.ramp()
    assert len(a.entries)==8 and len(b.entries)==2*expected-8
    # A later upward probe reclaims its share, without changing B's own target.
    w.target(a,100);a.tick();b.tick()
    assert b.target==100 and len(b.retired)==expected-8
    assert b.snapshot['fairness']['returning']==len(b.retired)
    assert {r['worker_id'] for r in b.snapshot['workers'] if r['draining']}==b.retired
    with a.permit() as allowed:assert not allowed  # grants aren't free RSS yet
    w.ramp()
    assert [len(a.entries),len(b.entries)]==[expected,expected]
    # A finishing lane releases its entitlement; B borrows the entire pool.
    w.target(a,0,done=True);a.tick();w.ramp()
    assert not a.entries and len(b.entries)==2*expected
    events=[json.loads(s) for s in b.events.read_text().splitlines()]
    assert any(r['reason']=='retire_requested_fair_share' for r in events)
    assert w.memory()[0]>=20*GIB


def test_late_unknown_campaign_reclaims_canary_from_full_borrower(tmp_path,monkeypatch):
    w=World(tmp_path,monkeypatch);a=w.lane('old');w.ramp(60)
    assert len(a.entries)==50
    b=w.lane('new',calibrated=False);a.tick()
    assert a.retired and not b.entries
    assert not w.spawn(b)
    w.boundary(a);w.sample()
    assert w.spawn(b)
    assert not w.spawn(b)  # unknown peak: one real canary generation
    w.boundary(b,list(b.entries));w.ramp()
    assert len(a.entries)==len(b.entries)==25


def test_different_profiles_and_low_target_borrow_unused_entitlement():
    cap=dict(ram=200,vram=10800)
    costs={'a':dict(ram=2,vram=400),'b':dict(ram=2,vram=100)}
    q=allocate(cap,dict(a=100,b=100),costs)
    assert q['a']>=13 and q['b']>=50
    assert all(sum(q[n]*costs[n][r] for n in q)<=cap[r] for r in cap)
    q=allocate(cap,dict(a=2,b=100),costs)
    assert q==dict(a=2,b=98)
    # The indivisible last worker is rotated, rather than permanently won by A.
    assert allocate(dict(ram=1),dict(a=1,b=1),{'a':dict(ram=1),'b':dict(ram=1)},epoch=0)==dict(a=1,b=0)
    assert allocate(dict(ram=1),dict(a=1,b=1),{'a':dict(ram=1),'b':dict(ram=1)},epoch=1)==dict(a=0,b=1)


def test_global_gpu_pending_counts_different_fleet_roots_and_observed_children():
    states=[dict(worker_vram_mib=530,workers=[dict(pid=1,egl_device=3,gpu_pids=[1,2])]),
            dict(worker_vram_mib=133,workers=[dict(pid=4,egl_device=3)])]
    cards={3:dict(processes=[dict(pid=2,used_mib=400)])}
    assert pending_gpu(states,cards)=={3:263}


def test_finished_empty_supervisor_cannot_block_a_peer_after_heartbeat_stops(tmp_path,monkeypatch):
    w=World(tmp_path,monkeypatch);a=w.lane('finished');b=w.lane('working')
    w.target(a,0,done=True);a.tick()
    w.time+=20;w.target(b,100)
    assert b.tick()['reason']!='memory_unknown'
    assert w.spawn(b)


def test_prepared_frozen_count_and_attested_hard_limit(tmp_path):
    from exp.offline_search.rounds.r17.autoscale_1017.defaults import apply
    original=dict(domain='metaworld',worker_nodes=dict(timan1=20,timan107=20),worker_maxima_explicit=True)
    c=apply(copy.deepcopy(original))
    assert c['autoscale_node_maxima']==dict(timan1=332,timan107=440)
    assert c['worker_nodes']==original['worker_nodes']
    capped=dict(original,worker_hard_limits={'timan1':dict(maximum=16,reason='fixture: only 16 EGL contexts supported')})
    assert apply(capped)['autoscale_node_maxima']==dict(timan1=16,timan107=440)
    with pytest.raises(ValueError,match='technical reason'):
        apply(dict(original,worker_hard_limits={'timan1':dict(maximum=16)}))
    atomic(tmp_path/'PREPARED.json',{})
    assert apply(copy.deepcopy(original),tmp_path)==original


def test_driver_intent_reclaims_borrowed_share_and_recovers_after_peer_finishes(tmp_path,monkeypatch):
    from exp.offline_search.rounds.r17.autoscale_1017.driver import DriverController
    w=World(tmp_path,monkeypatch);a=w.lane('H100');b=w.lane('4090')
    controllers=[]
    for g in (a,b):
        c=dict(g.campaign,worker_nodes={'timan107':100},workers=100)
        d=DriverController(SimpleNamespace(scheduler=SimpleNamespace(all_done=lambda:False)),c,g.run,
                           clock=lambda:w.time,wall=w.wall)
        controllers.append(d)
    def step(count):
        for _ in range(count):
            w.time+=2
            for g,d in zip((a,b),controllers):
                d.ingest({'node':g.tick()});d.tick()
            for g in (a,b):w.boundary(g);w.spawn(g)
    step(90)
    assert len(a.entries)==len(b.entries)==25
    controllers[0].tuner.target=8
    step(30)
    assert len(a.entries)==8 and len(b.entries)==42
    controllers[0].tuner.target=100
    step(30)
    assert len(a.entries)==len(b.entries)==25
    assert controllers[0].control['demand_targets']=={'timan107':100}
    controllers[0].driver.scheduler.all_done=lambda:True
    step(40)
    assert len(a.entries)==0 and len(b.entries)==50
    assert controllers[0].control['demand_targets']=={'timan107':0}


@pytest.mark.parametrize('pipeline',['R15','R16'])
def test_native_legacy_preparation_promotes_historical_counts(tmp_path,monkeypatch,pipeline):
    from exp.offline_search.rounds.r17.autoscale_1017 import legacy_release
    from exp.r17_integration.ops import formal_eval
    from exp.offline_search.closed_loop.ops.h100 import control
    monkeypatch.setattr(legacy_release,'build',lambda root:dict(digest='fixture',archive='fixture.tar',archive_sha256='fixture',files={}))
    monkeypatch.setattr(formal_eval,'transfer',lambda *a:None)
    monkeypatch.setattr(control,'remote',lambda *a:None)
    result=legacy_release.stage(tmp_path,'timan107',pipeline=pipeline,maximum=4,base='/frozen/fixture')
    assert result['maximum']==104
    result=legacy_release.stage(tmp_path,'timan107',pipeline=pipeline,maximum=4,base='/frozen/fixture',
        hard_limits={'timan107':dict(maximum=4,reason='fixture: fixed licensed EGL context limit')})
    assert result['maximum']==4


def test_splitting_a_campaign_into_fleets_does_not_multiply_its_share(tmp_path,monkeypatch):
    w=World(tmp_path,monkeypatch)
    a=w.lane('a1');a.fairness_campaign='A';a.tick()
    a2=w.lane('a2');a2.fairness_campaign='A';a2.tick()
    b=w.lane('B');w.ramp(70)
    assert len(a.entries)+len(a2.entries)==len(b.entries)==25
    assert b.snapshot['fairness']['campaigns']==2


def test_gpu_admission_charges_pending_workers_from_another_run_root(tmp_path,monkeypatch):
    import threading,time
    from exp.offline_search.rounds.r17.autoscale_1017.fleet import Slots
    from exp.r17_integration.ops import worker_gpu
    slot=object.__new__(Slots)
    slot.c={'worker_node':'timan107'};slot.run=tmp_path/'second-root'/'B';slot.run.mkdir(parents=True)
    slot.record={'worker_vram_mib':1000};slot.assignments={}
    slot.observation_lock=threading.Lock();slot.observed_at=time.monotonic()
    slot.observed=[dict(device=0,uuid='GPU',usable=True,total_mib=8000,used_mib=1000,
                        processes=[dict(pid=1,used_mib=1000)])]
    other=dict(schema=2,run=str(tmp_path/'first-root'/'A'),worker_vram_mib=1000,
               workers=[dict(pid=p,egl_device=0) for p in range(1,8)])
    slot.governor=SimpleNamespace(live_states=[other])
    monkeypatch.setattr(worker_gpu,'reservations',lambda *a:({0:0},[]))
    assert not slot.acquire(0)  # six promised contexts consume the free VRAM
    other['workers'].pop()    # only a completed boundary frees the promise
    assert slot.acquire(0)
    assert slot.assignments[0]['uuid']=='GPU'


def test_libero_and_metaworld_profiles_share_then_borrow_and_finish(tmp_path,monkeypatch):
    w=World(tmp_path,monkeypatch,gpu=True)
    libero=w.lane('LIBERO',peak=3*GIB,vram=530)
    mw=w.lane('MetaWorld',peak=GIB,vram=133)
    w.ramp(100)
    assert (len(libero.entries),len(mw.entries))==(10,41)
    assert sum(w.gpu_sizes.values())<=10800
    w.target(libero,4);libero.tick();w.ramp(35)
    assert (len(libero.entries),len(mw.entries))==(4,65)
    w.target(libero,0,done=True);libero.tick();w.ramp(30)
    assert not libero.entries and len(mw.entries)==81
    assert w.memory()[0]>=20*GIB


def test_one_live_peak_outlier_is_not_reserved_for_every_worker(tmp_path,monkeypatch):
    w=World(tmp_path,monkeypatch);a=w.lane('A');b=w.lane('B');w.ramp()
    # A single old process grows by 2 GiB. Return enough real RAM at boundaries,
    # without converting every 2-GiB worker into a 4-GiB reservation.
    p=next(iter(a.entries.values()))['pid'];w.sizes[p]=(4*GIB,4*GIB)
    w.ramp(40)
    assert len(a.entries)>=23 and len(b.entries)>=24
    assert len(a.entries)+len(b.entries)>=48
    assert w.memory()[0]>=20*GIB


def test_actual_gpu_growth_is_charged_to_its_lane_before_borrowing(tmp_path,monkeypatch):
    w=World(tmp_path,monkeypatch,gpu=True);a=w.lane('A');b=w.lane('B');w.ramp()
    p=next(iter(a.entries.values()))['pid'];w.gpu_sizes[p]=400
    w.ramp(30)
    assert (len(a.entries),len(b.entries))==(26,27)
    assert sum(w.gpu_sizes.values())<=10800


def test_gpu_fragments_cannot_be_combined_into_extra_workers():
    from exp.offline_search.rounds.r17.autoscale_1017.fairness import plan
    cards=[dict(device=d,total_mib=4444,used_mib=0,usable=True,processes=[]) for d in range(8)]
    states=[dict(schema=2,fairness_campaign=n,target=100,projected_peak=GIB,worker_vram_mib=3000,
                 workers=[],fleet=n,gpu_view=dict(time=0,cards=cards)) for n in ('A','B')]
    result=plan(states,120*GIB,20*GIB,now=0)
    assert result['quotas']=={'A':4,'B':4}  # one context fits on each of eight cards


def test_failed_memory_reader_holds_peer_admission_without_a_recovery_latch(tmp_path,monkeypatch):
    w=World(tmp_path,monkeypatch);a=w.lane('A');b=w.lane('B');assert w.spawn(a)
    def unavailable():raise OSError('simulated /proc failure')
    a.reader=unavailable;a.tick()
    assert not a.snapshot['memory_sample_valid']
    assert a.snapshot['projected_peak']==2*GIB  # known growth is retained
    assert not w.spawn(b)
    assert b.snapshot['reason']=='memory_unknown'
    assert b.snapshot['memory_sample_valid']  # its own reader still works
    a.reader=w.memory;w.sample();w.sample()
    assert w.spawn(b)


def test_peer_signal_failure_never_suppresses_known_local_critical_retirement(tmp_path,monkeypatch):
    w=World(tmp_path,monkeypatch);a=w.lane('A');b=w.lane('B');w.ramp()
    w.total-=15*GIB  # an unrelated allocation takes MemAvailable to 5 GiB
    def unavailable():raise OSError('simulated /proc failure')
    a.reader=unavailable;a.tick();b.tick()
    # 14-5 GiB is stopped at once (shed); boundary retirement still covers 24-5 GiB.
    assert b.snapshot['reason']=='shed_kill' and len(b.snapshot['kill'])==5 and len(b.retired)==10
    assert not b.snapshot['admit']
    w.boundary(b)
    assert w.memory()[0]>=24*GIB
