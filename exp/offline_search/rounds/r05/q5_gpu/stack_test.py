"""CPU-only fake stack: real plugin/orchestrator, K9 tensor math, no model/server."""
import argparse,json,sys,threading
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from common import *
from exp.offline_search.rounds.r05.q5_gpu.dev import gpu_retrieval as bridge
from exp.offline_search.closed_loop import selftest
import openpi.cache.config as cc
from openpi.cache.orchestrator import CacheOrchestrator

torch.set_num_threads(1)
class CPUConnection:
    def __init__(self,module):self.module=module;self.prev=None;self.stuck=0
    def run(self,keys,*,step,task,prev_hit,prev_action):
        if not step:self.prev=None;self.stuck=0
        prev=self.prev or {k:torch.zeros_like(v) for k,v in keys.items()}
        vals=[keys['vision_0'],keys['vision_1'],keys['robot_state'],torch.tensor(task),torch.tensor(step),torch.tensor(prev_hit),
              torch.zeros(10,32) if prev_action is None else torch.from_numpy(np.array(prev_action)),
              prev['vision_0'],prev['vision_1'],prev['robot_state'],torch.tensor(self.stuck)]
        with torch.inference_mode():o=self.module(*[v.unsqueeze(0) for v in vals])
        out={k:v.numpy()[0].copy() if v.ndim>1 else float(v[0]) for k,v in o.items()}
        out['rs']=keys['robot_state'].numpy().copy()
        self.prev={k:v.clone() for k,v in keys.items()}
        if self.module.judge:self.stuck=int(out['stuck_n'])
        return out,dict(event_ms=0.,wall_ms=0.,final_d2h_bytes=0,fake_cpu=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--mode',choices=['shadow','serve'],required=True);p.add_argument('--out',required=True);a=p.parse_args()
    c=config(a.config);root=Path(a.out);root.mkdir(parents=True,exist_ok=False)
    opts,_=plugin.parse_cli(['--os-method',c['spec'],'--os-kwargs',json.dumps(c['kwargs']),'--os-cell',c['cell'],
              '--os-root',str(STORE),'--os-log-dir',str(root),'--os-tag','test','--os-fit-artifact',c['path'],
              '--os-no-shadow-native','--os-tokens','off','--os-gpu-retrieval',a.mode]+(['--os-judge','guard_only'] if c['family']=='MixedJudge' else []))
    bridge.DeviceRuntime.connection=lambda self:CPUConnection(self.module)
    rt=plugin.PluginRuntime(opts,'pi05')
    cfg=cc.load_cache_config('exp/trace_dual/config/tr_pi05_l10_cache.yaml');shared=cc.build_shared_storage(cfg)
    qc=store.QueryCell(STORE,'pi05_l10_inf')
    def factory(_,bundle):
        comps=cc.build_per_connection_components(cfg,shared,quiet=True)
        comps=rt.attach(comps,cfg)
        kb=comps['key_builder'];s=plugin._TLS.new_sessions[-1]
        original_run=kb.graph.run
        def locked_run(*args,**kwargs):
            assert s.lock._is_owned() and not rt.decision_lock._is_owned()
            return original_run(*args,**kwargs)
        kb.graph.run=locked_run
        kb.inner.collect=lambda *a,**k:None
        kb.inner._slice=lambda:{'vision_0':torch.from_numpy(np.array(qc.key_v0[kb.row])),
                               'vision_1':torch.from_numpy(np.array(qc.key_v1[kb.row])),
                               'robot_state':torch.from_numpy(np.array(qc.rs[kb.row]))}
        kb.inner._reduce_vision=lambda x:x
        orch=CacheOrchestrator(storage=comps['storage'],key_builder=kb,gates=comps['gates'],judges=comps['judges'],search_strategies=comps['search_strategies'],
                              timer=comps['timer'],write_policy=comps.get('write_policy'),offline_writers=comps.get('offline_writers',()),library_stats=comps.get('library_stats'))
        return selftest.FakePolicy(orch,kb,qc.a_inf)
    conns=[plugin._wrap_factory(factory)(None,str(i)) for i in range(8)]
    assert len({id(c._osp_lock) for c in conns})==8
    count=0;miss=0
    # Each connection reuses itself for two tasks; reference sees actual served history.
    for reset in range(2):
        for idx,conn in enumerate(conns):
            e=next(e for e in qc.episodes if e['task_id']==(idx+reset)%10)
            conn.on_episode_start(task=e['task'],extra_metadata={'task_id':e['task_id'],'orig_init_state_idx':e['init'],'task_uid':e['uid']})
            s=conn._osp_sessions[0];ref,_=plugin.clone_method(rt.method,strict=True)
            for row in range(e['start'],min(e['end'],e['start']+6)):
                # Serve must not call CPU retrieval. Shadow must return exact CPU action.
                if a.mode=='serve':
                    s.method.query=lambda q:(_ for _ in ()).throw(AssertionError('CPU retrieval called in serve'))
                output=conn.infer({'_row':row,'observation/state':np.array(qc.raw_state[row]),'prompt':e['task']})
                step=s.step-1;q=plugin.OnlineQueryView(s,step,e['task_id'],s.ep)
                if step==0:ref.reset(s.ep)
                if a.mode=='shadow':
                    expect=ref.query(q)
                    actual=np.asarray(output['actions'])
                    if output['hit_type']=='FULL_HIT':assert np.array_equal(actual,expect.action)
                    else:assert np.array_equal(actual,np.array(qc.a_inf[row],np.float32));miss+=1
                else:
                    assert np.isnan(s.b_v0.a[step]).all() and np.isnan(s.b_v1.a[step]).all()
                count+=1
            conn.on_episode_end(True)
    # A failed admitted decision must require explicit lifecycle reset, then recover.
    conn=conns[0];s=conn._osp_sessions[0];e=qc.episodes[0];row=e['start']
    def begin():
        conn.on_episode_start(task=e['task'],extra_metadata={'task_id':e['task_id'],'task_uid':e['uid']})
    begin();obs={'_row':row,'observation/state':np.array(qc.raw_state[row]),'prompt':e['task']}
    original=conn._osp_inner.infer
    def fail(obs):
        original(obs)
        raise ValueError('injected after commit')
    conn._osp_inner.infer=fail
    try:conn.infer(obs)
    except ValueError as error:assert 'injected' in str(error)
    else:raise AssertionError('missing injected failure')
    conn._osp_inner.infer=original
    try:conn.infer(obs)
    except RuntimeError as error:assert 'new episode' in str(error)
    else:raise AssertionError('failed graph connection reused without reset')
    begin();conn.infer(obs);conn.on_episode_end(True)
    # Packet-only CPU guard reconstruction against the original MixedJudge on real CPU features.
    checks=[]
    for k,v in [('os_tokens','on'),('os_blind',True),('os_no_shadow_native',False),('os_log_inputs',True)]:
        test=SimpleNamespace(**vars(opts));setattr(test,k,v)
        if k=='os_log_inputs':test.os_gpu_retrieval='serve'
        try:bridge.validate_options(test,'pi05')
        except ValueError:checks.append(k)
        else:raise AssertionError(k)
    try:bridge.validate_options(opts,'groot')
    except ValueError:checks.append('groot')
    else:raise AssertionError('groot')
    test=SimpleNamespace(**vars(opts));test.judge=SimpleNamespace(mode='quantile')
    try:bridge.validate_options(test,'pi05')
    except ValueError:checks.append('quantile')
    else:raise AssertionError('quantile')
    for kind in ('top1','top16_set','top16_order','chunk','confidence'):
        if a.mode=='shadow':assert all(kind+'_agree' in r['gpu_retrieval'] for r in map(json.loads,rt.dec_path.read_text().splitlines()) if r['ev']=='dec')
    report=dict(PASS=True,config=a.config,mode=a.mode,decisions=count,miss=miss,connections=8,reset_episodes=16,invalid_options=checks,
                shadow_actions_exact=a.mode=='shadow',serve_cpu_retrieval_poisoned=a.mode=='serve',
                per_connection_lock_audited=True,failure_requires_reset=True)
    dump(root/'report.json',report);print(json.dumps(report))
if __name__=='__main__':main()
