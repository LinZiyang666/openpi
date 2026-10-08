"""CPU replay of every dev resident method against its standalone runtime.

Recorded B-library observations; real plugin/guards/corrector/knob/tail,
synthetic expert velocity forwards through the real Euler loops. No A outcomes,
checkpoint loads, subprocesses, network, servers, or CUDA contexts.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import MethodType, SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch

from exp.offline_search.closed_loop import plugin
from exp.offline_search.closed_loop.selftest import FakeKB, FakePolicy
from exp.offline_search.rounds.r15.descent.mechanics import Harness, semantic
from exp.offline_search.rounds.r15.descent.proof import cpu_fence
from openpi.cache.orchestrator import CacheOrchestrator
from openpi.cache.types import CheckpointID
from openpi.cache.components.judge import HitType
from .replay import synthetic_chunk, untimed, boundaries
from .resident import install
from .conductor import read_group
from .prepare import HERE, write_json


def connection(h,rt,row,build,*,resident=False):
    lib=h.lib;cfg=h.cc.load_cache_config(row['yaml']);k=row['r16_k'];name=row['arm']
    class CPUFake(FakePolicy):
        def stage1(self,obs):pass
        def infer(self,obs):
            self.stage1(obs);r=int(obs['_row']);self.kb.row=r
            result=self.orch.check(CheckpointID.CP1,stage1=None)
            if result.hit_type==HitType.FULL_HIT:chunk=result.payload.action_chunk
            elif result.hit_type==HitType.MISS:
                chunk=synthetic_chunk(rt.model,torch.from_numpy(np.array(self.policy[r],dtype=np.float32))[None],k)[0]
            else:raise AssertionError(result.hit_type)
            self.orch.broadcast_action(chunk);self.orch.clear()
            return dict(winner=result.entry_id,hit_type=result.hit_type.name,actions=np.asarray(chunk),
                        diag=(result.factor_outputs or {}).get('osplug'))
        def _osp_prepare_blind(self,obs):return np.array(lib.rs[obs['_row']],copy=True),None
        def _osp_blind_output(self,action,state):return dict(actions=action.copy())
        def _osp_shadow_chunk(self,obs):return np.array(lib.action[obs['_row']],np.float32,copy=True)
    def factory(base,bundle_id):
        comps=build(cfg,h.shared,quiet=True,yaml_id=name)
        if not resident:comps=rt.attach(comps,cfg,yaml_id=name)
        kb=FakeKB(NS(key_v0=lib.key_v0,key_v1=lib.key_v1,rs=lib.rs))
        for sess in plugin._TLS.new_sessions:sess.kb=kb
        orch=CacheOrchestrator(storage=comps['storage'],key_builder=kb,gates=comps['gates'],
            judges=comps['judges'],search_strategies=comps['search_strategies'],timer=comps['timer'],
            write_policy=comps.get('write_policy'),offline_writers=comps.get('offline_writers',()),
            library_stats=comps.get('library_stats'))
        return CPUFake(orch,kb,lib.action)
    conn=plugin._wrap_factory(factory)(None,name)
    conn._osp_sessions[0].miss_steps=k  # the synthetic policy has no live head; real boundary tested separately
    method=conn._osp_sessions[0].method
    progress=method.inner._progress
    def forced(owner,q,top1):
        result=progress(q,top1)
        if q.step in (2,8):owner._noprog_span=owner.noprog_n-1;return owner._noprog_span
        return result
    method.inner._progress=MethodType(forced,method.inner)
    return conn


def standalone(row,path):
    args=['--os-method',row['method'],'--os-kwargs',json.dumps(row['kwargs']),'--os-cell',row['cell'],
          *row['plugin_args'],'--os-log-dir',str(path),'--os-tag',row['arm']+'_replay']
    opts,rest=plugin.parse_cli(args);assert not rest;opts.os_miss_steps=row['r16_k']
    return plugin.PluginRuntime(opts,row['model'])


def run(index,out):
    import openpi.cache.config as cc
    from openpi.serving import websocket_policy_server as wps
    original_build=cc.build_per_connection_components;original_server=wps.WebsocketPolicyServer
    records={};memory={};seen={'guard':0,'knob':0,'tail':0,'pure':0}
    with cpu_fence(), tempfile.TemporaryDirectory(prefix='r16_resident_replay_') as scratch:
        scratch=Path(scratch)
        for group_path in json.loads(Path(index).read_bytes())['groups']:
            group=read_group(group_path);label=f'{group["model"]}_k{group["k"]}'
            entries_by_arm={e['row']['arm']:e for e in group['entries']}
            local=copy.deepcopy(group);local['root']=str(scratch/label/'resident')
            registry=install(local,group['model'],'replay')
            memory[label]=json.loads((Path(local['root'])/'resident/replay.json').read_bytes())
            harnesses={s:Harness(f'{group["model"]}_{s}_100',scratch/label/s) for s in ('l10','spatial')}
            def capture(rt):
                rows=[];original=rt.emit
                def emit(row):
                    offset=rt.dec_path.stat().st_size if rt.dec_path.exists() else 0
                    original(row)
                    if row.get('ev')=='dec':
                        with rt.dec_path.open('rb') as stream:
                            stream.seek(offset);rows.append(json.loads(stream.readline()))
                rt.emit=emit;return rows
            baseline={};live={};logs={};owned=[]
            try:
                for entry in group['entries']:
                    row=entry['row'];name=row['arm'];s=row['suite_short'];h=harnesses[s]
                    rt=standalone(row,scratch/label/'single'/name);owned.append(rt);rows=capture(rt)
                    conns=[connection(h,rt,row,original_build) for _ in range(2)]
                    identities=[next((t,i) for t,i in entry['pairs'] if t==task) for task in (0,1)]
                    actions={}
                    for conn,(t,i) in zip(conns,identities):
                        conn.on_episode_start(task=h.lib.meta['tasks'][str(t)],episode_id=i,
                            extra_metadata=dict(task_uid=f'{name}:eval:{t}:{i}',task_id=t,orig_init_state_idx=i,attempt=1))
                    for step in range(12):
                        for conn,(t,i) in zip(conns,identities):
                            obs=h.obs(t,step,init=i)
                            actions[t,step]=np.asarray(conn.infer(obs)['actions']).tobytes()
                    for conn in conns:conn.on_episode_end(success=False)
                    baseline[name]=(actions,rows)
                    rr=registry.runtimes[name];logs[name]=capture(rr)
                    live[name]=([connection(h,rr,row,cc.build_per_connection_components,resident=True) for _ in range(2)],identities,h,row)
                    for conn,(t,i) in zip(live[name][0],identities):
                        conn.on_episode_start(task=h.lib.meta['tasks'][str(t)],episode_id=i,
                            extra_metadata=dict(task_uid=f'{name}:eval:{t}:{i}',task_id=t,orig_init_state_idx=i,attempt=1))
                    assert conns[0]._osp_sessions[0].method is not live[name][0][0]._osp_sessions[0].method
                # Every method is live at once; rotate order and interleave peers.
                actions={name:{} for name in live}
                for step in range(12):
                    order=sorted(live,reverse=bool(step%2))
                    for name in order:
                        conns,identities,h,row=live[name]
                        for peer in (1,0):
                            t,i=identities[peer];conn=conns[peer]
                            actions[name][t,step]=np.asarray(conn.infer(h.obs(t,step,init=i))['actions']).tobytes()
                for name,(conns,identities,h,row) in live.items():
                    for conn in conns:conn.on_episode_end(success=False)
                    assert actions[name]==baseline[name][0],f'Action bytes differ for {name}'
                    def semantic_rows(rows):
                        normalized={}
                        for d in rows:
                            assert d['yaml_id']==d['bundle']==name and d['uid'].startswith(name+':eval:')
                            data=semantic(d);data.pop('arm',None)
                            data['r14']=untimed(d.get('r14',{}))
                            data['method']=d['method'];data['tag']=d['tag'];data['stage1_calls']=d['stage1_calls']
                            normalized[d['task_id'],d['step']]=json.dumps(data,sort_keys=True,default=plugin._json_default)
                        return normalized
                    assert semantic_rows(baseline[name][1])==semantic_rows(logs[name]),f'Decision mismatch: {name}'
                    assert all(d['arm']==name for d in logs[name])
                    fit_hash=entries_by_arm[name]['fit_sha256']
                    assert all(d['r14']['fit_sha256']==fit_hash for d in baseline[name][1]+logs[name]),'Wrong replay fit binding'
                    for d in logs[name]:
                        seen['guard']+=not d['hit'] and d.get('r14',{}).get('guard',False)
                        seen['knob']+=not d['hit'] and d.get('r14',{}).get('knob',False)
                        seen['tail']+=d.get('source',d.get('src'))=='policy_tail'
                        seen['pure']+=row['r14_descent_level']=='c10' and not d['hit']
                    boundaries(row['model'],row['r16_k'])
                    records[name]=dict(PASS=True,actions_byte_equal=True,decisions_semantic_equal=True,
                        interleaved_connections=2,decisions=len(logs[name]),k=row['r16_k'],
                        fit_sha256=fit_hash,original_uids=True,per_decision_arm=True)
            finally:
                for rt in owned:rt.flush_all()
                for h in harnesses.values():h.rt.flush_all()
                registry.flush_all();cc.build_per_connection_components=original_build
                wps.WebsocketPolicyServer=original_server;plugin.RUNTIME=None
    assert len(records)==176 and all(seen.values()),seen
    result=dict(PASS=True,arms=len(records),decisions=sum(r['decisions'] for r in records.values()),
                records=records,resident_cpu_memory=memory,observed_paths=seen,
                test_a_outcome_reads=0,servers_launched=0,gpu_calls=0,
                numerical_scope='Real Euler loops with synthetic velocity forwards, real lower layers, recorded B observations; GPU checkpoint parity remains coordinator smoke')
    write_json(out,result);print(f'RESIDENT_REPLAY_PASS arms={len(records)} decisions={result["decisions"]}')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--groups',type=Path,default=HERE/'conductor_dev_groups.json')
    ap.add_argument('--out',type=Path,default=HERE/'conductor_runtime_proof.json');args=ap.parse_args();run(args.groups,args.out)


if __name__=='__main__':main()
