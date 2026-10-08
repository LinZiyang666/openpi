"""CPU replay with real plugin/lower layers and synthetic action-expert forwards.

No checkpoint loads, A queries/outcomes, subprocesses, network, or GPU calls.
Default/no-override and explicit default K share noise and every other input.
"""
from __future__ import annotations

import contextlib
import json
import hashlib
import importlib.util
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS, MethodType
from unittest.mock import patch

import numpy as np
import torch
import yaml

from exp.offline_search.closed_loop import plugin, stage_overrides
from exp.offline_search.closed_loop.selftest import FakePolicy
from exp.offline_search.rounds.r15.descent.mechanics import Harness, load, semantic
from exp.offline_search.rounds.r15.descent.proof import cpu_fence
from openpi.cache.components.judge import HitType
from openpi.cache.types import CheckpointID, groot_n15_schedule
from openpi.cache.interceptor import InferenceInterceptor as CacheInterceptor
from openpi.cache.groot.interceptor import GrootCacheInterceptor
from openpi.cache.groot import staged, batcher as gb
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Stage1Output, Stage2Output
from openpi.serving.batching_coordinator import Pi05StageBatcher
from openpi.serving.batching_core import Stage3MissPayload
from .prepare import HERE, write_json


class TinyPi:
    run_stage3 = PI0Pytorch.run_stage3
    _stage3_with_intermediates = PI0Pytorch._stage3_with_intermediates
    _stage3_action_expert = PI0Pytorch._stage3_action_expert

    def __init__(self):
        self.calls = 0

    def denoise_step(self,state,pads,past,x,t):
        self.calls += 1
        return x*.125 + t[:,None,None]*.01


def pi_stage2():
    s1 = Stage1Output(state=torch.zeros(1,8),prefix_embs=torch.zeros(1,2,4),
                     prefix_pad_masks=torch.ones(1,2,dtype=torch.bool),
                     prefix_att_2d_masks_4d=torch.zeros(1,1,2,2),prefix_position_ids=torch.arange(2)[None])
    return Stage2Output(stage1=s1,past_key_values=())


def synthetic_chunk(model,noise,k):
    """Real Euler implementations, stand-in deterministic velocity evaluations."""
    if model=='pi05':
        head = TinyPi()
        out = head.run_stage3(pi_stage2(),noise=noise,num_steps=k,return_intermediates=True)
        assert head.calls==k
        return out.action_chunk
    head = NS(num_inference_timesteps=k,num_timestep_buckets=1000,
              process_backbone_output=lambda b:NS(backbone_features=b),
              state_encoder=lambda s,e:s)
    calls = []
    def step(h,vl,state,e,x,t,dt):
        calls.append(t.clone())
        return x+dt*(x*.125+t[:,None,None]*.00001)
    out = staged.denoise_loop(head,torch.zeros(1,2,4),dict(embodiment_id=torch.zeros(1,dtype=torch.long),
        state=torch.zeros(1,1,8)),noise=noise,num_steps=head.num_inference_timesteps,step_fn=step)
    assert len(calls)==k
    return out


def untimed(value):
    if isinstance(value,dict):
        return {k:untimed(v) for k,v in value.items() if not k.endswith(('_ms','_us'))}
    if isinstance(value,list):return [untimed(v) for v in value]
    return value


def boundaries(model,k):
    """Actual interceptor and blind adapter step-count boundaries; mismatch fails."""
    rt = NS(model=model); session = NS(rt=rt)
    orch = NS(_twins=None)
    if model=='pi05':
        p = NS(_orchestrator=orch,_miss_num_steps=k,_stage1_fn=lambda x:x)
        p._miss_steps = MethodType(CacheInterceptor._miss_steps,p)
        assert p._miss_steps()==k
    else:
        runner = NS(_model=NS(action_head=NS(num_inference_timesteps=k)),run_stage1=lambda x:x)
        p = NS(_orchestrator=orch,_runner=runner)
        assert GrootCacheInterceptor._require_miss_steps(p,k)==k
        try: GrootCacheInterceptor._require_miss_steps(p,k+1)
        except RuntimeError: pass
        else: raise AssertionError('GR00T mismatch accepted')
    plugin._BlindAdapter(p,session)
    assert session.miss_steps==k


def batchers():
    evidence = {}
    for k in (1,2,10):
        head = TinyPi(); s2 = pi_stage2(); noise = torch.arange(50*32,dtype=torch.float32).reshape(1,50,32)/1000
        direct = head.run_stage3(s2,noise=noise,num_steps=k,return_intermediates=True).action_chunk
        head.calls=0
        reply = Pi05StageBatcher(head,torch.device('cpu')).run_stage3_miss(
            [Stage3MissPayload(s2,noise[0],k)],num_steps=k,save_timesteps_per_request=[None])[0]
        assert head.calls==k and torch.equal(reply.action_chunk,direct)
        evidence[f'pi05_k{k}'] = dict(calls=k,batcher_direct_byte_equal=True,
            action_sha256=hashlib.sha256(direct.detach().numpy().tobytes()).hexdigest())
    # Existing full-shape GR00T upstream stub, real runner and batcher.
    from tests.cache.groot.test_groot_stage3 import _model_with_flow_head, _fixed_noise, _upstream_with_noise
    for k in (1,2,8):
        model = _model_with_flow_head(k); runner = staged.GrootStagedRunner(model,verify_upstream=False)
        with runner.session():
            s2 = runner.run_stage2_llm(runner.run_stage1(model.build_inputs()))
            noise = _fixed_noise()
            direct = runner.run_stage3(s2,noise=noise).action_pred
            upstream = _upstream_with_noise(model.action_head,runner._head_inputs(s2),s2.action_inputs,noise)
        cond = gb.stage3_input(runner,s2)
        reply = gb.GrootStageBatcher(runner).run_stage3_miss([Stage3MissPayload(cond,noise[0],k)],
                    num_steps=k,save_timesteps_per_request=[None])[0]
        assert torch.equal(reply.action,direct) and torch.equal(direct,upstream)
        evidence[f'groot_k{k}'] = dict(calls=k,batcher_direct_byte_equal=True,upstream_transcription_byte_equal=True,
            action_sha256=hashlib.sha256(direct.detach().numpy().tobytes()).hexdigest())
    return evidence


def schedule_default_proof():
    from openpi.cache import types as current
    spec=importlib.util.spec_from_loader('r16_before_types',loader=None)
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old
    exec(compile((HERE/'before/types.py.txt').read_bytes(),'r16_before_types','exec'),old.__dict__)
    proof={}
    for k in (2,4,8,10):
        before=old.groot_n15_schedule(k);after=current.groot_n15_schedule(k)
        assert vars(before)==vars(after)
        assert before.timesteps==after.timesteps
        assert [before.replay_timestep(i) for i in range(k)]==[after.replay_timestep(i) for i in range(k)]
        proof[str(k)]=dict(schedule_bytes_equal=json.dumps(vars(before),sort_keys=True)==json.dumps(vars(after),sort_keys=True),
                          timesteps_equal=True,replay_times_equal=True)
    try:old.groot_n15_schedule(1)
    except ValueError:pass
    else:raise AssertionError('Original unexpectedly accepted k1')
    one=current.groot_n15_schedule(1)
    assert one.timesteps==() and one.snapshot_t(0)==0.
    try:one.remaining_steps(.5)
    except ValueError:pass
    else:raise AssertionError('k1 schedule exposes warm start')
    return dict(defaults=proof,k1_only_new_explicit_behavior=True,k1_warm_start_refused=True)


def replay(directory):
    records = {}; seen = {'guard':0,'knob':0,'tail':0,'pure':0}
    with cpu_fence():
        for model in ('pi05','groot'):
            default = 10 if model=='pi05' else 8
            for level in ('off','ir45','c10'):
                h = Harness(f'{model}_l10_100',Path(directory)/f'{model}_{level}')
                original_connection = h.connection
                current_k = default
                def connection(method,**kwargs):
                    conn = original_connection(method,**kwargs)
                    # Only the synthetic interceptor stand-in is configured here;
                    # the actual interceptor/adapter and batcher boundaries are tested above.
                    conn._osp_sessions[0].miss_steps = current_k
                    conn._osp_adapter.policy.r16_k = current_k
                    return conn
                h.connection = connection
                def fake_infer(self,obs):
                    r = int(obs['_row']); self.kb.row=r
                    result = self.orch.check(CheckpointID.CP1,stage1=None)
                    if result.hit_type==HitType.FULL_HIT: chunk=result.payload.action_chunk
                    elif result.hit_type==HitType.MISS:
                        z=torch.from_numpy(np.array(self.policy[r],dtype=np.float32))[None]
                        chunk=synthetic_chunk(model,z,self.r16_k)[0]
                    else: raise AssertionError(result.hit_type)
                    self.orch.broadcast_action(chunk); self.orch.clear()
                    return dict(winner=result.entry_id,hit_type=result.hit_type.name,actions=np.asarray(chunk),
                                diag=(result.factor_outputs or {}).get('osplug'))
                try:
                    with patch.object(FakePolicy,'infer',fake_infer):
                        runs=[]
                        for k in (None,default,1,2):
                            path=Path(directory)/f'{model}_{level}_{k}.yaml'
                            path.write_text(yaml.safe_dump({} if k is None else dict(miss=dict(num_steps=k))))
                            current_k=stage_overrides.miss_steps_from_yaml(path,model,default if model=='groot' and k is None else k if model=='groot' else None)
                            boundaries(model,current_k)
                            actions,decisions,rows=h.run(load(f'{model}_l10_100',level),tasks=range(2),steps=16,force_guards=True)
                            # Include lower-layer audit values in the default equality proof.
                            decisions={(d['task_id'],d['step']):json.dumps(dict(semantic(d),r14=untimed(d.get('r14'))),
                                       sort_keys=True,default=plugin._json_default) for d in rows}
                            assert all(d['hit'] or d['miss_k']==current_k for d in rows)
                            assert all(d['vision'] or d['miss_k'] is None for d in rows)
                            tails=[d for d in rows if d.get('source')=='policy_tail']
                            for d in tails:
                                assert d['hit'] and not d['vision']
                                s=d['step']; t=d['task_id']
                                prior=np.frombuffer(actions[t,s-1],np.float32).reshape(h.rt.H,32)
                                tail=np.frombuffer(actions[t,s],np.float32).reshape(h.rt.H,32)
                                from exp.offline_search.closed_loop.blind import policy_tail_chunk
                                assert tail.tobytes()==policy_tail_chunk(prior).tobytes()
                            runs.append((actions,decisions,rows))
                        assert runs[0][0]==runs[1][0]
                        assert [(d['method'],d['tag'],d['stage1_calls']) for d in runs[0][2]]==[
                            (d['method'],d['tag'],d['stage1_calls']) for d in runs[1][2]]
                        if runs[0][1]!=runs[1][1]:
                            key=next(x for x in runs[0][1] if runs[0][1][x]!=runs[1][1][x])
                            before=json.loads(runs[0][1][key]);after=json.loads(runs[1][1][key])
                            changes={x:[str(before.get(x))[:300],str(after.get(x))[:300]] for x in set(before)|set(after)
                                     if before.get(x)!=after.get(x)}
                            raise AssertionError((model,level,key,changes))
                        reduced=runs[2][2]+runs[3][2]
                        seen['guard'] += sum(not d['hit'] and d.get('r14',{}).get('guard',False) for d in reduced)
                        seen['knob'] += sum(not d['hit'] and d.get('r14',{}).get('knob',False) for d in reduced)
                        seen['tail'] += sum(d.get('source')=='policy_tail' for d in reduced)
                        if level=='c10': seen['pure'] += sum(not d['hit'] for d in reduced)
                        records[f'{model}_{level}']=dict(default_steps=default,decisions=32,
                            default_action_bytes_equal=True,default_decision_bytes_equal=True,
                            method_tag_stage1_call_counts_equal=True,
                            lower_layer_audit_equal=True,canonicalization='Only clocks/timings and connection identities excluded',
                            reduced_k_checked=[1,2],synthetic_forward=True)
                finally: h.rt.flush_all()
    assert all(seen.values()),seen
    return dict(PASS=True,records=records,covered_paths=seen,batchers=batchers(),shared_runtime_edits=1,
                schedule_before_after=schedule_default_proof(),
                outcome_reads=0,servers=0,gpu_calls=0,limitations='CPU deterministic forward stubs; real-weight smoke remains coordinator work')


def main():
    with tempfile.TemporaryDirectory(prefix='r16_replay_',dir='/tmp') as directory:
        result=replay(directory)
    write_json(HERE/'runtime_proof.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
