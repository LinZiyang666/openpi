"""Opt-in evidence wrapper for NEW formal worker processes only.

Native socket calls/ACKs/results are unchanged. Observed state is policy wire
state, not hidden simulator state. Camera hashes can expose renderer drift.
"""
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid
import numpy as np


def install(root,client_class=None,record_limit=0):
    if client_class is None:
        # Both frozen LIBERO/MT50 worker factories instantiate StockClient.
        # Wrapping only the R17 subclass would miss their actual traffic.
        from openpi_client.websocket_client_policy import WebsocketClientPolicy as client_class
    if getattr(client_class,'_xgpu_trace_installed',False):raise RuntimeError('trace installed twice')
    client_class._xgpu_trace_installed=True
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    native_infer=client_class.infer;native_start=client_class.episode_start
    native_end=client_class.episode_end
    native_control=getattr(client_class,'_send_ctrl',None)
    lock=threading.Lock();states={}
    def state(client):
        # Weak ownership is unnecessary for bounded worker lifetimes, but key by
        # object itself rather than id() to avoid id reuse between connections.
        if client not in states:
            name=str(os.getpid())+'_'+uuid.uuid4().hex
            states[client]=dict(clock=0,records=0,path=root/(name+'.jsonl'),name=name,
                execution_path=root/(name+'.executed.jsonl'),pending=None,recorder=None)
        return states[client]
    def start(self,*args,**kwargs):
        response=native_start(self,*args,**kwargs)
        with lock:
            s=state(self);s['clock']=0;s['episode_metadata']=kwargs;s['pending']=None
        return response
    def executed(s,ack,terminal=False):
        pending=s['pending']
        if pending is None:raise ValueError('execution ACK has no recorded returned chunk')
        count=ack.get('executed_count')
        if (type(count) is not int or not 0<=count<=len(pending['actions']) or
                ack.get('decision_seq')!=pending['request']['decision_seq']):
            raise ValueError('execution ACK does not identify a valid returned prefix')
        row=dict(schema='xgpu.client_execution.v1',request=pending['request'],
            wire_delivery=pending['request']['decision_seq'],env_step=pending['env_step'],
            decision_step=pending['decision_step'],actions=pending['actions'][:count].tolist(),
            action_width=pending['actions'].shape[1],executed_count=count,execution=dict(ack),
            terminal_ack=terminal,scope='ACK-confirmed returned prefix; no simulator hidden state')
        with s['execution_path'].open('a') as stream:
            stream.write(json.dumps(row,allow_nan=False,sort_keys=True)+'\n')
        s['clock']+=count;s['pending']=None
    def infer(self,obs):
        began=time.monotonic();response=native_infer(self,obs)
        request=obs.get('__request__')
        if request is None:return response  # driver control probes are not episodes
        with lock:
            s=state(self);execution=obs.get('__execution__')
            if execution:executed(s,execution)
            hit=response.get('__hit_meta__',{});timing=response.get('server_timing',{})
            actions=np.asarray(response['actions'])
            if not np.isfinite(actions).all():raise ValueError('nonfinite native actions in trace')
            row=dict(schema='xgpu.client_trace.v1',request=request,
                episode_metadata=s.get('episode_metadata'),wire_delivery=request['decision_seq'],
                decision_step=hit.get('decision_index',request['decision_seq']),
                env_step=s['clock'],server_env_step=hit.get('env_step_index'),
                server_clock_agrees=(hit['env_step_index']==s['clock'])if 'env_step_index'in hit else None,
                state=np.asarray(obs['observation/state']).tolist(),
                actions=actions.tolist(),actions_dtype=str(actions.dtype),
                actual_noise_seed=hit.get('decision_seed',(hit.get('rng_receipt') or {}).get('initial_seed')),
                cache_decision={k:hit.get(k) for k in ('source','hit_type','donor_rows','corrector_applied',
                    'stall_call','stall_status','knob_call','segment_head','clock_unit')},
                camera_sha256={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
                    for k,v in obs.items() if isinstance(v,np.ndarray) and v.ndim==3},
                server_timing=timing,client_latency_seconds=time.monotonic()-began,
                state_scope='observed policy wire state, not full simulator state')
            with s['path'].open('a') as stream:stream.write(json.dumps(row,allow_nan=False,sort_keys=True)+'\n')
            s['pending']=dict(request=dict(request),actions=actions.copy(),
                env_step=row['env_step'],decision_step=row['decision_step'])
            if s['records']<record_limit:
                from capture import Recorder
                if s['recorder'] is None:
                    s['recorder']=Recorder(root/(s['name']+'_inputs'),limit=record_limit,
                        provenance=dict(kind='formal_socket_observer',physical_trajectory=True,
                            history_coverage='ordered wire frames; stateful server must reconstruct history',
                            seed_scope='wire seed; actual segment-head seed retained in client trace'))
                s['recorder'].add({k:v for k,v in obs.items() if not k.startswith('__')},request,
                    history={},episode_start=request['decision_seq']==0,
                    executed_controls_since_last_query=0 if execution is None else int(execution['executed_count']))
                s['records']+=1;s['recorder'].close()
        return response
    def end(self,*args,**kwargs):
        response=native_end(self,*args,**kwargs)
        with lock:
            if kwargs.get('execution') is not None:
                executed(state(self),kwargs['execution'],terminal=True)
        return response
    def control(self,message,*args,**kwargs):
        response=native_control(self,message,*args,**kwargs)
        # Frozen benchmarks.execution.end_episode calls _send_ctrl directly.
        # Observe the actual control once, including callers of episode_end;
        # wrapping both would duplicate the terminal prefix.
        ack=message.get('__execution__')
        if ack is not None and message.get('__ctrl__') in ('episode_end','acknowledge_execution'):
            with lock:executed(state(self),ack,terminal=message['__ctrl__']=='episode_end' or ack.get('terminal')is True)
        return response
    client_class.episode_start=start;client_class.infer=infer
    if native_control is None:client_class.episode_end=end
    else:client_class._send_ctrl=control


def from_environment():
    root=os.environ.get('XGPU_TRACE_ROOT')
    if root:install(root,record_limit=int(os.environ.get('XGPU_RECORD_LIMIT','0')))
