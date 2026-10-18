"""Real frozen WebSocket client/server, CPU fake arithmetic, no GPU or workers.

Exercises concurrent episode/select/infer/ACK/terminal protocol. The fake policy
is only a transport fixture and cannot establish model or cache equivalence.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import socket
import sys
import threading
import numpy as np
from artifacts import write_json
from concurrent_clients import drive
from concurrent_server import checked_pin
from replay import verify_runtime


class Policy:
    requires_context=True
    def __init__(self):self._awaiting=None;self.seq=0;self.controls=0
    def on_episode_start(self,**kwargs):self.__init__()
    def acknowledge_execution(self,ack):
        if self._awaiting!=ack['decision_seq'] or ack['executed_count']!=5:
            raise ValueError('wrong execution ACK')
        self._awaiting=None;self.controls+=5
        return dict(ack)
    def infer(self,observation):
        if '__execution__' in observation:self.acknowledge_execution(observation['__execution__'])
        request=observation['__request__']
        if self._awaiting is not None or request['decision_seq']!=self.seq:
            raise ValueError('request clock mismatch')
        self._awaiting=self.seq;self.seq+=1
        actions=np.full((5,7),(request['explicit_seed']%100)/100.,np.float32)
        return dict(actions=actions,__hit_meta__=dict(source='policy',decision_seed=request['explicit_seed'],
            decision_index=request['decision_seq'],env_step_index=self.controls,segment_head=True,
            clock_unit='segment_decision'))
    def on_episode_end(self,success=False):
        if self._awaiting is not None:raise ValueError('terminal ACK missing')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--trace',action='store_true')
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('mask CUDA for CPU protocol test')
    bundle=Path(__file__).parent/'concurrent/bundles/dp'
    pin=checked_pin(bundle);campaign=json.loads((bundle/'source_campaign.json').read_text())
    runtime=Path(campaign['runtime']);verify_runtime(runtime,pin['runtime_identity'])
    sys.path[:0]=[str(runtime),str(runtime/'src'),str(runtime/'packages/openpi-client/src')]
    from exp.r17_integration.common import websocket_policy_server as protocol
    if not Path(protocol.__file__).resolve().is_relative_to(runtime.resolve()):
        raise RuntimeError('CPU protocol test did not import the frozen module')
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1',0));port=reservation.getsockname()[1]
    ready=threading.Event();errors=[]
    server=protocol.WebsocketPolicyServer(Policy(),host='127.0.0.1',port=port,concurrent=True,
        connection_policy_factory=lambda base,bundle_id:Policy(),allow_dynamic_bundles=False,
        ready_callback=ready.set,metadata={'cpu_transport_fixture':True})
    def serve():
        try:asyncio.run(server.run(stop_on_request=True))
        except BaseException as error:errors.append(repr(error));ready.set()
    thread=threading.Thread(target=serve,daemon=True);thread.start()
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    if a.trace:
        from client_trace import install
        install(output/'client_trace')
    counts=[]
    try:
        if not ready.wait(15) or errors:raise RuntimeError(errors or 'CPU server did not bind')
        rows=[dict(observation={'observation/image':np.zeros((2,2,3),np.uint8),
            'observation/state':np.zeros(8,np.float32)},
            request=dict(task_name='cpu_transport_fixture',task_id=0,original_init_idx=0,benchmark_protocol='fixture'))]
        for repeat in range(2):
            records=drive(lambda:protocol.WebsocketClientPolicy('127.0.0.1',port),rows,
                dict(benchmark='libero',replan_steps=5),clients=8,deliveries=8,round_index=repeat,
                bundle_id='default',output=output/('round_'+str(repeat)))
            counts.append(len(records))
    finally:
        server.request_stop();thread.join(15)
    if thread.is_alive() or errors:raise RuntimeError(errors or 'CPU server did not close')
    import torch
    if torch.cuda.is_initialized():raise RuntimeError('CPU transport test initialized CUDA')
    report=dict(status='CPU_PROTOCOL_PASS',requests_per_round=counts,clients=8,
        runtime_identity=pin['runtime_identity'],protocol_module=protocol.__file__,
        cuda_initialized=False,server_closed=True,model_evidence=False,cache_evidence=False)
    if a.trace:
        paths=list((output/'client_trace').glob('*.jsonl'))
        queries=[json.loads(line)for path in paths if not path.name.endswith('.executed.jsonl')for line in path.read_text().splitlines()]
        executions=[json.loads(line)for path in paths if path.name.endswith('.executed.jsonl')for line in path.read_text().splitlines()]
        terminals=sum(row['terminal_ack'] for row in executions)
        if len(queries)!=sum(counts) or len(executions)!=sum(counts) or terminals!=16:
            raise ValueError('frozen socket trace missed requests or terminal ACKs')
        if any(row['server_clock_agrees'] is not True for row in queries):
            raise ValueError('wire execution clock differs from CPU server clock')
        report['trace']=dict(queries=len(queries),executions=len(executions),terminal_acks=terminals,
            hook='actual stock client base and direct _send_ctrl; includes R17 subclass calls')
    write_json(output/'RECEIPT.json',report);print(json.dumps(report))


if __name__=='__main__':main()
