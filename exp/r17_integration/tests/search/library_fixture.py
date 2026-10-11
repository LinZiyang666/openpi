"""Small images/keys, genuine P6 writer path, explicitly synthetic B500."""
from dataclasses import asdict

import numpy as np

from exp.r17_integration.common.contracts import DecisionCapture, RequestContext
from exp.r17_integration.benchmarks.spec import digest as task_data_digest
from exp.r17_integration.debug.library_capture import LibraryCaptureBridge, LibraryDecisionSink
from exp.r17_integration.debug.validate import digest
from exp.r17_integration.ops.library_alignment import array_identity, SEED
from exp.r17_integration.ops.formal_eval import benchmark
from exp.r17_integration.tests.debug.fixtures import arrays, binding, record


class Poison:
    def __array__(self,*args,**kwargs):
        raise AssertionError('non-allowlisted product was copied')


def synthetic_capture(path, *, episodes=500, suite='libero_spatial', decisions=2, bad_key_dtype=False, episode_offset=0, attempt=1):
    b = binding('dp','libero',key_width=3)
    names = benchmark().ordered_tasks
    offset = 0 if suite=='libero_spatial' else 30
    order = {str(t):np.random.default_rng(SEED+t).permutation(np.arange(t*50,(t+1)*50)).tolist() for t in range(10)}
    witnesses, cohort = [], []
    for t in range(10):
        for j in range(50):
            init = np.array([t,j,.125],np.float64)
            witnesses.append(dict(task_id=t,orig_init_state_idx=j,episode_id=t*50+j,
                task_file=names[offset+t].split('/',1)[1],init_array=array_identity(init)))
    audit = dict(status='PASS',init_pool='B',suite=suite,rows=witnesses,subset_seed=SEED,
        selections={str(n):{str(t):order[str(t)][:n//10] for t in range(10)} for n in (50,100,200,300,400,500)})
    audit['digest'] = digest(audit)
    bridge = LibraryCaptureBridge(path,enabled=True,binding=b)
    for w in witnesses[episode_offset:episode_offset+episodes]:
        t,j,eid = w['task_id'],w['orig_init_state_idx'],w['episode_id']
        name = names[offset+t]
        ep = dict(record(j,task=name,n_decisions=decisions,attempt=attempt),benchmark_protocol=b['benchmark_protocol'],
            init_pool='B',split='library',library_episode_id=eid,episode_id=eid,suite_task_id=t,task_id=t+offset,
            init_state_sha256=w['init_array']['state_sha256'],
            task_data_hash=task_data_digest(dict(w['init_array'],task_name=name)),
            closed_ok=True,success=bool(eid%2),error=None,secret='DROP')
        cohort.append(ep)
        sink = LibraryDecisionSink(bridge,ep)
        for seq in range(decisions):
            payload,keys = arrays(b,eid*decisions+seq)
            # Poison checks happen before serialization, not by stripping NPZ.
            images = [np.arange(4*4*3,dtype=np.uint8).reshape(4,4,3)+np.uint8(eid%255),
                      np.full((4,4,3),seq+1,np.uint8)]
            sink.bind_observation(dict(zip(('observation/image','observation/wrist_image'),images)))
            request = RequestContext(ep['campaign'],b['benchmark_protocol'],name,j,seq,
                arm=ep['arm'],bundle_generation=ep['bundle_generation'],attempt=attempt,task_id=t+offset)
            keys['robot_state'] = payload['state']
            if bad_key_dtype: keys['vision_0']=keys['vision_0'].astype(np.float64)
            capture = DecisionCapture(request,'policy',0,0,(1,1,5),5,
                payload['normalized_actions'],payload['wire_actions'],dict(keys,not_needed=Poison()),
                diagnostics={'not_needed':Poison()},model_capture={'not_needed':Poison()})
            sink(capture)
            # Mutating the original after binding must not change saved bytes.
            images[0][:] = 0
            sink.on_execution(dict(request=asdict(request),decision_seq=seq,executed_count=5,cursor=5,
                                   terminal=seq==decisions-1))
    assert bridge.close()['drained']
    return b,cohort,audit,names
