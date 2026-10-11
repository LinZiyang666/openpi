"""Same weights, recorded raw observation, official get_action vs R17 stages."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np

DEFAULT_OBSERVATIONS = '/data/mw_finetune/qualification/observations/cosmos_libero_raw_v1'


def run_parity(adapter, directory=DEFAULT_OBSERVATIONS, steps=(5, 3, 1, 2)):
    from .record_observations import validate
    from .protocols import PROPRIO_KEY
    from exp.r17_integration.common.contracts import RequestContext
    from exp.r17_integration.common.serving import full_call
    from cosmos_policy.experiments.robot.libero.run_libero_eval import prepare_observation
    from cosmos_policy.utils.utils import set_seed_everywhere
    import torch
    receipt = validate(directory)
    rows=[]
    with np.load(Path(directory)/'raw_observations.npz',allow_pickle=False) as data:
        for check, k in enumerate(steps):
            i = check % 3
            raw = {'agentview_image':data['agent_raw'][i], 'robot0_eye_in_hand_image':data['wrist_raw'][i],
                   'robot0_gripper_qpos':data['proprio'][i,:2], 'robot0_eef_pos':data['proprio'][i,2:5],
                   'robot0_eef_quat':data['proprio'][i,5:]}
            native = prepare_observation(raw,256,flip_images=True)
            wire = {'observation/image':np.ascontiguousarray(data['agent_raw'][i,::-1,::-1]),
                    'observation/wrist_image':np.ascontiguousarray(data['wrist_raw'][i,::-1,::-1]),
                    'observation/state':data['state'][i], PROPRIO_KEY:data['proprio'][i], 'prompt':str(data['prompt'][i])}
            request=RequestContext('cosmos-native-parity',adapter.metadata().allowed_benchmark_protocols[0],
                                   'libero_spatial/task0',0,i,explicit_seed=195)
            set_seed_everywhere(0)
            started=time.monotonic()
            # ndarray embedding selects the official cached-embedding branch,
            # with exactly the sealed prompt bytes; no text encoder is loaded.
            native_sigmas=[]
            def count_native_calls(module, args):
                x0_fn = args[0]
                def counted(x, sigma):
                    native_sigmas.append(sigma.detach().float().cpu().tolist())
                    return x0_fn(x, sigma)
                return (counted, *args[1:])
            hook=adapter.model.sampler.register_forward_pre_hook(count_native_calls)
            try:
                reference=adapter.cu.get_action(adapter.cfg,adapter.model,adapter.stats,native,
                    adapter.embeddings[wire['prompt']].float().numpy(),seed=195,
                    randomize_seed=False,num_denoising_steps_action=k,
                    generate_future_state_and_value_in_parallel=True)['actions']
            finally:
                hook.remove()
            torch.cuda.synchronize()
            native_seconds=time.monotonic()-started
            set_seed_everywhere(0)
            started=time.monotonic()
            sampled,result=full_call(adapter,wire,request,sampler=adapter.sampler_for_steps(k))
            torch.cuda.synchronize()
            staged_seconds=time.monotonic()-started
            expected,actual=np.asarray(reference),np.asarray(result.actions)
            error=float(np.max(np.abs(expected-actual)))
            exact=expected.dtype==actual.dtype and expected.tobytes()==actual.tobytes()
            if not exact:
                raise AssertionError(f'Cosmos native parity failed at row{i} K{k}: max_abs={error}; no tolerance assumed')
            if len(native_sigmas) != sampled.actual_nfe:
                raise AssertionError('Cosmos native/staged actual denoiser counts differ')
            rows.append(dict(row=i,k=k,actual_nfe=sampled.actual_nfe,native_actual_nfe=len(native_sigmas),
                native_denoiser_sigmas=native_sigmas,execution_sampler=asdict(sampled.sampler),
                arm_label=sampled.sampler.config.get('arm_label',f'K{k}'),shape=list(actual.shape),dtype=str(actual.dtype),
                bytes_exact=exact,max_abs=error,native_seconds=native_seconds,staged_seconds=staged_seconds,
                action_sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
    return dict(status='PASS',passed=True,upstream='cosmos_utils.get_action + LIBERO.prepare_observation',
                observations=receipt,checks=rows,tolerance=0.0,model_identity=adapter.metadata().identity,
                gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda)


def main():
    import argparse
    from .loader import load_adapter
    p=argparse.ArgumentParser();p.add_argument('--descriptor',required=True);p.add_argument('--output',required=True)
    p.add_argument('--observations',default=DEFAULT_OBSERVATIONS)
    p.add_argument('--steps',nargs='+',type=int,choices=(1,2,3,5),default=(5,3,1,2));a=p.parse_args()
    adapter=load_adapter(json.loads(Path(a.descriptor).read_text()))
    report=run_parity(adapter,a.observations,steps=a.steps)
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__': main()
