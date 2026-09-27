"""Current GR00T serving split, batch 1, warmed eager path, same-noise K wiring."""
from __future__ import annotations
import argparse,json,os,pathlib,subprocess,time
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',choices=['spatial','l10'],default='spatial')
    p.add_argument('--reps',type=int,default=30);p.add_argument('--warmup',type=int,default=5)
    a=p.parse_args()
    adm=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.total,name','--format=csv,noheader,nounits'],text=True).strip()
    free,total,*_=adm.split(',')
    if int(free)<14336:raise SystemExit('NOT_ADMITTED: '+adm)
    import torch
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(9.5*1024/int(total))
    from exp.libero_groot.bench_profile import load_policy
    from exp.libero_groot.policy_adapter import build_groot_observation
    from openpi.cache.groot.interceptor import _unsqueeze_values
    from openpi.cache.groot.staged import GrootStagedRunner
    from exp.offline_search.harness.store import QueryCell
    ckpt=pathlib.Path('/data/ckpt/n15_libero_'+('10' if a.suite=='l10' else 'spatial'))
    policy=load_policy(ckpt,device='cuda',denoising_steps=8)
    runner=GrootStagedRunner(policy.model,compile_vision=False)
    qc=QueryCell('/dev/shm/offline_search_store',f'groot_{a.suite}_cache')
    row=int(qc.tok_rows[0]);ti=int(qc.tok_index[row]);ep=qc.episodes[int(qc.ep[row])]
    wire={'observation/image':np.asarray(qc.tok('img0')[ti]),'observation/wrist_image':np.asarray(qc.tok('img1')[ti]),
          'observation/state':np.asarray(qc.raw_state[row],np.float64),'prompt':str(ep['task'])}
    obs=_unsqueeze_values(build_groot_observation(wire))
    inp=policy.apply_transforms({k:v if isinstance(v,np.ndarray) else np.array(v) for k,v in obs.items()})
    out={'admission':adm,'suite':a.suite,'checkpoint':str(ckpt),'batch':1,'row':row,'prompt':ep['task'],
         'dtype':'production session autocast bfloat16','backend':'eager; start_server.sh omits --compile-stage1',
         'warmup':a.warmup,'repetitions':a.reps,'affinity':sorted(os.sched_getaffinity(0)),
         'torch':torch.__version__,'cuda':torch.version.cuda,'ms':{},'parity':[],
         'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    def timing(fn):
        for _ in range(a.warmup):fn()
        torch.cuda.synchronize();vals=[]
        for _ in range(a.reps):
            b,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            b.record();r=fn();e.record();e.synchronize();vals.append(b.elapsed_time(e))
        return {'mean':float(np.mean(vals)),'median':float(np.median(vals)),
                'p10':float(np.percentile(vals,10)),'p90':float(np.percentile(vals,90)),'samples':vals}
    with runner.session():
        s1=runner.run_stage1(inp);s2=runner.run_stage2_llm(s1)
        out['prefix_tokens']=int(s1.input_embeds.shape[1])
        for tag,fn in [('stage1_full',lambda:runner.run_stage1(inp)),('stage2_unpacked',lambda:runner.run_stage2_llm(s1))]:
            out['ms'][tag]=timing(fn);print(tag,out['ms'][tag]['median'],flush=True)
        for K in (1,2,4,8):
            policy.model.action_head.num_inference_timesteps=K
            torch.manual_seed(1903);noise=runner.sample_noise(s2)
            explicit=runner.run_stage3(s2,noise=noise).action_pred
            torch.manual_seed(1903);prod=runner.run_stage3(s2).action_pred
            out['parity'].append({'K':K,'same_noise_max_abs':float((explicit-prod).abs().max()),
                                  'exact':torch.equal(explicit,prod)})
            out['ms'][f'stage3_K{K}']=timing(lambda:runner.run_stage3(s2))
            print('stage3',K,out['ms'][f'stage3_K{K}']['median'],out['parity'][-1],flush=True)
        policy.model.action_head.num_inference_timesteps=8
        out['ms']['stage23_K8']=timing(lambda:runner.run_stage2(s1))
    out['peak_allocated_bytes']=torch.cuda.max_memory_allocated();out['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    out['completed_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
    (HERE/f'results/groot_{a.suite}.json').write_text(json.dumps(out,indent=2))
    assert all(r['exact'] for r in out['parity'])

if __name__=='__main__':
    try:main()
    except Exception as e:
        if 'out of memory' in str(e).lower():
            (HERE/'results/GPU_OOM_STOP').write_text(str(e));raise SystemExit('OOM: GPU work stopped; no retries')
        raise
