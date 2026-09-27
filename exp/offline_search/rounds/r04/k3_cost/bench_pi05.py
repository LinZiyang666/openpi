"""Batch-1 warmed eager serving-path stage costs and same-noise parity; no server/ports."""
from __future__ import annotations
import argparse, dataclasses, hashlib, json, os, pathlib, subprocess, time
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/dev/shm/offline_search_store')
    p.add_argument('--reps', type=int, default=30)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--out', default=str(HERE/'results/pi05.json'))
    a = p.parse_args()
    admission = subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.total,name','--format=csv,noheader,nounits'], text=True).strip()
    free,total,*name = admission.split(',')
    if int(free)<14336: raise SystemExit('NOT_ADMITTED: '+admission)
    import torch, jax
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(9.5*1024/int(total))
    from openpi.training import config
    from openpi.policies import policy_config
    from openpi.models.model import Observation
    from openpi.cache.components.key_builder import CP1SpatialPool16KeyBuilder
    from openpi.cache.types import CheckpointID
    from exp.offline_search.harness.store import QueryCell
    from exp.offline_search.rounds.r04.k3_cost.dev.stage_overrides import Pi05Override
    cfg = config.get_config('pi05_libero')
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    ckpt='/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch'
    policy=policy_config.create_trained_policy(cfg, ckpt, pytorch_device='cuda')
    model=policy._model.eval()
    dummy=Pi05Override('dummy_cached'); wrist=Pi05Override('wrist_only')
    results={'admission':admission,'checkpoint':ckpt,'batch':1,'dtype':'serving mixed bfloat16/float32',
             'backend':'eager SDPA; concurrent Pi05StageAdapter calls run_stage1/2/3 directly; no CUDA graphs',
             'return_intermediates':True,'warmup':a.warmup,'repetitions':a.reps,'affinity':sorted(os.sched_getaffinity(0)),
             'torch':torch.__version__,'cuda':torch.version.cuda,'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
             'parity':[],'ms':{}}
    def keys(s1):
        kb=CP1SpatialPool16KeyBuilder(enabled_fields=['vision_0','vision_1','robot_state'])
        kb.collect(CheckpointID.CP1,stage1=s1)
        return kb.build(CheckpointID.CP1)
    def timing(fn):
        for _ in range(a.warmup): fn()
        torch.cuda.synchronize()
        vals=[]
        for _ in range(a.reps):
            b,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            b.record();out=fn();e.record();e.synchronize();vals.append(b.elapsed_time(e))
        return {'mean':float(np.mean(vals)),'median':float(np.median(vals)),'p10':float(np.percentile(vals,10)),
                'p90':float(np.percentile(vals,90)),'samples':vals}
    with torch.inference_mode():
        for suite in ('spatial','l10'):
            qc=QueryCell(a.root,f'pi05_{suite}_cache')
            eps=[e for e in qc.episodes if qc.tok_index[e['start']]>=0]
            for ei in (0,len(eps)//2):
                ep=eps[ei]
                for row in (ep['start'],min(ep['start']+1,ep['end']-1),ep['end']-1):
                    ti=int(qc.tok_index[row])
                    raw={'observation/image':np.asarray(qc.tok('img0')[ti]),'observation/wrist_image':np.asarray(qc.tok('img1')[ti]),
                         'observation/state':np.asarray(qc.raw_state[row],np.float64),'prompt':str(ep['task'])}
                    inp=policy._input_transform(raw)
                    obs=Observation.from_dict(jax.tree.map(lambda x:torch.as_tensor(np.array(x),device='cuda')[None],inp))
                    full=model.run_stage1(obs);dc=dummy.stage1(model,obs);wo=wrist.stage1(model,obs);complete=wrist.complete(model,wo)
                    kf,kd,kw=keys(full),keys(dc),keys(wo)
                    record={'suite':suite,'row':int(row),'step':int(qc.step[row]),'episode':ep['uid'],
                            'dummy_prefix_exact':torch.equal(full.prefix_embs,dc.prefix_embs),
                            'complete_prefix_exact':torch.equal(full.prefix_embs,complete.prefix_embs),
                            'dummy_keys_exact':all(torch.equal(kf[k],kd[k]) for k in kf),
                            'wrist_key_exact':torch.equal(kf['vision_1'],kw['vision_1']),
                            'stored_wrist_max_abs':float(np.max(np.abs(kw['vision_1'].cpu().numpy()-qc.key_v1[row])))}
                    noise=torch.randn((1,model.config.action_horizon,model.config.action_dim),device='cuda',generator=torch.Generator(device='cuda').manual_seed(17+row))
                    for K in (2,10):
                        base2=model.run_stage2(full)
                        ref=model.run_stage3(base2,noise=noise,num_steps=K,return_intermediates=True).action_chunk
                        for tag,x in [('dummy',dc),('wrist_complete',complete),('packed',dummy.pack(dc))]:
                            alt2=model.run_stage2(x);alt=model.run_stage3(alt2,noise=noise,num_steps=K,return_intermediates=True).action_chunk
                            record[f'{tag}_K{K}_exact']=torch.equal(ref,alt)
                            record[f'{tag}_K{K}_max_abs']=float((ref-alt).abs().max())
                            del alt2,alt
                        del base2,ref
                    print(json.dumps(record),flush=True);results['parity'].append(record)
        full=model.run_stage1(obs);dc=dummy.stage1(model,obs);wo=wrist.stage1(model,obs)
        packed=dummy.pack(dc)
        for tag,fn in [('stage1_full',lambda:model.run_stage1(obs)),('stage1_dummy_cached',lambda:dummy.stage1(model,obs)),
                       ('stage1_wrist_only',lambda:wrist.stage1(model,obs)),('stage1_wrist_completion',lambda:wrist.complete(model,wo)),
                       ('stage2_unpacked',lambda:model.run_stage2(dc)),('stage2_packed',lambda:model.run_stage2(packed))]:
            results['ms'][tag]=timing(fn);print(tag,results['ms'][tag]['median'],flush=True)
        s2=model.run_stage2(full)
        for K in (1,2,5,10):
            results['ms'][f'stage3_K{K}']=timing(lambda:model.run_stage3(s2,noise=noise,num_steps=K,return_intermediates=True))
            print('stage3',K,results['ms'][f'stage3_K{K}']['median'],flush=True)
        # Exercise the actual class hooks and coordinator tensor split/rebatch.
        from openpi.serving import stage_io
        wrist.install()
        a1=model.run_stage1(obs)
        b1=model.run_stage1(obs)
        shards=stage_io.split_stage1_output(stage_io.stack_stage1_output([b1,a1]),2)
        checks=[]
        for shard in shards:
            c=wrist.complete(model,shard.to('cuda'))
            checks.append(torch.equal(c.prefix_embs,full.prefix_embs))
            got=model.run_stage3(model.run_stage2(shard),noise=noise,num_steps=2,return_intermediates=True).action_chunk
            ref=model.run_stage3(s2,noise=noise,num_steps=2,return_intermediates=True).action_chunk
            checks.append(torch.equal(got,ref))
        results['installed_split_rebatch_parity']=checks
        assert all(checks)
        results['peak_allocated_bytes']=torch.cuda.max_memory_allocated();results['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    results['completed_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
    pathlib.Path(a.out).write_text(json.dumps(results,indent=2))
    required=['dummy_prefix_exact','complete_prefix_exact','dummy_keys_exact','wrist_key_exact','dummy_K2_exact','dummy_K10_exact','wrist_complete_K2_exact','wrist_complete_K10_exact']
    assert all(r[k] for r in results['parity'] for k in required)

if __name__=='__main__':
    try: main()
    except Exception as exc:
        if 'out of memory' in str(exc).lower():
            (HERE/'results/GPU_OOM_STOP').write_text(str(exc));raise SystemExit('OOM: GPU work stopped; no retries')
        raise
