"""Isolated real pi05 stage1 + deployed pool + retrieval, one config/process.

No server, hooks, compiler cache or policy path changes. CPU input transforms and
H2D of the input observation are outside timing in both designs. Pool in the
original stage dtype, then float() on GPU: preserves the deployed bf16 rounding.
"""
import argparse, dataclasses, gc, time
from common import *
import torch
from gpu_awm import GPUAWM, verdict_packet
from benchmark import inputs, capture, timing

class KeyView:
    def __init__(self,q,keys):self.q=q;self.keys=keys
    def __getattr__(self,name):
        if name in self.keys:return self.keys[name]
        return getattr(self.q,name)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run',required=True)
    p.add_argument('--reps',type=int,default=40);a=p.parse_args()
    cfg=next(c for c in configs() if c['id']==a.config);suite=cfg['cell'].split('_')[1]
    m=load(cfg['path']);b=getattr(m,'base',m)
    lib=store.LibraryView(STORE,f'pi05_{suite}',b.cand_name)
    mod=GPUAWM(m,lib).eval()
    seq=sequences(suite,episodes_per_task=1,max_steps=4);ep,qs=seq[0];q=qs[0]
    qc=store.QueryCell(STORE,f'pi05_{suite}_cache');e=next(e for e in qc.episodes if e['uid']==ep.uid)
    row=e['start'];ti=int(qc.tok_index[row]);assert ti>=0
    watch=GPUWatch(f'stage_{a.config}_{a.run}')
    # JAX is confined to CPU by JAX_PLATFORMS=cpu in every driver invocation.
    import jax
    from openpi.training import config
    from openpi.policies import policy_config
    from openpi.models.model import Observation
    from openpi.models_pytorch.stage_device_placement import StageDeviceConfig,relocate_model_stages
    from openpi.cache.components.key_builder import CP1SpatialPool16KeyBuilder,_spatial_pool_tokens
    from openpi.cache.types import CheckpointID
    from stage1_adapter import Stage1Adapter
    pcfg=config.get_config('pi05_libero')
    pcfg=dataclasses.replace(pcfg,model=dataclasses.replace(pcfg.model,pytorch_compile_mode=None))
    ckpt='/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch'
    cache=pathlib.Path('/tmp/k9_scratch/stage1_policy.pt')
    cache_key=(os.stat(ckpt+'/model.safetensors').st_size,os.stat(ckpt+'/model.safetensors').st_mtime_ns)
    if cache.exists():
        saved=torch.load(cache,map_location='cpu',weights_only=False)
        assert saved['checkpoint']==ckpt and saved['cache_key']==cache_key
        model,input_transform=saved['model'],saved['input_transform']
    else:
        policy=policy_config.create_trained_policy(pcfg,ckpt,pytorch_device='cpu')
        model=policy._model.eval();input_transform=policy._input_transform
        relocate_model_stages(model,StageDeviceConfig.create('cpu','meta','meta'))
        torch.save(dict(model=model,input_transform=input_transform,checkpoint=ckpt,cache_key=cache_key),cache)
    print('loaded exact stage1 weights; cache',str(cache),flush=True)
    watch.check(admit=True)
    relocate_model_stages(model,StageDeviceConfig.create('cuda:0','meta','meta'))
    adapter=Stage1Adapter(model).eval()
    gc.collect()
    raw={'observation/image':np.asarray(qc.tok('img0')[ti]),'observation/wrist_image':np.asarray(qc.tok('img1')[ti]),
         'observation/state':np.asarray(qc.raw_state[row],np.float64),'prompt':str(ep.task)}
    inp=input_transform(raw)
    obs=Observation.from_dict(jax.tree.map(lambda x:torch.as_tensor(np.array(x),device='cuda')[None],inp))
    mod.cuda();ii=inputs(q)
    result=dict(config=cfg,run=a.run,checkpoint=ckpt,row=row,uid=ep.uid,step=0,batch=1,
                torch=torch.__version__,cuda=torch.version.cuda,command=sys.argv,
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.py')},
                gpu_stage_placement='s1 CUDA, unused s2/s3 meta; same loaded stage1 weights',
                input_h2d_bytes=sum(x.numel()*x.element_size() for x in jax.tree.leaves(obs.to_dict()) if isinstance(x,torch.Tensor)),
                latency={},parity=[])
    kb=CP1SpatialPool16KeyBuilder(enabled_fields=['vision_0','vision_1','robot_state'])
    def pool(s):
        pref=s.prefix_embs[0]
        return (_spatial_pool_tokens(pref[:256],16,4),_spatial_pool_tokens(pref[256:512],16,4))
    def retrieve(s):
        v0,v1=pool(s)
        return mod(v0.float()[None],v1.float()[None],s.state,*ii[3:])
    def cpu_path(s):
        kb.collect(CheckpointID.CP1,stage1=s);keys=kb.build(CheckpointID.CP1)
        qq=KeyView(q,dict(key_v0=keys['vision_0'].numpy(),key_v1=keys['vision_1'].numpy(),rs=keys['robot_state'].numpy()))
        m.reset(ep)
        return m.query(qq)
    def packed(s):
        out=retrieve(s)
        return out['action'],verdict_packet(out)
    with torch.inference_mode():
        # Same stage output verifies the key boundary before timing it.
        s=model.run_stage1(obs);v=pool(s);ref=cpu_path(s);gpu=retrieve(s)
        kb.collect(CheckpointID.CP1,stage1=s);keys=kb.build(CheckpointID.CP1)
        result['pool_dtype']=str(v[0].dtype);result['prefix_shape']=list(s.prefix_embs.shape)
        result['bytes']=dict(current_keys_d2h=sum(x.numel()*x.element_size() for x in v)+s.state.numel()*s.state.element_size(),
                             host_keys_materialized=sum(keys[k].numel()*keys[k].element_size() for k in keys),
                             gpu_output_d2h=gpu['action'].numel()*gpu['action'].element_size()+verdict_packet(gpu).numel()*8,
                             possible_action_h2d_beyond_timed_boundary=ref.action.nbytes,
                             new_regime_h2d_min=3*8,
                             image_state_prompt_h2d_common=result['input_h2d_bytes'])
        result['pool_matches_cpu_keybuilder']=all(torch.equal(v[i].cpu().float(),keys[f'vision_{i}']) for i in range(2))
        result['parity'].append(dict(kind='same_eager_stage_output',top1=bool(ref.topk[0]==int(gpu['topk'][0,0])),
                                    order=np.array_equal(ref.topk,gpu['topk'][0].cpu().numpy()),
                                    action_max_abs=float(np.max(np.abs(ref.action-gpu['action'][0].cpu().numpy()))),
                                    confidence_abs=abs(ref.confidence-float(gpu['confidence'][0]))))
        del s,v,ref,gpu,keys;gc.collect();torch.cuda.empty_cache()
        g1,s1=capture(lambda:adapter(obs),watch)
        # Stage-only graph and combined graph measured from identical resident inputs.
        gr,grout=capture(lambda:retrieve(s1),watch)
        host_action=torch.empty_like(grout['action'],device='cpu',pin_memory=True)
        packet=verdict_packet(grout)
        host_packet=torch.empty_like(packet,device='cpu',pin_memory=True)
        del packet
        def final_copy(action,packet):
            host_action.copy_(action,non_blocking=True);host_packet.copy_(packet,non_blocking=True)
        gcmb,combined=capture(lambda:packed(adapter(obs)),watch)
        def gpu_eager():final_copy(*packed(model.run_stage1(obs)))
        def stage_graph_retrieval_eager():g1.replay();final_copy(*packed(s1))
        def separate_graphs():g1.replay();gr.replay();final_copy(grout['action'],verdict_packet(grout))
        def combined_graph():gcmb.replay();final_copy(*combined)
        fns=[('stage1_eager',lambda:model.run_stage1(obs)),('stage1_graph',g1.replay),
             ('today_eager_stage_cpu',lambda:cpu_path(model.run_stage1(obs))),
             ('today_graph_stage_cpu',lambda:(g1.replay(),cpu_path(s1))),
             ('gpu_all_eager',gpu_eager),('gpu_graph_stage_eager_retrieval',stage_graph_retrieval_eager),
             ('gpu_separate_graphs',separate_graphs),('gpu_combined_graph',combined_graph)]
        for name,fn in fns:
            result['latency'][name]=timing(fn,watch,reps=a.reps,warm=5)
            print(a.config,a.run,name,result['latency'][name]['event']['p50'],result['latency'][name]['wall']['p50'],flush=True)
        # Alternate-order paired increments reduce drift, but retain both signed samples.
        pairs=[]
        st,en=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        for i in range(a.reps):
            values={}
            for name,fn in ([('stage',g1.replay),('combined',gcmb.replay)] if i%2==0 else [('combined',gcmb.replay),('stage',g1.replay)]):
                t=time.perf_counter_ns();st.record();fn();en.record();en.synchronize()
                values[name]=dict(event=st.elapsed_time(en),wall=(time.perf_counter_ns()-t)/1e6)
            pairs.append(values)
        result['paired_no_d2h']=dict(samples=pairs,event_increment=stats([r['combined']['event']-r['stage']['event'] for r in pairs]),
                                    wall_increment=stats([r['combined']['wall']-r['stage']['wall'] for r in pairs]))
        # Replay graph with six distinct real images across three tasks, same tensor shapes.
        for task in (0,4,9):
            e=next(e for e in qc.episodes if e['task_id']==task and qc.tok_index[e['start']]>=0)
            for rr in (e['start'],min(e['start']+1,e['end']-1)):
                ti=int(qc.tok_index[rr]);assert ti>=0
                raw={'observation/image':np.asarray(qc.tok('img0')[ti]),'observation/wrist_image':np.asarray(qc.tok('img1')[ti]),
                     'observation/state':np.asarray(qc.raw_state[rr],np.float64),'prompt':str(e['task'])}
                new=input_transform(raw)
                for old,x in zip(jax.tree.leaves(obs.to_dict()),jax.tree.leaves(Observation.from_dict(jax.tree.map(lambda x:torch.as_tensor(np.array(x),device='cuda')[None],new)).to_dict())):
                    if isinstance(old,torch.Tensor):old.copy_(x)
                ii[3].fill_(task)
                g1.replay();expected=model.run_stage1(obs)
                prefix_diff=float((s1.prefix_embs.float()-expected.prefix_embs.float()).abs().max())
                all_stage_fields_exact=all(torch.equal(getattr(s1,n),getattr(expected,n)) for n in
                    ('state','prefix_embs','prefix_pad_masks','prefix_att_2d_masks_4d','prefix_position_ids'))
                rr_eager=packed(expected);gcmb.replay();torch.cuda.synchronize()
                result['parity'].append(dict(kind='changed_image_task_replay',task=task,row=int(rr),prefix_max_abs=prefix_diff,
                                             all_stage_fields_exact=all_stage_fields_exact,
                                             graph_eager_action_max_abs=float((combined[0]-rr_eager[0]).abs().max()),
                                             graph_eager_packet_max_abs=float((combined[1]-rr_eager[1]).abs().max())))
        result['gpu']=watch.finish()
    dump(OUT/f'stage_{a.config}_r{a.run}.json',result)

if __name__=='__main__':main()
