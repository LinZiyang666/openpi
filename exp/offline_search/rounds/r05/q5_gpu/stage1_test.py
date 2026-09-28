"""Real STAGE1_ONLY pi05 -> real orchestrator -> final plugin. No network/server."""
import argparse,os
os.environ['JAX_PLATFORMS']='cpu'
from common import *
from gpu_replay import Watch
import torch
import jax
from openpi.models.model import Observation
from openpi.models_pytorch.stage_device_placement import StageDeviceConfig,relocate_model_stages
from openpi.cache.components.key_builder import CP1SpatialPool16KeyBuilder
from openpi.cache.types import CheckpointID
from openpi.cache.orchestrator import CacheOrchestrator
import openpi.cache.config as cc

def main():
    p=argparse.ArgumentParser();p.add_argument('--scale',type=int,required=True);p.add_argument('--mode',required=True);a=p.parse_args()
    c=config(f'pi05_l10_{a.scale}_AWM');outdir=B/f'results/stage1_{a.scale}_{a.mode}';outdir.mkdir(exist_ok=True)
    for old in outdir.glob('decisions_*.jsonl'):old.unlink()
    opts,_=plugin.parse_cli(['--os-method',c['spec'],'--os-kwargs',json.dumps(c['kwargs']),'--os-cell',c['cell'],
         '--os-root',str(STORE),'--os-log-dir',str(outdir),'--os-tag','s1','--os-fit-artifact',c['path'],
         '--os-no-shadow-native','--os-tokens','off','--os-gpu-retrieval',a.mode])
    rt=plugin.PluginRuntime(opts,'pi05')
    saved=torch.load('/tmp/k9_scratch/stage1_policy.pt',map_location='cpu',weights_only=False)
    checkpoint='/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch'
    assert saved['checkpoint']==checkpoint
    assert saved['cache_key']==(os.stat(checkpoint+'/model.safetensors').st_size,os.stat(checkpoint+'/model.safetensors').st_mtime_ns)
    model=saved['model'].eval();transform=saved['input_transform'];watch=Watch()
    torch.set_float32_matmul_precision('high')  # real model constructor's setting
    relocate_model_stages(model,StageDeviceConfig.create('cuda:0','meta','meta'))
    cfg=cc.load_cache_config('exp/trace_dual/config/tr_pi05_l10_cache.yaml');shared=cc.build_shared_storage(cfg)
    qc=store.QueryCell(STORE,'pi05_l10_cache');checks=[]
    ref,_=plugin.clone_method(rt.method,strict=True)
    class ModelPolicy:
        def __init__(self,orch,session):self.orch=orch;self.s=session
        def on_episode_start(self,task='',extra_metadata=None,**kwargs):
            self.orch.on_episode_start(task_key=task,episode_id='stage_test',extra_metadata=extra_metadata)
        def on_episode_end(self,success):self.orch.on_episode_end()
        def infer(self,raw):
            inputs=transform(dict(raw));obs=Observation.from_dict(jax.tree.map(lambda x:torch.as_tensor(np.array(x),device='cuda')[None],inputs))
            with torch.inference_mode():stage=model.run_stage1(obs)
            response=self.orch.check(CheckpointID.CP1,stage1=stage)
            assert response.hit_type.name=='FULL_HIT'
            chunk=response.payload.action_chunk
            # Independent deployed builder D2H happens only in this test, after graph final copies.
            kb=CP1SpatialPool16KeyBuilder(enabled_fields=['vision_0','vision_1','robot_state'])
            kb.collect(CheckpointID.CP1,stage1=stage);cpu_keys=kb.build(CheckpointID.CP1)
            keys=response.query_keys
            assert all(np.array_equal(keys[n].cpu().float().numpy(),cpu_keys[n].numpy()) for n in keys)
            s=self.s;q=plugin.OnlineQueryView(s,s.step-1,s.ep.task_id,s.ep)
            if q.step==0:ref.reset(s.ep)
            if a.mode=='shadow':assert chunk.numpy().tobytes()==ref.query(q).action.tobytes()
            else:assert np.isnan(q.key_v0).all()
            checks.append(dict(task=q.task_id,step=q.step,pool_exact=True,**s._dec['gpu_retrieval']))
            self.orch.broadcast_action(chunk);self.orch.clear()
            return {'actions':chunk.numpy().copy()}
    def factory(_,bundle):
        comps=rt.attach(cc.build_per_connection_components(cfg,shared,quiet=True),cfg)
        session=plugin._TLS.new_sessions[-1]
        orch=CacheOrchestrator(storage=comps['storage'],key_builder=comps['key_builder'],gates=comps['gates'],
              judges=comps['judges'],search_strategies=comps['search_strategies'],timer=comps['timer'],
              write_policy=comps.get('write_policy'),offline_writers=comps.get('offline_writers',()),library_stats=comps.get('library_stats'))
        return ModelPolicy(orch,session)
    conn=plugin._wrap_factory(factory)(None,'stage_test')
    for task in (0,4,9):
        e=next(e for e in qc.episodes if e['task_id']==task and qc.tok_index[e['start']]>=0)
        conn.on_episode_start(task=e['task'],extra_metadata={'task_id':task,'task_uid':e['uid']})
        for row in range(e['start'],e['start']+2):
            ti=int(qc.tok_index[row]);assert ti>=0
            raw={'observation/image':np.array(qc.tok('img0')[ti]),'observation/wrist_image':np.array(qc.tok('img1')[ti]),
                 'observation/state':np.array(qc.raw_state[row],np.float64),'prompt':str(e['task'])}
            conn.infer(raw)
        conn.on_episode_end(True)
    record=dict(PASS=True,scale=a.scale,mode=a.mode,checkpoint=checkpoint,stage_placement='cuda/meta/meta',
                real_orchestrator=True,observations=6,checks=checks,gpu=watch.finish(),source_sha256=hashlib.sha256((B/'dev/gpu_retrieval.py').read_bytes()).hexdigest())
    assert torch.backends.cuda.matmul.allow_tf32
    dump(outdir/'report.json',record);print(json.dumps({k:v for k,v in record.items() if k!='gpu'}))
if __name__=='__main__':main()
