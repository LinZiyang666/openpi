"""Real STAGE1_ONLY pi05 through final ResidentKeys and plugin. No network/server."""
import argparse,os,time
os.environ['JAX_PLATFORMS']='cpu'
from common import *
from gpu_replay import Watch
import torch
import jax
from openpi.models.model import Observation
from openpi.models_pytorch.stage_device_placement import StageDeviceConfig,relocate_model_stages
from openpi.cache.components.key_builder import CP1SpatialPool16KeyBuilder
from openpi.cache.types import CheckpointID
from exp.offline_search.rounds.r05.q5_gpu.dev import gpu_retrieval as bridge

def main():
    p=argparse.ArgumentParser();p.add_argument('--scale',type=int,required=True);p.add_argument('--mode',required=True);a=p.parse_args()
    c=config(f'pi05_l10_{a.scale}_AWM');outdir=B/f'results/stage1_{a.scale}_{a.mode}';outdir.mkdir(exist_ok=True)
    opts,_=plugin.parse_cli(['--os-method',c['spec'],'--os-kwargs',json.dumps(c['kwargs']),'--os-cell',c['cell'],
         '--os-root',str(STORE),'--os-log-dir',str(outdir),'--os-tag','s1','--os-fit-artifact',c['path'],
         '--os-no-shadow-native','--os-tokens','off','--os-gpu-retrieval',a.mode])
    rt=plugin.PluginRuntime(opts,'pi05')
    saved=torch.load('/tmp/k9_scratch/stage1_policy.pt',map_location='cpu',weights_only=False)
    checkpoint='/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch'
    assert saved['checkpoint']==checkpoint
    assert saved['cache_key']==(os.stat(checkpoint+'/model.safetensors').st_size,os.stat(checkpoint+'/model.safetensors').st_mtime_ns)
    model=saved['model'].eval();transform=saved['input_transform'];watch=Watch()
    # Loading the cached object skips __init__; reproduce the real constructor's setting.
    torch.set_float32_matmul_precision('high')
    relocate_model_stages(model,StageDeviceConfig.create('cuda:0','meta','meta'))
    graph=rt.gpu.connection()
    native=SimpleNamespace();storage=SimpleNamespace()
    kb=CP1SpatialPool16KeyBuilder(enabled_fields=['vision_0','vision_1','robot_state'])
    s=plugin.PluginSession(rt,native,storage,kb,'stage_test')
    resident=bridge.ResidentKeys(kb,s,graph);s.kb=resident
    qc=store.QueryCell(STORE,'pi05_l10_cache');checks=[]
    for task in (0,4,9):
        e=next(e for e in qc.episodes if e['task_id']==task and qc.tok_index[e['start']]>=0)
        s.client_episode_start(dict(task=e['task'],extra_metadata={'task_id':task,'task_uid':e['uid']}))
        ref,_=plugin.clone_method(rt.method,strict=True)
        for row in range(e['start'],e['start']+2):
            ti=int(qc.tok_index[row]);assert ti>=0
            raw={'observation/image':np.array(qc.tok('img0')[ti]),'observation/wrist_image':np.array(qc.tok('img1')[ti]),
                 'observation/state':np.array(qc.raw_state[row],np.float64),'prompt':str(e['task'])}
            inputs=transform(dict(raw));obs=Observation.from_dict(jax.tree.map(lambda x:torch.as_tensor(np.array(x),device='cuda')[None],inputs))
            s.set_obs(raw)
            with torch.inference_mode():stage=model.run_stage1(obs)
            resident.collect(CheckpointID.CP1,stage1=stage)
            keys=resident.build(CheckpointID.CP1)
            # Verify deployed bf16 pooling exactly, separate reference D2H after the GPU packet.
            cpu_keys=kb.build(CheckpointID.CP1)
            assert all(np.array_equal(keys[n].cpu().float().numpy(),cpu_keys[n].numpy()) for n in keys)
            ctx=SimpleNamespace(query_keys=keys,current_step=s.step,task_key=e['task'],checkpoint_id=CheckpointID.CP1)
            s.on_search(ctx)
            q=plugin.OnlineQueryView(s,s.step-1,task,s.ep)
            if q.step==0:ref.reset(s.ep)
            if a.mode=='shadow':
                expected=ref.query(q)
                assert s._dec['served'].tobytes()==expected.action.tobytes()
            else:assert np.isnan(q.key_v0).all()
            checks.append(dict(task=task,step=q.step,pool_exact=True,**s._dec['gpu_retrieval']))
            s.on_executed(s._dec['served']);s.after_infer(0.,True);resident.clear()
        s.client_episode_end(True)
    record=dict(PASS=True,scale=a.scale,mode=a.mode,checkpoint=checkpoint,stage_placement='cuda/meta/meta',
                observations=6,checks=checks,gpu=watch.finish(),source_sha256=hashlib.sha256((B/'dev/gpu_retrieval.py').read_bytes()).hexdigest())
    assert torch.backends.cuda.matmul.allow_tf32
    dump(outdir/'report.json',record);print(json.dumps({k:v for k,v in record.items() if k!='gpu'}))
if __name__=='__main__':main()
