"""Bounded vision-only numerical mismatch measurement, no serving/rollouts.

Uses paired batch-two / batch-one passes on the SAME reconstructed B images.
56px recordings cannot reproduce original pixels. The resulting perturbations
are a sensitivity test, not exact one-camera deployment keys.
"""
from .dev_validation import *
import subprocess

def main():
    dest=HERE/'directive2'/'groot_mismatch';dest.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('TMPDIR',str(dest));os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(dest/'torch'))
    # Required workspace interpreter, deployed GR00T torch/FA2/transformers
    # packages together (their C++ ABIs must match). No installs or writes.
    sys.path.insert(0,'/home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/lib/python3.11/site-packages')
    import torch
    from transformers import SiglipVisionConfig, SiglipVisionModel
    from safetensors import safe_open
    from PIL import Image
    status=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip()
    free,total=map(int,status.split(','));assert free>=5000,status
    # Reserve at most 4 GiB for tensors; CUDA context leaves total far below 10 GB.
    torch.cuda.set_per_process_memory_fraction(4*1024/total)
    modeldir=Path('/home/weiland/projects/openpi_ext/third_party/gr00t_n15/gr00t/model/backbone/eagle2_hg_model')
    cfg=json.loads((modeldir/'config.json').read_text());vc=SiglipVisionConfig(**cfg['vision_config'])
    vc._attn_implementation='flash_attention_2'
    torch.manual_seed(0)
    vision=SiglipVisionModel(vc).eval()
    connector=torch.nn.Linear(1152,2048).eval()
    checkpoint=Path('/data/ckpt/n15_libero_10');index=json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    for part,mod in [('vision_model',vision),('mlp1',torch.nn.Sequential(connector))]:
        prefix=f'backbone.eagle_model.{part}.';state={}
        for name in sorted(set(index[k] for k in index if k.startswith(prefix))):
            with safe_open(checkpoint/name,framework='pt',device='cpu') as f:
                for k in index:
                    if k.startswith(prefix) and index[k]==name:state[k[len(prefix):]]=f.get_tensor(k)
        mod.load_state_dict(state,strict=True)
    vision=vision.to('cuda');connector=connector.to('cuda')
    def enc(x):
        with torch.inference_mode(),torch.autocast('cuda',torch.bfloat16):
            return connector(vision(pixel_values=x,return_dict=True).last_hidden_state).float()
    def pool(x):
        return x.reshape(-1,4,4,4,4,2048).mean((2,4)).reshape(-1,32768)
    bases={}
    for size in (50,400):
        with np.load(REPO/f'exp/offline_search/rounds/r10/pca/groot_l10/{size}/v1.npz') as z:bases[size]=z['basis'].astype(np.float32)
    deltas={50:[],400:[]};metadata=[];raw=[]
    for root in ('r12_devlooks_50L','r12_devlooks_400'):
        arms=json.loads((RUNS/root/'arms.json').read_text())
        arm=next(a for a in arms if a['model']=='groot');run=RUNS/root/'runs'/arm['arm']
        counts={};total_kept=0
        for directory in sorted(run.glob('server_*/looks')):
            for look in read_looks(directory):
                t=int(look['task_id']);s=int(look['step'])
                # Fixed task/step quotas, chosen before encoder results.
                if counts.get(t,0)>=8 or s%8!=0:continue
                counts[t]=counts.get(t,0)+1
                images=[]
                for cam in (0,1):
                    a=Image.fromarray(look[f'img{cam}']).resize((256,256),Image.Resampling.BILINEAR)
                    a=a.crop((6,6,249,249)).resize((224,224),Image.Resampling.BILINEAR)
                    images.append(np.asarray(a).copy())
                x=torch.from_numpy(np.stack(images)).permute(0,3,1,2).to('cuda',torch.float32)/127.5-1
                full=enc(x);half=enc(x[1:]);delta=(pool(half)-pool(full[1:])).cpu().numpy()[0]
                for size,B in bases.items():deltas[size].append(delta@B)
                raw.append(dict(token_max=float((half-full[1:]).abs().max()),token_rms=float((half-full[1:]).square().mean().sqrt()),key_rms=float(np.sqrt(np.mean(delta**2)))))
                metadata.append(dict(root=root,uid=look['meta']['uid'],step=s,task=t,init=int(look['init'])))
                total_kept+=1
                if total_kept%20==0:print(root,total_kept,raw[-1],flush=True)
                if len(counts)==10 and min(counts.values())>=8:break
        assert total_kept==80,(root,counts)
    np.savez_compressed(dest/'deltas.npz',**{f'P{size}':np.array(a) for size,a in deltas.items()})
    dump(dest/'summary.json',dict(timestamp=now(),count=len(raw),gpu_before=status,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        torch=torch.__version__,attention=vc._attn_implementation,checkpoint=str(checkpoint),
        raw_summary={k:dict(mean=float(np.mean([r[k] for r in raw])),max=float(max(r[k] for r in raw))) for k in raw[0]},
        projection={str(size):dict(rms=float(np.sqrt(np.mean(np.array(a)**2))),max=float(np.max(np.abs(a)))) for size,a in deltas.items()},
        raw=raw,observations=metadata,caveat=__doc__))
    dump(dest/'reads.json',sorted(READS))
    print('done peak MiB',torch.cuda.max_memory_reserved()/2**20,flush=True)

if __name__=='__main__':main()
