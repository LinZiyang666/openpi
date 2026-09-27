"""Own pi05 SigLIP checkpoint only: CPU feasibility/parity probe and guarded GPU tower microbenchmark.
GPU run never starts unless nvidia-smi reports >=16 GiB free, allocator capped below 8 GiB.
This benchmarks vision + key pooling, NOT language/mask prep or full stage1 wall time.
"""
import os,sys,json,time,pathlib,argparse,subprocess
import numpy as np
OUT=pathlib.Path(__file__).resolve().parent
CKPT='/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/model.safetensors'
ROOT=pathlib.Path('/home/weiland/trace_runs/offline_search_store')

def gpu_check():
 p=subprocess.run(['nvidia-smi','--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True)
 if p.returncode:raise RuntimeError('GPU unavailable: '+p.stdout+p.stderr)
 free,total=map(int,p.stdout.splitlines()[0].split(','))
 if free<16384:raise RuntimeError(f'GPU not admitted: {free} MiB free < 16384')
 return free,total

def load(device='cpu',depth=27):
 import torch
 from safetensors import safe_open
 from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
 from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer
 cfg=SiglipVisionConfig(hidden_size=1152,intermediate_size=4304,num_hidden_layers=depth,num_attention_heads=16,image_size=224,patch_size=14,hidden_act='gelu_pytorch_tanh',vision_use_head=False)
 cfg._attn_implementation='sdpa'
 with torch.device('meta'):
  tower=SiglipVisionTransformer(cfg)
  proj=torch.nn.Linear(1152,2048)
 d={};pd={}
 with safe_open(CKPT,framework='pt',device='cpu') as f:
  for k in f.keys():
   pref='paligemma_with_expert.paligemma.model.vision_tower.vision_model.'
   pp='paligemma_with_expert.paligemma.model.multi_modal_projector.linear.'
   if k.startswith(pref):
    short=k[len(pref):]
    if short.startswith('encoder.layers.') and int(short.split('.')[2])>=depth:continue
    d[short]=f.get_tensor(k)
   elif k.startswith(pp):pd[k[len(pp):]]=f.get_tensor(k)
 tower.load_state_dict(d,assign=True,strict=True);proj.load_state_dict(pd,assign=True,strict=True)
 tower.embeddings.position_ids=torch.arange(256).expand((1,-1))
 if device=='cpu':tower=tower.float();proj=proj.float()
 else:
  tower=tower.to(dtype=torch.bfloat16);proj=proj.to(dtype=torch.bfloat16)
  tower.embeddings=tower.embeddings.float()
 tower=tower.to(device).eval();proj=proj.to(device).eval()
 return tower,proj

def encode(tower,proj,ims,depth=27,res=224,pool=4,project=True):
 import torch
 from torch.nn import functional as F
 device=next(tower.parameters()).device
 x=torch.as_tensor(np.asarray(ims).copy(),device=device).permute(0,3,1,2).float()/127.5-1
 if res!=224:x=F.interpolate(x,size=(res,res),mode='bilinear',align_corners=False,antialias=True)
 h=tower.embeddings(x,interpolate_pos_encoding=res!=224)
 h=h.to(tower.encoder.layers[0].self_attn.q_proj.weight.dtype)
 for layer in tower.encoder.layers[:depth]:h=layer(h,attention_mask=None)[0]
 if depth==27:
  h=tower.post_layernorm(h)
  if project:h=proj(h)
 g=res//14
 return F.adaptive_avg_pool2d(h.reshape(len(ims),g,g,-1).permute(0,3,1,2).float(),(pool,pool)).permute(0,2,3,1).flatten(1)

def main():
 p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=p.parse_args()
 meta={'device':args.device,'checkpoint':CKPT,'affinity':sorted(os.sched_getaffinity(0))}
 if args.device=='cuda':
  try:free,total=gpu_check();meta['admission_free_MiB']=free
  except RuntimeError as e:
   meta['status']='blocked';meta['reason']=str(e);(OUT/'vision_gpu.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta));return
 import torch
 torch.set_num_threads(1)
 if args.device=='cuda':torch.cuda.set_per_process_memory_fraction(7.5*1024/total)
 t,pr=load(args.device)
 d=ROOT/'queries/pi05_spatial_cache'
 im=np.load(d/'tok/img0.npy',mmap_mode='r');rows=np.load(d/'tok/rows.npy');key=np.load(d/'key_v0.npy',mmap_mode='r')
 variants=[('full27_224',27,224,4,True),('early6_224',6,224,4,False),('early12_224',12,224,4,False),('early18_224',18,224,4,False),('full27_168',27,168,4,True),('full27_112',27,112,4,True),('pool2',27,224,2,True),('pool1',27,224,1,True)]
 out=[]
 with torch.inference_mode():
  for name,depth,res,pool,proj in variants:
   vals=[]
   for rep in range(3):
    st=time.perf_counter();v=encode(t,pr,im[rep:rep+1],depth,res,pool,proj)
    if args.device=='cuda':torch.cuda.synchronize()
    vals.append(1000*(time.perf_counter()-st))
   rec={'name':name,'ms':vals,'num_features':v.shape[1]}
   if name=='full27_224':
    v=encode(t,pr,im[:1]).float().cpu().numpy()[0];ref=key[rows[0]]
    rec['stored_key_relative_RMS']=float(np.sqrt(np.mean((v-ref)**2)/np.mean(ref**2)))
    rec['stored_key_cosine']=float(v@ref/(np.linalg.norm(v)*np.linalg.norm(ref)))
   out.append(rec);print(rec,flush=True)
 meta.update(status='ok',results=out,notes='CPU FP32 timing is feasibility only; GPU eager sample timing is NOT a CUDA-graph stage1 measurement.')
 if args.device=='cuda':meta['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
 (OUT/f'vision_{args.device}.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':
 try:main()
 except Exception as e:
  if 'out of memory' in str(e).lower():
   (OUT/'GPU_OOM_STOP').write_text(str(e));raise SystemExit('OOM: stopped GPU work; no retry allowed.')
  raise
