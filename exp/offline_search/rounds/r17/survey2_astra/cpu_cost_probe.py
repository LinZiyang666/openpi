"""Synthetic CPU proxy: released metadata and tensor-header shapes; no pretrained weights.
Upstream DP TransformerForDiffusion is used verbatim. FiLM vision is reconstructed,
so these timings are not a benchmark of the released checkpoint or GPU estimates.
"""
import pathlib,sys,time,json,statistics,platform,math
O=pathlib.Path(__file__).parent
sys.path.insert(0,str(O/'code'/'real-stanford__diffusion_policy'))
import torch
from torch import nn
from torchvision.models import resnet18
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(0)
class Vision(nn.Module):
 def __init__(self):
  super().__init__();r=resnet18(weights=None,norm_layer=lambda n:nn.GroupNorm(n//16,n));self.stem=nn.Sequential(r.conv1,r.bn1,r.relu,r.maxpool);self.blocks=nn.ModuleList([b for l in [r.layer1,r.layer2,r.layer3,r.layer4] for b in l]);self.film=nn.ModuleList([nn.Linear(768,2*c) for c in [64,64,128,128,256,256,512,512]]);self.poolconv=nn.Conv2d(512,32,1);self.fc=nn.Linear(64,64)
 def forward(self,x,lang):
  x=self.stem(x)
  for block,film in zip(self.blocks,self.film):
   x=block(x);g,b=film(lang).chunk(2,-1);x=x*(1+g[:,:,None,None])+b[:,:,None,None]
  x=self.poolconv(x).flatten(2).softmax(-1);pos=torch.linspace(-1,1,3);px=pos.repeat(3);py=pos.repeat_interleave(3);return self.fc(torch.cat([(x*px).sum(-1),(x*py).sum(-1)],-1))
vision=nn.ModuleList([Vision(),Vision()]).eval();im=torch.rand(1,3,84,84);lang=torch.rand(1,768)
model=TransformerForDiffusion(input_dim=7,output_dim=7,horizon=16,n_obs_steps=1,cond_dim=904,n_layer=12,n_head=8,n_emb=512,p_drop_emb=0.,p_drop_attn=.3,causal_attn=True,time_as_cond=True,obs_as_cond=True,n_cond_layers=4).eval()
header=json.loads((O/'raw'/'brandon_libero_safetensors_header.json').read_text());expected={k.removeprefix('_policy.model.'):v['shape'] for k,v in header.items() if k.startswith('_policy.model.')};actual={k:list(v.shape) for k,v in model.state_dict().items()};mismatch={k:[expected.get(k),actual.get(k)] for k in set(expected)|set(actual) if expected.get(k)!=actual.get(k)}
scheduler=DDPMScheduler(num_train_timesteps=100,beta_schedule='squaredcos_cap_v2',beta_start=.0001,beta_end=.02,variance_type='fixed_small',prediction_type='epsilon',clip_sample=True)
cond=torch.rand(1,1,904);x=torch.rand(1,16,7)
def enc():return torch.cat([v(im,lang) for v in vision],-1)
def head(steps):
 scheduler.set_timesteps(steps);y=torch.randn_like(x)
 for t in scheduler.timesteps:y=scheduler.step(model(y,t,cond),t,y).prev_sample
 return y
keys=torch.nn.functional.normalize(torch.rand(5000,128),dim=-1);vals=torch.rand(5000,16,7);query=torch.nn.functional.normalize(torch.rand(128),dim=-1)
def lookup():return vals[torch.argmax(keys@query)]
def measure(fn,n):
 for _ in range(2):fn()
 ts=[]
 for _ in range(n):
  t=time.perf_counter();fn();ts.append((time.perf_counter()-t)*1000)
 return {'median_ms':statistics.median(ts),'min_ms':min(ts),'max_ms':max(ts),'n':n,'samples_ms':ts}
with torch.inference_mode():
 result={'cpu':platform.processor(),'threads':4,'torch':torch.__version__,'dtype':'float32','batch':1,'weights':'random; no learned weights downloaded','model_shape_mismatches':mismatch,'model_parameters':sum(p.numel() for p in model.parameters()),'vision_parameters':sum(p.numel() for p in vision.parameters()),'vision_2cam':measure(enc,10),'lookup_5000x128':measure(lookup,30),'denoiser_1':measure(lambda:model(x,torch.tensor(50),cond),10),'head_100':measure(lambda:head(100),3),'head_2':measure(lambda:head(2),10),'head_1':measure(lambda:head(1),10)}
 # FLOPs for multiply-add dense operations, custom vision proxy + exact head.
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as prof:enc()
 result['vision_flops']=sum(e.flops for e in prof.key_averages())
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as prof:model(x,torch.tensor(50),cond)
 result['denoiser_flops_profiler_partial']=sum(e.flops for e in prof.key_averages())
(O/'raw'/'cpu_cost_proxy.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
