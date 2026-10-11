import os,time,json,hashlib,numpy as np,pathlib
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and os.environ['MUJOCO_GL']=='osmesa'
from robocasa.utils.env_utils import create_env

def rollout(sparse):
 env=create_env(env_name='OpenDrawer',split=None,obj_instance_split='target',layout_ids=[1],style_ids=[1],camera_widths=128,camera_heights=128,seed=17)
 env.reset();image_keys=[k for k in env._observables if k.endswith('_image')];out=[]
 render_flag=[True]
 if sparse:
  for key in image_keys:
   observable=env._observables[key];original=observable._sensor
   def sensor(cache, original=original, key=key):
    return original(cache) if render_flag[0] else cache.get(key,np.zeros((128,128,3),dtype=np.uint8))
   sensor.__modality__=original.__modality__
   observable._sensor=sensor
 t=time.perf_counter()
 for i in range(24):
  render=not sparse or i%8>=6
  render_flag[0]=render
  action=np.zeros(env.action_dim);action[0]=0.002*np.sin(i);action[1]=0.002*np.cos(i);action[6]=-1
  obs,r,d,info=env.step(action)
  if i%8>=6:
   out.append({'step':i+1,'images':{k:hashlib.sha256(obs[k].tobytes()).hexdigest() for k in image_keys},'qpos':env.sim.data.qpos.copy().tolist(),'qvel':env.sim.data.qvel.copy().tolist(),'reward':r,'done':d})
 elapsed=time.perf_counter()-t;env.close();return out,elapsed
full,tf=rollout(False);sparse,ts=rollout(True)
r={'task':'OpenDrawer','seed':17,'size':128,'steps':24,'query_period':8,'history_frames':2,'full_s':tf,'sparse_s':ts,'exact_equal':full==sparse,'sampled_steps':[r['step'] for r in full],'compared_images':sum(len(r['images']) for r in full),'full':full,'sparse':sparse}
p=pathlib.Path(__file__).parent/'rc_render_parity_v3.json';p.write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k not in ['full','sparse']})
