"""CPU-only, minimal one-model rsync allowlists; no model or simulator starts."""
import argparse
import importlib.metadata as md
import json
from pathlib import Path
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from a100_closedloop_prepare_1018 import HERE,OUT,REMOTE,write

LOCAL=Path('/data/mw_finetune/qualification/local_workers')
SIMDEPS=LOCAL/'dependencies/libero311_42f0e80603054ed9f4bab84a6ce71bdbc71958c6c5c0329f9b5c892d917b3fe9'
SIMASSETS=LOCAL/'assets/libero_3eb6e3e7ee3a03663dba84ccc6d6cf9dbb4ae21fb20558ca4097abc066be57dc/assets'


def prepare(model):
 root=OUT/model;dest=REMOTE/'models'/model/'rootfs';files=set();symlinks=set()
 def add(p):
  p=Path(p)
  if p.suffix in ('.h5','.hdf5','.mp4','.tar') or '__pycache__' in p.parts or p.suffix=='.pyc':return
  if p.is_symlink():symlinks.add(p)
  elif p.is_file():files.add(p)
 def tree(p):
  p=Path(p)
  if not p.exists():raise ValueError('required local resource missing: '+str(p))
  for f in ([p] if p.is_file() else p.rglob('*')):
   if '.git' not in f.parts and '.cache' not in f.parts and '.venv' not in f.parts:add(f)
 def prior(name,predicate=lambda p:True):
  for line in (HERE/'concurrent'/name).read_text().splitlines():
   p=Path(line)
   if predicate(p):add(p)
 mappings={'/data':str(dest/'data')};python='3.11';worker_extra=[]
 if model=='pi05':return
 island='groot_island_complete' if model=='groot' else 'fastwam_island_complete' if model in ('dp','fastwam') else 'cosmos_island_complete'
 idoc=json.loads((HERE/'concurrent'/island/'ISLAND.json').read_text());site=Path(idoc['site'])
 # Reuse the previous successfully loaded, same-version serving island list.
 # All paths are exact regular wheel files; no general pip/HF cache transfer.
 for line in (HERE/'concurrent'/island/'site.files').read_text().splitlines():add(line)
 for p in site.glob('websockets*'):
  for f in p.rglob('*') if p.is_dir() else [p]:add(f)
 if model=='groot':
  prior('groot_data.files',lambda p:str(p).startswith('/data/mw_finetune/groot/4090_mt50/checkpoint-60000/'))
  vendor=Path('/home/weiland/projects/openpi_ext/third_party/gr00t_n15');tree(vendor)
  remote_vendor=str(dest/str(vendor).lstrip('/'))
  mappings.update({'/home/exouser/gr00t_n15':remote_vendor,str(vendor):remote_vendor})
 elif model=='dp':
  python='3.12';tree('/data/mw_finetune/dp/libero_policy_100000');tree('/data/mw_finetune/dp/clip_text')
  tree('/data/mw_finetune/qualification/observations/dp_libero_v2')
 elif model=='fastwam':
  python='3.12'
  prior('fastwam_data.files',lambda p:str(p).startswith(('/data/mw_finetune/models/fastwam_libero/','/data/xwam/')))
  tree('/data/mw_finetune/vendor/FastWAM/src');tree('/data/mw_finetune/vendor/lerobot/src')
 elif model=='cosmos':
  python='3.10'
  prior('cosmos_data.files',lambda p:str(p).startswith(('/data/cosmos/hf_home/','/data/mw_finetune/models/cosmos_libero/','/data/mw_finetune/qualification/observations/')))
  vendor=Path('/home/weiland/projects/openpi_ext/lines/cosmos-policy')
  # Frozen vendor source allowlist excludes its model/environment/download trees.
  for line in (HERE/'concurrent/cosmos_vendor.include').read_text().splitlines():
   if not line.endswith('/'):add(vendor/line.lstrip('/'))
  mappings['/data/cosmos/cosmos-policy']=str(dest/str(vendor).lstrip('/'))
  for directory in (HERE/'concurrent').glob('cosmos_*'):
   listing=directory/'site.files'
   if listing.exists():
    for line in listing.read_text().splitlines():add(line)
  # A few native dynamic registration dependencies reside in the installed
  # serving environment but were absent from its first import census.
  for name in ('ftfy','webdataset','decord','wandb','multistorageclient'):
   for f in (site/name).rglob('*') if (site/name).is_dir() else []:add(f)
 if model in ('dp','fastwam','cosmos'):
  # Preserve the exact installed LIBERO Python source, BDDL/init pools, visual
  # assets and worker wheel versions used by the existing local formal fleet.
  tree(SIMDEPS);tree(SIMASSETS)
  pending=['mujoco','gymnasium','PyOpenGL','glfw','numba','lxml','h5py','opencv-python',
           'pillow','imageio','imageio-ffmpeg','scipy','cloudpickle','tyro','websockets','msgpack','msgpack-numpy']
  seen=set();worker_sites=set()
  while pending:
   name=canonicalize_name(pending.pop())
   if name in seen:continue
   seen.add(name)
   try:dist=md.distribution(name)
   except md.PackageNotFoundError:raise ValueError('worker dependency not installed: '+name)
   base=Path(dist.locate_file('')).resolve();worker_sites.add(str(dest/str(base).lstrip('/')))
   for f in dist.files or []:
    p=Path(dist.locate_file(f))
    if p.is_file() and p.resolve().is_relative_to(base):add(p)
   for requirement in dist.requires or []:
    req=Requirement(requirement)
    if req.marker is None or req.marker.evaluate({'extra':''}):pending.append(req.name)
  remote_deps=str(dest/str(SIMDEPS).lstrip('/'));remote_assets=str(dest/str(SIMASSETS).lstrip('/'))
  worker_extra=[remote_deps,*sorted(worker_sites)]
  config=dict(assets=remote_assets,benchmark_root=remote_deps+'/libero/libero',
       bddl_files=remote_deps+'/libero/libero/bddl_files',init_states=remote_deps+'/libero/libero/init_files',datasets=str(REMOTE/'unused_datasets'))
  import yaml
  private=root/'libero_config';private.mkdir(exist_ok=True);(private/'config.yaml').write_text(yaml.safe_dump(config))
  worker=dict(python=str(REMOTE/'liberoenv/bin/python'),extra_paths=worker_extra,env=dict(
      LIBERO_CONFIG_PATH=str(REMOTE/model/'libero_config'),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
      MUJOCO_EGL_DEVICE_ID='0',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',OMP_NUM_THREADS='1',
      OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
 else:
  worker=dict(python=str(REMOTE/'simenv/bin/python'),extra_paths=[],env=dict(
      MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0',OMP_NUM_THREADS='1',
      OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
 write(root/'WORKER.json',worker);write(root/'MODEL_PATHS.json',mappings)
 # Native cache files use exact read aliases, so original YAML bytes and
 # scientific identities remain intact across both old and current recipes.
 for arm in ('pure','cache'):
  d=root/arm;c=json.loads((d/'campaign_template.json').read_text())
  c['server_python']=str(REMOTE/'models'/model/'env/bin/python');write(d/'campaign_template.json',c)
  dep=json.loads((d/'DEPENDENCIES.json').read_text())
  write(d/'READMAP.json',{old:str(REMOTE/'deps'/old.lstrip('/')) for old in dep})
 # Do not follow the installed LIBERO assets symlink into a second copy.
 # Its absolute source link is replaced by a link within this new A100 tree.
 symlinks={p for p in symlinks if p!=SIMDEPS/'libero/libero/assets'}
 listing=root/'BULK.files';listing.write_text(''.join(str(p).lstrip('/')+'\n' for p in sorted(files|symlinks)))
 sizes={str(p):p.stat().st_size for p in files}
 spec=dict(model=model,python_version=python,interpreter='/home/exouser/xgpu_1017/'+
       ('python' if python=='3.11' else 'python312' if python=='3.12' else 'python310')+'/bin/python'+python,
       site=str(dest/str(site).lstrip('/')),destination=str(dest),
       files=len(files),symlinks=len(symlinks),bytes=sum(sizes.values()),
       bulk_list=str(listing),model_paths=mappings,worker=worker,
       libero_asset_link=str(dest/str(SIMDEPS/'libero/libero/assets').lstrip('/')) if model!='groot' else None,
       libero_assets=str(dest/str(SIMASSETS).lstrip('/')) if model!='groot' else None)
 write(root/'RESOURCES.json',spec)
 print(json.dumps({k:spec[k] for k in ('model','files','bytes','python_version')}))
 return spec


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('model',choices=['groot','dp','fastwam','cosmos']);a=p.parse_args();prepare(a.model)
