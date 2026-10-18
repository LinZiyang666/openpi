"""Freeze a UID subset of an existing formal plan, never regenerate its namespace."""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time

HERE=Path(__file__).resolve().parent
OUT=HERE/'a100_closedloop_1018'
REMOTE=Path('/home/exouser/xgpu_1017/cl18')
SOURCES={
 'pi05':dict(pure='tpi7_0536',cache='pimk1_fuld',cache_file='cache_yamls/n250_mt50_crg_k50.yaml',
             replicas=2,memory=20,batch=8,domain='metaworld',reference_segments=['pimk1_ful','pimk1_fulb','pimk1_fulc4','pimk1_fuld']),
 'groot':dict(pure='tgr7h1_full',cache='ff1015_groot_mt_10',cache_file='cache_yamls/n1000_mt50_c.yaml',
              replicas=4,memory=28,batch=1,domain='metaworld'),
 'dp':dict(pure='hs7dpctl_full',cache='ff1015_dp_li_14_u',cache_file='cache_yamls/n50_libero_spatial_c.yaml',
           replicas=4,memory=40,batch=8,domain='libero'),
 'fastwam':dict(pure='hs6fw_res_152931',cache='ff1015_fastwam_li_18',cache_file='cache_yamls/n50_libero_spatial_c.yaml',
                replicas=1,memory=35,batch=16,domain='libero'),
 'cosmos':dict(pure='hs2_cosmos_h100_k2incl_c16_t107_20261010_115120',cache='fa_co_r_u_l',cache_file='cache_yamls/n50_sp_crg_k50.yaml',
               replicas=2,memory=19,batch=1,domain='libero'),
}


def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(p,v):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')


def prepare(model,arm):
 cfg=SOURCES[model];source=Path('/data/mw_finetune/eval')/cfg[arm]
 target=OUT/model/arm;target.mkdir(parents=True,exist_ok=False)
 c=json.loads((source/'campaign.json').read_text())
 original_assets=Path(c['local_assets']);assets=target/'assets';assets.mkdir()
 for name in ('benchmark.json','model_spec.json','model_manifest.json','manifest.json','load.json','full.yaml',
              'CACHE_STUDY.json','ffk_bootstrap.py','execution_sampler.json'):
  p=original_assets/name
  if p.is_file():shutil.copyfile(p,assets/name)
 if (source/'SERVER_ATTESTATIONS.json').exists():
  shutil.copyfile(source/'SERVER_ATTESTATIONS.json',target/'SOURCE_ATTESTATIONS.json')
 group=None
 if arm=='cache':
  seal=json.loads((assets/'CACHE_STUDY.json').read_text())
  matches=[g for g in seal['groups'] if g['file']==cfg['cache_file']]
  if len(matches)!=1:raise ValueError('one preregistered original group required')
  group=matches[0];p=assets/group['file'];p.parent.mkdir(parents=True,exist_ok=True)
  shutil.copyfile(original_assets/group['file'],p)
 runtime=Path(c['runtime'])
 legacy_files=None
 if not runtime.is_dir():
  archive=source/'runtime.tar';identity=sha(archive)
  if not runtime.name.endswith(identity):raise ValueError('legacy runtime archive identity mismatch')
  restored=OUT/'legacy_runtimes'/runtime.name;restored.mkdir(parents=True,exist_ok=True)
  legacy_files={}
  with tarfile.open(archive) as tf:
   for m in tf.getmembers():
    rel=Path(m.name)
    if not m.isfile() or rel.is_absolute() or '..' in rel.parts:raise ValueError('unsafe legacy runtime archive')
    raw=tf.extractfile(m).read();q=restored/rel;q.parent.mkdir(parents=True,exist_ok=True)
    if q.exists() and q.read_bytes()!=raw:raise ValueError('legacy runtime restore changed')
    if not q.exists():q.write_bytes(raw)
    legacy_files[m.name]=hashlib.sha256(raw).hexdigest()
  runtime=restored;c['runtime']=str(runtime)
  write(target/'LEGACY_RUNTIME_FILES.json',dict(archive_sha256=identity,files=legacy_files))
 sys.path[:0]=[str(runtime),str(runtime/'src'),str(runtime/'packages/openpi-client/src')]
 # The process has one frozen source runtime only. Preserve campaign/plan names,
 # original A50 benchmark, group identities, num_trials and all seed rules.
 c={k:v for k,v in c.items() if not k.startswith(('cache_resume','resume_'))}
 c['local_assets']=str(assets);c['assets']=str(assets)
 from exp.r17_integration.ops import formal_driver
 if arm=='cache':
  from exp.r17_integration.ops import cache_study
  native_groups=cache_study.selected_groups
  def groups(c):
   s,gs=native_groups(c)
   selected=[g for g in gs if g['bundle_id']==group['bundle_id']]
   if len(selected)!=1:raise ValueError('group identity changed')
   return s,selected
  cache_study.selected_groups=groups
 benchmark,spec,tasks,endpoints=formal_driver.plan(c)
 limit=2 if cfg['domain']=='metaworld' else 10
 selected=[t for t in tasks if t.orig_init_state_idx<limit and
           (cfg['domain']=='metaworld' or t.experiment=='libero_spatial')]
 serial=[asdict(t) for t in selected]
 expected=98 if model=='groot' and arm=='cache' else 100
 if len(serial)!=expected or len({t.task_uid for t in selected})!=expected:
  raise ValueError(f'fixed {expected}-episode population missing: {len(serial)}')
 ep=source/('FULL_EPISODE_PLAN.json' if (source/'FULL_EPISODE_PLAN.json').exists() else 'EPISODE_PLAN.json')
 before=Path('/proc/pressure/io').read_text()
 def task_key(t):return tuple(t[k] for k in ('experiment','task_id','orig_init_state_idx','episode_idx','phase','bundle_id'))
 original_by_key={task_key(t):t for t in json.loads(ep.read_text())['tasks']}
 mapped=[];namespace=[]
 for task in selected:
  t=asdict(task);old=original_by_key.get(task_key(t))
  ignored=('server_host','server_port','task_uid','yaml_id')
  if old is None or {k:v for k,v in old.items() if k not in ignored}!={k:v for k,v in t.items() if k not in ignored}:
   raise ValueError('native planner changed scientific fields')
  if old['task_uid']!=t['task_uid']:
   namespace.append(dict(native_uid=t['task_uid'],original_uid=old['task_uid'],
       reason='source coordinator and frozen planner use different hash namespaces; scientific fields equal'))
  mapped.append(replace(task,task_uid=old['task_uid'],yaml_id=old['yaml_id']))
 selected=mapped;serial=[asdict(t) for t in selected]
 write(target/'ORIGINAL_NAMESPACE.json',namespace)
 # Freeze this selection BEFORE inspecting historical outcomes.
 write(target/'PLAN.json',dict(source=str(source),tasks=serial,expected_episodes=expected,
     benchmark_protocol=benchmark.identity,model=spec,cache_enabled=arm=='cache'))
 write(target/'UIDS.json',[t.task_uid for t in selected])
 write(target/'source_campaign.json',json.loads((source/'campaign.json').read_text()))
 local_plan={t.task_uid:asdict(t) for t in selected}
 # Cross-check the sealed source plan, using the full preregistered population
 # when the accepted source is a formal continuation.
 original={t['task_uid']:t for t in original_by_key.values() if t['task_uid'] in local_plan}
 for t in serial:
  old=original.get(t['task_uid'])
  if old is None or {k:v for k,v in old.items() if k not in ('server_host','server_port')}!={k:v for k,v in t.items() if k not in ('server_host','server_port')}:
   raise ValueError('native planner no longer matches sealed source task '+t['task_uid'])
 # Sequential readers; exact original UIDs only. Copied resume rows resolve by
 # earliest accepted timestamp, the existing formal provenance rule.
 wanted={t.task_uid for t in selected};accepted={};origins={};audits=[]
 segments=cfg.get('reference_segments',[cfg[arm]]) if arm=='cache' else [cfg[arm]]
 for name in segments:
  journal=source.parent/name/'driver/journal.jsonl';h=hashlib.sha256();initial=journal.stat()
  with journal.open('rb') as f:
   for line in f:
    h.update(line);r=json.loads(line)
    if r.get('accepted') is not True or r['task_uid'] not in wanted:continue
    if r.get('error'):raise ValueError('selected reference has unresolved infrastructure error')
    uid=r['task_uid']
    if uid not in accepted or r['ts']<accepted[uid]['ts']:
     accepted[uid]=r;origins[uid]=name
  if initial.st_size!=journal.stat().st_size:raise ValueError('reference journal is still appending')
  audits.append(dict(file=str(journal),bytes=initial.st_size,sha256=h.hexdigest()))
 if set(accepted)!=wanted:raise ValueError(f'fixed source subset has {len(wanted-set(accepted))} missing accepted results')
 write(target/'REFERENCE.json',dict(outcomes=accepted,origins=origins,journals=audits))
 dependencies=set()
 if group:
  import yaml
  recipe=yaml.safe_load((assets/group['file']).read_text())['recipe']
  for key in ('fitted_path','stall_path'):
   if recipe.get(key):
    path=Path(recipe[key]).resolve(strict=True)
    for f in ([path] if path.is_file() else path.rglob('*')):
     if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc':
      if f.suffix in ('.h5','.hdf5','.mp4','.tar'):raise ValueError('no bulk teacher data')
      dependencies.add(f)
  store=Path(recipe['store_path'])
  while (store/'VIEW.json').exists():
   dependencies.update([(store/'VIEW.json').resolve(),(store/'row_indices.npy').resolve()])
   view=json.loads((store/'VIEW.json').read_text());store=(store/view['parent_path']).resolve()
  dependencies.add((store/'READY.json').resolve())
 write(target/'DEPENDENCIES.json',{str(f):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(dependencies)})
 runtime_identity=(identity if legacy_files is not None else c.get('runtime_identity') or json.loads((source/'RUNTIME_RELEASE.json').read_text())['identity'])
 rp=REMOTE/model/arm
 c.update(assets=str(rp/'assets'),local_assets=str(rp/'assets'),
          runtime=str(REMOTE/'runtimes'/runtime.name),coordinator_runtime=str(REMOTE/'runtimes'/runtime.name),
          server_node='local',server_python='/home/exouser/xgpu_1017/env/bin/python',
          replicas=cfg['replicas'],memory_budget_gib=cfg['memory'],workers=8,execution_workers=8,
          worker_nodes={'a100':8},execution_worker_nodes={'a100':8},live_fleets=False,
          endpoint_host='127.0.0.1',backend_host='127.0.0.1',endpoint_port=29700,public_port=29700,
          endpoint='127.0.0.1:29700',backend_ports=list(range(29701,29701+cfg['replicas'])),
          driver_port=29730,driver_host='127.0.0.1',seconds=14400,retries=2,
          xgpu_subset=str(rp/'PLAN.json'),xgpu_group=group,xgpu_arm=arm)
 write(target/'campaign_template.json',c)
 write(target/'PREREGISTRATION.json',dict(created=time.time(),model=model,arm=arm,episodes=expected,
     cold_starts=2,margin=.02,confidence=.95,selection='all tasks, original init 0..'+str(limit-1),
     original_uid_namespace=True,capacity=dict(replicas=cfg['replicas'],max_batch=cfg['batch'],
       allocator_gib=cfg['memory'],local_workers=8,renderer='A100 EGL'),
     source=str(source),runtime_identity=runtime_identity,group=group,
     runtime_kind='legacy_archive' if legacy_files is not None else 'manifest',
     runtime_source=str(runtime),plan_sha256=sha(target/'PLAN.json'),
     source_receipt_status=json.loads((source/'RECEIPT.json').read_text()).get('status') if (source/'RECEIPT.json').exists() else None,
     io_before_plan=before,io_after=Path('/proc/pressure/io').read_text(),
     limitations=['100 pairs cannot certify ±2pp even with zero discordance',
       'A100-local worker placement and explicit capacity arm differ from reference; comparison is deployment-level']))
 print(json.dumps(dict(model=model,arm=arm,episodes=expected,source=str(source),runtime=runtime_identity)))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('model',choices=list(SOURCES));p.add_argument('arm',choices=['pure','cache'])
 a=p.parse_args();prepare(a.model,a.arm)
