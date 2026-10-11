import ast,json
from pathlib import Path
from datetime import datetime,timezone
from exp.r17_integration.ops import cache_study as study,cache_capacity as capacity,local_workers,worker_memory,worker_gpu
from exp.r17_integration.ops.formal_process import write_json
from exp.r17_integration.ops.formal_prestage import validate_prepared
from exp.r17_integration.ops.immutable_release import sha,verify,identity
from exp.r17_integration.ops.build_client_bundle import verified_members
from exp.r17_integration.ops.driver_release import planned_tasks,validate as validate_driver
from exp.r17_integration.ops.cache_control import check

lane=Path('/data/mw_finetune/eval/cszmdp'); plan=study.read(lane/'LANE_PLAN.json')
assert study.read(lane/'LANE_STATE.json')['status']=='PREPARED'
p=Path(plan['jobs'][0]['run']); snapshot=lane/'inputs_dp'
c,deployment=validate_prepared(p); capacity.validate_authorization(c); check(c); validate_driver(c,p)
assert c['execution_worker_nodes']=={'local':60} and c['execution_workers']==60 and c['replicas']==4
assert list(deployment)==['local'] and deployment['local']['workers']==60
assert all(not (p/n).exists() for n in ['LAUNCH.json','RECEIPT.json','OWNED.json'])
assert sha(Path(plan['after_lane']['directory'])/'LANE_PLAN.json')==plan['after_lane']['plan_sha256']
assert c['local_worker_exception']==local_workers.authorize(c,today='2026-10-13')
source=study.read(snapshot/'campaign.json')
native=[n for n in study.read(snapshot/'PREPARED.json')['files'] if n.startswith('assets/') or n in ('EPISODE_PLAN.json','RUNTIME_RELEASE.json','FROZEN_RUNTIME.json')]+['runtime.tar']
assert all(sha(p/n)==sha(snapshot/n) for n in native)
scientific=('checkpoint','mps','dp_batching','max_batch','batch_wait_ms','memory_budget_gib','worker_max_episodes','worker_nodes','workers','worker_memory_mib','worker_vram_mib')
assert all(c.get(k)==source.get(k) for k in scientific)
manifest=study.validate_manifest(plan['study']); audit=study.read(study.HERE/'evidence/sizes_cpu_audit.json')
assert manifest['library_sizes']==[50,100,200,300,400] and not manifest['artifacts_pending']
assert audit['status']=='PASS' and audit['study_digest']==manifest['digest']
assert sha(plan['study'])==plan['study_sha256']
ep=study.read(p/'EPISODE_PLAN.json'); generated=planned_tasks(c,p)
assert generated==ep['tasks'] and len(generated)==40000
print('Frozen CPU driver generated all 40000 original-A cells identically',flush=True)
frozen=Path(c['driver_runtime'])/'exp/r17_integration/ops/cache_study.py'
for name in ['plan','run_driver']:
 a=next(n for n in ast.parse(frozen.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
 b=next(n for n in ast.parse((study.ROOT/'exp/r17_integration/ops/cache_study.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
 assert ast.dump(a)==ast.dump(b)
assert "capacity=campaign.get('execution_workers',campaign['workers'])" in frozen.read_text()
worker=study.read(p/'local.json'); item=deployment['local']; payload,_=verified_members(p/'workers.tar','PAYLOAD.json')
installed=verify(item['payload'],'PAYLOAD.json',identity(payload)); assert installed['present']
ops=study.read(p/'LOCAL_WORKER_ORCHESTRATION.json')
assert all(payload['files'][n]==pin for n,pin in ops['unchanged'].items())
assert payload['files']['exp/r17_integration/ops/worker_memory.py']==sha(study.ROOT/'exp/r17_integration/ops/worker_memory.py')
assert worker['node_workers']==60 and worker['worker_memory_mib']==worker['worker_ram_reservation_mib']==3584
assert worker['worker_vram_mib']==717 and worker['worker_memory_headroom_mib']==8192
assert worker['policy_endpoint']=='127.0.0.1:23400' and worker['worker_egl_devices']==[0]
roster=study.read(p/'WORKER_ROSTER.json')['workers']
assert len(roster)==60 and len({x['worker_id'] for x in roster})==60 and {x['node'] for x in roster}=={'local'}
ram=study.read(p/'local_MEMORY_ADMISSION.json'); gpu=study.read(p/'local_GPU_ADMISSION.json'); disk=study.read(p/'local_DISK_ADMISSION.json')
assert ram['worker_memory_mib']==3584 and ram['headroom_bytes']==8192*1024**2 and disk['admitted']
idle=worker_gpu.admission(worker,Path(worker['worker_run']),observed=[dict(device=0,uuid='projected-local',total_mib=49140,used_mib=0,usable=True)])
# Current worker reservations also remain in this real-node projection. The
# mathematical upper bound removes those reservations, retaining GPU guards.
absolute_idle_slots=(49140-8192-4914)//717
assert absolute_idle_slots==50
record=dict(schema='r17.dp60_cpu_preparation.v1',status='PASS',generated_at=datetime.now(timezone.utc).isoformat(),cpu_only=True,
 launched=False,armed=False,coordinator_owns_process_control=True,lane=str(lane),run=str(p),fleet={'local':60},replicas=4,
 reservations=dict(ram_mib=3584,renderer_mib=717,renderer_gib_requested=0.7,global_ram_headroom_mib=8192,ram_accounting='sum(max(per-worker reservation, measured RSS)); charge only reservation shortfall beyond MemAvailable'),
 immutable_native_files_verified=len(native),native_worker_payload_files_verified=len(ops['unchanged']),original_A_episode_cells=40000,
 study_digest=manifest['digest'],study_sha256=sha(plan['study']),real_library_audit=dict(status='PASS',path=str(study.HERE/'evidence/sizes_cpu_audit.json'),sha256=sha(study.HERE/'evidence/sizes_cpu_audit.json'),configs=240,queries=960,reused_unchanged_scientific_inputs=True),
 prepared_sha256=sha(p/'PREPARED.json'),episode_plan_sha256=sha(p/'EPISODE_PLAN.json'),worker_roster_sha256=sha(p/'WORKER_ROSTER.json'),
 after_lane=plan['after_lane'],wait_for_pass=True,live_admission_preview=dict(memory=ram,gpu=gpu,disk=disk),
 gpu_remaining_limit=dict(physical_mib=49140,renderer_total_mib=60*717,server_growth_mib=8192,gpu_headroom_mib=4914,required_mib_even_without_current_use=60*717+8192+4914,absolute_idle_worker_upper_bound=absolute_idle_slots,sixty_can_admit=False),
 unchanged_Q_recycle_helper_pins={n:sha(study.ROOT/('exp/r17_integration/ops/'+n+'.py')) for n in ('formal_workers','timan1_workers','cache_capacity')},
 same_reservations_for_other_new_lanes=dict(ram_gib=3.5,renderer_gib=0.7,global_ram_headroom_gib=8),
 arm_command='nohup bash /home/weiland/projects/openpi/exp/r17_integration/ops/cache_study/sizes/run_4090.sh > /data/mw_finetune/eval/cszmdp/COORD_ARMED_60.log 2>&1 < /dev/null &')
write_json(study.HERE/'sizes/evidence/dp60_cpu_validation.json',record)
write_json(lane/'CPU_PREPARED_60.json',record)
print('DP60 CPU validation PASS; live RAM/GPU admission deferred; existing GPU guards upper bound50',flush=True)
