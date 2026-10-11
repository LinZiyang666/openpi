"""Concrete slot catalog and standalone executable wrappers (CPU only)."""
from __future__ import annotations
import json
import hashlib
import re
from pathlib import Path
from .assets import write_json


def require_owner_fix2(q):
    proofs={};missing=[]
    for number in range(1,5):
        path=q.parent/'handbacks'/f'HANDBACK_P{number}.md'
        content=path.read_text() if path.is_file() else ''
        if not re.search(r'^## Fix round 2(?:\s|$)',content,re.MULTILINE):
            missing.append(f'P{number}')
        else:
            proofs[f'P{number}']=dict(path=str(path),sha256=hashlib.sha256(content.encode()).hexdigest())
    if missing:
        raise RuntimeError('final slot regeneration waits for Fix round 2 handbacks: '+', '.join(missing))
    return proofs


def main():
    root=Path('/home/weiland/projects/openpi')
    q=root/'exp/r17_integration/qualification'
    owner_proofs=require_owner_fix2(q)
    review_proofs={name:dict(path=str(q.parent/'review'/name),
        sha256=hashlib.sha256((q.parent/'review'/name).read_bytes()).hexdigest())
        for name in ('REVIEW_A3.md','REVIEW_B3.md')}
    base={p.stem.removeprefix('asset_'):json.loads(p.read_text()) for p in (q/'specs').glob('asset_*.json')}
    for k in (2,1):
        key='groot_mw_k'+str(k)
        base[key]=dict(base['groot_mw'],asset_id=key,live_k=k,
            assets=base.get(key,{}).get('assets','/data/mw_finetune/qualification/assets/'+key))
        write_json(q/'specs'/('asset_'+key+'.json'),base[key])
    future={
        'groot_mw_final60000':('groot_mw','/data/mw_finetune/groot/4090_mt50/checkpoint-60000'),
        'fastwam_mw_final16001':('fastwam_mw','/data/mw_finetune/fastwam/mt50_frozen_b128/policy_016001'),
        'dp_mw_final100000':('dp_mw','/data/mw_finetune/dp/mt50_policy_100000')}
    for key,(origin,path) in future.items():
        base[key]=dict(base[origin],asset_id=key,checkpoint=path,future_checkpoint=True,
            assets=base[origin]['assets'] if origin=='dp_mw' else
                '/data/mw_finetune/qualification/assets/'+key+'_fix2_b3a3_v1')
        write_json(q/'specs'/('asset_'+key+'.json'),base[key])
    catalog=[]
    deployment=json.loads((q/'worker_deployment.json').read_text())
    def slot(name,key,phase,peak,minutes,**extra):
        job=dict(base[key],slot=name,phase=phase,expected_minutes=minutes,estimated_peak_mib=peak*1024,
            required_free_mib=(peak+4)*1024,teacher_store=base[key]['assets']+'/synthetic_teacher_store',
            library_artifact=base[key]['assets']+'/synthetic_library.pkl',fit_store=base[key]['assets']+'/synthetic_fit',
            synthetic_library=True,run_root='/data/mw_finetune/qualification/runs',deadline_seconds=890,
            port_wait_seconds=60,**extra)
        formal=20 if job['host']=='local' else 34
        job['control_python']='/home/weiland/projects/openpi/.venv/bin/python' if job['host']=='local' else '/home/exouser/openpi/.venv/bin/python'
        job['worker_python']='/scratch/zixuans8/mw_r17/envs/sim/bin/python'
        job['worker_payload']=deployment['nodes'][job['worker_node']]['destination']
        job['observations_dir']=job['assets']
        if job['domain']=='libero':
            job['simulator_node']='weilandserver'
            job['simulator_python']='/home/weiland/projects/openpi_ext/envs/libero_sim/bin/python'
            job['simulator_renderer']='osmesa'
        job['formal_eval_budget_mib']=formal*1024
        job['share_eval_safe']=phase!='batch_disabled' and peak+formal+6<=(48 if job['host']=='local' else 80)
        if phase=='batch_disabled':job['required_free_mib']=0
        if key.startswith('fastwam_libero'):
            job['share_eval_safe']=False  # Two-view activation budget has no measured GPU certificate yet.
        if phase=='replica':
            job['backend_ports']=[23160,23161] if job['host']=='local' else [23260,23261]
        criteria={
            'native':['owner native probe exits 0 and reports PASS/passed','checkpoint/norm/sampler manifest equals CPU seal',
                'π10/2/1 or DP100/2/(explicit labelled lossy1 study) or Fast10/2/1; GR fixed live K',
                'DP status pass, skipped0, passed equals all reported passing checks; no fixed check count',
                'upstream/staged normalization and finite native wire geometry',
                'debug on/off bytes exact; native suffix exact; recorded owner tolerance only for GPU arithmetic/B2'],
            'mechanical':['native server ready and closes/drains without forced kill','two actual WS clients: full/debug byte parity',
                'two hot full/retrieval bundles; unknown bundle and wrong norm rejected','forced synthetic library HIT has (1,0,0), NFE0',
                'exactly two accepted error-free real episodes (full index0 / retrieval index1)','MW exactly two attested workers, original reach-v3 kernel/seed7/160 controls',
                'MW driver bounded shutdown drain PASS; two worker exit codes0 with normal_exit, sealed snapshot and exactly two accepted error-free journal rows',
                'remote closed archive pulled from exact run-scoped allowed /tmp stage; SHA/P6 verified before exact owned source-stage cleanup; H100 relay push also stages in /tmp',
                'LIBERO original spatial task0 kernel/seed7/220 controls and two real views via OSMesa',
                'closed checksums and contiguous accepted decisions; actual applied control counts ≤5 per decision',
                'all server execution counts/cursors including final terminal ack match contiguous actual controls',
                'HTML/PNG/CSV/JSON reports; no quality or isolated pricing claim'],
            'g3':['9 actual GPU decisions on fixed inputs (recorded paired256 LIBERO, synthetic MW); byte parity','closed NPZ→H5 exact dtype/bytes',
                '3 complete fixed-input episodes→sealed library','library-only AWM/guard/LOEO corrector/periodic fit',
                'numeric reload action/donor/weights and replay diagnostics exact','new stock artifact/bundle forced HIT',
                'freeze/accepted identity/error audit and four report formats','prints PARTIAL even if subset PASS: full advanced/dev/test/semantic G3 pending'],
            'batch':['real native S3 B2 matches two single requests at recorded rtol=atol=1e-3',
                'per-request noise identical, output buffers independent','S1/S2 GR batching remains unsupported'],
            'replica':['two resident endpoints each attest exact sealed ModelSpec','stock sticky proxy routes two actual clients',
                'hot bundle broadcast, full/debug bytes and synthetic HIT pass','both owned servers close/drain; no throughput claim'],
            'tail':['five synthetic one-control acknowledgments consume original H5 without new model calls',
                'four suffixes byte-exact with (0,0,0), NFE0','prints PARTIAL: real separate replan1 physical protocol pending']}
        criteria['batch_disabled']=['CPU-only retired slot: Fast-WAM B>1 is disabled',
            'prints PARTIAL with receipt.gpu_executed=false; never starts a model or CUDA command']
        if job['model']=='pi05' and phase=='native':
            criteria['native'] += [
                'π contract r17.pi05.parity.r16_eager.v1: R16 eager normalized/wire, debug and suffix bytes exact',
                'K10 alternate stock staged bytes exact; K2/K1 compiled diagnostic admits only P2 sealed checkpoint/norm/archive profile',
                'compiled-only max/mean limits K2 .08/.02, K1 .13/.03, rtol0; no quality or general parity tolerance']
        if job['model']=='dp' and phase=='native':
            criteria['native'] += [
                'real_production_B1_only passes: max_batch_size1, B1 queue/direct bytes exact, all three production batch capabilities unsupported',
                'B2 composition measurements are labelled diagnostic_only and never count as production support or parity checks']
        job['pass_criteria']=criteria[phase]
        write_json(q/'specs'/(name+'.json'),job)
        script=f'''#!/usr/bin/env bash
# Generated concrete Q slot. CPU launcher; GPU admission happens inside run_slot.
set -uo pipefail
cd /home/weiland/projects/openpi || exit 1
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home/weiland/projects/openpi:/home/weiland/projects/openpi/src:/home/weiland/projects/openpi/packages/openpi-client/src
run_dir="/data/mw_finetune/qualification/runs/{name}_$(date +%Y%m%d_%H%M%S)_$$"
args=("$@")
for ((i=0; i<${{#args[@]}}; i++)); do
  if [[ "${{args[i]}}" == --run && $((i+1)) -lt ${{#args[@]}} ]]; then run_dir="${{args[i+1]}}"; fi
  if [[ "${{args[i]}}" == --run=* ]]; then run_dir="${{args[i]#--run=}}"; fi
done
/usr/bin/timeout --signal=TERM --kill-after=5s 885s /home/weiland/projects/openpi/.venv/bin/python -u -m exp.r17_integration.qualification.run_slot --slot /home/weiland/projects/openpi/exp/r17_integration/qualification/specs/{name}.json --run "$run_dir" "$@"
rc=$?
if [[ "$rc" == 124 || "$rc" == 137 || "$rc" == 143 ]]; then
  /home/weiland/projects/openpi/.venv/bin/python -m exp.r17_integration.qualification.timeout_receipt "$run_dir" '{name}'
  exit 2
fi
exit "$rc"
'''
        path=q/(name+'.sh');path.write_text(script);path.chmod(0o755)
        catalog.append(job)
    slot('slot_G1_pi05_mw_A','pi05_mw','native',12,'3–7')
    slot('slot_G1_pi05_mw_B','pi05_mw','mechanical',14,'4–10')
    slot('slot_G1_pi05_mw_C_tail','pi05_mw','tail',12,'3–6')
    for key,peak,minutes in [('groot_mw',18,'4–9'),('dp_mw',6,'1–4'),('dp_libero',6,'1–4'),
                              ('fastwam_libero',40,'7–14'),('fastwam_mw',32,'7–14')]:
        slot('slot_G2_'+key+'_A',key,'native',peak,minutes)
        slot('slot_G2_'+key+'_B',key,'mechanical',peak+4,'4–14')
    for k in (2,1): slot('slot_G2_groot_mw_A_k'+str(k),'groot_mw_k'+str(k),'native',18,'4–8')
    slot('slot_G2_groot_mw_C_B2','groot_mw','batch',22,'4–8')
    for key in ('fastwam_libero','fastwam_mw'):
        slot('slot_G2_'+key+'_C_B2',key,'batch_disabled',0,'CPU only (<1)',batch2=False)
    for key,peak in [('groot_mw',36),('dp_mw',12),('dp_libero',12),('fastwam_mw',68)]:
        slot('slot_G2_'+key+'_R_replicas',key,'replica',peak,'5–14')
    for key,peak in [('pi05_mw',14),('groot_mw',20),('dp_mw',8),('dp_libero',8),('fastwam_libero',44),('fastwam_mw',36)]:
        slot('slot_G3_'+key,key,'g3',peak,'5–14')
    for key in future:
        peak=18 if key.startswith('groot') else 32 if key.startswith('fastwam') else 6
        slot('slot_G2_'+key+'_A',key,'native',peak,'CPU seal ≤3 + native remainder ≤11')
        slot('slot_G2_'+key+'_B',key,'mechanical',peak+4,'CPU seal ≤3 + mechanics remainder ≤11')
    write_json(q/'SLOTS.json',dict(schema='r17.Q.slots.v1',slots=catalog,actual_gpu_executed=False,
        owner_fix2_handbacks=owner_proofs,final_review_proofs=review_proofs))
    print(f'PASS {len(catalog)} concrete bounded scripts')


if __name__=='__main__': main()
