"""Concrete per-model host entry. --run starts only this probe's residents."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',choices=['4090','h100','a100'],required=True)
    p.add_argument('--model',choices=['pi05','groot','dp','fastwam','cosmos'],required=True)
    p.add_argument('--deployment',choices=['canonical','fw4090_reduced','longseg_recal1018'],default='canonical')
    p.add_argument('--work')
    p.add_argument('--output')
    p.add_argument('--port',type=int,default=29600)
    p.add_argument('--run',action='store_true')
    p.add_argument('--capacity-arm',action='store_true',help='explicitly allow the documented nonnative replica topology')
    p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--deliveries',type=int,default=24)
    p.add_argument('--batch-telemetry',action='store_true')
    p.add_argument('--checkpoint-view',help='verified private pi05 H100 checkpoint; preserves frozen content identity')
    p.add_argument('--minutes',type=int,help='explicit watchdog override; A100 is dedicated')
    p.add_argument('--replicas',type=int,help='new capacity arm only')
    p.add_argument('--memory-gib',type=float,help='new allocator budget, never a silent change')
    p.add_argument('--free-gib',type=float,help='free-memory admission request')
    p.add_argument('--numerics',action='append',default=[],choices=['tf32_off','highest','sdpa_math','te_unfused','cosmos_math','compile_off','graphs_off','deterministic','full_reduction'])
    a=p.parse_args()
    w=Path(a.work or (Path(__file__).parent if a.host=='4090' else '/home/exouser/xgpu_1017')).resolve()
    tools=w if a.host=='4090' else w/'tools'
    if a.deployment=='fw4090_reduced' and a.model!='fastwam':p.error('FW deployment requires --model fastwam')
    if a.deployment=='longseg_recal1018' and a.model not in ('dp','groot','fastwam'):
        p.error('recalibrated long-segment profiles exist for DP/GR/FW only')
    bundle_name=('fastwam_4090_reduced' if a.deployment=='fw4090_reduced' else
        a.model+'_longseg_recal1018' if a.deployment=='longseg_recal1018' else a.model)
    bundle=w/'concurrent/bundles'/bundle_name
    c=json.loads((bundle/'source_campaign.json').read_text())
    if a.checkpoint_view and (a.host!='h100' or a.model!='pi05'):
        p.error('--checkpoint-view is only for the pi05 H100 relocation')
    if a.run and a.host=='h100' and a.model=='pi05':
        if not a.checkpoint_view:p.error('pi05 H100 requires --checkpoint-view after CPU staging/preflight')
        from pi05_checkpoint_view import verify
        verify(a.checkpoint_view,json.loads((bundle/'assets/model_manifest.json').read_text()))
    if a.host=='4090':
        python={'pi05':'/home/weiland/projects/openpi/.venv/bin/python','dp':'/home/weiland/projects/openpi/.venv/bin/python',
            'groot':'/home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/bin/python',
            'fastwam':'/data/mw_finetune/envs/fastwam/bin/python',
            'cosmos':'/home/weiland/projects/openpi_ext/lines/cosmos-policy/.venv/bin/python'}[a.model]
        capacities={'dp':(4,40,8,15),'pi05':(4,40,44,25),'groot':(6,42,42,25),
                    'fastwam':(2,54,47.5,30),'cosmos':(8,76,47,30)}
    elif a.host=='h100':
        python={'pi05':'/home/exouser/openpi/.venv/bin/python','dp':'/home/exouser/openpi/.venv/bin/python',
            'groot':'/home/exouser/gr00t_n15_venv/.venv/bin/python',
            'fastwam':'/data/mw_finetune/envs/fastwam/bin/python',
            'cosmos':'/data/cosmos/cosmos-policy/.venv/bin/python'}[a.model]
        capacities={'dp':(4,40,8,15),'pi05':(4,40,44,25),'groot':(11,76,76,25),
                    'fastwam':(2,70,70,30),'cosmos':(8,76,76,30)}
    else:
        python=str(w/'env/bin/python')
        capacities={'dp':(4,40,8,25),'pi05':(4,40,39.4,45),'groot':(5,36,36,30),
                    'fastwam':(1,36,36,35),'cosmos':(8,76,39.4,35)}
    replicas,budget,free,minutes=capacities[a.model]
    if a.deployment=='fw4090_reduced':
        replicas,budget,free,minutes=2,54,39.4 if a.host=='a100' else 42,35
    if a.replicas is not None:
        if a.replicas<1:p.error('positive replicas required')
        replicas=a.replicas
    if a.memory_gib is not None:budget=a.memory_gib
    if a.free_gib is not None:free=a.free_gib
    if min(budget,free)<=0:p.error('positive memory bounds required')
    if a.minutes is not None:
        if a.minutes<=0:p.error('positive watchdog required')
        minutes=a.minutes
    changed=replicas!=c['replicas'] or budget!=c['memory_budget_gib']
    if changed and not a.capacity_arm:
        p.error('this host cannot safely reproduce the selected source topology/budget; --capacity-arm labels a NEW topology, not deployment equivalence')
    inp='inputs_mt50_real3' if c['domain']=='metaworld' else 'inputs_cosmos3' if a.model=='cosmos' else 'inputs_dp3'
    label='' if a.deployment=='canonical' else a.deployment+'_'
    output=Path(a.output) if a.output else w/'results'/f'{a.host}_{a.model}_concurrent_{label}{time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())}'
    execution_tools=tools
    cache_before=None
    cache_relocations=[('/data/mw_finetune/eval',str(w/'rootfs/data/mw_finetune/eval'))] if a.host=='h100' else []
    if a.run:
        from cache_files import verify as verify_cache_files
        cache_before=verify_cache_files(bundle,cache_relocations)
        from harness_snapshot import seal
        execution_tools=seal(tools,w/'harnesses')
    argv=[sys.executable,str(execution_tools/'concurrent_server.py'),'--bundle',str(bundle),'--inputs',str(w/inp),
        '--output',str(output),'--python',python,'--mps-root',str(w),'--port',str(a.port),
        '--rounds',str(a.rounds),'--deliveries',str(a.deliveries),'--deadline-seconds',str(minutes*60),
        '--required-free-gib',str(free)]
    if a.deployment=='fw4090_reduced':argv+=['--clients','32']
    if changed:argv+=['--replicas',str(replicas),'--memory-gib',str(budget)]
    reference_name=a.model if a.deployment=='longseg_recal1018' else bundle_name
    reference=w/'concurrent/references'/(reference_name+'.json')
    if reference.exists():argv+=['--reference-attestation',str(reference)]
    if a.host=='h100':
        argv+=['--relocate','/data/mw_finetune/eval='+str(w/'rootfs/data/mw_finetune/eval')]
        if a.checkpoint_view:argv+=['--relocate',c['checkpoint']+'='+str(Path(a.checkpoint_view).resolve())]
        if a.model=='dp':
            argv+=['--graph-probe',str(w/'rootfs/data/mw_finetune/qualification/assets/dp_libero_gpu_fix1_v1/wire_observations.npz')]
    if a.host=='4090' and a.model=='groot':
        argv+=['--relocate','/home/exouser/gr00t_n15=/home/weiland/projects/openpi_ext/third_party/gr00t_n15']
        # The source is H100; make its pinned vendor imports find the local copy.
        runtime=c['runtime']
        argv+=['--env','PYTHONPATH='+':'.join([runtime+'/packages/openpi-client/src',runtime+'/src',runtime,
            '/home/weiland/projects/openpi_ext/third_party/gr00t_n15'])]
    if a.host=='4090' and a.model=='cosmos':
        argv+=['--relocate','/data/cosmos/cosmos-policy=/home/weiland/projects/openpi_ext/lines/cosmos-policy']
    if a.host=='a100' and a.model=='pi05':
        argv+=['--env','OPENPI_DATA_HOME='+str(w/'tokenizer')]
    if a.host=='a100':
        for key,relative in [('HF_MODULES_CACHE','model_cache/hf_modules'),('TORCH_HOME','model_cache/torch'),
            ('TORCHINDUCTOR_CACHE_DIR','model_cache/inductor'),('TRITON_CACHE_DIR','model_cache/triton'),
            ('XDG_CACHE_HOME','model_cache/xdg')]:
            argv+=['--env',key+'='+str(w/relative)]
        if a.model!='cosmos':argv+=['--env','HF_HOME='+str(w/'model_cache/hf')]
    for option in a.numerics:argv+=['--numerics',option]
    if a.batch_telemetry:argv.append('--batch-telemetry')
    if a.run:argv.append('--run')
    print(json.dumps(dict(host=a.host,model=a.model,deployment=a.deployment,free_gib=free,watchdog_minutes=minutes,
        topology_changed=replicas!=c['replicas'],allocation_budget_changed=budget!=c['memory_budget_gib'],argv=argv)),flush=True)
    code=subprocess.call(argv)
    if a.run and (output/'RUN.json').exists():
        audit=dict(before=cache_before,status='PASS')
        try:audit['after']=verify_cache_files(bundle,cache_relocations)
        except Exception as error:
            audit.update(status='FAILED',error=repr(error));code=code or 1
        with (output/'CACHE_FILES_VERIFICATION.json').open('x') as f:
            json.dump(audit,f,indent=2);f.write('\n')
        # Child cleanup is complete; preserve the exact observer/client source
        # identity alongside its run. Never overwrite an existing receipt.
        with (output/'EXECUTION_HARNESS.json').open('x') as f:
            json.dump(dict(snapshot=str(execution_tools),
                receipt=json.loads((execution_tools/'SNAPSHOT.json').read_text())),f,indent=2)
            f.write('\n')
    raise SystemExit(code)


if __name__=='__main__':main()
