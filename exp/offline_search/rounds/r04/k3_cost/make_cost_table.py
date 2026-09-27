"""Publish measured stage costs in K4's documented schema (atomic replacement)."""
import hashlib,json,os,pathlib,time
HERE=pathlib.Path(__file__).resolve().parent
REPO=HERE.parents[4]
DEST=REPO/'exp/offline_search/closed_loop/ops/cost_table.json'

def read(name):return json.loads((HERE/'results'/name).read_text())
def stages(d,k):
    ms={n:v['median'] for n,v in d['ms'].items()}
    full={'s1':ms['stage1_full'],'s2':ms['stage2_unpacked'],'s3':ms[f'stage3_K{k}'],'k':k}
    total=sum(full[x] for x in ('s1','s2','s3'))
    return {'full':full,'modes':{'full':dict(full)},'full_cost_ms':total,
            'shares':{x:full[x]/total for x in ('s1','s2','s3')},'stage3_per_step_ms':full['s3']/k,
            'measured_stage3_ms_by_k':{n.removeprefix('stage3_K'):v for n,v in ms.items() if n.startswith('stage3_K')},
            'measurement':d}

def main():
    pi=stages(read('pi05.json'),10);gsp=stages(read('groot_spatial.json'),8);gl10=stages(read('groot_l10.json'),8)
    pd=pi['measurement'];ms={n:v['median'] for n,v in pd['ms'].items()}
    for mode in ('dummy_cached','wrist_only'):
        pi['modes'][mode]={**pi['full'],'s1':ms['stage1_'+mode],'miss_s1_extra':ms['stage1_wrist_completion'] if mode=='wrist_only' else 0.0}
        pi['modes'][mode]['s1_share_of_full']=pi['modes'][mode]['s1']/pi['full_cost_ms']
    pi['prefix_packing']={'deployable':False,'stage2_ms':ms['stage2_packed'],'prefix_tokens_before':968,'prefix_tokens_after':712,
                          'reason':'bf16 same-noise action parity failed; --os-pack-prefix is refused',
                          'max_action_abs_K2':max(r['packed_K2_max_abs'] for r in pd['parity']),
                          'max_action_abs_K10':max(r['packed_K10_max_abs'] for r in pd['parity'])}
    # The flat GR00T entry averages the two measured suite shapes; retain each
    # separately so a future suite-aware ledger can select instead of averaging.
    gf={k:(gsp['full'][k]+gl10['full'][k])/2 for k in ('s1','s2','s3')};gf['k']=8
    gt=sum(gf[k] for k in ('s1','s2','s3'))
    groot={'full':gf,'modes':{'full':dict(gf)},'full_cost_ms':gt,'shares':{k:gf[k]/gt for k in ('s1','s2','s3')},
           'stage3_per_step_ms':gf['s3']/8,'aggregation':'arithmetic mean of spatial and l10 warmed per-stage medians',
           'by_suite':{'spatial':gsp,'l10':gl10}}
    sources=['src/openpi/models_pytorch/pi0_pytorch.py','src/openpi/serving/batching_coordinator.py',
             'scripts/serve_policy.py','src/openpi/cache/groot/staged.py','exp/libero_groot/serve_groot_libero.py',
             'exp/offline_search/closed_loop/stage_overrides.py']
    obj={'schema':'offline_search.r04.stage_costs.v1','units':'milliseconds','created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
         'models':{'pi05':pi,'groot':groot},'provenance':{'gpu':'NVIDIA GeForce RTX 4090','batch_size':1,
            'warmup':5,'repetitions':30,'statistic':'median CUDA-event elapsed time of eager serving stage (includes CPU launch gaps)',
            'cpu_affinity':'26-29,70-73','OMP_NUM_THREADS':1,'OPENBLAS_NUM_THREADS':1,'MKL_NUM_THREADS':1,
            'cuda_graphs':False,'cuda_graphs_reason':'Concurrent pi05 disables compile and calls staged methods eagerly; GR00T launcher omits --compile-stage1.',
            'source_sha256':{p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in sources},
            'scope':'cost-accounting microbenchmark only; no transport, retrieval, simulator, or queueing study',
            'memory_policy':'one GPU process; admission >=14 GiB; cap 9.5 GiB; process released after each benchmark'},
         'ledger':'sum(vision*s1[mode] + miss*(miss_s1_extra[mode]+s2[mode]+s3[mode]*K/k_ref))/(N*full_cost_ms); multiply 5/L for controls',
         'limitations':['Only batch 1 measured; no claim about dynamic batching throughput.',
            'Stage 3 is scaled linearly in the required ledger. Direct K measurements are retained to expose prologue/launch overhead.',
            'Current eager measurements differ from the historical compiled/CUDA-graph reference. Never combine their milliseconds or shares.',
            'GR00T flat constants average two suite/prompt shapes. Suite-specific measurements are retained in by_suite.',
            'Prefix packing is measured but not a deployable mode. No packed saving should enter the ledger.',
            'Wrist MISS completes the missing base camera; miss_s1_extra must be charged.'],
         'historical_reference':{'pi05':{'ms':{'s1':10.26,'s2':27.69,'s3':29.57},'shares':{'s1':.152,'s2':.410,'s3':.438},'k':10,
                                           'source':'rounds/r04/FINDINGS.md; historical CUDA-graph owner constants, not this run'}}}
    tmp=DEST.with_name('.'+DEST.name+'.k3.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,DEST)
    audit=HERE/'results/cost_install.json';audit.write_text(json.dumps({'file':str(DEST),'installed_at':obj['created_at'],
        'sha256':hashlib.sha256(DEST.read_bytes()).hexdigest()},indent=2))
    print(json.dumps({k:{x:v[x] for x in ('full','shares','stage3_per_step_ms')} for k,v in obj['models'].items()},indent=2))
if __name__=='__main__':main()
