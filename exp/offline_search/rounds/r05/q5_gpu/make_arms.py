from common import *
import shlex
from exp.offline_search.rounds.r05.q5_gpu.dev.gpu_retrieval import validate_method,validate_options,UNSUPPORTED_BLIND
arms=[];audits=[]
for scale in (50,500):
    c=config(f'pi05_l10_{scale}_AWM');m=load(c);validate_method(m)
    name=f'r5q5_p_l10_{scale}_cl2_shadow'
    flags=['--os-root',str(STORE),'--os-gpu-retrieval','shadow','--os-no-shadow-native','--os-tokens','off','--os-fit-artifact',c['path']]
    arm=dict(name=name,model='pi05',suite='l10',mode='plugin',method=c['spec'],kwargs=c['kwargs'],full_model=False,
             plugin_args=flags,cost_ledger=True)
    arms.append(arm)
    b=getattr(m,'base',m);lib=store.LibraryView(STORE,'pi05_l10',b.cand_name)
    deployed=pathlib.Path(rt_source) if (rt_source:=lib.meta.get('sources',{}).get('pkl')) else None
    native=pathlib.Path(store.LibraryView(STORE,'pi05_l10','current').meta['sources']['pkl'])
    audits.append(dict(name=name,scale=scale,fit_path=c['path'],fit_bytes=pathlib.Path(c['path']).stat().st_size,
                       fit_sha256=hashlib.sha256(pathlib.Path(c['path']).read_bytes()).hexdigest(),method=c['spec'],kwargs=c['kwargs'],cell=c['cell'],
                       candidate_library=b.cand_name,rows=len(b.act),library_array_bytes=sum(p.stat().st_size for p in lib.dir.glob('*.npy')),
                       deployed_pkl=str(deployed) if deployed else None,deployed_pkl_bytes=deployed.stat().st_size if deployed and deployed.exists() else None,
                       served_native_pkl=str(native),served_native_pkl_bytes=native.stat().st_size,
                       borrowed_information='none beyond the chosen fitted library',
                       exact_flags=shlex.join(['--os-method',c['spec'],'--os-kwargs',json.dumps(c['kwargs']),'--os-cell',c['cell'],
                                             '--os-log-dir',f'<RUN>/server_logs/{name}',*flags])))
    opts,_=plugin.parse_cli(['--os-method',c['spec'],'--os-kwargs',json.dumps(c['kwargs']),'--os-cell',c['cell'],'--os-log-dir','/tmp/q5_arm_validate',*flags])
    validate_options(opts,'pi05')
refused=[]
for scale in (50,500):
    c=config(f'K7_r4k7_p_l10_{scale}_tail1ug');m=load(c)
    try:validate_method(m)
    except ValueError as e:assert str(e)==UNSUPPORTED_BLIND
    else:raise AssertionError('K7 accepted')
    refused.append(dict(scale=scale,method=c['spec'],kwargs=c['kwargs'],fit=c['path'],status='unsupported; no runnable arm emitted',reason=UNSUPPORTED_BLIND))
dump(B/'arms_q5.json',arms);dump(B/'arms_unsupported.json',refused);dump(B/'results/arms_validation.json',dict(PASS=True,arms=audits,refused=refused,new_prefits=0))
(B/'prefit.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n# No new fits: exact spec/kwargs/cell checked by make_arms.py and runtime.\ntaskset -c 34-37,78-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r05/q5_gpu/make_arms.py\n')
print(json.dumps(audits,indent=2))
