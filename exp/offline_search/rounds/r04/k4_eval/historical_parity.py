"""Compare copied R3 implementations with current ops; historical roots read-only."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import warnings

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
ROOT=Path('/home/weiland/trace_runs/os_closed_loop')
STORE='/home/weiland/trace_runs/offline_search_store'
BASE=REPO/'exp/offline_search/closed_loop/ops' if '--installed' in sys.argv else HERE/'dev'


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s)
    sys.modules[name]=m; s.loader.exec_module(m); return m


def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k not in ('collected_at','parse_s')}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    return x


load('exp.offline_search.closed_loop.ops.remote.run_gtp_subset',BASE/'remote/run_gtp_subset.py')
new=load('exp.offline_search.closed_loop.ops.collect',BASE/'collect.py')
old=load('old_collect',HERE/'before/collect.py')
kn=load('new_kpi',BASE/'kpi.py'); ko=load('old_kpi',HERE/'before/kpi.py')
records=[]; total=0; t0=time.time()
warnings.filterwarnings('ignore',category=RuntimeWarning)
with tempfile.TemporaryDirectory(prefix='k4_parity_') as tmp:
    for tag in ['r02_g50','r02_g500','r03_full','r03_mx','r03_pilot','r03_smoke']:
        root=ROOT/tag
        if not (root/'arms.json').exists():continue
        arms=json.loads((root/'arms.json').read_text())
        local=Path(tmp)/tag; local.mkdir(); (local/'arms.json').write_text(json.dumps(arms))
        for meta in arms:
            arm=meta['arm']; src=root/'runs'/arm
            if not (src/'client/journal.jsonl').exists():continue
            dst=local/'runs'/arm; dst.mkdir(parents=True)
            (dst/'client').symlink_to(src/'client',target_is_directory=True)
            for server in src.glob('server_*'):
                if server.is_dir(): (dst/server.name).symlink_to(server,target_is_directory=True)
            a=old.summarize(local,arm); b=new.summarize(root,arm,write=False)
            assert clean(a)==clean(b),(tag,arm,'collect output changed')
            rec=dict(run=tag,arm=arm,n=b['complete'],success=b['success'],sr=b['sr'],
                     ir=b.get('mixed',{}).get('ir_pi05_formula',.152 if meta.get('model')=='pi05' else None))
            recorded=src/'summary.json'
            if recorded.exists():
                prior=json.loads(recorded.read_text())
                rec['recorded_sr_match']=prior['sr']==b['sr']
                if 'mixed' in prior:
                    rec['recorded_ir_match']=prior['mixed']['ir_pi05_formula']==b['mixed']['ir_pi05_formula']
                assert rec['recorded_sr_match'],rec
                assert rec.get('recorded_ir_match',True),rec
            total+=b['complete']; records.append(rec)
            print(tag,arm,rec['n'],rec['sr'],rec['ir'],flush=True)
# Full KPI output parity for eight CL2 arms and four mixed examples, using shared read-only library caches.
krecords=[]
for tag,arms in [('r02_g50',[f'oscl50_{m}_{s}_cl2' for m in ['p','g'] for s in ['sp','l10']]),
                 ('r02_g500',[f'oscl500_{m}_{s}_cl2' for m in ['p','g'] for s in ['sp','l10']]),
                 ('r03_mx',['r3mx_p_l10_g','r3mx_p_l10_g500','r3mx_p_l10_perk5','r3mx_p_sp_g'])]:
    for arm in arms:
        kw=dict(store=STORE,tasks=None,episodes=None,act_eps=.1,recon_override=None,cache_dir=HERE/'results/pause_cache',want_client=True)
        a=ko.load_arm(ROOT/tag,arm,**kw)
        kn._LIBS=ko._LIBS
        b=kn.load_arm(ROOT/tag,arm,**kw)
        oa=ko._to_native(ko.arm_kpis(a,None)); ob=kn._to_native(kn.arm_kpis(b,None))
        assert clean(oa)==clean(ob),(tag,arm,'KPI output changed')
        rec=dict(run=tag,arm=arm,n=ob['episodes'],sr=ob['sr'],decisions=ob['decisions'],
                 ir=ob.get('mixed_mode',{}).get('ir_formula_pi05',.152 if ob['model']=='pi05' else None))
        krecords.append(rec); print('KPI',rec,flush=True)
        ko._LIBS.clear(); kn._LIBS.clear()
report=dict(collect_arms=len(records),collect_episodes=total,collect=records,kpi_arms=len(krecords),kpi=krecords,
            pass_all=True,wall_s=time.time()-t0)
(HERE/'results/historical_parity.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('collect','kpi')}),flush=True)
