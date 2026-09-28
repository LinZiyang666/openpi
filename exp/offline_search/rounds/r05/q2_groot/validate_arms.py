"""Emitter/parser/CacheConfig, exact fit metadata and own-library footprint."""
import hashlib,json,pickle
from pathlib import Path
from exp.offline_search.closed_loop import plugin
from exp.offline_search.closed_loop.ops.emit_arms import main as emit
from exp.offline_search.harness import store
import openpi.cache.config as cc
B=Path(__file__).resolve().parent
R=Path('/tmp/q2_arm_validation');R.mkdir(exist_ok=True)
spec=json.loads((B/'arms_q2.json').read_text().replace('<RUN>',str(R)))
path=R/'spec.json';path.write_text(json.dumps(spec,indent=2))
emit(['--run-root',str(R),'--spec',str(path)])
rows=json.loads((R/'arms.json').read_text());fits=[]
assert len(rows)==6
for r in rows:
    cc.load_cache_config(r['yaml'])
    opts,rest=plugin.parse_cli(['--os-method',r['method'],'--os-kwargs',json.dumps(r['kwargs']),'--os-cell',r['cell'],'--os-log-dir',str(R/r['arm']),*r['plugin_args']])
    assert not rest and r['full_model'] and r['cost_ledger']
    assert r['client_overrides']['resize_size']==256
    if r.get('pure_inference'):
        assert opts.judge.mode=='periodic' and opts.judge.k==1 and not opts.os_blind
        assert r['replan_steps']==10 and r['server_seed']==5101
        continue
    assert r['replan_steps']==5 and opts.os_blind and opts.os_policy_tail and opts.os_policy_tail_blocks==1
    assert opts.judge.mode=='guard_only'
    fit=Path('/tmp/q2_fits')/(r['arm']+'.pkl')
    with fit.open('rb') as f:blob=pickle.load(f)
    assert {k:blob[k] for k in ('spec','kwargs','cell')}==dict(spec=r['method'],kwargs=r['kwargs'],cell=r['cell'])
    m=blob['method'];lib=store.LibraryView(opts.os_root,store.lib_key(r['cell']),m.cand_name)
    deployed=Path(lib.meta['sources']['pkl'])
    fits.append(dict(arm=r['arm'],path=str(fit),bytes=fit.stat().st_size,sha256=hashlib.sha256(fit.read_bytes()).hexdigest(),library=m.cand_name,library_path=str(lib.dir),rows=lib.L,episodes=len(set(lib.episode.tolist())),library_top_level_npy_bytes=sum(p.stat().st_size for p in lib.dir.glob('*.npy')),library_npy_bytes=sum(p.stat().st_size for p in lib.dir.rglob('*.npy')),deployed_pkl=str(deployed),deployed_pkl_bytes=deployed.stat().st_size if deployed.exists() else None,representation_bytes_per_entry=m.bytes_per_entry(),borrowed_big_library_information=False))
(B/'results/arms_validation.json').write_text(json.dumps(dict(PASS=True,arms=6,fits=fits),indent=2))
print(json.dumps(dict(PASS=True,arms=6,fits=fits),indent=2))
