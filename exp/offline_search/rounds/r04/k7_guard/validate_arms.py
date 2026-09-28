"""Use installed emitter/config/parser without launching anything."""
import json
import pickle
import subprocess
from exp.offline_search.closed_loop import plugin
from openpi.cache.config import load_cache_config
from exp.offline_search.rounds.r04.k7_guard.prepare import HERE, PREFIX

rows=json.loads((HERE/'arms_k7.json').read_text())
run=HERE/'results/emitted'
resolved=json.loads(json.dumps(rows).replace('<RUN>',str(run)))
p=HERE/'results/arms_resolved.json';p.write_text(json.dumps(resolved,indent=2))
cmd=PREFIX+['-m','exp.offline_search.closed_loop.ops.emit_arms','--run-root',str(run),'--spec',str(p)]
subprocess.run(cmd,check=True)
emitted=json.loads((run/'arms.json').read_text())
for s,r in zip(resolved,emitted):
    load_cache_config(r['yaml'])
    opts,rest=plugin.parse_cli(['--os-method',s['method'],'--os-kwargs',json.dumps(s['kwargs']),
        '--os-cell',r['cell'],'--os-log-dir',str(run/'logs')]+s['plugin_args'])
    assert not rest
    assert s['full_model'] and s['cost_ledger'] and opts.os_blind and opts.os_judge=='guard_only'
    assert r['full_model'] and r['cost_ledger'] and r['judge']=='guard_only'
    assert opts.os_root=='/home/weiland/trace_runs/offline_search_store'
    with open('/tmp/k7_guard_fits/'+s['name']+'.pkl','rb') as f:blob=pickle.load(f)
    assert blob['spec']==s['method'] and blob['kwargs']==s['kwargs'] and blob['cell']==r['cell']
report=dict(PASS=True,arms=len(rows),configs=len(emitted),metadata_matches=True,command=cmd)
(HERE/'results/arms_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
