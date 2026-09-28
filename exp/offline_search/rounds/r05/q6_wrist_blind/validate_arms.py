"""Installed emitter and strict real artifact metadata checks; no launch."""
import argparse
import json
import pickle
import subprocess
from exp.offline_search.closed_loop import plugin, stage_overrides
from openpi.cache.config import load_cache_config
from .common import HERE, FITS, PREFIX, write_json

def main():
    p=argparse.ArgumentParser(); p.add_argument('--tag',default='final'); a=p.parse_args()
    rows=json.loads((HERE/'arms_q6.json').read_text())+json.loads((HERE/'arms_phase2.json').read_text())
    run=HERE/'results'/('emitted_'+a.tag)
    resolved=json.loads(json.dumps(rows).replace('<RUN>',str(run)))
    path=HERE/'results'/('arms_resolved_'+a.tag+'.json'); write_json(path,resolved)
    cmd=PREFIX+['-m','exp.offline_search.closed_loop.ops.emit_arms','--run-root',str(run),'--spec',str(path)]
    subprocess.run(cmd,check=True)
    emitted=json.loads((run/'arms.json').read_text())
    for s,r in zip(resolved,emitted):
        load_cache_config(r['yaml'])
        stage,args=stage_overrides.parse_flags(s['plugin_args'])
        opts,rest=plugin.parse_cli(['--os-method',s['method'],'--os-kwargs',json.dumps(s['kwargs']),
            '--os-cell',r['cell'],'--os-log-dir',str(run/'logs')]+args)
        stage_overrides.validate_method(opts,stage.os_stage1_mode)
        assert not rest and stage.os_stage1_mode == 'wrist_only'
        assert s['full_model'] and s['cost_ledger'] and r['full_model'] and r['cost_ledger']
        assert opts.os_blind and opts.os_tokens == 'off' and opts.os_judge == 'guard_only'
        assert stage_overrides.miss_steps_from_yaml(r['yaml'],'pi05') == 10
        with (FITS/(s['name']+'.pkl')).open('rb') as f: blob=pickle.load(f)
        assert blob['spec']==s['method'] and blob['kwargs']==s['kwargs'] and blob['cell']==r['cell']
    result=dict(PASS=True,primary_arms=4,phase_variant_arms=4,configs=len(emitted),
                exact_artifact_metadata=True,full_model=True,miss_steps=10,cost_ledger=True,command=cmd)
    write_json(HERE/'results/arms_validation.json',result); print(result)

if __name__ == '__main__': main()
