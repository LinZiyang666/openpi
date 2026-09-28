"""Bounded CPU-only final test matrix; all child commands are affinity-prefixed."""
import argparse
import concurrent.futures
import json
import subprocess
from .common import HERE, ROOT, FITS, SPEC, PREFIX, arm_name, kwargs, write_json

def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['evidence','plugin','existing','smoke'])
    p.add_argument('--tag', default='final'); a = p.parse_args()
    out = HERE/'results'/a.tag; out.mkdir(parents=True, exist_ok=True)
    commands = []
    if a.mode == 'evidence':
        for suite in ('l10','spatial'):
            for scale in (50,500):
                commands.append((f'evidence_{suite}_{scale}', PREFIX+['-m',
                    'exp.offline_search.rounds.r05.q6_wrist_blind.evidence', '--suite', suite, '--scale', str(scale)]))
    elif a.mode == 'plugin':
        for variant in ('tail','phase2'):
            for suite in ('l10','spatial'):
                for scale in (50,500):
                    name = arm_name(suite, scale, variant)
                    commands.append(('plugin_'+name, PREFIX+['-m',
                        'exp.offline_search.rounds.r05.q6_wrist_blind.plugin_selftest',
                        '--os-blind','--os-stage1-mode','wrist_only','--os-tokens','off',
                        '--os-judge','guard_only','--cell',f'pi05_{suite}_cache', '--root',str(ROOT),
                        '--yaml',f"exp/trace_dual/config/tr_pi05_{'sp' if suite == 'spatial' else suite}_cache.yaml",
                        '--method',SPEC,'--kwargs',json.dumps(kwargs(scale,variant)),
                        '--fit-artifact',str(FITS/(name+'.pkl')),'--out',str(out/('plugin_'+name))]))
    elif a.mode == 'smoke':
        for suite in ('l10','spatial'):
            for scale in (50,500):
                for arm in ('inf','cache'):
                    name=f'smoke_{suite}_{scale}_{arm}'
                    commands.append((name,PREFIX+['-m','exp.offline_search.harness.smoke',
                        '--method',SPEC,'--kwargs',json.dumps(kwargs(scale)), '--cell',f'pi05_{suite}_{arm}',
                        '--episodes','2','--root',str(ROOT),'--no-ref','--out',str(out/name)]))
    else:
        cases=[]
        for model in ('pi05','groot'):
            for judge in (None,'periodic:1'):
                cases.append((f'legacy_{model}_{judge or "hit"}', model,
                    'exp.offline_search.closed_loop.probe:ProbeB0',{}, judge,False,None))
        cases += [
            ('k1_full','pi05','exp.offline_search.rounds.r04.k1_blind.judge:BlindMixedJudge',
             dict(base_kwargs=dict(lib='current',kref=5,budget=2),events='none',ncal=64),'guard_only',True,None),
            ('k1_wrist','pi05','exp.offline_search.rounds.r04.k1_blind.wrist:BlindWristMixedJudge',
             dict(base_kwargs=dict(lib='current',kref=5,budget=2),events='none',ncal=64),'guard_only',True,'wrist'),
            ('k3_wrist','pi05','exp.offline_search.rounds.r04.k3_cost.method:WristMixedJudge',
             dict(lib='current',events='none',ncal=64),'periodic:2',False,None),
            ('k7_full','pi05','exp.offline_search.rounds.r04.k7_guard.judge:VisionConfirmedBlindMixedJudge',
             dict(base_kwargs=dict(lib='current',kref=5,budget=2),events='none',ncal=64),'guard_only',True,None)]
        for name,model,spec,kw,judge,blind,mode in cases:
            mod='exp.offline_search.closed_loop.selftest' if mode is None else 'exp.offline_search.rounds.r05.q6_wrist_blind.plugin_selftest'
            cmd=PREFIX+['-m',mod,'--cell',f'{model}_spatial_cache','--root',str(ROOT),
                '--yaml',f'exp/trace_dual/config/tr_{model}_sp_cache.yaml','--method',spec,
                '--kwargs',json.dumps(kw),'--episodes','2','--out',str(out/name),'--no-shadow']
            if mode:
                cmd += ['--os-blind','--os-stage1-mode','wrist_only','--os-tokens','off','--os-judge',judge]
            else:
                if blind: cmd += ['--blind']
                if judge: cmd += ['--judge',judge]
            commands.append((name,cmd))
    def run(item):
        name,cmd=item
        with (out/(name+'.log')).open('w') as f:
            rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
        result=dict(name=name,returncode=rc,command=cmd)
        print(name,rc,flush=True); return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(run,commands))
    write_json(out/(a.mode+'_commands.json'),results)
    assert all(r['returncode']==0 for r in results),results

if __name__ == '__main__': main()
