"""Emit ready-to-emit_arms R4 batches 2 and 4; no server/simulator launch."""
import argparse,copy,json,pathlib,shlex
HERE=pathlib.Path(__file__).resolve().parent
ROOT=pathlib.Path('/home/weiland/trace_runs/os_closed_loop/r03_mx')
WRIST='exp.offline_search.rounds.r04.k3_cost.method:WristMixedJudge'
FIT='/home/weiland/trace_runs/offline_search_store/derived/r04/k3_cost/fits'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-root',default='/home/weiland/trace_runs/os_closed_loop/r04_cost')
    a=ap.parse_args();run=pathlib.Path(a.run_root)
    prior={r['name']:r for file in ('arms_in.json','arms_in_500.json','arms_in_ctrl2.json') for r in json.loads((ROOT/file).read_text())}
    out=[]
    def add(r,name,batch,stage='full'):
        r=copy.deepcopy(r);r['name']=name;r['full_model']=True;r['_r4_batch']=batch;r['stage1_mode']=stage
        r['yaml_patch']={'miss':{'num_steps':2,'evidence_dir':str(run/'evidence'/name)},'write_policy':{'type':'never'}}
        if stage!='full':r['plugin_args'] += ['--os-stage1-mode',stage]
        out.append(r);return r
    for parent in ('r3mx_p_l10_g500','r3mx_p_l10_g','r3mx_p_l10_perk5','r3mx_p_sp_g'):
        add(prior[parent],parent.replace('r3mx','r4b2')+'_k2',2)
    # Match both scales in spatial and periodic families as supplementary controls.
    spbig=copy.deepcopy(prior['r3mx_p_l10_g500']);spbig['suite']='spatial'
    spbig['plugin_args'][1]=str(ROOT/'fits/r3mx_p_sp_awm500_h70.pkl')
    add(spbig,'r4b2_p_sp_g500_k2',2)
    perbig=copy.deepcopy(prior['r3mx_p_l10_perk5']);perbig['kwargs']={'lib':'big','kref':8}
    perbig['plugin_args'][1]=str(run/'fits/r4b2_perbig.pkl')
    add(perbig,'r4b2_p_l10_perk5_500_k2',2)
    for suite,short in [('spatial','sp'),('l10','l10')]:
        for seed in (1101,1102):
            add({'model':'pi05','suite':suite,'mode':'plugin','pure_inference':True,'server_seed':seed,
                 'method':'exp.offline_search.harness.baselines:B0Current','kwargs':{},'plugin_args':['--os-no-shadow-native']},
                f'r4b2_p_{short}_inf_k2_s{seed}',2)
        for lib,suffix in [('current','50'),('big','500')]:
            parent=prior['r3mx_p_l10_g500' if lib=='big' else f'r3mx_p_{short}_g']
            parent=copy.deepcopy(parent);parent['suite']=suite
            if lib=='big' and suite=='spatial':parent['plugin_args'][1]=str(ROOT/'fits/r3mx_p_sp_awm500_h70.pkl')
            add(parent,f'r4b4_p_{short}_g{suffix}_k2_dummy',4,'dummy_cached')
            wrist={'model':'pi05','suite':suite,'mode':'plugin','method':WRIST,
                   'kwargs':{'lib':lib,'guards':True,'events':'none'},
                   'plugin_args':['--os-fit-artifact',f'{FIT}/pi05_{suite}_{lib}.pkl','--os-judge','guard_only',
                                  '--os-no-shadow-native','--os-tokens','off']}
            add(wrist,f'r4b4_p_{short}_g{suffix}_k2_wrist',4,'wrist_only')
            for mode in ('dummy_cached','wrist_only'):
                method=('exp.offline_search.rounds.r04.k1_blind.wrist:BlindWristMixedJudge' if mode=='wrist_only'
                        else 'exp.offline_search.rounds.r04.k1_blind.judge:BlindMixedJudge')
                name=f'r4b4_p_{short}_b2g{suffix}_k2_'+('wrist' if mode=='wrist_only' else 'dummy')
                r={'model':'pi05','suite':suite,'mode':'plugin','method':method,
                   'kwargs':{'base_kwargs':{'lib':lib,'kref':5 if lib=='current' else 8,'serving':'phase_particles','budget':2,'gates':'all'},
                             'guards':True,'events':'none','progress_guard':'noprog_span'},
                   'plugin_args':['--os-fit-artifact',str(run/'fits'/f'{name}.pkl'),'--os-judge','guard_only',
                                  '--os-blind','--os-no-shadow-native','--os-tokens','off'],
                   '_note':'Prespecified B2 composition; coordinator selects B1/B2 winner after batch three, then re-prefits if kwargs change.'}
                add(r,name,4,mode)
    (HERE/'arms_r4.json').write_text(json.dumps(out,indent=2)+'\n')
    commands=['#!/bin/bash','set -euo pipefail','cd /home/weiland/projects/openpi',
              '# Run after K1/K2 installation. Existing fits are reused; no GPU/server work.']
    seen=set()
    for r in out:
        pargs=r['plugin_args']
        if '--os-fit-artifact' not in pargs:continue
        art=pargs[pargs.index('--os-fit-artifact')+1]
        if art in seen:continue
        seen.add(art)
        args=['taskset','-c','26-29,70-73','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
              '.venv/bin/python','-m','exp.offline_search.closed_loop.plugin','--os-method',r['method'],
              '--os-kwargs',json.dumps(r['kwargs'],separators=(',',':')),'--os-cell',f"pi05_{r['suite']}_cache",
              '--os-root','/dev/shm/offline_search_store','--os-log-dir',str(run/'prefit'), '--os-fit-artifact',art]
        commands.append(f'[ -e {shlex.quote(art)} ] || '+shlex.join(args))
    (HERE/'prefit_arms.sh').write_text('\n'.join(commands)+'\n')
    print(json.dumps({'arms':len(out),'batch2':sum(r['_r4_batch']==2 for r in out),'batch4':sum(r['_r4_batch']==4 for r in out)},indent=2))
if __name__=='__main__':main()
