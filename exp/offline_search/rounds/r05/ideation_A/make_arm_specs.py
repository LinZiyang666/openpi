"""Write REVIEW-ONLY arm recipes. Does not fit methods or launch experiments.
Names under rounds.r05.horizon_controller and two marked flags are proposed APIs,
not implemented by this ideation deliverable. Existing controls are identified.
"""
from pathlib import Path
import json,copy
O=Path(__file__).parent
P='exp.offline_search.rounds.r05.horizon_controller:'
K7='exp.offline_search.rounds.r04.k7_guard.judge:VisionConfirmedBlindMixedJudge'
K10='exp.offline_search.rounds.r04.k10_policy_tail.judge:PolicyTailJudge'
arms=[]
def add(model,suite,scale,suffix,method,kw,flags,phase,requirements=(),L=5):
    short='sp' if suite=='spatial' else suite
    name=f'r5a_{model}_{short}_{scale}_{suffix}'
    arms.append(dict(arm=name,mode='plugin',model=model,suite=suite,cell=f'{model}_{suite}_cache',
                     library_episodes=scale,method=method,kwargs=kw,full_model=True,replan_steps=L,
                     client_overrides={'replan_steps':L},server_seed=5101,
                     plugin_args=['--os-root','/home/weiland/trace_runs/offline_search_store',
                                  '--os-no-shadow-native','--os-log-r4','--os-tokens','off',
                                  '--os-fit-artifact',f'<RUN>/fits/{name}.pkl',*flags],
                     cost_ledger=True,phase=phase,requires_implementation=list(requirements),
                     selection={'screen':{'tasks':list(range(10)),'inits':list(range(10))},
                                'confirmation':{'tasks':list(range(10)),'inits':list(range(50))}}))
for suite in ('l10','spatial'):
    for scale in (50,500):
        b=dict(lib='current' if scale==50 else 'big',fit_data='same',kref=5 if scale==50 else 8,
               serving='anchor_tail',budget=1,gates='budget_only')
        kw=dict(base_kwargs=b,progress_guard='noprog_span',memo_reset_after_miss=False,
                guards=True,events='none',stuck_guard='vision_confirmed')
        f=['--os-blind','--os-judge','guard_only']
        add('pi05',suite,scale,'k7',K7,copy.deepcopy(kw),f,'P1 control; reuse completed matching arms')
        k5=copy.deepcopy(kw);k5['base_kwargs']['budget']=0
        add('pi05',suite,scale,'replan5',K7,k5,f,'P3 all-vision ordinary replanning control')
        add('pi05',suite,scale,'k10',K10,copy.deepcopy(kw),f+['--os-policy-tail'],'P1 control')
        c={**copy.deepcopy(kw),'policy_tail_gate':'lifecycle','monitor':'off'}
        add('pi05',suite,scale,'commit',P+'CommitJudge',c,f+['--os-policy-tail'],'P1', ['CommitJudge'])
        add('pi05',suite,scale,'commit_resid',P+'CommitJudge',{**copy.deepcopy(c),'monitor':'loeo_xyz99'},
            f+['--os-policy-tail'],'P1 optional interrupt variant',['CommitJudge','LOEO monitor'])
        for rule in ('always','event_or_residual'):
            add('pi05',suite,scale,'inspect_'+rule,P+'InspectCommitJudge',
                {**copy.deepcopy(c),'checkpoint_rule':rule,'monitor':'loeo_xyz99'},
                f+['--os-policy-tail','--os-tail-look',rule],
                'P3, after P1',['InspectCommitJudge','--os-tail-look; persistent original wire buffer'])
        add('pi05',suite,scale,'bridge1',P+'BridgeCommitJudge',
            {**copy.deepcopy(c),'bridge_blocks':1,'bridge_source':'cache','join_quantile':.95,
             'minority_mass_max':.2,'monitor':'loeo_xyz99','require_all_successors':True},
            f+['--os-policy-tail'],'P4 conditional screen only', ['BridgeCommitJudge'])
        for B in (0,1,2):
            g=dict(base_kwargs={**b,'budget':B},cycle_k=4,cycle_first_miss=True,
                   commit_blocks=B+1,monitor='off',guards=False)
            flags=['--os-blind','--os-judge','guard_only']
            if B:flags+=['--os-policy-tail','--os-policy-tail-blocks',str(B)]
            add('groot',suite,scale,f'cycle{B+1}_k4',P+'CycleTail',g,flags,
                'P2 optional L15' if B==2 else 'P2',
                ['CycleTail','GR00T policy-tail adapter' if B else 'CycleTail anchor clock']+
                (['--os-policy-tail-blocks'] if B else []))
            if B==1:
                add('groot',suite,scale,'cycle2_k4_passive',P+'CycleTail',
                    {**copy.deepcopy(g),'checkpoint_rule':'passive'},flags+['--os-tail-look','passive'],
                    'P2 optional execution/vision ablation',
                    ['CycleTail','GR00T policy-tail adapter','--os-tail-look passive without new anchor/call'])
    for model,Ls in (('pi05',(5,10)),('groot',(5,10,15))):
        for L in Ls:
            add(model,suite,50,f'inf_L{L}',
                'exp.offline_search.rounds.r04.k4_eval.seeded_inference:SeededInference',{},
                ['--os-judge','periodic:1'],'horizon control; library-independent (also 500)',L=L)
out=dict(status='DESIGN RECIPES ONLY; do not feed unimplemented entries to chain.sh',
         policy_steps={'pi05':10,'groot':8},cost_basis='IR=0.152*v+0.848*m per five controls; full-cost MISS',
         notes=['Fit each exact kwargs/spec; do not rename existing fit artifacts.',
                'Use --os-log-inputs on the small instrumentation pilot; it saves full a_exec and wire_actions.',
                'Repeated seeds 5101 and 5102 are server seed bases, not guarantees of paired policy noise.',
                'Do not launch this entire catalog: follow the staged pilot order in REPORT.md.',
                'Use coordinator run/emission tooling to supply yaml, matrix, library path and actual run root.'],
         arms=arms)
(O/'ARM_SPECS.json').write_text(json.dumps(out,indent=2)+'\n')
print('Review recipes:',len(arms),'unique:',len({a['arm'] for a in arms}))
