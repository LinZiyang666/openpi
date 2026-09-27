"""CPU-only ops integration fixtures; no server, simulator, network, or historical writes."""
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
OPS = REPO / 'exp/offline_search/closed_loop/ops'
DEV = HERE / 'dev'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def modules(base):
    remote = load('exp.offline_search.closed_loop.ops.remote.run_gtp_subset', base/'remote/run_gtp_subset.py')
    col = load('exp.offline_search.closed_loop.ops.collect', base/'collect.py')
    kpi = load('exp.offline_search.closed_loop.ops.kpi', base/'kpi.py')
    emit = load('eval_emit',base/'emit_arms.py')
    emit.REPO, emit.SRC = REPO, REPO/'exp/trace_dual/config'
    return remote,col,kpi,emit


BASE = OPS if '--installed' in sys.argv else DEV
if '--installed' in sys.argv:
    sys.argv.remove('--installed')
remote,col,kpi,emit = modules(BASE)
from exp.offline_search.rounds.r04.k4_eval.cost_ledger import ledger
from exp.offline_search.rounds.r04.k4_eval.estimators import design_estimate


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='k4_eval_')
        self.p = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def manifest(self):
        data = {'strata':[{'stratum':0,'N':4,'n':2,'task':0,'population_weight':.25},
                          {'stratum':1,'N':12,'n':3,'task':1,'population_weight':.75}],
                'selected':[{'task':t,'init':e,'stratum':t,'inclusion_probability':pi}
                            for t,n,pi in [(0,2,.5),(1,3,.25)] for e in range(n)]}
        p=self.p/'manifest.json'; p.write_text(json.dumps(data))
        return p,remote.load_manifest(p)

    def test_manifest_design_and_pair_variance(self):
        _,m=self.manifest()
        values={(0,0):1,(0,1):0,(1,0):1,(1,1):1,(1,2):0}
        r=design_estimate(m,values)
        self.assertEqual(r['estimate'],.625)
        self.assertAlmostEqual(r['design_variance'],.0546875)
        self.assertEqual(r['strata'][0]['inclusion_probability'],.5)
        partial=design_estimate(m,{(0,0):1})
        self.assertIsNone(partial['estimate']); self.assertIsNone(partial['design_variance'])
        ref={(0,0):0,(0,1):0,(1,0):1,(1,1):1,(1,2):1}
        d=design_estimate(m,{p:values[p]-ref[p] for p in values})
        self.assertAlmostEqual(d['estimate'],-.125)
        self.assertAlmostEqual(d['design_variance'],.0546875)
        def arm(name,v):
            pairs=list(v)
            return {'info':{'arm':name},'manifest':m,'ep':{'task':np.array([p[0] for p in pairs]),
                    'ep_idx':np.array([p[1] for p in pairs]),'success':np.array(list(v.values()),bool)}}
        out=kpi.paired(arm('r',ref),arm('a',values),100,0)
        self.assertEqual(out['delta_sr'],0)
        self.assertAlmostEqual(out['weighted']['delta_sr'],-.125)

    def test_census_singleton_and_bad_probability(self):
        path,m=self.manifest()
        for h in m['data']['strata']: h['N']=h['n']; h.pop('population_weight')
        for r in m['selected'].values(): r['inclusion_probability']=1
        est=design_estimate(m,{p:1 for p in m['selected']})
        self.assertEqual(est['estimate'],1); self.assertEqual(est['design_variance'],0)
        m['selected'][(0,0)]['inclusion_probability']=.1
        with self.assertRaises(ValueError): design_estimate(m,{p:1 for p in m['selected']})
        simple=self.p/'simple.json'; simple.write_text('[[0, 0], [0, 0], [1, 2]]')
        x=remote.load_manifest(simple); self.assertEqual(len(x['selected']),2)
        self.assertIsNone(design_estimate(x,{(0,0):1,(1,2):0})['estimate'])
        simple.write_text('[[0,50]]')
        with self.assertRaises(ValueError): remote.load_manifest(simple)

    def test_all_24_supplied_manifests(self):
        paths=list((HERE.parent/'ideation_C').glob('pilot_manifest_*.json'))
        self.assertEqual(len(paths),24)
        for p in paths:
            m=remote.load_manifest(p)
            self.assertEqual(len(m['selected']),m['data']['n'])
            out=design_estimate(m,{k:1 for k in m['selected']})
            self.assertAlmostEqual(out['estimate'],1); self.assertEqual(out['design_variance'],0)

    def test_exact_selection_count_and_old_cartesian(self):
        path,m=self.manifest()
        tasks=[SimpleNamespace(task_id=t,episode_idx=e) for t in range(10) for e in range(50)]
        self.assertEqual(len(remote.selected_tasks(tasks,m)),5)
        self.assertEqual(len(remote.selected_tasks(tasks,None,{0,1},{0,1,2})),6)
        self.assertEqual(len(remote.selected_tasks(tasks,None,{0},set())),10)
        journal=self.p/'journal.jsonl'
        rows=[dict(task_uid=f'a:eval:{t}:{e}',accepted=True,status='done',success=True) for t,e in m['selected']]
        rows += [rows[0],dict(task_uid='a:eval:9:49',accepted=True,status='done',success=True),
                 dict(task_uid='b:eval:0:0',accepted=True,status='failed',success=False)]
        journal.write_text('\n'.join(json.dumps(r) for r in rows)+'\n{bad\n')
        res=subprocess.check_output(['taskset','-c','30-33,74-77',sys.executable,str(BASE/'remote/count.py'),str(journal),
                                     '--manifest',str(path),'--arm','a'],text=True).strip()
        self.assertEqual(res,'5 5 8')
        old=subprocess.check_output(['taskset','-c','30-33,74-77',sys.executable,str(BASE/'remote/count.py'),str(journal)],text=True).strip()
        self.assertEqual(old,'7 6 8')

    def test_cost_full_blind_k2_L10_and_groot(self):
        for model,kfull in [('pi05',10),('groot',8)]:
            full=ledger({'model':model},[dict(uid='a',hit=False,vision=True,miss_k=kfull)])
            self.assertAlmostEqual(full['ir_per_five_controls'],1)
            x=ledger({'model':model,'replan_steps':10},[dict(uid='a',hit=False,vision=True,miss_k=kfull)])
            self.assertAlmostEqual(x['ir_per_five_controls'],.5)
            self.assertEqual(x['controls_per_episode'],10)
        rows=[dict(uid='a',hit=True,vision=True),dict(uid='a',hit=True,vision=False,src='cache_blind'),
              dict(uid='a',hit=False,vision=True,miss_k=2)]
        c=ledger({'model':'pi05'},rows,table=self.p/'absent.json')
        self.assertAlmostEqual(c['ir_per_five_controls'],(.152*2+.410+.438*.2)/3)
        self.assertEqual(c['v'],2/3); self.assertEqual(c['m'],1/3)
        for K,L,expected in [(10,10,.5),(2,10,.3248),(2,5,.6496)]:
            c=ledger({'model':'pi05','client_overrides':{'replan_steps':L}},[dict(uid='a',hit=False,miss_k=K)],table=self.p/'absent.json')
            self.assertAlmostEqual(c['ir_per_five_controls'],expected)
        c=ledger({'model':'pi05'},[dict(uid='a',hit=True),dict(uid='a',hit=False)])
        self.assertAlmostEqual(c['ir_per_five_controls'],.576)
        with self.assertRaises(ValueError): ledger({'model':'pi05'},[dict(hit=False,vision=False)])

    def test_cost_table_and_startup_per_process(self):
        path=self.p/'cost.json'; path.write_text(json.dumps({'models':{'pi05':{
          'full':{'s1':.2,'s2':.3,'s3':.5,'k':10},'full_cost_ms':100,
          'modes':{'wrist_only':{'s1':.1,'miss_s1_extra':.1}}}}}))
        rows=[dict(uid='a',tag='a',hit=True,vision=True,s1_ms=10),
              dict(uid='a',tag='a',hit=False,vision=True,s1_ms=10,s23_ms=50)]
        starts=[dict(tag='a',stage1_mode='wrist_only',miss_steps=2)]
        c=ledger({'model':'pi05'},rows,starts,table=path)
        self.assertAlmostEqual(c['ir_per_five_controls'],.35)
        self.assertAlmostEqual(c['ir_measured_per_five_controls'],.35)
        self.assertEqual(c['k_per_miss']['mean'],2)

    def test_k3_installed_cost_table_and_legacy(self):
        path=OPS/'cost_table.json'
        if not path.exists(): self.skipTest('K3 table not yet installed')
        table=json.loads(path.read_text())
        for model in ['pi05','groot']:
            for suite in ['spatial','l10']:
                data=table['models'][model]
                data=data.get('by_suite',{}).get(suite,data)
                full=data['full']; total=sum(full[x] for x in ['s1','s2','s3'])
                for mode,prices in data['modes'].items():
                    c=ledger({'model':model,'suite_short':suite},[dict(vision=True,hit=False,miss_k=2)],
                             [dict(stage1_mode=mode,miss_steps=2)])
                    expected=(prices['s1']+prices.get('miss_s1_extra',0)+prices['s2']+prices['s3']*2/prices['k'])/total
                    self.assertAlmostEqual(c['ir_per_five_controls'],expected)
                    self.assertTrue(c['cost_source'].startswith(str(path)))
        old=ledger({'model':'pi05'},[dict(hit=True),dict(hit=False)])
        self.assertEqual(old['ir_per_five_controls'],.576)
        self.assertTrue(old['legacy_owner_pricing'])

    def test_emission_legacy_bytes_and_frontier(self):
        old=load('before_emit',HERE/'before/emit_arms.py'); old.REPO=REPO; old.SRC=emit.SRC
        spec=self.p/'spec.json'; spec.write_text(json.dumps([dict(name='legacy',model='pi05',suite='l10',mode='native')]))
        out=self.p/'run'; old.main(['--run-root',str(out),'--spec',str(spec)])
        before={p.name:p.read_bytes() for p in [out/'arms.json',*(out/'config').iterdir()]}
        emit.main(['--run-root',str(out),'--spec',str(spec)])
        after={p.name:p.read_bytes() for p in [out/'arms.json',*(out/'config').iterdir()]}
        self.assertEqual(before,after)
        emit.main(['--run-root',str(out),'--spec',str(HERE/'arms_frontier.json')])
        arms=json.loads((out/'arms.json').read_text()); self.assertEqual(len(arms),18)
        for row in arms[1:]:
            cfg=yaml.safe_load(Path(row['yaml']).read_text()); self.assertNotIn('trace',cfg)
            if row.get('pure_inference'): self.assertEqual(row['judge'],'periodic:1'); self.assertTrue(row['full_model'])
            if row.get('yaml_patch'):
                self.assertEqual(cfg['miss']['num_steps'],row['yaml_patch']['miss']['num_steps'])
                self.assertTrue(cfg['miss']['evidence_dir'])
            mx=yaml.safe_load(Path(row['matrix']).read_text())['arms'][0]
            if row.get('client_overrides'): self.assertEqual(mx['client_overrides'],row['client_overrides'])
        r=emit.expand_replicates([dict(name='rep',seeds=[1,2])]); self.assertEqual([x['name'] for x in r],['rep_s1','rep_s2'])
        self.assertEqual(emit.merge_yaml({'a':{'b':1,'c':2}}, {'a':{'b':3}}),{'a':{'b':3,'c':2}})

    def test_emitted_yaml_config_validation_both_models(self):
        from openpi.cache.config import load_cache_config
        specs=[]
        for model in ['pi05','groot']:
            for suite in ['l10','spatial']:
                specs.append(dict(name=f'{model}_{suite}',model=model,suite=suite,pure_inference=True,
                                  yaml_patch={'miss':{'num_steps':2}},client_overrides={'replan_steps':10}))
        sp=self.p/'spec.json'; sp.write_text(json.dumps(specs)); out=self.p/'run'
        emit.main(['--run-root',str(out),'--spec',str(sp)])
        for row in json.loads((out/'arms.json').read_text()):
            cfg=load_cache_config(row['yaml'],check_files=False)
            self.assertEqual(cfg.miss.num_steps,2)
            if row['model']=='groot': self.assertEqual(row['server_env']['GROOT_DENOISING_STEPS'],2)

    def test_process_policy_seed_and_prefit(self):
        code=('import json, random, numpy as np, torch; '
              'from exp.offline_search.rounds.r04.k4_eval.seeded_inference import seed_process; '
              's=seed_process(); print(json.dumps([s,random.random(),float(np.random.random()),float(torch.rand(()))]))')
        seeds=[1*65536+23180,1*65536+23181,2*65536+23180]
        outputs=[]
        for seed in [*seeds,seeds[0]]:
            text=subprocess.check_output(['taskset','-c','30-33,74-77',sys.executable,'-c',code,'--os-seed',str(seed)],text=True)
            outputs.append(json.loads(text.splitlines()[-1]))
        self.assertEqual(outputs[0],outputs[3])
        self.assertEqual(len(set(tuple(o) for o in outputs[:3])),3)

    def test_collect_kpi_end_to_end(self):
        # Both library scales/models: exact served heads override static action reconstruction.
        for model in ['pi05','groot']:
            for scale in [50,500]:
                run=self.p/f'{model}{scale}'; arm='a'; cd=run/'runs'/arm/'client'; cd.mkdir(parents=True)
                sd=run/'runs'/arm/'server_1'; sd.mkdir()
                store=self.p/'store'; lib=store/'library'/f'{model}_l10'/'current'; lib.mkdir(parents=True,exist_ok=True)
                rng=np.random.default_rng(0); action=rng.normal(size=(scale,10,32)).astype('float32')
                for key,a in dict(action=action,episode=np.arange(scale),step=np.zeros(scale,int),ep_len=np.ones(scale,int),
                                 next=np.full(scale,-1),task_id=np.zeros(scale,int),rs=np.zeros((scale,8)),
                                 key_v0=rng.normal(size=(scale,8)),key_v1=rng.normal(size=(scale,8))).items():
                    np.save(lib/f'{key}.npy',a)
                (run/'arms.json').write_text(json.dumps([dict(arm=arm,model=model,suite='libero_10',mode='plugin',judge='periodic:3')]))
                rows=[]; jr=[]; ps=[]
                for e in range(2):
                    uid=f'a:eval:0:{e}'; jr.append(dict(task_uid=uid,accepted=True,status='done',success=bool(e),attempt=2))
                    for step in range(4):
                        head=np.ones((5,7)); head[:,6]=1 if step%2 else -1
                        rows.append(dict(ev='dec',uid=uid,step=step,attempt=2,ts=e*10+step,tag='t',hit=step!=2,
                                         vision=step!=1,src='policy' if step==2 else ('cache_blind' if step==1 else 'cache'),
                                         miss_k=2 if step==2 else None,top1=0,topk=[0],scores=[1],lib='current',
                                         served_head=head.tolist(),s1_ms=1 if step!=1 else None,s23_ms=2 if step==2 else None))
                        ps.append(dict(task_uid=uid,step_idx=step,accepted=True,hit_type='MISS' if step==2 else 'FULL_HIT'))
                # Earlier failed attempt and retransmitted decision must not alter ledger.
                extra={**rows[0],'attempt':1,'hit':False,'miss_k':10}
                (sd/'decisions_test.jsonl').write_text('\n'.join(json.dumps(r) for r in [extra,*rows,rows[-1]]))
                (cd/'journal.jsonl').write_text('\n'.join(json.dumps(r) for r in jr))
                (cd/'per_step.jsonl').write_text('\n'.join(json.dumps(r) for r in ps))
                c=col.summarize(run,arm,write=False)
                self.assertEqual(c['complete'],2); self.assertEqual(c['sr'],.5)
                self.assertEqual(c['cost_ledger']['decisions'],8)
                A=kpi.load_arm(run,arm,store=str(store),tasks=None,episodes=None,act_eps=.1,recon_override=None,
                               cache_dir=None,want_client=True)
                out=kpi.arm_kpis(A,None)
                self.assertEqual(out['sr'],.5); self.assertEqual(out['gripper_flips']['g0_per_ep'],3)
                self.assertEqual(out['mixed_mode']['h_per_regime']['n_step0'],2)
                self.assertEqual(out['mixed_mode']['h_per_regime']['n_after_miss'],2)
                self.assertEqual(c['cost_ledger'],out['cost_ledger'])
                kpi._LIBS.clear()


if __name__=='__main__':
    unittest.main()
