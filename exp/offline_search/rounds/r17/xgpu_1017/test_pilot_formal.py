"""CPU checks of native command/plan boundaries; no GPU, remote call or fleet."""
import copy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import types
import unittest
from unittest.mock import patch

from pilot_campaigns import W,read,write,scientific,NODES
from pilot_formal import context,fixed,worker_rewrite,server_code,renderer_admission,sealed
from pilot_trace import child_command,fleet_prefix
from pilot_readmap import install

class PilotFormalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs=read(W/'closedloop_launch_1018/CAMPAIGNS.json')['jobs']
        cls.fixture=W/'evidence'/('pilot_cpu_'+str(time.time_ns()))
        cls.fixture.mkdir()

    def test_twenty_native_plans_pair_exactly(self):
        from normalize_formal import index_plan,validate_manifest
        self.assertEqual(len(self.jobs),20)
        by={};paired={}
        for j in self.jobs:
            p=Path(j['root']);c=read(p/'campaign.json');fixed(c)
            tasks=read(p/'PLANNED_TASKS.json');self.assertEqual(len(tasks),500)
            self.assertEqual(tasks,read(p/'EPISODE_PLAN.json')['tasks'])
            by[j['host'],j['model'],j['arm']]=scientific(tasks)
            plan=read(p/'PAIRED_PLAN.json')
            indexed=index_plan(plan,read(p/'assets/benchmark.json'))
            validate_manifest(indexed,dict(tasks=tasks))
            paired[j['host'],j['model'],j['arm']]=plan['identity']
            self.assertEqual(c['execution_worker_nodes'],NODES[j['model']])
            self.assertFalse((p/'LAUNCH.json').exists())
            self.assertFalse((p/'PREPARED.json').exists())
        for model in NODES:
            for arm in ('pure','cache'):
                self.assertEqual(by['4090',model,arm],by['h100',model,arm])
                self.assertEqual(paired['4090',model,arm],paired['h100',model,arm])

    def test_actual_frozen_mps_all_campaigns(self):
        rows=[]
        for j in self.jobs:
            c=read(Path(j['root'])/'campaign.json')
            paths=runpy.run_path(c['runtime']+'/exp/r17_integration/models/fastwam/ops/mps.py')['paths']
            actual=paths(Path(c['server_run']).parent,'GPU-01234567-0123-0123-0123-0123456789ab')
            size=len(os.fsencode(actual[1]));self.assertLessEqual(size,85)
            rows.append(dict(run=j['name'],bytes=size,pipe=str(actual[1])))
        write(W/'evidence/pilot_mps_CPU_1018.json',rows)

    def test_off_is_process_local_and_native_counts_cannot_change(self):
        from exp.offline_search.rounds.r17.autoscale_1017 import defaults,switch
        from exp.r17_integration.ops import formal_eval as f
        before=switch.enabled
        with patch.object(switch,'enabled',lambda:True),patch.object(defaults,'enabled',lambda:True):
            with context(Path(self.jobs[0]['root'])):
                self.assertFalse(switch.enabled());self.assertFalse(defaults.enabled())
                self.assertIsNone(f.worker_resilience)
            self.assertTrue(switch.enabled())
        self.assertIs(switch.enabled,before)
        c=read(Path(self.jobs[0]['root'])/'campaign.json')
        for change in (dict(autoscale={'enabled':True}),dict(execution_workers=1),
                       dict(execution_worker_nodes={'timan108':112})):
            with self.assertRaises(ValueError):fixed(dict(c,**change))

    def test_child_prefix_executes_and_preserves_argv(self):
        marker=self.fixture/'child_trace_installed'
        source='def from_environment():\n from pathlib import Path\n Path('+repr(str(marker))+').write_text("yes")\n'
        command=child_command([sys.executable,'-B','-c','import sys,json;print(json.dumps(sys.argv[1:]))','--seed','17'],source)
        p=subprocess.run(command,capture_output=True,text=True,check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        self.assertEqual(json.loads(p.stdout),['--seed','17']);self.assertEqual(marker.read_text(),'yes')

    def test_native_fleet_factory_is_wrapped_after_gpu2_and_for_recycles(self):
        from exp.r17_integration.ops import formal_workers as f
        generated=[]
        def native_main():
            for i in (0,1,0):generated.append(f.worker_command({},i))
        def factory(c,i):return str(i),[sys.executable,'-B','-c','print("guard then worker")','-m','native','--id',str(i)]
        native_runpy=runpy.run_module
        with patch.object(f,'main',native_main),patch.object(f,'worker_command',factory),patch.object(runpy,'run_module',native_runpy):
            scope={};exec(fleet_prefix('def from_environment():pass'),scope)
            runpy.run_module('exp.r17_integration.ops.formal_workers',run_name='__main__')
            self.assertEqual(len(generated),3)
            for identity,argv in generated:
                self.assertEqual(argv[-4:],['-m','native','--id',identity])
                self.assertIn('guard then worker',argv[3]);compile(argv[3],'<child>','exec')

    def test_real_native_worker_bootstraps_compile(self):
        from exp.offline_search.rounds.r17.filter_failed.launch import WORKER
        source=(W/'client_trace.py').read_text()
        for node in ('timan1','timan107'):
            cmd=worker_rewrite(['/python','-u','-m',WORKER,'--campaign','/own/campaign.json'],node,source)
            self.assertIn(WORKER,cmd)
            compile(cmd[cmd.index('-c')+1],'<fleet>','exec')
            self.assertEqual(cmd[-2:],['--campaign','/own/campaign.json'])
            if node=='timan1':self.assertIn('-S',cmd)

    def test_native_asset_aliases_are_read_only_and_restore(self):
        root=self.fixture/'aliases';root.mkdir();target=root/'real.json';target.write_text('{"pin":17}')
        alias=root/'missing'/'fit.json'
        with install({str(alias):str(target)}):
            self.assertTrue(alias.exists());self.assertTrue(alias.parent.is_dir())
            self.assertEqual(read(alias),{'pin':17})
            with self.assertRaises(PermissionError):alias.write_text('overwrite')
        self.assertFalse(alias.exists());self.assertEqual(read(target),{'pin':17})

    def test_every_h100_alias_map_constructs_without_ambiguity(self):
        for j in self.jobs:
            c=read(Path(j['root'])/'campaign.json')
            with install(c['xgpu_pilot']['read_aliases']):pass
            compile(server_code(c,Path(j['root'])),'<formal-server>','exec')

    def test_real_fit_loader_accepts_exact_byte_alias(self):
        from exp.r17_integration.search.fit import load_fitted_recipe
        fit=Path('/data/mw_finetune/eval/ffk1015_dp_libero_full_formal_A_original/assets/cache_inputs/libero_spatial/n50_C10/fit')
        alias=self.fixture/'missing_fit';ready=read(fit/'READY.json')
        with install({str(alias/name):str(fit/name) for name in ('READY.json','fit.npz')}):
            recipe=load_fitted_recipe(alias,expected_binding=ready['receipt']['binding'])
            self.assertEqual(recipe.action.shape,(1255,16,7))
        self.assertFalse(alias.exists())

    def test_renderer_refusal_happens_without_launch_and_no_forbidden_node(self):
        from exp.r17_integration.ops import formal_eval as f
        root=self.fixture/'admission';root.mkdir();c=read(Path(self.jobs[4]['root'])/'campaign.json')
        write(root/'campaign.json',c)
        write(root/'WORKER_DEPLOYMENT.json',{'timan107':dict(worker_run='/own/workers',payload='/payload',python='/python')})
        write(root/'timan107.json',dict(c,node_workers=48,worker_node='timan107',worker_run='/own/workers'))
        with patch.object(f,'remote',return_value=json.dumps(dict(gpu={'admitted':False},ram={'admitted':True}))) as remote:
            with self.assertRaisesRegex(RuntimeError,'no launch claimed'):renderer_admission(root)
            self.assertEqual(remote.call_args.args[0],'timan107')
            self.assertEqual([call.args[0] for call in remote.call_args_list],['timan107'])
        self.assertFalse((root/'LAUNCH.json').exists())

    def test_prepare_seal_detects_mutation(self):
        root=self.fixture/'seal';root.mkdir();(root/'input').write_text('changed')
        write(root/'PREPARED.json',dict(status='PASS',files={'input':'0'*64}))
        with self.assertRaisesRegex(ValueError,'prepared input changed'):sealed(root)

    def test_native_stage_body_with_only_transport_worker_leaves_stubbed(self):
        import inspect
        import shutil
        from exp.r17_integration.ops import formal_eval as f,formal_ports
        from exp.offline_search.rounds.r17.filter_failed import stage
        for job in (self.jobs[0],self.jobs[15]):
            source=Path(job['root']);root=self.fixture/('stage_'+source.name);root.mkdir()
            c=read(source/'campaign.json')
            for name in ('campaign.json','runtime.tar','workers.tar','EPISODE_PLAN.json'):
                shutil.copyfile(source/name,root/name)
            # Assets referenced by the plan remain the sealed study files.
            shutil.copytree(source/'assets',root/'assets')
            targets=[]
            def fake_remote(node,*a,**k):
                targets.append(node);self.assertIn(node,('h100','timan1','timan107'));return '{}'
            def workers(c,d,**kw):
                self.assertEqual(c['execution_worker_nodes'],NODES[job['model']])
                self.assertNotIn('autoscale',c)
                out={}
                for node,count in c['execution_worker_nodes'].items():
                    p='/own/'+c['run_name']+'/'+node
                    out[node]=dict(stage=p,env={},worker_run=p+'/workers')
                    write(Path(d)/(node+'.json'),dict(c,node_workers=count,worker_node=node,worker_run=p+'/workers'))
                    write(Path(d)/(node+'_MEMORY_ADMISSION.json'),dict(admitted=True))
                return out
            with patch.object(f,'remote',fake_remote),patch.object(f,'transfer',lambda *a,**k:None),patch.object(f,'prepare_workers',workers),patch.object(formal_ports,'prepare_admission',lambda c:None):
                with context(root):inspect.unwrap(stage._stage)(root)
            sealed(root)
            self.assertFalse((root/'LAUNCH.json').exists())
            if job['host']=='h100':self.assertIn('h100',targets)

    def test_dp_cpu_replay_explains_all_budget_failures(self):
        r=read(W/'evidence/dp_cache_localisation_1018.json')
        self.assertEqual((r['failed_logical_requests'],r['recordings']),(51,408))
        for row in r['rows']:
            self.assertTrue(row['donors_all_identical'])
            self.assertEqual(row['max_reported_correction'],0)
            self.assertEqual(row['max_weight_replay_error'],0)
            self.assertLessEqual(row['cross_delta_reconstruction_error'],2.4e-7)
            self.assertLessEqual(row['max_decode_residual'],2.4e-7)
            self.assertTrue(all(v['decision']==0 for v in row['recordings']))

if __name__=='__main__':unittest.main()
