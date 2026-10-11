"""CPU regressions for warm deployment, one-shot launch and queued preparation."""
from argparse import Namespace
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from unittest import mock

from exp.r17_integration.ops import formal_eval as launch, formal_prestage as stage, formal_queue as queue
from exp.r17_integration.ops import immutable_release as release
from exp.r17_integration.ops.formal_process import write_json


class PreparedTests(unittest.TestCase):
    def archive(self,path,body=b'original',mtime=0):
        with tarfile.open(path,'w') as tar:
            member=tarfile.TarInfo('module.py');member.size=len(body);member.mtime=mtime
            tar.addfile(member,io.BytesIO(body))
        return release.identity(release.archive_manifest(path))

    def test_tar_metadata_changes_reuse_same_readonly_runtime_without_rewrite(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);pin=self.archive(p/'first.tar')
            first=release.install(p/'first.tar',p/'runtime',release.sha(p/'first.tar'),pin)
            before=(p/'runtime/module.py').stat()
            self.assertFalse(first['reused'])
            second_pin=self.archive(p/'second.tar',mtime=999)
            self.assertEqual(second_pin,pin)
            self.assertNotEqual(release.sha(p/'first.tar'),release.sha(p/'second.tar'))
            second=release.install(p/'second.tar',p/'runtime',release.sha(p/'second.tar'),pin)
            self.assertTrue(second['reused'])
            self.assertEqual((p/'runtime/module.py').stat().st_mtime_ns,before.st_mtime_ns)
            self.assertEqual((p/'runtime/module.py').stat().st_ino,before.st_ino)
            self.assertEqual((p/'runtime/module.py').stat().st_mode & 0o222,0)

    def test_corrupt_runtime_is_refused_not_repaired_over_live_release(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);pin=self.archive(p/'runtime.tar')
            release.install(p/'runtime.tar',p/'runtime',release.sha(p/'runtime.tar'),pin)
            f=p/'runtime/module.py';f.chmod(0o644);f.write_text('corrupt')
            with self.assertRaisesRegex(ValueError,'immutable release changed'):
                release.install(p/'runtime.tar',p/'runtime',release.sha(p/'runtime.tar'),pin)
            self.assertEqual(f.read_text(),'corrupt')

    def test_release_bootstrap_and_verifier_support_python310_model_islands(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);pin=self.archive(p/'runtime.tar')
            with mock.patch.object(hashlib,'file_digest',side_effect=AssertionError('unavailable on3.10')):
                release.install(p/'runtime.tar',p/'runtime',release.sha(p/'runtime.tar'),pin)
            code='import hashlib; delattr(hashlib,"file_digest"); '+stage.verify_code(p/'runtime','RUNTIME_RELEASE.json',pin)
            result=json.loads(subprocess.check_output([sys.executable,'-c',code],text=True))
            self.assertTrue(result['present'])

    def test_client_cache_hit_checks_actual_bytes_and_skips_all_pushes(self):
        from exp.r17_integration.ops.build_client_bundle import build_client_bundle
        from exp.r17_integration.ops.worker_installs import install_client
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'worker.py').write_text('cpu only')
            payload=build_client_bundle(p,['worker.py'],p/'workers.tar')
            install_client(p/'workers.tar',p/'installed',payload['archive_sha256'])
            calls=[]
            def remote(node,argv,**kwargs):
                calls.append(argv)
                return subprocess.check_output([sys.executable,'-c',argv[-1]],text=True)
            fake=Namespace(remote=remote,transfer=mock.Mock(),ROOT=p)
            result=stage.client_release(fake,'timan107',p,'/tmp/unused',str(p/'installed'),payload)
            self.assertTrue(result['present']);self.assertTrue(result['reused'])
            fake.transfer.assert_not_called()
            self.assertEqual(len(calls),1)

    def fixture(self,p):
        (p/'assets').mkdir();write_json(p/'assets/model_spec.json',{'identity':'sealed'})
        campaign=dict(runtime='/immutable',runtime_identity='identity',local_assets=str(p/'assets'))
        for n,value in (('campaign.json',campaign),('PREFLIGHT.json',{'status':'PASS'}),
                        ('EPISODE_PLAN.json',{'tasks':[]}),('WORKER_DEPLOYMENT.json',{})):
            write_json(p/n,value)
        stage.seal_prepared(campaign,p,{})

    def test_asset_or_config_changes_fail_before_dispatch_and_launch_is_one_shot(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);self.fixture(p)
            stage.validate_prepared(p)
            (p/'assets/model_spec.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'content changed'):
                stage.validate_prepared(p)
            self.fixture_without_mkdir(p)
            write_json(p/'LAUNCH.json',{'pid':1})
            with self.assertRaisesRegex(ValueError,'already launched'):
                stage.validate_prepared(p)

    def fixture_without_mkdir(self,p):
        (p/'assets/model_spec.json').write_text('{"identity": "sealed"}\n')
        c=json.loads((p/'campaign.json').read_text());stage.seal_prepared(c,p,{})

    def test_prepared_launch_rejects_cli_overrides(self):
        good=launch.parse_args(['--run','--prepared-run','/prepared'])
        self.assertEqual(good.prepared_run,Path('/prepared'))
        for argv in (['--prepare-only'],['--run','--workers','4'],['--run','--max-batch','8']):
            with self.assertRaises(SystemExit):launch.parse_args([*argv,'--prepared-run','/prepared'])

    def test_cpu_preparation_allocates_no_cosmos_tunnels(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(launch,'FORMAL_ROOT',Path(temp)), \
                mock.patch.object(launch,'prepare',return_value={}),mock.patch.object(launch,'open_tunnels') as tunnels, \
                mock.patch.object(launch,'run_campaign') as run:
            self.assertEqual(launch.main(['--model','cosmos_libero','--prepare-only','--run-name','cosmos_cpu']),0)
            tunnels.assert_not_called();run.assert_not_called()

    def test_defaults_preserve_measured_recycling_and_renderer_spread(self):
        for model,width in (('dp_libero',4),('fastwam_libero',2),('cosmos_libero',1)):
            args=launch.parse_args(['--model',model,'--replicas','4','--prepare-only'])
            self.assertEqual((args.worker_max_episodes,args.worker_memory_mib),(4,6144))
            self.assertEqual(args.replica_startup_width,width)
            self.assertIsNone(args.worker_egl_devices)
            self.assertEqual(args.worker_vram_mib,1024)
        args=launch.parse_args(['--model','dp_mw','--prepare-only'])
        self.assertEqual(args.worker_max_episodes,0)

    def test_real_prepared_boundary_reuses_cpu_seal_and_claims_only_once(self):
        with tempfile.TemporaryDirectory() as temp:
            parent=Path(temp);p=parent/'prepared';p.mkdir();self.fixture(p)
            pin=self.archive(parent/'runtime.tar')
            release.install(parent/'runtime.tar',parent/'runtime',release.sha(parent/'runtime.tar'),pin)
            c=json.loads((p/'campaign.json').read_text())
            c.update(runtime=str(parent/'runtime'),runtime_identity=pin,server_node='local',model='dp',
                run_name='prepared',endpoint='unused:1',endpoint_host='unused',public_port=1,
                prepare_only=True,worker_nodes={})
            write_json(p/'campaign.json',c);stage.seal_prepared(c,p,{})
            with mock.patch.object(launch,'FORMAL_ROOT',parent), \
                 mock.patch('exp.r17_integration.ops.formal_driver.plan',return_value=(None,None,[],None)), \
                 mock.patch.object(launch,'prepare') as prepare,mock.patch.object(launch,'run_campaign',return_value=0) as run:
                self.assertEqual(stage.launch_prepared(p),0)
                prepare.assert_not_called();run.assert_called_once()
                self.assertFalse(run.call_args.args[0]['prepare_only'])
                with self.assertRaisesRegex(ValueError,'already launched'):stage.launch_prepared(p)

    def test_four_residents_spawn_before_first_readiness_wait_with_owned_cleanup(self):
        from exp.r17_integration.ops.tests.test_formal_eval import FormalCampaignTests
        from exp.r17_integration.common.websocket_policy_server import WebsocketClientPolicy
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);c=FormalCampaignTests().startup_campaign(p,'local')
            base=c['backend_ports'][0]
            c.update(model='dp',backend_ports=list(range(base,base+4)),replica_startup_width=4,
                local_assets=str(p),memory_budget_gib=14,ports=[1,2])
            write_json(p/'model_spec.json',{})
            children=[mock.Mock(pid=pid,poll=mock.Mock(return_value=None),wait=mock.Mock(return_value=0)) for pid in range(100,105)]
            popen=mock.Mock(side_effect=children)
            sockets=[];worker_spawns=[]
            deployment={'timan107':dict(python=sys.executable,stage='/tmp/cpu_owned_fleet',env={},
                payload=str(launch.ROOT),worker_run='/tmp/cpu_owned_fleet/run')}
            def remote(node,argv,**kwargs):
                mode=argv[argv.index('exp.r17_integration.ops.formal_process')+1]
                if mode=='spawn':
                    self.assertEqual(popen.call_count,0)
                    worker_spawns.append(mode)
                    return json.dumps(dict(status='RUNNING',child=dict(pid=999,pgid=999,start_ticks='cpu')))
                self.assertEqual(mode,'stop')
                return json.dumps(dict(alive=False))
            def connect(address,**kwargs):
                sockets.append(address)
                if len(sockets)==1:
                    self.assertEqual(popen.call_count,4)
                    self.assertEqual(worker_spawns,['spawn'])
                if len(sockets)==5:raise RuntimeError('CPU stop before dispatch')
                return mock.MagicMock()
            def execute(argv,**kwargs):
                return '\n'.join(str(pid)+', 1' for pid in range(100,104)) if argv[0]=='nvidia-smi' else ''
            with mock.patch.object(launch,'host_python',return_value=sys.executable), \
                 mock.patch.object(launch,'server_env',return_value={}), \
                 mock.patch.object(launch,'execute',side_effect=execute),mock.patch.object(launch.subprocess,'Popen',popen), \
                 mock.patch.object(launch,'remote',side_effect=remote), \
                 mock.patch.object(launch,'refresh_remote_owned',return_value=0), \
                 mock.patch.object(launch,'ticks',return_value=1),mock.patch.object(launch,'alive',return_value=False), \
                 mock.patch.object(launch,'stop_group',return_value={'alive':False}), \
                 mock.patch.object(launch.socket,'socket'),mock.patch.object(launch.socket,'create_connection',side_effect=connect), \
                 mock.patch.object(WebsocketClientPolicy,'__init__',return_value=None), \
                 mock.patch.object(WebsocketClientPolicy,'close'), \
                 mock.patch.object(WebsocketClientPolicy,'get_server_metadata',return_value={'r17_model_spec':{},'cuda_graph_self_check':{'passed':True}}):
                self.assertEqual(launch.run_campaign(c,p,deployment),2)
            report=json.loads((p/'RECEIPT.json').read_text())
            self.assertTrue(report['all_owned_closed'])
            self.assertEqual(len(json.loads((p/'SERVER_ATTESTATIONS.json').read_text())),4)
            self.assertTrue(all('--cuda-graph' in e['argv'] for e in report['owned_local'][:4]))

    def test_deferred_control_must_close_before_any_gpu_launch(self):
        from exp.r17_integration.models.fastwam import formal_batching
        campaign=dict(model='fastwam',domain='libero',prepare_only=False,local_assets='/unused',
                      fastwam_batching={'max_batch':8})
        with mock.patch.object(formal_batching,'check_control',side_effect=ValueError('control not closed')):
            with self.assertRaisesRegex(ValueError,'control not closed'):
                stage.control_gate(campaign,Path('/unused'))


class QueueTests(unittest.TestCase):
    def jobs(self,p):
        return [dict(name=n,argv=['--run-name',n],args=Namespace(paired_control=None,seconds=60),run=str(p/n)) for n in ('control','arm')]

    def test_next_cpu_prepare_overlaps_current_run_and_switch_requires_closed_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);jobs=self.jobs(p);events=[];started=threading.Event();prepared=threading.Event()
            class FakeRunner:
                stopped=False
                def invoke(self,job,kind,argv):
                    events.append((job['name'],kind))
                    path=Path(job['run']);path.mkdir(exist_ok=True)
                    if kind=='prepare' and job['name']=='arm':
                        self.assertion=started.wait(2);prepared.set()
                    elif kind=='run' and job['name']=='control':
                        started.set()
                        if not prepared.wait(2):raise AssertionError('no overlap')
                    if kind=='run':write_json(path/'RECEIPT.json',dict(status='PASS',all_owned_closed=True))
                    return str(path)
            runner=FakeRunner()
            with mock.patch.object(stage,'validate_prepared'):
                result=queue.run_jobs(jobs,runner)
            self.assertTrue(runner.assertion);self.assertEqual(len(result),2)
            self.assertLess(events.index(('arm','prepare')),events.index(('arm','run')))

    def test_no_gpu_switch_after_failed_or_unclosed_campaign(self):
        for receipt in ({'status':'FAIL','all_owned_closed':True},{'status':'PASS','all_owned_closed':False}):
            with self.subTest(receipt=receipt),tempfile.TemporaryDirectory() as temp:
                p=Path(temp);jobs=self.jobs(p);calls=[]
                class FakeRunner:
                    stopped=False
                    def invoke(self,job,kind,argv):
                        calls.append((job['name'],kind));path=Path(job['run']);path.mkdir(exist_ok=True)
                        if kind=='run':write_json(path/'RECEIPT.json',receipt)
                        return str(path)
                with mock.patch.object(stage,'validate_prepared'),self.assertRaisesRegex(RuntimeError,'refuses switch'):
                    queue.run_jobs(jobs,FakeRunner())
                self.assertNotIn(('arm','run'),calls)

    def test_queue_rejects_gpu_mixing_shell_text_and_forward_controls(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'jobs.json'
            for value in ('shell command', [['--run-name','a'],['--run-name','b','--server-node','h100']],
                          [['--run-name','a','--paired-control','@missing']], [['--run-name','a','--run']]):
                p.write_text(json.dumps(value))
                with self.assertRaises(ValueError):queue.jobs_from_file(p)
            p.write_text(json.dumps([['--run-name','control','--model','fastwam_libero','--server-node','h100','--max-batch','8'],
                ['--run-name','arm','--model','fastwam_libero','--server-node','h100','--max-batch','8','--paired-control','@control']]))
            jobs=queue.jobs_from_file(p)
            self.assertIn(str(launch.FORMAL_ROOT/'control'),jobs[1]['argv'])
