"""CPU-only regressions of the 10-10 OOM and overlapping worker deployments."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from exp.r17_integration.ops import formal_eval, worker_installs, worker_memory
from exp.r17_integration.ops.formal_process import alive, ticks, write_json

GIB = 1024**3
ROOT = formal_eval.ROOT


class ConcurrentFleetTests(unittest.TestCase):
    def test_node_admission_accounts_for_running_legacy_fleet_and_startup_reservations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = dict(status='RUNNING', workers=24, pids=[dict(pid=i) for i in range(24)])
            write_json(root/'legacy/receipt.json', old)
            rss = lambda entry: 2*GIB if entry['pid'] < 24 else None
            campaign = dict(domain='libero', node_workers=24)
            refused = worker_memory.admission(campaign,root/'second', memory=(220*GIB,115*GIB),rss_reader=rss)
            self.assertFalse(refused['admitted'])
            self.assertEqual(refused['other_pending_bytes'],48*GIB)
            self.assertEqual(refused['requested_bytes'],96*GIB)
            self.assertIn('reduce --workers',refused['reason'])
            self.assertTrue(worker_memory.admission(dict(campaign,node_workers=8),root/'second',
                memory=(220*GIB,115*GIB),rss_reader=rss)['admitted'])
            # A receipt published under the node lock reserves even children
            # not launched yet. A closed fleet and a reused PID reserve nothing.
            write_json(root/'legacy/receipt.json',dict(old,status='PASS'))
            write_json(root/'starting/receipt.json',dict(status='RUNNING',workers=24,pids=[],
                fleet_owner=dict(pid=0),worker_memory_mib=4096))
            self.assertFalse(worker_memory.admission(campaign,root/'second',
                memory=(220*GIB,160*GIB),rss_reader=rss)['admitted'])
            self.assertTrue(worker_memory.admission(campaign,root/'second',
                memory=(220*GIB,160*GIB),rss_reader=lambda p:None)['admitted'])

    def test_replacement_preserves_own_remaining_worker_reservations(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)/'own'
            write_json(run/'receipt.json',dict(status='RUNNING',workers=24,
                fleet_owner=dict(pid=100),pids=[dict(pid=i) for i in range(23)],worker_memory_mib=4096))
            rss = lambda p: 2*GIB if p['pid'] < 23 else GIB
            decision = worker_memory.admission(dict(domain='libero',node_workers=1),run,
                memory=(220*GIB,40*GIB),rss_reader=rss,replacement=True)
            self.assertFalse(decision['admitted'])
            self.assertEqual(decision['other_pending_bytes'],46*GIB)
            self.assertTrue(decision['other_fleets'][0]['own_fleet'])

    def test_rss_requires_pid_birth_identity(self):
        child = subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'])
        try:
            entry = dict(pid=child.pid,start_ticks=ticks(child.pid))
            self.assertIsNotNone(worker_memory.rss(entry))
            self.assertIsNone(worker_memory.rss(dict(entry,start_ticks='reused')))
        finally:
            child.terminate(); child.wait(timeout=5)

    def test_cli_exposes_validated_node_ram_and_bounded_restart_controls(self):
        args = formal_eval.parse_args(['--prepare-only'])
        self.assertEqual((args.worker_memory_mib,args.worker_restart_limit),(6144,2))
        mw = formal_eval.parse_args(['--prepare-only','--model','dp_mw'])
        self.assertEqual(mw.worker_memory_mib,2048)
        for flag,value in (('--worker-memory-mib','0'),('--worker-restart-limit','-1')):
            with self.assertRaises(SystemExit):formal_eval.parse_args(['--prepare-only',flag,value])

    def test_paths_configs_and_b_inputs_are_campaign_private(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'original.yaml'; source.write_text('assets: /original\n')
            a = worker_installs.private_config(source,root/'a/config')
            b = worker_installs.private_config(source,root/'b/config')
            self.assertNotEqual(a['path'],b['path'])
            (root/'b/config/config.yaml').write_text('assets: /different\n')
            self.assertEqual((root/'a/config/config.yaml').read_text(),source.read_text())
            for node in ('timan107','timan108'):
                first = formal_eval.worker_paths(dict(run_name='first'),node)
                second = formal_eval.worker_paths(dict(run_name='second'),node)
                self.assertFalse(set(first.values()) & set(second.values()))
                self.assertTrue(first['stage'].startswith('/tmp/'))
            for bad in ('../other','','.','..','a/b'):
                with self.assertRaises(ValueError):formal_eval.worker_paths(dict(run_name=bad),'timan107')
            # Exercise the actual launcher path construction for identical B
            # archives on two campaigns, without touching either worker node.
            for name in ('first','second'):
                camp = dict(run_name=name,domain='libero',worker_nodes={'timan107':1},worker_egl_device=0,
                    local_assets=str(root),library=True,bpool_archive='B.tar',
                    bpool_transfer=dict(archive_sha256='same',manifest={}))
                commands = []
                def remote(node,argv,**kw):
                    commands.append(argv)
                    code = argv[-1]
                    if 'private_config' in code:return '{"path":"/private/config"}'
                    if 'worker_memory import admission' in code:return '{"admitted":true}'
                    if 'install_init_archive' in code:return '{"manifest":{},"directories":{}}'
                    return '{}'
                with mock.patch.object(formal_eval,'remote',side_effect=remote), \
                     mock.patch('exp.r17_integration.ops.worker_disk.probe',return_value={'admitted':True}), \
                     mock.patch.object(formal_eval,'transfer'), \
                     mock.patch.object(formal_eval,'install_worker_dependencies',return_value=dict(
                        private_dependencies='/immutable/deps',config_source='/original/config.yaml')), \
                     mock.patch('exp.r17_integration.ops.build_client_bundle.worker_payload_files',return_value=[]), \
                     mock.patch('exp.r17_integration.ops.build_client_bundle.build_client_bundle',
                        return_value=dict(archive_sha256='same')):
                    formal_eval.prepare_workers(camp,root)
                installer = next(c[-1] for c in commands if 'install_init_archive' in c[-1])
                self.assertIn(formal_eval.worker_paths(camp,'timan107')['private']+'/bpool_same',installer)

    def test_immutable_dependency_publication_is_atomic_and_does_not_rewrite_a_live_release(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'source'
            package = source/'libero/libero'; package.mkdir(parents=True)
            module = package/'__init__.py'; module.write_text('VERSION=1\n')
            assets = root/'assets'; assets.mkdir()
            config = root/'original.yaml'; config.write_text('assets: '+str(assets)+'\n')
            store = root/'deps'
            def install():return worker_installs.install_dependencies(source,store,config)
            with ThreadPoolExecutor(max_workers=2) as pool:
                first,second = list(pool.map(lambda _:install(),range(2)))
            self.assertEqual(first,second)
            old = Path(first['private_dependencies']); marker = old/'DEPENDENCIES.json'
            before = (marker.read_bytes(),marker.stat().st_mtime_ns,(old/'libero/libero/__init__.py').stat().st_mtime_ns)
            self.assertEqual(install(),first)
            self.assertEqual(before,(marker.read_bytes(),marker.stat().st_mtime_ns,
                                     (old/'libero/libero/__init__.py').stat().st_mtime_ns))
            self.assertFalse(stat.S_IMODE(old.stat().st_mode)&0o222)
            self.assertEqual((old/'libero/libero/assets').resolve(),assets)
            module.write_text('VERSION=2\n')
            self.assertNotEqual(install()['private_dependencies'],str(old))
            self.assertEqual((old/'libero/libero/__init__.py').read_text(),'VERSION=1\n')
            # tempfile cleanup needs write permission only on these owned CPU fixtures.
            for path in store.rglob('*'):
                if path.is_dir() and not path.is_symlink():path.chmod(0o755)

    def test_immutable_payload_bootstrap_reuses_without_rewriting_and_imports_cpu_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            formal_eval.prepare_workers(dict(worker_nodes={}),root)
            from exp.r17_integration.ops.build_client_bundle import sha256
            destination = root/'client'; archive = root/'workers.tar'
            command = [sys.executable,str(ROOT/'exp/r17_integration/qualification/bootstrap_install.py'),
                       str(archive),str(destination),sha256(archive),'--immutable']
            first = subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(first.returncode,0,first.stderr)
            before = (destination/'PAYLOAD.json').stat().st_mtime_ns
            second = subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(second.returncode,0,second.stderr)
            self.assertEqual(before,(destination/'PAYLOAD.json').stat().st_mtime_ns)
            code = ('import sys; sys.path.insert(0,'+repr(str(destination))+'); '
                'from exp.r17_integration.ops import formal_workers,worker_memory,worker_installs; '
                'assert not any(m in sys.modules for m in ("torch","jax","mujoco","lerobot","gr00t"))')
            result = subprocess.run([sys.executable,'-I','-c',code],cwd=destination,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            for path in destination.rglob('*'):
                if path.is_dir():path.chmod(0o755)
            destination.chmod(0o755)

    def fleet(self,root,name,*,fail=False,restart_limit=2,refuse=False,seconds=25):
        campaign = dict(run_name=name,worker_node='cpu',worker_run=str(root/'runs'/name),
                        domain='libero',node_workers=2,seconds=seconds,worker_restart_limit=restart_limit)
        path = root/(name+'.json'); write_json(path,campaign)
        fake = root/'worker.py'
        if not fake.exists():
            fake.write_text('''import json,os,signal,sys,time
from pathlib import Path
workspace=Path.cwd(); run=workspace.parents[2]
index=sys.argv[1]; generation=workspace.name.split('_')[-1]
failure=sys.argv[4]=='True' and index=='0' and generation=='0'
marker=run/('started_'+index+'_'+generation)
marker.write_text(json.dumps(dict(pid=os.getpid(),cwd=str(workspace),
    tmp=os.environ['TMPDIR'],cache=os.environ['XDG_CACHE_HOME'],hf=os.environ['HF_HOME'])))
release=Path(sys.argv[3] if failure else sys.argv[2])
while not release.exists(): time.sleep(.02)
if failure: os.kill(os.getpid(),signal.SIGKILL)
''')
        harness = f'''import sys
from pathlib import Path
from exp.r17_integration.ops import formal_workers as f,worker_memory as m
m.admission=lambda *a,**kw: {dict(admitted=not refuse,reason='CPU refusal')!r}
root=Path({str(root)!r})
def command(c,i):
    return c['run_name']+str(i),[sys.executable,str(root/'worker.py'),str(i),
        str(root/(c['run_name']+'_release')),str(root/(c['run_name']+'_fail')),{str(fail)!r}]
f.worker_command=command
sys.argv=['cpu','--campaign',{str(path)!r}]
f.main()
'''
        env = {**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(ROOT), 'PYTHONDONTWRITEBYTECODE':'1'}
        log = (root/(name+'.log')).open('wb')
        return subprocess.Popen([sys.executable,'-c',harness],cwd=ROOT,env=env,stdout=log,stderr=log),log

    def wait_for(self,predicate,seconds=10):
        until = time.monotonic()+seconds
        while not predicate() and time.monotonic()<until:time.sleep(.02)
        self.assertTrue(predicate(),'CPU fleet condition timed out')

    def test_two_fleets_one_worker_sigkill_is_replaced_and_other_fleet_remains_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first,log1 = self.fleet(root,'first',fail=True)
            second,log2 = self.fleet(root,'second')
            try:
                self.wait_for(lambda:all((root/'runs'/name/'started_1_0').exists() for name in ('first','second')))
                before = json.loads((root/'runs/second/receipt.json').read_text())
                healthy = json.loads((root/'runs/first/receipt.json').read_text())['pids'][1]
                (root/'first_fail').touch()
                self.wait_for(lambda:(root/'runs/first/started_0_1').exists())
                self.assertTrue(alive(healthy))
                (root/'first_release').touch()
                self.assertEqual(first.wait(timeout=12),0,(root/'first.log').read_text())
                report = json.loads((root/'runs/first/receipt.json').read_text())
                self.assertEqual(report['status'],'PASS')
                self.assertEqual([f['exit_code'] for f in report['worker_failures']],[-9])
                self.assertEqual(len(report['pids']),3)
                self.assertTrue(all(not alive(p) for p in report['pids']))
                self.assertIsNone(second.poll())
                self.assertTrue(all(alive(p) for p in before['pids']))
                self.assertFalse({p['pid'] for p in before['pids']} & {p['pid'] for p in report['pids']})
                spaces = []
                for entry in report['pids']:
                    data = json.loads((root/'runs/first'/f"started_{entry['index']}_{entry['generation']}").read_text())
                    spaces.append(data['cwd'])
                    self.assertEqual(data['cwd'],data['tmp'])
                    self.assertEqual(data['cache'],data['cwd']+'/cache')
                    self.assertEqual(data['hf'],data['cwd']+'/hf_cache')
                self.assertEqual(len(set(spaces)),3)
                (root/'second_release').touch()
                self.assertEqual(second.wait(timeout=12),0,(root/'second.log').read_text())
                self.assertEqual(json.loads((root/'runs/second/receipt.json').read_text())['status'],'PASS')
                self.assertTrue((root/'runs/first.transfer.json').exists())
            finally:
                for child,log in ((first,log1),(second,log2)):
                    if child.poll() is None:child.terminate();child.wait(timeout=12)
                    log.close()

    def test_admission_refusal_spawns_nothing_and_real_failure_stays_partial(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            child,log = self.fleet(root,'refused',refuse=True)
            try:self.assertEqual(child.wait(timeout=12),2)
            finally:log.close()
            report = json.loads((root/'runs/refused/receipt.json').read_text())
            self.assertEqual(report['pids'],[])
            self.assertEqual(report['status'],'PARTIAL')
            child,log = self.fleet(root,'fatal',fail=True,restart_limit=0)
            try:
                self.wait_for(lambda:(root/'runs/fatal/started_1_0').exists())
                (root/'fatal_fail').touch()
                self.assertEqual(child.wait(timeout=12),2)
                report = json.loads((root/'runs/fatal/receipt.json').read_text())
                self.assertEqual(report['reason'],'worker_restart_budget_exhausted')
                self.assertTrue(all(not alive(p) for p in report['pids']))
                self.assertEqual(report['pids'][1]['exit_code'],-signal.SIGTERM)
            finally:
                if child.poll() is None:child.terminate();child.wait(timeout=12)
                log.close()

    def test_failed_fleet_cleanup_terminates_only_its_own_children(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first,log1 = self.fleet(root,'fatal_owner',fail=True,restart_limit=0)
            second,log2 = self.fleet(root,'survivor')
            try:
                self.wait_for(lambda:all((root/'runs'/name/'started_1_0').exists()
                                        for name in ('fatal_owner','survivor')))
                witness = json.loads((root/'runs/survivor/receipt.json').read_text())
                (root/'fatal_owner_fail').touch()
                self.assertEqual(first.wait(timeout=12),2,(root/'fatal_owner.log').read_text())
                self.assertIsNone(second.poll())
                self.assertTrue(all(alive(p) for p in witness['pids']))
                report = json.loads((root/'runs/fatal_owner/receipt.json').read_text())
                self.assertTrue(all(not alive(p) for p in report['pids']))
                self.assertEqual(report['pids'][1]['exit_code'],-signal.SIGTERM)
                (root/'survivor_release').touch()
                self.assertEqual(second.wait(timeout=12),0,(root/'survivor.log').read_text())
            finally:
                for child,log in ((first,log1),(second,log2)):
                    if child.poll() is None:child.terminate();child.wait(timeout=12)
                    log.close()
            child,log = self.fleet(root,'deadline',seconds=1)
            try:
                self.assertEqual(child.wait(timeout=12),2)
                report = json.loads((root/'runs/deadline/receipt.json').read_text())
                self.assertEqual(report['reason'],'worker_deadline')
                self.assertTrue(all(not alive(p) for p in report['pids']))
            finally:
                if child.poll() is None:child.terminate();child.wait(timeout=12)
                log.close()

    def test_failed_fleet_diagnostics_reads_only_the_exact_campaign_receipt(self):
        item = dict(worker_run='/owned/campaign_a',python='/cpu/python')
        with mock.patch.object(formal_eval,'remote',return_value=json.dumps(dict(
            reason='worker_restart_budget_exhausted',worker_failures=[dict(pid=123,exit_code=-9)]))) as remote:
            result = formal_eval.failed_fleet('timan107',item,dict(exit_code=2))
        self.assertEqual(result['fleet_receipt']['worker_failures'][0]['exit_code'],-9)
        self.assertIn('/owned/campaign_a/receipt.json',remote.call_args.args[1][-1])
        self.assertTrue(remote.call_args.kwargs['replay_safe'])


if __name__ == '__main__':
    unittest.main()
