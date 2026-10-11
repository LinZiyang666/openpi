"""CPU-only frozen baseline and queue regressions (no CUDA/model loading)."""
from argparse import Namespace
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

from exp.r17_integration.ops import formal_eval as launch, formal_prestage as stage, formal_queue as queue
from exp.r17_integration.ops import formal_runtime as frozen, immutable_release as release
from exp.r17_integration.ops.build_client_bundle import build_client_bundle
from exp.r17_integration.ops.formal_process import write_json


class FrozenRuntimeTests(unittest.TestCase):
    def fixture(self,p):
        source=p/'control';source.mkdir();(source/'assets').mkdir()
        code=p/'checkout';code.mkdir();(code/'model.py').write_text('BASELINE = True\n')
        archive=source/'runtime.tar'
        with tarfile.open(archive,'w') as tar:tar.add(code/'model.py',arcname='model.py')
        pin=release.identity(release.archive_manifest(archive));root=p/'release'
        release.install(archive,root,release.sha(archive),pin)
        campaign=dict(model='dp',domain='libero',checkpoint='/immutable/weights',server_node='local',
            official_50=True,smoke=False,runtime=str(root),runtime_identity=pin)
        write_json(source/'campaign.json',campaign)
        write_json(source/'RUNTIME_RELEASE.json',dict(identity=pin,root=str(root),node='local',archive_sha256=release.sha(archive)))
        payload=build_client_bundle(code,['model.py'],source/'workers.tar')
        write_json(source/'WORKER_DEPLOYMENT.json',{'timan107':dict(installed=payload['manifest'])})
        spec={'checkpoint_identity':'baseline'}
        for name,doc in [('model_spec.json',spec),('model_manifest.json',dict(spec=spec,content_digests={'code:model':'original'})),
                         ('benchmark.json',{'protocol':'original'})]:write_json(source/'assets'/name,doc)
        return source,code,campaign

    def test_checkout_edit_and_new_mps_file_cannot_change_frozen_server_or_worker_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);source,code,campaign=self.fixture(p)
            frozen.attach(campaign,source)
            (code/'model.py').write_text('CHANGED = True\n');(code/'mps.py').write_text('NEW = True\n')
            destination=p/'arm';destination.mkdir()
            frozen.copy_runtime(campaign,destination/'runtime.tar')
            self.assertEqual((destination/'runtime.tar').read_bytes(),(source/'runtime.tar').read_bytes())
            payload=frozen.worker_payload(campaign,destination)
            self.assertEqual(payload['archive_sha256'],release.sha(source/'workers.tar'))
            self.assertNotIn('mps.py',release.archive_manifest(destination/'runtime.tar')['files'])
            self.assertEqual((source/'campaign.json').read_text(),json.dumps(
                {k:v for k,v in campaign.items() if not k.startswith('runtime_source') and k!='runtime_from'},indent=2,sort_keys=True)+'\n')

    def test_missing_tampered_or_unbound_archive_is_refused(self):
        for change in ('missing','archive','release','worker'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temp:
                p=Path(temp);source,_,campaign=self.fixture(p)
                if change=='missing':(source/'runtime.tar').unlink()
                elif change=='archive':
                    with (source/'runtime.tar').open('ab') as stream:stream.write(b'changed')
                elif change=='release':
                    r=frozen.read(source/'RUNTIME_RELEASE.json');r['identity']='wrong';write_json(source/'RUNTIME_RELEASE.json',r)
                else:
                    d=frozen.read(source/'WORKER_DEPLOYMENT.json');d['timan107']['installed']['files']['model.py']='wrong'
                    write_json(source/'WORKER_DEPLOYMENT.json',d)
                with self.assertRaises((ValueError,FileNotFoundError)):frozen.attach(campaign,source)

    def test_checkpoint_bundle_batching_and_mps_mismatch_are_refused(self):
        for key,value in [('checkpoint','other'),('model','fastwam'),('server_node','h100'),
                          ('dp_batching',{'max_batch':8}),('mps',{'enabled':True})]:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as temp:
                p=Path(temp);source,_,campaign=self.fixture(p);campaign[key]=value
                with self.assertRaisesRegex(ValueError,'campaign mismatch: '+key):frozen.attach(campaign,source)

    def test_changed_source_or_vendor_model_seal_fails_before_dispatch(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);source,_,campaign=self.fixture(p);frozen.attach(campaign,source)
            campaign.update(local_assets=str(source/'assets'),coordinator_runtime='/verified')
            frozen.check_assets(campaign,p)
            write_json(source/'assets/model_spec.json',{'changed':True})
            with self.assertRaisesRegex(ValueError,'source changed'):frozen.check_pins(campaign)

    def test_frozen_execution_release_must_exist_even_when_archive_is_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);source,_,campaign=self.fixture(p);frozen.attach(campaign,source)
            campaign['run_name']='arm';directory=p/'arm';directory.mkdir()
            fake=Namespace(ROOT=p)
            with mock.patch.object(stage,'verify',return_value={'present':False}), \
                 mock.patch.object(stage,'install') as install:
                with self.assertRaisesRegex(ValueError,'execution-host runtime missing'):
                    stage.runtime_release(fake,campaign,directory)
                install.assert_not_called()

    def test_running_control_status_is_not_changed_or_required_closed_for_preparation(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);source,_,campaign=self.fixture(p)
            write_json(source/'RECEIPT.json',dict(status='RUNNING',all_owned_closed=False))
            before=(source/'RECEIPT.json').read_bytes()
            frozen.attach(campaign,source)
            self.assertEqual((source/'RECEIPT.json').read_bytes(),before)

    def test_native_seal_cwd_imports_frozen_code_not_checkout(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);source,code,campaign=self.fixture(p)
            (code/'model.py').write_text('BASELINE = False\n')
            output=launch.execute([sys.executable,'-c','import model; print(model.BASELINE)'],
                env={'PYTHONPATH':str(code)},cwd=campaign['runtime'])
            self.assertEqual(output.strip(),'True')

    def test_queue_automatically_pins_each_paired_arm_and_dp_shim_forwards_runtime_from(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'jobs.json'
            write_json(p,[['--run-name','arm','--model','fastwam_libero','--max-batch','8',
                           '--paired-control','/control','--commit-horizon','20']])
            job=queue.jobs_from_file(p)[0]
            self.assertEqual(job['argv'][-2:],['--runtime-from','/control'])
            from exp.r17_integration.models.dp import formal_launch
            with mock.patch.object(launch,'main',return_value=0) as main:
                self.assertEqual(formal_launch.main(['--model','dp_libero','--max-batch','16',
                    '--runtime-from','/dp_control','--prepare-only']),0)
                self.assertIn('--runtime-from',main.call_args.args[0])

    def test_dp_batching_hook_populates_config_before_frozen_binding(self):
        from exp.r17_integration.models.dp import formal_launch
        from exp.r17_integration.models.dp.batch_runtime import BatchConfig
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);source,_,campaign=self.fixture(p)
            campaign['dp_batching']=BatchConfig(16,5).metadata()
            write_json(source/'campaign.json',campaign)
            with mock.patch.object(launch,'FORMAL_ROOT',p), \
                 mock.patch.object(stage,'runtime_release',side_effect=RuntimeError('CPU stop after binding')) as runtime:
                with self.assertRaisesRegex(RuntimeError,'CPU stop after binding'):
                    formal_launch.main(['--model','dp_libero','--max-batch','16','--checkpoint','/immutable/weights',
                        '--runtime-from',str(source),'--official-50','--prepare-only','--run-name','dp_arm'])
                bound=runtime.call_args.args[1]
                self.assertEqual(bound['dp_batching'],campaign['dp_batching'])
                self.assertEqual(bound['runtime_source_identity'],campaign['runtime_identity'])

    def test_pending_control_from_another_release_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);source,_,campaign=self.fixture(p);campaign['paired_control']=str(p/'other')
            with self.assertRaisesRegex(ValueError,'must equal paired control'):frozen.attach(campaign,source)

    def test_fastwam_uses_control_implementation_pin_and_still_rejects_topology_changes(self):
        from exp.r17_integration.models.fastwam import formal_batching as batching
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);source,_,old=self.fixture(p)
            old.update(model='fastwam',workers=32,replicas=2,worker_nodes={'timan107':32},
                worker_egl_device=0,worker_egl_devices=None,worker_vram_mib=768,
                worker_memory_mib=3072,worker_max_episodes=4,worker_restart_limit=2)
            batching.bind(old,8,5)
            write_json(source/'campaign.json',old)
            arm={k:v for k,v in old.items() if not k.startswith('fastwam_batch')}
            arm.update(commit_horizon=20,prepare_only=True)
            with mock.patch.object(batching,'implementation_identity',return_value='edited-checkout'):
                batching.bind(arm,8,5,source)
                self.assertNotEqual(arm['fastwam_batch_treatment_identity'],old['fastwam_batch_treatment_identity'])
                frozen.attach(arm,source)
                self.assertEqual(arm['fastwam_batch_treatment_identity'],old['fastwam_batch_treatment_identity'])
                self.assertEqual(arm['commit_horizon'],20)
                arm['workers']=24
                with self.assertRaisesRegex(ValueError,'batching/topology mismatch'):frozen.attach(arm,source)

    def test_prepared_queue_launch_reuses_preparation_and_stops_on_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);jobs=[dict(name=n,run=str(p/n)) for n in ('one','two')]
            calls=[]
            class Runner:
                def invoke(self,job,phase,argv):
                    calls.append((job['name'],phase,argv));directory=Path(job['run']);directory.mkdir()
                    write_json(directory/'RECEIPT.json',dict(status='FAIL',all_owned_closed=True))
            with self.assertRaisesRegex(RuntimeError,'refuses switch'):queue.run_prepared_jobs(jobs,Runner())
            self.assertEqual(calls,[('one','run',['--run','--prepared-run',str(p/'one')])])

    def test_prepared_queue_validates_all_existing_seals_and_rejects_gpu_mixing(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);directory=p/'queue';directory.mkdir()
            jobs=[dict(name=n,run=str(p/n),argv=['--run-name',n]) for n in ('one','two')]
            write_json(directory/'QUEUE_PLAN.json',{'jobs':jobs})
            write_json(directory/'QUEUE_RECEIPT.json',dict(status='PASS',cpu_only=True,all_owned_closed=True,
                results=[dict(name=j['name'],prepared=j['run']) for j in jobs]))
            campaigns=[({'run_name':n,'server_node':'local','device':'cuda:0','seconds':60}, {}) for n in ('one','two')]
            with mock.patch.object(launch,'FORMAL_ROOT',p),mock.patch.object(stage,'validate_prepared',side_effect=campaigns) as verify:
                self.assertEqual(len(queue.prepared_jobs(directory)),2)
                self.assertEqual(verify.call_count,2)
            campaigns[1][0]['server_node']='h100'
            with mock.patch.object(launch,'FORMAL_ROOT',p),mock.patch.object(stage,'validate_prepared',side_effect=campaigns):
                with self.assertRaisesRegex(ValueError,'one execution host and GPU'):queue.prepared_jobs(directory)


if __name__=='__main__':unittest.main()
