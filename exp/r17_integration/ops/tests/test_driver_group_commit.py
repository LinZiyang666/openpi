"""CPU-only durability, lock/fence and orchestration-pin regressions."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from exp.r17_integration.ops.formal_driver_fast import AsyncCampaignDriver
from exp.r17_integration.ops.group_commit import GroupCommit
from exp.r17_integration.ops import driver_release, immutable_release
from exp.r17_integration.ops.formal_process import write_json


def fixture(directory,count=8,cls=AsyncCampaignDriver,writer=None):
    from openpi.conductor.task import EpisodeTask,ServerEndpoint
    from exp.r17_integration.conductor.manifest import EndpointSpec
    from exp.r17_integration.conductor.strategy import MetaWorldStrategy
    endpoint=ServerEndpoint('fake',1234)
    tasks=[EpisodeTask(f'yaml:eval:0:{i}','yaml','eval','libero_spatial',0,i,i,endpoint.host,endpoint.port,
        'formal_full_v1',extra=dict(model_identity='model',norm_identity='norm',layout_identity='layout',
        benchmark_protocol='benchmark',num_steps=100,original_init_idx=i)) for i in range(count)]
    endpoints=[EndpointSpec(endpoint,'model','norm','layout','benchmark',allowed_k=(100,),capacity=24)]
    strategy=MetaWorldStrategy(tasks,endpoints=endpoints)
    kwargs=dict(dispatch_path=directory/'dispatch.jsonl',rejection_path=directory/'rejections.json',expected_workers=24,
        shutdown_seconds=.1,yaml_weights={'yaml':1},servers=[endpoint],journal_path=str(directory/'journal.jsonl'),
        ctl_factory=lambda ep:mock.Mock(close=lambda:None),scheduler_kwargs=dict(eval_concurrency=24,max_episode_retries=2))
    if cls is AsyncCampaignDriver:kwargs['evidence_directory']=directory
    else:kwargs['per_step_writer']=writer
    driver=cls(strategy,**kwargs);driver.drive_stages_once()
    return driver


def result(task,**kwargs):
    return dict(task_uid=task['task_uid'],attempt=task['attempt'],success=True,n_steps=5,duration_s=.01,
        per_step_rows=[dict(decision_seq=0,capture_enabled=False,actual_nfe=100,source='policy',stage_calls=[1,1,100])],**kwargs)


def wait_until(predicate,timeout=5):
    until=time.monotonic()+timeout
    while not predicate():
        if time.monotonic()>until:raise AssertionError('CPU condition timeout')
        time.sleep(.005)


class CommitTests(unittest.TestCase):
    def test_fsync_never_holds_scheduler_lock_and_results_count_only_after_barrier(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);blocked=threading.Event();release=threading.Event();real_fsync=os.fsync
            def fsync(fd):
                name=os.readlink('/proc/self/fd/'+str(fd))
                if name.endswith('per_step.jsonl'):
                    blocked.set();release.wait(5)
                real_fsync(fd)
            with mock.patch('exp.r17_integration.ops.formal_driver_fast.GroupCommit',
                            side_effect=lambda d:GroupCommit(d,interval=.02,max_events=4,fsync=fsync)):
                driver=fixture(p)
            _,first=driver.handle_pull('fake:1234');driver.handle_result(result(first['task']))
            self.assertTrue(blocked.wait(5))
            self.assertEqual(driver.scheduler.progress()['episodes_done'],0)
            started=time.monotonic();_,second=driver.handle_pull('fake:1234')
            self.assertLess(time.monotonic()-started,.1);self.assertIn('task',second)
            self.assertFalse(driver.scheduler.mark_result(first['task']['task_uid'],success=False,retriable=True,attempt=1))
            driver.scheduler.requeue_timed_out(timeout_s=.000001,now=time.monotonic()+100)
            self.assertIn(first['task']['task_uid'],driver.scheduler.committing)
            self.assertFalse((p/'journal.jsonl').read_text())
            release.set();wait_until(lambda:driver.scheduler.progress()['episodes_done']==1)
            driver.close()
            ending=json.loads((p/'journal.jsonl').read_text())
            self.assertTrue(ending['accepted']);self.assertTrue((p/'attempts.jsonl').read_text())

    def test_failure_before_evidence_barrier_never_writes_terminal_or_counts_episode(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);real_fsync=os.fsync
            def fsync(fd):
                if os.readlink('/proc/self/fd/'+str(fd)).endswith('per_step.jsonl'):raise OSError('disk error')
                real_fsync(fd)
            with mock.patch('exp.r17_integration.ops.formal_driver_fast.GroupCommit',
                            side_effect=lambda d:GroupCommit(d,interval=.01,max_events=4,fsync=fsync)):
                driver=fixture(p)
            _,task=driver.handle_pull('fake:1234');driver.handle_result(result(task['task']))
            wait_until(lambda:driver.evidence.failure is not None)
            self.assertEqual(driver.scheduler.progress()['episodes_done'],0)
            self.assertFalse((p/'journal.jsonl').read_text())
            with self.assertRaisesRegex(RuntimeError,'commit failed'):driver.close()

    def test_journal_fsync_failure_has_durable_prerequisites_but_no_in_memory_acceptance(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);synced=[];real_fsync=os.fsync
            def fsync(fd):
                name=Path(os.readlink('/proc/self/fd/'+str(fd))).name
                if name=='journal.jsonl':raise OSError('terminal sync error')
                real_fsync(fd);synced.append(name)
            writer=GroupCommit(p,interval=.01,fsync=fsync)
            callback=mock.Mock()
            writer.submit({'per_step.jsonl':[{'task_uid':'one'}],'attempts.jsonl':[{'task_uid':'one'}],
                           'journal.jsonl':[{'task_uid':'one','accepted':True}]},callback)
            wait_until(lambda:writer.failure is not None)
            self.assertIn('per_step.jsonl',synced);self.assertIn('attempts.jsonl',synced)
            callback.assert_not_called()
            with self.assertRaisesRegex(RuntimeError,'commit failed'):writer.close()

    def test_duplicate_timeout_and_stale_generation_are_fenced_during_commit(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);driver=fixture(p)
            _,task=driver.handle_pull('fake:1234');live=task['task']
            driver.handle_result(result(live));driver.handle_result(result(live))
            stale=dict(live,attempt=0);driver.handle_result(result(stale))
            driver.close()
            journal=[json.loads(line) for line in (p/'journal.jsonl').read_text().splitlines()]
            self.assertEqual(sum(row['accepted'] is True for row in journal),1)
            self.assertEqual(driver.scheduler.progress()['episodes_done'],1)
            steps=[json.loads(line) for line in (p/'per_step.jsonl').read_text().splitlines()]
            self.assertEqual(sum(row['accepted'] is True for row in steps),1)

    def test_retry_then_real_failure_and_success_preserve_generation_and_outcome(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);driver=fixture(p,count=1)
            _,task=driver.handle_pull('fake:1234')
            failure=result(task['task']);failure.update(success=False,error='transient EGL')
            driver.handle_result(failure)
            wait_until(lambda:not driver.scheduler.committing)
            _,next_task=driver.handle_pull('fake:1234');self.assertEqual(next_task['task']['attempt'],2)
            driver.handle_result(result(next_task['task']));driver.close()
            self.assertEqual(driver.scheduler.progress()['episodes_done'],1)
            attempts=[json.loads(line) for line in (p/'attempts.jsonl').read_text().splitlines()]
            self.assertEqual([r['accepted'] for r in attempts],[True,True]);self.assertEqual(attempts[0]['error'],'transient EGL')
            self.assertEqual(len((p/'journal.jsonl').read_text().splitlines()),1)

    def test_grouping_drains_at_shutdown_and_does_not_sync_per_step(self):
        with tempfile.TemporaryDirectory() as temp:
            writer=GroupCommit(Path(temp),max_events=64)
            calls=[]
            for i in range(32):writer.submit({'per_step.jsonl':[{'i':i,'step':s} for s in range(50)],
                'attempts.jsonl':[{'i':i}],'journal.jsonl':[{'i':i,'accepted':True}]},lambda:calls.append(True))
            stats=writer.close()
            self.assertEqual(len(calls),32);self.assertEqual(stats['accepted_terminal'],32)
            self.assertLess(stats['fsyncs'],10)

    def test_restart_requires_fresh_resume_segment_and_default_mode_is_legacy(self):
        from exp.r17_integration.ops.formal_eval import parse_args
        self.assertEqual(parse_args(['--prepare-only']).driver_mode,'legacy')
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'journal.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'in-place restart'):fixture(p)

    def test_pure_fixed_bundle_dispatch_has_no_control_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            driver=fixture(Path(temp),count=1)
            driver._strategy.ensure_bundle=mock.Mock(side_effect=AssertionError('serialized control path'))
            _,assignment=driver.handle_pull('fake:1234')
            self.assertIn('task',assignment)
            driver._strategy.ensure_bundle.assert_not_called()
            driver.close()

    def test_retry_budget_exhaustion_is_durable_terminal_error(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);driver=fixture(p,count=1)
            for attempt in range(1,4):
                _,task=driver.handle_pull('fake:1234')
                self.assertEqual(task['task']['attempt'],attempt)
                value=result(task['task']);value.update(success=False,error='transient EGL')
                driver.handle_result(value)
                wait_until(lambda:not driver.scheduler.committing)
            driver.close()
            rows=[json.loads(line) for line in (p/'journal.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),1);self.assertTrue(rows[0]['accepted'])
            self.assertEqual(rows[0]['status'],'failed')
            self.assertIn('FatalEpisodeError: retry budget exhausted;',rows[0]['error'])

    def test_hard_exit_before_and_after_terminal_barrier_replays_only_safe_episodes(self):
        from openpi.conductor.journal import Journal
        code='''
import os,sys,time
from pathlib import Path
from exp.r17_integration.ops.group_commit import GroupCommit
directory=Path(sys.argv[1]); boundary=sys.argv[2]; real=os.fsync
def sync(fd):
    name=Path(os.readlink('/proc/self/fd/'+str(fd))).name
    if boundary=='before_evidence' and name=='per_step.jsonl':os._exit(77)
    real(fd)
    if boundary=='after_terminal' and name=='journal.jsonl':os._exit(78)
writer=GroupCommit(directory,interval=.01,fsync=sync)
def accepted():
    (directory/'COUNTED').write_text('1');os._exit(79)
writer.submit({'per_step.jsonl':[{'task_uid':'one','accepted':True}],
               'attempts.jsonl':[{'task_uid':'one','accepted':True}],
               'journal.jsonl':[{'task_uid':'one','status':'done','accepted':True}]},accepted)
writer.thread.join(10)
raise RuntimeError('crash boundary not reached')
'''
        for boundary,rc in (('before_evidence',77),('after_terminal',78),('after_acceptance',79)):
            with self.subTest(boundary=boundary),tempfile.TemporaryDirectory() as temp:
                process=subprocess.run([sys.executable,'-c',code,temp,boundary],timeout=15,capture_output=True)
                self.assertEqual(process.returncode,rc,process.stderr)
                done=Journal(Path(temp)/'journal.jsonl').replay_done_uids()
                self.assertEqual(done,set() if boundary=='before_evidence' else {'one'})
                if done:
                    self.assertEqual(json.loads((Path(temp)/'attempts.jsonl').read_text())['task_uid'],'one')
                    self.assertEqual(json.loads((Path(temp)/'per_step.jsonl').read_text())['task_uid'],'one')
                self.assertEqual((Path(temp)/'COUNTED').exists(),boundary=='after_acceptance')


class OrchestrationTests(unittest.TestCase):
    def test_resume_merge_never_discards_the_only_durable_raw_segment(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'driver').mkdir()
            names=('journal.jsonl','attempts.jsonl','per_step.jsonl','DRIVER_RECEIPT.json')
            for name in names:(p/'driver'/name).write_text('durable original '+name+'\n')
            proof=driver_release.preserve_resume_segment(p)
            for name in names:(p/'driver'/name).write_text('merged '+name+'\n')
            for name in ('EPISODE_PLAN.json','RESUME_MERGE.json'):(p/name).write_text('{}\n')
            synced=[];real=os.fsync
            def sync(fd):synced.append(os.readlink('/proc/self/fd/'+str(fd)));real(fd)
            with mock.patch.object(driver_release.os,'fsync',side_effect=sync):driver_release.sync_resume_merge(p)
            self.assertEqual(len(synced),8)
            for name,pin in proof['files'].items():
                self.assertEqual(immutable_release.sha(p/'durable_segment'/name),pin)
                self.assertNotEqual(immutable_release.sha(p/'driver'/name),pin)
            with self.assertRaises(FileExistsError):driver_release.preserve_resume_segment(p)

    def test_endpoint_replanning_uses_the_recorded_runtime_without_gpu_visibility(self):
        from exp.r17_integration.ops import formal_eval as launch
        with mock.patch.object(driver_release,'validate'),mock.patch.object(launch,'execute',return_value='[]') as execute:
            self.assertEqual(driver_release.planned_tasks({'driver_runtime':'/frozen/driver'},'/own/prepared'),[])
        self.assertEqual(execute.call_args.kwargs['cwd'],'/frozen/driver')
        self.assertEqual(execute.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
        self.assertIn('/frozen/driver',execute.call_args.kwargs['env']['PYTHONPATH'])

    def test_only_driver_overlay_changes_and_future_checkout_edits_cannot_change_pin(self):
        import tarfile
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);source=p/'source';source.mkdir();directory=p/'run';directory.mkdir()
            for name in driver_release.OVERLAY:
                file=source/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_text('pinned driver\n')
            file=source/'models/policy.py';file.parent.mkdir();file.write_text('unchanged native model\n')
            with tarfile.open(directory/'runtime.tar','w') as tar:
                tar.add(file,arcname='models/policy.py')
            base=immutable_release.archive_manifest(directory/'runtime.tar')
            campaign=dict(runtime_identity=immutable_release.identity(base),driver_mode='group-commit')
            def install(archive,root,archive_sha,pin):
                campaign['_test_driver_root']=str(p/'driver')
                return immutable_release.install(archive,p/'driver',archive_sha,pin)
            with mock.patch.object(driver_release,'install',side_effect=install):driver_release.prepare(campaign,directory,source)
            archived=immutable_release.archive_manifest(directory/'driver_runtime.tar')
            self.assertEqual(archived['files']['models/policy.py'],base['files']['models/policy.py'])
            self.assertEqual(campaign['orchestration']['overlay_files'],list(driver_release.OVERLAY))
            record=campaign['orchestration'];record['driver_runtime']=campaign['_test_driver_root']
            campaign['driver_runtime']=record['driver_runtime'];write_json(directory/'ORCHESTRATION.json',record)
            driver_release.validate(campaign,directory)
            for name in driver_release.OVERLAY:(source/name).write_text('edited later\n')
            driver_release.validate(campaign,directory)
            self.assertEqual(record['baseline']['mode'],'legacy');self.assertEqual(record['selected']['mode'],'group-commit')


class PreparedRefreshTests(unittest.TestCase):
    def test_completed_shutdown_recovery_cannot_hide_failed_group_commit(self):
        from exp.r17_integration.ops.formal_eval import driver_completion
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'driver').mkdir()
            write_json(p/'campaign.json',dict(driver_mode='group-commit',orchestration={'identity':'driver'}))
            write_json(p/'EPISODE_PLAN.json',{'tasks':[{'task_uid':'one'}]})
            for name,row in (('journal',dict(task_uid='one',accepted=True,status='done',success=True,attempt=1)),
                             ('attempts',dict(task_uid='one',accepted=True,success=True,attempt=1))):
                (p/'driver'/f'{name}.jsonl').write_text(json.dumps(row)+'\n')
            receipt=dict(status='PARTIAL',all_done=True,rejected_worker_probes=0,
                         worker_shutdown={'status':'PARTIAL'},orchestration={'identity':'driver'},
                         error='journal fsync failed',group_commit={'closed':False})
            write_json(p/'driver/DRIVER_RECEIPT.json',receipt)
            with self.assertRaisesRegex(RuntimeError,'grouped evidence commit'):
                driver_completion(p,2)
            receipt.update(error=None,group_commit={'closed':True})
            write_json(p/'driver/DRIVER_RECEIPT.json',receipt)
            self.assertTrue(driver_completion(p,2)['verified_completion'])

    def test_refresh_explicitly_selects_grouped_mode_before_sealing_driver(self):
        from exp.r17_integration.ops import formal_prestage as stage,formal_eval as launch
        from exp.r17_integration.ops.tests.test_formal_prestage import PreparedTests
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p=root/'run';p.mkdir();PreparedTests().fixture(p)
            c=json.loads((p/'campaign.json').read_text());c['server_node']='local';c['driver_mode']='legacy'
            write_json(p/'campaign.json',c);stage.seal_prepared(c,p,{})
            (p/'runtime.tar').write_bytes(b'unchanged');(p/'workers.tar').write_bytes(b'unchanged')
            def prepare(campaign,directory,checkout):
                self.assertEqual(campaign['driver_mode'],'group-commit')
                campaign.update(driver_runtime='/immutable/driver',orchestration={
                    'baseline':{'identity':'baseline'},'selected':{'identity':'selected'}})
            with mock.patch.object(launch,'FORMAL_ROOT',root),mock.patch.object(stage,'verify',return_value={'present':True}), \
                 mock.patch.object(driver_release,'prepare',side_effect=prepare),mock.patch.object(driver_release,'validate'), \
                 mock.patch.object(launch,'execute',return_value='{"status":"PASS","episodes":0}'):
                report=stage.refresh_driver(p,'group-commit')
            self.assertEqual(report['driver_mode'],'group-commit')
            stage.validate_prepared(p)
            self.assertEqual((p/'workers.tar').read_bytes(),b'unchanged')

    def test_refresh_and_launch_cannot_hold_the_preparation_lock_together(self):
        from exp.r17_integration.ops.formal_prestage import preparation_lock
        with tempfile.TemporaryDirectory() as temp:
            with preparation_lock(temp):
                with self.assertRaises(BlockingIOError):
                    with preparation_lock(temp):raise AssertionError('two owners acquired seal')

    def test_refresh_refuses_a_launched_campaign_without_writing_any_driver_files(self):
        from exp.r17_integration.ops import formal_prestage,formal_eval
        from exp.r17_integration.ops.tests.test_formal_prestage import PreparedTests
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);directory=root/'run';directory.mkdir();PreparedTests().fixture(directory)
            write_json(directory/'LAUNCH.json',{'pid':1})
            with mock.patch.object(formal_eval,'FORMAL_ROOT',root),self.assertRaisesRegex(ValueError,'already launched'):
                formal_prestage.refresh_driver(directory)
            self.assertFalse((directory/'driver_runtime.tar').exists())

    def test_queue_refresh_keeps_native_job_arguments_and_fails_closed_on_partial_refresh(self):
        from exp.r17_integration.ops import formal_queue,formal_prestage
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);jobs=[dict(name=n,run='/runs/'+n,argv=['--run-name',n,'--max-batch','8']) for n in ('a','b')]
            write_json(p/'QUEUE_PLAN.json',{'jobs':jobs})
            write_json(p/'QUEUE_RECEIPT.json',dict(status='PASS',cpu_only=True,all_owned_closed=True))
            with mock.patch.object(formal_prestage,'refresh_driver',side_effect=[{'status':'PASS'},ValueError('bad pin')]):
                with self.assertRaisesRegex(ValueError,'bad pin'):
                    formal_queue.refresh_prepared_queue(p,jobs,'group-commit')
            self.assertEqual(json.loads((p/'QUEUE_RECEIPT.json').read_text())['status'],'FAIL')
            self.assertEqual(jobs[0]['argv'],['--run-name','a','--max-batch','8','--driver-mode','group-commit'])
            self.assertEqual(len(list(p.glob('QUEUE_RECEIPT.json.before_driver_*'))),1)


if __name__=='__main__':unittest.main()
