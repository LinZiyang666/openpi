"""CPU regressions for formal cohorts, result fences and owned cleanup."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from exp.r17_integration.ops import formal_eval as launch
from exp.r17_integration.ops.formal_process import ticks, alive, stop_group, write_json
from exp.r17_integration.ops.formal_driver import plan, probe_validator


class FormalCampaignTests(unittest.TestCase):
    def fixture_plan(self, benchmark, domain):
        import yaml
        from exp.r17_integration.common.fake_adapter import fake_provider, fake_spec
        from exp.r17_integration.common.benchmarks import register_benchmark
        from exp.r17_integration.common.qualification import config_for_probe
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            register_benchmark(benchmark)
            spec = replace(fake_spec(), allowed_benchmark_protocols=(benchmark.identity,))
            provider = fake_provider('formal.fixture.'+root.name, spec)
            config = config_for_probe(provider, 4, 5)
            config.enabled = False
            config.checkpoints = {}
            config.recipe.ordered_tasks = list(benchmark.ordered_tasks)
            config.recipe.task_table_identity = benchmark.task_table_identity
            (root/'full.yaml').write_text(yaml.safe_dump(asdict(config)))
            write_json(root/'benchmark.json', benchmark.manifest())
            write_json(root/'model_spec.json', asdict(spec))
            campaign = dict(local_assets=str(root), domain=domain, endpoint_host='example.invalid',
                public_port=23160, smoke=False, run_name='fixture', workers=24)
            _, _, tasks, _ = plan(campaign)
            return tasks

    def test_sealed_cohorts_are_exactly_400_2000_500_and_canonical(self):
        from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
        for benchmark, domain, expected, indices in (
            (launch.benchmark(), 'libero', 400, set(range(10))),
            (launch.benchmark(True), 'libero', 2000, set(range(50))),
            (FORMAL_SPEC, 'metaworld', 500, set(range(10)))):
            tasks = self.fixture_plan(benchmark, domain)
            self.assertEqual(len(tasks), expected)
            self.assertEqual(len({t.task_uid for t in tasks}), expected)
            self.assertEqual({t.orig_init_state_idx for t in tasks}, indices)
            self.assertEqual({t.extra['task_name'] for t in tasks}, set(benchmark.ordered_tasks))
            self.assertEqual({t.extra['partition'] for t in tasks}, {'test'})

    def test_libero_worker_admission_binds_the_selected_benchmark(self):
        from exp.r17_integration.benchmarks.libero_bridge import LiberoBenchmarkAdapter
        from exp.r17_integration.conductor.manifest import validate_worker_probe
        spec = launch.benchmark()
        probe = LiberoBenchmarkAdapter(spec).worker_probe(dict(libero_code_digest='sealed-native-source',
            task_data=dict(state_sha256='original-state')))
        # Regression of the first real probe: default validator is MetaWorld.
        with self.assertRaisesRegex(ValueError, 'simulator package mismatch'):
            validate_worker_probe(probe)
        probe_validator(spec)(probe)
        with self.assertRaisesRegex(ValueError, 'LIBERO protocol/task map mismatch'):
            probe_validator(launch.benchmark(True))(probe)

    def test_official50_is_a_separate_sealed_partition(self):
        spec = launch.benchmark()
        official = launch.benchmark(True)
        self.assertEqual(spec.formal_indices, tuple(range(10)))
        self.assertEqual(official.formal_indices, tuple(range(50)))
        self.assertEqual(official.library_indices, ())
        self.assertEqual(official.dev_indices, ())
        self.assertNotEqual(spec.identity, official.identity)
        self.assertEqual(spec.tasks, official.tasks)
        self.assertEqual(spec.wire_size, 256)
        self.assertEqual(official.manifest()['expected_formal_episodes'], 2000)

    def test_formal_defaults_and_port_collision_rejection(self):
        args = launch.parse_args(['--prepare-only'])
        self.assertEqual((args.replicas, args.workers, args.memory_budget_gib), (12, 24, 14))
        for argv in (['--replicas','19'], ['--port','23150'], ['--driver-port','23161'],
                     ['--model','dp_mw','--official-50'], ['--replicas','15']):
            with self.assertRaises(SystemExit):
                launch.parse_args(['--prepare-only', *argv])

    def test_aggregate_fences_stale_rows_and_counts_retries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'driver').mkdir()
            tasks = [dict(task_uid='a', experiment='libero_spatial', extra=dict(task_name='task0')),
                     dict(task_uid='b', experiment='libero_spatial', extra=dict(task_name='task1'))]
            write_json(root / 'EPISODE_PLAN.json', dict(tasks=tasks, benchmark_protocol='sealed'))
            write_json(root / 'campaign.json', dict(run_name='test', smoke=True))
            write_json(root / 'driver/DRIVER_RECEIPT.json', dict(status='PASS', elapsed_seconds=2))
            journal = [dict(task_uid='a', status='done', accepted=True, attempt=2, success=True, error=None),
                       dict(task_uid='a', status='failed', accepted=False, attempt=1, success=False, error='stale'),
                       dict(task_uid='b', status='failed', accepted=True, attempt=1, success=False, error=None)]
            (root / 'driver/journal.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in journal))
            attempts = [dict(worker_id='w1', accepted=True, attempt=1, error='retry'),
                        dict(worker_id='w2', accepted=True, attempt=2, error=None)]
            (root / 'driver/attempts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in attempts))
            per_step = [dict(task_uid='a', accepted=True, attempt=2),
                        dict(task_uid='a', accepted=False, attempt=1)]
            (root / 'driver/per_step.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in per_step))
            summary = launch.aggregate(root)
            self.assertEqual(summary['status'], 'PASS')
            self.assertEqual(summary['suites']['libero_spatial']['sr'], .5)
            self.assertEqual(summary['terminal_errors'], 0)
            self.assertEqual((summary['error_attempts'], summary['retry_attempts']), (1, 1))
            self.assertEqual(summary['decisions'], 1)
            self.assertEqual(summary['accepted_worker_ids'], ['w1', 'w2'])

    def test_retried_error_completion_requires_exact_closed_journal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'driver').mkdir()
            tasks = [dict(task_uid=uid, experiment='libero_spatial', extra=dict(task_name=uid))
                     for uid in ('a', 'b')]
            write_json(root / 'EPISODE_PLAN.json', dict(tasks=tasks, benchmark_protocol='sealed'))
            write_json(root / 'campaign.json', dict(run_name='retried', smoke=True))
            journal = [dict(task_uid='a', status='done', accepted=True, attempt=2, success=True, error=None),
                       dict(task_uid='b', status='failed', accepted=True, attempt=1, success=False, error=None)]
            attempts = [dict(task_uid='a', worker_id='w1', accepted=True, attempt=1, n_steps=0,
                             error='Offscreen framebuffer is not complete, error 0x8cdd'),
                        dict(task_uid='a', worker_id='w2', accepted=True, attempt=2, error=None),
                        dict(task_uid='b', worker_id='w2', accepted=True, attempt=1, error=None)]
            receipt = dict(status='PARTIAL', all_done=True, rejected_worker_probes=0,
                           worker_shutdown=dict(status='PARTIAL'), elapsed_seconds=2)
            def evidence(rows=journal, outcomes=attempts, driver_receipt=receipt):
                for name, records in [('journal', rows), ('attempts', outcomes),
                                      ('per_step', [dict(task_uid='a', accepted=True, attempt=2)])]:
                    (root / 'driver' / (name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records))
                write_json(root / 'driver/DRIVER_RECEIPT.json', driver_receipt)
            evidence()
            completion = launch.driver_completion(root, 2)
            self.assertEqual(completion['accepted_episodes'], 2)
            self.assertTrue(completion['verified_completion'])
            self.assertTrue(completion['recovered_retried_errors'])
            self.assertEqual([(r['task_uid'],r['failed_attempt'],r['resolved_attempt']) for r in
                              completion['resolved_retry_errors']], [('a',1,2)])
            summary = launch.aggregate(root, driver_returncode=2)
            self.assertEqual(summary['status'], 'PASS')
            self.assertEqual(summary['suites']['libero_spatial']['sr'], .5)
            self.assertEqual((summary['error_attempts'], summary['retry_attempts']), (1, 1))
            self.assertEqual(summary['driver_completion'], completion)
            self.assertEqual(json.loads((root / 'SUMMARY.json').read_text())['status'], 'PASS')
            # Keep the raw shutdown/exit evidence, rather than rewriting it.
            self.assertEqual(json.loads((root / 'driver/DRIVER_RECEIPT.json').read_text()), receipt)
            for rc in (-15, 1, 3):
                with self.subTest(returncode=rc), self.assertRaisesRegex(RuntimeError, 'driver incomplete'):
                    launch.driver_completion(root, rc)
            cases = [
                ('missing', journal[:1], attempts, receipt),
                ('unaccepted', [journal[0], dict(journal[1], accepted=False)], attempts, receipt),
                ('duplicate', journal+[journal[0]], attempts, receipt),
                ('extra', journal+[dict(journal[0], task_uid='unexpected')], attempts, receipt),
                ('unresolved', [dict(journal[0], error='fatal'), journal[1]], attempts, receipt),
                ('nonterminal', [dict(journal[0], status='pending'), journal[1]], attempts, receipt),
                ('no_retry_error', journal, attempts[1:], receipt),
                ('stale_error_only', journal, [dict(attempts[0], accepted=False)], receipt),
                ('unresolved_attempt', journal, [dict(attempts[0], attempt=3)], receipt),
                ('unknown_error', journal, [dict(attempts[0], task_uid='unexpected')], receipt),
                ('not_done', journal, attempts, dict(receipt, all_done=False)),
                ('probe_rejected', journal, attempts, dict(receipt, rejected_worker_probes=1)),
                ('receipt_fail', journal, attempts, dict(receipt, status='FAIL')),
            ]
            for name, rows, outcomes, driver_receipt in cases:
                with self.subTest(case=name):
                    evidence(rows, outcomes, driver_receipt)
                    with self.assertRaisesRegex(RuntimeError, 'driver incomplete'):
                        launch.aggregate(root, driver_returncode=2)
            evidence(driver_receipt=dict(receipt, status='PASS'))
            self.assertEqual(launch.aggregate(root, driver_returncode=0)['status'], 'PASS')
            evidence(rows=journal[:1], driver_receipt=dict(receipt, status='PASS'))
            with self.assertRaisesRegex(RuntimeError, 'driver incomplete'):
                launch.aggregate(root, driver_returncode=0)

    def test_cleanup_never_signals_a_reused_pid(self):
        child = subprocess.Popen(['sleep','30'], start_new_session=True)
        entry = dict(pid=child.pid, pgid=child.pid, start_ticks=ticks(child.pid))
        try:
            wrong = dict(entry, start_ticks='different')
            self.assertFalse(alive(wrong))
            stop_group(wrong)
            self.assertIsNone(child.poll())
            result = stop_group(entry, grace=1)
            child.wait(timeout=3)
            self.assertEqual(result['status'], 'closed')
            self.assertFalse(alive(entry))
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait()

    def test_legacy_command_without_assignment_preserves_device_and_capture_off(self):
        from exp.r17_integration.ops.formal_workers import worker_command
        campaign = dict(run_name='formal', worker_node='timan107', endpoint='ziyanglin.com:23160',
            driver_host='ziyanglin.com', driver_port=23179, domain='libero',
            worker_benchmark='/tmp/benchmark.json', worker_egl_device=0)
        ids = set()
        for index in range(24):
            worker_id, command = worker_command(campaign, index, '/private/python')
            ids.add(worker_id)
            self.assertEqual(command[command.index('--egl-device')+1], '0')
            self.assertNotIn('--debug-directory', command)
            self.assertEqual(command[0], '/private/python')
        self.assertEqual(len(ids),24)

    def test_cosmos_nvrtc_paths_are_scoped_to_its_resident(self):
        campaign = dict(model='cosmos', server_node='h100', vendor='/private/cosmos')
        with mock.patch.dict(os.environ, LD_LIBRARY_PATH='/existing/lib'):
            env = launch.server_env(campaign, '/runtime')
            nvidia = '/private/cosmos/.venv/lib/python3.10/site-packages/nvidia'
            self.assertEqual(env['CUDA_HOME'], nvidia+'/cuda_nvrtc')
            self.assertIn(nvidia+'/cuda_nvrtc/lib', env['LD_LIBRARY_PATH'].split(':'))
            self.assertTrue(env['LD_LIBRARY_PATH'].endswith('/existing/lib'))
            self.assertEqual(os.environ['LD_LIBRARY_PATH'], '/existing/lib')
            other = launch.server_env(dict(campaign, model='fastwam'), '/runtime')
            self.assertNotIn('CUDA_HOME', other)
            self.assertNotIn('LD_LIBRARY_PATH', other)

    def test_libero_config_is_selected_per_worker_node_in_mixed_fleets(self):
        for model in ('fastwam', 'dp', 'cosmos'):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                campaign = dict(model=model, domain='libero', run_name='cpu_worker_config',
                    worker_nodes={'timan108': 2, 'timan107': 1}, worker_egl_device=0,
                    local_assets=str(root))
                calls = []
                def remote(node, argv, **kwargs):
                    calls.append((node, argv))
                    if 'private_config' in argv[-1]:
                        return json.dumps(dict(path=launch.worker_paths(campaign,node)['private']+'/libero_config'))
                    if 'worker_memory import admission' in argv[-1]:
                        return '{"admitted":true}'
                    return '{}'
                with mock.patch('exp.r17_integration.ops.build_client_bundle.worker_payload_files', return_value=[]), \
                     mock.patch('exp.r17_integration.ops.build_client_bundle.build_client_bundle',
                                return_value=dict(archive_sha256='cpu_fixture')), \
                     mock.patch.object(launch, 'remote', side_effect=remote), \
                     mock.patch('exp.r17_integration.ops.worker_disk.probe', return_value={'admitted':True}), \
                     mock.patch.object(launch, 'transfer'), \
                     mock.patch.object(launch, 'install_worker_dependencies', side_effect=lambda node,*args:
                        dict(copied=['libero'], private_dependencies='/immutable/'+node,
                             config_source='/original/'+node+'/config.yaml')):
                    report = launch.prepare_workers(campaign, root)
                paths = {node:launch.worker_paths(campaign,node)['private']+'/libero_config'
                         for node in campaign['worker_nodes']}
                for node, path in paths.items():
                    self.assertEqual(report[node]['env']['LIBERO_CONFIG_PATH'], path)
                    worker = json.loads((root/(node+'.json')).read_text())
                    self.assertEqual(worker['worker_memory_mib'], 6144)
                    self.assertEqual(worker['worker_max_episodes'], 4)
                    probes = [argv for n, argv in calls if n == node and 'CUDA_VISIBLE_DEVICES=' in argv]
                    self.assertEqual(len(probes), 2)
                    for argv in probes:
                        self.assertIn('LIBERO_CONFIG_PATH='+path, argv)
                        self.assertIn('CUDA_VISIBLE_DEVICES=', argv)

    def startup_campaign(self, root, node):
        import socket
        sockets = [socket.socket() for _ in range(3)]
        try:
            for sock in sockets:
                sock.bind(('127.0.0.1', 0))
            ports = [s.getsockname()[1] for s in sockets]
        finally:
            for sock in sockets:
                sock.close()
        return dict(model='cosmos', server_node=node, runtime=str(launch.ROOT),
            run_name='cpu_replica_death', seconds=30, ports=ports, backend_ports=[ports[1]],
            public_port=ports[0], driver_port=ports[2], server_campaign=str(root/'campaign.json'),
            server_run=str(root/'servers'), endpoint_host='127.0.0.1', backend_host='127.0.0.1')

    def test_real_local_replica_exit_is_fail_with_log_and_owned_cleanup(self):
        real_popen = subprocess.Popen
        def cpu_child(argv, *args, **kwargs):
            if 'exp.r17_integration.ops.formal_server' in argv:
                argv = [sys.executable, '-u', '-c',
                        'import sys; print("test resident import failure", flush=True); sys.exit(7)']
            return real_popen(argv, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            campaign = self.startup_campaign(root, 'local')
            started = time.monotonic()
            with mock.patch.object(launch, 'host_python', return_value=sys.executable), \
                 mock.patch.object(launch, 'server_env', return_value={}), \
                 mock.patch.object(launch.subprocess, 'Popen', side_effect=cpu_child):
                self.assertEqual(launch.run_campaign(campaign, root, {}), 2)
            receipt = json.loads((root/'RECEIPT.json').read_text())
            self.assertLess(time.monotonic()-started, 5)
            self.assertEqual(receipt['status'], 'FAIL')
            self.assertTrue(receipt['all_owned_closed'])
            self.assertIn('test resident import failure', receipt['error'])
            self.assertIn('exit_code=7', receipt['error'])
            self.assertTrue(all(not alive(entry) for entry in receipt['owned_local']))

    def test_proxy_readiness_precedes_tunnel_for_library_and_exposed_endpoints(self):
        cases = (
            ('dp', True, '127.0.0.1', False, True),
            ('dp', True, 'tunnel.invalid', True, True),
            ('dp', False, 'tunnel.invalid', False, True),
            ('dp', False, '127.0.0.1', True, True),
            ('dp', False, '127.0.0.1', False, False),
            ('cosmos', False, '127.0.0.1', False, True),
        )
        for model, library, host, different_port, direct_wait in cases:
            with self.subTest(model=model, library=library, host=host, different_port=different_port), \
                 tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                campaign = self.startup_campaign(root, 'local')
                endpoint_port = campaign['public_port'] + 1 if different_port else campaign['public_port']
                campaign.update(model=model, library=library, endpoint_host=host,
                    local_assets=str(root), memory_budget_gib=14)
                if different_port:
                    campaign['endpoint_port'] = endpoint_port
                write_json(root/'model_spec.json', {})
                direct = ('127.0.0.1', campaign['public_port'])
                expected = [('127.0.0.1', campaign['backend_ports'][0])]
                if direct_wait:
                    expected.append(direct)
                expected.append((host, endpoint_port))
                connections = []
                def connect(address, **kwargs):
                    connections.append(address)
                    if len(connections) == len(expected):
                        # Stop at the public endpoint, before any workers launch.
                        raise RuntimeError('CPU readiness probe complete')
                    return mock.MagicMock()
                children = [mock.Mock(pid=42, poll=mock.Mock(return_value=None),
                                      wait=mock.Mock(return_value=0)),
                            mock.Mock(pid=43, poll=mock.Mock(return_value=None),
                                      wait=mock.Mock(return_value=0))]
                metadata = dict(r17_model_spec={}, cuda_graph_self_check=dict(passed=True),
                    native_parity_self_check=dict(passed=True))
                with mock.patch.object(launch, 'host_python', return_value=sys.executable), \
                     mock.patch.object(launch, 'server_env', return_value={}), \
                     mock.patch.object(launch, 'execute', return_value='42, 1'), \
                     mock.patch.object(launch.subprocess, 'Popen', side_effect=children), \
                     mock.patch.object(launch, 'ticks', return_value=1), \
                     mock.patch.object(launch, 'stop_group', return_value=dict(alive=False)), \
                     mock.patch.object(launch, 'alive', return_value=False), \
                     mock.patch.object(launch.socket, 'create_connection', side_effect=connect), \
                     mock.patch('exp.r17_integration.common.websocket_policy_server.WebsocketClientPolicy') as client:
                    client.return_value.get_server_metadata.return_value = metadata
                    self.assertEqual(launch.run_campaign(campaign, root, {}), 2)
                self.assertEqual(connections, expected)
                receipt = json.loads((root/'RECEIPT.json').read_text())
                self.assertIn('CPU readiness probe complete', receipt['error'])
                self.assertTrue(receipt['all_owned_closed'])

    def test_remote_replica_death_fails_even_with_batched_lease_renewal_and_exit_zero(self):
        for code in (0, 1):
            with self.subTest(exit_code=code), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                campaign = self.startup_campaign(root, 'h100')
                calls = []
                def remote(node, argv, seconds=60, **kwargs):
                    calls.append(argv)
                    if argv[0] == 'tail':
                        return 'test H100 NVRTC failure tail'
                    if '-c' in argv:
                        if 'formal_process import controls' in argv[-1]:
                            return json.dumps(dict(replica_0=dict(status='CLOSED', exit_code=code,
                                alive=False, child_alive=False)))
                        return json.dumps({'free': campaign['ports'][:-1]})
                    mode = argv[argv.index('exp.r17_integration.ops.formal_process')+1]
                    if mode == 'spawn':
                        return json.dumps(dict(status='RUNNING', child=dict(pid=123), supervisor=dict(pid=456)))
                    if mode == 'status':
                        return json.dumps(dict(status='CLOSED', exit_code=code, alive=False, child_alive=False))
                    if mode == 'stop':
                        return json.dumps(dict(status='CLOSED', exit_code=code, alive=False, child_alive=False))
                    self.fail('dead resident must not renew a lease: '+mode)
                with mock.patch.object(launch, 'host_python', return_value=sys.executable), \
                     mock.patch.object(launch, 'server_env', return_value={}), \
                     mock.patch.object(launch, 'PROBE_SECONDS', 0), \
                     mock.patch.object(launch, 'remote', side_effect=remote):
                    self.assertEqual(launch.run_campaign(campaign, root, {}), 2)
                receipt = json.loads((root/'RECEIPT.json').read_text())
                self.assertEqual(receipt['status'], 'FAIL')
                self.assertTrue(receipt['all_owned_closed'])
                self.assertIn('test H100 NVRTC failure tail', receipt['error'])
                self.assertIn('exit_code='+str(code), receipt['error'])
                self.assertEqual(len(receipt['owned_remote']), 1)

    def test_live_supervisor_does_not_hide_dead_replica_child(self):
        with self.assertRaisesRegex(RuntimeError, 'replica_0.*exit_code=None'):
            launch.require_service_alive('replica_0', dict(status='RUNNING', alive=True, child_alive=False),
                                         'owned/replica_0.log', lambda: 'resident is gone')


if __name__ == '__main__':
    unittest.main()
