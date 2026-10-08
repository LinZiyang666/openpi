"""Coordinator benchmark driver using the deployed, unchanged R16 worker agent.

The only new strategy is a warmup barrier followed by a fixed B cohort. Uses
the R16 queue, result fencing, worker/pool attestation, and lease transport.
This command operates timan107 only when the owner starts the GPU package.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import threading
import time
import uuid

from openpi.conductor.strategy import ExperimentStrategy
from openpi.conductor.task import EpisodeTask, ServerEndpoint, Stage, TaskGraph, make_task_uid
from exp.decouple.package_config import write_json, worker_command
from exp.offline_search.rounds.r16.conductor import R16Driver, preload, read_group, completed_rows

FLEET_LOCK = '/scratch/zixuans8/openpi_trace_h100/os_cl/locks/fleet.lock'


def measurement_uids(case):
    name = case['row']['arm']
    return {make_task_uid(name, 'eval', t, i) for t, i in case['manifest']['pairs']}


class BenchmarkStrategy(ExperimentStrategy):
    def __init__(self, group, case, out, workers=64):
        self.group, self.case, self.out, self.workers = group, case, Path(out), workers
        self.smoke_episodes = None
        self.driver = None
        self.window = dict(manifest_sha256=case['manifest_sha256'], measured_uids=sorted(measurement_uids(case)),
                           warmup_episodes=len(case['manifest']['warmup_pairs']), start=None, end=None)

    def plan(self, yamls, server_assignment):
        graph = TaskGraph()
        name = self.case['row']['arm']
        server = server_assignment[name]
        for phase, pairs in [('warmup', self.case['manifest']['warmup_pairs']), ('eval', self.case['manifest']['pairs'])]:
            episodes = [EpisodeTask(make_task_uid(name, phase, t, i), name, 'eval', 'libero_spatial', t, i, i,
                                    server.host, server.port, name,
                                    extra=dict(num_trials_per_task=50, init_pool='B',
                                               manifest_sha256=self.case['manifest_sha256'])) for t, i in pairs]
            graph.add_stage(Stage(phase, name, 'eval', server, episodes, setup=dict(kind=phase)))
        graph.add_dependency('warmup', 'eval')
        return graph

    def on_stage_begin(self, stage, ctl, ctx):
        if stage.stage_id == 'eval':
            deadline = time.monotonic() + 120
            while len(self.driver.worker_census) < self.workers:
                if self.driver.fatal_error:
                    raise RuntimeError(self.driver.fatal_error)
                if time.monotonic() > deadline:
                    self.driver.fatal_error = '64 attested workers did not join before measurement'
                    raise ValueError('FatalEpisodeError: ' + self.driver.fatal_error)
                time.sleep(.1)
            self.window['start'] = time.monotonic()
            write_json(self.out / 'measurement.json', self.window)
            print('MEASUREMENT_BEGIN warmup_excluded=1 workers=64', flush=True)

    def on_stage_complete(self, stage, ctl, ctx):
        records = completed_rows(self.out / 'journal.jsonl')
        if any(ep.task_uid not in records for ep in stage.episodes):
            self.driver.fatal_error = 'benchmark cohort has a missing/error episode'
            raise ValueError('FatalEpisodeError: ' + self.driver.fatal_error)
        if stage.stage_id == 'eval':
            self.window['end'] = time.monotonic()
            write_json(self.out / 'measurement.json', self.window)
            print(f'MEASUREMENT_END accepted={len(stage.episodes)}', flush=True)


def remote_preflight(config, remote):
    island = config['worker_island']
    expected = hashlib.sha256(config['worker_sha256'].encode()).hexdigest()
    script = ('import hashlib;from pathlib import Path;'
              f'p=Path({island!r})/"worker.sha256";'
              f'assert hashlib.sha256(p.read_bytes()).hexdigest()=={expected!r};'
              'print("WORKER_MANIFEST_IDENTITY_OK")')
    remote('timan107', ['python3', '-c', script], timeout=60, attempts=1)
    remote('timan107', ['bash', '-c', 'cd ' + shlex.quote(island) + ' && sha256sum --quiet -c worker.sha256'],
           timeout=180, attempts=1)
    remote('timan107', ['flock', '-n', FLEET_LOCK, 'true'], timeout=30, attempts=1)


@contextlib.contextmanager
def agent_lease(config, model, out, remote):
    lease = config['worker_island'] + '/decouple_agents/' + Path(out).name + '/lease.json'
    token = uuid.uuid4().hex
    stop = threading.Event()
    def renew(active):
        script = ('import json,os,time;from pathlib import Path;'
                  f'p=Path({lease!r});p.parent.mkdir(parents=True,exist_ok=True);'
                  'q=p.with_suffix(".new");'
                  f'q.write_text(json.dumps(dict(token={token!r},active={active!r},ts=time.time())));'
                  'os.replace(q,p)')
        remote('timan107', ['python3', '-c', script], timeout=30, attempts=1)
    renew(True)
    command = worker_command(config, model, lease, token)
    argv = ['env', *[f'{k}={v}' for k, v in config['worker_environment'].items()], *command]
    log = str(Path(lease).parent / 'agent.log')
    script = ('cd ' + shlex.quote(config['worker_island']) + ' && nohup ' + shlex.join(argv) +
              ' >' + shlex.quote(log) + ' 2>&1 </dev/null &')
    def heartbeat():
        while not stop.wait(15):
            try:
                renew(True)
            except Exception as error:
                print(f'AGENT_LEASE_RENEW_FAILED {error}', flush=True)
    thread = threading.Thread(target=heartbeat, daemon=True)
    try:
        # A lost launch reply does not mean no fleet started. Match R16's
        # four tether retries; its fleet lock refuses duplicate agents.
        remote('timan107', ['bash', '-c', script], timeout=30)
        thread.start()
        write_json(Path(out) / 'agent_lease.json', dict(host='timan107', lease_path=lease, token=token, log=log,
                                                      argv=command, timeout_s=90, new_worker_bytes=False))
        yield
    finally:
        stop.set()
        if thread.ident:
            thread.join(35)
        renew(False)
        # Fail closed if our worker fleet has not released; never stop another
        # agent or launch the next arm in a lease-expiry gap.
        deadline = time.monotonic() + 100
        while True:
            try:
                remote('timan107', ['flock', '-n', FLEET_LOCK, 'true'], timeout=10, attempts=1)
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise RuntimeError('own worker lease did not release; wait 90 seconds before resume')
                time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model', choices=('groot', 'pi05'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        parser.error('GPU package only; this command starts the deployed worker agent')
    config = json.loads(args.config.read_text())
    case = config['cases'][args.model]
    group = read_group(args.out / 'group.json')
    if group['init_pool'] != 'B' or group['benchmark_manifest_sha256'] != case['manifest_sha256']:
        raise ValueError('changed benchmark cohort/pool')
    from exp.offline_search.closed_loop.ops.h100.control import remote
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    remote_preflight(config, remote)
    endpoint = f"{config['advertised_host']}:{args.port}"
    endpoints = {endpoint: ['libero_spatial']}
    # Local control path avoids hairpin NAT; worker tasks retain the already
    # forwarded public hostname and port used by R16.
    preload(group, {f'127.0.0.1:{args.port}': ['libero_spatial']})
    strategy = BenchmarkStrategy(group, case, args.out, config['workers'])
    def ctl(server):
        return WebsocketClientPolicy(host='127.0.0.1', port=args.port)
    def per_step(name, rows):
        with (args.out / 'per_step.jsonl').open('a') as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')
    driver = R16Driver(strategy, endpoints, yaml_weights={case['row']['arm']:len(case['manifest']['pairs'])},
                       servers=[ServerEndpoint(config['advertised_host'], args.port)],
                       journal_path=args.out / 'journal.jsonl', ctl_factory=ctl,
                       bind_host='0.0.0.0', bind_port=config['driver_port'], per_step_writer=per_step,
                       episode_timeout_s=600)
    stopped = threading.Event()
    def interrupt(signum, frame):
        stopped.set()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupt)
    try:
        with agent_lease(config, args.model, args.out, remote):
            started = time.monotonic()
            def stop():
                if time.monotonic() - started > 120 and not driver.worker_census:
                    driver.fatal_error = 'no attested timan107 workers'
                return stopped.is_set() or bool(driver.fatal_error)
            driver.run(stop)
            if stopped.is_set() or driver.fatal_error or not driver.scheduler.all_done():
                raise RuntimeError(driver.fatal_error or 'benchmark interrupted')
            if any(rt.fatal for rt in driver.scheduler._rt.values()):
                raise RuntimeError('benchmark stage failed')
            write_json(args.out / 'DONE.json', dict(accepted=len(case['manifest']['pairs']), manifest_sha256=case['manifest_sha256'],
                                                   run_id=driver.run_id))
    finally:
        write_json(args.out / 'worker_census.json', driver.worker_census)
        for client in driver._ctls.values():
            client.close()


if __name__ == '__main__':
    main()
