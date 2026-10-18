"""Dynamic legacy adapter, using the driver's ordinary pull TCP channel."""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import socket
import threading
import time

from .node import LocalGovernor, identity, atomic
from .driver import DriverController


class LegacyDriver:
    """Discover the single legacy fleet's explicit maximum before any dispatch.

    R15 has driver and agent in one process, R16 on different machines. A
    supervisor pull carries no episode request and works when active == 0.
    """
    def __init__(self, driver):
        self.driver, self.inner = driver, None
        self.lock = threading.RLock()
        self.root = driver.journal.path.parent/'autoscale'

    def ingest(self, data):
        row = data.get('node', {})
        if self.inner is None and row.get('node') in ('timan1', 'timan107', 'local'):
            maximum = int(row.get('maximum', 0))
            if maximum < 1 or not row.get('campaign'):
                return
            c = dict(run_name=row['campaign'], domain='libero', workers=maximum,
                     worker_nodes={row['node']:maximum}, autoscale=dict(mode=os.environ.get('OPENPI_WORKER_AUTOSCALE', 'dynamic')))
            self.inner = DriverController(self.driver, c, self.root)
        if self.inner:
            self.inner.ingest(data)

    def pull(self, payload):
        with self.lock:
            self.ingest(payload.get('_autoscale', {}))
            return self.inner.pull(payload) if self.inner else (False, {})

    def tick(self):
        with self.lock:
            if self.inner:self.inner.tick()

    def result(self, payload, accepted):
        if self.inner:self.inner.result(payload, accepted)


def driver_from_environment(driver):
    return LegacyDriver(driver)


class AgentGovernor:
    def __init__(self, agent):
        self.agent = agent
        node = socket.gethostname().split('.')[0]
        if node not in ('timan107', 'timan1'):node = 'local'
        run = Path(os.environ.get('OPENPI_AUTOSCALE_RUN', '/tmp/openpi-autoscale-%d/legacy-%d' % (os.getuid(), os.getpid())))
        campaign = dict(run_name=os.environ.get('OPENPI_AUTOSCALE_CAMPAIGN', run.name),
                        model=os.environ.get('OPENPI_AUTOSCALE_MODEL', 'legacy-libero'), domain='libero',
                        checkpoint=os.environ.get('OPENPI_AUTOSCALE_CHECKPOINT', ''),
                        worker_node=node, node_workers=len(agent._specs), worker_max_episodes=4,
                        worker_max_seconds=1800, worker_vram_mib=530,
                        worker_egl_devices=sorted({int(s.gpu_id) for s in agent._specs}),
                        autoscale_node_config=__import__('json').loads(os.environ.get('OPENPI_AUTOSCALE_NODE_CONFIG','{}')),
                        autoscale_external_legacy_nodes=list(__import__('json').loads(os.environ.get('OPENPI_AUTOSCALE_NODE_CONFIG','{}'))),
                        autoscale=dict(mode=os.environ.get('OPENPI_WORKER_AUTOSCALE', 'dynamic')))
        self.local = LocalGovernor(campaign, run)
        from .fleet import Slots
        if node == 'timan1':
            from exp.offline_search.rounds.r17.filter_failed.timan1_gpu2 import install_runtime
            install_runtime()
        self.slots = Slots(campaign, run)
        self.slots.governor=self.local
        self.local.slots=self.slots
        self.registered, self.generation = {}, {}
        self.last = 0
        self.gpu_retry_after = 0
        self.heartbeat = threading.Thread(target=self.supervise, daemon=True)
        self.heartbeat.start()

    def exchange(self):
        from openpi.conductor import protocol as p
        with socket.create_connection((self.agent._driver_host, self.agent._driver_port), timeout=3) as conn:
            snapshot = dict(self.local.snapshot, maximum=self.local.maximum)
            p.send_message(conn, p.MSG_PULL, dict(_autoscale_supervisor=True, _autoscale=dict(node=snapshot)))
            _, payload = p.recv_message(conn)
            control = payload.get('_autoscale_control')
            if control and control.get('campaign') == self.local.campaign['run_name']:
                atomic(self.local.control_path, control)

    def supervise(self):
        from openpi.conductor.protocol import ProtocolError
        try:
            while not self.agent._stop.is_set():
                try:self.exchange()
                except (OSError, ValueError, ProtocolError):pass
                if self.agent._stop.wait(15):break
        finally:
            try:
                if getattr(self, "local", None) is not None:
                    from .kernel_evidence import collect
                    collect(self.local.run, self.local.started_wall)
            finally:
                self.slots.close()

    def close(self):
        # Three shutdown-only kernel reads have 20s timeouts; never poll them.
        self.heartbeat.join(timeout=65)
        if self.heartbeat.is_alive():
            atomic(self.local.run / "AUTOSCALE_KERNEL.json", dict(status="UNKNOWN", coverage=False,
                oom_lines=[], reason="bounded shutdown evidence timeout"))

    def tick(self):
        now = self.local.clock()
        if self.local.snapshot and now - self.last < self.local.config.sample_s:return
        self.last = now
        self.local.tick()
        for worker_id in list(self.registered):
            handle = self.agent._handles.get(worker_id)
            if handle is not None and handle.poll() is not None:
                with self.local.lock():
                    self.local.finished(worker_id, normal=handle.poll() == 75)
                self.slots.release(worker_id)
                del self.registered[worker_id]

    def spawn(self, spec, spawn_fn, host, port):
        self.tick()
        if (not self.local.snapshot.get('admit') or len(self.local.entries) >= self.local.target
                or self.local.retired or self.local.clock() < self.gpu_retry_after):
            return None
        with self.local.permit() as allowed:
            if not allowed:return None
            if not self.slots.acquire(spec.worker_id):
                self.gpu_retry_after = self.local.clock() + self.local.config.sample_s
                return None
            assignment = self.slots.assignments[spec.worker_id]
            env = dict(spec.env, **self.local.worker_env(spec.worker_id))
            env.update(MUJOCO_EGL_DEVICE_ID=str(assignment['device']),
                       OPENPI_AUTOSCALE_GPU_UUID=assignment['uuid'])
            env['R17_WORKER_RECYCLE_RECEIPT'] = str(self.local.run / (spec.worker_id + '.recycle.json'))
            try:handle = spawn_fn(replace(spec, gpu_id=str(assignment['device']), env=env), host, port)
            except BaseException:
                self.slots.release(spec.worker_id)
                raise
            entry = dict(worker_id=spec.worker_id, pid=handle.pid, start_ticks=identity(handle.pid),
                         egl_device=assignment['device'],gpu_uuid=assignment['uuid'],
                         generation=self.generation.get(spec.worker_id, 0), recycle_receipt=env['R17_WORKER_RECYCLE_RECEIPT'])
            self.generation[spec.worker_id] = entry['generation'] + 1
            self.registered[spec.worker_id] = entry
            self.local.register(entry)
            return handle


def from_environment(agent):
    return AgentGovernor(agent) if os.environ.get('OPENPI_WORKER_MEMORY_GUARD') == '1' else None
