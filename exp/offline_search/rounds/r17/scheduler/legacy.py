"""R15/R16 (conductor WorkerAgent fleets) on the central scheduler.

Driver hook `driver_from_environment` (conductor driver.py, OPENPI_WORKER_MEMORY_GUARD=1): the batch's
BatchClient. Agent hook `from_environment` (conductor agent.py, OPENPI_WORKER_GOVERNOR): the batch's
NodeExecutor on this node, spawning the agent's fixed WorkerSpecs. Both read the campaign file named by
OPENPI_AUTOSCALE_CONFIG; its directory holds AUTOSCALE_CONTROL.json / AUTOSCALE_NODE.json.

Launcher side (4090): `campaign` / `node_campaign` describe the batch, `install` puts this package and the
conductor hooks into the frozen R15/R16 worker islands. Legacy batches run only on nodes with such an island:
R15 timan107 (driver and agent in one run_gtp process), R16 timan107 and the 4090 (its servers are local).
"""
from __future__ import annotations

from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import tarfile
import threading
import time

from .client import BatchClient
from .node import SAMPLE_S, NodeExecutor, identity, read

REPO = Path(__file__).resolve().parents[5]
# The islands' conductor predates the hooks (driver/agent/worker.py differ from it only by them);
# Slots takes its GPU inventory from exp.r17_integration.ops. The rest is this package.
RUNTIME = ('src/openpi/conductor/driver.py', 'src/openpi/conductor/agent.py', 'src/openpi/conductor/worker.py',
           'exp/r17_integration/ops/worker_gpu.py', 'exp/r17_integration/ops/formal_process.py')
HEADROOM = {'local': 5120, 'h100': 4096}  # MiB kept on the server's own machine for its server (owner)
EXCHANGE_S = 15


def campaign(batch, server_node, nodes):
    """Batch description for BatchClient and NodeExecutors, narrowed to the nodes the batch has agents on."""
    from .defaults import apply
    c = apply(dict(run_name=batch, domain='libero', server_node=server_node))
    return dict(run_name=batch, domain='libero', server_host=c['server_host'], scheduler_url=c['scheduler_url'],
                scheduler_nodes={n: v for n, v in c['scheduler_nodes'].items() if n in nodes},
                scheduler_node_config={n: v for n, v in c['scheduler_node_config'].items() if n in nodes},
                worker_vram_mib=c['worker_vram_mib'], worker_max_episodes=4, worker_max_seconds=1800)


def node_campaign(c, node):
    local = c['scheduler_nodes'][node]['class'] == 'local'
    return dict(c, worker_node=node, node_workers=c['scheduler_nodes'][node]['max'],
                worker_server_vram_headroom_mib=HEADROOM.get(node, 0) if local else 0)


def payload(extra=()):
    from .defaults import files
    return [(REPO / n, n) for n in sorted({*files(), *RUNTIME})] + list(extra)


def install(root, islands, extra=()):
    """Pack the payload as root/workers.tar (defaults.require checks it at launch); unpack it into each island."""
    from exp.offline_search.closed_loop.ops.h100.control import remote
    from exp.r17_integration.ops.formal_eval import transfer
    archive = Path(root) / 'workers.tar'
    with tarfile.open(archive, 'w') as tar:
        for source, name in payload(extra):
            tar.add(source, arcname=name, recursive=False)
    for node, island in islands.items():
        target = str(archive)
        if node != 'local':
            target = '/tmp/openpi_sched_%s.tar' % hashlib.sha256(archive.read_bytes()).hexdigest()
            transfer(['rsync', 'push', str(archive), node + ':' + target])
        # rsync renames each file into place: a process starting meanwhile never imports a half-written module.
        script = 'd=$(mktemp -d) && tar -xf %s -C "$d" && rsync -a "$d"/ %s/ && rm -rf "$d"' % (
            shlex.quote(target), shlex.quote(str(island)))
        remote(node, ['bash', '-c', script + ('' if node == 'local' else ' ' + shlex.quote(target))])


def driver_from_environment(driver):
    path = Path(os.environ['OPENPI_AUTOSCALE_CONFIG'])
    return BatchClient(driver, json.loads(path.read_text()), path.parent)


class AgentGovernor:
    """The batch's NodeExecutor driving a WorkerAgent: spawn/tick/close are the agent's governor hooks."""

    def __init__(self, agent, campaign, run):
        from .fleet import Slots
        self.agent, self.campaign, self.run = agent, campaign, Path(run)
        self.executor = NodeExecutor(campaign, run)
        self.slots = Slots(campaign, run)
        self.slots.governor, self.executor.slots = self.executor, self.slots
        self.paths = self.executor.worker_env('')
        self.handles, self.sampled, self.gpu_retry = {}, float('-inf'), 0.
        self.supervisor = threading.Thread(target=self.supervise, daemon=True)
        self.supervisor.start()

    def tick(self):
        """Reap exited workers on every call (before a respawn reuses the id); sample the node every SAMPLE_S."""
        for worker_id, handle in list(self.handles.items()):
            code = handle.poll()
            if code is not None:
                with self.executor.lock():
                    self.executor.finished(worker_id, normal=code == 75)
                self.slots.release(worker_id)
                del self.handles[worker_id]
        if time.monotonic() - self.sampled >= SAMPLE_S:
            self.sampled = time.monotonic()
            self.executor.tick()

    def spawn(self, spec, spawn_fn, host, port):
        """The agent offers its specs in order (R15/R16: round-robin over servers); one spawn per node tick."""
        self.tick()
        if time.monotonic() < self.gpu_retry:
            return None
        with self.executor.permit() as allowed:
            if not allowed:
                return None
            if not self.slots.acquire(spec.worker_id):
                self.gpu_retry = time.monotonic() + SAMPLE_S
                return None
            card = self.slots.assignments[spec.worker_id]
            receipt = str(self.run / (spec.worker_id + '.recycle.json'))
            env = dict(spec.env, **self.executor.worker_env(spec.worker_id), MUJOCO_EGL_DEVICE_ID=str(card['device']),
                       OPENPI_AUTOSCALE_GPU_UUID=card['uuid'], R17_WORKER_RECYCLE_RECEIPT=receipt)
            try:
                handle = spawn_fn(replace(spec, gpu_id=str(card['device']), env=env), host, port)
            except BaseException:
                self.slots.release(spec.worker_id)
                raise
            self.executor.register(dict(worker_id=spec.worker_id, pid=handle.pid, start_ticks=identity(handle.pid),
                                        egl_device=card['device'], gpu_uuid=card['uuid'], recycle_receipt=receipt))
            self.handles[spec.worker_id] = handle
            return handle

    def exchange(self):
        """Supervisor pull: node snapshot to the driver, control back (works with no live worker)."""
        from openpi.conductor import protocol
        with socket.create_connection((self.agent._driver_host, self.agent._driver_port), timeout=3) as conn:
            protocol.send_message(conn, protocol.MSG_PULL, dict(
                _autoscale_supervisor=True, _autoscale=dict(node=read(self.paths['OPENPI_AUTOSCALE_NODE'], {}))))
            _, reply = protocol.recv_message(conn)
        control, path = reply.get('_autoscale_control') or {}, self.paths['OPENPI_AUTOSCALE_CONTROL']
        if control.get('campaign') == self.campaign['run_name']:
            with open(path + '.lock', 'a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if control.get('time', 0) > (read(path, {}) or {}).get('time', 0):
                    # Own part name: in R15 the BatchClient publishes this file from the same process.
                    part = Path(path + '.supervisor.part')
                    part.write_text(json.dumps(control, sort_keys=True) + '\n')
                    os.replace(part, path)

    def supervise(self):
        from openpi.conductor.protocol import ProtocolError
        try:
            while True:
                try:
                    self.exchange()
                except (OSError, ValueError, ProtocolError):
                    pass  # driver not up yet / restarting
                if self.agent._stop.wait(EXCHANGE_S):
                    return
        finally:
            self.slots.close()
            from .kernel_evidence import collect
            collect(self.run, self.executor.started_wall)

    def close(self):
        self.supervisor.join(timeout=90)  # the kernel evidence reads have 20 s timeouts


def from_environment(agent):
    path = Path(os.environ['OPENPI_AUTOSCALE_CONFIG'])
    return AgentGovernor(agent, json.loads(path.read_text()), path.parent)
