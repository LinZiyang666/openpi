"""Worker instrumentation; transport metadata only, never changes an EpisodeTask."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import os
import resource
import socket
from pathlib import Path
import time

from .node import atomic, identity, read


@contextmanager
def creation_lock():
    root = Path(os.environ.get('OPENPI_AUTOSCALE_LEDGER', '/tmp/openpi-autoscale-%d' % os.getuid()))
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    gpu = os.environ.get('OPENPI_AUTOSCALE_GPU_UUID')
    if not gpu:
        minor = os.environ.get('MUJOCO_EGL_DEVICE_ID', '0')
        for info in Path('/proc/driver/nvidia/gpus').glob('*/information'):
            fields = dict(line.split(':', 1) for line in info.read_text().splitlines() if ':' in line)
            if fields.get('Device Minor', '').strip() == minor:
                gpu = fields.get('GPU UUID', '').strip()
                break
    # Unknown physical mapping serializes all unknown devices, conservatively.
    key = hashlib.sha256((gpu or 'unknown-physical-gpu').encode()).hexdigest()[:24]
    with (root / ('creation-' + key + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


class TimedClient:
    def __init__(self, native, observer):
        self.native, self.observer = native, observer

    def __getattr__(self, name):
        value = getattr(self.native, name)
        if name not in ('infer', 'episode_start', 'episode_end', 'select_bundle', 'reset') or not callable(value):
            return value
        def call(*args, **kwargs):
            started = time.monotonic()
            try:
                result = value(*args, **kwargs)
                if name == 'infer':
                    timing = result.get('server_timing', {}) if isinstance(result, dict) else {}
                    cost = timing.get('infer_ms')
                    if isinstance(cost, (int, float)) and cost >= 0:
                        self.observer.service_s += cost / 1000
                        self.observer.service_samples += 1
                    self.observer.infer_calls += 1
                return result
            finally:
                self.observer.wait_s += time.monotonic() - started
        return call


class Observer:
    def __init__(self, worker_id, runner):
        self.worker_id, self.runner = worker_id, runner
        self.started = time.monotonic()
        self.completed = 0
        self.wait_s = self.service_s = 0.
        self.infer_calls = self.service_samples = 0
        self.episode = None
        self.generation = '%s:%s' % (os.getpid(), identity(os.getpid()))
        if hasattr(runner, '_client_factory'):
            factory = runner._client_factory
            # Bind now: the environment wrappers below reuse the factory variable.
            runner._client_factory = lambda *a, _factory=factory, **kw: TimedClient(_factory(*a, **kw), self)
        for name in ('_env_factory', '_episode_setup'):
            if hasattr(runner, name):
                factory = getattr(runner, name)
                def serialized(*a, _factory=factory, **kw):
                    with creation_lock():
                        return _factory(*a, **kw)
                setattr(runner, name, serialized)

    def begin(self, task):
        self.wait_s = self.service_s = 0.
        self.infer_calls = self.service_samples = 0
        self.episode_started = time.monotonic()
        self.episode = dict(task_uid=task.task_uid, attempt=task.attempt,
                            cohort=task.yaml_id, generation=self.generation)

    def result(self):
        elapsed = time.monotonic() - self.episode_started
        self.completed += 1
        receipt = os.environ.get('R17_WORKER_RECYCLE_RECEIPT')
        if receipt:
            atomic(Path(receipt).with_name('PEAK.json'), dict(pid=os.getpid(), start_ticks=identity(os.getpid()),
                   peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024))
        return dict(self.episode, node=socket.gethostname().split('.')[0],
                    sim_render_s=max(0, elapsed - self.wait_s), server_wait_s=self.wait_s,
                    service_s=self.service_s if self.service_samples == self.infer_calls and self.infer_calls else None,
                    infer_calls=self.infer_calls, duration_s=elapsed)

    def payload(self):
        snapshot = read(os.environ.get('OPENPI_AUTOSCALE_NODE', ''), {}) or {}
        return dict(node=snapshot, worker_id=self.worker_id, generation=self.generation,
                    retire=self.worker_id in snapshot.get('retired', []))

    def response(self, payload):
        control = payload.get('_autoscale_control')
        path = os.environ.get('OPENPI_AUTOSCALE_CONTROL')
        if path and control and control.get('campaign') == os.environ.get('OPENPI_AUTOSCALE_CAMPAIGN'):
            # Concurrent workers may receive out of order replies. Serialize and
            # compare the driver's epoch+sequence before replacing shared state.
            with Path(path + '.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                old = read(path, {})
                if control.get('time', 0) > old.get('time', 0):
                    atomic(path, control)

    def stop_requested(self):
        # This predicate runs only between episodes, including idle backoff.
        self.recycle()
        snapshot = read(os.environ.get('OPENPI_AUTOSCALE_NODE', ''), {}) or {}
        return bool(snapshot.get('done') or self.worker_id in snapshot.get('retired',[]))

    def recycle(self):
        maximum = int(os.environ.get('OPENPI_AUTOSCALE_MAX_EPISODES', '32'))
        seconds = float(os.environ.get('OPENPI_AUTOSCALE_MAX_SECONDS', '1800'))
        if self.completed and (self.completed >= maximum or time.monotonic() - self.started >= seconds):
            path = os.environ.get('R17_WORKER_RECYCLE_RECEIPT')
            if path:
                atomic(path, dict(schema='r17.worker_recycle.v1', pid=os.getpid(), start_ticks=identity(os.getpid()),
                                  completed_episodes=self.completed, maximum_episodes=maximum,
                                  reason='episode_boundary', elapsed_seconds=time.monotonic() - self.started,
                                  maximum_seconds=seconds))
            raise SystemExit(75)
