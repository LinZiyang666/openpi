"""Wall-clock admission, interruptible cutoff, and bounded owned-child cleanup."""
from __future__ import annotations

import contextlib
from datetime import datetime
import os
import signal
import threading
import time
from zoneinfo import ZoneInfo

DEFAULT_DEADLINE = '2026-10-08 15:15 America/Chicago'
DEADLINE_ENV = 'OPENPI_DECOUPLE_DEADLINE'
CHICAGO = ZoneInfo('America/Chicago')


class PackageDeadlineReached(TimeoutError):
    pass


def parse_deadline(value=DEFAULT_DEADLINE):
    """Accept ISO 8601 (offset/Z optional), or a local time plus IANA zone."""
    value = value.strip()
    if not value:
        raise ValueError('deadline must be a date/time; an empty override cannot disable it')
    parts = value.rsplit(' ', 1)
    zone = CHICAGO
    if len(parts) == 2 and '/' in parts[-1]:
        zone = ZoneInfo(parts[-1])
        value = parts[0]
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=zone)
    return parsed.timestamp()


def deadline_label(epoch):
    return datetime.fromtimestamp(epoch, CHICAGO).isoformat()


class DeadlineBudget:
    def __init__(self, epoch, window_s, reserve_s, *, clock=time.monotonic, wall_clock=time.time):
        self.clock, self.wall_clock = clock, wall_clock
        self.epoch, self.reserve_s = epoch, reserve_s
        # Backward wall-clock adjustments cannot extend the original window.
        self.hard_end = min(clock() + window_s, clock() + epoch - wall_clock())

    def hard_remaining(self):
        return min(self.hard_end - self.clock(), self.epoch - self.wall_clock())

    def remaining(self):
        return self.hard_remaining() - self.reserve_s

    def check(self):
        remaining = self.remaining()
        if remaining <= 0:
            raise PackageDeadlineReached('deadline/window shutdown cutoff reached')
        return remaining

    def fits(self, seconds):
        return seconds <= self.remaining()

    def arm_fields(self):
        return dict(package_deadline=self.hard_end - self.reserve_s,
                    package_wall_deadline=self.epoch - self.reserve_s,
                    package_shutdown_deadline=self.hard_end,
                    package_wall_shutdown_deadline=self.epoch)


@contextlib.contextmanager
def deadline_watchdog(budget, *, enabled=True):
    """Interrupt even blocking gate waits/preflights; leave the reserve for finally."""
    if not enabled:
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)
    stop = threading.Event()
    fired = threading.Event()
    def interrupt(signum, frame):
        # Never interrupt shutdown a second time.
        fired.set()
        raise PackageDeadlineReached('deadline/window shutdown cutoff reached')
    def watch():
        while not stop.wait(min(0.25, max(0.001, budget.remaining()))):
            if budget.remaining() <= 0:
                if not fired.is_set():
                    signal.pthread_kill(threading.main_thread().ident, signal.SIGALRM)
                return
    signal.signal(signal.SIGALRM, interrupt)
    thread = threading.Thread(target=watch, name='package-deadline', daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(1)
        signal.signal(signal.SIGALRM, previous)


@contextlib.contextmanager
def defer_deadline_alarm():
    # A queued cutoff is delivered after all owned groups have been stopped.
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


@defer_deadline_alarm()
def stop_owned_groups(processes, *, hard_deadline=float('inf'), wall_deadline=float('inf'),
                      clock=time.monotonic, wall_clock=time.time, sleep=time.sleep,
                      killpg=os.killpg):
    """TERM every newly created session together, then KILL by a shared deadline.

    Model/relay groups get ten seconds; the driver gets up to 100 seconds to
    revoke its own remote lease. Never wait for remote cleanup before TERMing
    the GPU. KILL also covers descendants if their supervisor has exited.
    """
    started = clock()
    interrupted = None
    groups = []
    def send(pid, sig):
        try:
            killpg(pid, sig)
        except ProcessLookupError:
            pass
    for role, process in processes:
        if process is not None:
            groups.append((process, started + (100 if role == 'worker' else 10)))
            send(process.pid, signal.SIGTERM)
    pending = list(groups)
    while pending:
        try:
            now = clock()
            # Leave five seconds for reaping, NVML close and summary writes.
            hard_stop = min(hard_deadline - 5, now + wall_deadline - wall_clock() - 5)
            for process, grace in list(pending):
                if process.poll() is not None or now >= min(grace, hard_stop):
                    send(process.pid, signal.SIGKILL)
                    pending.remove((process, grace))
            if pending:
                sleep(min(0.1, max(0.001, min(min(grace, hard_stop) for _, grace in pending) - clock())))
        except PackageDeadlineReached as error:
            # A cutoff can arrive during a normal arm's cleanup. Finish owned
            # cleanup before propagating it to the package summary/marker.
            interrupted = error
    for process, _ in groups:
        try:
            remaining = min(2, hard_deadline - clock(), wall_deadline - wall_clock())
            process.wait(timeout=max(0, remaining))
        except TimeoutError:
            pass
        except Exception as error:
            import subprocess
            if not isinstance(error, subprocess.TimeoutExpired):
                raise
    if interrupted:
        raise interrupted
