"""Set parent-death ownership in the child, safely before exec and imports.

The fleet has a GPU sampler thread. A Python preexec_fn in that multithreaded
parent can deadlock after fork; execute this tiny wrapper instead.
"""
import ctypes
import os
import signal
import sys


def main():
    expected = int(os.environ['OPENPI_AUTOSCALE_PARENT_PID'])
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_PDEATHSIG failed')
    if os.getppid() != expected:
        raise SystemExit(73)
    argv = sys.argv[1:]
    if argv and argv[0] == '--':argv = argv[1:]
    if not argv:raise ValueError('worker command required')
    os.execvpe(argv[0], argv, os.environ)


if __name__ == '__main__':main()
