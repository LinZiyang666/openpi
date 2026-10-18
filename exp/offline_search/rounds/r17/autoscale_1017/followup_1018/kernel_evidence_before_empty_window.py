"""One local end-of-run kernel read; transported with existing worker evidence."""
from pathlib import Path
import json
import re
import subprocess
import time

from .node import atomic

OOM = re.compile(r'out of memory|oom-kill|killed process|memory cgroup out of memory', re.I)


def collect(run, started, *, execute=subprocess.run, now=time.time):
    ended = now()
    result = dict(started=started, ended=ended, status='UNKNOWN', oom_lines=[], source=None)
    commands = [(['journalctl', '-k', '--since', '@%.6f' % started, '--until', '@%.6f' % ended,
                 '--no-pager', '-o', 'json'], 'journal'),
                (['dmesg', '--time-format', 'raw'], 'dmesg')]
    uptime = float(Path('/proc/uptime').read_text().split()[0])
    boot_epoch = ended - uptime
    for argv, source in commands:
        try:
            output = execute(argv, text=True, capture_output=True, timeout=20)
            if output.returncode or not output.stdout.strip() or any(
                    hint in output.stderr.lower() for hint in ('permission', 'not seeing', 'no journal')):
                continue
            messages, covered = [], False
            for line in output.stdout.splitlines():
                if source == 'journal':
                    row = json.loads(line)
                    stamp = int(row['__REALTIME_TIMESTAMP']) / 1e6
                    message = row.get('MESSAGE', '')
                else:
                    match = re.match(r'^\[\s*([0-9.]+)\]\s*(.*)$', line)
                    if not match:continue
                    stamp, message = boot_epoch + float(match[1]), match[2]
                covered |= stamp <= started
                if started <= stamp <= ended and OOM.search(message):
                    messages.append(dict(time=stamp, message=message))
            # Absence is proof only when retention reaches the run's start.
            # This is a second bounded read at shutdown, never a polling loop.
            if source == 'journal' and not messages:
                anchor = execute(['journalctl', '-k', '-b', '--until', '@%.6f' % started,
                                  '--no-pager', '-n', '1', '-o', 'json'],
                                 text=True, capture_output=True, timeout=20)
                covered = not anchor.returncode and any(
                    int(json.loads(line)['__REALTIME_TIMESTAMP']) / 1e6 <= started
                    for line in anchor.stdout.splitlines() if line.startswith('{'))
            if messages or covered:
                result.update(status='FAIL' if messages else 'PASS', oom_lines=messages,
                              source=source, command=argv, coverage=covered)
                break
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            continue
    atomic(Path(run) / 'AUTOSCALE_KERNEL.json', result)
    return result
