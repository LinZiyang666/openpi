"""Start/end kernel evidence, transported with the existing worker snapshots."""
from pathlib import Path
import json
import re
import subprocess
import time

from .node import atomic

OOM = re.compile(r'out of memory|oom-kill|killed process|memory cgroup out of memory', re.I)


def counters(proc='/proc', cgroup='/sys/fs/cgroup'):
    """Two ordinary-user reads per lane, never a polling loop; standalone RPC safe."""
    import os
    import socket
    import time
    from pathlib import Path
    proc, cgroup = Path(proc), Path(cgroup)
    result = dict(time=time.time(), host=socket.gethostname().split('.')[0], uid=os.getuid(),
                  vmstat={}, cgroups={}, errors={})
    for key, path in [('boot_id', proc/'sys/kernel/random/boot_id'), ('uptime', proc/'uptime')]:
        try:result[key] = path.read_text().strip().split()[0]
        except OSError as exc:result['errors'][str(path)] = str(exc)
    if 'uptime' in result:result['boot_epoch'] = result['time'] - float(result['uptime'])
    try:
        values = dict(line.split() for line in (proc/'vmstat').read_text().splitlines())
        result['vmstat']['oom_kill'] = int(values['oom_kill'])
    except (OSError, ValueError, KeyError) as exc:result['errors']['vmstat'] = str(exc)
    # Select the persistent user slice that actually contains this process.
    # A same-named slice elsewhere is not evidence for this process's children.
    try:
        own = next(line.split(':', 2)[2] for line in (proc/'self/cgroup').read_text().splitlines() if line.startswith('0::'))
        parts = Path(own).parts[1:]
        user = 'user-%d.slice' % os.getuid()
        if user in parts:
            directory = cgroup.joinpath(*parts[:parts.index(user)+1])
            path = directory/'memory.events'
            st = path.stat()
            values = dict(line.split() for line in path.read_text().splitlines())
            result['cgroups'][str(path)] = dict(device=st.st_dev, inode=st.st_ino,
                oom=int(values['oom']), oom_kill=int(values['oom_kill']), covers_self=True)
        else:result['errors']['cgroup'] = 'current process is not inside a persistent user slice'
    except (OSError, ValueError, KeyError, StopIteration) as exc:result['errors']['cgroup'] = str(exc)
    return result


def begin(run, *, execute=subprocess.run, snapshot=None):
    import hashlib
    snapshot = counters() if snapshot is None else dict(snapshot)
    # Capture before workers spawn, not at prepare time. No kernel-log polling.
    try:
        output = execute(['dmesg'], text=True, capture_output=True, timeout=20)
        lines = [(line, re.match(r'^\[\s*([0-9.]+)\]\s*(.*)$', line))
                 for line in output.stdout.splitlines()]
        lines = [(line, match) for line, match in lines if match]
        if output.returncode or not lines:
            raise ValueError(output.stderr[-600:] or 'no readable dmesg records')
        line, match = lines[-1]
        snapshot['dmesg_cursor'] = dict(status='PASS', boot_id=snapshot.get('boot_id'),
            monotonic=float(match[1]), record_sha256=hashlib.sha256(line.encode()).hexdigest(),
            retained_from_monotonic=float(lines[0][1][1]), parsed_lines=len(lines))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        snapshot['dmesg_cursor'] = dict(status='UNKNOWN', error=str(exc)[-600:])
    atomic(Path(run)/'AUTOSCALE_KERNEL_START.json', snapshot)
    return snapshot


def counter_evidence(started, ended, before, after):
    """No invented historical baseline: positive lifetime counters need a pair."""
    sources = []
    same_boot = bool(before and before.get('boot_id') and before.get('boot_id') == after.get('boot_id'))
    paired = (same_boot and before.get('host') == after.get('host') and
              before.get('time', float('inf')) <= started and after.get('time', 0) >= ended)
    live_window = after.get('boot_epoch', float('inf')) <= started <= ended <= after.get('time', 0)
    value = after.get('vmstat', {}).get('oom_kill')
    old = (before or {}).get('vmstat', {}).get('oom_kill')
    if paired and isinstance(old, int) and isinstance(value, int) and value >= old:
        sources.append(dict(source='/proc/vmstat:oom_kill', scope='node', before=old, after=value, delta=value-old))
    elif live_window and value == 0:
        sources.append(dict(source='/proc/vmstat:oom_kill', scope='node', before=0, after=0, delta=0,
                            basis='zero since boot; covers the entire lane window'))
    if paired and before.get('uid') == after.get('uid'):
        for path, last in after.get('cgroups', {}).items():
            first = before.get('cgroups', {}).get(path, {})
            if (first.get('covers_self') and last.get('covers_self') and
                all(first.get(k) == last.get(k) for k in ('device', 'inode')) and
                all(isinstance(first.get(k), int) and isinstance(last.get(k), int) and last[k] >= first[k] for k in ('oom', 'oom_kill'))):
                sources.append(dict(source=path, scope='user_slice',
                    before={k:first[k] for k in ('oom', 'oom_kill')}, after={k:last[k] for k in ('oom', 'oom_kill')},
                    delta={k:last[k]-first[k] for k in ('oom', 'oom_kill')}))
    nonzero = any(any(s['delta'].values()) if isinstance(s['delta'], dict) else s['delta'] for s in sources)
    return dict(status=('FAIL' if nonzero else 'PASS') if sources else 'UNKNOWN', coverage=bool(sources),
                source='counters' if sources or after.get('vmstat') or after.get('cgroups') else None, counter_sources=sources,
                counter_start=before, counter_end=after,
                reason='counter increase in covering interval' if nonzero else
                       'zero counter delta over covering interval' if sources else 'no comparable start/end counters covering lane')


def collect(run, started, *, execute=subprocess.run, now=time.time, snapshot=None, ended_at=None, persist=True):
    import hashlib
    ended = now() if ended_at is None else ended_at
    result = dict(started=started, ended=ended, status='UNKNOWN', oom_lines=[], source=None,
                  read_errors=[], log_evidence=[])
    baseline = Path(run)/'AUTOSCALE_KERNEL_START.json'
    try:before = json.loads(baseline.read_text())
    except (OSError, ValueError):before = None
    after = counters() if snapshot is None else snapshot
    # Plain dmesg is already boot-relative [seconds]. timan107 rejects
    # --time-format raw (raw is not a supported time-format there).
    commands = [(['dmesg'], 'dmesg'),
                (['journalctl', '-k', '--since', '@%.6f' % started, '--until', '@%.6f' % ended,
                  '--no-pager', '-o', 'json'], 'journal')]
    uptime = float(Path('/proc/uptime').read_text().split()[0])
    boot_epoch = now() - uptime
    for argv, source in commands:
        try:
            output = execute(argv, text=True, capture_output=True, timeout=20)
            if output.returncode or (source == 'dmesg' and not output.stdout.strip()) or any(
                    hint in output.stderr.lower() for hint in ('permission', 'not seeing', 'no journal')):
                result['read_errors'].append(dict(source=source,returncode=output.returncode,stderr=output.stderr[-600:]))
                continue
            messages, covered, stamps = [], False, []
            lines = output.stdout.splitlines()
            cursor = (before or {}).get('dmesg_cursor', {})
            anchor = None
            if (source == 'dmesg' and cursor.get('status') == 'PASS' and
                    cursor.get('boot_id') and cursor['boot_id'] == after.get('boot_id') and
                    before.get('time', float('inf')) <= started and after.get('time', 0) >= ended):
                anchor = next((i for i, line in enumerate(lines)
                    if hashlib.sha256(line.encode()).hexdigest() == cursor['record_sha256']), None)
            if anchor is not None:
                covered = True
            for index, line in enumerate(lines):
                if not line.strip() or (source == 'journal' and line.strip() == '-- No entries --'):
                    continue
                if source == 'journal':
                    row = json.loads(line)
                    stamp = int(row['__REALTIME_TIMESTAMP']) / 1e6
                    message = row.get('MESSAGE', '')
                else:
                    match = re.match(r'^\[\s*([0-9.]+)\]\s*(.*)$', line)
                    if not match:continue
                    stamp, message = boot_epoch + float(match[1]), match[2]
                stamps.append(stamp)
                covered |= stamp <= started
                in_window = started <= stamp <= ended if anchor is None else index > anchor and stamp <= ended
                if in_window and OOM.search(message):
                    messages.append(dict(time=stamp, message=message))
            # Absence is proof only when retention reaches the run's start.
            # This is a second bounded read at shutdown, never a polling loop.
            if source == 'journal' and not messages:
                anchor = execute(['journalctl', '-k', '-b', '--until', '@%.6f' % started,
                                  '--no-pager', '-n', '1', '-o', 'json'],
                                 text=True, capture_output=True, timeout=20)
                covered = not anchor.returncode and not any(
                    hint in anchor.stderr.lower() for hint in ('permission', 'not seeing', 'no journal')) and any(
                    int(json.loads(line)['__REALTIME_TIMESTAMP']) / 1e6 <= started
                    for line in anchor.stdout.splitlines() if line.startswith('{'))
            if stamps:
                result['log_evidence'].append(dict(source=source,command=argv,retained_from=min(stamps),
                    retained_until=max(stamps),coverage=covered,parsed_lines=len(stamps),
                    start_cursor_retained=anchor is not None))
            if messages or covered:
                result.update(status='FAIL' if messages else 'PASS', oom_lines=messages,
                              source=source, command=argv, coverage=covered)
                break
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            result['read_errors'].append(dict(source=source,error=str(exc)[-600:]))
            continue
    fallback = counter_evidence(started, ended, before, after)
    result['counter_sources'] = fallback['counter_sources']
    if result['status'] == 'UNKNOWN':
        result.update(fallback)
        if fallback['status']=='UNKNOWN' and result['log_evidence']:
            # Preserve the actually readable source even when retention no
            # longer covers the historical lane. Counter UNKNOWN is not zero.
            observed=result['log_evidence'][0]
            result.update(source=observed['source'],reason='readable kernel log does not cover lane start; '+fallback['reason'])
    else:
        result.update(counter_start=before, counter_end=after)
        if fallback['status'] == 'FAIL':
            result.update(status='FAIL', coverage=True, reason=fallback['reason'])
        elif (before and before.get('boot_id') and after.get('boot_id') and
              before['boot_id'] != after['boot_id'] and result['status'] != 'FAIL'):
            result.update(status='UNKNOWN', coverage=False, reason='node rebooted during lane')
    if persist:atomic(Path(run) / 'AUTOSCALE_KERNEL.json', result)
    return result
