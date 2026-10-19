"""Continue the serial A100 closed-loop queue after the 18:54 DP staging failure.

DP -> Fast-WAM -> Cosmos, same fixed plans/arms/capacity as pi05/GR00T. Reuses
the original stage / A100 queue / collect code. Differences from the stopped
coordinator (a100_closedloop_all_1018.py):

* nothing is deleted: the old cleanup() (shutil.rmtree of model copies) is not
  called. Study replica files that must leave the 58 GB A100 disk are MOVED to
  this coordinator's /tmp with rsync --remove-source-files through a temporary
  A100 daemon (port 29770/29771, never 231xx), closed by PID afterwards;
* a model rootfs that the next model shares (same fastwam island / LIBERO
  sources) is renamed on the same filesystem instead of being re-transferred.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a100_closedloop_all_1018 as Q  # noqa: E402  launch/state/wait/collect
from a100_closedloop_prepare_1018 import OUT, REMOTE, write  # noqa: E402

PUBLIC = '149.165.152.232'
ME = '192.80.109.240'
# Replicas of local originals go to /tmp (non-unique copies only); the A100's
# own unique raw results go to a durable /home location.
ASIDE = Path('/tmp/xgpu_1017_a100_aside')
DURABLE = Path('/home/weiland/xgpu_1017_a100_aside')
JOURNAL = OUT / 'CONTINUE_1018b.jsonl'
STUDY = Path('/home/exouser/xgpu_1017')


def log(event, **kw):
    row = dict(time=time.time(), event=event, **kw)
    with JOURNAL.open('a') as f:
        f.write(json.dumps(row, sort_keys=True) + '\n')
    print(json.dumps(row, sort_keys=True), flush=True)


def tx(code, seconds='60s'):
    r = subprocess.run(['tether', 'exec', '--timeout', seconds, 'a100', '--', 'python3', '-c', code],
                       text=True, capture_output=True)
    if r.returncode:
        raise RuntimeError('A100 control failed: ' + r.stderr[-2000:])
    return r.stdout


def stage(model):
    log_path = OUT / (model + '_stage.log')
    if log_path.exists():
        log_path.rename(log_path.with_name(log_path.stem + '_failed_' + str(int(time.time())) + '.log'))
    with log_path.open('xb') as f:
        subprocess.run([sys.executable, str(HERE / 'a100_closedloop_stage_1018.py'), model],
                       stdout=f, stderr=subprocess.STDOUT, check=True)
    log('staged', model=model)


def start(model):
    if Q.state(model) is not None:
        raise RuntimeError(model + ' A100 queue receipt already exists; never rerun accepted episodes')
    stage(model)
    launch = Q.launch(model)
    log('launched', model=model, launch=launch)
    return launch


def relaunch(model):
    """Native --resume-startup: only for a FAILED queue with no accepted episode."""
    s = Q.state(model)
    assert s and s['status'] == 'FAILED' and not s['completed'], s
    c = json.loads((OUT / model / 'pure/campaign_template.json').read_text())
    argv = [c['server_python'], '-u', str(REMOTE / model / 'execution_tools/a100_closedloop_queue_1018.py'),
            '--model', model, '--resume-startup']
    name = model + '_queue_resume_' + str(int(time.time())) + '.log'
    code = ('from pathlib import Path;import subprocess,json; '
            f'r=Path({str(REMOTE)!r});log=(r/{name!r}).open("xb"); '
            f'p=subprocess.Popen({argv!r},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); '
            'print(json.dumps(dict(pid=p.pid)))')
    launch = Q.control(code)
    log('relaunched_startup', model=model, launch=launch, log=name, failed_state=s)
    return launch


def launch_wait(model):
    """Base queue queued directly behind the current GPU holder (blocking lock)."""
    if Q.state(model) is not None:
        raise RuntimeError(model + ' A100 queue receipt already exists')
    c = json.loads((OUT / model / 'pure/campaign_template.json').read_text())
    argv = [c['server_python'], '-u', str(REMOTE / model / 'execution_tools/a100_closedloop_queue_1018.py'),
            '--model', model, '--wait-lock']
    name = model + '_queue.log'
    code = ('from pathlib import Path;import subprocess,json; '
            f'r=Path({str(REMOTE)!r});log=(r/{name!r}).open("xb"); '
            f'p=subprocess.Popen({argv!r},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); '
            'print(json.dumps(dict(pid=p.pid)))')
    launch = Q.control(code)
    log('launched_wait_lock', model=model, launch=launch, log=name)
    return launch


def finish(model):
    s = Q.wait(model)
    log('queue_complete', model=model, state=s)
    paired = Q.collect(model)
    log('collected', model=model, paired=str(paired))


def free_port():
    out = subprocess.run(['ss', '-ltn'], text=True, capture_output=True, check=True).stdout
    remote = subprocess.run(['tether', 'exec', '--timeout', '30s', 'a100', '--', 'ss', '-ltn'],
                            text=True, capture_output=True, check=True).stdout
    for port in (29770, 29771, 29772):
        if f':{port} ' not in out and f':{port} ' not in remote:
            return port, out, remote
    raise RuntimeError('no study transfer port free outside 23100-23199')


def aside(name, base, relpaths, writable_roots=(), root=ASIDE):
    """Move study replica files (relative to base on A100) to local /tmp."""
    base = Path(base)
    assert str(base).startswith(str(STUDY)) and relpaths
    stamp = name + '_' + str(int(time.time()))
    work = OUT / 'transfer' / ('aside_' + stamp)
    work.mkdir(parents=True)
    listing = work / 'files.txt'
    listing.write_text(''.join(p.lstrip('/') + '\n' for p in relpaths))
    destination = root / stamp
    destination.mkdir(parents=True)
    # Own replica trees keep their source modes (some r-x). Grant owner-write on
    # directories only (recorded), so the sending daemon may unlink moved files;
    # surviving directories get their recorded modes back afterwards.
    roots = [str(r) for r in writable_roots]
    assert all(r.startswith(str(base)) for r in roots)
    modes = str(STUDY / 'cl18/transfer' / ('modes_' + stamp + '.json'))
    changed = json.loads(tx('import os,stat,json\nout=[]\n'
        f'for root in {roots!r}:\n'
        ' for d,ds,fs in os.walk(root):\n'
        '  m=stat.S_IMODE(os.stat(d).st_mode)\n'
        '  if not m&stat.S_IWUSR: os.chmod(d,m|stat.S_IWUSR); out.append([d,m])\n'
        f'open({modes!r},"x").write(json.dumps(out))\n'
        'print(json.dumps(len(out)))\n', '300s'))
    record_modes = dict(changed_directories=changed, a100_mode_file=modes)
    port, ss_local, ss_remote = free_port()
    daemon = STUDY / 'cl18/transfer' / ('aside_' + stamp)
    config = (f'use chroot = false\nread only = false\nlist = false\nhosts allow = {ME}\n'
              f'pid file = {daemon}/daemon.pid\nlog file = {daemon}/daemon.log\n'
              f'[aside]\npath = {base}\nread only = false\nlist = false\n')
    code = ('from pathlib import Path; import subprocess,json\n'
            f'd=Path({str(daemon)!r});d.mkdir(parents=True,exist_ok=False)\n'
            f'c=d/"rsyncd.conf";c.write_text({config!r})\n'
            'f=(d/"stdio.log").open("ab")\n'
            f'p=subprocess.Popen(["rsync","--daemon","--no-detach","--config="+str(c),"--port={port}"],'
            'stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)\n'
            'print(json.dumps({"pid":p.pid}))\n')
    pid = json.loads(tx(code, '30s'))['pid']
    record = dict(name=name, base=str(base), files=len(relpaths), destination=str(destination), port=port,
                  daemon_pid=pid, **record_modes, started=time.time(), io_before=Path('/proc/pressure/io').read_text())
    write(work / 'ASIDE.json', record)
    try:
        time.sleep(.5)
        cmd = ['rsync', '-a', '--remove-source-files', '--bwlimit=16384', '--contimeout=10', '--timeout=300',
               '--files-from=' + str(listing), f'rsync://{PUBLIC}:{port}/aside/', str(destination) + '/']
        record['command'] = cmd
        subprocess.run(cmd, check=True)
        record['status'] = 'MOVED'
    finally:
        subprocess.run(['tether', 'exec', '--timeout', '30s', 'a100', '--', 'kill', '-TERM', str(pid)], check=False)
        record.update(closed=time.time(), io_after=Path('/proc/pressure/io').read_text(),
                      daemon_closed='SIGTERM by PID')
        write(work / 'ASIDE.json', record)
    # Verify every listed path arrived locally and is absent on A100.
    local_bytes = 0
    missing = []
    for p in relpaths:
        q = destination / p.lstrip('/')
        if q.is_symlink():
            continue
        if not q.is_file():
            missing.append(p)
        else:
            local_bytes += q.stat().st_size
    record.update(local_bytes=local_bytes, missing_locally=missing, verified=not missing)
    write(work / 'ASIDE.json', record)
    if missing:
        raise RuntimeError(f'{len(missing)} moved files missing locally; see {work}')
    # Remove now-empty directories only (no file deletion); restore modes.
    left = json.loads(tx('import os,json\nn=0\n'
        f'for root in {roots!r}:\n'
        ' for d,ds,fs in os.walk(root,topdown=False):\n'
        '  if not os.listdir(d): os.rmdir(d)\n'
        '  else: n+=len(fs)\n'
        f'for d,m in json.load(open({modes!r})):\n'
        ' os.path.isdir(d) and os.chmod(d,m)\n'
        'print(json.dumps(n))\n', '300s'))
    record.update(files_left_under_roots=left)
    write(work / 'ASIDE.json', record)
    usage = tx('import shutil,json;print(json.dumps(shutil.disk_usage("/home/exouser")._asdict()))', '30s')
    log('aside', name=name, files=len(relpaths), local_bytes=local_bytes, destination=str(destination),
        a100_disk=json.loads(usage), receipt=str(work / 'ASIDE.json'))


EXT_TOOLS = ('a100_closedloop_run_1018.py', 'a100_closedloop_ext_queue_1018b.py', 'client_trace.py', 'pilot_readmap.py',
             'paired_sr.py', 'replay.py', 'artifacts.py', 'capture.py', 'inventory.py', 'concurrent_server.py',
             'dp_distance_capture_1018.py', 'dp_distance_replay_1018.py', 'filter_failed_server_cli.py')


def daemon(module_path, tag):
    port, _, _ = free_port()
    d = STUDY / 'cl18/transfer' / (tag + '_' + str(int(time.time())))
    config = (f'use chroot = false\nread only = false\nlist = false\nhosts allow = {ME}\n'
              f'pid file = {d}/daemon.pid\nlog file = {d}/daemon.log\n[m]\npath = {module_path}\nread only = false\nlist = false\n')
    code = ('from pathlib import Path; import subprocess,json\n'
            f'd=Path({str(d)!r});d.mkdir(parents=True,exist_ok=False)\n'
            f'Path({str(module_path)!r}).mkdir(parents=True,exist_ok=True)\n'
            f'c=d/"rsyncd.conf";c.write_text({config!r})\n'
            'f=(d/"stdio.log").open("ab")\n'
            f'p=subprocess.Popen(["rsync","--daemon","--no-detach","--config="+str(c),"--port={port}"],'
            'stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)\n'
            'print(json.dumps({"pid":p.pid}))\n')
    pid = json.loads(tx(code, '30s'))['pid']
    time.sleep(.5)
    return port, pid


def close(pid):
    subprocess.run(['tether', 'exec', '--timeout', '30s', 'a100', '--', 'kill', '-TERM', str(pid)], check=False)


def push_ext(model, names, snapname='execution_tools_ext'):
    import shutil, hashlib
    snap = OUT / model / snapname
    snap.mkdir(exist_ok=True)
    for n in EXT_TOOLS:
        shutil.copyfile(HERE / n, snap / n)
    write(OUT / model / (snapname.upper() + '.json'),
          {q.name: hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(snap.glob('*.py'))})
    port, pid = daemon(REMOTE, 'push_ext_' + model)
    try:
        inc = ['--include=/' + model + '/', '--include=/' + model + '/' + snapname + '/***',
               '--include=/' + model + '/' + snapname.upper() + '.json']
        inc += ['--include=/' + model + '/' + n + '/***' for n in names]
        subprocess.run(['rsync', '-a', '--bwlimit=16384', '--contimeout=10', '--timeout=120',
                        '--exclude=__pycache__', '--exclude=*.pyc', *inc, '--exclude=*',
                        str(OUT) + '/', f'rsync://{PUBLIC}:{port}/m/'], check=True)
    finally:
        close(pid)
    log('pushed_ext', model=model, names=names, port=port, daemon_pid=pid, daemon_closed='SIGTERM by PID')


def launch_ext(model, tag, runs, snapname='execution_tools_ext'):
    c = json.loads((OUT / model / 'pure/campaign_template.json').read_text())
    argv = [c['server_python'], '-u', str(REMOTE / model / snapname / 'a100_closedloop_ext_queue_1018b.py'),
            '--model', model, '--tag', tag, '--runs', *runs]
    name = f'{model}_ext_{tag}.log'
    code = ('from pathlib import Path;import subprocess,json; '
            f'r=Path({str(REMOTE)!r});log=(r/{name!r}).open("xb"); '
            f'p=subprocess.Popen({argv!r},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); '
            'print(json.dumps(dict(pid=p.pid)))')
    launch = Q.control(code)
    log('launched_ext', model=model, tag=tag, runs=runs, launch=launch, log=name)
    return launch


def pull(includes, dest):
    """Read-back of derived A100 files (never per_step bulk) through a temporary daemon."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    port, pid = daemon(REMOTE, 'pull')
    try:
        inc = []
        for i in includes:
            parts = i.strip('/').split('/')
            inc += ['--include=/' + '/'.join(parts[:k]) + '/' for k in range(1, len(parts))]
            inc.append('--include=/' + i.strip('/'))
        subprocess.run(['rsync', '-a', '--bwlimit=16384', '--contimeout=10', '--timeout=120', *inc, '--exclude=*',
                        f'rsync://{PUBLIC}:{port}/m/', str(dest) + '/'], check=True)
    finally:
        close(pid)
    log('pulled', includes=includes, destination=str(dest), port=port, daemon_pid=pid, daemon_closed='SIGTERM by PID')


def alias(name, target):
    """Point the study's own A100 canonical alias /data/<name> at a staged replica tree."""
    out = tx('import os,json\n'
             f'l="/data/{name}";t={str(target)!r}\n'
             'old=os.readlink(l) if os.path.islink(l) else None\n'
             'assert old is None or old.startswith("/home/exouser/xgpu_1017/"), old\n'
             'assert os.path.isdir(t)\n'
             'tmp=l+".new";os.symlink(t,tmp);os.replace(tmp,l)\n'
             'print(json.dumps(dict(link=l,old=old,new=t)))\n', '30s')
    with (HERE / 'evidence/a100_operations_1018b.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=time.time(), op='canonical_alias_repoint', **json.loads(out))) + '\n')
    log('alias', **json.loads(out))


def fw_to_cosmos():
    while True:
        s = Q.state('fastwam')
        if s and s['status'] in ('COMPLETE', 'FAILED'):
            break
        time.sleep(30)
    if s['status'] != 'COMPLETE':
        raise RuntimeError('fastwam base queue failed; handoff not started')
    Q.collect('fastwam')
    log('collected', model='fastwam')
    fw_only = sorted(bulk('fastwam') - bulk('cosmos') - bulk('dp'))
    aside('fastwam_only', REMOTE / 'models/dp/rootfs', fw_only, [REMOTE / 'models/dp/rootfs'])
    stage('cosmos')
    alias('cosmos', REMOTE / 'models/cosmos/rootfs/data/cosmos')
    launch_wait('cosmos')


def push_tree(local, remote, tag):
    """Public rsync push of one local replica tree into an A100 study path."""
    port, pid = daemon(remote, 'push_' + tag)
    try:
        subprocess.run(['rsync', '-a', '--bwlimit=16384', '--contimeout=10', '--timeout=300',
                        str(local).rstrip('/') + '/', f'rsync://{PUBLIC}:{port}/m/'], check=True)
    finally:
        close(pid)
    log('pushed_tree', local=str(local), remote=str(remote), port=port, daemon_pid=pid, daemon_closed='SIGTERM by PID')


def cosmos_to_pigr():
    while True:
        s = Q.state('cosmos')
        if s and (s['status'] == 'COMPLETE' or (s['status'] == 'FAILED' and len(s['completed']) == 4)):
            break
        if s and s['status'] == 'FAILED':
            raise RuntimeError('cosmos base queue failed before 4 runs; no handoff')
        time.sleep(30)
    if s['status'] != 'COMPLETE':
        # Runs complete; only the pairing step failed: redo it in the openpi island.
        tx('import subprocess\nsubprocess.run(["/home/exouser/xgpu_1017/env/bin/python",'
           '"/home/exouser/xgpu_1017/cl18/cosmos/execution_tools/a100_closedloop_queue_1018.py","--model","cosmos","--analyse-only"],check=True)\n', '300s')
    Q.collect('cosmos')
    log('collected', model='cosmos')
    co = sorted(bulk('cosmos'))
    aside('cosmos_all', REMOTE / 'models/cosmos/rootfs', co, [REMOTE / 'models/cosmos/rootfs'])
    stage('pi05')
    push_tree('/data/ckpt/pi05_metaworld_rlinf', STUDY / 'rootfs/data/ckpt/pi05_metaworld_rlinf', 'pi05_ckpt')
    stage('groot')
    for m in ('groot', 'pi05'):
        push_ext(m, ['pure_x1', 'cache_x1', 'pure_x2', 'cache_x2'])
    launch_ext('groot', 'x1', ['pure_x1:1', 'cache_x1:1', 'pure_x1:2', 'cache_x1:2'])


def bulk(model):
    return {l for l in (OUT / model / 'BULK.files').read_text().splitlines() if l}


def rename_rootfs(src, dst):
    s, d = REMOTE / 'models' / src / 'rootfs', REMOTE / 'models' / dst / 'rootfs'
    tx('import os\nfrom pathlib import Path\n'
       f's,d=Path({str(s)!r}),Path({str(d)!r})\n'
       'assert s.is_dir() and not s.is_symlink() and not d.exists()\n'
       'd.parent.mkdir(parents=True,exist_ok=True);os.rename(s,d);print("RENAMED",s,"->",d)\n', '30s')
    log('rootfs_renamed', source=str(s), target=str(d))


def listing(tops):
    return json.loads(tx('import os,json\n'
                         f'b={str(STUDY)!r};out=[]\n'
                         f'for top in {list(tops)!r}:\n'
                         ' for d,ds,fs in os.walk(os.path.join(b,top)):\n'
                         '  out += [os.path.relpath(os.path.join(d,f),b) for f in fs]\n'
                         'print(json.dumps(out))\n', '120s'))


def pre_dp_aside():
    # Unique A100 raw open-loop results: durable local copy, removed from A100.
    tops = ['results']
    aside('a100_openloop_results', STUDY, listing(tops), [STUDY / t for t in tops], root=DURABLE)
    # Finished pi05/GR00T replicas (MetaWorld sim env, their cache inputs).
    tops = ['cl18/simenv', 'cl18/deps/data/mw_finetune/eval/ff1015_groot_mt_10',
            'cl18/deps/data/mw_finetune/eval/pimk1_ful']
    aside('finished_pi05_groot_replicas', STUDY, listing(tops), [STUDY / t for t in tops])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=['pre_dp_aside', 'dp', 'fastwam', 'fastwam_handoff', 'cosmos', 'ext', 'pull', 'fw_to_cosmos', 'cosmos_to_pigr'])
    p.add_argument('--include', nargs='*'); p.add_argument('--dest')
    p.add_argument('--model'); p.add_argument('--tag'); p.add_argument('--snap', default='execution_tools_ext'); p.add_argument('--names', nargs='*'); p.add_argument('--runs', nargs='*')
    p.add_argument('--resume-startup', action='store_true'); p.add_argument('--wait-lock', action='store_true')
    a = p.parse_args()
    if a.step == 'pre_dp_aside':
        pre_dp_aside()
    elif a.step == 'cosmos_to_pigr':
        cosmos_to_pigr()
    elif a.step == 'fw_to_cosmos':
        fw_to_cosmos()
    elif a.step == 'pull':
        pull(a.include, a.dest)
    elif a.step == 'ext':
        push_ext(a.model, a.names, a.snap)
        launch_ext(a.model, a.tag, a.runs, a.snap)
    elif a.step in ('dp', 'fastwam', 'cosmos'):
        # DP rootfs already transferred; the stage rerun is idempotent (rsync skips).
        if a.wait_lock:
            launch_wait(a.step)
            return
        elif a.resume_startup:
            relaunch(a.step)
        elif Q.state(a.step) is None:
            start(a.step)
        finish(a.step)
    elif a.step == 'fastwam_handoff':
        # DP and Fast-WAM share models/dp/rootfs. Only Fast-WAM-only replicas
        # (weights, T5, VAE, vendor) leave; the fastwam island + LIBERO files stay
        # for DP filler runs. Cosmos stages into its own models/cosmos/rootfs.
        assert Q.state('fastwam')['status'] == 'COMPLETE'
        fw_only = sorted(bulk('fastwam') - bulk('cosmos') - bulk('dp'))
        aside('fastwam_only', REMOTE / 'models/dp/rootfs', fw_only, [REMOTE / 'models/dp/rootfs'])

if __name__ == '__main__':
    main()
