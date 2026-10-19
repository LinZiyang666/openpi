"""Serial A100 pipeline after the DP/Fast-WAM base arms (1018b).

Cosmos base -> Cosmos x1 -> [Cosmos aside; stage pi05 + GR00T] -> GR00T x1 -> pi05 x1
-> [pi05/GR00T aside; re-stage Fast-WAM-only] -> Fast-WAM x1 -> [Fast-WAM aside; re-stage
pi05 + GR00T] -> GR00T x2 -> pi05 x2.

The GPU is never left idle while a transfer runs: preregistered DP same-card
replicate runs (DP files stay on the A100 in the shared rootfs) are launched one
run at a time whenever no queue holds or waits for the exclusive GPU lock.
Every main block is launched as a blocking-lock waiter while its predecessor runs.
"""
import json
import threading
import time
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a100_closedloop_continue_1018b as C  # noqa: E402

# r2 (4 runs) was queued directly at 19:13 after the Cosmos startup failure.
FILLERS = [('pure_r3', 1), ('pure_r3', 2)]
filled = []


def qstate(name):
    return json.loads(C.tx('from pathlib import Path\n'
                           f'p=Path("/home/exouser/xgpu_1017/cl18/queue/{name}.json")\n'
                           'print(p.read_text() if p.exists() else "null")\n', '45s'))


def gpu_claimed():
    out = json.loads(C.tx('import json,glob,os\n'
                          'q="/home/exouser/xgpu_1017/cl18/queue/"\n'
                          'run=[p for p in glob.glob(q+"*.json") if json.load(open(p)).get("status")=="RUNNING"]\n'
                          'wait=glob.glob(q+"*.waiting")\n'
                          'print(json.dumps(dict(running=run,waiting=wait)))\n', '45s'))
    return bool(out['running'] or out['waiting'])


def wait_for(name, statuses=('COMPLETE',), allow_failed_after=None, feed=False, since=None):
    while True:
        s = qstate(name)
        if s and since is not None and s.get('started', 0) < since:
            s = None  # an earlier (startup-failed) receipt still awaiting the native resume rename
        if s and s['status'] in statuses:
            return s
        if s and s['status'] == 'FAILED':
            if allow_failed_after is not None and len(s['completed']) >= allow_failed_after:
                return s
            raise RuntimeError(name + ' failed: ' + str(s.get('error')))
        if feed:
            feed_gpu()
        time.sleep(30)


def feed_gpu():
    """Start exactly one filler run if nothing holds or waits for the GPU."""
    if gpu_claimed():
        return
    remaining = [f for f in FILLERS if f not in filled]
    if not remaining:
        C.log('filler_exhausted')
        return
    name, cold = remaining[0]
    filled.append((name, cold))
    C.launch_ext('dp', f'{name}c{cold}', [f'{name}:{cold}'], 'execution_tools_cb')


def transfer_in_background(fn):
    box = {}

    def run():
        try:
            fn()
            box['ok'] = True
        except BaseException as e:  # surfaced by the main loop
            box['error'] = repr(e)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, box


def keep_fed_until(thread, box):
    while thread.is_alive():
        feed_gpu()
        time.sleep(30)
    if 'error' in box:
        raise RuntimeError('transfer failed: ' + box['error'])


def sync_analysis():
    C.pull(['analysis/*_PAIRED.json', 'analysis/dp_DISTANCE.json'], C.OUT / 'collected_raw')
    for p in sorted((C.OUT / 'collected_raw/analysis').glob('*.json')):
        if p.name.endswith(('_EXT_cache_PAIRED.json', '_EXT_pure_PAIRED.json')):
            continue  # base-arm re-pairings already contained in <model>_PAIRED.json
        (C.OUT / 'collected' / p.name).write_bytes(p.read_bytes())
    C.log('synced_analysis')


def archive_runs(prefixes):
    tops = json.loads(C.tx('import os,json\nr="/home/exouser/xgpu_1017/cl18/runs"\nout=[]\n'
                           f'for d in sorted(os.listdir(r)):\n if d.startswith({tuple(prefixes)!r}):\n'
                           '  st=lambda x:json.load(open(os.path.join(r,x,"RUN.json"))).get("status") if os.path.exists(os.path.join(r,x,"RUN.json")) else None\n'
                           '  if st(d) not in ("COMPLETE","FAILED"): continue\n'
                           '  if "_cold" in d and "startup_failed" not in d:\n'
                           '   sib=d[:-1]+("2" if d.endswith("1") else "1")\n'
                           '   if st(d)=="COMPLETE" and st(sib)!="COMPLETE": continue  # keep until its pair is analysed\n'
                           '  out.append("cl18/runs/"+d)\n'
                           'print(json.dumps(out))\n', '60s'))
    if tops:
        C.aside('runs_' + '_'.join(p.strip('_') for p in prefixes), C.STUDY, C.listing(tops),
                [C.STUDY / t for t in tops], root=C.DURABLE)


RUNS = {'x1': ['pure_x1:1', 'cache_x1:1', 'pure_x1:2', 'cache_x1:2'],
        'x2': ['pure_x2:1', 'cache_x2:1', 'pure_x2:2', 'cache_x2:2']}


def chain(blocks):
    """Launch each block as the single waiter behind its running predecessor."""
    previous = None
    for model, tag in blocks:
        if previous:
            wait_for(previous, statuses=('RUNNING', 'COMPLETE'))
        C.launch_ext(model, tag, RUNS[tag])
        previous = f'{model}_ext_{tag}'
    for model, tag in blocks:
        wait_for(f'{model}_ext_{tag}', allow_failed_after=4)
        sync_analysis()


def main():
    # 1. Cosmos base is already queued behind the DP replicate; Cosmos x1 right behind it.
    C.push_ext('cosmos', ['pure_x1', 'cache_x1'])
    since = float(sys.argv[1]) if len(sys.argv) > 1 else None
    wait_for('cosmos', statuses=('RUNNING', 'COMPLETE'), since=since)
    C.launch_ext('cosmos', 'x1', ['pure_x1:1', 'cache_x1:1', 'pure_x1:2', 'cache_x1:2'])
    s = wait_for('cosmos', allow_failed_after=4, since=since)
    if s['status'] != 'COMPLETE':
        C.tx('import subprocess\nsubprocess.run(["/home/exouser/xgpu_1017/env/bin/python",'
             '"/home/exouser/xgpu_1017/cl18/cosmos/execution_tools/a100_closedloop_queue_1018.py",'
             '"--model","cosmos","--analyse-only"],check=True)\n', '300s')
    C.Q.collect('cosmos')
    C.log('collected', model='cosmos')
    sync_analysis()
    (C.OUT / 'COSMOS_BASE_DONE').write_text(str(time.time()) + '\n')
    # 2. Cosmos x1; then Cosmos replicas leave and pi05 + GR00T are staged.
    wait_for('cosmos_ext_x1', allow_failed_after=4)
    sync_analysis()

    def stage_pigr():
        C.stage('pi05')
        C.push_tree('/data/ckpt/pi05_metaworld_rlinf', C.STUDY / 'rootfs/data/ckpt/pi05_metaworld_rlinf', 'pi05_ckpt')
        C.stage('groot')
        for m in ('groot', 'pi05'):
            C.push_ext(m, ['pure_x1', 'cache_x1', 'pure_x2', 'cache_x2'])

    def gap_a():
        archive_runs(['cosmos_', 'fastwam_', 'dp_'])
        C.aside('cosmos_all', C.REMOTE / 'models/cosmos/rootfs', sorted(C.bulk('cosmos')), [C.REMOTE / 'models/cosmos/rootfs'])
        stage_pigr()
    t, box = transfer_in_background(gap_a)
    keep_fed_until(t, box)
    chain([('groot', 'x1'), ('pi05', 'x1')])

    # 3. pi05/GR00T replicas leave; Fast-WAM-only replicas return (shared DP rootfs).
    def gap_b():
        archive_runs(['groot_', 'pi05_', 'dp_'])
        C.aside('groot_all', C.REMOTE / 'models/groot/rootfs', sorted(C.bulk('groot')), [C.REMOTE / 'models/groot/rootfs'])
        tops = ['rootfs/data/ckpt/pi05_metaworld_rlinf', 'cl18/simenv', 'cl18/deps/data/mw_finetune/eval/ff1015_groot_mt_10',
                'cl18/deps/data/mw_finetune/eval/pimk1_ful']
        C.aside('pi05_ckpt_simenv_deps', C.STUDY, C.listing(tops), [C.STUDY / t for t in tops])
        C.stage('fastwam')
        C.push_ext('fastwam', ['pure_x1', 'cache_x1'])
    t, box = transfer_in_background(gap_b)
    keep_fed_until(t, box)
    chain([('fastwam', 'x1')])

    # 4. Fast-WAM-only replicas leave; pi05 + GR00T return for the long x2 blocks.
    def gap_c():
        archive_runs(['fastwam_', 'dp_'])
        fw_only = sorted(C.bulk('fastwam') - C.bulk('cosmos') - C.bulk('dp'))
        C.aside('fastwam_only_2', C.REMOTE / 'models/dp/rootfs', fw_only, [C.REMOTE / 'models/dp/rootfs'])
        stage_pigr()
    t, box = transfer_in_background(gap_c)
    keep_fed_until(t, box)
    chain([('groot', 'x2'), ('pi05', 'x2')])
    C.log('pipeline_complete')


if __name__ == '__main__':
    main()
