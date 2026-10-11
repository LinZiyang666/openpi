"""Bounded render-only native LIBERO probe, in an isolated owned directory.

No policy endpoint, campaign control, GPU reset or unrelated process signals.
Uses the stock environment/state/camera path; reports physical GPU memory and
node memory alongside per-process RSS/PSS. Run under the native worker env.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def memory(pid):
    out = {}
    for name in ('status', 'smaps_rollup'):
        try:
            for line in Path(f'/proc/{pid}/{name}').read_text().splitlines():
                key, _, value = line.partition(':')
                if key in ('VmRSS', 'Pss', 'Rss', 'Private_Dirty'):
                    out[key] = int(value.split()[0])*1024
        except OSError:
            pass
    return out


def child(args):
    # Must select before importing robosuite/mujoco/libero (EGL display globals).
    os.environ.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
                      MUJOCO_EGL_DEVICE_ID=str(args.device), CUDA_VISIBLE_DEVICES='')
    import numpy as np
    from libero.libero import benchmark
    from examples.libero import main as native
    root = Path(args.output)
    rows = []
    # Include spatial and the heavy long-horizon task implicated in M's burst.
    for suite_name, task_id in [('libero_spatial', 0), ('libero_10', 9)]:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        states = suite.get_task_init_states(task_id)
        for idx in (0, 1):
            # Native episode setup constructs a fresh environment per episode.
            env, prompt = native._get_libero_env(suite.get_task(task_id), 256, 0)
            try:
                env.reset()
                obs = env.set_init_state(states[idx])
                for _ in range(10):
                    obs, _, _, _ = env.step([0., 0., 0., 0., 0., 0., -1.])
                arrays = {k: np.asarray(obs[k]).copy() for k in
                          ('agentview_image', 'robot0_eye_in_hand_image',
                           'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos')}
                file = root/f'{args.index}_{suite_name}_{task_id}_{idx}.npz'
                np.savez(file, **arrays)
                row = dict(suite=suite_name, task=task_id, init=idx, prompt=prompt,
                           hashes={k: hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()},
                           memory=memory(os.getpid()))
                rows.append(row)
                write(root/f'worker_{args.index}.json', dict(pid=os.getpid(), device=args.device, rows=rows))
                for _ in range(args.controls):
                    env.step([0., 0., 0., 0., 0., 0., -1.])
                if suite_name == 'libero_10' and idx == 1 and args.hold_context:
                    write(root/f'worker_{args.index}.json', dict(pid=os.getpid(), device=args.device,
                        rows=rows, memory=memory(os.getpid()), holding_context=True))
                    while time.time() < args.until:
                        # Hold a native rendered state without stepping past
                        # robosuite's episode horizon (a probe, not an eval).
                        env.env.sim.render(height=256, width=256, camera_name='agentview')
                        env.env.sim.render(height=256, width=256, camera_name='robot0_eye_in_hand')
                        time.sleep(.01)
            finally:
                env.close()
    write(root/f'worker_{args.index}.json', dict(pid=os.getpid(), device=args.device, rows=rows,
                                               memory=memory(os.getpid()), completed=True))
    # Hold all owned contexts alive for the plateau; parent supplies a bound.
    while time.time() < args.until:
        time.sleep(.5)


def sample(devices, children):
    fields = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        k, _, v = line.partition(':')
        if k in ('MemTotal', 'MemAvailable', 'AnonPages', 'Slab', 'VmallocUsed'):
            fields[k] = int(v.split()[0])*1024
    query = subprocess.run(['nvidia-smi', '-i', ','.join(map(str,devices)), '-q', '-x'],
        capture_output=True, text=True, timeout=10)
    if query.returncode:
        raise RuntimeError('nvidia-smi failed: '+query.stderr)
    # Read only exact live-fleet receipt PIDs, for background recycle context.
    background=[]
    for receipt in Path('/scratch/zixuans8/mw_r17/formal/runs').glob(
            'hs2_cosmos_h100_k2incl_c16_t107_*/receipt.json'):
        for entry in json.loads(receipt.read_text()).get('pids', []):
            pid=entry['pid']
            try:
                fields_stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
                if fields_stat[19] != str(entry['start_ticks']) or fields_stat[0] == 'Z': continue
                background.append(dict(pid=pid,generation=entry['generation'],**memory(pid)))
            except OSError: pass
    owned={c.pid for c in children}
    cards=[]
    for card in ET.fromstring(query.stdout).findall('gpu'):
        cards.append(dict(index=int(card.findtext('minor_number')),
            total_mib=int(card.findtext('fb_memory_usage/total').split()[0]),
            used_mib=int(card.findtext('fb_memory_usage/used').split()[0]),
            processes=[dict(pid=int(proc.findtext('pid')),used_mib=int(proc.findtext('used_memory').split()[0]))
                       for proc in card.findall('processes/process_info')
                       if int(proc.findtext('pid')) in owned]))
    return dict(time=time.time(), meminfo=fields, background=background, gpu=cards,
        workers=[dict(pid=c.pid, alive=c.poll() is None, **memory(c.pid)) for c in children])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--devices', default='1,2,3,4,5,6,7')
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--seconds', type=int, default=240)
    p.add_argument('--controls', type=int, default=150)
    p.add_argument('--hold-context', action='store_true')
    p.add_argument('--child', action='store_true')
    p.add_argument('--device', type=int)
    p.add_argument('--index', type=int)
    p.add_argument('--until', type=float)
    args=p.parse_args()
    if args.child:
        child(args); return
    if not 30 <= args.seconds <= 540 or not 1 <= args.workers <= 16:
        raise ValueError('bounded probe: 30..540 seconds, <=16 workers')
    devices=[int(x) for x in args.devices.split(',')]
    if 0 in devices:
        raise ValueError('this probe excludes the protected live fleet GPU0')
    root=Path(args.output); root.mkdir(exist_ok=False, parents=True)
    children=[]; logs=[]; trace=[]
    started=time.time(); until=started+args.seconds
    write(root/'CONFIG.json', vars(args))
    try:
        for _ in range(4):
            trace.append(sample(devices, [])); write(root/'TRACE.json',trace); time.sleep(5)
        for i in range(args.workers):
            log=(root/f'worker_{i}.log').open('wb'); logs.append(log)
            env=dict(os.environ, TMPDIR=str(root), XDG_CACHE_HOME=str(root/f'cache_{i}'),
                     OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            command=[sys.executable, __file__, '--child', '--output', str(root),
                     '--index', str(i), '--device', str(devices[i%len(devices)]),
                     '--controls', str(args.controls), '--until', str(until)]
            if args.hold_context: command += ['--hold-context']
            children.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
            write(root/'OWNED_PIDS.json', [c.pid for c in children])
        while time.time()<until and any(c.poll() is None for c in children):
            trace.append(sample(devices, children)); write(root/'TRACE.json',trace)
            # Stop our probe if it consumes the node safety reserve.
            if trace[-1]['meminfo']['MemAvailable'] < 30*1024**3:
                raise RuntimeError('node available RAM fell below30GiB; stopping own probe')
            time.sleep(5)
        for c in children: c.wait(timeout=10)
    finally:
        # Exactly the child objects launched here; no name-based process kills.
        for c in children:
            if c.poll() is None: c.terminate()
        for c in children:
            try: c.wait(timeout=10)
            except subprocess.TimeoutExpired:
                c.kill(); c.wait(timeout=5)
        for log in logs: log.close()
        write(root/'EXITS.json', [dict(pid=c.pid,code=c.poll()) for c in children])
        for _ in range(4):
            trace.append(sample(devices, [])); write(root/'TRACE.json',trace); time.sleep(5)
    print(json.dumps(dict(output=str(root),elapsed=time.time()-started, exits=[c.poll() for c in children])))


if __name__=='__main__': main()
