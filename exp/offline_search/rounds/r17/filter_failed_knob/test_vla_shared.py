"""A busy legacy fleet must not block an independently admitted formal-A fleet."""
import fcntl
import inspect
import json
from pathlib import Path
import types

import pytest

from . import vla_shared as s, vla_workers as v


def test_busy_host_lock_scoped_remote_cleanup_and_capacity(tmp_path, monkeypatch):
    from exp.offline_search.closed_loop.ops.h100 import control as c, node
    gate = tmp_path/'locks/h100_chain.lock';gate.parent.mkdir()
    monkeypatch.setattr(c, 'CHAIN_LOCK', gate)
    monkeypatch.setattr(c, 'legacy_chains', lambda **kw: [])
    root = tmp_path/'fa_pi_f_u_o';root.mkdir()
    other = tmp_path/'fa_gr_f_u_l';other.mkdir()
    with c.host_lock('timan107').open('a') as busy:
        fcntl.flock(busy, fcntl.LOCK_EX|fcntl.LOCK_NB)
        with s.fleet_lock(root), s.fleet_lock(other):
            with pytest.raises(BlockingIOError):
                with s.fleet_lock(root): pass
            with gate.with_name(gate.name+'.gate').open('a') as maintenance:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(maintenance, fcntl.LOCK_EX|fcntl.LOCK_NB)

    island = tmp_path/'island';(island/'os_cl').mkdir(parents=True)
    native = c.HERE/'run_arm.sh'
    (island/'os_cl/run_arm.sh').write_bytes(native.read_bytes())
    n = types.ModuleType('cpu_native_node')
    n.__dict__.update(node.__dict__);n.ISLAND = island
    saved=[];n.atomic=lambda path,value:saved.append(value)
    lock=s.worker_lock(root,island);s.configure_node(n,lock,root.name)
    spec=dict(role='driver',cmd=['bash',str(island/'os_cl/run_arm.sh'),'libero_10','arm',
                                'h100:23230,h100:23231,h100:23232','13,13,13','/out'],
              env={'OSCL_MANIFEST':'/sealed/manifest.json','OSCL_ROLLOUT_SEED':'7'})
    n.atomic('/cpu/job.json',spec);actual=saved[0]
    assert spec['env']=={'OSCL_MANIFEST':'/sealed/manifest.json','OSCL_ROLLOUT_SEED':'7'}
    assert actual['cmd'][3:]==spec['cmd'][1:]
    script=actual['cmd'][2]
    assert script.replace('exec 9>"${FFK_WORKER_LOCK:?}"',
                          'exec 9>"$R/os_cl/locks/fleet.lock"')==native.read_text()
    assert actual['env']['FFK_WORKER_LOCK']==lock
    assert actual['env']['FFK_WORKER_FLEET']==root.name
    assert s.worker_lock(other,island)!=lock
    assert lock in n.wait_fleet.__code__.co_consts
    for i,marker in enumerate((root.name,other.name)):
        p=tmp_path/'proc'/str(i);p.mkdir(parents=True)
        (p/'cmdline').write_bytes(b'python\0-m\0examples.libero.worker_entry\0')
        (p/'environ').write_bytes(('FFK_WORKER_FLEET='+marker).encode()+b'\0')
    assert n.island_workers(tmp_path/'proc')==1
    assert (island/'os_cl/run_arm.sh').read_bytes()==native.read_bytes()
    # Native WorkerAgent retains the ownership marker through conda run. It
    # still uses the identical GPU, seed, replan and init-pool arguments.
    from openpi.conductor import agent
    spawned=[]
    monkeypatch.setenv('FFK_WORKER_FLEET', root.name)
    monkeypatch.setattr(agent.subprocess,'Popen',lambda argv,**kw:spawned.append((argv,kw)))
    agent._default_spawn(agent.WorkerSpec('w0','h100:23230','0',
        conda_env='/frozen/sim',seed=7,replan_steps=10,init_states_dir='/sealed/apool'),
        'localhost',23159)
    assert spawned[0][1]['env']['FFK_WORKER_FLEET']==root.name
    assert spawned[0][1]['env']['CUDA_VISIBLE_DEVICES']=='0'
    assert spawned[0][0][-2:]==['--seed','7']

    job=dict(pipeline='R15',workers={'timan108':39})
    admission=dict(capacity=50,cards={str(i):dict(total=8192,used=3500,pending=0) for i in range(8)})
    monkeypatch.setattr(v,'probe',lambda h,p:dict(usable=h=='timan107',admission=admission,host=h))
    assert v.choose(job,require_capacity=True)[0]=='timan107'
    admission['capacity']=11
    assert v.choose(job)[0]=='timan107'  # prepare never dispatches renderers
    assert v.choose(job,require_capacity=True)[0]=='timan107'  # runtime admits actual workers
    assert admission['capacity']==11  # never falsify the measured memory envelope
    # R16 uses exactly the same scoped native remote lock as its preflight.
    boot=v.r16_agent_bootstrap('timan107','/native/island',lock)
    assert repr(lock) in boot
    compile(boot,'<cpu-r16-agent>','exec')
