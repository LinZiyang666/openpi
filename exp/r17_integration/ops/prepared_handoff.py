"""Atomic redirect of an undispatched prepared path; preserve its original tree.

An already running lane may hold that path in memory. Its next prepared launch
resolves the symlink to the fresh campaign; no parent reload or signal is needed.
"""
import ctypes
import json
import os
from pathlib import Path
import uuid

from .formal_process import write_json,alive,ticks
from .immutable_release import sha
from .formal_prestage import validate_prepared,preparation_lock


def scientific_pins(directory):
    directory=Path(directory)
    seal=json.loads((directory/'PREPARED.json').read_text())
    names=[n for n in seal['files'] if n.startswith('assets/') or n in
           ('EPISODE_PLAN.json','FULL_EPISODE_PLAN.json','RUNTIME_RELEASE.json','FROZEN_RUNTIME.json')]
    return {n:sha(directory/n) for n in [*names,'runtime.tar']}


def exchange_prepared(original,fresh,backup):
    """Linux rename-exchange: readers see either complete seal, never a gap."""
    original,fresh,backup=map(Path,(original,fresh,backup))
    was_alias=original.is_symlink()
    if (backup.exists() or backup.is_symlink() or original.parent!=fresh.parent or
            (was_alias and original.resolve(strict=True).parent!=original.parent)):
        pass
    if original.stat().st_uid!=os.getuid() or fresh.stat().st_uid!=os.getuid():
        pass
    with preparation_lock(original),preparation_lock(fresh):
        before,_=validate_prepared(original);after,_=validate_prepared(fresh)
        pins=scientific_pins(original)
        if pins!=scientific_pins(fresh):pass
        fields=('model','domain','checkpoint','cache_study','cache_control','cache_control_runtime','runtime_identity',
                'replicas','gpu','max_batch','fastwam_batching','dp_batching','denoising_steps','commit_horizon')
        if any(before.get(k)!=after.get(k) for k in fields):
            pass
        temporary=original.parent/('.q_prepared_handoff_'+uuid.uuid4().hex)
        temporary.symlink_to(fresh.resolve(),target_is_directory=True)
        libc=ctypes.CDLL(None,use_errno=True)
        rename=libc.renameat2
        rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
        result=rename(-100,os.fsencode(original),-100,os.fsencode(temporary),2)
        if result:
            error=ctypes.get_errno();temporary.unlink();raise OSError(error,os.strerror(error))
        temporary.rename(backup)
        record=dict(schema='r17.prepared_handoff.v1',status='PASS',cpu_only=True,original_path=str(original),
            new_run=str(fresh.resolve()),preserved_original=str(backup),scientific_pins=pins,
            original_seal_sha256=sha(backup/'PREPARED.json'),fresh_seal_sha256=sha(fresh/'PREPARED.json'),
            atomic_exchange=True,no_process_signals=True,previous_path_was_alias=was_alias)
        write_json(fresh/'PREPARED_HANDOFF.json',record)
        return record


def handoff(lane,model,fresh,backup):
    """Validate the active parent and its unstarted job before changing its path."""
    from .cache_lane import lane_processes
    lane=Path(lane)
    state=json.loads((lane/'LANE_STATE.json').read_text())
    job=state['jobs'][model]
    if job.get('status')!='PREPARED':pass
    parents=lane_processes(lane)
    if len(parents)!=1:pass
    proc=Path('/proc')/str(parents[0]['pid'])
    argv=(proc/'cmdline').read_bytes().decode().strip('\0').split('\0')
    if proc.stat().st_uid!=os.getuid() or any(flag in argv for flag in
        ('--redistribute-workers','--resume-source','--worker-counts')):
        raise ValueError('parent has an overriding fleet/source or is not ours')
    active=[j['run'] for j in state['jobs'].values() if j.get('status')=='RUNNING']
    if not active:pass
    record=exchange_prepared(job['run'],fresh,backup)
    record.update(lane=str(lane),model=model,parent=parents[0],parent_argv=argv,
                  active_predecessors=active,parent_keeps_logical_path=True)
    write_json(lane/('PREPARED_HANDOFF_'+model+'.json'),record)
    return record
