"""Preserve a closed startup failure's evidence and re-stage its prepared run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil

from .audit_counts import read
from .build import write


def prestart_failure(root, verified_dead=False):
    receipt = read(root/'RECEIPT.json')
    if receipt.get('status') != 'FAIL' or (receipt.get('all_owned_closed') is not True and not verified_dead):
        raise ValueError('reset requires a closed failed startup')
    owned = receipt.get('owned_local', [])+receipt.get('owned_remote', [])
    if owned:
        # Coordinator 2026-10-15 explicitly authorized resetting the closed
        # model-load failure. No driver may have dispatched any test episode.
        from exp.r17_integration.ops.formal_process import alive
        if any(p.get('label') == 'driver' for p in owned) or (root/'driver').exists():
            raise ValueError('driver execution present; use an explicit continuation')
        cleanup = receipt.get('cleanup', [])
        if verified_dead:
            # A fleet that crashed before writing its receipt is recorded as not
            # closed. Re-check every recorded local pid instead of the proof.
            if any(p.get('node') not in (None, 'local') for p in receipt.get('owned_remote', [])):
                raise ValueError('--verified-dead covers local processes only')
            for process in owned:
                for rec in [process, *[p for p in cleanup if p.get('label') == process.get('label')]]:
                    for item in (rec, rec.get('child'), rec.get('supervisor')):
                        if not (isinstance(item, dict) and item.get('pid')):
                            continue
                        cmdline = Path('/proc/%d/cmdline' % item['pid'])
                        if alive(item) if 'start_ticks' in item else (cmdline.exists() and root.name in cmdline.read_bytes().decode(errors='ignore')):
                            raise ValueError('owned process still alive: '+str(item.get('pid')))
            owned_checked = True
        else:
            owned_checked = False
        for process in ([] if owned_checked else owned):
            closure = [p for p in cleanup if p.get('label') == process.get('label') and p.get('closure_confirmed')]
            if not closure or (process.get('pid') and alive(process)):
                raise ValueError('owned process closure is not proved')
            for proof in closure:
                if (proof.get('alive') or proof.get('child_alive') or proof.get('supervisor_alive')
                        or proof.get('owned_worker_survivors')):
                    raise ValueError('owned worker survivors present')
                for key in ('child', 'supervisor'):
                    if proof.get(key) and alive(proof[key]):
                        raise ValueError('owned child still alive')
    elif receipt.get('gpu_bound') is not False and receipt.get('cleanup'):
        # An empty cleanup with no owned process means nothing was spawned.
        raise ValueError('missing pre-start failure proof')
    import json
    phases = [json.loads(line) for line in (root/'PHASE_TIMINGS.jsonl').read_text().splitlines() if line]
    validation = [row for row in phases if row['phase'] == 'prepared_validation']
    spawned = any(row['phase'] == 'owned_spawn' for row in phases)
    if not owned and (spawned or (receipt.get('cleanup') and (not validation or validation[-1]['status'] != 'FAIL'))):
        raise ValueError('missing proof that failure preceded run_campaign; reset refused')
    if (not owned and any((root/name).exists() for name in ('servers', 'driver', 'GPU_RELEASED.json', 'SUMMARY.json'))) or (root/'SUMMARY.json').exists():
        raise ValueError('execution artifacts present; reset refused')
    claim = read(root/'LAUNCH.json')
    try:
        os.kill(claim['pid'], 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError('launch process still exists; reset refused')
    return receipt


def reset(root, verified_dead=False):
    from exp.r17_integration.ops.formal_prestage import preparation_lock
    from .stage import _stage
    root = Path(root).resolve(strict=True)
    with preparation_lock(root):
        receipt = prestart_failure(root, verified_dead)
        suffix = '.failed_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        restored = []
        snapshot = root/'.startup_inputs'
        if (snapshot/'STARTUP_INPUTS.json').exists():
            from exp.r17_integration.ops.immutable_release import sha
            for name, pin in read(snapshot/'STARTUP_INPUTS.json')['files'].items():
                target = root/name
                if target.exists() and sha(target) == pin:
                    continue
                if target.exists():
                    target.rename(target.with_name(target.name+suffix))
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(snapshot/name, target)
                restored.append(name)
        external = []
        if receipt.get('owned_local') or receipt.get('owned_remote'):
            # Preserve local fleet receipts too: formal_process.spawn reuses a
            # receipt at the same path, so old closed receipts must be renamed.
            for node, item in read(root/'WORKER_DEPLOYMENT.json').items():
                if node != 'local':
                    # Same renames on the worker node (closed fleet, e.g. refused by
                    # the frozen worker's EGL admission), plus its sync-back sidecars.
                    from exp.r17_integration.ops import formal_eval as launch
                    script = ('for p in "$1"/*.process.json "$1"/*.lease "$1"/*.log "$2" "$2.tar" "$2.transfer.json"; '
                              'do if [ -e "$p" ]; then mv -n "$p" "$p$3" && echo "$p"; fi; done')
                    renamed = launch.remote(node, ['bash', '-c', script, 'reset', item['stage'], item['worker_run'], suffix])
                    external += [dict(node=node, original=p, saved=p+suffix) for p in renamed.split()]
                    continue
                for path in [*Path(item['stage']).glob('*.process.json'), *Path(item['stage']).glob('*.lease'),
                             *Path(item['stage']).glob('*.log'), Path(item['worker_run'])]:
                    if path.exists():
                        target = path.with_name(path.name+suffix)
                        path.rename(target)
                        external.append(dict(original=str(path), saved=str(target)))
        # A remote server node (H100) keeps its process receipts in the
        # server stage; formal_process.spawn refuses an existing receipt.
        campaign = read(root/'campaign.json')
        if campaign.get('server_node') not in (None, 'local'):
            from exp.r17_integration.ops import formal_eval as launch
            stage = '/tmp/r17_formal/'+campaign['run_name']+'/server'
            script = ('for p in "$1"/*.process.json "$1"/*.lease "$1"/*.log; '
                      'do if [ -e "$p" ]; then mv -n "$p" "$p$2" && echo "$p"; fi; done')
            renamed = launch.remote(campaign['server_node'], ['bash', '-c', script, 'reset', stage, suffix])
            # The server node also mirrors the run directory (MPS control, servers).
            script = ('for p in "$1"/MPS_READY.json "$1"/mps "$1"/mps_control "$1"/servers; '
                      'do if [ -e "$p" ]; then mv -n "$p" "$p$2" && echo "$p"; fi; done')
            renamed += launch.remote(campaign['server_node'], ['bash', '-c', script, 'reset', str(root), suffix])
            external += [dict(node=campaign['server_node'], original=p, saved=p+suffix) for p in renamed.split()]
        names = {'LAUNCH.json', 'RECEIPT.json', '.startup_inputs', 'LAUNCH_WORKER_DEPLOYMENT.json',
                 'PHASE_TIMINGS.jsonl', 'TETHER_RETRY_RECEIPT.json', 'TETHER_CALLS.jsonl', 'BATCHING_CONTROL.json',
                 'servers', 'mps', 'mps_control', 'BACKEND_EVIDENCE.json', 'PORT_RELEASE.json',
                 'SERVER_ATTESTATIONS.json', 'GPU_MEMORY.json', 'GPU_RELEASED.json', 'PROXY_METADATA.json',
                 'BUNDLE_ACK.json', 'STUDY_BUNDLE_ACKS.json'}
        names.update(p.name for pattern in ('*_LAUNCH_ADMISSION.json', '*_launch.json', 'MPS_*.json',
                                           'replica_*.log', 'mps_*.log') for p in root.glob(pattern))
        moved = {}
        for name in sorted(names):
            path = root/name
            if path.exists():
                target = path.with_name(name+suffix)
                path.rename(target)
                moved[name] = target.name
        record = root/('RESET'+suffix+'.json')
        metadata = dict(run=str(root), original_failure=receipt, renamed=moved, restored_inputs=restored,
                        external_renames=external, driver_started=False, reset_started_processes=False)
        write(record, dict(status='RESTAGING', **metadata))
        # Reuse the original seal and native releases; repair staged controls.
        _stage(root)
        write(record, dict(status='PREPARED', **metadata))
    print(str(record))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--verified-dead', action='store_true',
                   help='local-only startup failure whose receipt could not prove closure; every recorded pid is re-checked')
    a = p.parse_args()
    reset(a.run, a.verified_dead)
