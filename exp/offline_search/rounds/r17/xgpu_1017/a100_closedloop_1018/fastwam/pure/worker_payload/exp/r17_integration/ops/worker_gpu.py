"""Physical EGL GPU admission; never signal another fleet or initialize CUDA.

LIBERO reserves per-worker VRAM under a short node-wide admission lock. Both
graphics and compute memory are read through nvidia-smi; live receipt-bound
legacy/orphan workers remain accounted for even when their parent died.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from .formal_process import alive, ticks, write_json

DEFAULT_VRAM_MIB = 1024
PROTECTED = {'timan108': {1, 2}}


def inventory(node, devices=None):
    """Skip protected GPUs before making any NVIDIA query; fail closed per card."""
    excluded = PROTECTED.get(node, set())
    if devices is None:
        devices = []
        for path in Path('/proc/driver/nvidia/gpus').glob('*/information'):
            for line in path.read_text().splitlines():
                if line.startswith('Device Minor:'):
                    devices.append(int(line.split(':', 1)[1]))
    result = []
    for device in sorted(set(devices)-excluded):
        try:
            query = subprocess.run(['nvidia-smi', '-i', str(device), '-q', '-x'],
                capture_output=True, text=True, timeout=8, check=True)
            card = ET.fromstring(query.stdout).find('gpu')
            if card is None or int(card.findtext('minor_number')) != device:
                raise ValueError('physical NVIDIA ordinal differs from device minor')
            memory = card.find('fb_memory_usage')
            number = lambda value: int(value.split()[0])
            row = dict(device=device, uuid=card.findtext('uuid'), name=card.findtext('product_name'),
                       total_mib=number(memory.findtext('total')), used_mib=number(memory.findtext('used')),
                       processes=[], usable=True)
            for proc in card.findall('processes/process_info'):
                pid = int(proc.findtext('pid'))
                try:
                    uid = Path(f'/proc/{pid}').stat().st_uid
                except FileNotFoundError:
                    continue
                amount = proc.findtext('used_memory')
                row['processes'].append(dict(pid=pid, uid=uid,
                    used_mib=number(amount) if amount and amount != 'N/A' else None))
                if uid != os.getuid():
                    row.update(usable=False, reason='other user GPU allocation')
            result.append(row)
        except (OSError, ValueError, ET.ParseError, subprocess.SubprocessError) as exc:
            result.append(dict(device=device, usable=False, reason=repr(exc)))
    return result


def reservations(run, cards, *, ignore_index=None):
    """Future VRAM beyond already observed physical usage, including recycle gaps."""
    pending = {d: 0 for d in cards}
    evidence = []
    for path in Path(run).parent.glob('*/manifest.json'):
        config = json.loads(path.read_text())
        if 'worker_egl_device' not in config:
            continue
        receipt_path = path.parent/'receipt.json'
        data = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        live = [e for e in data.get('pids', []) if alive(e)]
        reservation_path = path.parent/'GPU_RESERVATION.json'
        reservation = json.loads(reservation_path.read_text()) if reservation_path.exists() else {}
        owner = reservation.get('owner') or data.get('fleet_owner')
        holding = owner and alive(owner) and not reservation.get('released', False)
        assignments = reservation.get('assignments', [])
        if holding:
            devices = [(e['index'], e['device']) for e in assignments] if assignments else [
                (i, config['worker_egl_device']) for i in range(config.get('node_workers', data.get('workers',len(live))))]
        else:
            devices = [(e.get('index'), e.get('egl_device', config['worker_egl_device'])) for e in live]
        per_worker = reservation.get('worker_vram_mib', config.get('worker_vram_mib', DEFAULT_VRAM_MIB))
        for device in cards:
            selected = [i for i,d in devices if d == device and
                        not (path.parent == Path(run) and i == ignore_index)]
            if not selected:
                continue
            pids = {e['pid'] for e in live if e.get('egl_device', config['worker_egl_device']) == device and
                    not (path.parent == Path(run) and e.get('index') == ignore_index)}
            actual = sum(p.get('used_mib') or 0 for p in cards[device].get('processes', []) if p['pid'] in pids)
            future = max(0, len(selected)*per_worker-actual)
            pending[device] += future
            evidence.append(dict(run=path.parent.name, device=device, workers=len(selected),
                                 reservation_mib=len(selected)*per_worker, observed_mib=actual, pending_mib=future))
    return pending, evidence


def admission(campaign, run, *, observed=None, ignore_index=None, count=None):
    per_worker = campaign.get('worker_vram_mib', DEFAULT_VRAM_MIB)
    if type(per_worker) is not int or per_worker < 1:
        raise ValueError('positive integer per-worker VRAM reservation required')
    requested_devices = campaign.get('worker_egl_devices')
    protected = PROTECTED.get(campaign.get('worker_node'), set())
    observed = inventory(campaign.get('worker_node'), requested_devices) if observed is None else observed
    cards = {c['device']: c for c in observed if c.get('usable') and c['device'] not in protected and
             (requested_devices is None or c['device'] in requested_devices)}
    pending, evidence = reservations(run, cards, ignore_index=ignore_index)
    slots = {}
    for device, card in cards.items():
        reserve = max(512, (card['total_mib']+9)//10)
        card = dict(card, headroom_mib=reserve, pending_mib=pending[device])
        cards[device] = card
        slots[device] = max(0, (card['total_mib']-card['used_mib']-reserve-pending[device])//per_worker)
    order = sorted(slots)
    preferred = campaign.get('worker_egl_device', 0)
    order = [d for d in order if d >= preferred]+[d for d in order if d < preferred]
    assignments = []
    required = campaign['node_workers'] if count is None else count
    while len(assignments) < required:
        before = len(assignments)
        for device in order:
            if slots[device] and len(assignments) < required:
                assignments.append(dict(index=len(assignments), device=device, uuid=cards[device]['uuid'],
                                        worker_vram_mib=per_worker))
                slots[device] -= 1
        if len(assignments) == before:
            break
    return dict(admitted=len(assignments)==required, assignments=assignments, cards=list(cards.values()),
                observed=observed, other_fleets=evidence, worker_vram_mib=per_worker,
                excluded_devices=sorted(protected), requested_workers=required,
                reason=None if len(assignments)==required else
                f'EGL VRAM admission refused: {required} workers requested, {len(assignments)} fit; '
                'includes global GPU use, live fleet reservations and 10%/512MiB headroom')


class Reservation:
    def __init__(self, campaign, run, result):
        self.campaign, self.run, self.result = campaign, Path(run), result
        self.record = dict(owner=dict(pid=os.getpid(), start_ticks=ticks(os.getpid())),
                           assignments=result['assignments'], worker_vram_mib=result['worker_vram_mib'], released=False)
        write_json(self.run/'GPU_RESERVATION.json', self.record)

    def check(self, index):
        """A replacement consumes its existing reservation; recheck external usage."""
        device = self.result['assignments'][index]['device']
        with (self.run.parent/'.egl_admission.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = admission(dict(self.campaign, worker_egl_devices=[device]), self.run,
                               ignore_index=index, count=1)
        return result['admitted']

    def close(self):
        self.record['released'] = True
        write_json(self.run/'GPU_RESERVATION.json', self.record)


def acquire(campaign, run):
    if 'worker_egl_device' not in campaign:
        return None
    if campaign.get('domain') != 'libero':
        return _exclusive(campaign, run)
    run = Path(run)
    with (run.parent/'.egl_admission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = admission(campaign, run)
        if not result['admitted']:
            raise RuntimeError(result['reason'])
        return Reservation(campaign, run, result)


def live_workers(run):
    receipt = Path(run) / 'receipt.json'
    data = json.loads(receipt.read_text())
    return [entry for entry in data.get('pids', []) if alive(entry)]


def fleet_evidence(request):
    """A closed formal_workers parent does not prove its children closed."""
    argv = request.get('argv', ())
    if 'exp.r17_integration.ops.formal_workers' not in argv:
        return {}
    config = json.loads(Path(argv[argv.index('--campaign') + 1]).read_text())
    run = Path(config['worker_run'])
    manifest = json.loads((run / 'manifest.json').read_text())
    if (manifest['run_name'], manifest['worker_node']) != (config['run_name'], config['worker_node']):
        raise ValueError('worker closure manifest differs from owned campaign')
    return dict(worker_run=str(run), owned_worker_survivors=live_workers(run))


def _exclusive(campaign, run):
    # CPU fleet fixtures supply their own worker command and have no EGL device.
    # Every formal/library native worker campaign carries this explicit field.
    if 'worker_egl_device' not in campaign:
        return None
    device = campaign['worker_egl_device']
    if type(device) is not int or device < 0:
        raise ValueError('nonnegative integer EGL device required')
    run = Path(run)
    lock = (run.parent / f'.egl_gpu_{device}.lock').open('a')
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'EGL GPU{device} is leased by another R17 worker fleet') from None
        # Legacy fleets lack the new lock; failed parents may also release it
        # while leaving D-state children. Check live receipt-bound descendants
        # regardless of the receipt's reported status.
        for receipt in run.parent.glob('*/receipt.json'):
            if receipt.parent == run:
                continue
            manifest = json.loads((receipt.parent / 'manifest.json').read_text())
            if manifest.get('worker_egl_device') != device:
                continue
            survivors = live_workers(receipt.parent)
            data = json.loads(receipt.read_text())
            owner = data.get('fleet_owner')
            if survivors or owner and alive(owner):
                raise RuntimeError(f'EGL GPU{device} still belongs to {receipt.parent.name}; '
                                   f'live worker PIDs={[e["pid"] for e in survivors]}')
        return lock
    except BaseException:
        lock.close()
        raise
