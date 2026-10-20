"""Command-only timan1 GPU2/GPU1 island for ff/ffk; frozen releases stay unchanged.

NVML index 2 is kernel minor 1 / native NVIDIA EGL index 1. As with the
original island, an EGL alias preserves the lease ordinal inside a namespace
that exposes only the UUID-authorized cards. No renderer is initialized here.

Owner 2026-10-18: NVML index 1 (PCI 0000:41:00.0, kernel minor 2) is also
authorized, shared with other users by measured free VRAM; GPU2 stays first.
A lease without ``devices`` (prepared or launched before that ruling) keeps the
GPU2-only namespace, guard, admission and capacity exactly as before.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import types
from unittest.mock import patch
import xml.etree.ElementTree as ET

UUID = 'GPU-9bdcf7cc-8a05-a7f9-b1c3-4d53d0a229df'
PCI = '0000:81:00.0'
DEVICE, MINOR = 2, 1
# NVML index -> identity; dict order is the fill order. GPUs 0/3 belong to others.
CARDS = {2: dict(uuid=UUID, pci=PCI, minor=MINOR, render='renderD129'),
         1: dict(uuid='GPU-3c43a0c0-b32f-2dfd-b530-615cb00ec924', pci='0000:41:00.0',
                 minor=2, render='renderD130')}
ORDER = tuple(CARDS)
LEASE, CARD_LEASE = 'r17.timan1_gpu2_lease.v1', 'r17.timan1_gpu12_lease.v1'
ISLAND = 'R17_TIMAN1_EGL_DEVICES'  # set only by namespace(); absent means GPU2 only
GUARD = '/scratch/zixuans8/mw_r17/timan1_provision/gpu3_guard'
WORKER = 'exp.r17_integration.ops.formal_workers'
# Actual framebuffer: 49140 MiB per card, idle use: 10 MiB. RAM budget: 450 GiB.
SIZES = {'libero': (530, 2048), 'metaworld': (133, 512)}


def caps(devices=ORDER):
    return {d: min(len(devices)*((49140-4914-10)//v), (450*1024*9//10)//r)
            for d, (v, r) in SIZES.items()}


GPU2_CAPS = caps((DEVICE,))
CAPS = caps()


def cards(devices):
    devices = list(devices)
    if DEVICE not in devices or len(set(devices)) != len(devices) or set(devices) - set(CARDS):
        raise ValueError('timan1 workers need GPU2 and may add only GPU1: '+repr(devices))
    return [d for d in ORDER if d in devices]


def label(devices):
    return '/'.join('GPU'+str(d) for d in devices)


def leased(c):
    """Devices of an issued lease; a lease without them keeps GPU2 only."""
    lease = c.get('timan1_worker_lease') or {}
    return cards(lease['devices']) if lease.get('schema') == CARD_LEASE else [DEVICE]


def island():
    """Cards exposed by this process's namespace (see namespace())."""
    value = os.environ.get(ISLAND)
    return cards(int(v) for v in value.split(',')) if value else [DEVICE]


def authorize(c, devices=ORDER):
    n = c.get('execution_worker_nodes', c.get('worker_nodes', {})).get('timan1', 0)
    if not n:
        return None
    devices = cards(devices)
    domain = c['domain']
    vram, ram = SIZES[domain]
    # n is the node's hard cap; the central scheduler decides active counts.
    lease = dict(schema=LEASE, workers=n, device=DEVICE,
                 device_minor=MINOR, uuid=UUID, pci=PCI, excluded_devices=[0, 1, 3],
                 worker_memory_mib=ram, worker_vram_mib=vram,
                 # No recycling on timan1 (owner 10-19: timan1 workers need no restart). LIBERO workers do
                 # not leak per episode; MetaWorld workers share one EGL context (13 MiB over 150 episodes).
                 # Spawns here take 5-9 s each, so MetaWorld's 32-episode/30-min generations started together
                 # and exited together, leaving timan1 at a few live workers of 50-70 targets (10-19 21:4x).
                 worker_max_episodes=0,
                 maximum_workers=caps(devices)[domain])
    if devices != [DEVICE]:
        lease.update(schema=CARD_LEASE, devices=devices,
                     excluded_devices=sorted({0, 1, 2, 3}-set(devices)),
                     cards=[dict(device=d, **CARDS[d]) for d in devices])
    return lease


def route(worker, campaign, devices=ORDER):
    if worker.get('worker_node') != 'timan1':
        return worker
    lease = dict(authorize(campaign, devices), workers=worker['node_workers'])
    worker['worker_max_seconds'] = 1800 if lease['worker_max_episodes'] else 0
    worker.update(worker_egl_device=DEVICE, worker_egl_devices=list(lease.get('devices', [DEVICE])),
                  timan1_worker_lease=lease,
                  worker_ram_reservation_mib=lease['worker_memory_mib'])
    worker.update({k: lease[k] for k in
                   ('worker_memory_mib', 'worker_vram_mib', 'worker_max_episodes')})
    return worker


def island_route(worker, campaign):
    """Inside the namespace: keep the device set the coordinator leased."""
    return route(worker, campaign, leased(campaign))


def environment(env):
    env = dict(env)
    env['PYTHONPATH'] = ':'.join(p for p in env.get('PYTHONPATH', '').split(':') if p != GUARD)
    env.update(MUJOCO_EGL_DEVICE_ID=str(DEVICE), CUDA_VISIBLE_DEVICES='',
               MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', PYTHONDONTWRITEBYTECODE='1')
    env.pop('R17_GPU3_DEVICE_ISOLATION', None)
    env.pop(ISLAND, None)
    return env


def query(device):
    """One UUID-selected NVML read; fail closed for this card only."""
    card = CARDS[device]
    try:
        reply = subprocess.run(['nvidia-smi', '-i', card['uuid'], '-q', '-x'],
                               capture_output=True, text=True, timeout=8, check=True)
        gpu = ET.fromstring(reply.stdout).find('gpu')
        pci = gpu.findtext('pci/pci_bus_id', '').lower().replace('00000000:', '0000:')
        if (gpu.findtext('uuid') != card['uuid'] or pci != card['pci'] or
                int(gpu.findtext('minor_number')) != card['minor']):
            raise ValueError('timan1 GPU'+str(device)+' UUID/PCI/minor differs from lease')
        memory = gpu.find('fb_memory_usage')
        number = lambda value: int(value.split()[0])
        processes = [dict(pid=int(p.findtext('pid')), used_mib=number(p.findtext('used_memory')))
                     for p in gpu.findall('processes/process_info')
                     if p.findtext('used_memory') not in (None, 'N/A')]
        return dict(device=device, device_minor=card['minor'], pci=card['pci'], uuid=card['uuid'], usable=True,
                    total_mib=number(memory.findtext('total')), used_mib=number(memory.findtext('used')),
                    processes=processes, namespace_guard_required=True)
    except (OSError, ValueError, AttributeError, ET.ParseError, subprocess.SubprocessError) as exc:
        return dict(device=device, usable=False, reason=repr(exc))


def inventory(node, devices=None):
    """Only cards present in this namespace; never query another user's card."""
    if devices is not None and set(devices) - set(CARDS):
        return []
    return [query(d) for d in island() if devices is None or d in devices]


def admit(c, run, rows, present, *, ignore_index=None, count=None):
    """Measured free VRAM minus headroom per card; GPU2 is filled first."""
    from exp.r17_integration.ops import worker_gpu as g
    requested = c.get('worker_egl_devices') or []
    allowed = ([d for d in ORDER if d in present and d in requested]
               if c.get('worker_egl_device') == DEVICE and not set(requested) - set(CARDS) else [])
    selected = {}
    for d in allowed:
        identity = (d, CARDS[d]['uuid'], CARDS[d]['pci'], CARDS[d]['minor'])
        for r in rows:
            if r.get('usable') and (r.get('device'), r.get('uuid'), r.get('pci'), r.get('device_minor')) == identity:
                selected[d] = r
    pending, evidence = g.reservations(run, selected, ignore_index=ignore_index)
    vram = c['worker_vram_mib']
    if type(vram) is not int or vram < min(v for v, _ in SIZES.values()):
        raise ValueError('measured timan1 per-worker VRAM reservation required')
    required = c['node_workers'] if count is None else count
    assignments = []
    for d, card in selected.items():
        headroom = max(512, (card['total_mib']+9)//10) + c.get('worker_server_vram_headroom_mib', 0)
        card.update(headroom_mib=headroom, pending_mib=pending[d])
        fit = max(0, (card['total_mib']-card['used_mib']-headroom-pending[d])//vram)
        assignments += [dict(index=len(assignments)+i, device=d, uuid=CARDS[d]['uuid'], worker_vram_mib=vram)
                        for i in range(min(fit, required-len(assignments)))]
    return dict(admitted=len(assignments) == required, assignments=assignments,
                cards=list(selected.values()), observed=rows, other_fleets=evidence,
                worker_vram_mib=vram, excluded_devices=sorted({0, 1, 2, 3}-set(present)),
                requested_workers=required,
                reason=None if len(assignments) == required else
                'timan1 '+label(present)+' capacity/device admission refused')


def install_runtime():
    """Patch only modules in this process, including old LIBERO-only archives."""
    from exp.r17_integration.ops import timan1_workers as t, worker_gpu as g, worker_memory as m
    present = island()
    t.UUID, t.PCI = UUID, PCI
    t.authorize, t.route = (lambda c: authorize(c, leased(c))), island_route
    g.PROTECTED['timan1'] = {0, 1, 2, 3}-set(present)
    native_inventory, native_admission = g.inventory, g.admission
    g.inventory = lambda node, devices=None: (inventory(node, devices) if node == 'timan1'
                                             else native_inventory(node, devices))

    def admission(c, run, *, observed=None, ignore_index=None, count=None):
        if c.get('worker_node') != 'timan1':
            return native_admission(c, run, observed=observed, ignore_index=ignore_index, count=count)
        rows = g.inventory('timan1', c.get('worker_egl_devices')) if observed is None else observed
        return admit(c, run, rows, present, ignore_index=ignore_index, count=count)
    g.admission = admission
    reservation, memory_admission = m.reservation_mib, m.admission
    m.reservation_mib = lambda c: (c['worker_ram_reservation_mib'] if 'worker_ram_reservation_mib' in c
                                  else reservation(c))

    def memory(c, run, **kw):
        result = memory_admission(c, run, **kw)
        result['headroom_bytes'] = max(result['headroom_bytes'], (result['total_bytes']+9)//10)
        result['admitted'] = (result['requested_bytes']+result['other_pending_bytes']+
                              result['headroom_bytes'] <= result['available_bytes'])
        if not result['admitted']:
            result['reason'] = 'timan1 RAM capacity with 10% headroom refused'
        return result
    m.admission = memory


# Multi-card guard edits on top of the GPU2 rewrite; each must match exactly once.
EGL_SINGLE = """            if len(native) != 1:
                deny('isolated EGL must expose exactly one NVIDIA device')
            query_binary = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))(module.eglGetProcAddress('eglQueryDeviceBinaryEXT'))
            value = (ctypes.c_ubyte * 16)(); size = ctypes.c_int()
            if not query_binary(native[0][1], 0x335C, 16, value, ctypes.byref(size)) or size.value != 16:
                deny('EGL physical device UUID query failed')
            actual = bytes(value).hex()
            if actual != UUID.removeprefix('GPU-').replace('-', ''):
                deny('isolated EGL UUID differs from physical GPU2: '+actual)
            os.environ['R17_GPU2_EGL_UUID_HEX'] = actual
            os.environ['R17_GPU2_EGL_NATIVE_INDEX'] = str(native[0][0])
"""
EGL_CARDS = """            if not 1 <= len(native) <= len(UUIDS):
                deny('isolated EGL must expose only the authorized NVIDIA devices')
            query_binary = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))(module.eglGetProcAddress('eglQueryDeviceBinaryEXT'))
            hexes = {u.removeprefix('GPU-').replace('-', '') for u in UUIDS.values()}
            wanted = UUIDS[os.environ['MUJOCO_EGL_DEVICE_ID']].removeprefix('GPU-').replace('-', '')
            selected = []
            for index, device in native:
                value = (ctypes.c_ubyte * 16)(); size = ctypes.c_int()
                if not query_binary(device, 0x335C, 16, value, ctypes.byref(size)) or size.value != 16:
                    continue  # an unverifiable card is never selected
                actual = bytes(value).hex()
                if actual not in hexes:
                    deny('isolated EGL exposes an unauthorized device: '+actual)
                if actual == wanted:
                    selected.append((index, device))
            if len(selected) != 1:
                deny('EGL physical device UUID query failed')
            native = selected
            actual = wanted
            os.environ['R17_GPU2_EGL_UUID_HEX'] = actual
            os.environ['R17_GPU2_EGL_NATIVE_INDEX'] = str(native[0][0])
"""


def guard_source(devices=(DEVICE,)):
    """Rebind the pinned guard in memory; never install another sitecustomize."""
    path = Path(GUARD)/'sitecustomize.py'
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != '4bb5b8280cfa21328c9d07c4aae4b8220fc9b68605d6b3160288b4fab98bb2b8':
        raise ValueError('original timan1 guard changed')
    source = body.decode().replace('GPU-ba276422-a0b1-e371-8327-f744be942f15', UUID)
    for old, new in (("'3'", "'2'"), ('--egl-device=3', '--egl-device=2'), (' != 3', ' != 2'), ('[3]', '[2]'),
                     ('[native[0][1]] * 4', '[native[0][1]] * 3'),
                     ('for ordinal in (1, 2, 3):', 'for ordinal in (0, 2, 3):'),
                     ('/dev/nvidia0', '/dev/nvidia1'), ('195, 0', '195, 1'),
                     ('GPU3', 'GPU2'), ('gpu3', 'gpu2')):
        source = source.replace(old, new)
    devices = cards(devices)
    if devices == [DEVICE]:
        return source
    minors = tuple(CARDS[d]['minor'] for d in devices)
    uuids = {str(d): CARDS[d]['uuid'] for d in devices}
    for old, new in (
            ("UUID = '"+UUID+"'\n", "UUID = '"+UUID+"'\nUUIDS = "+repr(uuids)+"\n"),
            ("os.environ.get('MUJOCO_EGL_DEVICE_ID') != '2':",
             "os.environ.get('MUJOCO_EGL_DEVICE_ID') not in UUIDS:"),
            ("os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', UUID):",
             "os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', UUIDS.get(os.environ.get('MUJOCO_EGL_DEVICE_ID'))):"),
            ("argv[i+1] != '2'", "argv[i+1] != os.environ.get('MUJOCO_EGL_DEVICE_ID')"),
            ("arg != '--egl-device=2'", "arg != '--egl-device='+os.environ.get('MUJOCO_EGL_DEVICE_ID', '')"),
            ("campaign.get('worker_egl_devices') != [2]:",
             "not set(campaign.get('worker_egl_devices') or [-1]) <= "+repr(set(devices))+":"),
            ("a.get('device') != 2 for a", "a.get('device') not in "+repr(tuple(devices))+" for a"),
            (EGL_SINGLE, EGL_CARDS),
            ('for ordinal in (0, 2, 3):', 'for ordinal in '+repr(tuple(m for m in range(4) if m not in minors))+':'),
            ("if os.stat('/dev/nvidia1').st_rdev != os.makedev(195, 1):",
             "if any(os.stat('/dev/nvidia'+str(m)).st_rdev != os.makedev(195, m) for m in "+repr(minors)+"):")):
        if source.count(old) != 1:
            raise ValueError('original timan1 guard changed: '+old.strip().splitlines()[0])
        source = source.replace(old, new)
    return source


def exposed(args, kind, devices=None):
    """Identity-checked cards for one namespace; GPU2 is required, an absent GPU1 is dropped."""
    if devices is None:
        devices = [DEVICE]
        if kind == 'fleet' and '--campaign' in args:
            try:
                devices = leased(json.loads(Path(args[args.index('--campaign')+1]).read_text()))
            except (OSError, ValueError, IndexError):
                devices = [DEVICE]
    devices = cards(devices)
    present = []
    for d in devices:
        card = CARDS[d]
        info = Path('/proc/driver/nvidia/gpus/'+card['pci']+'/information')
        if d != DEVICE and not info.exists():
            continue
        info = info.read_text()
        if card['uuid'] not in info or int(info.split('Device Minor:', 1)[1].split()[0]) != card['minor']:
            raise ValueError('timan1 GPU'+str(d)+' PCI/UUID/minor mapping changed')
        render = '/dev/dri/by-path/pci-'+card['pci']+'-render'
        if os.path.realpath(render) != '/dev/dri/'+card['render']:
            raise ValueError('timan1 GPU'+str(d)+' render node changed')
        present.append(d)
    return present


def namespace(python, args, kind, devices=None):
    if os.uname().nodename.split('.')[0] != 'timan1':
        raise ValueError('GPU2 namespace requires timan1')
    present = exposed(args, kind, devices)
    env = environment(os.environ)
    env['R17_GPU2_DEVICE_ISOLATION'] = '1'
    if present != [DEVICE]:
        env[ISLAND] = ','.join(map(str, present))
    argv = ['/usr/bin/bwrap', '--unshare-user', '--die-with-parent', '--bind', '/', '/', '--dev-bind', '/dev', '/dev']
    minors = {'nvidia'+str(CARDS[d]['minor']) for d in present}
    for dev in Path('/dev').glob('nvidia[0-9]*'):
        if dev.name not in minors:
            argv += ['--ro-bind', '/dev/null', str(dev)]
    allowed = {p.name for d in present for p in Path('/sys/bus/pci/devices/'+CARDS[d]['pci']+'/drm').iterdir()}
    for dev in Path('/dev/dri').glob('*'):
        if dev.is_char_device() and dev.name not in allowed:
            argv += ['--ro-bind', '/dev/null', str(dev)]
    code = 's='+repr(SOURCE)+';g={"SOURCE":s};exec(s,g);g["run"]('+repr(kind)+')'
    os.execve(argv[0], [*argv, '--', python, '-B', '-u', '-c', code, *args], env)


def run(kind):
    # Legacy VLA workers use the same device namespace/renderer guard, without
    # importing or changing the R17 fleet implementation in their R16 island.
    if kind != 'legacy':
        install_runtime()
    # The old guard's orig_argv campaign check reads the sealed GPU3 config.
    # Validate the routed in-memory campaign instead; keep the OS argv marker
    # intact for native fleet ownership/closure probes.
    sys.orig_argv = [a for a in sys.orig_argv if a != WORKER]
    guard = types.ModuleType('sitecustomize')
    sys.modules['sitecustomize'] = guard
    exec(compile(guard_source(island()), '<timan1_gpu2_guard>', 'exec'), guard.__dict__)
    if kind == 'fleet':
        from exp.r17_integration.ops import formal_workers as f
        proxy = types.SimpleNamespace(**vars(json))
        def loads(value, *a, **kw):
            c = json.loads(value, *a, **kw)
            return island_route(c, c) if isinstance(c, dict) and c.get('worker_node') == 'timan1' and 'worker_run' in c else c
        proxy.loads = loads
        f.json = proxy
        command = f.worker_command
        def worker_command(c, index, python=sys.executable):
            worker_id, argv = command(c, index, python)
            assigned = c.get('worker_gpu_assignments')
            if assigned and argv[argv.index('--egl-device')+1] != str(assigned[index]['device']):
                raise ValueError('timan1 worker command differs from its GPU assignment')
            # All children inherit the island. Direct native children preserve
            # stock lifetime receipts and exit75 classification without a relay.
            argv[0] = python
            target = ('worker_lifetime' if c.get('worker_max_episodes') else
                      'cache_study_worker' if c.get('cache_study') else None)
            argv[argv.index('-m')+1] = ('exp.r17_integration.ops.'+target if target else
                                      'exp.r17_integration.conductor.worker_entry')
            # Reinstall the in-memory renderer guard before native imports.
            return worker_id, [python, '-B', '-u', '-c',
                's='+repr(SOURCE)+';g={"SOURCE":s};exec(s,g);g["run"]("native")', *argv[2:]]
        f.worker_command = worker_command
        del sys.argv[1]
        f.main()
    elif sys.argv[1] == '-m':
        module = sys.argv[2]
        sys.argv = [module, *sys.argv[3:]]
        runpy.run_module(module, run_name='__main__', alter_sys=True)
    elif sys.argv[1] == '-c':
        code = sys.argv[2]
        sys.argv = ['-c', *sys.argv[3:]]
        exec(compile(code, '<timan1_gpu2_command>', 'exec'), {'__name__': '__main__'})
    else:
        raise ValueError('unsupported GPU2 Python command')


def bootstrap(python, args, kind, devices=None):
    """devices=None: a fleet follows its campaign's lease, anything else GPU2 only."""
    source = Path(__file__).read_text()
    call = repr(python)+','+repr(args)+','+repr(kind)+('' if devices is None else ','+repr(cards(devices)))
    code = 's='+repr(source)+';g={"SOURCE":s};exec(s,g);g["namespace"]('+call+')'
    # -S prevents the original guard running before the command can override it.
    return ['/usr/bin/python3', '-B', '-S', '-c', code, *args]


def auxiliary(value):
    """Cover prepare's import probes and CPU admission in the same island."""
    if not isinstance(value, list):
        return value
    for i, arg in enumerate(value):
        if isinstance(arg, str) and arg.endswith(('/python-gpu3', '/python-mw-gpu3')):
            python = ('/scratch/zixuans8/mw_r17/envs/sim/bin/python' if arg.endswith('/python-mw-gpu3')
                      else '/scratch/zixuans8/openpi/.venv/bin/python')
            return value[:i]+bootstrap(python, value[i+1:], 'native', ORDER)
        if arg == '-c' and i and i+1 < len(value):
            # Ownership probes embed recorded argv, including this module's
            # source, as strings. Only an actual admission import needs GPU2.
            import ast
            try:
                tree = ast.parse(value[i+1])
            except SyntaxError:  # e.g. bash -c, not a Python admission probe
                return value
            if any(isinstance(n, ast.ImportFrom) and n.module in (
                    'exp.r17_integration.ops.worker_gpu',
                    'exp.r17_integration.ops.worker_memory') for n in ast.walk(tree)):
                return value[:i-1]+bootstrap(value[i-1], value[i:], 'native', ORDER)
            # Remaining -c tokens belong to the program's argv, not Python.
            return value
    return value


def capacity_cards():
    """PCI -> (UUID, kernel minor, NVML index) for worker_resilience.capacity."""
    return {CARDS[d]['pci']: (CARDS[d]['uuid'], CARDS[d]['minor'], d) for d in ORDER}


@contextmanager
def coordinator():
    """Scope new authorization to these entry points, including unstaged maps."""
    from exp.r17_integration.ops import timan1_workers as t, local_workers as l, formal_eval as f, worker_memory as m
    counts, remote, reservation = l.counts, f.remote, m.reservation_mib
    def parse(value):
        parts = value.split(',')
        timan1 = [p for p in parts if p.startswith('timan1=')]
        if not timan1:
            return counts(value)
        n = int(timan1[0].split('=', 1)[1])
        if not 1 <= n <= max(CAPS.values()):
            raise ValueError('invalid timan1 GPU2/GPU1 count')
        result = counts(','.join('timan1=1' if p.startswith('timan1=') else p for p in parts))
        return dict(result, timan1=n)
    def call(node, argv, *a, **kw):
        return remote(node, auxiliary(argv) if node == 'timan1' else argv, *a, **kw)
    def ram(c):
        return SIZES[c['domain']][1] if c.get('worker_node') == 'timan1' else reservation(c)
    with patch.object(t, 'authorize', authorize), patch.object(t, 'route', route), \
            patch.object(t, 'environment', environment), patch.object(l, 'counts', parse), \
            patch.object(m, 'reservation_mib', ram), patch.object(f, 'remote', call):
        yield
