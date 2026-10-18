"""Command-only GPU2 island for ff/ffk; frozen releases stay unchanged.

NVML index 2 is kernel minor 1 / native NVIDIA EGL index 1. As with the
original island, an EGL alias preserves the lease ordinal inside a namespace
that exposes only the UUID-authorized card. No renderer is initialized here.
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
GUARD = '/scratch/zixuans8/mw_r17/timan1_provision/gpu3_guard'
WORKER = 'exp.r17_integration.ops.formal_workers'
# Actual framebuffer: 49140 MiB, idle use: 10 MiB. RAM budget: 450 GiB.
SIZES = {'libero': (530, 2048), 'metaworld': (133, 512)}
CAPS = {d: min((49140-4914-10)//v, (450*1024*9//10)//r)
        for d, (v, r) in SIZES.items()}


def authorize(c):
    n = c.get('execution_worker_nodes', c.get('worker_nodes', {})).get('timan1', 0)
    if not n:
        return None
    domain = c.get('domain')
    from exp.offline_search.rounds.r17.autoscale_1017.switch import enabled
    dynamic = enabled() and bool(c.get('autoscale'))
    if (domain not in SIZES or domain == 'libero' and not c.get('cache_study') or
            type(n) is not int or n < 1 or not dynamic and n > CAPS[domain]):
        raise ValueError('timan1 requires LIBERO cache / MetaWorld workers within GPU2 capacity')
    vram, ram = SIZES[domain]
    return dict(schema='r17.timan1_gpu2_lease.v1', workers=n, device=DEVICE,
                device_minor=MINOR, uuid=UUID, pci=PCI, excluded_devices=[0, 1, 3],
                worker_memory_mib=ram, worker_vram_mib=vram,
                worker_max_episodes=4 if domain == 'libero' else 32 if dynamic else 0,
                maximum_workers=CAPS[domain])


def route(worker, campaign):
    if worker.get('worker_node') != 'timan1':
        return worker
    lease = authorize(campaign)
    from exp.offline_search.rounds.r17.autoscale_1017.switch import enabled
    dynamic = enabled() and bool(campaign.get('autoscale'))
    if lease and dynamic:lease = dict(lease, workers=worker['node_workers'])
    if not lease or worker['node_workers'] != lease['workers']:
        raise ValueError('timan1 worker count differs from GPU2 lease')
    if dynamic:worker['worker_max_seconds'] = 1800
    else:worker.pop('worker_max_seconds', None)
    worker.update(worker_egl_device=DEVICE, worker_egl_devices=[DEVICE],
                  timan1_worker_lease=lease,
                  worker_ram_reservation_mib=lease['worker_memory_mib'])
    worker.update({k: lease[k] for k in
                   ('worker_memory_mib', 'worker_vram_mib', 'worker_max_episodes')})
    return worker


def environment(env):
    env = dict(env)
    env['PYTHONPATH'] = ':'.join(p for p in env.get('PYTHONPATH', '').split(':') if p != GUARD)
    env.update(MUJOCO_EGL_DEVICE_ID=str(DEVICE), CUDA_VISIBLE_DEVICES='',
               MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', PYTHONDONTWRITEBYTECODE='1')
    env.pop('R17_GPU3_DEVICE_ISOLATION', None)
    return env


def inventory(node, devices=None):
    if devices is not None and set(devices) != {DEVICE}:
        return []
    try:
        query = subprocess.run(['nvidia-smi', '-i', UUID, '-q', '-x'],
                               capture_output=True, text=True, timeout=8, check=True)
        card = ET.fromstring(query.stdout).find('gpu')
        pci = card.findtext('pci/pci_bus_id', '').lower().replace('00000000:', '0000:')
        if (card.findtext('uuid') != UUID or pci != PCI or
                int(card.findtext('minor_number')) != MINOR):
            raise ValueError('timan1 GPU2 UUID/PCI/minor differs from lease')
        memory = card.find('fb_memory_usage')
        number = lambda value: int(value.split()[0])
        processes = [dict(pid=int(p.findtext('pid')), used_mib=number(p.findtext('used_memory')))
                     for p in card.findall('processes/process_info')
                     if p.findtext('used_memory') not in (None, 'N/A')]
        return [dict(device=DEVICE, device_minor=MINOR, pci=PCI, uuid=UUID, usable=True,
                     total_mib=number(memory.findtext('total')), used_mib=number(memory.findtext('used')),
                     processes=processes, namespace_guard_required=True)]
    except (OSError, ValueError, AttributeError, ET.ParseError, subprocess.SubprocessError) as exc:
        return [dict(device=DEVICE, usable=False, reason=repr(exc))]


def install_runtime():
    """Patch only modules in this process, including old LIBERO-only archives."""
    from exp.r17_integration.ops import timan1_workers as t, worker_gpu as g, worker_memory as m
    t.UUID, t.PCI, t.authorize, t.route = UUID, PCI, authorize, route
    g.PROTECTED['timan1'] = {0, 1, 3}
    native_inventory, native_admission = g.inventory, g.admission
    g.inventory = lambda node, devices=None: (inventory(node, devices) if node == 'timan1'
                                             else native_inventory(node, devices))

    def admission(c, run, *, observed=None, ignore_index=None, count=None):
        if c.get('worker_node') != 'timan1':
            return native_admission(c, run, observed=observed, ignore_index=ignore_index, count=count)
        rows = g.inventory('timan1', c.get('worker_egl_devices')) if observed is None else observed
        allowed = c.get('worker_egl_device') == DEVICE and c.get('worker_egl_devices') == [DEVICE]
        cards = {DEVICE: r for r in rows if allowed and r.get('usable') and
                 (r.get('device'), r.get('uuid'), r.get('pci'), r.get('device_minor')) ==
                 (DEVICE, UUID, PCI, MINOR)}
        pending, evidence = g.reservations(run, cards, ignore_index=ignore_index)
        vram = c['worker_vram_mib']
        if type(vram) is not int or vram < min(v for v, _ in SIZES.values()):
            raise ValueError('measured timan1 per-worker VRAM reservation required')
        required = c['node_workers'] if count is None else count
        fit = 0
        for card in cards.values():
            headroom = max(512, (card['total_mib']+9)//10) + c.get('worker_server_vram_headroom_mib', 0)
            card.update(headroom_mib=headroom, pending_mib=pending[DEVICE])
            fit = max(0, (card['total_mib']-card['used_mib']-headroom-pending[DEVICE])//vram)
        assignments = [dict(index=i, device=DEVICE, uuid=UUID, worker_vram_mib=vram)
                       for i in range(min(required, fit))]
        return dict(admitted=len(assignments) == required, assignments=assignments,
                    cards=list(cards.values()), observed=rows, other_fleets=evidence,
                    worker_vram_mib=vram, excluded_devices=[0, 1, 3], requested_workers=required,
                    reason=None if len(assignments) == required else 'timan1 GPU2 capacity/device admission refused')
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


def guard_source():
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
    return source


def namespace(python, args, kind):
    if os.uname().nodename.split('.')[0] != 'timan1':
        raise ValueError('GPU2 namespace requires timan1')
    info = Path('/proc/driver/nvidia/gpus/'+PCI+'/information').read_text()
    if UUID not in info or int(info.split('Device Minor:', 1)[1].split()[0]) != MINOR:
        raise ValueError('timan1 GPU2 PCI/UUID/minor mapping changed')
    render = '/dev/dri/by-path/pci-'+PCI+'-render'
    if os.path.realpath(render) != '/dev/dri/renderD129':
        raise ValueError('timan1 GPU2 render node changed')
    env = environment(os.environ)
    env['R17_GPU2_DEVICE_ISOLATION'] = '1'
    argv = ['/usr/bin/bwrap', '--unshare-user', '--die-with-parent', '--bind', '/', '/', '--dev-bind', '/dev', '/dev']
    for dev in Path('/dev').glob('nvidia[0-9]*'):
        if dev.name != 'nvidia'+str(MINOR):
            argv += ['--ro-bind', '/dev/null', str(dev)]
    allowed = {p.name for p in Path('/sys/bus/pci/devices/'+PCI+'/drm').iterdir()}
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
    exec(compile(guard_source(), '<timan1_gpu2_guard>', 'exec'), guard.__dict__)
    if kind == 'fleet':
        from exp.r17_integration.ops import formal_workers as f
        proxy = types.SimpleNamespace(**vars(json))
        def loads(value, *a, **kw):
            c = json.loads(value, *a, **kw)
            return route(c, c) if isinstance(c, dict) and c.get('worker_node') == 'timan1' and 'worker_run' in c else c
        proxy.loads = loads
        f.json = proxy
        command = f.worker_command
        def worker_command(c, index, python=sys.executable):
            worker_id, argv = command(c, index, python)
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


def bootstrap(python, args, kind):
    from exp.offline_search.rounds.r17.autoscale_1017.switch import source as preparation_source
    source = preparation_source(__file__).decode()
    code = 's='+repr(source)+';g={"SOURCE":s};exec(s,g);g["namespace"]('+repr(python)+','+repr(args)+','+repr(kind)+')'
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
            return value[:i]+bootstrap(python, value[i+1:], 'native')
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
                return value[:i-1]+bootstrap(value[i-1], value[i:], 'native')
            # Remaining -c tokens belong to the program's argv, not Python.
            return value
    return value


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
            raise ValueError('invalid timan1 GPU2 count')
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
