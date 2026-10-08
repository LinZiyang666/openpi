"""Explicit R16 entry point; ordinary plugin/server installations are untouched."""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .prepare import write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ArrayPool:
    """Intern frozen fit arrays, which the existing clone_method already shares.

    No device transfer, precision conversion, fit, or flag mutation. Each
    connection still deep-copies every mutable method container using the
    original plugin clone. CPU tensors are left alone; CUDA fits are refused.
    """
    def __init__(self):
        self.arrays = {}

    def intern(self, obj, seen=None):
        seen = {} if seen is None else seen
        if id(obj) in seen:
            return seen[id(obj)]
        if isinstance(obj, np.ndarray):
            if obj.dtype.hasobject:
                raise ValueError('Object arrays cannot be resident fit buffers')
            data = np.ascontiguousarray(obj)
            key = (obj.dtype.str, obj.shape, hashlib.sha256(memoryview(data).cast('B')).hexdigest())
            old = self.arrays.setdefault(key, obj)
            assert np.array_equal(old, obj, equal_nan=True)
            return old
        if hasattr(obj, 'device') and str(obj.device).startswith('cuda'):
            raise ValueError('Resident fits must be CPU fits')
        seen[id(obj)] = obj
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                obj[key] = self.intern(value, seen)
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                obj[i] = self.intern(value, seen)
        elif isinstance(obj, tuple):
            obj = tuple(self.intern(v, seen) for v in obj)
            seen[id(obj)] = obj
        elif hasattr(obj, '__dict__') and type(obj).__module__.startswith('exp.'):
            for key, value in list(vars(obj).items()):
                setattr(obj, key, self.intern(value, seen))
        return obj

    @property
    def bytes(self):
        return sum(a.nbytes for a in self.arrays.values())


class Registry:
    def __init__(self, runtimes):
        self.runtimes = runtimes
        self.api = next(iter(runtimes.values())).api

    def attach(self, comps, config, yaml_id=None):
        if yaml_id not in self.runtimes:
            raise ValueError(f'ConfigValidationError: unregistered R16 yaml_id {yaml_id!r}')
        return self.runtimes[yaml_id].attach(comps, config, yaml_id=yaml_id)

    def flush_all(self):
        for runtime in self.runtimes.values():
            runtime.flush_all()


def install(group, model, server_tag, suite=None):
    from exp.offline_search.closed_loop import plugin
    import openpi.cache.config as cc
    from openpi.serving import websocket_policy_server as wps

    if plugin.RUNTIME is not None:
        raise RuntimeError('Plugin already installed')
    pool = ArrayPool()
    runtimes = {}
    entries = [e for e in group['entries'] if suite is None or e['row']['suite'] == suite]
    if not entries or group['model'] != model:
        raise ValueError('Empty/wrong resident model set')
    for entry in entries:
        row = entry['row']; name = row['arm']
        fit = row['plugin_args'][row['plugin_args'].index('--os-fit-artifact')+1]
        if digest(fit) != entry['fit_sha256'] or not Path(fit).is_file():
            raise ValueError('Frozen fit changed/missing; refitting is forbidden')
        if digest(row['yaml']) != entry['yaml_sha256']:
            raise ValueError('Resident YAML changed')
        args = ['--os-method', row['method'], '--os-kwargs', json.dumps(row['kwargs']),
                '--os-cell', row['cell'], *row['plugin_args'],
                '--os-log-dir', str(Path(group['root'])/'runs'/name/f'server_{server_tag}'),
                '--os-tag', f'{name}_{server_tag}']
        opts, rest = plugin.parse_cli(args)
        if rest or getattr(opts, 'os_gpu_retrieval', None):
            raise ValueError('Unexpected resident plugin options')
        opts.os_miss_steps = group['k']
        # The startup argv must remain the original arm's plugin binding.
        previous = sys.argv
        try:
            sys.argv = [previous[0], *args]
            rt = plugin.PluginRuntime(opts, model)
        finally:
            sys.argv = previous
        rt.method = pool.intern(rt.method)
        rt.tables = pool.intern(rt.tables)
        rt.cur_action = pool.intern(rt.cur_action)
        emit = rt.emit
        def arm_emit(data, *, emit=emit, name=name):
            emit(dict(data, arm=name))
        rt.emit = arm_emit
        rt.emit(dict(ev='resident_binding', yaml_id=name, arm=name, fit_sha256=entry['fit_sha256'],
                     group_sha256=group['group_sha256'], miss_steps=group['k']))
        runtimes[name] = rt
    registry = Registry(runtimes)
    plugin.RUNTIME = registry
    original = cc.build_per_connection_components
    def build(config, shared_storage, *args, **kwargs):
        return registry.attach(original(config, shared_storage, *args, **kwargs), config,
                               yaml_id=kwargs.get('yaml_id'))
    build.__wrapped__ = original
    cc.build_per_connection_components = build
    base = wps.WebsocketPolicyServer
    class ResidentServer(base):
        def __init__(self, *args, **kwargs):
            factory = kwargs.get('connection_policy_factory')
            if factory is None:
                raise ValueError('Resident serving requires concurrent mode')
            kwargs['connection_policy_factory'] = plugin._wrap_factory(factory)
            super().__init__(*args, **kwargs)
    wps.WebsocketPolicyServer = ResidentServer
    atexit.register(registry.flush_all)
    write_json(Path(group['root'])/'resident'/f'{server_tag}.json', dict(
        model=model, k=group['k'], suite=suite, methods=list(runtimes),
        unique_cpu_array_bytes=pool.bytes, additional_fit_gpu_bytes=0,
        group_sha256=group['group_sha256']))
    return registry


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--registry', type=Path, required=True)
    ap.add_argument('--server-tag', required=True)
    ap.add_argument('--resident-suite')
    args, rest = ap.parse_known_args()
    from .conductor import read_group
    group = read_group(args.registry)
    from exp.offline_search.closed_loop import plugin
    plugin.install = lambda opts, model: install(group, model, args.server_tag, args.resident_suite)
    sys.argv = [sys.argv[0], *rest]
    if group['model'] == 'pi05':
        from exp.offline_search.closed_loop.serve_pi05 import main as serve
    else:
        from exp.offline_search.closed_loop.serve_groot import main as serve
    serve()


if __name__ == '__main__':
    main()
