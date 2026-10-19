"""One CPU regression against the actual, unchanged LIBERO and MW archives."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import tarfile
import types
from unittest.mock import Mock

import pytest

from . import timan1_gpu2 as p
from .launch import worker_command


def test_stdlib_closure_argv_does_not_trigger_gpu2_bootstrap():
    import inspect
    from exp.r17_integration.ops.library_salvage import process_probe
    from exp.r17_integration.ops.formal_process import alive, ticks
    entries = [dict(pid=123, start_ticks='456', pgid=123,
                    argv=['/frozen/python', '-c',
                          'from exp.r17_integration.ops.worker_gpu import admission'])]
    code = ('from pathlib import Path; import json\n'+inspect.getsource(ticks)+'\n'+
            inspect.getsource(alive)+'\n'+inspect.getsource(process_probe)+'\n'+
            'print(json.dumps(process_probe('+repr(entries)+',"fa_parent")))')
    assert 'ops.worker_gpu import' in code  # the exact false match in the receipt
    argv = ['env', 'CUDA_VISIBLE_DEVICES=', 'python3', '-c', code]
    assert p.auxiliary(argv) == argv
    for module in ('worker_gpu', 'worker_memory'):
        real = ['env', 'PYTHONPATH=/frozen', '/frozen/python', '-c',
                'from exp.r17_integration.ops.'+module+' import admission; print(admission)']
        assert p.auxiliary(real) != real
    # Reapplying the two nested coordinator contexts is idempotent.
    wrapped = p.auxiliary(real)
    assert p.auxiliary(wrapped) == wrapped


def test_frozen_gpu2_island_admission_and_native_commands(tmp_path, monkeypatch):
    from exp.r17_integration import ops
    archives = [Path('/data/mw_finetune/eval')/name/'workers.tar' for name in
                ('ff1015_cosmos_li_04', 'ffk1015_dp_mt50_reduced_descentc1')]
    pins = [hashlib.sha256(a.read_bytes()).hexdigest() for a in archives]
    guard = (Path(__file__).parents[4]/'r17_integration/diagnostics/timan1_provision/gpu3_sitecustomize.py').read_bytes()
    source = Path(p.__file__).read_text()
    monkeypatch.setattr(p, 'SOURCE', source, raising=False)
    read_bytes = Path.read_bytes
    assert p.GPU2_CAPS == {'libero': 83, 'metaworld': 332}
    assert p.CAPS == {'libero': 166, 'metaworld': 664}  # GPU2 + GPU1
    for archive, domain in zip(archives, ('libero', 'metaworld')):
        with monkeypatch.context() as m, tarfile.open(archive) as tar:
            modules = {}
            for name in ('formal_process', 'worker_memory', 'worker_gpu', 'timan1_workers',
                         'worker_lifetime', 'formal_workers'):
                fullname = 'exp.r17_integration.ops.'+name
                member = next(x for x in tar if x.name.endswith('/ops/'+name+'.py'))
                module = types.ModuleType(fullname)
                module.__package__ = 'exp.r17_integration.ops'
                module.__file__ = str(archive)+'/'+member.name
                m.setitem(sys.modules, fullname, module)
                m.setattr(ops, name, module, raising=False)
                exec(compile(tar.extractfile(member).read(), module.__file__, 'exec'), module.__dict__)
                modules[name] = module
            g, t, f = (modules[n] for n in ('worker_gpu', 'timan1_workers', 'formal_workers'))
            m.setattr(sys, 'meta_path', list(sys.meta_path))
            m.setitem(sys.modules, 'sitecustomize', types.ModuleType('sitecustomize'))
            m.setattr(sys, 'orig_argv', ['python', '-c', 'bootstrap', p.WORKER])
            m.setattr(os, 'uname', lambda: types.SimpleNamespace(nodename='timan1'))
            m.setattr(Path, 'read_bytes', lambda path: guard if str(path) == p.GUARD+'/sitecustomize.py' else read_bytes(path))
            m.setattr(os, 'environ', p.environment(dict(os.environ, R17_GPU2_DEVICE_ISOLATION='1')))
            n = p.GPU2_CAPS[domain]  # a lease from before GPU1 keeps the GPU2-only cap
            c = dict(domain=domain, cache_study='frozen', execution_worker_nodes={'timan1': n},
                     worker_node='timan1', node_workers=n, worker_egl_device=3,
                     worker_egl_devices=[3], timan1_worker_lease={'device': 3},
                     run_name='cpu', worker_run=str(tmp_path/domain), endpoint='unused:1',
                     driver_host='unused', driver_port=1, worker_benchmark='unused')
            before = json.dumps(c)
            m.setattr(f, 'main', lambda: None)
            m.setattr(sys, 'argv', ['-c', p.WORKER, '--campaign', 'sealed-unused.json'])
            p.run('fleet')
            routed = f.json.loads(before)
            assert c['worker_egl_device'] == 3  # no sealed input mutation
            assert routed['worker_egl_devices'] == [2]
            assert routed['timan1_worker_lease']['uuid'] == p.UUID
            assert t.authorize(routed)['workers'] == n
            assert t.authorize(dict(c, execution_worker_nodes={'timan1': n+1}))['workers'] == n+1
            # Desired maximum is not an allocation; physical admission below still rejects n+1.

            xml = ('<nvidia_smi_log><gpu><uuid>'+p.UUID+'</uuid><minor_number>1</minor_number>'
                   '<pci><pci_bus_id>00000000:81:00.0</pci_bus_id></pci><fb_memory_usage>'
                   '<total>49140 MiB</total><used>10 MiB</used></fb_memory_usage></gpu></nvidia_smi_log>')
            query = Mock(return_value=types.SimpleNamespace(stdout=xml))
            m.setattr(p.subprocess, 'run', query)
            rows = g.inventory('timan1', [2])
            assert rows[0]['usable'] and rows[0]['device_minor'] == 1
            assert query.call_args.args[0][2] == p.UUID
            for devices in ([0], [1], [3], [0, 1, 2, 3]):
                query.reset_mock()
                assert g.inventory('timan1', devices) == []
                query.assert_not_called()
            for bad in (xml.replace(p.UUID, 'GPU-wrong'), xml.replace('81:00.0', 'c1:00.0'),
                        xml.replace('<minor_number>1', '<minor_number>2')):
                query.return_value.stdout = bad
                assert not g.inventory('timan1', [2])[0]['usable']
            observed = rows+[dict(rows[0], device=d) for d in (0, 1, 3)]
            result = g.admission(routed, tmp_path/domain, observed=observed)
            assert result['admitted'] and len(result['assignments']) == n
            assert {a['device'] for a in result['assignments']} == {2}
            assert {a['uuid'] for a in result['assignments']} == {p.UUID}
            assert not g.admission(dict(routed, node_workers=n+1), tmp_path/domain, observed=observed)['admitted']
            for device in (0, 1, 3):
                refused = g.admission(dict(routed, worker_egl_device=device, worker_egl_devices=[device]),
                                      tmp_path/domain, observed=observed)
                assert not refused['admitted'] and not refused['assignments']
            with m.context() as pending:
                pending.setattr(g, 'reservations', lambda *a, **kw: ({2: 1024}, []))
                assert not g.admission(routed, tmp_path/domain, observed=observed)['admitted']
            memory = modules['worker_memory'].admission(routed, tmp_path/domain,
                        memory=(500*1024**3, 450*1024**3))
            assert memory['admitted'] and memory['worker_memory_mib'] == p.SIZES[domain][1]
            assert memory['headroom_bytes'] == 50*1024**3

            _, argv = f.worker_command(routed, 0, '/unused/python')
            assert argv[:4] == ['/unused/python', '-B', '-u', '-c']
            assert argv[argv.index('--egl-device')+1] == '2'
            target = 'worker_lifetime'  # Both benchmarks recycle at episode boundaries
            assert argv[argv.index('-m')+1] == 'exp.r17_integration.ops.'+target
            m.setattr(sys, 'argv', ['-c', *argv[5:]])
            m.setattr(sys, 'orig_argv', argv)
            native = Mock()
            m.setattr(runpy, 'run_module', native)
            exec(argv[4], {})  # actual serialized child guard/bootstrap, CPU only
            native.assert_called_once_with('exp.r17_integration.ops.'+target, run_name='__main__', alter_sys=True)
            runtime_guard = sys.modules['sitecustomize']
            m.setattr(runtime_guard, 'deny', Mock(side_effect=ValueError('guard refused')))
            for device in ('0', '1', '3'):
                with m.context() as env:
                    env.setitem(os.environ, 'MUJOCO_EGL_DEVICE_ID', device)
                    with pytest.raises(ValueError, match='guard refused'):
                        runtime_guard.check()
                with m.context() as cli:
                    cli.setattr(sys, 'orig_argv', ['python', '--egl-device', device])
                    with pytest.raises(ValueError, match='guard refused'):
                        runtime_guard.check()
            if domain == 'libero':
                marker = tmp_path/'RECYCLE.json'
                marker.write_text(json.dumps(dict(schema='r17.worker_recycle.v1', pid=123, start_ticks='456',
                    completed_episodes=4, maximum_episodes=4, reason='episode_boundary')))
                assert modules['worker_lifetime'].planned_recycle(
                    dict(pid=123, start_ticks='456', recycle_receipt=str(marker)), 75)

    request = dict(argv=['/frozen/python', '-u', '-m', p.WORKER, '--campaign', '/sealed.json'],
                   env={'PYTHONPATH': p.GUARD+':/frozen'})
    command = ['python', '-m', 'formal_process', '--request', json.dumps(request)]
    rewritten = worker_command(command, 'timan1')
    inner = json.loads(rewritten[-1])['argv']
    assert inner[:4] == ['/usr/bin/python3', '-B', '-S', '-c'] and p.WORKER in inner
    assert source in inner[4] or repr(source) in inner[4]
    assert worker_command(command, 'timan107')[-1] != rewritten[-1]
    from exp.r17_integration.ops import local_workers, timan1_workers, worker_memory
    native_authorize = timan1_workers.authorize
    with p.coordinator():
        assert local_workers.counts('timan1=332,timan107=40') == {'timan1': 332, 'timan107': 40}
        with pytest.raises(ValueError):
            local_workers.counts('timan108=40')
        with pytest.raises(ValueError):
            local_workers.counts('timan1=40,timan1=40')
        for domain, n in p.CAPS.items():
            c = dict(domain=domain, cache_study='frozen', worker_node='timan1', node_workers=n,
                     execution_worker_nodes={'timan1': n})
            timan1_workers.route(c, c)
            assert worker_memory.reservation_mib(c) == p.SIZES[domain][1]
            assert c['worker_egl_devices'] == [2, 1]  # new coordinator leases add GPU1
    assert timan1_workers.authorize is native_authorize
    assert pins == [hashlib.sha256(a.read_bytes()).hexdigest() for a in archives]
