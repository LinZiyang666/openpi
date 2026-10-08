"""CPU collection regressions using real archive packing and mocked transport.

The helper/collector round trip checks bounded transfers, exact bytes, SHA
verification, cleanup and deterministic refusal without touching live hosts.
"""
import contextlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from exp.offline_search.closed_loop.ops import collect as collector
from exp.offline_search.closed_loop.ops.h100 import control, node


def _write(root, files):
    for name, data in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists() or p.read_bytes() != data:
            p.write_bytes(data)


def _tree(root):
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in root.rglob('*') if p.is_file() and not p.is_symlink()}


@pytest.fixture
def transport(tmp_path, monkeypatch):
    """Use only test-owned paths and forbid every unmocked external operation."""
    root = tmp_path / 'run'
    base, worker = tmp_path / 'h100', tmp_path / 'worker'
    stage = tmp_path / 'stage'
    for p in (root, base / '.rpc', stage):
        p.mkdir(parents=True)
    worker_src = worker / 'os_cl/runs/run/sample'
    server_src = base / 'runs/run/runs/sample'
    monkeypatch.setattr(control, 'BASE', base)
    monkeypatch.setattr(control, 'island', lambda host: worker)
    monkeypatch.setattr(control, 'T107_STAGE', stage)
    monkeypatch.setattr(control, 'T108_STAGE', stage)
    monkeypatch.setenv('WORKER_HOST', 'timan107')
    monkeypatch.setenv('SERVER_HOST', 'h100')
    monkeypatch.setattr(control, 'push', lambda *a, **k: pytest.fail('no remote push'))
    monkeypatch.setattr(control.time, 'sleep', lambda _: pytest.fail('unexpected retry sleep'))
    state = SimpleNamespace(root=root, worker=worker_src, server=server_src,
                            requests=[], replies=[], pulls=[], cleaned=[], summarized=[],
                            corrupt=None, error=None, unsafe=None, cleanup_error=False)

    def rpc(host, action, spec, **kw):
        assert action == 'archive'
        assert spec['max_bytes'] == control.COLLECT_ARCHIVE_MAX_BYTES
        state.requests.append((host, spec))
        if state.unsafe is not None:
            path = Path(spec['archive'])
            with tarfile.open(path, 'w') as tf:
                tf.addfile(state.unsafe, io.BytesIO(b'bad'))
            meta = dict(archives=[dict(sha256=node.sha(path), parts=[dict(
                archive=str(path), sha256=node.sha(path), size=path.stat().st_size)])])
        else:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                node.archive(spec)
            meta = json.loads(out.getvalue())
        state.replies.append(meta)
        return json.dumps(meta)

    def invoke(cmd, **kw):
        assert cmd[:3] == ['tether', 'pull', '--force']
        assert kw['timeout'] == 590
        path = Path(cmd[3].split(':', 1)[1])
        data = path.read_bytes()
        state.pulls.append((cmd[3], data))
        if state.error is not None:
            return SimpleNamespace(returncode=75, stdout='', stderr=state.error)
        if state.corrupt == len(state.pulls):
            data += b'corrupt'
        Path(cmd[-1]).write_bytes(data)
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    def cleanup(host, cmd, **kw):
        assert cmd[:2] == ['rm', '-f']
        if state.cleanup_error:
            raise RuntimeError('cleanup transport unavailable')
        owned = {part['archive'] for meta in state.replies
                 for volume in meta['archives'] for part in volume['parts']}
        for name in cmd[2:]:
            assert name in owned
            Path(name).unlink()
            state.cleaned.append((host, name))
        return ''

    def summarize(runroot, arm, manifest=None):
        state.summarized.append((runroot, arm, manifest, _tree(root / 'runs/sample')))
        return dict(complete=1, success=1, sr=1.0)

    monkeypatch.setattr(control, 'rpc', rpc)
    monkeypatch.setattr(control.subprocess, 'run', invoke)
    monkeypatch.setattr(control, 'remote', cleanup)
    monkeypatch.setattr(collector, 'summarize', summarize)
    return state


def _round_trip(t, worker_files, server_files, manifest=None):
    _write(t.worker, worker_files)
    _write(t.server, server_files)
    expected = {f'client/{name}': data for name, data in worker_files.items()}
    expected.update(server_files)
    assert control.collect(t.root, 'sample', manifest=manifest)['complete'] == 1
    assert _tree(t.root / 'runs/sample') == expected
    assert t.summarized == [(t.root, 'sample', manifest, expected)]
    assert len(t.cleaned) == len(t.pulls)
    assert all(not Path(name).exists() for _, name in t.cleaned)
    assert all(len(data) <= control.COLLECT_ARCHIVE_MAX_BYTES for _, data in t.pulls)
    return expected


def test_collect_small_run_one_archive_per_source_exact_bytes(transport, tmp_path):
    """Small bounded archives match the legacy tar and extraction byte for byte."""
    t = transport
    worker = {'journal.jsonl': b'journal\n', 'nested/per_step.bin': bytes(range(256))}
    server = {'server_23230/log.bin': b'\x00\xfflogs', 'server_23231/nested/frame': b'image'}
    _write(t.worker, worker)
    _write(t.server, server)
    (t.server / 'ignored_link').symlink_to(t.server / 'server_23230/log.bin')
    legacy = []
    for index, (src, prefix) in enumerate(((t.worker, 'client'), (t.server, ''))):
        path = tmp_path / f'legacy_{index}.tar'
        with contextlib.redirect_stdout(io.StringIO()):
            node.archive(dict(root=str(src), archive=str(path), prefix=prefix))
        legacy.append(path.read_bytes())
    _round_trip(t, worker, server, manifest={'selected': [[0, 0]]})
    assert [data for _, data in t.pulls] == legacy
    assert [len(meta['archives']) for meta in t.replies] == [1, 1]
    assert len(t.pulls) == 2


@pytest.mark.parametrize('layout', ['several_servers', 'one_large_directory', 'long_pax_paths'])
def test_collect_oversized_member_set_bounded_tars_exact_extraction(transport, monkeypatch, layout):
    """Split both whole member sets and a single large top-level directory."""
    monkeypatch.setattr(control, 'COLLECT_ARCHIVE_MAX_BYTES', 32_000)
    server = {}
    for i in range(7):
        directory = f'server_{23230 + i}' if layout == 'several_servers' else 'server_23230'
        if layout == 'long_pax_paths':
            directory += '/' + 'long' * 40
        server[f'{directory}/frame_{i}.bin'] = bytes([i]) * 9_000
    _round_trip(transport, {'nested/journal': b'worker'}, server)
    volumes = transport.replies[1]['archives']
    assert len(volumes) > 1
    assert all(len(v['parts']) == 1 and v['size'] <= 32_000 for v in volumes)
    for _, data in transport.pulls:
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            assert all(m.isfile() for m in tf.getmembers())


def test_collect_single_oversized_file_verified_parts_exact_extraction(transport, monkeypatch):
    """Even one file over the budget transfers in bounded verified tar parts."""
    monkeypatch.setattr(control, 'COLLECT_ARCHIVE_MAX_BYTES', 32_000)
    _round_trip(transport, {'journal': b'worker'}, {
        'server_23230/a_recording.bin': bytes(range(256)) * 400,
        'server_23230/z_sidecar': b'metadata'})
    volumes = transport.replies[1]['archives']
    assert len(volumes) == 2
    assert len(volumes[0]['parts']) > 1
    assert all(p['size'] <= 32_000 for p in volumes[0]['parts'])
    joined = b''.join(data for _, data in transport.pulls[1:-1])
    with tarfile.open(fileobj=io.BytesIO(joined)) as tf:
        assert tf.extractfile('server_23230/a_recording.bin').read() == bytes(range(256)) * 400


def test_collect_empty_run_one_archive_per_source(transport):
    """Keep empty archives compatible with existing missing/empty source behavior."""
    _round_trip(transport, {}, {})
    assert len(transport.pulls) == 2
    assert all(len(data) == tarfile.RECORDSIZE for _, data in transport.pulls)


@pytest.mark.parametrize('code', [1, 69, 70, 75, 255])
@pytest.mark.parametrize('stream', ['stdout', 'stderr'])
def test_tether_too_large_never_retried(monkeypatch, code, stream):
    """Payload refusal wins over transient return codes and timeout diagnostics."""
    calls = []
    message = 'pull refused: code=too_large size=943206400; timeout'

    def invoke(cmd, **kw):
        calls.append(cmd)
        result = dict(returncode=code, stdout='', stderr='')
        result[stream] = message
        return SimpleNamespace(**result)

    monkeypatch.setattr(control.subprocess, 'run', invoke)
    monkeypatch.setattr(control.time, 'sleep', lambda _: pytest.fail('too_large must not sleep'))
    with pytest.raises(control.CommandError, match='code=too_large') as exc:
        control.run(['tether', 'pull', '--force', 'h100:logs.tar', 'logs.tar'])
    assert not exc.value.transient
    assert len(calls) == 1


def test_collect_too_large_never_retried_and_cleans_all_parts(transport, monkeypatch):
    """The outer eight-attempt collection loop also refuses immediately."""
    monkeypatch.setattr(control, 'COLLECT_ARCHIVE_MAX_BYTES', 32_000)
    _write(transport.worker, {f'data/{i}': b'x' * 9_000 for i in range(7)})
    transport.error = 'pull refused: code=too_large; timeout'
    with pytest.raises(control.CommandError, match='code=too_large'):
        control.collect(transport.root, 'sample')
    assert len(transport.pulls) == 1
    assert len(transport.cleaned) == sum(len(v['parts']) for v in transport.replies[0]['archives'])
    assert not transport.summarized


def test_collect_plain_runtime_too_large_never_retried(transport, monkeypatch):
    """Fail fast for existing transports raising plain RuntimeError as well."""
    calls = []

    def invoke(cmd, **kw):
        calls.append(cmd)
        raise RuntimeError('pull refused: code=too_large')

    monkeypatch.setattr(control, 'run', invoke)
    with pytest.raises(RuntimeError, match='code=too_large'):
        control.collect(transport.root, 'sample')
    assert len(calls) == 1 and len(transport.cleaned) == 1


@pytest.mark.parametrize('pull', [1, 2])
def test_collect_checks_each_part_sha_before_extracting(transport, monkeypatch, pull):
    """Corrupt initial and continuation parts never reach extraction or summary."""
    monkeypatch.setattr(control, 'COLLECT_ARCHIVE_MAX_BYTES', 32_000)
    _write(transport.worker, {'large.bin': b'x' * 90_000})
    transport.corrupt = pull
    with pytest.raises(RuntimeError, match='collection archive SHA mismatch'):
        control.collect(transport.root, 'sample')
    assert len(transport.pulls) == pull
    assert not (transport.root / 'runs/sample').exists()
    assert not transport.summarized
    assert len(transport.cleaned) == len(transport.replies[0]['archives'][0]['parts'])


@pytest.mark.parametrize('kind', ['absolute', 'traversal', 'symlink', 'hardlink', 'directory'])
def test_collect_split_preserves_unsafe_member_refusal(transport, kind):
    """Preserve all existing regular-file and path checks after splitting."""
    names = {'absolute': '/escape', 'traversal': '../escape'}
    info = tarfile.TarInfo(names.get(kind, 'entry'))
    info.size = 3
    if kind == 'symlink':
        info.type, info.linkname, info.size = tarfile.SYMTYPE, '/escape', 0
    elif kind == 'hardlink':
        info.type, info.linkname, info.size = tarfile.LNKTYPE, '/escape', 0
    elif kind == 'directory':
        info.type, info.size = tarfile.DIRTYPE, 0
    transport.unsafe = info
    with pytest.raises(ValueError, match='unsafe collected archive member'):
        control.collect(transport.root, 'sample')
    assert not transport.summarized
    assert len(transport.cleaned) == 1
    assert _tree(transport.root / 'runs/sample') == {}


def test_collect_cleanup_failure_preserves_too_large_error(transport):
    """A second transport failure cannot obscure the non-retryable refusal."""
    transport.error = 'pull refused: code=too_large'
    transport.cleanup_error = True
    with pytest.raises(control.CommandError, match='code=too_large'):
        control.collect(transport.root, 'sample')
    assert len(transport.pulls) == 1


def test_archive_failure_cleans_created_parts(tmp_path, monkeypatch):
    """Do not leave earlier volumes behind if packing a subsequent file fails."""
    root, dest = tmp_path / 'root', tmp_path / 'logs.tar'
    _write(root, {f'{i}.bin': b'x' * 9_000 for i in range(5)})
    original = tarfile.TarFile.addfile
    calls = []

    def addfile(tf, info, source=None):
        calls.append(info.name)
        if len(calls) == 5:
            raise OSError('read failed')
        return original(tf, info, source)

    monkeypatch.setattr(tarfile.TarFile, 'addfile', addfile)
    with pytest.raises(OSError, match='read failed'):
        node.archive(dict(root=str(root), archive=str(dest), prefix='', max_bytes=32_000))
    assert not dest.exists()
    assert not list(tmp_path.glob('logs.tar.part*'))
    assert not list(tmp_path.glob('*.tmp'))
