"""Write the segment-boundary fix into a runtime archive's source files.

Long-segment campaigns inherit frozen runtime archives that predate the fix.
Instead of patching methods at server start, rewrite the archived modules so
every launch path imports only the fixed methods. Each listed method is
replaced by the working-tree definition; every other byte is kept.
"""
import ast
import io
import tarfile
from pathlib import Path

from .runtime import source

REPO = Path(__file__).resolve().parents[5]


def _module_path(module):
    return module.replace('.', '/') + '.py'


def _span(lines, node):
    start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
    return start, node.end_lineno


def _methods(tree, name):
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name]
    if len(cls) != 1:
        raise ValueError(f'class {name} not found exactly once')
    return cls[0], {n.name: n for n in cls[0].body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def fix_text(frozen, fixed, name, methods):
    """Return frozen module text with `methods` of class `name` taken from `fixed`."""
    new_lines, old_lines = fixed.splitlines(True), frozen.splitlines(True)
    _, new = _methods(ast.parse(fixed), name)
    cls, old = _methods(ast.parse(frozen), name)
    missing = [m for m in methods if m not in new]
    if missing:
        raise ValueError(f'{name}: fixed source lacks {missing}')
    edits = []
    for method in methods:
        a, b = _span(new_lines, new[method])
        body = ''.join(new_lines[a:b])
        if method in old:
            edits.append((*_span(old_lines, old[method]), body))
        else:
            edits.append((cls.end_lineno, cls.end_lineno, '\n' + body))
    for a, b, body in sorted(edits, reverse=True):
        old_lines[a:b] = [body]
    out = ''.join(old_lines)
    _, check = _methods(ast.parse(out), name)
    for method in methods:
        if ast.dump(check[method]) != ast.dump(new[method]):
            raise ValueError(f'{name}.{method} differs from fixed source after rewrite')
    return out


def patched_files(read):
    """Map archive member -> fixed text, given read(member) -> frozen text."""
    result = {}
    for item in source():
        member = _module_path(item['module'])
        fixed = (REPO / member).read_text()
        frozen = read(member)
        text = fix_text(frozen, fixed, item['name'], item['methods'])
        compile(text, member, 'exec')
        result[member] = text
    return result


def apply(archive):
    """Rewrite `archive` in place; return {member: (old_sha, new_sha)}."""
    import hashlib
    archive = Path(archive)
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        data = {m.name: tar.extractfile(m).read() for m in members}
    fixed = patched_files(lambda member: data[member].decode())
    receipt = {}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as out:
        for m in members:
            body = data[m.name]
            if m.name in fixed:
                new = fixed[m.name].encode()
                receipt[m.name] = (hashlib.sha256(body).hexdigest(), hashlib.sha256(new).hexdigest())
                body = new
                m.size = len(body)
            out.addfile(m, io.BytesIO(body))
    tmp = archive.with_suffix('.tar.fixing')
    tmp.write_bytes(buffer.getvalue())
    tmp.replace(archive)
    return receipt


def stage(launch, campaign, root):
    """filter_failed.stage path: run the fixed copy of the inherited archive.

    The campaign's runtime.tar (the frozen source release) is rewritten, then
    installed as its own content-addressed release locally and, for H100
    servers, on H100; the campaign points to that release. Returns its sha.
    """
    import json
    from exp.r17_integration.ops.formal_prestage import verify_code
    from exp.r17_integration.ops.formal_process import write_json
    from exp.r17_integration.ops.immutable_release import archive_manifest, identity, install, sha
    root = Path(root)
    archive = root/'runtime.tar'
    members = apply(archive)
    document = archive_manifest(archive)
    pin, archive_pin = identity(document), sha(archive)
    release = '/data/mw_finetune/qualification/formal_runtime_'+pin
    install(archive, release, archive_pin, pin)
    if campaign['server_node'] == 'h100':
        directory = '/tmp/r17_formal/'+campaign['run_name']+'/server'
        launch.remote('h100', ['mkdir', '-p', directory], replay_safe=True)
        present = json.loads(launch.remote('h100', [launch.host_python(campaign), '-c',
            verify_code(release, 'RUNTIME_RELEASE.json', pin)], replay_safe=True))['present']
        if not present:
            launch.transfer(['rsync', 'push', str(archive), 'h100:'+directory+'/runtime.tar'])
            launch.transfer(['rsync', 'push', str(REPO/'exp/r17_integration/ops/immutable_release.py'),
                             'h100:'+directory+'/release_install.py'])
            launch.remote('h100', [launch.host_python(campaign), directory+'/release_install.py',
                '--root', release, '--identity', pin, '--archive', directory+'/runtime.tar',
                '--archive-sha', archive_pin])
    campaign.update(runtime=release, runtime_identity=pin)
    write_json(root/'RUNTIME_RELEASE.json', dict(node=campaign['server_node'], archive_sha256=archive_pin,
        identity=pin, root=release, reused=False, files=len(document['files'])))
    write_json(root/'LONGSEG_RUNTIME_FIX.json', dict(members=members, runtime=release, identity=pin,
                                                    archive_sha256=archive_pin))
    return archive_pin
