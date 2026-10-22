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
# Whole working-tree files written into (or added to) inherited releases, which predate them.
# formal_server.py: B1 requests run on one model thread per replica (pi0.5/GR00T campaigns).
# With `stage_batching` also cross-connection stage batching: formal_server stage_batching,
# common.batch_runtime, the GR00T/pi0.5 stage hooks, GR00T S1/S2 batches and its transform lock.
SERVER_FILES = ('exp/r17_integration/ops/formal_server.py',)
BATCHING_FILES = SERVER_FILES + ('exp/r17_integration/common/batch_runtime.py',
                  'exp/r17_integration/models/groot/provider.py', 'exp/r17_integration/models/groot/batcher.py',
                  'exp/r17_integration/models/pi05/adapter.py', 'src/openpi/cache/groot/batcher.py',
                  'src/openpi/cache/groot/staged.py')
# fastwam_text_table: the resident is built without umT5-xxl and reads prompt encodings from the table.
# formal_batching.py: older frozen runtimes still raise on the batch implementation identity, which the
# text-table server changes (fa_ls_v08_1 10-22 10:43); the working copy only records it.
FASTWAM_TEXT_FILES = ('exp/r17_integration/models/fastwam/formal_batch_server.py',
                      'exp/r17_integration/models/fastwam/formal_batching.py',
                      'exp/r17_integration/models/fastwam/loader.py',
                      'exp/r17_integration/models/fastwam/text_table.py',
                      'exp/r17_integration/qualification/model_runtime.py',
                      'exp/r17_integration/common/provider.py',
                      'exp/r17_integration/models/fastwam/ops/library_compatibility.py')
# Older frozen runtimes still raise on identity checks the owner removed (10-10): the adapter_binding
# checkpoint/norm/layout/feature identities (the checkpoint identity digests the fastwam code, so any overlay
# or new release breaks it; fa_ls_v08_1c1/c2 10-22) and the batch implementation identity.
# Without the table nothing under models/fastwam/*.py is overlaid: those files are digested into the checkpoint
# identity, so replacing one made load_adapter refuse the sealed content (fa_ls_v05_1 10-22 11:20).
RELAXED_FILES = ('exp/r17_integration/common/provider.py',
                 'exp/r17_integration/models/fastwam/ops/library_compatibility.py')


def overlay_files(campaign):
    if campaign.get('stage_batching'):
        return BATCHING_FILES
    if campaign.get('fastwam_text_table'):
        return FASTWAM_TEXT_FILES
    if campaign.get('model') == 'fastwam':
        return RELAXED_FILES
    return SERVER_FILES if campaign.get('model') in ('groot', 'pi05') else ()


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


def apply(archive, files=()):
    """Rewrite `archive` in place; return {member: (old_sha, new_sha)}.

    `files` are working-tree paths written whole (added when the archive lacks them),
    after the method rewrites, so they carry their own complete definitions.
    """
    import copy
    import hashlib
    archive = Path(archive)
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        data = {m.name: tar.extractfile(m).read() for m in members}
    fixed = patched_files(lambda member: data[member].decode())
    fixed.update({name: (REPO / name).read_text() for name in files})
    receipt = {}
    buffer = io.BytesIO()
    names = {m.name for m in members}
    template = next(m for m in members if m.isfile())
    for name in files:
        if name not in names:
            m = copy.copy(template)
            m.name = name
            members.append(m)
            data[name] = b''
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
    members = apply(archive, overlay_files(campaign))
    document = archive_manifest(archive)
    pin, archive_pin = identity(document), sha(archive)
    release = '/data/mw_finetune/qualification/formal_runtime_'+pin
    # Mark the rewrite before any install/transfer, so a retry after a failed
    # transfer recognizes runtime.tar as the fixed copy (apply is idempotent).
    write_json(root/'LONGSEG_RUNTIME_FIX.json', dict(members=members, runtime=release, identity=pin,
                                                    archive_sha256=archive_pin, installed=False))
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
