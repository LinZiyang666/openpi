"""Explicit orchestration-only overlays for a new, unlaunched formal campaign."""
import hashlib
import json
from pathlib import Path
import tarfile

from .bundle import files


def driver_files():
    return files() + ['src/openpi/conductor/driver.py', 'src/openpi/conductor/worker.py',
                      'exp/r17_integration/qualification/driver.py']


def worker_files():
    return files() + ['src/openpi/conductor/worker.py', 'exp/r17_integration/conductor/worker_entry.py'] + [
        'exp/r17_integration/ops/' + n + '.py' for n in
        ('formal_process', 'formal_workers', 'worker_lifetime', 'worker_gpu', 'worker_memory',
         'local_workers', 'timan1_workers', 'worker_installs', 'worker_disk',
         # formal_workers starts local fleets through this entry (routing only);
         # frozen 10-10 payloads predate it.
         'local_worker_entry')]


def overlay_worker(source, destination, checkout):
    from exp.r17_integration.ops.build_client_bundle import verified_members, json_bytes, add_bytes, sha256
    original, content = verified_members(source, 'PAYLOAD.json')
    for name, data in content.items():
        if hashlib.sha256(data).hexdigest() != original['files'][name]:
            raise ValueError('source worker hash mismatch: ' + name)
    additions = {name: (Path(checkout) / name).read_bytes() for name in worker_files()}
    content.update(additions)
    manifest = dict(schema='r17.client_payload.v1', python='3.11', files={
        n: hashlib.sha256(b).hexdigest() for n, b in sorted(content.items())})
    manifest['digest'] = hashlib.sha256(json_bytes(manifest)).hexdigest()
    with Path(destination).open('xb') as out, tarfile.open(fileobj=out, mode='w') as archive:
        for name, data in sorted(content.items()):add_bytes(archive, name, data)
        add_bytes(archive, 'PAYLOAD.json', json_bytes(manifest))
    unchanged = {n: pin for n, pin in original['files'].items() if n not in additions}
    assert all(manifest['files'][n] == pin for n, pin in unchanged.items())
    return dict(original_sha256=sha256(source), archive_sha256=sha256(destination),
                unchanged=unchanged, ops_files={n: manifest['files'][n] for n in additions})


def require_patch(checkout):
    # Post-apply validation checks the capability, not historical pre-patch
    # hashes. Subsequent reviewed shared-code fixes are expected.
    required = {
        'src/openpi/conductor/worker.py': 'OPENPI_WORKER_OBSERVER',
        'src/openpi/conductor/driver.py': 'OPENPI_AUTOSCALE_DRIVER_MODULE',
        'exp/r17_integration/ops/formal_eval.py': 'autoscale_1017.defaults',
    }
    for path, anchor in required.items():
        if anchor not in (Path(checkout)/path).read_text():
            raise ValueError('autoscale integration capability missing: ' + path)



def refresh_worker(campaign, directory, checkout, payload):
    """Freeze an ops-only overlay on the native payload, with an auditable pin."""
    from .node import atomic
    from exp.r17_integration.ops.build_client_bundle import verified_members, sha256
    import shutil
    directory = Path(directory)
    source = directory/'workers.tar'
    # A unique destination makes repeated preparation safe without overwriting
    # an immutable archive that an earlier preparation might reference.
    import uuid
    destination = directory/('autoscale_workers_' + uuid.uuid4().hex + '.tar')
    proof = overlay_worker(source, destination, checkout)
    shutil.copyfile(destination, source)
    manifest, _ = verified_members(source, 'PAYLOAD.json')
    campaign['autoscale_overlay'] = True
    atomic(directory/'AUTOSCALE_RELEASE.json', dict(proof, maxima=campaign['autoscale_node_maxima']))
    atomic(directory/'LOCAL_WORKER_ORCHESTRATION.json', dict(proof, schema='r17.autoscale_worker_ops.v1'))
    return dict(archive_sha256=sha256(source), manifest=manifest)
