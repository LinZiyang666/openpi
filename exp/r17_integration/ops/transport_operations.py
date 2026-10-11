"""Idempotent by-value control helpers for already frozen remote runtimes."""
from __future__ import annotations

import inspect
import json


def cleanup_archive(request):
    from pathlib import Path
    import json
    from exp.r17_integration.qualification.archive_stage import cleanup_stage, stage_identity
    # Formal campaigns can exceed the qualification helper's 512-MiB bound.
    # This process-local orchestration override never edits a frozen release.
    if 'max_bytes' in request:
        if request['max_bytes'] != 8 * 1024**3:
            raise ValueError('invalid formal evidence size bound')
    path, key, sha = request['archive'], request['key'], request['sha256']
    if stage_identity(path,key,'workers') != sha:
        raise ValueError('cleanup stage ownership mismatch')
    marker = json.loads(Path(path).with_suffix('.stage.json').read_text())
    if (marker['archive'] != path or marker['archive_sha256'] != sha or
            marker['run_key'] != key or marker['kind'] != 'workers'):
        raise ValueError('cleanup stage receipt mismatch')
    if not Path(path).exists():
        if marker.get('stage_removed') is not True or marker.get('verified_pull_sha256') != sha:
            raise ValueError('cleanup missing stage lacks verified closure receipt')
        return dict(status='PASS', **marker)
    from exp.r17_integration.qualification import archive_stage
    previous = archive_stage.MAX_BYTES
    try:
        archive_stage.MAX_BYTES = request.get('max_bytes',previous)
        return cleanup_stage(path,key,'workers',sha)
    finally:
        archive_stage.MAX_BYTES = previous


def cleanup(node, archive, key, sha256, *, python, env, remote, max_bytes=None):
    request = dict(archive=archive,key=key,sha256=sha256)
    if max_bytes is not None:
        request['max_bytes'] = max_bytes
    code = inspect.getsource(cleanup_archive)+'\nimport json; print(json.dumps(cleanup_archive('+repr(request)+')))'
    return json.loads(remote(node,['env',*env,python,'-c',code],replay_safe=True))
