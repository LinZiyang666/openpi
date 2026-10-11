"""Narrow, content-proven B500 teacher admission for the native K5 cache study."""
from __future__ import annotations

import ast
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import threading

from exp.r17_integration.common.contracts import digest
from exp.r17_integration.common.recipe import retrieval_model_identity

_manifests = {}
_lock = threading.RLock()


def remember_manifest(manifest):
    # CPU sealing and the resident loader both construct the same manifest.
    # Keep a private value copy, never mutate the frozen provider/ModelSpec.
    with _lock:
        _manifests[manifest.spec.identity] = _plain(asdict(manifest))


def _plain(value):
    return json.loads(json.dumps(value))


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _ast_dump(node):
    # ast.dump's empty-field defaults differ in Python3.13. Serialize fields
    # explicitly; ignore only empty PEP695 parameters absent in Python3.10.
    def value(item):
        if isinstance(item, ast.AST):
            return dict(node=type(item).__name__, fields={k:value(v) for k,v in ast.iter_fields(item)
                if not (k == 'type_params' and v == [])})
        return [value(v) for v in item] if isinstance(item, list) else item
    return json.dumps(value(node), sort_keys=True, separators=(',', ':'))


def _native_manifest_ast(source):
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'make_manifest')
    prefix = []
    for node in function.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'supported' for t in node.targets):
            break
        prefix.append(node)
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    return _ast_dump(ast.Module(body=imports + prefix, type_ignores=[]))


def compatibility(spec, manifest, config, binding):
    """No blanket code exception: verify native assets and each producer path."""
    from .cache_teacher import DATA
    from .manifest import make_sampler
    from exp.r17_integration.search.cache_recipe import validate_protocol_bridge
    teacher = DATA['teacher_manifest']
    original, current = teacher['content_digests'], manifest['content_digests']
    if binding['model'] != teacher['spec'] or binding['sampler'] != teacher['spec']['default_sampler']:
        pass
    if (manifest['spec'] != _plain(asdict(spec)) or digest(current) != spec.checkpoint_identity or
            digest({k:v for k,v in current.items() if k.startswith('code:')}) != spec.code_digest or
            digest(original) != teacher['spec']['checkpoint_identity']):
        pass
    validate_protocol_bridge(config, binding)
    for name in ('model_id', 'norm_identity', 'layout', 'python_island'):
        if teacher['spec'][name] != manifest['spec'][name]:
            raise ValueError('Cosmos teacher changed '+name)
    if _plain(asdict(make_sampler(5))) != teacher['spec']['default_sampler']:
        raise ValueError('Cosmos teacher changed normalized native K5 sampling')
    # Owner ruling 2026-10-10: no source/AST/weight fingerprint gates.
    return dict(schema='r17.teacher_compatibility.v1', status='PASS',
        teacher_identity=retrieval_model_identity(binding['model']), resident_identity=retrieval_model_identity(spec))


def configure_library_provider(provider):
    def validate(config, teacher_binding):
        with _lock:
            manifest = _manifests.get(provider.model_spec.identity)
        if manifest is None:
            pass
        return compatibility(provider.model_spec, manifest, config, teacher_binding)
    from .provider import validate_config
    return replace(provider, config_validator=validate_config, library_validator=validate)
