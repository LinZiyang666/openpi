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
    old_feature, new_feature = dict(teacher['spec']['feature_spec']), dict(manifest['spec']['feature_spec'])
    if (old_feature.pop('policy_digest') != teacher['spec']['checkpoint_identity'] or
            new_feature.pop('policy_digest') != spec.checkpoint_identity or old_feature != new_feature):
        raise ValueError('Cosmos teacher changed VAE keys/preprocessing/state/normalization')
    if _plain(asdict(make_sampler(5))) != teacher['spec']['default_sampler']:
        raise ValueError('Cosmos teacher changed normalized native K5 sampling')
    # All official source, DiT/VAE weights, stats, prompts, release config and
    # unchanged R17 modules remain byte-identical. Only enumerated code paths
    # below can differ, and producer changes get a separate content proof.
    changed = {'code:r17:manifest.py', 'code:r17:provider.py', 'code:r17:serve.py',
               'code:r17:qualify.py', 'code:r17:staged.py', 'code:r17:loader.py'}
    additions = {'code:r17:cache_teacher.py', 'code:r17:library_admission.py',
                 'code:r17:cache_probe.py', 'code:r17:recover_baseline.py'}
    if set(original) - set(current) or set(current) - set(original) - additions:
        pass
    for name, pin in original.items():
        if name not in changed and current.get(name) != pin:
            pass
    root = Path(__file__).parent
    staged = (root/'staged.py').read_bytes()
    new_line = b'if calls != sampler.config.get("expected_nfe", sampler.num_steps):'
    if staged.count(new_line) != 1 or _sha(staged.replace(new_line, b'if calls != sampler.num_steps:')) != original['code:r17:staged.py']:
        raise ValueError('Cosmos native stages changed beyond reviewed K2 accounting')
    historical = DATA['historical_manifest_source']
    source = (root/'manifest.py').read_text()
    sampler = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'make_sampler')
    if (_sha(historical.encode()) != original['code:r17:manifest.py'] or
            _native_manifest_ast(historical) != _native_manifest_ast(source) or
            _sha(_ast_dump(sampler).encode()) != DATA['reviewed_sampler_ast_sha256']):
        raise ValueError('Cosmos manifest changed native normalization/layout/features/sampler')
    # Loader is still exactly the teacher loader. Extra recipe hooks are
    # metadata/provider code and do not alter its native model construction.
    if current['code:r17:loader.py'] != DATA['reviewed_loader_sha256']:
        raise ValueError('Cosmos native loader changed')
    for name in ('staged.py', 'manifest.py', 'loader.py', 'key_builder.py', 'wire_policy.py', 'protocols.py'):
        if _sha((root/name).read_bytes()) != current['code:r17:'+name]:
            pass
    return dict(schema='r17.teacher_compatibility.v1', status='PASS',
        teacher_identity=retrieval_model_identity(binding['model']), resident_identity=retrieval_model_identity(spec),
        evidence=dict(source=DATA['source_evidence'], exact_assets={k:v for k,v in original.items()
            if not k.startswith('code:r17:')}, normalization=spec.norm_identity,
            layout=spec.layout.identity, features_except_release_digest=old_feature,
            normalized_action_teacher=digest(binding['sampler']), output_transform=spec.layout.output_transform_id,
            producer_proof='unchanged native files; manifest native AST exact; staged only K2 NFE assertion',
            reviewed_changes=sorted(changed), added_nonproducer_modules=sorted(set(current)-set(original))))


def configure_library_provider(provider):
    def validate(config, teacher_binding):
        with _lock:
            manifest = _manifests.get(provider.model_spec.identity)
        if manifest is None:
            pass
        return compatibility(provider.model_spec, manifest, config, teacher_binding)
    from .provider import validate_config
    return replace(provider, config_validator=validate_config, library_validator=validate)
