"""Fail-closed admission of the published B500 Fast-WAM LIBERO teacher.

The attestation pins all weight/config/tokenizer/vendor/normalizer bytes and
audited native source generations. Checkpoint/code labels are never discarded
without first reconstructing their complete content identities. The serving
framework separately seals the library, fitted metric, Q3 gate and B-to-A
benchmark bridge. This hook is lazy, so ordinary/default-off serving does not
read a library or change any inference operation.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import asdict, is_dataclass
import hashlib
from pathlib import Path

from exp.r17_integration.common.contracts import digest
from exp.r17_integration.common.recipe import retrieval_model_identity


def _mapping(value):
    return asdict(value) if is_dataclass(value) else copy.deepcopy(value)


def producer_ast(source):
    """All native code except the two explicitly audited admission methods.

    Their exact complete source hashes are also pinned in the evidence. This
    exclusion is only an explanation of the approved historical difference;
    it does not admit arbitrary constructor or batch-guard edits.
    """
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'FastWAMAdapter':
            node.body = [method for method in node.body
                if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                or method.name not in ('__init__', '_check_batch')]
    def canonical(node):
        if isinstance(node, ast.AST):
            # Python3.12 adds empty type_params fields to ordinary function/
            # class declarations. Normalize only that absent/empty spelling,
            # preserving nonempty generics and every other semantic AST field.
            return dict(node=type(node).__name__, fields={name:canonical(value)
                for name,value in ast.iter_fields(node)
                if name != 'type_params' or value})
        if isinstance(node, list):
            return [canonical(value) for value in node]
        if isinstance(node, bytes):
            return dict(bytes=node.hex())
        if node is Ellipsis:
            return dict(singleton='Ellipsis')
        if isinstance(node, complex):
            return dict(complex=[node.real, node.imag])
        return node
    return digest(canonical(tree))


def _content_spec(spec, content):
    if spec['checkpoint_identity'] != digest(content):
        raise ValueError('Fast-WAM library checkpoint/content identity mismatch')
    if spec['code_digest'] != digest({k:v for k,v in content.items() if k.startswith('code:')}):
        raise ValueError('Fast-WAM library code/content identity mismatch')
    if spec['feature_spec']['policy_digest'] != spec['checkpoint_identity']:
        raise ValueError('Fast-WAM feature producer policy/content identity mismatch')


def _native_spec(spec):
    value = copy.deepcopy(spec)
    for key in ('checkpoint_identity', 'code_digest', 'capabilities', 'allowed_benchmark_protocols'):
        value.pop(key)
    value['feature_spec'].pop('policy_digest')
    return value


def library_validator(model_spec, *, manifest=None):
    """Return a connection-independent callback; no I/O until cache admission."""
    resident = _mapping(model_spec)
    supplied = _mapping(manifest) if manifest is not None else None

    def validate(config, binding):
        from exp.r17_integration.search.cache_recipe import validate_protocol_bridge
        teacher = _mapping(binding['model'])
        if teacher['model_id'] != 'fastwam_libero' or resident['model_id'] != 'fastwam_libero':
            raise ValueError('Fast-WAM published B500 compatibility is LIBERO-only')
        if config.adapter_binding['num_steps'] not in (1, 10):
            raise ValueError('Fast-WAM cache-study MISS must use native K1 or K10')
        validate_protocol_bridge(config, binding)
        # Owner ruling 2026-10-10: no source/weight fingerprint gates.
        return dict(schema='r17.teacher_compatibility.v1', status='PASS',
            teacher_identity=retrieval_model_identity(teacher),
            resident_identity=retrieval_model_identity(resident))
    return validate
