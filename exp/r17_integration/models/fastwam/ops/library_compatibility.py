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
        from .library_evidence import EVIDENCE
        from exp.r17_integration.search.cache_recipe import validate_protocol_bridge
        record = copy.deepcopy(EVIDENCE)
        record_digest = record.pop('digest')
        if digest(record) != record_digest or record['schema'] != 'r17.fastwam.library_sources.v1':
            raise ValueError('Fast-WAM library source attestation digest mismatch')
        teacher = _mapping(binding['model'])
        if teacher['model_id'] != 'fastwam_libero' or resident['model_id'] != 'fastwam_libero':
            raise ValueError('Fast-WAM published B500 compatibility is LIBERO-only')
        teacher_content = record['teacher_content_digests']
        _content_spec(teacher, teacher_content)
        if digest(_native_spec(teacher)) != digest(_native_spec(record['teacher_model_spec'])):
            raise ValueError('Fast-WAM library teacher differs from published sealed model semantics')
        approved = [item for item in record['resident_content_digests']
                    if digest(item) == resident['checkpoint_identity']]
        if len(approved) != 1:
            raise ValueError('Fast-WAM resident source/weights generation is not audited for B500')
        content = approved[0]
        _content_spec(resident, content)
        if supplied is not None and (supplied['spec'] != resident or supplied['content_digests'] != content):
            raise ValueError('Fast-WAM live manifest differs from audited resident content')
        # Every non-R17 input, including vendor/loader code, must be identical.
        native = lambda pins: {k:v for k,v in pins.items() if not k.startswith('code:r17:')}
        if native(teacher_content) != native(content):
            raise ValueError('Fast-WAM weights/norm/config/tokenizer/vendor producer bytes changed')
        for filename in ('key_builder.py', 'wire_policy.py', 'loader.py', 'manifest.py', 'protocols.py'):
            key = 'code:r17:'+filename
            if teacher_content[key] != content[key]:
                raise ValueError('Fast-WAM key/preprocessing/output-transform source changed: '+filename)
        if digest(_native_spec(teacher)) != digest(_native_spec(resident)):
            raise ValueError('Fast-WAM library norm/preprocessing/features/action transform/geometry mismatch')
        if resident['default_sampler']['num_steps'] != 10 or digest(binding['sampler']) != digest(teacher['default_sampler']):
            raise ValueError('Fast-WAM B500 teacher must be native K10')
        if config.adapter_binding['num_steps'] != 10:
            raise ValueError('Fast-WAM cache-study stall MISS must use native K10')
        if (binding['benchmark_protocol'] not in teacher['allowed_benchmark_protocols'] or
                config.adapter_binding['benchmark_protocol'] not in resident['allowed_benchmark_protocols']):
            raise ValueError('Fast-WAM library/serving benchmark is not allowlisted')
        validate_protocol_bridge(config, binding)
        # Detect unsealed local edits rather than approving an old spec against
        # a different executing key producer. Ancillary files are exact pins too.
        root = Path(__file__).parents[1]
        current = {'code:r17:'+p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in root.glob('*.py')}
        if current != record['executing_r17_digests']:
            raise ValueError('Fast-WAM executing native source differs from reviewed B500 attestation')
        if producer_ast((root/'staged.py').read_text()) != record['native_producer_ast_sha256']:
            raise ValueError('Fast-WAM executing VAE/key/prefill/head producer changed')
        return dict(schema='r17.teacher_compatibility.v1', status='PASS',
            teacher_identity=retrieval_model_identity(teacher),
            resident_identity=retrieval_model_identity(resident),
            evidence=dict(schema=record['schema'], source_receipt_digest=record_digest,
                source_receipt='exp/r17_integration/models/fastwam/ops/library_evidence.py',
                teacher_checkpoint=digest(teacher_content), resident_checkpoint=digest(content),
                teacher_staged_sha256=teacher_content['code:r17:staged.py'],
                resident_staged_sha256=content['code:r17:staged.py'],
                native_producer_ast_sha256=record['native_producer_ast_sha256'],
                weights_norm_vendor_unchanged=True, native_sources=copy.deepcopy(content),
                qualification_scope='content/source equivalence; GPU cache smoke pending'))
    return validate
