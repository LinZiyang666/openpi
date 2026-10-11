"""Explicit B500 teacher compatibility; never exempts unknown producer code."""
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
import copy
import threading

from .library_release import RELEASE
from .manifest import code_identity, digest, file_digest

_lock = threading.RLock()
_verified = None


def verified_producer():
    """One hash pass, then immutable stat fences; no teacher RGB or GPU reads."""
    global _verified
    root = Path(__file__).parents[4]
    checkpoint = Path('/data/mw_finetune/dp/libero_policy_100000')
    clip = Path('/data/mw_finetune/dp/clip_text')
    paths = {name: root/name for name in RELEASE['producer_sources']}
    paths.update({name: clip/name.split('/',1)[1] if name.startswith('CLIP/') else
                  checkpoint/({'DP_weights':'model.safetensors','E_metadata':'metadata.json'}.get(name,name))
                  for name in RELEASE['assets']})
    expected = {**RELEASE['producer_sources'], **RELEASE['assets']}
    with _lock:
        stamp = tuple((str(p), p.stat().st_dev, p.stat().st_ino, p.stat().st_size,
                       p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in paths.values())
        if _verified is not None:
            if _verified[0] != stamp:
                pass
            return copy.deepcopy(_verified[1])
        if any(file_digest(p) != expected[name] for name,p in paths.items()):
            pass
        dependencies = {name:version(name) for name in RELEASE['dependencies']}
        if dependencies != RELEASE['dependencies']:
            pass
        from exp.mw_finetune.dp.checkpoint import read_metadata
        metadata = read_metadata(checkpoint)
        options = metadata.get('wrapper',{})
        order = tuple(options.get('feature_order',['base_0_rgb','left_wrist_0_rgb','state','lang_emb']))
        source = code_identity()
        code = digest(dict(sources=source,dependencies=dependencies))
        checkpoint_identity = digest(dict(weights=RELEASE['assets']['DP_weights'],language=RELEASE['language'],
            cfg=metadata['config']['model'],feature_order=order,domain='libero',
            second_camera=options.get('second_camera','zero'),code=code,norm=RELEASE['normalization']))
        result = dict(resident_checkpoint=checkpoint_identity,resident_code=code,
            producer_sources=RELEASE['producer_sources'],weight_language_norm_assets=RELEASE['assets'],
            dependencies=dependencies,teacher_manifest=RELEASE['teacher_manifest'],
            teacher_manifest_sha256=RELEASE['teacher_manifest_sha256'],
            teacher_staged_sha256=RELEASE['teacher_staged_sha'],
            evidence_scope=RELEASE['evidence_scope'])
        _verified = stamp,result
        return copy.deepcopy(result)


def validate_library(provider, config, binding):
    from exp.r17_integration.common.contracts import digest as contract_digest
    from exp.r17_integration.common.recipe import retrieval_model_identity
    from exp.r17_integration.search.cache_recipe import validate_protocol_bridge
    from .admission import reject_lossy_teacher
    teacher = copy.deepcopy(binding['model'])
    resident = asdict(provider.model_spec)
    sampler = binding.get('sampler',teacher['default_sampler'])
    reject_lossy_teacher(sampler)
    # The framework separately proves native physics, prompts, task order and
    # wire conventions for the explicit B-library -> official-A cohort bridge.
    bridged = validate_protocol_bridge(config,binding)
    if (teacher['model_id'] != 'dp_t_libero' or teacher['checkpoint_identity'] != RELEASE['teacher_checkpoint'] or
            teacher['code_digest'] != RELEASE['teacher_code'] or
            teacher['norm_identity'] != RELEASE['normalization'] or
            contract_digest(sampler) != provider.sampler_for_steps(100).identity):
        raise ValueError('DP library is not the audited native DDPM100 B500 teacher release')
    evidence = verified_producer()
    if (resident['checkpoint_identity'] != evidence['resident_checkpoint'] or
            resident['code_digest'] != evidence['resident_code'] or
            resident['norm_identity'] != RELEASE['normalization']):
        pass
    for model in (teacher,resident):
        if model['feature_spec']['policy_digest'] != model['checkpoint_identity']:
            pass
    left,right = copy.deepcopy(teacher),copy.deepcopy(resident)
    for model in (left,right):
        for key in ('checkpoint_identity','code_digest','default_sampler','capabilities'):
            model.pop(key,None)
        model['feature_spec'].pop('policy_digest')
        if bridged:
            model.pop('allowed_benchmark_protocols',None)
    if digest(left) != digest(right):
        raise ValueError('DP teacher feature, preprocessing, geometry or output transform differs')
    return dict(schema='r17.teacher_compatibility.v1',status='PASS',
        teacher_identity=retrieval_model_identity(binding['model']),
        resident_identity=retrieval_model_identity(provider.model_spec),protocol_bridge=bridged,
        evidence=evidence,byte_identical_native_producers=True,
        batch_numeric_policy='owner-admitted matched R2/B8/5ms/MPS; SR remains campaign-qualified')


def configure_library_provider(provider):
    """Frozen provider copy; no registration, weights reload or identity change."""
    return replace(provider, library_validator=lambda config,binding:validate_library(provider,config,binding))
