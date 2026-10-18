"""Explicit source closure for newly prepared releases; no running release edits."""
from contextlib import contextmanager
from pathlib import Path

PREFIX = 'exp/offline_search/rounds/r17/autoscale_1017/'
MODULES = ('__init__', 'policy', 'node', 'worker', 'driver', 'fleet', 'legacy', 'bundle', 'bootstrap', 'kernel_evidence', 'defaults', 'fairness', 'switch')


def files():
    return [PREFIX + name + '.py' for name in MODULES] + [PREFIX + 'ENABLED',
        'exp/offline_search/rounds/r17/filter_failed/timan1_gpu2.py']


@contextmanager
def runtime_sources(campaign):
    if not campaign.get('autoscale'):
        yield
        return
    from exp.r17_integration.qualification import deploy_cpu
    import tarfile
    native = deploy_cpu.runtime_archive
    def build(root, archive):
        names = native(root, archive)
        with tarfile.open(archive, 'a') as out:
            for name in files():
                out.add(Path(root) / name, arcname=name, recursive=False)
        return names | {Path(root) / name for name in files()}
    deploy_cpu.runtime_archive = build
    try:
        yield
    finally:
        deploy_cpu.runtime_archive = native
