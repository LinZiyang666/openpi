"""Read-only DP CPU diagnosis through the frozen server's unchanged loader.

Run this file with the execution host's Python, frozen PYTHONPATH and cwd.
The trace observes the failing guard's inputs; it never changes the guard.
"""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback


def diagnose(root):
    from exp.r17_integration.qualification.model_runtime import load
    from exp.r17_integration.models.dp import loader
    import torch
    campaign = json.loads((root/'campaign.json').read_text())
    if campaign['model'] != 'dp':
        raise ValueError('DP diagnosis only')
    sealed = json.loads((Path(campaign['assets'])/'model_manifest.json').read_text())
    load_path = Path(campaign['assets'])/'load.json'
    expected = json.loads(load_path.read_text())
    path = Path(loader.__file__)
    pin = hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(run=str(root), python=sys.executable, prefix=sys.prefix,
                  loader=str(path), loader_sha256=pin, expected=expected,
                  sealed_dependency_versions=sealed['dependency_versions'], cpu_only=True)

    def capture(adapter, policy=None):
        result.update(actual_spec=asdict(adapter.metadata()), dependency_versions=adapter.dependency_versions,
                      source_digests=adapter.source_digests, weight_identity=adapter.weight_identity,
                      language_identity=adapter.language_identity, model_config=adapter.model.cfg,
                      feature_order=list(adapter.model._policy.obs_encoder.feature_order),
                      second_camera=adapter.second_camera)
        if policy is not None:
            result['strict_load'] = policy.strict_load

    def trace(frame, event, arg):
        if frame.f_code.co_filename != str(path):
            return None
        if event == 'exception' and frame.f_locals.get('adapter') is not None:
            capture(frame.f_locals['adapter'], frame.f_locals.get('policy'))
        return trace

    sys.settrace(trace)
    try:
        adapter, _ = load(dict(campaign, benchmark_spec=str(Path(campaign['assets'])/'benchmark.json')), 'cpu')
        capture(adapter, adapter.legacy_policy)
        result['status'] = 'PASS'
    except Exception as exc:
        result.update(status='FAIL', error=repr(exc), traceback=traceback.format_exc())
    finally:
        sys.settrace(None)
    result['cuda_initialized'] = torch.cuda.is_initialized()
    result['loader_unchanged'] = hashlib.sha256(path.read_bytes()).hexdigest() == pin
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = diagnose(args.run.resolve())
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2)+'\n')
    print(json.dumps({k:result.get(k) for k in ('status', 'error', 'python', 'cuda_initialized', 'loader_unchanged')}))
