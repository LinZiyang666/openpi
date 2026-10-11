"""P2's real upstream/native π probe plus exact suffix and P1 dispatch checks."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job', required=True)
    p.add_argument('--run', required=True)
    a = p.parse_args()
    job, run = json.loads(Path(a.job).read_text()), Path(a.run)
    from .assets import write_json
    report = dict(status='PARTIAL', synthetic_input=True, quality_claim=False, checks=[])
    write_json(run / 'native.json', report)
    started = time.monotonic()
    try:
        import numpy as np
        import torch
        from .model_runtime import load, wire_observations
        from exp.r17_integration.common.contracts import CaptureSpec, RequestContext
        from exp.r17_integration.common.request_rng import RequestRNG
        from exp.r17_integration.common.serving import full_call
        from exp.r17_integration.common.qualification import run_probe
        from exp.r17_integration.models.pi05.adapter import sampler_for_steps
        from exp.r17_integration.models.pi05.parity import (
            r16_reference, assert_native_parity, compiled_fixture_limits,
        )
        from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC, decision_seed
        adapter, provider = load(job)
        alternate_limits = compiled_fixture_limits(
            adapter.metadata(),
            Path(job.get('observations_dir', job['assets'])) / 'wire_observations.npz',
        )
        if time.monotonic()-started > 180:
            raise TimeoutError('startup exceeded 180 seconds')
        obs = wire_observations(job)[0]
        req = RequestContext('Q-pi-native', FORMAL_SPEC.identity, 'reach-v3', 0, 0, arm='qualification',
            attempt=1, bundle_generation='native-v1', task_id=43, explicit_seed=decision_seed('reach-v3',0,0))
        for k in (10,2,1):
            if time.monotonic()-started > 760:
                raise TimeoutError('native phase deadline')
            adapter.policy._sample_kwargs = dict(num_steps=k)
            adapter.policy._staged_inference = k == 10
            alternate = adapter.policy.infer(obs, noise=RequestRNG(req).normal((5,32)))['actions']
            baseline_normalized, baseline = r16_reference(adapter, obs, req, k)
            plain, wire = full_call(adapter, obs, req, sampler=sampler_for_steps(k))
            debug, debug_wire = full_call(adapter, obs, req, sampler=sampler_for_steps(k), capture=CaptureSpec(True,True))
            parity = assert_native_parity(
                baseline_normalized, baseline, plain, wire, debug, debug_wire,
                alternate_wire=alternate, alternate_limits=alternate_limits,
            )
            assert plain.actual_nfe == debug.actual_nfe == k
            assert wire.actions.shape == (5,4) and np.isfinite(wire.actions).all()
            prepared = adapter.prepare(obs, req)
            stage2 = adapter.run_stage2(adapter.run_stage1(prepared))
            for snapshot in debug.snapshots:
                suffix = adapter.resume(stage2,snapshot,debug.sampler,RequestRNG(req),CaptureSpec())
                from .g3 import equal_native
                assert equal_native(suffix.normalized_chunk, debug.normalized_chunk)
            report['checks'].append(dict(k=k, actual_nfe=k, **parity,
                native_grid=list(debug.sampler.native_timesteps), suffixes=len(debug.snapshots)))
            write_json(run / 'native.json', report)
        framework = run_probe(provider, [obs], steps=(10,2,1), deadline_seconds=max(1,760-(time.monotonic()-started)))
        write_json(run / 'framework.json', framework)
        report.update(status='PASS', framework=True, commit=5, model_spec=asdict(adapter.metadata()))
    except BaseException as exc:
        report.update(status='PARTIAL' if isinstance(exc, TimeoutError) else 'FAIL', error=str(exc))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        write_json(run / 'native.json', report)


if __name__ == '__main__':
    main()
