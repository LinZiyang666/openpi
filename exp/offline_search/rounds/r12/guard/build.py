"""Rebuild Opus's risk model from B-library LOEO replay, then freeze pilot fits.

The risk coefficient reference is an assertion only. Thresholds are explicitly
frozen B-dev calibration inputs. Deployed cache and Opus W/WL fits are reused
from attested library-only artifacts without modifying the running M1 pilot.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import pickle
import time
import numpy as np

from exp.offline_search.harness import api
from exp.offline_search.closed_loop.guard import R12Guard, _H2Judge, FEATURES, feature_matrix
from exp.offline_search.closed_loop.lookhalf_math import freeze_fit
from exp.offline_search.rounds.r10.data import STORE, sha
from exp.offline_search.rounds.r10.recipe.fitting import library_boundary, fit_path
from exp.offline_search.rounds.r12.lookhalf.build import CELLS

HERE = Path(__file__).resolve().parent
R12 = HERE.parent
SPEC = 'exp.offline_search.closed_loop.guard:R12Guard'
PILOT_CELLS = (('pi05', 'l10', 50, 'off'), ('pi05', 'spatial', 50, 'off'),
               ('groot', 'l10', 50, 'off'), ('groot', 'spatial', 50, 'off'),
               ('pi05', 'l10', 50, 'periodic'))


def _load(path):
    fit_path(path, STORE)
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


def rebuild_risk():
    """Fit both logistic models afresh on the two suites' size-50 B rows."""
    from exp.offline_search.rounds.r12.opus.tools.guard_lib import lib_table
    from exp.offline_search.rounds.r12.opus.tools.guard_design import fit_logit
    from exp.offline_search.rounds.r12.opus.tools.guard_table import mismatch_map
    from exp.offline_search.rounds.r12.opus.tools.analyze_replay import load
    from exp.offline_search.rounds.r12.opus.tools import common as cm
    reference = json.loads((cm.OUT/'guard_lib.json').read_text())
    thresholds = json.loads((cm.OUT/'guard_m2.json').read_text())['thetas']
    result = dict(schema='offline_search.r12_h2_fit.v1', library_only=True,
                  thresholds_provenance='frozen per-cell B-dev calibration, guard_m2.json; no pilot refit',
                  normalization='library LOEO non-step-zero median; dev replay used arm median',
                  coefficient_reference_sha256=sha(cm.OUT/'guard_lib.json'),
                  threshold_reference_sha256=sha(cm.OUT/'guard_m2.json'), models={}, cells={})
    for model in ('pi05', 'groot'):
        tables = []
        reads_all = []
        for suite in ('l10', 'spatial'):
            directory = cm.core.lib_dir(model, suite)
            with library_boundary(STORE, directory) as reads:
                table = lib_table(model, suite)
            tables.append(table)
            reads_all.extend(sorted(reads))
        merged = {k: np.concatenate([table[k] for table in tables]) for k in tables[0]}
        X = feature_matrix(merged)
        y = (merged['D'] > np.quantile(merged['D'], .75)).astype(float)
        _, coef = fit_logit(X, y)
        coef = dict(feats=list(FEATURES), **coef)
        ref = reference[f'{model}|coef']
        maximum = 0.
        for key in ('mu', 'sd', 'w'):
            maximum = max(maximum, float(np.max(np.abs(np.asarray(coef[key])-ref[key]))))
            np.testing.assert_allclose(coef[key], ref[key], rtol=1e-10, atol=1e-10)
        result['models'][model] = dict(coef=coef, coefficient_max_abs=maximum,
            training_rows=len(X), library_reads=reads_all, target='D above pooled library 75th percentile')
    for model, suite, size in CELLS:
        directory = cm.core.lib_dir(model, suite)
        with library_boundary(STORE, directory) as reads:
            B, scale = mismatch_map(model, suite)
            replay = load(model, suite, size)
        if replay is None:
            raise ValueError(f'missing library replay: {model}:{suite}:{size}')
        median = float(np.median(replay['W|d1'][replay['step'] > 0]))
        available = [(abs(int(arm.rsplit('_', 1)[1])-size), arm) for arm in thresholds
                     if arm.startswith(f'r11_dev_{model}_{suite}_')]
        arm = min(available)[1]
        files = sorted((cm.OUT/'replay').glob(f'{model}_{suite}_{size}_f*.npz'))
        result['cells'][f'{model}_{suite}_{size}'] = dict(mismatch_B=B.tolist(), mismatch_scale=scale,
            distance_median=median, **thresholds[arm], threshold_source_arm=arm,
            library_reads=sorted(reads), replay_sources=[dict(path=str(p), sha256=sha(p)) for p in files])
    (HERE/'risk_model.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def _wrap(base, kwargs, half_fit, risk):
    obj = R12Guard(**kwargs)
    constructor = vars(obj).copy()
    vars(obj).update(vars(base))
    for key in ('guard_method', 'lookhalf_method', 'knob_method', 'target', 'half_fit', 'risk_fit',
                '_guard_history', '_pending_guard', 'next_camera_mode', '_camera_mode', '_memory',
                '_lh_step', '_lh_result', 'lookhalf_diagnostics', '_execution', '_previous_look_called'):
        setattr(obj, key, constructor[key])
    obj.half_fit, obj.risk_fit = half_fit, risk
    if obj.guard_method != 'off':
        obj.inner = _H2Judge(obj.inner)
        freeze_fit(obj.risk_fit)
        freeze_fit(obj.half_fit)
    obj.reset(api.EpisodeView('r12-guard-build', '', 0, 0, 0, 0))
    obj.name = f'R12Guard_{obj.guard_method}__{base.name}'
    return obj


def build_cell(model, suite, size, knob, fit):
    """Emit off, H2 and M2 artifacts while preserving all frozen lower layers."""
    key = f'{model}_{suite}_{size}'
    suffix = key+('_periodic_ir32' if knob == 'periodic' else '')
    dest = HERE/'artifacts'/suffix
    dest.mkdir(parents=True, exist_ok=True)
    source = R12/'lookhalf/artifacts'/key/'opus_m1.pkl'
    half_blob = _load(source)
    half = half_blob['method']
    if not half.fit_info.get('lookhalf_library_only'):
        raise ValueError('unattested camera fit')
    hfit = half.half_fit
    hfit = dict(metrics={k: v for k, v in hfit['metrics'].items() if k.startswith('opus')},
                heads={'opus_w': hfit['heads']['opus_w']}, scales={'opus_w': hfit['scales']['opus_w']})
    hfit['bytes'] = freeze_fit(hfit)
    if knob == 'periodic':
        base_source = R12.parent/'r11/knob/artifacts'/f'r11_{key}_periodic_ir32.pkl'
    else:
        base_source = R12.parent/'r10/recipe/artifacts'/f'r10_recipe_{key}.pkl'
    blob = _load(base_source)
    base = blob['method']
    if not base.fit_info['library_only'] or not np.array_equal(base.row_subset, half.row_subset):
        raise ValueError('guard/cache library selection differs')
    kwargs = {k: v for k, v in blob['kwargs'].items() if k in ('library', 'episode_subset')}
    kwargs.update(knob_method=knob)
    if knob == 'periodic':
        kwargs['target'] = .32
    risk = dict(fit['cells'][key], coef=fit['models'][model]['coef'])
    risk['mismatch_B'] = np.asarray(risk['mismatch_B'], np.float64)
    artifacts = []
    for method in ('off', 'h2', 'm2'):
        kw = dict(kwargs, method=method)
        obj = _wrap(copy.deepcopy(base), kw, None if method == 'off' else hfit,
                    None if method == 'off' else copy.deepcopy(risk))
        obj.fit_info = dict(obj.fit_info, guard_library_only=True, guard_fit_model=model,
            guard_coefficient_max_abs=fit['models'][model]['coefficient_max_abs'],
            guard_thresholds_provenance=fit['thresholds_provenance'],
            guard_normalization=fit['normalization'], guard_library_reads=risk['library_reads'],
            full_fit=str(base_source), full_fit_sha256=sha(base_source),
            camera_fit=str(source), camera_fit_sha256=sha(source), risk_fit_sha256=sha(HERE/'risk_model.json'))
        api.check_method_attrs(obj)
        path = dest/f'{method}.pkl'
        with path.open('wb') as stream:
            pickle.dump(dict(spec=SPEC, kwargs=kw, cell=f'{model}_{suite}_cache', method=obj,
                             registered={}, fit_s=0.), stream, protocol=4)
        artifacts.append(dict(method=method, artifact=str(path), sha256=sha(path), bytes=path.stat().st_size))
    (dest/'build.json').write_text(json.dumps(dict(artifacts=artifacts, fit_info=obj.fit_info), indent=2)+'\n')
    print(json.dumps(dict(cell=suffix, methods=3)), flush=True)


def fit_wrapper(recipe, ctx):
    """Require a prebuilt, attested frozen risk artifact for online fitting."""
    if recipe.guard_method == 'off':
        if recipe.knob_method == 'off':
            from exp.offline_search.rounds.r10.recipe.fitting import fit_recipe
            fit_recipe(recipe, ctx)
        else:
            from exp.offline_search.rounds.r11.knob.fitting import fit_knob
            fit_knob(recipe, ctx)
        return
    # Fitting a risk model pools both suites and is intentionally an explicit
    # coordinator CPU build, never an accidental serving-start operation.
    raise api.ContractError('R12Guard requires --os-fit-artifact from rounds.r12.guard.build')


def main():
    """Rebuild risk and emit all requested cells, without starting services."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    fit = rebuild_risk()
    for model, suite, size in CELLS:
        build_cell(model, suite, size, 'off', fit)
    build_cell('pi05', 'l10', 50, 'periodic', fit)
    (HERE/'BUILD.json').write_text(json.dumps(dict(seconds=time.monotonic()-started,
        coefficients_rebuilt=True, sources='B-library LOEO only; frozen B-dev threshold constants', launches=0), indent=2)+'\n')


if __name__ == '__main__':
    main()
