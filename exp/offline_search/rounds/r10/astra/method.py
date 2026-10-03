"""GC_dist: unchanged G + Stage 2b LOEO heads; scalar distance-dependent strength."""
from __future__ import annotations
import json
import numpy as np

from exp.offline_search.rounds.r10.method import SizeController
from exp.offline_search.rounds.r10.data import assert_fit_input, sha
from exp.offline_search.rounds.r09.recipe.recipe import RecipeCorrectedBase


def multiplier(ratio, step, rule):
    if int(step) == 0 or not np.isfinite(ratio):
        return 0.
    return float(2 * rule['peak'] * np.clip(
        (rule['cutoff'] - ratio) / (rule['cutoff'] - rule['plateau']), 0., 1.))


class DistanceCorrectedBase(RecipeCorrectedBase):
    family = 'r10_distance_corrected_cache'

    def _correction(self, q, action):
        # os_synth and standalone query have already installed the exact retrieved anchor.
        # This extra _dist call is stateless and uses the unchanged deployed metric.
        # Geometric distance stays meaningful for the rare regime-1 continuity ranking.
        if int(q.step) == 0:
            return np.zeros((10, self.chans), np.float32)
        T, _, _, _, _, _, _, d, _, _, _ = self._dist(q)
        anchor = self._anchor
        if anchor is None:
            return np.zeros((10, self.chans), np.float32)
        positions = np.searchsorted(T.rows, anchor['rows'])
        if np.any(positions >= len(T.rows)) or not np.array_equal(T.rows[positions], anchor['rows']):
            raise ValueError('distance corrector anchor is not in the query task')
        distance = float(np.min(d[positions]))
        factor = multiplier(distance / self.distance_scale, q.step, self.distance_rule)
        if factor == 0:
            return np.zeros((10, self.chans), np.float32)
        return super()._correction(q, action) * np.float32(factor)


class DistanceController(SizeController):
    family = 'r10_distance'

    def __init__(self, size=500, variant='GC_dist', base_kwargs=None, guard_kwargs=None,
                 source_fit='', head_path='', calibration_path=''):
        if variant != 'GC_dist':
            raise ValueError('DistanceController supports only GC_dist')
        super().__init__(size, 'GC_loeo', base_kwargs, guard_kwargs, source_fit, head_path)
        self.variant = variant
        self.calibration_path = str(calibration_path)
        self.name = f'R10_GC_dist_{size}'

    def fit(self, lib, ctx):
        calibration = json.loads(assert_fit_input(self.calibration_path).read_text())
        if calibration['fit_pool'] != 'B' or calibration['source_fit_sha256'] != sha(self.source_fit):
            raise ValueError('distance calibration / unchanged G provenance mismatch')
        if not np.isfinite(calibration['scale']) or calibration['scale'] <= 0:
            raise ValueError('invalid B distance scale')
        rule = calibration['rule']
        if set(rule) != {'peak', 'plateau', 'cutoff'} or not 0 < rule['peak'] <= .5 or not 0 <= rule['plateau'] < rule['cutoff']:
            raise ValueError('invalid monotone correction-strength rule')
        # Reuse sol's fit composition and all its provenance/row/head checks verbatim.
        self.variant = 'GC_loeo'
        try:
            super().fit(lib, ctx)
        finally:
            self.variant = 'GC_dist'
        self.inner.base.__class__ = DistanceCorrectedBase
        self.inner.base.distance_scale = float(calibration['scale'])
        self.inner.base.distance_rule = dict(rule)
        self.fit_info.update(distance_revision=calibration['revision'],
            distance_calibration_sha256=sha(self.calibration_path), distance_scale=calibration['scale'],
            distance_rule=rule, stage2b_head_unchanged=True, guard_unchanged=True)
