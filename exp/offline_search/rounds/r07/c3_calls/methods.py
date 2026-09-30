"""CU/CT: unchanged R6 CalibratedRescue decisions with an optional lottery tilt.

The inherited uniform parameter is updated after retrieval and before R6 draws
its coin. All stall/cooldown/LOOK/tail decisions remain in the original query.
With tilt=False this method delegates directly, including all Result extras.
"""
from __future__ import annotations

import json
from pathlib import Path

from exp.offline_search.harness import api
from exp.offline_search.rounds.r06.ideation_Q1.method_c.methods import CalibratedRescue
from .common import VERSION, sha
from .tilt import factors, stage_features


class CallController(CalibratedRescue):
    family = 'r7_calls'

    def __init__(self, rho, placement='uniform', stall_model_path=None, calibration_path=None,
                 random_seed=0, randomization_key='R6-C-v2', cooldown_scope='stall', tilt=False):
        if type(tilt) is not bool:
            raise ValueError('tilt must be bool')
        if tilt and (placement != 'uniform' or not stall_model_path):
            raise ValueError('CT requires uniform placement and a calibrated stall model')
        super().__init__(rho, placement, stall_model_path, calibration_path,
                         random_seed, randomization_key, cooldown_scope)
        self.tilt = tilt
        self.cache_extension = None
        self.name = ('CT' if tilt else 'CU') + f'_rho{rho:g}'

    def _load(self, cell, *, allow_dryrun=False):
        cal = json.loads(Path(self.calibration_path).read_text())
        meta = cal.get('r7_calls')
        if meta and (meta['version'] != VERSION or bool(meta['tilt']) != self.tilt):
            raise ValueError('R7 calibration/controller tilt mismatch')
        if self.tilt and not meta:
            raise ValueError('CT requires its re-solved stage calibration')
        super()._load(cell, allow_dryrun=allow_dryrun)
        self.lambda_ = self.parameter
        self.stage_table = None
        if self.tilt:
            from exp.offline_search.rounds.r07.stages.stages import StageTable
            path = Path(self.calibration_path).parent / meta['stage_path']
            if sha(path) != meta['stage_sha256']:
                raise ValueError('stage artifact content differs from CT calibration')
            self.stage_table = StageTable.load(path)
            if self.stage_table.fingerprint != meta['stage_fingerprint']:
                raise ValueError('stage table differs from CT budget calibration')
            if self.stage_table.retrieval_fingerprint != self.bank['retrieval_fingerprint']:
                raise ValueError('stage calibration used a different A retrieval')
        self.fit_info.update(r7_calls_version=VERSION, r7_calls_tilt=self.tilt,
                             r7_calls_lambda=self.lambda_)

    def reset(self, episode):
        super().reset(episode)
        self._deviation_latched = False
        self._tilt_log = {}
        if hasattr(self, 'lambda_'):
            self.parameter = self.lambda_
        if self.cache_extension is not None:
            self.cache_extension.reset(episode)

    def _query_with_metric_code(self, q):
        result, code = super()._query_with_metric_code(q)
        if self.tilt:
            anchor = self.base._anchor
            feature = stage_features(self.stage_table, anchor['rows'], anchor['weights'], q.rs, q.task_id)
            weight, self._deviation_latched, entry = factors(
                feature['event_mass'], feature['h'], feature['high'], feature['h_dev'], self._deviation_latched)
            self.parameter = min(1., self.lambda_ * weight)
            self._tilt_log = dict(os_c3_weight=weight, os_c3_lambda=self.lambda_,
                os_c3_event_mass=feature['event_mass'], os_c3_h=feature['h'], os_c3_h_dev=feature['h_dev'],
                os_c3_dev_entry=float(entry), os_c3_dev_latched=float(self._deviation_latched),
                os_c3_unanimous=float(feature['unanimous']), os_c3_unknown=feature['unknown_mass'])
            for key in ('deviation', 'p75'):
                if feature[key] is not None:
                    self._tilt_log['os_c3_' + key] = feature[key]
        return result, code

    def query(self, q):
        result = super().query(q)
        if self.tilt:
            self._last_log.update(self._tilt_log)
            # New decisions and force-MISS inputs precede legacy diagnostics in
            # the plugin's 40-scalar mixed-mode log budget.
            result.extras = {**self._tilt_log, **result.extras}
        if self.cache_extension is not None:
            self.cache_extension.on_anchor(self, q, result, call=bool(self._last_log['os_c_call']))
        return result

    def _carried_log(self):
        log = super()._carried_log()
        return {**self._tilt_log, **log} if self.tilt else log

    def install_follow_extension(self, component):
        """Install C1's fitted FollowExtension via the ready composition bridge."""
        from .composition import FollowCacheExtension
        self.install_cache_extension(FollowCacheExtension(component))

    def install_cache_extension(self, extension):
        """Composition hook for C1 SA; see COMPOSITION.md.

        Install a fitted, per-controller extension with reset/on_anchor/
        blind_step/invalidate_anchor. It gets the frozen original A base and
        does not replace retrieval or policy-tail handling. Recalibrate SA's
        changed cadence/camera costs separately before deployment.
        """
        if self._episode_identity is not None:
            raise api.ContractError('install the cache extension before reset')
        self.cache_extension = extension

    def invalidate_anchor(self):
        super().invalidate_anchor()
        if self.cache_extension is not None:
            self.cache_extension.invalidate_anchor()

    def blind_step(self, bq):
        # R6's scheduled ambiguous LOOK always has priority over an extension.
        if self._look_due_step == int(bq.step) or self.cache_extension is None:
            return super().blind_step(bq)
        return self.cache_extension.blind_step(self, bq, super().blind_step)
