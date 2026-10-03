"""Swap only the corrector in the frozen r3c stacks. Standard PluginRuntime."""
import json
import numpy as np
from exp.offline_search.harness import api
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM
from exp.offline_search.rounds.r09.explore_fable.round3.tools.methods import (
    CorrectedCacheJ, NpGraspStack3, NpGraspStackGroot3, NpGraspEsc3)
from .numeric import predict


class RowCorrectedCache(BlindAWM):
    """One shared head plus one library-row table; neither takes a task ID."""
    family = 'r9a7_retrieval_residual'
    uses_nonlibrary_action = True
    _AR7_OWN = ('ar7_head', 'ar7_table', 'ar7_meta', 'ar7_blend')
    _OLD_HEAD = ('heads', 'head_meta', 'head_path', 'sig_head', 'chans', 'blend', 'correct_gripper')

    @classmethod
    def from_corrected(cls, source, path, cell):
        if type(source) is not CorrectedCacheJ:
            raise api.ContractError('expected the established judge-path corrector')
        if source.blend != .5 or source.correct_gripper:
            raise api.ContractError('expected half-strength motion-only corrector')
        clash = set(cls._AR7_OWN) & set(vars(source))
        # Check callable shadowing too; copying fitted objects must never hide a new method.
        clash |= {k for k in vars(source) if k in vars(cls) and callable(vars(cls)[k])}
        if clash:
            raise api.ContractError('corrector attribute collision: ' + str(sorted(clash)))
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['meta_json']))
            head = {k[5:]: np.array(z[k]) for k in z.files if k.startswith('head_')}
            table = np.array(z['table'])
        if (meta['cell'].rsplit('_', 1)[0] + '_cache' != cell or
                meta['fit_inits'] != list(range(20)) or meta['task_id_input'] or meta['task_dispatch'] or
                meta['input_dimension'] not in (207, 231) or meta['confidence_gain'] not in (0., .1, .25)):
            raise api.ContractError('incorrect cell, provenance or task-id input')
        if table.shape != (len(source.act), 60) or not np.isfinite(table).all():
            raise api.ContractError('invalid residual table')
        if head and (head['mean'].shape != (meta['input_dimension'],) or not all(np.isfinite(v).all() for v in head.values())):
            raise api.ContractError('invalid observation head')
        obj = cls.__new__(cls)
        vars(obj).update({k: v for k, v in vars(source).items() if k not in cls._OLD_HEAD})
        obj.ar7_head, obj.ar7_table, obj.ar7_meta, obj.ar7_blend = head, table, meta, .5
        for v in [table, *head.values()]:
            v.flags.writeable = False
        assert not any(k in vars(obj) for k in cls._OLD_HEAD)
        obj.name = 'R9A7_shared_retrieval_residual'
        obj.invalidate_anchor()
        return obj

    def _ar7_action(self, q, action, rows, weights):
        """Correction API intentionally needs no task/episode/init identity."""
        xv = np.r_[self.B0T @ np.asarray(q.key_v0, np.float32) - self.muB0,
                   self.B1T @ np.asarray(q.key_v1, np.float32) - self.muB1]
        x = np.r_[xv, np.asarray(q.rs, np.float32)[:8], np.asarray(action[:10, :7]).ravel(),
                  min(int(q.step), 120) / 120].astype(np.float32)[None]
        correction = predict(self.ar7_head, self.ar7_table, x,
                             np.asarray(rows)[None], np.asarray(weights)[None], self.act, self.ar7_meta)[0]
        out = np.array(action, dtype=np.float32, copy=True)
        out[:10, :6] += correction
        return out

    def os_synth(self, q, rows, w):
        action = BlindAWM.os_synth(self, q, rows, w)
        action = self._ar7_action(q, action, rows, w)
        # The blind tail must serve the SAME corrected anchor, exactly once.
        self._remember_anchor(q, rows, np.asarray(w) / np.sum(w), action)
        return action

    def query(self, q):
        res = BlindAWM.query(self, q)
        anchor = self._anchor
        res.action = self._ar7_action(q, res.action, anchor['rows'], anchor['weights'])
        self._remember_anchor(q, anchor['rows'], anchor['weights'], res.action)
        return res


class _SwapCorrector:
    def __init__(self, residual_path, **kwargs):
        super().__init__(**kwargs)
        if 'ar7_residual_path' in vars(self):
            raise api.ContractError('stack attribute collision')
        self.ar7_residual_path = str(residual_path)

    def fit(self, lib, ctx):
        # Validate against the *source* judge before its state is copied by r3c.
        from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
        with open(self.gm_onlynp_fit, 'rb') as f:
            source = FitUnpickler(f).load()['method']
        if 'ar7_residual_path' in vars(source):
            raise api.ContractError('stack attribute shadows frozen judge')
        path = self.ar7_residual_path
        super().fit(lib, ctx)  # r3c verifies gm_* conflicts, judge family and burst.
        assert self.ar7_residual_path == path
        self.base = RowCorrectedCache.from_corrected(self.base, path, ctx.cell)
        self.fit_info.update(corrector_path='os_synth', ar7_task_id_input=False,
                             ar7_residual_path=path, ar7_config=self.base.ar7_meta)
        assert self.burst == source.burst == 0
        assert tuple(self.disabled_guards) == tuple(source.disabled_guards)
        assert not any(k in vars(self.base) for k in RowCorrectedCache._OLD_HEAD)


class TaskFreeStackGroot(_SwapCorrector, NpGraspStackGroot3):
    family = 'r9a7_task_free_stack_groot'
