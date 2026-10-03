"""Meaningful admission, integration and regression checks for the frozen arms."""
import importlib
import json
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from exp.offline_search.closed_loop.plugin import clone_method
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
from exp.offline_search.rounds.r09.explore_fable.round3.tools import methods as f
from .methods import Pace12, StackTakeover
from .safe import HERE, RUN, identity, records, sha, dump


def load(name):
    with (RUN/'fits'/f'r9a5_{name}.pkl').open('rb') as h:
        return FitUnpickler(h).load()


class AdmissionTests(unittest.TestCase):
    def test_reject_before_json_payload_parse(self):
        path=RUN/'testdata/admission.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        # Synthetic poison values are deliberately invalid JSON. They must never
        # reach json.loads, nor be printed by the reader.
        path.write_text('{"task_id":0,"init":30,"payload":INVALID}\n'
                        '{"task_id":0,"init":49,"payload":INVALID}\n'
                        '{"task_id":0,"init":20,"ok":true}\n')
        self.assertEqual(list(records(path)),[((0,20),dict(task_id=0,init=20,ok=True))])
        self.assertIsNone(identity('{"task_uid":"a:eval:0:49"}'))

    def test_prediction_precedes_freeze(self):
        frozen=json.loads((HERE/'FROZEN.json').read_text())
        self.assertEqual(sha(HERE/'PREDICTION.md'),frozen['prediction_sha256'])
        self.assertLess((HERE/'PREDICTION.md').stat().st_mtime,(RUN/'arms.json').stat().st_mtime)


class FrozenTests(unittest.TestCase):
    def test_artifacts_and_source_pins(self):
        freeze=json.loads((HERE/'FROZEN.json').read_text())
        for row in freeze['arms']:
            self.assertEqual(sha(row['artifact']),row['sha256'])
        for model in ['pi05','groot']:
            b=load(model+'_l10_50_latch');obj=b['method']
            for mod,h in obj.tk_pins['sources'].items():
                self.assertEqual(sha(importlib.import_module(mod).__file__),h)
            for path,h in obj.tk_pins['assets'].items():self.assertEqual(sha(path),h)
            self.assertEqual(obj.tk_force_at,())
            self.assertEqual(obj.tk_stack.burst,0)
            self.assertEqual(obj.tk_stack.gm_max_calls,0)
            self.assertFalse(set(vars(obj)) & set(vars(obj.tk_stack)) - {'name','prof','fit_info'})

    def test_corrector_active_in_os_synth_and_corrected_anchor(self):
        for model in ['pi05','groot']:
            for mode in ['control','latch','pace12']:
                m=load(model+'_l10_50_'+mode)['method']
                stack=m if mode=='control' else m.tk_stack
                base=stack.base
                self.assertIs(type(base),f.CorrectedCacheJ)
                self.assertIs(type(base).os_synth,f.CorrectedCacheJ.os_synth)
                self.assertEqual(base.blend,.5)
                rows=np.asarray(base.tasks[0].rows[:16]);w=np.ones(16)
                q=SimpleNamespace(step=3,task_id=0,key_v0=np.zeros(32768,np.float32),
                    key_v1=np.zeros(32768,np.float32),rs=np.zeros(32,np.float32),
                    episode=SimpleNamespace(uid='test:0:0'))
                plain=f.CorrectedCache.os_synth(base,q,rows,w)
                corrected=base.os_synth(q,rows,w)
                self.assertGreater(np.max(np.abs(corrected[:10,:6]-plain[:10,:6])),1e-4)
                np.testing.assert_array_equal(corrected[:,6],plain[:,6])
                np.testing.assert_array_equal(base._anchor['action'],corrected)

    def test_fit_rejects_frozen_dependency_change(self):
        b=load('pi05_l10_50_latch');m=StackTakeover(**b['kwargs'])
        with patch('exp.offline_search.rounds.r09.explore_astra.round5.tools.methods.digest',return_value='changed'):
            with self.assertRaisesRegex(Exception,'frozen source changed'):
                m.fit(None,SimpleNamespace(cell=b['cell'],model='pi05'))

    def test_detector_distance_matches_cache_diagnostic(self):
        for model in ['pi05','groot']:
            base=load(model+'_l10_50_latch')['method'].tk_stack.base
            ep=SimpleNamespace(uid='test:0:0',task_id=0,init=0)
            base.reset(ep)
            q=SimpleNamespace(step=0,task_id=0,episode=ep,prev_hit=None,
                key_v0=np.zeros(32768,np.float32),key_v1=np.zeros(32768,np.float32),
                rs=np.zeros(32,np.float32),hist_a_exec=np.zeros((0,50 if model=='pi05' else 16,32),np.float32),
                hist_has_vision=np.zeros(0,bool))
            _,_,_,_,_,xv,_,d,med,_,_=base._dist(q)
            result=base.query(q)
            self.assertAlmostEqual(float(np.min(d))/med,result.extras['d1_rel'],places=12)
            np.testing.assert_array_equal(xv,np.r_[base.B0T@q.key_v0-base.muB0,base.B1T@q.key_v1-base.muB1])

    def test_reset_clone_and_anchor_invalidation_preserve_takeover(self):
        for model in ['pi05','groot']:
            m=load(model+'_l10_50_latch')['method']
            ep=SimpleNamespace(uid='test:0:0',task_id=0,init=0)
            m.reset(ep)
            m.tk_gate.observe(1,12);m.tk_gate.observe(1,14)
            self.assertEqual(m.tk_gate.used,1)
            m.invalidate_anchor()
            self.assertEqual(m.tk_gate.used,1)
            self.assertTrue(m.tk_gate.active)
            other,_=clone_method(m,strict=True)
            other.reset(ep)
            self.assertEqual(other.tk_gate.used,0)
            self.assertEqual(m.tk_gate.used,1)
            self.assertIsNot(m.tk_stack._s,other.tk_stack._s)

    def test_pace12_matches_imported_entry_and_stops_exactly(self):
        base=SimpleNamespace(lib_step=np.array([0]))
        gate=Pace12(base);ep=SimpleNamespace(uid='test:0:0')
        q=lambda s:SimpleNamespace(step=s,episode=ep)
        self.assertEqual(gate.observe(q(10),0),(False,False))
        self.assertEqual(gate.observe(q(12),0),(True,True))
        for step in range(14,36,2):self.assertEqual(gate.observe(q(step),0),(True,False))
        self.assertEqual(gate.used,12)
        for step in [36,38,80,90]:self.assertEqual(gate.observe(q(step),0),(False,False))
        late=Pace12(base)
        self.assertEqual(late.observe(q(82),0),(False,False))

    def test_preserved_guard_verdict_and_cap_do_not_suppress_guard(self):
        from exp.offline_search.harness import api
        m=load('pi05_l10_50_pace12')['method'];ep=SimpleNamespace(uid='test:0:0',task_id=0,init=0)
        m.reset(ep);m.tk_gate.used=12
        m.tk_stack.base._anchor=dict(step=0)
        q=SimpleNamespace(step=0,episode=ep)
        result=api.Result(np.array([0]),np.array([0.]),0.,action=np.zeros((50,32),np.float32),
            library='current',extras=dict(os_force_miss=1.,os_reason=4.,os_flags=8.))
        with patch.object(type(m.tk_stack),'query',return_value=result):
            out=m.query(q)
        self.assertEqual(out.extras['os_force_miss'],1.)
        self.assertEqual(out.extras['os_reason'],4.)
        self.assertEqual(out.extras['os_flags'],8.)
        self.assertEqual(out.extras['r9a5_added'],0.)


if __name__=='__main__':unittest.main(verbosity=2)
