"""Layer 4 on the byte-preserved R10Recipe. All fitted knobs live in the pickle."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import numpy as np
from exp.offline_search.harness import api
from exp.offline_search.rounds.r10.recipe.recipe import R10Recipe
from .controller import Ledger, MissController

METHODS = ('off', 'random', 'periodic', 'periodic_pgt1', 'random_tail2',
           'distance', 'disagreement', 'error_hybrid', 'adaptive_error_hybrid')
REASONS = dict(zip(METHODS[1:], range(61, 69)))
METHOD_IDS = dict(zip(METHODS, range(9)))


def uniform(key, seed, task, init, step, domain):
    # R6's complete payload includes VERSION. Preserve it literally as required
    # by the reference uniform function; R11 domain/key strings distinguish arms.
    if init < 0 or task < 0:
        raise api.ContractError('R11 coins require original nonnegative task/init metadata')
    payload = ['Q2-deploy-v1', key, int(seed), int(task), int(init), int(step), domain]
    b = hashlib.sha256(json.dumps(payload, separators=(',', ':'), ensure_ascii=True).encode()).digest()[:8]
    return (int.from_bytes(b, 'big') >> 11) * 2.**-53


def history(q):
    s = int(q.step)
    hv = np.asarray(q.hist_has_vision, bool)
    hit = np.asarray(q.hist_hit)
    if len(hv) != s or len(hit) != s or np.any((hit != 0) & (hit != 1)):
        raise api.ContractError('knob needs accepted histories for every preceding decision')
    anchors = np.flatnonzero(hv)
    calls = anchors[hit[anchors] == 0]
    last = int(calls[-1]) if len(calls) else -1
    run = int(np.count_nonzero(anchors > last))
    return anchors, last, run, Ledger(s, len(anchors), len(calls))


def score_features(actions, weights, distances, sigma):
    """Exact experiment.retrieve feature arithmetic, including float32 means."""
    a = np.asarray(actions, np.float32)[None, :, :10, :7]
    w = np.asarray(weights, np.float32)[None]
    ds = np.asarray(distances, np.float64)[None]
    sig = np.asarray(sigma, np.float32)
    chunk = np.einsum('qk,qktc->qtc', w, a, optimize=False)
    var = np.einsum('qk,qk->q', w, np.mean(((a[:, :, :, :6]-chunk[:, None, :, :6])/sig[:6])**2, axis=(2, 3)))
    energy = np.mean((chunk[:, :, :6]/sig[:6])**2, axis=(1, 2))
    g = a[:, :, :, 6] >= 0
    trans = np.einsum('qk,qk->q', w, (g != g[:, :, :1]).any(2))
    vote = np.einsum('qk,qkt->qt', w, g.astype(np.float32))
    disagreement = np.mean(4*vote*(1-vote), axis=1)
    cg = chunk[:, :, 6] >= 0
    ct = (cg != cg[:, :1]).any(1).astype(float)
    gap = np.log1p((ds[:, -1]-ds[:, 0])/np.maximum(ds[:, 0], 1e-6))
    effective = 1/np.sum(w*w, 1)/16
    return np.column_stack((np.log1p(ds[:, 0]), np.log1p(var), np.log1p(energy),
                            trans, disagreement, ct, gap, effective))[0]


def predictor_score(head, features):
    z = np.clip((features-np.asarray(head['mean']))/np.asarray(head['std']), -8, 8)
    return float(np.r_[1., z] @ np.asarray(head['coef']))


class R11Knob(R10Recipe):
    family = 'r11_knob'

    def __init__(self, library='current', episode_subset=None, method='off', target=None):
        super().__init__(library, episode_subset)
        if method not in METHODS:
            raise ValueError(method)
        if method != 'off' and (target is None or not np.isfinite(target) or not 0 < target <= .5):
            raise ValueError('active knob needs a target IR in (0,.5]')
        self.knob_method, self.target = method, target
        self.knob_settings = None
        self._controller = None
        self._guard_steps, self._trigger_steps = set(), set()
        self._proposal_step, self._proposal = -1, None

    def fit(self, lib, ctx):
        from .fitting import fit_knob
        fit_knob(self, ctx)

    def configure(self, settings, model):
        self.knob_settings = copy.deepcopy(settings)
        self.knob_model = model
        if self.knob_method in METHODS[5:]:
            mode = 'adaptive' if self.knob_method == 'adaptive_error_hybrid' else (
                'hybrid' if self.knob_method == 'error_hybrid' else 'threshold')
            self._controller = MissController(model=model, method=mode, dose=settings['dose'], target=self.target,
                threshold=settings.get('threshold'), tie_probability=settings.get('tie_probability', 0.),
                score_reference=settings.get('score_reference'), beta=settings.get('beta', .5),
                eta=settings.get('eta', .2))
        self.name = f'R11Knob_{self.knob_method}_{self.target}__{self.name}'

    def reset(self, episode):
        super().reset(episode)
        self._guard_steps, self._trigger_steps = set(), set()
        self._proposal_step, self._proposal = -1, None
        if self._controller is not None:
            self._controller.reset(getattr(episode, 'uid', episode))

    def bytes_per_entry(self):
        cfg = self.knob_settings or {}
        extra = sum(v.nbytes for v in cfg.get('predictor', {}).values() if isinstance(v, np.ndarray))
        ref = cfg.get('score_reference')
        if isinstance(ref, np.ndarray): extra += ref.nbytes
        if self._controller is not None and self._controller.reference is not ref:
            if isinstance(self._controller.reference, np.ndarray):
                extra += self._controller.reference.nbytes
        return super().bytes_per_entry() + extra / len(self.row_subset)

    def _score(self, q):
        b = self.inner.base
        T, _, _, _, _, _, _, d, _, _, _ = b._dist(q)
        # These are the actual selected donors and weights, before correction.
        a = b._anchor
        pos = np.searchsorted(T.rows, a['rows'])
        ds = np.asarray(d[pos], np.float64)
        if self.knob_method == 'distance':
            return float(np.min(d))
        X = score_features(b.act[a['rows']], a['weights'], ds, b.sig)
        if self.knob_method == 'disagreement':
            return float(np.expm1(X[1]))
        return predictor_score(self.knob_settings['predictor'], X)

    def query(self, q):
        if self.knob_method == 'off':
            return super().query(q)  # no extra fields or memo changes, even on retries
        if int(q.step) == self._proposal_step:
            return copy.deepcopy(self._proposal)
        if int(q.step) < self._proposal_step:
            raise api.ContractError('reset knob on a new episode')
        res = super().query(q)  # retains policy gate, guard memo and corrected cache
        ex = dict(res.extras)
        guard = bool(ex.get('os_force_miss', 0))
        anchors, last, run, ledger = history(q)
        method, cfg = self.knob_method, self.knob_settings
        knob, u, cap, score, p, nonfinite = False, -1., -1., 0., 0., False
        sampled = False
        setting = cfg.get('setting', cfg.get('dose', 0.))
        reason = REASONS[method]
        if guard:
            self._guard_steps.add(int(q.step))
        if method in METHODS[1:5]:
            task, init, step = int(q.task_id), int(q.episode.init), int(q.step)
            prev = int(anchors[-1]) if len(anchors) else -1
            if method.startswith('periodic'):
                u = uniform('R11-gap-v1', 0, task, init, last, 'cap')
                floor = math.floor(setting)
                cap = floor+int(setting-floor > 0 and u < setting-floor)
                tail = method == 'periodic_pgt1' and prev in self._guard_steps and ledger.calls > 0 and last == prev
                knob = not guard and (tail or run >= cap)
                p = float(tail or run >= cap)
            else:
                u = uniform('R11-random-v1', 0, task, init, step, 'knob-anchor')
                tail = method == 'random_tail2' and prev in self._trigger_steps and last == prev
                knob = not guard and (tail or u < setting)
                p = 1. if tail else setting
                if knob and not tail:
                    self._trigger_steps.add(step)
                    # Internal trigger memo implements opus's reason-61 marker;
                    # the task requires a unique external code for each method.
        else:
            score = self._score(q)
            out = self._controller.propose(step=int(q.step), score=score, guard=guard, ledger=ledger)
            knob = not guard and out['knob']
            sampled = out['knob']
            p, setting, nonfinite = out['probability'], out['dose'], out['nonfinite']
        if knob:
            ex.update(os_force_miss=1., os_reason=float(reason))
        diagnostics = dict(os_r11_version=1., os_r11_method=float(METHOD_IDS[method]),
            os_r11_guard=float(guard), os_r11_eligible=float(not guard), os_r11_knob_call=float(knob),
            os_r11_setting=float(setting), os_r11_coin=float(u), os_r11_run=float(run),
            os_r11_cap=float(cap), os_r11_calls_before=float(ledger.calls), os_r11_p=float(p),
            os_r11_score=float(score) if np.isfinite(score) else 0., os_r11_nonfinite=float(nonfinite),
            os_r11_N=float(ledger.decisions), os_r11_V=float(ledger.looks), os_r11_M=float(ledger.calls))
        # Put all scalar audit fields before lower-priority non-os diagnostics.
        aliases = (dict(r11_knob=float(sampled), r11_guard=float(guard), r11_p=float(p),
                        r11_score=diagnostics['os_r11_score'], r11_dose=float(setting))
                   if method in METHODS[5:] else {})
        res.extras = {**{k:v for k,v in ex.items() if k.startswith('os_')}, **diagnostics, **aliases,
                      **{k:v for k,v in ex.items() if not k.startswith('os_')}}
        self._proposal_step, self._proposal = int(q.step), copy.deepcopy(res)
        return res
