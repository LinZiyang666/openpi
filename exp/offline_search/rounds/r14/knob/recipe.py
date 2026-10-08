"""Independent R14 partitions, with the byte-preserved R10/R11 off path."""
from __future__ import annotations
import copy
import time
import numpy as np
from exp.offline_search.harness import api
from exp.offline_search.rounds.r10.recipe.recipe import R10Recipe, _CorrectedCache, predict
from exp.offline_search.rounds.r11.knob.recipe import R11Knob, history
from .controller import propose, REASON_TASK_PERIODIC

SWITCHES = ('scope', 'metric', 'head', 'continuity', 'guard')

class _PartitionCache(_CorrectedCache):
    """AWM arithmetic over a table embedded in the query's coordinates."""
    def reset(self, episode):
        super().reset(episode)
        self._distance_key = None
        self._distance_value = None
        self._retrieval_ns = self._correction_ns = 0
        self._correction_audit = {}

    def _features(self, q, xv, rs8, action):
        x = super()._features(q, xv, rs8, action)
        return x[:, :-10] if self.r14_head == 'pool' else x

    def _dist(self, q):
        key = (id(q), int(q.step), int(q.task_id), getattr(q.episode, 'uid', None))
        if getattr(self, '_distance_key', None) == key:
            return self._distance_value
        started = time.perf_counter_ns()
        value = super()._dist(q)
        self._retrieval_ns = time.perf_counter_ns() - started
        self._distance_key, self._distance_value = key, value
        return value

    def _correction(self, q, action):
        started = time.perf_counter_ns()
        corr = super()._correction(q, action)
        self._correction_ns = time.perf_counter_ns() - started
        a = self._anchor
        distance = gate = 0.
        if int(q.step) and a is not None:
            T, *_, d, med, c, dt = self._dist(q)
            positions = np.searchsorted(T.rows, a['rows'])
            distance = float(np.min(d[positions]))
            ratio = distance / self.distance_scale
            gate = 0. if not np.isfinite(ratio) else .5 * float(np.clip((2. - ratio) / 1.25, 0., 1.))
        self._correction_audit = dict(distance=distance, gate=gate,
            norm=float(np.linalg.norm(corr[:, :6])), motion_rms=float(np.sqrt(np.mean(corr[:, :6] ** 2))))
        return corr


class R14Knob(R11Knob):
    family = 'r14_partition'

    def __init__(self, library='current', episode_subset=None, *, scope='task', metric='task',
                 head='task', continuity='task', guard='task', method='off', target=None):
        if method not in ('off', 'periodic', 'task_periodic'):
            raise ValueError(method)
        super().__init__(library, episode_subset, 'periodic' if method == 'task_periodic' else method, target)
        self.knob_method = method
        for name, value in dict(scope=scope, metric=metric, head=head, continuity=continuity, guard=guard).items():
            if value not in ('task', 'pool'):
                raise ValueError(f'{name} must be task|pool')
            setattr(self, 'r14_' + name, value)
        if method != 'off' and any(getattr(self, 'r14_' + s) != 'task' for s in SWITCHES):
            raise ValueError('primary task allocation uses unchanged lower layers')
        self.r14_fit = None
        self.r14_diagnostics = None
        self._accepted_prefix = ((), ())
        self._fallback_counts = {}

    def fit(self, lib, ctx):
        from .fitting import fit_recipe
        fit_recipe(self, ctx)

    def configure(self, settings, model):
        if self.knob_method == 'task_periodic':
            K=np.asarray(settings.get('K'),float)
            if K.shape!=(10,) or not np.isfinite(K).all() or np.any(K<0):
                raise ValueError('need ten finite nonnegative task caps')
            if len(settings.get('priority',()))!=10:
                raise ValueError('need ten frozen task priorities')
        super().configure(settings,model)

    def reset(self, episode):
        super().reset(episode)
        self.r14_diagnostics = None
        self._accepted_prefix = ((), ())
        self._fallback_counts = {}

    def query(self, q):
        lower_started=time.perf_counter_ns()
        if self.knob_method == 'task_periodic':
            if int(q.step) == self._proposal_step:
                return copy.deepcopy(self._proposal)
            if int(q.step) < self._proposal_step:
                raise api.ContractError('reset R14 on a new episode')
            # Validate accepted history without counting any proposed action.
            history(q)
            hv, hh = np.asarray(q.hist_has_vision), np.asarray(q.hist_hit)
            if hv.ndim != 1 or hh.ndim != 1 or np.any((hv != 0) & (hv != 1)):
                raise api.ContractError('malformed R14 accepted history')
            oldv, oldh = self._accepted_prefix
            if tuple(hv[:len(oldv)]) != oldv or tuple(hh[:len(oldh)]) != oldh:
                raise api.ContractError('R14 accepted history was rewritten')
            res = R10Recipe.query(self, q)
            lower_ns=time.perf_counter_ns()-lower_started
            ex = dict(res.extras)
            out = propose(q, bool(ex.get('os_force_miss', 0)), self.knob_settings['K'])
            if out['knob']:
                ex.update(os_force_miss=1., os_reason=float(REASON_TASK_PERIODIC))
            res.extras = ex
            self._accepted_prefix = tuple(map(bool, hv)), tuple(map(int, hh))
            self._proposal_step, self._proposal = int(q.step), copy.deepcopy(res)
        else:
            res = super().query(q)
            lower_ns=time.perf_counter_ns()-lower_started
            out = None
        self._diagnostics(q, res, out)
        if self.r14_diagnostics is not None:
            self.r14_diagnostics['lower_query_ms']=lower_ns/1e6
        return res

    def _diagnostics(self, q, res, allocation):
        base = self.inner.base
        anchor = base._anchor
        if not hasattr(q, 'hist_has_vision'):
            # Off has precisely the R10 input contract. Diagnostics never add
            # an accepted-history requirement to that original path.
            self.r14_diagnostics = None
            return
        _, _, _, ledger = history(q)
        record = dict(schema='offline_search.r14.v1', step=int(q.step), task_id=int(q.task_id),
            N=ledger.decisions, V=ledger.looks, M=ledger.calls, target=self.target,
            method=self.knob_method, switches={s:getattr(self, 'r14_' + s) for s in SWITCHES},
            fit_sha256=(self.r14_fit or {}).get('fit_sha256'),
            parent_donor_rows=self.row_subset[anchor['rows']].tolist(),
            donor_task_ids=self.inner.C.task[anchor['rows']].astype(int).tolist(),
            branch='early' if int(q.step) == 0 else ('fresh' if q.prev_hit is False else 'stale'),
            active_guard_reason=float(res.extras.get('os_reason', 0)),
            lifecycle_fallback_counts=dict(self._fallback_counts))
        if isinstance(base, _PartitionCache):
            record.update(retrieval_ms=base._retrieval_ns / 1e6, correction_ms=base._correction_ns / 1e6,
                          corrector=dict(base._correction_audit))
        else:
            record.update(original_corrector_audit(base,q,res), retrieval_ms=None, correction_ms=None)
        if allocation is not None:
            cfg = self.knob_settings
            record.update(allocation, reference_K=cfg['reference_K'], priority=cfg['priority'][int(q.task_id)],
                          gain=cfg['gain'], bias=cfg['bias'])
            record['active_guard_reason'] = float(res.extras.get('os_reason', 0)) if allocation['guard'] else 0.
        self.r14_diagnostics = record

    def _fallback(self, result):
        from exp.offline_search.closed_loop.blind import LookReason
        if isinstance(result, LookReason):
            self._fallback_counts[result.name] = self._fallback_counts.get(result.name, 0) + 1
        return result

    def blind_step(self, q):
        return self._fallback(super().blind_step(q))

    def policy_tail_step(self, q):
        return self._fallback(super().policy_tail_step(q))


def original_corrector_audit(base, q, res):
    """Read-only audit of R10's actual donor gate; never a new action proposal."""
    started=time.perf_counter_ns()
    anchor=base._anchor
    distance=gate=norm=rms=0.
    if int(q.step) and anchor is not None:
        T,_,_,_,_,xv,_,ds,_,_,_=base._dist(q)
        positions=np.searchsorted(T.rows,anchor['rows'])
        if np.any(positions>=len(T.rows)) or not np.array_equal(T.rows[positions],anchor['rows']):
            raise api.ContractError('R14 diagnostic donor mapping failed')
        distance=float(np.min(ds[positions]));ratio=distance/base.distance_scale
        factor=0. if not np.isfinite(ratio) else float(np.clip((2.-ratio)/1.25,0.,1.))
        gate=.5*factor
        if factor:
            # The original judge returns the proposal's normalized kernel
            # scores. Reconstruct that same raw proposal solely for logging;
            # the serving head/gate already used these actual donor rows.
            w=np.exp(-(res.scores[0]-res.scores)/base.synth_T)
            raw=base._mix(anchor['rows'],w)[2]
            x=base._features(q,xv,np.asarray(q.rs,np.float32)[:8],raw)
            corr=predict(base.heads[str(int(q.task_id))],x)[0].reshape(10,base.chans)*base.sig_head[:base.chans]
            corr=corr*np.float32(factor)
            rms=float(np.sqrt(np.mean(corr[:,:6]**2)));norm=float(np.linalg.norm(corr[:,:6]))
    return dict(audit_reconstruction_ms=(time.perf_counter_ns()-started)/1e6,
                corrector=dict(distance=distance,gate=gate,motion_rms=rms,norm=norm,blend=.5))


def reference_diagnostics(method, q, res):
    """Opt-in side record for the unchanged original R11 periodic artifact."""
    if getattr(method,'knob_method',None)!='periodic':
        return {}
    b=method.inner.base;a=b._anchor;x=res.extras
    _,_,_,ledger=history(q)
    out=dict(schema='offline_search.r14.v1',step=int(q.step),task_id=int(q.task_id),
        N=ledger.decisions,V=ledger.looks,M=ledger.calls,target=method.target,method='periodic',
        reference_K=float(method.knob_settings['setting']),K=float(method.knob_settings['setting']),
        cap=x['os_r11_cap'],coin=x['os_r11_coin'],run=x['os_r11_run'],
        guard=bool(x['os_r11_guard']),knob=bool(x['os_r11_knob_call']),priority=0.,gain=0.,bias=0.,
        parent_donor_rows=method.row_subset[a['rows']].tolist(),
        donor_task_ids=method.inner.C.task[a['rows']].astype(int).tolist(),
        branch='early' if int(q.step)==0 else ('fresh' if q.prev_hit is False else 'stale'),
        active_guard_reason=float(x['os_reason']) if x['os_r11_guard'] else 0.)
    out.update(original_corrector_audit(b,q,res))
    return out
