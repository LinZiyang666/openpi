"""Pre-stage-one light lookup and actual-work ledger, imported only when opted in."""
from __future__ import annotations
import copy
import time
from types import SimpleNamespace
import numpy as np
from exp.offline_search.closed_loop import plugin as p


class LookInputs:
    """All causally available data. Images are wire agentview/wrist in library order."""
    def __init__(self, session, rs):
        self.s = session
        self.rs = p._ro(np.array(rs, np.float32, copy=True))
        self.step, self.task_id, self.episode = session.step, session.ep.task_id, session.ep
        self.instruction = session.ep.task
        self.raw_state = p._ro(session._raw_state().copy())
        self.img0 = p._ro(np.array(session.cur_obs['observation/image'], copy=True))
        self.img1 = p._ro(np.array(session.cur_obs['observation/wrist_image'], copy=True))
        self.hist_looks = tuple(session._qk_looks)
        self.hist_images = tuple((x['img0'], x['img1']) for x in self.hist_looks)
        self.hist_features = tuple(x['features'] for x in self.hist_looks)
        self.hist_a_exec = session.b_aex.view(0, session.step)
        self.hist_hit = p._ro(np.asarray(session.hits, np.int8))
        self.hist_has_look = p._ro(np.asarray(session.has_vision, bool))
        self.hist_rs = session.b_rs.view(0, session.step)
        self.hist_raw_state = session.b_raw.view(0, session.step)
        self.prev_hit = None if not session.step else bool(session.hits[-1])
        self.prev_a_exec = None if not session.step else session.b_aex.view(session.step-1,session.step)[0]
        self.has_look, self.model = True, session.rt.model
        self.last_policy_output = getattr(session, '_qk_last_policy', None)
        self.last_policy_wire_output = getattr(session, '_qk_last_policy_wire', None)


class LightQueryView(p.OnlineQueryView):
    key_kind = 'light_symmetric'
    has_look = property(lambda s: bool(s._s.has_vision[s.step]))
    hist_has_look = property(lambda s: p._ro(np.asarray(s._s.has_vision[:s.step], bool)))
    light_features = property(lambda s: p._ro(np.stack([s._s.b_v0.a[s.step], s._s.b_v1.a[s.step]])))
    hist_light_features = property(lambda s: p._ro(np.stack([s._s.b_v0.view(0, s.step), s._s.b_v1.view(0, s.step)], 1)))
    has_tok = property(lambda s: False)
    img0 = property(lambda s: p._ro(np.asarray(s._s.cur_obs['observation/image'])))
    img1 = property(lambda s: p._ro(np.asarray(s._s.cur_obs['observation/wrist_image'])))
    representation_sha256 = property(lambda s: s._s.method.light_key_spec['representation_sha256'])
    hist_looks = property(lambda s: tuple(s._s._qk_looks[:-1]))
    hist_images = property(lambda s: tuple((x['img0'], x['img1']) for x in s.hist_looks))
    hist_features = property(lambda s: tuple(x['features'] for x in s.hist_looks))
    last_policy_output = property(lambda s: s._s._qk_last_policy)
    last_policy_wire_output = property(lambda s: s._s._qk_last_policy_wire)
    instruction = property(lambda s: s.episode.task)
    def unavailable(self):
        raise LookupError('teacher keys/tokens unavailable in a light image look')
    key_v0 = property(unavailable)
    key_v1 = property(unavailable)
    tok_v0 = property(unavailable)
    tok_v1 = property(unavailable)
    hist_key_v0 = property(unavailable)
    hist_key_v1 = property(unavailable)


def initialize(rt):
    from .encoder import FrozenEncoder
    if (rt.model != 'pi05' or not rt.blind or rt.shadow_native or rt.gpu is not None
            or rt.judge is None or rt.judge.mode != 'guard_only' or rt.lookhalf is not None
            or rt.cache_ablate is not None or rt.request_cameras):
        raise ValueError('QK requires pi05 blind guard_only, no shadows/camera modes')
    if not getattr(rt.method, 'light_key_spec', None):
        raise ValueError('QK requires a complete light-key artifact')
    spec, method = rt.method.light_key_spec, rt.method
    if (spec['cell'] != f'{rt.model}_{rt.suite}_{method.size}' or not spec['library_only']
            or spec['completed_epochs'] != spec.get('declared_epochs',80) or not method.fit_info['library_only']
            or method.episode_subset != method.fit_info['episode_ids_by_task']):
        raise ValueError('QK wrong-cell/subset or incomplete frozen artifact')
    base = method.inner.base
    if (set(base.heads) != set(map(str, range(10))) or set(base.tasks) != set(range(10))
            or not np.isfinite(base.distance_scale) or base.distance_scale <= 0
            or base.B0T.shape != (64,64) or base.B1T.shape != (64,64)
            or len(method.row_subset) != sum(len(t.rows) for t in base.tasks.values())):
        raise ValueError('QK partial metric/head/row artifact')
    if rt.opts.os_tokens != 'off' or rt.opts.os_log_inputs:
        raise ValueError('QK teacher tokens and legacy input recording must be off')
    rt.light_encoder = FrozenEncoder(rt.method.light_key_spec)


def infer(s, adapter, obs, full_call):
    # _try_blind has validated identity, state, transaction and execution receipt.
    if not (obs.get('__extra__') or {}).get('exec_ok', True):
        s.method.invalidate_anchor()
        s._policy_tail = None
    t0 = time.perf_counter_ns()
    look = LookInputs(s, s._prepared_rs)
    features = s.rt.light_encoder.encode(look)
    elapsed = (time.perf_counter_ns() - t0) / 1e6
    s._qk_looks.append(dict(step=s.step, img0=look.img0, img1=look.img1,
                           features=p._ro(features.copy())))
    ctx = SimpleNamespace(task_key=s.ep.task, current_step=s.step, checkpoint_id='CP1',
                          query_keys={'light_features': features, 'robot_state': look.rs})
    results = s.on_search(ctx)
    d = s._dec
    d.update(vision=True, light_encoder_ran=True, light_encoder_ms=elapsed)
    from .recipe import FeatureGeometry
    query = FeatureGeometry(LightQueryView(s,d['step'],s.ep.task_id,s.ep))
    distances = s.method.inner.base._dist(query)[7]
    d['qk_distance_ratio'] = float(np.min(distances)) / s.method.inner.base.distance_scale
    if d['hit']:
        try:
            out = adapter.output(d['served'], adapter.prepare(obs)[1])
        except Exception:
            d.update(hit=False,judge='light_output_preflight_failed')
            d['extras'] = dict(d['extras'],os_force_miss=1.,os_reason=8.)
            s.method.invalidate_anchor()
    if not d['hit']:
        # The ordinary full-policy path searches only this already-decided MISS.
        # It does not query, append an input, or commit a provisional action.
        s._qk_full = True
        d['policy_full_call'] = True
        before = s.stage1_calls
        started = time.perf_counter_ns()
        try:
            out = full_call()
        finally:
            s._qk_full = False
            d['policy_full_ms'] = (time.perf_counter_ns() - started) / 1e6
        if s.stage1_calls != before + 1:
            raise RuntimeError('QK MISS must run exactly one tower')
        s._qk_last_policy_wire = p._ro(np.array(out['actions'], copy=True))
        return out
    action = d['served']
    p.commit_external_hit(adapter.orch, look.rs, step=d['step'], winner=d['winner'])
    import torch
    adapter.orch.broadcast_action(torch.from_numpy(action.copy()))
    adapter.orch.clear()
    out['__hit_meta__'] = dict(hit_type='FULL_HIT', start_t=None, winner_id=d['winner'],
        cp1_score=None, checkpoint='CP1', score=None, searched=True, source='cache',
        factor_outputs={'osplug': s.wire_diag()})
    if adapter.fake:
        out.update(winner=d['winner'], hit_type='FULL_HIT', diag=s.wire_diag())
    return out


def ledger(row, d, s, ok):
    look = bool(d.get('vision', True))
    q = int(d.get('light_encoder_ran', False))
    m = int(d.get('policy_full_call', False))
    t = int(s.stage1_calls - getattr(s, '_qk_tower_before', s.stage1_calls))
    if ok and (t != m or q != int(look)):
        raise RuntimeError('QK actual-work receipt differs from decision')
    epsilon = float(s.method.light_key_spec['epsilon'])
    delta = dict(Q=q, T=t, M=m, N=int(ok))
    for key, value in delta.items():
        s._qk_counts[key] += value
    counts = dict(s._qk_counts)
    row.update(look=look, has_look=look, key_kind='light_symmetric' if look else 'none',
        representation_sha256=s.method.light_key_spec['representation_sha256'],
        light_encoder_ran=bool(q), light_encoder_ms=d.get('light_encoder_ms'),
        teacher_tower_ran=bool(t), teacher_tower_ms=s._s1_ms if t else None,
        policy_full_call=bool(m), policy_full_ms=d.get('policy_full_ms'),
        policy_remainder_ms=max(0., d.get('policy_full_ms', 0.) - (s._s1_ms or 0.)) if m else None,
        epsilon=epsilon, qk_work=delta, Q=counts['Q'], T=counts['T'], M=counts['M'], N=counts['N'],
        IR_new=(epsilon * counts['Q'] + counts['M']) / counts['N'] if counts['N'] else None,
        decision_cost=epsilon * q + m, nominal_controls=5 * counts['N'],
        vision=False if not t else True, hist_has_vision_alias='logical look, controller only',
        distance_ratio=d.get('qk_distance_ratio'),
        parent_donor_rows=(s.method.row_subset[d['rows']] if look else d['rows']) if 'rows' in d else None,
        donor_weights=d.get('weights'),
        guard_reason=d['extras'].get('os_reason', 0.), fixed_cap=d['extras'].get('os_r11_cap', 0.),
        diagnostics_geometry='student features; original teacher token diagnostics unavailable')
