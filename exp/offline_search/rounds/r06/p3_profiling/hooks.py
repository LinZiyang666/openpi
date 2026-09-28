"""Opt-in, connection-local staged-policy tap and reusable MISS output.

No recursive infer(), shadow orchestrator, broadcast, history push, global RNG
seed/fork, or second policy forward on injection. Stage 1 is shared exactly.
"""
import copy
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

EXTENSIONS = {}  # register before connections: namespace -> (schema, fn(session, record))


def register_extension(namespace, schema, fn):
    if not namespace or namespace in EXTENSIONS or not callable(fn):
        raise ValueError("unique extension namespace and callable required")
    EXTENSIONS[namespace] = (schema, fn)


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.array(value, np.float32, copy=True)


def sync():
    import torch
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()


class Proxy:
    def __init__(self, target, **overrides):
        self._target, self._overrides = target, overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._target, name)


class Engine:
    def __init__(self, conn):
        self.conn, self.s = conn, conn._osp_sessions[0]
        s, rt = self.s, self.s.rt
        if (not rt.blind or not rt.policy_tail or rt.policy_tail_blocks != 1 or rt.gpu is not None
                or rt.randomized or rt.judge is None or rt.judge.mode != "guard_only"
                or rt.judge.cap or rt.judge.burst != 1 or rt.judge.step0 != "judge"
                or getattr(rt.opts, "os_stage1_mode", "full") != "full"):
            raise ValueError("P3 requires full-stage A/B, blind, policy_tail blocks=1, uncapped guard_only")
        if s.miss_steps != (10 if rt.model == "pi05" else 8):
            raise ValueError("P3 requires the full policy denoising schedule")
        self.p = conn._osp_adapter.policy
        if getattr(self.p, "_trace", None) is not None or getattr(self.p, "_warm_reset", None) is not None:
            raise ValueError("P3 needs untraced ordinary full-policy inference")
        self.stage1 = self.stage2 = self.stage3 = self.noise = None
        self.reuse = False
        self.profile = None
        self.timings = {}
        self.fake = conn._osp_adapter.fake
        if self.fake:
            if not hasattr(self.p, "_p3_policy"):
                raise ValueError("CPU fixture must provide _p3_policy(seed) and reuse callback")
        elif rt.model == "pi05":
            p = self.p
            self.f1, self.f2, self.f3 = p._stage1_fn, p._stage2_fn, p._stage3_fn
            self.model = p._model
            if str(p._stage3_device) == "meta":
                raise ValueError("P3 needs a full model, including p=0")
            p._stage1_fn = self.capture_stage1
            p._stage2_fn = lambda *a, **kw: self.stage2 if self.reuse else self.f2(*a, **kw)
            p._stage3_fn = lambda *a, **kw: self.stage3 if self.reuse else self.f3(*a, **kw)
            # Non-coordinator MISS calls _model.run_stage3 directly; never
            # mutate the shared model object or its bound stage functions.
            p._model = Proxy(self.model,
                run_stage3=lambda *a, **kw: self.stage3 if self.reuse else self.model.run_stage3(*a, **kw),
                sample_noise=lambda *a, **kw: self.noise if self.reuse else self.model.sample_noise(*a, **kw))
        else:
            self.runner = self.p._runner
            self.f1 = self.runner.run_stage1
            self.p._runner = Proxy(self.runner, run_stage1=self.capture_stage1,
                                   run_stage2=self.groot_reuse)
        original_search, original_after = s.on_search, s.after_infer
        original_executed = s.on_executed
        original_diag = s.wire_diag

        def search(ctx):
            result = original_search(ctx)
            self.anchor()
            return result

        def after(ms, ok, err=None):
            self.last_log_ok = False
            try:
                self.last_marker = dict(p3_anchor=self.profile is not None,
                                        p3_step=s._dec["step"] if s._dec is not None else None)
                if self.profile is not None:
                    self.finish(ms, ok, err)
                result = original_after(ms, ok, err)
                self.last_log_ok = True
                return result
            finally:
                self.stage1 = self.stage2 = self.stage3 = self.noise = None
                self.reuse = False
                self.profile = None
                self.timings = {}

        def diag():
            out = original_diag()
            if out is not None:
                out["p3_anchor"] = bool(self.profile is not None)
                out["p3_step"] = s._dec["step"]
            return out

        def executed(chunk):
            if self.profile is not None:
                expected = self.profile["policy_chunk"] if self.profile["assignment"]["executed_policy"] else self.profile["cache_chunk"]
                if array(chunk).reshape(s.rt.H, 32).tobytes() != expected.tobytes():
                    raise RuntimeError("P3 interceptor did not reuse the selected chunk")
            return original_executed(chunk)

        s.on_search, s.after_infer, s.wire_diag, s.on_executed = search, after, diag, executed
        # Production interceptors omit factor_outputs from wire hit metadata.
        # Attach a small explicit marker after _osp_infer's finally/logging,
        # still under the connection lock; never infer anchors from step parity.
        if hasattr(conn, "_osp_infer"):
            original_infer = conn._osp_infer
            def infer_with_marker(*args, **kwargs):
                result = original_infer(*args, **kwargs)
                if not self.last_log_ok:
                    raise RuntimeError("P3 decision logging failed; refusing to serve an unlogged chunk")
                result["__p3__"] = dict(self.last_marker)
                return result
            conn._osp_infer = infer_with_marker
        rt.emit(dict(ev="p3_startup", schema="r6p3.config.v1", conn=s.conn, tag=rt.tag,
                     p=s.method.p, seed=s.method.seed, replicate=s.method.replicate,
                     strata=s.method.strata, policy_noise="private SHA256 task/init/replicate/step seed",
                     policy_forward="shared stage1 + full stage2/3 once per anchor",
                     calibrated_from="own candidate-library LOEO in fitted metric codes",
                     extensions={k: v[0] for k, v in EXTENSIONS.items()}))

    def capture_stage1(self, *a, **kw):
        self.stage1 = self.f1(*a, **kw)
        return self.stage1

    def measured(self, key, fn):
        sync()
        t = time.perf_counter()
        value = fn()
        sync()
        self.timings[key] = 1000 * (time.perf_counter() - t)
        return value

    def forward(self, seed):
        import torch
        p = self.p
        if self.fake:
            return self.measured("stage23_ms", lambda: p._p3_policy(seed))
        if self.stage1 is None:
            raise RuntimeError("P3 did not capture the live stage1")
        if self.s.rt.model == "pi05":
            with torch.no_grad():
                self.stage2 = self.measured("stage2_ms", lambda: self.f2(self.stage1))
                if p._stage_config is not None and p._stage_config.needs_relocation:
                    self.stage2 = self.stage2.to(p._stage3_device)
                gen = torch.Generator(device=p._stage3_device).manual_seed(seed)
                self.noise = self.model.sample_noise((1, self.s.rt.H, 32), p._stage3_device, generator=gen)
                self.stage3 = self.measured("stage3_ms", lambda: self.f3(
                    self.stage2, noise=self.noise, num_steps=10, return_intermediates=True))
                chunk = getattr(self.stage3, "action_chunk", self.stage3)
        else:
            with self.runner.session():
                self.stage2 = self.measured("stage2_ms", lambda: self.runner.run_stage2_llm(self.stage1))
                device = self.stage2.backbone_features.device
                gen = torch.Generator(device=device).manual_seed(seed)
                # Avoid runner._noise_dtype's global RNG fork on first use.
                # Its prologue is deterministic in eval mode. All P3 noise is
                # explicit; no process RNG state is reset across connections.
                head = self.runner._model.action_head
                if head.training or int(head.num_inference_timesteps) != 8:
                    raise ValueError("P3 requires eval-mode GR00T with full K8 head")
                processed = head.process_backbone_output(self.runner._head_inputs(self.stage2))
                from openpi.cache.groot.staged import _processed_features
                dtype = _processed_features(processed).dtype
                self.noise = torch.randn((1, self.s.rt.H, 32), device=device, dtype=dtype, generator=gen)
                self.stage3 = self.measured("stage3_ms", lambda: self.runner.run_stage3(self.stage2, noise=self.noise))
                chunk = self.stage3.action_pred
        return array(chunk).reshape(self.s.rt.H, 32)

    def groot_reuse(self, stage1):
        if not self.reuse:
            return self.runner.run_stage2(stage1)
        return replace(self.stage2, action_pred=self.stage3.action_pred)

    def anchor(self):
        s, d = self.s, self.s._dec
        r = copy.deepcopy(s.method.record)
        if r is None or r["assignment"]["step"] != d["step"]:
            raise RuntimeError("missing per-anchor method features")
        baseline_hit = bool(d.get("hit", True))
        a = r["assignment"]
        forward_start = time.perf_counter()
        chunk = array(self.forward(a["policy_seed"]))
        forward_ms = (time.perf_counter() - forward_start) * 1000
        if chunk.shape != (s.rt.H, 32) or not np.isfinite(chunk).all():
            raise ValueError("nonfinite or malformed shadow chunk")
        inject = baseline_hit and a["assigned_call"]
        executed_policy = not baseline_hit or inject
        a.update(eligible=baseline_hit, baseline_hit=baseline_hit, injected=inject,
                 executed_policy=executed_policy, actual_propensity=a["propensity"] if baseline_hit else 1.)
        if inject:
            d.update(hit=False, judge="p3_randomized:CALL")
        self.reuse = executed_policy
        if self.fake:
            self.p._p3_reuse = chunk if executed_policy else None
        cache = array(d["served"])
        diff = cache[:, :7].astype(float) - chunk[:, :7]
        r.update(ev="p3_anchor", uid=s.ep_meta["uid"], attempt=s.ep_meta.get("attempt"),
                 conn=s.conn, tag=s.rt.tag, task_id=s.ep.task_id, init=s.ep.init,
                 step=d["step"], lib=d["lib"], model=s.rt.model, suite=s.rt.suite,
                 cache_chunk=cache, policy_chunk=chunk,
                 distance=dict(rms=float(np.sqrt(np.mean(diff ** 2))),
                    per_step_rms=np.sqrt(np.mean(diff ** 2, axis=1)),
                    commit10_rms=float(np.sqrt(np.mean(diff[:10] ** 2))),
                    per_step_l2=np.linalg.norm(diff, axis=1)))
        r["timing"].update(self.timings, policy_total_ms=forward_ms,
                           stage1_ms=s._s1_ms, retrieval_ms=d["q_us"] / 1000)
        # Ledger uses dec vision/hit. Physical profiling-only cost = stage2/3
        # on CACHE anchors. Selected chunks are already paid ledger MISSes.
        r["cost"] = dict(full_policy_forwards=1, additional_forward_for_injection=0,
                          ledger_miss=int(executed_policy),
                          profiling_only_policy_calls=int(not executed_policy),
                          profiling_only_s23_ms=0. if executed_policy else forward_ms)
        for name, (schema, fn) in EXTENSIONS.items():
            r["extensions"][name] = dict(schema=schema, data=fn(s, copy.deepcopy(r)))
        self.profile = r

    def finish(self, ms, ok, err):
        s, r, d = self.s, self.profile, self.s._dec
        r["timing"]["request_ms"] = ms
        r.update(ok=ok, error=err)
        if ok:
            executed = array(s.b_aex.a[d["step"]])
            expected = r["policy_chunk"] if r["assignment"]["executed_policy"] else r["cache_chunk"]
            if executed.tobytes() != expected.tobytes():
                raise RuntimeError("P3 served chunk differs from selected cache/policy chunk")
            r.update(executed_chunk=executed, served_wire_chunk=array(d["wire_actions"]),
                     executed_head=executed[:5, :7], commit_controls=10)
        s.rt.emit(r)


def install():
    from exp.offline_search.closed_loop import plugin
    if getattr(plugin._ConnPolicy, "_p3_installed", False):
        return
    original = plugin._ConnPolicy.__init__

    def init(self, inner, sessions, bundle_id):
        original(self, inner, sessions, bundle_id)
        enabled = [s for s in sessions if getattr(s.method, "family", None) == "r6_p3_profile"
                   and s.method.enabled]
        if enabled:
            if len(sessions) != 1:
                raise ValueError("P3 needs one CP1 session per connection")
            self._p3_engine = Engine(self)

    plugin._ConnPolicy.__init__ = init
    plugin._ConnPolicy._p3_installed = True
