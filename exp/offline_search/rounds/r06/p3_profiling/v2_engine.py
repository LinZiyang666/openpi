"""V2 connection-local instrumentation, with observational blind probes.

The live key builder/orchestrator/history are never called by a blind probe.
Profiling stage counters are separate from the deployment decision ledger.
"""
import copy
import hashlib
import time
from pathlib import Path

import numpy as np

from .hooks import Engine as V1Engine, Proxy, array
from .assignment import seed_for
from .design import uniform
from .snapshots import atomic_npz


class Engine(V1Engine):
    def __init__(self, conn):
        super().__init__(conn)
        self.last_record = None
        self.last_anchor = None
        self.last_uid = None
        self.stage_counts = dict(stage1=0, stage2=0, stage3=0)
        def counted(name, fn):
            def call(*args, **kw):
                self.stage_counts[name] += 1
                return fn(*args, **kw)
            return call
        if not self.fake:
            self.f1 = counted("stage1", self.f1)
            if self.s.rt.model == "pi05":
                self.f2 = counted("stage2", self.f2)
                self.f3 = counted("stage3", self.f3)
            else:
                self.runner = Proxy(self.runner,
                    run_stage2_llm=counted("stage2", self.runner.run_stage2_llm),
                    run_stage3=counted("stage3", self.runner.run_stage3))
        orch = getattr(conn._osp_adapter, "orch", None)
        if orch is not None:
            kb = orch._real.key_builder
            orch._real.key_builder = Proxy(kb, build=lambda *a, **kw: self.measured("key_build_ms", lambda: kb.build(*a, **kw)))
        if hasattr(self.s, "_verdict"):
            verdict = self.s._verdict
            self.s._verdict = lambda *a, **kw: self.measured("verdict_ms", lambda: verdict(*a, **kw))
        self.s.rt.emit(dict(ev="p3_v2_startup", schema="r6p3.config.v2", conn=self.s.conn,
            tag=self.s.rt.tag, design=self.s.method.design_config,
            p=getattr(self.s.method, "nominal_p", self.s.method.p), seed=self.s.method.seed,
            replicate=self.s.method.replicate, strata=self.s.method.strata,
            resample_p=self.s.method.resample_p, resample_draws=self.s.method.resample_draws,
            blind_shadow=self.s.method.blind_shadow, catalog_sha256=self.s.method.catalog_sha256,
            catalog=self.s.method.catalog,
            implementation_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
                Path(__file__), Path(__file__).with_name("v2.py"), Path(__file__).with_name("design.py"),
                Path(__file__).with_name("method.py"), Path(__file__).with_name("hooks.py"),
                Path(__file__).resolve().parents[3] / "closed_loop/plugin.py")}))
        # after_infer sees the complete decision before stock clears _dec/obs.
        old_after = self.s.after_infer
        def after(ms, ok, err=None):
            s, d = self.s, self.s._dec
            self.last_log_ok = False
            try:
                if d is not None and ok:
                    self.decision(d)
                old_after(ms, ok, err)
                if d is not None:
                    self.last_marker.update(p3_version=2, source=d.get("source", "cache" if d.get("hit", True) else "policy"),
                        parent_anchor=self.last_anchor, commit_controls=s.method.commit_controls,
                        catalog_sha256=s.method.catalog_sha256)
            finally:
                self.stage_counts = dict(stage1=0, stage2=0, stage3=0)
                if not self.last_log_ok:
                    self.stage1 = self.stage2 = self.stage3 = self.noise = self.profile = None
                    self.reuse = False
        self.s.after_infer = after

    def anchor(self):
        super().anchor()  # one shadow s2/3; v1 shared-stage1 interception
        s, r, d = self.s, self.profile, self.s._dec
        a = s.method.design.resolve(r["assignment"], r["assignment"]["baseline_hit"])
        r["assignment"] = a
        call = a["executed_policy"]
        d.update(hit=not call, judge="p3_v2:" + a["override"] + (":CALL" if call else ":CACHE"))
        self.reuse = call
        if self.fake:
            self.p._p3_reuse = r["policy_chunk"] if call else None
        s.method.commit_controls = a["commit_controls"]
        r["cost"].update(ledger_miss=int(call), profiling_only_policy_calls=int(not call),
                         profiling_only_s23_ms=0. if call else r["timing"]["policy_total_ms"])
        self.last_anchor = d["step"]
        # Every selection RNG domain is independent of action sampling/source.
        u = uniform(s.method.seed, s.ep.task_id, s.ep.init, s.method.replicate, d["step"], "resample_select")
        r["resampling"] = dict(selection_p=s.method.resample_p, selection_u=u, selected=u < s.method.resample_p,
                               seeds=[a["policy_seed"]], extra_chunks=[])
        if u < s.method.resample_p:
            st2, st3, noise, reuse = self.stage2, self.stage3, self.noise, self.reuse
            t = time.perf_counter()
            try:
                self.reuse = False
                for j in range(1, s.method.resample_draws):
                    seed = seed_for(s.method.seed, s.ep.task_id, s.ep.init, s.method.replicate, d["step"], f"resample_{j}")
                    r["resampling"]["seeds"].append(seed)
                    r["resampling"]["extra_chunks"].append(self.draw_head(seed))
            finally:
                self.stage2, self.stage3, self.noise, self.reuse = st2, st3, noise, reuse
            r["resampling"]["extra_head_ms"] = (time.perf_counter() - t) * 1000
        samples = [r["policy_chunk"], *r["resampling"]["extra_chunks"]]
        r["resampling"]["dispersion_per_step"] = np.std(np.asarray(samples)[:, :, :7], axis=0).mean(1) if len(samples) > 1 else None
        r["cost"].update(extra_stage3_forwards=len(samples) - 1, extra_stage2_forwards=0)
        # Keep both alternative output-transformed chunks for future branches.
        _, state = self.conn._osp_adapter.prepare(s.cur_obs)
        r["candidate_wire"] = {key: array(self.conn._osp_adapter.output(r[key + "_chunk"], state)["actions"])
                               for key in ("cache", "policy")}
        r["provenance"] = dict(catalog_sha256=s.method.catalog_sha256,
            chunk_sha256={key: hashlib.sha256(r[key + "_chunk"].tobytes()).hexdigest() for key in ("cache", "policy")},
            policy_state_contract="feed-forward stages; explicit per-decision private noise; no shadow broadcast/history push",
            split="calibration" if s.ep.init % s.method.design_config["split_modulus"] == 0 else "validation",
            run_block=s.method.replicate, accepted_attempt_required=True)

    def draw_head(self, seed):
        import torch
        if self.fake:
            return array(self.p._p3_policy(seed))
        if self.s.rt.model == "pi05":
            with torch.no_grad():
                gen = torch.Generator(device=self.p._stage3_device).manual_seed(seed)
                noise = self.model.sample_noise((1, self.s.rt.H, 32), self.p._stage3_device, generator=gen)
                out = self.f3(self.stage2, noise=noise, num_steps=10, return_intermediates=True)
                return array(getattr(out, "action_chunk", out)).reshape(self.s.rt.H, 32)
        with self.runner.session():
            gen = torch.Generator(device=self.noise.device).manual_seed(seed)
            noise = torch.randn(self.noise.shape, device=self.noise.device, dtype=self.noise.dtype, generator=gen)
            return array(self.runner.run_stage3(self.stage2, noise=noise).action_pred).reshape(self.s.rt.H, 32)

    def blind_probe(self, obs, seed):
        s, p = self.s, self.p
        if self.fake:
            # Fixture uses the very same recorded observation row as the live
            # call, without invoking the fake orchestrator/search/stage counter.
            old = p.kb.row
            try:
                p.kb.row = int(obs["_row"])
                keys = p.kb.qc if hasattr(p.kb, "qc") else p.kb.q
                vectors = {"vision_0": np.asarray(keys.key_v0[p.kb.row]),
                           "vision_1": np.asarray(keys.key_v1[p.kb.row]),
                           "robot_state": np.asarray(keys.rs[p.kb.row])}
                return array(p._p3_policy(seed)), vectors
            finally:
                p.kb.row = old
        import torch
        from openpi.cache.types import CheckpointID
        # COPY the builder; no cache/state latch on the live builder is touched.
        kb = copy.copy(s.kb)
        kb._cache = {}
        if hasattr(kb, "_state_index") and kb._state_index is not None:
            kb._state_index = kb._state_index.clone()
        counted, s1_ms = s.stage1_calls, s._s1_ms
        try:
            if s.rt.model == "pi05":
                import jax
                from openpi.models import model
                inputs = p._input_transform(dict(obs))
                if p._coordinator is None:
                    inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(p._pytorch_device)[None, ...], inputs)
                    observation = model.Observation.from_dict(inputs)
                else:
                    observation = jax.tree.map(lambda x: torch.from_numpy(np.array(x)) if not torch.is_tensor(x) else x, inputs)
                with torch.no_grad():
                    self.stage1 = self.measured("blind_stage1_ms", lambda: self.f1(observation))
            else:
                from exp.libero_groot.policy_adapter import build_groot_observation
                from openpi.cache.groot.interceptor import _unsqueeze_values
                inputs = _unsqueeze_values(build_groot_observation(obs))
                inputs = {k: v if isinstance(v, np.ndarray) else np.array(v) for k, v in inputs.items()}
                norm = p._policy.apply_transforms(inputs)
                with self.runner.session():
                    self.stage1 = self.measured("blind_stage1_ms", lambda: self.f1(norm))
            def build_keys():
                kb.collect(CheckpointID.CP1, stage1=self.stage1)
                return kb.build(CheckpointID.CP1)
            vectors = {k: array(v).reshape(-1) for k, v in self.measured("blind_key_build_ms", build_keys).items()}
            return self.forward(seed), vectors
        finally:
            s.stage1_calls, s._s1_ms = counted, s1_ms
            kb.clear()

    def decision(self, d):
        s, m, step = self.s, self.s.method, d["step"]
        uid = s.ep_meta["uid"]
        if uid != self.last_uid:
            self.last_uid = uid
            self.last_anchor = step if d.get("vision", True) else None
        vision = bool(d.get("vision", True))
        seed = seed_for(m.seed, s.ep.task_id, s.ep.init, m.replicate, step, "policy")
        t = time.perf_counter()
        if vision:
            vectors = dict(vision_0=s.b_v0.a[step], vision_1=s.b_v1.a[step], robot_state=s.b_rs.a[step])
            shadow = self.profile["policy_chunk"]
        elif m.blind_shadow:
            shadow, vectors = self.blind_probe(s.cur_obs, seed)
        else:
            shadow, vectors = None, {}
        executed = array(s.b_aex.a[step])
        source = d.get("source", "cache" if d.get("hit", True) else "policy")
        key = hashlib.sha256(uid.encode()).hexdigest()[:24]
        rel = Path("p3_inputs") / f"{key}_a{s.ep_meta.get('attempt', 1)}" / f"step_{step:06d}.npz"
        values = {k: np.array(v, copy=True) for k, v in vectors.items()}
        values.update(raw_state=s.b_raw.a[step], executed_chunk=executed,
                      wire_chunk=np.asarray(d["wire_actions"]), policy_chunk=shadow if shadow is not None else np.empty((0, 32)))
        if vision:
            from .method import awm_of
            values["neighbour_chunks"] = np.asarray(awm_of(m.base).act[self.profile["retrieval"]["rows"]])
        blind_features = None
        if not vision and shadow is not None:
            from exp.offline_search.closed_loop import plugin
            clone, _ = plugin.clone_method(m)
            view = plugin.OnlineQueryView(s, step, s.ep.task_id, s.ep)
            view = Proxy(view, key_v0=vectors["vision_0"], key_v1=vectors["vision_1"], rs=vectors["robot_state"])
            start = time.perf_counter()
            proposal = clone.query(view)
            blind_features = {k: clone.record[k] for k in ("retrieval", "guards", "calibration")}
            blind_features["scope"] = "isolated diagnostic reobservation; live history/cadence unchanged"
            blind_features["query_ms"] = (time.perf_counter() - start) * 1000
            values["diagnostic_cache_chunk"] = np.asarray(proposal.action)
            del clone
        atomic_npz(Path(s.rt.opts.os_log_dir) / rel, values)
        row = dict(ev="p3_decision", schema="r6p3.decision.v2", uid=uid, attempt=s.ep_meta.get("attempt"),
            tag=s.rt.tag, conn=s.conn, step=step, task_id=s.ep.task_id, init=s.ep.init,
            vision=vision, parent_anchor=self.last_anchor, source=source, policy_seed=seed,
            input_archive=str(rel), input_sha256=None, shadow_available=shadow is not None,
            shadow_is_profiler_only=not vision, controller_observes_shadow=False,
            full_policy_forwards=1 if shadow is not None else 0,
            timing=dict(self.timings), input_and_blind_probe_ms=(time.perf_counter() - t) * 1000,
            executed_vs_shadow_per_step=None if shadow is None else np.sqrt(np.mean((executed[:, :7] - shadow[:, :7]) ** 2, axis=1)),
            commit_controls=m.commit_controls, catalog_sha256=m.catalog_sha256)
        row["blind_features"] = blind_features
        extra = len(self.profile["resampling"]["extra_chunks"]) if vision else 0
        if not self.fake:
            want = dict(stage1=int(shadow is not None), stage2=int(shadow is not None),
                        stage3=int(shadow is not None) + extra)
            if self.stage_counts != want:
                raise RuntimeError(f"unexpected physical stage invocation count: {self.stage_counts} != {want}")
        row["stage_invocations"] = dict(self.stage_counts) if not self.fake else None
        row["stage_invocations_scope"] = "per-request dispatch; coordinator batches may share physical kernels"
        if self.profile is not None:
            self.profile["input_archive"] = str(rel)
        self.last_record = row
        s.rt.emit(row)

    def finish(self, ms, ok, err):
        s, r, d = self.s, self.profile, self.s._dec
        r["timing"]["request_ms"] = ms
        r.update(ok=ok, error=err)
        if ok:
            executed = array(s.b_aex.a[d["step"]])
            expected = r["policy_chunk"] if r["assignment"]["executed_policy"] else r["cache_chunk"]
            if executed.tobytes() != expected.tobytes():
                raise RuntimeError("V2 selected chunk mismatch")
            r.update(executed_chunk=executed, served_wire_chunk=array(d["wire_actions"]),
                     executed_head=executed[:5, :7], commit_controls=s.method.commit_controls)
        s.rt.emit(r)
