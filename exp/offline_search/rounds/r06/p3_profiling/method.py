"""Exact A/B delegation plus opt-in profiling. No shared file is modified.

Profile(enabled=False) delegates verbatim. enabled=True installs connection-local
instrumentation through hooks.install(), also when loading a pickled fit.
"""
import copy
import pickle
import time

import numpy as np

from exp.offline_search.harness import api, dims
from exp.offline_search.closed_loop.blind import BlindResult, LookReason, policy_tail_chunk
from exp.offline_search.closed_loop import plugin
from .assignment import assign


def awm_of(method):
    for _ in range(8):
        if hasattr(method, "_dist") and hasattr(method, "tasks"):
            return method
        method = getattr(method, "base", None)
    raise ValueError("profiling needs an AWM/BlindAWM base")


def load_fit(path, spec, kwargs, cell):
    with open(path, "rb") as f:
        b = pickle.load(f)
    if any(b.get(k) != v for k, v in dict(spec=spec, kwargs=kwargs, cell=cell).items()):
        raise ValueError(f"base fit provenance mismatch: {path}")
    if b.get("registered"):
        raise ValueError("registered action libraries need an explicit profile adapter")
    return plugin.clone_method(b["method"])[0]


def loeo_cdf(awm, block=128):
    """Own deployed candidates only, all rows vs other episodes of same task.

    Float64 distances between the stored float32 metric codes; no evaluation
    trajectory or success labels. Sorted samples give an empirical midrank CDF.
    """
    result = {}
    for task, T in awm.tasks.items():
        codes = {"main": np.asarray(T.Z, np.float64)}
        if awm.early:
            early = np.asarray(T.Z0, np.float64) if T.Z0 is not None else codes["main"] @ T.A0
            if T.Z0 is None and T.As0 is not None:
                early = early + np.asarray(T.RS, float) @ T.As0
            codes["early"] = early
        for regime, z in codes.items():
            eps = awm.lib_ep[T.rows]
            if len(np.unique(eps)) < 2:
                raise ValueError("LOEO needs two library episodes per task")
            z2 = (z * z).sum(1)
            ds = []
            for start in range(0, len(z), block):
                zz = z[start:start + block]
                d2 = (zz * zz).sum(1)[:, None] + z2 - 2 * (zz @ z.T)
                d2[eps[start:start + block, None] == eps[None, :]] = np.inf
                ds.extend(np.sqrt(np.maximum(d2.min(1), 0)).tolist())
            result[int(task), regime] = np.sort(np.asarray(ds, np.float64))
            result[int(task), regime].flags.writeable = False
    return result


class Profile(api.Method):
    tier = "T1"
    family = "r6_p3_profile"
    uses_gt = False
    uses_nonlibrary_action = False  # inference belongs to the plugin, as for B

    def __init__(self, base_spec, base_kwargs=None, base_fit="", enabled=False,
                 p=0.1, seed=603, replicate=0, guard_spec="", guard_kwargs=None,
                 guard_fit="", strata=None):
        self.base_spec, self.base_kwargs, self.base_fit = base_spec, base_kwargs or {}, base_fit
        if type(enabled) is not bool:
            raise ValueError("enabled must be a JSON boolean")
        self.enabled, self.p, self.seed, self.replicate = enabled, float(p), seed, replicate
        assign(seed, 0, 0, replicate, 0, self.p)
        self.guard_spec, self.guard_kwargs, self.guard_fit = guard_spec, guard_kwargs or {}, guard_fit
        # Optional pre-treatment d1-quantile bins, portable across action/state scales.
        self.strata = strata
        if strata is not None:
            if set(strata) != {"edges", "p"} or len(strata["p"]) != len(strata["edges"]) + 1:
                raise ValueError("strata = {edges: sorted interior quantiles, p: probabilities}")
            if list(strata["edges"]) != sorted(set(strata["edges"])) or any(not 0 < x < 1 for x in strata["edges"]):
                raise ValueError("stratum edges must be strictly increasing in (0,1)")
            for p0 in strata["p"]:
                assign(seed, 0, 0, replicate, 0, p0)
        cls, _ = plugin.load_method_class(base_spec)
        self.base = cls(**self.base_kwargs)
        self.name = "P3__" + self.base.name
        self.guard = None
        self.record = None
        self.tail_anchor = None

    def fit(self, lib, ctx):
        if self.base_fit:
            self.base = load_fit(self.base_fit, self.base_spec, self.base_kwargs, ctx.cell)
        else:
            self.base.prof = api.NULL_PROFILER
            self.base.fit(lib, ctx)
        self.tier = self.base.tier
        if not self.enabled:
            return
        if not self.guard_spec and not hasattr(self.base, "guards"):
            raise ValueError("superset profiling of A requires a shadow B guard_spec/guard fit")
        awm = awm_of(self.base)
        if (awm.serving, awm.budget, awm.gates) != ("anchor_tail", 1, "budget_only"):
            raise ValueError("P3 requires A/B's 10-step anchor_tail budget=1 budget_only")
        self.cdf = loeo_cdf(awm)
        deployed = lib if awm.cand_name == "current" else ctx.open_library(awm.cand_name)
        self.ep_len = np.array(deployed.ep_len, np.int32)
        self.ep_len.flags.writeable = False
        if self.guard_spec:
            if self.guard_fit:
                self.guard = load_fit(self.guard_fit, self.guard_spec, self.guard_kwargs, ctx.cell)
            else:
                cls, _ = plugin.load_method_class(self.guard_spec)
                self.guard = cls(**self.guard_kwargs)
                self.guard.prof = api.NULL_PROFILER
                self.guard.fit(lib, ctx)

    def bytes_per_entry(self):
        return self.base.bytes_per_entry()

    def reset(self, episode):
        self.base.reset(episode)
        if self.guard is not None:
            self.guard.reset(episode)
        self.record = self.tail_anchor = None

    def invalidate_anchor(self):
        self.base.invalidate_anchor()
        if self.guard is not None:
            self.guard.invalidate_anchor()

    def blind_step(self, bq):
        return self.base.blind_step(bq)

    def policy_tail_step(self, bq):
        if hasattr(self.base, "policy_tail_step"):
            return self.base.policy_tail_step(bq)
        a, self.tail_anchor = self.tail_anchor, None
        if (a is None or bq.prev_hit is not False or bq.blind_age != 0
                or a["step"] != bq.step - 1 or a["episode"] != bq.episode.uid
                or a["task"] != bq.task_id or getattr(bq, "executed_steps", 5) != 5
                or len(bq.hist_hit) != bq.step or len(bq.hist_has_vision) != bq.step
                or not bq.hist_has_vision[-1] or not np.isfinite(bq.rs).all()
                or not np.isfinite(bq.raw_state).all()):
            return LookReason(6, "p3_policy_tail_lifecycle")
        return BlindResult(policy_tail_chunk(bq.prev_a_exec), a["rows"].copy(),
                           a["weights"].copy(), awm_of(self.base).cand_name,
                           {"policy_tail": 1., "policy_anchor_step": float(a["step"])})

    def query(self, q):
        res = self.base.query(q)
        if not self.enabled:
            return res
        t0 = time.perf_counter()
        awm = awm_of(self.base)
        self.tail_anchor = copy.deepcopy(awm._anchor)
        # Read-only re-evaluation obtains unsquashed internals without changing
        # AWM's synthesis, tie selection, gripper memo or anchor.
        T, _, regime, _, _, xv, rs, d, med, continuity, ranked = awm._dist(q)
        rows = np.asarray(res.topk, np.int64)
        pos = np.searchsorted(T.rows, rows)
        if not np.array_equal(T.rows[pos], rows):
            raise ValueError("task row index is not sorted")
        anchor = awm._anchor
        weights = anchor["weights"]
        chunks = np.asarray(awm.act[rows, :, :7], np.float64)
        center = np.tensordot(weights.astype(float), chunks, 1)
        dispersion = np.sqrt(np.sum(weights[:, None] * np.mean((chunks - center) ** 2, axis=2), axis=0))
        metric = "early" if regime == 0 and awm.early else "main"
        samples = self.cdf[int(q.task_id), metric]
        d1 = float(d.min())
        quantile = float((np.searchsorted(samples, d1, "left") + np.searchsorted(samples, d1, "right")) / (2 * len(samples)))
        guard_res = self.guard.query(q) if self.guard is not None else res
        guard = self.guard if self.guard is not None else self.base
        # Preserve every guard extra, without plugin's scalar truncation.
        guards = dict(guard_res.extras or {})
        thresholds = {k: getattr(guard, k) for k in ("m_thr", "c_thr", "stuck_thr", "lag_thr", "noprog_n", "prog_eps") if hasattr(guard, k)}
        hv = np.asarray(getattr(q, "hist_has_vision", []), bool)
        prev = np.flatnonzero(hv)
        cosines = None
        if len(prev) and hasattr(guard, "M0"):
            from exp.offline_search.rounds.r03.h3_judge.judge import core
            j = int(prev[-1])
            cosines = [core.centred_cos(q.key_v0, q.hist_key_v0[j], guard.M0[q.task_id]),
                       core.centred_cos(q.key_v1, q.hist_key_v1[j], guard.M1[q.task_id])]
        p, stratum = self.p, 0
        if self.strata is not None:
            stratum = int(np.searchsorted(self.strata["edges"], quantile, side="right"))
            p = self.strata["p"][stratum]
        assignment = assign(self.seed, int(q.task_id), int(q.episode.init), self.replicate, int(q.step), p)
        assignment["stratum"] = stratum
        feat = awm.feat0 if metric == "early" else awm.features
        code = np.concatenate([xv, rs]) if feat == "joint" else xv
        code = code @ T.W0f - T.c0 if metric == "early" else code @ T.Wf - T.shift
        self.record = dict(schema="r6p3.anchor.v1", assignment=assignment,
            retrieval=dict(rows=rows, distances=d[pos], ranked_distances=ranked[pos],
                nearest_distances=np.sort(d)[:16], weights=weights, episodes=awm.lib_ep[rows],
                progress=awm.lib_step[rows] / np.maximum(self.ep_len[rows] - 1, 1),
                steps=awm.lib_step[rows], phase_rows=anchor["phase"],
                phase_steps=awm.lib_step[anchor["phase"]], metric_code=code, metric=metric,
                d1=d1, d1_loeo_quantile=quantile, loeo_n=len(samples), regime=regime,
                distance_median=med, continuity=None if continuity is None else continuity[pos],
                dispersion_per_step=dispersion, dispersion_rms=float(np.sqrt(np.mean(dispersion ** 2)))),
            guards=dict(inputs_outputs=guards, thresholds=thresholds, endpoint_cosines=cosines,
                dense_motion=None if not q.step else float(np.linalg.norm(rs - dims.valid_state(q.hist_rs[-1], awm.model))),
                shadow_only=self.guard is not None),
            state=dict(normalized=np.asarray(q.rs), raw=np.asarray(q.raw_state)),
            timing=dict(features_ms=(time.perf_counter() - t0) * 1000), extensions={})
        return res


# Installation affects only connections whose fitted method is enabled Profile.
from .hooks import install
install()
