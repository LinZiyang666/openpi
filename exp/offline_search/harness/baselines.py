"""Fixed reference methods (protocol §4.3). Run them like any method, e.g.

    python -m exp.offline_search.harness.run --method exp.offline_search.harness.baselines:B0Current --out ...

B0Current   online weighted-score-sum kNN (per-field cosine / -L2 -> 0.5*(tanh((x-mu)/sigma)+1) -> sum w*n), task-
            filtered; reproduces the recorded top-1 (float32 arithmetic mirrors the torch backend).
B1LDA       same formula, LDA weights from exp/weighted_sum/config/fusion_ablation/manifest.json.
B2Random    uniform random candidate within the task (episode-seeded), random confidence. Lower bound.
B3Oracle    argmin err over the task candidates. Uses GT (uses_gt=True): reference only, the regret budget.
B4Single    one field only: kwargs {"field": "v0" | "v1" | "rs"} -> names B4_v0 / B4_v1 / B4_rs; confidence = that
            field's raw similarity of the top-1 (cosine, or -L2 for rs).

Run every reference method over a store (one results dir, scoreboard rows appended):
    python -m exp.offline_search.harness.baselines --root <store> --out exp/offline_search/results/r00 [--workers <#CPUs in affinity mask>]
        [--cells all] [--only B0Current,B3Oracle]
"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__":      # CLI: pin threads / hide GPUs before numpy loads (importers are left alone)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = "1"

import numpy as np  # noqa: E402

from . import api, dims, metrics  # noqa: E402

_EPS = np.float32(1e-8)


class _Block:
    __slots__ = ("rows", "V0", "n0", "V1", "n1", "RS", "A")


def _task_blocks(lib, fields=("v0", "v1", "rs"), with_action=False, sigma=None) -> dict:
    """Per-task candidate matrices in library order (float32, contiguous) + float32 row norms (as torch)."""
    blocks = {}
    for t in lib.tasks():
        b = _Block()
        b.rows = lib.rows_of_task(t)
        if "v0" in fields:
            b.V0 = np.ascontiguousarray(lib.key_v0[b.rows], dtype=np.float32)
            b.n0 = np.linalg.norm(b.V0, axis=1).astype(np.float32)
        if "v1" in fields:
            b.V1 = np.ascontiguousarray(lib.key_v1[b.rows], dtype=np.float32)
            b.n1 = np.linalg.norm(b.V1, axis=1).astype(np.float32)
        if "rs" in fields:
            b.RS = np.ascontiguousarray(lib.rs[b.rows], dtype=np.float32)
        if with_action:
            b.A = metrics.seg(lib.action[b.rows])
        blocks[t] = b
    return blocks


def _cos(V, n, q):
    q = np.asarray(q, np.float32)
    qn = np.float32(np.sqrt(np.dot(q, q)))
    return (V @ q) / np.maximum(n * qn, _EPS)


def _l2(RS, q):
    d = RS - np.asarray(q, np.float32)
    return np.sqrt(np.einsum("ij,ij->i", d, d))


def _zt(x, mu, sigma):
    """ZScoreNormalizer: 0.5 * (tanh((x - mu) / sigma) + 1) in float32 (x already oriented; L2 -> -distance)."""
    sigma = sigma if sigma > 1e-12 else 1.0
    return np.float32(0.5) * (np.tanh((x - np.float32(mu)) / np.float32(sigma)) + np.float32(1.0))


class FusedKNN(api.Method):
    """Weighted score sum over (vision_0 cosine, vision_1 cosine, robot_state -L2) with zscore+tanh normalizers."""

    tier = "T0"
    family = "baseline"

    def __init__(self, weights="current", name=None, extras=True):
        self.weights_src = weights
        self.name = name or {"current": "B0_current", "lda": "B1_lda"}.get(weights if isinstance(weights, str) else "",
                                                                           "fused_custom")
        self.want_extras = bool(extras)

    def fit(self, lib, ctx):
        p = ctx.current_params()
        if self.weights_src == "current":
            w = p["weights"]
        elif self.weights_src == "lda":
            w = ctx.lda_weights()
            if w is None:
                raise api.SkipCell(f"no LDA weights for {ctx.model} {ctx.suite} in the fusion_ablation manifest")
        else:
            w = tuple(float(x) for x in self.weights_src)
        self.w = tuple(float(x) for x in w)
        self.mu, self.sg = p["mu"], p["sigma"]
        self.blocks = _task_blocks(lib)
        self.rs_dim = lib.rs.shape[1]

    def score_all(self, q):
        """(rows, fused f32[c], (cos0, cos1, dist, s0, s1, s2)) over the task candidates of q."""
        b = self.blocks[q.task_id]
        with self.prof.section("sim_v0"):
            c0 = _cos(b.V0, b.n0, q.key_v0)
        with self.prof.section("sim_v1"):
            c1 = _cos(b.V1, b.n1, q.key_v1)
        with self.prof.section("sim_rs"):
            d = _l2(b.RS, q.rs)
        with self.prof.section("fuse"):
            s0 = _zt(c0, self.mu[0], self.sg[0])
            s1 = _zt(c1, self.mu[1], self.sg[1])
            s2 = _zt(-d, self.mu[2], self.sg[2])
            f = np.zeros(c0.shape[0], np.float32)       # torch: final_scores = zeros(n); += w * s (field order)
            f += self.w[0] * s0
            f += self.w[1] * s1
            f += self.w[2] * s2
        return b.rows, f, (c0, c1, d, s0, s1, s2)

    def query(self, q):
        rows, f, parts = self.score_all(q)
        with self.prof.section("rank"):
            o = np.argsort(-f, kind="stable")[: api.TOPK_SAVE]
        ex = None
        if self.want_extras:
            j = o[0]
            c0, c1, d, s0, s1, s2 = parts
            ex = {"cos_v0": float(c0[j]), "cos_v1": float(c1[j]), "dist_rs": float(d[j]), "n_v0": float(s0[j]),
                  "n_v1": float(s1[j]), "n_rs": float(s2[j]),
                  "margin": float(f[o[0]] - f[o[1]]) if o.size > 1 else float("nan")}
        return api.Result(topk=rows[o], scores=f[o], confidence=float(f[o[0]]), extras=ex)

    def bytes_per_entry(self):
        return float(2 * dims.KEY_DIM * 4 + self.rs_dim * 4)


class B0Current(FusedKNN):
    def __init__(self, extras=True):
        super().__init__("current", "B0_current", extras)


class B1LDA(FusedKNN):
    def __init__(self, extras=True):
        super().__init__("lda", "B1_lda", extras)


class B2Random(api.Method):
    name = "B2_random"
    tier = "T0"
    family = "baseline"

    def fit(self, lib, ctx):
        self.rows = {t: lib.rows_of_task(t) for t in lib.tasks()}

    def reset(self, episode):
        self.rng = episode.rng

    def query(self, q):
        r = self.rows[q.task_id]
        o = self.rng.permutation(r.size)[: api.TOPK_SAVE]
        conf = float(self.rng.random())
        return api.Result(topk=r[o], scores=np.linspace(1.0, 0.0, o.size), confidence=conf)

    def bytes_per_entry(self):
        return 0.0


class B3Oracle(api.Method):
    """argmin err against GT over the task candidates -- uses GT, reference only (regret budget)."""

    name = "B3_oracle"
    tier = "T0"
    family = "baseline"
    uses_gt = True

    def fit(self, lib, ctx):
        self.sigma = ctx.action_sigma
        self.blocks = _task_blocks(lib, fields=(), with_action=True)

    def query(self, q):
        b = self.blocks[q.task_id]
        e = metrics.oracle_one(metrics.seg(q._reference_gt()), b.A, self.sigma)
        o = np.argsort(e, kind="stable")[: api.TOPK_SAVE]
        return api.Result(topk=b.rows[o], scores=-e[o], confidence=float(-e[o[0]]))

    def bytes_per_entry(self):
        return 0.0


class B4Single(api.Method):
    tier = "T0"
    family = "baseline"

    def __init__(self, field="v0"):
        if field not in ("v0", "v1", "rs"):
            raise ValueError(f"B4Single field must be v0, v1 or rs, got {field!r}")
        self.field = field
        self.name = f"B4_{field}"

    def fit(self, lib, ctx):
        self.blocks = _task_blocks(lib, fields=(self.field,))
        self.rs_dim = lib.rs.shape[1]

    def query(self, q):
        b = self.blocks[q.task_id]
        with self.prof.section("sim"):
            if self.field == "v0":
                s = _cos(b.V0, b.n0, q.key_v0)
            elif self.field == "v1":
                s = _cos(b.V1, b.n1, q.key_v1)
            else:
                s = -_l2(b.RS, q.rs)
        o = np.argsort(-s, kind="stable")[: api.TOPK_SAVE]
        return api.Result(topk=b.rows[o], scores=s[o], confidence=float(s[o[0]]))

    def bytes_per_entry(self):
        return float(self.rs_dim * 4 if self.field == "rs" else dims.KEY_DIM * 4)


class B4V0(B4Single):
    def __init__(self):
        super().__init__("v0")


class B4V1(B4Single):
    def __init__(self):
        super().__init__("v1")


class B4RS(B4Single):
    def __init__(self):
        super().__init__("rs")


ALL_BASELINES = ("B0Current", "B1LDA", "B2Random", "B3Oracle", "B4V0", "B4V1", "B4RS")


# ----------------------------------------------------------------------------- gate helper (not a method)
def fused_for_rows(model, suite, qv0, qv1, qrs, lib, rows, weights=None):
    """Vectorized online fused score of (query i, library row rows[i]) pairs, float32 as the online backend.
    Used by the gates to score the recorded top-1 without giving B0 access to it."""
    from .store import current_params

    p = current_params(model, suite)
    w = p["weights"] if weights is None else weights
    rows = np.asarray(rows, np.int64)
    L0 = np.asarray(lib.key_v0[rows], np.float32)
    L1 = np.asarray(lib.key_v1[rows], np.float32)
    LR = np.asarray(lib.rs[rows], np.float32)
    q0 = np.asarray(qv0, np.float32)
    q1 = np.asarray(qv1, np.float32)
    qr = np.asarray(qrs, np.float32)
    c0 = np.einsum("ij,ij->i", L0, q0) / np.maximum(np.linalg.norm(L0, axis=1) * np.linalg.norm(q0, axis=1), _EPS)
    c1 = np.einsum("ij,ij->i", L1, q1) / np.maximum(np.linalg.norm(L1, axis=1) * np.linalg.norm(q1, axis=1), _EPS)
    d = np.sqrt(np.einsum("ij,ij->i", LR - qr, LR - qr))
    s = (_zt(c0, p["mu"][0], p["sigma"][0]), _zt(c1, p["mu"][1], p["sigma"][1]), _zt(-d, p["mu"][2], p["sigma"][2]))
    f = np.zeros(rows.shape[0], np.float32)
    for wi, si in zip(w, s):
        f += wi * si
    return f, np.stack(s, 1)


def main(argv=None):
    import argparse

    from . import run, store

    ap = argparse.ArgumentParser(description="run the reference methods B0..B4")
    ap.add_argument("--root", default=str(store.FULL_ROOT))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--cells", default="all")
    ap.add_argument("--only", default=",".join(ALL_BASELINES))
    ap.add_argument("--scoreboard", default="default")
    ap.add_argument("--profile", action="store_true")
    a = ap.parse_args(argv)
    bad = 0
    for cls in a.only.split(","):
        r = run.run(f"exp.offline_search.harness.baselines:{cls.strip()}", {}, a.cells, root=a.root, out=a.out,
                    workers=a.workers, scoreboard=a.scoreboard, profile=a.profile)
        bad += bool(r["errors"])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
