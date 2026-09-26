"""R1 family F3 "B0-plus": keep B0's ranking (harness.baselines.FusedKNN, bit-identical fused scores), change what
is done with it. Shared machinery in f3_core.py (see its docstring for the library-fitted scales).

M4 B0TopkConsensus        B0 top-k -> synthesis: mean | kernel (w ~ exp(-(S1 - Si)/sd_S)) | med (per-step median of
                          dims 0..5 + gripper majority, dims 7.. from top-1). Confidence = B0 fused score of top-1.
M5 B0ShortlistContRerank  B0 top-10; after a MISS (q.prev_hit False) at step >= 1 re-order by rank_B0 + lam*rank_cont,
                          cont = RMS_sigma(head[:5,:7] - prev_a_exec[5:10,:7]) (lam = inf: pure continuity order);
                          otherwise B0 order. Synthesis med over the first k. Confidence = B0 fused score of the pick.
M6 ConsistencyConfidence  base selector (b0 top-1 | m4k5med | m5l2k3) + conf = -w1*cont/s_c - w2*disp_k/s_a + w3*g,
                          g = (B0 score of the pick - mu_S)/sd_S. weights "equal": (1,1,1); "split": (1,1,1) after a
                          MISS, (0,1,1) after a HIT / unknown. Step 0 always drops continuity. A dropped continuity term
                          is replaced by its library-typical value 1 (s_c is the LOEO median), so regimes stay on one
                          scale. veto: flip10 (fraction of B0's top-10 whose gripper flips vs prev_a_exec[4,6]) >= tau
                          -> conf -= 1000.
M8 B0BigLibCons           B0's exact formula (same yaml mu/sigma/weights) against the big library (bpool_cs pi0.5 /
                          bpool_all GR00T); synthesis top1 | mean-k. Keys are zero-copy views of the shared memmap.

Every method returns Result.extras (see f3_core.B0Plus.base_extras): regime (0 step0 / 1 after MISS / 2 after HIT /
3 unknown), b0_top1, b0_pick, rank_b0, margin, cos_v0, cos_v1, dist_rs, cont0 (pick), cont_b0 (B0 top-1), cont_act
(returned action), disp_set, disp5_b0, flip10, pick_ep, pick_step, k (+ w_top1, k_eff for kernel; t_cont, t_disp,
t_g, veto for M6).
"""
from __future__ import annotations

try:
    from . import f3_core as core
except ImportError:                      # loaded by file path (harness.run.load_method_class puts this dir on sys.path)
    import f3_core as core

VETO = 1000.0
SYNTHS = ("top1", "mean", "kernel", "med")


class B0TopkConsensus(core.B0Plus):
    """M4 b0_topk_consensus."""

    def __init__(self, k=5, synth="mean"):
        if synth not in ("mean", "kernel", "med"):
            raise ValueError(f"synth must be mean | kernel | med, got {synth!r}")
        self.k = int(k)
        self.how = synth
        super().__init__(f"M4_b0cons_k{self.k}_{synth}")
        self.need_libstats = synth == "kernel"

    def query(self, q):
        T = self.stats["sd_S"] if self.how == "kernel" else None
        r = self.run_base(q, self.k, self.how, kernel_T=T)
        return self.result(r, float(r["f"][r["o"][0]]), self.base_extras(r))


class B0ShortlistContRerank(core.B0Plus):
    """M5 b0_shortlist_cont_rerank."""

    def __init__(self, lam=2.0, k=3):
        self.lam = core.parse_lam(lam)
        self.k = int(k)
        super().__init__(f"M5_b0sl_cont_l{core.lam_tag(self.lam)}_k{self.k}")

    def query(self, q):
        r = self.run_base(q, self.k, "med", lam=self.lam)
        return self.result(r, float(r["f"][r["order"][0]]), self.base_extras(r))


class ConsistencyConfidence(core.B0Plus):
    """M6 consistency_confidence (confidence head over a base selector)."""

    need_libstats = True
    BASES = {"b0": dict(k=1, how="top1", lam=None),
             "m4k5med": dict(k=5, how="med", lam=None),
             "m5l2k3": dict(k=3, how="med", lam=2.0)}

    def __init__(self, base="b0", weights="split", veto=False, tau=0.3):
        if base not in self.BASES:
            raise ValueError(f"base must be one of {sorted(self.BASES)}, got {base!r}")
        if weights not in ("equal", "split"):
            raise ValueError(f"weights must be equal | split, got {weights!r}")
        self.base = base
        self.weights = weights
        self.veto = bool(veto)
        self.tau = float(tau)
        tag = "eq" if weights == "equal" else "split"
        vt = (f"_veto{self.tau:g}".replace(".", "p")) if self.veto else ""
        super().__init__(f"M6_ccf_{base}_{tag}{vt}")

    def query(self, q):
        b = self.BASES[self.base]
        r = self.run_base(q, b["k"], b["how"], lam=b["lam"])
        ex = self.base_extras(r)
        st = self.stats
        reg = r["regime"]
        disp = ex["disp5_b0"] if self.base == "b0" else ex["disp_set"]
        use_cont = reg != core.REGIME_STEP0 and (self.weights == "equal" or reg == core.REGIME_MISS)
        t_cont = ex["cont_act"] / st["s_c"][self.base] if use_cont else 1.0
        t_disp = disp / st["s_a"][self.base]
        t_g = (ex["b0_pick"] - st["mu_S"]) / st["sd_S"]
        conf = -t_cont - t_disp + t_g
        vetoed = self.veto and reg != core.REGIME_STEP0 and ex["flip10"] >= self.tau
        if vetoed:
            conf -= VETO
        ex.update(t_cont=float(t_cont), t_disp=float(t_disp), t_g=float(t_g), veto=float(vetoed))
        return self.result(r, conf, ex)


class B0BigLibCons(core.B0Plus):
    """M8 b0_biglib_cons: B0 formula on the 10x library (bpool_cs pi0.5 / bpool_all GR00T)."""

    def __init__(self, k=1, synth="top1"):
        if synth not in ("top1", "mean"):
            raise ValueError(f"synth must be top1 | mean, got {synth!r}")
        self.k = 1 if synth == "top1" else int(k)
        self.how = synth
        super().__init__("M8_b0big_top1" if synth == "top1" else f"M8_b0big_k{self.k}_{synth}")

    def fit(self, lib, ctx):
        p = ctx.current_params()
        self.w = tuple(float(x) for x in p["weights"])
        self.mu, self.sg = p["mu"], p["sigma"]
        big = core.BIG_LIB[ctx.model]
        B = ctx.open_library(big)
        with ctx.prof.section("big_blocks"):
            self.blocks = core.big_blocks(B)
        self.rs_dim = B.rs.shape[1]
        self._fit_payload(B, ctx, big)

    def query(self, q):
        r = self.run_base(q, self.k, self.how)
        return self.result(r, float(r["f"][r["o"][0]]), self.base_extras(r))
