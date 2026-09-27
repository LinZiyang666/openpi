"""R2 family G3 "closed-loop mechanisms": two WRAPPERS around any base selector (README.md: base contract).

V6 StuckRecovery (B-P2)
  (a) blend   after a HIT, next(previous served row) joins the kernel mean with the top weight (blend_w="top"; or a
              fixed final share blend_w = beta, the top-(k-1) others sharing 1 - beta) (skipped when the
              served row is terminal); the served row ("anchor") of the previous decision is the library row whose
              chunk was executed (exact match of prev_a_exec[:, :7] in the candidate library -- offline cache arms:
              the recorded B0 pick) or, when the executed chunk is synthesized (the closed loop), this method's own
              previous top-1 (anchor="self" forces the latter).
  (b) detector, online data only:
              still_t   = |rs_t - rs_{t-1}|[:8] < m_thr (library m_pct-th pct of per-decision motion) AND
                          min over cameras of cos(key_t - m_task, key_{t-1} - m_task) >= c_thr (library c_pct-th pct
                          of the same cosine between consecutive library decisions)
              stuck_n   = consecutive still decisions; overtime = step / median library ep_len of the task;
              terminal  = the served row has no next (after a HIT); lag = step - mean library step of the base top-5
              stuck     = stuck_n >= 2 OR terminal OR (overtime > 1 AND lag > 5)
                          [ot_still=True: the last clause only while stuck_n >= 1 -- a stricter closed-loop option]
  (c) recovery, after a HIT, escalating over the stuck decisions since the last reset (esc = 2 decisions / level):
              R1 exclude the library episodes served in the last excl_win decisions, re-select with the base;
              R2 + restrict to library progress <= ref_prog - rewind (ref_prog = progress of the served row when the
                 stuck spell began; frozen for the spell, so successive R2 decisions do not rewind further);
              R3 top-1 of the R2 set (a library row served as is) instead of the kernel mean.
              Escalation resets after 2 consecutive decisions with stuck_n = 0 (motion resumed).
V7 DriftCalibratedConfidence (B-P3)
  confidence = -predicted err of a fixed-weight z-sum of observation-side features,
      stale / step 0: -d_nn, -dispersion(top-5), -overtime, max vis_v0 + max vis_v1, -min(stuck_n, 5), -|lag|
      fresh (after a MISS): -continuity(served head vs previous tail), -dispersion, vision
  z-scales and the per-regime monotone map (isotonic PAV or 10-bin) are fitted on library LOEO pseudo-queries of
  the base's fit library (failed library episodes included when the library has them); + 1e-6 * zsum breaks the
  ties of the flat map pieces, so within a regime the ranking is exactly the z-sum's.

Both inherit the base's library (base.os_library) and fit data (base.os_fit_library); thresholds and calibration are
computed on the fit library. Per-episode state lives in self._s and is reset in reset(). Extras expose every flag.
"""
from __future__ import annotations

import json
import math

import numpy as np

from exp.offline_search.harness import api, dims

try:
    from . import g3_core as core
except ImportError:                      # loaded by file path (the file's dir is on sys.path)
    import g3_core as core

NAN = float("nan")
REG_STEP0, REG_FRESH, REG_STALE = 0, 1, 2
STALE_FEATS = (("dnn", -1.0), ("disp", -1.0), ("overtime", -1.0), ("vis", 1.0), ("stuck", -1.0), ("abslag", -1.0))
FRESH_FEATS = (("cont", -1.0), ("disp", -1.0), ("vis", 1.0))
FEATS = {REG_STEP0: STALE_FEATS, REG_STALE: STALE_FEATS, REG_FRESH: FRESH_FEATS}
MIN_CAL = 30                 # a regime with fewer calibration pseudo-queries borrows the stale map
TIE_EPS = 1e-6
EXTRA_BYTES = 4 + 4 + 2 + 4 + 32   # next, episode, step, progress, rs[:8] (shared with the base if it stores rs)


def _rebuild(cls, base_spec):
    """Unpickle / deepcopy helper: import the base module (file-path modules are not importable by name) BEFORE the
    state that holds the fitted base is restored."""
    core.load_class(base_spec)
    return cls.__new__(cls)


def _tag(x) -> str:
    return f"{x:g}".replace(".", "p").replace("-", "m")


class G3Wrapper(api.Method):
    family = "g3_recovery"
    tier = "T1"
    PREFIX = "G3w"

    def __init__(self, base, base_kwargs=None, blend=False, recover=False, conf="base", calib="iso", ncal=3000,
                 m_pct=10.0, c_pct=95.0, excl_win=3, rewind=0.2, esc=2, anchor="exec", lag_thr=5.0, stuck_thr=2,
                 ot_still=False, blend_w="top"):
        if conf not in ("base", "score", "v7"):
            raise ValueError(f"conf must be base | score | v7, got {conf!r}")
        if calib not in ("iso", "bin10"):
            raise ValueError(f"calib must be iso | bin10, got {calib!r}")
        if anchor not in ("exec", "self"):
            raise ValueError(f"anchor must be exec | self, got {anchor!r}")
        self.base_spec = str(base)
        self.base_kwargs = dict(base_kwargs or {})
        self.base = core.load_class(self.base_spec)(**self.base_kwargs)
        api.check_method_attrs(self.base)
        self.blend, self.recover, self.conf, self.calib = bool(blend), bool(recover), conf, calib
        self.ncal, self.m_pct, self.c_pct = int(ncal), float(m_pct), float(c_pct)
        self.excl_win, self.rewind, self.esc = int(excl_win), float(rewind), int(esc)
        self.anchor_mode, self.lag_thr, self.stuck_thr = anchor, float(lag_thr), int(stuck_thr)
        self.ot_still = bool(ot_still)       # True: the (overtime AND lag) clause only counts while stuck_n >= 1
        self.blend_w = "top" if blend_w in ("top", None) else float(blend_w)   # "top" | final weight share of next
        self.tier = "T2" if getattr(self.base, "tier", "T0") == "T2" else "T1"
        self.name = self._make_name()
        self._s = None

    # ------------------------------------------------------------------------------------------ naming
    def _suffix(self, with_conf=True):
        d = []
        if self.m_pct != 10.0:
            d.append(f"m{_tag(self.m_pct)}")
        if self.c_pct != 95.0:
            d.append(f"c{_tag(self.c_pct)}")
        if self.excl_win != 3:
            d.append(f"x{self.excl_win}")
        if self.rewind != 0.2:
            d.append(f"rw{_tag(self.rewind)}")
        if self.esc != 2:
            d.append(f"e{self.esc}")
        if self.lag_thr != 5.0:
            d.append(f"lg{_tag(self.lag_thr)}")
        if self.stuck_thr != 2:
            d.append(f"sn{self.stuck_thr}")
        if self.ot_still:
            d.append("ots")
        if self.blend_w != "top":
            d.append(f"bw{_tag(self.blend_w)}")
        if self.anchor_mode != "exec":
            d.append(f"a{self.anchor_mode}")
        if not with_conf:
            pass
        elif self.conf == "v7":
            d.append(f"cv7{self.calib}" + (f"n{self.ncal}" if self.ncal != 3000 else ""))
        elif self.conf == "score":
            d.append("cscore")
        return ("_" + "_".join(d)) if d else ""

    def _make_name(self):
        return f"{self.PREFIX}_bl{int(self.blend)}rc{int(self.recover)}{self._suffix()}__{self.base.name}"

    def __reduce__(self):
        return (_rebuild, (type(self), self.base_spec), self.__dict__)

    def bytes_per_entry(self):
        b = float(self.base.bytes_per_entry())
        return b + EXTRA_BYTES

    # --------------------------------------------------------------------------------------------- fit
    def fit(self, lib, ctx):
        b = self.base
        b.prof = self.prof
        b.fit(lib, ctx)
        b.prof = api.NULL_PROFILER
        self.model = ctx.model
        self.sigma = np.asarray(ctx.action_sigma, np.float64)
        libname = getattr(b, "os_library", None)
        if not isinstance(libname, str):
            raise api.ContractError(f"base {b.name}: attribute os_library (the library its rows index) is missing")
        if libname in ctx.registered:
            raise api.ContractError(f"base {b.name}: method-built library {libname!r} is not supported by G3")
        fitname = getattr(b, "os_fit_library", None) or libname
        try:
            self.k = int(getattr(b, "synth_k"))
            self.T = float(getattr(b, "synth_T"))
        except AttributeError as e:
            raise api.ContractError(f"base {b.name}: synth_k / synth_T missing ({e})") from None
        self.libname, self.fitname = libname, fitname
        self.has_synth = callable(getattr(b, "os_synth", None))
        self.has_conf = callable(getattr(b, "os_confidence", None))
        Lv = lib if libname == "current" else ctx.open_library(libname)
        with self.prof.section("g3_tables"):
            self.C = core.Tables(Lv, ctx.model, self.sigma, want_hash=(self.anchor_mode == "exec"))
        if fitname == libname:
            Fv, FT = Lv, self.C
        else:
            Fv = lib if fitname == "current" else ctx.open_library(fitname)
            FT = core.Tables(Fv, ctx.model, self.sigma, want_hash=False)
        with self.prof.section("g3_thresholds"):
            self._fit_thresholds(Fv, FT)
        info = {"libname": libname, "fitname": fitname, "k": self.k, "T": self.T, "m_thr": self.m_thr,
                "c_thr": self.c_thr, "med_len": {str(t): v for t, v in self.med_len.items()},
                "thr_stats": self.thr_stats}
        self.cal = None
        if self.conf == "v7":
            with self.prof.section("g3_calibrate"):
                self.cal, cinfo = self._calibrate(Fv, FT)
            info["calibration"] = cinfo
        self._F_still = None
        self.fit_info = info
        try:
            (ctx.scratch / f"g3_fit_{self.name[:80]}.json").write_text(json.dumps(info, indent=1, default=float))
        except Exception:
            pass

    def _fit_thresholds(self, Fv, FT):
        ep_rows = core.episode_rows(FT)
        tasks = np.unique(FT.task)
        ok = FT.nxt >= 0
        motion_next = np.linalg.norm(FT.rs8[FT.nxt[ok]] - FT.rs8[ok], axis=1)
        self.m_thr = float(np.percentile(motion_next, self.m_pct))
        self.M0 = core.task_means(Fv.key_v0, FT.task, tasks)
        self.M1 = core.task_means(Fv.key_v1, FT.task, tasks)
        vc = core.consecutive_vcos(Fv.key_v0, Fv.key_v1, self.M0, self.M1, FT, ep_rows)
        fin = np.isfinite(vc)
        self.c_thr = float(np.percentile(vc[fin], self.c_pct))
        self.med_len = {}
        for t in tasks:
            eps = np.unique(FT.ep[FT.task == t])
            self.med_len[int(t)] = float(np.median([FT.ep_len[ep_rows[int(e)][0]] for e in eps]))
        hp = FT.prev >= 0
        motion_prev = np.full(FT.L, np.inf)
        motion_prev[hp] = np.linalg.norm(FT.rs8[hp] - FT.rs8[FT.prev[hp]], axis=1)
        still = (motion_prev < self.m_thr) & (np.where(fin, vc, -np.inf) >= self.c_thr)
        self._F_still = core.run_lengths(still, FT, ep_rows)
        self.thr_stats = {"motion_pcts": [float(x) for x in np.percentile(motion_next, [5, 10, 25, 50])],
                          "vcos_pcts": [float(x) for x in np.percentile(vc[fin], [50, 90, 95, 99])],
                          "lib_still_frac": float(still[hp].mean()),
                          "lib_stuck2_frac": float((self._F_still[hp] >= 2).mean()),
                          "n_rows": int(FT.L)}

    # ------------------------------------------------------------------------------------- calibration
    def _calibrate(self, Fv, FT):
        C = self.C
        same = FT is C
        if same:
            own_ep = FT.ep
        else:                                       # map fit-library episodes to candidate episodes by stem
            cmap = {s: i for i, s in enumerate(C.stems or [])}
            fmap = np.asarray([cmap.get(s, -1) for s in (FT.stems or [])], np.int64)
            own_ep = fmap[FT.ep] if fmap.size else np.full(FT.L, -1, np.int64)
        ctasks = set(np.unique(C.task).tolist())
        ep_rows = core.episode_rows(FT)
        pos_of = np.empty(FT.L, np.int64)
        for rows in ep_rows.values():
            pos_of[rows] = np.arange(rows.size)
        in_task = np.asarray([int(t) in ctasks for t in FT.task])
        s0 = np.flatnonzero(in_task & (FT.step == 0))
        later = np.flatnonzero(in_task & (FT.step > 0) & (FT.prev >= 0))
        rng = np.random.default_rng(0)
        sub = np.sort(rng.choice(later, size=min(self.ncal, later.size), replace=False)) if later.size else later
        uses_prev = bool(getattr(self.base, "os_uses_prev", True))
        acc = {REG_STEP0: ([], []), REG_FRESH: ([], []), REG_STALE: ([], [])}
        n_calls = 0
        for r in np.r_[s0, sub]:
            r = int(r)
            ep_rows_r = ep_rows[int(FT.ep[r])]
            regs = (REG_STEP0,) if FT.step[r] == 0 else (REG_STALE, REG_FRESH)
            cached = None
            for reg in regs:
                if cached is None or uses_prev:
                    pq = core.PseudoQuery(Fv, FT, ep_rows_r, pos_of[r], self.model, prev_hit=(reg != REG_FRESH))
                    cached = core.base_scores(self.base, pq)
                    n_calls += 1
                rows, S, aux = cached
                valid = C.ep[rows] != own_ep[r]
                if not valid.any():
                    continue
                sel = core.topk_pos(S, self.k, valid)
                w = core.kernel_weights(S[sel], self.T)
                head = self._head(core.synth(C.act, rows[sel], w, math.isinf(self.T)), rows[sel[0]])
                err = float(np.sqrt(np.mean((head - FT.HD[r]) ** 2)))
                tail = None
                if reg == REG_FRESH:
                    tail = self._tail_sig(FT.act[FT.prev[r]])
                f = self._features(reg, int(FT.step[r]), int(FT.task[r]), FT.rs8[r], rows, S, aux, valid, head, tail,
                                   min(int(self._F_still[r]), 5))
                acc[reg][0].append([f[n] for n, _ in FEATS[reg]])
                acc[reg][1].append(err)
        cal, info = {}, {"n_calls": n_calls, "uses_prev": uses_prev, "same_library": same}
        for reg in (REG_STALE, REG_FRESH, REG_STEP0):
            X = np.asarray(acc[reg][0], np.float64).reshape(-1, len(FEATS[reg]))
            e = np.asarray(acc[reg][1], np.float64)
            if e.size < MIN_CAL and reg != REG_STALE:
                cal[reg] = dict(cal[REG_STALE], borrowed=True) if reg == REG_STEP0 else None
                info[str(reg)] = {"n": int(e.size), "borrowed_stale": reg == REG_STEP0}
                continue
            vi = [n for n, _ in FEATS[reg]].index("vis")
            if not np.isfinite(X[:, vi]).any():
                raise api.ContractError(f"base {self.base.name}: os_score_all exposes no per-candidate visual "
                                        f"similarity (aux 'vis_v0'/'vis_v1' or 'vis'); V7 needs it (vision is "
                                        f"mandatory in every regime)")
            fill = np.where(np.isfinite(X).any(0), np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0), 0.0)
            X = np.where(np.isfinite(X), X, fill)
            mu = X.mean(0)
            sd = X.std(0)
            wgt = np.asarray([sg if s > 1e-9 else 0.0 for (_, sg), s in zip(FEATS[reg], sd)])
            sd = np.where(sd > 1e-9, sd, 1.0)
            z = ((X - mu) / sd) @ wgt
            kx, ky = core.pav_decreasing(z, e) if self.calib == "iso" else core.bin_map(z, e)
            cal[reg] = {"mu": mu, "sd": sd, "w": wgt, "kx": kx, "ky": ky, "borrowed": False}
            pred = np.interp(z, kx, ky)
            info[str(reg)] = {"n": int(e.size), "err_mean": float(e.mean()), "n_knots": int(kx.size),
                              "corr_z_err": float(np.corrcoef(z, e)[0, 1]) if e.size > 2 else NAN,
                              "aurc_insample": _aurc(-pred + TIE_EPS * z, e), "aurc_opt": _aurc(-e, e),
                              "pred_range": [float(ky.min()), float(ky.max())], "feats": [n for n, _ in FEATS[reg]],
                              "mu": mu.tolist(), "sd": sd.tolist(), "w": wgt.tolist()}
        if cal.get(REG_FRESH) is None:
            cal[REG_FRESH] = None
        return cal, info

    # ------------------------------------------------------------------------------------------ per query
    def reset(self, episode):
        self.base.reset(episode)
        self._s = {"prev_top1": -1, "stuck_n": 0, "move_n": 0, "esc_n": 0, "ref_prog": NAN, "recent": []}

    def _head(self, action, row):
        """sigma-scaled executed block (35,) of the served action (synthesized, or the library row itself)."""
        if action is None:
            return self.C.HD[row].astype(np.float64)
        return (np.asarray(dims.valid_action(action), np.float64) / self.sigma).reshape(core.NH)

    def _tail_sig(self, prev_chunk):
        a = np.asarray(prev_chunk, np.float64)[dims.EXEC_STEPS:2 * dims.EXEC_STEPS, dims.ACT_VALID]
        return (a / self.sigma).reshape(core.NH)

    def _self_change(self, q):
        if q.step == 0:
            return NAN, NAN
        rs = dims.valid_state(np.asarray(q.rs, np.float32), self.model)
        pr = dims.valid_state(np.asarray(q.hist_rs[-1], np.float32), self.model)
        motion = float(np.linalg.norm(rs - pr))
        t = int(q.task_id)
        if t not in self.M0:
            return motion, NAN
        v0 = core.centred_cos(q.key_v0, q.hist_key_v0[-1], self.M0[t])
        v1 = core.centred_cos(q.key_v1, q.hist_key_v1[-1], self.M1[t])
        return motion, min(v0, v1)

    def _anchor(self, q):
        """(row of the previous decision's served chunk in the candidate library, source 1 exec-match / 0 self)."""
        if q.step == 0:
            return -1, -1
        if self.anchor_mode == "exec":
            pa = q.prev_a_exec
            if pa is not None:
                a7 = np.ascontiguousarray(dims.valid_action_chunk(np.asarray(pa, np.float32)))
                r = self.C.find_chunk(a7, int(q.task_id))
                if r >= 0:
                    return r, 1
        p = self._s["prev_top1"]
        return p, (0 if p >= 0 else -1)

    def _features(self, reg, step, t, rs8, rows, S, aux, valid, head, tail, stuck, t5=None):
        C = self.C
        d = C.rs8_of(rows) - rs8
        d2 = np.einsum("ij,ij->i", d, d)
        dnn = float(np.sqrt((d2 if valid is None else d2[valid]).min()))
        if t5 is None or valid is not None:
            t5 = core.topk_pos(S, 5, valid)
        disp = core.dispersion(C.HD[rows[t5]])
        lag = step - float(C.step[rows[t5]].mean())
        f = {"dnn": dnn, "disp": disp, "overtime": step / self.med_len.get(t, NAN), "vis": core.vis_of(aux, valid),
             "stuck": float(stuck), "abslag": abs(lag), "lag_pool": lag,
             "cont": float(np.sqrt(np.mean((head - tail) ** 2))) if tail is not None else NAN}
        return f

    def _v7(self, reg, f):
        c = self.cal.get(reg) if self.cal else None
        if c is None:
            c = self.cal[REG_STALE]
            reg = REG_STALE
        x = np.asarray([f[n] for n, _ in FEATS[reg]], np.float64)
        x = np.where(np.isfinite(x), x, c["mu"])
        z = float(((x - c["mu"]) / c["sd"]) @ c["w"])
        pred = float(np.interp(z, c["kx"], c["ky"]))
        return -pred + TIE_EPS * z, z, pred

    def query(self, q):
        s = self._s
        C = self.C
        step, t = int(q.step), int(q.task_id)
        ph = q.prev_hit
        reg = REG_STEP0 if step == 0 else (REG_FRESH if ph is False else REG_STALE)
        with self.prof.section("base_score"):
            rows, S, aux = core.base_scores(self.base, q)
        with self.prof.section("g3_detect"):
            motion, vself = self._self_change(q)
            still = step > 0 and motion < self.m_thr and vself >= self.c_thr
            s["stuck_n"] = s["stuck_n"] + 1 if still else 0
            s["move_n"] = s["move_n"] + 1 if s["stuck_n"] == 0 else 0
            anchor, asrc = self._anchor(q)
            if anchor >= 0:
                s["recent"] = (s["recent"] + [int(C.ep[anchor])])[-self.excl_win:]
            t8 = core.topk_pos(S, max(self.k, 5))
            t5 = t8[:5]
            lag = step - float(C.step[rows[t5]].mean())
            overtime = step / self.med_len.get(t, NAN)
            terminal = reg == REG_STALE and anchor >= 0 and C.nxt[anchor] < 0
            behind = overtime > 1.0 and lag > self.lag_thr and (s["stuck_n"] >= 1 or not self.ot_still)
            stuck = (s["stuck_n"] >= self.stuck_thr) or terminal or behind
        level, relaxed, valid, n_excl = 0, 0, None, 0
        if self.recover and reg == REG_STALE:
            with self.prof.section("g3_recover"):
                if s["move_n"] >= 2:
                    s["esc_n"] = 0
                if stuck:
                    if s["esc_n"] == 0:
                        s["ref_prog"] = float(C.prog[anchor]) if anchor >= 0 else min(1.0, overtime)
                    s["esc_n"] += 1
                    level = min(3, 1 + (s["esc_n"] - 1) // self.esc)
                if level >= 1:
                    bad = np.isin(C.ep[rows], np.asarray(s["recent"], np.int64))
                    if level >= 2:
                        bad2 = bad | (C.prog[rows] > s["ref_prog"] - self.rewind)
                        if bad2.all():
                            relaxed = 1
                        else:
                            bad = bad2
                    if bad.all():
                        relaxed = 2
                        bad[:] = False
                    n_excl = int(bad.sum())
                    valid = ~bad if n_excl else None
        with self.prof.section("g3_select"):
            k = 1 if level == 3 else self.k
            sel = t8[:k] if valid is None else core.topk_pos(S, k, valid)
            w = core.kernel_weights(S[sel], self.T)
            forced, nx_in, blend_used = False, 0, 0
            if self.blend and level == 0 and reg == REG_STALE and anchor >= 0 and C.nxt[anchor] >= 0:
                p = np.flatnonzero(rows == C.nxt[anchor])
                if p.size:
                    p = int(p[0])
                    j = np.flatnonzero(sel == p)
                    wt = w[0]
                    if self.blend_w != "top":             # next gets the final share beta, the top-(k-1) others 1-beta
                        beta = self.blend_w
                        others = sel[sel != p][:max(self.k - 1, 1)]
                        s0 = max(float(S[others].max()), float(S[p]))
                        wo = np.ones(others.size) if math.isinf(self.T) else np.exp(-(s0 - S[others]) / self.T)
                        sel = np.r_[others, p]
                        w = np.r_[wo / wo.sum() * (1.0 - beta), beta]
                        nx_in = int(j.size > 0)
                    elif j.size:
                        nx_in = 1
                        w = w.copy()
                        w[j[0]] = wt
                    elif sel.size >= 2:
                        sel = sel.copy()
                        w = w.copy()
                        sel[-1] = p
                        w[-1] = wt
                    else:
                        sel = np.r_[sel, p]
                        w = np.r_[w, wt]
                    forced, blend_used = True, 1
            top = sel[int(np.argmax(S[sel]))]                  # the anchor for the next decision = S-top-1
            o = np.lexsort((np.arange(sel.size), -S[sel], -w, sel != top))
            sel, w = sel[o], w[o]
            srows = rows[sel]
            if srows.size == 1:
                action = None
            elif self.has_synth:
                action = self.base.os_synth(q, srows, w)
            else:
                action = core.synth(C.act, srows, w, math.isinf(self.T) and not forced)
            s["prev_top1"] = int(srows[0])
        with self.prof.section("g3_conf"):
            head = self._head(action, srows[0])
            tail = self._tail_sig(q.prev_a_exec) if (reg == REG_FRESH and q.prev_a_exec is not None) else None
            rs8 = dims.valid_state(np.asarray(q.rs, np.float32), self.model)
            f = self._features(reg, step, t, rs8, rows, S, aux, valid, head, tail, min(s["stuck_n"], 5), t5)
            zsum = pred = NAN
            if self.conf == "v7":
                conf, zsum, pred = self._v7(reg, f)
            elif self.conf == "base" and self.has_conf:
                conf = float(self.base.os_confidence(q, rows, S, aux, srows, w))
            else:
                conf = float(S[sel[0]])
        wn = w / w.sum()
        ex = {"regime": float(reg), "motion": motion, "vself": vself, "still": float(still),
              "stuck_n": float(s["stuck_n"]), "overtime": overtime, "terminal": float(terminal), "lag": lag,
              "stuck": float(stuck), "level": float(level), "esc_n": float(s["esc_n"]), "relaxed": float(relaxed),
              "n_excl": float(n_excl), "n_cand": float(rows.size), "blend": float(blend_used), "nx_in": float(nx_in),
              "w_nx": float(wn[np.flatnonzero(sel == p)[0]]) if blend_used else NAN,
              "anchor": float(anchor), "anchor_src": float(asrc),
              "anchor_ep": float(C.ep[anchor]) if anchor >= 0 else -1.0,
              "top1": float(srows[0]), "top1_ep": float(C.ep[srows[0]]), "top1_step": float(C.step[srows[0]]),
              "top1_prog": float(C.prog[srows[0]]), "k_sel": float(srows.size), "k_eff": float(1.0 / np.sum(wn * wn)),
              "dnn": f["dnn"], "disp": f["disp"], "vis": f["vis"], "cont": f["cont"], "zsum": zsum, "pred_err": pred}
        ex = {kk: v for kk, v in ex.items() if math.isfinite(v)}     # NaN = "not applicable": left out (the harness
        #                          stores missing keys as NaN; the closed-loop verifiers compare present keys with ==)
        return api.Result(topk=srows, scores=S[sel], confidence=conf, action=action, library=self.libname, extras=ex)


class StuckRecovery(G3Wrapper):
    """V6 stuck_recovery (B-P2): blend + detector + escalating recovery (defaults: all on, confidence = base's)."""
    PREFIX = "V6sr"

    def __init__(self, base, base_kwargs=None, blend=True, recover=True, conf="base", **kw):
        super().__init__(base, base_kwargs, blend=blend, recover=recover, conf=conf, **kw)


class DriftCalibratedConfidence(G3Wrapper):
    """V7 drift_calibrated_confidence (B-P3): base selection unchanged, confidence = -predicted err (LOEO-calibrated)."""
    PREFIX = "V7dc"

    def __init__(self, base, base_kwargs=None, calib="iso", blend=False, recover=False, **kw):
        kw.pop("conf", None)
        super().__init__(base, base_kwargs, blend=blend, recover=recover, conf="v7", calib=calib, **kw)

    def _make_name(self):
        b = f"_bl{int(self.blend)}rc{int(self.recover)}" if (self.blend or self.recover) else ""
        suf = self._suffix(with_conf=False)
        return f"{self.PREFIX}_{self.calib}{'' if self.ncal == 3000 else f'n{self.ncal}'}{b}{suf}__{self.base.name}"


class Passthrough(G3Wrapper):
    """Contract check: the wrapper with every mechanism off must reproduce base.query() (tools/contract_check.py)."""
    PREFIX = "G3pt"

    def __init__(self, base, base_kwargs=None, **kw):
        super().__init__(base, base_kwargs, blend=False, recover=False, conf="base", **kw)


def _aurc(conf, err):
    o = np.lexsort((np.arange(conf.size), -conf))
    e = err[o]
    return float((np.cumsum(e) / np.arange(1, e.size + 1)).mean())
