"""R3 family H3 -- MixedJudge: judge signals for a mixed HIT/MISS cache, as a wrapper around any G3-contract base.

Selection and synthesis are the BASE's (served action bit-identical to base.query(): tools/contract_check_mx.py).
The wrapper adds, per decision,

  confidence   V7 DriftCalibratedConfidence on the base (r02 g3_recovery/wrappers.py): -pred_err + 1e-6 * zsum,
               library-LOEO calibration (no trace data; deployable). Higher = more confident = HIT side of the
               plugin's threshold / quantile controller. During the "return" phase after a burst (below) the
               confidence is lowered by ret_margin (a stricter return threshold, implemented on the method side
               because the plugin's tau is unknown to the method).
  guards       (A-P4 + B-F5; V6 detector with ot_still)             -> Result.extras["os_force_miss"] = 1
     1 stuck        stuck_n >= stuck_thr (2): consecutive decisions with state motion < library 10th pct AND
                    min-camera task-centred key cosine to the previous decision >= library 95th pct
     2 terminal     the base's top-1 is the last row of its library episode AND the previously EXECUTED gripper
                    (q.prev_a_exec[4, 6]) is closed (>= 0)
     3 overtime     step / median library episode length > 1 AND lag (step - mean library step of the top-5) >
                    lag_thr (5) AND stuck_n >= 1
     4 no-progress  the library progress of the base's top-1 did not advance over the last noprog_n (3) decisions:
                    every consecutive difference, in library steps ((prog_t - prog_{t-1}) * (ep_len(top1_t) - 1)),
                    is <= prog_eps (0.5 step)
  events       (C-P3 early warnings)                                 -> os_force_miss = 1
     5 dispersion   disp (mean pairwise RMS of the top-5 heads, sigma units) >= disp_thr: an absolute threshold in
                    sigma units (disp_abs, default 1.0: the top-5 neighbours disagree by one action std) or, with
                    disp_q set, that quantile of the library LOEO pseudo-query dispersions (does not transfer to
                    closed-loop states: q.9 flags 25-30 % of the R2 closed-loop decisions, see README)
     6 gripper      the served action's step-0 gripper sign differs from the previously executed sign
                    (q.prev_a_exec[4, 6]) while |vote| < vote_thr (.8); vote = sum_i wn_i sign(g_i[0]) over the
                    EXACT kernel members with the base's kernel weights
  burst        (C-P3) a guard / event forced MISS at step s keeps forcing MISS at s+1 .. s+burst-1 (reason 7).
               Then the "return" phase: confidence -= ret_margin until ret_n HITs have been observed in q.hist_hit
               after the burst (or ret_hold decisions elapsed); a new guard / event restarts the burst.

Result.extras (os_* first: the plugin log keeps the first 24 scalars): os_force_miss (0/1), os_reason (0 none, 1-7
above; the lowest code that fires), os_flags (bitmask of every firing condition, bit r-1 for reason r), os_phase
(0 normal, 1 burst, 2 return), os_conf_raw (confidence before the return penalty), then pred_err, zsum, regime,
stuck_n, overtime, lag, term1, gexec, gprop, vote, disp, dnn, vis, top1_prog, noprog_n, burst_left, motion, vself.

Mixed-mode correctness (CODING_BRIEF): every per-episode quantity that depends on what was executed comes from the
QueryView -- gexec from q.prev_a_exec, stuck_n from q.rs / q.key_* vs q.hist_*, the burst / return state from
q.hist_hit. The two memos kept in self._s (top-1 progress per step for guard 4; own force flags per step for the
burst window) are functions of the earlier QueryViews only (os_score_all is stateless, and the QueryView at step
s < t is a prefix of q.hist_*), i.e. equivalent to re-scoring q.hist_*; they are indexed by step and dropped when a
step is missing (a skipped decision), never assumed to have been executed.

kwargs: base (spec), base_kwargs, guards (bool), events (list of "disp" / "grip", or "all" / "none"), burst (int,
0 = off), ret_margin, ret_n, ret_hold, disp_abs (1.0) | disp_q (None), vote_thr, stuck_thr, lag_thr, noprog_n,
prog_eps, calib, ncal, m_pct, c_pct. ot_still is always on.
Name: MXJ_g<0|1>_ev<0|D|G|DG>[_b<burst>_rm<ret_margin>...]__<base name>.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

from exp.offline_search.harness import api, dims

REPO = pathlib.Path(__file__).resolve().parents[5]
G3_DIR = REPO / "exp" / "offline_search" / "rounds" / "r02" / "g3_recovery"
if str(G3_DIR) not in sys.path:
    sys.path.insert(0, str(G3_DIR))
import g3_core as core  # noqa: E402  (the r02 module, by its plain name: pickles of its Tables refer to it)

WRAPPERS_SPEC = str(G3_DIR / "wrappers.py") + ":DriftCalibratedConfidence"
V7 = core.load_class(WRAPPERS_SPEC)
_W = sys.modules[V7.__module__]
REG_STEP0, REG_FRESH, REG_STALE = _W.REG_STEP0, _W.REG_FRESH, _W.REG_STALE

DEFAULT_BASE = "exp/offline_search/rounds/r02/g1_awm/awm.py:AWM"
DEFAULT_BASE_KWARGS = {"lib": "current", "kref": 5}
NAN = float("nan")
R_STUCK, R_TERM, R_OVERTIME, R_NOPROG, R_DISP, R_GRIP, R_BURST = 1, 2, 3, 4, 5, 6, 7
G_IDX0 = dims.GRIPPER_DIM                                   # index of head[0, 6] in the (35,) sigma-scaled head
EVENTS = ("disp", "grip")


def _tag(x) -> str:
    return f"{float(x):g}".replace(".", "p").replace("-", "m")


def _rebuild_mx(cls, base_spec):
    """Unpickle helper: import the r02 wrappers module and the base module before the state is restored."""
    core.load_class(WRAPPERS_SPEC)
    core.load_class(base_spec)
    return cls.__new__(cls)


class MixedJudge(V7):
    """V7 confidence + guards + early events + burst around any G3-contract base (see module docstring)."""

    PREFIX = "MXJ"
    family = "h3_judge"

    def __init__(self, base=DEFAULT_BASE, base_kwargs=None, guards=True, events="all", burst=0, ret_margin=0.1,
                 ret_n=1, ret_hold=4, disp_abs=1.0, disp_q=None, vote_thr=0.8, stuck_thr=2, lag_thr=5.0, noprog_n=3,
                 prog_eps=0.5, calib="iso", ncal=3000, m_pct=10.0, c_pct=95.0):
        if base_kwargs is None:
            base_kwargs = dict(DEFAULT_BASE_KWARGS) if base == DEFAULT_BASE else {}
        if isinstance(events, str):
            events = list(EVENTS) if events == "all" else ([] if events in ("none", "") else [events])
        events = tuple(sorted(set(events)))
        for e in events:
            if e not in EVENTS:
                raise ValueError(f"events must be a subset of {EVENTS}, got {e!r}")
        self.guards, self.events = bool(guards), events
        self.burst, self.ret_margin = max(int(burst), 0), float(ret_margin)
        self.ret_n, self.ret_hold = max(int(ret_n), 1), max(int(ret_hold), 1)
        self.disp_abs = float(disp_abs)
        self.disp_q = None if disp_q is None else float(disp_q)
        self.vote_thr = float(vote_thr)
        self.noprog_n, self.prog_eps = max(int(noprog_n), 2), float(prog_eps)
        self._disp_pool = None
        super().__init__(base, base_kwargs, calib=calib, ncal=ncal, m_pct=m_pct, c_pct=c_pct, lag_thr=lag_thr,
                         stuck_thr=stuck_thr, ot_still=True, anchor="self")

    # ------------------------------------------------------------------------------------------ naming
    def _make_name(self):
        ev = "".join(sorted(e[0].upper() for e in self.events)) or "0"
        d = [f"g{int(self.guards)}", f"ev{ev}"]
        if self.burst:
            d.append(f"b{self.burst}")
            d.append(f"rm{_tag(self.ret_margin)}")
            if self.ret_n != 1:
                d.append(f"rn{self.ret_n}")
            if self.ret_hold != 4:
                d.append(f"rh{self.ret_hold}")
        if "disp" in self.events:
            if self.disp_q is not None:
                d.append(f"dq{_tag(self.disp_q)}")
            elif self.disp_abs != 1.0:
                d.append(f"da{_tag(self.disp_abs)}")
        if "grip" in self.events and self.vote_thr != 0.8:
            d.append(f"vt{_tag(self.vote_thr)}")
        if self.guards:
            if self.stuck_thr != 2:
                d.append(f"sn{self.stuck_thr}")
            if self.lag_thr != 5.0:
                d.append(f"lg{_tag(self.lag_thr)}")
            if self.noprog_n != 3:
                d.append(f"np{self.noprog_n}")
            if self.prog_eps != 0.5:
                d.append(f"pe{_tag(self.prog_eps)}")
        if self.calib != "iso":
            d.append(self.calib)
        if self.ncal != 3000:
            d.append(f"n{self.ncal}")
        if self.m_pct != 10.0:
            d.append(f"m{_tag(self.m_pct)}")
        if self.c_pct != 95.0:
            d.append(f"c{_tag(self.c_pct)}")
        return f"{self.PREFIX}_{'_'.join(d)}__{self.base.name}"

    def __reduce__(self):
        return (_rebuild_mx, (type(self), self.base_spec), self.__dict__)

    # --------------------------------------------------------------------------------------------- fit
    def fit(self, lib, ctx):
        self._disp_pool = []                       # _features() collects the pseudo-query dispersions
        super().fit(lib, ctx)
        pool = np.asarray(self._disp_pool, np.float64)
        self._disp_pool = None
        pool = pool[np.isfinite(pool)]
        if self.disp_q is not None:
            self.disp_thr = float(np.quantile(pool, self.disp_q)) if pool.size else math.inf
        else:
            self.disp_thr = self.disp_abs
        qs = np.round(np.arange(0.50, 1.0001, 0.005), 3)          # quantile grid of the pool (diagnostics / replay)
        self.disp_qgrid = {float(q): float(v) for q, v in zip(qs, np.quantile(pool, qs))} if pool.size else {}
        self.fit_info["judge"] = {"guards": self.guards, "events": list(self.events), "burst": self.burst,
                                  "ret_margin": self.ret_margin, "ret_n": self.ret_n, "ret_hold": self.ret_hold,
                                  "disp_abs": self.disp_abs, "disp_q": self.disp_q, "disp_thr": self.disp_thr,
                                  "disp_pool_n": int(pool.size),
                                  "disp_pool_pcts": [float(x) for x in np.percentile(pool, [50, 75, 90, 95, 99])]
                                  if pool.size else [], "vote_thr": self.vote_thr, "stuck_thr": self.stuck_thr,
                                  "lag_thr": self.lag_thr, "noprog_n": self.noprog_n, "prog_eps": self.prog_eps}
        try:
            (ctx.scratch / f"mxj_fit_{self.name[:80]}.json").write_text(json.dumps(self.fit_info, indent=1, default=float))
        except Exception:
            pass

    def _features(self, *a, **kw):
        f = super()._features(*a, **kw)
        if self._disp_pool is not None:
            self._disp_pool.append(f["disp"])
        return f

    # ------------------------------------------------------------------------------------------ per query
    def reset(self, episode):
        super().reset(episode)
        s = self._s
        s["prog"] = []             # memo: (progress, ep_len - 1) of the base's top-1 per step (function of the
        #                            earlier QueryViews: os_score_all is stateless in the history)
        s["flag"] = []             # memo: 1 if a guard / event fired at that step
        s["burst_end"] = 0         # forced window [s, burst_end) of the last guard / event
        s["ret_end"] = 0           # return phase ends at this step at the latest

    def _memo_sync(self, s, step):
        """Keep the per-step memos aligned with the QueryView: entries beyond the current step (a repeated query)
        are dropped; missing steps (skipped decisions) are padded with NaN / 0 so nothing is assumed."""
        for key, fill in (("prog", (NAN, NAN)), ("flag", 0)):
            m = s[key]
            if len(m) > step:
                del m[step:]
            while len(m) < step:
                m.append(fill)

    def query(self, q):
        res = super().query(q)                                   # base selection / synthesis, V7 confidence, extras
        s, C = self._s, self.C
        step = int(q.step)
        ex = res.extras
        srows = np.asarray(res.topk, np.int64)
        top1 = int(srows[0])
        stuck_n = int(ex.get("stuck_n", 0))
        overtime, lag = float(ex.get("overtime", NAN)), float(ex.get("lag", NAN))
        disp = float(ex.get("disp", NAN))
        with self.prof.section("mxj_judge"):
            # -- executed gripper sign (from the QueryView), proposed sign and exact-member vote
            pa = q.prev_a_exec
            gexec = 0.0
            if step > 0 and pa is not None:
                gexec = 1.0 if float(pa[dims.EXEC_STEPS - 1, dims.GRIPPER_DIM]) >= 0 else -1.0
            w = core.kernel_weights(np.asarray(res.scores, np.float64), self.T)
            wn = w / w.sum()
            gsign = np.where(C.HD[srows, G_IDX0] >= 0, 1.0, -1.0)
            vote = float(wn @ gsign)
            g0 = float(res.action[0, dims.GRIPPER_DIM]) if res.action is not None else float(C.HD[top1, G_IDX0])
            gprop = 1.0 if g0 >= 0 else -1.0
            # -- top-1 progress memo (guard 4)
            self._memo_sync(s, step)
            prog = float(C.prog[top1])
            s["prog"].append((prog, float(max(int(C.ep_len[top1]) - 1, 1))))
            noprog = 0
            for j in range(step, 0, -1):                           # consecutive non-advancing transitions
                (a, n_a), (b, _) = s["prog"][j], s["prog"][j - 1]
                if not (math.isfinite(a) and math.isfinite(b)) or (a - b) * n_a > self.prog_eps:
                    break                                          # advanced by more than prog_eps library steps
                noprog += 1
            term1 = C.nxt[top1] < 0
            flags = 0
            if self.guards:
                if stuck_n >= self.stuck_thr:
                    flags |= 1 << (R_STUCK - 1)
                if term1 and gexec > 0:
                    flags |= 1 << (R_TERM - 1)
                if math.isfinite(overtime) and overtime > 1.0 and lag > self.lag_thr and stuck_n >= 1:
                    flags |= 1 << (R_OVERTIME - 1)
                if noprog >= self.noprog_n - 1:
                    flags |= 1 << (R_NOPROG - 1)
            if "disp" in self.events and math.isfinite(disp) and disp >= self.disp_thr:
                flags |= 1 << (R_DISP - 1)
            if "grip" in self.events and gexec != 0.0 and gprop != gexec and abs(vote) < self.vote_thr:
                flags |= 1 << (R_GRIP - 1)
            fired = flags != 0
            s["flag"].append(int(fired))
            # -- burst / return state
            reason, phase, conf = 0, 0, float(res.confidence)
            if fired:
                reason = (flags & -flags).bit_length()              # lowest firing reason code
                if self.burst > 1:
                    s["burst_end"] = step + self.burst
                    s["ret_end"] = s["burst_end"] + self.ret_hold
            elif step < s["burst_end"]:
                reason, phase = R_BURST, 1
            elif step < s["ret_end"]:
                hh = np.asarray(q.hist_hit)[s["burst_end"]:step]        # executed outcomes after the burst
                if int((hh == 1).sum()) < self.ret_n:
                    phase = 2
                    conf = conf - self.ret_margin
                else:
                    s["ret_end"] = 0
            force = int(reason != 0)
            burst_left = max(s["burst_end"] - step - 1, 0) if (fired or phase == 1) else 0
        out = {"os_force_miss": float(force), "os_reason": float(reason), "os_flags": float(flags),
               "os_phase": float(phase), "os_conf_raw": float(res.confidence),
               "pred_err": ex.get("pred_err", NAN), "zsum": ex.get("zsum", NAN), "regime": ex.get("regime", NAN),
               "stuck_n": float(stuck_n), "overtime": overtime, "lag": lag, "term1": float(term1), "gexec": gexec,
               "gprop": gprop, "vote": vote, "disp": disp, "dnn": ex.get("dnn", NAN), "vis": ex.get("vis", NAN),
               "top1_prog": prog, "noprog_n": float(noprog), "burst_left": float(burst_left),
               "motion": ex.get("motion", NAN), "vself": ex.get("vself", NAN)}
        out = {k: v for k, v in out.items() if math.isfinite(v)}
        return api.Result(topk=res.topk, scores=res.scores, confidence=conf, action=res.action, library=res.library,
                          extras=out)
