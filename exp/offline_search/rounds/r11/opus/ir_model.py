"""Shared offline owner-IR accounting model for R11 layer 4 (knob -> theoretical owner IR on the 3-layer base).

Library only. Inputs are the whole-episode-out anchor tables written by ``loeo_anchor_table.py`` (R10 nested
subsets of the B-pool libraries, 5 folds by permutation position % 5, the deployed retrieval rule). Nothing under
trace_runs/os_closed_loop is read. See IR_MODEL.md for the derivation, assumptions and what cannot be estimated.

Owner IR (ledger form, per arm):  IR = (c_v * V + c_m * M) / N
  N = decision slots (one per 5 executed controls, blind and policy-tail slots included),
  V = vision decisions (cache LOOKs + policy calls), M = policy calls (MISS; a MISS is also a vision decision),
  pi05 (c_v, c_m) = (.152, .848), GR00T = (.148, .852). A ten-control unit costs c_v (LOOK) or c_v + c_m = 1 (MISS)
  for 2 slots, i.e. ~.076 / .5 per slot; pure policy with a ten-control commit = .5.

Cadence (3-layer base, ten-control anchor-tail commit, one committed policy tail): anchors (vision decisions) sit
at even decision steps 0, 2, 4, ...; the odd slot is the cache's blind tail or the policy tail. Hence
v = V/N ~= sum ceil(n_ep/2) / sum n_ep (lifecycle LOOKs ignored; R6 ledgers .502-.514). With f = M/V the anchor
miss fraction, IR = v * (c_v + c_m * f).

Guard: the frozen R8 only-no-progress rule replayed on each held-out library episode (span rule over the top-1's
library progress, noprog_n 3, prog_eps .5). It fires at anchor i iff the retrieved top-1 progress has not advanced
by more than half a library step (in the top-1's episode-normalised units) for >= 2 decision steps.

Knobs act on anchors only. The default overlap rule is "guard first": a knob can only add calls on anchors where
the guard did not fire. Knob calls do not change later guard flags (open-loop replay; R6 B + random-dose arms
matched this independence within +-.006 in m, see IR_MODEL.md).
"""
from __future__ import annotations

import glob
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ANCHOR_DIR = HERE / "out" / "anchor"
COSTS = {"pi05": (0.152, 0.848), "groot": (0.148, 0.852)}
PROG_EPS, NOPROG_N = 0.5, 3
DIST_RULE = dict(peak=.5, plateau=.75, cutoff=2.)


# ----------------------------------------------------------------------------------------------------- accounting
def owner_ir(model, V, M, N):
    c_v, c_m = COSTS[model]
    return (c_v * V + c_m * M) / N


def ir_from_fraction(model, v, f):
    """IR from v = V/N and the anchor miss fraction f = M/V."""
    c_v, c_m = COSTS[model]
    return v * (c_v + c_m * f)


def fraction_for_ir(model, v, target):
    """Anchor miss fraction f that gives owner IR = target at vision rate v (clipped to [0, 1])."""
    c_v, c_m = COSTS[model]
    return float(np.clip((target / v - c_v) / c_m, 0.0, 1.0))


def random_closed_form(model, v, g, rho):
    """Guard-first independent coin: f = g + (1 - g) rho."""
    return ir_from_fraction(model, v, g + (1 - g) * rho)


def solve_rho(model, v, g, target):
    """Random knob rho for a target owner IR, given v and the base guard rate g (clipped to [0, 1])."""
    f = fraction_for_ir(model, v, target)
    return float(np.clip((f - g) / (1 - g), 0.0, 1.0)) if g < 1 else 0.0


# --------------------------------------------------------------------------------------------- episode tables
@dataclass
class Episode:
    ep: int
    task: int
    success: bool
    n_slots: int
    steps: np.ndarray          # anchor decision steps (0, 2, 4, ...)
    guard: np.ndarray          # bool per anchor: frozen only-no-progress guard fires (open-loop replay)
    risk: np.ndarray           # per anchor: LOEO policy-vs-(GC_dist-corrected) cache motion error, sigma units
    risk_raw: np.ndarray       # same, uncorrected cache
    grip: np.ndarray           # gripper sign disagreement fraction over 10 controls (uncorrected cache)
    d1norm: np.ndarray         # distance ratio used by the corrector gate (R10 opus scale)
    extra: dict = field(default_factory=dict)


def guard_flags(steps, prog, eplen, prog_eps=PROG_EPS, noprog_n=NOPROG_N):
    """Frozen R4/R8 noprog_span rule on the anchor sequence (R4 BlindMixedJudge._progress)."""
    span, out = 0, np.zeros(len(steps), bool)
    for i in range(1, len(steps)):
        n = max(int(eplen[i]) - 1, 1)
        span = span + int(steps[i] - steps[i - 1]) if (prog[i] - prog[i - 1]) * n <= prog_eps else 0
        out[i] = span >= noprog_n - 1
    return out


def corrected_err(e_none, dot, cc, d1norm, step):
    """GC_dist-strength corrected error from R10 opus's LOEO head terms: s = .5 * clip((2 - r) / 1.25, 0, 1)."""
    s = DIST_RULE["peak"] * np.clip((DIST_RULE["cutoff"] - d1norm) / (DIST_RULE["cutoff"] - DIST_RULE["plateau"]),
                                    0.0, 1.0)
    s = np.where(step == 0, 0.0, s)
    return np.sqrt(np.maximum(e_none ** 2 - (2 * s * dot - s * s * cc) / 60.0, 0.0))


def cell_name(model, suite, size):
    return f"{model}_{suite}_{size}"


def load_episodes(model, suite, size, anchor_dir=ANCHOR_DIR):
    files = sorted(glob.glob(str(Path(anchor_dir) / f"{cell_name(model, suite, size)}_f[0-4].npz")))
    if len(files) != 5:
        raise FileNotFoundError(f"{cell_name(model, suite, size)}: {len(files)} fold files")
    zs = [np.load(f) for f in files]
    keys = ("ep", "task", "step", "ep_len", "success", "top1_prog", "top1_eplen", "e10", "grip_mis",
            "r10|full|loeo|dot", "r10|full|loeo|cc", "r10|d1norm")
    d = {k: np.concatenate([z[k] for z in zs]) for k in keys}
    o = np.lexsort((d["step"], d["ep"]))
    d = {k: v[o] for k, v in d.items()}
    risk = corrected_err(d["e10"], d["r10|full|loeo|dot"], d["r10|full|loeo|cc"], d["r10|d1norm"], d["step"])
    eps = []
    bounds = np.flatnonzero(np.r_[True, d["ep"][1:] != d["ep"][:-1], True])
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        st = d["step"][lo:hi]
        if not np.array_equal(st, np.arange(hi - lo)):
            raise ValueError("episode rows must be complete 0..len-1")
        a = np.flatnonzero(st % 2 == 0) + lo
        steps = d["step"][a]
        eps.append(Episode(ep=int(d["ep"][lo]), task=int(d["task"][lo]), success=bool(d["success"][lo]),
                           n_slots=int(hi - lo), steps=steps,
                           guard=guard_flags(steps, d["top1_prog"][a], d["top1_eplen"][a]),
                           risk=risk[a], risk_raw=d["e10"][a], grip=d["grip_mis"][a], d1norm=d["r10|d1norm"][a]))
    return eps


# ---------------------------------------------------------------------------------------------- knob schedules
def _u(*key):
    """Deterministic uniform in [0, 1) (keyed SHA256, as R6 Q2's coins)."""
    b = hashlib.sha256(json.dumps(key, separators=(",", ":")).encode()).digest()[:8]
    return (int.from_bytes(b, "big") >> 11) * 2.0 ** -53


class Knob:
    """Per-episode schedule. ``calls(ep, rep)`` returns (guard_call, knob_call) bool arrays over anchors."""
    name = "off"

    def calls(self, ep, rep=0):
        return ep.guard.copy(), np.zeros(len(ep.steps), bool)

    deterministic = True


class Random(Knob):
    """Owner's random miss: each anchor where the guard did not fire calls the policy with probability rho.
    tail: a trigger also forces the next tail-1 anchors (a policy segment of `tail` chunks);
    refractory: no knob trigger within r anchors after a guard call; step0: whether anchor 0 is eligible."""
    deterministic = False

    def __init__(self, rho, tail=1, refractory=0, step0=True, seed=0):
        self.rho, self.tail, self.refractory, self.step0, self.seed = float(rho), int(tail), int(refractory), step0, seed
        self.name = f"random(rho={rho:g},tail={tail},refr={refractory})"

    def calls(self, ep, rep=0):
        n = len(ep.steps)
        g, k = ep.guard.copy(), np.zeros(n, bool)
        left, since_guard = 0, 10 ** 9
        for i in range(n):
            if g[i]:
                since_guard = 0
                left = 0          # a guard call ends any knob segment (it is itself a call)
                continue
            since_guard += 1
            if left > 0:
                k[i], left = True, left - 1
                continue
            if i == 0 and not self.step0:
                continue
            if since_guard <= self.refractory:
                continue
            if _u("r11-random", self.seed, rep, ep.ep, int(ep.steps[i])) < self.rho:
                k[i], left = True, self.tail - 1
        return g, k


class SigmaDelta(Knob):
    """Even-spacing (periodic) miss with an exact rate, per episode, no task index.

    scope='total': target anchor miss fraction F counts every policy call (guard + knob): at a non-guard anchor i
      (0-based), call iff calls_before + 1 <= F * (i + 1) + beta. Guard bursts are thereby refractory and the
      realized IR tracks T = v (c_v + c_m F) whenever the guard alone spends less than T.
    scope='knob': F is the knob's own fraction among non-guard anchors (deterministic twin of Random(rho=F)).
    phase: beta = .5 ('round'), or a per-episode keyed uniform ('random'). tail as in Random (a trigger forces the
    next tail-1 anchors; for 'total' they also count against the budget)."""

    def __init__(self, F, scope="total", phase="random", tail=1, seed=0):
        self.F, self.scope, self.phase, self.tail, self.seed = float(F), scope, phase, int(tail), seed
        self.deterministic = phase != "random"
        self.name = f"sigmadelta(F={F:.3f},{scope},{phase},tail={tail})"

    def calls(self, ep, rep=0):
        n = len(ep.steps)
        g, k = ep.guard.copy(), np.zeros(n, bool)
        beta = 0.5 if self.phase == "round" else _u("r11-sd-phase", self.seed, rep, ep.ep)
        calls, elig, left = 0, 0, 0
        for i in range(n):
            if g[i]:
                left = 0
                if self.scope == "total":
                    calls += 1
                continue
            elig += 1
            if left > 0:
                k[i], left = True, left - 1
                calls += 1
                continue
            denom = (i + 1) if self.scope == "total" else elig
            if calls + 1 <= self.F * denom + beta:
                k[i], left = True, self.tail - 1
                calls += 1
        return g, k


class GapCap(Knob):
    """Max cache run: call when `cap` consecutive cache anchors have passed since the last policy call of any
    source (guard calls reset the counter). Integer cap; the existing plugin cap counts slots instead and can force
    early vision, so this anchor-level version is the one to implement."""

    def __init__(self, cap):
        self.cap = int(cap)
        self.name = f"gapcap({cap})"

    def calls(self, ep, rep=0):
        n = len(ep.steps)
        g, k = ep.guard.copy(), np.zeros(n, bool)
        run = 0
        for i in range(n):
            if g[i]:
                run = 0
                continue
            if run >= self.cap:
                k[i], run = True, 0
            else:
                run += 1
        return g, k


class DitheredGapCap(Knob):
    """Owner's periodic miss, guard-aware ("regular intervals since the last policy call").

    K >= 0 is the mean number of cache anchors allowed between two policy calls. After every call (guard or knob)
    and at episode start, the next allowed run c is floor(K) or floor(K)+1 (prob frac(K), keyed uniform per
    episode and anchor index); the knob calls at the first non-guard anchor with run >= c. Without guard calls the
    anchor miss fraction is ~1/(K+1); guard calls reset the run, so knob calls never follow a guard call sooner
    than c anchors (built-in refractory) and no cache-only stretch exceeds ceil(K) anchors."""

    def __init__(self, K, seed=0, post_guard_tail=0):
        self.K, self.seed, self.pgt = float(K), seed, int(post_guard_tail)
        self.deterministic = float(K).is_integer()
        self.name = f"gapcap(K={K:.3f}{',pgt=' + str(post_guard_tail) if post_guard_tail else ''})"

    def _cap(self, ep, rep, i):
        base = int(np.floor(self.K))
        frac = self.K - base
        return base + (1 if frac and _u("r11-gap", self.seed, rep, ep.ep, int(i)) < frac else 0)

    def calls(self, ep, rep=0):
        n = len(ep.steps)
        g, k = ep.guard.copy(), np.zeros(n, bool)
        run, cap, tail = 0, self._cap(ep, rep, -1), 0
        for i in range(n):
            if g[i]:
                run, cap, tail = 0, self._cap(ep, rep, i), self.pgt
                continue
            if tail > 0:                       # optional escalation: policy also at the next anchor(s)
                k[i], tail = True, tail - 1
                run, cap = 0, self._cap(ep, rep, i)
                continue
            if run >= cap:
                k[i], run, cap = True, 0, self._cap(ep, rep, i)
            else:
                run += 1
        return g, k


def perturb_guard(episodes, factor, seed=0):
    """Stress copy of the episodes with the guard rate scaled by ~factor (thinning or random additions)."""
    out = []
    G = sum(e.guard.sum() for e in episodes)
    A = sum(len(e.guard) for e in episodes)
    g = G / A
    q = 0.0 if factor <= 1 else min(1.0, g * (factor - 1) / max(1 - g, 1e-9))
    for e in episodes:
        new = e.guard.copy()
        for i in range(len(new)):
            u = _u("r11-perturb", seed, e.ep, i)
            if new[i] and factor < 1:
                new[i] = u < factor
            elif not new[i] and i > 0 and factor > 1:
                new[i] = u < q
        out.append(Episode(e.ep, e.task, e.success, e.n_slots, e.steps, new, e.risk, e.risk_raw, e.grip, e.d1norm))
    return out


class FrontLoad(Knob):
    """First n0 anchors (steps 0, 2, ..) are policy calls, then `inner` (default: nothing)."""

    def __init__(self, n0, inner=None):
        self.n0, self.inner = int(n0), inner or Knob()
        self.deterministic = getattr(self.inner, "deterministic", True)
        self.name = f"front({n0})+{self.inner.name}"

    def calls(self, ep, rep=0):
        g, k = self.inner.calls(ep, rep)
        k = k.copy()
        k[: self.n0] |= ~g[: self.n0]
        return g, k


# ---------------------------------------------------------------------------------------------- simulation
def simulate(model, episodes, knob, reps=None):
    """Slot-weighted owner IR, anchor miss fraction and offline value diagnostics of a knob on the library."""
    reps = reps or (1 if knob.deterministic else 8)
    V = M = N = G = K = 0.0
    risk_k = risk_g = risk_all_ng = n_ng = 0.0
    max_runs, runs_ge4 = [], 0.0
    per_ep_ir = []
    for ep in episodes:
        for r in range(reps):
            g, k = knob.calls(ep, r)
            call = g | k
            V += len(ep.steps); M += call.sum(); N += ep.n_slots; G += g.sum(); K += k.sum()
            risk_k += ep.risk[k].sum(); risk_g += ep.risk[g].sum()
            risk_all_ng += ep.risk[~g].sum(); n_ng += (~g).sum()
            # cache-only stretches (consecutive anchors without a call)
            run, mx, ge4 = 0, 0, 0
            for c in call:
                run = 0 if c else run + 1
                mx = max(mx, run)
                ge4 += (run == 4)
            max_runs.append(mx); runs_ge4 += ge4
            per_ep_ir.append((owner_ir(model, len(ep.steps), call.sum(), ep.n_slots), ep.n_slots, ep.success))
    reps_eps = len(episodes) * reps
    mean_ng = risk_all_ng / max(n_ng, 1)
    pe = np.array(per_ep_ir)
    return dict(knob=knob.name, IR=owner_ir(model, V, M, N), v=V / N, f=M / V, g=G / V, knob_frac=K / V,
                knob_frac_of_eligible=K / max(V - G, 1),
                capture_knob=(risk_k / K) / mean_ng if K else float("nan"),
                capture_guard=(risk_g / G) / mean_ng if G else float("nan"),
                mean_max_cache_run=float(np.mean(max_runs)), p90_max_cache_run=float(np.percentile(max_runs, 90)),
                cache_runs_ge4_per_ep=runs_ge4 / reps_eps,
                ir_ep_sd=float(np.sqrt(np.average((pe[:, 0] - owner_ir(model, V, M, N)) ** 2, weights=pe[:, 1]))),
                ir_success_eps=float(np.average(pe[pe[:, 2] == 1, 0], weights=pe[pe[:, 2] == 1, 1]))
                if (pe[:, 2] == 1).any() else float("nan"),
                ir_failed_eps=float(np.average(pe[pe[:, 2] == 0, 0], weights=pe[pe[:, 2] == 0, 1]))
                if (pe[:, 2] == 0).any() else float("nan"))


def base_stats(model, episodes):
    """v, g (slot-weighted ratio over anchors), base IR, by success label too."""
    out = simulate(model, episodes, Knob())
    return dict(v=out["v"], g=out["g"], IR=out["IR"], ir_success_eps=out["ir_success_eps"],
                ir_failed_eps=out["ir_failed_eps"], n_ep=len(episodes),
                success_frac=float(np.mean([e.success for e in episodes])),
                mean_slots=float(np.mean([e.n_slots for e in episodes])))


def solve_knob(model, episodes, make, target, lo=0.0, hi=1.0, iters=30):
    """Bisection on a monotone scalar knob so that the simulated library IR equals target."""
    f_lo, f_hi = simulate(model, episodes, make(lo))["IR"], simulate(model, episodes, make(hi))["IR"]
    if target <= f_lo:
        return lo, f_lo
    if target >= f_hi:
        return hi, f_hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        if simulate(model, episodes, make(mid))["IR"] < target:
            lo = mid
        else:
            hi = mid
    x = (lo + hi) / 2
    return x, simulate(model, episodes, make(x))["IR"]
