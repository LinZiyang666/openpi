"""R11 post-sweep extraction (opus): arm discovery + journal / ledger / decision-log parsing, with a per-arm cache.

Read-only on every run root. Each complete arm is reduced to two numpy tables:
  episodes: one row per (task, init) in the accepted attempt  (success, N slots, V looks, M calls, G guard calls,
            K knob calls, O other calls, audit counters)
  anchors:  one row per vision decision (episode index, step, kind 0 look / 1 guard call / 2 knob call / 3 other call,
            retrieved top-1 library progress, knob probability p, knob score, periodic run/cap, knob setting q,
            top-1 row id, look reason)
plus arm-level audit counters (ledger match, cadence, schedule rule, keyed-coin determinism).

The cache key is the parser version plus (path, size, mtime) of every input file, so a rerun after more arms finish
only parses the new arms.
"""
from __future__ import annotations

import glob
import gzip
import hashlib
import json
import math
import os
import pickle
from collections import defaultdict

import numpy as np

PARSER_VERSION = 3
BASE = "/home/weiland/trace_runs/os_closed_loop"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "..", ".."))
CALIB = os.path.join(REPO, "exp/offline_search/rounds/r11/knob/calibration")

# r11_knob_4 (h100) holds the last 5 arms of r11_local_k4's list; r11_local_k4 keeps an aborted partial copy of one
# of them (no summary.json). discover() keeps exactly one copy per test arm name: the first complete one.
TEST_ROOTS = ["r11_knob_1", "r11_knob_2", "r11_local_k3", "r11_local_k4", "r11_knob_4"]
IDG_ROOT = "r11_local_idg"
R10_ROOTS = ["r10_corr3_pi05", "r10_corr3_groot", "r10_corr3_groot_b"]
R8_ROOT = "r08_main"
CELLS = ["pi05_l10_50", "groot_l10_50", "pi05_spatial_50", "groot_spatial_50",
         "pi05_l10_200", "groot_l10_200", "pi05_l10_500", "groot_l10_500"]
COSTS = {"pi05": (0.152, 0.848), "groot": (0.148, 0.852)}
HOSTS = {"vla-cache": "H100", "weilandserver": "4090"}
FULL_N = 500
MIN_N = FULL_N - 5          # same completeness rule as the coordinator scorer

KIND_LOOK, KIND_GUARD, KIND_KNOB, KIND_OTHER = 0, 1, 2, 3
SRC = {"cache": 0, "cache_blind": 1, "policy": 2, "policy_tail": 3}

ANCHOR_DT = np.dtype([("ep", "i4"), ("step", "i2"), ("kind", "i1"), ("prog", "f4"), ("p", "f4"), ("score", "f4"),
                      ("run", "i2"), ("cap", "i2"), ("q", "f4"), ("top1", "i4"), ("lr", "i1"), ("reason", "i2")])
EP_DT = np.dtype([("task", "i2"), ("init", "i2"), ("success", "i1"), ("N", "i4"), ("V", "i4"), ("M", "i4"),
                  ("G", "i4"), ("K", "i4"), ("O", "i4"), ("attempt", "i2"), ("cad_bad", "i2"), ("has_dec", "i1")])


# ------------------------------------------------------------------------------------------------ keyed coins
def uniform(key, seed, task, init, step, domain):
    """R6 Q2 keyed uniform, exactly as served by knob/recipe.py (payload prefix 'Q2-deploy-v1')."""
    payload = ["Q2-deploy-v1", key, int(seed), int(task), int(init), int(step), domain]
    b = hashlib.sha256(json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode()).digest()[:8]
    return (int.from_bytes(b, "big") >> 11) * 2.0 ** -53


# ------------------------------------------------------------------------------------------------ discovery
def _settings_index():
    out = {}
    for p in sorted(glob.glob(os.path.join(CALIB, "*.json"))):
        d = json.load(open(p))
        for s in d.get("settings", []):
            t = s.get("target")
            out[(s["cell"], s["method"], None if t is None else round(float(t), 4))] = s
    return out


def _complete(root, arm):
    d = f"{BASE}/{root}/runs/{arm}"
    return os.path.exists(f"{d}/summary.json") and os.path.exists(f"{d}/client/journal.jsonl")


def discover():
    """All arms the analysis may use, complete or not. Returns list of dict specs."""
    settings = _settings_index()
    specs = []
    for root in TEST_ROOTS + [IDG_ROOT]:
        p = f"{BASE}/{root}/arms.json"
        if not os.path.exists(p):
            continue
        for r in json.load(open(p)):
            cell = f"{r['model']}_{r['suite_short']}_{r.get('r10_size')}"
            meth = r.get("r11_method") or "off"
            tgt = r.get("target_ir")
            s = settings.get((cell, meth, None if tgt is None else round(float(tgt), 4)), {})
            specs.append(dict(root=root, arm=r["arm"], kind="idg" if root == IDG_ROOT else "r11", model=r["model"],
                              cell=cell, method=meth, target=tgt, pred=r.get("pred_IR_lib"),
                              setting=s.get("setting", s.get("dose")), threshold=s.get("threshold"),
                              tie=s.get("tie_probability"), beta=s.get("beta"), lib_v=s.get("v"), lib_g=s.get("g"),
                              complete=_complete(root, r["arm"])))
    # one copy per test arm name: prefer a complete copy (root order), else the first listed
    test, other = [], []
    for sp in specs:
        (test if sp["kind"] == "r11" else other).append(sp)
    keep = {}
    for sp in test:
        cur = keep.get(sp["arm"])
        if cur is None or (sp["complete"] and not cur["complete"]):
            keep[sp["arm"]] = sp
    specs = [sp for sp in test if keep[sp["arm"]] is sp] + other
    for cell in CELLS:
        model, suite, size = cell.split("_")
        arm = f"r10_{model}_{suite}_{size}_GC_dist"
        for root in R10_ROOTS:
            if _complete(root, arm):
                specs.append(dict(root=root, arm=arm, kind="r10", model=model, cell=cell, method="off_r10", target=None,
                                  pred=None, setting=None, threshold=None, tie=None, beta=None, lib_v=None, lib_g=None,
                                  complete=True))
                break
    for model in ("pi05", "groot"):
        for suite in ("l10", "spatial"):
            arm = f"r8_{model}_{suite}_P10"
            specs.append(dict(root=R8_ROOT, arm=arm, kind="r8", model=model, cell=f"{model}_{suite}", method="pure",
                              target=None, pred=None, setting=None, threshold=None, tie=None, beta=None, lib_v=None,
                              lib_g=None, complete=_complete(R8_ROOT, arm)))
    return specs


# ------------------------------------------------------------------------------------------------ parsing
def _files(spec):
    d = f"{BASE}/{spec['root']}/runs/{spec['arm']}"
    fs = [f"{d}/client/journal.jsonl", f"{d}/summary.json"]
    if spec["kind"] != "r8":
        fs += sorted(glob.glob(f"{d}/server_*/decisions_*.jsonl"))
    return fs


def _cache_key(spec):
    h = hashlib.sha256(f"v{PARSER_VERSION}".encode())
    for f in _files(spec):
        st = os.stat(f)
        h.update(f"{f}|{st.st_size}|{int(st.st_mtime)}".encode())
    for k in ("setting", "threshold", "tie", "beta"):
        h.update(f"{k}={spec.get(k)}".encode())
    return h.hexdigest()[:20]


def read_journal(path):
    out = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        u = r.get("task_uid") or ""
        if ":eval:" not in u or "success" not in r or r.get("error") not in (None, ""):
            continue
        t, i = u.split(":eval:")[1].split(":")[:2]
        out[(int(t), int(i))] = (bool(r["success"]), int(r.get("attempt") or 1))
    return out


def _ex(e, k, d=0.0):
    v = e.get(k, d)
    return d if v is None else v


def parse_arm(spec):
    """Reduce one complete arm to its tables. Pure function of the input files (cached by the caller)."""
    d = f"{BASE}/{spec['root']}/runs/{spec['arm']}"
    jr = read_journal(f"{d}/client/journal.jsonl")
    summ = json.load(open(f"{d}/summary.json"))
    led = summ.get("cost_ledger") or {}
    res = dict(spec=dict(spec), n_journal=len(jr), ledger=dict(decisions=led.get("decisions"),
               vision=led.get("vision_decisions"), misses=led.get("misses"), v=led.get("v"), m=led.get("m")),
               host=None, sr_summary=summ.get("sr"))
    keys = sorted(jr)
    if spec["kind"] == "r8":
        ep = np.zeros(len(keys), EP_DT)
        for j, k in enumerate(keys):
            ep[j]["task"], ep[j]["init"] = k
            ep[j]["success"] = jr[k][0]
            ep[j]["attempt"] = jr[k][1]
        res["episodes"], res["anchors"], res["audit"] = ep, np.zeros(0, ANCHOR_DT), {}
        return res

    rows = defaultdict(dict)          # (task, init, attempt) -> step -> tuple
    hosts = set()
    for f in sorted(glob.glob(f"{d}/server_*/decisions_*.jsonl")):
        with open(f, "rb") as fh:
            for line in fh:
                if line.startswith(b'{"ev": "startup"'):
                    try:
                        hosts.add(json.loads(line).get("host"))
                    except Exception:
                        pass
                    continue
                if not line.startswith(b'{"ev": "dec"'):
                    continue
                r = json.loads(line)
                u = r["uid"]
                t, i = u.split(":eval:")[1].split(":")[:2]
                key = (int(t), int(i), int(r.get("attempt") or 1))
                e = r.get("extras") or {}
                rec = (bool(r["vision"]), bool(r["hit"]), SRC.get(r.get("src"), 9), r.get("look_reason"),
                       float(_ex(e, "os_reason")), float(_ex(e, "os_force_miss")),
                       float(_ex(e, "os_r11_guard", -1.0)), float(_ex(e, "os_r11_knob_call", -1.0)),
                       float(_ex(e, "os_r11_p", np.nan)), float(_ex(e, "os_r11_score", np.nan)),
                       float(_ex(e, "os_r11_run", -1.0)), float(_ex(e, "os_r11_cap", -1.0)),
                       float(_ex(e, "os_r11_setting", np.nan)), float(_ex(e, "os_r11_coin", np.nan)),
                       float(_ex(e, "top1_prog", np.nan)), int(r.get("top1") if r.get("top1") is not None else -1),
                       float(r.get("ts") or 0.0))
                old = rows[key].get(int(r["step"]))
                if old is None or rec[-1] >= old[-1]:
                    rows[key][int(r["step"])] = rec
    res["host"] = ",".join(sorted(HOSTS.get(h, str(h)) for h in hosts))
    by_ep = defaultdict(dict)
    for (t, i, a), st in rows.items():
        by_ep[(t, i)][a] = st

    meth = spec["method"]
    setting = spec.get("setting")
    audit = dict(rule_checked=0, rule_bad=0, coin_checked=0, coin_bad=0, run_bad=0, cap_bad=0, p_checked=0, p_bad=0,
                 guard_flag_bad=0, missing_dec=0, dup_attempts=0, step_gaps=0, miss_reasons={})
    ep = np.zeros(len(keys), EP_DT)
    anchors = []
    for j, k in enumerate(keys):
        succ, att = jr[k]
        e = ep[j]
        e["task"], e["init"], e["success"], e["attempt"] = k[0], k[1], succ, att
        atts = by_ep.get(k)
        if not atts:
            audit["missing_dec"] += 1
            continue
        if len(atts) > 1:
            audit["dup_attempts"] += 1
        st = atts.get(att) or atts[max(atts)]
        steps = sorted(st)
        if steps != list(range(len(steps))):
            audit["step_gaps"] += 1
        e["has_dec"] = 1
        e["N"] = len(steps)
        seq = [st[s] for s in steps]
        # history for schedule audits
        last_call = -1
        prev_anchor = -1
        prev_anchor_trigger = False      # random_tail2: previous anchor was a (non-tail) knob trigger
        prev_anchor_guard = False        # periodic_pgt1: previous anchor was a guard call
        run = 0
        for s, rec in zip(steps, seq):
            vis, hit, src, lr, reason, force, g_flag, k_flag, p, score, lrun, lcap, q, coin, prog, top1, _ = rec
            if not vis:
                continue
            miss = not hit
            if g_flag >= 0:
                guard = g_flag == 1.0
                knob = k_flag == 1.0
                if guard and not (force == 1.0 and miss):
                    audit["guard_flag_bad"] += 1
            else:                       # knob-off arms: every forced call is a guard call
                guard = miss and force == 1.0
                knob = False
            if miss:
                kind = KIND_GUARD if guard else (KIND_KNOB if knob else KIND_OTHER)
                rk = str(int(reason))
                audit["miss_reasons"][rk] = audit["miss_reasons"].get(rk, 0) + 1
            else:
                kind = KIND_LOOK
                if guard or knob:
                    audit["guard_flag_bad"] += 1
            # ---- schedule audits (opus methods, eligible anchors only)
            if meth in ("random", "random_tail2", "periodic", "periodic_pgt1") and not guard and setting is not None:
                if meth.startswith("periodic"):
                    u = uniform("R11-gap-v1", 0, k[0], k[1], last_call, "cap")
                    K = float(setting)
                    base = math.floor(K)
                    cap = base + int(K - base > 0 and u < K - base)
                    tail = meth == "periodic_pgt1" and prev_anchor_guard and last_call == prev_anchor and prev_anchor >= 0
                    exp_call = tail or run >= cap
                    audit["coin_checked"] += 1
                    audit["coin_bad"] += int(not (abs(coin - u) < 1e-12))
                    audit["cap_bad"] += int(lcap != cap)
                    audit["run_bad"] += int(lrun != run)
                else:
                    u = uniform("R11-random-v1", 0, k[0], k[1], s, "knob-anchor")
                    tail = meth == "random_tail2" and prev_anchor_trigger and last_call == prev_anchor
                    exp_call = tail or u < float(setting)
                    audit["coin_checked"] += 1
                    audit["coin_bad"] += int(not (abs(coin - u) < 1e-12))
                audit["rule_checked"] += 1
                audit["rule_bad"] += int(exp_call != knob)
            elif meth in ("distance", "disagreement", "error_hybrid") and not guard and spec.get("threshold") is not None:
                thr, tie, beta, dose = spec["threshold"], spec["tie"] or 0.0, spec["beta"], float(spec["setting"])
                high = float(score > thr) + float(score == thr) * tie
                p_exp = high if beta == 1.0 else (1 - beta) * dose + beta * high
                audit["p_checked"] += 1
                audit["p_bad"] += int(abs(p_exp - p) > 1e-6)
            anchors.append((j, s, kind, prog, p, score, lrun, lcap, q, top1, -1 if lr is None else int(lr), int(reason)))
            # ---- update history
            if meth == "random_tail2":
                was_tail = prev_anchor_trigger and last_call == prev_anchor
                prev_anchor_trigger = knob and not was_tail
            prev_anchor_guard = guard
            if miss:
                last_call = s
                run = 0
            else:
                run += 1
            prev_anchor = s
        vis = np.array([r[0] for r in seq])
        hit = np.array([r[1] for r in seq])
        src = np.array([r[2] for r in seq])
        missv = vis & ~hit
        e["V"], e["M"] = int(vis.sum()), int(missv.sum())
        # cadence: every call is followed by exactly one policy-tail slot, then a fresh vision decision
        bad = 0
        for s in np.flatnonzero(missv):
            if s + 1 < len(seq) and src[s + 1] != SRC["policy_tail"]:
                bad += 1
            elif s + 2 < len(seq) and not vis[s + 2]:
                bad += 1
        e["cad_bad"] = bad
    an = np.array(anchors, ANCHOR_DT) if anchors else np.zeros(0, ANCHOR_DT)
    if len(an):
        cnt = lambda kind: np.bincount(an["ep"][an["kind"] == kind], minlength=len(ep))
        ep["G"], ep["K"], ep["O"] = cnt(KIND_GUARD), cnt(KIND_KNOB), cnt(KIND_OTHER)
    res["episodes"], res["anchors"], res["audit"] = ep, an, audit
    return res


def load_arm(spec):
    os.makedirs(CACHE, exist_ok=True)
    key = _cache_key(spec)
    p = os.path.join(CACHE, f"{spec['root']}__{spec['arm']}__{key}.pkl.gz")
    if os.path.exists(p):
        with gzip.open(p, "rb") as fh:
            return pickle.load(fh)
    res = parse_arm(spec)
    tmp = p + f".tmp{os.getpid()}"
    with gzip.open(tmp, "wb", compresslevel=3) as fh:
        pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, p)
    for old in glob.glob(os.path.join(CACHE, f"{spec['root']}__{spec['arm']}__*.pkl.gz")):
        if old != p:
            os.remove(old)
    return res
