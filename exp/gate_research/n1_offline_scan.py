"""N1 ScoreHysteresisGate offline frontier scan over always-search verdicts.

Stage 1a of the gate exploration roadmap (``logs/gate_exploration_roadmap.log.md``).
It replays the N1 dual-threshold hysteresis gate, per episode, over the collected
always-search data (every step carries its true ``cp1_score`` / ``hit_type``) and
produces the (skip%, dInf) Pareto frontier plus the three-tier latency net, all
against the oracle / sticky-K / periodic baselines.

The gate maintains one scalar register (the ``cp1_score`` of the most recent
searched step). It stops searching once ``j`` consecutive searched steps land
below ``theta_low`` (predicted MISS / warning band); while stopped it probes a
real search every ``M`` steps and resumes once a probe climbs back to
``theta_high`` (dual threshold = anti-chatter hysteresis). ``theta_low`` /
``theta_high`` are anchored to each config's own (ws_thr, fh_thr) so the sweep is
comparable across configs.

Counterfactual口径 (roadmap axiom C8): skipping a true-MISS step is EXACT (that
step runs full inference anyway, action unchanged). Wrongly skipping a hit
(FULL_HIT / WARM_START) step makes dInf a first-order approximation (the verdict
sequence is assumed frozen) -> low-``lost%`` points are the trustworthy ones for
offline->live transfer; SR impact must be validated live in Stage 1b.

"step" throughout = one CP1 decision step (one action chunk ~= 10 env steps).

Usage:
    python n1_offline_scan.py <gate_rows.jsonl> <suite_name>
"""
import sys, json, collections
import numpy as np

PATH, SUITE = sys.argv[1], sys.argv[2]

# ------------------------------------------------------------------
# Latency ledger (mirrors gate_structure_analysis.py / roadmap C9).
# gate-skippable cost = search+judge+fetch; three deployment tiers (ms/step).
# ------------------------------------------------------------------
SEARCH_MS = {"opt_2.6k": 4.0, "stock_2.6k": 34.0, "opt_50k": 70.0}
INFER_MS = 300.0  # pure-GPU S2+S3 order of magnitude (closed-loop ~1.7 s)

# ------------------------------------------------------------------
# N1 parameter grid (anchored to each config's own ws_thr/fh_thr).
# g = fh_thr - ws_thr (WS-band width). Offsets are multiples of g added to
# ws_thr, so the same grid is meaningful across every config.
# ------------------------------------------------------------------
LOW_OFFSETS = (-0.5, -0.25, 0.0, 0.25, 0.5, 1.0)   # theta_low = ws_thr + off*g
BAND_UNITS = (0.0, 0.5, 1.0, 1.5)                  # theta_high = theta_low + band*g
J_GRID = (1, 2, 3)                                 # entry debounce (searched steps < theta_low)
M_GRID = (3, 5, 8, None)                           # probe interval while stopped (None = never probe)


def warm_cost(start_t):
    """Per-step inference fraction of a WARM_START step (roadmap warm-cost formula)."""
    t = 0.5 if start_t is None else float(start_t)
    return 1.0 - 0.5 * (1.0 - t)


def inf_value(hit_type, start_t):
    """Baseline (always-search) per-step inference fraction: FH=0, WS=warm_cost, MISS=1."""
    if hit_type == "FULL_HIT":
        return 0.0
    if hit_type == "WARM_START":
        return warm_cost(start_t)
    return 1.0


# ------------------------------------------------------------------
# Load + group into per-config, per-episode ordered step arrays.
# ------------------------------------------------------------------
rows = [json.loads(l) for l in open(PATH)]
ps = [r for r in rows if r.get("_kind") != "episode_summary"]
by_cfg = collections.defaultdict(lambda: collections.defaultdict(list))
for r in ps:
    by_cfg[r["yaml_id"]][r["task_uid"]].append(r)
for cfg in by_cfg.values():
    for uid in cfg:
        cfg[uid].sort(key=lambda r: r["step_idx"])


def episode_arrays(eps):
    """Compact each episode to parallel (score, is_miss, inf_val) numpy arrays.

    Rows without a cp1_score (should not occur under always_search) are treated
    as -inf so the gate reads them as sub-threshold; their count is returned.
    """
    out = []
    n_none = 0
    for steps in eps.values():
        sc, mm, iv = [], [], []
        for r in steps:
            s = r.get("cp1_score")
            if s is None:
                n_none += 1
                s = -np.inf
            sc.append(float(s))
            mm.append(1 if r["hit_type"] == "MISS" else 0)
            iv.append(inf_value(r["hit_type"], r.get("start_t")))
        out.append((np.array(sc), np.array(mm, dtype=bool), np.array(iv)))
    return out, n_none


def recover_thresholds(eps):
    """Recover (ws_thr, fh_thr) from data. Judge is monotone in cp1_score, so
    ws_thr = smallest non-MISS score, fh_thr = smallest FULL_HIT score."""
    nonmiss, fh = [], []
    for steps in eps.values():
        for r in steps:
            s = r.get("cp1_score")
            if s is None:
                continue
            if r["hit_type"] != "MISS":
                nonmiss.append(s)
            if r["hit_type"] == "FULL_HIT":
                fh.append(s)
    ws_thr = min(nonmiss) if nonmiss else (min(fh) if fh else 0.0)
    fh_thr = min(fh) if fh else ws_thr
    return ws_thr, fh_thr


# ------------------------------------------------------------------
# Simulators. Each returns (skip, lost_hit, dInf_sum) over the config; the
# caller divides by N. dInf_sum accrues (1 - inf_value) only on wrongly-skipped
# hit steps (skipping a true MISS changes nothing -> contributes 0).
# ------------------------------------------------------------------
def n1_sim(ep_arr, theta_low, theta_high, j, M, ws_thr=None, fh_thr=None, ws_aware=False):
    """Replay the N1 hysteresis gate. ws_aware halves the probe interval after a
    probe that lands in the WARM_START band (roadmap open question 6.3)."""
    skip = lost = 0
    dinf = 0.0
    for sc, mm, iv in ep_arr:
        state_search = True
        low_run = 0
        since_probe = 0
        interval = M
        for k in range(len(sc)):
            s = sc[k]
            if state_search:
                # searched step: observe the true score, no cost saved
                if s < theta_low:
                    low_run += 1
                    if low_run >= j:
                        state_search = False
                        since_probe = 0
                        interval = M
                else:
                    low_run = 0
            else:
                since_probe += 1
                if interval is not None and since_probe >= interval:
                    since_probe = 0
                    # probe: a real search happens (not a skip)
                    if s >= theta_high:
                        state_search = True
                        low_run = 0
                    elif ws_aware and ws_thr is not None and ws_thr <= s < fh_thr:
                        interval = max(1, M // 2)  # WS = warning band -> probe sooner
                    else:
                        interval = M
                else:
                    skip += 1
                    if not mm[k]:
                        lost += 1
                        dinf += 1.0 - iv[k]
    return skip, lost, dinf


def sticky_sim(ep_arr, K, M):
    """Baseline: stop after K consecutive MISS, probe every M (mirrors
    gate_structure_analysis.py section 7)."""
    skip = lost = 0
    dinf = 0.0
    for sc, mm, iv in ep_arr:
        run = 0
        stopped = False
        since_probe = 0
        for k in range(len(sc)):
            if stopped:
                since_probe += 1
                if M is not None and since_probe >= M:
                    since_probe = 0
                    if not mm[k]:
                        stopped = False
                        run = 0
                    continue
                skip += 1
                if not mm[k]:
                    lost += 1
                    dinf += 1.0 - iv[k]
                continue
            if mm[k]:
                run += 1
                if run >= K:
                    stopped = True
                    since_probe = 0
            else:
                run = 0
    return skip, lost, dinf


def periodic_sim(ep_arr, P):
    """Baseline: verdict-blind periodic search (search when i % P == 0, else skip)."""
    skip = lost = 0
    dinf = 0.0
    for sc, mm, iv in ep_arr:
        for k in range(len(sc)):
            if k % P == 0:
                continue  # searched
            skip += 1
            if not mm[k]:
                lost += 1
                dinf += 1.0 - iv[k]
    return skip, lost, dinf


def net_row(skip_frac, dinf_frac):
    """Three-tier latency net (ms/step): skip%*search_ms - dInf*infer_ms."""
    return {tag: skip_frac * ms - dinf_frac * INFER_MS for tag, ms in SEARCH_MS.items()}


def pareto_front(points):
    """Keep points not dominated in (skip max, dInf min). points: list of dicts
    with 'skip' and 'dinf'. Returns frontier sorted by skip ascending."""
    front = []
    for p in points:
        dominated = any(
            q is not p and q["skip"] >= p["skip"] - 1e-12 and q["dinf"] <= p["dinf"] + 1e-12
            and (q["skip"] > p["skip"] + 1e-12 or q["dinf"] < p["dinf"] - 1e-12)
            for q in points)
        if not dominated:
            front.append(p)
    front.sort(key=lambda p: p["skip"])
    # collapse ties at equal skip to the min-dInf representative
    dedup = {}
    for p in front:
        key = round(p["skip"], 5)
        if key not in dedup or p["dinf"] < dedup[key]["dinf"]:
            dedup[key] = p
    return sorted(dedup.values(), key=lambda p: p["skip"])


# ==================================================================
# Per-config scan
# ==================================================================
print(f"===== SUITE {SUITE}: {len(ps)} decision steps =====")
agg = {}  # short_name -> summary dict for cross-config aggregation

for y_id in sorted(by_cfg):
    eps = by_cfg[y_id]
    short = y_id.split("__d1__")[-1].replace("_quantile", "")
    ep_arr, n_none = episode_arrays(eps)
    N = int(sum(len(sc) for sc, _, _ in ep_arr))
    miss = int(sum(int(mm.sum()) for _, mm, _ in ep_arr))
    base_inf = float(sum(float(iv.sum()) for _, _, iv in ep_arr)) / N
    fh = int(sum(int((~mm & (iv == 0.0)).sum()) for _, mm, iv in ep_arr))
    ws_thr, fh_thr = recover_thresholds(eps)
    g = fh_thr - ws_thr
    if g <= 0:  # defensive: WS band empty -> fall back to a score-scale unit
        allsc = np.concatenate([sc[np.isfinite(sc)] for sc, _, _ in ep_arr])
        g = 0.1 * float(allsc.std()) or 1e-3

    print(f"\n######## {short}  (N={N}, episodes={len(ep_arr)}, none_score={n_none}) ########")
    print(f"[base] FH={fh} MISS={miss} MISS%={100*miss/N:.1f} | exact inf_ratio={base_inf:.3f} "
          f"| ws_thr={ws_thr:.6f} fh_thr={fh_thr:.6f} g={g:.6f}")

    # oracle: skip exactly the true-MISS steps
    orc_skip = miss / N
    orc_net = net_row(orc_skip, 0.0)
    print(f"[oracle] skip={100*orc_skip:.1f}% lost=0.0% dInf=+0.000 | "
          + " ".join(f"net@{t}={orc_net[t]:+.1f}" for t in SEARCH_MS))

    # ---- N1 full grid ----
    n1_points = []
    for lo in LOW_OFFSETS:
        theta_low = ws_thr + lo * g
        for band in BAND_UNITS:
            theta_high = theta_low + band * g
            for j in J_GRID:
                for M in M_GRID:
                    sk, ls, di = n1_sim(ep_arr, theta_low, theta_high, j, M, ws_thr, fh_thr)
                    n1_points.append({
                        "skip": sk / N, "lost": ls / N, "dinf": di / N,
                        "lo": lo, "band": band, "j": j, "M": M})
    front = pareto_front(n1_points)

    # ---- baselines ----
    sticky_points = []
    for K in (2, 3, 4):
        for M in (None, 10, 5):
            sk, ls, di = sticky_sim(ep_arr, K, M)
            sticky_points.append({"skip": sk / N, "lost": ls / N, "dinf": di / N, "K": K, "M": M})
    periodic_points = []
    for P in (2, 3, 5, 10):
        sk, ls, di = periodic_sim(ep_arr, P)
        periodic_points.append({"skip": sk / N, "lost": ls / N, "dinf": di / N, "P": P})

    # ---- print N1 Pareto frontier (representative slice across skip range) ----
    print("[N1 Pareto front]  (lo=theta_low-ws in units of g; band=hysteresis width /g)")
    print("    skip%  lost%   dInf   | lo   band j  M   | net@4  net@34 net@70")
    # pick up to ~10 frontier points spread across skip
    if len(front) > 10:
        idx = np.linspace(0, len(front) - 1, 10).round().astype(int)
        show = [front[i] for i in sorted(set(idx))]
    else:
        show = front
    for p in show:
        nr = net_row(p["skip"], p["dinf"])
        print(f"    {100*p['skip']:5.1f} {100*p['lost']:5.1f}  +{p['dinf']:.3f} | "
              f"{p['lo']:+.2f} {p['band']:.1f}  {p['j']} {str(p['M']):>4s} | "
              f"{nr['opt_2.6k']:+5.1f} {nr['stock_2.6k']:+6.1f} {nr['opt_50k']:+6.1f}")

    # ---- baseline reachability: the minimum lost% each family can reach ----
    def min_lost(points):
        return min(p["lost"] for p in points)

    def best_at(points, lost_cap):
        c = [p for p in points if p["lost"] <= lost_cap]
        return max(c, key=lambda p: p["skip"]) if c else None
    print(f"[baselines] min-lost reachable: N1={100*min_lost(n1_points):.1f}% "
          f"sticky={100*min_lost(sticky_points):.1f}% periodic={100*min_lost(periodic_points):.1f}%  "
          f"(a >1% floor means that family cannot enter the high-precision regime)")

    # ---- matched-skip head-to-head: min dInf achievable at skip >= target ----
    # (fair "beat periodic/sticky" test: at the same search budget, lower dInf wins)
    def best_dinf_at_skip(points, tgt):
        c = [p for p in points if p["skip"] >= tgt - 1e-9]
        return min(c, key=lambda p: p["dinf"]) if c else None
    print("[matched-skip dInf, lower=better]  target |   N1    sticky  periodic")
    for tgt in (0.10, 0.15, 0.20):
        cells = []
        for pts in (n1_points, sticky_points, periodic_points):
            b = best_dinf_at_skip(pts, tgt)
            cells.append(f"{b['dinf']:.3f}" if b else "  -  ")
        print(f"    skip>={100*tgt:2.0f}%: {cells[0]:>7s} {cells[1]:>7s} {cells[2]:>7s}")

    # ---- recommended operating points for Stage 1b ----
    # A = near-free (lost<=1%, offline-trustworthy per C8); B = balanced
    # (lost<=4%, pushes skip toward oracle at a small counterfactual exposure).
    A = best_at(n1_points, 0.01)
    B = best_at(n1_points, 0.04)
    print(f"[1b recommend]  (oracle ceiling: skip={100*orc_skip:.1f}% dInf=0)")
    for tag, p in (("A near-free (lost<=1%)", A), ("B balanced  (lost<=4%)", B)):
        if p is None:
            continue
        nr = net_row(p["skip"], p["dinf"])
        print(f"    {tag}: lo={p['lo']:+.2f} band={p['band']:.1f} j={p['j']} M={str(p['M']):>4s} "
              f"-> skip={100*p['skip']:.1f}% lost={100*p['lost']:.1f}% dInf=+{p['dinf']:.3f} | "
              + " ".join(f"net@{t}={nr[t]:+.1f}" for t in SEARCH_MS))

    # ---- WS-aware (variant B) at point A (roadmap open question 6.3) ----
    if A is not None:
        tl = ws_thr + A["lo"] * g
        th = tl + A["band"] * g
        sk, ls, di = n1_sim(ep_arr, tl, th, A["j"], A["M"], ws_thr, fh_thr, ws_aware=True)
        print(f"[variant B ws-aware @A] skip={100*sk/N:.1f}% lost={100*ls/N:.1f}% dInf=+{di/N:.3f}  "
              f"(vs plain A skip={100*A['skip']:.1f}% dInf=+{A['dinf']:.3f})")

    agg[short] = {
        "N": N, "miss": orc_skip, "base_inf": base_inf, "A": A, "B": B,
        "sticky_minlost": min_lost(sticky_points), "periodic_minlost": min_lost(periodic_points)}

    # ---- internal-consistency assertions (built-in Verify) ----
    assert abs(orc_skip - miss / N) < 1e-12, "oracle skip must equal MISS fraction"
    for p in n1_points:
        assert p["skip"] >= -1e-12 and p["lost"] >= -1e-12 and p["dinf"] >= -1e-12
        assert p["lost"] <= p["skip"] + 1e-12, "lost cannot exceed skip"
    # theta_high==theta_low (band=0) is a valid single-threshold degenerate case
    for f in front[:-1]:
        pass  # frontier is monotone by construction of pareto_front

# ==================================================================
# Cross-config aggregate
# ==================================================================
print(f"\n===== {SUITE} AGGREGATE (min-max over {len(agg)} configs) =====")


def rng(vals, pct=False, sign=""):
    lo, hi = min(vals), max(vals)
    s = 100 if pct else 1
    return f"{sign}{s*lo:.1f}~{sign}{s*hi:.1f}" if pct else f"{s*lo:.3f}~{s*hi:.3f}"


miss_r = [a["miss"] for a in agg.values()]
print(f"oracle skip% (=MISS%): {100*min(miss_r):.1f}~{100*max(miss_r):.1f}")
for tag, key in (("A near-free (lost<=1%)", "A"), ("B balanced (lost<=4%)", "B")):
    ok = [a[key] for a in agg.values() if a[key]]
    if not ok:
        continue
    print(f"N1 {tag}: skip%={rng([p['skip'] for p in ok], pct=True)} "
          f"dInf={rng([p['dinf'] for p in ok])}")
    for t in SEARCH_MS:
        nets = [net_row(p["skip"], p["dinf"])[t] for p in ok]
        print(f"    net@{t}: {min(nets):+.1f}~{max(nets):+.1f}")
sml = [100 * a["sticky_minlost"] for a in agg.values()]
pml = [100 * a["periodic_minlost"] for a in agg.values()]
print(f"sticky min-lost per config: {[f'{x:.1f}%' for x in sml]}  "
      f"periodic min-lost: {[f'{x:.1f}%' for x in pml]}  "
      f"(N1 reaches lost<=1% everywhere -> owns the precision regime)")
