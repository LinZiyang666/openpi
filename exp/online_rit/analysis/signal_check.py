"""Q1: does the continuation disagreement carry information beyond the score?

Reads the M1 table and writes ``signal_check.json`` (numbers only; figures are
drawn by local scripts that are not committed). Trajectory-level split: the
``fit`` half fixes the score-decile edges and the per-decile 95% D
thresholds; every statistic below is computed on the ``test`` half.

Per tier:
  (a) Spearman(d, D) overall and the partial correlation after rank-regressing
      both on the score (``rank_residual_spearman``);
  (b) AUROC of ``d`` for ``D > q95(D | score decile)``, per decile with >= 5 of
      each class from >= 10 episodes, weighted by valid test rows;
  (c) episode-level AUROC of max / mean d for success (report only);
  (d) the d distribution against the ``d_self`` floor;
  (e) cross-tier quantiles of d at matched scores.

The release rule (plan §4 M1-4) is written as PASS / FAIL with reasons.
"""

from __future__ import annotations

import argparse

import numpy as np

from exp.online_rit import ladder3
from exp.online_rit.common import load_jsonl, write_json

MIN_PARTIAL = 0.2
MIN_AUROC = 0.65
FLOOR_RATIO = 0.10


def _rank(x: np.ndarray) -> np.ndarray:
    """One-based average ranks (ties share the mean rank), as the Mann-Whitney AUROC needs."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inv, ranks)
    return sums[inv] / counts[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3:
        return None
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


class RankResidualModel:
    """Linear rank-on-rank regressions of d and D on s, fitted once on the fit half.

    The test half is scored with the fit half's coefficients and its rank
    transform is anchored to the fit half's score values (empirical CDF), so
    nothing is re-estimated on test data.
    """

    def __init__(self, d: np.ndarray, y: np.ndarray, s: np.ndarray) -> None:
        if d.size < 5:
            raise ValueError("need at least five fit rows")
        self._s_sorted = np.sort(s)
        self._d_sorted = np.sort(d)
        self._y_sorted = np.sort(y)
        rs = self._cdf(self._s_sorted, s)
        self.beta_d = np.polyfit(rs, self._cdf(self._d_sorted, d), 1)
        self.beta_y = np.polyfit(rs, self._cdf(self._y_sorted, y), 1)

    @staticmethod
    def _cdf(sorted_ref: np.ndarray, v: np.ndarray) -> np.ndarray:
        # mid-rank empirical CDF against the fit sample: rank(v) in [0, 1]
        lo = np.searchsorted(sorted_ref, v, side="left")
        hi = np.searchsorted(sorted_ref, v, side="right")
        return (lo + hi) / (2.0 * sorted_ref.size)

    def residual_spearman(self, d: np.ndarray, y: np.ndarray, s: np.ndarray) -> float | None:
        rs = self._cdf(self._s_sorted, s)
        rd = self._cdf(self._d_sorted, d) - np.polyval(self.beta_d, rs)
        ry = self._cdf(self._y_sorted, y) - np.polyval(self.beta_y, rs)
        return spearman(rd, ry)


def rank_residual_spearman(d: np.ndarray, y: np.ndarray, s: np.ndarray, *, model: RankResidualModel) -> float | None:
    """Spearman between d and D after removing the fit-half rank dependence on s."""
    if d.size < 5:
        return None
    return model.residual_spearman(d, y, s)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Mann-Whitney AUROC with tie-averaged one-based ranks; None without both classes."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    pos = scores[labels]
    neg = scores[~labels]
    if pos.size == 0 or neg.size == 0 or not np.isfinite(scores).all():
        return None
    r = _rank(np.concatenate([pos, neg]))
    return float((r[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def tier_stats(fit: list[dict], test: list[dict], tier, *, deciles: int = 10) -> dict:
    dcol, ycol = tier.d_column, tier.y_column
    f = [r for r in fit if r.get(dcol) is not None and r.get(ycol) is not None]
    t = [r for r in test if r.get(dcol) is not None and r.get(ycol) is not None]
    if not f or not t:
        return {"n_fit": len(f), "n_test": len(t), "estimable": False}
    s_f = np.array([r["s"] for r in f])
    y_f = np.array([r[ycol] for r in f])
    d_f = np.array([r[dcol] for r in f])
    model = RankResidualModel(d_f, y_f, s_f)
    edges = np.quantile(s_f, np.linspace(0, 1, deciles + 1)[1:-1])
    thr = []
    for b in range(deciles):
        m = np.digitize(s_f, edges) == b
        thr.append(float(np.quantile(y_f[m], 0.95)) if m.sum() >= 5 else None)
    s_t = np.array([r["s"] for r in t])
    d_t = np.array([r[dcol] for r in t])
    y_t = np.array([r[ycol] for r in t])
    ep_t = np.array([r["trajectory_id"] for r in t])
    bands = np.digitize(s_t, edges)
    per_decile = []
    weighted = 0.0
    weight = 0
    for b in range(deciles):
        m = bands == b
        if thr[b] is None or m.sum() == 0:
            per_decile.append({"decile": b, "n": int(m.sum()), "auroc": None})
            continue
        lab = y_t[m] > thr[b]
        n_pos, n_neg = int(lab.sum()), int((~lab).sum())
        n_ep = len(set(ep_t[m]))
        a = auroc(d_t[m], lab) if n_pos >= 5 and n_neg >= 5 and n_ep >= 10 else None
        per_decile.append({"decile": b, "n": int(m.sum()), "n_pos": n_pos, "n_neg": n_neg, "n_episodes": n_ep, "auroc": a})
        if a is not None:
            weighted += a * m.sum()
            weight += int(m.sum())
    return {
        "n_fit": len(f), "n_test": len(t), "estimable": weight > 0,
        "spearman_d_D": spearman(d_t, y_t),
        "partial_spearman_given_s": rank_residual_spearman(d_t, y_t, s_t, model=model),
        "auroc_weighted": (weighted / weight) if weight else None,
        "auroc_per_decile": per_decile,
        "d_quantiles": {q: float(np.quantile(d_t, q)) for q in (0.05, 0.5, 0.95)},
    }


def episode_success_auroc(test: list[dict], tier) -> dict:
    by: dict[str, list] = {}
    succ: dict[str, bool] = {}
    for r in test:
        if r.get(tier.d_column) is None:
            continue
        by.setdefault(r["trajectory_id"], []).append(float(r[tier.d_column]))
        succ[r["trajectory_id"]] = bool(r["episode_success"])
    if not by:
        return {"n": 0, "auroc_max": None, "auroc_mean": None}
    ids = sorted(by)
    labels = np.array([not succ[i] for i in ids])  # positive = failure
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos < 5 or n_neg < 5:
        return {"n": len(ids), "n_fail": n_pos, "n_succ": n_neg, "auroc_max": None, "auroc_mean": None}
    mx = np.array([max(by[i]) for i in ids])
    mn = np.array([float(np.mean(by[i])) for i in ids])
    return {"n": len(ids), "n_fail": n_pos, "n_succ": n_neg, "auroc_max": auroc(mx, labels), "auroc_mean": auroc(mn, labels)}


def floor_stats(rows: list[dict], tier) -> dict:
    col = f"d_self_{tier.index}"
    vals = np.array([r[col] for r in rows if r.get(col) is not None], dtype=np.float64)
    d = np.array([r[tier.d_column] for r in rows if r.get(tier.d_column) is not None and not r.get("in_library")], dtype=np.float64)
    return {
        "n_self": int(vals.size),
        "self_median": float(np.median(vals)) if vals.size else None,
        "self_p95": float(np.quantile(vals, 0.95)) if vals.size else None,
        "query_median": float(np.median(d)) if d.size else None,
        "ratio": (float(np.median(vals) / np.median(d)) if vals.size and d.size and np.median(d) > 0 else None),
    }


def cross_tier(test: list[dict], tiers, edges: np.ndarray) -> list[dict]:
    out = []
    s = np.array([r["s"] for r in test if all(r.get(t.d_column) is not None for t in tiers)])
    if s.size == 0:
        return out
    rows = [r for r in test if all(r.get(t.d_column) is not None for t in tiers)]
    bands = np.digitize(s, edges)
    for b in range(len(edges) + 1):
        m = bands == b
        if m.sum() < 30:
            continue
        cell = {"band": b, "n": int(m.sum())}
        for t in tiers:
            v = np.array([rows[i][t.d_column] for i in np.where(m)[0]])
            cell[t.name] = {"p50": float(np.median(v)), "p95": float(np.quantile(v, 0.95))}
        out.append(cell)
    return out


def release(result: dict, tiers) -> dict:
    ok_tiers = 0
    reasons = []
    for t in tiers:
        st = result["tiers"][t.name]
        pc, au = st.get("partial_spearman_given_s"), st.get("auroc_weighted")
        if pc is not None and au is not None and pc >= MIN_PARTIAL and au >= MIN_AUROC:
            ok_tiers += 1
        else:
            reasons.append(f"{t.name}: partial={pc} auroc={au}")
        fl = result["floor"][t.name]
        if fl["ratio"] is None or fl["ratio"] >= FLOOR_RATIO:
            reasons.append(f"{t.name}: d_self floor ratio {fl['ratio']} not < {FLOOR_RATIO}")
    status = "PASS" if ok_tiers >= 2 and not any("floor" in r for r in reasons) else "FAIL"
    return {"status": status, "tiers_passing": ok_tiers, "reasons": reasons}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = [r for r in load_jsonl(args.table) if r.get("s") is not None]
    fit = [r for r in rows if r.get("split") == "fit" and not r.get("in_library")]
    test = [r for r in rows if r.get("split") == "test" and not r.get("in_library")]
    tiers = ladder3.warm_tiers()
    from exp.online_rit.common import sha256_file

    result = {"table": args.table, "table_sha256": sha256_file(args.table), "analysis_sha256": sha256_file(__file__), "n_rows": len(rows), "n_fit": len(fit), "n_test": len(test), "tiers": {}, "episode_success": {}, "floor": {}}
    for t in tiers:
        result["tiers"][t.name] = tier_stats(fit, test, t)
        result["episode_success"][t.name] = episode_success_auroc(test, t)
        result["floor"][t.name] = floor_stats(rows, t)
    s_f = np.array([r["s"] for r in fit]) if fit else np.array([0.0, 1.0])
    result["cross_tier"] = cross_tier(test, tiers, np.quantile(s_f, [0.25, 0.5, 0.75]))
    counts: dict[str, int] = {}
    for r in load_jsonl(args.table):
        key = f"task{r['task_id']}|{'succ' if r['episode_success'] else 'fail'}|{r.get('split')}"
        counts[key] = counts.get(key, 0) + 1
    result["rows_by_task_success_split"] = counts
    result["release"] = release(result, tiers)
    write_json(args.out, result)
    print(f"Q1 release: {result['release']}")


if __name__ == "__main__":
    main()
