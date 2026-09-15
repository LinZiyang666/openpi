"""Closed-form and pairwise-ranking weight fits on phase labels (companion of lcw_offline_stats).

Same population as the statistics script (leave-one-trajectory-out, same-task
candidates). A candidate is *phase-aligned* with its query when the normalized
trajectory progress differs by at most ``delta``. Three fits of the fusion
weights on that label:

* ``fisher``     diagonal LDA, ``w_f ∝ d_f / var_f`` (independent fields);
* ``lda``        full LDA, ``w = Σ⁻¹ d`` with the pooled within-class 3x3
                 covariance (accounts for redundancy between fields);
* ``pairwise``   logistic regression on (aligned − misaligned) score
                 differences inside each query's candidate list (RankNet with
                 a linear scorer), L2-regularised, unconstrained.

Negative components are reported raw and then clipped to the simplex, since
the served fusion only accepts non-negative weights. Each fit is read against
the closed-loop cells exactly like the priors in lcw_offline_stats.

The template is checked before anything is fitted: all three fields must carry
a zscore+tanh normalizer with finite mu and positive sigma (the only normalizer
the served fusion and this fit agree on), every key must be finite, and each
trajectory's steps must be numbered 0..len-1 (progress is step/(len-1)). The
primary fit (``lda@0.05``) must have at least one query with both classes and
a finite, non-zero clipped weight vector, otherwise the script stops instead of
writing a fallback. The output records the library and template by content
(sha256) and the per-field normalizers so a consumer can tell exactly which
inputs a fit belongs to.

Usage:
  python exp/weighted_sum/analysis/lcw_fit_weights.py --library <pkl> --template <yaml> \
      --cells <json> --output <dir> [--deltas 0.05,0.1,0.2] [--pairs 200000]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time

import numpy as np
import torch
import yaml

from lcw_offline_stats import FIELDS, load_entries, nearest_cell, normalize_w, normalized_scores, smoothed_sr


ACT = {"act": None}  # executed-horizon action block per entry, set by main()
PRIMARY = "lda@0.05"  # the fit consumers read; a degenerate primary aborts the run


def check_template(ss: dict) -> dict:
    """Return ``{field: {sim_type, method, params}}``; reject anything the fit cannot use."""
    norm = ss.get("score_normalization") or {}
    if norm.get("type") != "per_field":
        raise SystemExit(f"template score_normalization.type={norm.get('type')!r}, need per_field")
    out = {}
    for f in FIELDS:
        entry = (norm.get("fields") or {}).get(f)
        sim = (ss.get("field_similarity") or {}).get(f)
        if entry is None or sim is None:
            raise SystemExit(f"template lacks normalizer/similarity for {f!r}; the fit needs all of {FIELDS}")
        p = entry.get("params") or {}
        if entry.get("method") != "zscore" or p.get("squash", "tanh") != "tanh":
            raise SystemExit(f"template {f!r} uses {entry.get('method')}/{p.get('squash')}, the fit assumes zscore+tanh")
        mu, sigma = p.get("mu"), p.get("sigma")
        if not all(isinstance(x, (int, float)) and np.isfinite(x) for x in (mu, sigma)) or sigma <= 0:
            raise SystemExit(f"template {f!r} has unusable mu/sigma {mu}/{sigma}")
        out[f] = {"sim_type": sim["type"], "method": "zscore", "params": {"mu": float(mu), "sigma": float(sigma), "squash": "tanh"}}
    return out


def check_progress(traj: np.ndarray, step: np.ndarray) -> None:
    """Every trajectory must be numbered 0..len-1 without gaps, or progress is meaningless."""
    for t in np.unique(traj):
        steps = np.sort(step[traj == t])
        if not np.array_equal(steps, np.arange(len(steps))):
            raise SystemExit(f"trajectory index {t}: steps {steps[:5].tolist()}... are not 0..{len(steps) - 1}")


def degenerate(raw, w) -> bool:
    """A fit is unusable when anything is non-finite or the clipped vector has no mass."""
    raw = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(raw)):
        return True
    clipped = np.clip(raw, 0.0, None)
    return not (clipped.sum() > 0) or w is None or not np.all(np.isfinite(np.asarray(w, dtype=np.float64)))


def sha256_file(path: pathlib.Path) -> str:
    """Hash an input file without retaining another copy of a large library."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_normalize(v):
    """``normalize_w`` unless the clipped vector has no mass or a non-finite entry (then ``None``)."""
    v = np.asarray(v, dtype=np.float64)
    if not np.all(np.isfinite(v)) or not (np.clip(v, 0.0, None).sum() > 0):
        return None
    return normalize_w(v)


def per_query_labels(M, progress, delta, far=None, label="phase"):
    """Yield (query, candidate indices, positive mask, negative mask) per query.

    ``label="phase"``: positives are candidates within ``delta`` of the query's
    normalized progress; ``far`` widens the gap between the two classes so the
    separation measured is "right stage" vs "wrong stage".
    ``label="action"``: positives are the 10% of candidates whose executed
    action block is closest to the query's own (layout-aware, scale-free),
    negatives the farthest 50%.
    ``label="both"``: positives must satisfy both, negatives either.
    """
    for i in np.flatnonzero(M.any(axis=1)):
        cand = np.flatnonzero(M[i])
        gap = np.abs(progress[cand] - progress[i])
        pos_p = gap <= delta
        neg_p = (gap > far) if far is not None else ~pos_p
        if label == "phase":
            yield i, cand, pos_p, neg_p
            continue
        act = ACT["act"]
        dist = np.sqrt(((act[cand] - act[i]) ** 2).sum(axis=1))
        pos_a = dist <= np.quantile(dist, 0.1)
        neg_a = dist >= np.quantile(dist, 0.5)
        if label == "action":
            yield i, cand, pos_a, neg_a
        else:
            yield i, cand, pos_p & pos_a, neg_p | neg_a


def fisher_and_lda(N, M, progress, delta, far=None, label="phase"):
    d_sum = np.zeros(3)
    cov_sum = np.zeros((3, 3))
    var_sum = np.zeros(3)
    count = 0
    for i, cand, pos, neg in per_query_labels(M, progress, delta, far, label):
        if not pos.any() or not neg.any():
            continue
        X = np.stack([N[f][i, cand] for f in FIELDS], axis=1)  # [C, 3]
        mp, mn = X[pos].mean(axis=0), X[neg].mean(axis=0)
        d_sum += mp - mn
        cp = np.cov(X[pos].T, bias=True) if pos.sum() > 1 else np.zeros((3, 3))
        cn = np.cov(X[neg].T, bias=True) if neg.sum() > 1 else np.zeros((3, 3))
        cov_sum += 0.5 * (cp + cn)
        var_sum += 0.5 * (X[pos].var(axis=0) + X[neg].var(axis=0))
        count += 1
    if count == 0:
        return {"d": None, "var": None, "cov": None, "fisher_raw": None, "fisher": None,
                "lda_raw": None, "lda": None, "queries": 0}
    d, cov, var = d_sum / count, cov_sum / count, var_sum / count
    with np.errstate(divide="ignore", invalid="ignore"):
        fisher = d / var
    lda = np.linalg.solve(cov + 1e-6 * np.eye(3), d)
    return {"d": d.tolist(), "var": var.tolist(), "cov": cov.tolist(),
            "fisher_raw": fisher.tolist(), "fisher": safe_normalize(fisher),
            "lda_raw": lda.tolist(), "lda": safe_normalize(lda), "queries": count}


def pairwise_logistic(N, M, progress, delta, n_pairs, far=None, label="phase", seed=0, l2=1e-3, steps=300):
    rng = np.random.default_rng(seed)
    diffs = []
    per_query = []
    for i, cand, pos, neg in per_query_labels(M, progress, delta, far, label):
        if not pos.any() or not neg.any():
            continue
        per_query.append((i, cand[pos], cand[neg]))
    if not per_query:
        return {"raw": None, "w": None, "pairs": 0, "pair_accuracy": None}
    quota = max(1, n_pairs // len(per_query))
    for i, P, Q in per_query:
        a = rng.choice(P, size=quota)
        b = rng.choice(Q, size=quota)
        X = np.stack([N[f][i, a] - N[f][i, b] for f in FIELDS], axis=1)
        diffs.append(X)
    X = np.concatenate(diffs).astype(np.float64)
    w = np.zeros(3)
    # Plain gradient descent on the mean logistic loss; the problem is tiny and convex.
    lr = 2.0
    for _ in range(steps):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-z))
        grad = -(X * (1.0 - p)[:, None]).mean(axis=0) + l2 * w
        w -= lr * grad
    z = X @ w
    acc = float((z > 0).mean())
    return {"raw": w.tolist(), "w": safe_normalize(w), "pairs": int(len(X)), "pair_accuracy": acc}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", type=pathlib.Path, required=True)
    ap.add_argument("--template", type=pathlib.Path, required=True)
    ap.add_argument("--cells", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--deltas", default="0.05,0.1,0.2")
    ap.add_argument("--pairs", type=int, default=200000)
    ap.add_argument("--bandwidth", type=float, default=0.15)
    ap.add_argument("--action-dims", type=int, default=7, help="leading executed action dims (LIBERO: 7)")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()

    library_sha256 = sha256_file(args.library)
    entries, _ = load_entries(args.library)
    n = len(entries)
    template_bytes = args.template.read_bytes()
    template_sha256 = hashlib.sha256(template_bytes).hexdigest()
    tpl = yaml.safe_load(template_bytes)
    ss = tpl["checkpoints"]["cp1"]["search_strategy"]
    normalizers = check_template(ss)
    task_names = sorted({e.payload.task_key for e in entries})
    task = np.array([task_names.index(e.payload.task_key) for e in entries])
    traj_names = sorted({e.trajectory_id for e in entries})
    traj = np.array([traj_names.index(e.trajectory_id) for e in entries])
    step = np.array([int(e.step_idx) for e in entries])
    check_progress(traj, step)
    length = np.array([np.sum(traj == t) for t in traj])
    progress = step / np.maximum(length - 1, 1)
    N = {}
    for f in FIELDS:
        keys = np.stack([np.asarray(e.query_keys[f], dtype=np.float32) for e in entries])
        if not np.all(np.isfinite(keys)):
            raise SystemExit(f"library field {f!r} has non-finite keys")
        p = normalizers[f]["params"]
        N[f] = normalized_scores(keys, normalizers[f]["sim_type"], p["mu"], p["sigma"], device)
    M = (task[:, None] == task[None, :]) & (traj[:, None] != traj[None, :])
    actions = np.stack([np.asarray(e.payload.action_chunk, dtype=np.float32) for e in entries])
    live = np.arange(min(args.action_dims, actions.shape[-1])) if args.action_dims > 0 else \
        np.flatnonzero(actions.reshape(-1, actions.shape[-1]).std(axis=0) > 1e-2)
    ACT["act"] = actions[:, : min(5, actions.shape[1]), :][:, :, live].reshape(n, -1)
    cells = json.loads(args.cells.read_text())["cells"]
    leader = max(cells, key=lambda c: c["sr"])
    print(f"{n} entries; leader w={np.round(leader['w'], 3).tolist()} sr={leader['sr']:.3f}")

    # Both inputs are recorded by content, not only by path: a library rebuilt
    # at the same path, or a re-calibrated template, must not be able to claim
    # an older fit as its own.
    out = {"library": str(args.library), "library_sha256": library_sha256,
           "template": {"path": str(args.template), "sha256": template_sha256,
                        "normalizers": normalizers},
           "leader": leader, "fits": {}}
    fmt = lambda v: "/".join(f"{x:.2f}" for x in v)  # noqa: E731
    print(f"{'fit':22s} {'raw (v0/v1/rs)':26s} {'w':20s} {'nearest':20s} {'dist':>5s} {'cell':>6s} {'smooth':>7s} {'gap':>6s}")
    settings = [(float(x), None, "phase") for x in args.deltas.split(",")]
    settings += [(0.1, 0.2, "phase"), (0.1, 0.3, "phase"), (0.05, None, "action"), (0.1, None, "action"),
                 (0.05, None, "both"), (0.1, None, "both"), (0.1, 0.3, "both")]
    for delta, far, label in settings:
        fl = fisher_and_lda(N, M, progress, delta, far, label)
        pw = pairwise_logistic(N, M, progress, delta, args.pairs, far, label)
        tag = f"@{delta}" + (f"|far{far}" if far is not None else "") + ("" if label == "phase" else f"|{label}")
        for name, raw, w in [(f"fisher{tag}", fl["fisher_raw"], fl["fisher"]),
                             (f"lda{tag}", fl["lda_raw"], fl["lda"]),
                             (f"pairwise{tag}", pw["raw"], pw["w"])]:
            if raw is None or degenerate(raw, w):
                if name == PRIMARY:
                    raise SystemExit(f"{PRIMARY} is degenerate (queries={fl['queries']}, raw={raw}); "
                                     "no fallback weights are written")
                out["fits"][name] = {"raw": raw, "w": None, "degenerate": True}
                print(f"{name:22s} degenerate (queries={fl['queries']})")
                continue
            cell, dist = nearest_cell(w, cells)
            sm = smoothed_sr(w, cells, args.bandwidth)
            out["fits"][name] = {"raw": raw, "w": w, "nearest_cell": cell, "dist": dist,
                                 "smoothed_sr": sm, "gap_pp": 100 * (leader["sr"] - cell["sr"])}
            print(f"{name:22s} {fmt(raw):26s} {fmt(w):20s} {fmt(cell['w']):20s} {dist:5.2f} {cell['sr']:6.3f} {sm:7.3f} "
                  f"{100 * (leader['sr'] - cell['sr']):+6.1f}")
        out["fits"][f"detail{tag}"] = {"fisher_lda": fl, "pairwise": pw}
    out["seconds"] = time.perf_counter() - t0
    if sha256_file(args.library) != library_sha256 or sha256_file(args.template) != template_sha256:
        raise SystemExit("library or template changed during fitting; discard this computation and refit")
    (args.output / "lcw_fit_weights.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"done in {out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
