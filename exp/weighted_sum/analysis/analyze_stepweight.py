"""Analyze the trajectory step-weight screening run.

Screening deliverable (NOT a decisive verdict): per depth, rank every swept
``trajectory_weights`` shape by success rate and report a paired comparison vs
that depth's incumbent, so candidates carry uncertainty rather than a single
over-claimed winner.

Design points (all mandated by the plan):

- True weights are read back from the eval YAMLs, never parsed from the yaml_id.
- Shapes are classified by a mutually-exclusive precedence
  ``uniform -> decreasing -> increasing -> strict interior peak -> strict
  interior trough -> other``; a plateau (non-unique extremum) is ``other``, not
  peak/trough. ``current_dominant`` (``w[0] >= 0.6``) is an ORTHOGONAL boolean
  tag, not a shape class.
- The journal is first collapsed to the latest-``ts`` terminal per ``task_uid``
  (consistent with ``summarize.py``) BEFORE building ``(task_id, episode_idx)``
  paired keys, so a legitimate timeout/late-result duplicate row cannot create a
  wrong pairing. Same ``(task_id, episode_idx)`` is the same initial state across
  configs (deterministic held-out inits), giving valid paired binary outcomes.
- Completeness is cross-checked against the emitter's independent
  ``expected_ids()`` (the locked config set) — NOT against the input files' own
  agreement — and every config must carry exactly ``task_ids x range(trials)``
  keys, so a truncated batch is rejected.
- Every config gets a full paired stat: delta-SR vs incumbent + McNemar exact p +
  fixed-seed paired-bootstrap delta-SR CI. ALL configs are written to the
  machine-readable ``decision.json`` (in ``data/``); ``results.md`` shows a top-k
  summary. The d1 prior SR is wired in as a non-decisive annotation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from math import comb

import yaml

from exp.weighted_sum.emit_traj_weight_alloc import STEM, expected_ids

# d1 (depth-1, no trajectory) ceiling — prior-run non-decisive reference.
D1_NOTE = "non-decisive prior-run reference (same ziyang10 H200 series, not same-batch)"

SHAPE_CLASSES = ("uniform", "decreasing", "increasing", "peak", "trough", "other")


# ----------------------------------------------------------------------
# Shape classification (pure; unit-tested)
# ----------------------------------------------------------------------
def classify_shape(weights, tol: float = 0.02) -> str:
    """Mutually-exclusive shape label by first-match precedence.

    Precedence: uniform -> monotone decreasing -> monotone increasing -> STRICT
    interior peak -> STRICT interior trough -> other. A monotone class requires a
    net end-to-end change greater than ``tol``. A peak/trough requires a UNIQUE
    interior extremum (no plateau within ``tol``) that is strictly beyond both
    immediate neighbors; plateaus and multi-extrema fall through to ``other``.
    """
    w = list(weights)
    d = len(w)
    if d < 2:
        return "other"
    if max(w) - min(w) < tol:
        return "uniform"
    non_inc = all(w[i] - w[i + 1] >= -tol for i in range(d - 1))
    if non_inc and (w[0] - w[-1] > tol):
        return "decreasing"
    non_dec = all(w[i + 1] - w[i] >= -tol for i in range(d - 1))
    if non_dec and (w[-1] - w[0] > tol):
        return "increasing"
    mx = max(w)
    near_max = [i for i in range(d) if mx - w[i] <= tol]
    if len(near_max) == 1 and 0 < near_max[0] < d - 1:
        k = near_max[0]
        up = all(w[i + 1] - w[i] >= -tol for i in range(k))
        down = all(w[i] - w[i + 1] >= -tol for i in range(k, d - 1))
        if up and down and (w[k] - w[k - 1] > tol) and (w[k] - w[k + 1] > tol):
            return "peak"
    mn = min(w)
    near_min = [i for i in range(d) if w[i] - mn <= tol]
    if len(near_min) == 1 and 0 < near_min[0] < d - 1:
        k = near_min[0]
        down = all(w[i] - w[i + 1] >= -tol for i in range(k))
        up = all(w[i + 1] - w[i] >= -tol for i in range(k, d - 1))
        if down and up and (w[k - 1] - w[k] > tol) and (w[k + 1] - w[k] > tol):
            return "trough"
    return "other"


def is_current_dominant(weights, thr: float = 0.6) -> bool:
    """Orthogonal tag: is the current-step (newest) weight at least ``thr``."""
    return len(weights) > 0 and weights[0] >= thr


# ----------------------------------------------------------------------
# Journal parsing + pairing + stats (pure; unit-tested)
# ----------------------------------------------------------------------
def dedup_latest(records) -> dict[str, dict]:
    """Collapse terminal eval records to the latest ``ts`` per ``task_uid``.

    Mirrors ``summarize.summarize_journal``: only ``phase=="eval"`` +
    ``status in {done, failed}`` rows count; a retried/late duplicate keeps the
    latest-ts outcome so an episode is paired once at its final result.
    """
    final: dict[str, dict] = {}
    for rec in records:
        if rec.get("phase") != "eval" or rec.get("status") not in ("done", "failed"):
            continue
        uid = rec["task_uid"]
        ts = float(rec.get("ts", 0.0))
        prev = final.get(uid)
        if prev is None or ts >= float(prev.get("ts", 0.0)):
            final[uid] = rec
    return final


def _parse_uid(task_uid: str) -> tuple[str, int, int]:
    """``yaml_id:phase:task_id:episode_idx`` -> (yaml_id, task_id, episode_idx)."""
    parts = task_uid.rsplit(":", 3)
    if len(parts) != 4:
        raise ValueError(f"unexpected task_uid {task_uid!r}")
    return parts[0], int(parts[2]), int(parts[3])


def paired_by_yaml(records) -> dict[str, dict[tuple[int, int], bool]]:
    """yaml_id -> {(task_id, episode_idx): success} after latest-ts dedup."""
    out: dict[str, dict[tuple[int, int], bool]] = defaultdict(dict)
    for rec in dedup_latest(records).values():
        yaml_id, task_id, ep = _parse_uid(rec["task_uid"])
        out[yaml_id][(task_id, ep)] = bool(rec.get("success"))
    return out


def mcnemar(a_map: dict, b_map: dict) -> tuple[int, int, int, float]:
    """(n10, n01, n_pairs, p) on shared keys; exact two-sided binomial McNemar p.

    n10 = a-success & b-fail, n01 = b-success & a-fail.
    """
    keys = set(a_map) & set(b_map)
    n10 = sum(1 for k in keys if a_map[k] and not b_map[k])
    n01 = sum(1 for k in keys if b_map[k] and not a_map[k])
    n = n10 + n01
    if n == 0:
        return n10, n01, len(keys), 1.0
    k = min(n10, n01)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return n10, n01, len(keys), min(1.0, 2.0 * tail)


def paired_bootstrap_ci(
    base_map: dict, cfg_map: dict, *, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Fixed-seed paired-bootstrap CI on delta-SR = SR(cfg) - SR(base).

    Resamples the shared paired keys with replacement (paired: each key's
    (base, cfg) outcomes move together). Deterministic for a given ``seed`` so the
    CI is reproducible. Returns (point_delta, lo, hi).
    """
    keys = sorted(set(base_map) & set(cfg_map))
    n = len(keys)
    if n == 0:
        return 0.0, 0.0, 0.0
    diffs = [(1 if cfg_map[k] else 0) - (1 if base_map[k] else 0) for k in keys]
    point = sum(diffs) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = 0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return point, lo, hi


def acceptance_check(paired, weights_ids, expected, task_ids, trials: int) -> list[str]:
    """Fail-fast completeness gate; returns a list of problems (empty = pass).

    Cross-checks BOTH the eval-YAML set and the journal set against the emitter's
    independent ``expected`` id set (not against each other), and requires every
    config's paired keys to equal exactly ``task_ids x range(trials)`` — so a
    truncated batch (fewer configs, or an episode_idx outside 0..trials-1) fails.
    """
    errs: list[str] = []
    exp = set(expected)
    if set(weights_ids) != exp:
        errs.append(f"yaml-dir set != expected: missing={sorted(exp - set(weights_ids))}, "
                    f"extra={sorted(set(weights_ids) - exp)}")
    if set(paired) != exp:
        errs.append(f"journal config set != expected: missing={sorted(exp - set(paired))}, "
                    f"extra={sorted(set(paired) - exp)}")
    want = {(t, e) for t in task_ids for e in range(trials)}
    for yid, pmap in paired.items():
        if set(pmap.keys()) != want:
            per_task = Counter(t for (t, _e) in pmap)
            errs.append(f"{yid}: paired-key set != exact {len(task_ids)}x{trials} "
                        f"(have {len(pmap)} keys; per-task {dict(per_task)})")
    return errs


# ----------------------------------------------------------------------
# Inputs from disk
# ----------------------------------------------------------------------
def read_trajectory_weights(yaml_dir: Path) -> dict[str, list[float]]:
    """yaml_id -> trajectory_weights, read from the eval YAMLs (not the id)."""
    out: dict[str, list[float]] = {}
    for p in sorted(yaml_dir.glob("*.yaml")):
        cfg = yaml.safe_load(p.read_text())
        ss = cfg["checkpoints"]["cp1"]["search_strategy"]
        out[p.stem] = list(ss["trajectory_weights"])
    return out


def read_d1_prior(baseline_csv: Path, keybuilder: str = STEM) -> float | None:
    """d1 (depth-1) ceiling SR from the phase-2 results CSV (or None).

    The best regular-grid zscore config for ``keybuilder`` (dropping ``__norm2``
    normalizer-swap and ``__iso`` single-modality variants), averaged over repeat
    rows per yaml_id. This is the non-decisive prior-run d1 ceiling the trajectory
    depths are compared against — read robustly (max mean SR) rather than via a
    fragile hard-coded id, since phase-2 used a 2-decimal weight grid.
    """
    if not baseline_csv.exists():
        return None
    agg: dict[str, list[float]] = defaultdict(list)
    for r in csv.DictReader(baseline_csv.open()):
        yid = r.get("yaml_id", "")
        if r.get("keybuilder") != keybuilder or r.get("normalizer") != "zscore":
            continue
        if "__norm2" in yid or "__iso" in yid:
            continue
        agg[yid].append(float(r["success_rate"]))
    return max((sum(v) / len(v) for v in agg.values()), default=None)


# ----------------------------------------------------------------------
# Reporting (main)
# ----------------------------------------------------------------------
def _incumbent_id(yaml_ids, depth: int) -> str | None:
    for yid in yaml_ids:
        if yid.endswith("__incumbent") and f"__d{depth}__" in yid:
            return yid
    return None


def _sr(pmap: dict) -> tuple[float, int, int]:
    n = len(pmap)
    ns = sum(1 for v in pmap.values() if v)
    return (ns / n if n else 0.0), n, ns


def _plot(rows_by_depth, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cmap = plt.get_cmap("tab10")
    color = {c: cmap(i) for i, c in enumerate(SHAPE_CLASSES)}
    depths = sorted(rows_by_depth)
    fig, axes = plt.subplots(1, len(depths), figsize=(6 * len(depths), 5), squeeze=False)
    for ax, depth in zip(axes[0], depths):
        rows = sorted(rows_by_depth[depth], key=lambda r: r["sr"])
        ax.scatter(range(len(rows)), [r["sr"] for r in rows],
                   c=[color[r["shape"]] for r in rows], s=26)
        for r in rows:
            if r["is_incumbent"]:
                ax.axhline(r["sr"], ls="--", c="k", lw=1)
                ax.text(0, r["sr"], f" incumbent {r['sr']:.2f}", va="bottom", fontsize=8)
        ax.set_title(f"d{depth} — {len(rows)} configs")
        ax.set_xlabel("config (SR-sorted)")
        ax.set_ylabel("success_rate")
    handles = [Line2D([0], [0], marker="o", ls="", color=color[c], label=c) for c in SHAPE_CLASSES]
    handles.append(Line2D([0], [0], ls="--", color="k", label="incumbent"))
    fig.legend(handles=handles, loc="upper center", ncol=len(SHAPE_CLASSES) + 1, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "stepweight_sr_by_depth.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Analyze trajectory step-weight screening run")
    ap.add_argument("--journal", required=True)
    ap.add_argument("--yaml-dir", required=True)
    ap.add_argument("--out-dir", required=True, help="analysis dir (results.md + png)")
    ap.add_argument("--decision-out", required=True,
                    help="machine-readable decision.json path — MUST be under data/, not analysis/")
    ap.add_argument("--baseline-csv", default="exp/weighted_sum/data/libero_spatial/phase2/all_results.csv")
    ap.add_argument("--depths", default="3,4,5", help="depths whose locked expected_ids to cross-check")
    ap.add_argument("--task-ids", default="0-9")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dec_path = Path(args.decision_out)
    if "analysis" in dec_path.parts:
        raise SystemExit(f"--decision-out must land in data/, not analysis/: {dec_path}")

    if "-" in args.task_ids:
        lo, hi = args.task_ids.split("-")
        task_ids = list(range(int(lo), int(hi) + 1))
    else:
        task_ids = [int(x) for x in args.task_ids.split(",")]
    depths = tuple(int(d) for d in args.depths.split(",") if d != "")

    records = [json.loads(x) for x in Path(args.journal).read_text(encoding="utf-8").splitlines() if x.strip()]
    paired = paired_by_yaml(records)
    weights = read_trajectory_weights(Path(args.yaml_dir))

    errs = acceptance_check(paired, set(weights), expected_ids(depths), task_ids, args.trials)
    if errs:
        raise SystemExit("acceptance-gate FAILED:\n  " + "\n  ".join(errs))

    d1_sr = read_d1_prior(Path(args.baseline_csv))

    rows_by_depth: dict[int, list[dict]] = defaultdict(list)
    for yid, w in weights.items():
        sr, n, ns = _sr(paired[yid])
        rows_by_depth[len(w)].append(
            {"yaml_id": yid, "weights": w, "shape": classify_shape(w, args.tol),
             "current_dominant": is_current_dominant(w), "sr": sr, "n": n, "n_success": ns,
             "is_incumbent": yid.endswith("__incumbent")}
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decision: dict = {
        "note": "SCREENING — candidates with paired uncertainty, not a decisive verdict",
        "d1_prior_sr": d1_sr, "d1_prior_note": D1_NOTE,
        "bootstrap": {"n_boot": args.n_boot, "seed": args.seed}, "by_depth": {},
    }
    md = ["# Trajectory step-weight screening — results\n",
          "> Screening: ranked candidates + paired McNemar + bootstrap CI vs incumbent. NOT decisive.",
          f"> d1 prior SR = {d1_sr} ({D1_NOTE}).",
          "> Full per-config paired stats are in the machine-readable decision.json (data/); "
          "the tables below are the top-k summary.\n"]

    for depth in sorted(rows_by_depth):
        rows = rows_by_depth[depth]
        inc_id = _incumbent_id(weights, depth)
        inc_map = paired.get(inc_id, {})
        base_sr = _sr(inc_map)[0] if inc_id else None

        full = []
        for r in sorted(rows, key=lambda r: -r["sr"]):
            n10, n01, npair, p = mcnemar(inc_map, paired[r["yaml_id"]])
            dpt, lo, hi = paired_bootstrap_ci(inc_map, paired[r["yaml_id"]], n_boot=args.n_boot, seed=args.seed)
            full.append({"yaml_id": r["yaml_id"], "weights": r["weights"], "shape": r["shape"],
                         "current_dominant": r["current_dominant"], "sr": r["sr"], "n": r["n"],
                         "vs_incumbent": {"delta_sr": (r["sr"] - base_sr) if base_sr is not None else None,
                                          "mcnemar": {"n10": n10, "n01": n01, "n_pairs": npair, "p": p},
                                          "bootstrap_delta_ci": {"point": dpt, "lo": lo, "hi": hi}}})
        decision["by_depth"][depth] = {"incumbent_id": inc_id, "incumbent_sr": base_sr,
                                       "n_configs": len(rows), "configs": full}  # ALL configs
        md.append(f"\n## d{depth} — {len(rows)} configs (incumbent SR {base_sr})\n")
        md.append("| rank | id tail | shape | curr_dom | SR | ΔSR vs inc | boot CI | McNemar p |")
        md.append("|---|---|---|---|---|---|---|---|")
        for i, c in enumerate(full[: args.top_k], 1):
            v = c["vs_incumbent"]
            dsr = f"{v['delta_sr']:+.3f}" if v["delta_sr"] is not None else "—"
            ci = f"[{v['bootstrap_delta_ci']['lo']:+.3f}, {v['bootstrap_delta_ci']['hi']:+.3f}]"
            md.append(f"| {i} | `{c['yaml_id'].split('__d')[-1]}` | {c['shape']} | "
                      f"{'Y' if c['current_dominant'] else ''} | {c['sr']:.2f} | {dsr} | {ci} | "
                      f"{v['mcnemar']['p']:.3f} |")

    (out_dir / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    dec_path.parent.mkdir(parents=True, exist_ok=True)
    dec_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    _plot(rows_by_depth, out_dir)
    print(f"wrote {out_dir/'results.md'}, {dec_path}, {out_dir/'stepweight_sr_by_depth.png'}")


if __name__ == "__main__":
    main()
