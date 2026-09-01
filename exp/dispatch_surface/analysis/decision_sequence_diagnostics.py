"""Decision-sequence diagnostics on the shared A' episodes (exploratory).

Backs the empirical facts used in the amendment result (logs/dispatch_surface_rev2_amendment_result.md §6.2):

  1. determinism  -- arm pairs whose verdict sequence on an episode is identical
                     must agree on the outcome (any disagreement = stochastic rollout);
  2. episode difficulty distribution across arms, and the "rescue" rate of the
     cache arms on episodes where always-full inference failed;
  3. per-band marginal effect of extending WARM_START (fixed fh, ws k-1 -> k) or
     FULL_HIT (fixed ws, fh k -> k+10) to the next lower score band in the dense
     grid -- a regression-discontinuity reading of the damage of one more band,
     paired over the 300 episodes;
  4. whether the early score profile predicts episode difficulty (per-task
     difficulty spread is reported alongside).

Usage:
  python -m exp.dispatch_surface.analysis.decision_sequence_diagnostics \
      --tgrid-matrix <package>/arm_matrix_exploratory_tgrid.json --tgrid-journal ... --tgrid-per-step ... \
      --extra-journal <phase0 journal> --extra-journal <rev1 journal> --out <json>
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

FH = (20, 30, 40, 50, 60, 70, 80)
WS = (0, 10, 20, 30, 40)


def _successes(journal: str) -> dict[str, dict[tuple[int, int], int]]:
    out: dict[str, dict] = collections.defaultdict(dict)
    for line in open(journal):
        r = json.loads(line)
        if r.get("accepted") is True:
            u = r["task_uid"].split(":")
            out[r["yaml_id"]][(int(u[2]), int(u[3]))] = int(bool(r["success"]))
    return out


def _paired_diff(succ, a: str, b: str, eps):
    d = np.array([succ[b][e] - succ[a][e] for e in eps], dtype=float)
    return {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))),
            "n_up": int((d > 0).sum()), "n_down": int((d < 0).sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tgrid-matrix", required=True)
    ap.add_argument("--tgrid-journal", required=True)
    ap.add_argument("--tgrid-per-step", required=True)
    ap.add_argument("--extra-journal", action="append", default=[], help="other layers' journals on the same A' (phase0, rev1)")
    ap.add_argument("--early-steps", type=int, default=20)
    ap.add_argument("--early-arm", default="dsp_tg_fh20_ws0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    succ = _successes(args.tgrid_journal)
    for j in args.extra_journal:
        for arm, cells in _successes(j).items():
            if arm in succ:
                raise SystemExit(f"arm {arm} appears in two journals")
            succ[arm] = cells
    arms = sorted(succ)
    eps = sorted(next(iter(succ.values())))
    if any(sorted(succ[a]) != eps for a in arms):
        raise SystemExit("journals do not cover the same episode grid")

    # 1. determinism: identical verdict sequences on an episode -> identical outcome
    seq: dict[tuple, list] = collections.defaultdict(list)
    early: dict[tuple, list] = collections.defaultdict(list)
    for line in open(args.tgrid_per_step):
        r = json.loads(line)
        if r.get("accepted") is not True or "hit_type" not in r:
            continue
        key = (r["yaml_id"], r["task_id"], r["subset_init_state_idx"])
        seq[key].append((r["step_idx"], r["hit_type"]))
        if r["yaml_id"] == args.early_arm and r.get("cp1_score") is not None and r["step_idx"] < args.early_steps:
            early[(r["task_id"], r["subset_init_state_idx"])].append(float(r["cp1_score"]))
    by_ep: dict[tuple, list] = collections.defaultdict(list)
    for (arm, t, i), rows in seq.items():
        by_ep[(t, i)].append((arm, tuple(h for _, h in sorted(rows))))
    pairs = same = diff = 0
    for e, lst in by_ep.items():
        for x in range(len(lst)):
            for y in range(x + 1, len(lst)):
                if lst[x][1] == lst[y][1]:
                    pairs += 1
                    if succ[lst[x][0]][e] == succ[lst[y][0]][e]:
                        same += 1
                    else:
                        diff += 1
    determinism = {"identical_sequence_pairs": pairs, "same_outcome": same, "different_outcome": diff}

    # 2. difficulty distribution and rescue rate
    M = np.array([[succ[a][e] for a in arms] for e in eps], dtype=float)
    frac = M.mean(axis=1)
    edges = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001]
    hist, _ = np.histogram(frac, bins=edges)
    difficulty = {"n_episodes": len(eps), "n_arms": len(arms),
                  "histogram": {f"{edges[i]:.1f}-{min(edges[i + 1], 1.0):.1f}": int(h) for i, h in enumerate(hist)},
                  "always_succeeds": int((frac >= 0.999).sum()), "never_succeeds": int((frac <= 0.001).sum())}
    rescue = None
    if "always_full_inference" in arms:
        ai = arms.index("always_full_inference")
        others = [k for k in range(len(arms)) if k != ai]
        af = M[:, ai] == 0
        rescue = {"anchor_failures": int(af.sum()), "other_arms_success_rate_on_anchor_failures": float(M[af][:, others].mean()),
                  "anchor_successes": int((~af).sum()), "other_arms_success_rate_on_anchor_successes": float(M[~af][:, others].mean())}

    # 3. per-band marginal effects on the dense grid
    warm_bands, full_bands = {}, {}
    for fh in FH:
        for k in range(1, len(WS)):
            a, b = f"dsp_tg_fh{fh}_ws{WS[k - 1]}", f"dsp_tg_fh{fh}_ws{WS[k]}"
            if a in succ and b in succ:
                warm_bands[f"fh{fh}: ws{WS[k - 1]}->ws{WS[k]}"] = _paired_diff(succ, a, b, eps)
    for ws in WS:
        for k in range(1, len(FH)):
            a, b = f"dsp_tg_fh{FH[k - 1]}_ws{ws}", f"dsp_tg_fh{FH[k]}_ws{ws}"
            if a in succ and b in succ:
                full_bands[f"ws{ws}: fh{FH[k - 1]}->fh{FH[k]}"] = _paired_diff(succ, a, b, eps)

    # 4. early score profile vs difficulty; per-task spread
    x = np.array([np.mean(early[e]) if early.get(e) else np.nan for e in eps])
    ok = ~np.isnan(x)
    corr = float(np.corrcoef(x[ok], frac[ok])[0, 1]) if ok.sum() > 2 else None
    by_task = collections.defaultdict(list)
    for e, f in zip(eps, frac):
        by_task[e[0]].append(f)
    per_task = {str(t): {"mean": float(np.mean(v)), "sd": float(np.std(v))} for t, v in sorted(by_task.items())}

    out = {"protocol": "dispatch_surface_rev2_decision_sequence_diagnostics", "posthoc_exploratory": True,
           "arms": arms, "determinism": determinism, "difficulty": difficulty, "rescue": rescue,
           "warm_band_marginal_effect": warm_bands, "full_band_marginal_effect": full_bands,
           "early_profile": {"arm": args.early_arm, "steps": args.early_steps, "corr_mean_score_vs_difficulty": corr},
           "per_task_difficulty": per_task}
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))
    print("determinism:", determinism)
    print("difficulty:", difficulty["histogram"], "always/never:", difficulty["always_succeeds"], difficulty["never_succeeds"])
    print("rescue:", rescue)
    print("early-profile corr:", corr)
    for k, v in warm_bands.items():
        print(f"  WARM {k:22s} dSR {v['mean']:+.3f} ± {v['se']:.3f} (+{v['n_up']}/-{v['n_down']})")
    for k, v in full_bands.items():
        print(f"  FULL {k:22s} dSR {v['mean']:+.3f} ± {v['se']:.3f} (+{v['n_up']}/-{v['n_down']})")


if __name__ == "__main__":
    main()
