#!/usr/bin/env python3
"""Locate the knee of SR(p) and derive the lambda that targets it.

    python knee_and_lambda.py <sweep.json> [--batch-noise 0.027] [--n 200]

``sweep.json`` is ``sweep_mixture.py``'s output: one record per fixed teacher
share, with the realized share and the success rate.

Three questions, in order:

1. **Where does success stop improving?** Reported as the smallest measured p
   whose success is within one standard error of the best measured point. Below
   that the router is buying accuracy; above it, it is only buying cost.
2. **What does the cheapest acceptable policy cost?** The knee's mean per-step
   cost in the frozen M5a units, against the 50/50 warm start the runs used.
3. **Which lambda puts the optimum at the knee?** For a policy at share p the
   objective is ``SR(p) - lambda * n * cost(p) / T_max``; lambda is admissible
   if that is maximised at the knee rather than at either end, and *useful* if
   the objective's spread across the measured points clears the batch noise --
   a lambda whose optimum is right but whose gradient is below the noise floor
   trains no better than the one already tried.
"""
import argparse
import json
import math
import pathlib

C_T, C_S, C_C, T_MAX, N_STEPS = 1.0, 0.05565643123645517, 7.2165e-05, 520, 54.0


def cost_term(lam: float, p: float, cheap_cost: float = C_S) -> float:
    return lam * N_STEPS * (p * C_T + (1 - p) * cheap_cost) / T_MAX


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--batch-noise", type=float, default=0.027,
                    help="observed sd of batch mean reward in the M6 runs")
    # The cheap arm is a PARAMETER: this tool was written for the ts sweep and
    # hard-coded the student's cost, which is 770x the cache's (0.0556 vs
    # 7.2e-05, M5a). Fed a tc sweep, that tilts every objective by up to
    # lambda*K*0.055*(1-p) — enough to move a marginal argmax. R_tc must pass
    # --cheap-cost 7.2165e-05 (or the C_C constant).
    ap.add_argument("--cheap-cost", type=float, default=C_S,
                    help=f"per-step cost of the non-teacher arm: student {C_S}, "
                         f"cache {C_C}")
    ap.add_argument("--lambdas", default="0.05,0.2,0.5,1.0,1.5,2.0,3.0,5.0,8.0",
                    help="comma list for the objective table")
    args = ap.parse_args()

    rows = json.loads(pathlib.Path(args.sweep).read_text(encoding="utf-8"))
    rows = [r for r in rows if r.get("success_rate") is not None]
    rows.sort(key=lambda r: r["p_target"])

    print(f"{'p':>7}{'realized':>10}{'n':>6}{'SR':>9}{'SE':>8}")
    for r in rows:
        n = r["episodes"]
        sr = r["success_rate"]
        se = math.sqrt(max(sr * (1 - sr), 1e-9) / n)
        print(f"{r['p_target']:>7.2f}{r['p_realized']:>10.4f}{n:>6}{sr:>9.4f}{se:>8.4f}")

    best = max(rows, key=lambda r: r["success_rate"])
    se_best = math.sqrt(max(best["success_rate"] * (1 - best["success_rate"]), 1e-9)
                        / best["episodes"])
    knee = next(r for r in rows if r["success_rate"] >= best["success_rate"] - se_best)
    print(f"\nbest measured  p={best['p_target']:.2f}  SR={best['success_rate']:.4f}")
    print(f"KNEE (first p within 1 SE of best): p*={knee['p_target']:.2f}  "
          f"SR={knee['success_rate']:.4f}")

    cheap = float(args.cheap_cost)
    c_knee = N_STEPS * (knee["p_target"] * C_T + (1 - knee["p_target"]) * cheap) / T_MAX
    c_half = N_STEPS * (0.5 * C_T + 0.5 * cheap) / T_MAX
    print(f"\ncost at the knee vs the 50/50 warm start: {c_knee:.4f} vs {c_half:.4f} "
          f"({c_knee / c_half:.2f}x)")

    print(f"\n{'lambda':>8}{'argmax p':>10}{'spread':>10}{'spread/noise':>14}  verdict")
    for lam in (float(x) for x in args.lambdas.split(",")):
        objs = [(r["p_target"], r["success_rate"] - cost_term(lam, r["p_target"], cheap))
                for r in rows]
        pbest = max(objs, key=lambda t: t[1])[0]
        spread = max(o for _, o in objs) - min(o for _, o in objs)
        ratio = spread / args.batch_noise
        ok = (abs(pbest - knee["p_target"]) < 1e-9) and ratio >= 3.0
        print(f"{lam:>8.2f}{pbest:>10.2f}{spread:>10.4f}{ratio:>14.2f}  "
              f"{'USABLE' if ok else ''}")

    print("\nread: 'argmax p' must equal the knee (a lambda that optimises at an end "
          "point makes the router a constant), and spread/noise >= 3 or the gradient "
          "is below the per-batch noise floor and the policy will random-walk again.")


if __name__ == "__main__":
    main()
