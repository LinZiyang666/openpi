"""Paired-design analysis for the ablation study (Phase 5, plan §3).

Reads the conductor journal(s), joins arms on episode identity
``(task_id, orig_init_state_idx)``, and produces per-arm SR with Wilson CIs
plus PAIRED arm-vs-arm comparisons: exact McNemar, the paired risk difference
with a paired-bootstrap CI, and TOST equivalence on the paired risk difference
at the pre-registered margin (delta = 3pp). An attainable-precision check runs
BEFORE any verdict wording: an underpowered comparison is reported as
"insufficient evidence" (never as equivalence). Holm-Bonferroni is applied
over the pre-registered primary family.

Output: a machine-readable json + a markdown fragment for analysis.md.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib

import numpy as np
from scipy import stats

DELTA = 0.03  # pre-registered equivalence margin (3pp), plan §3
ALPHA = 0.05
BOOT = 10_000
BOOT_SEED = 20260811


def parse_task_uid(task_uid: str) -> tuple[int, int]:
    """Episode identity from the conductor's canonical uid
    ``<yaml_id>:<phase>:<task_id>:<episode_idx>`` (task.make_task_uid)."""
    parts = task_uid.rsplit(":", 3)
    if len(parts) != 4:
        raise ValueError(f"unrecognised task_uid {task_uid!r}")
    return int(parts[2]), int(parts[3])


def load_journal(paths: list[str]) -> dict[str, dict[tuple[int, int], bool]]:
    """arm -> {(task_id, episode_idx): success}.

    Consumes the conductor Journal contract (``task_uid/yaml_id/phase/status/
    success``): BOTH terminal statuses count — ``failed`` episodes are ordinary
    unsuccessful rollouts and must stay in the SR denominator. The last record
    per (arm, identity) wins (file order == append order).
    """
    arms: dict[str, dict[tuple[int, int], bool]] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("status") not in ("done", "failed"):
                    continue
                key = parse_task_uid(rec["task_uid"])
                arms.setdefault(rec["yaml_id"], {})[key] = bool(rec["success"])
    return arms


def check_paired_coverage(arms: dict[str, dict]) -> dict:
    """Coverage audit before statistics: identical identity sets per arm pair."""
    all_keys = {arm: set(eps) for arm, eps in arms.items()}
    universe = set().union(*all_keys.values()) if all_keys else set()
    report = {}
    for arm, keys in all_keys.items():
        missing = sorted(universe - keys)
        report[arm] = {"n": len(keys), "missing": len(missing)}
        if missing:
            report[arm]["missing_sample"] = missing[:5]
    return report


def wilson_ci(k: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (center - half, center + half)


def paired_compare(a: dict, b: dict, delta: float = DELTA) -> dict:
    """Paired stats for arms a vs b over their common episode identities."""
    keys = sorted(set(a) & set(b))
    xa = np.array([a[k] for k in keys], dtype=bool)
    xb = np.array([b[k] for k in keys], dtype=bool)
    n = len(keys)
    n01 = int(np.sum(~xa & xb))  # a fail, b success
    n10 = int(np.sum(xa & ~xb))  # a success, b fail
    diff = float(np.mean(xa.astype(float) - xb.astype(float)))  # paired risk difference a-b
    # Exact McNemar on the discordant pairs.
    m = n01 + n10
    mcnemar_p = float(stats.binomtest(min(n01, n10), m, 0.5).pvalue) if m > 0 else 1.0
    # Paired bootstrap CI for the risk difference (resample episode pairs).
    rng = np.random.RandomState(BOOT_SEED)
    d = xa.astype(float) - xb.astype(float)
    boots = np.array([d[rng.randint(0, n, n)].mean() for _ in range(BOOT)])
    ci = (float(np.quantile(boots, ALPHA / 2)), float(np.quantile(boots, 1 - ALPHA / 2)))
    # TOST at the pre-registered margin via the 90% bootstrap interval rule.
    tost_ci = (float(np.quantile(boots, ALPHA)), float(np.quantile(boots, 1 - ALPHA)))
    tost_pass = -delta < tost_ci[0] and tost_ci[1] < delta
    # Attainable precision: TOST cannot pass when the 90% half-width >= delta.
    half_width = (tost_ci[1] - tost_ci[0]) / 2
    powered = half_width < delta
    verdict = (
        "equivalent(TOST)" if (tost_pass and powered)
        else "insufficient evidence" if not powered
        else "not equivalent"
    )
    return {
        "n_pairs": n, "n01": n01, "n10": n10,
        "risk_diff": diff, "risk_diff_ci95": ci,
        "mcnemar_p": mcnemar_p, "tost_ci90": tost_ci,
        "tost_pass": bool(tost_pass), "powered": bool(powered),
        "verdict": verdict,
    }


def holm_adjust(pvals: list[tuple[str, float]]) -> dict[str, dict]:
    """Correct Holm-Bonferroni step-down: adjusted p = cumulative max of
    (m-i)*p_(i) over the ascending ordering; decisions stop at the first
    retained hypothesis (monotone — no later rejection after a retention)."""
    ordered = sorted(pvals, key=lambda t: t[1])
    m = len(ordered)
    out: dict[str, dict] = {}
    running_max = 0.0
    rejecting = True
    for i, (spec, p) in enumerate(ordered):
        adj = min(1.0, max(running_max, (m - i) * p))
        running_max = adj
        if adj >= ALPHA:
            rejecting = False
        out[spec] = {"p": p, "holm_adjusted_p": adj, "significant": rejecting and adj < ALPHA}
    return out


POWER_TARGET = 0.8  # preregistered TOST power target at true risk difference 0


def preflight(n: int, discordance: float, power_target: float = POWER_TARGET,
              delta: float = DELTA) -> dict:
    """Pre-launch TOST POWER gate (plan §3). Under the normal approximation
    with true paired risk difference 0, se = sqrt(q/n) and the probability of
    passing both one-sided tests at level alpha is
    ``power = max(0, 2*Phi(delta/se - z_{1-alpha}) - 1)``. Launch is blocked
    unless power >= power_target. Also reports the maximum discordance rate
    still meeting the target for this n."""
    z = float(stats.norm.ppf(1 - ALPHA))
    se = float(np.sqrt(discordance / n))
    power = max(0.0, 2.0 * float(stats.norm.cdf(delta / se - z)) - 1.0)
    z_beta = float(stats.norm.ppf((1 + power_target) / 2))
    max_q = n * (delta / (z + z_beta)) ** 2
    return {
        "n_pairs": n,
        "expected_discordance": discordance,
        "se": se,
        "delta": delta,
        "tost_power_at_zero_diff": float(power),
        "power_target": power_target,
        "decidable": bool(power >= power_target),
        "max_decidable_discordance": float(max_q),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journals", nargs="+", default=[])
    parser.add_argument("--primary", nargs="*", default=[],
                        help="primary-family comparisons as armA:armB (Holm corrected)")
    parser.add_argument("--preflight-n", type=int, default=None,
                        help="run the pre-launch power gate for n pairs and exit")
    parser.add_argument("--preflight-discordance", type=float, default=0.15)
    parser.add_argument("--allow-partial", action="store_true",
                        help="exploratory only: proceed despite coverage gaps")
    parser.add_argument("--preflight-artifact", default=None,
                        help="the immutable <per-step-out>.preflight.json written at "
                             "launch; its recorded delta/decision bind the analysis "
                             "to the launched design (mutable approval files are "
                             "NOT re-read here)")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    delta = DELTA
    launch_design = None
    if args.preflight_artifact:
        launch_design = json.loads(pathlib.Path(args.preflight_artifact).read_text())
        delta = float(launch_design["delta"])
    if args.preflight_n is not None:
        result = preflight(args.preflight_n, args.preflight_discordance, delta=delta)
        print(json.dumps(result, indent=2))
        if not result["decidable"]:
            raise SystemExit(
                f"UNDERPOWERED: TOST at delta={delta} not decidable with "
                f"n={args.preflight_n}, q={args.preflight_discordance} — do not launch."
            )
        return

    if not (args.journals and args.out_json and args.out_md):
        raise SystemExit("--journals/--out-json/--out-md required (or --preflight-n)")

    arms = load_journal(args.journals)
    coverage = check_paired_coverage(arms)
    incomplete = {a: c for a, c in coverage.items() if c["missing"]}
    if incomplete and not args.allow_partial:
        raise SystemExit(
            f"paired coverage incomplete: {incomplete} — refusing statistics "
            "(pass --allow-partial for exploratory reads only)."
        )
    report: dict = {"design": {"delta": delta,
                               "launch_preflight": launch_design},
                    "coverage": coverage,
                    "arms": {}, "pairs": {}, "primary_family": {}}
    for arm, eps in sorted(arms.items()):
        k, n = sum(eps.values()), len(eps)
        report["arms"][arm] = {"sr": k / n if n else None, "n": n, "wilson_ci95": wilson_ci(k, n)}
    for a, b in itertools.combinations(sorted(arms), 2):
        report["pairs"][f"{a}|{b}"] = paired_compare(arms[a], arms[b], delta=delta)
    if args.primary:
        pvals = []
        for spec in args.primary:
            a, b = spec.split(":")
            key = f"{a}|{b}" if f"{a}|{b}" in report["pairs"] else f"{b}|{a}"
            pvals.append((spec, report["pairs"][key]["mcnemar_p"]))
        report["primary_family"] = holm_adjust(pvals)

    pathlib.Path(args.out_json).write_text(json.dumps(report, indent=2))
    approval_note = ""
    if launch_design and launch_design.get("approval"):
        a = launch_design["approval"]
        approval_note = (f"\n> Owner-approved design: decision={a.get('decision')}, "
                         f"delta={delta}, approval sha256={a.get('sha256')}\n")
    lines = [f"Design margin delta = {delta}" + approval_note,
             "| arm | SR | n | Wilson 95% CI |", "|---|---|---|---|"]
    for arm, r in report["arms"].items():
        lo, hi = r["wilson_ci95"]
        lines.append(f"| {arm} | {r['sr']:.3f} | {r['n']} | [{lo:.3f}, {hi:.3f}] |")
    lines += ["", "| pair | risk diff | 95% CI | McNemar p | TOST verdict | powered |",
              "|---|---|---|---|---|---|"]
    for pair, r in report["pairs"].items():
        lo, hi = r["risk_diff_ci95"]
        lines.append(
            f"| {pair} | {r['risk_diff']:+.3f} | [{lo:+.3f}, {hi:+.3f}] "
            f"| {r['mcnemar_p']:.4f} | {r['verdict']} | {r['powered']} |"
        )
    if report["primary_family"]:
        lines += ["", "| primary comparison | p | Holm-adjusted p | significant |",
                  "|---|---|---|---|"]
        for spec, r in report["primary_family"].items():
            lines.append(f"| {spec} | {r['p']:.4f} | {r['holm_adjusted_p']:.4f} | {r['significant']} |")
    pathlib.Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(report["primary_family"] or report["coverage"], indent=2))


if __name__ == "__main__":
    main()
