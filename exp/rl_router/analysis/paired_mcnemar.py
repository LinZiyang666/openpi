#!/usr/bin/env python3
"""P3: the pre-registered paired verdict — trained router vs the constant policy
at its own realized share.

    python paired_mcnemar.py <router_journal> <constant_journal> \
        [--label-a router] [--label-b constant@p] [--json out.json]

What P3 asks (ops log §2.5k, fixed before the run started): after training, take
the realized teacher share of the last 50 batches as p-hat, run the FROZEN
router (sample mode) and a constant policy at p-hat over the SAME 2,000 (task,
init) slots, and test whether the router's success differs. This module is the
verdict half; the rollouts are produced by the runner.

Why paired rather than two independent rates: at these success levels an
unpaired 95% interval on the difference is ~0.03 wide, which is wider than any
effect this line has ever produced. Pairing removes the (task, init) difficulty
variance -- the dominant term -- and leaves only the discordant pairs, which is
what makes ~0.02 detectable at n=2,000.

Three things this refuses to do, each because a previous version of this
comparison got them wrong:

* **Guess at pairing.** The slots must match by ``task_uid``; the test fails
  loudly rather than silently comparing whatever the two files happen to share.
  §2.5h.1's lesson is that a comparison whose pairing is assumed is a comparison
  nobody can check.
* **Count a retried episode twice.** The terminal state per uid is the
  highest-attempt row (ConductorDriver replays failed uids, §3.22), so a retry
  contributes exactly one outcome.
* **Report a one-sided or uncorrected result.** The test is two-sided at
  alpha=0.05 and it is the run's ONLY primary; a nominal p from a point chosen
  after seeing the curve is a lead, not a finding (§2.5h.1, where p=0.043 did
  not survive Holm over six such points).
"""
import argparse
import json
import math
import pathlib


def terminal_outcomes(path: str | pathlib.Path) -> dict[str, bool]:
    """Map task_uid -> success of its TERMINAL attempt.

    A journal can carry several rows for one uid: the driver replays failed
    episodes, so the same slot appears once per attempt. Taking the last row by
    file order would depend on interleaving across workers; taking the highest
    ``attempt`` is order-independent and is what "the outcome of this slot"
    means.
    """
    best: dict[str, tuple[int, bool]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uid = row["task_uid"]
            attempt = int(row.get("attempt", 1))
            if uid not in best or attempt >= best[uid][0]:
                best[uid] = (attempt, bool(row.get("success")))
    return {uid: success for uid, (_, success) in best.items()}


def exact_mcnemar(n01: int, n10: int) -> float:
    """Two-sided exact (binomial) McNemar p-value.

    Exact rather than the chi-square approximation because the discordant count
    is what carries the test, and the pre-registered detectable effect (~0.02 at
    n=2,000) sits where the approximation's tail is least trustworthy.
    """
    m = n01 + n10
    if m == 0:
        return 1.0
    k = min(n01, n10)
    tail = sum(math.comb(m, i) for i in range(k + 1))
    return min(1.0, 2 * tail / (2 ** m))


def compare(a: dict[str, bool], b: dict[str, bool]) -> dict:
    """Paired comparison over the shared slots of two outcome maps."""
    common = sorted(set(a) & set(b))
    n = len(common)
    if n == 0:
        raise SystemExit(
            "no shared task_uids: the two arms did not run the same slots, so "
            "there is nothing to pair. P3 requires both arms drawn from one "
            "slot list (same seed, same pool)."
        )
    n11 = sum(1 for u in common if a[u] and b[u])
    n01 = sum(1 for u in common if a[u] and not b[u])
    n10 = sum(1 for u in common if b[u] and not a[u])
    n00 = n - n11 - n01 - n10
    m = n01 + n10
    diff = (n01 - n10) / n
    # SE of the paired difference: only discordant pairs carry information, so
    # sqrt(m)/n -- NOT the unpaired sqrt(p(1-p)/n), which would overstate the
    # uncertainty by roughly the ratio this design exists to remove.
    se = math.sqrt(m) / n if m else 0.0
    p_value = exact_mcnemar(n01, n10)
    return {
        "n_paired": n,
        "n_only_a": len(set(a) - set(b)),
        "n_only_b": len(set(b) - set(a)),
        "sr_a": sum(a[u] for u in common) / n,
        "sr_b": sum(b[u] for u in common) / n,
        "table": {"both": n11, "only_a": n01, "only_b": n10, "neither": n00},
        "discordant": m,
        "paired_diff": diff,
        "paired_se": se,
        "p_value": p_value,
        "significant_at_05": p_value < 0.05,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("journal_a", help="arm A journal.jsonl (the trained router)")
    ap.add_argument("journal_b", help="arm B journal.jsonl (the constant policy)")
    ap.add_argument("--label-a", default="router")
    ap.add_argument("--label-b", default="constant")
    ap.add_argument("--json", help="also write the verdict as json")
    args = ap.parse_args()

    a = terminal_outcomes(args.journal_a)
    b = terminal_outcomes(args.journal_b)
    r = compare(a, b)
    la, lb = args.label_a, args.label_b

    print(f"{la}: {len(a)} slots   {lb}: {len(b)} slots   paired: {r['n_paired']}")
    if r["n_only_a"] or r["n_only_b"]:
        # Unshared slots are dropped, not silently absorbed: a large unshared
        # count means the two arms were not run over one slot list and the
        # pairing premise is broken even though the test still "works".
        print(f"  WARNING: {r['n_only_a']} slots only in {la}, {r['n_only_b']} only in "
              f"{lb} -- dropped from the test; investigate before quoting this")
    print(f"marginal SR   {la} = {r['sr_a']:.4f}   {lb} = {r['sr_b']:.4f}")
    t = r["table"]
    print(f"paired table  both={t['both']}  only {la}={t['only_a']}  "
          f"only {lb}={t['only_b']}  neither={t['neither']}")
    print(f"paired diff = {r['paired_diff']:+.4f} +/- {r['paired_se']:.4f}"
          f"   exact McNemar two-sided p = {r['p_value']:.4f}")
    print(f"=> {'SIGNIFICANT' if r['significant_at_05'] else 'not significant'} "
          "at alpha=0.05 (P3 primary, pre-registered two-sided)")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(f"verdict written to {args.json}")


if __name__ == "__main__":
    main()
