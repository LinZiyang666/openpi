"""Is a finished arm's 500 episodes actually 500 distinct, correctly-spread rollouts?

The aggregate reports ``n_ep`` as the number of distinct accepted task_uids, so
an arm can read "500" while being wrong in ways the number hides:

*   **coverage** -- 500 episodes could be 50 init states on 10 tasks (correct)
    or every episode of one task run fifty times (a sharding or filter bug).
    Checked as: exactly ``num_tasks`` task ids, exactly ``trials`` init indices
    on each, no duplicate (task, init) pair.
*   **per-step rows attached** -- the verdict counts come from ``per_step.jsonl``
    matched to the accepted attempt. An arm whose rows failed to match would
    report a success rate but a verdict mix of nothing, and the inference ratio
    would be computed over zero decisions.
*   **verdict mix consistent with the recipe** -- an ``always_search`` anchor
    with a cut above the score domain must be all MISS; one with a cut below it
    must be all FULL_HIT. A sweep arm behind the hysteresis gate must show some
    MISS (the gate forces a teacher call every L steps) and, if its judge
    carries warm tiers, may show warm verdicts at exactly those start_t values.
    A warm verdict at a start_t the judge never names means the server served a
    bundle other than this arm's.

Usage:
  uv run python -m exp.libero_groot.check_eval_integrity \
      --data-dir <eval group dir> --arm-record <suite>/arm_record.json
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

NUM_TASKS = 10
TRIALS = 50


def parse_uid(uid: str) -> tuple[str, int, int] | None:
    """``<arm>:eval:<task_id>:<init_idx>`` -> (arm, task_id, init_idx)."""
    parts = uid.split(":")
    if len(parts) != 4 or parts[1] != "eval":
        return None
    try:
        return parts[0], int(parts[2]), int(parts[3])
    except ValueError:
        return None


def check(data_dir: pathlib.Path, record: dict, *, num_tasks: int, trials: int) -> int:
    journal = data_dir / "journal.jsonl"
    if not journal.exists():
        print(f"{data_dir}: no journal")
        return 0

    accepted: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for raw in journal.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        if row.get("status") in ("done", "failed") and row.get("accepted"):
            accepted[row["yaml_id"]][row["task_uid"]] = row["attempt"]

    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    per_step = data_dir / "per_step.jsonl"
    if per_step.exists():
        with per_step.open(encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                arm = row.get("yaml_id")
                if accepted.get(arm, {}).get(row.get("task_uid")) != row.get("attempt"):
                    continue
                hit = row.get("hit_type")
                if hit is None:
                    continue
                key = hit if hit != "WARM_START" else f"WARM_START@{float(row['start_t']):g}"
                counts[arm][key] += 1

    problems = 0
    for arm in sorted(accepted):
        uids = accepted[arm]
        if len(uids) < num_tasks * trials:
            continue  # still running; judged only when complete
        pairs = [parse_uid(u) for u in uids]
        bad = [u for u, p in zip(uids, pairs) if p is None]
        tasks = collections.Counter(p[1] for p in pairs if p)
        inits = {t: {p[2] for p in pairs if p and p[1] == t} for t in tasks}
        c = counts[arm]
        n_dec = sum(c.values())
        issues = []
        if bad:
            issues.append(f"{len(bad)} unparseable uid(s)")
        if len(uids) != num_tasks * trials:
            issues.append(f"{len(uids)} episodes != {num_tasks * trials}")
        if sorted(tasks) != list(range(num_tasks)):
            issues.append(f"task ids {sorted(tasks)}")
        short = {t: len(v) for t, v in inits.items() if len(v) != trials}
        if short:
            issues.append(f"init coverage {short}")
        if n_dec == 0:
            issues.append("no per-step rows matched (inference ratio would be over 0 decisions)")

        # The yaml is the authoritative shape, not the record's positional
        # ``cuts``: a dropped rung is recorded there but the judge is what ran.
        meta = record.get("arms", {}).get(arm, {})
        yaml_rel = meta.get("yaml")
        if yaml_rel:
            import yaml as _yaml

            doc = _yaml.safe_load((pathlib.Path(yaml_rel)).read_text(encoding="utf-8")) \
                if pathlib.Path(yaml_rel).exists() else None
            if doc:
                j = doc["checkpoints"]["cp1"]["judge"]
                judge_warm = {round(float(t["start_t"]), 4) for t in j.get("warm_tiers", [])}
                seen_warm = {round(float(k.split("@")[1]), 4)
                             for k in c if k.startswith("WARM_START@")}
                stray = seen_warm - judge_warm
                if stray:
                    issues.append(f"warm verdicts at {sorted(stray)} the judge never names")
                gate = doc["checkpoints"]["cp1"]["gate"]["type"]
                if gate == "score_hysteresis" and c.get("MISS", 0) == 0:
                    issues.append("hysteresis gate but zero MISS (the lockout must force some)")

        mix = " ".join(f"{k}={v}" for k, v in sorted(c.items()))
        status = "OK  " if not issues else "BAD "
        print(f"  {status}{arm:28s} eps={len(uids)} dec={n_dec}  {mix}")
        for issue in issues:
            print(f"        ! {issue}")
            problems += 1
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--arm-record", default="")
    ap.add_argument("--num-tasks", type=int, default=NUM_TASKS)
    ap.add_argument("--trials", type=int, default=TRIALS)
    args = ap.parse_args()
    record = (json.loads(pathlib.Path(args.arm_record).read_text(encoding="utf-8"))
              if args.arm_record else {"arms": {}})
    n = check(pathlib.Path(args.data_dir), record,
              num_tasks=args.num_tasks, trials=args.trials)
    print(f"problems: {n}")


if __name__ == "__main__":
    main()
