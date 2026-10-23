"""Strict freeze and accepted-attempt/cost coverage audit."""
from exp.r17_integration.debug.reader_profiles import accepted_episodes, accepted_join, merge_decision_ledgers


def audit_freeze(seal, dependencies):
    changed = sorted(k for k in (set(dependencies) | set(seal["dependencies"])) - {"code"}
                     if dependencies.get(k) != seal["dependencies"].get(k))
    if changed:
        raise ValueError("frozen dependencies changed: " + ", ".join(changed))
    return True


def audit_campaign(expected, journal, decisions, *, cost_rows=(), execution_rows=(), episode_manifests=()):
    journal = tuple(journal)
    accepted = accepted_episodes(journal)
    expected = tuple(expected)
    if len(expected) != len(set(expected)):
        raise ValueError("duplicate expected identity")
    actual = {row["task_uid"] for row in accepted.values()}
    if actual - set(expected):
        raise ValueError("unexpected accepted episode")
    joined = accepted_join(merge_decision_ledgers(decisions, cost_rows=cost_rows, execution_rows=execution_rows,
                                                episode_manifests=episode_manifests), journal)
    for row in joined:
        if (row.get("source") == "tail" or row.get("source", "").endswith("_tail")) and (
                tuple(row.get("stage_calls") or ()) != (0, 0, 0) or row.get("actual_nfe") != 0):
            raise ValueError("tail ledger contains model/vision calls")
    for row in accepted.values():
        own = [r for r in joined if r["task_uid"] == row["task_uid"] and r["attempt"] == row["attempt"]]
        seq = sorted(r["decision_seq"] for r in own)
        if seq != list(range(row["n_decisions"])):
            raise ValueError("missing/noncontiguous accepted decision ledger")
    missing_cost = [r["task_uid"] for r in joined if any(k not in r or r[k] is None for k in
                                                       ("stage_calls", "actual_nfe", "cost_ms", "cursor", "executed_count"))]
    return dict(accepted=len(accepted), missing=sorted(set(expected) - actual),
                errors=sum(r.get("status") == "error" for r in journal),
                cost_status="available" if joined and not missing_cost else "unsupported",
                missing_cost_rows=missing_cost, decisions=len(joined))
