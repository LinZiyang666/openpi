"""Review A round 3 (A2 N5): price on measured service time, report owner IR separately."""
import pytest

from exp.r17_integration.debug.reader_profiles import cost_fields
from exp.r17_integration.debug.validate import EPISODE_KEYS, digest
from exp.r17_integration.search.analyze import summarize
from exp.r17_integration.tests.debug.fixtures import binding, record


def ledger(*, service=True):
    """Two decisions held behind a busy shared model lock: wall 10ms = 6 queue + 4 service."""
    episode = dict(record(n_decisions=2), accepted=True, status="done")
    rows = []
    for seq, calls in enumerate(([1, 1, 4], [1, 0, 0])):
        row = record(seq=seq)
        for key in ("cost_ms", "queue_wait_ms", "service_ms", "pricing_time_field"):
            row.pop(key)
        row.update(stage_calls=calls, actual_nfe=calls[2], sampler_identity="miss-k4", executed_count=5,
                   cost_ms=10.)
        if service:
            row.update(queue_wait_ms=6., service_ms=4., pricing_time_field="service_ms")
        rows.append(row)
    return episode, rows


def baseline(**extra):
    value = {"binding_digest": digest(binding()), "absolute_ms": 40., "pricing_absolute_ms": 16.,
             "pricing_time_field": "service_ms", "capture_enabled": False, "pricing_eligible": True}
    value.update(extra)
    return value


def test_round3_pricing_sums_service_time_not_lock_queue_wall():
    episode, rows = ledger()
    report = summarize([episode["task_uid"]], [episode], rows, task_groups={"reach-v3": "easy"},
                       binding=binding(), full_baseline_ms=baseline())
    cost = report["cost"]
    assert cost["absolute_ms"] == 20.              # host wall, diagnostic
    assert cost["queue_wait_total_ms"] == 12.
    assert cost["pricing_absolute_ms"] == 8.       # service time only
    assert cost["pricing_time_field"] == "service_ms"
    assert cost["normalized_within_model"] == .5   # 8 / 16, never 20 / 40 wall ratio
    # A wall-only baseline cannot normalize service pricing.
    unpriced = summarize([episode["task_uid"]], [episode], rows, task_groups={"reach-v3": "easy"},
                         binding=binding(), full_baseline_ms=baseline(pricing_time_field=None))
    assert unpriced["cost"]["normalized_within_model"] is None


def test_round3_rows_without_queue_service_split_are_never_priced():
    episode, rows = ledger(service=False)
    report = summarize([episode["task_uid"]], [episode], rows, task_groups={"reach-v3": "easy"},
                       binding=binding(), full_baseline_ms=baseline())
    assert report["cost"]["status"] == "available" and report["cost"]["absolute_ms"] == 20.
    assert report["cost"]["pricing_status"] == "unsupported"
    assert report["cost"]["pricing_absolute_ms"] is None and report["cost"]["normalized_within_model"] is None


def test_round3_owner_ir_uses_stage_ledger_and_bound_sampler():
    episode, rows = ledger()
    costs = {"schema": "r17.ir_costs.v1", "vision": .148, "miss": .852, "sampler_identity": "miss-k4",
             "source_identity": "cpu-fixture", "benchmark_protocol": "p", "retrieval_model_identity": "m"}
    report = summarize([episode["task_uid"]], [episode], rows, task_groups={"reach-v3": "easy"},
                       binding=binding(), ir_costs=costs)
    # N=2, V=2, M=1 -> (.148*2 + .852*1) / 2
    assert report["cost"]["owner_ir"] == pytest.approx((.148 * 2 + .852) / 2)
    with pytest.raises(ValueError, match="different MISS sampler"):
        summarize([episode["task_uid"]], [episode], rows, task_groups={"reach-v3": "easy"},
                  binding=binding(), ir_costs=dict(costs, sampler_identity="miss-k2"))
    with pytest.raises(ValueError, match="r17.ir_costs.v1"):
        summarize([episode["task_uid"]], [episode], rows, task_groups={"reach-v3": "easy"},
                  binding=binding(), ir_costs={"vision": .1, "miss": .2})


def test_round3_writer_backpressure_and_episode_copies_do_not_replace_decision_timing():
    # Closed osdebug rows: top-level queue_wait_ms is the BlockWriter wait, and
    # episode metadata may carry an unrelated cost copy.
    row = dict(record(), cost_ms=2., queue_wait_ms=3.5, service_ms=2.,
               diagnostics={"cost_ms": 7., "queue_wait_ms": 1., "service_ms": 6., "pricing_time_field": "service_ms",
                            "capture_enabled": True, "pricing_eligible": False})
    fields = cost_fields(row)
    assert (fields["cost_ms"], fields["queue_wait_ms"], fields["service_ms"]) == (7., 1., 6.)
    assert fields["writer_queue_wait_ms"] == 3.5
    with pytest.raises(ValueError, match="service_ms \\+ queue_wait_ms"):
        cost_fields({"cost_ms": 10., "queue_wait_ms": 1., "service_ms": 4., "pricing_time_field": "service_ms"})
    with pytest.raises(ValueError, match="pricing_time_field"):
        cost_fields({"cost_ms": 10., "pricing_time_field": "service_ms"})
