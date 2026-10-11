"""Review A6/A10/A11 and B9: real ledger shape, final ack, timing bias."""
import pytest

from exp.r17_integration.debug.validate import EPISODE_KEYS, digest
from exp.r17_integration.search.analyze import summarize
from exp.r17_integration.search.audit import audit_campaign
from exp.r17_integration.tests.debug.fixtures import binding, record


def run_rows():
    episode = dict(record(n_decisions=2), accepted=True, status="done")
    decisions = [record(seq=i) for i in range(2)]
    costs, acks = [], []
    for i, row in enumerate(decisions):
        row.pop("cost_ms")
        row["executed_count"] = None
        identity = {k: row[k] for k in EPISODE_KEYS}
        identity["orig_init_state_idx"] = identity.pop("orig_init_idx")
        costs.append(dict(identity, decision_seq=i, source="policy", cursor=0,
                          stage_calls=[1, 1, 5], actual_nfe=5,
                          server_timing={"infer_ms": 2.5+i}, capture_enabled=False,
                          pricing_eligible=True, queue_wait_ms=.5, service_ms=2.+i,
                          pricing_time_field="service_ms"))
        acks.append(dict(identity, _kind="execution_ack", decision_seq=i,
                         executed_count=5 if i == 0 else 2, last_cursor=4 if i == 0 else 1,
                         terminal=i == 1))
    return episode, decisions, costs, acks


def test_actual_server_timing_and_final_client_ack_are_consumed():
    ep, decisions, costs, acks = run_rows()
    result = summarize([ep["task_uid"]], [ep], decisions, task_groups={"reach-v3": "easy"},
                       binding=binding(), cost_rows=costs, execution_rows=acks,
                       full_baseline_ms={"binding_digest": digest(binding()), "absolute_ms": 12.,
                                         "pricing_absolute_ms": 10., "pricing_time_field": "service_ms",
                                         "capture_enabled":False,"pricing_eligible":True})
    assert result["cost"]["status"] == "available"
    assert result["cost"]["pricing_absolute_ms"] == 5. and result["cost"]["queue_wait_total_ms"] == 1.
    assert result["cost"]["absolute_ms"] == 6.
    assert result["cost"]["executed_controls"] == 7
    assert result["cost"]["normalized_within_model"] == .5
    audit = audit_campaign([ep["task_uid"]], [ep], decisions, cost_rows=costs, execution_rows=acks)
    assert audit["cost_status"] == "available"


@pytest.mark.parametrize("source", ["tail", "policy_tail", "library_tail", "cache_tail"])
@pytest.mark.parametrize("calls,nfe", [([1, 0, 0], 0), ([0, 0, 0], 1)])
def test_every_tail_source_is_audited(source, calls, nfe):
    row = dict(record(n_decisions=1), source=source, stage_calls=calls, actual_nfe=nfe)
    with pytest.raises(ValueError, match="tail"):
        audit_campaign([row["task_uid"]], [dict(row, accepted=True, status="done")], [row])


@pytest.mark.parametrize("field,value", [("bundle_generation", "other"), ("task_data_hash", "other"),
                                         ("orig_init_state_idx", 7), ("attempt", 2)])
def test_client_ledger_cannot_silently_join_wrong_identity(field, value):
    ep, rows, costs, acks = run_rows()
    acks[1][field] = value
    with pytest.raises(ValueError, match="identity|unmatched"):
        summarize([ep["task_uid"]], [ep], rows, task_groups={"reach-v3": "easy"}, binding=binding(),
                  cost_rows=costs, execution_rows=acks)


def test_capture_timings_remain_diagnostic_and_never_enter_pricing():
    ep, rows, costs, acks = run_rows()
    costs[1].update(capture_enabled=True, pricing_eligible=True, capture_overhead_ms=.8)
    report = summarize([ep["task_uid"]], [ep], rows, task_groups={"reach-v3": "easy"}, binding=binding(),
                       cost_rows=costs, execution_rows=acks,
                       full_baseline_ms={"binding_digest": digest(binding()), "absolute_ms": 12.})
    assert report["cost"]["absolute_ms"] == 6.
    assert report["cost"]["pricing_status"] == "excluded_capture"
    assert report["cost"]["normalized_within_model"] is None
    assert report["cost"]["pricing_absolute_ms"] is None
    assert report["cost"]["capture_overhead_ms"] == .8


def test_unacknowledged_final_decision_and_missing_decision_cost_are_unknown():
    ep, rows, costs, acks = run_rows()
    report = audit_campaign([ep["task_uid"]], [ep], rows, cost_rows=costs, execution_rows=acks[:1])
    assert report["cost_status"] == "unsupported"
    result = summarize([ep["task_uid"]], [ep], [dict(record(), n_decisions=2)],
                       task_groups={"reach-v3": "easy"}, binding=binding())
    assert result["cost"]["status"] == "unsupported"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.])
def test_invalid_real_cost_is_rejected(value):
    ep, rows, costs, acks = run_rows()
    costs[0]["server_timing"]["infer_ms"] = value
    with pytest.raises(ValueError, match="cost|timing"):
        summarize([ep["task_uid"]], [ep], rows, task_groups={"reach-v3": "easy"}, binding=binding(),
                  cost_rows=costs, execution_rows=acks)


def test_client_and_server_execution_ack_must_agree():
    ep, rows, costs, acks = run_rows()
    rows[1].update(executed_count=5, cursor_end=5, execution_status="acknowledged")
    with pytest.raises(ValueError, match="execution"):
        audit_campaign([ep["task_uid"]], [ep], rows, cost_rows=costs, execution_rows=acks)


def test_capture_with_accepted_uid_attempt_but_wrong_generation_is_rejected():
    row=record(n_decisions=1)
    ep=dict(row,accepted=True,status="done")
    row["bundle_generation"]="wrong"
    with pytest.raises(ValueError,match="identity"):
        audit_campaign([ep["task_uid"]],[ep],[row])


def test_unmeasured_capture_overhead_stays_unknown():
    ep,rows,costs,acks=run_rows()
    costs[0]["capture_enabled"]=True
    report=summarize([ep["task_uid"]],[ep],rows,task_groups={"reach-v3":"easy"},binding=binding(),
                     cost_rows=costs,execution_rows=acks)
    assert report["cost"]["capture_overhead_ms"] is None
    assert report["cost"]["capture_overhead_status"]=="unknown"


def test_captured_or_unknown_reference_cost_cannot_normalize_plain_pricing():
    ep,rows,costs,acks=run_rows()
    for flags in ({},{"capture_enabled":True,"pricing_eligible":True}):
        report=summarize([ep["task_uid"]],[ep],rows,task_groups={"reach-v3":"easy"},binding=binding(),
                         cost_rows=costs,execution_rows=acks,
                         full_baseline_ms={"binding_digest":digest(binding()),"absolute_ms":12.,**flags})
        assert report["cost"]["pricing_status"]=="available"
        assert report["cost"]["normalized_within_model"] is None
        assert report["cost"]["normalization_status"]=="excluded_capture_or_unknown"


def test_sealed_journal_join_rejects_conflicting_original_index_alias():
    from exp.r17_integration.debug.reader_profiles import enrich_journal
    ep=record()
    row=dict(ep,orig_init_state_idx=7,accepted=True,status='done')
    with pytest.raises(ValueError,match='identity'):
        enrich_journal([row],[ep])
