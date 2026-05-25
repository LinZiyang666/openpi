"""WarmupEvalStrategy tests (M4): dump aggregation + graph construction."""

from __future__ import annotations

from exp.verdict_factor_judge.strategies.warmup_eval_strategy import WarmupEvalStrategy
from exp.verdict_factor_judge.strategies.warmup_eval_strategy import aggregate_dump
from openpi.conductor import task as T  # noqa: N812

S1 = T.ServerEndpoint("h", 8001)


# ------------------------------------------------------------------
# dump aggregation
# ------------------------------------------------------------------


def test_aggregate_dump_finite_only():
    content = b"\n".join(
        [
            b'{"factor_raw": {"a": 1.0, "b": 2.0}}',
            b'{"factor_raw": {"a": "nan", "b": 3.0}}',  # a parses to NaN -> skipped; b kept
            b'{"factor_raw": {"a": 4.0}}',
            b"",  # blank line tolerated
        ]
    )
    buf = aggregate_dump(content)
    assert buf["a"] == [1.0, 4.0]  # NaN value dropped
    assert buf["b"] == [2.0, 3.0]  # same-row valid value still collected


def test_aggregate_dump_caps_per_key():
    lines = [b'{"factor_raw": {"a": 1.0}}'] * 10
    buf = aggregate_dump(b"\n".join(lines), max_per_key=3)
    assert buf["a"] == [1.0, 1.0, 1.0]


def test_aggregate_dump_declared_keys_filter():
    content = b'{"factor_raw": {"a": 1.0, "b": 2.0}}'
    buf = aggregate_dump(content, declared_keys={"a"})
    assert "a" in buf
    assert "b" not in buf


# ------------------------------------------------------------------
# plan
# ------------------------------------------------------------------


def _strategy(**kw):
    return WarmupEvalStrategy(
        task_ids=[0, 1], warmup_trials=2, eval_trials=5, task_suite_name="libero_spatial", yaml_dir="/cfg", **kw
    )


def test_plan_builds_warmup_eval_chain():
    g = _strategy().plan(["y"], {"y": S1})
    g.validate()
    assert set(g.stages) == {"y:warmup", "y:eval"}
    assert g.stages["y:warmup"].produces_calib_id == "y"
    assert g.stages["y:eval"].consumes_calib_id == "y"
    assert g.upstreams("y:eval") == ["y:warmup"]
    # 2 tasks x 2 warmup trials = 4 warmup episodes; x5 eval = 10 eval episodes
    assert len(g.stages["y:warmup"].episodes) == 4
    assert len(g.stages["y:eval"].episodes) == 10
    # cleanup_id constraint: dump named <cleanup_id>__warmup
    assert g.calibrations["y"].dump_name == "y__warmup"


def test_plan_skip_warmup_has_no_warmup_stage():
    g = _strategy(skip_warmup=True).plan(["y"], {"y": S1})
    g.validate()
    assert set(g.stages) == {"y:eval"}
    assert g.stages["y:eval"].consumes_calib_id is None
