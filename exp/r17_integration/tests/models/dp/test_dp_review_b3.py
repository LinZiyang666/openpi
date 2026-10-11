"""Review B round-3 regressions for DP-T LIBERO benchmark-document consumption."""
from dataclasses import asdict, replace
import json

import pytest

from exp.r17_integration.benchmarks.libero_bridge import LiberoBenchmarkSpec, SUITES
from exp.r17_integration.tests.models.dp.conftest import observation

TASKS = tuple((f"{suite}/task{i}", f"pick object{i}", suite) for suite in SUITES for i in range(10))


def emitted(spec):
    """Exactly what `python -m exp.r17_integration.benchmarks.libero_bridge --output` writes."""
    return json.loads(json.dumps(spec.manifest(), sort_keys=True))


@pytest.mark.parametrize("wire", [96, 256])
def test_b3_dp_accepts_p2_emitted_libero_manifest_and_binds_wire(make_adapter, req_context, wire):
    spec = LiberoBenchmarkSpec(TASKS, wire_size=wire)
    document = emitted(spec)
    assert "protocol_identity" in document and "task_table_identity" in document
    adapter = make_adapter("libero", libero_benchmark_specs=[document])
    assert adapter.metadata().allowed_benchmark_protocols == (spec.identity,)
    request = replace(req_context, benchmark_protocol=spec.identity)
    obs = observation("libero")
    if wire == 256:
        with pytest.raises(ValueError, match="bound benchmark protocol"):
            adapter.prepare(obs, request)
    else:
        adapter.prepare(obs, request)
    # Dataclass documents (the older in-process form) still parse identically.
    same = make_adapter("libero", libero_benchmark_specs=[asdict(spec)])
    assert same.metadata().allowed_benchmark_protocols == (spec.identity,)


@pytest.mark.parametrize("key,value", [("protocol_identity", "sha256:forged"),
                                       ("task_table_identity", "sha256:forged"),
                                       ("ordered_tasks", ["libero_spatial/other"])])
def test_b3_dp_rejects_forged_p2_document_identities(make_adapter, key, value):
    document = {**emitted(LiberoBenchmarkSpec(TASKS, wire_size=96)), key: value}
    with pytest.raises(ValueError, match=key):
        make_adapter("libero", libero_benchmark_specs=[document])


@pytest.mark.parametrize("changes,field", [
    ({"seed": 42, "seed_mode": "fixed", "replan_steps": 32, "max_steps_by_suite": (280, 280, 300, 520),
      "protocol_id": "libero_released_reference_r17_v1", "wire_size": 256}, "replan_steps"),
    ({"replan_steps": 1}, "replan_steps"), ({"settle_steps": 0}, "settle_steps"),
    ({"seed": 8}, "seed"), ({"seed_mode": "fixed"}, "seed_mode"),
    ({"max_steps_by_suite": (280, 280, 300, 520)}, "max_steps_by_suite")])
def test_b3_dp_rejects_non_dp_libero_execution_profiles(make_adapter, changes, field):
    spec = LiberoBenchmarkSpec(TASKS, **{"wire_size": 96, **changes})
    for document in (emitted(spec), asdict(spec)):
        with pytest.raises(ValueError, match=field):
            make_adapter("libero", libero_benchmark_specs=[document])
