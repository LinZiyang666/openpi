import numpy as np
import pytest

from exp.r17_integration.common.qualification import run_probe
from exp.r17_integration.common.replicas import plan_replicas


def test_mechanical_probe_exercises_stock_hit_miss_capture_and_tail(provider):
    receipt=run_probe(provider,[{"state":np.arange(4,dtype=np.float32)}])
    assert receipt["status"]=="passed"
    assert [r["k"] for r in receipt["rows"] if "synthetic_full_hit" in r]==[4,2,1]
    assert all(r["tail"]=="passed" for r in receipt["rows"] if "tail" in r)
    with pytest.raises(TimeoutError):run_probe(provider,[{"state":np.arange(4,dtype=np.float32)}],deadline_seconds=0)


def test_replica_plan_does_not_start_processes(provider):
    plan=plan_replicas(("python","-m","model.serve"),public_port=31000,backend_ports=(31001,31002),model_spec=provider.model_spec)
    assert plan.commands[1][-2:]==("--port","31002")
    with pytest.raises(ValueError):plan_replicas(("python",),public_port=31000,backend_ports=(31000,),model_spec=provider.model_spec)
