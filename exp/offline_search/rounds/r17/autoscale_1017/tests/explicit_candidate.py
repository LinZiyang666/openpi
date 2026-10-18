"""Explicit test-only ON/OFF selection; never writes the owner's switch."""
import pytest
@pytest.fixture(autouse=True)
def candidate_policy(monkeypatch,request):
    on=request.node.path.name not in ('test_off_switch.py','test_reset_legacy_zero.py','test_validation_lane.py')
    from exp.offline_search.rounds.r17.autoscale_1017 import switch,defaults,legacy_release
    from exp.offline_search.rounds.r17.filter_failed_knob import vla_workers,formal_a_queue
    from exp.offline_search.closed_loop.ops.h100 import fleet
    for module in (switch,defaults,legacy_release,vla_workers,formal_a_queue,fleet):
        monkeypatch.setattr(module,'enabled',lambda:on)
    old=fleet.FLEETS['timan107']
    monkeypatch.setitem(fleet.FLEETS,'timan107',(*old[:2],104 if on else 64))
