import pytest


@pytest.fixture(autouse=True)
def no_real_island_install(monkeypatch):
    """sync() ends in install_scheduler, which pushes into the real worker islands (10-21: a test run
    rewrote timan107's live island). Tests never reach a machine."""
    from exp.offline_search.closed_loop.ops.h100 import control
    monkeypatch.setattr(control, "install_scheduler", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def no_real_transfer(monkeypatch):
    """push/collect move files with formal_eval.transfer (public-IP rsync, its own tether exec): in tests it goes
    through control.run, which tests mock, and never reaches a machine."""
    from exp.offline_search.closed_loop.ops.h100 import control
    from exp.r17_integration.ops import formal_eval
    monkeypatch.setattr(formal_eval, "transfer", lambda argv, **kw: control.run(["transfer", *map(str, argv)]))
