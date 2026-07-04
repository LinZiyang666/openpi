"""Project-root pytest plugins.

Two responsibilities for the ``manual`` / ``env_dependent`` markers (declared in
``pyproject.toml``):

1. **Default-skip the heavy set.** ``manual`` (GPU / model download / external
   services) and ``env_dependent`` (failures usually from local env limits —
   GPU memory, GCP credentials, un-downloaded model / tokenizer assets) tests
   are slow and environment-fragile. A bare ``uv run pytest`` should stay fast,
   so both are skipped unless ``--run-manual`` is passed. CI passes the flag on
   its provisioned runner (see ``.github/workflows/test.yml``).
2. **Explain env_dependent failures.** When the heavy set does run, the
   ``pytest_terminal_summary`` hook appends a dedicated section so a failure in
   an ``env_dependent`` test is not mistaken for a code regression.
"""

from __future__ import annotations

import pytest

# ------------------------------------------------------------------
# Collection-time snapshot of env_dependent markers
# ------------------------------------------------------------------
# Populated by ``pytest_collection_modifyitems`` and consumed by
# ``pytest_terminal_summary``. A module-level dict is sufficient because each
# pytest invocation runs in a fresh interpreter.

_env_reasons: dict[str, str] = {}


def pytest_addoption(parser):
    """Register ``--run-manual`` (off by default) to opt into the heavy set."""
    parser.addoption(
        "--run-manual",
        action="store_true",
        default=False,
        help=(
            "Run @pytest.mark.manual / @pytest.mark.env_dependent tests "
            "(GPU / model download / network / GCS). Off by default so a bare "
            "`pytest` stays fast; CI passes this flag."
        ),
    )


def pytest_collection_modifyitems(config, items):
    """Snapshot ``env_dependent`` reasons and default-skip the heavy set.

    Without ``--run-manual``, every ``manual`` / ``env_dependent`` test is
    marked skipped so a bare ``pytest`` never loads models / hits GPU / network
    / GCS. With the flag, nothing is skipped here (the caller still composes
    with ``-m`` — e.g. CI adds ``-m "not manual"`` to run env_dependent but
    exclude manual).
    """
    run_manual = config.getoption("--run-manual")
    skip_heavy = pytest.mark.skip(reason="needs --run-manual (manual/env_dependent)")
    for item in items:
        marker = item.get_closest_marker("env_dependent")
        if marker is not None:
            reason = marker.kwargs.get("reason") or (marker.args[0] if marker.args else "")
            _env_reasons[item.nodeid] = reason or "(no reason provided)"
        if not run_manual and ("manual" in item.keywords or "env_dependent" in item.keywords):
            item.add_marker(skip_heavy)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Append an env-dependent failures section to the pytest terminal summary."""
    failed = terminalreporter.stats.get("failed", [])
    env_failed = [(r.nodeid, _env_reasons[r.nodeid]) for r in failed if r.nodeid in _env_reasons]
    if not env_failed:
        return

    tr = terminalreporter
    tr.write_sep("=", "Environment-dependent failures — may not be code defects")
    tr.write_line(
        "The following tests carry @pytest.mark.env_dependent. They typically "
        "require a GPU with sufficient memory, GCP credentials, or pre-downloaded "
        "model / tokenizer assets — none of which are guaranteed on every "
        "developer machine. CI (`.github/workflows/test.yml`) runs on the "
        "`openpi-verylarge` runner where the required environment is provisioned, "
        "so a failure locally here does not necessarily indicate a code regression."
    )
    tr.write_line("Before investigating as a code bug, verify the listed environment prerequisite.")
    tr.write_line("")
    for nodeid, reason in env_failed:
        tr.write_line(f"  FAILED {nodeid}")
        tr.write_line(f"         reason: {reason}")
