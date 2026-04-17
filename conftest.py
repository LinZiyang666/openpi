"""Project-root pytest plugins.

Registers behavior for the ``env_dependent`` marker (declared in ``pyproject.toml``).
The marker flags tests whose failures are typically caused by local environment
limits — insufficient GPU memory, missing GCP credentials, un-downloaded model /
tokenizer assets — rather than real code defects.

The ``pytest_terminal_summary`` hook below appends a dedicated section to the
standard pytest summary so that contributors running ``uv run pytest`` locally
can immediately see which failures are most likely environmental and therefore
worth provisioning around (or ignoring) before investigating as bugs.
"""

from __future__ import annotations

# ------------------------------------------------------------------
# Collection-time snapshot of env_dependent markers
# ------------------------------------------------------------------
# Populated by ``pytest_collection_modifyitems`` and consumed by
# ``pytest_terminal_summary``. A module-level dict is sufficient because each
# pytest invocation runs in a fresh interpreter.

_env_reasons: dict[str, str] = {}


def pytest_collection_modifyitems(config, items):
    """Capture ``env_dependent`` markers keyed by nodeid."""
    for item in items:
        marker = item.get_closest_marker("env_dependent")
        if marker is None:
            continue
        reason = marker.kwargs.get("reason") or (marker.args[0] if marker.args else "")
        _env_reasons[item.nodeid] = reason or "(no reason provided)"


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
