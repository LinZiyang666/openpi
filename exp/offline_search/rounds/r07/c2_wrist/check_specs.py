"""Validate every public arm's bank load, fit metadata, and emitted config."""
import ast
import json
from pathlib import Path

from exp.offline_search.harness import api
from exp.offline_search.closed_loop.plugin import parse_cli, PluginRuntime
from .method import StageWrist
from .specs import make_specs


def main():
    arms = make_specs("/tmp/r7_C2")
    checks = []
    for row in arms:
        ctx = api.Context(root="/home/weiland/trace_runs/offline_search_store", cell=f"pi05_{row['suite']}_cache",
                          seed=0, scratch=Path("/tmp/r7_C2/spec_checks"))
        method = StageWrist(**row["kwargs"])
        method.fit(ctx.open_library("current"), ctx)
        assert method.wrist.tasks[0].Wf.shape == (72, 72)
        assert (Path("/tmp/r7_C2/emitted/config")/(row["name"]+".yaml")).exists()
        assert row["server_env"] == {"BATCHING_MAX_BATCH_SIZE": "1"}
        checks.append(row["name"])
    args = ["--os-method", arms[0]["method"], "--os-cell", "groot_spatial_cache", "--os-log-dir", "/tmp/r7_C2/refused",
            "--os-request-cameras", "--os-blind", "--os-no-shadow-native", "--os-tokens", "off"]
    opts, _ = parse_cli(args)
    try:
        PluginRuntime(opts, "groot")
    except ValueError as exc:
        assert "unavailable" in str(exc)
    else:
        raise AssertionError("unsupported GR00T encoder path was accepted")
    scripts = list(Path(__file__).parent.glob("*.py"))
    for script in scripts:
        ast.parse(script.read_text(), filename=str(script))
    report = dict(PASS=True, arm_bank_loads=len(checks), configs=len(checks), groot_refusals=1, scripts_parsed=len(scripts))
    Path("/tmp/r7_C2/spec_checks.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
