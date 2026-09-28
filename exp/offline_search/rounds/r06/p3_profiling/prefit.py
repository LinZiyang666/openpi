"""CPU-only coordinator prefit of selected campaign arms. Never starts a server."""
import argparse
import json
from pathlib import Path


def main():
    from exp.offline_search.closed_loop import plugin
    plugin._git_head = lambda: None
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--spec", type=Path, default=Path(__file__).with_name("arms.json"))
    ap.add_argument("--arms", nargs="*")
    a = ap.parse_args()
    rows = json.loads(a.spec.read_text().replace("<RUN>", str(a.run_root)))
    for row in rows:
        if a.arms and row["name"] not in a.arms:
            continue
        argv = ["--os-method", row["method"], "--os-kwargs", json.dumps(row["kwargs"]),
                "--os-cell", f'{row["model"]}_{row["suite"]}_cache',
                "--os-log-dir", str(a.run_root / "prefit_logs" / row["name"]),
                "--os-tag", row["name"], *row["plugin_args"]]
        opts, rest = plugin.parse_cli(argv)
        assert not rest
        runtime = plugin.PluginRuntime(opts, row["model"])
        print(row["name"], opts.os_fit_artifact, flush=True)
        del runtime


if __name__ == "__main__":
    main()
