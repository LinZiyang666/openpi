"""Emit the arm yamls, single-arm run_gtp matrices and arms.json of a closed-loop run.

Every arm serves the trace_dual pure-cache yaml of its model x suite (exp/trace_dual/config/tr_<m>_<s>_cache.yaml:
the historical retrieval section, always_search gate, threshold judge at -1e6 = FULL_HIT every decision) with the
trace block removed (no trace flags). What differs between arms is only the server side:

  mode native  -- native retrieval, served through the plugin entry in log-only mode (--os-method native): the B0
                  pure-cache control re-run with per-decision timing / winner logs
  mode stock   -- the same yaml through the stock entry point (no plugin code at all)
  mode plugin  -- the plugin serves --os-method <spec> --os-kwargs <json> (the yaml only provides key builder,
                  library pkl for the native shadow / payloads, and the always-FULL_HIT contract)

The arm name is the run_gtp yaml_id and hence the task_uid prefix: keep it unique per method x kwargs x model x suite.

    .venv/bin/python -m exp.offline_search.closed_loop.ops.emit_arms --run-root /home/weiland/trace_runs/os_closed_loop/<tag> \
        --spec <arms_in.json>

arms_in.json: [{"name": "oscl_pi05_sp_b0nat", "model": "pi05", "suite": "spatial", "mode": "native"},
               {"name": "oscl_pi05_sp_b0plug", "model": "pi05", "suite": "spatial", "mode": "plugin",
                "method": "exp.offline_search.harness.baselines:B0Current", "kwargs": {},
                "plugin_args": ["--os-no-shadow-native"]}, ...]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import re

import yaml

REPO = pathlib.Path(__file__).resolve().parents[4]
SRC = REPO / "exp" / "trace_dual" / "config"
SUITES = {"spatial": ("sp", "libero_spatial"), "l10": ("l10", "libero_10")}
REMOTE_CFG = "os_cl/cfg"          # relative to the timan107 island root /scratch/zixuans8/openpi_trace


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--spec", required=True)
    a = ap.parse_args(argv)
    run = pathlib.Path(a.run_root)
    cfg_dir = run / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads(pathlib.Path(a.spec).read_text())
    arms_path = run / "arms.json"
    arms = {r["arm"]: r for r in json.loads(arms_path.read_text())} if arms_path.exists() else {}
    for s in spec:
        name = s["name"]
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise SystemExit(f"arm name {name!r}: use [A-Za-z0-9_] only (it becomes the yaml_id / task_uid prefix)")
        model, suite, mode = s["model"], s["suite"], s["mode"]
        if model not in ("pi05", "groot") or suite not in SUITES or mode not in ("native", "stock", "plugin"):
            raise SystemExit(f"bad arm {s}")
        if mode == "plugin" and not s.get("method"):
            raise SystemExit(f"plugin arm {name} needs 'method'")
        short, full = SUITES[suite]
        src = SRC / f"tr_{model}_{short}_cache.yaml"
        cfg = yaml.safe_load(src.read_text())
        cfg = copy.deepcopy(cfg)
        cfg.pop("trace", None)
        cp1 = cfg["checkpoints"]["cp1"]
        assert cp1["gate"] == {"type": "always_search"}, cp1["gate"]
        assert cp1["judge"]["type"] == "threshold" and float(cp1["judge"]["threshold"]) <= -1e5, cp1["judge"]
        assert cfg["write_policy"] == {"type": "never"}
        y = cfg_dir / f"{name}.yaml"
        y.write_text(yaml.safe_dump(cfg, sort_keys=False))
        mx = cfg_dir / f"matrix_{name}.yaml"
        mx.write_text(yaml.safe_dump({"arms": [{"arm": name, "yaml": f"{REMOTE_CFG}/{name}.yaml", "suite": full}]},
                                     sort_keys=False))
        prev = arms.get(name)
        row = {"arm": name, "model": model, "suite": full, "suite_short": suite, "cell": f"{model}_{suite}_cache",
               "mode": mode, "method": s.get("method", "native" if mode == "native" else None),
               "kwargs": s.get("kwargs", {}), "plugin_args": s.get("plugin_args", []),
               "yaml": str(y), "matrix": str(mx), "remote_yaml": f"{REMOTE_CFG}/{name}.yaml",
               "remote_matrix": f"{REMOTE_CFG}/matrix_{name}.yaml", "src_yaml": str(src),
               "src_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
               "yaml_sha256": hashlib.sha256(y.read_bytes()).hexdigest(),
               "library": cfg["backend"]["in_memory"]["preload_path"]}
        if prev and prev != row:
            print(f"note: arm {name} redefined")
        arms[name] = row
    arms_path.write_text(json.dumps(list(arms.values()), indent=1))
    print(json.dumps([{k: r[k] for k in ("arm", "model", "suite", "mode", "method")} for r in arms.values()], indent=1))


if __name__ == "__main__":
    main()
