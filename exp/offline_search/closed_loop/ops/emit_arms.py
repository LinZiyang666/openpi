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

Mixed HIT/MISS arms (R3): add ``"full_model": true`` and the judge flags in plugin_args, e.g.
    {"name": "r3mx_p_sp_b0h70", "model": "pi05", "suite": "spatial", "mode": "plugin", "full_model": true,
     "method": "exp.offline_search.harness.baselines:B0Current", "kwargs": {},
     "plugin_args": ["--os-judge", "quantile:0.7:1000:0.974", "--os-fit-artifact", "<RUN>/fits/r3mx_p_sp_b0h70.pkl"]}
``full_model`` is carried into arms.json; chain.sh then starts that arm's servers with STAGE1_ONLY=0 (stage 2/3
loaded, full-model NEED_MB). A judge that can MISS without full_model is refused here (the server would die on its
first MISS). Pure-cache rows are emitted exactly as before (no new key unless the spec carries it).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import re

import yaml

REPO = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "exp/trace_dual/config").is_dir())
SRC = REPO / "exp" / "trace_dual" / "config"
SUITES = {"spatial": ("sp", "libero_spatial"), "l10": ("l10", "libero_10")}
REMOTE_CFG = "os_cl/cfg"          # relative to the timan107 island root /scratch/zixuans8/openpi_trace


def merge_yaml(base, patch):
    """Recursive dictionary merge; lists/scalars/null replace, never mutate the spec."""
    out = copy.deepcopy(base)
    for key, value in patch.items():
        out[key] = merge_yaml(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else copy.deepcopy(value)
    return out


def expand_replicates(spec):
    out = []
    for row in spec:
        seeds = row.get("seeds")
        if seeds is None:
            out.append(copy.deepcopy(row))
            continue
        if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be a nonempty list of distinct integers")
        for seed in seeds:
            r = copy.deepcopy(row)
            r.pop("seeds")
            r["name"] += f"_s{seed}"
            r["server_seed"] = seed
            out.append(r)
    names = [r["name"] for r in out]
    if any("seeds" in row or "server_seed" in row for row in spec) and len(set(names)) != len(names):
        raise ValueError("duplicate arm names in spec/seed expansion")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--spec", required=True)
    a = ap.parse_args(argv)
    run = pathlib.Path(a.run_root)
    cfg_dir = run / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    spec = expand_replicates(json.loads(pathlib.Path(a.spec).read_text()))
    arms_path = run / "arms.json"
    arms = {r["arm"]: r for r in json.loads(arms_path.read_text())} if arms_path.exists() else {}
    for s in spec:
        name = s["name"]
        if s.get("pure_inference"):
            s["mode"], s["full_model"] = "plugin", True
            s["method"] = ("exp.offline_search.rounds.r04.k4_eval.seeded_inference:SeededInference"
                           if "server_seed" in s else "exp.offline_search.harness.baselines:B0Current")
            s["kwargs"] = {}
            args = s.get("plugin_args", [])
            if any(x.startswith("--os-judge") for x in args):
                raise ValueError("pure_inference sets periodic:1; do not supply judge overrides")
            s["plugin_args"] = [*args, "--os-judge", "periodic:1"]
        if "server_seed" in s:
            seed = s["server_seed"]
            if type(seed) is not int or not 0 <= seed < 2 ** 32 - 65536:
                raise ValueError("server_seed must be an integer in [0, 2**32-65536)")
            if not s.get("pure_inference"):
                raise ValueError("server_seed currently requires pure_inference (policy RNG initializer)")
            if any(x.startswith("--os-seed") for x in s.get("plugin_args", [])):
                raise ValueError("server_seed conflicts with --os-seed")
        client = s.get("client_overrides", {})
        if not isinstance(client, dict) or set(client) - {"replan_steps", "resize_size"}:
            raise ValueError("client_overrides supports replan_steps and resize_size")
        client = copy.deepcopy(client)
        if "replan_steps" in s:
            if "replan_steps" in client and client["replan_steps"] != s["replan_steps"]:
                raise ValueError("conflicting replan_steps overrides")
            client["replan_steps"] = s["replan_steps"]
        if any(type(v) is not int or v < 1 for v in client.values()):
            raise ValueError("client overrides must be positive integers")
        if client.get("replan_steps", 5) > (50 if s["model"] == "pi05" else 16):
            raise ValueError("replan_steps exceeds policy horizon")
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise SystemExit(f"arm name {name!r}: use [A-Za-z0-9_] only (it becomes the yaml_id / task_uid prefix)")
        model, suite, mode = s["model"], s["suite"], s["mode"]
        if model not in ("pi05", "groot") or suite not in SUITES or mode not in ("native", "stock", "plugin"):
            raise SystemExit(f"bad arm {s}")
        if mode == "plugin" and not s.get("method"):
            raise SystemExit(f"plugin arm {name} needs 'method'")
        pargs = list(s.get("plugin_args", []))
        judge = next((pargs[i + 1] for i, x in enumerate(pargs[:-1]) if x == "--os-judge"), None)
        step0 = next((pargs[i + 1] for i, x in enumerate(pargs[:-1]) if x == "--os-judge-step0"), "judge")
        full_model = bool(s.get("full_model", False))
        if judge is not None and mode != "plugin":
            raise SystemExit(f"arm {name}: --os-judge needs mode plugin")
        can_miss = judge is not None and (judge.split(":")[0] != "always" or step0 == "miss")
        if can_miss and not full_model:
            raise SystemExit(f"arm {name}: --os-judge {judge} can MISS (stage 2/3 needed) -> set \"full_model\": true")
        short, full = SUITES[suite]
        src = SRC / f"tr_{model}_{short}_cache.yaml"
        cfg = yaml.safe_load(src.read_text())
        cfg = copy.deepcopy(cfg)
        cfg.pop("trace", None)
        cp1 = cfg["checkpoints"]["cp1"]
        assert cp1["gate"] == {"type": "always_search"}, cp1["gate"]
        assert cp1["judge"]["type"] == "threshold" and float(cp1["judge"]["threshold"]) <= -1e5, cp1["judge"]
        assert cfg["write_policy"] == {"type": "never"}
        if "yaml_patch" in s:
            if not isinstance(s["yaml_patch"], dict):
                raise ValueError("yaml_patch must be a dictionary")
            cfg = merge_yaml(cfg, s["yaml_patch"])
            # CacheConfig requires evidence_dir whenever miss.num_steps is present.
            if isinstance(cfg.get("miss"), dict) and cfg["miss"].get("num_steps") is not None:
                cfg["miss"].setdefault("evidence_dir", str(run / "evidence" / name))
        y = cfg_dir / f"{name}.yaml"
        y.write_text(yaml.safe_dump(cfg, sort_keys=False))
        mx = cfg_dir / f"matrix_{name}.yaml"
        matrix_row = {"arm": name, "yaml": f"{REMOTE_CFG}/{name}.yaml", "suite": full}
        if client:
            matrix_row["client_overrides"] = client
        mx.write_text(yaml.safe_dump({"arms": [matrix_row]},
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
        for field in ("yaml_patch", "pure_inference", "server_seed", "manifest", "cost_ledger", "server_env"):
            if field in s:
                row[field] = s[field]
        if "server_env" in row:
            if not isinstance(row["server_env"], dict) or any(not re.fullmatch(r"[A-Z][A-Z0-9_]*", k)
                                                            for k in row["server_env"]):
                raise ValueError("server_env must be a dict with uppercase environment variable names")
        if model == "groot" and isinstance(cfg.get("miss"), dict) and cfg["miss"].get("num_steps") is not None:
            row.setdefault("server_env", {}).setdefault("GROOT_DENOISING_STEPS", cfg["miss"]["num_steps"])
            if int(row["server_env"]["GROOT_DENOISING_STEPS"]) != int(cfg["miss"]["num_steps"]):
                raise ValueError("GROOT_DENOISING_STEPS must match miss.num_steps")
        if client:
            row["client_overrides"] = client
            row["replan_steps"] = client.get("replan_steps", 5)
        if "full_model" in s or judge is not None:
            row["full_model"] = full_model          # read by chain.sh (STAGE1_ONLY=0 + full-model NEED_MB)
            row["judge"] = judge
        if prev and prev != row:
            print(f"note: arm {name} redefined")
        arms[name] = row
    arms_path.write_text(json.dumps(list(arms.values()), indent=1))
    print(json.dumps([{k: r[k] for k in ("arm", "model", "suite", "mode", "method")} for r in arms.values()], indent=1))


if __name__ == "__main__":
    main()
