"""Run each real prompt in a fresh model process, and certify replay with nsys."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .latency import ROOT, STEPS, SUITES, certificate, sha

PYTHONS = {"dp_libero": "/data/mw_finetune/dp/env/bin/python",
           "fastwam_libero": "/data/mw_finetune/envs/fastwam/bin/python",
           "cosmos_libero": "/home/weiland/projects/openpi_ext/lines/cosmos-policy/.venv/bin/python"}


def parse_trace(text, marker):
    rows = list(csv.reader(text.splitlines()))
    if not marker or not any(v.strip().rsplit(":", 1)[-1] == marker for r in rows for v in r):
        raise ValueError("nsys trace does not contain this run's unique NVTX marker")
    count_index = name_index = None
    launches = captures = parsed = 0
    for row in rows:
        labels = [v.strip() for v in row]
        count = next((v for v in ("Num Calls", "Instances", "Calls") if v in labels), None)
        if "Name" in labels and count:
            if parsed:
                break  # Next report is NVTX, not another CUDA API table.
            name_index, count_index = labels.index("Name"), labels.index(count)
            continue
        if name_index is None:
            continue
        if not labels or len(labels) <= max(count_index, name_index):
            if parsed:
                break
            continue
        try:
            calls = int(labels[count_index].replace(",", ""))
        except ValueError:
            if parsed:
                break
            continue
        parsed += 1
        name = labels[name_index]
        if "cudaGraphLaunch" in name or "cuGraphLaunch" in name:
            launches += calls
        if any(v in name for v in ("StreamBeginCapture", "StreamEndCapture", "GraphInstantiate")):
            captures += calls
    if not parsed:
        raise ValueError("nsys trace contains no CUDA API call table")
    return {"cudagraph_launch_count": launches, "capture_calls_after_warmup": captures}


def certify_trace(receipt, text, path):
    counts = parse_trace(text, receipt["trace_marker"])
    result = dict(receipt)
    result["torch_cudagraph_launch_count"] = receipt["cudagraph_launch_count"]
    result["torch_capture_calls_after_warmup"] = receipt["capture_calls_after_warmup"]
    result.update(counts)
    result.update(certificate(receipt["expected_cudagraph_launch_count"], counts["cudagraph_launch_count"],
        counts["capture_calls_after_warmup"], receipt["cudagraph_skips"], receipt["compiles_after_warmup"], receipt["compile_count"],
        receipt.get("compiled_stages", 3)))
    result["valid"] = bool(result["certified"] and receipt["valid"])
    result.update(certification_source="nsys CUDA API trace + torch per-budget replay counts",
                  cuda_trace_path=str(path), cuda_trace_sha256=sha(path))
    return result


def commands(model, suite, directory, python, nsys=None, extra=()):
    out = directory / (model + "_" + suite + ".json")
    invocation = [python, "-m", "exp.r17_integration.profile.latency", "--model", model,
                  "--suite", suite, "--out", str(out), *extra]
    prefix = out.with_suffix("")
    profile = ([nsys, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--force-overwrite=true",
                "-o", str(prefix), *invocation] if nsys else invocation)
    stats = ([nsys, "stats", "--report", "cuda_api_sum,nvtx_pushpop_sum", "--format", "csv",
              "--force-export=true", str(prefix) + ".nsys-rep"] if nsys else None)
    return out, profile, stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=STEPS, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--suite", choices=(*SUITES, "both"), default="both")
    p.add_argument("--python", help="model's existing Python island; selected automatically by default")
    args, extra = p.parse_known_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    nsys = shutil.which("nsys")
    python = args.python or PYTHONS[args.model]
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join((str(ROOT), str(ROOT / "src"), str(ROOT / "packages/openpi-client/src"), env.get("PYTHONPATH", "")))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    # The Cosmos island needs the formal server's environment (LIBERO deps, config, HF cache, pip CUDA libraries).
    if args.model == "cosmos_libero":
        from exp.r17_integration.ops.formal_eval import server_env
        env.update(server_env(dict(model="cosmos", server_node="local",
                                   vendor="/home/weiland/projects/openpi_ext/lines/cosmos-policy"), str(ROOT)))
    ok = True
    for suite in SUITES if args.suite == "both" else (args.suite,):
        out, invocation, stats = commands(args.model, suite, args.out_dir, python, nsys, extra)
        print("Running", args.model, suite, "with", "nsys" if nsys else "torch graph counters", flush=True)
        with out.with_suffix(".log").open("w") as log:
            subprocess.run(invocation, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        receipt = json.loads(out.read_text())
        if stats:
            trace = out.with_suffix(".trace.csv")
            export = subprocess.run(stats, env=env, cwd=ROOT, text=True, capture_output=True, check=False)
            trace.write_text(export.stdout)
            try:
                if export.returncode:
                    raise ValueError(export.stderr.strip())
                receipt = certify_trace(receipt, export.stdout, trace)
            except ValueError as exc:
                receipt.update(certified=False, valid=False, trace_error=str(exc))
            out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        ok &= receipt["valid"]
        print(json.dumps({"out": str(out), "valid": receipt["valid"], "coefficients": receipt["coefficients"]}), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
