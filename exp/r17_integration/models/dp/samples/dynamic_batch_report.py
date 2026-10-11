"""CPU-only aggregation of real receipts; never fills an unmeasured cell."""
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from exp.r17_integration.models.dp.manifest import code_identity, framework_identity, file_digest

ROOT = Path("exp/r17_integration/models/dp")


def main():
    samples = ROOT / "samples"
    phases = ["dynamic_batch_gpu_20261010", "dynamic_batch_gpu_final_20261010",
              "dynamic_batch_gpu_persistent_20261010", "dynamic_batch_gpu_admission_20261010",
              "dynamic_batch_gpu_k5_completion_20261010"]
    budgets = {name: json.loads((samples / name / "completed.json").read_text()) for name in phases}
    budgets["dynamic_batch_gpu_cap32_final_20261010"] = json.loads((samples / "dynamic_batch_gpu_cap32_final_20261010/GPU_WINDOW.json").read_text())
    seconds = sum(doc["GPU_seconds"] for doc in budgets.values())
    assert seconds <= 1200, "owner's aggregate GPU allowance exceeded"
    rows = []
    for name in ("dynamic_batch_gpu_admission_20261010", "dynamic_batch_gpu_k5_completion_20261010"):
        for row in json.loads((samples / name / "results.json").read_text()):
            directory = samples / name / f"k{row['K']}_r{row['replicas']}_b{row['max_batch']}_c12"
            assert (directory / "result.json").is_file()
            rows.append(dict(row, receipt=str(directory / "result.json"), status="measured"))
    keys = [(k, r, b) for k in (100, 10, 5) for r in (1,2) for b in (1,4,8,16,24,32)]
    lookup = {(row["K"],row["replicas"],row["max_batch"]):row for row in rows}
    assert len(lookup)==len(rows)==35
    missing = [key for key in keys if key not in lookup]
    assert missing == [(5,2,32)]
    table = ["| K / replicas × max B | calls/s | actions/s | p50 / p90 ms | W | max abs diff |",
             "|---|---:|---:|---:|---:|---:|"]
    for key in keys:
        label = f"{key[0]} / {key[1]}×{key[2]}"
        if key not in lookup:
            table.append(f"| {label} | — | — | — | — | unmeasured |")
        else:
            row = lookup[key]
            diff = "0" if not row["max_diff"] else f"{row['max_diff']:.2e}"
            table.append(f"| {label} | {row['calls_per_second']:.2f} | {row['actions_per_second']:.2f} | {row['p50_ms']:.0f} / {row['p90_ms']:.0f} | {row['watts']:.1f} | {diff} |")
    (samples / "dynamic_batch_table.md").write_text("\n".join(table)+"\n")
    tests = {}
    for filename in ("cpu_dynamic_batch_full.xml", "cpu_dynamic_batch_legacy.xml"):
        suite = ET.parse(ROOT / filename).getroot().find("testsuite")
        tests[filename] = {key:int(suite.attrib[key]) for key in ("tests","errors","failures","skipped")}
        assert tests[filename]["errors"]==tests[filename]["failures"]==tests[filename]["skipped"]==0
    seal = json.loads((samples / "dynamic_batch_assets/receipt.json").read_text())
    assert seal["model_sources"] == code_identity()
    assert seal["framework_sources"] == framework_identity()
    graph_checks, noise_checks, native_audits = 0, 0, []
    for name, first in (("dynamic_batch_gpu_admission_20261010","k100"),("dynamic_batch_gpu_k5_completion_20261010","k5")):
        for replicas in ((1,2) if first=="k100" else (2,)):
            for index in range(replicas):
                path = samples / name / f"{first}_r{replicas}_b1_c12/replica{index}/ready.json"
                report = json.loads(path.read_text())["cuda_graph_self_check"]
                assert report["passed"] and all(row["wire_bytes_exact"] and row["rng_exact"] for row in report["checks"])
                graph_checks += len(report["checks"])
                for row in report["batch32_noise_audit"]:
                    assert row["private_rng_exact"] and row["batch"]==32
                    noise_checks += row["checked_requests"]
                    native_audits.append(dict(row, receipt=str(path)))
    recommended = lookup[(100,1,8)]
    summary = dict(status="PARTIAL", measured_configurations=len(rows), requested_configurations=36, missing=missing,
        aggregate_GPU_seconds=seconds, budget_seconds=1200, GPU_lock="/tmp/r17_gpu.lock",
        GPU="GPU-98d36ed2-a8e9-fb07-701c-41f337a6c1f9", phase_receipts=budgets, tests=tests,
        B1_graph_exact_comparisons=graph_checks, B32_exact_private_rng_requests=noise_checks, native_B32_audits=native_audits,
        emitted_requests_with_latency=sum(r["latency_samples"] for r in rows), measured_completions=sum(r["calls"] for r in rows),
        quality_sr_claim=False, max_server_committed_action_diff=max(r["max_diff"] for r in rows),
        recommended=dict(replicas=1,max_batch=8,batch_wait_ms=5,workers=12,worker_memory_mib=4096,
            actual_max_stage3_batch=recommended["max_stage3_batch"], mean_stage3_batch=recommended["mean_stage3_batch"],
            measured= recommended, latency_speedup_vs_1409_ms=1409/recommended["p50_ms"],
            inferred_old_capacity_calls_s=12/1.409, provisional_capacity_speedup=recommended["calls_per_second"]/(12/1.409)),
        sealed_model=seal, extra_runtime_sources={str(p):file_digest(p) for p in [Path("exp/mw_finetune/dp/cuda_graph.py"),
            ROOT/"formal_eval.sh",ROOT/"formal_python.sh",samples/"dynamic_batch_measure.py",samples/"dynamic_batch_persistent.py",
            samples/"node_ram_preflight.py",samples/"cap32_completion.py",
            Path("exp/r17_integration/ops/formal_eval.py"),Path("exp/r17_integration/ops/formal_eval.sh"),
            Path("exp/r17_integration/ops/formal_server.py")]})
    (samples / "dynamic_batch_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (samples / "dynamic_batch_results.json").write_text(json.dumps([lookup.get(key,dict(K=key[0],replicas=key[1],max_batch=key[2],status="unmeasured")) for key in keys],indent=2)+"\n")
    fields = ["K","replicas","max_batch","status","calls_per_second","actions_per_second","p50_ms","p90_ms","watts","max_diff","mean_stage3_batch","max_stage3_batch","measurement_seconds","latency_samples","receipt"]
    with (samples / "dynamic_batch_results.csv").open("w") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        for key in keys:
            writer.writerow(lookup.get(key,dict(K=key[0],replicas=key[1],max_batch=key[2],status="unmeasured")))
    print(json.dumps({key:summary[key] for key in ("status","measured_configurations","aggregate_GPU_seconds","B1_graph_exact_comparisons","B32_exact_private_rng_requests","max_server_committed_action_diff")}))


if __name__=="__main__":
    main()
