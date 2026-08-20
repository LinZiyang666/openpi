"""Render the latency matrix from the raw result JSONs in ``../data``.

Every number in ``analysis.md`` comes from here, so the report can be
regenerated from the artifacts rather than transcribed by hand.

Usage:
    uv run exp/ablation_study/latency_bench/analysis/summarize.py
"""

from __future__ import annotations

import json
import pathlib

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"


def load(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text())


def _fmt(x: float | None, unit: str = "") -> str:
    return "—" if x is None else f"{x:,.2f}{unit}"


def main() -> None:
    # ---- students: bench_students.py (eager) / bench_students_compile.py ----
    rows = [
        ("ACT", "51.6M", "RTX 4090", "eager", load("act_cuda_4090")["call_ms"]["median"],
         load("act_cuda_4090")["call_ms"]["p90"], None),
        ("ACT", "51.6M", "RTX 4090", "compile default",
         load("act_cuda_compile_default")["median_ms"], load("act_cuda_compile_default")["p90_ms"],
         load("act_cuda_compile_default")["gpu_util_median_pct"]),
        ("ACT", "51.6M", "RTX 4090", "compile CUDA-Graph",
         load("act_cuda_compile_ro")["median_ms"], load("act_cuda_compile_ro")["p90_ms"],
         load("act_cuda_compile_ro")["gpu_util_median_pct"]),
        ("ACT", "51.6M", "Xeon CPU", "eager", load("act_cpu_xeon")["call_ms"]["median"],
         load("act_cpu_xeon")["call_ms"]["p90"], None),
        ("SmolVLA", "450M", "RTX 4090", "eager", load("smolvla_cuda_4090")["call_ms"]["median"],
         load("smolvla_cuda_4090")["call_ms"]["p90"], None),
        ("SmolVLA", "450M", "RTX 4090", "compile default",
         load("smolvla_cuda_compile_default")["median_ms"], load("smolvla_cuda_compile_default")["p90_ms"],
         load("smolvla_cuda_compile_default")["gpu_util_median_pct"]),
        ("SmolVLA", "450M", "RTX 4090", "compile CUDA-Graph",
         load("smolvla_cuda_compile_ro_markstep")["median_ms"],
         load("smolvla_cuda_compile_ro_markstep")["p90_ms"],
         load("smolvla_cuda_compile_ro_markstep")["gpu_util_median_pct"]),
        ("SmolVLA", "450M", "Xeon CPU", "eager", load("smolvla_cpu_xeon")["call_ms"]["median"],
         load("smolvla_cpu_xeon")["call_ms"]["p90"], None),
    ]
    print("## Students (policy_fn call = obs->batch->predict_action_chunk->cpu)\n")
    print("| model | params | device | build | median ms | p90 ms | GPU util |")
    print("|---|---|---|---|---|---|---|")
    for name, par, dev, build, med, p90, util in rows:
        print(f"| {name} | {par} | {dev} | {build} | {med:,.2f} | {p90:,.2f} | "
              f"{'—' if util is None else f'{util:.0f}%'} |")

    # ---- teacher: bench_teacher.py (staged) + microbench_cost (cuda events) ----
    print("\n## Teacher pi0.5 (staged path; wall includes input/output transforms)\n")
    print("| build | wall ms | model total ms | Stage1 | Stage2 | Stage3 | GPU util |")
    print("|---|---|---|---|---|---|---|")
    for key, label in [("pi05_eager", "eager (3-stage)"),
                       ("pi05_compile_default", "compile default (3-stage)"),
                       ("pi05_compile_ro_3stage", "compile CUDA-Graph (3-stage)"),
                       ("pi05_fused_clean_default", "compile default (fused)"),
                       ("pi05_fused_clean_ro", "compile CUDA-Graph (fused)"),
                       ("pi05_eager_cpu", "eager, CPU (3-stage)")]:
        d = load(key)
        fused = d.get("fused", False)
        s1, s2, s3 = d["stage1_token_prep_ms"], d["stage2_llm_backbone_ms"], d["stage3_action_expert_ms"]
        # Fused + CUDA Graph: policy.infer's internal total_ms measures graph
        # SUBMISSION only (kernels still async) — the synchronized wall is the
        # only valid figure there.
        total = "n/a (async)" if fused and d["compile_mode"] == "reduce-overhead" else f"{d['model_total_ms']:,.2f}"
        stages = "— | — | —" if fused else f"{s1:,.2f} | {s2:,.2f} | {s3:,.2f}"
        util = d.get("gpu_util_median_pct")
        print(f"| {label} | {d['wall_ms']:,.2f} | {total} | {stages} | "
              f"{'—' if util is None else f'{util:.0f}%'} |")

    # ---- fused vs 3-stage, same compile mode on both sides -----------------
    print("\n## Fused (one graph) vs 3-stage (cache split), identical compile modes\n")
    print("| compile mode | 3-stage wall | fused wall | fused advantage |")
    print("|---|---|---|---|")
    for st_key, fu_key, label in [
        ("pi05_compile_default", "pi05_fused_clean_default", "default"),
        ("pi05_compile_ro_3stage", "pi05_fused_clean_ro", "CUDA-Graph"),
    ]:
        st, fu = load(st_key)["wall_ms"], load(fu_key)["wall_ms"]
        print(f"| {label} | {st:,.2f} ms | {fu:,.2f} ms | {st-fu:+,.2f} ms ({100*(st-fu)/st:.1f}%) |")

    ev = load("teacher_pi05_4090")["gpu"]
    print(f"\nCUDA-event cross-check (microbench_cost.py, eager): median "
          f"{ev['median_s']*1000:,.2f} ms, p90 {ev['p90_s']*1000:,.2f} ms, n={ev['n']}")

    # ---- derived: what a cache-HIT step costs vs a full teacher step ----
    print("\n## Derived: routed step (Stage1 + student) vs full teacher step\n")
    print("| build | Stage1 | +ACT | +SmolVLA | teacher step | ACT saving | SmolVLA saving |")
    print("|---|---|---|---|---|---|---|")
    for key, act_key, smol_key, label in [
        ("pi05_eager", "act_cuda_4090", "smolvla_cuda_4090", "eager"),
        ("pi05_compile_default", "act_cuda_compile_default", "smolvla_cuda_compile_default", "compile default"),
        ("pi05_compile_ro_3stage", "act_cuda_compile_ro", "smolvla_cuda_compile_ro_markstep", "compile CUDA-Graph"),
    ]:
        t = load(key)
        s1, full = t["stage1_token_prep_ms"], t["model_total_ms"]
        a = load(act_key)
        s = load(smol_key)
        act = a["call_ms"]["median"] if "call_ms" in a else a["median_ms"]
        smol = s["call_ms"]["median"] if "call_ms" in s else s["median_ms"]
        print(f"| {label} | {s1:,.2f} | {s1+act:,.2f} | {s1+smol:,.2f} | {full:,.2f} | "
              f"{100*(1-(s1+act)/full):+.0f}% | {100*(1-(s1+smol)/full):+.0f}% |")


if __name__ == "__main__":
    main()
