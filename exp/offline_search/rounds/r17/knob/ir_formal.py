"""Formal IR of the per-task knob TEST cells from the batch-1 latency profile (CPU only).

Every decision runs stage1 (the retrieval key); a model call adds stage2 + denoising. With the
full-step pure-inference decision as the unit (= 1):

    IR = (v * (policy + library) + m * policy) / (controls / C0)

m is the full-step miss coefficient for full-step knobs and the reduced-step one for reduced knobs.
Text encoding (owner rule 10-12): the benchmark prompts are fixed, so the cache system never pays an
encoder that only serves the text modality (numerator); the pure-inference denominator counts it only
when it runs on the GPU. DP's CLIP text encoder runs on the CPU: neither side counts it. Fast-WAM's T5
runs on the GPU inside stage2: the denominator counts it, the numerator uses the cached-text stage2.
Cosmos does not re-encode text per request.

MT50 uses results_mt50's supplied stage costs for all four models. DP's CPU
text time is excluded on both sides. MT50 Fast-WAM retains measured stage2;
the LIBERO-only cached-text stage2 profile is not substituted for MT50.

usage: ir_formal.py --test FULL_OR_REDUCED:path/test.json ... --out DIR
"""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from exp.offline_search.rounds.r17.knob import common as c

PROFILE = Path(__file__).resolve().parents[5] / "exp/r17_integration/profile"
MODELS = {"DP": "dp_libero", "Fast-WAM": "fastwam_libero", "Cosmos": "cosmos_libero"}
SUITES = {"libero_10": "长任务", "libero_spatial": "空间任务"}


def coefficients(model, suite):
    """(system, pure): per-decision costs in units of the full-step pure-inference decision."""
    bundle = c.MODEL_IDS[model]+"_mw" if suite == "mt50" else MODELS[model]
    directory = "results_mt50" if suite == "mt50" else "results"
    r = json.loads((PROFILE / directory / f"{bundle}_{suite}.json").read_text())
    stage1, stage2 = r["stage1_ms"], r["stage2_ms"]
    if model == "DP":  # CPU CLIP text: counted on neither side
        stage1 -= r["dp_cpu_text_encoder_ms"]["median"]
    system_stage2 = stage2
    if model == "Fast-WAM" and suite != "mt50":  # GPU T5 in stage2: pure inference pays it, the cache system does not
        system_stage2 = json.loads((PROFILE / "results_cached_text" / f"{MODELS[model]}_{suite}.json").read_text())["stage2_ms"]
    unit = stage1 + stage2 + r["stage3_full_loop_ms"]
    def coef(s2):
        return dict(vision=stage1 / unit, full_miss=(s2 + r["stage3_full_loop_ms"]) / unit,
                    reduced_miss=(s2 + r["stage3_reduced_loop_ms"]) / unit)
    return coef(system_stage2), coef(stage2)


def ir(cost, coef, miss_key, c0):
    decisions = cost["policy"] + cost["library"]
    return (coef["vision"] * decisions + coef[miss_key] * cost["policy"]) * c0 / cost["controls"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="append", required=True, help="full:path or reduced:path")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rows = []
    for item in a.test:
        steps, path = item.split(":", 1)
        miss_key = "reduced_miss" if steps == "reduced" else "full_miss"
        for cell in json.loads(Path(path).read_text())["cells"]:
            model, suite = cell["model"], cell["suite"]
            system, pure = coefficients(model, suite)
            c0 = c.C0[model]
            rows.append(dict(model=model, suite=suite, size=cell["size"], steps=steps, sr=cell["sr"],
                             control_sr=cell["control"]["sr"],
                             group_mean=cell.get("group_mean"), task_mean=cell.get("task_mean"),
                             control_group_mean=cell["control"].get("group_mean"),
                             control_task_mean=cell["control"].get("task_mean"), ir_proxy=cell["ir_proxy"],
                             vision_ratio=(cell["policy"] + cell["library"]) * c0 / cell["controls"] if cell["controls"] else None,
                             ir=ir(cell, system, miss_key, c0) if cell["cost_n"] == cell["n"] and cell["controls"] else None,
                             pure_ir=pure["vision"] + pure[miss_key]))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "ir_formal.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    lines = ["# 逐任务旋钮的正式 IR（按 batch=1 延迟实测）", "",
             "IR = [视觉系数 ×（调用次数 + 命中次数）+ 调用系数 × 调用次数] ÷（控制步数 ÷ C0），"
             "以全步数纯推理的一次决策为 1。减步版用减步调用系数。"
             "文本编码：缓存系统不计只服务文本模态的编码（提示词固定）；纯推理里在 GPU 上的计入、在 CPU 上的不计。"
             "LIBERO Fast-WAM 的 GPU T5 只计入纯推理；MT50 使用提供的实测 stage2，不替换成 LIBERO 的文本缓存成本。"
             "MT50 成功率先列四难度组等权均值，再列逐任务均值。纯推理一列是同步数纯推理的 IR。", "",
             "| 模型 | 步数 | 任务套件 | 库集数 | 成功率 | 纯推理成功率 | 调用比例（旧代理） | 视觉比例 | IR | 纯推理 IR |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        sr = (c.percent(r['group_mean'])+' / '+c.percent(r['task_mean'])) if r['suite']=='mt50' else c.percent(r['sr'])
        control_sr = (c.percent(r['control_group_mean'])+' / '+c.percent(r['control_task_mean'])) if r['suite']=='mt50' else c.percent(r['control_sr'])
        lines.append(f"| {r['model']} | {'减步' if r['steps'] == 'reduced' else '全步数'} | {c.SUITE_LABELS[r['suite']]} | {r['size']} | "
                     f"{sr} | {control_sr} | {c.number(r['ir_proxy'])} | {c.number(r['vision_ratio'])} | "
                     f"{c.number(r['ir'])} | {r['pure_ir']:.3f} |")
    (a.out / "IR_FORMAL.md").write_text("\n".join(lines) + "\n")
    print(len(rows), "cells")


if __name__ == "__main__":
    main()
