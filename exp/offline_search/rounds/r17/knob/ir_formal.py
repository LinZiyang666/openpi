"""Formal IR of the per-task knob TEST cells from the batch-1 latency profile (CPU only).

Every decision runs stage1 (the retrieval key); a model call adds stage2 + denoising. With the
full-step pure-inference decision as the unit (vision + full miss = 1):

    IR = (v * (policy + library) + m * policy) / (controls / C0)

m is the full-step miss coefficient for full-step knobs and the reduced-step one for reduced
knobs. Two coefficient sets: as served (DP re-encodes CLIP text on the CPU, Fast-WAM runs T5 in
stage2 on every request) and with the text encoded once per episode.

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
    receipt = json.loads((PROFILE / "results" / f"{MODELS[model]}_{suite}.json").read_text())
    served = receipt["coefficients"]
    once = dict(served)
    if model == "DP":
        stage1 = receipt["stage1_ms"] - receipt["dp_cpu_text_encoder_ms"]["median"]
        total = stage1 + receipt["stage2_ms"] + receipt["stage3_full_loop_ms"]
        once = dict(vision=stage1 / total, full_miss=1 - stage1 / total,
                    reduced_miss=(receipt["stage2_ms"] + receipt["stage3_reduced_loop_ms"]) / total)
    elif model == "Fast-WAM":
        once = json.loads((PROFILE / "results_cached_text" / f"{MODELS[model]}_{suite}.json").read_text())["coefficients"]
    return served, once


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
            served, once = coefficients(model, suite)
            c0 = c.C0[model]
            rows.append(dict(model=model, suite=suite, size=cell["size"], steps=steps, sr=cell["sr"],
                             control_sr=cell["control"]["sr"], ir_proxy=cell["ir_proxy"],
                             vision_ratio=(cell["policy"] + cell["library"]) * c0 / cell["controls"],
                             ir_served=ir(cell, served, miss_key, c0), ir_text_once=ir(cell, once, miss_key, c0),
                             pure_served=served["vision"] + served[miss_key],
                             pure_text_once=once["vision"] + once[miss_key]))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "ir_formal.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    lines = ["# 逐任务旋钮的正式 IR（按 batch=1 延迟实测）", "",
             "IR = [视觉系数 ×（调用次数 + 命中次数）+ 调用系数 × 调用次数] ÷（控制步数 ÷ C0），"
             "以全步数纯推理的一次决策为 1。减步版用减步调用系数。"
             "「服务端现状」按每次请求都重算文本编码计（DP 的 CPU CLIP、Fast-WAM 的 T5）；"
             "「文本每局一次」假定提示词编码每局只算一次。纯推理一列是同一口径下纯推理（减步版为减步纯推理）的 IR。", "",
             "| 模型 | 步数 | 任务套件 | 库集数 | 成功率 | 纯推理成功率 | 调用比例（旧代理） | 视觉比例 | IR（服务端现状） | 纯推理 IR（服务端现状） | IR（文本每局一次） | 纯推理 IR（文本每局一次） |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['model']} | {'减步' if r['steps'] == 'reduced' else '全步数'} | {SUITES[r['suite']]} | {r['size']} | "
                     f"{100*r['sr']:.1f}% | {100*r['control_sr']:.1f}% | {r['ir_proxy']:.3f} | {r['vision_ratio']:.3f} | "
                     f"{r['ir_served']:.3f} | {r['pure_served']:.3f} | {r['ir_text_once']:.3f} | {r['pure_text_once']:.3f} |")
    (a.out / "IR_FORMAL.md").write_text("\n".join(lines) + "\n")
    print(len(rows), "cells")


if __name__ == "__main__":
    main()
