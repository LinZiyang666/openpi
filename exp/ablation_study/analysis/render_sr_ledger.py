"""Render the ablation success-rate ledger (analysis/sr_ledger.md).

Owner directive: the ledger records ONLY full-scale EN-2 evaluation numbers —
every candidate cell measured on the official pruned_init set (n=50/task) and
the frozen selections derived from them. No legacy n=5 data, no collection
stats.

Inputs: the timan107 collector snapshot (collect_sr.py -> sr_snapshot.json)
plus select_freeze_*.yaml files. Regenerate after each selection batch.

Usage:
    python exp/ablation_study/analysis/render_sr_ledger.py \
        --snapshot <sr_snapshot.json> --config-dir exp/ablation_study/config \
        --out exp/ablation_study/analysis/sr_ledger.md
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib

import yaml

STEPS = ["002000", "004000", "006000", "008000", "010000", "012000",
         "014000", "016000", "018000", "020000"]
ANCHORS = {"libero_spatial": 0.95, "libero_10": 0.83}


def fmt(v) -> str:
    return "—" if v is None else f"{v['sr']:.2f}"


def act_matrix(snap: dict, suite: str) -> list[str]:
    rows = [f"### {suite} — ACT 候选 SR（n=50/格）\n",
            "| task | " + " | ".join(s.lstrip("0") or "0" for s in STEPS) + " |",
            "|---|" + "---|" * len(STEPS)]
    tasks = snap.get("en2_act_candidates", {}).get(suite, {})
    for t in [f"task_{i}" for i in range(10)]:
        cells = tasks.get(t, {})
        rows.append(f"| {t} | " + " | ".join(fmt(cells.get(s)) for s in STEPS) + " |")
    return rows


def selections(config_dir: str) -> list[str]:
    """EN-3 rev (owner ruling 2026-08-13): freezes are suite-uniform steps at
    the standard-recipe endpoint; the band is a disclosure on the suite
    aggregate, not a selector."""
    rows = ["## 冻结选择（EN-3：套件级统一 step=标准配方终点；band 为披露性字段）\n",
            "| suite | student | uniform step | 聚合 SR (n=500) | band | 带内 |",
            "|---|---|---|---|---|---|"]
    for f in sorted(glob.glob(f"{config_dir}/select_freeze_*.yaml")):
        doc = yaml.safe_load(pathlib.Path(f).read_text())
        sel = doc.get("selection") or {}
        if sel.get("mode") != "suite_uniform_step":
            continue
        student = doc.get("student", "?")
        if student == "act_per_task" and doc.get("task") != "task_0":
            continue  # aggregate row rendered once via task_0; per-task files share it
        band = sel["admission_band"]
        rows.append(
            f"| {doc.get('suite','?')} | {student} | {sel['uniform_step']} | "
            f"{sel['suite_aggregate_succ']}/{sel['suite_aggregate_n']}={sel['suite_aggregate_sr']:.3f} | "
            f"[{band[0]:.3f},{band[1]:.3f}] | {'✓' if sel['suite_admitted'] else '✗（披露）'} |")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--config-dir", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    snap = json.load(open(a.snapshot))

    lines = [
        "# Ablation Study — 全量评估成功率账本（sr_ledger）\n",
        "> 仅记录 EN-2 全量评估数据：每个学生候选在**官方 pruned_init 测试集**（LIBERO benchmark",
        "> 默认 init，50/任务；与训练差集池逐字节零重叠）上的 SR，及由此冻结的选择。",
        "> 由 `render_sr_ledger.py` 从 timan107 `collect_sr.py` 快照再生成，随评估推进持续更新。",
        "> EN-3（owner 裁决 2026-08-13）：冻结=套件级统一 step **020000**（标准配方终点：SmolVLA 官方",
        "> 微调预算 20k；ACT 处 SR 平台），四组合全表同 step，零选择偏差；band 降级为聚合披露字段。",
        f"> Anchors（teacher 测试集协议锚）：libero_spatial={ANCHORS['libero_spatial']}，libero_10={ANCHORS['libero_10']}\n",
    ]
    lines += selections(a.config_dir)
    lines.append("")
    for suite in ("libero_spatial", "libero_10"):
        lines += act_matrix(snap, suite)
        lines.append("")
    sm = snap.get("en2_smolvla_candidates", {})
    lines.append("## SmolVLA 候选 SR（粗筛 {4k,8k,12k,16k,20k}）\n")
    if not any(sm.values() if isinstance(sm, dict) else []):
        lines.append("（评估 job 挂起中，ACT drain 后释放）")
    else:
        for suite, steps in sm.items():
            lines.append(f"### {suite}")
            lines.append("| step | " + " | ".join(f"t{i}" for i in range(10)) + " | 聚合 |")
            lines.append("|---|" + "---|" * 11)
            for step, tasks in sorted(steps.items()):
                srs = [tasks.get(f"task_{i}") for i in range(10)]
                tot_s = sum(v["succ"] for v in srs if v)
                tot_n = sum(v["n"] for v in srs if v)
                agg = f"{tot_s}/{tot_n}={tot_s/tot_n:.3f}" if tot_n else "—"
                lines.append(f"| {step} | " + " | ".join(fmt(v) for v in srs) + f" | {agg} |")
    pathlib.Path(a.out).write_text("\n".join(lines) + "\n")
    print("ledger written:", a.out)


if __name__ == "__main__":
    main()
