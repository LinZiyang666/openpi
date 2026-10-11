"""Select per-task R17 knobs on B; read-only input lanes, CPU only."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from exp.offline_search.rounds.r17.knob import common as c


def descend(levels, tolerance):
    first = next((level for level in c.LEVELS[1:]
                  if levels[level]["passes"][str(tolerance)] is False), None)
    incomplete = any(levels[level]["n"] != 30 for level in c.LEVELS[1:])
    recoveries = [level for level in c.LEVELS[c.LEVELS.index(first)+1:]
                  if levels[level]["passes"][str(tolerance)] is True] if first else []
    # Report all available outcomes; incomplete tasks have no frozen choice.
    selected = None if incomplete else c.LEVELS[c.LEVELS.index(first)-1] if first else "koff"
    return dict(selected=selected, first_failure=first, non_monotone_recoveries=recoveries,
                incomplete=incomplete)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, default=c.LIBRARY_ROOT)
    args = parser.parse_args(argv)
    cells, planned = c.load_knob(args.lane, "B")
    references = {(model, suite): c.teacher_reference(args.library_root, model, suite)
                  for model, suite in sorted({(m, s) for m, s, _ in cells})}
    tables = {tol: {} for tol in (0, 2, 5)}
    result = dict(primary_tolerance_pp=5, cells=[], references=[])
    lines = ["# 逐任务旋钮下降", "", "开发集使用B 池每任务起点0–29，纯推理参考来自教师库。主容差为5 个点，另给0 和 2 个点表。",
             "首次失败即选择上一档；低档恢复只记录，不改变选择。缺局的任务标为不完整，不写入冻结表，也不补齐结果。",
             "推理开销代理为模型调用数除以执行控制数折算的决策数；DP 和 Fast-WAM 每 5 步折算一次，Cosmos 每 16 步折算一次。纯推理参考按一计；拼接开销由所选轨迹的总调用数和总执行控制数计算。",
             "缺少完整执行计数时开销记为暂无。开发集拼接结果用于选择，尚未独立测试。", "",
             "档位数字是标定的目标 miss 比例，表中开销是实测值。", ""]
    for (model, suite), ref in references.items():
        result["references"].append(dict(model=model, suite=suite,
            inits_0_29=c.stats({k: e for k, e in ref.items() if k[2] < 30}, model),
            inits_0_49=c.stats(ref, model)))
    totals = []
    for cell, arms in sorted(cells.items()):
        model, suite, size = cell
        ref = {k: e for k, e in references[model, suite].items() if k[2] < 30}
        names = sorted({k[1] for k in ref})
        if set(planned[cell]) - set(names):
            raise ValueError(f"{cell}: lane task names absent from teacher reference")
        tasks, selected_rows, selected_ref = {}, {}, {}
        model_label = c.MODEL_LABELS[model]
        lines += [f"## {model_label} · {c.SUITE_LABELS[suite]} · 库 {size} 集", "",
                  "| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |",
                  "|---|---|---|---|---|"]
        for task in names:
            pure = {k: e for k, e in ref.items() if k[1] == task}
            pure_stats = c.stats(pure, model)
            levels = {"pure": dict(pure_stats, missing_inits=[], passes={str(t): True for t in tables})}
            for level in c.LEVELS[1:]:
                data = {k: e for k, e in arms.get(level, {}).items() if k in pure}
                st = c.stats(data, model)
                paired_pure = sum(pure[k]["success"] for k in data)
                gains = sum(e["success"] and not pure[k]["success"] for k, e in data.items())
                losses = sum(pure[k]["success"] and not e["success"] for k, e in data.items())
                levels[level] = dict(st, paired_pure_successes=paired_pure, gains=gains, losses=losses,
                    missing_inits=sorted(k[2] for k in pure if k not in data),
                    passes={str(t): 100*(st["successes"]-pure_stats["successes"]) >= -t*30
                            if st["n"] == 30 else None for t in tables})
            decisions = {str(t): descend(levels, t) for t in tables}
            tasks[task] = dict(levels=levels, tolerances=decisions, **decisions["5"])
            for tol in tables:
                if decisions[str(tol)]["selected"] is not None:
                    c.nested_set(tables[tol], cell, task, decisions[str(tol)]["selected"])
            choice = decisions["5"]
            level = choice["selected"]
            if level is not None:
                selected_rows.update(pure if level == "pure" else {k: arms[level][k] for k in pure})
                selected_ref.update(pure)
            lines.append(f'| {task} | {c.LABELS.get(level, "未冻结")} | {c.LABELS.get(choice["first_failure"], "无")} | '
                         f'{"、".join(c.LABELS[x] for x in choice["non_monotone_recoveries"]) or "无"} | '
                         f'{"不完整" if choice["incomplete"] else "完整"} |')
        stitched = c.stats(selected_rows, model)
        baseline = c.stats(selected_ref, model)
        row = dict(model=model, suite=suite, size=size, tasks=tasks, selected=stitched,
                   paired_pure=baseline, pure=c.stats(ref, model), complete_tasks=stitched["n"]//30)
        result["cells"].append(row)
        totals.append((model_label, suite, size, row))
        lines += ["", f'已冻结 {row["complete_tasks"]}/10 个任务；拼接 {stitched["n"]} 局，成功率 {c.percent(stitched["sr"])}，'
                  f'推理开销代理 {c.number(stitched["ir_proxy"])}；同任务纯推理成功率 {c.percent(baseline["sr"])}，开销代理 1.000。'
                  f'全十任务纯推理成功率 {c.percent(row["pure"]["sr"])}。', "",
                  "| 任务 | 档位 | 已有局数 | 成功局数 | 同起点纯推理成功局数 | 推理开销代理 | 五点容差结果 |",
                  "|---|---|---:|---:|---:|---:|---|"]
        for task, task_result in tasks.items():
            for level, st in task_result["levels"].items():
                passed = st["passes"]["5"]
                lines.append(f'| {task} | {c.LABELS[level]} | {st["n"]} | {st["successes"]} | '
                             f'{st.get("paired_pure_successes", st["successes"])} | {c.number(st["ir_proxy"])} | '
                             f'{"通过" if passed is True else "失败" if passed is False else "不完整"} |')
        lines.append("")
    lines += ["## 汇总", "", "| 模型 | 任务套件 | 库集数 | 已冻结任务数 | 拼接局数 | 拼接成功率 | 推理开销代理 | 同任务纯推理成功率 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for label, suite, size, row in totals:
        st = row["selected"]
        lines.append(f'| {label} | {c.SUITE_LABELS[suite]} | {size} | {row["complete_tasks"]} | {st["n"]} | '
                     f'{c.percent(st["sr"])} | {c.number(st["ir_proxy"])} | {c.percent(row["paired_pure"]["sr"])} |')
    args.out.mkdir(parents=True, exist_ok=True)
    for tol, table in tables.items():
        c.write_json(args.out / ("levels.json" if tol == 5 else f"levels_tol{tol}pp.json"), table)
    c.write_json(args.out / "descent.json", result)
    (args.out / "DESCENT.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote descent tables and report to {args.out}")
    return result


if __name__ == "__main__":
    main()
