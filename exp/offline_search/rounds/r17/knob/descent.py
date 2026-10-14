"""Select R17 knobs on LIBERO B or directly on MT50 seed-7 TEST; CPU only."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from exp.offline_search.rounds.r17.knob import common as c


def descend(levels, tolerance, count=30):
    first = next((level for level in c.LEVELS[1:]
                  if levels[level]["passes"][str(tolerance)] is False), None)
    # Once a complete level fails, no lower level is needed to freeze the choice.
    visited = c.LEVELS[:c.LEVELS.index(first)+1] if first else c.LEVELS
    incomplete = any(levels[level]["n"] != count for level in visited)
    recoveries = [level for level in c.LEVELS[c.LEVELS.index(first)+1:]
                  if levels[level]["passes"][str(tolerance)] is True] if first else []
    # Report all available outcomes; incomplete tasks have no frozen choice.
    selected = None if incomplete else c.LEVELS[c.LEVELS.index(first)-1] if first else "koff"
    return dict(selected=selected, first_failure=first, non_monotone_recoveries=recoveries,
                incomplete=incomplete)


def analyze_mt50(cells, references, output, miss_steps, control_overrides=()):
    """Freeze measured seed-7 levels directly, with a two-success tolerance."""
    from exp.offline_search.rounds.r17.knob.ir_formal import coefficients, ir
    tables = {}
    control_names = dict(c.MW_REDUCED_CONTROL_NAMES if miss_steps == "reduced" else c.MW_CONTROL_NAMES)
    control_names.update({c.model_label(m):(run,) for m,run in (item.split("=",1) for item in control_overrides)})
    result = dict(benchmark="mt50", seed=7, inits=50, miss_steps=miss_steps,
                  tolerance_successes=2, controls=control_names, cells=[])
    lines = ["# MT50 per-task knob descent", "",
             "Seed 7, idx 0–49. The same-step pure baseline supplies the pure level. "
             "A level passes with at most two fewer successes of 50. The first failure "
             "freezes the previous level; its measured episodes are the final result.", "",
             "SR is the equal-weight mean of the four difficulty groups first, then the per-task mean. "
             "A complete koff cell reaching the pure group-mean SR is used directly.", "",
             "| Model | Size | Frozen tasks | Group mean | Task mean | Pure group mean | Pure task mean | Formal IR | No knob |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    details = []
    for cell, arms in sorted(cells.items()):
        model, suite, size = cell
        ref = references.get((model, suite), {})
        names = sorted({k[1] for k in ref} | {k[1] for data in arms.values() for k in data})
        expected = {(suite, task, i) for task in names for i in range(50)}
        off = arms.get("koff", {})
        no_knob = (len(names) == 50 and set(off) == set(ref) == expected and
                   c.stats(off, model)["group_mean"] >= c.stats(ref, model)["group_mean"])
        tasks, selected, selected_ref = {}, {}, {}
        details += [f"## {model} · size {size}", "",
                    "| Task | Frozen level | First failure | Successes | Pure successes | Complete |",
                    "|---|---|---|---:|---:|---|"]
        for task in names:
            pure = {k:e for k,e in ref.items() if k[1] == task and 0 <= k[2] < 50}
            baseline = c.stats(pure, model)
            levels = {"pure": dict(baseline, passes={"4": True if baseline["n"] == 50 else None})}
            for level in c.LEVELS[1:]:
                data = {k:e for k,e in arms.get(level, {}).items() if k in pure}
                st = c.stats(data, model)
                levels[level] = dict(st, passes={"4": baseline["successes"]-st["successes"] <= 2
                                    if st["n"] == baseline["n"] == 50 else None},
                                    missing_inits=sorted(set(range(50))-{k[2] for k in data}))
            decision = (dict(selected="koff", first_failure=None, non_monotone_recoveries=[], incomplete=False)
                        if no_knob else descend(levels, 4, count=50))
            level = decision["selected"]
            tasks[task] = dict(levels=levels, **decision)
            if level is not None:
                data = pure if level == "pure" else {k:arms[level][k] for k in pure}
                selected.update(data)
                selected_ref.update(pure)
                c.nested_set(tables, cell, task, level)
            details.append(f'| {task} | {level or "pending"} | {decision["first_failure"] or "none"} | '
                           f'{levels[level]["successes"] if level else "pending"} | {baseline["successes"]} | '
                           f'{not decision["incomplete"]} |')
        st = c.stats(selected, model)
        baseline = c.stats(selected_ref, model)
        system, pure_cost = coefficients(model, suite)
        miss_key = "reduced_miss" if miss_steps == "reduced" else "full_miss"
        formal_ir = ir(st, system, miss_key, c.C0[model]) if st["n"] and st["cost_n"] == st["n"] and st["controls"] else None
        row = dict(model=model, suite=suite, size=size, **st, control=baseline,
                   tasks=tasks, no_knob=no_knob, complete_tasks=st["n"]//50,
                   incomplete=st["n"] != 2500, ir=formal_ir,
                   pure_ir=pure_cost["vision"]+pure_cost[miss_key])
        result["cells"].append(row)
        lines.append(f'| {model} | {size} | {row["complete_tasks"]}/50 | {c.percent(st.get("group_mean"))} | '
                     f'{c.percent(st.get("task_mean"))} | {c.percent(baseline.get("group_mean"))} | '
                     f'{c.percent(baseline.get("task_mean"))} | {c.number(formal_ir)} | {no_knob} |')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    c.write_json(output/"levels.json", tables)
    c.write_json(output/"descent.json", result)
    # The TEST entry point and formal-IR tool read this same final measurement.
    c.write_json(output/"test.json", result)
    (output/"DESCENT.md").write_text("\n".join(lines+[""]+details)+"\n")
    print(f"Wrote final MT50 measurements and levels to {output}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, default=c.LIBRARY_ROOT)
    parser.add_argument("--miss-steps", choices=("full", "reduced"), default="full")
    parser.add_argument("--control-root", type=Path, default=c.CONTROL_ROOT)
    parser.add_argument("--control", action="append", default=[], help="MODEL=RUN: same-step seed-7 pure baseline")
    parser.add_argument("--reference", choices=("teacher", "pure-arm"), default=None)
    parser.add_argument("--reference-lane",type=Path,action="append",help="Standalone same-step pure B baseline; repeat to combine models")
    args = parser.parse_args(argv)
    mt50 = any(c.campaign_bundle(c.cache.Sources().obj(lane/"campaign.json")) in c.study.MW_MODELS for lane in args.lane)
    cells, planned = c.load_knob(args.lane, "A" if mt50 else "B")
    if mt50:
        from exp.offline_search.rounds.r17.knob.test import load_controls
        references = load_controls(args.control_root, {m for m,_,_ in cells}, args.miss_steps, args.control, suites=("mt50",))
        return analyze_mt50(cells, references, args.out, args.miss_steps, args.control)
    args.reference = args.reference or ('pure-arm' if args.reference_lane or args.miss_steps == 'reduced' or any(m in c.VLA_MODELS and s != 'mt50' for m,s,_ in cells) else 'teacher')
    references = c.pure_references(args.reference_lane) if args.reference_lane else {(model, suite): (cells.get((model, suite, 500), {}).get("pure", {})
                  if args.reference == "pure-arm" else c.teacher_reference(args.library_root, model, suite))
                  for model, suite in sorted({(m, s) for m, s, _ in cells})}
    tables = {tol: {} for tol in (0, 2, 5)}
    result = dict(primary_tolerance_pp=5, cells=[], references=[])
    lines = ["# 逐任务旋钮下降", "", "开发集使用B 池每任务起点0–29，纯推理参考来自教师库。主容差为5 个点，另给0 和 2 个点表。",
             "首次失败即选择上一档；低档恢复只记录，不改变选择。缺局的任务标为不完整，不写入冻结表，也不补齐结果。",
             "推理开销代理为模型调用数除以执行控制数折算的决策数；DP、Fast-WAM、π0.5 和 GR00T 每 5 步折算一次，Cosmos 每 16 步折算一次。纯推理参考按一计；拼接开销由所选轨迹的总调用数和总执行控制数计算。",
             "缺少完整执行计数时开销记为暂无。开发集拼接结果用于选择，尚未独立测试。", "",
             "档位数字是标定的目标 miss 比例，表中开销是实测值。", ""]
    if args.reference == "pure-arm" or args.reference_lane:
        result["reference"] = "pure-lane" if args.reference_lane else "pure-arm"
        lines[2] = "开发集使用B 池每任务起点0–29，参考为同起点同一步数纯推理；主容差为5 个点，首次失败即冻结上一档。"
        if args.reference_lane:result['reference_lanes'] = list(map(str,args.reference_lane))
    for (model, suite), ref in references.items():
        result["references"].append(dict(model=model, suite=suite,
            inits_0_29=c.stats({k: e for k, e in ref.items() if k[2] < 30}, model),
            inits_0_49=c.stats(ref, model)))
    totals = []
    for cell, arms in sorted(cells.items()):
        model, suite, size = cell
        ref = {k: e for k, e in references.get((model, suite),{}).items() if k[2] < 30}
        names = sorted({k[1] for k in ref})
        if args.reference == "pure-arm":
            names = sorted(set(names) | set(planned[cell]))
        elif set(planned[cell]) - set(names):
            raise ValueError(f"{cell}: lane task names absent from teacher reference")
        tasks, selected_rows, selected_ref = {}, {}, {}
        model_label = c.MODEL_LABELS[model]
        lines += [f"## {model_label} · {c.SUITE_LABELS[suite]} · 库 {size} 集", "",
                  "| 任务 | 选择档位 | 首次失败 | 失败后的恢复档位 | 完整性 |",
                  "|---|---|---|---|---|"]
        for task in names:
            pure = {k: e for k, e in ref.items() if k[1] == task}
            pure_stats = c.stats(pure, model)
            levels = {"pure": dict(pure_stats, missing_inits=sorted(set(range(30)) - {k[2] for k in pure}),
                                   passes={str(t): True if pure_stats["n"] == 30 else None for t in tables})}
            for level in c.LEVELS[1:]:
                data = {k: e for k, e in arms.get(level, {}).items() if k in pure}
                st = c.stats(data, model)
                paired_pure = sum(pure[k]["success"] for k in data)
                gains = sum(e["success"] and not pure[k]["success"] for k, e in data.items())
                losses = sum(pure[k]["success"] and not e["success"] for k, e in data.items())
                levels[level] = dict(st, paired_pure_successes=paired_pure, gains=gains, losses=losses,
                    missing_inits=sorted(set(range(30)) - {k[2] for k in data}),
                    passes={str(t): 100*(st["successes"]-pure_stats["successes"]) >= -t*30
                            if st["n"] == pure_stats["n"] == 30 else None for t in tables})
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
        lines += ["", f'已冻结 {row["complete_tasks"]}/{c.task_count(suite)} 个任务；拼接 {stitched["n"]} 局，成功率 {c.percent(stitched["sr"])}，'
                  f'推理开销代理 {c.number(stitched["ir_proxy"])}；同任务纯推理成功率 {c.percent(baseline["sr"])}，开销代理 1.000。'
                  f'全任务纯推理成功率 {c.percent(row["pure"]["sr"])}。', "",
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
