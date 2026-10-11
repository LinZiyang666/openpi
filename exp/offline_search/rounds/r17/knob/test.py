"""Evaluate frozen per-task knobs on official A; CPU only, no campaign actions."""
import argparse
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from exp.offline_search.rounds.r17.knob import common as c

BOOT, SEED = 5000, 20261005


def raised_level(level, successes, control_successes, n):
    deficit = 100 * (control_successes - successes)
    if level == "pure" or deficit < 6*n:
        return level
    return "pure" if deficit >= 10*n else c.LEVELS[c.LEVELS.index(level)-1]


def load_controls(root, models):
    references = {}
    for model in sorted(models):
        # Both Cosmos segments (and the Fast-WAM ancestor) are read by load_lane.
        lane = Path(root) / c.CONTROL_NAMES[model][-1]
        loaded = c.cache.load_lane(lane, c.cache.receipt_snapshot(lane), c.cache.Sources(), c.cache.SIZES)
        for (actual_model, _, group, suite), result in loaded.items():
            if actual_model != model or group != "control":
                raise ValueError(f"{lane}: expected {model} pure-inference control")
            for episode in result["data"].values():
                episode["pure"] = True
            references[model, suite] = result["data"]
    return references


def paired(data, ref, indices, names):
    keys = sorted(data.keys() & ref.keys())
    gains = sum(data[k]["success"] and not ref[k]["success"] for k in keys)
    losses = sum(ref[k]["success"] and not data[k]["success"] for k in keys)
    draws = np.zeros(BOOT)
    rng = np.random.default_rng(SEED)
    for t, task in enumerate(names):
        ks = [k for k in keys if k[1] == task]
        if not ks:
            continue
        delta = np.array([int(data[k]["success"]) - int(ref[k]["success"]) for k in ks])
        # Full cells use the shared R16 suite/task/init resampling. Partial
        # cells describe the observed pairs only, with their coverage reported.
        idx = indices[:, t, :] if len(ks) == 50 else rng.integers(len(ks), size=(BOOT, len(ks)))
        draws += delta[idx].sum(axis=1)
    n = len(keys)
    return dict(n=n, gains=gains, losses=losses, difference=(gains-losses)/n if n else None,
                bootstrap_ci95=np.quantile(draws/n, [.025, .975]).tolist() if n else None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, default=c.CONTROL_ROOT)
    args = parser.parse_args(argv)
    cells, planned = c.load_knob(args.lane, "A")
    references = load_controls(args.control_root, {m for m, _, _ in cells})
    rng = np.random.default_rng(SEED)
    indices = {suite: rng.integers(0, 50, size=(BOOT, 10, 50))
               for suite in ("libero_10", "libero_spatial")}
    result = dict(bootstrap=dict(draws=BOOT, seed=SEED, method="within-task paired init resampling"), cells=[])
    raises = {}
    lines = ["# 逐任务旋钮测试", "", "使用 A 集官方起点 0–49，每任务固定一个档位，与同起点纯推理对照配对。",
             "推理开销代理按模型调用数除以执行控制数折算的决策数计算；DP 和 Fast-WAM 每 5 步折算一次，Cosmos 每 16 步折算一次，纯推理按一计。",
             "成功率差的95% 区间在每个任务内按起点配对重采样5,000 次，随机种子为20261005。",
             "任务成功率低于对照至少 10 个点，提到纯推理；低 6–9 个点，提一档，0.5 档提到纯推理；纯推理保持。阈值用成功局数作整数比较。",
             "缺局只报告已有配对，不补齐、不按部分结果提档；缺少完整执行计数时开销记为暂无。提档清单供后续复测，当前结果仍是提档前的测量。", "",
             "| 模型 | 任务套件 | 库集数 | 已有局数 | 配对局数 | 成功率 | 同起点纯推理成功率 | 差（个点） | 95% 区间（个点） | 新增成功 | 丢失成功 | 推理开销代理 | 完整性 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|"]
    detail = []
    for cell, arms in sorted(cells.items()):
        model, suite, size = cell
        ref = references.get((model, suite), {})
        names = sorted({k[1] for k in ref} | planned[cell].keys())
        if len(names) != 10:
            raise ValueError(f"{cell}: expected 10 task names from control and test plan")
        data, tasks = {}, {}
        detail += [f"## {c.MODEL_LABELS[model]} · {c.SUITE_LABELS[suite]} · 库 {size} 集", "",
                   "| 任务 | 当前档位 | 已有局数 | 配对局数 | 成功率 | 同起点纯推理成功率 | 差（个点） | 复测档位 | 完整性 |",
                   "|---|---|---:|---:|---:|---:|---:|---|---|"]
        for task in names:
            choices = planned[cell].get(task, set())
            if len(choices) > 1:
                raise ValueError(f"{cell}/{task}: TEST requires one frozen level, got {sorted(choices)}")
            level = next(iter(choices), None)
            observed = {k: e for k, e in arms.get(level, {}).items() if k[1] == task}
            control = {k: e for k, e in ref.items() if k[1] == task and 0 <= k[2] < 50}
            matched = observed.keys() & control.keys()
            st = c.stats(observed, model)
            base = c.stats({k: control[k] for k in matched}, model)
            candidate_s = sum(observed[k]["success"] for k in matched)
            delta = (candidate_s-base["successes"])/len(matched) if matched else None
            complete = len(observed) == len(control) == len(matched) == 50
            new_level = raised_level(level, candidate_s, base["successes"], 50) if complete else None
            if new_level is not None and new_level != level:
                c.nested_set(raises, cell, task, new_level)
            tasks[task] = dict(level=level, **st, paired_n=len(matched), control=base, difference=delta,
                               retest_level=new_level, incomplete=not complete,
                               missing_inits=sorted(set(range(50)) - {k[2] for k in observed}),
                               missing_control_inits=sorted(set(range(50)) - {k[2] for k in control}))
            data.update(observed)
            detail.append(f'| {task} | {c.LABELS.get(level, "未安排")} | {st["n"]} | {len(matched)} | '
                          f'{c.percent(st["sr"])} | {c.percent(base["sr"])} | {"暂无" if delta is None else f"{100*delta:+.1f}"} | '
                          f'{c.LABELS[new_level] if new_level != level and new_level is not None else "无需提档" if complete else "待完整"} | '
                          f'{"完整" if complete else "不完整"} |')
        st = c.stats(data, model)
        comparison = paired(data, ref, indices[suite], names)
        paired_control = c.stats({k: ref[k] for k in data.keys() & ref.keys()}, model)
        row = dict(model=model, suite=suite, size=size, **st, comparison=comparison,
                   control=paired_control, tasks=tasks, incomplete=any(t["incomplete"] for t in tasks.values()))
        result["cells"].append(row)
        ci = comparison["bootstrap_ci95"]
        interval = "暂无" if ci is None else f"[{100*ci[0]:+.1f}, {100*ci[1]:+.1f}]"
        diff = comparison["difference"]
        lines.append(f'| {c.MODEL_LABELS[model]} | {c.SUITE_LABELS[suite]} | {size} | {st["n"]} | {comparison["n"]} | '
                     f'{c.percent(st["sr"])} | {c.percent(paired_control["sr"])} | {"暂无" if diff is None else f"{100*diff:+.1f}"} | '
                     f'{interval} | {comparison["gains"]} | {comparison["losses"]} | {c.number(st["ir_proxy"])} | '
                     f'{"不完整" if row["incomplete"] else "完整"} |')
        detail.append("")
    args.out.mkdir(parents=True, exist_ok=True)
    c.write_json(args.out / "raise.json", raises)
    c.write_json(args.out / "test.json", result)
    (args.out / "TEST_A.md").write_text("\n".join(lines + [""] + detail) + "\n")
    print(f"Wrote test report and retest levels to {args.out}")
    return result


if __name__ == "__main__":
    main()
