"""Plot two controlled T/N comparisons from the original accepted episode journals.

Run with the repository's .venv Python. Writes only to this figure directory.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(HERE / ".mplconfig")
helper_path = HERE.parent / "reproduce.py"
spec = importlib.util.spec_from_file_location("report_data", helper_path)
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)
import matplotlib.pyplot as plt
import numpy as np


def aligned_outcomes(tasks: list[str], arms: list[str]) -> np.ndarray:
    """Load the same 50 environment identities in every cell of one comparison."""
    result = np.empty((len(tasks), 50, len(arms)), dtype=np.int8)
    for ti, task in enumerate(tasks):
        reference = None
        for ai, arm in enumerate(arms):
            rows = data.cell(data.choose_root(arm, task), arm, task)
            identities = [tuple(r.get(k) for k in data.KEYS) for r in rows]
            assert reference is None or identities == reference, (task, arm)
            reference = identities
            result[ti, :, ai] = [r["success"] for r in rows]
    return result


def summarize(raw: np.ndarray) -> dict:
    """Compute fixed-task means and within-task paired bootstrap intervals."""
    rng = np.random.default_rng(20260922)
    draws = np.zeros((20000, raw.shape[2]))
    for task_rows in raw:
        indices = rng.integers(0, 50, size=(20000, 50))
        draws += task_rows[indices].mean(axis=1) / raw.shape[0]
    return {"task_rates": raw.mean(axis=1).tolist(),
            "mean": raw.mean(axis=(0, 1)).tolist(),
            "successes": raw.sum(axis=(0, 1)).astype(int).tolist(),
            "n_per_arm": int(raw.shape[0] * raw.shape[1]),
            "ci95": np.percentile(draws, [2.5, 97.5], axis=0).T.tolist(),
            "differences": [{"a": a, "b": b, "delta": float(raw[:,:,a].mean() - raw[:,:,b].mean()),
                             "ci95": np.percentile(draws[:,a] - draws[:,b], [2.5,97.5]).tolist()}
                            for a in range(raw.shape[2]) for b in range(a)]}


def task_label(name: str) -> str:
    """Keep full task names while wrapping the long pick-and-place prefix."""
    return name.replace("PickPlace", "PickPlace\n")


def plot_comparison(tasks: list[str], result: dict, labels: list[str], colors: list[str],
                    title: str, subtitle: str, filename: str, conclusion: str) -> None:
    """Plot task-level points and a separate aggregate panel on a common percentage scale."""
    n_methods = len(labels)
    rates = np.array(result["task_rates"]) * 100
    means = np.array(result["mean"]) * 100
    intervals = np.array(result["ci95"]) * 100
    height = 10.4 if len(tasks) == 13 else 9.0
    fig = plt.figure(figsize=(15, height), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[2.25, 1], wspace=.22,
                           left=.235, right=.975, top=.785, bottom=.12)
    ax = fig.add_subplot(grid[0])
    ax_macro = fig.add_subplot(grid[1])
    bar_height = .29 if n_methods == 2 else .23
    offsets = (np.arange(n_methods) - (n_methods-1)/2) * (bar_height+.025)
    y = np.arange(len(tasks))
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.barh(y+offsets[i], rates[:,i], height=bar_height, color=color, label=label, zorder=3)
        for ti, value in enumerate(rates[:,i]):
            ax.text(value+1.0, ti+offsets[i], f"{value:.0f}%", va="center", ha="left",
                    fontsize=9, color="#243b50")
    ax.set(yticks=y, yticklabels=[task_label(t) for t in tasks], xlim=(0,111),
           xticks=[0,20,40,60,80,100], xlabel="Success rate (%)")
    ax.set_title("Task-level results", fontsize=13, fontweight="bold", loc="left", pad=16)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, labelsize=10, pad=9)
    ax.grid(axis="x", color="#e7edf2", lw=.8, zorder=0)
    for ti in range(len(tasks)):
        if ti % 2 == 0:
            ax.axhspan(ti-.47, ti+.47, color="#f5f8fa", zorder=0)
    ax_macro.bar(range(n_methods), means, color=colors, width=.59, zorder=3)
    ax_macro.errorbar(range(n_methods), means,
                      yerr=[means-intervals[:,0], intervals[:,1]-means],
                      fmt="none", ecolor="#263e52", capsize=5, lw=1.5, zorder=4)
    for i, value in enumerate(means):
        ax_macro.text(i, intervals[i,1]+2.3, f"{value:.2f}%", ha="center", va="bottom",
                      fontsize=13, fontweight="bold", color=colors[i])
        ax_macro.text(i, max(value*.5, 10), f"{result['successes'][i]} /\n{result['n_per_arm']}",
                      ha="center", va="center", color="white", fontsize=10)
    ax_macro.set(xticks=range(n_methods), xticklabels=[label.split(" | ")[0] for label in labels],
                 ylim=(0,100), yticks=[0,20,40,60,80,100], ylabel="Success rate (%)")
    ax_macro.grid(axis="y", color="#e7edf2", lw=.8, zorder=0)
    ax_macro.set_title(f"Mean across the same {len(tasks)} tasks", fontsize=12, fontweight="bold", pad=16)
    for axis in (ax,ax_macro):
        for side in ("top","right","left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color("#cbd5df")
        axis.tick_params(colors="#42566b")
    fig.text(.04,.964,title,fontsize=21,fontweight="bold",color="#172f44",va="top")
    fig.text(.04,.920,subtitle,fontsize=12,color="#546c80",va="top")
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc="upper left",bbox_to_anchor=(.035,.888),
               ncol=n_methods,frameon=False,fontsize=11,handlelength=1.6,columnspacing=2.3)
    fig.text(.04,.050,conclusion,fontsize=12,fontweight="bold",color="#254a60")
    fig.text(.04,.019,"50 matched episodes per task and setting. Task bars are point estimates; mean error bars are exploratory 95% bootstrap intervals.",
             fontsize=9,color="#667d8e")
    for extension in ("png","svg","pdf"):
        fig.savefig(HERE/f"{filename}.{extension}",dpi=200,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def main() -> None:
    """Generate both comparisons, their data, and an archive with source fingerprints."""
    previous_path = HERE.parent / "statistics.json"
    previous = json.loads(previous_path.read_text())
    tasks13 = [row["task"] for row in previous["tasks"]]
    tasks8 = previous["matched_eight_tasks"]["tasks"]
    arms_t = ["warmreset_t0.2", "resetfinal_t0.2"]
    arms_n = ["resetfinal_t0.1", "resetfinal_t0.2", "resetfinal_t0.3"]
    raw_t = aligned_outcomes(tasks13, arms_t)
    raw_n = aligned_outcomes(tasks8, arms_n)
    result_t, result_n = summarize(raw_t), summarize(raw_n)
    assert result_t["successes"] == [460,460]
    assert result_n["successes"] == [174,271,238]
    for i, arm in enumerate(arms_t):
        assert abs(result_t["mean"][i]-previous["primary"][arm]["sr"]) < 1e-12
    for i, arm in enumerate(arms_n):
        assert abs(result_n["mean"][i]-previous["matched_eight_tasks"][arm]) < 1e-12
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"svg.fonttype":"none",
                         "axes.labelcolor":"#42566b","text.color":"#243b50","pdf.fonttype":42})
    plot_comparison(tasks13,result_t,["T = 0.2 | warmreset","T = 0 | resetfinal"],
                    ["#138b80","#8260ac"],
                    "Snapshot choice: vary T, hold N = 2",
                    "13 tasks | Both methods restart at t = 1 and run two updates to t = 0; only the cached initialization rule changes.",
                    "01_snapshot_T_fixed_N2",
                    "Task-level point estimates differ despite equal aggregate success (70.77%).")
    plot_comparison(tasks8,result_n,["N = 1 | resetfinal","N = 2 | resetfinal","N = 3 | resetfinal"],
                    ["#5282ad","#8260ac","#c28a36"],
                    "Inference budget: vary N, hold T = 0",
                    "Same 8 tasks at every budget | All settings start from the cached final action; the uniform step size is -1/N.",
                    "02_budget_N_fixed_T0",
                    "Two steps have the highest observed mean on this eight-task set; task-level trends differ.")
    results = {"fixed_N2": {"tasks":tasks13,"arms":arms_t,"T":[.2,0],"N":2,**result_t},
               "fixed_T0": {"tasks":tasks8,"arms":arms_n,"T":0,"N":[1,2,3],**result_n},
               "bootstrap":{"draws":20000,"seed":20260922,
                            "method":"paired resampling within each fixed task; percentile 95%; exploratory, no multiplicity correction"},
               "cohort":"init_idx 0..49; environment seed 2,000,000+idx; macro13-only warmreset; disjoint rc/macro13 task batches for resetfinal"}
    (HERE/"plot_data.json").write_text(json.dumps(results,indent=2)+"\n")
    with (HERE/"task_rates.csv").open("w",newline="") as handle:
        writer=csv.writer(handle)
        writer.writerow(["comparison","task","arm","T","N","successes","episodes","success_rate"])
        for name,tasks,arms,raw in [("fixed_N2",tasks13,arms_t,raw_t),("fixed_T0",tasks8,arms_n,raw_n)]:
            for ti,task in enumerate(tasks):
                for ai,arm in enumerate(arms):
                    t = ([.2,0][ai] if name=="fixed_N2" else 0)
                    n = (2 if name=="fixed_N2" else ai+1)
                    writer.writerow([name,task,arm,t,n,int(raw[ti,:,ai].sum()),50,float(raw[ti,:,ai].mean())])
    note = """# T / N ablation figures

中文说明：

1. **固定 N=2，改变 T**：13 个任务，每个设置每任务 50 集。T=0.2 使用 warmreset，T=0 使用 resetfinal。两者均从 t=1 更新到 t=0，只改变缓存初始化的选择规则；总体均为 460/650 = 70.77%，逐任务结果不同。
2. **固定 T=0，改变 N**：同一组 8 个任务，每个设置每任务 50 集。N=1、2、3 的成功率分别为 174/400 = 43.50%、271/400 = 67.75%、238/400 = 59.50%。此处步数及其配套步长 -1/N 一起变化。不能将第二张图的八任务均值直接与第一张图的十三任务均值比较。

逐任务柱子显示点估计；总体均值误差线是固定任务内对环境身份进行 20,000 次配对 bootstrap 得到的探索性 95% 区间，没有多重比较校正。环境种子配对不意味着模型采样噪声相同。两图均仅展示已测设置，不构成完整 T×N 网格搜索。

取数沿用双语报告的固定 50 集口径。warmreset 只取 rc_macro13；resetfinal 取 rc 的两个先导任务与 rc_macro13 的其余任务，不平均或重复计入先导复测。图 1 CloseFridge 的 warmreset 为宏观轮 25/50，而早期阶梯轮为 26/50。

English captions:

**Figure 1. Snapshot-time ablation at a fixed two-step budget.** Warmreset initializes from the cached intermediate at T=0.2; resetfinal initializes from the cached final action at T=0. Both restart flow time at t=1 and perform two updates to t=0. Results use the same 13 tasks and 50 matched environment identities per setting/task. The two methods have identical aggregate success, but task-specific results differ.

**Figure 2. Step-budget ablation at a fixed final-action initialization (T=0).** Resetfinal is evaluated at N=1,2,3, with a uniform update step of -1/N, on the same eight tasks and 50 matched environment identities per setting/task. Two steps have the highest observed aggregate success in this set; additional updates can help or harm individual tasks. These eight-task means should not be compared directly with the thirteen-task means in Figure 1.

In both figures, task bars show point estimates and mean error bars show exploratory percentile 95% intervals from 20,000 paired bootstrap draws within fixed tasks. Intervals are not corrected for multiple comparisons. The figures cover only tested settings, not a full T-by-N sweep.

Files: each figure is supplied as PNG, SVG, and PDF. `task_rates.csv` and `plot_data.json` hold the plotted values. `source_manifest.json` records the raw inputs. The plotting script reads the existing report's `reproduce.py` helper in its parent directory; the figure-only archive requires that report directory to rerun.
"""
    (HERE/"README.md").write_text(note)
    source_files=dict(data.sources)
    for source in (helper_path,previous_path):
        source_files[str(source.relative_to(data.REPO))]=hashlib.sha256(source.read_bytes()).hexdigest()
    for rel,digest in source_files.items():
        assert hashlib.sha256((data.REPO/rel).read_bytes()).hexdigest()==digest,rel
    (HERE/"source_manifest.json").write_text(json.dumps({"repository":str(data.REPO),"files":source_files},indent=2)+"\n")
    files=sorted(p for p in HERE.iterdir() if p.is_file() and p.suffix!=".zip")
    with zipfile.ZipFile(HERE/"tn_ablation_figures.zip","w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path,path.name)
    print(json.dumps({"fixed_N2":result_t,"fixed_T0":result_n,"source_files":len(source_files)},indent=2))


if __name__=="__main__":
    main()
