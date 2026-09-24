"""Write equivalent Chinese and English reports, standalone HTML, and a shareable bundle."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from markdown_it import MarkdownIt

OUT = Path(__file__).resolve().parent
S = json.loads((OUT / "statistics.json").read_text())
P = S["primary"]
ARMS = S["arms"]


def table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a Markdown table with explicit columns."""
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def main_table(lang: str) -> str:
    """Summarize the aligned 650-episode cohort."""
    headers = ["实验臂", "每次决策去噪次数", "成功 / 总数", "成功率", "95% 区间"] if lang == "zh" else [
        "Arm", "Denoising calls / decision", "Successes / episodes", "Success rate", "95% interval"]
    return table(headers, [[f"`{a}`", "10" if a == "full" else "2", f"{P[a]['successes']} / 650",
                           f"{100*P[a]['sr']:.2f}%", f"[{100*P[a]['ci95'][0]:.2f}, {100*P[a]['ci95'][1]:.2f}]%"] for a in ARMS])


def contrast_table(lang: str) -> str:
    """Summarize paired differences without interpreting interval overlap as equivalence."""
    headers = ["配对比较 A − B", "差值（百分点）", "95% 区间（百分点）"] if lang == "zh" else [
        "Paired contrast A − B", "Difference (pp)", "95% interval (pp)"]
    return table(headers, [[f"`{d['a']}` − `{d['b']}`", f"{100*d['delta']:+.2f}",
                           f"[{100*d['ci95'][0]:+.2f}, {100*d['ci95'][1]:+.2f}]"] for d in S["contrasts"]])


def tasks_table(lang: str) -> str:
    """Use one row per task and the exact same cohort as the main figure."""
    headers = ["任务" if lang == "zh" else "Task", "Full", "Plain", "Resume", "Reset", "Final", "Shoot"]
    return table(headers, [[r["task"], *[f"{r[a]*100:.0f}%" for a in ARMS]] for r in S["tasks"]])


def ladder_table(lang: str) -> str:
    """Show the original diagnostic step ladder, separately from the macro batch."""
    headers = ["任务", "步数", "Plain", "Resume", "Reset", "Final"] if lang == "zh" else [
        "Task", "Steps", "Plain", "Resume", "Reset", "Final"]
    rows = []
    for task, r in S["ladder"].items():
        for n in (1, 2, 3):
            rows.append([task, str(n)])
            rows[-1].extend(f"{100*r[f][n-1]:.0f}%" for f in ("plain", "warm", "warmreset", "resetfinal"))
    return table(headers, rows)


def robustness_table(lang: str) -> str:
    """Compare the original seed block, alternate seed block, and larger follow-up."""
    headers = ["任务", "实验臂", "2M：原始 50 集", "1M：50 集", "2M：500 集"] if lang == "zh" else [
        "Task", "Arm", "2M: original 50", "1M: 50", "2M: 500"]
    return table(headers, [[task, f"`{arm}`", *[f"{100*r[k]:.1f}%" for k in ("2m_50", "1m_50", "2m_500")]]
                          for task, arms in S["robustness"].items() for arm, r in arms.items()])


def sensitivity_table(lang: str) -> str:
    """Make the alternative use of all recorded replicates transparent."""
    headers = ["实验臂", "本文统一 50 集口径", "仓库全部数据、身份等权口径"] if lang == "zh" else [
        "Arm", "This report: matched 50", "Repository: all data, identity-balanced"]
    return table(headers, [[f"`{a}`", f"{100*P[a]['sr']:.3f}%", f"{100*S['all_data_identity_balanced'][a]:.3f}%"] for a in ARMS])


zh = f"""# π0.5 / RoboCasa 推理消融实验报告

**中文版本 · 2026-09-22 · 基于当前工作区的独立数据复核**

本报告解释 `exp/step_diag/` 中完整推理、减少去噪步数与缓存初始化变体的实验设计和结果。范围是固定的 13 个 RoboCasa 任务、π0.5、当前检查点与缓存库；不是 RoboCasa365 全任务基准。英文版与本版使用完全相同的统计数据和图表。

**主要发现。** 在每臂每任务固定 50 个环境身份的统一口径下，10 步 full 成功率为 **54.77%（356/650）**；两步 warmreset 与两步 resetfinal 均为 **70.77%（460/650）**。warmreset 相对 full 的配对差为 **+16.00 个百分点，探索性 95% 区间 [+11.54, +20.46]**。这个差距在去掉早期重复评测后仍存在。最终动作与中间快照两种起点的总体成功率相同，但逐任务、逐 episode 结果并不相同。

## 1. 实验要回答什么

这组消融拆开三个因素：**计算预算、动作张量起点、去噪时间安排**。所有方法都根据当前观测产生动作；warm 系列额外从固定缓存库检索一个候选起点。缓存提供的是其他已收集轨迹中的信息，不是当前评测 episode 的未来动作。

这里的“步数”指动作去噪网络的调用次数 NFE，不是机器人环境步数。`t` 是模型的 flow time：标准推理从 `t=1` 向 `t=0` 更新。每次 Euler 更新可写为 `x ← x + Δt · vθ(x, t, 当前观测)`。相同的张量配上不同的 `t`，网络预测与后续动作都可能改变。

| 实验臂 | 初始动作张量 | NFE | 网络看到的 t | 每步 Δt | 主要对照目的 |
| --- | --- | --- | --- | --- | --- |
| `full` | 随机噪声 | 10 | 1.0, 0.9, …, 0.1 | −0.1 | 完整推理参考 |
| `plain_k2` | 随机噪声 | 2 | 1.0, 0.5 | −0.5 | 单纯减少计算 |
| `warm_t0.2` | 缓存快照 x₀.₂ | 2 | 0.2, 0.1 | −0.1 | 沿原时间网格续跑 |
| `warmreset_t0.2` | 缓存快照 x₀.₂ | 2 | 1.0, 0.5 | −0.5 | 缓存初始化后重新跑短循环 |
| `resetfinal_t0.2` | 缓存最终动作 x₀ | 2 | 1.0, 0.5 | −0.5 | 去掉中间快照依赖 |
| `warmshoot_t0.2` | 缓存快照 x₀.₂ | 2 | 0.2, −0.3 | −0.5 | 保留初始 t、只放大步长 |

**warmreset 与 resetfinal 的直接区别就是初始化张量。** 在同一个检索条目上，前者取中间去噪快照，后者取最终动作；两者随后都使用当前观测，执行 `t=1, 0.5` 的两次更新。重置 t 不会清空缓存，也不会重新添加随机噪声。闭环轨迹分开以后，各臂实际检索到的条目可能不同，因此“起点控制”指算法规则，不代表所有后续决策始终检索到同一条记录。

![图 1：各臂的推理起点与时间安排](figures/00_inference_schedules.png)

*图 1。实心点表示一次网络调用，空心点表示最后一次更新后的时间。warmshoot 的第二次调用落在负时间。图中时间按理想小数展示，代码沿用 float32 网格累加。实现依据见 [S1]、[S2]。*

## 2. 样本、共同设置与分析口径

主比较固定为 **13 任务 × 50 episodes × 6 臂 = 3,900 条已接受的终结记录**。环境种子为 `2,000,000 + init_idx`，`init_idx=0…49`。按任务、初始化序号、环境种子、lane、pin、layout、style 配对；每臂恰有 650 个身份，无缺失或重复身份。

各臂使用同一 π0.5 配置/检查点身份，layout/style 均为 1，执行动作的重规划间隔为 5 个环境步。warm 系列使用同一 W13 库、top-1 检索和强制 warm-start 规则，库写入关闭。成功与失败都进入分母。环境种子配对不等于模型采样噪声逐次配对；各臂也不是同一服务进程同时运行。

| 方法 | 本报告取数来源 | 每任务取数 |
| --- | --- | --- |
| full / plain_k2 / warm_t0.2 | `rc` 原 7 任务 + `rc_macro13` 新增 6 任务 | 只取 idx 0…49；plain/warm 另有部分 50…99 数据，留作全量口径对照 |
| warmreset_t0.2 / warmshoot_t0.2 | 只取 `rc_macro13` 的完整 13 任务轮 | 50；不混入 `rc` 的两任务先导复测 |
| resetfinal_t0.2 | `rc` 的 CloseFridge、PickPlaceCounterToStove + `rc_macro13` 其余 11 任务 | 50；两批任务不重叠 |

因此，resetfinal 的 **13 × 50 已全部完成**；全量实验并没有整套重复两遍。扩展轮复用了既有任务，并对部分方法、部分任务进行了复测。

所有表格的总体成功率按任务等权平均；主表每个 cell 样本量相等，因此也等于总成功数除以 650。区间由本报告独立重算：固定 13 个任务，在每个任务内部对 50 个环境身份进行 **20,000 次配对 bootstrap**，同一次抽样同步用于各臂，随机种子为 `20260922`。区间是未经多重比较校正的探索性 percentile 95% 区间，条件于这些固定任务，不能解释为对任意新任务总体的置信区间。后续变体是看到初始结果后设计的探索性实验。

## 3. 13 任务总体结果

{main_table('zh')}

{contrast_table('zh')}

![图 2：总体成功率与配对差](figures/01_main_comparison.png)

*图 2。左图为同一 13 任务、每任务 50 个环境身份的成功率；右图为配对差及探索性 95% 区间。括号中的 10/2 表示每次决策的去噪调用次数。差值单位是百分点。*

**预算消融：full → plain。** 从 10 步减到 2 步，成功率由 54.77% 降为 48.00%，差 −6.77 个百分点。这是随机噪声初始化下的预算代价，不代表每个任务都受损。

**初始化消融：plain → warmreset。** 两者都使用 `t=1,0.5`、相同两步循环，主要改变随机噪声与检索缓存两种起点；warmreset 高 22.77 个百分点。这支持缓存初始化提供了有用信息。full 不使用这个缓存先验，因此不能把 warmreset 的优势概括成“同一种推理做两步天然比十步好”。

**续跑方式消融：warm resume → warmreset。** 相同缓存类型与预算下，重新开始短循环高 42.92 个百分点。这里同时改变了 t 序列和 Δt；实验支持这套组合方案，尚不能把收益全部归给单独一个旋钮。“精确续跑”只表示沿用原时间网格；缓存来自另一场景，不是当前观测下完整轨迹的后两步。

**中间快照消融：warmreset → resetfinal。** 两者均为 460/650，但有 66 对环境是 final 成功、reset 失败，另有 66 对方向相反，总计 **132 对结果不同**。差值区间为 [−3.38, +3.38] 个百分点。证据支持“当前配置下最终动作是有效替代起点”，不等于证明两方法等价，也不等于证明中间快照永远无用。

**步长/时间一致性对照：warmshoot。** 两步 warmshoot 仅 3/650 成功。其第二次网络调用进入负时间，这与失败相容；仅凭这组实验还不能把所有损伤唯一归因于负时间，因为其更新路径也整体改变了。

## 4. 逐任务差异

![图 3：13 任务逐任务成功率](figures/02_task_heatmap.png)

*图 3。每个格子均为 n=50 的成功率点估计。此处不依据逐格颜色作显著性判定。所有列使用与图 2 相同的取数规则。*

{tasks_table('zh')}

warmreset 相对 plain 的点估计在 11 个任务上更高、1 个持平、1 个更低；相对 full 则在 10 个任务上更高、1 个持平、2 个更低。它并非所有任务都获益。

CloseFridge 上，宏观轮 warmreset 为 50%，resetfinal 为 76%；PickPlaceCounterToStove 上，两者分别为 82% 与 72%。总体相同可以由方向相反的任务差异抵消。本文逐任务表的 CloseFridge warmreset 采用宏观轮 25/50；早期阶梯实验用的是先导轮 26/50，两个数字对应不同运行，不能交替当作同一条测量。

## 5. 步数阶梯：1、2、3 步

最初在 CloseFridge 与 PickPlaceCounterToStove 两个诊断任务上，四类方法都跑了 1/2/3 步，每格 50 集。它们分别代表该批数据中减步受损明显和减步较稳定的情形；这两个任务经过选择，不是随机抽取的任务样本。

![图 4：两个诊断任务的步数阶梯](figures/03_step_ladder.png)

*图 4。使用 `rc` 原始阶梯轮，点为成功率、连线仅帮助阅读，不表示中间预算已测试。虚线为 full 的 10 步参考，均为 50 集。*

{ladder_table('zh')}

CloseFridge 上，resetfinal 随 1/2/3 步从 24% → 76% → 84%；搬运任务上则从 100% → 72% → 42%。更多更新可能帮助一个任务，同时损害另一个任务，不能据此制定“步数越多越好”的统一规则。

warm/warmreset 跨档时，预算和缓存快照时间一起变化（0.1/0.2/0.3）；resetfinal 始终取缓存最终动作，跨档比较更直接地考察步数与其配套网格的变化。即便如此，它也不是固定 Δt 下只增加迭代次数的实验。

一步、三步 resetfinal 后续各扩展到 8 个任务。为了避免把 8 任务均值与 13 任务均值混比，本报告在**相同的 8 任务**上重新对齐：

| 方法 | 相同 8 任务的成功率 | episodes |
| --- | --- | --- |
| full，10 步 | 56.50% | 400 |
| resetfinal，1 步 | 43.50% | 400 |
| resetfinal，2 步 | 67.75% | 400 |
| resetfinal，3 步 | 59.50% | 400 |

这八个任务是 CloseBlenderLid、CloseFridge、OpenCabinet、PickPlaceCounterToCabinet、PickPlaceCounterToStove、PickPlaceDrawerToCounter、PickPlaceSinkToCounter、PickPlaceToasterToCounter。两步在这组已测预算中具有最高点估计；这不是在独立验证集上证明的最优超参数。

## 6. 500 集扩样与换种子复测

两个诊断任务另跑了四臂 × 每任务 500 集：plain、warm resume、warmreset、warmshoot。**这轮没有 full 或 resetfinal。** 同时，四臂在另一段环境种子 `1,000,000+idx` 上各做了 50 集复测。

![图 5：500 episode 扩样](figures/04_500_episode_followup.png)

*图 5。每根柱子为 n=500，误差线为单个任务/臂的 95% Wilson 区间。虚线仍是原先 n=50 的 full 参考，不是新跑的 n=500 full。*

{robustness_table('zh')}

500 集结果中，warmreset 相对 warm resume 在两个任务分别高 32.2、62.8 个百分点；相对 plain，则在 CloseFridge 高 54.4 个百分点、在搬运任务低 15.8 个百分点。换种子轮保持同样的主要方向。这些复测增强了两个已选任务上的可重复性证据，没有验证 13 任务全体在第二套种子上的效果。

扩样轮 idx 0…49 与原始变体试验重叠，部分 plain/warm 原先已有 idx 0…99。原报告给出排除前 50 集的 450 集敏感性分析。本报告还独立检查了更严格的 **idx 100…499，共 400 集**：warmreset/plain/warm resume 在 CloseFridge 分别为 65.5% / 6.5% / 30.5%，在搬运任务为 78.0% / 94.5% / 16.25%，主要方向不变。排除已用环境不能消除事后选择任务、方法和预算带来的选择效应。

## 7. 代码复核与结果解释边界

此前对工作区实现的审查没有发现能解释主要差距的 full 步数被削弱、成功判定偏袒 warmreset 或当前评测未来信息泄漏证据。full 显式固定为 10 步；warm 系列使用当前观测的 stage-2 条件，缓存只提供初始化张量；各臂通过同一评测循环判定成功。历史成功轨迹进入固定缓存库，是该方法明确增加的信息来源，应在比较中公开描述。[S1–S4]

报告生成时重新检查了 launch/journal 身份与终结记录、3,900 行主比较的跨臂配对，以及已保存的模型/环境身份与 NFE 准入结果。它没有重新运行机器人实验，也没有在这次制表中重放所有服务端数组验证。不同批次的服务代码摘要或机器可能不同；模型身份摘要不是全部大权重字节的独立逐字节重验，因此不能把跨批次可比性表述为所有条件逐位相同。

此前指出的复测覆盖问题，在当前读取的 `warm_variants.py` 中已经改为按环境身份保留多次结果并取平均；该文件也包含跨臂身份检查和退化区间处理。本报告没有修改该汇总器，也不将这次阅读等同于对其全部修订重新验收。为使主比较直观，主图采用固定 50 集口径；另独立重算当前仓库“全部数据、身份等权”的结果：

{sensitivity_table('zh')}

两个口径的主要排序与约 16 个百分点差距一致。它们之间的小数差异来自使用哪些环境身份、是否平均重复评测，不能混用不同口径的 SR 与配对差。

当前结果支持三点：缓存初始化与重新开始的短循环组合有效；中间快照在本配置的平均表现上没有显示出优于最终动作的优势；预算效果明显依赖任务。它们尚未证明“当前观测会按某个固定比例纠正缓存”“检索命中一定更接近正确动作”或“缓存中间快照普遍可删除”。

要进一步分离机制，尚需当前主矩阵未包含的对照：直接执行检索动作的零更新臂、随机/打乱检索起点、缓存初始化的 10 步 reset，以及固定起点下独立控制 t 起点和积分步长的设计。NFE 从 10 降为 2 也不等于端到端快 5 倍；编码、检索与每集决策次数都需单独计时。

## 8. 数据与复现

报告的图表直接由日志重算，未使用生成式图像工具。图内统一保留英文算法名和轴标；中英文图注分别解释含义。每张图同时提供 PNG 和 SVG。

| 文件 | 内容 |
| --- | --- |
| `report.zh.md` / `report.en.md` | 中英文完整报告 |
| `report.zh.html` / `report.en.html` | 图片内嵌的独立 HTML，可离线打开、浏览器打印 |
| `statistics.json` | 全部主结果、配对区间、阶梯与复测数字 |
| `primary_episodes.csv` | 主比较的 3,900 条选中记录、环境身份、原日志文件和行号 |
| `task_success_rates.csv` | 13 任务 × 6 臂成功率 |
| `source_manifest.json` | 137 个工作区输入文件的 SHA-256，固定本次读取快照 |
| `reproduce.py` / `write_reports.py` | 复算和制图脚本、双语文档渲染脚本 |

在仓库根目录依次运行：

```bash
.venv/bin/python -B exp/step_diag/analysis/ablation_report_20260922/reproduce.py
.venv/bin/python -B exp/step_diag/analysis/ablation_report_20260922/write_reports.py
```

脚本只写本报告目录。运行依赖当前本地原始数据及已有 Python 环境；下载包没有复制模型或原始服务端数组。复算读取的是运行当时的工作区，应对照输入清单确认是否与本版一致。

**代码与原始报告索引**

- [S1] `exp/step_diag/pi05.py:71–115, 134–147, 180–190`：变体更新循环、full/plain 步数绑定、最终动作起点替换。
- [S2] `src/openpi/models_pytorch/pi0_pytorch.py:644–769`：随机噪声初始化、标准 stage-3 与缓存续跑。
- [S3] `exp/step_diag/config/arms/pi05_rc/warm_t0.2.yaml:24–35, 72–75`：强制 top-1 warm start、固定库与禁止写入。
- [S4] `exp/robocasa365/episode_runner.py:530–625`：环境重置、动作执行与成功判定；`exp/step_diag/analysis/aggregate_arms.py:56–112, 207–337`：终结记录与准入。
- [S5] `exp/step_diag/analysis/step_vs_warmstart.md:142–295`：变体、阶梯、扩样、换种子和 13 任务扩展的原始叙述。本文的独立重算和解释边界优先于其中较强的机制推断。
- [S6] `exp/step_diag/analysis/warm_variants.py:38–83, 174–199`：当前复测汇总、配对区间与跨臂门控；`exp/step_diag/data/analysis/warm_variants_pi05_macro13.json`：已保存的准入和全数据统计。
"""

en = f"""# π0.5 / RoboCasa Inference Ablation Report

**English edition · 2026-09-22 · Independent analysis of the current working tree**

This report explains the full-inference, reduced-step, and cache-initialized variants in `exp/step_diag/`. Its scope is a fixed set of 13 RoboCasa tasks, π0.5, and the checkpoint and cache used in these runs. It is not an evaluation of all RoboCasa365 tasks. The Chinese edition uses exactly the same statistics and figures.

**Main finding.** With a consistent cohort of 50 environment identities per task and arm, 10-step full inference achieves **54.77% (356/650)**. Two-step warmreset and two-step resetfinal each achieve **70.77% (460/650)**. The paired warmreset-minus-full difference is **+16.00 percentage points, with an exploratory 95% interval of [+11.54, +20.46]**. The gap remains when the earlier repeated evaluations are excluded. The two reset variants have equal aggregate success counts, but their task-level and episode-level outcomes differ.

## 1. Questions and interventions

The ablations separate three factors: **compute budget, the initial action tensor, and the denoising time schedule**. Every method conditions action generation on the current observation. The warm variants additionally retrieve an initialization from a fixed library of previously collected trajectories, rather than obtaining future actions from the current evaluation episode.

Here, a “step” means one denoising-network evaluation, or NFE, not one environment action. The model's flow time `t` normally progresses from 1 toward 0. An Euler update is `x ← x + Δt · vθ(x, t, current observation)`. Changing `t` while retaining the same tensor can change both the network prediction and the eventual action.

| Arm | Initial action tensor | NFE | Times seen by the network | Δt per update | Main purpose |
| --- | --- | --- | --- | --- | --- |
| `full` | Random noise | 10 | 1.0, 0.9, …, 0.1 | −0.1 | Full-inference reference |
| `plain_k2` | Random noise | 2 | 1.0, 0.5 | −0.5 | Reduce inference compute |
| `warm_t0.2` | Cached intermediate x₀.₂ | 2 | 0.2, 0.1 | −0.1 | Resume on the original time grid |
| `warmreset_t0.2` | Cached intermediate x₀.₂ | 2 | 1.0, 0.5 | −0.5 | Restart a short run from the cache |
| `resetfinal_t0.2` | Cached final action x₀ | 2 | 1.0, 0.5 | −0.5 | Remove the need for an intermediate snapshot |
| `warmshoot_t0.2` | Cached intermediate x₀.₂ | 2 | 0.2, −0.3 | −0.5 | Enlarge the step while retaining the initial time |

**Warmreset and resetfinal differ directly in the initialization tensor.** For a given retrieved record, one takes the intermediate snapshot and the other takes the final action. Both then perform two updates at `t=1,0.5`, conditioned on the current observation. Resetting the time neither erases the cached tensor nor adds fresh noise. Once their closed-loop trajectories diverge, the methods may retrieve different records; the controlled change is the algorithmic rule, not an assurance of identical retrieval winners at every later decision.

![Figure 1: Initial tensors and time schedules](figures/00_inference_schedules.png)

*Figure 1. Filled dots are network evaluations; open circles mark the time after the final update. Warmshoot evaluates the network at a negative time on its second call. Times are shown as ideal decimals; the implementation replays float32 grid accumulation. Implementation evidence: [S1], [S2].*

## 2. Cohort, common settings, and statistical method

The main comparison contains **13 tasks × 50 episodes × 6 arms = 3,900 accepted terminal records**. Environment seeds are `2,000,000 + init_idx`, with `init_idx=0…49`. Pairing uses task, initialization index, environment seed, lane, pin, layout, and style. Each arm has exactly 650 identities, with no missing or repeated identity in the selected cohort.

The arms share the π0.5 configuration/checkpoint identity, layout/style 1, and a replanning interval of five environment steps. Warm variants share the W13 library, top-1 retrieval, and forced warm-start rule, with library writes disabled. Successful and unsuccessful episodes both enter the denominator. Matching environment seeds does not imply matching model-noise samples. The arms were not all run simultaneously in a single serving process.

| Methods | Source selection for this report | Episodes per task |
| --- | --- | --- |
| full / plain_k2 / warm_t0.2 | Original 7 tasks from `rc`; remaining 6 from `rc_macro13` | idx 0…49 only; additional idx 50…99 in some plain/warm cells enter the all-data sensitivity comparison |
| warmreset_t0.2 / warmshoot_t0.2 | Complete 13-task batch from `rc_macro13` only | 50; the earlier two-task pilot in `rc` is excluded |
| resetfinal_t0.2 | CloseFridge and PickPlaceCounterToStove from `rc`; remaining 11 tasks from `rc_macro13` | 50; the two task sets are disjoint |

Thus, **all 13 × 50 resetfinal episodes are complete**. The entire matrix was not run twice. The expansion reused existing task evaluations and repeated only some method/task combinations.

Aggregate success is the equally weighted mean of task success rates. Because all primary cells contain 50 episodes, this also equals total successes divided by 650. This report independently recomputes intervals with **20,000 paired bootstrap draws**, resampling the 50 environment identities within each of the 13 fixed tasks. The same sampled indices are used across arms in each draw; the random seed is `20260922`. These are exploratory percentile 95% intervals, without multiplicity correction, conditional on the fixed task roster. They are not intervals for an arbitrary population of unseen tasks. The follow-up variants were designed after the initial results were observed.

## 3. Main results on 13 tasks

{main_table('en')}

{contrast_table('en')}

![Figure 2: Aggregate success and paired differences](figures/01_main_comparison.png)

*Figure 2. Left: success on the same 13 tasks and 50 environment identities per task. Right: paired differences with exploratory 95% intervals. Parenthetical 10/2 denotes denoising calls per decision. Differences are percentage points.*

**Budget ablation: full → plain.** Reducing the budget from 10 to 2 evaluations lowers success from 54.77% to 48.00%, a difference of −6.77 percentage points. This estimates the cost of step reduction with random-noise initialization; it does not imply that every task deteriorates.

**Initialization ablation: plain → warmreset.** Both use the same two-update loop at `t=1,0.5`; the main intervention replaces random noise with a retrieved tensor. Warmreset improves success by 22.77 percentage points. This supports the value of cache initialization. Full inference does not receive that cache prior, so the result cannot be reduced to a claim that two steps intrinsically outperform ten steps of the same procedure.

**Continuation ablation: warm resume → warmreset.** With the same cache type and compute budget, restarting the short loop improves success by 42.92 percentage points. Both the time sequence and Δt change, so the experiment supports their combination rather than identifying the effect of either parameter alone. “Exact resume” refers to retaining the original time grid; the cached tensor was generated in another scene and is not the final two steps of the current observation's full trajectory.

**Snapshot ablation: warmreset → resetfinal.** Both succeed in 460/650 episodes, but final succeeds where reset fails in 66 paired environments, and reset succeeds where final fails in another 66: **132 paired outcomes disagree**. The difference interval is [−3.38, +3.38] percentage points. The evidence supports final actions as a viable alternative initialization in this setting. It does not establish equivalence or prove that intermediate snapshots are universally unnecessary.

**Step/time consistency control: warmshoot.** Two-step warmshoot succeeds in only 3/650 episodes. Its second network call uses a negative time, which is consistent with the observed failure. This experiment alone does not identify negative time as the unique cause, because the update trajectory also changes.

## 4. Task-level heterogeneity

![Figure 3: Success rates for all 13 tasks](figures/02_task_heatmap.png)

*Figure 3. Each cell is a point estimate from 50 episodes. Colors are not a task-wise significance test. Source selection is identical to Figure 2.*

{tasks_table('en')}

Warmreset has a higher point estimate than plain on 11 tasks, ties on one, and is lower on one. Against full, it is higher on 10, ties on one, and is lower on two. The improvement is not universal.

For CloseFridge, the macro-batch warmreset rate is 50%, versus 76% for resetfinal. For PickPlaceCounterToStove, the corresponding rates are 82% and 72%. Opposite task-specific changes can cancel in the aggregate. The primary table uses the macro-batch CloseFridge warmreset result, 25/50. The original step ladder uses the earlier pilot result, 26/50; these are different runs and should not be treated as the same measurement.

## 5. The one-, two-, and three-step ladder

The original diagnostic ladder evaluated four method families at 1/2/3 steps on CloseFridge and PickPlaceCounterToStove, with 50 episodes per cell. They represent a task strongly affected by step reduction and a task relatively insensitive to it in these data. The two tasks were selected diagnostically, not sampled randomly from a task population.

![Figure 4: Step-budget ladders on the two diagnostic tasks](figures/03_step_ladder.png)

*Figure 4. Original ladder runs from `rc`. Points are observed success rates; connecting lines do not imply that intermediate budgets were tested. Dashed lines are the 10-step full reference, also based on 50 episodes.*

{ladder_table('en')}

On CloseFridge, resetfinal rises from 24% → 76% → 84% at 1/2/3 steps. On the pick-and-place task it falls from 100% → 72% → 42%. Additional updates can help one task and harm another; the results do not support a universal “more steps is better” rule.

For warm resume and warmreset, changing the step budget also changes the cached snapshot time, 0.1/0.2/0.3. Resetfinal always initializes from the final action, so its ladder more directly tests the budget and associated grid. It still changes the step size with the number of updates, rather than adding iterations at a fixed Δt.

One- and three-step resetfinal were later expanded to eight tasks. To avoid comparing an eight-task mean with a thirteen-task mean, this report aligns all budgets on **the same eight tasks**:

| Method | Success on the same eight tasks | Episodes |
| --- | --- | --- |
| Full, 10 steps | 56.50% | 400 |
| Resetfinal, 1 step | 43.50% | 400 |
| Resetfinal, 2 steps | 67.75% | 400 |
| Resetfinal, 3 steps | 59.50% | 400 |

The eight tasks are CloseBlenderLid, CloseFridge, OpenCabinet, PickPlaceCounterToCabinet, PickPlaceCounterToStove, PickPlaceDrawerToCounter, PickPlaceSinkToCounter, and PickPlaceToasterToCounter. Two steps have the highest point estimate among these tested budgets. This is not an optimum established on an independent validation set.

## 6. Larger-sample and alternate-seed follow-ups

The two diagnostic tasks received an additional four-arm evaluation with 500 episodes per task: plain, warm resume, warmreset, and warmshoot. **Neither full nor resetfinal was included in that batch.** The same four arms were also evaluated for 50 episodes per task using the alternate environment-seed block `1,000,000+idx`.

![Figure 5: The 500-episode follow-up](figures/04_500_episode_followup.png)

*Figure 5. Each bar uses n=500; error bars are 95% Wilson intervals for individual task/arm cells. Dashed lines retain the original n=50 full reference and are not new n=500 full evaluations.*

{robustness_table('en')}

In the 500-episode batch, warmreset improves over warm resume by 32.2 and 62.8 percentage points on the two tasks. Against plain, it improves CloseFridge by 54.4 points but lowers success on the pick-and-place task by 15.8 points. The alternate seed block retains these principal directions. These follow-ups strengthen reproducibility evidence for the two selected tasks; they do not replicate all 13 tasks on a second seed block.

Indices 0…49 in the larger batch overlap with the original variant pilot, and some original plain/warm cells also contain indices 50…99. The original report gives a 450-episode sensitivity analysis excluding the first 50. This report additionally checks the stricter **idx 100…499 subset, n=400**: warmreset/plain/warm resume score 65.5% / 6.5% / 30.5% on CloseFridge and 78.0% / 94.5% / 16.25% on the pick-and-place task. The main directions remain unchanged. Removing previously used environments does not remove post-hoc task, method, or budget selection.

## 7. Code review context and limits of interpretation

The earlier working-tree review found no evidence that weakened full-inference step counts, an arm-specific success rule, or access to the current episode's future explains the main gap. Full explicitly uses ten steps; warm methods use stage-2 conditioning from the current observation and obtain only their initialization tensor from the cache. The arms share the evaluation loop's success criterion. A fixed library of historical successful trajectories is an explicit additional information source and should be disclosed as part of the method. [S1–S4]

Report generation rechecks launch/journal identities and terminal completeness, pairing of the 3,900 selected records, and the saved model/environment-identity and NFE admission results. It does not run new robot evaluations or repeat every server-array validation. Batches may differ in serving-source digests or machines. The recorded model-identity digest is not an independent bytewise rehash of all large weight files; comparability should therefore not be described as every execution condition being bit-identical.

The previously identified replicate-overwrite behavior has been changed in the currently read `warm_variants.py`: repeated outcomes are retained and averaged per environment identity. The file also contains cross-arm identity checks and handling for degenerate intervals. This report does not modify that aggregator or treat this reading as a complete revalidation of its revisions. The primary figures use a fixed 50-episode cohort for clarity; the repository's current all-data, identity-balanced estimates were also independently reconstructed:

{sensitivity_table('en')}

Both estimands preserve the principal ordering and the approximately 16-point gap. Small numerical differences reflect which identities are included and whether repeated evaluations are averaged. Success rates and paired contrasts from different estimands should not be mixed.

The evidence supports the combination of cache initialization and a restarted short loop; it does not show an aggregate advantage for intermediate snapshots over final actions in this configuration; and budget effects vary substantially by task. It does not establish a fixed mathematical mixture of cached and observation-derived actions, prove that retrieved actions are necessarily closer to the correct action, or justify deleting intermediate snapshots in every setting.

Further mechanism isolation would require controls absent from the current main matrix: zero-update execution of the retrieved action, random or shuffled retrieval, a ten-step reset from a cached initialization, and designs that separately vary initial time and integration step size while holding the start tensor fixed. Reducing denoising calls from ten to two does not establish a fivefold end-to-end speedup; encoding, retrieval, and the number of decisions per episode need their own timing measurements.

## 8. Data and reproduction

Figures are plotted directly from recomputed log statistics, not produced with a generative image model. Both editions share English method labels and axes, with captions in the edition's language. Every figure is supplied as PNG and SVG.

| File | Contents |
| --- | --- |
| `report.zh.md` / `report.en.md` | Complete Chinese and English reports |
| `report.zh.html` / `report.en.html` | Standalone HTML with embedded images; opens offline and supports browser printing |
| `statistics.json` | Main estimates, paired intervals, step ladders, and follow-up statistics |
| `primary_episodes.csv` | All 3,900 selected records, environment identities, source journal paths and line numbers |
| `task_success_rates.csv` | Success rates for 13 tasks and six arms |
| `source_manifest.json` | SHA-256 hashes of 137 working-tree input files defining this read snapshot |
| `reproduce.py` / `write_reports.py` | Recalculation/plotting and bilingual rendering scripts |

From the repository root, run:

```bash
.venv/bin/python -B exp/step_diag/analysis/ablation_report_20260922/reproduce.py
.venv/bin/python -B exp/step_diag/analysis/ablation_report_20260922/write_reports.py
```

The scripts write only to this report directory. Recalculation requires the local raw data and existing Python environment. The download bundle does not copy the model or raw server arrays. Reruns read the working tree as it exists at that time; compare the input manifest to establish whether it matches this edition.

**Implementation and source-report index**

- [S1] `exp/step_diag/pi05.py:71–115, 134–147, 180–190`: variant update loop, full/plain step binding, and final-action substitution.
- [S2] `src/openpi/models_pytorch/pi0_pytorch.py:644–769`: random-noise initialization, standard stage 3, and cached continuation.
- [S3] `exp/step_diag/config/arms/pi05_rc/warm_t0.2.yaml:24–35, 72–75`: forced top-1 warm start, fixed library, and disabled writes.
- [S4] `exp/robocasa365/episode_runner.py:530–625`: environment reset, action execution, and success detection; `exp/step_diag/analysis/aggregate_arms.py:56–112, 207–337`: terminal records and admission.
- [S5] `exp/step_diag/analysis/step_vs_warmstart.md:142–295`: original account of the variants, budget ladder, larger sample, alternate seeds, and thirteen-task expansion. The independent calculations and interpretation limits here take precedence over stronger mechanism claims in that account.
- [S6] `exp/step_diag/analysis/warm_variants.py:38–83, 174–199`: current replicate aggregation, paired intervals, and cross-arm checks; `exp/step_diag/data/analysis/warm_variants_pi05_macro13.json`: saved admission and all-data estimates.
"""

CSS = """
:root{--ink:#1c3043;--muted:#5b6d7e;--accent:#127f77;--line:#dce6ec;--paper:#fff}
*{box-sizing:border-box}body{margin:0;background:#edf2f5;color:var(--ink);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans CJK SC","Microsoft YaHei",sans-serif;font-size:16px;line-height:1.8}
header{max-width:1160px;margin:28px auto 0;padding:18px 52px;background:var(--ink);color:white;border-radius:12px 12px 0 0;display:flex;justify-content:space-between;align-items:center;gap:20px}
header span{font-size:13px;letter-spacing:.07em}header button{border:1px solid #9bb9c5;background:transparent;color:white;border-radius:6px;padding:8px 14px;cursor:pointer}
main{max-width:1160px;margin:0 auto 40px;background:var(--paper);padding:34px 52px 52px;box-shadow:0 12px 35px #23374d0d;border-radius:0 0 12px 12px}
h1{font-size:32px;line-height:1.3;margin:10px 0 18px;color:#142d41;letter-spacing:-.03em}h2{font-size:24px;line-height:1.4;margin:46px 0 16px;border-top:1px solid var(--line);padding-top:24px;color:var(--accent)}
p{margin:15px 0}strong{font-weight:650}a{color:var(--accent)}img{display:block;width:100%;height:auto;margin:24px auto 8px;border:1px solid var(--line);border-radius:8px}p:has(>em:only-child){font-size:14px;line-height:1.65;color:var(--muted);margin-top:8px}
.table-wrap{overflow-x:auto;margin:22px 0}table{width:100%;border-collapse:collapse;font-size:13px;line-height:1.6}th{background:#edf5f4;text-align:left;color:#165950;font-weight:650}th,td{padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}tr:nth-child(even) td{background:#f8fafb}td:first-child{font-weight:500}code{font-family:"SFMono-Regular",Consolas,monospace;font-size:.87em;overflow-wrap:anywhere;background:#f0f4f7;padding:2px 4px;border-radius:3px}pre{background:#172b3d;color:#e6eff6;padding:20px;border-radius:7px;overflow:auto;line-height:1.55}pre code{background:none;color:inherit;overflow-wrap:normal;padding:0}li{margin:9px 0}nav{background:#f5f9fa;padding:14px 22px;border-radius:8px;margin:25px 0}nav a{display:inline-block;margin:3px 18px 3px 0;font-size:14px;text-decoration:none}
@media(max-width:750px){header{margin:0;padding:14px 20px;border-radius:0}main{padding:22px 20px;border-radius:0}h1{font-size:26px}h2{font-size:21px}body{font-size:15px}th,td{padding:8px;font-size:12px}}
@media print{@page{size:A4 landscape;margin:14mm}body{background:white;font-size:10pt;line-height:1.55}header,nav{display:none}main{padding:0;margin:0;max-width:none;box-shadow:none}h1{font-size:23pt}h2{font-size:17pt;break-after:avoid}table{font-size:8pt}tr,img{break-inside:avoid}img{max-height:155mm;object-fit:contain;border:0}pre{white-space:pre-wrap}a{color:inherit;text-decoration:none}p:has(>em:only-child){font-size:9pt}.table-wrap{overflow:visible}}
"""


def render(md: str, lang: str) -> str:
    """Embed all figures so the HTML remains portable as one file."""
    body = MarkdownIt("commonmark").enable("table").render(md)
    headings = []

    def heading(match):
        idx = len(headings) + 1
        title = match.group(1)
        headings.append((idx, title))
        return f'<h2 id="section-{idx}">{title}</h2>'

    body = re.sub(r"<h2>(.*?)</h2>", heading, body)

    def embed(match):
        path = OUT / match.group(1)
        assert path.is_file(), path
        encoded = base64.b64encode(path.read_bytes()).decode()
        return 'src="data:image/png;base64,' + encoded + '"'

    body = re.sub(r'src="(figures/[^\"]+\.png)"', embed, body)
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    nav = '<nav>' + ''.join(f'<a href="#section-{i}">{t}</a>' for i, t in headings) + '</nav>'
    pos = body.find("<h2")
    body = body[:pos] + nav + body[pos:]
    title = "π0.5 / RoboCasa 推理消融实验报告" if lang == "zh" else "π0.5 / RoboCasa Inference Ablation Report"
    print_label = "打印 / 保存 PDF" if lang == "zh" else "Print / Save PDF"
    return f'<!doctype html>\n<html lang="{"zh-CN" if lang == "zh" else "en"}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title><style>{CSS}</style></head><body><header><span>STEP DIAG · ABLATION STUDY · 2026-09-22</span><button onclick="window.print()">{print_label}</button></header><main>{body}</main></body></html>\n'


def main() -> None:
    """Write two complete editions and a manifest-backed portable archive."""
    for lang, text in (("zh", zh), ("en", en)):
        (OUT / f"report.{lang}.md").write_text(text, encoding="utf-8")
        (OUT / f"report.{lang}.html").write_text(render(text, lang), encoding="utf-8")
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and not any(part.startswith(".") for part in p.relative_to(OUT).parts)
                   and p.name not in ("bundle_manifest.json", "ablation_report_bilingual.zip"))
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                "files": {str(p.relative_to(OUT)): {"bytes": p.stat().st_size,
                           "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}}
    manifest_path = OUT / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(OUT / "ablation_report_bilingual.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [*files, manifest_path]:
            archive.write(path, str(path.relative_to(OUT)))
    print(json.dumps({"reports": ["report.zh.md", "report.en.md", "report.zh.html", "report.en.html"],
                      "figures": 5, "bundle": "ablation_report_bilingual.zip", "files": len(files) + 1,
                      "bundle_bytes": (OUT / "ablation_report_bilingual.zip").stat().st_size}))


if __name__ == "__main__":
    main()
