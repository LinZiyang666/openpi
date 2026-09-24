# T / N ablation figures

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
