## 给负责人的结论

已经做出不输入任务编号、也不按任务选择纠正头的替换方案。它先用当前观察预测动作偏差，再把本次检索到的示例行上的残差加回来。四个配置单元使用同一套规则，不指定哪些任务困难。检索本身沿用原来的按任务缓存。

离线结果是“保留大部分改善，但还没有追平”。只在初始状态 0–19 上拟合，用 20–29 评估。相比不纠正的缓存，新方案的动作均方误差下降约 38%–47%；相对当前按任务纠正器，它保留了约 87%–96% 的误差改善。这个比例不是闭环成功率，也不能证明保住了成功率收益。

| 配置单元 | 相对不纠正缓存的误差下降 | 比当前纠正器多出的误差 | 保留的误差改善 |
|---|---:|---:|---:|
| π0.5，LIBERO-10，50 示例 | 40.5% | 11.2% | 87.1% |
| π0.5，Spatial，50 示例 | 45.8% | 11.5% | 89.2% |
| GR00T，LIBERO-10，50 示例 | 46.7% | 9.1% | 91.3% |
| GR00T，Spatial，50 示例 | 37.5% | 2.4% | 96.2% |

上述数字来自纯缓存轨迹上相同观察的策略影子比较。当前纠正器还使用了更多训练路径和标签，因此不能把全部差距解释成“不知道任务编号”的代价。新方案在共享观测头的基础上增加检索残差后，四个单元的误差又下降了约 3%–6%。这说明局部残差有用，但单独依赖库行残差仍弱于两者组合。

四个单元各冻结了一个原纠正器对照和一个替换臂，共八个臂；每臂十个任务、初始状态 20–29，共 100 回合。防停滞规则、升级规则、客户端参数和纠正强度保持配对一致。π0.5 长任务保留原升级规则，其余三个单元保持原防停滞栈。预期替换臂成功率接近或略低于同批对照；是否接受，以已写好的闭环判断标准为准。

已完成十项结构与数值测试、十六项真实 CPU 插件自测，以及标准控制程序的本地计划检查。全部通过。生成的臂使用标准数据仓库，没有另建服务仓库，也没有自定义控制程序。没有启动同步、闭环、服务器、工作进程、GPU 或网络任务。没有查看初始状态 30–49 的实验结果或接触禁用运行目录。

## Frozen implementation

Selected `shared_row1`: a 207-input shared observation head (768 random Fourier features, ridge alpha 100), plus a 60-output table indexed only by the already retrieved library rows (ridge lambda 1). The final ten-control motion correction has strength 0.5. The gripper, remaining controls and padding channels are unchanged.

The correction head receives two projected visual keys, eight robot-state values, the cached 10×7 action chunk and capped elapsed decision index. It receives no task ID, task one-hot, init ID, task name, per-task scale or per-task head selector. One numerical artifact is fitted per model/suite/library cell; the configuration is identical across cells. The row table is permutation-equivariant with the retrieval row numbering. It stores no task map or task-specific thresholds.

This guarantee concerns the **new correction layer**. The requested existing retrieval, guard and escalation code is preserved; the four controls intentionally retain the old per-task heads. The full inherited stack should not be described as universally free of task metadata.

`RowCorrectedCache.os_synth` applies the correction where the judge actually obtains an action and remembers the corrected anchor for blind continuation. The separate direct-query path applies it once too. Fit-time checks reject attribute collisions before source state is copied, validate the training split and input dimension, remove the old head fields, and preserve judge `burst=0`. `TaskFreeEsc`, `TaskFreeStack`, and `TaskFreeStackGroot` reuse the corresponding existing stack query/reset behavior.

## Evidence and handoff

- [DATA_ANALYSIS.md](DATA_ANALYSIS.md): split, training objective, selection, metrics, uncertainty and limits.
- [PREDICTION.md](PREDICTION.md): frozen at **2026-10-02 10:37:06 UTC**, before artifact freeze/emission at **10:38:16 UTC**; point forecasts and screen criteria.
- [HANDBACK.md](HANDBACK.md): exact coordinator source staging and standard plan/sync/chain commands.
- [FROZEN.json](FROZEN.json), [H100_SOURCES.json](H100_SOURCES.json), [H100_SOURCES.sha256](H100_SOURCES.sha256): artifact/source provenance; 66 experiment Python source dependencies, four new serving files.
- [results/unit_tests.log](results/unit_tests.log), [results/plugin_selftests.json](results/plugin_selftests.json), [results/standard_control_plan.log](results/standard_control_plan.log): local passing evidence.
- [INTEGRITY.json](INTEGRITY.json): final file inventory and SHA-256 seal.

Run root: `/home/weiland/trace_runs/os_closed_loop/r09_astra_r6/`.
The standard plan contains 102 files / 23,870,705,656 bytes (22.231 GiB); existing assets may already be present remotely. Source staging and actual remote execution remain coordinator work. No remote compatibility or closed-loop success claim is made from these local checks.
