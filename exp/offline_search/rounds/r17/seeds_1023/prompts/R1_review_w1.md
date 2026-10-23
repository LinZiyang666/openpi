# R1：独立审查 W1（R17 线新种子产物与合并建批）

先读 seeds_1023/prompts/COMMON.md（必须遵守：本机禁止任何运算，只读/改文件和 py_compile，不提交）、seeds_1023/DESIGN.md、W1 任务书 W1_r17_seeds.md。

## 审查对象
提交 0c987196（`git show 0c987196`）：新模块 seeds_1023/{inputs,build,workflow,regress}.py，以及 filter_failed/stage.py、filter_failed_knob/{build,common,evidence,formal_a_analysis,runtime}.py、knob/common.py、longseg_1017/{recalibrate,staging}.py、longseg_off_1022/{build,fit500,workflow}.py、ops/{cache_knob,cache_study}.py 的改动。

## 只审影响数字或判定、或会让批次跑挂的问题。重点核实：
1. 新种子子集：LIBERO 每任务 `default_rng(种子+任务号).permutation` 取前 N/10、嵌套；DP/Fast-WAM/Cosmos 库大小 50/100/200/300/400；MT50 用 seed 9/10 库自带 subsets（size_5..50 = 250..2500）。
2. 每格产物按新库重新标定：C10 拟合 + 段头 Q3 + 周期（只用成功局）；C5（仅长执行段模型）拟合 + C5 Q3、周期沿用同种子 C10；Cosmos C16、π0.5 MT50 C5 无长执行段组；kept = 视图局 ∩ 成功局；selection 文件按 fit digest 放在 fit 旁边、并在 assets/cache_inputs 里随产物推送。
3. 合并 campaign：组名/YAML 名/分析键/evidence 键都带种子和段长，两种子 × 库大小 × 段长不会互相覆盖；每个 YAML 的 fitted/stall/store/selection 都指向本种子本格；fit_identity 重算正确。
4. 两步：LIBERO 判格用 formal_a_queue.classify；MT50 判据（2,500 完整 + 四难度组均值不低于同卡同步数纯推理）实现无误；只给要调的格建 k50..k10；参照选择正确（`longseg_off_1022/REFERENCES.json`、Cosmos PURE；GR00T/FW MT50 减步若只有 pod 参照要明确报出来）。
5. runtime.py 惰性 selection：过滤批次缺 selection 必报错；未过滤批次不受影响；periodic_gap==0 的行为与旧代码一致。
6. 旧种子回归（regress.py）比对的对象与判定是否真能证明新代码在旧种子上逐位复现。

## 做法
- 明确的 bug 直接修（保持 W1 结构与风格，改动最小），每处写清原因；修完对改过的文件 py_compile。
- 不确定或需要设计决定的，只列出来。
- 交付：发现清单（file:line、严重性、是否已修、一句话理由）＋ COMMON.md 要求的交付。
