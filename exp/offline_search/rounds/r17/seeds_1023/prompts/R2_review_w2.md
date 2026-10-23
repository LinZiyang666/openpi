# R2：独立审查 W2（π0.5/GR00T LIBERO 新子集种子）

先读 seeds_1023/prompts/COMMON.md（必须遵守：本机禁止任何运算，只读/改文件和 py_compile，不提交）、seeds_1023/DESIGN.md、W2 任务书 W2_vla_libero.md。

## 审查对象（W2 的改动，未提交）
`git diff -- exp/offline_search/rounds/r15/descent/fits.py exp/offline_search/rounds/r17/filter_failed_knob/q3_port_1020 exp/offline_search/rounds/r17/filter_failed_knob/formal_a_queue.py exp/offline_search/rounds/r17/filter_failed_knob/vla.py` 和新目录 `exp/offline_search/rounds/r17/seeds_1023/vla/`（fit.py、build.py、check.py）。

## 只审影响数字或判定、或会让批次跑挂的问题。重点核实：
1. 空任务（库子集里某任务没有成功局）钉 c10：在旋钮关和各档批次里，c10 是否保证该任务每次决策都调用模型、从不读那张未过滤的占位表（顺着 plugin/knob/awm 代码确认）；声明与实际不符时是否必报错。
2. `seeds_1023/vla/fit.py` 的 `old_gate()`：旧门放行只在拟合进程内、拟合后剥掉 judge，产出的 fit 里不含旧门对象；服务端加载路径仍禁旧门。
3. `row_episodes` 与旧 `schedule_episodes` 在种子 20261002 上给 ScheduleReplay 的输入是否等价（ep/task/n_slots/偶数步）。
4. Q3/PERIODS：带种子路径后「已存在就复用」只在同种子内生效；PERIODS 合并不会把旧种子的周期用到新种子格。
5. 建批：R15 单位（全步）与 R16 组（减步，24 entry）的 plugin_args 开/关只差 `--os-blind --os-policy-tail --os-policy-tail-blocks 1`，fit 路径指向本种子；分析键（`q3_port_1020/analysis.py` load_job key）含种子与开/关；单位/根名与现有不冲突；参照选择（关 `fa_lo_*_p5_q`，开 H100 `fa_lo_{gr_f,pi_r,gr_r}_p_q` 与现有）正确。
6. 库大小只做 50/100/200；两个种子都覆盖；kept 列表按种子算。

## 做法
- 明确的 bug 直接在上面这些 W2 文件里修（保持 W2 的结构与风格，改动最小），每处写清原因。
- 不确定或需要设计决定的，列出来不改。
- 交付：发现清单（file:line、严重性、是否已修、一句话理由）＋ COMMON.md 要求的交付。
