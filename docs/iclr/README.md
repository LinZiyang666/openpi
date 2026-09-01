# iclr/

ICLR 2027 投稿（TIER: experience-tiered inference）的论文工作文档：提纲、实验设计、评审博弈记录。文献书目见 [`../papers/`](../papers/README.md)。

> **状态（2026-08-30）**：owner 变更论文目标——旧 gate 文档（`ICLR_PAPER_BLOCKING_TODO.md`）与 `.old` 规划稿已删除；新方向见 `logs/dispatch_surface_sonly_pivot_report.md`（s-only 风险索引转向）与 [`latex/sonly_note.tex`](latex/sonly_note.tex)。

| File | Description |
|------|-------------|
| [paper_rethink_discussion.md](paper_rethink_discussion.md) | **现行讨论纪要（2026-08-22 起；同日晚整体重构为逻辑结构）**：owner 判决放弃旧 TIER 叙事回归 VLA cache 本身；novelty 叙事（A 冗余度骨架 + B 跨场景惊喜 + teacher 自身表征加粗点）；应用故事/经济账/题目；中心量 R(ε) 四节（三层定义/数学 v2/创新性核查/体量裁决）；**工作提纲 v0.1**（§5，吸收全部裁决）；章节-实验对照表（含 oracle 臂详解 + §5.3 实测复核）；defense 弹药库三线（小模型四道防线+regime 地图 / trained router / cheapening+warm-start 三方对照）；部署生命周期四阶段；Table 1 与 serving mini-bench 裁决；未决问题状态表 |
| [actioncache_response_plan.md](actioncache_response_plan.md) | **ActionCache（arXiv 2607.06370，concurrent work）攻防方案（2026-08-26）**：三路专家核实整合——逐项核实（key=VLM 输出、地板 45–53% vs 我们 14%、端到端仅 1.26×、LIBERO 被 NFE=1 裸基线支配）；ICLR 规则双保险（arXiv-only 豁免）；写作定位（concurrent and independent 门面 + 三层攻防分工 + RW 英文草稿）；12 条弹药 + 还功八点；**四臂对照方案（Arm1=激活 CP2，核心新增仅 ~3,000 ep）**含 2×4 courtesy 校准与六条预注册分支 |
| [latex/](latex/) 合作者材料四件套 | ICLR 模板英文 note（各配 PDF）：**redundancy_note**（R(ε) 全量形式化 + 实验呼应逐条）；**dispatch_note v3**（免训练分级调度：代理事件 NP 正名 + 耦合噪声可执行协议 + $\tau=0$ fallback 与有限样本条件 + episode 级同时共形覆盖 + split 职责/coverage 口径/start_t 约定，两轮外审修订）；**experiment_list**（E0–E13 状态表，E1/E2 同图成组）；**paper_outline**（提纲 v0.2 英文版含 E13 真机节） |
| [dispatch_defense_plan.md](dispatch_defense_plan.md) | **Dispatch surface 线攻防思路（2026-08-27）**：守住 novelty 的三条件（surface 实测增益 / CP2-AC→CP1-threshold→surface→+stateful 递进链 / 不声称动态去噪分配为新——D3P、DVAC 划界）；note v1→v2 七处数学修正记录 + 建模四修正；工程锚点与在线三档缺口；增益预检先行的行动序 |
| [redundancy_structure_fig.html](redundancy_structure_fig.html) | §5.3 冗余结构三 panel 草图（真数据自包含页 + [PNG](redundancy_structure_fig.png)）：episode 条带 / 粘滞散点 / run-length 生存曲线；配套讲法与裁决点见纪要 §10.2 |
