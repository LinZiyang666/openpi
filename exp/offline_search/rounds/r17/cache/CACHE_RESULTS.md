# R17 六尺寸缓存研究：封存 lane 增量快照

UTC：2026-10-11T01:25:29.627427+00:00。仅分析显式指定尺寸 [500]。

F500 与 Fsize 各固定40比较/5%；不随已到达 lane 缩小族，不声称合并族5%。中间尺寸与敏感性均为描述性。
Cosmos ch/chr/chg/chrg 分别复用 c/cr/cg/crg。CI为逐点双侧95% CP；差值LB为单侧。
配对按原任务/初态字节；近邻固定剔除32例（spatial剩489，10剩479）。没有PASS的lane不读取outcome。

## Lane 封存状态

| Lane | 状态 | 收据 |
|---|---|---|
| /data/mw_finetune/eval/csl10d | PENDING_SEAL | RECEIPT.json:FAIL |
| /data/mw_finetune/eval/csl10f | PENDING_SEAL | RECEIPT.json:PARTIAL |
| /data/mw_finetune/eval/csl10c | PENDING_SEAL | 未出现 |

当前 0/240 个不同组×尺寸×suite 可分析；未到齐不宣布研究完成。

## SR 与成本

| model | N | group | suite | cohort | n | SR% [CP95%] | fresh hit% | MISS/control | NFE/control | calls/ep |
|---|---:|---|---|---|---:|---|---:|---:|---:|---:|

## 配对比较（差值/LB单位pp）

| model/N/group/suite | cohort | reference | family | Δ | g/l | CP LB family | boot LB family | CP/boot LB未校正 | NI |
|---|---|---|---|---:|---|---:|---:|---|---|

## 两套件主判定

| family | model | group | NI（两套件） | 成本/采用 |
|---|---|---|---|---|
| F500 | DP | c | PENDING | PENDING |
| F500 | DP | cr | PENDING | PENDING |
| F500 | DP | ch | PENDING | PENDING |
| F500 | DP | chr | PENDING | PENDING |
| F500 | DP | cg | PENDING | PENDING |
| F500 | DP | crg | PENDING | PENDING |
| F500 | DP | chg | PENDING | PENDING |
| F500 | DP | chrg | PENDING | PENDING |
| F500 | Fast-WAM | c | PENDING | PENDING |
| F500 | Fast-WAM | cr | PENDING | PENDING |
| F500 | Fast-WAM | ch | PENDING | PENDING |
| F500 | Fast-WAM | chr | PENDING | PENDING |
| F500 | Fast-WAM | cg | PENDING | PENDING |
| F500 | Fast-WAM | crg | PENDING | PENDING |
| F500 | Fast-WAM | chg | PENDING | PENDING |
| F500 | Fast-WAM | chrg | PENDING | PENDING |
| F500 | Cosmos | c | PENDING | PENDING |
| F500 | Cosmos | cr | PENDING | PENDING |
| F500 | Cosmos | cg | PENDING | PENDING |
| F500 | Cosmos | crg | PENDING | PENDING |
| Fsize | DP | c | PENDING | PENDING |
| Fsize | DP | cr | PENDING | PENDING |
| Fsize | DP | ch | PENDING | PENDING |
| Fsize | DP | chr | PENDING | PENDING |
| Fsize | DP | cg | PENDING | PENDING |
| Fsize | DP | crg | PENDING | PENDING |
| Fsize | DP | chg | PENDING | PENDING |
| Fsize | DP | chrg | PENDING | PENDING |
| Fsize | Fast-WAM | c | PENDING | PENDING |
| Fsize | Fast-WAM | cr | PENDING | PENDING |
| Fsize | Fast-WAM | ch | PENDING | PENDING |
| Fsize | Fast-WAM | chr | PENDING | PENDING |
| Fsize | Fast-WAM | cg | PENDING | PENDING |
| Fsize | Fast-WAM | crg | PENDING | PENDING |
| Fsize | Fast-WAM | chg | PENDING | PENDING |
| Fsize | Fast-WAM | chrg | PENDING | PENDING |
| Fsize | Cosmos | c | PENDING | PENDING |
| Fsize | Cosmos | cr | PENDING | PENDING |
| Fsize | Cosmos | cg | PENDING | PENDING |
| Fsize | Cosmos | crg | PENDING | PENDING |

## 限制与待到达

NFE是动作头计算量，不含视觉/检索/修正器；零NFE不等于零耗时。service无完整可比证据时不宣称墙钟加速。
Cosmos旧控制含timan108与timan107 renderer；JSON按旧控制节点分层，不能把跨节点差异解释为纯缓存因果效应。
Q3实际校准来自各尺寸B成功episode LOEO，不是独立B验证录制；门不可用任务在JSON逐组列出。

未到达 cell 清单与输入 SHA256 见 CACHE_RESULTS.json；CSV含逐比较界、成本原始计数；曲线未封存点留空。
本次运行没有GPU/模型/campaign操作，没有读取未选尺寸outcome。
