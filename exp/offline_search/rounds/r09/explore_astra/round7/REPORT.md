负责人说明：这轮已经备好两个只用于 GR00T 的无任务编号纠正器，并为两个测试单元各放入同批的原纠正器对照，共六个方案、六百个计划回合。预测先写入并冻结，随后才发包。没有启动评测、同步远端或改变现有进程。

主要发现是共享纠正头容量不足，训练轨迹种类较少也可能有影响；没有证据表明差距来自抓手正负号、32 维补齐通道或动作长度错误。两个单元的检索行都有训练数据，但这不等于所有状态都覆盖充分。多出来的模型调用全部由原来的“无进展”保护触发，主要集中在新方案失败而对照成功的回合；这支持轨迹质量变差的解释，但日志没有在线标准答案，不能把每次调用都解释为前一个动作错误。

第一方案扩大共享头，保持一半纠正强度。第二方案使用同一套拟合参数，在检索动作更一致时适度加大强度。离线长任务／Spatial 的动作误差相对对照分别为第一方案 +0.8%／−2.4%，第二方案 −6.8%／−6.5%。第二方案的 Spatial 平均误差较低，但只有 46/100 个回合的误差低于对照。闭环成功率是否补回来，仍需这次同批实验判断。

# Frozen package

| Cell | Same-batch per-task control | Capacity candidate | Confidence candidate |
|---|---|---|---|
| GR00T LIBERO-10-50 | `r9a7_groot_l10_50_control` | `r9a7_groot_l10_50_capacity` | `r9a7_groot_l10_50_confidence` |
| GR00T Spatial-50 | `r9a7_groot_spatial_50_control` | `r9a7_groot_spatial_50_capacity` | `r9a7_groot_spatial_50_confidence` |

One shared same-batch control per cell serves both candidate comparisons. Controls are byte-identical to round6. Both candidates use a 3072-RFF, alpha-100 shared head with 231 observation/action inputs and a retrieval-row residual table. Fixed strength .5 versus `.5+.25/(1+dispersion/train_median)`. No task id, one-hot, task head, task selector or task threshold is introduced. Both models fit only 0–19; all emitted manifests select exactly tasks 0–9 × inits 20–29.

Round6 loss diagnosis: long −6 pp, concentrated in tasks 0/8; Spatial −5 pp, largest on task 9. All extra calls are no-progress calls. Long offline deficit is greatest after release and during carry; spatial offline deficit is stronger during carry and in the farthest retrieval-distance bin. Gripper-event deficits do not explain the pattern on their own. See [DATA_ANALYSIS.md](DATA_ANALYSIS.md) for task/phase/distance tables and causal limits.

# Validation and handoff

- Fifteen unit tests passed: identity-first rejection, train/eval separation, task-input poison, row permutation, padding/tail invariance, strength bounds, exact control copies, unchanged non-corrector state, and judge correction/anchor behavior.
- Twelve standard-plugin CPU selftests passed: all six production artifacts plus forced-call fixtures; each covers two connections, two training episodes, 48 decisions. Forced fixtures exercise policy miss/tail paths; emitted arms retain `max_calls=0` and no forced trigger indices.
- The standard controller `plan` passed with six arms, 44 files, 7,375,204,959 bytes. Standard store `/home/weiland/trace_runs/offline_search_store`; standard emitter and standard controller; no serving-store subset or control adapter.
- `H100_SOURCES.json` and `H100_SOURCES.sha256` record the 69-file experiment source import closure; `H100_NEW_SOURCES.sha256` lists the four new serving files. Static inventory and local hashes are complete. Remote hashes were not checked.
- Local correction latency: median 1.24–1.31 ms; p95 1.35–1.55 ms. No H100 timing or round7 closed-loop outcome is claimed.

Run root: `/home/weiland/trace_runs/os_closed_loop/r09_astra_r7/`.
Prediction: [PREDICTION.md](PREDICTION.md), frozen before emission. Source staging, checks and coordinator-only standard sync/chain commands: [HANDBACK.md](HANDBACK.md). Local integrity seal: `INTEGRITY.json`.

This is an iterative development screen on the reused 20–29 evaluation population. Forecasts remain conservative: capacity −2 pp in each cell; confidence −1 pp long / −2 pp spatial against the new same-batch controls. No claim of closed-loop parity is made before those runs.
