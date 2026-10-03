# 给负责人的说明

连续接管在旧的纯缓存底座上有收益，但**目前没有证据证明它能进一步改善现有最好方案，也不能说它比按节奏落后接管更好**。这一轮已准备好每个模型三组对照，尚未运行新的闭环实验。

π0.5 的成功率从 69% 到 80%，逐对比较是新增 11 次成功、没有新增失败。不过，其中 6 次新增成功根本没有触发接管。真正属于“缓存失败、实验中触发接管、最后成功”的是 5 次，占实际触发且缓存失败的 25 次的 **20%**。GR00T 的对应比例是 **8/32，即 25%**；其总成功率与随机安排三次调用相同，开销更高。不能把总分差全部算作接管救回。

两种模型此前实际接管后救回的案例，在当前最好方案中都已经成功。连续接管通常也偏晚：第一次告警的中位数分别是第 58、62 个决策，约第 290、310 个控制步。因此，我更看重另一个小改动：沿用现有的节奏落后告警，只把连续接管限制为十二次策略调用，之后仍让原保护机制正常工作。它没有减少保护机制的调用，也没有改变观察频率。

下一步应在同一批次比较原方案、连续接管方案、十二次接管方案。配置、预测、核验程序和交接命令均已准备完毕。十二组本地插件检查全部通过，包括强制触发；没有启动服务器、工作进程、同步或远程任务，没有读取保留初始状态的结果。

## Frozen comparison

Only LIBERO-10, 50-demo library, both models, tasks 0–9 × evaluation inits 20–29. Six arms, no task-dependent takeover parameter or hard-task input.

| Model | Arm suffix | Composition |
|---|---|---|
| π0.5 | `control` | Exact copied r3c artifact: only-no-progress + `CorrectedCacheJ` .5 + persistent lag12/deadline80 escalation |
| π0.5 | `latch` | Same guard/corrector; replace escalation with frozen 2b detector + `LatchedGate` |
| π0.5 | `pace12` | Same guard/corrector; imported pace trigger, one entry, at most 12 fresh policy calls |
| GR00T | `control` | Exact copied r3c artifact: only-no-progress + `CorrectedCacheJ` .5 |
| GR00T | `latch` | Add the frozen 2b detector/latch |
| GR00T | `pace12` | Add the same pace-triggered 12-call takeover |

The extra variant family is `pace12` only. The latch uses its original model-level threshold (.49085291028022776 / .34257609844207804), two high fresh scores after decision 12, minimum three calls, two scores below half threshold to release, six-anchor cooldown, and 12 total takeover calls. All calls retain the ten-control policy-tail contract. Guard-overlapping calls consume the takeover budget; ordinary guard calls remain unrestricted after it expires.

The prescribed frozen corrector already selects fitted heads by task ID. It is imported and preserved exactly, including those heads; this package adds no task-indexed head, routing, library, budget, strength, or threshold. Task identity remains available to the existing cache/judge machinery. The new monitor/gates never receive a task difficulty label. This distinction is explicit rather than describing the whole inherited stack as task-blind.

## What the evidence supports

The 2b paired results reproduce the supplied summary. The latch-versus-three-call advantage is only +5/−2 for π0.5 (p=.4531) and +5/−4 for GR00T (p=1). Against random placement it is +8/−2 (p=.1094) and +5/−5 (p=1). The detector's selection advantage is therefore unconfirmed, even on the old cache base.

On the logged π0.5 leading-stack paths, `pace12` would first change the action source in six episodes after its cap. Under identical exogenous randomness, 90 successes are on unchanged paths and the six divergent outcomes are unidentified, giving only a pathwise [.90,.96] bound. This is **not** a new-run success estimate. GR00T has 21 potentially divergent episodes and a much less informative [.78,.99] bound. Standard logs lack camera keys needed for the learned monitor, so no exact latch replay on the corrected stack or latch-versus-escalation SR claim is possible.

## Validation and handoff

- **9 unit/regression tests pass:** identity admission, prediction ordering, hashes, frozen-field checks, correction active in `os_synth`, distance-input parity, reset/clone isolation, cap/deadline, and guard preservation.
- **12/12 real CPU plugin selftests pass:** six production and six forced probes; 576 decisions, 288 fresh looks, 288 blind decisions, 120 policy-tail checks. New takeover arms each trigger on all four connection/episode resets. Control forced probes use the existing r3c debug grasp hook; their production artifacts are unchanged.
- Wrapper `fit` verifies 16 pinned source modules and four frozen assets per model, compares all 60/61 inherited π0.5/GR00T judge fields except the explicitly swapped base/name/profiler/fit-info, asserts `judge.burst == 0`, `gm_max_calls == 0`, and `CorrectedCacheJ` at strength .5. Wrapper state uses `tk_*` and lives outside the judge.
- Local control plan passes: **59 assets, 17,127,761,562 bytes (15.951 GiB logical)**. A small adapter supplies a parser-filtered, identity-only metadata store under the new run root; its remote destination is also isolated to this run.
- **69 H100 source files with full SHA-256** are listed in `H100_SOURCES.sha256` and `H100_SOURCES.json`. No source install or asset sync was executed.

Predictions were timestamped at **2026-10-02T08:59:55.754479+00:00**, before arm emission; `FROZEN.json` pins the prediction hash. The final integrity inventory also pins the serving source, six fit artifacts, configurations, analysis and tests. See `DATA_ANALYSIS.md` for definitions and limitations and `HANDBACK.md` for exact coordinator commands.
