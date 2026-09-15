# online_rit — 分析产物说明

计划：`logs/online_rit_groot_plan.log.md`。本目录只放分析脚本与报告；运行数值一律在 `../data/<suite>/`。

| 文件 | 内容 |
|---|---|
| `signal_check.py` | M1 Q1 信号检验（`data/<suite>/offline/signal_check.json`） |
| `results.md` | 实验报告（M1/M2 结果填入；未跑前为结构占位） |

流水线（每 suite）：

1. `library_prep scales` / `s3b` / `pools`（主 venv，CPU；pools 写 adapt/terminal/smoke 三池、父池 SHA/映射 manifest 与三份 apool 记录）
2. `ops/run_table.sh` → 先 `build_disagreement_table --parity-only`（200 行 parity + 两噪声地板门，非 PASS 停），再 `--parity-gate` 建 d 表（岛 venv，GPU）
3. `analysis/signal_check.py --table ... --out data/<suite>/offline/signal_check.json`（Q1 门；calibration 人群）
4. `fit_init_curves.py knots --table --out data/<suite>/offline/knots.json`（calibration fit 半分位结点 + 分数段）
5. `replay_sim.py --table --knots --delta --gate-theta` → `data/<suite>/offline/replay.json`（估计器门，真实 gate 逐事件）
6. `fit_init_curves.py init --signal --parity --replay ...` → `init_state.json`（通过 `<table>.record.json` 核验 parity→表→Q1/knots/replay 的无环链与实际尺度身份）；`fit_init_curves.py rprime` → `rprime_fit.json`
7. `bench_fb_cost.py --mode eager --checkpoint ... --library-pkl ... --template-yaml ... --scales ... --knots ... --gate-theta ... --out ...`（岛 venv，v2 实测阶梯 + 完整反馈/CPU/文件成本）→ `data/cost/<hw_mode>/cost.json`
8. `ir_replay.py`（online fm1/fm0；`--rprime` 自动按无反馈计费）→ δ 阶梯与可达目标
9. `emit_online_arms.py --run-tag formal` → `config/<suite>/arms/*.yaml` + 四正式矩阵（rprime / frozen / online_a500 / online_adapt，各带 cohort）；smoke 使用 `--smoke --run-tag smoke`，只产 O-init/O-cold、每任务 1 集的 smoke cohort
10. `ops/launch_server_online.sh` + `ops/launch_clients.sh <matrix>`（按 cohort 绑池/trials/单端点/并发 1）；O-cold 结束后 `pick_terminal_state.py --state-dir ... --source-yaml ... --out-yaml ... --terminal-copy ... --data-dir <adapt_run> --pool-manifest ...`，输出冻结 yaml 及 `.matrix.yaml`，用 terminal 池记录/目录和同一 manifest 启动
11. `aggregate_online.py --ledger --knots [--pool-manifest --pool-key] [--pair A,B]` → `data/<suite>/<run>/aggregate.json`

画图脚本按 owner 规则不入库。

真件测试在岛上设置 `ONLINE_RIT_CKPT`、`ONLINE_RIT_LIBRARY`、`ONLINE_RIT_SCALES`、`ONLINE_RIT_GATE_OUT`，执行 `python -m pytest tests/cache/groot/test_online_rit_real_model.py --run-manual -q`。它检查 eager、64 条快照、batch1/2/3、RNG、capture 和实际尺度有限性；固定 conditioning 的压力测试不称 d_self，原条件数值地板仍由 M1 parity/打标输出。

成本 reader 对正式运行要求 `online_rit_cost_v2`，缺项或非 eager 账本拒绝。`stage3_ladder_ms` 直接计价，head/step 线性拟合仅诊断；CPU 检索/门/judge 与 commit 使用冷/半满/满窗的已测最大值，文件快照按 200 学习批次及集末摊销。因此这是统一参考计价，另报实际端到端延迟；不称每个在线请求的精确实测成本。`ir_replay --frozen` 可单独检查冻结成本；R′ 自动 none，FM-0 仍有执行反馈开销。

`--pair` 使用 aggregate 的结构化 episode 记录（suite、父池 SHA、task、原初态、success、逐档违规计数/分母）；旧 uid→bool 输出缺少身份，拒绝直接配对。每臂输出决策前 cuts/支持率的 `dynamics`，配对报告同时包含 SR 和风险区间。`launch_clients.sh` 的 cohort 参数不接受额外参数覆盖；子池必须提供 manifest，实际池由 `run_gtp` 重算内容摘要并核验映射。
