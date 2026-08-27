# Session Handoff — Dispatch Surface 执行期（compact 交接 v2）

> 2026-08-27 更新。**代码阶段已全部完成**：G1 五轮 APPROVED、G2 四轮 APPROVED（含 owner 越权轮的 reviewer 直接修复，已复核归档）、Verify 通过、commit `1985271` 已 push 到 `origin/Ziyang`。**接手后的工作 = 按 plan §7 预注册时序执行实验**，不写新功能代码。评审史不必重读（永久记录在 plan 的 Review Log）。

## 0. 唯一权威文档

`logs/dispatch_surface_plan.log.md` —— 设计、§7 执行编排、§4.5 机械 δ 规则、§4.6 臂表与序贯门控判据、止损点、G1/G2 全部裁决。数学背景才查 `docs/iclr/latex/dispatch_note.tex`；预检判读的论文归宿见 `docs/iclr/dispatch_defense_plan.md` §5。

## 1. 已交付的执行入口（全在 `exp/dispatch_surface/`）

按 §7 顺序：`split_init_pools.py` → `rebuild_dispatch_library.py` → `collect_query_cohort.py`（plan/launch/verify 三子命令）→ `noise_sensitivity.py` → `build_dispatch_table.py` → `fit_surface.py`（SV 再 `--s-only --frozen-record`）→ `power_sim_cost_blocks.py` → `emit_precheck_yamls.py` + `run_precheck.py` → `run_cost_bench.py`（两次 server launch）→ `analysis/analyze_precheck.py`。每个脚本 docstring 有用法；所有 provenance/契约失配都是 fail-fast，报错信息即诊断。

## 2. 执行时必然踩到的关卡（不是 bug，是设计）

1. **标定 yaml 尚未写**（`exp/dispatch_surface/config/` 空）。需要一份 `calibration_retrieval.yaml`：spatial16 builder + `weighted_score_sum_knn` + `preload_path` 指重建库 + `write_policy: never` + judge 用 `always_hit`。**其 keys/key_builder/search_strategy 段必须与 emit 用的 gtp 模板（`exp/gate_research/config/libero_spatial/n4_server/*.yaml`）逐字段一致**——surface artifact 的契约 digest 从标定 yaml 算，预检臂 yaml 检索段与它不同构会在 load 期被契约拒绝。
2. **官方 A-pool 目录**：`exp/common/data/db_init/libero/libero_spatial_apool`（50 init/task）。本机若无 → `exp/ablation_study/cache_size/materialize_apool.py` 物化（需 libero client env）。
3. **power sim 的 variance source** 是 schema 文件不是 CLI 数字：`{"schema_version": 1, "sigma_compute": <..>, "sigma_latency": <..>}`，σ 从 E0 成本方差先验推（`exp/data_authority/records/latency_bench__libero_spatial__executor_costs.json`）；seed/n_sim/n_boot 已冻结为模块常量，validator 会确定性重放。
4. **cohort 采集**：server `serve_policy.py --collect`（纯 teacher），client 走 `collect_query_cohort.py launch` 打印的命令（`examples/libero/main.py --cohort-plan ... --num-workers 1 --init-states-dir <C池>`），四字段身份 metadata 自动进 H5 attrs；`verify` 成功产 manifest 才算采完。
5. **预检/成本 launch 契约**：runner 要 `--replan-steps 5`（== artifact h_exec）且从 server metadata 读 `policy_fingerprint` 比对（server 启动时对实际 checkpoint 逐文件哈希，首启会多花 ~20s）；cost compute pass 要 server `OPENPI_MONITOR_LEVEL=SNAPSHOT` 且 metadata `stage_probe_backends` 三 stage 全 "cuda"，latency pass 要 OFF——**两个独立 server launch**，不能同进程切。
6. checkpoint = `pi05_libero`（`~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch` 先例路径）；重建/标定/拟合与 server 必须同一 checkpoint，否则 fingerprint 拒。

## 3. 冻结判据（执行期零自由度，違反 = 流程违规）

- 配额 5(D_lib)/5(fit)/10(cal)/30(test) per task；`--trials 30`；cal |E|=100。
- δ\*：P10–P90 去重 grid + init mod 5 fold + OOF 部署 verdict 评估 + 写死的选择/tie-break/fallback；s-only 必须 `--frozen-record` 继承。
- **止损点 A**（fit 期）：sparse 格梯 (12,6)→(8,4)→(6,3) 全笛卡尔 cell ≥8 穷尽、(A1) 违反率 >20%、accuracy 门全败、cal<19 —— exit 3，跳过闭环，负结果归档。
- **止损点 B**：power sim R=15 仍不足，或 Gate 1 不胜。
- 裁决 = analyzer 硬编码：Gate 1（D_sr p5≥0、D_c p95≤−5%、D_l p95≤0）→ Gate 2（ΔSR p2.5>0、Δcompute p97.5≤+5%、Δlatency p97.5≤0）→ 不胜即停，无 Gate 3；SV± 只描述。三分结论对应并入 SV / 记 "v 未证" / 线降级反哺纪要 §7.2。
- E5 第三项 = 预检后描述性分析，不参与任何选点。

## 4. 环境与预算

- 拓扑与监控按 `experiment-lifecycle` skill（server 侧 weilandserver 4090、client 侧 timan107 worker 等）；吞吐锚点 ~66 ep/min。
- 预算：重建库 ~10 min GPU；cohort 采集 150 ep 数小时（一次性）；标定表 ~1–2 GPU 时；预检 5–7 臂 × 300 ep ≈ 1 h；成本 bench 5 臂 × R × 10 ep × 2 pass（R=5 时 ~500 ep）。
- 实验产物义务（WA）：`exp/data_authority` 收编 + MANIFEST + `records/` 台账（重建库、cohort、预检、cost 各一）；结果 `analysis/analysis.md` 按 skill 模板（§5 反 narrative + §8 layout 硬要求）；判读后同步 `docs/iclr/dispatch_defense_plan.md` §5 与 `actioncache_response_plan.md` 状态行 + `docs/iclr/README.md`。

## 5. 已知无关事项（勿误判）

- `tests/exp/test_prebuilt_matrix_backend.py` 两个 bit-identical 断言在 HEAD 上**既有失败**，与本线零因果（score 路径无改动），另行排查。
- 工作区仍有其他会话的 unstaged 改动（robocasa365 / rl_router / docs/iclr 等），与本线无关，勿动勿提交。
- 本线不再有代码待写；若实验暴露缺陷需改 `src/`，按 WA 重新走流程（L1/L2 视范围）。
