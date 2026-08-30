# Session Handoff — Dispatch Surface（Rev 2 第二计划 Code 完成，待 G2；2026-08-29 晚）

> **当前状态一句话**：Phase 0 已跑完并裁决（Decision Gate A 未全部满足，不进 C）；codex 与执行方就"接下来做什么"达成共识；第二计划（budget-mixture estimand / dense threshold grid / 完整裁决 power MC / v 离线指标 / H1-only gate / fresh-init C 与封存）**G1 APPROVED（R3）→ Code 完成 → 待 G2**。G2 通过前：**不 emit tgrid 矩阵、不 rollout、不 materialize P/C**；commit/push 需 owner 明确指示（**不加 AI 署名**）。
> 本文只写「状态 / 下一步 / 坑」。推导与评审史在 §0 的文档里。

## 0. 权威文档（按需读）

| 文档 | 用途 |
|---|---|
| [`dispatch_surface_rev2_confirmation_plan.log.md`](dispatch_surface_rev2_confirmation_plan.log.md) | **第二计划（Rev 4，G1 APPROVED R3）**：§3 设计（3.1 budget-mixture LP、3.2 tgrid、3.3 amendment analyzer + C roster 选择器、3.4 power MC、3.5 v 指标、3.6 confirmation discipline/analyzer、3.7 fresh-init/seal、3.8 Action Cache）、§6 测试、§7 放行顺序、§8 BLOCKING_TODO 挂接、§9 三轮 G1 review log、**§10 Code 记录（交付清单、供 G2 审的实现决定、验证证据）** |
| [`dispatch_surface_rev2_protocol_draft.md`](dispatch_surface_rev2_protocol_draft.md) | Rev 2 协议；**§13 FROZEN（G1 R3）**：分式混合 LP estimand、H1 唯一 primary、pre-C gate、fresh pool/seal/unseal、Action Cache record；之后只允许 `amendment_result` 填输出值 |
| [`dispatch_surface_rev2_phase0_result.md`](dispatch_surface_rev2_phase0_result.md) | Phase 0 结果（§0–§8）、codex 裁决 §9、执行方核验 §10、共识 §11/§12、接收 §13 |
| [`../docs/iclr/ICLR_PAPER_BLOCKING_TODO.md`](../docs/iclr/ICLR_PAPER_BLOCKING_TODO.md) | owner/codex 的论文 Gate（P0-A…E）；§1.3 已改为分式混合构造、无 Pareto 剪枝；fresh C 挂在其后 |
| `exp/dispatch_surface/config/confirmation_freeze_record.json` | G1 冻结的三份文档 SHA + 全部常量（测试断言与代码一致） |
| [`dispatch_surface_rev2_phase0_plan.log.md`](dispatch_surface_rev2_phase0_plan.log.md) / [`dispatch_surface_rev1_aprime_result.md`](dispatch_surface_rev1_aprime_result.md) | Phase 0 工具链 / Rev 1 负结果（永久保留） |

## 1. 拓扑与运行时（照着做）

```
timan107 (client, 48 worker, 8×1080 EGL)   ──►  weilandserver (server :23150 直连, 4090, 4 replica)
/scratch/zixuans8/openpi_dispatch                /data/openpi_dispatch ; tmux srv0 ; /tmp/srv0.log
./precheck_t107.sh <suite> <layer> [extra]       tether exec 必须 export HOME=/home/weiland
产物 /tmp/dsp_precheck/<suite>_<layer>/          /tmp/dsp_shared/ 三机同路径（本地 WSL 也有）
```

- 两台远端是**独立克隆**：改完代码 tar + `tether push` 到 `/tmp` 再解压，逐文件 `sha256sum -c`（timan107 allow_roots 只有 /home /tmp /srv）。上次同步的是 `b1609b9` 的 30 个文件；**本次 Code 的 ~20 个新/改文件尚未同步到任何远端**（G2 通过后再同步）。
- `/tmp/dsp_shared/<suite>/rev1_discipline/`（归档包副本）、`/tmp/dsp_shared/<suite>/exploratory/{sv,s0}/`（7 个正式导出）、`/tmp/dsp_shared/config/precheck_<suite>_exploratory/`（Phase 0 矩阵）三机一致；`/tmp/dsp_shared/<suite>/lib.pkl` 只在 weilandserver（l10 1.04 GB，spatial 425 MB；**emit 必须在 weilandserver 做**，因为 emitter 要哈希 lib.pkl；tgrid emit 同理）。
- server 重启：`tmux new -s srv0 -d 'cd /data/openpi_dispatch && export HOME=/home/weiland && export PYTHONPATH=/data/openpi_dispatch/src:/data/openpi_dispatch && export OPENPI_SERVER_GPU_MEMORY_LOCK=0 && /home/weiland/openpi/.venv/bin/python scripts/serve_policy.py --replicas 4 --replica-spawn-batch 2 --port 23150 policy:checkpoint --policy.config pi05_libero --policy.dir /home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch 2>&1 | tee /tmp/srv0.log'`；ready 签名 `replica_proxy listening on 0.0.0.0:23150`。**server 现在仍开着**（等 owner 说关；关 = `tmux kill-session -t srv0` + `ps` 取 pid kill，勿 `pkill -f`）。
- 吞吐 ≈ 22.8 ep/min（anchor 全推理臂 ≈ 17）；Phase 0 2700 ep 用了 1 h 36 min。监控：timan107 `/tmp/dsp_phase0_health.sh`（L1）+ Monitor 条件触发（milestone/ALERT/STALL/DONE 才推）+ 20 min cron；rollout 脚本模板 `/tmp/dsp_phase0_rest.sh`（串行多 launch，末尾 echo DONE 签名）。
- 数据回拉：`$CLAUDE_JOB_DIR/tmp/pull_phase0.sh <suite>`（tar → pull → 解压到 `exp/dispatch_surface/data/aprime_rev1/<suite>_exploratory/` → sha 核）。本地分析用本地归档包。
- tether 坑：`pull` 固定 5 min 上限（大文件不要拉，改 tar 或反向 push）；`exec` 本地 exit 0 ≠ 远端跑完；同一脚本里别出现要 grep 的字面量（`"[s]erve_polic""y"` 拆法）。

## 2. 已完成（2026-08-29）

- **Phase 0**：两 suite anchor A-4 通过（全 MISS，解析成本 67.518595；成本全部解析、无实测计时）；l10（exact-cost estimand）`[41.7,47.7]`，H1 +0.072 q05 +0.022 LOTO 全正但冻结口径功效 0.26（−1 约定占 89% 方差），H2 +0.041 功效 0.13 ⇒ A-6 ❌；spatial A-2 负、支配剪枝致 H1/H2 缺支撑 ⇒ A-5 ❌。数据 `data/aprime_rev1/<suite>_exploratory/`（journal/per_step/ledger + `cost_map_frozen.json` l10 `da17c198…`/spatial `1c2004ae…` + `phase0_outcome_design.json`）。**永久保留、不覆盖。**
- **codex 裁决 + 执行方核验**：六项事实全部独立复现；budget envelope 诊断表八个数一致；被支配臂可成为最优基（反例 A/B/C @B=30：全臂 0.9607 vs 剪枝 0.6111）；estimand 改为 **分式混合 LP**（直线 hull 使 l10 H1 低估 0.94 pp：0.0722 → 0.0816）。
- **第二计划 G1**：R1 7 blocking、R2 5 blocking、R3 APPROVED + reviewer 四处直接修订（task-plan/seal 无环、identity 走 task_uid→task plan、Simpson 只作抽样审计、术语统一）。执行方核验后已暂存接纳；freeze record 记录三份文档 SHA（plan `0b357190…`、协议 `59e05870…`、TODO `d26fee90…`）。
- **Code**（本地，未同步远端；详见 plan §10）：`analysis/budget_mixture.py`、`h1_verdict.py`（唯一裁决实现）、`estimator_version.py`；tgrid：`phase0_roster.py` 网格常量、`emit_precheck_yamls.emit_tgrid`、`run_precheck`（`exploratory_tgrid` + `confirmation` 层）、`phase0_discipline.validate_tgrid`、`tgrid_package.py`、`finalize_tgrid_package.py`；分析：`budget_cost_map.py`（cost-only）、`budget_outcome_design.py`（roster 选择器 + 止损 + `c_roster.json`）、`confirmation_power_mc.py`、`v_offline_metric.py`；确认：`generate_fresh_inits.py`、`build_confirmation_task_plan.py`、`seal_confirmation.py`、`confirmation_io.py`、`confirmation_discipline.py`、`confirmation_analyzer.py`、`action_cache_decision.py`、`register_confirmation_records.py`；`config/confirmation_freeze_record.json`；测试 `tests/dispatch_surface/test_rev2_confirmation.py` + `fixtures/budget_mixture_dev_stats.json`。
- **验证**：Phase 0 六个产物两 suite 字节等价；`frontier_hull.py` 未改（sha `99f8a962…`）；全量回归 **4594 passed / 45 skipped**（排除 review_tests、既有 `test_prebuilt_matrix_backend.py` 两个已知失败、单独跑的新文件）；ruff 全净；真实 l10 归档包 `v_offline_metric` 跑通（h10 OOF pinball SV−S0 −0.018 [−0.023,−0.013]，coverage 0.948/0.951）。新测试文件最终 **31 passed / 0 failed**（618 s；`pytest tests/dispatch_surface/test_rev2_confirmation.py`，≈10 min，模块 fixture 构建 R=10000 合成 design）。上一轮的 1 个失败已修：Rev 1 threshold 臂进 C roster 时 seal 缺 pair → 现从归档 yaml 读 pair。

## 3. 下一步（严格按 plan §7）

1. **G2**：已到达 gate（执行方已声明 `G2 gate reached. Please initiate a separate Review Authority session for code audit.`）。G2 修订先独立核验再接纳。
2. G2 通过 → 同步代码到两远端并 sha 核 → 在 weilandserver `emit_tgrid`（`--layer exploratory_tgrid --suite libero_10 --rev1-package-manifest /tmp/dsp_shared/libero_10/rev1_discipline/MANIFEST.json --table <归档表 data/aprime_rev1/inputs/libero_10/dispatch_table_fresh.jsonl 先同步过去；SHA 须等于 fit.sv input_digests.table> --template exp/gate_research/config/libero_10/eval/cp1_spatial_pool_16__grid3_vision_0@56_vision_1@25_robot_state@18__d1__fh40_ws40_quantile.yaml --library-pkl /tmp/dsp_shared/libero_10/lib.pkl --out-dir /tmp/dsp_shared/config/precheck_libero_10_exploratory_tgrid`）→ 分发 → timan107 `./precheck_t107.sh libero_10 exploratory_tgrid --dry-validate`。
3. **owner 放行** tgrid rollout（29 臂 × 300 = 8700 ep ≈ 6.4 h，`--workers 48`，可分批 `--arms`，同一矩阵/ledger）→ 回拉 → `finalize_tgrid_package` 到 `exp/dispatch_surface/data/tgrid_dev/libero_10/` → `register_confirmation_records tgrid`。
4. `budget_cost_map`（l10 带 `--tgrid-package-manifest`；spatial 不带）→ 记 SHA → `budget_outcome_design --budget-cost-map-sha256 … --out-roster c_roster.json`（l10）。verdict：`proceed_to_power` / `stop_before_C`（止损，停）/ `roster_overflow`（回 G1）。
5. `confirmation_power_mc`（先 `--smoke` 核时长；formal 在 weilandserver CPU `--workers 80`）→ N；60 不过 ⇒ `underpowered_stop` 交 owner。
6. `v_offline_metric`（可并行）；owner 签 Action Cache decision record（schema 见 `action_cache_decision.py`；`inclusion=yes` 需独立 G1/G2 包）；写 `amendment_result` artifact（输出值）；BLOCKING_TODO P0-A/B/D 勾选证据。
7. `generate_fresh_inits`（weilandserver 有 LIBERO 环境：`--pool P` / `--pool C`，`--state-dim 47`，`--apool-dir exp/common/data/db_init/libero/libero_10`，`--bddl-root <libero bddl 目录>`，`--exclude-manifest` 做 P↔C 互斥）→ timan107 重生成 `compare_manifests` 相等并写 `cross_machine: {verified: true, problems: []}` → 真实 round-trip smoke → P pilot（anchor 100 ep，`|SR−0.847| ≤ 10 pt`，一次性，pilot record `{arm,pool_id:"P",sr,attempt:1}`）→ `build_confirmation_task_plan` → `seal_confirmation seal`。
8. **owner 放行 l10 C**（`run_precheck --layer confirmation --seal … --task-plan … --pool-dir <C pool> --trials N --replan-steps 5`）→ `confirmation_discipline` → `seal_confirmation unseal` → `confirmation_analyzer`（无 unseal 只能 `--cost-only`）。H1 fail ⇒ 停线。

## 4. 等 owner / codex 定夺

- G2 放行；tgrid rollout 放行；Action Cache `inclusion` 取值；C 放行；server 何时关；何时 commit/push。工作区未提交：docs/iclr（TODO 新建 + README）、logs 五个文件、exp/dispatch_surface 20 个新/改文件、tests 两个新文件。
- 论文 framing：主张 = "经验风险校准的三档 dispatch surface"，`v` 为辅助特征（H2 非 gating）；"tuned threshold" 只有 dense grid 后才能用。

## 5. 坑与勿误判

- **分式成本**：mixture 成本是 `Σp T / Σp D`，不是 `Σp (T/D)`；任何"(cost,SR) 上 Pareto 剪枝/同成本去重"都错；被支配臂可进最优基。旧 `frontier_hull.py` 只供 Phase 0 复算，不得改。
- **两阶段解封**：`budget_cost_map` 不 import `budget_mixture`/outcome 模块、不读 success/status（AST 锁）；`budget_outcome_design.run` 强制 `replicates=10000, seed=20260829`；`design()` 本身 R 无关（测试用）。
- **扫描分片**是生产路径（K=32 时 11 ms/replicate）；全量 O(K³) 枚举只作参照；Simpson 只审计（G2 随机集、full sample + ≤100 replicate、power inner-0），差 > 1e-8 fail closed。
- **C 的身份**：per-step 行 schema 不变；`orig_init_state_idx` 必须 null（`EpisodeTask` wire 是 `dataclasses.asdict`，None 可透传）；identity 靠 `task_uid → confirmation_task_plan.json`（无 seal 字段）+ ledger 的 seal/task-plan 双 SHA；surface 臂 artifact 按内容绑定（yaml 内是历史绝对路径）；Rev 1 threshold 臂的 pair 从归档 yaml 读。
- **seed**：`np.random.seed` 只收 32 位 → authority(256 位) → `SeedSequence` → `uint32`；attempt a=0 base + 1..4 retry；failed/collision 占位；P 10/10、C 60/60 无余量。
- **runner**：`--layer confirmation` 需 `--seal --task-plan --pool-dir --trials N`；`--arm-matrix` 对 confirmation 不用；tgrid 走 `precheck_t107.sh libero_10 exploratory_tgrid`（`--trials 30` 冻结）。
- **测试**：新文件模块 fixture ≈10 min；合成 threshold 网格成功率故意弱于 SV（否则 verdict=stop_before_C，confirmation 链会 skip）。既有 `tests/exp/test_prebuilt_matrix_backend.py` 两个失败是历史遗留。
- `tests/review_tests/` 对执行方密封；Rev 1 冻结项（配额、δ\*、单价、`--trials 30`）不动；`exp/` 不放设计文档；`logs/archive/` 不删。
- 远端保留物：`/tmp/dsp_shared`、`/tmp/dsp_precheck`（含 Phase 0 原始数据）、`/tmp/srv0.log`/`srv0_0a.log`、健康/启动脚本；timan107 有来历不明的 `/tmp/verify_gate.py`，没动。
