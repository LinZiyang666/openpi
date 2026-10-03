# Trace 双模型对照采集：π0.5 × GR00T，纯 inference 与纯 cache 下发

> 2026-09-23 22:52 – 2026-09-24 02:23 CDT，weilandserver（4090）↔ timan107，8 组 × 500 集全部跑完。
> 数据根目录：`/home/weiland/trace_runs/dual_20260923/`（home SSD）。代码版本为 HEAD `2dc90d0`（本次运行没有改动任何代码）。

## 1. 实验设计

| 维度 | 取值 |
|---|---|
| 模型 | π0.5（`pi05_libero`，`pi05_libero_pytorch`）；GR00T N1.5（`ckpt_n15_libero_spatial` / `ckpt_n15_libero_10`，8 步去噪） |
| suite | libero_spatial、libero_10 |
| 集数 | 每组 500：官方 pruned_init 测试集（A-pool，10 任务 × 50）。A-pool rollup：spatial `0eeece46…`，l10 `52457a37…`，timan107 上重新哈希后一致 |
| 纯 inference 组（`_inf`） | gate `always_search`，threshold judge 阈值 +1e6，永远判 MISS。**下发**完整推理；检索和 top-1 缓存动作在 shadow 里算出并记录 |
| 纯 cache 组（`_cache`） | 同上，但阈值为 −1e6，每步都判 FULL_HIT。**下发** top-1 缓存动作；完整推理在 shadow 里算出并记录 |
| debug | `--trace-out <dir> --trace-build-cache`：每步的所有模块都运行并落盘，noise_action 全部记录。π0.5 另外打开 `record_model_images` |
| 拓扑 | 本机 4 个独立的单 replica server（`ziyanglin.com:23100-23103`），timan107 起 64 个 worker（每个 server 16 个），driver 为 `run_gtp` role=all |

两组配置只差 judge 阈值这一个数，gate、检索和库完全相同。

## 2. 配置（沿用历史上用得最多的检索配置）

| 模型 | 来源 yaml | key builder | 库 | keys 权重 (v0 / v1 / rs) |
|---|---|---|---|---|
| π0.5 spatial | `exp/gate_threshold_pareto/config/libero_spatial/eval/gtp_ws_sp_fh30.yaml` | `cp1_spatial_pool_16` | 权威版 `cp1_spatial_pool_16.pkl`（`36cd0f3b…`） | 0.0625 / 0.5 / 0.4375 |
| π0.5 l10 | `…/libero_10/eval/gtp_ws_l10_fh30.yaml` | 同上 | 权威版（`f13517ad…`） | 0.5625 / 0.25 / 0.1875 |
| GR00T spatial | `exp/libero_groot/config/rit/libero_spatial/source_template.yaml` | `cp1_groot_libero_spatial_pool_16` | `/data/libero_cache/libraries/libero_spatial/libero_spatial_sp16_S3.pkl`（`182ef39b…`） | 0.4167 / 0.3333 / 0.25 |
| GR00T l10 | `…/libero_10/source_template.yaml` | 同上 | `…/libero_10/libero_10_sp16_S3.pkl`（`e8e41965…`） | 0.5 / 0.4167 / 0.0833 |

四个检索段都是 `weighted_score_sum_knn`，top_k=1，d1；vision 用 cosine，robot_state 用 l2 距离取负（−d）；per_field zscore + tanh。yaml 里的 `to_similarity: {exp, tau: 1.0}` 在 per_field 模式下不生效（`score_normalizers.py` 只在 percentile / none 路径读 tau）。μ/σ 从来源 yaml 原样复制。生成器是 `exp/trace_dual/ops/emit_configs.py`，8 份 yaml 在 `exp/trace_dual/config/`，`arms.json` 记录了来源 sha。8 份 yaml 均通过 `load_cache_config`、`validate_effective_trace` 和 run_gtp 的 `validate_arms` 校验。

⚠ **π0.5 库换成了权威版**。主树 `exp/common/data/cache_artifacts/<suite>/cp1_spatial_pool_16.pkl` 是旧的 WSL 副本（sha `09cacb81` / `de51731d`）。逐条目比对后，两版内容相同，只是 NaN 因子在 pickle 里的表示不同，文件差 8 字节。本实验用的是权威版，已复制到 `libs/<suite>/`，所以 trace 里记录的 `library_sha256` 就是权威 sha。

## 3. 结果

| 组 | 成功 / 500 | 成功率 | 决策数 | 决策 / 集 | trace | MiB / 决策 | 运行时长（server 就绪 → 500） |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tr_pi05_sp_inf` | 493 | 0.986 | 10,798 | 21.6 | 42.7 GiB | 4.05 | 12 min |
| `tr_pi05_sp_cache` | 334 | 0.668 | 14,621 | 29.2 | 57.8 GiB | 4.05 | 14 min |
| `tr_pi05_l10_inf` | 422 | 0.844 | 29,406 | 58.8 | 115.2 GiB | 4.01 | 28 min |
| `tr_pi05_l10_cache` | 226 | 0.452 | 40,127 | 80.2 | 157.4 GiB | 4.02 | 37 min |
| `tr_groot_sp_inf` | 470 | 0.940 | 11,338 | 22.7 | 30.5 GiB | 2.75 | 11 min |
| `tr_groot_sp_cache` | 368 | 0.736 | 13,820 | 27.6 | 37.1 GiB | 2.75 | 13 min |
| `tr_groot_l10_inf` | 435 | 0.870 | 29,065 | 58.1 | 77.5 GiB | 2.73 | 28 min |
| `tr_groot_l10_cache` | 234 | 0.468 | 39,669 | 79.3 | 105.8 GiB | 2.73 | 36 min |
| **合计** | | | **188,844** | | **624 GiB** | | 约 3.5 h（含换组） |

逐任务成功数见 `summary.json` 的 `per_task_success`。本报告只做数据清点，不做分析。

**数据质量核查**（逐步数值体检、GPU 重放、跨组一致性）见 [`data_quality.md`](data_quality.md)：没有发现采集错误；GR00T 动作块的填充维等三个分析注意点写在那份报告的 §5。

## 4. 完整性核验

- **trace 文件**：每组 500 个 `.h5`，全部 `trace_closed_ok=True`、`trace_terminal=True`、`trace_write_errors=0`，没有残留的 `.tmp` 或 `.reserved`（`ops/check_trace.py` → `runs/<arm>/check_trace.jsonl`）。
- **身份**：每组 trace 的 `trace_task_uid` 集合与 journal 里 accepted 终态的 uid 集合完全相等，没有重复 uid；trace 与 journal 的 success 标记逐集一致（`ops/summarize.py` → `summary.json`）。
- **零重试**：每组 journal 都正好 500 行、attempt 全为 1，所以没有中断重跑留下的非 terminal 文件。
- **下发语义**（`ops/verify_served.py` → `served_check.json`）：全部 188,844 个决策都已逐步核对。`_inf` 组 100% `executed_arm=full_inference`（MISS），`_cache` 组 100% `executed_arm=full_hit`（FULL_HIT），没有例外。
- **服务收尾**：32 次 server 停机都是 SIGTERM，日志末尾都是 `trace writer … drained cleanly` 和 `SERVER_EXIT=0`。运行期间没有发生过 server 退出或 OOM。

## 5. 数据布局

```
/home/weiland/trace_runs/dual_20260923/
  config/        8 份 arm yaml + 单臂 matrix + arms.json（与 exp/trace_dual/config 相同）
  libs/          π0.5 权威库副本（spatial 36cd0f3b / l10 f13517ad）
  ops/           运行脚本（与 exp/trace_dual/ops 相同）
  runs/<arm>/trace/<suite>/episode_<id>_<µs时间戳>_p<pid>.h5 (+ .trace.jsonl 侧车)
  runs/<arm>/client/   timan107 回收的 journal.jsonl / per_step.jsonl / driver.log / launch.json（sha 校验过）
  runs/<arm>/server_<port>.log, check_trace.jsonl
  runs/chain.log, summary.json, served_check.json
  smoke/         两模型 smoke 的 trace（不计入正式数据）
```

每个 `step_XXXX` 组里有：
- 旧版采集器的全部字段：`vision_0..2`、`prompt_emb`、`robot_state`、`clean_action`、`noise_action_0..N-1`（π0.5 为 10 个，GR00T 为 8 个）、`input_images/*`。
- `trace/` 子树：
  - `actions/{executed, full_hit, full_inference}`，两种动作都在；
  - `raw_images`、`raw_state`、`tokenized_prompt`；
  - π0.5 另有 `model_images`；
  - `query_keys/*`；
  - `search/`：real 与 twin 的 top-k id 和分数、逐字段分数与 margin；
  - 组 attrs：`executed_arm`、`hit_type`、`score`、`cp1_score`、`verdict_json`、`timing_json` 等。

文件级 attrs 带 `trace_task_uid`、`trace_attempt`、`trace_yaml_id`、`trace_yaml_sha256`、`trace_library_sha256` 和 `success`。

## 6. 运行中的偏差与处理

1. **完成判据写错**：`chain.sh` 最初只把 `status=done` 算作完成，把正常失败集（accepted、`status=failed`、无 error）当成了缺集。结果第一组 driver 结束时判为 493/500，并触发了续跑。23:21 修正为"accepted 的 done 或 failed，且无 error"。第一组核实已满 500 后手工补记 `GROUP_DONE`，编排从第二组重新进入。数据不受影响：续跑那一次 run_gtp 判定全部已完成，没有派发任何一集。
2. **π0.5 库换成权威版**（见 §2），trace 里记录的库 sha 是权威 sha。
3. **显存**：π0.5 l10 组 4 个进程合计峰值约 45.6 / 48.5 GB。server 开了显存锁，峰值会被保留，但没有出现 OOM。GR00T 约 26.5 GB。

## 7. 复现

```bash
# 本机：生成配置 → 串行编排（每组：起 4 个 server → 在 timan107 跑 run_gtp → 核对 500 → SIGTERM 停机）
.venv/bin/python exp/trace_dual/ops/emit_configs.py
bash exp/trace_dual/ops/chain.sh tr_pi05_sp_inf tr_pi05_sp_cache tr_pi05_l10_inf tr_pi05_l10_cache \
     tr_groot_sp_inf tr_groot_sp_cache tr_groot_l10_inf tr_groot_l10_cache
# 收尾
bash exp/trace_dual/ops/collect.sh <arm ...>; .venv/bin/python exp/trace_dual/ops/check_trace.py runs/<arm>/trace
.venv/bin/python exp/trace_dual/ops/summarize.py; .venv/bin/python exp/trace_dual/ops/verify_served.py
```

脚本里的路径写死为 run 根 `/home/weiland/trace_runs/dual_20260923/`。timan107 上的岛树是 `/scratch/zixuans8/openpi_trace`：由 HEAD 的相关 `.py` 文件、A-pool 和 `run_cfg/` 组成，driver 入口是 `run_group.sh`。

## 8. 待 owner 确认

- §2 的「标准 yaml」认定：π0.5 用 gtp_ws 族，GR00T 用 rit source_template。
- 是否把 `exp/trace_dual/`（ops、config、analysis）和 handoff 入库。按规矩，commit 要等 owner 当次指示。
