# 融合权重 ablation 运行交接（L1，纯 cache，A 池 500 全集）

> 2026-09-14。owner 裁定：三臂（均匀 / 7 维动作误差网格 / 相位判别 LDA）× 四 suite，只跑 A 池 `pruned_init` 500 全集，
> 网格榜首不重跑（沿用历史）。目的 = 论文附录 C.2 的 ablation 表（`docs/iclr/modality_weight_selection.md` §5–§6）。
> 级别 L1：不改 `src/`，只出 yaml + 复用两条现成运行链；无 G1/G2。

## 1. 臂（12 个，`exp/weighted_sum/config/fusion_ablation/`，`emit_fusion_ablation.py` 生成，权重从离线 JSON 读出不手抄）

| 臂 | π0.5 Spatial | π0.5 LIBERO-10 | GR00T Spatial | GR00T LIBERO-10 |
|---|---|---|---|---|
| uniform | 1/3 ×3 | 1/3 ×3 | 1/3 ×3 | 1/3 ×3 |
| acterr（7 维动作误差 1/48 网格 argmin，**在被服务的库上算**） | 0.125/0.750/0.125 | 0.167/0.792/0.042 | 0.208/0.125/0.667 | 0.333/0.188/0.479 |
| lda（相位判别 δ=0.05） | 0.308/0.412/0.280 | 0.407/0.321/0.272 | 0.331/0.385/0.284 | 0.647/0.200/0.153 |
| 历史网格榜首（不跑，对照） | 0.06/0.44/0.50 → 0.740 (B 池 held-out n=300) | 0.12/0.88/0.00 → 0.520 (B 池 n=100) | 0.17/0.58/0.25 → 0.766 (A 池 500) | 0.67/0.17/0.17 → 0.454 (A 池 500) |

- 配方 = 各 suite 历史网格自己的 yaml，只换三个权重：π0.5 用 `cache_prune/config/search_<suite>.yaml`（normalizer 与 phase-2 网格逐字段相同），
  库 `cp1_spatial_pool_16.pkl`；GR00T 用 r1 粗扫格 `v0@2_v1@2_rs@2.yaml`（**原 S3 库** `libraries/<suite>/<suite>_sp16_S3.pkl` + `calib_input_S3` 标定）。
- ⚠ acterr 的 GR00T 权重在 W13 库上算是另一组（spatial 0.25/0.04/0.71，l10 0.69/0.02/0.29）；本实验服务的是原 S3 库，所以用原 S3 上的 argmin。
  两次同配方采集之间 acterr 的解漂移这么大，本身是一条记录（LDA 在两库上 0.669/0.188/0.143 ↔ 0.647/0.200/0.153）。
- GR00T uniform 在 r1 里是精确格（0.750 / 0.412）；按 owner 令仍跑，兼作同批次复现读数。
- π0.5 的历史网格在 B 池 held-out、已下线的 H200 上；本实验三臂在 A 池、h100 上跑，所以 π0.5 那两列只能臂间互比，与历史榜首的对照要注明池与 GPU 都不同。

## 2. 拓扑（两条 lane 并行）

| lane | server 节点 | 运行链 | 车队 | 顺序 |
|---|---|---|---|---|
| P（π0.5） | h100 `149.165.153.233`，5 × `serve_policy --replicas 1`（stage 2/3 meta）端口 23280–23284，driver 23290 | `exp/weighted_sum/ops/fw_lane_pi05.sh`（= cache_prune 09-12 直跑配方，`run_size_eval --role driver`） | timan108 `cache_prune/ops/launch_fleet.sh`，5 端点 × 12 = 60 worker，GPU 0,1,2 | spatial → libero_10（server 按 suite 重起） |
| G（GR00T） | weilandserver 4090，代码树 **`/data/openpi_lg`**（HEAD 同本地，`serve_groot_libero.py` sha 5d47762f）端口 23160–23162，岛 venv 起 server | `exp/weighted_sum/ops/fw_lane_groot.sh`（= r1 网格自己的 harness `orchestrate_search.py`） | timan107，harness 自带：3 slot × 12 worker，client = `/scratch/zixuans8/openpi` 的 `main.py --resize-size 256 --replan-steps 5`，seed 7 | spatial → libero_10（ckpt 切换） |

预算：π0.5 3,000 集 + GR00T 3,000 集。估算 lane P ≈ 45 min（09-12 实测 spatial 110 集/min），lane G ≈ 1.5 h（r1 实测 0.31 集/s/slot）。

## 3. 步骤

1. 推文件（tether push，已存在加 `--force`，推完对 sha）：
   - h100 `/home/weiland/openpi/exp/weighted_sum/config/fusion_ablation/pi05/**` + `ops/fw_lane_pi05.sh`
   - weilandserver `/data/libero_cache/search/<suite>/abl/fw_groot_<suite>_{uniform,acterr,lda}.yaml` + `/data/openpi_lg/exp/weighted_sum/ops/fw_lane_groot.sh`
2. smoke：lane P `--smoke --trials 1 --arms fw_pi05_libero_spatial_lda`（10 集）；lane G `--tasks 10 --trials 1 --expect 10 --workers 2` 到 `abl_smoke_results/`。
3. 正式：lane P 车队 `launch_fleet.sh 149.165.153.233 23290 23280,23281,23282,23283,23284 12 0,1,2 timan108` → h100 tmux `fwlane` 跑 spatial 再 l10；
   lane G weilandserver tmux `fwg` 跑 `fw_lane_groot.sh libero_spatial && fw_lane_groot.sh libero_10`。
4. 巡检：每 20 min 一行（journal 行数 / results 文件数 / GPU / 错误 grep），不做额外分析。
5. 收尾：拉回 h100 `/data/openpi/weighted_sum/fusion_ablation/<suite>/{journal,per_step}.jsonl` 与 weilandserver `abl_results/*.json`
   到 `exp/weighted_sum/data/fusion_ablation/`（gitignored）；关 server、停车队（`stop_fleet.sh`）；`exp/weighted_sum/analysis/lcw_ablation_summary.py` 出表。

## 4. 产物与判读

- 每臂 500 集 SR + Wilson 区间；同 suite 三臂在同一 init 集上配对（sign-flip）；GR00T 与 r1 榜首格逐集配对（同 init、同 seed）。
- FULL_HIT 率必须 1.000（π0.5 由 driver 见证；GR00T 由 always_hit 配方保证）。
- 表进 `docs/iclr/modality_weight_selection.md` §4 与附录 C.2；表注写明 π0.5 历史榜首的池/GPU 差异与 GR00T 榜首的 winner's curse。

## 5. 结果（2026-09-14 14:28 全部完成，本地时间）

- 12 + 4 臂全部 500/500，FULL_HIT 全 1.000，基础设施错误 0。四 suite 四臂终表与配对检验见 `docs/iclr/modality_weight_selection.md` §4.5，
  数据 `exp/weighted_sum/data/fusion_ablation/{pi05,groot,groot_conductor}/`，聚合 `summary_final.json`。
- 运行中的两处偏离 §2 计划：① owner 中途加第四臂（网格榜首重跑）；② GR00T 榜首臂改走 conductor（weilandserver 5 replica + timan107/108 两支车队 120 worker），
  因为 r1 harness 一 cell 一 server 且不能与在跑的 orchestrator 并发（其启动 reap 会杀全部 timan107 worker）。`run_size_eval` 加了 `--resize-size/--replan-steps` 透传（未 commit）。
- 跨 harness 一致率 81%（同权重：conductor 0.736 vs 网格 0.766，Spatial），同 harness 逐集确定（Spatial 100%，LIBERO-10 93.8%）。
- 收尾：h100 / weilandserver 显存 0，timan107 / timan108 车队已停，残留 worker 0。

## 6. 追加：1/6 网格全量重做（owner 选方案 B，2026-09-14 16:10 起）

- 目标：四 suite 同一套 1/6 闭单纯形网格 28 格 × A 池 500 集 = 56,000 集，替代历史暴力搜索（π0.5 历史在 B 池 n=100、GR00T 在 A 池但 eager server）。
- 臂：`exp/weighted_sum/emit_grid6.py` → `exp/weighted_sum/config/grid6/<teacher>/<suite>/g6_*_v0@a_v1@b_rs@c.yaml`（模板同 §1）；GR00T cell 在 weilandserver `/data/libero_cache/search/grid6/<suite>/cells/`。
- server 改动（本轮，未 commit）：`serve_groot_libero.py --compile-stage1` 移植；`staged.py` 等价门改分位数判据 + 首连接线程内完成编译/预热/录制。编译版单 replica 59 集/分（eager 20）。
- 拓扑：Phase 1 GR00T = weilandserver 6 个编译 replica（23160–23165）+ conductor，车队 timan107 60 + timan108 60；spatial → l10。Phase 2 π0.5 = h100 5 server + 同两支车队。
- 产物：weilandserver `/data/libero_cache/search/grid6/<suite>/{journal,per_step}.jsonl`；h100 `/data/openpi/weighted_sum/grid6/<suite>/`。

### 6.1 结果（2026-09-15 00:33 全部完成）

- 4 × 28 格 × 500 = 56,000 集全部完成，FULL_HIT 全 1.000，driver/server 零错误。地形与候选落点见 `docs/iclr/modality_weight_selection.md` §4.6，数据 `exp/weighted_sum/data/grid6/{pi05,groot}/<suite>/`，汇总 `summary_grid6.json`。
- 运行事实：GR00T Spatial 16:08 起（前 30 分钟为全模型编译 replica，16:35 起按 owner 令切 stage1-only 续跑，显存 35 → 12 GB）17:36 完；
  GR00T LIBERO-10 首起失败（另一 session 17:00 推的 HEAD 版覆盖了含 `--stage1-only` 的文件，已合并回推）18:00 重起 → 00:33 完，同卡另一线的 loto/online-RIT 进程分走算力，约 40 集/分；
  π0.5 两 suite 18:25 → 21:48 在 h100 + timan108 完成（spatial 约 200 集/分、l10 约 110 集/分）。
- 吞吐参考（编译 + stage1-only，6 replica）：GR00T Spatial 约 176 集/分（eager 22）；单 replica 上限约 59 集/分（36 worker 打满）。
- 收尾：weilandserver 6 replica 已关（残留的 4 个 GR00T server :23181–23184 与 timan108 上 48 个 worker 属另一 session 的 online RIT 线，未动）；h100 显存 0；timan107 车队停。


## 7. 追加：key builder × LDA 补充实验（plan `logs/keybuilder_lda_supplement_plan.log.md`，§4 Code 阶段 2026-09-15）

### 7.1 准备阶段事实（已完成，未跑闭环）

- **libero_10 LLM 库**（plan §4.1）：原料 = 本地 / weilandserver `/data/openpi/exp_common/db/libero_cache/libero_10` 的 50 个 H5（与所有现役 libero_10 库同一批：本地全部库的 `(task_key, trajectory_id, step_idx)` 集合与 `action_chunk` 摘要完全一致），经 weilandserver 临时只读 rsync daemon（:23190，用完即停）拉到 h100 `/data/openpi/exp_common/db/libero_cache/libero_10`，50 文件 sha256 与本地逐一相同。
  h100 建库 10:48–10:53（本地时间；`--builder-type cp1_llm_layer_extract --extract-layer 0 --prefix-reducer-type prefix_mean_pool --checkpoint-dir /data/openpi/checkpoints/pi05_libero_pytorch --config-name pi05_libero --device cuda`，tokenizer self-check max abs diff 0），产物 `exp/common/data/cache_artifacts/libero_10/llm_layer_extract/cp1_llm_l0_prefix_mean_pool.pkl`（sha256 `93d3d35d9e75…bd59b`，57.5 MB），已拉回本地同路径。
  验收：2,640 条 / 50 轨迹 / 10 任务；与 `cp1_spatial_pool_16.pkl` 的 id 集合相等，`action_chunk` 逐条 shape 相同且 `allclose(rtol=0, atol=0)`；`vector_dims == {vision_0: 2048, robot_state: 32}`；vision_0 全有限。
- **补标定**（`exp/common/calibrate_score_normalizers.py`，单库目录，默认 300 LOEO query，seed 42）→ `exp/weighted_sum/data/keybuilder_lda/calibration/<suite>/<stem>.json`：libero_10 clip_vit_b_32（三字段全 zscore）、两 suite cp1_llm_l0_prefix_mean_pool（vision_0 / robot_state 均 zscore）。兼容门（zscore+tanh、μ 有限、σ>0）八个三字段库全过。
- **模板与拟合**：`emit_keybuilder_lda.py templates` → 8 份三字段模板（权重 1/3，normalizer = 各自 selected）+ 诊断格 `diagnostic_cells/<suite>.json`（来自 `summary_grid6.json` 的 spatial_16 A 池网格，只做诊断输出）。`lcw_fit_weights.py` 八次全部非退化（6 分钟，本机；G2 R1 后按 B2 重跑一遍，fit 文件新增库 sha256 与模板 sha256 身份，权重逐位不变）。
- **LDA 权重**（`lda@0.05`，v0 / v1 / rs；raw = Σ⁻¹d 未截断；queries = 参与的 query 数）：

| 臂 | 库 | w（截断归一） | raw | queries |
|---|---|---|---|---|
| `kb_libero_spatial_cp1_mean_pool_lda` | cp1_mean_pool | 0.189 / 0.447 / 0.365 | 2.43 / 5.75 / 4.70 | 1018 |
| `kb_libero_spatial_cp1_max_pool_lda` | cp1_max_pool | 0.249 / 0.423 / 0.329 | 3.16 / 5.36 / 4.17 | 1018 |
| `kb_libero_spatial_cp1_spatial_pool_64_lda` | cp1_spatial_pool_64 | 0.255 / 0.424 / 0.321 | 3.37 / 5.61 / 4.25 | 1018 |
| `kb_libero_spatial_clip_vit_b_32_lda` | clip_vit_b_32 | 0.302 / 0.369 / 0.329 | 3.98 / 4.88 / 4.35 | 1018 |
| `kb_libero_spatial_cp1_llm_l0_prefix_mean_pool` | cp1_llm_l0_prefix_mean_pool | v0 1.0 / rs 0.0（单 key，无权重） | — | — |
| `kb_libero_10_cp1_mean_pool_lda` | cp1_mean_pool | 0.296 / 0.334 / 0.370 | 3.27 / 3.70 / 4.09 | 2640 |
| `kb_libero_10_cp1_max_pool_lda` | cp1_max_pool | 0.290 / 0.335 / 0.375 | 3.09 / 3.58 / 4.01 | 2640 |
| `kb_libero_10_cp1_spatial_pool_64_lda` | cp1_spatial_pool_64 | 0.388 / 0.303 / 0.308 | 4.68 / 3.66 / 3.72 | 2640 |
| `kb_libero_10_clip_vit_b_32_lda` | clip_vit_b_32 | 0.507 / 0.254 / 0.239 | 9.63 / 4.83 / 4.55 | 2640 |
| `kb_libero_10_cp1_llm_l0_prefix_mean_pool` | cp1_llm_l0_prefix_mean_pool | v0 1.0 / rs 0.0（单 key，无权重） | — | — |

  所有截断前分量均为正（无字段被截到 0）。每臂的库 sha256 / 标定 json sha256 / 模板 sha256 / d、Σ、raw 全在 `exp/weighted_sum/data/keybuilder_lda/weights.json`。
- **正式 yaml**：`emit_keybuilder_lda.py final` → `exp/weighted_sum/config/keybuilder_lda/<suite>/{kb_*.yaml, matrix_<suite>_pool.yaml（4 臂）, matrix_<suite>_llm.yaml（1 臂）}`，10 份全部通过 `load_cache_config` + 逐字段断言（builder、权重、normalizer 与模板逐字段一致、top_k=1 / depth=1 / task_scoped、always_search / always_hit / write never、vector_dims = 库全字段、CLIP 库变体 = ViT-B-32/openai）。`active_manifest.json` 固定 10 个新臂 + 每 suite 1 个参考臂（`fw_pi05_<suite>_lda`）、A 池 rollup、库/yaml 身份；两轮 round 的 launch 走 `launch_binding: run_id`（driver 每次启动写 `per_step.jsonl.launch.<run_id>.json`，汇总按 `launch_glob` 逐 run 绑定），参考臂 `legacy`（2026-09-14 的单份记录）。参考臂用真实 journal/launch 过严格入口检查（launch 4 臂、A 池 rollup 一致、500 集覆盖；libero_10 参考臂 12 集为 attempt 2）。
- LLM 库 vision_0 的 LOEO 余弦极集中（libero_10 μ=0.9974、σ=0.00063；spatial μ=0.9977、σ=0.00067）：单字段 top-1 与 normalizer 无关，但说明整段 prefix 均值 key 的区分度天然很低，判读时要连同 §3.5 的时延口径一起看。
- 机器现状（10:55 本地）：h100 上另一 session 起了 4 个 GR00T server（tmux `ort_srv_frontier_rprime_2320[1-4]`，显存 30 GB、util 96%），本线 pool 轮的显存/吞吐预检（plan §6）必须在其占用下实测；weilandserver 显存 44/49 GB 同样被占。


### 7.2 G2 复审与 owner 授权修复（2026-09-15 12:27 CDT）

G2 Round 3 **APPROVED**，详见 [计划 Review Log](keybuilder_lda_supplement_plan.log.md)。按 owner 本轮明确授权，审查方先复审，再直接修复残余的 driver/summary 联接不一致、跨 run 与重复记录冲突检查、census 被续跑覆盖、拟合时输入身份时序问题。正式 driver 与汇总现在共享 `exp/common/conductor_evidence.py`；launch/census 都按 run 留存；fit 记录读入时的输入摘要并在计算结束复核。

- 相关默认回归共 **458 passed / 5 manual skipped**；沙箱限制导致的 7 项 socket 失败已在沙箱外复跑通过。另有 **18 项独立审查测试通过**。新增公开测试 17 项已包含在 458 项内。
- 实际八份 fit 的输入摘要与 LDA 解、十份 YAML 与四份 matrix 全部通过回读。两个历史 reference 各 500 集的真实 FULL_HIT 证据通过共享读取器（Spatial 14,448 步；L10 40,131 步）。
- 执行方原有本任务改动已暂存；本轮审查方追加的实现、测试和审查记录按 owner 指示留在暂存区外。
- 本节只记录本地代码验收；项目完整 Verify、远端部署、GPU 预检、LLM manual parity 与新十臂正式 rollout 均未执行。
