# Session handoff

> 多条线共用本文件。§0 为常驻初始化（不动）。**§1 = 减步基线线（RoboCasa365 / Cosmos / X-WAM / DP 四条阶梯全部收工，2026-09-17 07:45；compact 后从 §1 恢复）**；旧 §7（key builder × LDA）已删——其状态见记忆 `project_fusion_weight_lda_ablation` 与 `logs/keybuilder_lda_supplement_plan.log.md`。

## 0. 初始化方式（不变）

owner 的常驻指令，逐字有效：

> 「开始进行实验，我离开了，期间由你独断专行，不要问我任何问题，注意监控体系定位，
> corn 负责定时巡检，monitor 负责条件触发，不要职责混淆，停止条件是完全做完实验，不做完不停」

**三条必须保持的纪律**：

1. ⛔ **不得用 git commit / push 同步代码和数据**。要同步任何代码或数据一律走 **tether**。
   git 只在里程碑收口、且 owner 当次明确指示时才动。
2. **server 只在 server 节点跑，worker 只在 timan 族跑**（见 §2）。
3. **读 `experiment-lifecycle` skill 但不挂载**：只取 tether 用法（`dist_experiment_control/docs/usage.md`）
   与设备清单（`docs/devices.md`），**不要执行它的 §0 初始化**，不要索要 agentchat 账号 / token / 房间。

### 开工步骤（照做，不要问 owner）

1. **读文档**，按这个顺序，都读完再动手：
   - `logs/cache_prune_run_handoff.md`（整份，运行入口的唯一权威）
   - 本文件 §1–§5
   - `/home/weiland/projects/dist_experiment_control/docs/devices.md`（拓扑与机器约束）
   - `/home/weiland/projects/dist_experiment_control/docs/usage.md`（tether 命令）
   - 需要时再查 `logs/cache_prune_plan.log.md`（算法与统计定义）
2. **探设备**：`tether node ls -a`，确认 h100 / weilandserver / timan107 / timan108 四台
   都 ONLINE；逐台查 GPU 占用、RAM、磁盘余量、残留 tmux 与进程。⚠ timan108 只有 3 张卡
   （见 §2），别按 4 张排。
3. **对齐代码**：四台的仓要和本地同一 commit。**走 tether push，不走 git push**；
   已存在的文件要 `--force`；`/scratch` 不在 timan 的 allow_roots，经 `/tmp` 中转。
   推完逐文件 `sha256sum` 对账，**比对内容不要带路径**（`cat A B | sha256sum`，
   `sha256sum A B` 会把路径算进去，我为此误判过一次）。
4. **建任务表**（TaskCreate），把 §1 的准备项与三相位拆成条目，边做边更新状态。
5. **搭监控**（§4）：cron 定时一行巡检 + Monitor 条件触发。**先搭好再放量**。
6. **先 smoke 再放量**：任何新拓扑都先跑一个最小 cell 验证到底（起 1 个 server、少量 worker、
   跑通一集并核对产物），再按 §2 的规格扩到目标规模。上一条线每次跳过这步都出事。

**其它长期约束**：

- commit message 全英文；**绝不加 `Co-Authored-By: Claude`** 或任何 AI 署名（作者恒为
  `LinZiyang666 <3177267975@qq.com>`）。未经 owner 当次指示不得 `git add`。
- ⛔ **画图脚本一律不许 commit**（`build_*figure*` / `plot_*` / `render_*` / `edit_*figure*` /
  `*_editor.html`）。`exp/rit_pareto/build_figure.py` 与 `edit_figure.py` 属于另一个 session，**碰都不要碰**。
- ⛔ 不要 `rm -rf`（带 -f 的递归强删）；`rm` / `rm -r` 不带 -f 可以用；删除前先 `wc -l` 核对规模。
- ⛔ **共享机上不要 `pkill -f`** 裸模式：它会匹配到发起命令的 shell 自己。用字符类
  `[w]orker_entry`，且**脚本正文任何地方（含注释、echo）都不能出现裸的匹配串**。
- 无人值守期间**禁起 `run_in_background` 后台任务**（触发审批弹窗阻塞会话），用 Monitor。
- 报时刻用本机本地时间（America/Chicago）。
- **巡检就只巡检**：贴 PROBE 行，不要顺手做额外分析（owner 明确要求过）。

## 1. 减步基线线 — 本 session 唯一运行手册（2026-09-16 14:20 写；compact 后从这里恢复）

> 权威记录：`logs/nfe_baseline_rc365_plan.log.md` §8（RoboCasa365，**live**）、`logs/cosmos_nfe_plan.log.md` §4（Cosmos × LIBERO，已收工）、`logs/nfe_baseline_plan.log.md` §5（π0.5/GR00T × LIBERO，已收工）、`logs/cache_direction_discussion_20260915.log.md`（方向讨论）、`logs/cache_transfer/summary.md`（跨领域调研）。
> 记忆：`project_nfe_baseline_live_run`、`feedback_no_unassigned_machines`。代码：`exp/nfe_baseline/`（已 commit bf36867）、`exp/cosmos_nfe/`（**未 commit**）。Stop hook 目标：「有序进行所有实验，不做完不停」。

### 1.1 owner 裁定（逐条有效）

- 只用 h100 / weilandserver / timan107 / timan108；timan107、timan108 各可跑 ≥30 worker（9-16 14:1x）。不碰 timan1。
- 报时用 America/Chicago；巡检只贴 PROBE 行；不 commit/push 除非 owner 当次说；画图脚本不入库（9-16 那次 bf36867 是 owner 明令的一次性例外）。
- 减步成本口径 = RIT 同式 IR(k) = (s1+s2+s3(k))/MISS；RoboCasa 用 `exp/nfe_baseline/config/rc_cost_{groot_tp,pi05}.json`（GR00T IR(k=1/2/3) = 74.4/83.0/91.5%，hit 地板 23%）。
- LIBERO 图：π0.5 l10 各点 +0.04（GR00T 不加）已写进 `exp/rit_pareto/analysis/figures/*.json`，编辑页 Save JSON 已关（`figure_editor.html` hidden + `edit_figure.py` /api/save 403）。
- Cosmos 线：**libero_10 k=2 跑完即全线停（不跑 k=1）**，机器转 RoboCasa365；RoboCasa365 减步**每个 k 都跑**（GR00T 1,2,3；π0.5 1…9），13 任务 × 50 seed。
- 引用 RIT RoboCasa 数字一律从原始台账 `~/tmp_rit/frontier_{groot,k1_groot,…}_all13.json` 取，`exp/robocasa365/analysis/figures/rit_*_all13.json` 是编辑页改过的（IR=100 换锚点、k1 IR84.6 的 0.694 被压到 0.62 等）。
- 下一篇方向 owner 倾向换更难 benchmark；候选表见讨论纪要 §（RoboCasa365 主场 / VLABench π0.5-ft 官方 ckpt / Cosmos 用 RoboCasa-2024 / robomimic+DP）。HF token（owner 聊天给出、授权使用）存 h100 `/data/cosmos/hf_home/token`（600），建议 owner 事后撤销。

### 1.2 结果终态

**LIBERO（500 集/点，A 池官方 init）**
| | k=1 | 2 | 3 | 4 | 5 | 6 | 7 | 锚 |
|---|---|---|---|---|---|---|---|---|
| π0.5 spatial | 0.988 | 0.982 | | | | | | 0.99 |
| π0.5 l10 | 0.824 | 0.828 | 0.836 | 0.850 | 0.842 | 0.838 | 0.840 | 0.92 |
| GR00T spatial | 0.932 | 0.918 | 0.946 | | | | | 0.946 |
| GR00T l10 | 0.856 | 0.872 | 0.862 | 0.886 | 0.858 | 0.842 | 0.858 | 0.868 |
| Cosmos spatial（5 步为原样） | 0.974 | 0.974 | 0.984 | | 0.986 | | | 论文 98.1 |
| Cosmos l10 | 未跑 | 0.972 | partial 0.975@277 | | 0.980 | | | 论文 97.6 |

Cosmos 数据 `exp/cosmos_nfe/data/{results/,aggregate.json}`。**Cosmos 的 k 步 = k+1 次网络前向**（`res_sampler.py` `sample_clean=True` 末尾多一次 σ_min 评估；dummy 实测 nfe=1→2 次），k=1 与 k=2 逐集结果相同；真正单前向需 `sample_clean=False`（未跑）。
结论：LIBERO 上三个 teacher 减步都近乎无损——EDM/流匹配 x₀-预测的一步 = 条件均值，单峰示范无损；ActionCache Table 7 同。

**RoboCasa365（GR00T，昨天 partial）**：k=1 主 lane 8 任务 387 集 = 0.532 vs 4 步参考 0.640（−11 pp；掉在 SlideDishwasherRack/OpenCabinet/TurnOnSinkFaucet）。同 IR≈74 上 RIT cache K=1 档 0.705（原始台账），IR<74 只有 cache 能去。图（原始台账、8 任务、Wilson CI）：`exp/nfe_baseline/analysis/figures/rc365_groot_cache_vs_steps.png`（脚本 `exp/nfe_baseline/analysis/plot_rc365_cache_vs_steps.py`，数据 `exp/nfe_baseline/data/rc365_groot_k1_main_partial.json`）。

### 1.3 live 拓扑（14:17 起跑）

| lane | server | client | 状态/日志 |
|---|---|---|---|
| RoboCasa GR00T k=1,2,3 | h100 `nfeladder_grc`（`/tmp/nfe/switch_rc_h100.sh` → `ladder_server.sh groot_rc … 1,2,3 23230`，`serve_groot_rc_ksweep`，ckpt `/home/exouser/ckpt/n15_robocasa_tp/.../atomic_seen/checkpoint-60000`）；`/tmp/nfe/ladder_grc.log`、`/tmp/nfe/nfesrv23230.log` | timan108 `nfelane_grc`（`/tmp/nfe/switch_rc_t108.sh` → `ladder_rc_client.sh groot_tp 1,2,3 149.165.153.233:23230 **30** 0,1,2 /scratch/zixuans8/nfe/rc_results`）；`/tmp/nfe/lane_grc.log`、`/tmp/nfe/nfercli_run_groot_tp_k<k>_<lane>.log` | k=1 main 已完成（400/400，macro 0.5175 on 8 任务）→ pnp 14:20 起（30 worker）→ k=2 → k=3；每 lane 完成写 `summary_nfek<k>-teacher__l1s1_groot_tp.json`（`complete:true`） |
| RoboCasa π0.5 k=1…9 | weilandserver `nfeladder_prc`（`/tmp/nfe/switch_rc_wls.sh` → `ladder_server.sh pi05_rc … 1,…,9 23170`，`serve_pi05_ksweep --cache`，ckpt `/home/weiland/ckpt_pi05_robocasa_pytorch`）；`/tmp/nfe/ladder_prc.log`、`/tmp/nfe/nfesrv23170.log` | timan107 `nfelane_prc`（`/tmp/nfe/switch_rc_t107.sh` → `ladder_rc_client.sh pi05 1..9 ziyanglin.com:23170 **32** 0-7 /scratch/zixuans8/nfe/rc_results`）；`/tmp/nfe/lane_prc.log` | k=1 起跑；k=9 main 有 36 集旧 journal（会续） |

- 驱动逻辑：server 侧 `ladder_server.sh` 见过连接后空闲 400 s 判 k 结束 → `stop_servers.sh` 按 PID 停 → 下一 k；client 侧 `ladder_rc_client.sh` 用 `probe_metadata.py` 对表 `nfe_num_steps` 后起 `run_rc_client.sh`（`run_ws_search --cid teacher`，journal 续跑），main 完成再 pnp，两 lane 都 `complete:true` 才进下一 k。已完成的 k 自动跳过。
- 验证步数生效：GR00T 日志 `KSWEEP num_inference_timesteps=k`；π0.5 `KSWEEP run_stage3 first call num_steps=k`；`__hit_meta__` 恒 MISS（无库）。
- 监控（session-only，compact 后重挂）：Monitor 四机（RCLANE/LADDER 变化、Traceback、timan108 显存>92%、15 min 心跳）+ cron `:17/:37/:57` 四条 PROBE。
- 时长估计：GR00T 30 worker ≈ 8–10 集/分 → 三个 k 约 3.5 h；π0.5 32 worker ≈ 20 集/分 → 九个 k 约 5 h → 约 19:30 全完。
- Cosmos 残留（已全停，不再跑）：weilandserver `/home/weiland/cosmos-policy`（venv 13 GB）+ `/home/weiland/cosmos_home` + `/home/weiland/cosmos_exp`，模型 `/data/cosmos/hf_home`（12 GB），旧 venv `/data/cosmos/cosmos-policy`（13 GB，可删）；h100 `/data/cosmos/*`；timan107 `/scratch/zixuans8/cosmos/*`（/scratch 只剩 15 GB）。结果已拉回本地。

**再均衡（owner 14:50 裁定「groot 跑完之后把所有资源都给 pi，要做再均衡」）**：GR00T 三档跑完后 h100/t108 自动切成 pair B 跑 π0.5 降序 9,8,…,2（h100 :23231，t108 30 worker）；pair A（wls/t107）升序 1..9。两对靠阈值文件分工：t107+wls `/tmp/nfe/prc_stop_after_k`（初值 5，A 跑完该 k 即停）、t108+h100 `/tmp/nfe/prc2_stop_after_k`（初值 6，B 跑完该 k 即停）；看门 tmux `nfestopk_prc`/`nfestopk_prc2`/`nfeswitch_prc2`。每个 k 边界按两对速度改阈值（只能改尚未 DONE 的 k）；两对 /scratch 不共享，pull 时 pair B 的 k 从 t108 拉。脚本副本 `exp/nfe_baseline/data/tmp/{stopk_wls,stopk_t107,switch_prc2_h100,switch_prc2_t108}.sh`，本地四机探针 `$CLAUDE_JOB_DIR/tmp/probe4.sh`。

**Cosmos × RoboCasa-2024 全量（owner 20:0x 裁定，π0.5 结束后自动开跑）**：四机已武装 `cosmos_arm_rc24` tmux（`exp/cosmos_nfe/ops/rc24/arm_full_*.sh`）：wls 在 π0.5 pair A STOPPED 后起 4 replica :23180-23183，h100 在 pair B STOPPED 后起 10 replica :23250-23259；t107 跑 trial 40-49（32 worker→wls），t108 跑 0-39（30 worker→h100），k=1..5 各 24 任务。进度看 `/tmp/cosmos/rc24_full.log`（`RC24 k=<k> UP/DONE`）与 `results_rc24/k*/`。完成后：打包两台 client 的 `results_rc24` 拉回 `exp/cosmos_nfe/data/results_rc24_{t107,t108}/` → `python -m exp.cosmos_nfe.aggregate_cosmos_rc24 --root … --root … --out exp/cosmos_nfe/data/aggregate_rc24.json` → 写 `logs/cosmos_nfe_plan.log.md` §5。h100 上 :23250 冒烟 replica 已在跑（保留即可）。Cosmos LIBERO 件已删（wls 7 项、h100 1 项）。

**[23:10 更新] RoboCasa365 减步阶梯全部完成**：GR00T k=1/2/3 = 0.572/0.652/0.665（参考 0.680）；π0.5 k=1..9 = 0.391/0.488/0.525/0.523/0.554/0.551/0.569/0.554/0.595（参考 0.557）。聚合 `exp/nfe_baseline/data/agg_{groot,pi05}_rc.json`，原始 `runs/rc/<teacher>/{main,pnp}/k*/`，终图 `exp/nfe_baseline/analysis/figures/rc365_cache_vs_steps_all13.{png,pdf}`（脚本 `plot_rc365_cache_vs_steps_all13.py`，不入库）。π0.5/GR00T 的 nfe tmux（ladder/lane/stopk/sig）都已退出；wls :23170、h100 :23231 已停。

**[00:15 已停] Cosmos RC24**：owner 裁定 k=1（macro 0.6725，官方 5 步 0.671）无折损即停，四机 Cosmos 进程全清；下面这段是当时拓扑，仅供追溯。原 **Cosmos RC24 全量在跑**（22:33 wls/t107 侧、22:58 h100/t108 侧起）：h100 10 replica :23250-23259（`--no-future-decode 1 --cuda-graph 1`，6.5 GB 各）、wls 4 replica :23180-23183；t107 `cosmos_arm_rc24` tmux 里的 ladder 跑 trial 40-49（24 分片）、t108 `cosmos_ladder_rc24` tmux（W=24，30 会把 A5000 顶爆 EGL）跑 trial 0-39。每 k：h100 侧 ≈ 31 min × (1,1.24,1.47,1.71,1.95)，预计 ~03:00 全完；wls 侧 ~01:55。巡检 cron ba2997bd（四机 `/tmp/cosmos/rc24_probe.sh`）、Monitor `rc24_watch.sh`。收尾：`$CLAUDE_JOB_DIR/tmp/pull_rc24.sh timan107|timan108` → `python -m exp.cosmos_nfe.aggregate_cosmos_rc24 --root exp/cosmos_nfe/data/results_rc24_timan107 --root exp/cosmos_nfe/data/results_rc24_timan108 --out exp/cosmos_nfe/data/aggregate_rc24.json` → 写 `logs/cosmos_nfe_plan.log.md` §5 → 停 replica（h100 `stop_rc24_servers.sh 10 23250`、wls `stop_rc24_servers.sh 4 23180`）、杀 `cosmos_*` tmux、删 cron/Monitor；**不 commit 不 push**。

**[23:55] 新排队（owner）：Cosmos RC24 → X-WAM × RoboCasa-2024 → DP DDPM-100 × robomimic**，环境已在装（记录 `logs/xwam_dp_nfe_plan.log.md`）：wls/h100 `xwam_setup` tmux（server：torch2.8+flash-attn+ckpt 下载）、t108 `xwam_setup`（client）+ `dp_setup`（micromamba）+ `dp_ckpts`（5 个 4 GB ckpt）。RC24 结束后：先 X-WAM 冒烟（broker+server+client 三件，`--action_denoise_steps` 阶梯），再 DP。Task list #10–#13。

**[2026-09-17 07:45 更新] X-WAM 与 DP 阶梯全部完成，全线收工**（记录 `logs/xwam_dp_nfe_plan.log.md`）：
- X-WAM × RoboCasa-2024（24 任务 × 50 集，`--action_denoise_steps`）：k=10/5/3/2/1 macro = **0.7917/0.7808/0.7858/0.7683/0.7933**（官方 79.2% 复现；1 步无损）。数据 `exp/xwam_nfe/data/results_k{10,5,3,2,1}/` + `aggregate_xwam.json`。h100 两 server + broker、wls server、t108 client 全停，三机 GPU 归零。
- DP DDPM-100 × robomimic/PushT（5 任务 × 6 档 × 50 集）：均值 DDPM-100/DDIM-10/4/2/1/DDPM-10 = **0.85/0.88/0.81/0.60/0.03/0.03**（ε-DDPM 头有硬悬崖）。数据 `exp/dp_nfe/data/results/<task>/<cell>/summary.json` + `aggregate_dp.json`（`python -m exp.dp_nfe.aggregate_dp --root exp/dp_nfe/data/results`）。batch-1 延迟已在空卡 4090 重测（DDPM-100 ≈1.33 s、DDIM-10 ≈145 ms、DDIM-4 ≈65 ms、DDIM-2 ≈38 ms、DDIM-1 ≈25 ms，≈13 ms/步），见 log 与 `aggregate_dp.json`。
- 环境留存（owner 未说删）：h100 `/data/xwam`（ckpt 109 GB + Wan 32 GB）、wls `/home/weiland/x-wam` + `/data/xwam` + `/home/weiland/dp` + `/data/dp`、t108 `/scratch/zixuans8/{xwam,dp}`。cron `ea271db6` 已删；本地 tmux `xwam_ladder` 已退出。
- **未 commit 未 push**（exp/cosmos_nfe、exp/xwam_nfe、exp/dp_nfe、logs/* 全在工作树），等 owner 指示。

### 1.4 待做（按顺序）

-1. （9-18 13:5x）**x₀-head × 标签过滤数据实验（`logs/x0_multimodal_plan.log.md`，L2）G2 R2 APPROVED（owner 授权 codex 直接修码）→ 我逐文件复核 + Verify → 已 commit/push（见 git log）。** 测试：本机 `uv run pytest tests/dp_nfe -rs` 108 passed/2 skipped；wls 固定 DP 环境 `bash /tmp/x0/run_ws_test.sh` 120 passed（`/tmp/x0/ws_test_r2_final.log`）；全仓裸 `uv run pytest` 5765 passed / 9 失败 = 9-15 HEAD 基线原班（日志 job tmp `verify_full_x0_20260918.log`）。`plot_x0.py` 按画图脚本规则未提交（工作树保留）。代码同步用 `bash exp/dp_nfe/ops/sync_x0_code.sh weilandserver /home/weiland/dp`（h100 尚未装 DP 环境）。诊断产物（official square ckpt 50 集：官方 diffusers DDIM-100 0.78 = DDPM-100 0.78；v2 trailing DDIM-100 0.72 / DDPM-100 0.74）已挪到 wls `/data/dp/x0_multimodal/diagnostics/`，`results_trailing/` 为空；正式 P0 等 G2 放行后由队列重跑。wls GPU/tmux 已归零。G2 通过后：Verify（裸全量 `uv run pytest`）→ h100 DP 环境 → 数据下载（kitchen/blockpush/square image）→ 子集 + 标签抽查 → 冻结 normalizer → pilot → 冻结矩阵 → 队列。
0. （9-17 07:55）DP 延迟重测也已收尾（`aggregate_dp.json` 的 `batch1_latency_ms_idle_gpu`，log §5 收工状态）；四机 GPU/tmux 归零。等 owner 裁 commit/push 与后续（jepa-wms/LeWM latent-MPC、Cosmos planning、`sample_clean=False`、撤销 HF token）。以下 1–3 为历史步骤，均已完成。
1. 每个 k 两 lane 都完成 → `tether exec` 打包 `/scratch/zixuans8/nfe/rc_results/<teacher>/{main,pnp}/k<k>/summary_*` → `tether pull` 到 `exp/nfe_baseline/data/runs/rc/`（目录结构 `<teacher>/<lane>/k<k>/`）→ `uv run python -m exp.nfe_baseline.aggregate_nfe --policy groot_rc|pi05_rc --summaries-root exp/nfe_baseline/data/runs/rc --ks … --per-task 50 --anchor-sr 0.680|0.557 --anchor-source "RIT teacher-only reference arm (~/tmp_rit/teacher_ref_*_50ep.json)" --out … --figure-out-dir exp/nfe_baseline/data/figures`。
2. 全部跑完：画 RoboCasa 的"cache vs 减步"对比图（13 任务，原始台账 `~/tmp_rit/frontier_*_all13.json`，可仿 `plot_rc365_cache_vs_steps.py`）；结果写 `logs/nfe_baseline_rc365_plan.log.md` §8 与本文件；停 cron/Monitor；四台 `nfe*` tmux 归零、显存归零；**不 commit 不 push**，等 owner。
3. 若 owner 要：Cosmos 补 `sample_clean=False` 的真单前向点；Cosmos 换 RoboCasa-2024（需 moojink fork）；VLABench π0.5-ft。

### 1.5 坑

- h100 `tether exec` 双执行 → 脚本全部幂等（mkdir 锁 + has-session）。`pgrep -f` 用字符类。timan `/scratch` 不在 tether allow_roots，经 `/tmp` 中转。tmux server 不继承调用 shell 的 env（PYTHONPATH/HOME 要写进命令串）。
- **切档用信号不用空闲超时**（owner 20:3x）：`kdone_listener.py` + `ladder_server2.sh` + `rc/sig_client.sh`，pair B 已上线（h100 :23232）；v1 空档 8–9 min/档是 owner 明确不接受的。
- **timan108 的 robocasa365 checkout 缺 pinned_objects 补丁**（pnp lane 起跑即 TypeError）：已 `git apply exp/robocasa365/patches/robocasa_pnp_pinned_objects.patch`（工作树改动，重 clone 会丢）；timan107 早已打过。
- RoboCasa worker 显存：GR00T 峰值 3.5 GB、π0.5 峰值 3.1 GB；owner 说 30 个没问题，但 timan108 有 98.7% 压死驱动先例，Monitor 设了 92% 线。`run_ws_search --server` 必须在 `config/rc_timan.env` 的 `*_SERVERS` 里；pnp lane 冻结身份 5 任务 × 50，不能缩小 smoke。
- 「停」的解释：owner 9-15 的"k=7 跑完就停"指全线；9-16 的"k=2 跑完就停"也指 Cosmos 全线。哪个 teacher 加 0.04 owner 说反过一次，改数前复述。
- Cosmos 环境：NVIDIA 预编译索引不需 nvcc/Docker；egl-probe 要 `cmake<4`；TE import 需 `CUDA_HOME=<venv>/nvidia CUDNN_HOME=<venv>/nvidia/cudnn`；libero 包无 `~/.libero/config.yaml` 会交互式提问卡死；gated 仓库要 owner 在 HF 点 Agree + token。
