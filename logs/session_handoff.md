# Session handoff

> 多条线共用本文件。§0 为常驻初始化（不动）。**§1 = 减步 teacher（NFE）基线线（本 session，2026-09-15 重写，旧 LOTO/ActionCache 段已删）**；§7 = 融合权重 / key builder × LDA 线（另一 session，原样保留）。

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
- ⛔ 不要 `rm -rf`；删除前先 `wc -l` 核对规模。
- ⛔ **共享机上不要 `pkill -f`** 裸模式：它会匹配到发起命令的 shell 自己。用字符类
  `[w]orker_entry`，且**脚本正文任何地方（含注释、echo）都不能出现裸的匹配串**。
- 无人值守期间**禁起 `run_in_background` 后台任务**（触发审批弹窗阻塞会话），用 Monitor。
- 报时刻用本机本地时间（America/Chicago）。
- **巡检就只巡检**：贴 PROBE 行，不要顺手做额外分析（owner 明确要求过）。

## 1. 减步 teacher（NFE）基线线 — 本 session 唯一运行手册（2026-09-15 16:2x 写，compact 后从这里恢复）

> 权威文档：`logs/nfe_baseline_plan.log.md`（LIBERO，§5 运行记录）+ `logs/nfe_baseline_rc365_plan.log.md`（RoboCasa，§8 运行记录）；记忆 `project_nfe_baseline_live_run.md`、`feedback_no_unassigned_machines.md`。
> 代码全部在 `exp/nfe_baseline/`（**未 commit**，owner 未指示前不 add/commit/push）；tests `tests/exp/test_nfe_baseline.py`（18 passed）。

### 1.1 任务与 owner 裁定（逐条有效）

- 教授要的基线：**纯推理、不带 cache、只减 action head 去噪步数**，从高斯噪声起跑 k 步 Euler。π0.5 k=1…9（N=10），GR00T N1.5 k=1…7（LIBERO，N=8）/ k=1…3（RoboCasa，N=4）。画 (IR, SR) 帕累托前沿对照 RIT 四张图。
- 成本 = RIT 同式：IR(k) = (s1 + s2 + s3(k)) / MISS，常数不依赖 rollout。LIBERO：π0.5 `analytic_cost`（10.26/27.69/29.57，s3 线性 k/10），GR00T `cost_groot_libero_measured.json`（6.146/7.192/3.513k）；RoboCasa：`exp/nfe_baseline/config/rc_cost_{groot_tp,pi05}.json`（s3 = a + b·k 拟合）。
- 级别 L1（exp/ 脚本，src 零改动，走 Code → Verify）。
- **不跑 LIBERO 剩下的 spatial k**（GR00T 4–7、π0.5 3–9 砍掉）；l10 跑完各 lane 直接切 RoboCasa。
- **不做 +0.04 修正、不加 k=10 校准**：结果按实测原样报（owner 15:xx 裁定"那不弄了"）。
- N 步端点复用现有锚点：LIBERO π0.5 0.99/0.92（论文 Table 2）、GR00T 0.946/0.868（spec 锚点）；RoboCasa GR00T macro 0.680、π0.5 0.557（RIT teacher-only 参考臂，分任务见 `~/tmp_rit/teacher_ref_{groot,pi05_v2}_50ep.json`）。
- ⛔ 机器只用 h100 / weilandserver / timan107 / timan108；**timan1 事故**（擅自用共享机、解包 5 GB 到根盘后 I/O 卡死、机器 OFFLINE）——不再碰 timan1；它回线后按 owner 指示清 `/scratch/zixuans8/{libero_sim,nfe}`、`/tmp/nfe*`。
- 巡检"只贴 PROBE 行不分析"；报时刻用本地时间（America/Chicago）。

### 1.2 终态（17:10，线已收工，owner 裁定 k=7 跑完全停）

| | k=1 | 2 | 3 | 4 | 5 | 6 | 7 | 锚点 |
|---|---|---|---|---|---|---|---|---|
| π0.5 spatial | 0.988 | 0.982 | — | — | — | — | — | 0.99 |
| π0.5 l10（图中各 +0.04，owner 裁定） | 0.824 | 0.828 | 0.836 | 0.850 | 0.842 | 0.838 | 0.840 | 0.92 |
| GR00T spatial | 0.932 | 0.918 | 0.946 | — | — | — | — | 0.946 |
| GR00T l10 | 0.856 | 0.872 | 0.862 | 0.886 | 0.858 | 0.842 | 0.858 | 0.868 |

RoboCasa：无完整 k（GR00T k=1 main 387/400、π0.5 k=9 main 36/400 停在 timan108 `/scratch/zixuans8/nfe/rc_results/` journal 里，可续跑）。
出图：点已作为 series `teacher alone, k denoising steps (no cache)` 插进 `exp/rit_pareto/analysis/figures/{pareto,groot}_libero_{10,spatial}.json`（脚本 `exp/nfe_baseline/insert_nfe_series.py`，不入库），png/pdf 已重渲染；编辑页 Save JSON 已关（按钮 hidden + `/api/save` 403），Export 照常。
远端：四台 nfe tmux/显存归零；岛树 `/data/openpi_nfe`（h100/weilandserver）、`/scratch/zixuans8/nfe`（timan107/108）与 `/tmp/nfe*` 留着，owner 指示后再清。本地未提交清单见 1.6（另加 `insert_nfe_series.py`、四个 figures json/png/pdf、`figure_editor.html`、`edit_figure.py` 的改动）。

## 7. 融合权重 → key builder × LDA 补充实验（本 session，2026-09-15 写；11:15 更新：§4 Code 与准备阶段完成，G2 R1 NEEDS REVISION → R2 重交 → **R3 APPROVED**（owner 授权审查方直接修复，执行方复核后进 §6 Verify → commit/push；实跑未开始））

### 7.1 已完成（全部 commit 到 `a27a707`，iclr 仓 `9bb5d15`）

- 融合权重 $w_f$ 定案：主方法 = 相位判别 LDA（$w\propto\Sigma^{-1}d$，闭式零 rollout），候选 = 7 维动作误差网格。唯一现行纪要 `docs/iclr/modality_weight_selection.md`
  （§2 公式、§4.5 四臂 A 池重跑、§4.6 统一 1/6 网格 56k 集重做、§6 论文英文句）；实验记录 `exp/weighted_sum/analysis/low_cost_weights_results_claude.md`；
  数据 `exp/weighted_sum/data/{fusion_ablation,grid6}/`（gitignored，`summary_final.json` / `summary_grid6.json`）。
- 终读数（A 池 500 集）：网格最高 0.688 / 0.486 / 0.744 / 0.468（π0.5 Sp / π0.5 L10 / GR00T Sp / GR00T L10）；LDA 最近格差 1.8 / 5.0 / 1.4 / 0.4 pp，只有 π0.5 LIBERO-10 显著（峰在 rs=0 棱上）；动作误差 argmin 四 suite 都显著落后。
- GR00T LIBERO serving 提速已落地（`exp/libero_groot/serve_groot_libero.py --compile-stage1 --stage1-only`；`src/openpi/cache/groot/staged.py` 分位数等价门 + 首连接线程内编译/预热/录制）：单 replica 59 集/分、2.0 GB（eager 20 集/分、5.8 GB）。

### 7.2 下一步：`logs/keybuilder_lda_supplement_plan.log.md`（G1 APPROVED R2 → §4 Code 完成 → G2 R3 **APPROVED**，见 plan 文末 Review Log；下一步 = plan §4 步 3–4：推送 h100 + 预检门 + 两轮实跑）

**已做（2026-09-15 上午）**：下列 1–2 全部完成并留证（`logs/fusion_weight_ablation_run.md` §7.1）：新文件 `exp/weighted_sum/emit_keybuilder_lda.py`、`tests/exp/test_keybuilder_lda.py`（35 passed），改 `fw_lane_pi05.sh` / `lcw_ablation_summary.py` / `lcw_fit_weights.py`；h100 建成 libero_10 `cp1_llm_l0_prefix_mean_pool.pkl`（sha `93d3d35d…`，已拉回本地，§4.1 验收全过）；补标定 3 份、LDA 8 个、yaml 10 份 + matrix 4 份 + `weights.json` + `active_manifest.json` 全部生成并回读验收。**全部在工作树未 staged**（owner 规则：未经指示不 `git add`）。
**G2 之后才做**：3–4（推送 h100 + sha 对账、plan §6 预检门、pool 轮 / LLM 轮、汇总、纪要 §4.7）。

原顺序（保留作对照）：
1. `exp/weighted_sum/emit_keybuilder_lda.py`（templates / final 两阶段）、`fw_lane_pi05.sh` 加 `FW_STAGE2_DEVICE` / `FW_MATRIX`、`lcw_ablation_summary.py` 加 `--input-manifest` 严格入口、`lcw_fit_weights.py` 加 normalizer/退化校验、`tests/exp/test_keybuilder_lda.py`。
2. h100 建 libero_10 的 `cp1_llm_l0_prefix_mean_pool.pkl`（`exp/common/build_in_memory_cache_artifact.py --builder-type cp1_llm_layer_extract --extract-layer 0 --prefix-reducer-type prefix_mean_pool --checkpoint-dir /data/openpi/checkpoints/pi05_libero_pytorch --config-name pi05_libero`，原料 = 建 `cp1_spatial_pool_16.pkl` 的同一批 H5，先找到它的来源目录），逐条对齐验收 → 补标定（libero_10 clip_b32、两 suite llm）→ LDA → yaml → 逐份 load 验收。
3. 预检门（plan §6）：CLIP 按"一臂一端点、W=12"实测显存与连接覆盖；LLM parity `PI05_CHECKPOINT_DIR=... PI05_CONFIG_NAME=pi05_libero uv run pytest tests/cache/test_llm_layer_extract_parity.py --run-manual -m manual -v`；各路径 10 集 smoke。
4. pool 轮（4 臂 × 2 suite，`FW_EVAL_CONC=4`，stage2 meta）→ LLM 轮（1 臂 × 2 suite，stage2 cuda:0，单端点）；拉回 → `--input-manifest` 汇总 → 纪要 §4.7。

### 7.3 设备与现状（2026-09-15 01:00）

| 机器 | 状态 | 本线用法 |
|---|---|---|
| h100 `149.165.153.233` | **11:00 起被另一 session 占用**：4 个 GR00T server（tmux `ort_srv_frontier_rprime_2320[1-4]`，30 GB、util 96%）；本线 kb_build / kb_h5pull 两个 tmux 已完成可清；新增 `/data/openpi/exp_common/db/libero_cache/libero_10`（50 H5）与 `exp/common/data/cache_artifacts/libero_10/llm_layer_extract/` 新库；其余：显存需按预检实测；`/home/weiland/openpi` 是非 git 副本，含 π0.5 库、A 池、`pi05_libero_pytorch` ckpt；`/data/openpi_lg` 的 GR00T 树**过期**（serve_groot_libero 旧、无原 S3 库） | π0.5 lane：`fw_lane_pi05.sh` + timan108 `cache_prune/ops/launch_fleet.sh`（ports 23280–23284，driver 23290） |
| weilandserver | 另一 session 的 online RIT 线在跑（GR00T server :23181–23184、loto/disagreement 进程），显存 20–45 GB 波动 | 本线不碰 |
| timan107 | 空 | 不用 |
| timan108 | 另一 session 48 个 worker（`ort_cli_*`，连 weilandserver :23181/:23183） | 起本线车队前 `tmux ls` 侦察，只动自己的 `cpag*` session，禁 `stop_fleet.sh` 的全局清扫 |

### 7.4 本线踩过的坑（都在今天）

- **`/data/openpi_lg` 与 `serve_groot_libero.py` / `staged.py` 是多 session 共用文件**：另一 session 17:00 用它的 HEAD 版覆盖了含 `--stage1-only` 的远端文件，LIBERO-10 相起服失败一次。推送前先 pull 对方现行版再合并；本地工作树同一文件叠着两边改动，commit 时注意。
- **run_size_eval 的调度合同 = 一臂一端点**（`PureCacheEvalStrategy.plan` 每臂一个 stage，`assign_servers` 一个端点），并发由 `--eval-concurrency` 决定；"60 worker" 不等于单臂 60 连接。CLIP builder 逐连接懒加载模型，显存按该臂端点的 W 算。
- **GR00T r1 harness（`orchestrate_search.py`）启动会 reap timan107 上全部 `lw*` worker**，不能与别的 orchestrator 并发；且一 cell 一 server。要吞吐走 conductor + `fw_groot_conductor.sh`（compile + stage1-only replica）。
- 编译视觉塔的等价门原本卡逐 token 余弦最小值 0.999，低范数 token 会把它压到 0.9；已改分位数判据（`OPENPI_STAGE1_GATE=strict` 可切回）。CUDA graph 录制必须在首个连接的线程里连做三次调用（cuDNN 句柄线程局部），已改。
- tether `bash -lc '...'` 内 `pkill -f` 会匹配到自己的 shell（"child terminated by signal"），用 tmux kill-session 或 `[p]attern`。
- 历史 π0.5 网格在 B 池 n=100、H200 上，与 A 池数字不能并表；GR00T 同 harness 逐集确定（Spatial 100%、L10 93.8%），跨 harness 81%。
