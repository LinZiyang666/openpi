# Session Handoff — RIT-Pareto（K=2 四组已完成；K=3 四组进行中，group 1 在跑；2026-09-01 20:25 CDT）

> **一句话状态**：RIT-Pareto 线正在跑 **K=3 阶梯（FULL / WARM@0.3 / WARM@0.5，无 gate）的 4 组帕累托极限**：顺序 ① spatial-RIT（**20:17 起在跑**，timan108 tmux `k3_g1`）→ ② spatial-GST → ③ l10-RIT → ④ l10-GST；每组每臂 500 集（官方 pruned-500 池），RIT 16 臂、GST 34 格。K=2 的四组（spatial/l10 × no-gate/H-gate）今天 05:24–18:45 全部跑完、已本地化、已出图与报告。**owner /goal 授权：独断、不弹阻塞窗口、不做完不停、监控体系必须在。** 所有代码改动**未 commit**（owner 明示后才 commit/push）。
> **术语**：GST = 网格搜索阈值（分位切）；RIT = 风险索引阈值（单 δ 切风险曲线）；RIT-PL = 分段线性估计量；K3 = 三档阶梯。tau1 = 标定参考（winner 的 x₀.₉ 在查询观测下补完 9 步）。
> **规则**：未经 owner 明示绝不 `git add`/commit/push；不写记忆文件（owner 明令）；handoff 只按要求写；作者 LinZiyang666、英文 why 式 commit、无 AI 署名；共享机杀进程 ps→kill pid；中文回复英文注释；**图 = 散点 + 非支配点前沿线（owner 纠正：不逐点连线）**；server 4 replica / client 48 worker 固定（owner）。

## 0. 权威文档

| 文档 | 内容 |
|---|---|
| `exp/rit_pareto/analysis/figures/*.json` | **图的唯一数据源**（figure spec，schema `rit_pareto.figure/v1`）：六份 —— 四张主图 + 两张 `pareto_acb_<suite>`（ActionCache CP2 baseline，owner 2026-09-05 提供，**保持独立**）。每点 `x/y` + 标注 + `n_ep`，加 series 的样式与图例模板；K=3 两张内含 owner 2026-09-04 跑的 GST-K3 + H gate 格点作为自有 series |
| `exp/rit_pareto/analysis/figures/pareto_{rit,k3}_{libero_spatial,libero_10}.{png,pdf}` | 四张前沿图（K=2 no gate vs H gate + GST 参考；K=3 RIT vs GST + K=2 虚线参考） |
| ~~`exp/rit_pareto/analysis/analysis.md`~~ | **已删除**（owner 2026-09-03 裁定：本线只留图，不留报告）。历史版本见 git `0cfd905`；设计/拓扑/事故记录仍在本文件 §2–§7，K=2 结果摘要在 §8 |
| `logs/rit_pl_ir_ladder_plan.log.md` | RIT-PL 计划（commit `ca15b5e`，G1/G2 记录） |
| `logs/dispatch_surface_sonly_pivot_report.md`、`docs/iclr/latex/sonly_note.tex` | 转向报告与数学 note |
| 本文件 §5–§7 | 拓扑、恢复规程、事故记录 |

## 0.6 RIT-K3 + H gate（规则 `rith`）两组全部完成（2026-09-05 17:50），等 owner 裁决

- **定义**：K=3 RIT 的 16 个 IR 寻址臂（切点与 `k3/rit/*.yaml` 逐字相同，judge 块 diff 为 0）+ K=2 H gate（θ 同 gsth：spatial 0.9773518443107605 / l10 0.9922123551368713，j=3/probe=3/L=6）。规则名 `rith`，臂名 `k3_<sp|l10>_rith_irNN`，每组 16×500 = **8000**。导出：`export_k3.py --rules rit --gate-theta θ`（记录改为按规则命名：`export_record_rith.json`；同时把之前的 `export_record_hg.json` 三处改名为 `export_record_gsth.json`）。yaml 已在 weilandserver 与 timan107 同路径 `/tmp/dsp_shared/rit_pareto/<suite>/k3/rith/`（tgz sha f790866c…，本地副本 `exp/rit_pareto/data/k3_rith/`）。
- **runner**：`run_group_k3.sh` 规则 `rith` → `--eval-gate score_hysteresis`（已推 timan107，含 `export HOME=/home/zixuans8`）。启动：`HOST=timan107 bash exp/rit_pareto/ops/launch_k3_group.sh 8 libero_spatial rith`，之后 `9 libero_10 rith`。健康：`rit_health_k3.sh <suite> rith 8000`。
- **设备空闲检测**：`bash exp/rit_pareto/ops/devices_free.sh`（timan107：最忙卡显存 <1.5 GB、无他人 GPU 进程、无 run_gtp、RAM ≥120 GB、load <8；weilandserver：srv0 在 或 GPU 有 ≥34 GB 可起）→ `DEVICES FREE|BUSY`，exit 0/1。23:30 实测 BUSY：timan107 有另一会话的 run_gtp（48 worker，load 9.4），weilandserver GPU 38 GB（srv0 30 GB + 他用）。
- **闹钟**：Monitor `bxygnkcag`（persistent）：本地 sleep 到 2026-09-05 10:00 后每 30 min 跑一次 `devices_free.sh`，每次发一行 `ALARM check: BUSY|FREE`，FREE 时发 `DEVICES_FREE_START_RITH` 退出 → 由本会话接手：起 srv0（若停，`start_eval_server.sh` 等 `replica_proxy listening on`）→ `HOST=timan107 bash exp/rit_pareto/ops/launch_k3_group.sh 8 libero_spatial rith` → 核验（runner/workers/err/门生效 searched:false）→ 重建运行期 Monitor（120 s 条件触发，DONE 标记 K3_G8_SPATIAL_RITH_DONE）+ 20 min cron 巡检 → g8 DONE 后 audit/pull/aggregate → `launch_k3_group.sh 9 libero_10 rith` → 同样收尾。不画图、不 commit、server 不关。**compact/重启后若 Monitor 丢失，按此重建**（原一次性 cron 已撤，避免双链轮询）。
- **2026-09-05 10:00 闹钟结果 FREE**（另一会话的 ActionCache 已跑完并关掉了 srv0）→ 10:00 `start_eval_server.sh` 重起 srv0（10:04 就绪）→ **10:04 `k3_g8` = libero_spatial rith RUNNING**（timan107 tmux `k3_g8`，日志 `/tmp/rit_k3_libero_spatial_rith.log`，journal `/tmp/dsp_precheck/rit_pareto/libero_spatial_k3_rith/`，8000；冒烟：48 worker、err 0、HOME=/home/zixuans8、门生效）。spatial ≈ 50–90 ep/min → 预计 **~12:30** 完成。
- **g8 = libero_spatial rith：DONE（12:05）**，8000/8000，ok 7596（SR 0.950 总体），审计 OK（16 臂各 500、0 dup、截断 0、attempt 集合一致），门生效（25373 步被门跳过搜索 / 159350 步搜索），全程 cexc/restarts/bigW/zomb 0；已 pull 到 `exp/rit_pareto/data/runs/libero_spatial_k3_rith/`（sha 校验）并 aggregate。
- **g9 = libero_10 rith：DONE（17:50）**（12:05:53 启动，一次跑完），8000/8000，ok 6099（SR 0.762 总体），审计 OK（16 臂各 500、0 dup、截断 0、attempt 集合一致），门生效（78548 步被门跳过搜索 / 442951 步搜索），全程 cexc/restarts/bigW/zomb 0；已 pull 到 `exp/rit_pareto/data/runs/libero_10_k3_rith/`（sha 校验）并 aggregate。
- **收官状态**：两组 raw + aggregate.json 均在本地（`libero_spatial_k3_rith/`、`libero_10_k3_rith/`）；**按 owner 裁定不画图**；cron/Monitor 已全部撤除（17:52）；**weilandserver `srv0` server 已于 20:02 按 owner 指令关闭**（按 PID kill router 2428801 + 4 replica，2 s 内退出，tmux srv0 消失，:23150–23154 不监听，显存 0）；timan107 车队随 runner 退出（0 worker、0 runner、GPU 8 卡各 2 MiB、可用内存 216 GB）；本地改动未 commit（见下）。待 owner：① ~~是否关 server~~（已关）；② 是否 commit（`export_k3.py` 记录命名 / `ops/run_group_k3.sh` rith / `ops/devices_free.sh` 新 / 两组 aggregate 由 gitignore 排除）；③ 是否把两组 rith 叠进四张图（点已导出：`build_figure.py` 新增子命令 `k3-rith`，两组各一份独立 spec `analysis/figures/pareto_k3_rith_{libero_spatial,libero_10}.json`，series `RIT-K3 + H gate` 16 点 + 指向 `pareto_k3_<suite>` 的 `RIT-K3` 无门前沿引用；未渲染 png/pdf 进 figures/，只在 scratch 验证过可渲染；叠进现有四张图需照 gsth 先例改 `build_figure k3/rit`，由画图会话 `b2bf9376` 维护）；④ timan107 `/tmp/dsp_precheck/rit_pareto/*_k3_rith/` 与 weilandserver `/tmp/dsp_shared/rit_pareto/<suite>/k3/rith/` 临时文件是否清理。
- **监控**：已全部撤除（17:52）。运行期：g8 Monitor `brhk9n34k` + cron `d865814a`，g9 Monitor `b0l3wk3c0` + cron `98ad8a81`，均为 120 s 条件触发 + 20 min 巡检，两组全程无一次 ALERT。看守会话 = `4c29be2a`（原 `d61863e2` 的后台 fork，11:47 起接管）。
- 本地改动（未 commit）：`export_k3.py`（记录命名）、`ops/run_group_k3.sh`（rith）、`ops/devices_free.sh`（新）、`data/k3_rith/`（gitignore）；临时测试副本在 `~/.claude/scratch_rit/test_k3_gsth_export.py`（配 `git show HEAD:tests/rit_pareto/test_k3_export.py` 与 `test_rit_k.py` 一起跑，14 passed，含 rith 用例）。
- **WSL 重启后恢复清单（owner 2026-09-05 11:0x 预告重启）**：实验在 timan107/weilandserver 上不受影响；本会话丢失的只有 Monitor `b3rvcr6vc` 与 cron `99bc63a9`。恢复步骤：① `tether exec --timeout 60s timan107 -- bash -lc 'bash /tmp/dsp_precheck/rit_pareto/rit_health_k3.sh <suite> rith 8000'` 看当前组（先 spatial 后 l10）；② 若 `RIT GROUP DONE` 且 runner=0 → 走组末流程（audit/pull/aggregate/切组）；③ 否则按 §4 模板重建 Monitor（120 s 条件触发）与 */20 cron（prompt 指明当前组）；④ 若 timan107 的 tmux/runner 不在而 journal 未满 → audit 后 `launch_k3_group.sh N <suite> rith` 同 journal resume。

## 0.5 K3-GSTH（GST 格 + H gate）两组全部完成（2026-09-04 15:48），等 owner 裁决

- **定义**：K=3 GST 的 34 格切点原样不动（judge 块与 `k3/gst/*.yaml` 逐字相同），cp1.gate 换成 K=2 的 H gate（`score_hysteresis`，θ_low=θ_high=K=2 `export_tau1/export_record.json` 的 `gate_theta`：spatial 0.9773518443107605 / l10 0.9922123551368713，j=3 / probe_interval=3 / L=6）。规则名 `gsth`，臂名 `k3_<sp|l10>_gsth_f..w..v..`。导出：`export_k3.py --rules gst --gate-theta θ --out-dir <suite>/k3`（新增选项，输出 `k3/gsth/`、`arm_matrix_gsth.yaml`、`export_record_hg.json`，与 rit/gst 共存，拒绝覆盖）。yaml 在 weilandserver 与 timan107 同路径 `/tmp/dsp_shared/rit_pareto/<suite>/k3/gsth/`（本地副本 `exp/rit_pareto/data/k3_gsth/`，tgz sha c35f354d…）。
- **拓扑**：server = weilandserver `srv0` 4 replica :23150（22:29 重启，`start_eval_server.sh`）；client = **timan107** 48 worker / 8× GTX1080（`--gpus 8`），代码 = `git archive f11a381`（运行时与 0cfd905 等价）解在 `/scratch/zixuans8/openpi_dispatch_k3`，A-pool `/scratch/zixuans8/openpi/exp/common/data/db_init/libero/<suite>_apool`，libero_sim 原生 EGL（无 nvidia-gl 钩子）。`run_group_k3.sh` 已按 hostname 分支（timan107/timan108），规则 `gsth` → `--eval-gate score_hysteresis --judge-type threshold --warm-tiers 0.3,0.5`。ops 脚本在 timan107 `/tmp/dsp_precheck/rit_pareto/`（run_group_k3.sh / rit_health_k3.sh / audit_k3_group.py / excise_uids.py）；本地 `launch_k3_group.sh` 用 `HOST=timan107`。
- **g6 = libero_spatial gsth：DONE（02:51）**，17000/17000，ok 16550（SR 0.974 总体），审计 OK（34 臂各 500、0 dup、截断 0），门生效（42603 步被门跳过搜索 / 339109 步搜索）；已 pull 到 `exp/rit_pareto/data/runs/libero_spatial_k3_gsth/`（sha 校验）并 aggregate（`_GST_CELL` 已兼容 gsth）。对照无门 gst：低 IR 格加门后 IR 高 1–4 pt、SR 高 1–4 pt（如 (80,0,0) 41.4%/0.944 vs 40.4%/0.908；(40,0,40) 50.6%/0.980 vs 48.7%/0.960），fh=0 的格两者几乎重合。
- **g7 = libero_10 gsth：DONE（15:48）**（02:51 首启因资产缺失全部 raised、02:53 停掉、03:01 重启后一次跑完），17000/17000，ok 13781（SR 0.811 总体），审计 OK（34 臂各 500、0 dup、截断 0），门生效（131940 步被门跳过 / 933957 步搜索）；已 pull 到 `exp/rit_pareto/data/runs/libero_10_k3_gsth/`（sha 校验）并 aggregate。
- **点导出（owner 2026-09-04 16:0x 指令）**：`build_figure.py` 新增子命令 `k3-gsth`，两组各一份独立 spec JSON：`exp/rit_pareto/analysis/figures/pareto_k3_gsth_libero_spatial.json`、`pareto_k3_gsth_libero_10.json`（schema `rit_pareto.figure/v1`，series `GST-K3 + H gate` 34 点 x=IR% / y=SR / label=fh/w3/w5 / n_ep=500，另含指向 `pareto_k3_<suite>` 的 `GST-K3` 无门前沿的引用 series，只存指针不拷点）。重建：`uv run python -m exp.rit_pareto.build_figure k3-gsth --suite <suite> --gsth $R/<suite>_k3_gsth/aggregate.json --out-dir $F`。未渲染 png/pdf 进 figures/（仅在 scratch 验证过可渲染）。
- **收官状态**：两组 raw + aggregate.json 均在本地；**按 owner 裁定不画图**；cron/Monitor 已全部撤除；**weilandserver `srv0` server 已于 20:02 按 owner 指令关闭**（按 PID kill router 2428801 + 4 replica，2 s 内退出，tmux srv0 消失，:23150–23154 不监听，显存 0）；timan107 车队随 runner 退出（0 worker、0 runner、GPU 8 卡各 2 MiB、可用内存 216 GB）；本地改动未 commit（见下）。待 owner：① ~~是否关 server~~（已关）；② 是否 commit（export_k3/aggregate_rit/ops 四处改动 + 两组 aggregate 由 gitignore 排除）；③ 是否要把两组 gsth 加进图（`build_figure`/`render_figure` 链路由另一会话维护）；④ timan107 `/tmp` 与 weilandserver `k3_gsth_arms.tgz` 等临时文件是否清理。
- **监控**：已全部撤除（15:50）。运行期用的是 Monitor 120 s 条件触发（ALERT/cexc↑/bigW≥4/zomb≥2/低内存/STALL/DONE）+ cron */20 定时巡检，模板见 §4。
- **本轮本地改动（未 commit）**：`exp/rit_pareto/export_k3.py`（`--rules/--gate-theta`）、`aggregate_rit.py`（`_GST_CELL` 兼容 gsth，1 处）、`ops/run_group_k3.sh`（hostname 分支 + gsth）、`ops/launch_k3_group.sh`（HOST 变量）、`data/k3_gsth/`（gitignore）。新增 gsth 导出测试暂放 `~/.claude/jobs/d61863e2/tmp/tests_head/test_k3_gsth_export.py`（因工作树里 `tests/rit_pareto/` 已被另一会话暂存删除，未入库；用 HEAD 旧测试 + 新测试跑 13 passed）。

## 1. 当前状态：K=3 四组全部完成（2026-09-02 21:34），等 owner 裁决

- **实验已收官（owner 21:40 裁定）**：无任务在跑，cron 已撤、Monitor 已停，TaskList #14/#15 完成；**weilandserver `srv0` server 已于 21:44 关闭**（:23150 不监听、显存 0、tmux 会话消失）；timan108 车队已随 runner 退出（0 worker）。全部改动已作为**单一 commit `Add RIT-Pareto K=2 and K=3 frontier experiments on the pruned-500 pools` push 到 origin/Ziyang**（§6 Verify：全量 pytest 4981 passed / 14 failed / 60 skipped，14 例全为 HEAD 既有或密封 review_tests，与本次改动无关）。
- 四组结果（全部审计 OK：唯一 uid = 总量、每臂 500、0 dup、截断 0、attempt 集合一致）：
  | 组 | 集数 | ok | 本地 raw + aggregate.json |
  |---|---|---|---|
  | g1 spatial RIT | 8000 | 7048 | `exp/rit_pareto/data/runs/libero_spatial_k3_rit/` |
  | g2 spatial GST | 17000 | 16378 | `exp/rit_pareto/data/runs/libero_spatial_k3_gst/` |
  | g3 l10 RIT | 8000 | 5186 | `exp/rit_pareto/data/runs/libero_10_k3_rit/` |
  | g4+g5 l10 GST | 17000 | 13221 | `exp/rit_pareto/data/runs/libero_10_k3_gst/`（含 `contaminated_uids_1011.json`；g4 事故后剔除 30 截断集 + resume 补跑 9111 集，§7） |
- **图（owner 裁定只画图不写报告）**：`exp/rit_pareto/analysis/figures/pareto_k3_libero_spatial.{png,pdf}`（RIT 前沿 13 点 / GST 前沿 9 点 / K=2 虚线）、`pareto_k3_libero_10.{png,pdf}`（RIT 前沿 14 点 / GST 前沿 17 点 / K=2 虚线）；散点画全部臂，折线只连非支配点。重画见 §1.1 的三段链路（2026-09-03 起 `aggregate_rit` 不再画图）。
- 两 suite 共同观察（未写入 analysis.md，供 owner 参考）：低/中 IR 段 GST 前沿高于 RIT-K3（spatial IR 40–55：GST 0.91–0.97 vs RIT 0.80–0.82；l10 IR 43–66：GST 0.61–0.78 vs RIT 0.49–0.68），RIT-K3 在该段把 18–29% 决策放到 WARM@0.5，而 GST 最优格几乎不用 W0.5；高 IR 段两法齐平。
- **仍待 owner（非阻塞）**：① 是否要 analysis.md §10–12（owner 已裁只画图）；② 根治僵尸 worker 的 src 改动（`examples/libero/episode_runner.py` 在 ConnectionClosed 时重建 client）是否立项；③ 远端临时文件是否清理：timan108 `/tmp/dsp_precheck/rit_pareto/`（raw 已 sha 校验拉回本地，含 g4 备份 `*.pre_excise_142132`、`*.old` 日志）与 weilandserver `/tmp/dsp_shared/rit_pareto/`（shadow H5 唯一副本 41 GB、`h5_shards/`、`table_tau1_k3.part*`）。
- **timan108 远端 raw**：`/tmp/dsp_precheck/rit_pareto/<suite>_k3_<rule>/` + tgz/sha；本地已 sha 校验一致。

### 1.1 画图链路（2026-09-03 改造，三段式）

取数与画图已拆开，`aggregate_rit` 只剩 `aggregate`（raw → `aggregate.json`），画图链路为：

| 阶段 | 模块 | 输入 → 输出 |
|---|---|---|
| ① 取数 | `exp/rit_pareto/build_figure.py` | `aggregate.json`（+ GTP `plot_data.json`）→ `analysis/figures/<figure_id>.json` |
| ② 画图 | `exp/rit_pareto/render_figure.py` | `<figure_id>.json` → `<figure_id>.{png,pdf}`（默认写在 spec 同目录 = 原地覆盖）。**磁盘上没有对应图的 spec 会被拒**（纯数据源），要新建图须显式 `--new` |
| ③ 交互 | `exp/rit_pareto/edit_figure.py` + `figure_editor.html` | 本地网页，**无参数启动**，下拉框只列**已成图的 spec**（纯数据源如 `pareto_acb_*` 不出现、直接点名也 404）；拖点/拖标注/数值精修/隐藏臂/整条 series on-off；Ctrl-Z 撤销；Save 覆盖当前图 json（**引用线的改动写回持有它的 json**），Export 再重渲染所有写过的 spec **及引用它们的图** |

- **所有"认识实验"的语义只在 ①**（哪个 run 喂哪条 series、arm 名 → 标注、成本口径、样式与图例文案）；② 与 ③ 只认 spec。补臂或重跑后只需重建 spec，手工摆过的点位在再次渲染时保留。
- **帕累托前沿永远是导出量**，不进 json：② 每次渲染现算，③ 拖动时实时重算；网页只是预览，Export 走服务端 matplotlib，与静态渲染同一条路径。
- **引用是指针不是拷贝，且可就地编辑**：一条 series 只存 `source_figure` + `source_series` 时，渲染/编辑都从 owner spec 现读，同一批测量只有一处副本。**在任何宿主图里都能直接拖这些点**，Save 时写回 owner json（状态栏列出写了哪些文件），Export 再把宿主、owner、以及引用它们的图全部重渲染。写回前校验坐标有限（坏值 400 拒写，两个文件都不动），且拒绝引用链（owner 必须自己持有点）。指向的 spec 缺失/改名/自身又是引用时，渲染打 warning 并省掉该线，不炸整张图。
- **GST-K3 + H gate 叠加线**（紫色菱形，owner 2026-09-04 裁定：叠在现有图内，不单独成图）：**K=3 spec 持有**这 34 个点（`build_figure k3 --gsth data/runs/<suite>_k3_gsth/aggregate.json` 重建时带上）；**K=2 rit 图里是引用**（`build_figure rit --gsth`）。
- **ActionCache baseline 叠加线**（owner 2026-09-05 裁定：加进 4 张图、可编辑、**json 保持独立**）：`pareto_acb_<suite>.json` 持有每 suite 2 条 series（50-traj lib × N_hit=0/1，各 5 臂，橙 `#e07b39`，圆=N0 方=N1）；四张主图各以**引用**方式叠加这 2 条（`build_figure {rit,k3} --acb`，读同目录 `pareto_acb_<suite>.json` 的样式与图例文案）。在主图里直接拖即写回 acb json。**S6 full lib 两条已按 owner 2026-09-05 指令从全部 6 个 spec 删除**（备份见会话 scratchpad）。
- 现有 spec 一律**原地插入**新 series，不 rebuild —— owner 手改过的 y 值（spatial 32 处、l10 20 处）必须保留。网页里每条 series 前有 `on` 总开关可整条隐藏。
- ⚠ **只有 4 张主图有 png/pdf**（owner 2026-09-05：不要新图）。`pareto_acb_<suite>` 只是数据源，不成图。两处都堵住了：`render_figure` CLI 对磁盘上无图的 spec 直接拒绝（须 `--new` 才新建），网页 Export 的连带重渲染也只刷新已存在的图 —— 所以整目录循环渲染不会凭空造图。图间引用已成环（rit ↔ k3、acb → rit、rit/k3 → acb），连带渲染只做一层，不递归、不死循环。
- ⚠ **图例已 16 条**（8 series × 散点+前沿），`lower right` 遮挡明显；待 owner 定：关掉部分线 / 改 `legend.loc` / 调 `figsize`。
- **成功率吸附**：网页里 y 只能落在 1/n_ep 的整数倍上（500 集臂 ⇒ 0.002 的整数倍，且钳在 [0,1]），拖动/方向键/手输一律吸附；方向键一次正好一集。吸附按 `round(y·n)/n` 算，所以拖回原值与实测值逐位相同、`edited` 标记不会假阳。x（成本）不吸附。
- **写入保护**：写前校验 schema/figure_id/坐标有限，原子写（tmp + `os.replace`），坏状态不会截断好 json。spec 只存「画什么」（点、标注、样式、图例模板，外加 `n_ep` 供吸附），要回实测值就用 `build_figure` 重建。
- **撤销的基准是「上一次保存」**：单点 `↺` 与整图 `Revert to saved` 都回到最近一次 Save/Export 的状态，不是回到实测值；dirty 标志按与该快照的 diff 实算（手动改回去会自动变回无改动）。要回实测值 = 用 `build_figure` 重建 spec。
- 复现四张图（`$R = exp/rit_pareto/data/runs`，`$F = exp/rit_pareto/analysis/figures`）：
  ```bash
  uv run python -m exp.rit_pareto.build_figure rit --suite <suite> \
      --nogate $R/<suite>_ng/aggregate.json --hgate $R/<suite>_hg/aggregate.json \
      --gst-plot-data exp/gate_threshold_pareto/analysis/plot_data.json --out-dir $F
  uv run python -m exp.rit_pareto.build_figure k3 --suite <suite> \
      --rit $R/<suite>_k3_rit/aggregate.json --gst $R/<suite>_k3_gst/aggregate.json --out-dir $F
  uv run python -m exp.rit_pareto.render_figure --figure $F/<figure_id>.json
  uv run python -m exp.rit_pareto.edit_figure      # 无参数；四张图在网页里切换
  ```
  顺序有要求：K=3 的参考线现读 `pareto_rit_<suite>.json`，所以先 build `rit` 再 build/render `k3`。
- 保真核对：用改造前 HEAD(`e6b0ef4`) 的 `plot_suite` 就同一批 aggregate 重画，与新链路输出**逐像素一致**（四张 maxdiff 0）。已入库的两张 K=2 png 与新图有图例框位置差（文案与数据像素相同），是那两张图早于当日后续绘图改动生成所致，以新链路输出为准。

## 2. K=3 实验定义（owner 裁定）

- 阶梯：FULL_HIT → WARM@0.3（3 步）→ WARM@0.5（5 步，新增）→ MISS；verdict 取最便宜的可入档（`ThresholdJudge`：先 `threshold`，再按 `warm_tiers` 列表顺序）。**无 gate**（`always_search`）。
- 成本（`rit_k.tier_cost`，常数来自 `analytic_cost`）：FULL 10.260 / W0.3 46.818 / W0.5 52.733 / MISS 67.519 ms；IR% = Σcost / (N·MISS)。
- 两法同一部署形式（GTP 模板 + `judge: threshold` + `warm_tiers [{θ_w03,0.3},{θ_w05,0.5}]`，同一 ws pkl），只差阈值来源：
  - **RIT-K3**：`exp/rit_pareto/rit_k.py` 三层联合 pinball LP（嵌套 q_w05 ≤ q_w03 ≤ q_full，ε=0.02，α=0.05，结点梯子同 rit_pl；K=2 与 rit_pl 逐位一致有测试），单 δ → 三切点；IR 寻址 20…95 步 5 → 16 臂；切点 +∞ 或被更险档遮蔽的档从 yaml 省略。
  - **GST-K3**：百分比三元组 (fh,w3,w5)，步长 20，fh+w3+w5 ≤ 80，去 (0,0,0) → **34 格**（owner 拍板）；θ = 同一 shadow 表 s 的降序分位（`derive_thresholds` 惯例，累积份额）；部署切点重合的格去重（两 suite 均 0 跳过）。
- 标定：复用 Part I 的 150 集 tau1 shadow cohort（seed 20260901，每 task 15 个 pruned init），`build_dispatch_table --extra-warm-tiers 0.5` 重建表 `table_tau1_k3.jsonl`（多一列 `y_tau5`；guard 证明 s/y7/y10 与 K=2 表逐位一致、行序一致）。spatial 3193 行 / l10 9008 行；y5 中位 < y7 < y10 两 suite 均成立。
- 导出结果：两 suite RIT 16 臂 |gap| ≤ 0.04 pt、GST 34/34 格；l10 RIT ir20 因 full=w03 切点重合只部署 full+warm05。

## 3. 资产位置

- **weilandserver**（server 机，`ziyanglin.com:23150` 1:1 直连；`tether exec` 须 `export PATH=/usr/local/bin:/usr/bin:/bin; export HOME=/home/weiland`）：
  - tmux `srv0`：`serve_policy --replicas 4 --replica-spawn-batch 2 --port 23150`（pi05_libero，~31 GB；`/tmp/srv0.log`；重启脚本 `/tmp/dsp_shared/rit_pareto/start_eval_server.sh`，ready 签名 `replica_proxy listening on`，~4 min）。
  - `/tmp/dsp_shared/rit_pareto/<suite>/`：`shadow_manifest.json`、`shadow_pool/`、`cohort_plan.json`、`cohort_manifest.json`、`h5/shard*/`（150 H5，spatial 11 GB / l10 30 GB，**唯一副本**）、`calibration_retrieval.yaml`、`table_tau1.jsonl`（K=2）、`table_tau1_k3.jsonl`（K=3，+`.weights.npz`）、`export_tau1/`（K=2 RIT 工件）、`arms/`（K=2 yaml）、**`k3/`（K=3：`export_record.json`、`rit/*.yaml`、`gst/*.yaml`、`arm_matrix_{rit,gst}.yaml`）**；l10 另有 `table_tau1_k3.{done,part0-3}.jsonl` 与 `h5_shards/`（并行建表遗留，可删）。
  - `/tmp/dsp_shared/rit_pareto/*.sh`：全部运维脚本；`/data/rit_stage/`：给 timan108 建车队用的 staging 包（可删）；`/data/openpi_dispatch`：代码克隆（与本地工作树同步，含 K3 代码）。
- **timan108**（client 车队，`tether exec` 用 `export PATH=/scratch/zixuans8/dsp_bin:/usr/local/bin:/usr/bin:/bin`）：
  - 环境：`/scratch/zixuans8/libero_sim`（从 weilandserver 搬迁的 conda prefix，EGL 535 hook 指向 `/scratch/zixuans8/nvidia-gl`）、`/scratch/zixuans8/dsp_bin/conda`（shim）、`/scratch/zixuans8/openpi_dispatch`（代码 + `packages/openpi-client`，runner PYTHONPATH 必须前置 `packages/openpi-client/src`）、runner venv `/scratch/zixuans8/openpi/.venv`（py3.12）、A-pool `/scratch/zixuans8/openpi_dispatch/exp/common/data/db_init/libero/<suite>_apool`。
  - `/tmp/dsp_shared/rit_pareto/<suite>/{export_tau1,arms,k3}`：与 weilandserver 同路径同 sha（arm yaml 内为绝对路径）。
  - `/tmp/dsp_precheck/rit_pareto/`：`run_group.sh`（K=2）、`run_group_k3.sh`、`rit_health.sh`、`rit_health_k3.sh`；K=2 四组 raw（`<suite>_{ng,hg}/` + tgz/sha）；K=3 `<suite>_k3_<rule>/`；`libero_10_ng/journal.jsonl.pre_srvdown` 与 `contaminated_uids_srvdown_1130.json`（11:30 事故备份）。
- **timan107**（2026-09-03 起 K3-GSTH client）：`/scratch/zixuans8/{libero_sim(原生 EGL), openpi/.venv(py3.11), openpi(旧克隆+A-pool), dsp_bin, openpi_dispatch_k3(f11a381 归档)}`；旧 `openpi_dispatch` 克隆已损坏勿用；`/tmp/dsp_shared/rit_pareto/<suite>/k3/gsth`、`/tmp/dsp_precheck/rit_pareto/`。
- **本地（全部未 commit）**：`exp/rit_pareto/`（`shadow_cohort.py`、`export_rit.py`、`emit_arms.py`、`rit_k.py`、`export_k3.py`、`aggregate_rit.py`、`config/task_order_*.json`、`ops/*.sh` 21 个运维脚本副本、`build_figure.py`、`render_figure.py`、`edit_figure.py`、`figure_editor.html`、`analysis/figures/`（6 spec + 4 png + 4 pdf）、`data/shadow/`、`data/runs/{libero_spatial,libero_10}_{ng,hg}/` 四组 raw + aggregate.json）；`tests/rit_pareto/`（3 文件 44 例，全绿）；additive 改动 `exp/gate_threshold_pareto/run_gtp.py`（`--judge-type/--eval-gate/--gpu-ids/--warm-tiers`）、`exp/dispatch_surface/build_dispatch_table.py`（`--noise-sidecar` 可选、`--extra-warm-tiers`）；`logs/README.md`、本文件。定级 L1（exp 脚本 + additive 标志，未动 src）。

## 4. 监控体系（compact/重启后必须重建，session 级）

- **L1** `rit_health_k3.sh <suite> <rit|gst> <total>`（timan108）：一行 progress/ok/runner/workers/server/err/**cexc**（main.py `Caught exception` 次数 = 被截断集数）/**restarts**（worker 重启数）/**bigW**（RSS ≥ 6 GB 的 worker 数）/**zomb**（没有到 :23150 established 连接的 worker 数）/freeGB；`RIT GROUP DONE`；`ALERT runner exited|server DOWN|runner dead|low memory(<30 GB)|ballooned workers(bigW≥4)|zombie workers(zomb≥2)`。err 是累计 traceback 行数，g4 已被 1011 刷到 3.6 万，**看 cexc 不看 err**。
- **L2 Monitor**（persistent，180 s，**owner 22:20 裁定：只做条件触发，不推里程碑**）：每 3 min 跑 L1，只在 ALERT、STALL(6 轮 = 18 min 冻结)、DONE、tether 连续 3 次无回复时发事件；DONE 退出。写法：`tether exec --timeout 60s timan108 -- bash -lc 'bash /tmp/dsp_precheck/rit_pareto/rit_health_k3.sh <suite> <rule> <total>'`，`grep ALERT` 直出、`d==prev` 计 stall、`RIT GROUP DONE` → `K3_GN_<SUITE>_<RULE>_DONE` 退出。
- **L3 cron** `*/20 * * * *`（**定时巡检归 cron**）：同一 L1 + `tether node ls -a`；prompt 含处置规程（§6）；compact 后存活。
- 当前挂载：无（rith 两组 17:50 全部完成，cron/Monitor 已撤）。TaskList：#16–#20 全部完成。

## 5. 拓扑速查

server weilandserver 4090 48 GB：4 replica，`--replica-spawn-batch 2`，`OPENPI_SERVER_GPU_MEMORY_LOCK=0`，bundle 由 conductor 每臂热切；client timan108 3×A5000，48 worker（每 EGL worker ≈ 0.65 GB CPU 内存、~8 GB 显存/卡），`--gpus 3`；吞吐 spatial ≈ 95 ep/min、l10 ≈ 30 ep/min。tether exec 静默 10 min 上限，长跑一律 tmux；`tether push` 远端路径须绝对、父目录先建；timan108 `allow_roots=[/home /tmp /srv]`（不能 push 到 /scratch，先 push /tmp 再 mv）。

## 6. 故障处置规程（已实战验证）

- **server 死**（`pgrep -af "[s]erve_policy.py --replicas"` 为 0，`:23150` 不监听）：runner 会把在途集记成 `failed, accepted`（同一 drain ts 的一批）。步骤：停 runner（`kill -INT`，等退出，`kill -9` worker）→ 从 journal/per_step 剔除该批 uid（按 `max(ts of failed)` ±2 s 选 uid，写备份 `.pre_srvdown` 与 uid 清单）→ `start_eval_server.sh` 等 ready → 轮换日志 → 同 journal 续跑（resume 跳过完整臂，episode 级续跑）。
- **weilandserver OOM 根因**（11:30）：另一会话经 tether 在同机跑 tether 仓库 Go 测试（5789 个 `exe` 进程，unit 峰值 247 GB）→ kernel OOM 杀 replica、tmux server、tether agent（systemd `Restart=on-failure` 5 s 自动拉回）。若再发生：同上规程；建议那边加 `systemd-run -p MemoryMax=`。
- **timan107 首启 l10 全部 `episode … raised`（MuJoCo `Error opening file …/.cache/libero/assets/...msh`）**（2026-09-04 02:51）：libero 把资产解析到 `$HOME/.cache/libero/assets`；通过 tether exec 新建的 tmux 服务器继承 tether agent 的 `HOME=/srv/local/zixuans8/tether-home`，那里只有一份缺 l10 场景文件的部分拷贝（spatial 能跑、l10 不能）；timan108 的 tmux 服务器早已存在、HOME 是 NFS 家目录，其 `/home/zixuans8/.cache/libero/assets`（407 MB）完整，所以此前 l10 组没事。修复：`run_group_k3.sh` 显式 `export HOME=/home/zixuans8`，并 `rsync` NFS 拷贝覆盖 /srv/local 拷贝。症状识别：启动 1 分钟内 raised 数百、progress=0、runner 很快 exit 0（队列被重试耗尽）；处置：停 runner（kill -INT + kill -9 worker）、修资产、同 journal 重启（无终态记录则无需剔除）。另：`tether push` 会丢可执行位，launch 脚本已改为 `bash <script>` 调用并 chmod。
- **timan108 runner/worker 死或 OOM**：`tail -40 /tmp/rit_k3_<suite>_<rule>.log`、`dmesg -T | grep -i killed`；轮换日志后同 journal 续跑。
- **GPU 被他人占满 → EGL FatalError 假失败**（timan107 事故）：`run_gtp --gpu-ids` 跳过该卡；受污染的 journal 整体作废重跑。
- **timan108 worker 内存膨胀 → OOM → 1011 截断 + 僵尸 worker**（g4 10:46–10:57 实战）：症状 = freeGB 骤降到个位数、bigW 十几个（RSS 7–12 GB，GPU 显存同涨 2–5 GB，渲染器泄漏，起因不明）、dmesg OOM kill、随后 server 端 keepalive ping 20 s 无 pong 批量 1011 关连接（同一秒一批）。后果两类：① main.py 步内 `except Exception → break`，集被截断记成终态 `failed`（client_timing `steps` < 上限）= **污染，必须剔除补跑**；② `episode_start` 在死连接上抛 → worker 报 `episode … raised`，driver 判 retriable 只重派不落 journal，但该 worker 永远拿着死连接空转刷 traceback（僵尸）。**僵尸的真正危害**：`episode_runner._ensure_client → select_bundle` 在死连接上立即抛 → driver 判 retriable 重派 → 同一僵尸再领再抛，每集 `max_episode_retries=3`（4 次）耗尽后 scheduler 静默记 `done_fail` 且**不落 journal**；6 个僵尸 19 分钟烧掉 9081 集，runner 之后以 exit 0 "完成"。因此 **zomb ≥ 2 立即按 PID kill**；runner 若在 progress < total 时退出（`ALERT runner exited`），不是故障而是队列被烧光：audit → excise 截断集 → `launch_k3_group.sh N suite rule` 同 journal resume 即可补齐。处置：`ps`→按 PID `kill -9` 膨胀（≥6 GB）与僵尸（无 :23150 连接，两次采样取交集）worker，conductor 自动重启并重派在途集（无终态记录、无损失）。不要用 pkill。根治需改 `examples/libero/episode_runner.py` 在 ConnectionClosed 时重建 client（src 改动，等 owner 裁）。
- **完整性审计**（每组完成后）：journal terminal 行 = unique uid = 总量、dup 0；failed 集的 per_step 决策数不异常少（spatial <42、l10 <100 判可疑）；per_step attempt 集合 == journal attempt；**failed 且 client_timing.steps < 上限（l10 500 / spatial 200）= 截断污染**（`audit_k3_group.py` 已内置，输出 `truncated_failed_uids`）；四组 K=2 与 K3 g1–g3 均 0 异常。

## 7. 事故记录（时间线）

01:54 K=2 group 1 在 timan107 起跑，GPU 5/7 被占 → 7 次 EGL FatalError 假失败 → 作废重跑（`--gpu-ids 0,1,2,3,4,6`）；02:50 timan107 宕机（12:05 才重启）→ 03:19 weilandserver 本机 20 worker 跑 870 集（唯一偏离 48 的区间）→ 03:37 自建 timan108 车队接管，同 journal 续跑；11:30 weilandserver 被另一会话 Go 测试打爆内存，OOM 杀 replica，group 3 剔除 48 个在途假失败后续跑；K=2 四组 18:45 全部完成，审计 0 异常。K=3：18:56 建表（并行 4 片后 20:07 完成），20:15 srv0 重启，20:16 smoke 三档均出现，20:17 group 1 起跑。

- **2026-09-02 10:41–11:15（g4 l10 GST）**：10:41 freeGB 174 → 11:01 10；19 个 worker RSS 膨胀到 7–12 GB（GPU 显存同涨），10:46–10:49 内核 OOM 杀 5 个；10:52:31/10:52:51/10:53:11/10:55:51/10:57:11 五批 server 1011 keepalive 关连接 → 30 集截断记 failed（臂 f00w20v20）+ 6 僵尸 worker。11:08 按 PID 杀 19 膨胀 worker（free 回 153 GB、显存回 10 GB/卡），11:11 杀 6 僵尸（raised 行停增），conductor 共重启 30 个 worker，cexc 定格 30，进度未中断。健康脚本/审计/剔除脚本随即固化（§4/§6）。**14:19** runner exit 0 提前退出于 7919/17000：事后从 raised 行分布（每 5 万行均匀 2632 条）确认 6 僵尸在 10:52–11:11 已把后续臂的 9081 集全部重试耗尽（36418 raised ≈ 9100 × 4）；14:21 剔除 30 截断集，14:22 `k3_g5` 同 journal resume 补跑 9111 集，**21:34 完成**（resume 段 cexc/restarts/bigW/zomb 全程 0），审计 OK。
## 8. K=2 结果摘要（详见 analysis.md Part I）

spatial：no gate 实测 IR 37→93 %、SR 0.772→0.998；H gate 40→93 %、0.906→0.992；H gate 在 40–60 % 段与 GST 同库参考持平到 −2 pt，>70 % 一致。l10：no gate 27→93 %、0.468→0.872；H gate 40→93 %、0.662→0.860；H gate 在 40–66 % 比 no gate 高 8–20 pt，vs GST 50–62 % 低 2–5 pt、74–86 % 高 2–4 pt（均在 CI 内）；顶点 ≈ 纯 teacher 0.868。
