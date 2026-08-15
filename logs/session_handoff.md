# Session Handoff — ablation_study（cache 有效性双方向消融）

> 更新：2026-08-14 09:15（**🏁 全线收官：主矩阵 7000 + 4b 5000 + 延迟 pass 140 ep 全满账；Phase 5 报告终稿 `exp/ablation_study/analysis/analysis.md`；plan log EN-5 已记；等 owner commit 指示**。存活服务：wls srv0/srv8001(BASIC lat 版)/acts7012+sml7011(spatial)、zy10 srv8005(单 replica BASIC)+双 sidecar+7 占位器、农场闲置——均未关（等 owner 令）；守护全部自然退场。延迟锚点：teacher 步 690ms(4090)/114ms(H200)，hit→ACT 步 −78/−79%，hit→SmolVLA 4090 −24% 但 H200 **+36% 反亏**。历史头部：）（**Phase 4 主矩阵两套件 7000/7000 满账完成**：spatial 3500 SR=0.900、l10 3500 SR=0.685。l10 臂级：cache_baseline .704 / hit_act .888 / hit_smolvla .830 / miss_act .466 / miss_smolvla .474 / pure_act .794 / pure_smolvla .640——hit 替换超 baseline、miss 替换重降质、pure 与独立基线一致。spatial 臂级早查：cache_baseline .930 / hit_act .990 / FULL_HIT 率 ~60%。**spatial 4b 正在 wls 本机 conductor 跑**（tmux p4bsp，8 workers→127.0.0.1:8001，journal wls 本地 data/runs/p4b_sp_journal.jsonl；srv8001 已重启为 v2 载入 EN-4 config——旧进程内存白名单曾拒 composite；worker spawn 需 PATH 前置 ~/miniconda3/bin）。l10 4b 待 zy10 回线（回线后 **--replicas 2 --replica-spawn-batch 1 铁律**，B=2 加载峰爆 32G 两次实证 pod 自动重启）或 spatial 4b 完后 wls 串行。守护：Monitor bckojodb7(p4bsp)+bcx238pd3(zy10 回线)+cron cfc035b9。⚠⚠ kill 与 relaunch 严禁同一 shell 调用（pkill/pgrep 正则匹配 relaunch 文本自杀，一天三次实证）
> Plan：`logs/ablation_study_plan.log.md`（G1/G2 APPROVED + Execution Notes **EN-1..EN-4**，全部裁决与偏差记录在文末）
> 账本：`exp/ablation_study/analysis/sr_ledger.md`（全部全量评估 SR 的单一 tracked 记录）
> Memory：`project_ablation_phase0_live_run.md`（拓扑与陷阱全集，与本文互补）
> Goal 状态：owner goal「前进 phase 3 于 phase 4 门停止」**已达成**——不得启动任何 Phase 4/4b 臂，等 owner 放行口令。

## 一、状态总览（2026-08-13 晚）

- **Phase 0 ✅** 蒸馏采集：两套件各 500 ep，内容校验 PASS，三端 sha 一致。
- **Phase 1 ✅ + EN-3 重冻结**：全部候选 EN-2 全量评估（官方 pruned_init，与训练差集池逐字节零交集）；冻结=**套件级统一 step 020000**（标准配方终点，零选择偏差，band 降披露；旧 per-task/002000 表达已按 owner 令删除，演化记录唯一存于 plan log EN-3）。
- **Phase 2 ✅**：权重汇集（20/20 ACT 020000 sha 对 freeze 校验 PASS→wls `act_selected/` 链接布局）；`act_manifest_<suite>.json` ×2 装配；正式 1-cell conductor smoke（hit_act spatial 10/10 ep，FULL_HIT=override=sidecar 计时=144 三方相等，命中率 70.9%）；`uv run pytest tests/ablation_study/test_manual_e2e.py -m manual --run-manual` **PASSED**。
- **Phase 3 ✅**：teacher anchor 复跑 + 学生锚点 + 曲线补全（下表）。
- **Phase 4b 已批（EN-4）**：kinematic 学生路由帕累托，工件全备（见 §三-3）。

### 锚点参考系（Phase 4/5 对照用，全部官方 init 50/任务）

| 锚点 | libero_spatial | libero_10 |
|---|---|---|
| **teacher（Pi0.5 本次同协议复测）** | **0.974**（500 ep） | **0.868**（500 ep，t8 最难 0.46） |
| 学生 ACT @20000（冻结即锚点） | 0.966 | 0.766 |
| 学生 SmolVLA @20000（冻结即锚点） | 0.954 | 0.630 |
| 历史协议锚（仅参考，勿用于对照） | 0.95 | 0.83 |

anchor json 本地存档：`exp/ablation_study/data/anchors/libero_{spatial,10}_teacher/results_*.json`（t107 亦有正本）。SmolVLA l10 曲线已完整单调：0.344→0.404→0.496→0.504→0.550→0.630（2k→20k，账本）。

## 二、存活拓扑（compact 后接手先核对）

| 机器 | 现役 |
|---|---|
| **weilandserver (4090, tether 节点 weilandserver)** | tmux `srv0`=pi05 纯推理:8000→expose 14008（**owner 明令勿关**）；`srv0b`=pi05 第二实例:8002→14026（anchor 加速件，Phase 4 延迟 pass/4b 可复用，可按需关）；`srv8001`=routed server:8001→**14025**（spatial hit_act bundle 已载，**换臂只需 conductor 热切换勿重启**）；`acts7002`=ACT sidecar:7002（**spatial** manifest 全 10 任务）；lerobot venv `~/lerobot_venv`；repo `~/openpi`@d2e4293+工作树（含 4b bundle 已解包） |
| **ziyang10 (H200, tether 节点 jupyter-ziyang10)** | **Phase 4 l10 全栈 live**（setsid 无 tmux）：pi05 routed server :8000（log /tmp/zy10_srv8000.log）→ expose `zy10-p4`=**linziyang.top:14022**；ACT sidecar :7002（l10 manifest，/tmp/zy10_act7002.{log,jsonl}）；SmolVLA sidecar :7001（l10 020000，/tmp/zy10_sml7001.{log,jsonl}）；栈占 ~12G 显存（owner 授权 ≤48G）；config.py 已更 a371f35f；l10 pkl/act_selected 链接/SmolVLA+ACT t6-9 权重（sha 5/5 对 freeze 匹配）/d1 校准 jsonl 全落位；⚠ 无 rsync 二进制（传输走 tar-over-ssh@wls-ssh:14024）；训练正本勿删；HOME=/home/ziyang10 |
| **timan107 (48核+8×1080, tether 节点 timan107)** | 农场 `/scratch/zixuans8/equeue/` + 58 tmux worker w0-57 **空转待命**（Phase 4 可复用为 conductor 车队）；repo `/scratch/zixuans8/openpi`@d2e4293+4b bundle（config.py sha 与本地一致 a371f35f）；libero_sim conda + lerobot venv + uv .venv；`export HOME=/home/zixuans8` 必须；sshd expose 14010 |
| **timan107 双 conductor** | tmux `p4sp`（spatial 主矩阵，journal `exp/ablation_study/data/runs/p4_libero_spatial_journal.jsonl`，log /tmp/p4sp.log，servers 14025）+ tmux `p4l10`（l10 主矩阵，journal `p4_libero_10_journal.jsonl`，log /tmp/p4l10.log，servers 14022）；各 --workers 8 --gpus 8 --trials 50；健康脚本 /tmp/p4{sp,l10}_health.sh（TOTAL=3500）；smoke journal p4sp_smoke(10/10 成功)/p4l10_smoke(19/20) |
| 守护件 | L2 Monitor `bun0qign6`(spatial)+`bhgrxnzcd`(l10)（300s 轮询，milestone/ALERT/STALL/DONE）+ L3 cron `8a8c673b`（6,20,34,48 分兜底巡检，含 conductor 同 journal relaunch/expose 断重建/Monitor 重装指令）。Monitor 会因进程重启静默死，cron 必须兜底重装 |

三端 `src/openpi/cache/config.py`（EN-4 改动）sha256 前缀 `a371f35f` 一致。

## 三、Phase 4 → 5 执行手册（owner 放行后照做）

1. **Phase 4 主矩阵（每套件 7 臂 × 500 ep，SR 主跑）**。conductor 在 t107 tmux：
   ```
   export HOME=/home/zixuans8; cd /scratch/zixuans8/openpi && PYTHONPATH=. .venv/bin/python \
     exp/ablation_study/run_ablation_eval.py \
     --arm-matrix exp/ablation_study/config/arm_matrix_<suite>.yaml \
     --task-suite <suite> --servers linziyang.top:14025 \
     --workers 8 --gpus 8 --trials 50 \
     --journal exp/ablation_study/data/runs/p4_<suite>_journal.jsonl \
     --per-step-out exp/ablation_study/data/runs/p4_<suite>_per_step.jsonl \
     --expected-discordance 0.15 \
     --preflight-approval exp/ablation_study/config/common/preflight_approval.yaml \
     --conda-env /scratch/zixuans8/libero_sim
   ```
   preflight 在 n=500,q=0.15 下功效 ~7% 必判 underpowered → O7 预批 `underpowered_ok` 放行（审计 json 自动落盘）。ConductorDriver ep 级 resume：崩溃后同 journal 直接 relaunch。
2. **sidecar 侧按套件切换（关键！）**：跑 <suite> 前 wls 必须（a）`acts7002` 载对应 `act_manifest_<suite>.json`（现载 spatial；换 l10 要重启 sidecar 换 --manifest）；（b）SmolVLA 臂需起 `:7001` sidecar 载 `<suite>/smolvla/checkpoints/020000/pretrained_model`（**当前未起**；换套件同样要换 checkpoint 重启）。sidecar 重启等 CUDA teardown 释放端口。
2a-新. **端口迁移（2026-08-13 晚，owner 令"端口不能重合"）**：另一 session 共用 ziyang10（markov_sufficiency :8020 双 replica + 一套 15:58 起的同款 ablation 三件套占 8000/7001/7002，归属未明勿动）与 wls（markov :8010）。**我方 l10/4b 栈全部迁新端口：server=8005、SmolVLA=7011、ACT=7012**；small_at_hit/small_at_miss/kin_route 全部 yaml 的 routing 已 sed 成 7011/7012（local+t107 已同步；spatial 主矩阵历史数据用旧端口跑完，不受影响）。wls 现役 l10 sidecar tmux=`acts7012`/`sml7011`。⚠ ziyang10 的 ss 输出可能整体为空不可信，判监听用**新命名 log 文件**的签名；杀进程只按 PID（/tmp/zy10_our_pids_v2.txt）。spatial 主矩阵 **3500/3500 满账完成**（success 3151=0.900；pure_smolvla 曾缺 150 由 resume 补齐）。
2b. **owner 令（2026-08-13）——p4sp 完成后 wls 承接 l10**：wls sidecars 切 l10（acts7002 换 act_manifest_libero_10.json、sml7001 换 l10 smolvla 020000；wls 已备 l10 pkl/act_selected×10/manifest/双套件 d1 jsonl），p4l10 conductor 同 journal 重启为 `--servers linziyang.top:14022,linziyang.top:14025 --workers 20 --server-workers 12,8`。⚠ conductor 是**臂粒度**分 server（`server_assignment[yaml_id]`）→ 记录 wls 分到的臂并写 plan log **EN-5 跨硬件披露**（部分 l10 臂在 4090、其余在 H200）。l10 主矩阵完 → l10 4b 走 ziyang10(14022)、spatial 4b 走 wls(14025, acts7002 切回 spatial manifest)，并行。
3. **Phase 4b（主矩阵后）**：同拓扑，conductor 换 `--arm-matrix exp/ablation_study/config/arm_matrix_4b_<suite>.yaml`（5 阈值臂/套件，kinroute FULL_HIT→ACT:7002，校准 jsonl 已在 wls `exp/weighted_sum/data/<suite>/kinematic_phase5/d1/`）。产出与历史 cache 帕累托（`exp/weighted_sum/analysis/<suite>/kinematic_phase5/`）同图比较。
4. **延迟 pass**（SR 主跑后另跑）：`--workers 1` + server 侧 `OPENPI_MONITOR_LEVEL=BASIC` + `--timing_csv_dir`；srv0b 可作纯推理延迟对照端。
5. **Phase 5**：`analyze_ablation.py --preflight-artifact <per_step>.preflight.json` → `exp/ablation_study/analysis/analysis.md`。叙事=EN-3/EN-4 口径：标准配方自然强度谱系（0.630-0.966）、无害替换（spatial 双格+l10 ACT）/降质可测（l10 SmolVLA）双 regime、teacher 对照用 0.974/0.868；caveat：三格出带披露、锚点共用测试集（冻结零选择偏差）、underpowered_ok、EN-4 的 broadcast-前置披露。

## 四、陷阱清单（血泪浓缩；详表在 memory）

- `conda run` 块缓冲：client 日志/tmux pane 全程空白、results json **结束才写**——判活用 **ps CPU time 两次采样**，勿误判挂死；判 server 活用 TCP 探测，勿 grep websockets 握手 Traceback（良性）。
- pytest manual 用例必须 `--run-manual` 旗标（conftest auto-skip），加 `-m manual`。
- t107 农场端口表有**两份**：worker2/3 读 `sm_ports.json`、worker4 读 `sm_ports2.json`——新增映射两表都写（原子写：tmp+os.replace）；port-map 改动不影响 in-flight client（要重排必须 kill+requeue，equeue 幂等）。
- pkill 模式与自身命令行重合会自杀（用 `task_[5]` 方括号）；`conda run` wrapper+child 双条目非双跑；equeue claim mtime 不可判无主（用进程表重建）。
- t107 tether allow_roots 仅 /home /tmp /srv（/scratch 文件先 cp /tmp 再 pull）；WSL→wls 直连仅 22 端口可靠；ziyang10 无 tmux（setsid）、32G RAM 静默 OOM、pod 不定期回收。
- 权威工件：cache pkl 在 wls `exp/common/data/cache_artifacts/`（spatial=36cd0f3b/430MB、l10=1.1GB；**本地 WSL 副本目录互换勿用**）；norm_stats 权威=c0ee3c1a（已同步）；所有训练系列 owner 明令保留不删。
- EN-4 语义：hit 路径 `broadcast_action` 在 executor override **之前**（interceptor L761）——kinematic verdict 的动作历史=cache/teacher 侧（与 cache 系统同构，Phase 5 披露）；composite 进 routing 必须空 WARM tier（config.py 静态校验强制）。

## 五、未 commit 工作树（owner 指示后 commit；勿 git add）

`exp/ablation_study/`：`train_student.py`（EN-1）、`train_{act,smolvla}.py`、`select_student_checkpoint.py`、`config/common/{split_*,stem_map_*,preflight_approval}.yaml`、22 个 `select_freeze_*.yaml`（EN-3 020000 终版）、`config/act_manifest_*.json` ×2、`config/kin_route/` ×10 + `config/arm_matrix_4b_*.yaml` ×2（EN-4）、`analysis/{render_sr_ledger.py,sr_ledger.md}`、`data/anchors/`（新，teacher anchor json ×6）。
`src/openpi/cache/config.py`（EN-4 白名单+composite 校验；tests/cache+ablation 1152 passed）。
`logs/`：plan log（EN-1..4+状态头）、本 handoff。
`assets/.../norm_stats.json`（更正+.bak）；`tests/ablation_study/`（用户/linter 微调过 test_router_hooks 与 test_distill_builder——保留勿回退）。
