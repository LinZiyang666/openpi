# Session Handoff — RoboCasa365 跨场景 cache 实验（T7 ws_search 收口版 v7）

> ⚠ **本文件不是 `logs/session_handoff.md`**（那份属 X14/X15 RL Router session，勿动）。

**Status**: `T7 ws_search Round-1 已收口（2026-08-22 13:3x 本地）`。owner /goal 终点改令执行完毕：**132/132 筛选轮 + 机理解剖 + pi05 Stage-0 容量探测（4 replica PASS）**；锚点臂/加密轮/pi05 全臂按改令取消（cache 将重构）。本 handoff 是下一轮（cache 重构后）的进入点。

## 0. 本轮定论（一屏读完）

1. **权重空间**：v2 唯一强载体（边际单调升，0 权 0.117 → 高权 0.18-0.20）；rs 有>0 即饱和的辅助（0.120→~0.18 平台）；v0/v1 单调有害（1.0 权直落 0.019）。榜首 `grid2 v2@87.5/rs@12.5`=0.269，但 **48/132 与之统计打平**（8 trials 只能圈"好区域"）。
2. **任务两极化机理**（详见 `analysis/robocasa365_seed_anatomy.md`）：seed 重抽"物体类别+位置+干扰物+臂初始 ±4°"；固定家具类天然对位（0.43-0.48，且**反超 teacher**）；4 个 PickPlace 因 n=5 库仅覆盖 19% 类别而结构性死亡（~1.5%，teacher 0.4-1.0）——判"纯 cache 不可迁移"，从搜索指标剔除（双口径）。
3. **pi05 服务容量**（`analysis/t7_capacity_probe.txt` pi05 节）：weilandserver **4 replica 线性 PASS**（165→625 ep/h，0 err，余 17G），timan107 32 worker 无瓶颈。
4. 全部产物：`analysis/ws_search_round1.md`（主报告）+ `robocasa365_seed_anatomy.md`（解剖）+ `ws_search_matrix.html`/`pickplace_first_frames.html`（证据页）+ `ws_anatomy.json`/`ws_object_probe.json`（探测数据）。数据面：132 cell journal/summary 双源（weilandserver `~/openpi/exp/robocasa365/data/ws_search/groot_tp/` 为主 113 + timan107 `/scratch/zixuans8/openpi_rc365/...` 旧纪元 19）。

## 1. 下一轮（cache 重构）直接可用的资产

- **设施**：编排器全套升级已 commit——共享工作队列（无尾部空转）、`--cid-order stratified`（防族偏样本）、`--run-prefix`（多轮不互踩）、`--agent-c-host`（第三工作机）、`pack_agent_c_gpus`（**timan1 单卡优先打满铁律**）。
- **机器**：weilandserver（keepwarm 铁律不关；pi05 4 replica 已验）+ timan107（32 worker 编制）+ **timan1 就绪**（4×A6000 共享机，EGL 原生，岛 174630 文件验收过；`--timan2-workers N --agent-c-host timan1 --timan2-gpu-order 0,2,3`，GPU1 他人常占避开）+ timan108（仍卡管理员重启）。
- **统计工装**：`analyze_ws_search_stats.py`（配对符号翻转/初始状态支配/权重边际/--exclude-tasks 双口径）；配对合法性根基=同 (task,seed) 同初始状态。
- **重构建议**（round1 报告 §5）：库按物体池×位姿定容而非统一 n5；特征需编码目标位置（物体中心/相对目标）；prompt_emb 应入检索（RoboCasa 指令随物体变，与 LIBERO 不同）；v2 主载体先验需跨场景重验。
- **加密轮设计保留**：配对打平 top-8 + 2 阴性对照 × 32 trials、`--run-prefix ws2`、9 任务集——重构后直接复用。

## 2. 纪律（不变项）

- keepwarm 恒温不关；他 session 进程/端口（23152-55、23166/lab23166）不动；`tests/review_tests/` 密封；共享机禁宽杀按 PID；复制校验四判据；**汇报用本地时间**（America/Chicago）；两 session 共享 checkout——**stage 只按显式路径清单**（X15 曾把本线未 commit 文件扫进其暂存快照）。
- 陷阱全集：plan §9（时区鬼影/banner 假阳/GL 帧缓冲耗竭（≥13 env 同进程必炸，逐任务子进程）/会话事件递送冻结（恢复后先全量对账）/诊断先跑控制组/长探测逐步落盘）。

## 3. 快速定位

主报告与证据全在 `exp/robocasa365/analysis/`；plan=`logs/robocasa365_ws_search_plan.log.md`（§9 全史 §10 截断令）；本轮终局 commit 见 git log（Round-1 closeout）。
