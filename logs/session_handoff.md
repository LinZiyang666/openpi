# Session Handoff — gate_threshold_pareto **已收官** + 4090 故障处置（2026-08-21 07:15 更新）

> Authority: Execution。owner 已 override G1/G2（见 §6）。无人值守 mandate 生效中。

---

## 0. 实验已完成（2026-08-21 06:56 EVAL DONE）

**32,000 ep 全部跑完**，两 suite 各 16,000。异常臂 `gtp_ws_sp_fh30` 已用干净数据重跑替换（493/500 = 98.6%）。

- **终版报告与图**：`exp/gate_threshold_pareto/analysis/`（analysis.md + plot_data.json + 两 suite 各 png/pdf），已按规则收编进 `exp/data_authority/analysis/gate_threshold_pareto/`（6 文件 MANIFEST，validate + verify 三类检查全空）。图另存 `C:\Users\lzy66\Downloads\gtp_logs\`。
- **核心结论**：spatial 上 ws 前沿全程支配 cs（同 teacher ratio 高 0.7–1.9 个百分点）；l10 上 ws 在中段占优（0.50 处 +2.3 个百分点）但 ~0.67 处两线交叉、右端 cs 略反超。省教师调用的代价在 l10 显著更高。
- **已决（2026-08-21）**：owner 定义 `inference_ratio = (N_req·s1 + N_miss·(s2+s3))/(N_req·(s1+s2+s3))`，s 取 CUDA-Graph 档三段延迟（=0.152+0.848·teacher_ratio），四图两轴并存于 analysis/；原始 journal/per_step 尚未拉回本地、尚未在 `records/` 登记指针台账（EVAL DONE 后文件已定，可随时做）。
- **待授权**：commit/push（工作树新增 `exp/gate_threshold_pareto/analysis/`、`exp/gate_threshold_pareto/analyze_gtp.py`、`exp/data_authority/analysis/gate_threshold_pareto/`）；关 server 需 owner 确认。⚠ **keepwarm 保持常驻，不要关**（owner 裁定，后续其它负载也要用）。
- **监控已全部收口**：cron `2a2efd38` 与两条 Monitor 均已停。timan107 tmux `gtpe` 已退出；weilandserver `srv0`（4 replica）与 `keepwarm` 仍在跑。

---

## 1. 实验拓扑与配置（历史记录，已收官）

**gate_threshold_pareto 实验**（plan: `logs/gate_threshold_pareto_plan.log.md`，含 §5.2 事故全记录）：

- 设计：老 threshold-pareto 重做。**只 d=1**；gate 换 **N4 混合门**（`score_hysteresis`, θ_low=θ_high=各库 warmup 上 0.85 分位, j=3, probe_interval=3, **L=6**）；**warm_start 整条关闭**（verdict 二值）；`f_FH ∈ {0.05..0.80}` 16 档；**4 个库**（ws=weighted_sum 底座 / cs=cache_size S3 × spatial/l10）；评测集=**完整 A-pool 500 init**（泄漏已实测排除 0/50）。**4×16×500 = 32,000 ep**。
- `inference_ratio` 新算法 owner 说暂缓——per-step 全量落盘（hit_type/searched/cp1_score），任何口径可事后重算。
- 进度（19:56:44）：**spatial 14,215/16,000（44.4%），success 13,589，err=0，~79 ep/min**；l10 未开始（cron 自动发射）。预计 spatial ~20:20 收口，l10 过夜（~8-12 h）。
- 代码：`exp/gate_threshold_pareto/`（libraries.py=四库唯一声明处 / emit_gtp_yamls.py / solve_gtp.py / run_gtp.py）。**commit `ef9d9cc` 已 push origin/Ziyang**。`run_gtp.py` 的 `arms_with_work_left()` resume 过滤（按 distinct task_uid）**必须保留**——resume 时为已完成臂走 stage 会触发 bundle 切换风暴（曾打死 GPU，§5.2）。
- 工作树有 3 个**他人改动未 commit**（勿动勿提交）：`exp/ablation_study/latency_bench/analysis/analysis.md`、`logs/rl_router_operations.log.md`、`logs/session_handoff.md`(本文件)。

### 1.1 拓扑

```
timan107 (tmux gtpe, 64 worker, conda /scratch/zixuans8/libero_sim)
   │  公网直连（不走 tether broker）
   ▼
ziyanglin.com:23100  （交换机 NAT 1:1 → weilandserver:23100；ufw 已放行 23100-23199）
   ▼
weilandserver tmux srv0: serve_policy --replicas 4 --port 23100
   boot bundle = exp/gate_threshold_pareto/config/libero_spatial/warmup/gtpw_ws_sp.yaml
   日志 = /home/weiland/gtp_logs/srv_<mmdd_hhmm>.log（持久路径，勿放 /tmp——重启会清）
weilandserver tmux keepwarm: GPU 保温 v6（§3，实验全程必须陪跑）
```

- journal/per-step：timan107 `/scratch/zixuans8/openpi/exp/gate_threshold_pareto/data/eval/libero_spatial/`（l10 将落 `.../libero_10/`）
- timan107 client 日志：`/tmp/gtp_eval_sp.log`（l10: `/tmp/gtp_eval_l10.log`）
- weilandserver sudo 密码：`/home/weiland/.claude/jobs/204fbb99/tmp/weilandserver_sudo.txt`（repo 外，勿写入任何 tracked 文件）；dmesg 必须 sudo 才有内容（无 sudo 返回空≠零 Xid）

### 1.2 l10 发射命令（cron 会自动做；手动 fallback 用此模板）

发射前**必须**：keepwarm tmux 存活 且 GPU 温度 ≥50 °C。

```bash
tether exec timan107 -- bash -lc '
cd /scratch/zixuans8/openpi
tmux kill-session -t gtpe 2>/dev/null
tmux new -s gtpe -d "cd /scratch/zixuans8/openpi && PYTHONPATH=. /shared/nas/data/m1/zixuans8/miniconda3/bin/uv run python -m exp.gate_threshold_pareto.run_gtp --arm-matrix exp/gate_threshold_pareto/config/libero_10/eval_matrix.yaml --phase eval --task-suite libero_10 --servers ziyanglin.com:23100 --workers 64 --gpus 8 --trials 50 --journal exp/gate_threshold_pareto/data/eval/libero_10/journal.jsonl --per-step-out exp/gate_threshold_pareto/data/eval/libero_10/per_step.jsonl --apool-record exp/ablation_study/cache_size/config/apool_libero_10.yaml --apool-dir /scratch/zixuans8/openpi/exp/common/data/db_init/libero/libero_10_apool --conda-env /scratch/zixuans8/libero_sim 2>&1 | tee /tmp/gtp_eval_l10.log"'
```

l10 的库：ws_l10 pkl 在 weilandserver `exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl`、cs_l10 在 `/data/openpi/ablation_study/cache_size/artifacts/cache_size_libero_10_all_S3.pkl`——都已在 server 侧，内容摘要与权威副本核验一致（`exp/data_authority/records/` 有台账）。

---

## 2. GPU 故障全案（今日主线事故，已收敛）

**这张 48G 改装 clamshell 4090 有确定性硬件缺陷**：冷启动（≤36 °C）陡热爬坡 × 满带宽显存流量 ⇒ 显存接口时序余量塌陷 ⇒ **静默算错**（4069 万/209 万 compare errors，零 Xid 零报警）或静默算死挂起，进而 `Xid 31 MMU Fault @0x0 → 175 GSP timeout → 154 Reset Required`。当日 5 次推理崩溃 + 3/3 冷烧机全灭；**≥44 °C 起温 2/2 全过**；memtest 稳态与有效陡坡下都干净=无静态坏点，是接口级问题。软件全洗清（bundle 风暴假说/batching 假说/驱动/背板均排除或证伪）。

- **英文送修报告**：`C:\Users\lzy66\Downloads\gtp_logs\REPAIR_REPORT.md` + weilandserver `/home/weiland/gpu_fault_reports/`（含全部取证日志）。
- **判烧机三件套**：errors + **proc'd 是否推进** + 终判行——只看 errors 会被"静默挂死后的僵尸零读数"骗（实测被骗一次）。
- **wedge 后软重启不可靠**：D 态进程 + GSP RPC 串行超时会把关机钉死（实测一次，OFFLINE 不回）。软重启 5 分钟不回 ONLINE ⇒ 提请 owner 物理重启，勿反复软重启。
- 该卡承担本实验是 **owner 知情决定**（保温协议护航）；静默算错前科记录在案。
- memory 有两条：`reference_weilandserver_4090_unstable`、`feedback_keepwarm_pure_thermostat`。

---

## 3. 保温脚本（实验的生命线）

`/home/weiland/gtp_logs/gpu_keepwarm.py`（v6），tmux `keepwarm`，日志 `/home/weiland/gtp_logs/keepwarm.log`。

- **纯恒温器，温度是唯一输入**（owner 裁定：不看进程/利用率——轻负载压不住温度时就该并行加热；对重负载不干扰由物理保证）。
- 参数：**54 °C 介入 / 62 °C 让出**（零 kernel）；≤60 °C 近带区紧轮询 ~0.5 s（负载骤停结温 2 s 自由落体 10 °C）；实测谷值 50-53 °C，距 44 安全线 6 °C 余量。
- 显存 **408 MiB**（1536² 矩阵 + `CUBLAS_WORKSPACE_CONFIG=:16:8` + torch context 地板 ~400M）。再往下唯有裸 CUDA C（~250M），owner 未要求。
- **待验证**：1536 矩阵的加热功率未在真实空闲窗口实测（部署时实验正跑）——spatial→l10 切换空隙它会自然上场，看 log 里 engage→disengage 爬升时长即可。功率弱也不破坏安全（最坏=持续加热平台停低点）。
- **规程（两份 devices.md 已写入，2026-08-20 22:40 更新）**：实验前必开并确认 ≥50 °C；**owner 裁定为常驻服务，不随实验结束关闭**（缺陷与具体实验无关，后续其它负载同样依赖）——只有送修/拆机/长期停用才关。⚠ 重启后 tmux 会话消失、不自动恢复，而冷启动正是最危险窗口：重启后第一件事是重挂保温 + 确认 ≥50 °C，之后才起任何 GPU 负载。

重启模板：
```bash
tether exec weilandserver -- bash -lc 'export HOME=/home/weiland; tmux kill-session -t keepwarm 2>/dev/null; sleep 2; tmux new -s keepwarm -d "cd /home/weiland/openpi && /home/weiland/.local/bin/uv run python /home/weiland/gtp_logs/gpu_keepwarm.py 2>&1 | tee -a /home/weiland/gtp_logs/keepwarm.log"'
```

---

## 4. 监控体系 v2（当前在跑的全部）

### L1 — 健康脚本（timan107 `/tmp/gtp_health.sh`；本地母本 `/tmp/claude-1000/-home-weiland-projects-openpi/2973268f-bb39-47cb-9ba5-97851cd2667a/scratchpad/gtp_health.sh`）

一行总览：`progress/sp/l10/success/cond/active/srv/err/ram_avail/swap_free/orphan_workers` + 终态行（`EVAL DONE` / `PHASE_DONE libero_spatial` / `ALERT ...`）。要点：`SRV_HOST=ziyanglin.com SRV_PORT=23100`；孤儿判据跟随 `$SRV_PORT`（勿硬编码端口）；client 日志错误计数用**结构化屏蔽**（`Exception ignored in:` 块跳过，不是数量相减）；RAM<20G / 孤儿>0 / dmesg 新增 OOM 均 ALERT。调用：`tether exec timan107 -- bash -lc 'bash /tmp/gtp_health.sh'`。

### L5 探针脚本（weilandserver `/home/weiland/gtp_logs/gtp_srverr.sh`；母本 job tmp `/home/weiland/.claude/jobs/204fbb99/tmp/gtp_srverr.sh`）

输出 `SRVERR/HARD/REAL_TB/HANDSHAKE_NOISE/ALIVE/FREE/TEMP/FAN/THROTTLE`。要点：**握手噪声结构化屏蔽**（TCP 探活会造 `opening handshake failed`+链式 Traceback，awk 按块跳过）；**ALIVE=监听端口计数**（`ss -tln | grep -cE ":(2310[1-4]|800[1-4])"`——replica 是 spawn 子进程，cmdline 无 serve_policy.py，进程名计数不可靠）；nvidia-smi 全部带 `timeout 20`，挂起输出 `GPU=UNRESPONSIVE`（区分"GPU 死"与"节点掉"）。

### L2 — Monitor `bh6te733d`（persistent，300 s 轮询 L1）

每 10% 里程碑（高水位 numeric sticky，从 30% 起）/ ALERT / STALL（6 轮=30 min 冻结，报一次即重置）/ `PHASE_DONE` / `EVAL DONE`（触发即 exit）。探针空返回也报 ALERT。

### L5 — Monitor `bewj7nhod`（persistent，90 s 轮询 gtp_srverr.sh + keepwarm 存活）

新增 SRVERR / ALIVE<4（连续 2 次）/ **KEEPWARM=0（连续 2 次 ⇒ "cold-ramp protection lost"）** / THROTTLE / 持续 ≥80 °C（连续 2 次）/ ≥76 °C 新高水位（每档一次）/ `GPU=UNRESPONSIVE` 单列。

### L3 — Cron `b4b29292`（`13,53 * * * *`，session-only）

兜底巡检 + **spatial 收口自动发射 l10**（条件：`PHASE_DONE libero_spatial` 或 sp≥16000 且 cond=0 且 l10<16000；**发射前强制验 keepwarm 存活 + 温度 ≥50**，命令见 §1.2）+ ALERT 按 skill §4.5 先诊断（不擅自重启 server；GPU wedge 时软重启 5 min 不回=提请物理重启）+ progress=32000 时标记 EVAL DONE 并 CronDelete 自停。

### L4 — agentchat：owner 明示跳过，未挂。

### 已收口的后台任务：无其它存活（历史 watcher 全部结束）。

---

## 5. 今日其它交付（均已完成）

1. **`exp/data_authority/` 权威数据登记地**（G1+G2 owner 双豁免，见 `logs/data_authority_plan.log.md` §0 与 Review Log）：records/ 台账 5 条 + analysis/<任务>/ 收编规则（artifact_layout.md §1.2 注册 registry 类目录）+ registry/verify/collect 三工具 + 79 测试。随 `ef9d9cc` 入库。
2. **4 pkl 考据**：均 50 轨迹（ws_sp 例外=49，caveat 已登记）；weilandserver 副本与权威副本 pickle 字节不同但**内容摘要逐位相同**（replicas 字段已记）。
3. **ziyanglin.com:23100-23199 直连入口**：devices.md §2.5.1/§4.0（优先级：weilandserver 先直连后 tether expose）；ufw 规则已加；实测双向通、区间边界正确。
4. **devices.md 两份同步**（本地 `/home/weiland/projects/dist_experiment_control/docs/` + weilandserver `/home/weiland/dist_experiment_control/docs/`，sha 一致）：直连入口 + 4090 故障与保温协议 v5/v6 + 软重启警告 + 烧机三件套判据。
5. **REPAIR_REPORT.md** 英文送修报告（Downloads + weilandserver 两份）。

---

## 6. 流程状态与边界

- **G1/G2**：data_authority 与 gate_threshold_pareto 两案均由 owner 明令 override/豁免（原话与范围记录在各自 plan log §0/Review Log）。**豁免≠APPROVED，勿转述为"审过"**。
- **无人值守 mandate**：不弹阻塞窗口；commit/push/关 server/关 keepwarm 仍需 owner 明示。
- **goal 已 clear**（原"有序开展实验，不做完不停"）。
- EVAL DONE 之后的路径：skill §5（关 server 提议须 owner 确认 → 数据 pull 回本地 + sha 校验 → 本地分析 → results.md → commit 授权）。inference_ratio 新口径等 owner 给定义。

## 7. 接管 checklist（compact 后第一件事）

1. `tether exec timan107 -- bash -lc 'bash /tmp/gtp_health.sh'` — 看 progress/err/srv。
2. `tether exec weilandserver -- bash -lc 'bash /home/weiland/gtp_logs/gtp_srverr.sh; tmux ls'` — 看 SRVERR/ALIVE/TEMP + keepwarm/srv0 会话在不在。
3. 确认 Monitor `bh6te733d`/`bewj7nhod` 与 Cron `b4b29292` 存活（compact 不杀它们；若丢失按 §4 命令重挂）。
4. 若 spatial 已收口而 l10 未起：按 §1.2 手动发射（先验 keepwarm+温度）。
5. 出现 `Xid`/`GPU requires reset`：按 §2 处置（保数据 → 软重启 → 5 min 不回提请物理重启 → 恢复时 server 起 23100 + conductor resume，resume 过滤自动生效）。
