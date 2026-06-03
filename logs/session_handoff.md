# SESSION HANDOFF — weighted_sum libero_10 复刻（Stage 4 进行中，全量恢复手册）

> ⚠ **compact 后第一件事：从头读完本文件全文**，再读 plan `logs/weighted_sum_libero10_replication.log.md`。
> 假设你对之前对话**零记忆**——本文件 = 你的全部上下文。最后全量重写：**2026-06-01 14:08 CST**。
> **当前核心**：Stage 4 进行中。**d1 完整 DONE**（kinematic 237cell, SR 峰 **0.91**/mean 0.69, aggregate+analyze+报房间带 Pareto 图）。**d3 run-eval 跑中**（双 server, tmux `s4eval`, journal ~6200/23700 ~26%）。**无人值守 mandate 生效**（完全权限，不起 background，沟通走聊天室 `019e749f-c5f4-7ce0-9666-4b9a5d8e9af3`，owner 必回）。
>
> ⚡ **2026-06-01 全量更新（compact 必读，覆盖下文一切旧描述）**：
> **A. 进度**：Stage 4 跑 d1+d3 两 base。d1 ✅全 DONE。d3: warmup✅+verify-raw✅+emit-eval✅+run-eval🔄(**故障恢复后单 server 续跑 ~48%+**)。
> **✅ 故障已恢复(2026-06-01)**：12:09 双 server(ziyang10+xuanlel2)同时 OFFLINE→conductor 退出(完成 114 yaml/journal 11311)。owner 重 pod ziyang，**恢复(起 server+relaunch)是 agent 职责非 owner**(owner 严厉纠正"自己查 skill")。我自主恢复：起 ziyang server(`weiland.top:14000`, 3replica pi05, handoff §6)+relaunch **单 server worker40** 同 journal resume。**⭐认知修正**: ConductorDriver 是 **ep-level crash-safe resume**(driver.py:134 `replay_done_uids` 含 done+failed，按 task_uid skip)→**直接同 journal resume 即可，不需过滤 journal**(旧铁律⑥"过滤保留完整 yaml"是误解；per_step `open(w)` overwrite 安全)→skip 11311 ep 续跑剩余 12389(112 完整 yaml 不重跑+2 部分 yaml 自动补齐+123 新 yaml)。当前**单 server**(ziyang 14000)，xuanle 仍 OFFLINE(上线再扩双)。journal 备份 bak_failover。监控: L1 单server版/L2 `bgyzor6bt`/L3 cron 见 CronList(`32cad226`)/L4 `bsg489p2g`。failover relaunch: `--servers weiland.top:14000 --workers 40 --server-workers 40 --journal 同路径`。
> **⚠ GPU 应急(owner 2026-06-01 授权)**: ziyang 是共享 GPU，20:40 被他人占用 free 骤降到 2-3G(OOM 风险)。owner 储备进程 **PID 756239**(`./cuda/gpu -g 5` 占 5.5G，我方容器可见可 kill)，GPU free 逼近上限时 owner 授权 `tether exec jupyter-ziyang10 -- bash -lc 'kill 756239'` 释放 5.5G 应急。已纳入 cron(`113203be`)：free<5G→kill 756239；我方 server OOM 崩→先 kill 756239 再判断 free 够否重启(他人占满 free<15G 则等释放不盲目重启)。**GPU 显存实时 Monitor `bh6vop1pq`**(45s 轮询，free<6G emit 通知我手动 kill；owner 2026-06-01 20:44 要求"OOM 前反应"，auto checker 禁自动 kill daemon 故 kill 由 agent 手动点)。**⚠ 756239 已于 20:52 kill 用掉(gone)，GPU free 回升 13G**；之后再逼近上限**无更多储备**，需 owner 重起储备进程或协调占用方(Monitor emit 时我房间报 owner 协调，别再盲目 kill 已 gone 的 756239)。
> **⚠⚠ timan107 维护故障(2026-06-01 ~21:48)**: 管理员维护 timan107(client)，tether **exec channel 卡死**(节点 ONLINE 但 exec 命令不返回，最简 echo 90s Terminated)，**L1/L2/cron 依赖 exec 全失效**，只能靠 `tether exec jupyter-ziyang10` 看 ziyang server 推理活动(WARM_START/step N)间接判断 d3 在跑。owner 称 **timan107 client 随时可能下线**(下线后上线时 owner 教恢复)。**数据备份受阻**: d3 数据(journal 11311+/per_step 114 yaml)在 timan107 `/scratch`，tether pull 文件 channel 可用但 allow_roots 只 `/home /tmp /srv`(/scratch 被拒)，exec 卡无法 cp 到 /tmp 备份。**待 owner 判断 /scratch 是否清盘**: 不清→client 上线后 relaunch conductor 同 journal ep-level resume 零丢失；可能清→owner 在 timan107 本地 `cp -r .../d3 /home/zixuans8/d3_backup` 我再 pull 备份。conductor 仍在 timan107 tmux `s4eval`(单 server worker40)。
> **⚡ 2026-06-02 更新**: ①conductor 跑到 **138/237 yaml** 后被 **owner 关闭**(d3 暂停，数据 /scratch 保留)。②owner 重启 ziyang server→**新端口 `weiland.top:14002`**(name=s4-d3-resume，我重新 expose；旧 14000 owner 重启后 REVOKED)。③timan107 **exec 仍卡**(无法 relaunch)，已给 owner ssh relaunch 命令(`--servers weiland.top:14002 --workers 40`)。④**cron `999cda9e` 监控 timan107 exec 恢复**(恢复后我自己 relaunch 用 14002，cond=0 则起 cond>0 则确认 progress)。⑤**exec 卡时用 pull 监控**(关键发现): `tether pull timan107:/tmp/s4_d3_eval.log` 数 `grep -c "flushed per_step"`=relaunch 后完成 yaml(总=114+flush) + `tether exec jupyter-ziyang10` 看 ziyang server 推理活动(pull 文件 channel 可用，只 exec channel 卡；**timan107 比 ziyang 慢 ~5h14min 时区**，别因 conductor 日志时间戳"旧"误判卡)。⑥**session 重启会丢 cron/Monitor**(已发生：cron 全丢，靠 handoff 恢复)，L4 watcher 裸进程跨 session 存活。
> **⏸ 实验已暂停(owner 2026-06-02 21:34"暂停实验,可以继续的时候我叫你")**: d3 停在 **138/237**(数据 /scratch 保留)，**ziyang server 留着不动**(weiland.top:14002, name=s4-d3-resume, 6replica ready, owner 21:35"留着服务器不动")。timan107 exec 仍卡。**cron `10e68ac3` 暂停待命模式**(每小时:17，只验 L4 watcher 存活 + 兜底查房间 owner 消息，**不自动 relaunch、不试 exec**)。**等 owner 明确说"继续"才 relaunch**。L4 watcher `bsg489p2g` 推 owner 消息。
> **✅ 实验已恢复续跑(2026-06-02 17:25, owner"timan107上线了重启实验")**: timan107 exec 恢复。发现 owner ssh 跑的 conductor 卡死在 90%(21412)——**连旧端口 14000**(server 重启后 REVOKED)。**timan107 维护破坏环境(NFS `/shared/nas` 没挂载→uv+conda 命令都没了)**。**3 个 fix**: ①新端口 **`weiland.top:14002`**(我重 expose name=s4-d3-resume) ②**`.venv/bin/python` 替代 uv**(`/scratch/zixuans8/openpi/.venv/bin/python`，uv 在 NFS 没了) ③**conda shim `/tmp/cshim/conda`**(假 conda，把 `conda run -p <env>` 转发到 `<env>/bin/python`；owner 授权装临时 conda 实验后清；shim 最轻量，测试 import libero/openpi_client OK)。**relaunch 命令关键 3 点**: `.venv/bin/python` + `export PATH=/tmp/cshim:$PATH` + `--servers weiland.top:14002`(完整见 cron `2c930e5a`)。resume 生效 journal 续涨(21412→21480, util 32%)。worker 数 ~116(76 旧连 14000 死 worker 自生自灭 + 40 新连 14002)。监控: L1(`/tmp/s4_d3_health.sh` s1 探 14002) / cron `2c930e5a`(20min 进度监控+完成 aggregate+清痕迹) / L4 `bsg489p2g`。**⚠ 实验结束清 conda shim 痕迹**: `rm -rf /tmp/cshim /tmp/conda_shim_conda /tmp/s4_d3_relaunch*`(owner 要求)。
> **🚨 STALL 卡死等 owner 重启 timan107(2026-06-02 18:00+)**: relaunch 后续跑到 **21544/23700(90.9%)** 又卡死。**根因**: timan107 8 卡 GTX1080 GPU 显存被 **116 worker 占满**(76 旧连 14000 死端口没退 + 40 新)→新 worker MuJoCo EGL offscreen framebuffer 分配不到显存崩(288 次)。**worker 卡在 GPU 操作内核态(EGL driver call)，pkill -9/xargs kill 全 timeout 40-90s 杀不动**(SIGKILL 排队等 GPU call 返回；pgrep 只读秒级，kill 要动就卡)。我无法清 worker/释放 GPU。**已发 urgent 求 owner 重启 timan107 或 nvidia-smi --gpu-reset**。数据 journal 21544+per_step 在 /scratch 保留。**cron `e9cb036a` 盯守**(检测 timan107 重启 worker=0 → 自动: 重 push+部署 conda shim[/tmp 重启丢] + 确认 ziyang 14002 + relaunch[.venv/python+PATH cshim+14002] + 验 EGL=0/util非0)。owner 处理完 timan107 我自动接管恢复。
> **✅ STALL 已解决续跑(2026-06-03 00:17)**: owner **本地 sudo 杀 worker**(116→4，绕过 tether kill 卡 GPU 内核态问题，本地权限强；非重启故 /tmp/cshim conda shim 仍在)→GPU 释放(8110 free)。我 relaunch(.venv/python+PATH cshim+14002)→**EGL 错误 0**(之前 288，GPU 有显存不崩)+journal 续涨(21544→21553)+cond 2+workers 44(40 新+4 残留)+util 30%。从 90.9% 续跑剩 ~22 yaml。进度监控 cron `75823cee`(20min，含 **GPU 显存监控防再满**：EGL 暴涨+journal 停→报 owner 本地杀 worker；完成→aggregate→analyze→报 SR→画重叠图→清 conda shim 痕迹)。L2 进度 Monitor 已 TaskStop。
> **✅✅ d3 EVAL DONE + 收尾完成(2026-06-03 ~02:40)**: d3 从 90.9% 续跑到 **23700/23700(100%)** conductor 自然退出(all stages done)。**SR 峰 0.88**(cell `ws_d1_kin_g5_p2__fh0.2_ws0.4`)/mean 0.72/min 0.50(对比 d1 kinematic 峰 0.91, Stage3 threshold d3 峰 0.97)。收尾已完成: aggregate-summary(237 rows)✅ + analyze(g1-g5 decision + pareto_overlay.png/pdf)✅ + 发房间(DONE+SR+pareto图+decision结论)✅ + CronDelete `75823cee`✅ + 停 L2 monitor✅ + 清 timan107 残留 4 worker(GPU 释放 8110 free)✅ + conductor 已退✅ + **d3 数据全量本地化**(`data/kinematic_phase5/libero_10/d3/`: journal 23700 + per_step/237 + g1-g5_decision + per_yaml_summary 237 + pareto png/pdf；删了空 per_step.jsonl，真实 per_step 在 per_step/ 目录)✅。**decision 洞察**: g1-g4 大部分组内 inconclusive(Δ<0.05 noise floor=100ep 噪声)，少数 conclusive(g3 p1-state pat-1-2→0.88, g4 p2-action drop-off-jerk→0.75)。**⏳ 待 owner 定 always-inf baseline**(已问房间：跑则保留 conda shim+force-MISS 全/抽样，不跑则清痕迹)→然后 commit(等 owner 明确)。**⚠ conda shim 仍在 `/tmp/cshim`**(baseline 决定后清: `rm -rf /tmp/cshim /tmp/conda_shim_conda /tmp/s4_d3_*relaunch* /tmp/s4_d3_clean*`)。**⚠ always_warm_journal 缺失**(d3 没跑 always-warm anchor，relaunch 时漏，故 d3 无 base policy anchor 对照；d1 有)。timan107 worker=0/conductor=0/GPU free 8110，agentchat watcher(pid 14337)保留。
> **B. 端口/server**：ziyang=`weiland.top:14003`(load libero_10 2640) + xuanle=`weiland.top:14002`(name=xlt2)。eval 双 server 80(40,40)，两台 cache 都 libero_10 2640。warmup 单 server ziyang(带 `--warmup_dump_root /tmp/s4_warmup_dumps`，eval 不需)。
> **B2. xuanle failover 预案(owner 2026-06-01 预授权)**：xuanle 机器**真 OFFLINE**(`tether node ls jupyter-xuanlel2`=OFFLINE，**区分** broker 瞬断/expose 失效——那种先重连/重 expose 不 failover) → 自主 failover 到 ziyang 单 server(不再问 owner)：①kill s4eval + `pkill -9 -f "[w]orker_entry"` ②relaunch `--servers weiland.top:14003 --workers 40 --server-workers 40` **同一 journal**(resume 跳过已 done) ③验 resume(log "skipped N done") ④房间报。单 server worker40 吞吐减半但已 done 不丢。已内置 L3 cron step4(id 见 CronList)。
> **C. d1 结果**：237cell SR max **0.91**/mean 0.69/min 0.50；峰值 cell `ws_d1_kin_g5_p1__fh0.2_ws0.2`=0.91；最省高SR `ws_d1_kin_g5_p2__fh0.2_ws0.5` inf_ratio 0.658(省34%推理)仍保 0.91。数据 `data/kinematic_phase5/libero_10/d1/`(journal 23700 / per_step 237 yaml / per_yaml_summary 237行 / g1-g5_decision.json / pareto_overlay.png)。对比 Stage3 threshold d1 峰 0.93(kinematic 接近 2pp)。重叠图脚本 `/tmp/plot_d1_overlay.py`(Stage3 4-frontier + kinematic d1, WARM=0.75 两边一致)；图 `/tmp/d1_kinematic_vs_stage3.png` 已发房间。
> **D. d3 现状**：cfg=`spatial16_ws_d3_best_libero10`(无robot_state/depth3/tw=[.5,.3,.2]，v2_spec 已加+assert==d3 winner yaml+pytest27)。warmup super raw **26730行**(dump winner_id 全命中); verify-raw **HARD GATE PASSED**; emit-eval **237 yaml 0 skip**; run-eval 双 server tmux `s4eval`, journal=`data/.../libero_10/d3/journal.jsonl`(~26%, ~7h 剩)。
> **E. 今天 9 条铁律(踩坑总结，最重要)**：①super warmup run-warmup/emit-warmup **必带 `--cfg-id`**(否则用 libero_spatial 默认→检索不到→全 NaN) ②换实验/换 cfg/换 cache **必重启 server**(load_cache_config 不 reload 已有 backend，旧 cache 残留) ③emit-eval/长任务用 **tmux**(tether exec 10min 超时切断，进程会继续但 exec 断) ④**tmux kill-session 不杀脱离的 worker**→必接 `pkill -9 -f "[w]orker_entry"`(echo/pgrep 里用 `[w]orker_entry` char-class，否则 pkill 自匹配 shell argv→被 signal 杀) ⑤**worker 启动连不上 server 就卡死，不自动重连**→server 崩必 relaunch conductor ⑥**relaunch 前过滤 journal 保留完整 yaml**(kinematic per_step per-yaml flush survives crash; done=per_step 目录 *.jsonl basename, journal 只留 yaml_id∈done，备份原 journal，清部分 yaml 让整跑) ⑦**server kill 用 setsid detached**(pkill/kill serve_policy 会 signal 杀 exec shell) ⑧**expose 端口循环试拿固定端口**(broker 轮询，多 expose 几次命中 14002，rm 非目标的) ⑨**inference_ratio 公式两边一致 WARM=0.75**(=WS@t with t=0.5，owner 确认；非两套公式)。
> **F. monitor = skill §4.3 完整 5 层架构**(owner 2026-06-01 亲自在线指示重启所有 monitor，要 skill 多层职责分离，**不要全塞一个 cron**)：
> - **L1** health script `/tmp/s4_d3_health.sh`(timan107，**2 行**：①progress=done/23700 success cond s1 s2 err ②**SYS ram/disk_free/load1/workers/zombie**[client 系统健康]；信号行 EVAL DONE / ALERT[实验类: 双server DOWN/cond=0; 系统类: RAM<12G/disk<20G/orphan workers>240(正常160=80×2)/zombie>20]；改字段 `tether push --force`)。
> - **L2** 进度 Monitor `task b110qtxof`(persistent，每180s 轮询 L1，sticky milestone[50/75/100%]/ALERT/STALL[6轮=18min]/DONE 才 echo；high 起步 25 因已过)。
> - **L3** cron(每15min 8,23,38,53，**id 见 CronList**[当前 `981fcaab`，会随 prompt 调整变化]，**兜底层 + 系统健康度 + xuanle failover**：①跑 L1[实验+client 系统健康] ②`tether exec jupyter-ziyang10`/`jupyter-xuanlel2` 查 **server GPU vram + serve_policy 进程数**[正常 ~58G used/85G free；**⚠srv 进程数 4-6 transient 不作崩溃信号**——曾因 srv=4 误判差点误杀，server 崩看端口持续DOWN/vram突增>75G/progress不涨+err涨] ③验 L4 watcher 存活 ④**xuanle OFFLINE→failover ziyang 单 server worker40**(见 B2) ⑤兜底查房间 ⑥ALERT 分类发房间 ⑦DONE 兜底推进；健康只记一行不抢 L2 播报)。
> - **L4** agentchat watcher Monitor `task bsg489p2g`(persistent，`agentchat watch state` jq 过滤 unread>0，房间消息 daemon push)。
> - **L5** ad-hoc Bash 按需，无常驻。
> - **关于红线1**(无人值守禁 Monitor/run_in_background)：核心顾虑是"Monitor 启动弹窗阻塞会话"。本次趁 **owner 在线**启动好全部 Monitor(启动弹窗当场放行，进程跨 compact 存活、之后不再弹窗)，owner 走后进入无人值守时各层已在跑、不阻塞——正好规避红线1。owner 当前指令 override 之前的"cron only"做法。**无人值守期间仍不得新起 Monitor**(只复用已跑的)。
> - 历史 cron(4ba96e8f/b30ccc08/c9a332fa/0f9b966f/6233dce7/9460ea91/2230c9e8) 已删。
> **G. 后续**：d3 run-eval 完(journal≈23700+"all stages done")→aggregate-summary(--data-dir .../d3)→analyze(--summary .../d3/per_yaml_summary.jsonl)→拉 pareto 图→发房间 d3 DONE+SR+对比 d1→**画 d1+d3 vs Stage3 重叠图**(扩展 /tmp/plot_d1_overlay.py 加 d3 黑线，对比 Stage3 d3 峰 0.97)→**always-inf baseline**(§7，问 owner force-MISS vs 标准 eval；500ep×3, server 不关, worker)→**收尾 RESULTS + commit**(等 owner 房间发话)。
> **H. 未提交文件(等 owner commit，别 revert)**：`exp/verdict_factor_judge/common/v2_spec.py`(d1+d3 CFG)、`exp/weighted_sum/kinematic/super_warmup.py`+`runner.py`(cfg_id fail-fast 透传 + 双 server guard 移除 + 清 dead import + ruff format)、`tests/exp/test_weighted_sum_libero10.py`+`tests/test_kinematic_super_warmup.py`、`exp/weighted_sum/run_phase2.py`、`src/openpi/conductor/agent.py`、`scripts/serve_policy.py`、`logs/*`。

---

## 目录
- §0 身份 / 权限 / 红线 / 无人值守 mandate
- §1 ★★ 当前正在跑什么（Stage 4 d1 run-warmup）
- §2 ★ Stage 1/2/3 已完成结果
- §3 ★★ Stage 4 完整流程 + 命令（核心，跑 d1+d3）
- §4 ★ owner 决策历史（关键，别重新纠结）
- §5 ★ 走过的坑 + 修复（全量）
- §6 设备拓扑 + server 重启（必带 warmup_dump_root）
- §7 Stage 4 后：always-inference baseline
- §8 收尾 + commit
- §9 监控 + cron
- §10 命令速查 + 红线 checklist

---

## §0 身份 / 权限 / 红线 / 无人值守 mandate

- **Authority = Execution**。只读 `protocols/execution_authority.md`，**绝不读** `protocols/review_authority.md`。
- 项目宪法：`WORKING_AGREEMENT.md` + `CLAUDE.md`（auto-loaded）。plan = `logs/weighted_sum_libero10_replication.log.md`（L2，G1+G2 APPROVED，代码 commit `9519a79`）。
- **无人值守 mandate 生效**（owner 多次授权，最近 2026-05-31）：可自主改任何脚本/跑任何命令；所有沟通走 agentchat **libero10 房间 `019e749f-c5f4-7ce0-9666-4b9a5d8e9af3`**；**owner 任何消息必回**。
- ⚠⚠ **红线1：绝不起 run_in_background Bash / Monitor 后台任务**（owner 2026-05-31 最高级别警告——后台任务弹审批窗口阻塞会话）。监控长跑改用 **L3 cron（CronCreate）+ cron 触发时主动查**。远端 tmux 不算（那是远端进程不是 Claude 后台任务）。也绝不用 AskUserQuestion/EnterPlanMode/任何交互窗口。
- ⚠ **红线2：git commit/push/stash/reset/rebase 等写历史**——必须 owner 房间显式发话才动。commit 规约：英文 message、**无 Co-Authored-By Claude**、author=`LinZiyang666 <3177267975@qq.com>`、owner 偏好一次结构化大 commit。
- ⚠ **红线3：聊天室只发重要事务**（①需 owner 决策 ②ALERT/危险 ③Stage/milestone DONE ④server 起关）。routine 进度/吞吐/诊断细节**不主动发**，只主会话记一行，owner 问才答（owner 2026-05-31 反馈）。
- ⚠ **红线4：Stage 4 进入 + 各 base 推进，G2 已被 owner 依 WA line7 override**（详 §4）——**不需独立 Review session**，走"有条件预决策"。但**这不豁免 git commit 红线**（commit 仍等发话）。
- **语言**：对话/plan/handoff 中文；代码注释/docstring 英文（0 汉字）；commit message 英文。
- 当前 working tree 未提交改动：`src/openpi/conductor/agent.py`(task_suite+MALLOC)、`exp/weighted_sum/run_phase2.py`(task_suite)、`exp/verdict_factor_judge/common/v2_spec.py`(**d1 CFG 填值**)、`tests/exp/test_weighted_sum_libero10.py`(trajectory_depth 断言)、`logs/*`、`scripts/serve_policy.py`。**别 revert，等 owner commit。**

---

## §1 ★★ 当前正在跑什么（→ 实际状态见顶部 2026-06-01 全量更新块）

⚠ **当前实际状态在顶部 2026-06-01 块**：d1 已完整 DONE（SR 峰 0.91），d3 run-eval 跑中（双 server, tmux s4eval, ~26%）。本节以下是 d1 run-warmup 的**历史命令**——d3 warmup/eval 复用相同命令结构（路径换 `d3/`、`--cfg-id` 换 `spatial16_ws_d3_best_libero10`、端口 14003、eval 双 server 14003+14002），具体见顶部 D/G 段 + cron `b30ccc08`。

**Stage 1/2/3 全完。Stage 4 = kinematic 237 cell，跑 d1+d3 两 base 串行。**

### 1.1 d1 run-warmup（正在跑）
- **client = timan107，tmux `s4warmup`**，连 **ziyang10:14000 单 server**（§2.3 决策阶段钉死单 server）。
- 进度：`/tmp/s4_d1_warmup.log` 里 `Wxx: done N/520`。super raw 在 run-warmup 结束时写出（不是实时）。
- 输出 super raw：`exp/weighted_sum/data/kinematic_phase5/libero_10/d1/super_warmup_raw.jsonl`（timan107 repo `/scratch/zixuans8/openpi/...`）。
- **验活：**
```bash
tether exec timan107 -- bash -lc 'tmux ls 2>&1|grep s4warmup; grep -oE "[0-9]+/520" /tmp/s4_d1_warmup.log|tail -1; echo "raw:$(wc -l < /scratch/zixuans8/openpi/exp/weighted_sum/data/kinematic_phase5/libero_10/d1/super_warmup_raw.jsonl 2>/dev/null||echo 0)"; P=run; pgrep -f "${P}er.py\|kinematic.runner"|wc -l'
nc -zv weiland.top 14000 2>&1|tail -1
```
- **★完整 run-warmup 启动命令（死了/重发用；ziyang10 必须先带 --warmup_dump_root 起好，见 §6）：**
```bash
tether exec timan107 -- bash -lc '
  export HOME=/home/zixuans8
  tmux kill-session -t s4warmup 2>/dev/null; rm -f /tmp/s4_d1_warmup.log
  tmux new -s s4warmup -d "cd /scratch/zixuans8/openpi && export HOME=/home/zixuans8 && export PYTHONPATH=/scratch/zixuans8/openpi && export MALLOC_ARENA_MAX=2 && export MALLOC_TRIM_THRESHOLD_=134217728 && /shared/nas/data/m1/zixuans8/miniconda3/bin/uv run python -m exp.weighted_sum.kinematic.runner --mode run-warmup --host weiland.top --port 14000 --warmup-yaml exp/weighted_sum/config/kinematic_phase5/libero_10/d1/ws_d1_kin_super_warmup.yaml --super-raw exp/weighted_sum/data/kinematic_phase5/libero_10/d1/super_warmup_raw.jsonl --trials-per-task 15 --task-suite libero_10 --task-ids 0,1,2,3,4,5,6,7,8,9 --preload-pkl-override exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl --conda-env /scratch/zixuans8/libero_sim --num-workers 40 2>&1 | tee /tmp/s4_d1_warmup.log"
'
```
- run-warmup 完成信号：log 出现 `[run-warmup] super warmup raw at <path>` + tmux 进程退出 + super_warmup_raw.jsonl 有内容。

### 1.2 run-warmup 完成后立即做（d1 后续 mode）
顺序：**verify-raw → emit-eval-yamls → run-eval → aggregate-summary → analyze**（命令 §3.3，flag 现读 runner 确认）。然后切 d3（§3.4）。

---

## §2 ★ Stage 1/2/3 已完成结果

- **Stage 1**（weight search）：winner = `cp1_spatial_pool_16` d1-best **SR 0.520**（grid3_v0@56_v1@25_rs@18，有 robot_state）。all_results.csv = `exp/weighted_sum/data/phase2/libero_10/all_results.csv`。
- **Stage 2**（trajectory，37200 ep）：**结论 trajectory depth 不增益=negative result**。各 depth max SR：d1=.520>d3=.510>d5=.490>d4=.460>d6=.420。journal/results = `exp/weighted_sum/data/stage2/libero_10/`。
- **Stage 3**（threshold-pareto，4 base × 5000ep）：**WINNER=d3**（逐-x 称霸 54% > d1 43% > d4 2% > d5 1%）。peak SR d1=.93/d3=.97/d4=.97/d5=.95；anchor(纯base policy all-MISS)=0.79/0.84/0.86/0.89 ≈ **~0.83 合理**（见 memory `reference_pi05_libero10_baseline`：pi05 libero_10 官方 93%/社区复现 55-83%/我们 0.83 在高端；anchor 7pp 差=pi05 采样随机 73% 一致；t8 both_moka_stove 拖低）。
  - 4 base 数据：`exp/weighted_sum/data/threshold_pareto/libero_10/{d1,d3,d4,d5}/{journal.jsonl,eval_per_step.jsonl,results.json,inf_ratio.json}`。
  - 图：`/tmp/stage3_pareto_4base.png`（脚本 `/tmp/stage3_winner_plot.py`）已发 owner。
  - **4 base 的 winner yaml**（Stage 2 各 depth SR 最优）：d1=`grid3_vision_0@56_vision_1@25_robot_state@18__d1`(有rs)、d3=`grid_vision_0@62_vision_1@37__d3`(无rs)、d4=`grid3_v0@25_v1@43_rs@31`、d5=`grid_v0@50_v1@50`。在 `exp/weighted_sum/config/stage2/libero_10/cp1_spatial_pool_16__<stem>.yaml`。

---

## §3 ★★ Stage 4 完整流程 + 命令（核心，跑 d1+d3）

**owner 定：Stage 4 kinematic 跑 2 个 base 串行 = 先 d1(有robot_state,depth1) 后 d3(无robot_state,depth3)。** 每 base 走 kinematic 6 mode 全流程（~24000 ep/base，单 server）。

### 3.1 CFG_SPECS 现状（`exp/verdict_factor_judge/common/v2_spec.py`）
- ✅ **d1 已填+核验**：`CFG_SPECS["spatial16_ws_d1_best_libero10"]`：
  - keys.weight: vision_0=**0.5625** / vision_1=**0.25** / robot_state=**0.1875**（vision_2/prompt_emb=0）
  - score_normalization μσ: vision_0(mu=0.9739899923664463,sigma=0.0061831533438692935) / vision_1(0.9659078322399228,0.006527797454113087) / robot_state(-1.9584325681212513,0.7484941685797242)，全 zscore+tanh
  - trajectory_depth=**1**，trajectory_weights=_exp_decay_weights(1)=[1.0]；field_similarity 含 robot_state(l2)
  - 核验：程序化 assert CFG==d1 winner yaml 全 OK；27 tests pass（`pytest tests/exp/test_weighted_sum_libero10.py`）。
- ⬜ **d3 待新增**：`CFG_SPECS["spatial16_ws_d3_best_libero10"]`（d1 完后填）。d3 winner = `grid_v0@62_v1@37`（**无 robot_state**！）：
  - keys.weight: vision_0=**0.62** / vision_1=**0.37** / robot_state=**0.0**（其余 0）
  - field_similarity: **只 vision_0/vision_1 cosine（去掉 robot_state 的 l2 block）**
  - score_normalization: **只 vision_0(mu=0.9739899923664463,sigma=0.0061831533438692935) / vision_1(0.9659078322399228,0.006527797454113087)（去掉 robot_state block）**
  - trajectory_depth=**3**，trajectory_weights=**[0.5, 0.3, 0.2]**（d3 winner yaml 实际值，**非** _exp_decay_weights(3)！）
  - 导出 + assert：读 d3 winner yaml（`config/stage2/libero_10/cp1_spatial_pool_16__grid_vision_0@62_vision_1@37__d3.yaml`，结构：keys 在顶层 `y['keys']`，search_strategy 在 `y['checkpoints']['cp1']['search_strategy']`）→ assert CFG==yaml → 跑测试（测试断言可能要再调，d3 无 robot_state）。
  - 新增 CFG 条目用 `--cfg-id spatial16_ws_d3_best_libero10` 切换。kinematic spec/runner 透传 `cfg["search_strategy"]`（spec.py:288/runner.py:312）+ `cfg["keys"]`，所以加新条目即可，--cfg-id CLI 已就绪。

### 3.2 §8.4 pkl 验证（已澄清，pkl OK，别重 build）
- pkl `entries[i].payload.factors` 有 **64 个全 offline** factor（path_length/jerk/direction/dispersion _offline_ × state/action × 8 window），这是**正确的**。
- kinematic `super_warmup_declared_keys()`=50 = 18 offline + 32 online。**18 offline 全在 pkl ✅**。32 online（jerk_online/dispersion_online）= **super warmup RUNTIME 才算**（要 trajectory walk_next 邻居 + 调优），本就不该在 pkl。**pkl 现状正确，不重 build。**（owner 2026-05-31 纠正我之前的"缺32=blocker"误判）。
- pkl 路径：本地 + ziyang10(`/home/ziyang10/openpi/...`) + xuanle(`/home/xuanlel2/openpi/...`) 各一份 `exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl`(1.1GB)。

### 3.3 kinematic 6 mode（`exp/weighted_sum/kinematic/runner.py`，--mode 选）
- `emit-warmup`：写 super_warmup.yaml（本地，no server）。✅ d1 已做：`exp/weighted_sum/config/kinematic_phase5/libero_10/d1/ws_d1_kin_super_warmup.yaml`（12 factor blocks，depth=1）。命令：`uv run python -m exp.weighted_sum.kinematic.runner --mode emit-warmup --warmup-yaml <out> --preload-pkl-override <pkl> --cfg-id spatial16_ws_d1_best_libero10`
- `run-warmup`：server 跑 super warmup，runtime 算 online factor，fetch+extract → super raw。🔄 d1 正跑（§1.1）。
- `verify-raw`：7 检 hard-gate（验 super raw）。命令需读 runner verify-raw mode flag（--super-raw 等）。
- `emit-eval-yamls`：per-cell reconstruct_scores + derive_thresholds + 写 237 yaml。需 super raw + thresholds-dir + eval-dir。
- `run-eval`：dispatch 到 run_phase2，跑 237×100=23700ep（单 server）+ always-WARM 3×100。
- `aggregate-summary` + `analyze`：rebuild summary + decision-gate dump + 4-frontier Pareto overlay。
- ⚠ 每个 mode 的确切 flag 现读 runner.py 确认（line 87+ 是各 mode 实现）。runner CLI flags：--mode --host --port --task-suite --task-ids --cfg-id --preload-pkl-override --warmup-yaml --super-raw/--super-raw-relpath --eval-dir --always-warm-dir --thresholds-dir --summary --data-dir --journal --per-step-dir --servers --conda-env --num-workers --trials-per-task --eval-trials。

### 3.4 §8.6 ★ 路径/suite override 全集（防覆盖 libero_spatial Stage4 产物）
runner 默认烘焙 `kinematic_phase5` + `libero_spatial`。libero_10 跑必须全套 override 到独立目录：
- `--task-suite libero_10 --init-map exp/common/data/db/libero_cache/libero_10_init_map.json --task-ids 0,1,2,3,4,5,6,7,8,9`
- `--cfg-id spatial16_ws_d1_best_libero10`（d3 时换 `_d3_`）`--preload-pkl-override exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl`
- `--data-dir`/`--eval-dir`/`--thresholds-dir`/`--always-warm-dir`/`--journal`/`--per-step-dir`/`--summary`/`--super-raw`/`--super-raw-relpath` 全指 `kinematic_phase5/libero_10/d1/`（d3 换 `/d3/`）——两 base + 两 suite 的产物绝不同路径相撞。
- 预算 §8.5：super warmup ~150ep + 237×100 + always-WARM 3×100 = **~24000 ep/base**。d1+d3 = ~48000 ep，单 server，多日级。

### 3.5 d3 流程（d1 全 6 mode 完后）
1. 新增 CFG `spatial16_ws_d3_best_libero10`（§3.1，无 robot_state，depth3，traj_w=[0.5,0.3,0.2]）+ assert + 测试。
2. push v2_spec.py 到 timan107（若 d3 流程在 timan107 读 CFG）。
3. emit-warmup（--cfg-id _d3_，--warmup-yaml .../libero_10/d3/）→ run-warmup（同 §1.1 改 d3 路径）→ verify-raw → emit-eval → run-eval → aggregate → analyze。

---

## §4 ★ owner 决策历史（关键，别重新纠结）

1. **G2 override（2026-05-31，依 WA line 7）**：owner 是 "Project Owner: Ziyang Lin. Holds absolute authority over this Working Agreement and all project matters. May override any process at will."（WA 第 7 行真实存在，我核对过）。owner 行使此权 override 了"Stage 4 CFG 真值填入需重过独立 G2"的要求。**替代 = "有条件预决策"**：Stage 4 base 从 Stage 3 Pareto winner 确定性导出（weight+depth+μσ 从 winner yaml 机械读+assert 相等），结构性消除手填错值。记录于 plan §8.3 OWNER OVERRIDE 块 + task #6。**不需独立 G2 session。**（我曾据 CLAUDE.md "WA outranks user" 拒绝，但核对 WA 发现第7行授予 owner override 权，纠正了立场——owner override 是 WA 授权行为非违反。）
2. **用 winner 的 depth + 改代码支持**（owner 纠正我"固定depth=1"的误解）：Stage 4 用 winner 的 trajectory_depth。后端 weighted_score_sum_knn 已支持 depth>1（Stage 2 d3/d4/d5 验证），kinematic 透传 CFG search_strategy，所以"支持 winner depth"只是 CFG 加 trajectory_depth 字段（数据非后端改）。已在 CFG 加 trajectory_depth 参数化 + 测试断言改成 verify-exists。
3. **Stage 4 跑 d1+d3 两个 base**（owner 2026-05-31，化解 robot_state 纠结）：d3 winner 恰好无 robot_state（grid_v0@62_v1@37 SR0.510 > 含robot_state最高 grid3_v0@43_v1@37_rs@18 SR0.500，差1pp噪声内）。owner 决定 d1(有rs)+d3(无rs)都跑对比。**先 d1 后 d3。**
4. **§8.4 pkl 纠正**（owner 2026-05-31）：pkl 都是 offline 正确，online runtime 算 + super warmup 调优，不重 build（§3.2）。
5. **always-inference baseline**（owner 2026-05-31，§7）：Stage 4 后跑。
6. **Stage 3 完先发图等指令**（owner 2026-05-31）：已执行（发了 4-base 图 + winner=d3），owner 回"我选depth=3"批准进 Stage 4。

---

## §5 ★ 走过的坑 + 修复（全量，别重犯）

- **warmup_dump_root（Stage4 关键）**：server 启动**必须加 `--warmup_dump_root <path>`**，否则 run-warmup 报 `load_cache_config: server was started without --warmup-dump-root, deferred dumps cannot be resolved`。Stage 2/3 的 server 没带（不需要），Stage 4 super warmup 必须带。ziyang10 dump root = `/tmp/s4_warmup_dumps`（已建 chmod 700）。
- **§8.4 pkl online**：见 §3.2。pkl 只存 offline，online runtime 算，**别误判"pkl 缺 factor=blocker"，别重 build pkl**。
- **HOME 铁律**：所有 timan107 命令必 `export HOME=/home/zixuans8`（否则 libero input() EOFError）；ziyang10/xuanle 必 `export HOME=/home/<user>`。
- **MALLOC**：`MALLOC_ARENA_MAX=2 + MALLOC_TRIM_THRESHOLD_=134217728`（已是 agent.py 默认 + launch export）。
- **per-step 不能 journal-resume**：Stage 3/4 eval 的 per-step（hit_type/cp1_score）在 conductor 退出时一次写，journal-resume 跳过的 episode 不重产 per-step → 中途崩要整批重跑（清 journal+per-step）。所以分 base 一次跑完。
- **keepalive ping timeout = 良性自愈噪音**（深检索单请求 >20s）：worker 断连被 agent 重启 + episode requeue + journal-resume 零损失。证据：conductor etime 无异常重启 + worker 满额 + journal 涨 + server FULL_HIT 无 OOM。别误判 fatal。
- **集群不稳**：jupyter-ziyang10/xuanlel2 ~1h 掉一次（node STALE→OFFLINE，exec 不进 + 隧道死）。节点 OFFLINE → urgent 喊 owner 重启 pod；节点 ONLINE serve 没了 → 自主重启（§6，**带 warmup_dump_root**）。
- **expose 端口会变**（broker 重分配）：re-expose 后看实际端口（ziyang10 这次仍 14000，xuanle=14002），改 health 探针 + --port/--host。
- **pkill 自杀**：变量法 `P=serve; pkill -9 -f "${P}_policy"`，echo 不含字面，kill/launch 拆 exec。
- **xuanle 无系统 tmux/fuser**：用 `/home/xuanlel2/miniforge3/bin/tmux` + `pkill -9 -f "[s]erve_policy"` + `ss`。NFS uid 坑 dump root 用 /tmp。
- **tether allow_roots 不含 /scratch**：push 走 /tmp 中转再 cp。
- **96 worker RAM 累积**（Stage 2/3 client）：MALLOC 后稳态但深检索期累积，曾从 96 缩 80(40,40)。Stage 4 run-warmup 用 40 worker（小）。
- **d6 调度排尾**（Stage 2 已消解）：conductor 按 yaml 字典序，spatial d6 排巨量 spatial wsweep 后，落末尾跑。

---

## §6 设备拓扑 + server 重启（必带 warmup_dump_root）

| 角色 | 节点 | 关键 |
|---|---|---|
| server(Stage4 单) | jupyter-ziyang10 | H200；HOME=/home/ziyang10；repo=/home/ziyang10/openpi；uv=/home/ziyang10/.local/bin/uv；ckpt=~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch；tmux=srv0；**3 replica + --warmup_dump_root /tmp/s4_warmup_dumps**；NAT→**weiland.top:14000**；git 9519a79(旧版无 d1 CFG，但 server 用 yaml 不读 CFG_SPECS，OK) |
| server 备 | jupyter-xuanlel2 | 另块 H200；HOME=/home/xuanlel2；无系统tmux(用 miniforge3/bin/tmux)；NAT→**weiland.top:14002**；当前 srv0 在跑但**无 warmup_dump_root**(Stage4 用要重启加) |
| client | timan107 | Xeon 48核/220G/8×GTX1080；repo=/scratch/zixuans8/openpi；uv=/shared/nas/data/m1/zixuans8/miniconda3/bin/uv；conda=/scratch/zixuans8/libero_sim；**HOME 必 export /home/zixuans8**；allow_roots 无 /scratch |
| broker | pc732(weiland.top→155.98.36.32) | 公网端口段 14000-14999 |
| a100 | OFFLINE | failover 需 owner |

**★ ziyang10 server 重启命令（带 warmup_dump_root，崩了/重启 pod 后用）：**
```bash
# 1. 清残留 + 建 dump root
tether exec jupyter-ziyang10 -- bash -lc 'export HOME=/home/ziyang10; tmux kill-session -t srv0 2>/dev/null; P=serve; pkill -9 -f "${P}_policy" 2>/dev/null; sleep 3; rm -rf /tmp/s4_warmup_dumps; mkdir -p /tmp/s4_warmup_dumps; chmod 700 /tmp/s4_warmup_dumps; ss -tlnp|grep :8000||echo free'
# 2. 起 server(3rep + warmup_dump_root)
tether exec jupyter-ziyang10 -- bash -lc '
  export HOME=/home/ziyang10; cd /home/ziyang10/openpi
  tmux new -s srv0 -d "cd /home/ziyang10/openpi && export HOME=/home/ziyang10 && export OPENPI_SERVER_GPU_MEMORY_LOCK=0 && /home/ziyang10/.local/bin/uv run scripts/serve_policy.py --replicas 3 --replica-spawn-batch 2 --port 8000 --warmup_dump_root /tmp/s4_warmup_dumps --cache_config /home/ziyang10/stage1_placeholder.yaml policy:checkpoint --policy.config=pi05_libero --policy.dir=/home/ziyang10/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch 2>&1 | tee /tmp/srv0.log"
'
# 3. 等 ready(~3-4min, 3rep 加载 pi05 + 1.1GB pkl), ready 签名 "replica_proxy listening on 0.0.0.0:8000 -> [8001,8002,8003]"
tether exec jupyter-ziyang10 -- bash -lc 'export HOME=/home/ziyang10; for i in $(seq 1 20); do sleep 10; grep -q "replica_proxy listening" /tmp/srv0.log && { echo READY; break; }; done; tail -3 /tmp/srv0.log'
# 4. re-expose(broker 可能给新端口, 看输出)
tether expose jupyter-ziyang10 --local 8000 --name ws-s4-zy2   # 看实际端口, 期望 14000
nc -zv weiland.top 14000 2>&1|tail -1
# placeholder 缺失则: tether push <任一 stage2 spatial_16 d1 yaml> jupyter-ziyang10:/home/ziyang10/stage1_placeholder.yaml
```

---

## §7 Stage 4 后：always-inference baseline（owner 2026-05-31 指令，记住）

Stage 4(d1+d3)全完后跑：**always-inference baseline = 500ep(50 trial/subtask × 10 task) × 同时跑 3 次**（量 pi05 采样方差，因 pi05 随机），**server 不关、worker 仍 80(40,40)**。目的=1500ep 可靠纯-inference ceiling 对照 cache 实验，确证 anchor ~0.83。实现 (a)force-MISS held-out 同协议 vs (b)标准 libero eval 标准 init，**到时问 owner**。参考 pi05 libero_10 baseline 官方93%/复现55-83%/我们 0.83（memory `reference_pi05_libero10_baseline`）。

---

## §8 收尾 + commit（task #7）

Stage 4 + always-inf baseline 后：汇总 4 Stage 分析 + RESULTS.md + commit（**等 owner 房间显式发话**，英文/无 Co-Authored/author=LinZiyang666/单大 commit）。未提交改动见 §0。

---

## §9 监控 + cron

- **cron 当前：暂停/待重建**（Stage 3 完时删了 728ed282）。Stage 4 d1 run-warmup 跑起来后**需重建 Stage 4 cron**（监控 s4warmup 进度 + run-warmup 完成→推进 verify-raw/emit-eval/run-eval；server 14000 存活；不起 background）。cron schedule 用 `8,23,38,53 * * * *`(15min 错峰)，CronCreate。
- **旧 Monitor**（owner 警告前起的，已在跑可不动）：L4 agentchat watcher(`pgrep -fc 'agentchat watch state'`)、L5 server-log。若死**不重挂**（不起 background），改 cron 手动 tail/read 替代。
- **L4 agentchat watcher 是 owner 消息推送源**——若死，每次 cron 主动 `agentchat read 019e749f-c5f4-7ce0-9666-4b9a5d8e9af3 --json` 检查。

---

## §10 命令速查 + 红线 checklist

**全栈验活：**
```bash
tether node ls -a | grep -E "ziyang10|xuanlel2|timan107"
nc -zv weiland.top 14000 2>&1|tail -1     # ziyang10 Stage4 server
tether exec timan107 -- bash -lc 'tmux ls 2>&1|grep s4warmup; grep -oE "[0-9]+/520" /tmp/s4_d1_warmup.log|tail -1'
agentchat read 019e749f-c5f4-7ce0-9666-4b9a5d8e9af3 --json | head -20
pgrep -fc 'agentchat watch state'   # L4
```

**恢复 checklist（compact 后）：**
1. 读本文件全文 + plan。
2. `git -C /home/weiland/projects/openpi log -1 --oneline`(9519a79) + `git status`(看 §0 未提交，**别 revert/commit**)。
3. `tether node ls -a`：ziyang10/xuanlel2/timan107 ONLINE？a100 OFFLINE。掉了走 §6。
4. 验 d1 run-warmup（§1.1）：在跑→等完成；完成(super raw 写出)→§1.2/§3.3 推进 verify-raw→emit-eval→run-eval。
5. server(14000)存活？崩了→§6 重启(**带 warmup_dump_root**)。
6. `agentchat read` 看 owner 有无新指令(必回)。

**红线（最重要）：**
1. **绝不起 run_in_background/Monitor 后台任务**（owner 最高警告）；监控用 cron+手动查。
2. **绝不 git commit/push/stash/reset**——等 owner 房间发话。
3. **server 重启必带 `--warmup_dump_root`**（Stage4 super warmup 必须）。
4. **聊天室只发重要事务**（决策/危险/DONE/server起关），owner 消息必回。
5. **不弹任何交互/权限窗口**（AskUserQuestion/EnterPlanMode）。
6. **Stage 4 G2 已 owner override**（WA line7，§4），不需独立 G2，但 commit 红线不豁免。
7. **Stage 4 单 server**（§2.3 决策阶段，跨 GPU ~7pp 会翻转 cell 排序）。
8. **per-step 不 resume**→eval 中途崩整批重跑。
9. **d3 winner 无 robot_state**（§3.1）——填 CFG 要去掉 robot_state 的 field_sim+score_norm，trajectory_weights=[0.5,0.3,0.2] 非 exp_decay。
10. **HOME=/home/zixuans8(timan107) / MALLOC / 单server / warmup_dump_root** 是铁律。

---

## §11 关键路径/值速查

- plan: `logs/weighted_sum_libero10_replication.log.md`
- CFG: `exp/verdict_factor_judge/common/v2_spec.py` → `CFG_SPECS["spatial16_ws_d1_best_libero10"]`(已填d1) / 待加 `_d3_`
- kinematic runner: `exp/weighted_sum/kinematic/runner.py`(6 mode) / spec: `kinematic/spec.py`(super_warmup_declared_keys, generate_all_cells) / super_warmup.py
- d1 super warmup yaml: `exp/weighted_sum/config/kinematic_phase5/libero_10/d1/ws_d1_kin_super_warmup.yaml`
- d1 super raw(输出): `exp/weighted_sum/data/kinematic_phase5/libero_10/d1/super_warmup_raw.jsonl`
- d1/d3 winner yaml: `exp/weighted_sum/config/stage2/libero_10/cp1_spatial_pool_16__grid3_vision_0@56_vision_1@25_robot_state@18__d1.yaml` / `...__grid_vision_0@62_vision_1@37__d3.yaml`
- init_map: `exp/common/data/db/libero_cache/libero_10_init_map.json`(server端 timan107) / 本地同
- pkl: `exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl`(64 offline factor, 正确) + `.pre_phase5.bak.pkl`(1.1GB备份)
- Stage3 数据: `exp/weighted_sum/data/threshold_pareto/libero_10/{d1,d3,d4,d5}/`
- 房间: `019e749f-c5f4-7ce0-9666-4b9a5d8e9af3`(libero10)
- d1 CFG 值: weight v0=0.5625/v1=0.25/rs=0.1875; μσ v0(0.9739899923664463,0.0061831533438692935)/v1(0.9659078322399228,0.006527797454113087)/rs(-1.9584325681212513,0.7484941685797242); depth=1
- d3 CFG 值(待填): weight v0=0.62/v1=0.37/rs=0(去rs); μσ v0/v1 同上(去rs); depth=3; traj_w=[0.5,0.3,0.2]

---

---

## §12 ★ Stage 4 各 mode 详细命令模板（verbatim，compact 后照抄改路径）

> 全部在 timan107 跑（client），连 ziyang10:14000 单 server。所有命令前缀：
> `tether exec timan107 -- bash -lc 'export HOME=/home/zixuans8; cd /scratch/zixuans8/openpi && export PYTHONPATH=/scratch/zixuans8/openpi && export MALLOC_ARENA_MAX=2 && export MALLOC_TRIM_THRESHOLD_=134217728 && UV=/shared/nas/data/m1/zixuans8/miniconda3/bin/uv; <CMD>'`
> 其中 `<CMD>` = `$UV run python -m exp.weighted_sum.kinematic.runner ...`。长跑(run-eval)用 tmux。

### 12.1 emit-warmup（本地或 timan107，no server）— d1 已做
```bash
$UV run python -m exp.weighted_sum.kinematic.runner \
  --mode emit-warmup \
  --warmup-yaml exp/weighted_sum/config/kinematic_phase5/libero_10/d1/ws_d1_kin_super_warmup.yaml \
  --preload-pkl-override exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl \
  --cfg-id spatial16_ws_d1_best_libero10
# 输出: "[emit-warmup] wrote super warmup yaml -> ... (12 dump factor blocks)"
```

### 12.2 run-warmup（server，~150ep/520单位）— d1 正跑
见 §1.1 完整命令。要点：--mode run-warmup --host weiland.top --port 14000 --warmup-yaml <d1 yaml> --super-raw <d1 raw> --trials-per-task 15 --task-suite libero_10 --task-ids 0,1..9 --preload-pkl-override <pkl> --conda-env /scratch/zixuans8/libero_sim --num-workers 40。完成信号 log `[run-warmup] super warmup raw at <path>`。

### 12.3 verify-raw（7 检 hard-gate，no server）
```bash
# 先读 runner.py line ~172+ 的 _mode_verify_raw 确认 flag, 大概:
$UV run python -m exp.weighted_sum.kinematic.runner \
  --mode verify-raw \
  --super-raw exp/weighted_sum/data/kinematic_phase5/libero_10/d1/super_warmup_raw.jsonl \
  --cfg-id spatial16_ws_d1_best_libero10
# 7 检全过才继续; 不过则看哪检失败(declared keys 覆盖/finite/分布等)
```

### 12.4 emit-eval-yamls（237 cell，no server）
```bash
$UV run python -m exp.weighted_sum.kinematic.runner \
  --mode emit-eval-yamls \
  --super-raw exp/weighted_sum/data/kinematic_phase5/libero_10/d1/super_warmup_raw.jsonl \
  --eval-dir exp/weighted_sum/config/kinematic_phase5/libero_10/d1/eval \
  --thresholds-dir exp/weighted_sum/data/kinematic_phase5/libero_10/d1/thresholds \
  --always-warm-dir exp/weighted_sum/config/kinematic_phase5/libero_10/d1/always_warm \
  --cfg-id spatial16_ws_d1_best_libero10 \
  --preload-pkl-override exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl
# 产 237 eval yaml(per-cell threshold 解算) + always-warm yaml
```

### 12.5 run-eval（237×100=23700ep + always-WARM 3×100，单 server，tmux）
```bash
tether exec timan107 -- bash -lc '
  export HOME=/home/zixuans8
  tmux new -s s4eval -d "cd /scratch/zixuans8/openpi && export HOME=/home/zixuans8 && export PYTHONPATH=/scratch/zixuans8/openpi && export MALLOC_ARENA_MAX=2 && export MALLOC_TRIM_THRESHOLD_=134217728 && /shared/nas/data/m1/zixuans8/miniconda3/bin/uv run python -m exp.weighted_sum.kinematic.runner --mode run-eval --servers weiland.top:14000 --eval-dir exp/weighted_sum/config/kinematic_phase5/libero_10/d1/eval --always-warm-dir exp/weighted_sum/config/kinematic_phase5/libero_10/d1/always_warm --journal exp/weighted_sum/data/kinematic_phase5/libero_10/d1/journal.jsonl --per-step-dir exp/weighted_sum/data/kinematic_phase5/libero_10/d1/per_step --task-suite libero_10 --task-ids 0,1,2,3,4,5,6,7,8,9 --eval-trials 10 --conda-env /scratch/zixuans8/libero_sim --num-workers 40 --cfg-id spatial16_ws_d1_best_libero10 --preload-pkl-override exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl 2>&1 | tee /tmp/s4_d1_eval.log"
'
# ⚠ run-eval flag 须读 runner.py _mode_run_eval 确认(--servers 单 endpoint! 双 endpoint fail-fast)
```

### 12.6 aggregate-summary + analyze（no server）
```bash
$UV run python -m exp.weighted_sum.kinematic.runner --mode aggregate-summary --journal <d1 journal> --per-step-dir <d1 per_step> --summary exp/weighted_sum/data/kinematic_phase5/libero_10/d1/per_yaml_summary.jsonl ...
$UV run python -m exp.weighted_sum.kinematic.runner --mode analyze --summary <...> ...   # decision-gate + 4-frontier Pareto overlay
```

---

## §13 ★ d3 CFG 待填完整 dict（d1 全完后，加进 v2_spec.py CFG_SPECS）

```python
    "spatial16_ws_d3_best_libero10": {
        "key_builder_type": "cp1_spatial_pool_16",
        "vector_dims": {"vision_0": 32768, "vision_1": 32768, "prompt_emb": 2048, "robot_state": 32},
        "keys": {
            # Stage 3 winner d3 (grid_v0@62_v1@37) — NO robot_state (weight 0).
            "vision_0":   {"enabled": True,  "weight": 0.62},
            "vision_1":   {"enabled": True,  "weight": 0.37},
            "vision_2":   {"enabled": False, "weight": 0.0},
            "prompt_emb": {"enabled": False, "weight": 0.0},
            "robot_state": {"enabled": True, "weight": 0.0},
        },
        "preload_pkl": "exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl",
        "search_strategy": {
            "type": "weighted_score_sum_knn",
            "top_k": 1,
            "step_filter": "all",
            "trajectory_depth": 3,
            "trajectory_weights": [0.5, 0.3, 0.2],   # d3 winner yaml 实际值, 非 _exp_decay_weights(3)
            "field_similarity": {                      # 只 vision_0/vision_1, 去 robot_state
                "vision_0": {"type": "cosine"},
                "vision_1": {"type": "cosine"},
            },
            "score_normalization": {
                "type": "per_field",
                "fields": {                            # 只 vision_0/vision_1, 去 robot_state
                    "vision_0": {"method": "zscore", "params": {"mu": 0.9739899923664463, "sigma": 0.0061831533438692935, "squash": "tanh"}},
                    "vision_1": {"method": "zscore", "params": {"mu": 0.9659078322399228, "sigma": 0.006527797454113087, "squash": "tanh"}},
                },
            },
        },
    },
```
> 加完 assert: 读 d3 yaml(`config/stage2/libero_10/cp1_spatial_pool_16__grid_vision_0@62_vision_1@37__d3.yaml`, keys 顶层/search_strategy 在 checkpoints.cp1) → CFG==yaml → 跑 pytest(d3 无 robot_state, test_cfg_specs_libero10_entry_structure 断言 robot_state 的可能要放宽或该测试只测 d1 条目)。

---

## §14 ★ Stage 1/2/3 详细数据（写 RESULTS 用）

### 14.1 Stage 2 各 depth 全局 SR（37200ep, 372 yaml）
- max SR: d1=0.520 / d3=0.510 / d5=0.490 / d4=0.460 / d6=0.420（随 depth 递减，trajectory 不增益）
- mean SR: d1=0.387 / d4=0.393 / d3=0.378 / d5=0.370 / d6=0.343
- yaml 分布: d1=78(仅2b wsweep) / d3=d4=d5=93(2a 15+2b 78) / d6=15(仅2a)。total 372。

### 14.2 Stage 3 4 base Pareto（各 5000ep=49cell+anchor）
| base | winner yaml(weight) | 逐-x称霸 | Pareto峰SR@ratio | anchor(纯base) | 前沿点 |
|---|---|---|---|---|---|
| d1 | grid3 v0=.5625 v1=.25 rs=.1875 (有rs,depth1) | 43% | 0.93@0.83 | 0.79 | 9 |
| **d3 WINNER** | grid v0=.62 v1=.37 (无rs,depth3,tw=[.5,.3,.2]) | **54%** | 0.97@0.89 | 0.84 | 13 |
| d4 | grid3 v0=.25 v1=.43 rs=.31 (depth4) | 2% | 0.97@0.89 | 0.86 | 12 |
| d5 | grid v0=.50 v1=.50 (depth5) | 1% | 0.95@0.91 | 0.89 | 13 |
- inference_ratio 定义(summarize_inf_ratio.py): FULL_HIT→0 / WARM_START@t→1-0.5(1-t) / MISS→1。winner 选取: 逐-x「ratio<=x 预算下最优 SR」称霸最长(脚本 /tmp/stage3_winner_plot.py)。
- 核心发现: 少量高置信 cache 辅助 SR(0.93-0.97) > 纯 base policy(0.79-0.89) > 过度用 cache(0.50-0.60)。Stage2 always_hit 全程替代 SR 仅 0.52(全程 cache 伤 SR)。

### 14.3 anchor 数据质疑(owner)的结论
- 三 anchor 全 MISS(纯 base policy)实锤; SR 0.79/0.84/0.86 差异=pi05 随机采样(同 init 三次 73% 一致, 27% 翻转)。
- ~0.83 合理: pi05 libero_10 官方 93%/社区复现 55-83%/我们 0.83 在高端。t8(both_moka_stove)仅 3/10 拖低(最难 long-horizon task)。
- libero_10 task: 0 tomato_sauce/1 cream_cheese+butter/2 turn_on_stove_moka/3 black_bowl_drawer/4 white_mug/5 book_caddy/6 white_mug+chocolate/7 alphabet_soup+cream_cheese/8 both_moka_stove/9 yellow_white_mug_microwave。

---

## §15 always-inference baseline 命令草案（Stage 4 后, §7）
```bash
# (a) force-MISS 同协议(held-out init, 跟 cache 实验可比) — 3 次:
for run in 1 2 3; do
  # emit force-MISS anchor yaml(任一 base 的 threshold@2.0 anchor) 或复用 stage3 anchor
  # run_phase2 --eval-trials 50 --task-ids 0-9 (=500ep) --journal .../baseline/run$run/journal.jsonl --servers weiland.top:14000 --task-suite libero_10 --workers 80(40,40) --conda-env ...
done
# (b) 标准 libero eval: examples/libero/main.py 纯 inference 标准 init
# 到时问 owner 选 (a)/(b); server 不关 worker 80。
```

---

## §16 ★ 历次关键命令/事件历史（verbatim 参考）
- Stage 2 conductor(已完): run_phase2 --yaml-dir /tmp/stage2_libero10_yamls --servers weiland.top:14000,14002 --server-workers 40,40(曾96缩80) --eval-concurrency 2 --task-suite libero_10。
- Stage 3 分 base eval(已完): run_phase2 --yaml-dir /tmp/by_base/<base> --per-step-out .../eval_per_step.jsonl --eval-trials 10 --server-workers 40,40。per-step 不 resume→每 base 一次跑完。pull→summarize.py+summarize_inf_ratio.py→画 Pareto。
- Stage 3 warmup(已完): emit_threshold_yamls --mode warmup(force-MISS)→run_phase2 --per-step-out→按 yaml_id 拆 per_base 分布→emit eval(11×5-6=49cell 满格)+anchor。
- d6 调度排尾(已消解): grep journal 各 depth 完成数确认。
- RAM 处置史: 96w 累积冲 197G→滚动重启泄压(kill 197G→14G 全归还)+缩80(40,40)→稳态 106-138G。
- keepalive timeout 1409 次(Stage2): 良性自愈(d5 检索 >20s ping timeout, worker 重启 requeue, 0 损失)。

---

## §17 ★ 测试 / 验证命令
```bash
# CFG 改动后跑 libero10 测试:
cd /home/weiland/projects/openpi && uv run pytest tests/exp/test_weighted_sum_libero10.py -q   # 应 27 passed
# d1 CFG assert(已过): 读 d1 yaml(keys 顶层/ss 在 checkpoints.cp1) vs CFG_SPECS, assert weight/μσ/depth 相等
# §8.4 pkl: super_warmup_declared_keys() offline 子集 ⊆ pkl entries[0].payload.factors(64 offline) → YES
# pkl 结构: dict{key_builder_type,checkpoint_id,vector_dims,entries(CacheEntry list),library_stats}; CacheEntry.payload(CachePayload).factors
```

---

## §18 ★ kinematic factor 列表（权威, super_warmup_declared_keys=50）
- declared 50 = 18 offline + 32 online。
- offline descs(pkl有64全集): jerk/direction/dispersion/path_length × state/action × 8 window{(0,3)(0,5)(1,1)(2,2)(3,0)(3,3)(5,5)(7,7)}=64。
- online descs(runtime算): jerk/dispersion(只这2个有online) × state/action × 8 window=32。
- 源: verdict_factor_judge/phase5/spec.py OFFLINE_DESCS=("jerk","direction","dispersion","path_length") ONLINE_DESCS=("jerk","dispersion")。online 算引擎: src/openpi/cache/components/factors/online.py(需 history.actions/states + walk_next(F))。

---

## §19 ★ Stage 4 cron 重建模板（run-warmup 跑起来后, 不起 background）
CronCreate, schedule `8,23,38,53 * * * *`, prompt 要点:
```
[Stage4 cron 15min] 巡检: tether exec timan107 'tmux ls|grep s4warmup/s4eval; grep N/520 或 journal 进度; nc 14000'; free -g。
处理: 全健康记一行不发房间。run-warmup 完成(super raw 写出)→读 runner 跑 verify-raw→emit-eval-yamls→run-eval(tmux s4eval, 单server 14000, 23700ep)→aggregate→analyze→发房间报 d1 DONE。d1 全完→切 d3(加 CFG _d3_ §13+assert+test→emit-warmup→run-warmup→...→237cell)。
server(14000)崩→§6 重启(带 --warmup_dump_root!)。绝不起 run_in_background。commit 等 owner。Stage4 完→always-inf baseline(§7,问owner)。
```

---

## §20 ★ owner 决策时间线（避免重新纠结）
1. unattended mandate(完全权限, 不起 background[最高警告], 沟通走聊天室, 聊天室只发重要事务)。
2. Stage 3 改 per-depth 4 base(d1/3/4/5)。
3. WA line7 override G2(有条件预决策)。我曾拒绝→核对 WA 第7行→纠正接受。
4. 用 winner depth(非固定 depth=1)。
5. §8.4 pkl: offline 正确 online runtime 算, 不重 build。
6. Stage3 完先发图等指令→发了→owner "我选depth=3"。
7. Stage4 跑 d1+d3 两个 base(化解 robot_state), 先 d1。
8. always-inf baseline(Stage4后 500ep×3 server不关 worker80)。
9. ziyang10 掉线→owner 重启 pod→我重启 server(带 warmup_dump_root)。

---

---

## §21 ★ git 未提交改动详情（等 owner commit，别 revert）
```
M  exp/weighted_sum/run_phase2.py          # task_suite_name 透传(WorkerSpec)
MM logs/session_handoff.md                  # 本文件
M  logs/weighted_sum_libero10_replication.log.md  # §14 执行日志 + §8.3 OWNER OVERRIDE 块
M  scripts/serve_policy.py                  # (早期改动)
MM src/openpi/conductor/agent.py            # task_suite_name 字段 + MALLOC_ARENA_MAX/TRIM 默认
M  exp/verdict_factor_judge/common/v2_spec.py  # d1 CFG 填值(spatial16_ws_d1_best_libero10)
M  tests/exp/test_weighted_sum_libero10.py  # trajectory_depth 断言 not-in→verify-exists
?? tests/scripts/test_collect_isolation.py
```
- agent.py 改动: WorkerSpec 加 `task_suite_name: str = "libero_spatial"` + _default_spawn base_cmd 加 `"--task-suite-name", spec.task_suite_name` + env.setdefault MALLOC_ARENA_MAX=2/MALLOC_TRIM_THRESHOLD_=134217728。run_phase2: WorkerSpec(...task_suite_name=args.task_suite)。已 push timan107 应用。
- v2_spec.py: CFG_SPECS[spatial16_ws_d1_best_libero10] keys.weight + score_norm μσ + trajectory_depth=1(d1)。已 push timan107。
- commit 时机: Stage4+baseline 全完, owner 房间发话, 英文/无 Co-Authored/author=LinZiyang666/单大 commit。

## §22 ★ super warmup 机制(理解 run-warmup 在干嘛)
- emit-warmup 写的 yaml: judge.dump.factors 含 237 cell declared 的 factor block(12 个 type×window 组), super warmup 跑时 server 对每步算这些 factor(offline 从 pkl, online runtime 算)并 dump。
- run-warmup: client 跑 super warmup eval(force 全程收集), server fetch_dump 把 factor 值 dump 到 warmup_dump_root(故 server 必须 --warmup_dump_root!), client extract finite → super_warmup_raw.jsonl(每步一行, 含所有 factor 值)。
- super raw 用途: emit-eval-yamls 时 reconstruct_scores + derive_thresholds(per cell 算 factor 的 percentile 阈值), 生成 237 个 eval yaml(每 cell 一组 fh/ws 阈值)。
- SUPER_WARMUP_ID="ws_d1_kin_super_warmup"(237 cell 共享一个 super warmup, spec.py:56)。

## §23 ★ 关键文件行号速查
- v2_spec.py: CFG_SPECS dict line 70+; spatial16_ws_d1_best_libero10 条目 line 210+(已填); _exp_decay_weights line 35。
- kinematic/runner.py: VALID_MODES line 62-69; _mode_emit_warmup line 91; _mode_run_warmup line 160; verify-raw line 172+; 各 mode flag add_argument line 746-788。
- kinematic/spec.py: super_warmup_declared_keys line 195; generate_all_cells; SUPER_WARMUP_ID line 56; search_strategy 透传 line 288。
- kinematic/super_warmup.py: build_super_warmup_yaml; top_k=5(line 147); load_per_key_finite_history。
- summarize_inf_ratio.py: FH→0/WS@t→1-0.5(1-t)/MISS→1。
- emit_threshold_yamls.py: warmup/eval/anchor mode; 11×5 网格-6 退化=49 cell。

## §24 ★ memory 文件(持久, 已存)
- user_language / feedback_no_coauthor / feedback_commit_message_english / feedback_review_cycle_protocol / reference_remote_server / feedback_no_unsolicited_git_add / reference_cache_baselines / project_phase5_libero10_stage2_failover / feedback_phase5_yaml_default / reference_libero_concurrency / reference_device_topology / project_scaleout_serving / feedback_single_commit_preference / **feedback_chatroom_important_only**(聊天室只发重要) / **feedback_no_background_tasks_unattended**(禁 background) / **reference_pi05_libero10_baseline**(pi05 libero_10 官方93%/复现55-83%/我们0.83) / feedback_plan_verify_src_api。

---

## §25 ★ compact 后 30 秒 TL;DR
1. 我是 Execution agent, 无人值守跑 weighted_sum libero_10 复刻, 在 Stage 4(kinematic 237cell)。
2. Stage 1/2/3 全完。Stage3 winner=**d3**。owner 定 Stage4 跑 **d1+d3 两个 base 串行, 先 d1**。
3. 当前: **d1 run-warmup 在跑**(timan107 tmux s4warmup → ziyang10:14000 单 server)。d1 CFG 已填+核验(27test pass)。emit-warmup 已做。
4. d1 run-warmup 完(super raw 写出)→ verify-raw → emit-eval-yamls(237) → run-eval(23700ep tmux s4eval) → aggregate → analyze → 发房间报 d1 DONE。
5. d1 全完 → 切 d3: 加 CFG_SPECS[spatial16_ws_d3_best_libero10](§13, 无robot_state/depth3/tw=[.5,.3,.2])+assert+test → 同样 6 mode。
6. d1+d3 全完 → always-inf baseline(§7, 问owner) → 收尾 RESULTS + commit(等owner发话)。
7. ⚠ 红线: 不起 background/不擅自commit/server重启带warmup_dump_root/单server/聊天室只发重要事务/owner消息必回/HOME+MALLOC。
8. 沟通: agentchat 房间 019e749f-c5f4-7ce0-9666-4b9a5d8e9af3。监控: 重建 Stage4 cron(§19)。
9. 集群不稳~1h掉一次: 节点OFFLINE喊owner重pod, ONLINE serve没了自主重启(§6带warmup_dump_root)。
10. owner 是 WA line7 绝对权威, 已override Stage4 G2(有条件预决策, 不需独立review)。

---

## §26 ★ 补充：易错点 + 单 server 理由
- **单 server 理由(§2.3)**：237 cell 的 5pp 决策门 + Pareto dominance 是阶段内比较；offline-calib 只吸收检索分漂移不吸收 SR 漂移，跨 GPU ~7pp 会翻转 cell 排序。整批 237 cell 跑同一台(ziyang10:14000)。run-eval 的 --servers 必须单 endpoint(双 endpoint runner fail-fast, G2 R1 修过)。
- **kinematic factor 仪器是从 phase5 移植的固定仪器**(未在 libero_10 重调)，Option B 是"检索层忠实复刻"(weight+μσ 用 libero_10 的, factor recipe 用 phase5 的)。RESULTS 不 over-claim 跨 suite, 只声称 libero_10 内部可比。
- **xuanle 也要 warmup_dump_root**: 若 Stage4 改用 xuanle, 同样重启加 --warmup_dump_root。当前 xuanle srv0 在跑但无此参数(Stage2/3 遗留), 不能直接给 Stage4 run-warmup 用。
- **emit-eval-yamls 的 always-warm**: 237 cell 之外还有 always-WARM yaml(3×100ep), echo 里硬编码 libero_spatial_init_map → 须 override(§8.6)。
- **trials-per-task 15 → super warmup ~150ep**(但 log 进度条显示 N/520, 520 可能是内部 step/episode 单位, 不影响)。
- **CFG_ID_DEFAULT 绝不改**(破坏 libero_spatial Stage4 复现)；Option B 走新 cfg-id。

---

> 本文件全量恢复手册(§0-§26, 500+行)。compact 后：读全文 → §25 TL;DR → §10 checklist → 验 d1 run-warmup(§1.1) → 推进 Stage4 mode(§12) → d1 完切 d3(§13/§3.5) → Stage4 完 always-inf baseline(§7) → 收尾 commit(等 owner)。**铁律：① 绝不起 run_in_background/Monitor 后台任务(owner 最高警告) ② 绝不擅自 git commit/push ③ server 重启必带 --warmup_dump_root ④ Stage4 单 server ⑤ 聊天室只发重要事务 ⑥ owner 消息必回 ⑦ d3 winner 无 robot_state(填CFG去rs) ⑧ HOME=/home/zixuans8 + MALLOC ⑨ per-step 不resume中途崩整批重跑 ⑩ G2 已owner override(WA line7)无需独立review但commit红线不豁免。** 当前：d1 run-warmup 在跑(ziyang10:14000, tmux s4warmup, N/520, super raw 结束时写)。下一步 run-warmup 完→verify-raw→emit-eval→run-eval。
