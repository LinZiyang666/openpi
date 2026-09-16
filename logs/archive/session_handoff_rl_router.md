# Session Handoff — rl_router 线（X14 已收官 / X15 已 G1 APPROVED 待开工）

> 完全重写于 **2026-08-22 ~17:00 UTC**（compact 前）。本文=接手导航+资产清单；**X15 的唯一设计权威 = [`rl_router_v2_risk_router_plan.log.md`](rl_router_v2_risk_router_plan.log.md)（G1 APPROVED，commit `79f0e20` 已 push）**；X14 运行实录与全部 37 类实测缺陷 = [`rl_router_operations.log.md`](rl_router_operations.log.md)。冲突以 plan 为准。
> ⚠ 不要把本线状态写回共享的 `session_handoff.md`（被 gate_threshold_pareto 线占用）。

---

## 0. 接手第一步

1. 读 `WORKING_AGREEMENT.md` → 声明 **Execution Authority** → 只读 `protocols/execution_authority.md`（**绝不读 review_authority.md，绝不碰 tests/review_tests/**）。
2. 出状态卡，等 owner 确认再动：

```
WORKFLOW STATUS | Authority: Execution | Task: X15 代理监督风险门控 router | Level: L3
X14 ✅收官(b0288 关停,资产在位) → X15 Plan ✅ → G1 ✅APPROVED(R4)+polish+push → Code ⬚(U0→U6) → 逐单元 G2 ⬚ → Verify ⬚
```

3. 通读 X15 plan 全文（172 行，含 §11 L3 触点矩阵与全部冻结契约）。**四轮评审 21 条意见的历史在 git（79f0e20 之前的暂存快照）**，Review Log 已按 §3.1 polish 删除。

## 1. 现状一句话

**没有任何东西在跑。** X14 于 b0288 被 owner 令完全关停（2026-08-22 05:1x UTC，§3.37：全机零本线进程、expose 已归池、监控全撤）；X15 plan 已 G1 APPROVED（Round 4，11:36 CDT）并 commit+push（`79f0e20`，仅 plan+索引两文件）。**下一步 = §4 Code：U0→U6 逐单元实现、每单元过 G2**；实验发射需 owner 排 H200 资源窗口（预算 ~12,700 ep ≈ 13–14 h）。

## 2. X15 一页速览（细节以 plan 为准，此处只帮你定位）

- **目标**：teacher(pi05)/cache 二臂，SR→0.80 一带，teacher 份额 ≪ 盲混合的 ~0.73。**C1（唯一 primary）**= 近似匹配份额（|Δshare|≤0.02）下 A 池 500 配对赢过 global 常数；**C3 headline** = B-test 上独立重测的 iso-SR@0.80 份额节省。
- **方法**：shadow-teacher 逐步偏差标签（`u_t=‖a_C−a_T‖/σ_a`）→ 59 维检索侧特征 → ~25k 参数 MLP → isotonic(B-cal) → τ 闭环网格。与 X14 的关系=互补（代理标签监督 vs 语义正确 RL；outline Q1 按二分修订——owner 裁定 ④）。
- **六项 owner 裁决**（2026-08-22「按你的建议」，已记 plan §0-pre）：① t/T_max 属 A 档 ② B 档只作 ablation ③ headline=iso-SR ④ Q1 二分修订 ⑤ D1 功效兜底分支 ⑥ 驻留属执行器策略。
- **关键冻结契约**（都是评审换来的，实现时别走样）：backend 原子 `search_with_diagnostics` + 每连接 facade 快照（防 BackendPool 共享实例并发串线）；**双时间轴**（query `decision_idx×replan_steps` / 库侧 `CacheEntry.step_idx×library_replan_steps`，两者都是推理周期需换算）；shadow RNG 隔离（`sample_noise` 加 generator 参数、sha256 稳定 seed、同 device）；B 池四方互斥（gradient 300/δ 50/B-cal 50/**B-test=章程 B-val 50 零拟合**）；p̂ 与全部对照参数 B 侧冻结先于触碰 A；G0 止损门在 gradient 侧。
- **实施单元**：U0 backend 侧信道(L2) / U1 离线分数管线(L1) / U2 P0-b yamls+dump 扩展(L2) / U3 shadow 接线(**L3**，触点含 `pi0_pytorch.py`) / U4 特征+训练(L2) / U5 `risk_router` judge(**L3**) / U6 统计驱动(L2)。文档触点：`docs/architecture/cache_system.md`、`docs/cache/tutorial.md`。
- ⚠ **批准后发现的事实漂移（G2 时披露）**：plan §2/§4 写「现存 b0284–87 四批 ≈21.6k 步」作 U1 冒烟——实测 **只有 b0287 保有完整 .bin 特征**（201 文件/836 MB ≈5.4k 步），b0227–b0286 的 61 个目录只剩 jsonl 残骸（~3 MB/批，bin 已被 reclaim）。只影响管线冒烟集大小，不影响任何判据。

## 3. X14 遗产（收官终态）

- **结论**：ts 线打平；tc@λ=0.5 角点平凡解；tc@λ₃=5.0 跑到 **288 批/28,867 ep**（v288），b0195 起 reward 横盘（0.505 / SR 0.694 / teacher 0.289），**增量 100% 来自省成本、oracle 空隙只回收 ~3%**。§2.5l 的 b0400 停跑判据没等到数据（owner 于 b0288 外部指令停，报告口径已写清）。P3 从未运行。
- **必带边界**：信息屏蔽（`mlp_router` 冻结）；硬件台阶 b0227(4090→H200)；4090 静默算错窗 b0143–b0215；λ 不外推。
- **数据资产（未删一字节）**：ziyang10 `rl_router/art_m6/l10_tc_lam3_s0/`（4.5 G：checkpoint、state v288/288、metrics、weights v228–288、b0227–87 批产物）；**权重链跨机 v0–v227 在 wls** `rl_router/art_m6/.../weights/`；t107 `/scratch/zixuans8/rl_router/m6/l10_tc_lam3_s0/` 288 批 journal/client_rows；ziyang10 `dump_m6` 见 §2 漂移条；ziyang10 `rlr_backup_0821/`（迁移前 src 原件）。

## 4. 基础设施：全部已拆，恢复手册

**当前**：t107/ziyang10/wls 上本线进程=0；expose 池零占用；本会话 Monitor/cron 已全撤；无 stop-hook。他线在跑的别碰：ziyang10 `srv0/srv1`（h200 探针线）、wls `wsdrv*/wssrv*` 与 t107 端点 `ziyanglin.com:2316x` 的 worker（ws_search 线）、wls `keepwarm`。

**X15 发射时的恢复步骤**（次序即依赖序）：
1. `tether expose jupyter-ziyang10 --local 8999 --name rlr-inf` 与 `--local 2222 --name rlr-ssh`（**端口必变**，四处同步：发射脚本 `--servers`/`--remote-port`、`/tmp/rlr_health.sh` 与 `/tmp/reap_orphans.sh` 的 `SERVER_KEY`——必须与 worker argv 逐字一致）。⚠ broker 实测 15.01 MB/s vs 自建中继 3.85（§3.36）；**relay.py 已弃用**（文件还在 wls `gtp_logs/`、ziyang10 `tools/`，别再起）。**expose 用完即还**。
2. ziyang10 用户态 sshd **不自启**：`/home/ziyang10/sshenv/bin/sshd -f /home/ziyang10/sshd_run/sshd_config -E /home/ziyang10/sshd_run/sshd.log`。
3. pi05 server：`tmux new -s rlrsrv -d "cd /home/ziyang10/openpi && export HOME=/home/ziyang10 && /home/ziyang10/.local/bin/uv run scripts/serve_policy.py --replicas 1 --port 8999 --cache_config <X15 arm yaml> policy:checkpoint --policy.config=pi05_libero --policy.dir=/home/ziyang10/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch 2>&1 | tee /tmp/rlr_srv_h200.log"`（~3 min）。
4. **跨机/换通道后发全量前必跑 2×2 真推理冒烟**（§3.29 规程）；X14 发射脚本 `/tmp/run_m6_h200.sh` 端点已 stale，仅作模板。
5. 监控重挂套件见下方 §7（⚠ 旧版 handoff 从未 commit，此处是唯一存续副本）。

**硬约束**：ziyang10 **32 GiB RAM cgroup**（OOM 杀全 pod 含 tether agent；离线作业流式 ≤8 GiB RSS、不与 server 并发）；H200 显存不是瓶颈 host RAM 才是；单 pi05 replica。

**安全红线**（全有事故背书）：绝不 kill 非本会话进程；孤儿判据=父链 tmux/systemd **且按当时那次 run 的端点收窄**（先 `/tmp/inspect_workers.sh` 逐进程定性，DRYRUN 看 KEEP 侧再实清）；`pkill -f`/grep 自匹配坑（tmux server 的 argv 会含首条命令文本——凭 argv 判服务残留必配端口监听交叉验证）；err>0 先分类再定性（§3.34）；`ConnectionClosedError` 成片先查节点 OFFLINE（§3.35）；git 未授权不 add/commit/push、英文消息、无 AI 署名。

## 5. Git 状态

- **已入库**：`79f0e20`（X15 plan + logs/README.md 索引）→ origin/Ziyang。
- **本线未提交**（等 owner 授权）：`logs/rl_router_operations.log.md`（§2.5j–l + §3.24–3.37）、本文件、`exp/rl_router/run_rl_router.py`（resume 便宜清扫）、`tests/exp/test_rl_router_run_loop.py`（+5）。
- 工作树里 robocasa365/latency_bench/`src/openpi/cache/backends/in_memory_backend.py` 等改动是**他线的，不碰不提**。

## 6. 下一步（次序）

1. **§4 Code**：按 plan §11 U0→U6 实现（U0 先行——U1/U2 依赖它的 `StepRetrievalFeatures`），每单元 G2。
2. owner 排 H200 窗口 → 恢复拓扑（§4 步骤）→ P0-a 冒烟（b0287 一批）→ P0-b/P0-c → **G0 裁决**。
3. G0 过 → Phase A(2,000 ep) → B → C → D；G0 死 → 止损转 ws_search（cache 质量线），本线出负结果短报告。
4. 报告阶段：起草 outline Q1 二分修订稿（裁决 ④）+ X14 收官报告更新（`tc_line_results.md`）。

## 7. 监控重挂套件（X15 发射时照抄；Monitor/cron 均 session-only，重启/compact 后按此重建）

**Monitor（persistent，主跑期）**——⚠ 三代教训：fatal 正则必含 `ALERT|Traceback|exited 255`；必查 tmux 会话存在性（进程没了是最强终态信号）；必查 ziyang10 节点在线性（整机掉线时 traceback 只是间接症状）。偏移 `off` 每次重发后重取（`grep -n "resume:" <log> | tail -1`）：

```bash
prev=""; off=<现取>
while true; do
  cur=$(tether exec timan107 -- bash -lc "D=<out-dir>; L=<log>; alive=\$(tmux ls 2>/dev/null | grep -c '^<会话名>:'); fatal=\$(tail -n +$off \$L 2>/dev/null | grep -cE 'ALERT|Traceback|LAUNCH BLOCKED|CUDA error|OutOfMemory|exited 255'); fin=\$(tail -n +$off \$L | grep -c '########## RUN'); nb=\$(ls -d \$D/b0* 2>/dev/null | wc -l); t=\$(cat \$D/b0*/journal.jsonl 2>/dev/null | wc -l); echo \"alive=\$alive fatal=\$fatal fin=\$fin batches=\$nb decile=\$((nb/10)) episodes=\$t stamp=\$(date -u +%H:%M)Z\"" 2>/dev/null | tail -1)
  z=$(tether node ls -a 2>/dev/null | awk '$1=="jupyter-ziyang10"{print $2}'); [ "$z" = "ONLINE" ] || cur="$cur ZIYANG10=$z"
  x=$(tether ps -a 2>/dev/null | grep -c "rlr-inf\|rlr-ssh"); [ "$x" -ge 2 ] 2>/dev/null || cur="$cur EXPOSE=$x"
  key=$(echo "$cur" | sed -E 's/(episodes|batches|stamp)=[^ ]*//g')
  if [ -n "$cur" ] && [ "$key" != "$prev" ]; then echo "X15RUN $cur"; prev="$key"; fi
  case "$cur" in *fin=[1-9]*) echo "X15RUN finished"; break;; esac; sleep 240
done
```

**cron ①（45 min）t107 巡检**：跑 `bash /tmp/rlr_health.sh`（SERVER_KEY 先同步！），健康只回一行。行动条件六条（照 X14 终版）：(a) err>0 **先分类再定性**（§3.34；`ConnectionClosedError` 成片 ⇒ 先查节点 OFFLINE）；(b) saturated 真饱和才停；(c) 孤儿先 `inspect_workers.sh` 定性、**按当时端点收窄**、DRYRUN 看 KEEP 侧；(d) t107 vram<600 只观察；(e) 三拍无进展且 quiet>900 看现场（resume 有静默重建段，看 `reclaim --shards` 推进）；(f) alive=0 ⇒ 依次查节点在线性→expose 存活→再重发（重发前收孤儿、重发后推偏移）。**共享机绝不宽模式 pkill。**

**cron ②（每小时 :23）通道+H200 看门狗**：探测 expose 两条在位 + `healthz=200` + ziyang10 `VRAM` + tmux `rlrsrv`。行动：expose 掉了重建（**端口必变 ⇒ 四处同步再重发 conductor**）；sshd 掉了手动拉（不自启）；server 掉了重启（~3 min，conductor ep 级自愈）；节点 OFFLINE 只能等 owner 重开 jupyter；`VRAM<5000` 只观察绝不释放他人显存。
