# Session Handoff — X14 在线 RL Router 基线（TIER 论文线）

> 更新：**2026-08-18 12:4x CDT**。**本文完全覆盖旧版**，为 compact 后接手而写：读完这一篇 + `rl_router_operations.log.md` 就能接着干，不需要对话历史。
> 三份文档的分工：**本文 = 全部历史结论 + 现在怎么办**；[`rl_router_baseline_plan.log.md`](rl_router_baseline_plan.log.md) = **冻结设计（唯一权威）**；[`rl_router_operations.log.md`](rl_router_operations.log.md) = **运行期实录与全部实测缺陷**（本文引用的 §x.x 都指它）。冲突以 plan 为准。
> 正式结果报告（面向论文）：[`exp/rl_router/analysis/ts_line_results.md`](../exp/rl_router/analysis/ts_line_results.md) 与 [`tc_line_results.md`](../exp/rl_router/analysis/tc_line_results.md)。

---

## 0. 接手第一步（不可跳）

1. 读 `WORKING_AGREEMENT.md` → 声明 **Execution Authority** → 读 `protocols/execution_authority.md`。**绝不读 `review_authority.md`，绝不碰 `tests/review_tests/`**。
2. 出状态卡，等 owner 确认后再动。
3. 读 ops log §3（23+ 类实测缺陷）——每一条都是真机上咬过人的。

```
WORKFLOW STATUS | Authority: Execution | Task: X14 在线 RL router 基线 | Level: L3
实现线 ✅ commit 721d3fa 已 push
运行期：M5a✅ M5b✅ M4✅ M5c✅(标定无效) G-launch✅ → M6 ✅ 5/5 run 完成（全 ADMISSIBLE）
→ ts 线已结案 / tc 线已结案 / 吞吐与拓扑已标定 → **7 天重跑提案待 owner 批（§2.5i）**
```

---

## 1. 任务是什么

**论文线**：TIER（experience-tiered inference）投 ICLR 2027，thesis =「经验库的价值在索引不在 payload」。系统按检索相似度 + 双阈值把每个控制步派给 **teacher**（Pi0.5）/ **student**（蒸馏 ACT）/ **cache**（FULL_HIT 回放）。

**X14 回答**：「为什么不训一个 router？」——真训一个在线 RL router 当基线，给「训 router 要花多少交互、能买到什么」标价。三个变体 `R_ts`/`R_tc`/`R_tsc`；MLP 在 verdict 层，只看 `build()` 后的 `query_keys`，对库侧信息全屏蔽；batch on-policy REINFORCE，`R_ep = success − λ·(Σcost)/T_max`，批均值 baseline，每批恰一次 Adam step。

**口径**：训练只用 **B 池**（差集池 450/套件）；**A 池**（pruned_init 500/套件）只留最终一次性评测（M7）。

**冻结常数**：λ₁=0.5、λ₂=0.05；cost teacher **1.0** / student **0.055656** / cache **7.2165e-05**；`T_max=520`、`N_STEPS=54`（⇒ 成本系数 K=54/520=**0.10385**）；batch=100、lr=3e-4、β=0.01、clip=1.0、seed=0、`mode: sample`、temperature=1.0。

---

## 2. 实验历史与全部结果

### 2.1 基础件（全部冻结、可复用）

| 件 | 产物 | 关键数 |
|---|---|---|
| M5a cost | `art/arm_costs.json` | 上面那三个数，batch=1 GPU-time |
| M5b warm-start | `art/warmstart_l10.pt`（ts）/ `_tc.pt` / `_tsc.pt`（后两个为同 trunk 重 graft，`register_variant_warmstarts.py` 已登记 trunk_sha） | ts: CV 0.555<熵 0.656，部署 student 率 0.5000。**tc 版输出层全零 = 均匀常数，是 §3.8 预注册设计不是缺陷**（graft 把 success logit 放 student 行，tc 无 student） |
| M5c λ pilot | λ₁=0.5/λ₂=0.05 已写入矩阵 | **非单调、标定无效**（§2.5e），且只在 ts 上跑过 |
| M4 smoke | passed | 每步 dump 131,136 B |
| 锚点 | 纯 student 0.787（B 池 450ep）；纯 teacher **0.8650（同池，tc 扫点 p=1.0）**与 0.868（跨池 A 池）互证 | |

### 2.2 M6 五条 run 全部完成（全 40 批 / 4000 ep / ADMISSIBLE）

| run | λ | 起点 | success 均值 | 份额净移 | 结论归属 |
|---|---|---|---|---|---|
| `l10_ts_lam1_s0` | 0.5 | teacher 0.46 | 0.907 | student −0.109 | ts 线 |
| `l10_ts_lam1_s1` | 0.5 | 同权重换 seed | 0.904 | −0.045 | ts 线 |
| `l10_ts_lam2_s0` | 0.05 | 同 | 0.904 | −0.082 | ts 线 |
| `l10_ts_lam1_s0_knee` | 0.5 | **teacher 0.30（实测拐点）** | 0.895 | −0.115 | ts 线 |
| `l10_tc_lam1_s0` | 0.5 | **均匀 0.50** | **0.767** | cache −0.030（z=−6.17） | tc 线 |

五条 run 的共同点：**被优化的目标（reward）没有一条有超过 1σ 的上升**。每 episode 成功率方差逐位等于二项 p(1−p)（无结构可榨）；(task,init) 条件化 baseline 只去 33.8% 方差（=1.5× 有效样本）。每变体交互账：warm-start 450 + pilot 1800 + 训练 4000 = **6250 ep ≈ 10.7 GPU·h**。

### 2.3 ts 线判决（结案，报告 `ts_line_results.md`）

**SR(p) 扫点**（7 点 ×200ep，同一组 200 对 (task,init)）：0/0.775、0.048/0.810、0.097/0.875、0.151/0.845、0.203/0.850、**0.305/0.925**、0.446/0.910。**p≳0.10 后各点 CI 全重叠 = 平坦段**；拐点 p\*=0.30。

**三口径判读**（`router_vs_fixed.py`）：分箱 6 格 5 负；四 run 整程均值全负（−0.003~−0.028）；最宽容口径（任一 run 最好 1000ep 窗 vs 最好常数）**−0.008±0.021（z=−0.39）**。⇒ **打平**（强陈述只能到打平不能到更差——扫点绝对水平带子样本偏移）。

**机制**：固定效应斜率 dSR/d(teacher share)=+0.0044±0.0108（整 episode 口径 −0.70 是反向因果不能用）；策略均值游走而**跨状态 sd 0.1191→0.1172 纹丝不动**——从没学过区分状态。**knee run 证伪了「问题是起点」**：起点挪到拐点 reward 依旧不动，真因是平坦段无信号。trainer 无符号错误（dJ/dd 与权重位移同向）。

### 2.4 tc 线判决（结案，报告 `tc_line_results.md`）⭐ 本会话最重要结论

**SR(p) 扫点**（teacher/cache，7 点 ×200ep）：0/**0.520**、0.154/0.580、0.302/0.725、0.447/0.770、0.549/0.790、0.700/0.845、**1.000/0.865**。**单调、全域涨 0.345**——与 ts 的平完全不同形状。

**λ=0.5 下目标最优在角点 p=1.0**（J=0.8131；成本项在 p=1 才 0.052 ≪ SR 收益 0.345）⇒ **最优本来就是常数「永远用 teacher」，router 构造上无事可做**。而 **RL 方向是对的**（cache 0.507→0.476 朝角点），只是需 ~700 步只给了 40 步（走完 6%）。**tc 不是学不动，是问题被 λ 摆成了平凡解。**

**λ 决策表**（J=SR−λ·0.10385·p，全由扫点导出）：

| λ | argmax p | 角点边距 vs p=0 | 右侧 gap | 跨度/批噪声 |
|---|---|---|---|---|
| 3.0 | 0.302 | 0.111 | **0.0002 躺平** | 2.8 |
| 4.0 | 0.302 | 0.080 | 0.015 | 3.7 |
| **4.5–5.0** | **0.302** | 0.048–0.064 | 0.023–0.030 | 4.7–5.6 |
| 6.0 | 0.302 | **0.017 塌向纯cache** | 0.045 | 7.4 |

⚠ 角点边距 n=200 下只有 ~1σ ⇒ 发射前须加密扫点。**λ 网格是在 ts 上标定的，tc 必须重标——这是 7 天提案的根。**

**匹配占比对照**：三箱全负（−0.027/−0.010/−0.015），整程 −0.0148 ⇒ 打平。⚠ `z=−3.09` 那条是「router 工作点 vs 全局最好常数」，**不是**匹配比较，别混用。

**机制**：tc 从零头起步，跨状态 sd 0→**0.0079** 而均值移 0.0273 ⇒ 学到的 ~3/4 是全局偏置；**`argmax=cache` 恒 0 ⇒ M7 若用 argmax，tc router 与「永远用 teacher」不可区分——M7 前必须定死评测模式**。

### 2.5 配对检验与 Holm 教训（§2.5h.1）

ts 与 tc 扫点用**同一组 200 对**（task_uid 逐位一致）⇒ 可配对。「混合 p=0.305 (0.925) vs 纯 teacher (0.865)」配对 McNemar 名义 **p=0.0428**——**但那是从 6 个点挑 argmax，Holm 后 0.257 全否**（6 点差值三正三负）。⇒ 「混合优于纯 teacher」**只是线索**；要成立须预注册单点重测。**教训：凡从曲线挑最好点再检验，必须先多重校正。**

### 2.6 吞吐、扩容、拓扑（全部实测，§3.23/§3.23.1）

- **吞吐模型**：每 worker 每 episode 秒数 = **86 + 65.6·p**（86s 仿真可并行，65.6·p 是 teacher 推理争 GPU）。每 episode ~**65 次 server 往返**（action chunk ≈8 步/次），每往返上行 ~**346 KB**（msgpack 裸 numpy 无压缩），周期 **1.66–1.88 s/往返**。
- **实测吞吐**：N=16 稳态 **496 ep/h**、N=24 **671 ep/h**（**1.353×，90% 效率**）；训练态 = 稳态×0.88 ≈ **591 ep/h**（批间 trainer 步+worker 重生）。整程 tc run 437 ep/h。
- **显存**：每 worker **440 MiB**；**N=24 是共享 4090 的安全上限**（N=32 把空闲压到 4.5G，差点挤到他线 csmain）。server 7.8–9.2 G。
- **多 server 现在没用**：`assign_servers` 每 yaml 只映射一个 server，训练每批只有一个 yaml ⇒ 多 `--servers` 让其余 worker 空转。真多 server 要改 `RouterBatchStrategy.plan()` 成每 server 一 Stage（且 `RemoteRun.shards()` 单路径 ⇒ 须同机共享文件系统）。**先加 worker，再 `--replicas`，最后才考虑多 server。**
- **跨机链路（实测）**：wls→broker(racknerd) 14ms/21.5MB/s；ziyang10→broker 5ms/50.8MB/s；隧道端到端 41–115ms / 7.8–19.3MB/s（broker 是 1vCPU VPS，限的是它的网不是 CPU）。**旧的「跨机慢 7.8×」已推翻**——那是旧 broker 0.9MB/s 带宽打满（0.9/0.346=2.6RT/s=2.4ep/min 与旧记录逐位吻合）。跨机延迟惩罚仅 **+2~7%**；带宽需求 N=24≈4.3-4.9MB/s、N=32≈5.8-6.5、N=40≈7.2-8.2（贴边）。最大杠杆改进 = 观测 JPEG 压缩（10-20×）。
- ⚠ **t107→wls 这条具体路径没测过**（入口 `155.98.36.13:9000` frp 或 `linziyang.top:14007` expose），发射时 T0 必测。

### 2.7 我在本会话纠正过的数（勘误表，别再用错的）

| 错 | 对 | 出处 |
|---|---|---|
| 吞吐 833 ep/h | **437**（整程）/ 591（N=24 训练态） | §3.23（4000 除以只跑 2500 的窗口） |
| 跨机慢 7.8× | 带宽打满的历史产物，现 +2~7% | §3.23 |
| 每往返周期 1.06s | **1.66–1.88s**（把跨 worker 墙钟当单 worker 周期） | §3.23 修正 |
| 「混合优于纯 teacher，p=0.043」 | Holm 后否，只是线索 | §2.5h.1 |
| tc 零输出层是退化 | 是 §3.8 预注册设计（graft 无 student 行可放） | §3.18（我为此误停过一条正确 run） |
| 500 批 = 60 GPU·h/2.5 天 | **=101.5h/4.2 天 @N=24**（85h 系 437 时代旧算） | §2.5i |

---

## 3. 当前状态（快照 2026-08-18 ~21:15Z / 16:15 CDT）

- **⏸ 主跑 `l10_tc_lam3_s0` 已暂停（owner 2026-08-20 要跑别的任务，说「好了」才恢复——**不得自行恢复**）。断点：**143 批 / 14,324 ep 完成**，trainer_state.consumed_batches 到 b0142，权重链到 **v143.pt**，产物全在 wls `art_m6/l10_tc_lam3_s0/`（**绝不删**）。两台机器本线**零占用**：t107 worker 已收 48 个（DRYRUN 验名单后）、wls server 已停（显卡 48511 MiB 全空）。监控全撤（Monitor + 三条 cron），恢复时重挂。
  **恢复顺序**：① wls 起 server（handoff §5.1 命令，cache_config 用 `m6/l10_tc_lam1_s0/r_tc_train.yaml` 占位）→ ② t107 上 `curl -m8 http://linziyang.top:14007/healthz` 应 200，不通则查 `tether ps -a` 的 rlr-srv exposure（可 rm 后重建，端口通常原位复用）→ ③ t107 `tmux new -s rlrm6 -d "WORKERS=24 GPU_IDS=<按当时 nvidia-smi 现读> bash /tmp/run_m6_t107.sh l10_tc_lam3_s0 2>&1 | tee -a /tmp/rlr_m6_l10_tc_lam3_s0.log"`（批级 resume 从 b0143）→ ④ 重挂 Monitor（err 计数用 `tail -n +16818` 跳过 §3.25 崩溃期旧 Traceback）+ 两条巡检 cron（t107 45min / wls 每小时）+ P1 门一次性 cron。恢复后有 ~6 分钟静默重建段属正常（§3.25）。
- 发射全链条：T0（frp 死→broker 入口 6ms/5.6MB/s）→ T1 PASS（632.9 ep/h=0.943×wls）→ λ 加密扫点 4×400（0.5075/0.6875/0.6825/0.7200，0.25-0.302 段平）→ **λ₃=5.0**（§2.5j 预注册规则，min_z=1.41，p\*=0.25）→ 门禁 dry-run 0 problems → 发射。全记录 ops log §3.24/§2.5j/§2.5k。
- **监控**：wls 看门狗 cron `a0cdc845`（每小时 :23，主跑版）；t107 巡检 cron `70c105a4`（每 31min，探针 /tmp/rlr_health.sh t107 版）；条件触发 Monitor `bzp8ieqy5`（每 10 批/错误/完成）；P1 一次性 cron `3130e9f7`（8-19 18:47 CDT 查 batch150 遍历门）。cron 7 天过期，比主跑长。
- 同机他线（**绝不 kill**）：wls `csmain`/`cssrv`/`rc5run`/`rc5srv0`；t107 是共享机（他 key 的 5 天代孤儿域外不碰）。
- t107 工具箱 /tmp/：run_m6_t107.sh、run_t1_cal.sh、run_lam3_sweep.sh、reap_orphans.sh（SERVER_KEY=linziyang.top:14007）、rlr_health.sh、steady_rate.py、list_m6_runs.py。wls 新增 /tmp/run_m6_v3.sh（WORKERS/GPU_IDS 参数化，failover 用）。

## 4. 运行中判据与收尾（预注册 §2.5k，不许改）

1. **P1 遍历门**：batch 150 最近 10 批 teacher 份额均值 ≤0.46；不达 → 停跑改 p=0.35 偏置起点重发。
2. **P2 中期实测（暂停期离线算，ops log 有全表）**：判别度与份额下探是同一件事——b0040-b0110 七十批 sd 钉在 0.005-0.007 且份额卡 0.38 平台；b0120 后 sd 翻倍（0.0074→0.0117）份额才破位到 0.315（已摸到 λ₃ 目标带边缘）。P3 判定工具已落地：`exp/rl_router/analysis/paired_mcnemar.py`（+8 测试，`pytest tests/exp/` 1282 绿）。
3. **P3 主判定**：最后 50 批实测份额 p̂ → 同组 2000 slots「冻结 router(sample) vs 常数@p̂」配对 McNemar α=0.05。**预期：份额 0.50→~0.25-0.30，SR ~0.75-0.79 → ~0.69-0.72 = 成功的样子**。
3. **failover**：连续 3 批 <450 ep/h → 停 t107 conductor，wls 上 `WORKERS=24 bash /tmp/run_m6_v3.sh l10_tc_lam3_s0` 接跑（批级 resume §3.22，**绝不删 art_m6/**）。wls server 掉线 → 重启 server 即可，conductor ep 级重试自愈。
4. amendment 记录：矩阵行 `l10_tc_lam3_s0`（amendment 块）+ basis `exp/rl_router/data/amendment_lambda3.json`（三机同内容）。M7 评测模式仍待 owner 裁（argmax 使 tc 与纯 teacher 不可区分）。

## 5. 运维手册

### 5.1 常驻服务启动命令（进程表是唯一存放处，重启即失）

```bash
# pi05 routed server -> tmux rlrsrv, :8000（tc 工作不需要 sidecar）
cd /home/weiland/openpi && export HOME=/home/weiland && \
export OPENPI_SERVER_GPU_MEMORY_LOCK=0 && \
/home/weiland/.local/bin/uv run scripts/serve_policy.py \
  --replicas 1 --port 8000 \
  --cache_config exp/rl_router/config/libero_10/m6/l10_tc_lam1_s0/r_tc_train.yaml \
  policy:checkpoint --policy.config=pi05_libero \
  --policy.dir=/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch 2>&1 | tee /tmp/rlr_srv.log
# （ts/tsc 工作才需要）ACT sidecar libero_10 -> tmux rlrsc, :7002
/home/weiland/lerobot_venv/bin/python exp/ablation_study/sidecar_server.py \
  --policy act --manifest exp/ablation_study/config/act_manifest_libero_10.json \
  --host 0.0.0.0 --port 7002 --device cuda --timing-log /tmp/rlr_sidecar_act.jsonl 2>&1 | tee /tmp/rlr_sidecar.log
# spatial 版 -> rlrsc2 :7003，manifest 换 act_manifest_libero_spatial.json
```

conductor 侧 conda prefix `/home/weiland/libero_sim`；yaml 由 conductor 逐批热切换下发，占位配置不绑死 suite。

### 5.2 发 run / 恢复

- 发 run：`tmux new -s rlrm6 -d "bash /tmp/run_m6_v2.sh <rid> 2>&1 | tee /tmp/rlr_m6_<rid>.log"`（⚠ 目前 `--workers 16` 写死，上 24 须参数化）。
- **批级 resume**：同一条命令重发即从 `trainer_state.json` 的 `consumed_batches` 续跑，**绝不删 `art_m6/<rid>/`**。ep 级 replay 连未完成批内的进度都不丢（§3.22 实证）。
- 完成判定：日志尾 `RUN <rid> DONE` → `python3 /tmp/verify_m6.py <rid> <批数>`（六断言：链连续/逐批 model_sha 互异/零拒收/join 完整…）。

### 5.3 监控两层模板

L2 Monitor（persistent，240s 轮询）核心循环——每换 run 重挂一个：

```bash
h=$(tether exec weilandserver -- bash -lc 'bash /tmp/rlr_health.sh' | tail -1)
# 解析 err/policy_saturated/orphan_workers/vram_free_mb/progress 并逐项报警
# 完成: grep 'RUN <rid> DONE' 该 run 日志 → 推送并 break
```

L3 cron（20min）判据：(a) `err>0`（`EOFError ... HTTP request line` 是良性噪声）；(b) `policy_saturated=1`（**b0000 已豁免**——tc 的 v0 均匀常数是设计）；(c) `orphan_workers>0` → **先 `DRYRUN=1 bash /tmp/reap_orphans.sh` 看名单含 KEEP 侧**再实跑；(d) `vram_free_mb<3000` → `bash /tmp/gpu_attrib.sh` 看归属、只降本线；(e) **`progress` 连续三拍不变**（不是 `eps`；批间 trainer 步 quiet_s 可达 13 分钟属正常）。

### 5.4 安全红线（每条都有事故背书）

1. **绝不 kill 不是本会话起的进程**；共享卡显存不够=等，不抢。
2. **孤儿判据 = 父链先遇 tmux pane（活）还是 systemd/init（孤儿），且按 `SERVER_KEY`（本线 :8000）收窄**——白名单/直接父进程/朴素父链/子串匹配四代判据全都错过（§3.19/§3.20，误杀他线 20 worker + 自家 16 孤儿吃 8.4G 计数恒 0 + 差点杀掉执行通道）。改判据必 DRYRUN 且看 KEEP 侧。
3. **`pkill -f`/`pgrep -f` 自匹配**：模式含在脚本正文里就会杀到执行 shell（本会话踩 4 次）。用 char-class `[x]` 或独立 argv 元素判据（`grep -lzxF ... /proc/*/cmdline`）。
4. reaper **单趟只清一代**（杀包装层才让孙进程变孤儿）——已改多轮至干净 + 双采样防瞬态误杀。
5. `nvidia-smi --query-compute-apps` **看不见 LIBERO worker**（EGL graphics 上下文）——查占用用 `/tmp/gpu_attrib.sh`。
6. 长命令写文件 `tether push` 再执行，别嵌套 heredoc（引号断三次）；`tether push` 已存在文件要 `--force`。
7. 动手前**先读目标源码 docstring**（§3.18 没读 graft docstring 误停正确 run）。
8. 中途不判方向（从半段读方向错过 4 次）；预注册判据 + 等跑满。
9. git：**owner 未授权前不 add/commit/push**；commit 英文、无 AI 署名。

## 6. 工具与数据清单

**repo 内**（未提交）：`exp/rl_router/sweep_mixture.py`（`--cheap-arm` 已泛化+28 测试）、`analysis/{router_vs_fixed,knee_and_lambda,plot_reward_curves,plot_sr_vs_share}.py`（后者已参数化 `--cheap-arm/--run/--bin-edges/--title`）、`register_variant_warmstarts.py`、报告×2 + 图×3。

**wls `/tmp/`**（重启即失，ops log §3.11 有清单）：`rlr_health.sh`、`reap_orphans.sh`（DRYRUN=1 支持）、`gpu_attrib.sh`、`run_m6_v2.sh`、`run_sweep_tc.sh`、`run_cal_workers.sh`（N=? P=? 参数化）、`sweep_tc_probe.sh`、`verify_m6.py`、`run_outcomes.py`、`compare_runs.py`、`policy_drift2.py`（`--arm`）、`steady_rate.py`、`paired_mix_vs_teacher.py`、`all_mix_vs_teacher.py`。

**数据根（wls）**：`/home/weiland/rl_router/{art,art_m6,dump_m6,sweep_l10,sweep_l10_tc,cal_w24}`；run 侧 `exp/rl_router/data/m6/<rid>/`；A 池评测入口仍缺（M7，需 `materialize_apool.py` + §3.14 歧义裁决）。

## 7. 未提交改动（owner 从未授权 git add）

`logs/rl_router_operations.log.md`（新增 §2.5f–§2.5i、§3.18–§3.23.1）、`logs/session_handoff.md`（本文）、`exp/rl_router/analysis/*`（2 报告+3 图+4 工具）、`exp/rl_router/sweep_mixture.py`、`tests/exp/test_rl_router_sweep_mixture.py`（28 passed；`pytest tests/exp/` 全绿 1266）。
