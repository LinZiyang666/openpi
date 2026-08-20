# Session Handoff — RoboCasa365 跨场景 cache 实验（T5 采集就绪版）

> ⚠ **本文件不是 `logs/session_handoff.md`**。那一份属 X14 RL Router session，勿动、勿暂存。

**Status**: `Ready-to-collect` — 工程面收官，**owner 裁完 T4b 即开跑 T5 正式采集**。
**日期**: 2026-08-18（覆写第四版；旧版内容已被本版取代）

---

## 0. 一句话现状

**统一临时 G2 已 APPROVED（Round 11）+ §6 Verify 已过（1496 passed/0 failed）**。临时 G1 五轮 + 统一 G2 六轮共 30 项 blocking 全闭合；真机门禁全套通过（manual 5–8、老 T-8 9/9、G0-E 106/106 FULL_HIT + stage2 零采样），证据在 `exp/robocasa365/analysis/`（8 份）。本地与远端同在 git 头部附近（见 §6 待办）。**下一步 = owner 裁 T4b → 按 §3 runbook 开跑。**

## 1. 接手第一步

1. 本文件读完。
2. plan 权威 = [`logs/robocasa365_framework_integration.log.md`](robocasa365_framework_integration.log.md)：冻结契约（§4.3.x 全部 🔒 小节）、**逐 task N 表已回填 §7.2**、Review Log 含全部审查历史与真机证据轮次。
3. Authority = **Execution**（读 `protocols/execution_authority.md`，勿读 review 侧）。
4. ⚠ 正式采集属 T5，plan/G2 均已就绪，**只差 T4b 的 owner 裁决**（§4）。

## 2. 冻结的采集设计（不可再议的部分）

- **建库场景 (1,1)**，输出根 `/data/robocasa365_cache/build_l1s1`（**场景根**，collector 自己插 `<teacher>` 层）；最终 `build_l1s1/<teacher>/<Task>/episode_NNNN_aAA.h5`。
- **种子**：采集段 `base_seed=0`，评测段 `base_seed=1000000`；`env.reset(seed=base_seed+episode_idx)`；契约 = same initial state, fresh stochastic rollout（重试写新 `_aAA`，永不覆盖）。
- **完成规则**：每 task 跑 `N`（§7.2 表）= 点估计二项 0.90 分位；不足 20 条成功走**补批规程**（`--batch N+1` + `--episode-lo` 从上批末尾接续，新 run-plan 文件）。
- **拓扑（D-L）**：1 server 进程 ↔ 1 连接 ↔ 1 worker；横向扩 = 多 server 进程 + `--gpu-ids`；**一个 teacher 一次 driver 调用**（亲和门在 env-config 里按 `PI05_SERVERS`/`GROOT_TP_SERVERS` 分组）。
- **manifest**：只收 journal `accepted∧success∧error is None` 的 attempt、按 `episode_idx` 升序前 20、sha256 钉死；建库只消费 manifest。
- **审计是 T5 的阻塞后置**：每批跑完必跑 `verify_collection_artifacts.py`（h5 写失败是静默的，journal 会说谎）。

## 3. 🚀 Runbook（pi0.5 侧示例；GR00T 侧换 server 命令与 `--teacher groot_tp`）

```bash
# ① server（每路一个进程；多路换 --port 8011... 各自独立进程）
tmux new -s rc5srv0 -d "cd /home/weiland/openpi && .venv/bin/python scripts/serve_policy.py \
  --port 8010 --non-concurrent --collect --collect_dir /data/robocasa365_cache/build_l1s1 \
  policy:checkpoint --policy.config pi05_robocasa --policy.dir /home/weiland/ckpt_pi05_robocasa_pytorch \
  2>&1 | tee /tmp/rc5_srv0.log"
# 就绪标志：log 出现 "server listening on 0.0.0.0:8010"

# ② driver+agent+worker（--tasks 用 §7.2 的 N 表逐 task 冒号计数；按 T4b 裁决增删）
tmux new -s rc5run -d "export HOME=/home/weiland; cd /home/weiland/openpi && .venv/bin/python exp/robocasa365/run_collect.py \
  --role all --teacher pi05 --servers 127.0.0.1:8010 \
  --tasks CloseBlenderLid:126,CloseFridge:48,CoffeeSetupMug:48,OpenCabinet:40,OpenDrawer:83,OpenStandMixerHead:83,PickPlaceCounterToCabinet:40,PickPlaceCounterToStove:24,PickPlaceDrawerToCounter:33,PickPlaceSinkToCounter:28,PickPlaceToasterToCounter:61,SlideDishwasherRack:61,TurnOnSinkFaucet:40 \
  --layout 1 --style 1 --base-seed 0 --batch 1 \
  --collect-root /data/robocasa365_cache/build_l1s1 \
  --env-config exp/robocasa365/config/collect_weilandserver.env \
  --connect-deadline-s 60 --episode-deadline-s 900 --terminate-grace-s 30 \
  2>&1 | tee /tmp/rc5_run_pi05.log"
# 完成标志：log 出现 "[run_collect] driver finished"；崩溃直接重启同命令（journal 续跑 + run-plan hash 校验）

# ③ 审计 + manifest（跑完必做；多批时 --run-plan 重复给）
.venv/bin/python exp/robocasa365/verify_collection_artifacts.py \
  --root /data/robocasa365_cache/build_l1s1 --teacher pi05 \
  --journal exp/robocasa365/data/journal_collect_l1s1_pi05.jsonl \
  --run-plan exp/robocasa365/data/run_plan_collect_l1s1_pi05_b01.json \
  --target 20 --report-out exp/robocasa365/analysis/t5_audit_pi05.txt \
  --manifest-out exp/robocasa365/data/manifest_l1s1_pi05.json

# ④ provenance（run 结束后抓，单连接 server 跑时抓会被 1013 拒）
{ tr '\0' ' ' < /proc/$(pgrep -f "^[.]venv/bin/python scripts/serve_policy[.]py --port 8010")/cmdline; echo; \
  .venv/bin/python -c "from openpi_client.websocket_client_policy import WebsocketClientPolicy as W; c=W(host='127.0.0.1', port=8010); print(repr(c.get_server_metadata())); c.close()"; \
  sha256sum /home/weiland/ckpt_pi05_robocasa_pytorch/model.safetensors; } \
  > exp/robocasa365/analysis/t5_server_provenance_8010.txt

# ⑤ 软链（D-G，首次开跑前建一次）
ln -sT /data/robocasa365_cache /home/weiland/openpi/exp/robocasa365/data_symlink_to_data_disk
```

**GR00T 侧 server**（孤岛 B；worker 不变，`--teacher groot_tp` + `GROOT_TP_SERVERS` 端口）：
```bash
tmux new -s rc5gsrv -d "export HOME=/home/weiland; cd /home/weiland/openpi && \
  PYTHONPATH=/home/weiland/gr00t_n15:/home/weiland/openpi/src:/home/weiland/openpi \
  /home/weiland/gr00t_n15_venv/.venv/bin/python exp/robocasa365/serve_groot_n15.py \
  --collect-hdf5 /data/robocasa365_cache/build_l1s1 2>&1 | tee /tmp/rc5_gsrv.log"
```
✅ **T5 实测修正（2026-08-19）**：conductor 路径对 GR00T 完整可用——`run_collect --teacher groot_tp` 端到端验证通过（GrootTeacherAdapter 在 episode_runner TEACHERS 注册，journal/run-plan/审计全套同 pi05）。⚠ `--collect-hdf5` 与 pi05 的 `--collect_dir` **同语义 = 场景根**（`build_l1s1`）：GrootCacheCollector 内部就是 EpisodeDataCollector，episode_name 自带 `<teacher>/<Task>/` 层；传 teacher 根会得 `groot_tp/groot_tp/`（已实踩，episode 0 手工归位）。旧文本推荐的 `groot_tp_raw` 裸堆形态作废。

## 4. ⏳ T4b 待 owner 裁（每条给默认建议，裁「按默认」即可开跑）

| 项 | 建议默认 | 备注 |
|---|---|---|
| D-A 顺序 | **先 pi0.5 后 GR00T** | 你原话「先 pi 在 gt」；两侧独立，顺序不影响科学性 |
| D-C `TurnOnSinkFaucet` | **(a) tp 侧封顶 100 ep**（攒到几条算几条），pi0.5 侧照 N=40 收 | 剔除会连 pi0.5 侧 ŜR=0.6 的好数据一起丢；封顶省 156 ep≈4.2h |
| D-D 子集 | **宽·pooled 13**（§7.2 表即按此口径） | 12↔13 只差 TurnOnSinkFaucet，与 D-C 联动 |
| D-E 种子 | 采集 0 / 评测 1000000 | 已冻结在机制里，此处只确认数值 |
| D-F 失败轨迹 | 全落盘（现成行为），manifest 只收成功 | 磁盘代价已计入 790 GB 估算 |
| D-G 存储 | `/data/robocasa365_cache` + 软链（runbook ⑤） | 路径公式已冻结 |
| D-H L0 阶梯 | 纳入且第一个跑（评测段种子，另一次 run_collect 即可） | 廉价熔断 |
| D-I 物体实例 | 本计划不做（相反裁决须另开 plan+G1+G2） | §3.2.1 冻结 |
| D-J 场景水平 | 维持 2×2（layout∈{1,5}×style∈{1,7}） | 加水平每场景先付 ≈3.75h 准入门 |

**预算（默认口径）**：pi05 715 ep ≈19.2h + tp（封顶后）559 ep ≈15.0h ≈ **34h 单路**；双 server 进程对半。

## 5. 拓扑 / 纪律 / 陷阱

- **weilandserver 4090（49G）与 owner 其它 session 共享**：8000 端口 serve_policy、8030 serve_policy、两个 sidecar **绝不可动**；⚠ 每路 ≈21G（server 8 + sim client 13），开跑前 `nvidia-smi` 查 free，**不够不硬挤**（owner 明令不伤别的进程）；显存读数波动大，抢大块前连续 3 次 ≥ 需求。
- 端口：我方 pi0.5 用 8010+（多路 8011…）、GR00T 用 8020+；tmux 一律 `rc5*` 前缀；禁宽 pkill，pgrep 用 `^[.]venv/bin/python …` 锚定（sh 包装与自匹配双坑）。
- **同步方式（owner 裁定）**：迭代 `tether push --force` 直传 tracked 文件；git 只做里程碑收口（收口时远端先 `git checkout -- <直传文件>` 再 `ff-only pull`；**新文件必须先走 git**，untracked+incoming 同名会中止 pull）。
- 孤岛 A（sim client）：venv `/home/weiland/Isaac-GR00T/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv`，cwd 必须 `/home/weiland/Isaac-GR00T/external_dependencies/robocasa365`，EGL 三件套必需（run_collect 的 spawn 已封装，手跑 client 才需手动 export）。孤岛 B（GR00T）：venv `/home/weiland/gr00t_n15_venv/.venv`，`PYTHONPATH` 必含 `/home/weiland/gr00t_n15`。
- tether exec 单次 ~10min 硬上限：长跑一律 tmux+tee，本地 until-grep 短查。
- ⚠ h5 写失败静默 + journal 会把"完成"记在没有文件的 episode 上 ⇒ **审计不可跳过**；manifest 只认审计产物。
- ⚠ `--collect` server 单连接：run 进行中任何第二连接（包括 provenance 抓取）都会被 1013 拒。
- ⚠⚠ **serve_policy 采集模式偶发挂死**（T5 实录 3 次）：签名 = episode 写毕 + "Connection closed" 后进程全局失响，client 报 `1011 keepalive ping timeout`，driver 空转把剩余 uid 全 raise 光（不落账，resume 会重跑，只亏时间）；wchan 取证 215/219 线程 futex_wait = 用户态死锁，与邻居 GPU 高载时段相关。**处置**：监控 driver 日志 keepalive 计数 ≥4 即弹换（重起 server → 原样重发 driver）；根因修复是待立项的工程债。
- ⚠ **审计必须放 tmux**：全量 h5 校验 >10min，tether exec 上限会把进程杀在 report 与 manifest 之间（已实踩一次）。
- ⚠ pgrep/kill 的 shell 里**不得同时含重启命令字符串**（纯文本 `serve_xxx.py` 会被自己的 pgrep 匹配 → kill 自杀，已实踩）；清理与重启分两次 tether exec。
- ⚠ 图像 token 事实修正（2026-08-17 真机 A/B）：GR00T 模板把 instruction 排在图像块**后**，段起点恒 20/283/546；load-bearing = ≠ pi0.5 固定表 0/256/512。旧说法「随 prompt 浮动」已废。

## 6. 待办与工作树状态

- [ ] **owner 裁 T4b**（§4，可整体「按默认」）
- [ ] **收尾 commit**（待 owner 授权）：工作树里本线未提交 = plan（G2 APPROVED 条目+N 表回填+状态行）、本 handoff、`logs/README.md`、`key_builder.py` docstring 修正、provenance 尾空格、`t8_island_b_pytest.txt` 归档、两条记忆同步——commit 后远端 `ff-only pull` 收敛
- [x] **T5 pi0.5 侧收官（2026-08-19）**：批1 715 + 批2 59 = 774 ep，369+ 成功；审计 ok:True（13/13 task ≥20，零缺失零 schema 错）；`manifest_l1s1_pi05.json`（13×20）+ `t5_audit_pi05.txt` + provenance 归档；期间 server 挂死 3 次均按恢复流程闭环，零数据损失
- [x] **T5 GR00T tp 侧收官（2026-08-19）**：批1 559 + 批2 18 + 批3 2 = 579 ep；审计 ok:True（13/13 ≥20，零缺陷）；`manifest_l1s1_groot_tp.json`（13×20）+ `t5_audit_tp.txt` + `t5_server_provenance_groot_tp.txt`（四端同 ckpt 双 sha）归档；**双路→四路横向扩真机验证通过**（D-K 降级条款的双路真机证据已补齐，assign_servers 整 task 原子分派实测）；owner 授权的多路加速将 tp 侧从 ~15h 压到 ~9h
- [ ] **建库**（T6：从两侧 manifest 各取 13×20 成功轨迹 build cache artifacts）→ L0 阶梯评测（D-H 第一个跑）
- 最近提交：`def89fb`（远端已同步）；真机证据 commit `d961388`
- ⚠ 工作树混有其它 session 的 ~25 个未提交文件，提交必须逐文件点名

## 7. 证据索引（`exp/robocasa365/analysis/`）

`t2_parity.txt`（跨栈 sha256 逐位同+真 provenance）｜`t3_action_binding.txt`（孤岛 A 6 passed）｜`t3_audit_single.txt`+`t3_resume.txt`+`t3_server_provenance_8010.txt`（单路端到端+SIGKILL 续跑+provenance）｜`t8_island_b_pytest.txt`（9 passed）｜`g0e_closed_loop.txt`+`g0e_hit_log.jsonl`（106/106 FULL_HIT、stage2 零采样）
