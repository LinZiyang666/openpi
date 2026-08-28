# Session Handoff — ws2 检索搜索 Round 2(**全量实验已完成 2026-08-28**)

> 覆盖 2026-08-26/27 会话:text-IVF Round-2 从 plan 到主臂 live。
> **接手第一件事:确认实验还在跑**(§0)。其他线的 handoff 见 `session_handoff_{rl_router,cache_size,robocasa365}.md`,勿混淆。

---

## 0. 接手第一步(照抄)

```bash
export HOME=/home/weiland
timeout 150 tether exec weilandserver -- bash -lc 'export HOME=/home/weiland; /tmp/ws2_sample.sh'
```

输出形如:
```
01:48:49Z journal=2542/13728 (~24/132 cells) complete_at_last_finalize=13
  short +25 in 69s = 1304 ep/h | LONG +381 in 1485s = 923 ep/h | remaining=11186 ETA=12.1h (long)
```

⚠ **速率必须用这个脚本读,不要心算**。上一会话我在速率上错了三次(短窗口踩收尾峰值报高 1224、
时间间隔估错报低 86、把 40 分钟增量当 20 分钟算)。脚本把 `(时间戳, journal 行数, 完成格数)`
追加到 `/tmp/ws2_samples.tsv`,速率由文件算,机器算不会记错。

**读这三个数的正确姿势**(2026-08-27 加固后):
- **以 LONG 窗口为准**。short 是相邻两点,窗口若跨批次边界会系统性偏低——批尾只剩少数格在跑、
  大部分 worker 空转(实测同一时刻 short 655 / LONG 905)。ETA 用 LONG 算。
- **`complete_at_last_finalize` 是陈旧值**:132 个 summary 文件一开始就在,`complete: true`
  只在 finalize 时写。中途它不动是正常的,别当成"卡住了"。
- **活的进度是 journal 行数**,`~N/132 cells` 由它除以 104 推出;要刷新 complete 就手动跑一次
  `--finalize-only`(幂等)。
- ⚠ **想中途看成功率,必须先 `--finalize-only`**:`analyze` 走的 `load_journals` 读的是
  **per-cell** journal(`journal_ws2-<cid>__*.jsonl`),而那只在 finalize 时才写。不跑 finalize
  直接 compare,读到的是**上一次 finalize 的旧快照**,而且看不出是旧的(我踩过:中央 journal
  已 2719 行,compare 却只认 13 格)。finalize 后立刻升到 26 格。

三台机的存活检查:
```bash
timeout 100 tether exec weilandserver -- bash -lc 'export HOME=/home/weiland; tmux ls | grep ws2'
timeout 100 tether exec timan107 -- bash -lc 'export HOME=/home/zixuans8; echo "workers=$(pgrep -fc "[w]orker_entry") agents=$(tmux ls | grep -c ws2agent)"'
timeout 100 tether exec timan1   -- bash -lc 'export HOME=/home/zixuans8; echo "workers=$(pgrep -fc "[w]orker_entry") agents=$(tmux ls | grep -c ws2agent)"'
```
期望:weilandserver 6 个 `ws2s2316x` + 1 个 `ws2main`;timan107 **27 worker / 3 agent**;timan1 **15 worker / 3 agent**(合计 42)。

---

## 1a. ✅ 全量实验已完成(2026-08-28 11:05Z)

四个相位全部收官,每个都 `n_err=0 / n_missing=0`、每格恰好满集数:
GR00T 主臂 132/132、ws2c 控制臂 12/12、ws2e 加密臂 10/10、**pi05 全线 132/132**。
**正式报告:`exp/robocasa365/analysis/ws_search2_groot_results.md`**(七份证据文件同目录)。

四条结论:①库增长 +12.0pp 且 132 格无一退步;②拆开后 **94% 来自库、text-IVF 对 SR 中性
(0/12 显著)** ⇒ 按 owner 口径通过;③复现噪声地板 3.0%、头部格约 5pp 胜者诅咒
⇒ 禁用「最好格 vs 最好格」;④**字段权重不跨执行体迁移**(两 teacher 排序 ρ=+0.175,
最强字段相反)。⚠ 本轮**未测检索延迟**,"更快"这半边无实验支撑。

**收尾待办**:server/车队仍在运行,需 owner 决定是否停机;未 commit 的代码待授权。

## 1b. ✅ GR00T 主臂已完成(2026-08-27 14:17Z)

`finalize-only: 132/132`,journal 13,728 行、每格恰好 104 集、`n_err=0 / n_missing=0`。
**正式全矩阵结果**:平均 macro SR `0.158 → 0.278`(**+12.0 pp / +76%**),132 格无一变差、
95 格 p<0.05;榜首 0.385。⚠ **最优配比迁移**:round-1 冠军(全押 vision_2)跌到 0.279,
新前六名清一色 `robot_state 37–50 + 中等 vision_2`。PickPlace 家族仍未被救活(两格仍为负)。
详见 plan §6b-5,证据 `analysis/ws2_joint_full132.txt`。

**当前相位:ws2c 控制臂**(12 格 × 104 = 1,248 集,2 批 × 6 格,brute_force server 已换)。
采样器与守候进程已泛化为**相位自动识别**(按最新的 `journal_central_*.jsonl` 判定;
守候匹配 `DONE|INCOMPLETE phase=`),ws2e / pi05 相位无需再改。

## 1. 此刻在跑什么(2026-08-27 01:35Z 实测,主臂阶段的历史记录)

**GR00T 主臂 ws2**:132 格 × 104 集 = 13,728;当前 **journal 2360 集 / complete 13 格 / 批 2 of 11 / ~1050 ep/h / ETA ~10.8h**。

| 角色 | 位置 | 细节 |
|---|---|---|
| 服务池 | weilandserver | 6 × `serve_groot_n15 --concurrent --allow-dynamic-bundles`,tmux `ws2s23160..65`,**全在 GPU0**(`--cuda-device 0`);每台 ~6.1G VRAM / ~21.2G RSS |
| driver | weilandserver | tmux `ws2main`,`--cells-per-batch 12 --bind-port 23180`,日志 `/tmp/ws2main.log` |
| worker | timan107 | 3 fleet × 9 = 27,tmux `ws2agent0/1/2` → :23160/61/62,gpu-ids 错位轮转覆盖 8 张卡 |
| worker | timan1 | 3 fleet × 5 = 15,tmux `ws2agent3/4/5` → :23163/64/65,GPU 1/2/3(**避开 GPU0,他人常占**) |

数据落点:`/data/openpi_text_ivf_build/exp/robocasa365/data/ws_search2/groot_tp/`
(中心 journal + 逐 cell journal/summary/run_plan + per_step 证据流)。

**巡检 cron**(20 分钟一次)只活在当前会话,**新会话必须重挂**,否则这轮无人监管。
当前会话的 job id = `c03b5cd3`(`7,27,47 * * * *`);上一会话是 `3d77a18d`,已随会话消亡。

---

## 2. ⚠ 硬件已变更:缺陷卡拆了

owner 于本会话中途装新显卡,**把有冷启动静默算错缺陷的旧 4090 拆除**。现在:

- **单张 RTX 4090 48G**(开机瞬间 `nvidia-smi` 可能报两张,不可信;稳定后只有 GPU0)
- **不再需要 keepwarm,不再看 44°C 温度门**。`preflight_gate` 的 `--min-gpu-temp-c` 默认已改为 **0(关闭)**
- 2026-08-26 之前的日志/证据卡里凡提到"keepwarm 在岗""等温到 44°C",都是缺陷卡时期的产物
- 记忆 `reference_weilandserver_4090_unstable` 已改写为作废;`reference_weilandserver_public_gpu` 里那句"冷卡必须 keepwarm 常驻"**尚未更新,已过时**

---

## 3. 已入库的代码(commit `3598534`,已 push origin/Ziyang)

"Retrieve RoboCasa cells by prompt bucket instead of by task",23 文件 +4958/−10。
G1 四轮 APPROVED、G2 五轮 APPROVED(共 14 条 blocking 闭合)、§6 Verify 裸全量
`4404 passed / 11 failed`(11 条全为既有基线:5 条 HEAD + 6 条其他线的 review_tests 探针)。

**核心改动**:校验规则 6 按正向冻结集放开(`_TEXT_IVF_GROOT_BUILDERS` = 四个 RoboCasa GR00T
pool;`cp1_groot_libero_*` 仍拒);`episode_runner` 新增单次-reset 捕获 hook(默认空操作);
新增 `run_ws_search2`(双图 resume / 族交错 stage_id / 分批 + 失败非零退出)、
`emit_ws_search2_yamls`、`build_selection_manifest`、`build_bucket_variants`、
`analyze_ws2_vs_ws1`、`orchestrate_ws_search2`、`ws2_episode_runner`。

### ⚠ commit 之后还有未入库改动(工作树 27 个文件脏)

这些是部署期真机暴露的问题,**已推到服务器在用,但没 commit**:

1. `build_bucket_variants.py` — **桶键必须复刻 backend 的推导**。pkl 里 `query_keys` 是 numpy,
   backend 在 `load_artifact` 里 `torch.from_numpy(v).float()` 之后才算键;直接读 pkl 会算出
   与服务端索引对不上的桶号。已修 + 对拍测试(fp16 numpy 存档同时喂两条路径断言分组一致)。
2. `build_bucket_variants.py` — `_EPISODE_RE` 要容忍 `_aNN` 尝试后缀(真实 id 是
   `CloseBlenderLid/episode_0000_a01:0`);不容忍则 111 桶全部 unresolved、归因链产出空。
3. `emit_ws_search2_yamls.py` — 泛化到双 teacher(`TEACHERS` 表);pi05 必须带
   `prompt_masked_pool/prompt_instruction_span: true`(与其库 meta 对齐,配错方向启动期即拒)。
4. `orchestrate_ws_search2.py` — `texec` **按节点传 HOME**(worker 岛是 `zixuans8` 不是 `weiland`);
   新增 `--cuda-device`(preflight 用 `nvidia-smi -i <gpu>` 读目标卡);温度门默认关。
5. `orchestrate_ws_search2.py` — **按 teacher 泛化**(2026-08-27 补,原本写死 GR00T,pi05 相位
   会卡在这里):顶层 `--teacher` + `TEACHERS` 表(entry / 解释器 / PYTHONPATH / checkpoint /
   VRAM 门),`servers-down` 清扫锚点随 teacher 切换,`driver-up`/`agents-up` 透传 `--teacher`。
   三条真机结论写进 plan §6d ④:服务克隆无 venv(借主 checkout 解释器 + PYTHONPATH 嫁接)、
   tyro flag 必须在 `policy:checkpoint` 之前且拼写 `--cache_config`、**pi05 池子只能 5 台**
   (6×8 GiB 超 48G 卡,准入门会主动拒第 6 台)⇒ pi05 相位配 **5 个 fleet**。
   已 `tether push --force` 到服务克隆(sha 对齐 `30c332cb`)。

本地回归:`tests/robocasa365 tests/cache` = **1768 passed / 0 failed**,ruff 对改动文件净。
新增 14 条契约测试;`servers-down` 的 teacher 锚点做过变异验证(改回写死 GR00T ⇒ 两条 pi05
用例失败:漏杀自己的 server + 误杀对方的 server)。
**这些改动需要 owner 授权才能 commit。**

---

## 4. 已完成的门与证据(全在 `exp/robocasa365/analysis/`)

| 门 | 结论 | 证据卡 |
|---|---|---|
| S0-a 桶映射 | **PASS** 111 桶 / 0 unresolved / 0 ambiguous;`robocasa_commit be22d659…` + `camera 512×512` 证明 replay env 与正式 eval 同源;代表 prompt 与 object_class 互证(corn↔"the corn") | `ws2_s0a_bucket_variants.txt` |
| S1 冒烟 | **PASS** `complete=2/2`;1071 次推理全 FULL_HIT;8 集全 join 零缺口;桶落点 6/8 精确 | `ws2_s1_smoke.txt` |
| Stage-0 容量 | 6 worker 挤单 server=126 ep/h;**30 worker 摊 6 台=1224 ep/h**;瓶颈是单 server 推理排队不是 worker 数 | `ws2_stage0_capacity.txt` |

**S1 最有价值的观察**:同一 `OpenDrawer` 任务在不同 seed 下 prompt 是 "Open the **left** drawer."
vs "**right**"——语义不同的子目标,正是 text-IVF 的分桶依据。两个 `matched=False` 是**设计中的
最近代表回落**(eval 抽到 "hot dog",库内最近为 "hotdog bun"),不是缺陷,是待测现象。

**首批 12 格科学读数**:榜首 `grid4 v0@12 v1@12 v2@12 rs@62` **macro_sr 0.337**,已超 round-1
全矩阵最好成绩 0.269;`iso_vision_0/1` 垫底(0.096/0.077),与 round-1 "v0/v1 零正贡献" 一致。

---

## 5. 关键设计约束(改动前必读)

- **`--episodes` 是「每任务」不是「每格」**:任务集冻结为 13 个,所以 `--episodes 8` = 每格 104 集,
  `--episodes 32` = 每格 **416** 集。读错这一处会把 ws2e 的规模低估 13 倍(320 vs 4,160 集)。
- **worker 亲和把 fleet 绑死在一台 server**:N 台 server 必须配 N 个 fleet,否则其余 server 空转。
  GR00T 相位 N=6;**pi05 相位 N=5**(VRAM 只够 5 台,见 plan §6d ④)。
- **分批是未经 G1 批准的偏离,有实测依据**:scheduler 每次调用全表扫描,代价随 stage 数平方增长
  (实测 156 stage=3.3ms vs 1716 stage=435ms,全程持锁)。三条守卫:固定 `--bind-port`、
  非末批抑制 `MSG_SHUTDOWN`(否则车队被解散)、任一批失败非零退出且不 finalize。
  `--cells-per-batch 0` 可退回单图。**批 1→2 跃迁已在真机 30 worker 下验证通过。**
- **finalize 只在全 11 批跑完后自动调一次**;中途要看完成格数须手动 `--finalize-only`
  (幂等,可随时跑)。
- **两个 estimand 不得混用**:全 132 格只答联合效应;因子分解仅限 manifest 的 12 个匹配 cell。
  `analyze_ws2_vs_ws1 compare` 默认强制全覆盖,中途观察须显式 `--allow-partial`(输出自带
  NOT A FORMAL RESULT 抬头)。
- ⚠ **所有 agents-up / agents-down 必须显式带 `--tmux-prefix ws2`**:默认前缀是 `ws2s`,
  而实际会话叫 `ws2agent0..5`。漏了就一个 supervisor 都杀不到,只扫 worker,supervisor 立刻
  把它们拉回来——表现为「报 0 tmux session + LEFTOVER,30 秒后 worker 数原样复原」。
  (2026-08-27 又踩一次,见 plan §6b-4。)
- ⚠ **重启 server 必须同时循环车队**,顺序是 `agents-down 全部 → 起 driver → agents-up 全部`。
  只重启 server 会让 worker 握着死连接,每集瞬间失败 ×3 → 批次秒排空(无永久损失但白跑)。
- **车队必须用 `orchestrate_ws_search2 agents-up` 起**:`agents-down` 按 `<prefix>agent<fleet>`
  推导会话名,手写 tmux 名会让它找错会话,supervisor 存活并把刚扫掉的 worker 重新拉起
  (本会话踩过一次)。清扫身份是 `--server-key × --driver-host × --driver-port` 三锚点全中,
  且值经 `ere_literal` 转义(hostname 的 `.` 是正则通配符,不转义会误杀他人 fleet)。

---

## 6. 剩余工作与命令

**完整命令见 plan `logs/robocasa365_ws_search2_text_ivf_plan.log.md` §6c(恢复手册)/ §6d(相位切换)
/ §6e(车队原始参数)。**

**2026-08-27 起,后三个相位的前置条件已全部验完**:ws2c 的 12 个 control yaml 与 manifest 的
ws2c 段一一对应且确为 `brute_force` + 与主臂同库;六条 `agents-up` 的原始参数(错位轮转的
gpu-ids)已从 live 进程 `/proc/<pid>/cmdline` 读出记入 §6e;pi05 的解释器嫁接、tyro flag 顺序、
VRAM 只够 5 台、以及**启动绑定检查**(库 `prompt_pool={masked:T,span:T}` 对上 yaml 旋钮)全部
实测通过。另外 **round-1 的 132 个 journal 已从两台机器归并到单目录**
`/data/openpi_exp_data/robocasa365/ws_search/groot_tp`(即 `--ws1-dir`),并逐条对拍 manifest 的
sha256:`missing=0 / extra=0 / mismatch=0`。**剩下的只是等主臂排空**。

1. **主臂收官**:`--finalize-only` 报 `132/132` 才算完;INCOMPLETE 用 summarize 的 `--only` 串重发射。
   ⚠ **这一步已确定是必需的,不是保险**:2026-08-27 发现批 6–8 在**静默丢集**
   (33 格残缺、约 20 格已停滞),根因是 WS keepalive 掉线 → 重试耗尽 → **不写 journal 行** →
   批次以为跑完就推进。详见 plan §6b-3(含证据与给 owner 的两个处置选项)。
   **别把首次 11 批跑完当作主臂收官。**
2. **ws2c 控制臂**(12 格 ×104):⚠ 是 `brute_force` 指纹,与主臂不同 backend,同进程双份驻留 42G
   ⇒ **必须先全停 server 再重起**指向 `control/` 目录,`--run-prefix ws2c --manifest <manifest>`。
3. **ws2e 加密臂**(10 格 × **416 集** = `--episodes 32` × 13 任务):先
   `build_selection_manifest --segment ws2e` 追加段(append-only,既有段被改会拒;显著劣的
   cell 不足 2 个会 fail-fast),再用 main 的 yaml 跑 `--episodes 32`。共 4,160 集,别当成 320。
4. **pi05 全线**:标定与 **132 个 main yaml** 已就绪(另有 GR00T 的 12 个 control,合计 144 是
   GR00T 那侧的数;pi05 无 control 臂);server 用 `serve_policy.py`(已亲验默认支持动态
   bundle:`WebsocketPolicyServer` 默认 `allow_dynamic_bundles=True`,其
   `_connection_policy_factory` 同样接收 `bundle_id`)。⚠ **pi05 库 29.8G、单 server RSS ~32G,
   五台需 ~160G** ⇒ 必须先停光 GR00T 的 server,两个库塞不进 251G。
   ⚠ **池子是 5 台不是 6**(6×8 GiB 超 48G 卡),因此 pi05 相位配 **5 个 fleet**。

---

## 7. 本会话的运维教训(别再踩)

1. **速率必须用 `/tmp/ws2_sample.sh` 读**,别心算(我错了三次)。短窗口测长尾任务会系统性偏高。
2. **`tmux ls` 只看一眼会误判**:我曾据此报"只起来 1 台 server",实际编排器正在串行起第 2 台。
   判死之前先看进程 + 日志增量。
3. **加 worker 未必提速**:实测 30→42 worker(+40%)只涨 4%(1128→1176 ep/h),瓶颈是单卡服务
   6 个推理栈。要有量级提升得解决"单卡多栈",不是堆 worker。
4. **`grep -E` 的模式以 `-` 开头会被当选项**:清扫模式必须写 `[-]-port` / `[-]-driver-port`。
   本会话在 server 与 agent 两处清扫上各踩一次。
5. **冷启动 server 约 10 分钟/台**(20.5G 库从磁盘真读),page cache 热了之后约 45 秒/台。
   六台串行起,冷态要 30-40 分钟,别以为卡死了。

---

## 8. 持久记忆索引

`reference_weilandserver_4090_unstable`(**已改写为作废**:缺陷卡已拆)、
`reference_robocasa_build_l1s1_census`(库台账)、`reference_weilandserver_disks`、
`feedback_no_deferred_work`、`project_search_tuning_workflow`、
`reference_ws_search_cross_machine_topology`、`feedback_no_unsolicited_git_add`。
