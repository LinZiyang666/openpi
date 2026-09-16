# GR00T × LIBERO 加权和检索搜索 — 方案与口径

> 运行口径唯一来源。姊妹文件：`logs/libero_groot_collection.log.md`（采集与建库）。
> 前身方法学：`exp/weighted_sum`（pi0.5 × LIBERO）。**本实验不沿用 RoboCasa365 的任何量级选择**
> （owner 2026-08-23 令：不要被那条线污染视线），只沿用其已验证的**工程件**。

## 1. 问题

固定 GR00T N1.5 教师与固定库，扫加权和检索的权重空间，看**检索质量**能把纯 cache 的成功率推到哪里。
`always_search + always_hit + top_k=1 + write_policy never` —— 无教师回落，成功率是检索结果的纯函数。
**只做 d=1**（单步检索，不进轨迹链，owner 令）。

## 2. 打分口径（两层，转写自 pi0.5 线）

- **Layer-1**：每字段把原始相似度（vision=cosine、robot_state=l2）经单调有界 normalizer 映到 [0,1]。
  **不搜**，由 `exp/common/calibrate_score_normalizers.py` 离线定：**LOEO**（每条 entry 当 query，
  只对**不同 `trajectory_id`** 的库内条目打分，剔除自匹配与同链近重复，镜像"在线 episode 不在库里"），
  按 `J = mag_sep + β·intra_spread − λ·sat` 排序取第一。
- **Layer-2**：对归一化分数做加权和，**权重是唯一自变量**。

**实测标定结果**（`--max-queries 300`，三字段在两个 suite、两种 builder 上**全部选中 `zscore(tanh)`**）：

| suite / builder | vision_0 J | vision_1 J | robot_state J |
|---|---|---|---|
| spatial / mean_pool | 0.3541 | 0.2535 | 0.2421 |
| spatial / sp16 | **0.3962** | 0.3251 | 0.2463 |
| libero_10 / mean_pool | — | 0.4017 | 0.3626 |

J 只是诊断量，最终由 Phase-2 任务成功率裁决。

## 3. 搜索空间

**加权字段只有 3 个**：`vision_0`（agentview）、`vision_1`（wrist）、`robot_state`。
`prompt_emb` 权重恒 0 且不入扫描轴（任务内恒定；owner 令）；LIBERO 无 `vision_2`。

**闭单纯形**（含角点与棱），不用命名家族拼接 —— 角点=单模态独占、棱=两模态，都是网格点。
把边界留在网格里是有代价的选择：如果最优解落在棱上（某个模态零贡献），只扫内部会整个错过。

| 步长 | cells |
|---|---|
| 1/6 | 28 |
| 1/8 | 45 |
| 1/16 | 153 |

## 4. 评测集与防泄漏

**评测集 = A 池 `pruned_init`，每 cell 跑满 500 集**（10 任务 × 50 init；owner 令）。

防泄漏是结构性成立的，不需要 held-out 计算：**库 100% 来自 B 池差集，A 池与 B 池按构造互斥**。
（pi0.5 线需要 `init_holdout.py` 是因为它的库建自 B 池的一个小子集、评测也在 B 池内；
我们的 S6 库用掉了每任务约 45.6/50 的 B 池 init，B 池内已无足量无泄漏切片，A 池是唯一选择。）

⚠ 代价须写在最终报告里：A 池同时是本线的测试集，**权重是在它上面选的**，
所以最终数字带选择偏差；跨臂比较（cache vs 教师）仍然有效，因为教师侧不参与选择。

## 5. 两轮预算（每 suite ≈ 26,000 集）

| 轮次 | cells | 集/cell | 小计 | 说明 |
|---|---|---|---|---|
| 粗扫 | **28**（步长 1/6） | 500 | 14,000 | 3 角点 + 每棱 5 点 + 10 内部点，"各种权重大概都来一次" |
| 细扫 | **24**（固定数量，位置由粗扫定） | 500 | 12,000 | 最优区域上铺步长 1/24 局部网格 |

细扫的**数量提前钉死**、只有位置由数据决定 —— 否则预算随中途读数漂移。
500 配对集（同 task × 同 init）下，配对检验可分辨约 3–4 个百分点。
顺序：spatial 先，l10 随后。

## 6. 工程件：复用 vs 新建

**复用（owner 早前为 RoboCasa365 做的 GR00T 适配，已验证）**
- `serve_groot_n15.py --concurrent` 的 `_build_concurrent_factory` → 移植进 `serve_groot_libero.py`：
  共享 GPU policy（推理锁串行 infer）+ 只读 storage，其余每连接重建。
- 配套的 `_require_default_bundle`：GR00T 侧 `allow_dynamic_bundles=False`，配置身份由**进程**承载，
  调度器每 cell 重起 server；而 conductor 的 `LiberoEpisodeRunner._ensure_client` **每集都调**
  `select_bundle`，`select_bundle("default")` 在该模式下恰是幂等放行
  （`websocket_policy_server.py:462-465`）。**这两块是配套的，不能只抄一半。**
- `exp.weighted_sum.emit_yamls.build_eval_config` 逐字复用，recipe 形状与 pi0.5 线对齐。

**新建**
- `exp/libero_groot/emit_search_yamls.py` — 闭单纯形矩阵生成。
- 补 `examples/libero/worker_entry.py` 的 `--resize-size` / `--replan-steps`：原来固定用
  `m.Args()` 默认 224，GR00T 会双重裁剪（契约层会硬报错，但那时车队已经起来了）。

**顺手补上的两道守卫**（cache 路径原先没有）
- `validate_groot_cache_config`：拦的全是**静默失效** —— 不可满足的 WARM_START 被悄悄降级成 MISS
  （GR00T 库永远没有 intermediates），cp3 建了从不查询，非 `always_search` 的 gate 改变
  `searched` 的下游含义。
- `validate_artifact_identity`：`load_artifact` 只比 `vector_dims`，mean_pool 与 max_pool 维度完全相同，
  掉包无人会发现。

## 7. 踩过的坑

- **`__hit_meta__` 被当成非法 action key**：cache 拦截器把逐步命中元数据挂在 infer 结果上，
  而 adapter 的契约"拒绝一切未知 action key"。**采集路径没有这个键，所以直到第一次跑 cache 才暴露。**
  修法与 RoboCasa 版一致：dunder 前缀的 side-channel 在校验**之前**摘出、放在 `actions` 旁边，
  校验继续拒绝真正意外的 *action* 键。已加两条测试（透传 + 非 dunder 未知键仍拒）。

## 8. 容量实测与待办（2026-08-23 17:55）

**6 个 server 在一张 4090 上线性扩展**（从进度条直接算的每 server 吞吐）：

| | 每 worker 单集 | worker/server | 每 server 吞吐 |
|---|---|---|---|
| 1 server（试运行） | 59 s | 16 | 0.271 集/秒 |
| 6 server | 19.9–31.7 s | 8 | **0.310 集/秒** |

多 5 个 server 共享同卡，单 server 吞吐不降反升 ⇒ **`nvidia-smi` 报的 91% 利用率是 launch-bound 的假象**
（`reference_vla_latency_launch_bound`：batch=1 的 VLA 推理 6–15% util 即已打满延迟），这张卡远未到算力饱和。

**当前只用了单 server 天花板的 68%**：天花板 = 1/0.088 s ≈ 11.4 次推理/秒 ÷ 约 25 次/集 = 0.456 集/秒；
实测 0.310。缺的是 client 数 —— 每 worker 产出 0.0388 集/秒，顶满需 **≈12 个/slot**，现用 8 个。

> **待办（owner 裁定 2026-08-23：本轮不动，粗扫→细扫交界再改）**：`--workers 8` → **12**。
> 免费 +45%；timan107 成本 72 worker × 2.2 GB = 158/220 GB、load ~31/48，均在安全区。
> 交界处改零浪费；中途改要重起调度器、作废在飞 cell 的进度，净收益不抵风险。

**若加第二张 48G 4090**：server 侧大致可翻倍，但瓶颈立刻搬到仿真车队 ——
喂饱两张卡需 ≈144 个 sim worker = 317 GB RAM / ~62 核，**超出 timan107 的 220 GB / 48 核**。
网络不是墙（实测 rx 184 Mbps，翻倍约 370 Mbps，本机网卡 1 Gbps）；weilandserver CPU 也不是（12 server × 2.7 = 32/88 核）。
⇒ 第二张卡必须配第二个仿真宿主，timan1 合适（4×A6000、EGL 原生、GR00T 岛已验收；按 owner 铁律单卡优先打满不平摊）。

## 9. 无人值守夜跑的三次失效（2026-08-23 夜 → 08-24）

### 9.1 broker 传输并发上限吞掉已算完的 cell

`tether pull` 被拒 `too_many_in_flight`：6 个 slot 在同一分钟内完成并同时发起文件传输，broker 的并发上限把大部分拒了。
日志里 `merge='MERGED 500'` 证明 500 行已在远端合并好，**只是传不回来** —— 三个跑满 27 分钟的 cell 因此判失败。

**修**：`_PULL_LOCK` 全局锁串行化传输 + 6 次线性退避；失败时日志明写"数据已在远端 merged，可捞回不必重跑"，
并配 `recover_pulls.sh`（逐个 pull、5 次退避、校验 500 行才留）。
**教训**：几百 KB 的传输排队成本可以忽略，但它保护的是 27 分钟的算力。传输层的并发限制要当成**共享资源**对待，不是无限的。

### 9.2 链把"没在跑"当成"已完成"

重启链时 spatial 细扫并未在运行，脚本的 `if tmux has-session -t libsearch` 为假 ⇒ 整段跳过 ⇒
**直接分析只有 6/26 的结果并进入 l10**。我把"接管一个正在跑的阶段"写成了唯一路径。
**修**：else 分支查完成数，不足就 `run_stage` 启动它。

### 9.3 ⚠ 跨阶段僵尸 worker：同名 session 让孤儿判据完全失效

§9.2 那次错误启动 l10 时拉起了 l10 worker。我随后 reap 了 58 个，但**其中 28 个是在 reap 之后才被
`launch_clients` 拉起来的**，逃过清理，并且打的是同一批端口 23160-23165。

级联后果：
1. 每台 server 的推理锁上排两倍的队 ⇒ `infer_ms_per_call` **88 ms → 2325 ms**（26 倍）
2. cell 从 27 分钟拖到 **62–81 分钟**
3. 最要紧的：`main.py` 的 `client.episode_start()` 在 try 块**之外**，连接异常会让**整条 lane 崩掉**
   而不是只失败一集 ⇒ 那条 lane 一行不写 ⇒ cell 回来 459 行（正好差一条 lane 的 41 集）⇒ 判 partial。
   **5 个 partial 全部源于此，不是 5 个独立故障。**

**为什么孤儿判据查不出**：l10 worker 的 tmux session 命名与 spatial 完全相同（`lw<port>_<n>`），
"进程数 = session 数 × 2" 这个比例在两边都成立。**唯一有效的判据是按 `--task-suite-name` 分类计数**
—— 现役阶段只该出现一个 suite。已写进巡检指令。

**清除方式**：读 `/proc/<pid>/cmdline` 按 suite 判定、逐 PID `kill -TERM`。不用 pkill（自匹配踩过三次），
且同一条 argv 里不能出现被匹配串的明文（连中文注释里写 `main.py` 都会自匹配，踩过一次）。

### 9.4 告警设计的通用教训：绑变化，不绑状态

同一类错误在这一夜犯了五次（可用内存 45G 绝对阈值、worker <75%、单周期跌 12G、worker 连续两拍、
partial 持续存在）。根子是把"**当前处于某状态**"当成了"**刚刚出了事**"。

- cell 边界会让 worker 数、可用内存、吞吐三条线同时跳崖 ⇒ 它们是**相位信号**，不是故障信号
- partial cell 会一直存在到阶段末补跑 ⇒ 按存在报警等于每拍复报一件已处置的事，真告警会淹没在噪声里
- 稳态就在阈值附近时（72 worker 稳态可用 43G，阈值 45G），告警永远为真 = 等于没有告警

**现行四条判据全为转移量**：OOM 计数增量、孤儿数不匹配、partial 数**增加**、75 分钟无 cell 完成。

## 10. 四阶段终表（全链 2026-08-23 19:19 → 08-24 10:05，46,000 集）

92 个 cell × 500 集 A 池，全部 rows=500，**四阶段 partial 均为 0**。

| 阶段 | cells | 榜首 | 打平集合 | 结论 |
|---|---|---|---|---|
| spatial 粗扫 | 28 | `3/2/1` **0.760** | 18/27 | 好区域=三模态都参与 |
| spatial 细扫 | 26 | `2/7/3` **0.766** | **25/25** | 区域内部完全平坦 |
| libero_10 粗扫 | 28 | `4/1/1` **0.454** | 6/27 | 好区域更窄、可分辨 |
| libero_10 细扫 | 10 | `6/5/1` **0.454** | 6/9 | 未超越粗扫榜首 |

### 结论 1：权重调优的收益上限约 10 个百分点，且大半在第一格拿到

- spatial：最差 0.528（v0 独占）→ 最好 0.766，**但 0.75–0.766 之间 25 个 cell 全部统计打平**
- l10：最差 0.180（v0 独占）→ 最好 0.454，打平集合 6/27（粗）、6/9（细）

**细扫在两个 suite 上都没跑出比粗扫更高的成绩**（spatial 0.760→0.766 打平带内；l10 0.454→0.454 持平）。
粗扫已经把可得的收益全部拿到，加密只是确认了区域的平坦性 —— 这本身是有价值的负结果：
**不必为权重的第三位小数付出算力**。

### 结论 2：两个 suite 的最优权重结构不同，且可解释

| | spatial | libero_10 |
|---|---|---|
| 边际峰位（粗扫单位） | (2, 2, 1) 三模态均衡 | **(4, 5, 1) 双视觉重权** |
| `robot_state` 峰 | 1/6–2/6 后单调降 | 1/6 后单调降（更陡） |
| `vision_0` 独占 | 0.528 | **0.180** |

l10 是长程两段式任务（"把两个 moka 壶都放到炉子上"），**本体位姿在长轨迹里区分度低** ——
同一个手臂姿势会出现在任务的不同阶段，按本体检索会取回相位错误的轨迹。必须靠视觉判断"现在做到哪一步"。
spatial 十个任务都是单段"抓黑碗放盘子"，本体的相位信息就够用。

### 结论 3：库规模对长程任务的要求高得多

同为 5 条/任务的 S3 库：spatial 纯 cache 0.75–0.77（教师 0.912，达成率 **82–84%**），
l10 只有 0.42–0.45（教师 0.854，达成率 **49–53%**）。教师侧只差 6 个百分点，cache 侧差了 30 个。

⚠ **A 池同时是本线测试集，权重是在它上面选的**，最终数字带选择偏差。跨臂比较（cache vs 教师）仍有效，
因为教师侧不参与选择。
