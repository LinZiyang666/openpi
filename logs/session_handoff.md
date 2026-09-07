# Session Handoff —— WARM_START 引入 RoboCasa365（开发线）

> 2026-09-06 晚。**上一条实验线已收官并拆干净，不要再碰它**（见 §5）。
> 当前唯一活跃的工作是 **warm-start 开发线**，卡在 **G2 Round 4 待审**（R3 执行方答复已追加）。
> 范围已由 owner 扩到 **RoboCasa365 + LIBERO 两条 GR00T 线**。

---

## 0. 接手第一步

```bash
cd /home/weiland/projects/openpi
cat logs/robocasa365_warmstart_plan.log.md        # 权威 plan，含 G2 Review Log
uv run pytest tests/robocasa365/test_bench_groot_stages.py -q     # 应 32 passed
uv run pytest tests/review_tests/test_warmstart_w1_g2.py -q       # 应 12 passed（可跑不可读）
git status --short
```

**Authority = Execution。** 只读 `protocols/execution_authority.md`，不得读 review 那本。
G2 若返回 NEEDS REVISION：**先重读 §10.2**，每条 reviewer 意见必须恰好一条
`Accepted`/`Rejected` 答复，沉默或无理由驳回都是违规；Review Log 追加式。

---

## 1. 当前状态

```
Understand ✅ → Plan ✅ → G1 ✅ APPROVED(3轮) → §3.1 polish ✅(Review Log 已整节删)
→ Code ✅(W1) → G2 🔄 Round 3 执行方答复已追加，等 Round 4
```

**G2 R1（8 blocking + 1 nb）与 R2（5 blocking + 2 nb）全部 Accepted 并已落到代码。**
R2 那批修复由 Review Authority 会话（codex）直接写进工作区、未暂存；它**不能**为自己改的东西
签 APPROVED（WA §9.3），所以我作为 Execution Authority 逐条复核并接管，写在 R3 答复里。
复核实测：`tests/robocasa365/test_bench_groot_stages.py` **43 passed**；ruff check + format 全过；
`tests/review_tests/test_warmstart_w1_g2.py` **12 passed**（可跑不可读）。

R2 那轮的实质修复（记住这几条，别再退回去）：三段全部 `fullgraph=True`；stage1 改为
benchmark 侧等价体 + `index_copy`（生产 eager 那次调用独占所有 `Tensor.item()` 数据依赖 guard），
并新增 `RUN_STAGE1_SRC_SHA256` 漂移钉；stage3 改成**单步编译 + Python 外循环**（每步 `.clone()`）；
`compile_count` 读 Dynamo `stats.unique_graphs` 精确计数且必须恰好 +3；measurement 区间由
`cudaProfilerStart/Stop` 界定并压 NVTX cell marker，certify 时要求 `cudaGraphLaunch` **恰等于**
`iters·(2+k)` 且 capture 计数为 0；每段 parity 都吃**同一份 eager 上游**，另加 end-to-end 链路指标。

⚠ **提交发生在 G2 APPROVED 之前，这是 owner 的越门指示，不是流程走完了。**
owner 2026-09-06 明示「检查审核之后推进流程…前进到 commit push」，据 WA 抬头「Project Owner…
May override any process at will」执行。⇒ **G2 仍未 APPROVED**，Round 4 的裁决照旧要等；
若返回 NEEDS REVISION，**先重读 `protocols/execution_authority.md` §10.2**，每条意见恰好一条
`Accepted`/`Rejected`，Review Log 追加式。**不要**因为已经 commit/push 就当 G2 结了。

**唯一未完成项**：真模型 parity/trace 仍没跑。GPU 窗口现在是开的（两机全空），但
① 它是 §6 执行顺序里 W0→stage1 诊断→W2 三步，计划明确要 owner 起跑；
② `assert_host()` 钉死 weilandserver，**本机（WSL `Weiland`，3060 6G）会被脚本正确拒绝**。

---

## 2. 这条线在做什么（一句话 + 三个硬事实）

让 WARM_START（缓存命中后从中途去噪状态续跑）在 RoboCasa365 上对两个 teacher 都能用，
覆盖采集 → 建库 → 标定 → 正式实验。**但先测收益再动生产代码。**

| 事实 | 影响 |
|---|---|
| **pi0.5 侧 WARM_START 是现成能力**（517 个 `always_warm_start` yaml 在用） | 只需 emit 段换 judge |
| **GR00T 侧功能根本不存在**，不是"配置没开" | 四处同缺：采集写空 `noise_action_steps=[]`；`run_stage2` 里 `get_action` 是原子调用、无 partial 执行体；时间轴与 pi0.5 **方向相反**；`load_guard` 硬拒 warm judge |
| **时间方向相反** | pi0.5 t:1→0 十步；GR00T t:0→1 **四步**，可恢复点只有 0.25/0.5/0.75，剩余步数 **3/2/1**（照抄 pi0.5 的 `floor(start_t·N+0.5)` 会算反） |

**⚠ LIBERO×GR00T 是同一套代码，不是第二份实现**（owner 2026-09-06 扩范围，plan §1.8 / §3.2-W14）：
`cache/groot/{staged,interceptor,load_guard}.py` 两线共用，连采集器都共用 ——
LIBERO 的 server 在 `exp/libero_groot/serve_groot_libero.py:400-409` **直接构造 RoboCasa 的
`GrootCacheCollector`**，所以那行 `noise_action_steps=[]` 同时决定了两条线。
阶段 B 做一次两边都得到，**没有可以单独补给 LIBERO 的实现**。但有三处差异必须单独处理：

| # | 差异 | 位置 |
|---|---|---|
| L-1 | ⚠ **LIBERO 跑 8 步，不是 4 步** ⇒ schedule 主键必须带步数：`groot_n15_k4_v1` / `groot_n15_k8_v1`（原先冻结的 `groot_n15_v1` **已作废**） | `serve_groot_libero.py:60,380` |
| L-2 | **第二道守卫**：emitter 也拒 warm tier，必须与 `load_guard` 同时最后放开 | `emit_gate_yamls.py:177-182` |
| L-3 | 三个装配点带 `allow_hysteresis_gate=True` 既有豁免，放宽时别冲掉 | `serve_groot_libero.py:160,235,429` |

k=8 的 timesteps `{0.125…0.875}` 与 `CANONICAL_DENOISE_TIMESTEPS` **交集为空** ⇒ LIBERO 侧
没有 k=4 那个"碰巧全合法"的静默坑，失败是响亮的。
⚠ **术语陷阱**：`exp/libero_groot/` 里的 "warmup" 指**阈值标定的 force-MISS 臂**，
与 warm-**start**（中途续跑）毫无关系，只是共用一个英文词，且在同一条流水线里相邻出现。

**⚠ 最危险的静默失败**：GR00T 四步中间量被现有 builder 打成 t=0.9/0.8/0.7，
而这三个值**恰好都是合法 canonical 值** ⇒ config 与 orchestrator 双双放行、每步都返回
WARM_START 但拿的是错的 x_t；而真值 0.25/0.5/0.75 写进 yaml **反而被拒**。
**错的配置能跑通，对的配置被拒绝**，症状只有动作质量下降。

---

## 3. plan 里已冻结、不要重新讨论的裁定

- **D1 按精度立论，不按延迟。** warm-start 的结构上限是 `s1+s2` 地板（pi0.5 CUDA-Graph 档
  最多省 39.4%、永远降不到 FULL_HIT 的 15.2%），折到 episode 墙钟不足 4%；
  而 pi0.5/LIBERO 实测 warm-start 收回 FULL_HIT→teacher gap 的 **87-95%**。
- **D3 owner 三条硬约束**：① 只做 CUDA Graph（`reduce-overhead`），不写 eager/default 退路；
  ② **三个 stage 必须各自编译成独立的图**；③ **机器固定 weilandserver，且测量期间 GPU 必须完全空闲**
  （遇到他人占用只能等，不得驱逐）。
- **D4** 引入 `DenoiseSchedule` 身份对象（`schedule_id`/`num_steps`/**`direction` 显式字段**）
  贯穿 config、h5 attrs、artifact meta、`CachePayload`、装配期绑定五面。旧产物默认 `pi05_v1`。
  ⚠ 只按 timestep 值集合校验**挡不住方向反转**（`{0.1…0.9}` 在 t→1−t 下不变）。
- **D5** 起点噪声单开 `noise_action_0`，**只作基线复现、不进 `CachePayload`**；
  实现走"加字段"不动 `enumerate(start=1)`/`range(1,N)` 那对配套偏移。
- **D6** GR00T 采集走 `action_head.action_encoder` 的 forward hook，**不复刻循环**
  （`UPSTREAM_FORWARD_SHA256` 只钉 eagle forward，action head 无漂移守卫）。
- **D8 ⛔ 绝不就地改 `emit_ws_search_yamls.py` / `emit_ws_search2_yamls.py`**：
  `index_digest.json` 的 `source_sha256` 冻结的是**这两个源文件本身的 sha**，改一个字节会让
  待续跑的 `run_ws_search2.py` 在 preflight 直接 SystemExit。warm-start 的 emit 走新文件新 digest。
- **D9 放开 `load_guard.py` 白名单必须是最后一步** —— 它是目前唯一挡在静默坑前的**响亮**防线。

---

## 4. 下一步：阶段 A（零生产代码改动）

**待 owner 一句话裁决**：先按 §6 顺序跑阶段 A（W0→stage1 诊断→W2→W3→W4），
还是直接解除 D2 授权阶段 B（含 LIBERO 的 W14）并为其单独走一次 G1。
⚠ 我没写任何 W5-W14 生产代码，三条理由写在 plan 的 G2 Round 3 范围扩展条目里
（D2 未解除 / L3 实现细节未过 G1 / 未验证的 L3 改动不该随本次 push 上去）。
⚠ **W13 重采集义务翻倍**（RoboCasa 语料 + LIBERO 约 89 GB），W4 计价时要算进去。

**两机现在全空，GPU 窗口已开。** 顺序不可交换：

| # | 动作 | 备注 |
|---|---|---|
| 0 | W0 空窗确认 | 可停清单已冻结在 plan §3.0.1；**清单外的占用一律不动** |
| 1 | **stage1 等价门诊断** | `--mode diagnose-stage1`，`reduce-overhead` 与 `default` 并排 |
| 2 | W1 脚本已就绪 | `exp/robocasa365/bench_groot_stages.py` |
| 3 | W2 跑 G-M | k∈{1..4} × 3 进程 + pi0.5 同卡复标定；约 45 min |
| 4 | 判 G-M/G-T/G-S | 拟 `s2act(k)=a+b·k`，真实上限是 `3b/T` |
| 5 | W3 跑 G-A1 步数敏感筛查 | 约 1.2 h |
| 6 | **owner 唯一裁决点**：是否进阶段 B（L3 改造，含 LIBERO 的 W14） | |

**⚠ 已查明**：stage1 等价门那次 `cos=0.8716` 的 FAIL **与 CUDA Graph 无关** ——
`cudagraph_trees` 第一次调用走 warmup 不重放，而那道门只在第一次真实推理跑
⇒ **CUDA Graph 从未被真正试过**，退回 `default` 档也救不了它。根因二选一：
autocast dtype 处置（仓内实测同族 LayerNorm fp32/bf16 差 max|Δ|≈1.4e-2），或判据过脆
（min-cos 对低范数 token 极敏感）。**第 1 步就是判这个。**

**⚠ 真正的新问题不在编译，在 §7.2**：stage2/stage3 吃 `[1,N,2048]`，N 随 prompt 变，
CUDA Graph 要静态形状；每形状 247 MB lm_head 静态缓冲；左填充会稀释 `prompt_emb`
**改变所有 cache key**（唯一有"库不可比"风险的改动）。pi0.5 的先例在这条上不适用。

---

## 5. 上一条实验线：已收官，不要重启

pnp 定物体 GR00T 全线跑完并已**拆干净**（两机 GPU 全空、231xx 零监听、无 tmux）。

- cache 臂 `DONE complete=132/132`，5,280/5,280，`n_err=0`；teacher 基线臂 250/250，`macro_sr=0.752`
- 报告已写：`exp/robocasa365/analysis/pnp_pinned_results.md`
- 数据已归位本地 `exp/robocasa365/data/`（sha256 两端一致）
- **pi0.5 不跑**（owner 裁定）
- 结论摘要：teacher 75.2% vs cache 4.58%（16×）；**132 格权重搜索无信号**
  （实测最优 0.1250 < 纯噪声期望最大值 0.1502）；钉死验证成功但没救回 PickPlace 族

---

## 6. 提交状态

owner 2026-09-06 授权 commit + push，已提交（Ziyang 分支）：

```
exp/robocasa365/bench_groot_stages.py          W1 脚本
tests/robocasa365/test_bench_groot_stages.py   43 项测试
logs/robocasa365_warmstart_plan.log.md         plan（含 G2 R1-R3 Review Log）
logs/README.md                                 索引已同步（§4 章程红线）
exp/robocasa365/run_collect.py                 sys.path bootstrap（旧线遗留修复）
logs/session_handoff.md                        本文件
```

**仍未提交**（上一条实验线的产物，owner 未就它们下过指示，要保留就单开一次提交，
其中 `logs/pnp_run_progress.md` 进库前须按 §4 先补 `logs/README.md` 条目）：

```
?? exp/robocasa365/analysis/pnp_pinned_results.md   W10 报告
?? exp/robocasa365/config/ws_search2_pnp/           265 份 yaml + digest
?? exp/robocasa365/config/{calibration_normalizers_pnp_pinned.json,collect_pnp_*.env}
?? logs/pnp_run_progress.md
```

⚠ `exp/rit_pareto/*` 是**另一个 session** 的，而且它**已经被那边放进 index 了**
（`AM exp/rit_pareto/edit_figure.py`）。本次提交按路径逐一指定，**没有**扫进去 ——
以后在这个仓提交也别用裸 `git commit -a` / `git add .`。

---

## 7. 纪律（本线踩过的坑）

- **不 `git add`**，owner 明确指示才暂存；commit message 只用英文、**不加 AI 署名**。
- **共享机禁宽模式 `pkill`**，按 PID / 端口 / tmux 名定点；weilandserver 的 23100-23199 与
  `srvN` 是**多 session 共享命名空间**。
- **`tests/review_tests/` 可跑不可读**，失败 id 交 Review Authority。
- **声称"已改"必须回读确认** —— 本线在 G1 答复里两次写了"已完成"而正文没落。
- **删除类操作前先 `wc -l` 核对清单规模**，`tail` 截断的输出不是全集（491 个文件的教训）。
- `tether exec` 单次约 10 min 上限，长跑一律进 tmux；`push`/`pull` 目标已存在即拒，要 `--force`，
  **别把 stderr 重定向掉**；`/scratch` 不在 timan107 的 allow_roots，须经 `/tmp` 中转。
