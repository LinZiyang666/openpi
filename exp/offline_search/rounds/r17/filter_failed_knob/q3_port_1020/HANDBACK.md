# Q3 停滞门移植进 R15/R16：交回（10-20）

按 MAP.md「R16/R15 判定器换成 Q3」实现：fit 里只换 `R10Recipe.inner`，旋钮层原样。只新增文件、只复制新树；没有原地改主仓、`openpi_r16_k1ref_server`、`r16_local_island` 里任何文件；没起 GPU 进程或实验；tether 只用了 `exec`（读 H100 上的 md5 和 plugin.py、看进程）。没 commit。
路径默认相对 `/home/weiland/projects/openpi`。

## 1. 改动文件清单

**新文件（主仓）**
- `exp/offline_search/rounds/r06/ideation_Q1/method_c/stall_only.py`：`Q3StallInner(base, C, stall_model)`。
  - 每次新查询（段边界）先 `base.query`，再 `StallTracker.observe(q, step*5)`。时钟同 R6-C，原始 QueryView 由 `StallModel.encode` 投影。
  - `slow_confirmed` 且这个锚点不在冷却期 → 写 `os_force_miss=1`、`os_reason=63`。冷却只跟在停滞触发的调用后面，冷一个锚点；彩票为 0，这些都调 `stall_bridge.call_probability/starts_cooldown`。
  - `slow_ambiguous` 不插段中 LOOK（协调者决定 1）。
  - 其余部分：
    - `blind_step` 直接用 base 的，旧 noprog 盲尾否决就没了；
    - `policy_tail_step` 用 CommitJudge lifecycle 门面，照 methods.py；
    - 每次查询都记 `_policy_gate_anchor`，这样旋钮触发的 MISS 也有策略尾；
    - 同一步重复查询时返回备忘结果；
    - 其他属性一律转给 base；
    - 每连接克隆时 Q3 模型共享，不深拷贝。
  - 诊断字段写 `os_q3_*`（state/call/cooldown/anchor/delta_hat/e90/a10/W/…）。
- `exp/offline_search/rounds/r17/filter_failed_knob/q3_port_1020/`
  - `fit.py`：拟合 16 份 Q3，再用 Q3 掩码重解各档周期（第 2 节）。
  - `build.py`：建 fit 和批次定义，写 `QUEUE.json` 和 commands 脚本；`levels`（等旋钮关结果）；`prepare` / `dispatch` 入口。
  - `runner.py`：包装 R16 conductor_runner，把 `SERVER_TREE` 换成新树，GR00T 每副本启动预算 7200 MB。
  - `legacy.py`：包装 R15 legacy，本机（4090）服务端用新树（`local_server.TREE`）。
  - `analysis.py`：旋钮关和纯推理对比、选格，复用 R15/R16 原装载器，去掉 judge 名检查（见第 5 节）。
  - `smoke.py`：最小冒烟。
  - `QUEUE.json`：新批次索引。
- `exp/offline_search/rounds/r17/filter_failed_knob/commands/`，新增 92 个脚本：
  - R15 四个单位，单位级加 8 格：`{prepare,launch}_fa_{pi,gr}_f_{u,f}_o_q[_l50…_s500].sh`；
  - R16：`{prepare,launch}_fa_{pi,gr}_r_{u,f}_o_q.sh`、`{prepare,launch}_fa_cs_{pi,gr}_rg.sh`；
  - 生成各档：`build_levels_fa_{pi,gr}_{f,r}_{u,f}_l_q.sh`（8 个）。

**新代码树** `/home/weiland/projects/openpi_r16_q3_server`
- 由 `openpi_r16_k1ref_server` 用 rsync 只读复制：343 MB 纯代码，不含 .git/pkl/npz/npy/pt/jsonl/log；无硬链接；`.venv` 是指向主仓 .venv 的软链。
- 只改了两处：
  - 新增 `exp/offline_search/rounds/r06/ideation_Q1/method_c/stall_only.py`（与主仓同一份）；
  - `exp/offline_search/closed_loop/plugin.py` 加 `stall_only`，共 3 处。和主仓 plugin.py 的 diff：
    ```
    77:  JUDGE_MODES = (..., "periodic", "stall_only")
    259: if mode in ("always", "guard_only", "stall_only"):
    1028+: elif J.mode == "stall_only": hit, why = True, "stall_only"   # 有 forced 时在上面已判 MISS
    ```
    `guard_only` 照旧 SystemExit。
- R16 的四个旋钮关 group、带门两组的常驻服务端，以及 R15 GR00T 全步的本机服务端，都从这棵树起。
- 这棵树和主仓在 GR00T 服务端代码上的差异，只有 groot batcher/staged（09-23 版）和 msgpack（不压缩）。B=1 时行为等价：staged 只差 B>1 的拒绝，batcher 不启用，zlib 默认关。

**产物**（新写入都在 /nvme）
- 根目录 `/nvme/weiland/openpi/r17_q3_port_1020/`，并加了软链 `/home/weiland/projects/openpi_ext/r17_q3_port_1020` 指向它。
  - 加软链的原因：H100 同步只会镜像 `/home/weiland` 和 `/data` 下的输入，`/nvme` 路径会在 `relative_to(base)` 处失败。所以 fit 路径一律走这个 /home 别名。
- `q3/<cell>/`：`stall.pkl/json/loeo.csv` 加 `RECEIPT.json`（每格收据）。
- `Q3_FITS.json`：汇总；`PERIODS.json`：新周期。
- `fits/{original,filtered}/<cell>_off.pkl`：全步和减步共用。
- `fits/cs/<cell>_{c,cr}.pkl`：cg/chg 共用 c，crg/chrg 共用 cr。共 64 份，4.9 GB。
- `definitions/<group>/<arm>/`：R16 的 config.yaml、eval500.json、ROOT.json。
- `h100_push/`：要推给 H100 的两份文件（第 4 节）。`h100_orig/plugin.py` 是从 H100 读回的原件。
- `logs/`。
- 批次目录在 `/data/mw_finetune/eval/` 下，只写小文件：32 个 R15 单格根 `fa_*_f_*_o_q_<格>`，6 个 R16 group 根。

## 2. 拟合结果摘要

**Q3 拟合**
- 每 (模型, suite, 库大小) 一份。拟合用 `StallModel.fit(选定子集行, metric=cs 研究 _cr fit 的 base, commit_controls=10)`。
- 原库和成功过滤库共用，全步和减步共用。
- 拟合加掩码一共不到 1 分钟（16 进程）。

**触发率的口径**
- 在库的全部选定局上回放：只取偶数步锚点，用留一 Q3 状态，按"确认慢才调、冷一个锚点"统计强制调用。
- 成功局上的触发率接近 0；集中在失败局，这是预期。

**无效任务**（参考局少于 4 个，永不触发）
- 和 MAP 一致：π0.5 l10_50 任务 9、π0.5 spatial_50 任务 0、GR00T l10_50 任务 6 和 8、GR00T spatial_50 任务 0。

| 格 | Q3 触发率（每锚点） | 失败局/成功局触发率 | 参考局数（任务0–9） | 无效任务 | 旋钮关预测 IR 旧→新 |
|---|---:|---:|---|---|---:|
| pi05_l10_50 | 0.0298 | 0.141 / 0.000 | 5/5/4/5/4/4/5/5/4/3 | 9 | 0.187→0.089 |
| pi05_l10_100 | 0.0320 | 0.122 / 0.010 | 10/10/9/10/9/8/10/10/6/7 | — | 0.164→0.090 |
| pi05_l10_200 | 0.0546 | 0.186 / 0.027 | 20/20/19/19/19/18/18/20/12/16 | — | 0.158→0.100 |
| pi05_l10_500 | 0.0816 | 0.234 / 0.037 | 49/50/46/43/46/42/44/50/25/41 | — | 0.164→0.111 |
| pi05_spatial_50 | 0.0135 | 0.091 / 0.000 | 2/5/5/5/5/5/5/5/4/5 | 0 | 0.142→0.083 |
| pi05_spatial_100 | 0.0465 | 0.311 / 0.013 | 7/10/10/10/10/9/10/10/9/9 | — | 0.123→0.098 |
| pi05_spatial_200 | 0.0362 | 0.318 / 0.016 | 16/20/20/20/20/19/20/20/19/19 | — | 0.108→0.093 |
| pi05_spatial_500 | 0.0322 | 0.297 / 0.018 | 45/50/50/50/49/49/50/50/45/49 | — | 0.097→0.092 |
| groot_l10_50 | 0.0013 | 0.006 / 0.000 | 5/5/5/5/5/5/3/5/2/4 | 6,8 | 0.199→0.075 |
| groot_l10_100 | 0.0281 | 0.075 / 0.018 | 10/10/9/10/10/9/6/10/7/9 | — | 0.191→0.087 |
| groot_l10_200 | 0.0674 | 0.216 / 0.025 | 20/19/17/19/19/19/16/17/11/18 | — | 0.191→0.103 |
| groot_l10_500 | 0.0837 | 0.220 / 0.037 | 43/48/42/48/45/46/39/42/29/45 | — | 0.192→0.110 |
| groot_spatial_50 | 0.0163 | 0.091 / 0.000 | 2/4/5/5/5/5/5/5/4/5 | 0 | 0.141→0.083 |
| groot_spatial_100 | 0.0492 | 0.232 / 0.013 | 5/9/10/10/10/10/10/10/8/9 | — | 0.137→0.097 |
| groot_spatial_200 | 0.0498 | 0.256 / 0.019 | 14/19/20/20/20/19/20/20/16/18 | — | 0.126→0.097 |
| groot_spatial_500 | 0.0617 | 0.267 / 0.022 | 36/44/47/49/49/48/49/47/40/47 | — | 0.126→0.102 |

旧门（noprog）在库回放里每锚点触发 13–27%，Q3 只有 0.1–8%。所以旋钮关的预测 IR 从 0.10–0.20 降到 0.075–0.11。

**各档周期重解**
- 用 R15 `fits.schedule_episodes` 的偶数步锚点，把 guard 换成上面的 Q3 掩码，`ScheduleReplay.solve('periodic', 目标)` 原样。
- 回放核对：用旧 noprog 掩码重算，16 格 × 4 档全部逐位等于 formal-A 在用的周期（`r16/design.json` periodic_caps）。
- 新周期 16×4 全部 `ok`，取四位小数后的预测 IR 与目标最大偏差 1.1e-4。
- 两个库共用同一套周期，沿用 formal-A 的做法：成功过滤库只是候选行过滤，旋钮不变，Q3 也相同。
- 下表左边是旧周期，右边是新周期：

| 格 | ir45 | ir40 | ir32 | ir25 |
|---|---:|---:|---:|---:|
| pi05_l10_50 | 0.1957→0.1496 | 0.4386→0.3257 | 1.0657→0.7513 | 2.6125→1.4606 |
| pi05_l10_100 | 0.1801→0.1460 | 0.4031→0.3243 | 0.9491→0.7511 | 2.1038→1.4545 |
| pi05_l10_200 | 0.1776→0.1520 | 0.4001→0.3377 | 0.9305→0.7650 | 2.0000→1.4989 |
| pi05_l10_500 | 0.1825→0.1557 | 0.4057→0.3430 | 0.9459→0.7778 | 2.0924→1.5395 |
| pi05_spatial_50 | 0.1900→0.1659 | 0.3931→0.3357 | 0.8707→0.7266 | 1.7790→1.3970 |
| pi05_spatial_100 | 0.1840→0.1707 | 0.3805→0.3489 | 0.8314→0.7624 | 1.6623→1.4546 |
| pi05_spatial_200 | 0.1742→0.1677 | 0.3633→0.3483 | 0.7946→0.7636 | 1.5395→1.4355 |
| pi05_spatial_500 | 0.1712→0.1689 | 0.3524→0.3456 | 0.7833→0.7627 | 1.4757→1.4287 |
| groot_l10_50 | 0.2007→0.1421 | 0.4512→0.3108 | 1.0252→0.7341 | 2.7237→1.3941 |
| groot_l10_100 | 0.1925→0.1417 | 0.4333→0.3209 | 1.0030→0.7482 | 2.4851→1.4356 |
| groot_l10_200 | 0.1959→0.1515 | 0.4385→0.3362 | 1.0069→0.7664 | 2.4585→1.4994 |
| groot_l10_500 | 0.1965→0.1550 | 0.4338→0.3412 | 1.0011→0.7731 | 2.4699→1.5249 |
| groot_spatial_50 | 0.1906→0.1678 | 0.3931→0.3365 | 0.8576→0.7349 | 1.7146→1.3942 |
| groot_spatial_100 | 0.1788→0.1631 | 0.3782→0.3378 | 0.8456→0.7511 | 1.6988→1.4173 |
| groot_spatial_200 | 0.1725→0.1604 | 0.3721→0.3418 | 0.8221→0.7567 | 1.6398→1.4266 |
| groot_spatial_500 | 0.1771→0.1666 | 0.3722→0.3478 | 0.8314→0.7639 | 1.6294→1.4523 |

**CPU 冒烟**（只做了下面这些，没跑测试套件）
1. `smoke.py pi05_l10_50 groot_spatial_500`，结果在 `logs/smoke_main.json`：
   - 按插件方式 `pickle.load` 加载 fit：原库旋钮关、过滤库旋钮关、cs c、cs cr，再在内存里拼一份 ir32；
   - 每份跑一条失败库局的全部偶数步查询；
   - 5 种 fit 的 Q3 调用都和库留一掩码逐锚点相同（22/52 和 8/22 次）。这说明在线原始投影和库码一致；ir32 的周期旋钮能正常触发。
2. 在新树里：
   - `JudgeSpec.parse('stall_only')` 正常，`guard_only` 仍然报错；
   - fit 的类从新树加载，`clone_method` 后共享 Q3 模型；
   - 构造了 3 个 `PluginRuntime`（R16 过滤库旋钮关、cs chrg、R15 GR00T）；
   - `torch.cuda` 未初始化。
3. 6 个 R16 group 跑 `R16Strategy.plan`：分别规划出 4000/4000/4000/4000/16000/16000 局，task_uid 全部唯一。
4. H100 只读资产计划：`legacy plan --work-dir` 跑 `fa_pi_f_f_o_q_l50`。
   - 重定位后的 fit 能加载，`act` 指向 `/data/oscl_h100/store/...`，Q3 指纹不变；
   - 生成的副本已删除。

## 3. 新批次清单（各档等旋钮关结果出来再建）

| 白话（库 · 模型 · LIBERO · 步数 · 阶段） | 短名 | 局数 | 服务在哪 | GR00T/π0.5 副本 | 脚本（commands/） |
|---|---|---:|---|---|---|
| 原库 · π0.5 · 全步 · 旋钮关 | fa_pi_f_u_o_q（8 格 `_l50…_s500`） | 8×500=4,000 | H100（/data/oscl_h100/openpi） | π0.5 3（不变） | `prepare_/launch_fa_pi_f_u_o_q[_<格>].sh` |
| 成功过滤库 · π0.5 · 全步 · 旋钮关 | fa_pi_f_f_o_q | 4,000 | H100 | π0.5 3 | `…_fa_pi_f_f_o_q[_<格>].sh` |
| 原库 · GR00T · 全步 · 旋钮关 | fa_gr_f_u_o_q | 4,000 | 4090 本机（新树） | GR00T 5（PORTS 23100–23104，NEED_MB 7200） | `…_fa_gr_f_u_o_q[_<格>].sh` |
| 成功过滤库 · GR00T · 全步 · 旋钮关 | fa_gr_f_f_o_q | 4,000 | 4090（新树） | GR00T 5 | `…_fa_gr_f_f_o_q[_<格>].sh` |
| 原库 · π0.5 · 减步 K1 · 旋钮关（合成一个 group） | fa_pi_r_u_o_q | 4,000 | 4090 常驻（新树） | π0.5 3 | `prepare_/launch_fa_pi_r_u_o_q.sh` |
| 成功过滤库 · π0.5 · 减步 · 旋钮关 | fa_pi_r_f_o_q | 4,000 | 4090 常驻 | π0.5 3 | `…_fa_pi_r_f_o_q.sh` |
| 原库 · GR00T · 减步 · 旋钮关 | fa_gr_r_u_o_q | 4,000 | 4090 常驻 | GR00T 4（每 suite 2，7200 MB） | `…_fa_gr_r_u_o_q.sh` |
| 成功过滤库 · GR00T · 减步 · 旋钮关 | fa_gr_r_f_o_q | 4,000 | 4090 常驻 | GR00T 4 | `…_fa_gr_r_f_o_q.sh` |
| 原库 · π0.5 · 减步 · 缓存部件带门四组 cg/crg/chg/chrg | fa_cs_pi_rg | 8×4×500=16,000 | 4090 常驻 | π0.5 3 | `…_fa_cs_pi_rg.sh` |
| 原库 · GR00T · 减步 · 缓存部件带门四组 | fa_cs_gr_rg | 16,000 | 4090 常驻 | GR00T 4 | `…_fa_cs_gr_rg.sh` |
| 各档 ir45/40/32/25（8 个单位） | fa_{pi,gr}_{f,r}_{u,f}_l_q | 每个要调的格 2,000 | 同对应的旋钮关 | 同上 | 先 `build_levels_<名>.sh`，再 `prepare_/launch_<名>.sh` |

合计：旋钮关 32,000 局，带门四组 32,000 局；各档最多 8 单位 × 8 格 × 2,000 = 128,000 局，实际按新旋钮关结果选格。

**带门四组的配置**
- cg 和 crg 是 C5，不带 `--os-blind`，MISS 执行 5 步。
- chg 和 chrg 是 C10，带 `--os-blind --os-policy-tail --os-policy-tail-blocks 1`，MISS 执行满 10 步。
- 四组都带 `--os-judge stall_only`，`full_model=True`，K1（yaml 是 miss num_steps 1，GR00T 带 `GROOT_DENOISING_STEPS=1`）。门触发的调用是唯一的模型调用。

**旋钮层**
- 全部用 R15 `TaskDescentKnob`：十个任务同一档；off 的 cap 为 0，其他档的 cap 取 `PERIODS.json`。
- R15 全步原来用的是 R11Knob/R13Knob。两者的周期算术相同：同一枚 `R11-gap-v1` 硬币、同样的 cap 和 run 规则。
- 这样全步和减步走同一个类，fit 文件也共用。

**副本数**
- 按 owner 10-20 新规定：GR00T 不超过 5 个副本，每副本按 7.2 GB 算。
- 本机这张 4090 实测 49,140 MiB，所以 R15 GR00T 全步开 5 个副本（启动时需要 36 GB 空闲）。
- R16 常驻按 suite 分进程，副本数只能是偶数，所以取 4（2+2），并把每副本启动预算从 6500 MB 改成 7200 MB（`runner.py`）。
- π0.5 照原配置不变。

## 4. 建议的上机顺序

1. **现在就能跑：π0.5 全步旋钮关**（fa_pi_f_u_o_q、fa_pi_f_f_o_q，服务在 H100，worker 走中央调度）。前提是先把下面两个文件用公网 rsync 推到 H100 的 `/data/oscl_h100/openpi/`（相对路径）：
   - `exp/offline_search/rounds/r06/ideation_Q1/method_c/stall_only.py`（新文件）
   - `exp/offline_search/closed_loop/plugin.py`：这是 H100 自己那份（md5 d6b98008…）加上同样 3 行的补丁版，不是主仓那份。

   两份都已放在 `/nvme/weiland/openpi/r17_q3_port_1020/h100_push/` 下，目录结构相同。新 fit 用到的其他模块（stall.py、stall_bridge、r05 CommitJudge、r10/r11/r13/r14/r15 recipe、blind_awm、awm、g3_core、blind、api、dims）的 md5 已用 `tether exec` 核对，和主仓相同。H100 上现在没有进程在用这份副本。
   - 跑法：`bash commands/prepare_fa_pi_f_u_o_q.sh` 后接 `launch_…`。单位级脚本会逐格依次执行，也可以用单格脚本 `_l50` 等。
2. **等 fa_cs_pi_r 和 fa_cs_gr_r 都结束后，在 4090 上串行跑**：
   - 先 R16 减步旋钮关：fa_pi_r_u_o_q、fa_pi_r_f_o_q、fa_gr_r_u_o_q、fa_gr_r_f_o_q；
   - 再 R15 GR00T 全步旋钮关：fa_gr_f_u_o_q、fa_gr_f_f_o_q；
   - 最后带门四组：fa_cs_pi_rg、fa_cs_gr_rg。

   要等的原因：这些批次和 fa_cs 共用 4090 的显存和 23150–23159 端口；R16 launch 的端口和 fa_cs 相同（`--port 23159 --ports 2315x`）。
3. **各档**：某个旋钮关单位跑完后，执行对应的 `commands/build_levels_fa_<m>_<f|r>_<u|f>_l_q.sh`。
   - 只在 CPU 上做：装载结果，和同步数纯推理（原 formal-A 参照）比，选出要调的格，建 fit 和定义，写脚本。
   - 然后执行 `prepare_/launch_fa_<m>_<f|r>_<u|f>_l_q.sh`。
   - 有格没完成时会报 pending，不会建出任何东西。

## 5. 拔掉/绕开的检查（都在新文件里，没改原文件）

- `analysis.py` 在本进程里替换了 `r16.analyze.require`，跳过两条检查：
  - `'Changed judge'`：`r16/analyze.py:83`，原来要求 `--os-judge guard_only`；
  - `'Changed client/policy protocol'`：`r16/conductor_analysis.py:44`，这条把 replan、full_model、recording、seed、blind、policy-tail 和 guard_only 捆在一起检查。只影响用这个模块做的分析，其他检查照旧。
- 绕开的写死项：
  - `r15/descent/prepare.arm_spec` 里写死的 `guard_only`：由 `build.r15_spec` 自己生成插件参数；
  - `vla_workers.r15_command` 写死的 legacy 模块：在 `build.r15_run` 里临时替换成 `q3_port_1020.legacy`。
- 没有新增任何校验层、哈希钉或源码锁。新树服务端原有的 fit/yaml sha 核对还在，不挡路，没动。

## 6. 已知风险

1. **R15 GR00T 的服务端代码版本**：现在跑的是冻结 R16 树的代码，旧 R15 跑的是当时的主仓。差异在第 1 节，B=1 时应当等价，但这是实测前的推断。
2. **R15 GR00T 5 副本的显存**：启动要求空闲 ≥36 GB，等不到会在 900 s 后报 GPU_TIGHT。到时把 `q3_port_1020/QUEUE.json` 里对应 job 的 env 改成 `PORTS=23100,23101,23102,23103`、`LOCAL_SERVER_COUNT=4`，就回到 4 副本。
3. **C5 带门组的窗口**：cg/crg 每 5 个控制步观察一次，窗口跨度是 C10 标定时的一半。估计器按跨度归一，但统计性质和标定时不同。C10 各组和旋钮批次的锚点则和 L=10 的标定网格对齐。
4. **slow_ambiguous 不加看**：按协调者决定不插段中 LOOK。回放里这个状态不少（例如 pi05_l10_50 任务 2：50/159 个锚点）。冷却会让持续停滞时的调用隔一个锚点出现一次（例：41 次确认慢 → 22 次调用）。
5. **小库格的无效任务**：库大小 50 的格里，上面列出的无效任务 Q3 永不触发，这些任务上只有旋钮能调用。
6. **成功过滤库共用周期**：两个库共用原库的周期，沿用 formal-A。R17 longseg 是按库分别解周期的；如果要按成功局单独重解，改 `fit.solve_periods` 的入参即可。
7. **内存**：fit 比 cs 研究的大约一倍（内嵌 Q3：50 格约 24 MB，500 格约 60 MB）。R16 常驻要载入整组 32 个 entry。相同数组会被 ArrayPool 合并，没实测常驻内存。
8. **H100 推送前后**：π0.5 全步必须在两个文件推上去之后才能 prepare；否则 H100 服务端会因为不认识 `stall_only` 直接退出（SystemExit）。
9. **旧结论作废**：新的旋钮关 IR 会明显低于旧门时代（预测 0.075–0.11）。旧的 `formal_a_1016/off_decisions/{pi05,groot}_*` 作废；新的选格结果写到 `q3_port_1020/off_decisions/`。
