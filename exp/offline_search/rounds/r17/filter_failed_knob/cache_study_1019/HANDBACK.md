# #135 八组缓存部件研究（原库 · π0.5 / GR00T · LIBERO · 减步）— 交回 2026-10-19

只做了 CPU 上的定义和演练。没有起服务、worker、rsync，没有 tether，也没有跑 prepare/launch 包装脚本。
没有改共享代码，也没有动 `formal_a_1016/QUEUE.json`、在跑的链、`openpi_ext/scratch/r17_ops/` 和已有运行目录。

## 结论

- **能直接排的是四个不带停滞门的组**（c、cr、ch、chr）。每个模型一批，一批 8 个库格 × 4 组 × 500 局 = **16,000 局**，两批合计 **32,000 局**。
- **四个带停滞门的组（cg、crg、chg、chrg）没有建**。R16 这套代码里没有 Q3 停滞触发器，唯一的门是 R8 的「只看无进展」旧门，也就是 10-13 已经拔掉、禁止再用的 NoProgressGuard。要跑这四组，得先把 Q3 移植进 R16，这件事要 owner 或协调者拍板（见最后一节）。
- 八组全做是 2 × 8 × 8 × 500 = **64,000 局**，其中 32,000 局（带门四组）在等这个决定。

## 每组改了什么（R16 流程里的实际开关）

每个库格用的都是**现行 formal-A「原库 · 减步 · 旋钮关」那份 fit 本身**，没有重新拟合：库的行、PCA、度量、修正器头都不变（CPU 演练逐格比对了拟合数组，逐字节相同）。
改的只是外面那层：去掉 R8 旧门（`TriggerCommitJudge`），换成现成的 `R10Recipe` 包装。插件参数里也去掉 `--os-judge guard_only` 和 `--os-policy-tail`，所以四组都**不可能调用模型**。

| 组 | 执行段 | 修正器 | 停滞门 | fit 里的缓存类 | 插件开关 |
|---|---|---|---|---|---|
| c 纯缓存 | C5：每 5 步重新检索一次 | 关 | 关 | `BlindAWM`（只是把修正头的类换掉，数组不变） | 不带 `--os-blind` |
| cr 缓存+修正器 | C5 | 开（R10 GC_dist 残差头，强度 .5·clip((2−r)/1.25)） | 关 | `_CorrectedCache` 原样 | 不带 `--os-blind` |
| ch 缓存+长执行段 | C10：先执行 5 步，再用同一个 anchor 的无视觉尾段执行 5 步 | 关 | 关 | `BlindAWM` | `--os-blind`（anchor_tail，budget 1） |
| chr 缓存+长执行段+修正器 | C10 | 开 | 关 | `_CorrectedCache` | `--os-blind` |
| cg / crg / chg / chrg | — | — | Q3 | **未建** | — |

- C10 的做法和 R10 A、R15 AC_dist 完全一样（5 步加一个无视觉尾段）。C5 就是不开尾段，每次请求都重新检索、重新合成。
- 公共部分照搬 formal-A 的减步行：`--os-root /home/weiland/trace_runs/offline_search_store --os-no-shadow-native --os-tokens off`、`--os-rollout-seed 2026101401`、`--os-r14-audit`，YAML 原样复制（`miss.num_steps: 1`）。
- 每局成本：四组的动作头调用都是 0，只有新 anchor 时要算视觉。按 R16 的 D5 计价，π0.5 视觉一次 .152、GR00T 一次 .148；所以 C5 组 IR 约等于 .152 / .148，C10 组约为它的一半。

## 减步 K = 1，出处

- R16 设计 `exp/offline_search/rounds/r16/design.json` 只登记了 k=1/2 两档（`rounds/r16/cost.py::pricing` 只接受 1、2 和满步）。
- 现行的减步 formal-A 批次 `fa_pi_r_u_c / fa_pi_r_u_l / fa_gr_r_u_c / fa_gr_r_u_l`（`formal_a_1016/QUEUE.json`，mode=reduced）都是 group `k: 1`，行上是 `r16_k: 1`、`yaml_patch.miss.num_steps: 1`，模板 `r16_conductor_test_{model}_k1_v1`。
- 新的两批也是 `k: 1`。常驻服务会因此设 `os_miss_steps=1`，GR00T 服务还会带 `GROOT_DENOISING_STEPS=1`。
- 这四组不调用模型，所以 K 对它们的结果没有影响。保留 K1 是为了和旋钮批次一致，也给以后带门的组用。
- 同步数的纯推理参照（减步）用 R16 phase1 的 K1 纯推理 C10：`r16p1_{pi05,groot}_{l10,spatial}_k1_pure_c10`，在 `/home/weiland/trace_runs/os_closed_loop/r16_phase1_{model}_{suite}_k1`，跑在 4090 上，A 集每 suite 500 局。这和 `formal_a_analysis.vla_references(model,'reduced')` 用的是同一份。

## 库

都是现成的库，没有重建。

- YAML 里的 `preload_path`：
  - π0.5：`/home/weiland/trace_runs/dual_20260923/libs/{libero_10,libero_spatial}/cp1_spatial_pool_16.pkl`，软链到 `/data/r17_library/libero/pi05/{suite}/cp1_spatial_pool_16.pkl`。
  - GR00T：`/data/libero_cache/libraries/{suite}/{suite}_sp16_S3.pkl`，软链到 `/data/r17_library/libero/groot/{suite}/{suite}_sp16_S3.pkl`。
- 检索用的库在 fit 里：π0.5 是 `bpool_cs`，GR00T 是 `bpool_all`，按 `episode_subset` 取 50/100/200/500 局（每任务 5/10/20/50 局）。
- 每格的 fit 来源：`formal_a_1016/runs/ffk1015_{model}_libero_reduced_formal_A_original/definitions/*_{cell}_k1_off/fit.pkl`。

## 规模

- 库格和现行 formal-A 一样：libero_10 和 libero_spatial，各取库 50/100/200/500，共 8 格（l50 l100 l200 l500 s50 s100 s200 s500）。
- 每个 arm 跑官方 A 测试集：10 任务 × 初始状态 0–49，500 局，用 A 池初始状态。仿真种子和 rollout 种子都是 2026101401，和 formal-A 一样。
- `fa_cs_pi_r`（原库 · π0.5 · LIBERO · 减步 · 缓存部件四组）：32 个 arm，16,000 局。
- `fa_cs_gr_r`（原库 · GR00T · LIBERO · 减步 · 缓存部件四组）：32 个 arm，16,000 局。
- arm 命名是 `fa_cs_{pi,gr}_r_{格}_{组}`，例如 `fa_cs_pi_r_l50_chr`。

## 4090 队列要追加的链项

按顺序追加：

```
fa_cs_pi_r fa_cs_gr_r
```

- `chain_ffk2.run.sh` 会依次执行 `filter_failed_knob/commands/prepare_<名>.sh` 和 `launch_<名>.sh`。
- prepare 包装做的事和 `_dyn` 包装一样：带同一份 `OPENPI_AUTOSCALE_NODE_CONFIG`，给 timan107 推送动态 worker 的发布包（LEGACY_AUTOSCALE），用 tether exec 加公网直连 rsync。重复执行也可以。
- launch 包装：`VLA_WORKER_HOST=timan107`；从 `/home/weiland/projects/openpi_r16_k1ref_server` 起原生 R16 conductor；`--agent-island /scratch/zixuans8/openpi_r16_conductor_v1 --workers 64`。自适应开着时，64 会被换成 timan107 的物理上限 104，实际数量由运行时调。
- GR00T 那批另外挂了 `attach_local.sh`，就是本机 4090 的 R16 worker。这个脚本只认 GR00T，所以 π0.5 那批只用 timan107。
- 端口是 R16 本机那一套：服务 23150–23152（π0.5，3 个常驻）或 23150–23153（GR00T，每个 suite 2 个），driver 23159。两批**必须一前一后跑**，也不能和其他 R16 批次同时跑。以前的 `fa_*_r_*` 都已 DONE，目前没有冲突。
- 显存：π0.5 约 3 × 10 GB，GR00T 约 4 × 6.5 GB。
- 时长按以前的 R16 减步批次估：`fa_pi_r_u_l` 16,000 局用了 2.9 h，`fa_gr_r_u_l` 12,000 局用了 3.9 h。这两批不调用模型，估计 π0.5 约 3 h、GR00T 约 4–5 h。
- 中断后再跑同一个 launch 会从 group 自己的 journal 续跑。
- 链日志里 `accepted=` 一栏会显示 `-`，因为 R16 的 journal 在 `$E/<名>/journal.jsonl`，不在 `driver/` 下（以前的 R16 批次也一样，不影响运行）。

## CPU 演练（`PLAN.json`，`plan.log`）

- 原生 `read_group` 封印校验通过。原生 `R16Strategy.plan`：每批 16,000 局、320 个阶段；两批合计 32,000 个 task UID，全都不重复。
- 每个 arm 都核对了：
  - 500 对 (task, init) 正好是官方 A 模板，而且是 A 池。
  - `--os-blind` 有没有和组的 C5/C10 对得上；没有 `--os-judge`，也没有 `--os-policy-tail`。
  - fit 的类和组对得上（c/ch 是 `BlindAWM`，cr/chr 是 `_CorrectedCache`），spec、kwargs、cell 和行一致。
  - 拟合数组和来源 fit 逐字节相同。
- 每个模型在 CPU 上真的构建了两个插件运行时（l50 c 和 s500 chr），能载入，judge 是 None，blind 开关正确。

## 没解决、要 owner / 协调者定的事

1. **带门四组需要把 Q3 移植进 R16。** 目前选的是最保守的做法：不用旧门，先不建。可选方案：
   - **A（建议）：把 Q3 移植进 R16。** 要做三件事：
     - 写一个新包装：每个新 anchor 用 r06 的 `StallTracker` 观察检索码，按 `stall_bridge` 的规则判断。确认停滞时强制做一次 K1 调用；为了让 MISS 也执行满 C10，要带 `--os-judge guard_only --os-policy-tail --os-policy-tail-blocks 1`。可疑停滞时下一步多看一次图。停滞调用之后冷却一个 anchor。
     - 每格每个 C 用库内成功局做 LOEO，跑 `StallModel.fit` 标定。标定按决策步计时，照 10-18 的规则：标定要先对上，再跑。
     - 在 CPU 上做回放核对。

     这是方法移植，相当于在 R16 的库上重新标定 Q3，估计要半天到一天。
   - B：用 R16 现成的门（R8 `noprog_span`，n=3、eps=.5）。几分钟就能建好，但它就是被禁止的那个旧门，不建议。
   - C：π0.5/GR00T 这两个模型只报四个无门组。
2. **顺带发现**：现行 formal-A 里 π0.5/GR00T 的旋钮批次全都带着这个旧门，「旋钮关」和各档都是。
   - R16 减步：16 个 off fit 逐个看过，全是 `TriggerCommitJudge`/`TriggerGrootCommitJudge`，`progress_guard=noprog_span`、`noprog_n=3`、`prog_eps=0.5`。
   - R15 全步：插件参数同样是 `--os-judge guard_only`。

   所以这两个模型的「旋钮关 = 只开停滞门」，和其他模型的 Q3 门不是同一个门。这一点和 10-13「Q3 only，LIBERO 和 MetaWorld 都适用」的裁定不一致。我没改，需要 owner 知道。
3. **C5 + 修正器（cr）。** R16 的修正头一次输出 10 步残差（在 C10 下拟合）。C5 时每次请求都重新算一次，只执行前 5 步，没有为 C5 重新拟合头。R17 那边重新拟合，是因为它的头维度随 C 变；这里维度不变，所以可以直接用。但这算不算"同一个修正器"，要 owner 认可。
4. **C5 不开尾段的服务方式**在这个 R16 库上从没在 GPU 上跑过（R10 A / R15 AC_dist 都是 C10）。CPU 载入已经通过了。上线后看一眼前几局：`/data/mw_finetune/eval/fa_cs_*_r/runs/<arm>/server_*/decisions_*.jsonl`，c/cr 应该每次请求都是新检索，ch/chr 应该隔一次出现 `cache_blind`；所有 arm 都不应出现 MISS。
5. **还没写分析脚本。** 成功率可以直接从 `journal.jsonl` 按 `yaml_id`（就是 arm 名）统计，参照用上面的 r16p1 K1 纯推理。

## 文件

- 代码：`cache_study_1019/study.py`。`build` 和 `plan` 只写定义；`prepare` 和 `serve` 是链项入口。`serve` 和 `legacy_release.coordinator` 一样会重新 exec 一次，只是 job 从本目录的 `CAMPAIGNS.json` 读，不碰 `formal_a_1016/QUEUE.json`。
- 定义：`cache_study_1019/runs/fa_cs_{pi,gr}_r/definitions/`，每格两份 fit（`_c`、`_cr`），已被 gitignore；`CAMPAIGNS.json`、`PLAN.json`、`build.log`、`plan.log`、`cpu/`（CPU 载入日志）。
- 运行目录：`/data/mw_finetune/eval/fa_cs_{pi,gr}_r/`，里面只有 `group.json` 和 `ROOT.json`（R16 原生定义步要求放在这里）。
- 包装：`filter_failed_knob/commands/{prepare,launch}_fa_cs_{pi,gr}_r.sh`。
- 重做：
  - 删掉上面两个运行目录和 `runs/` 之后，执行 `PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.cache_study_1019.study build`，再执行 `... plan`。
  - 一旦 prepare 过，`build` 会拒绝重建。
