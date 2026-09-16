# 减步 teacher（NFE）基线 × RoboCasa365 — 计划

> 状态：`In Progress`（2026-09-15 14:4x owner 裁定 §6 B：LIBERO 剩余 spatial 不跑，l10 跑完即切 RoboCasa；smoke 进行中）
> 级别：**L1**（复用 `exp/nfe_baseline/` 与 `exp/robocasa365/` 现有机件，新增只有启动脚本与聚合器的 RoboCasa 输入分支；`src/` 零改动）
> 姊妹计划：[`nfe_baseline_plan.log.md`](nfe_baseline_plan.log.md)（LIBERO 两 suite，2026-09-15 在跑）。
> 动机：LIBERO 上两 VLA 一步去噪几乎无损（π0.5 spatial 0.988/0.99、l10 0.824/0.92；GR00T spatial 0.932/0.946、l10 0.856/0.868），
> ActionCache（arXiv 2607.06370）Table 1 显示 VLABench 上一步会崩（π0.5 38.8→6.8）。步数敏感性是 benchmark 属性，
> 我们第三个 benchmark RoboCasa365 上是哪一种，决定论文主图与 warm-start 叙事的去留。

## 1. 口径（全部沿用 RoboCasa RIT 线 `logs/rc365_rit_run_progress.md` §0 的冻结值）

| 项 | 值 |
|---|---|
| 任务 | 13 个 Atomic-Seen 任务：main 8（CloseBlenderLid, CloseFridge, CoffeeSetupMug, OpenCabinet, OpenDrawer, OpenStandMixerHead, SlideDishwasherRack, TurnOnSinkFaucet）+ pnp 5（PickPlaceCounterToCabinet, PickPlaceCounterToStove, PickPlaceDrawerToCounter, PickPlaceSinkToCounter, PickPlaceToasterToCounter，**钉物体** `exp/robocasa365/config/pnp_pinned_objects.json`） |
| 场景 / 种子 | `layout=1, style=1`，`replan_steps=5`，**eval seed = 1,000,000 + idx，idx 0…49**（`env.reset(seed=base_seed+orig_init_state_idx)`），与 RIT 全部臂及 teacher-only 参考臂逐集同 seed |
| 每组 | 13 任务 × 50 = **650 集**；两条 lane（main 8 任务 / pnp 5 任务）分开跑，同 RIT 线 |
| 阶梯 | **GR00T N1.5（target-posttrained, N=4）k = 1, 2, 3**；**π0.5（N=10）k = 1…9**；N 步端点复用 RIT 线 teacher-only 参考臂（GR00T macro 0.680 = contact-8 0.640 / pick-5 0.744；π0.5 v2 0.557） |
| 指标 | **任务级 macro SR**（先任务内 50 集平均、再 13 任务等权平均，RIT 线 D1 口径）；IR 为解析常数 |
| 成本（RIT 同式） | IR(k) = (s1 + s2 + s3(k)) / MISS，s3(k) = a + b·k 取 RIT 台账线性拟合（`~/tmp_rit/rc_cost_{groot_tp,pi05}.json`，将拷入 `exp/nfe_baseline/config/`）：GR00T 8.12 / 9.36 / 5.777+3.006k，MISS 35.28 → k=1…3 = **74.4 / 83.0 / 91.5 %**；π0.5 9.828 / 27.374 / 0.357+3.042k，MISS 67.98 → k=1…9 = **59.7 / 64.2 / 68.7 / 73.1 / 77.6 / 82.1 / 86.6 / 91.1 / 95.5 %**。台账里还有每 k 实测值（GR00T 1…4：8.38/11.28/13.73/17.58 ms；π0.5 1/3/5/10：3.39/9.47/15.59/30.77 ms），只作附注不改口径 |
| 纯推理 | 不带任何 cache_config：GR00T `serve_groot_n15_ksweep.py`（teacher-only，`_InferLockedPolicy`）；π0.5 `serve_pi05_ksweep.py --cache`（只装 interceptor + BatchingCoordinator，无库；`__hit_meta__` 恒 MISS） |

⚠ GR00T RoboCasa 的 stage 3 截距大（5.78 ms），k=1 也只到 IR 74.4%——减步基线在这个 teacher 上够不到 IR<74 的区间，而 FULL_HIT 地板是 23%。这本身就是一个要写进结果的事实。

## 2. 机件（现成 / 新增）

现成、不改：
- `exp/robocasa365/serve_groot_n15_ksweep.py`（G-A1 先例：`--denoising-steps k` 注入 `Gr00tPolicy`，打印 `KSWEEP num_inference_timesteps=k`）+ `serve_groot_n15.py --concurrent`；
- `exp/nfe_baseline/serve_pi05_ksweep.py`（LIBERO 线 wrapper，`policy:checkpoint --policy.config pi05_robocasa --policy.dir <ckpt>`；norm_stats 在 ckpt 目录自带的 `assets/`）；
- `exp/robocasa365/run_ws_search.py --teacher {pi05,groot_tp} --server host:port --cid teacher --run-prefix nfek<k> --tasks … --episodes 50 --layout 1 --style 1 --base-seed 1000000 --replan-steps 5 [--pinned-objects …] --env-config … --workers W --gpu-ids … --role all`（G-A1 与 RIT teacher-only 参考臂同款驱动：conductor journal + resume + 看门狗；产物 `journal_/run_plan_/summary_<run_id>.json`，summary 含每任务 SR 与 `complete` 标志）；
- `exp/robocasa365/config/collect_calib_timan107.env`（timan107 岛：`WORKER_PYTHON=/scratch/zixuans8/Isaac-GR00T/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python`、`ROBOCASA_CWD=…/external_dependencies/robocasa365`、`REPO_ROOT=/scratch/zixuans8/openpi_rc365`、系统 EGL）；timan108 同样有这三样（已核）。

新增（全部 `exp/nfe_baseline/`）：
- `ops/rc/launch_groot_rc_server.sh <k> <port>`（h100 或 weilandserver）、`ops/rc/launch_pi05_rc_server.sh <k> <port>`（weilandserver，`--cache` 批处理）；`ops/rc/ladder_rc_server.sh`（沿用 LIBERO 线"见连接后空闲即换 k"的驱动，只换起服命令）。
- `ops/rc/ladder_rc_client.sh <teacher> <ks> <server> <lane main|pnp> …`（timan 侧：对表握手步数 → `run_ws_search.py --cid teacher --run-prefix nfek<k>` → 等 `summary_*.json` 且 `complete:true` → 下一 k）。探针复用 `probe_metadata.py`（GR00T RoboCasa server 的 metadata 键待核，不在则改读 `KSWEEP` 行经 tether）。
- `aggregate_nfe.py` 加 `--policy {pi05_rc, groot_rc}`（`rit_cost_rc.StageCost` 读台账 JSON）与 `--summaries <dir>` 输入模式（读 `summary_nfek<k>-teacher__l1s1_<teacher>.json`，macro SR = 13 任务等权、要求每任务 `n_scored=50, err=0, missing=0`），spec `figure_id = nfe_<policy>_robocasa365`。
- `tests/exp/test_nfe_baseline.py` 加：RoboCasa 计价对 `rit_cost_rc.tier_cost`（GR00T k=1 ≡ WARM@0.75、k=2 ≡ WARM@0.50；π0.5 k=3 ≡ WARM@0.3、k=5 ≡ WARM@0.5）逐位相等；summary 合并的 13 任务/50 集/complete 校验。

## 3. 拓扑（推荐拓扑不变：lane A = h100 ↔ timan108，lane B = weilandserver ↔ timan107）

| lane | server | client | 内容 | 端口 |
|---|---|---|---|---|
| A | h100 `serve_groot_n15_ksweep.py`（ckpt `/home/exouser/ckpt/n15_robocasa_tp/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000`，gr00t venv 同 LIBERO 线）1 进程 | timan108 30 worker（RoboCasa 每上下文 1.5–1.6 GiB，3×24 GB 卡 ≤75%） | GR00T k=1,2,3 × {main, pnp} | 23230 |
| B | weilandserver `serve_pi05_ksweep.py --cache`（ckpt `/home/weiland/ckpt_pi05_robocasa_pytorch`，`pi05_robocasa`）1 进程，batch 16 | timan107 30 worker（8×8 GB 卡） | π0.5 k=1…9 × {main, pnp} | 23170 |

- 一个 k 一组 server；main 与 pnp 两条 lane 对同一组 server **顺序**跑（同 RIT teacher-only 臂：`main` 8 任务 → `pnp` 5 任务）。
- 每 k 换服（GR00T ~2 min、π0.5 ~2 min），`ladder_rc_server.sh` 按"见过连接后空闲 150 s"判 k 结束。
- ⚠ 与 LIBERO 线的资源冲突：timan108 现被 LIBERO GR00T lane 占（64 worker、显存 56%），timan107 被 LIBERO π0.5 lane 占（64 worker、内存 ~70 GB）；RoboCasa worker 每个 1.5 GiB 显存，两条 LIBERO lane 跑完前放不下 30 个。**开跑时点见 §6。**

## 4. 时长（按 devices.md RoboCasa 实测：30 worker ≈ 10.5 集/分）

- 每组 650 集 ≈ 62 min + 换 k 2 min → GR00T 3 组 ≈ **3.2 h**；π0.5 9 组 ≈ **9.6 h**。
- 若 GR00T 跑完后把 π0.5 的 k 拆到两台 client 机（h100 也起一个 π0.5 server：ckpt 已在 `/home/exouser/ckpt/pi05_robocasa_pytorch`），π0.5 ≈ 5 h；总 ≈ **8 h**。
- smoke：GR00T k=2 main 1 任务 × 5 集（~5 min），核 `KSWEEP` 行、summary 格式、seed 复现（与 RIT teacher-only 臂同 seed 的首集 obs 哈希可对）。

## 5. 产物

- 远端：`<REPO_ROOT>/exp/robocasa365/data/nfe/<teacher>/<lane>/{journal,run_plan,summary}_nfek<k>-teacher__l1s1_<teacher>.json`；拉回本地 `exp/nfe_baseline/data/runs/<policy>_robocasa365/`（gitignore）。
- `aggregate.json`（每 k：13 任务 SR、macro SR、IR、台账 sha）+ spec `exp/rit_pareto/analysis/figures/nfe_{pi05,groot}_robocasa365.json`（可叠 RIT 线 `rit_{groot,pi05}_all13.json` 的前沿作参照，同图链引用机制）；渲染 `render_figure --new`。
- 结果段追加到本文件 §7；不 commit 不 push（owner 裁）。

## 6. 顺序与依赖（需 owner 裁决）

1. **A. 等 LIBERO 收工再开**（GR00T lane ≈ 15:45、π0.5 lane ≈ 18:45）：零冲突；RoboCasa 总 8 h → 明晨结束。
2. **B. 提前腾资源**：LIBERO 剩下的 spatial k（GR00T 4…7、π0.5 3…9）曲线已平，可以砍掉/延后，把 timan108 在 GR00T l10 跑完（≈15:20）时直接切 RoboCasa GR00T；π0.5 l10 跑完（≈17:30）时切 RoboCasa π0.5。总提前 ~1.5–2 h。
3. 无论 A/B：先 5 集 smoke，再 GR00T 全阶梯 ∥ π0.5 全阶梯；每组 `complete:true` 才进下一 k；任何 `n_err>0` 停下报。

## 7. 风险

- GR00T RoboCasa server 在 h100 上从未以 ksweep 方式起过（RIT 线的 h100 服务是 LIBERO/或 k=1 补做），首起可能缺 `decord` 等依赖 → smoke 阶段暴露；备选 weilandserver（RIT 线原机，与 π0.5 server 共卡，显存 7.5 + 7.5 GB 够）。
- `run_ws_search.py` v1 单 server：π0.5 无 replicas，靠 coordinator 批处理；若 30 worker 把单进程压满（LIBERO 线实测 4 进程批处理 22 集/分），可在 weilandserver 起第 2 个进程、两 lane（main/pnp）各连一个。
- 步数生效证据：GR00T `KSWEEP num_inference_timesteps=k`（ksweep 断言）；π0.5 `KSWEEP run_stage3 first call num_steps=k`；两边响应无库信息（`__hit_meta__` 恒 MISS）。
- timan108 RoboCasa 上下文踩踏先例（45 worker 压死驱动）：固定 30、`RC365_ENV_BUILD_LOCK` 默认开。

## 8. 运行记录

（开跑后追加）
- **14:4x owner 裁定 §6 B**。14:53 起 h100 上起 RoboCasa smoke server：GR00T k=2 `:23230`（`serve_groot_rc_ksweep` → G-A1 wrapper，日志 `KSWEEP num_inference_timesteps=2`）、π0.5 k=3 `:23231`（wrapper `--cache`，`pi05_robocasa`）；timan108 起 2-worker smoke client（GR00T main CloseFridge×5、π0.5 main CloseFridge×5）。
  ⚠ 两个坑：① `run_ws_search` 的 `--server` 必须在 env 文件对应 teacher 的 `*_SERVERS` 组里（亲和门），`rc_timan.env` 已把 h100/weilandserver 的候选端口都列上；② pnp lane 有冻结身份检查（5 任务定序 × 50 集），不能缩小做 smoke，只能整跑。
  实现落地：`serve_groot_rc_ksweep.py`、`ops/rc/{launch_groot_rc_server,launch_pi05_rc_server,run_rc_client,ladder_rc_client}.sh`、`ladder_server.sh` 加 `groot_rc|pi05_rc`、`aggregate_nfe.py` 加 `--policy {pi05_rc,groot_rc} --summaries-root`、`config/rc_cost_{groot_tp,pi05}.json`（自 `~/tmp_rit/`）、`config/rc_timan.env`；tests 18 passed。切换脚本 `/tmp/nfe/switch_rc_{h100,wls,t108,t107}.sh`。
- **15:00 smoke 双过**（GR00T k=2 main CloseFridge×5 → 0.6；π0.5 k=3 main CloseFridge×5 → 0.2；均 `complete:true`、`KSWEEP` 标记在、零 Traceback）；smoke server 已停。
- **15:36 GR00T LIBERO l10 FINISHED → 切 RoboCasa GR00T**：h100 `nfeladder_grc`（k=1,2,3 @ :23230），timan108 `nfelane_grc`。
  ⚠ **timan108 显存**：RoboCasa GR00T 上下文实测 ≈2.8 GB（清单按 π0.5 估 1.5 GiB），30 worker → 88–90%、21 worker → 81–86%，均超 75% 线（该机有 98.7% 压死驱动先例）；
  15:41 / 16:09 两次在 lane 内停 worker（按 PID）并以 journal 续跑，最终 **15 worker（每卡 5 个 ≈ 14 GB）**。吞吐相应降到约 5–6 集/分，GR00T 三个 k 预计 ~21:45 完。
  π0.5 RoboCasa lane（timan107，8×8 GB）worker 数定 24（RIT 线同数）。

- **16:32 π0.5 RoboCasa 第二条子阶梯提前开跑（owner 16:28 告知 weilandserver 已空）**：weilandserver 4090 实测被本线 LIBERO 4 进程打满（42.5 GB / 100% util），塞不下第二 server；改用 h100（空 70 GB，已有 π0.5 RC ckpt + venv，:23231 冒烟过）：h100 `nfeladder_prc2`（`ladder_server.sh pi05_rc … 9,8 23231`）↔ timan108 `nfelane_prc2`（`ladder_rc_client.sh pi05 9,8 149.165.153.233:23231 6 0,1,2`，每卡 2 个 ≈1.5 GiB，与 15 个 GR00T worker 同卡，预估峰值 ≤72%）。主阶梯 `/tmp/nfe/switch_rc_{t107,wls}.sh` 相应缩为 **k=1…7**（:23170，24 worker）。启动脚本 `/tmp/nfe/start_prc2_{h100,t108}.sh`（本地 `exp/nfe_baseline/data/tmp/`）。
  GR00T k=1 main 进度 16:29：198/400（15 worker ≈3.9 集/分），三个 k 重估 ~00:00 完；GR00T 收工后可在 k 边界给 π0.5 子阶梯加 worker（journal 续跑）。
- **16:41 timan108 显存告警 → 子阶梯缩到 3 worker**：Monitor 报 GPU1 20.5 GB（83%）。逐进程看：GR00T worker 空闲 1.36 GB、峰值 3.48 GB；π0.5 RoboCasa worker 空闲 1.05 GB、峰值 **3.12 GB**（不是清单里的 1.5 GiB），5+2 每卡最坏 23.7 GB 会撞 24.5 GB。16:43 按 PID 停掉 6 个 π0.5 worker（`/tmp/nfe/resize_prc2_t108.sh`），journal 续跑 k=9 main，改 **3 worker（每卡 1 个）**，最坏 20.6 GB。子阶梯吞吐 ≈2 集/分，k=9 约 5.5 h；GR00T 收工后再加。⚠ timan107 主阶梯 24 worker（8 GB 卡 3 个）沿用 RIT 线实证数，峰值靠错峰。
- **17:00 全线停（owner 裁定"k=7 跑完之后就停，我们不跑了"，17:05 确认"管所有"）**：RoboCasa 两条 lane 全停（h100 :23230/:23231 server 阶梯、timan108 两个 client 阶梯；17:02 曾因我怀疑误读重启了 3 分钟，17:05 再停），journal 保留可续跑：GR00T k=1 main 387/400、π0.5 k=9 main 36/400；pnp lane 未开。RoboCasa 减步基线**无完整 k**，不出图。
