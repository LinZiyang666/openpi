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
- **2026-09-16 14:14 续跑（owner 裁定：把 π0.5 / GR00T 的 RoboCasa365 减步实验跑完，每个 k 都跑；timan107/108 各可 ≥30 worker）**：GR00T h100 `nfeladder_grc` :23230 ↔ timan108 `nfelane_grc` **30 worker**（k=1 main 从 387/400 续，→ pnp → k=2 → k=3）；π0.5 weilandserver `nfeladder_prc` :23170 ↔ timan107 `nfelane_prc` **32 worker**（k=1…9）。Monitor 四机 + cron :17/:37/:57。
- **14:18 GR00T k=1 main 完成**：400/400，macro 0.5175（8 任务；4 步参考 0.640）。**14:19 pnp lane 起跑即 2000 次 `TypeError: Kitchen.__init__() got an unexpected keyword argument 'pinned_objects'`**：timan108 的 robocasa365 checkout 没打 `exp/robocasa365/patches/robocasa_pnp_pinned_objects.patch`（timan107 打过，RIT 线 pnp 在那边跑的）。14:20 `git apply` 打上，重挂 `switch_rc_t108.sh`（main 已完成自动跳过），pnp 30 worker 连上 h100（server 仍在 k=1，没触发 400 s 空闲切换），journal 正常写。
- **14:50 owner 裁定「groot 跑完之后把所有资源都给 pi，要做再均衡」** → 装了再均衡机制（脚本本地副本 `exp/nfe_baseline/data/tmp/{stopk_wls,stopk_t107,switch_prc2_h100,switch_prc2_t108}.sh`）：
  - pair A（wls:23170 ↔ t107 32 worker）继续升序 1..9；两端各挂 `nfestopk_prc` 看门 tmux，阈值文件 `/tmp/nfe/prc_stop_after_k`（初值 5），出现 `LADDER/RCLANE k=<K> DONE` 即杀 ladder（client 侧此时只在 probe server、无 worker；server 侧再 `stop_servers.sh 23170 1`）。
  - pair B（h100:23231 ↔ t108 30 worker，gpus 0,1,2）：`nfeswitch_prc2` 看门等 GR00T `LADDER FINISHED`/`RCLANE FINISHED` 后起 `ladder_server.sh pi05_rc robocasa365 9,8,7,6,5,4,3,2 23231` 与 `ladder_rc_client.sh pi05 9,8,...,2`；`nfestopk_prc2` 阈值文件 `/tmp/nfe/prc2_stop_after_k`（初值 6）。t108 上 pi05/main/k9 旧 journal 36 行会被 resume。
  - 两对不共享 /scratch（t107 与 t108 各自的 rc_results），所以分工全靠阈值文件；默认 A=1..5、B=9..6 不重叠，每个 k 边界按两对实测速度改阈值再均衡（改阈值只对尚未 DONE 的 k 有效）。
  - 14:50 状态：GR00T k=1 pnp 171/250；π0.5 k=1 main 209/400（前 49 集全是 CloseBlenderLid 0/49，该任务 10 步参考 0.10）。
- **15:00 GR00T k=1 全 13 任务完成**（main 400 + pnp 250，全 complete）：macro **0.572**（IR 74.4%）vs 4 步参考 0.680（−10.8 pp）。逐任务 k=1 / 4 步参考（`~/tmp_rit/teacher_ref_groot_50ep.json`）：CloseBlenderLid 0.70/0.74、CloseFridge 0.46/0.48、CoffeeSetupMug 0.56/0.68、OpenCabinet 0.80/0.98、OpenDrawer 0.60/0.74、OpenStandMixerHead 0.72/0.74、SlideDishwasherRack 0.24/0.48、TurnOnSinkFaucet 0.06/0.28、PickPlaceCounterToCabinet 0.42/0.42、PickPlaceCounterToStove 0.90/0.94、PickPlaceDrawerToCounter 0.38/0.68、PickPlaceSinkToCounter 0.86/0.94、PickPlaceToasterToCounter 0.74/0.74。掉点集中在 PickPlaceDrawerToCounter(−0.30)/SlideDishwasherRack(−0.24)/TurnOnSinkFaucet(−0.22)/OpenCabinet(−0.18)。已拉回 `exp/nfe_baseline/data/runs/rc/groot_tp/{main,pnp}/k1/`，聚合 `exp/nfe_baseline/data/agg_groot_rc.json`（`--anchor-sr 0.680 --anchor-n-ep 650 --expect-episodes 650`）。server 空闲 400 s 后自动切 k=2。
- **15:06 π0.5 k=1 main lane 完成**（400/400，49 min，32 worker）：8 任务 143/400，macro **0.3575** vs 10 步参考（v2 台账）同 8 任务 0.4825（−12.5 pp）。逐任务 k=1/参考 v2：CloseBlenderLid 0.00/0.06、CloseFridge **0.00/0.62**、CoffeeSetupMug 0.14/0.34、OpenCabinet **0.28/0.70**、OpenDrawer 0.78/0.68、OpenStandMixerHead 0.42/0.26、SlideDishwasherRack 0.46/0.44、TurnOnSinkFaucet 0.78/0.76。→ flow-matching 头在 RoboCasa365 上 k=1 同样明显掉点（不是 GR00T 独有），且掉点任务与 GR00T 不同（π0.5 崩在 CloseFridge/OpenCabinet/CoffeeSetupMug，GR00T 崩在 PickPlaceDrawerToCounter/SlideDishwasherRack/TurnOnSinkFaucet）。pnp lane 接着跑。
- **15:36 π0.5 k=1 全 13 任务完成**（main 400 + pnp 250，1 h 19 min，32 worker）：macro **0.391**（IR 59.7%）vs 10 步参考 v2 0.557（**−16.6 pp**）。pnp 逐任务 k=1/参考：PickPlaceCounterToCabinet 0.56/0.74、PickPlaceCounterToStove 1.00/0.84、PickPlaceDrawerToCounter 0.10/0.30、PickPlaceSinkToCounter 0.56/1.00、PickPlaceToasterToCounter **0.00/0.50**。三个任务归零（CloseBlenderLid/CloseFridge/PickPlaceToasterToCounter），四个任务反而略升（OpenDrawer/OpenStandMixerHead/PickPlaceCounterToStove +0.10~0.16）——典型的一步=条件均值：均值落在可行动作里的任务不掉，多峰任务崩。对照 RIT 原始台账 K=1 cache 臂：IR 52.6→0.503、IR 66.8→0.548，同 IR 段 cache 比 1 步 teacher 高 11–16 pp。已拉回 `runs/rc/pi05/{main,pnp}/k1/`，聚合 `exp/nfe_baseline/data/agg_pi05_rc.json`（anchor 0.5569 v2）。server 空闲后自动切 k=2。
- **16:47 GR00T k=2 全 13 任务完成**（1 h 40 min）：macro **0.652**（IR 83.0%）vs 4 步 0.680（−2.8 pp，在 ±2.4 pp 抽样噪声边缘）。k=1 崩掉的任务在 k=2 基本回来：SlideDishwasherRack 0.24→0.48（参考 0.48）、TurnOnSinkFaucet 0.06→0.24（0.28）、PickPlaceDrawerToCounter 0.38→0.56（0.68）、OpenCabinet 0.80→0.90（0.98）；CoffeeSetupMug 仍 0.56（0.68）。RIT 台账同 IR 段：K=2 臂 IR 81.4→0.648、K=3 臂 IR 85.0→0.660、K=1 臂 IR 77.5→0.694——k=2 teacher 与 cache 打平，cache 的优势集中在 IR<75。聚合 `agg_groot_rc.json` 已含 k=1,2。server 空闲后切 k=3。
- **17:09 π0.5 k=2 全 13 任务完成**（1 h 26 min）：macro **0.488**（IR 64.2%）vs 10 步 0.557（−6.9 pp）。k=1→k=2 回升明显但没回满：CloseFridge 0.00→0.12（参考 0.62）、OpenCabinet 0.28→0.52（0.70）、PickPlaceToasterToCounter 0.00→0.34（0.50）、PickPlaceSinkToCounter 0.56→1.00（1.00）、CoffeeSetupMug 0.14→0.30（0.34）；PickPlaceDrawerToCounter 仍 0.12（0.30）。RIT 台账同 IR 段：K=1 臂 IR 66.8→0.548、K=3 臂 IR 65.4→0.526、K=2 臂 IR 67.0→0.517——k=2 teacher（0.488）仍低于所有 cache 臂 3–6 pp。聚合 `agg_pi05_rc.json` 已含 k=1,2。切 k=3。
- **18:20 再均衡阈值改为 A=6 / B=7**：pair A 实测每 k ≈ 1 h 20–30 min（k=1 1:19、k=2 1:26、k=3 main 58 min），GR00T lane（同 30 worker 拓扑）每 k ≈ 1 h 40 min，预期 pair B 跑 π0.5 也在这个量级 → 剩余 4..9 按 A=4,5,6 / B=9,8,7 三三分，双方都在 ~23:30 收工。B 第一个 k 出实测速度后再校。
- **18:41 GR00T k=3 全 13 任务完成，GR00T 阶梯收官**（k=3 1 h 47 min）：macro **0.665**（IR 91.5%）vs 4 步 0.680（−1.5 pp，噪声内）。GR00T RoboCasa365 阶梯终表：k=1/2/3/4 = **0.572 / 0.652 / 0.665 / 0.680** @ IR 74.4 / 83.0 / 91.5 / 100。悬崖只在 1→2 步之间（+8 pp），2→4 步平缓（+2.8 pp）。聚合 `agg_groot_rc.json` 三档齐、`runs/rc/groot_tp/{main,pnp}/k{1,2,3}/` 齐。t108 client ladder 已 RECHAIN 成 pair B（π0.5 9,8,…），h100 server 空闲 400 s 后 `LADDER FINISHED` → 看门自动起 π0.5 :23231。
- **18:49 π0.5 k=3 全 13 任务完成**（1 h 31 min）：macro **0.525**（IR 68.7%）vs 10 步 0.557（−3.2 pp）。CloseFridge 仍只有 0.18（参考 0.62）、OpenCabinet 0.54（0.70），其余任务基本回到参考。RIT 台账同 IR 段：K=1 臂 IR 66.8→0.548、K=3 臂 65.4→0.526、K=2 臂 67.0→0.517——k=3 teacher 与 K=2/K=3 cache 臂打平，仍低于 K=1 臂 2 pp。π0.5 阶梯至此 k=1/2/3 = 0.391/0.488/0.525。
- **18:49 pair B 起跑**：h100 :23231 `KSWEEP num_steps=9`，t108 30 worker 接上（31 连接），k=9 main 从 36 集续跑。pair A 切 k=4。
- **19:25 再均衡阈值改为 A=5 / B=6**：pair B k=9 main 400 集只用 35 min（h100 不是瓶颈，A5000 渲染快），估 B ≈ 1 h/k；pair A k=4 main 27 min 才 120 集（4090 100%，stage3 随 k 变贵），估 A ≈ 2 h/k。→ A 跑 4,5（~22:55 完），B 跑 9,8,7,6（~22:50 完）。
- **19:47 π0.5 k=9 全 13 任务完成（pair B 第一档，58 min 含续跑）**：macro **0.595**（IR 95.5%）vs 10 步 0.557（**+3.8 pp**，在噪声内偏上；CloseBlenderLid 0.20 vs 0.06、CoffeeSetupMug 0.48 vs 0.34）。⚠ 注意 pair B 用 h100 server（bf16 数值路径与 4090 不同）且 t108 worker 的 A5000 渲染；k=9 与 10 步参考差在 ±1 σ 内，暂按抽样噪声处理，全部跑完后看 k=6..9 是否整体高于参考再判是否有系统偏移。π0.5 阶梯至此 k=1/2/3/9 = 0.391/0.488/0.525/0.595。
- **20:35 π0.5 k=4 全 13 任务完成**（pair A，1 h 38 min）：macro **0.523**（IR 73.2%）vs 10 步 0.557（−3.4 pp），与 k=3（0.525）持平。π0.5 阶梯至此 k=1/2/3/4/9 = 0.391/0.488/0.525/0.523/0.595。图已更新（`rc365_cache_vs_steps_all13.png`）。
- **20:41 切档机制改为信号驱动（owner 20:3x 斥责空闲超时切档）**：新增 `exp/nfe_baseline/ops/kdone_listener.py`（server 机 TCP 监听，收到 `DONE k=<k>` 就 touch `/tmp/nfe/kdone_<policy>_<k>`）、`ladder_server2.sh`（等待循环里见到该文件立即 `LADDER k=<k> SIGNAL` 切档，空闲 400 s 只作兜底；已在跑的 server 直接接管）、`ops/rc/sig_client.sh`（client 机盯 lane 日志，新出现的 `RCLANE k=<k> DONE` 转发到 server 机）。已在 pair B 上线：h100 `nfesig_prc2`（:23232，tag pi05_rc）+ v1 ladder 换成 v2（ks 8,7,6，接管 k=8 server）；t108 `nfesig_prc2` 转发到 149.165.153.233:23232，TCP 连通已验。pair A 只剩 k=5 没有切档，不改。v1 的空档 ≈ 8–9 min/档（400 s 空闲判定 + 1.5–2 min 装模型 + 30 s 探测），v2 ≈ 2 min（只剩装模型）。**以后阶梯默认 v2；更彻底的是 server 按请求带 k（Cosmos 线已是）。**
- **20:49–20:54 pair A k=5 事故**：wls k=5 server 起来后 client 32 worker 全部 `keepalive ping timeout`（server 端 handshake 正常、GPU 0%、进程 12 min CPU 后空转、245 线程全在等锁）——batched 拦截器在 k=5 首调用期间（疑似 reduce-overhead 图捕获拖长）client 连接被 20 s keepalive 掐断后死锁，之后再也不出结果。处置：20:53 `stop_servers` + 重拉 k=5 server（v1 ladder 空闲计数未到 400 s，仍在 k=5），20:53:53 重开 t107 k=5 main lane（旧 worker 随 conductor 退出，无孤儿；旧日志改名 `stuck_k5_main_*.log`），首调用后 33 连接、GPU 100%、journal 正常增长、零 traceback。journal 里前一轮那 1 行 done 记录保留（resume）。
- **20:56 pair B k=8 完成，信号切档首次实跑**：client DONE 20:56:23 → 信号 20:56:26 → server `LADDER k=8 SIGNAL` 20:56:31 → k=7 UP 20:57:32 → client k=7 UP 20:57:54，**切档 1.5 min**（v1 是 8–9 min）。π0.5 k=8 = **0.554**（IR 91.0%）vs 参考 0.557（持平）。阶梯至此 k=1/2/3/4/8/9 = 0.391/0.488/0.525/0.523/0.554/0.595。
- **21:55 π0.5 k=7 完成（pair B，58 min）**：macro **0.569**（IR 86.6%）vs 参考 0.557（+1.2 pp，噪声内）。信号切档第二次：k=7 DONE 21:55:54 → server 切 21:56:05 → k=6 UP 21:57:06。pair A k=5 main 完成 21:55:47（重启后 62 min），pnp 起。阶梯至此 k=1/2/3/4/7/8/9 = 0.391/0.488/0.525/0.523/0.569/0.554/0.595，剩 k=5（A pnp 中）、k=6（B 刚起）。
- **22:30 π0.5 k=5 完成（pair A 收官，重启后 1 h 36 min）**：macro **0.554**（IR 77.6%）vs 参考 0.557（持平）。pair A 停：t107 lane 自停（STOPPED after k=5），wls 端我手动执行了看门动作（杀 v1 ladder + stop_servers，不等 400 s 空闲），22:32 Cosmos RC24 replica 开始在 :23180-23183 起。π0.5 阶梯至此 k=1/2/3/4/5/7/8/9 = 0.391/0.488/0.525/0.523/0.554/0.569/0.554/0.595，只剩 k=6（pair B，main 366/400）。
- **22:55 π0.5 k=6 完成 → π0.5 阶梯收官（pair B，58 min）**：k=6 = 0.551（IR 82.1%）。**π0.5 RoboCasa365 阶梯终表（13 任务 × 50 集，10 步参考 v2 0.557）**：k=1/2/3/4/5/6/7/8/9 = **0.391 / 0.488 / 0.525 / 0.523 / 0.554 / 0.551 / 0.569 / 0.554 / 0.595** @ IR 59.7/64.2/68.7/73.2/77.6/82.1/86.6/91.0/95.5。悬崖在 1→2→3 步（+13.4 pp），k≥5 全部落在参考 ±2.5 pp 的噪声带内（k=9 偏高 +3.8 pp）。h100 侧信号切档 + `LADDER FINISHED` → 看门自动起 Cosmos replica（23:55 h100 本地时 = 22:55）。聚合 `agg_pi05_rc.json` 九档齐、`runs/rc/pi05/{main,pnp}/k1..9/` 齐，终图 `rc365_cache_vs_steps_all13.png` 已重画。
