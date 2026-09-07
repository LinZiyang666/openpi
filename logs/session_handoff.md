# Session Handoff —— GR00T/pi0.5 warm-start 数据重采集（W13）

> 2026-09-07 凌晨。阶段 B 代码已入库放行，阶段 A 只留三段延迟。
> **下一步工作：三条线重采数据** —— RoboCasa365 × pi0.5、RoboCasa365 × GR00T、LIBERO × GR00T（spatial + libero_10）。
> 原因：现存语料全部**没有去噪快照**（`noise_action_*`）且没有 schedule 戳，warm-start 库无法从旧语料建出。

---

## 0. 接手第一步

```bash
cd /home/weiland/projects/openpi            # 分支 Ziyang，与 origin 同步
cat logs/robocasa365_warmstart_plan.log.md  # 权威 plan（§1.8 两线差异、§3.2 W14、Review Log）
uv run pytest tests/cache/groot tests/collect tests/robocasa365 tests/libero_groot -q   # 应全过（既有失败仅 test_prebuilt_matrix_backend ×2）
git status --short | grep -v rit_pareto      # exp/rit_pareto/* 是另一个 session 的，不碰
```

Authority = Execution，只读 `protocols/execution_authority.md`。W13 是**运行动作**，不需要新 G1/G2；
若要改采集代码，按 WA 定级（`exp/` 脚本 L1）。**不 `git add`** 除非 owner 明示；commit 英文、无 AI 署名。

---

## 1. 现状一句话

- 代码：`2e51b02`（阶段 B 全链路，G2 在 owner override 下 CODE APPROVED）+ 后续 `00d2739…ba0eb4e`（bench 修正、阶段 A 记录）。
- 阶段 A 结论只记三个数（CUDA Graph，三段各一张图）：**stage1 8.12 / stage2 9.36 / stage3 17.80 ms**。owner 裁定其余不记录。
- 未提交（owner 未指示）：`exp/robocasa365/analysis/pnp_pinned_results.md`、`exp/robocasa365/config/ws_search2_pnp/`、
  `config/calibration_normalizers_pnp_pinned.json`、`config/collect_pnp_{weilandserver,timan107}.env`、`logs/pnp_run_progress.md`。
  ⚠ 两个 `collect_pnp_*.env` 是上一轮采集的**精确部署参数**（端口组、worker 解释器、EGL 路径），重采要用，先入库。

---

## 2. 新采集 schema（v2）—— 重采要交付的东西

每个 h5 文件 file-level attrs：`denoise_schedule_id`、`denoising_num_steps`（取自**活值**，不是常量）。
每个 step group 除原有字段外新增：`noise_action_0`（纯噪声起点，D5）与 `noise_action_1..N-1`（第 i 步去噪前的 x_t）。

| 线 | schedule_id | N | 相机 / state 宽 | 谁写快照 |
|---|---|---|---|---|
| RoboCasa × pi0.5 | `pi05_v1` | 10 | 3 路 / 32 | `src/openpi/collect/collection_policy.py`（hook `action_in_proj`，写 `noise_action_0` + 戳，步数≠10 即拒） |
| RoboCasa × GR00T | `groot_n15_k4_v1` | **4**（ckpt 内置，server 不传步数） | 3 路 / 20 | `exp/robocasa365/groot_cache_collector.py`（hook `action_head.action_encoder`，每集重读活值，hook 计数≠N 即拒） |
| LIBERO × GR00T | `groot_n15_k8_v1` | **8**（`--denoising-steps 8` CLI 默认） | 2 路 / 8 | 同上一个采集器（`serve_groot_libero.py:400` 直接构造它） |

⚠ 步数是**运行时属性**：GR00T 的 id 由 `GrootStagedRunner.live_schedule()` 派生，代码里没有字面 4/8。
⚠ run-plan 的 hashed params 现固定带 `collect_schema: v2` ⇒ **旧 journal 不能续进新采集**；重采必须换新的 collect_root / run id，不要在旧目录上"续跑"。
⚠ RoboCasa pi0.5 旧语料本来就有 `noise_action_1..9`，缺的是 `noise_action_0` 与戳；GR00T 两线旧语料**零快照**。

---

## 3. 采集配方

### 3.1 RoboCasa365（两个 teacher，同一套 conductor）

- 入口：`exp/robocasa365/run_collect.py --role all --teacher {pi05|groot_tp} --servers … --tasks … --layout 1 --style 1
  --base-seed 0 --collect-root <scene-root> --env-config <env> [--pinned-objects exp/robocasa365/config/pnp_pinned_objects.json]`；
  完整说明 `docs/data_collection/guide.md` §"RoboCasa365 teacher-library collection"。
- 拓扑（**一 server ↔ 一连接 ↔ 一 worker**，`--collect` 与并发互斥）：server 全在 weilandserver（4090 48G），
  driver + sim worker 在 timan107（`/scratch/zixuans8/openpi_rc365`，8×1080，原生 EGL）。**两个 teacher 不能同时在 48G 卡上**
  （GR00T 5×8.5G / pi0.5 4×10.5G）。上一轮：pi0.5 4 server @8010-8013（公网 23110-23113），GR00T 5 server @8020-8024（23120-23124），
  参数全在 `config/collect_pnp_{weilandserver,timan107}.env`。
- server 命令：pi0.5 = `uv run scripts/serve_policy.py --port <p> --non-concurrent --collect --collect_dir <scene-root> policy:checkpoint --policy.config pi05_robocasa --policy.dir /home/weiland/ckpt_pi05_robocasa_pytorch`；
  GR00T = `serve_groot_n15.py --port <p> --collect-hdf5 <scene-root> --checkpoint /home/weiland/ckpt_n15_robocasa_tp/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000`
  （GR00T venv `/home/weiland/gr00t_n15_venv/.venv/bin/python`，`PYTHONPATH=/home/weiland/gr00t_n15:<repo>/src:<repo>`，`HF_HUB_OFFLINE=1`）。
- 上一轮（pnp 定物体线）的任务集与规模：5 个 PickPlace 任务、`pnp_pinned_objects.json`（pin_id 4d13ac5e…）、每任务目标 50 条成功、
  `--layout 1 --style 1 --base-seed 0`（评测段用 1,000,000，两段不相交），采集根 `/data/robocasa365_cache/build_l1s1_pnp`（CMR 盘 /data，**不要放叠瓦盘 /archive**）。
  ⚠ 是否仍限 PickPlace / 是否钉物体 / K 值，由 owner 定；**`--pinned-objects` 一旦用就必须传给该 run 的每一个 driver**。
- 审计与入库：`verify_collection_artifacts.py --run-plan … --journal … --require-denoise-schedule --manifest-out …`
  （新 flag 让无戳文件成为 schema 错误；戳存在时逐 step 核对 `noise_action_0..N-1` 的基数、连续性、shape/dtype 对 `clean_action`）
  → `build_in_memory_cache_artifact.py --builder-type {cp1_spatial_pool_16|cp1_groot_spatial_pool_16} --expected-schedule <id> --manifest …`
  ⚠ pi0.5 建库必须 `--vision-slots 3`（默认 2，第三路会被**静默丢掉**，到 emit 才炸）；GR00T builder 默认 (3, 20)。
  → `calibrate_score_normalizers.py`（按 pkl stem 键入）→ warm-start yaml 只由 `exp/robocasa365/emit_ws_warmstart_yamls.py` 产出
  （`--groot-steps` 无默认必须显式，产物根 `config/ws_warmstart_pnp/`，库 tag `pnp_warm`，自带 digest）。

### 3.2 LIBERO × GR00T（spatial + libero_10，weilandserver 单机闭环）

- 权威 runbook：`logs/libero_groot_collection.log.md`（§5 参数、§7 铁律、§10 建库）。一键：
  `bash exp/libero_groot/launch_collection.sh <suite> <ckpt> <out-dir> [lanes=6] [base-port=8030]`
  （spatial：`libero_spatial /home/weiland/ckpt_n15_libero_spatial /data/libero_cache/build_spatial`；
  l10：`libero_10 /home/weiland/ckpt_n15_libero_10 /data/libero_cache/build_libero10`）。**重跑即续跑**（按已落盘 h5 重算分片）
  ⇒ 重采必须换 out-dir（或清空旧目录），否则脚本会把旧无戳文件当成已采完。
- server 走 `serve_groot_libero.py --collect-hdf5 … --denoising-steps 8`（默认 8）；client 是 `examples/libero/main.py`，
  6 路 × `--episode-filter` 分片，B 池 init `exp/common/data/db_init/libero/<suite>`（**不得含 `.pruned_init`**，脚本会拒）。
  6 路实测 15.6 ep/min；spatial 500 集约 35 min，l10 约 1.5–2 h、95–100 GB。
- 审计/建库：`report_collection.py <h5dir> --trials 50 --num-tasks 10`；
  `build_size_libraries.py … --denoise-schedule groot_n15_k8_v1`（透传为 builder `--expected-schedule`，manifest 记 schedule）；
  `verify_libraries.py <manifest> --expected-schedule groot_n15_k8_v1`；warm yaml 由 `exp/libero_groot/emit_warmstart_yamls.py --suite … --library … --denoising-steps 8` 产出，
  eval 入口（`orchestrate_search.py` / `run_conductor.py`）会先 `verify_warm_sweep` 校验 index/digest/schedule/库。
- ⚠ GR00T→LIBERO 夹爪必须走官方 `normalize_gripper_action`（漏掉 = 静默 0%）；replan_steps=5；keepwarm 脚本不许关。

---

## 4. 远端仓状态（开工前必须先对齐）

| 机器 | 路径 | 状态 |
|---|---|---|
| weilandserver | `/home/weiland/openpi` | 分支 Ziyang 停在 **9da3983**（落后 origin 多个提交），且有**别人的未提交改动**，其中 **`exp/robocasa365/run_collect.py` 与我的提交重叠** ⇒ `git pull --ff-only` 被拒。**不要 stash 别人的东西**：先问 owner，或看那处改动能否直接 commit/丢弃。阶段 A 用的是 `/tmp/openpi-stageA` 浅克隆（可删）。 |
| timan107 | `/scratch/zixuans8/openpi_rc365` | 停在 **3598534**，没有新采集代码（`collect/`、`groot_cache_collector.py`、`run_collect.py`、`verify_collection_artifacts.py`、`cache/groot/staged.py`…）。上一轮是 tether push 逐文件补的；这次建议 `git pull`。`/scratch` 不在 tether allow_roots，push 须经 `/tmp` 中转。 |

两机对齐后用 sha256 逐文件对账（上一轮的做法），再起 server。

---

## 5. 纪律与本线踩过的坑

- **`pgrep -f`/`pkill -f` 自匹配**：模式本身要用字符类（`[w]orker_entry`），且**脚本正文任何地方（含注释、echo）都不能出现裸的匹配串**，
  否则 `bash -lc '<script>'` 的 argv 含该串 → 杀掉自己的 tether shell（本线连踩两次）。共享机禁宽模式 pkill，按端口/tmux 名/PID 定点。
- weilandserver 23100-23199 端口段与 `srvN` tmux 名是**多 session 共享**命名空间；自己的 tmux 起别的名字。
- `tether exec` 单次约 10 min 上限，长跑一律 tmux + `tee` 日志；`tether push` 目标已存在要 `--force`；别把 stderr 重定向掉。
- Bash 工具单次 120 s 超时，远端等待循环要拆短。
- 采集期间两机别的 session 可能在跑：起 server 前 `ss -tlnp` 侦察端口、`tmux ls` 侦察归属，别人的一律不动。
- 数据一律落 `/data`（weilandserver）；HDD 顺序流要串行化。
- 删除类操作前先 `wc -l` 核对清单规模。
