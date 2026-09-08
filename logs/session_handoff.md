# Session Handoff —— W13 语料重采完成，下一步做实验

> 2026-09-08 上午。**四条线的 warm-start 语料重采 + pkl 建库切分全部完成并验收通过**。
> 语料正由**另一个会话**搬到 `/archive`（原位留软链，路径不变），本会话不必管。
> 本会话的下一步：**用这批新 pkl 跑实验**。

---

## 0. 接手第一步

```bash
cd /home/weiland/projects/openpi            # 分支 Ziyang
cat logs/robocasa365_warmstart_plan.log.md  # 权威 plan（§1.8 两线差异、§3.2 W14、Review Log）
git status --short | grep -v rit_pareto      # exp/rit_pareto/* 属另一 session，永不碰
```

Authority = Execution，只读 `protocols/execution_authority.md`。跑实验是运行动作，不需要新 G1/G2。
纪律：**不 `git add`/commit/push** 除非 owner 当次明示；commit 英文、无 AI 署名；跨机同步一律 `tether`，不用 git。

---

## 1. 产物：四套 pkl（全部已回环校验，`bad=0`）

**RoboCasa** `/data/robocasa365_cache/cache_artifacts_w13/`

| stem | schedule | full | 档位（轨迹数） | 几何 |
|---|---|---|---|---|
| `groot_tp_spatial_pool_16_w13` | `groot_n15_k4_v1` | 650 集 / 46,216 条 / 19.0 GB | S1-S6 = 13/26/65/130/260/650 | 3 相机 / 20 维 state |
| `pi05_spatial_pool_16_w13` | `pi05_v1` | 650 集 / 60,096 条 / 28.0 GB | 同上 | 3 相机 / 32 维 state |

两条线各另有 `_pnp_pinned.pkl`（pick 族 5 任务 × 50 集，戳 `pin_id=4d13ac5e…`）：GR00T 17,493 条 / 7.2 GB，pi0.5 21,963 条 / 10.2 GB。

**LIBERO** `/data/libero_cache/libraries_w13/{libero_spatial,libero_10}/`

| stem | schedule | full | 档位（轨迹数） |
|---|---|---|---|
| `libero_spatial_w13` | `groot_n15_k8_v1` | 450 集 / 9,638 条 / 4.0 GB | S1-S6 = 10/20/50/100/200/450 |
| `libero_10_w13` | `groot_n15_k8_v1` | 436 集 / 22,662 条 / 9.5 GB | S1-S6 = 10/20/50/100/200/436 |

共 30 个 pkl / 185.8 GB。⚠ **S6 与 full 内容等价**（名义 k=50 在各任务成功数处触顶），保留两份只为将来重切档位时不必再扫语料。

**建库口径（复现/重建时必须照抄）**
- 切分：`seed=0` 确定性洗牌、**整条 episode 为单位**、每任务取 `min(k, n_t)`、逐档真包含；x 轴报**实测均值**不是名义 k。
- pi0.5 建库必须 `--vision-slots 3` + text-IVF 三旗标 `--prompt-masked-pool --prompt-instruction-span --discrete-state-input`（manifest 里 `prompt_pool={masked:True, instruction_span:True}` 是它生效的证据）；GR00T 侧不带这三个（prompt_emb 天然位稳定）。
- 全部用 `--trajectory-id-mode relpath`（RoboCasa 的 `<teacher>/<Task>/episode_NNNN_aAA` 会跨任务撞 stem）与 `--expected-schedule`。

---

## 2. 语料现状（另一会话在搬，路径不变）

| 语料 | 位置 | 规模 |
|---|---|---|
| RoboCasa W13（13 任务 × 2 teacher） | `/data/robocasa365_cache/build_l1s1_w13/<teacher>/<Task>/` | 1.2 T |
| RoboCasa 多采的 5 个任务 | `…/build_l1s1_w13_extra/`（已移出正式集） | 193 G |
| LIBERO spatial / libero_10 | `/data/libero_cache/build_{spatial,libero10}_w13/` | 26 G / 63 G |

归档惯例：搬到 `/archive/<同名相对路径>` 后**原位留软链**——LIBERO 在 build 根一条，RoboCasa 在**每个任务目录**一条（`<root>/<teacher>/<Task>` 级），所以 h5 的绝对路径逐字不变，读的人无感。`/archive` 是 host-managed SMR，单流串行。

**schema v2**：每个 h5 file-level 带 `denoise_schedule_id` / `denoising_num_steps`（取自活值），每 step 带 `noise_action_0..N-1`（0 是起点噪声）。四份审计全 `ok=True`，`schema_errors / pin_errors / missing_file / missing_terminal / multiple_accepted` 均 0。

---

## 3. 数据规模与成功率（决定 tier 上界）

- RoboCasa（13 任务，每任务恰好 50 条进 manifest）：GR00T 全部 ≥50；pi0.5 逐任务成功 50-91。
- LIBERO spatial：500/500 init 全覆盖，SR 90.0%，逐任务成功 38-49。
- LIBERO libero_10：500/500 全覆盖，SR 87.2%，逐任务成功 33-48（task 8 `both moka pots` 最低 33）。
- ⚠ **S5(k=20) 是最后一个所有任务都达名义值的档**；S6 触顶。做 cache-size 曲线时 x 轴用 manifest 里的 `realized_mean`。

---

## 4. 未入库的代码（本轮为完成任务所写，owner 未授权提交）

| 文件 | 作用 |
|---|---|
| `exp/robocasa365/merge_collection_manifests.py`（新） | 合并多批审计 manifest，`--only-tasks` 收窄到正式任务集 |
| `exp/robocasa365/build_size_libraries_rc.py`（新） | RoboCasa 版档位切分（复用 libero 版的 `verify_tier`/seed 口径，任务名从 relpath id 取） |
| `exp/robocasa365/verify_collection_artifacts.py`（改） | 加 `--only-tasks`：把期望 uid 集收窄到正式任务集 |
| `exp/robocasa365/config/collect_w13_timan107.env`（新） | W13 采集部署参数（6 GR00T + 6 pi0.5 端点） |

另有更早的未入库物：`analysis/pnp_pinned_results.md`、`config/ws_search2_pnp/`、`config/calibration_normalizers_pnp_pinned.json`、`config/collect_pnp_*.env`、`logs/pnp_run_progress.md`。

---

## 5. 下一步实验前必须知道的坑

1. **conductor 一个 yaml 绑一个 server**（`driver.assign_servers` 返回 `yaml_id -> 单个 endpoint`）⇒ 单个任务永远只用一个 worker。收尾只剩一个任务时并行度掉到 1/N；要并行只能起**多个 driver 各切一段**（本轮这样提速 5.7×）。
2. **`task_uid` 含任务序号**（`<run>__<Task>:eval:<task_ordinal>:<episode_idx>`）⇒ 补批必须保持**同样的任务顺序**（不需要的任务写 `:0` 占位），否则整批行与账本对不上。审计器要求各批 uid **不相交**。
3. **`pgrep -f` 自匹配**：命令正文任何位置出现匹配串（含注释、sed 模式）都会匹配自己 → 杀掉 tether shell。更狠的是 `tmux new-session -d "<整条命令>"` 派生的 **tmux server 进程 argv 里带着那条命令**，误杀它会带走该机所有 tmux 会话。定式：模式写在**远端脚本文件**里、只杀 `readlink /proc/<pid>/exe` 是 python 的进程（见 `/tmp/w13/kill_lane.sh`）。
4. **tether agent 的 HOME 是 `/srv/local/<user>/tether-home`**，tmux 会话继承它 ⇒ LIBERO 找不到 `~/.libero/config.yaml` 会**弹交互提示卡死**。跨机跑 LIBERO client 必须在 tmux 内层写死 `HOME=/home/zixuans8 LIBERO_CONFIG_PATH=/home/zixuans8/.libero`。
5. **`ionice -c 3`（idle）在有写盘负载时会被彻底饿死**（实测 0.9 MB/s），且非特权进程**无法把自己提回**普通优先级——只能杀掉重起。长任务别用 idle 类。
6. **EGL context 残留**：被杀的 LIBERO lane 会在那张 GPU 上留 ~2 GB context，新 lane 在同卡起会 `EGLError`。换一张空卡即可。
7. LIBERO 跨机（client@timan107 → server@ziyanglin.com）实测 **13-15 s/集**，比同机还快，网络不是瓶颈。
8. 机器纪律：server 只在 weilandserver（公网段 23100-23199，1:1 NAT），worker 只在 timan107；`tether exec` 单次 ~10 min 上限，长跑一律 tmux + tee。

---

## 6. 监控体系（本轮验证好用的形态）

- **cron 定时巡检**：固定间隔跑远端一行 health 脚本，健康只在主会话记一行。
- **Monitor 条件触发**：轮询同一个 health 脚本，只在 ALERT / DONE / STALL / 里程碑时出声；用 `until <条件>; do sleep; done` 等一次性事件。
- health 脚本写在远端 `/tmp/w13/h*.sh`，判「driver 退出」看日志**最后一行**是不是退出标记（不能数累计次数——重启会追加到同一日志）。
