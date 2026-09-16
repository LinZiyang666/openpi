# PickPlace 定物体重做 — 进度

图例：`[x]` 完成  `[~]` 进行中  `[ ]` 待做  `[!]` 阻塞

## 流程
- [x] Understand / Plan / G1(4 轮 15 blocking 全闭合)
- [x] Code(W1 W2 W3 W5 W7 W9)
- [x] G2(2 轮 6 blocking 全闭合，APPROVED)
- [x] §6 Verify 裸全量 pytest：5159 passed / 15 failed / 60 skipped（17m07s）
- [x] 15 条失败逐条归因：**零条属于本线**（5 条在 2026-08-24 干净 HEAD 实测清单上；1 条是 `docs/iclr` untrack 造成；10 条 review_tests 测的四个模块本线均未改动，id 交 Review Authority）
- [x] Commit `9da3983`（30 文件 +4638/−261，无 AI 署名，零 rit_pareto）
- [x] Push origin/Ziyang（`7e77ad2..9da3983`）

## 实验运行
- [x] 投递：weilandserver 15 文件、timan107(openpi_rc365) 7 文件，sha256 全对账
- [x] 拓扑改正：**server 只在 weilandserver**(5×GR00T @ ziyanglin.com:23120-23124)，**driver+worker 全在 timan107**
- [x] W4-groot 采集**完成**：448 attempts / 291 success，5/5 达标(Cabinet55 Stove54 Drawer68 Sink58 Toaster56)。b01 404 + 续批 b02 44(Cabinet only, --episode-lo :89)
  - 审计 PASS：`ok=true`，expected=448，admitted=291，**insufficient / pin_errors / schema_errors 三者全空**，missing_file=0 missing_terminal=0 multiple_accepted=0；pin_id 4d13ac5e… 一致，plan_hashes 两批；报告与 manifest 已归位 `/data/robocasa365_cache/build_l1s1_pnp/pnp_{audit,manifest}_groot.json`
- [x] W4-pi05 采集**完成**：483 attempts / 303 success，5/5 达标(Cabinet56 Stove59 Drawer53 Sink58 Toaster77)。b01 441 + b02 42(Cabinet:23@lo81 + Drawer:19@lo106)
  - 审计 PASS：`ok=true`，expected=483，admitted=303，**insufficient / pin_errors / schema_errors 三者全空**，missing_file=0 missing_terminal=0 multiple_accepted=0；manifest 五任务各 50 条；orphan_attempts=628 = 931 总 h5 − 303 admitted(含另一 teacher 目录，符合预期)；产物归位 `/data/robocasa365_cache/build_l1s1_pnp/pnp_{audit,manifest}_pi05.json`
- [~] W6 建库 ×2 + 重标定 + emit 132×2 + digest 冻结
  - a1 GR00T 建库**首次失败**：`TypeError: _CP1BaseKeyBuilder.__init__() got an unexpected keyword argument 'prompt_masked_pool'` —— weilandserver 的 openpi 落后 origin 35 commit，`build_in_memory_cache_artifact.py` 是本线投递的新版但它依赖的 `src/openpi/cache/` 是旧版
  - 修复 09-06 09:05：比对 `src/openpi/cache/` 全部 60 个 .py，找出 17 个差异(其中 7 个远端缺失)，逐个 `tether push --force` 投递后复算 sha256 **60/60 全等**；别人的两个脏文件 `cache/groot/{load_guard,staged}.py` 未触碰(git status 仅此两条)；`exp/common` 仅差一个与建库无关的 `data/db_init/sample_cache.py`
  - a1 **GR00T 库建成** 09-06 09:16：250/250 files、**17499 entries**、`groot_tp_spatial_pool_16_pnp_pinned.pkl` 7,068,321,704 B(6.58 GiB)，err 0。耗时约 9 分钟(88 核并行)
  - a2 **pi0.5 库建成** 09:34：250/250 files、**22148 entries**、`pi05_spatial_pool_16_pnp_pinned.pkl` 10,329,005,768 B(9.62 GiB)，err 0。两库合计 16.2 GiB(GR00T 17499 / pi05 22148 entries)
  - b warm up 重标定第一轮完成 10:12(7 字段全 zscore、sat 全 0)，但**基于错误的 pi05 库**，须随库重建重跑
  - ⚠ **发现并纠正:pi0.5 库漏了第三路相机** 09-06 10:17。`cp1_spatial_pool_16` 默认几何是 2 个 vision slot(help 明写 2 for Pi0.5 / 3 for cp1_groot_*)，而 RoboCasa365 的 pi0.5 h5 实际有三路(base + left_wrist + right_wrist，`vision_0/1/2` 各 (256,2048))。默认建出的库 `vector_dims` 只有 vision_0/1，**vision_2 被静默丢弃、不报错**，标定也只标了 3 个字段。直到 emit 才炸:`ValueError: cp1_spatial_pool_16: field 'vision_2' has no vector dim in calibration` —— 因为 132 格权重矩阵建在四字段上，grid3v 21 格 + grid4 35 格共 56 格专门覆盖三相机面。旧库已移到 /tmp/pi05_2slot_WRONG.pkl，正用 `--vision-slots 3` 重建(tmux bld1)
  - a2' **pi05 库(3-slot)重建完成** 10:34 并逐项验证：`vector_dims={vision_0/1/2:32768, prompt_emb:2048, robot_state:32}` **三路齐全**；`pin_id=4d13ac5e…` 与 `plan_hashes`(b01+b02)已 stamp；22148 entries；抽查 entry 的 `query_keys` 确认 vision_0/1/2 各 (32768,) float32 **实数据在**。⚠ 新旧库大小几乎相同(10,329,006,120 vs 10,329,005,768 B)——说明错误版**也存了 vision_2 的数据**，只是没写进 `vector_dims`，backend 查询时整字段略过，这正是源码注释说的 silent drop
  - b' **warm up 重标定第二轮运行中** 09-06 10:37(tmux calib0)，旧 json 已删
  - 巡检 09-06 11:15：重标定第二轮 **6/8 字段完成**。关键确认:pi05 段的字段列表已是 `['vision_0','vision_1','vision_2','robot_state']` **四个**(上轮只有三个)，修复生效。pi05 vision_0 J=.4073 / vision_1 J=.4015(上轮 vision_0 是 .4299，库结构变了故不同)。err 0，剩 pi05 的 vision_2 与 robot_state，约 20 min
- [~] W8 评测 132 格 ×2 teacher（GR00T 6 replica / pi05 4 replica，timan107 30 worker）
  - **GR00T cache 臂 09-06 12:28 CDT 点火**。拓扑：weilandserver 6× `serve_groot_n15 --concurrent --allow-dynamic-bundles --cache-config <bootstrap iso_robot_state.yaml>` @ **23160-23165**（tmux srv0-srv5，日志 `/tmp/w8g_srv{0..5}.log`）；timan107 driver（tmux drv0，pull port 23180，`--cells-per-batch 12` → 11 批）+ 6 队 agent × 5 worker = **30 worker**（tmux ag0-ag5，GPU 轮转铺满 8 张 1080）
  - 开跑前投递修复：weilandserver 的 `serve_groot_n15.py` 落后 origin、**没有 `_resolve_bundle` / `--allow-dynamic-bundles`**（先前跑着的 6 个是 teacher-only 形态，ws2 用不了）；连同 `serving/{batching_coordinator,stage_io}.py` 一并 push。timan107 补 7 个过期文件（含 `run_ws_search{,2}.py`、`emit_ws_search2_yamls.py`、`conductor/{agent,driver}.py`）+ 2 个缺失文件 + 265 份 yaml。两机 sha256 全对账
  - preflight（`--finalize-only`）PASS：整棵树**两个 teacher** 的 digest 校验通过、`pin_id=4d13ac5e…` 断言一致、132 份 run_plan 落盘
  - 首批实测：episode 时长 83-98 s，成功/失败均已出现；按 30 worker × ~90 s 估 5,280 集 ≈ **4.5 h**
  - 巡检 cron `517833c2`（每 23 分钟），含阶段推进指令（cache 臂 → 地板臂 → pi0.5）
  - **实测稳态 900 集/小时**（4 分钟窗口 137→197），episode p50 103.7 s / p90 138.7 s。212 条记录**全部 accepted，零 ERR、零重试、零 CUDA OOM、零 worker 重启**；`status: failed` 是任务未达成（SR 分母），非运行错误。族内 SR 10/197 ≈ 5.1%，与 round-1 GR00T 族内 macro SR 0.066 同量级
  - 编制上限的真正约束是 **timan107 八张 1080 的 8 GB 显存**（5 张只剩 16-123 MiB），不是算力（CPU 29% / load 17.6 of 48）⇒ 30 worker 已到顶
  - **owner 裁定 09-06 12:45：不启用 timan1 扩编**（瓶颈已转到网络侧），维持 weilandserver 6 server + timan107 30 worker 的现拓扑跑完
  - 时间账：cache 臂 5,280 集/teacher ≈ 5.9 h ×2，地板臂 250 集/teacher ≈ 0.3 h ×2，合计约 12.5 h
  - 告警层 Monitor `btxm8joou`（5 分钟一探，只在状态变化时出声）
  - 12:52 复测：**1000-1260 集/小时**（325→376 用 180 s），episode 时长无漂移（最近 30 集均值 110.4 s vs 最早 30 集 113.2 s）。中途一次"降速到 310/h"的判断是我把读数时刻记岔造成的误报，三机时钟实测完全一致，**无降速**
  - server 侧负载画像：weilandserver 六个 server 进程各 290-380% CPU（合计约 21 核 / 88）、GPU 利用率 52-87%、显存 35.5 GiB。检索代价随格子家族变化（iso 只查 20 维 robot_state，grid 家族查 3×32768 维），但未成为瓶颈
  - ⚠ 工具坑（两个脚本都踩了同一个）：`x=$(grep -c ... || echo 0)` 会输出**两行 0**，把后续字段整体位移；且 journal 的 status 值是**小写** `done/failed/err`，用大写 pattern 会恒为 0 —— 两个判据都会静默失效。`/tmp/w8_health_t107.sh` 与告警脚本均已修正
  - 修正后读数 12:52：journal 387，success 13 / failed 374 / **ERR 0**，SR ≈ 3.4%
- [ ] W8 teacher-only 地板臂 250 集/teacher
- [ ] W10 报告

## 已就位（准备阶段）
- [x] W1 补丁三机部署（be22d659 / 分支 pnp-pinned-objects / 他人文件零触碰）
- [x] 采集根 `/data/robocasa365_cache/build_l1s1_pnp`（CMR 盘，非叠瓦 /archive）
- [x] 两 teacher checkpoint + venv + 岛 A EGL 全部核实
- [x] weilandserver 端口段 23100-23199 与 tmux 全空
