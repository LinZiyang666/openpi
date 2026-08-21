# Session Handoff — RoboCasa365 跨场景 cache 实验（T7 评测就绪版）

> ⚠ **本文件不是 `logs/session_handoff.md`**（那份属 X14 RL Router session，勿动）。

**Status**: `Eval-prep` — 采集(T5)/建库(T6)/并发服务改造全部收官，**下一步 = T7 评测（L0 阶梯先行）**。
**日期**: 2026-08-21（覆写第五版；旧版内容已被本版取代。⚠ commit `9f9a1f0` 里的本文件是被共享工作树意外回退的 v4，以本版为准）

---

## 0. 一句话现状

数据、库、并发基础设施三线全清：双 teacher 各 13 task × ≥20 成功轨迹审计过 + manifest 钉死；{pi05, groot_tp}×{n20/n10/n5} 六件套 pkl 建成；GR00T 并发服务改造经外审 G2 四轮 APPROVED 并 ship（`9f9a1f0`，本地=origin）；timan107 sim 车队部署完毕、weilandserver 直连公网段可用。**T7 已开：并发双连接真机冒烟 PASS（2026-08-21）；下一步 = 评测臂设计裁决 + 薄 eval driver + L0 阶梯。**

## 1. 接手第一步

1. 本文件读完；Authority = Execution（读 `protocols/execution_authority.md`）。
2. plan 权威：主线 = [`robocasa365_framework_integration.log.md`](robocasa365_framework_integration.log.md)（冻结契约 §4.3.x、N 表 §7.2）；并发改造 = [`groot_concurrent_serving_plan.log.md`](groot_concurrent_serving_plan.log.md)（`Shipped`，含 3 专家自审 + 外审 G2 四轮 Review Log）。
3. T7 尚无 plan——评测臂设计含科学决策（§4 待裁），**流程形式先问 owner**（正常 L2 还是 fast-track）。

## 2. 资产清单（全部已验证在位）

### 数据（weilandserver `/data/robocasa365_cache/`，盘 3.6T 余 ~2.8T）

- **`build_l1s1/pi05/`** 774 ep（322G）：批1 715+批2 59；审计 ok:True（13/13 task ≥20 成功，零缺失/零 missing_terminal/零 multi/零 schema 错）。逐 task 实测 SR：CloseBlenderLid 0.14、CloseFridge 0.62、CoffeeSetupMug 0.48、OpenCabinet 0.82、OpenDrawer 0.72、OpenStandMixerHead 0.31、PP-CounterToCabinet 0.70、PP-CounterToStove 0.81、PP-DrawerToCounter 0.51、PP-SinkToCounter 0.93、PP-ToasterToCounter 0.43、SlideDishwasherRack 0.51、TurnOnSinkFaucet 0.82。
- **`build_l1s1/groot_tp/`** 579 ep（183G）：批1 559+批2 18+批3 2；审计 ok:True。逐 task：CBL 0.74、CF 0.43、CSM 0.60、OC 0.81、OD 0.77、OSMH 0.79、PPCC 0.62、PPCS 0.81、PPDC 0.60、PPSC 0.82、PPTC 0.79、SDR 0.55、TOSF 0.24（封顶 100）。
- **账本**（`exp/robocasa365/data/`，gitignored）：`journal_collect_l1s1_{pi05,groot_tp}.jsonl`、run-plan pi05 b01-b02 / tp b01-b03、**`manifest_l1s1_{pi05,groot_tp}.json`（各 13×20，sha256 钉死，建库唯一入口）**。
- T3 测试残留已归档 `_archive_t3_manual/`（勿混入正式数据）。示例视频 `videos_l1s1/` 13 mp4（pi05 manifest 首条成功集，三视角横拼；Windows 副本在 `C:\Users\lzy66\Downloads\videos_l1s1\`）。

### 库（T6，`/data/robocasa365_cache/cache_artifacts_l1s1/`，软链 `exp/robocasa365/data/cache_artifacts_l1s1`）

| pkl | entries | 大小 |
|---|---|---|
| `pi05_spatial_pool_16_n{20,10,5}` | 23793/11911/5901 | 11.1G/5.6G/2.8G |
| `groot_tp_spatial_pool_16_n{20,10,5}` | 19003/9683/4794 | 7.7G/3.9G/1.9G |

嵌套前缀（n5⊂n10⊂n20，manifest episode_idx 序，零选择偏差）；n20 全量构建、n10/n5 切片派生（builder 纯可加性验证过，`library_stats` 用同一 `enrich_artifact_with_factors` 在子集重算=与重建等价）；`lists/` 六份 episode 清单为 provenance。配方：pi05=`cp1_spatial_pool_16 --vision-slots 3`、tp=`cp1_groot_spatial_pool_16`，均 `--episode-list`+`--trajectory-id-mode relpath`（stem 模式会静默碰撞丢 90%，绝不可用）。

### 代码（git：本地 = origin/Ziyang = `9f9a1f0`）

- **并发服务（本次 ship）**：`serve_groot_n15.py --concurrent`（默认 off 显式 opt-in，保既有命令 byte-fidelity）= per-connection 工厂：共享面仅 GPU policy（模块级单推理锁，v1 并发模型=GPU 串行+sim/网络重叠）与 storage backend（`build_shared_storage`+每连接 facade）；每连接全新 key_builder/orchestrator/`GrootStagedRunner`/interceptor/adapter；CSV 落 `conn_<uuid8>/` 子目录。`WebsocketPolicyServer` 新公共参数 **`allow_dynamic_bundles=False`**：拒全部 `load_cache_config`、拒非 default `select_bundle`、`select_bundle("default")` 保留为启动配置幂等绑定（runner 每连接首发依赖）。`--collect-hdf5` 恒单连接（D-L 冻结）。16 条非 manual 测试；**正式 Verify（G2 后）`tests/robocasa365 tests/cache tests/serving` = 1527/27/0**。
- T5 机器（run_collect conductor 全套 + `verify_collection_artifacts.py` + `min_episodes_for_target`）与 GR00T cache 栈（staged/interceptor/key_builder）早前已 ship。
- **并发能力真机实证（2026-08-21 探测）**：pi05 concurrent server + 双真 robocasa episode 全程时间戳交错、零串扰（公网路径推理 1.1-1.2s/次，两集一集墙钟）；conductor `WorkerAgent(specs)` 任意 worker 数（`capacities=1` 只是采集冻结参数）；**GR00T `--concurrent` 真机双连接冒烟未做**（G2 记录的 manual 遗留，T7 真机段第一件事）。

### 基础设施

- **weilandserver**（4090 48G 改装卡，多 session 共享）：直连公网段 **`ziyanglin.com:23100-23199`**（交换机 NAT 1:1，**服务必须监听段内**，ufw 已放行，优先级高于 tether expose——devices.md §4.0/§2.5.1）；已占 23100-23103/23122/23150（他线）。⚠ **GPU 冷卡不稳定**（冷态拉载 28-62s 窗口静默算错/Xid 31）→ **tmux `keepwarm` 恒温脚本常驻，任何情况不许关**。实测占用：GR00T server 6.1G、pi05 server ~8G、sim worker ~0.5-1G 显存+6G RAM。
- **timan107 sim 岛（已部署+冒烟 SMOKE-OK）**：Isaac-GR00T@`376ba89`（与 weilandserver 同 commit）+ robocasa365 assets 23G + venv `/scratch/zixuans8/Isaac-GR00T/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv`（py3.12、numpy 2.2.5、openpi-client `--no-deps` 已装——⚠ 直装会被 numpy<2 pin 拖垮整个 venv）+ **干净浅克隆 `/scratch/zixuans8/openpi_rc365`**（本线 import 专用；勿动共享 `/scratch/zixuans8/openpi`，rl_router 线直传态在那里）；EGL 原生（`10_nvidia.json`，无需 weilandserver 那套 nvidia-gl 补丁）；容量 ≈35-40 worker（48 核 CPU 是瓶颈）；/scratch 余 ~32G。⚠ 机上有 yurenh2 训练与 zixuans8 LIBERO 车队。
- **weilandserver 仓库未收敛到 `9f9a1f0`**：rl_router 线活跃使用中，pull 暂缓——T7 启动前先协调再同步。

## 3. T7 评测 — 下一步工作（顺序）

1. ~~真机段第一件事：GR00T 并发双连接冒烟~~ **✅ DONE（2026-08-21）PASS**——weilandserver:23160 `--concurrent`+n5 库 × timan107 双客户端，八项判据全过；证据 `exp/robocasa365/analysis/t7_concurrent_smoke.txt`（含纯 cache 回放 SR 0/4 备注，供 judge 裁决参考）。timan107 venv 已补 websockets/msgpack（--no-deps，numpy 未动）。
2. **薄 eval driver**：复用 conductor（`WorkerAgent` capacities>1 + **真 ctl factory**——`run_collect.py` docstring 明记「EVAL runs must switch back to a real ctl factory (`examples.libero.episode_runner.default_client_factory`)，no-op ctl 仅对 collection 正确」）；评测段种子 `base_seed=1000000`；沿用 journal/审计对账纪律。
3. **L0 阶梯先行**（D-H 冻结默认）：场景 (1,1)+评测段种子，同分布熔断；天然结合 size 阶梯 {n5,n10,n20}。
4. **2×2 场景阵**（D-J：layout∈{1,5}×style∈{1,7}；同 seed 跨场景不同 ⇒ 配对在 task 层）。
5. **拓扑**：weilandserver 多 server 进程（231xx 公网口，显存三连读 ≥ 需求才抢）+ timan107 多 worker（broker/公网往返慢，靠并发摊薄）。

## 4. T7 前待裁/待办（owner 决策点）

- **评测臂设计**：每臂 episode 数与统计口径；judge/gate 配置——G0-E 用 `always_hit` 只验接线，**现有 τ=0.334717 是 pi0.5 尺度标定，GR00T 侧不可直接复用**；纯 cache 口径（`always_search`+`always_hit`，同 cache_size 消融）还是完整 judge 栈，须 owner 定。
- **⚠ pi05 侧在线 cache 通路未验证**：库按 3 相机建（`--vision-slots 3`），pi05 在线 key builder 是 LIBERO 双相机血统——serve_policy `--cache_config` 在 `pi05_robocasa` 上的在线 key 构建须先做结构验证（groot_cache_integration 的 P2 遗留）。
- **T7 流程形式**（L2 全流程 vs fast-track）。
- eval yaml 族未起草：GR00T 模板 = `exp/robocasa365/config/groot_cache_cp1.yaml`（artifact 换 T6 pkl；⚠ cp3 块即使 disabled 也须显式 `search_strategy: weighted_rrf_knn`，validator 坑实踩过）。

## 5. 纪律与陷阱（存活精选）

- 共享机红线：他 session 进程/端口（8030、23100-23103、23122、23150、tmux `csmain`/`cssrv`/`keepwarm`/`rlrsrv`/`srv0`/`pubfwd`、timan107 的训练与 LIBERO 车队）绝不可动；禁宽 pkill；pgrep 锚定 `^[.]venv/bin/python …`；**清理 shell 与重启命令不得同 shell**（pgrep 自匹配自杀，实踩）。
- tether exec ~10min 硬上限：长任务（审计/构建）一律 tmux+tee；**审计曾被掐死在 report 与 manifest 之间**（实踩）。
- **远端后台清场脚本是延时地雷**：TaskStop 杀不掉 tether 远端 shell，尾部 `tmux kill-session` 会炸后启的同名会话（实踩，损失一次构建）。
- **共享工作树风险成真过**：他 session 的 git 操作会洗掉本线未提交编辑（本文件 v4→回退实例）；重要文档编辑后尽快 commit，或先核对 mtime。
- serve_policy 采集模式偶发挂死档案（3 次同签名：episode 写毕+连接关闭后 futex 用户态死锁全局失响；处置=driver 日志 keepalive 计数 ≥4 即弹换 server 重发 driver，resume 零数据损失；工程债未立项）。
- `--collect-hdf5`/`--collect_dir` 都传**场景根**（collector 自插 teacher 层，传 teacher 根得双层，实踩）。
- 同步规矩：迭代 `tether push --force`（tracked）；git 只在里程碑收口；新文件必须先走 git。
- 图像 token 事实：GR00T 段起点恒 20/283/546 ≠ pi0.5 固定表 0/256/512。

## 6. 工作树与监控状态

- git：本地 = origin = `9f9a1f0`；工作树残留他线文件（latency_bench、rl_router 等），提交须逐文件点名；本文件 v5 为本线唯一未提交件（下个里程碑随批提交）。
- 无活跃 cron/Monitor（T5 巡检已清；T7 开跑时按「Monitor 管事件触发、cron 管按时巡检」双层重挂，Monitor 挂载须在被盯日志重置**之后**——旧日志虚警实踩过）。
- 记忆：`reference_weilandserver_public_gpu.md`（公网段+冷卡）、`project_robocasa365_framework_reconnect.md`（T5 收官笔记）已同步。

## 7. 证据索引

`exp/robocasa365/analysis/`：`t5_audit_pi05.txt`/`t5_audit_tp.txt`（双侧终审 ok:True）｜`t5_server_provenance_8010.txt`（含 4 实例注记）/`t5_server_provenance_groot_tp.txt`（四端同 ckpt 双 safetensors sha）｜`t2_parity.txt`、`t3_*`、`t8_island_b_pytest.txt`、`g0e_closed_loop.txt`+`g0e_hit_log.jsonl`（历史门禁全套）。并发改造全history：`logs/groot_concurrent_serving_plan.log.md` Review Log。
