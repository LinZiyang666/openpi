# LIBERO × GR00T N1.5 采集 — 运行参数与判决记录

> 本文件是 LIBERO 侧 cache 采集的**运行口径唯一来源**。改任何一个参数都要在这里改，
> 并说明理由；下游 `exp/libero_groot/launch_collection.sh` 是它的可执行形式。
> 姊妹文件：`logs/session_handoff_robocasa365.md`（接手入口）。

## 1. 目标

用官方登记的 GR00T N1.5 LIBERO 权重，在 **B 池差集 500 条**（10 任务 × 50 初始状态）上
同时采集 **CP1 cache 嵌入** 与 **rollout 成功率**。suite 顺序：`libero_spatial` → `libero_10`。

## 2. ⚠ 2026-08-23 事故：夹爪动作未归一化 —— 首轮 351 集全废

**症状**：351 集 `success=False`，且每集恰好 44 步（= 220 max_steps ÷ 5 replan_steps），
即 100% 跑满步数预算。无异常、无 NaN、日志全绿。

**根因**：官方 `examples/Libero/eval/run_libero_eval.py:_convert_to_libero_action` 在返回动作前
必调 `normalize_gripper_action(action, binarize=True)`：

```python
action[..., -1] = 1 - 2 * (action[..., -1] - 0.0) / (1.0 - 0.0)   # [0,1] -> [+1,-1]
action[..., -1] = np.sign(action[..., -1])
```

两边的夹爪约定是**反的**，不是缩放差：GR00T 动作头输出的是开度（1=张开），
robosuite 把 +1 读作"合"（其 no-op 动作是 `[0]*6 + [-1]`）。原样透传 ⇒ 命令方向相反
且幅值不饱和 ⇒ 机械臂伸过去但永远抓不住 ⇒ 全部 pick-and-place 任务必然 0%。

**修复**：`exp/libero_groot/policy_adapter.py:normalize_gripper_action`，落在
`chunk_to_libero_actions`（7 维数组唯一装配点），`infer` 与 `iter_step_actions` 共享。
测试 `tests/libero_groot/test_policy_adapter.py::TestAction` 三条锁死（含"位姿列不受影响"）。

**教训**：跨执行体适配必须逐函数对照官方 evaluator 的**输出后处理**，不能只对齐输入构造。
输入侧写错会抛异常，输出侧写错只会安静地降到 0%。

**处置**：34G 废数据已按 owner 令删除（全部 `success=False`，库构建器本就会丢弃）。

## 3. 与官方 evaluator 的偏差表（每条都要有理由）

| 项 | 官方 | 本实验 | 理由 |
|---|---|---|---|
| 夹爪后处理 | `normalize_gripper_action(binarize=True)` | **同** | 见 §2，必须一致 |
| 图像 | `obs[...][::-1, ::-1]` 原始 256 | **同**（`--resize-size 256`，`resize_with_pad` 256→256 走恒等短路） | 用 client 默认 224 会裁两次，视野与训练分布不一致 |
| state | `[eef_pos(3), quat2axisangle(eef_quat)(3), gripper_qpos(2)]` | **同**（8 维 wire → 6 标量 + `state.gripper`(2)） | 逐字转写 |
| data config | `LiberoDataConfig` | **同** | spatial/object/90/long 都是它；只有 goal 用 MeanStd |
| 去噪步数 | `--denoising-steps 8` | **8**（`serve_groot_libero.py --denoising-steps`，默认 8） | 官方 92% 是 8 步下测的；checkpoint config 自带值更低 |
| replan | 每次推理只执行 `chunk[0]`（≈replan 1） | **5** | 见 §4 配对 A/B：无差别但便宜 5×，且是本仓库全部 LIBERO 基线口径 |
| 初始状态 | A 池 `get_task_init_states()` | **B 池差集** | 铁律：A 池是冻结测试集，采集/训练绝不可用 |
| env seed | `env.seed(0)` | `env.seed(7)`（client 默认） | B 池初始状态在 seed 7 下生成，本仓库 LIBERO 线一贯用 7 |
| max_steps (l10) | 1000 | 520 | `main.py:_get_max_steps` 现有口径（最长训练 demo 505 步） |

**教师基线预期**：官方 spatial 报 46/50=92%（A 池）。本实验 B 池实测 ≈ **76%**（见 §4）。
差距来自初始状态池：A 池 `pruned_init` 是**被筛选过**的，B 池是未筛选的剩余，本就更难。
按 `feedback_teacher_is_not_the_variable` —— 教师是固定底座不是自变量，只需"够能干"且两臂同一个，76% 完全够用。

## 4. 配对 A/B：replan_steps 5 vs 1（2026-08-23 11:18–11:31）

同 5 任务 × 同 5 个初始状态索引（`--num-trials-per-task 5`，global id = task×5+idx ⇒ 逐集配对）。

| 臂 | SR | 每集推理次数 | 逐任务 (t0..t4) |
|---|---|---|---|
| replan 5 | **19/25 = 0.760** | 17–33 | 0.40 / 0.80 / 0.60 / 1.00 / 1.00 |
| replan 1 | 17/24 = 0.708 | 110–192 | 0.20 / 0.50 / 0.80 / 1.00 / 1.00 |

**裁决：replan 5**。n=25 上两者无可分差异，replan 1 贵约 5× 且不更好；replan 5 还是
`exp/warm_start` / gate_research / TRACER 等本仓库既有 LIBERO 基线的口径，换掉会让新数据与旧基线不可比。

## 5. 正式运行参数（spatial，2026-08-23 11:34 起）

**拓扑**：weilandserver 本机闭环，6 server × 6 client 单机 4090 48G。
显存 38.1/49 GB（每 server ≈6.0 G + 每 sim EGL ≈0.55 G + keepwarm 0.4 G），GPU 温度 56 °C。
6 路实测 ≈ 2.6 ep/min/lane ≈ **15.6 ep/min** 聚合（4 路时同样 2.6/lane，说明未到饱和）。

**server**（× 6，端口 8030–8035）：
```
PYTHONPATH=/home/weiland/gr00t_n15:/home/weiland/gr00t_n15/examples/Libero:\
/home/weiland/openpi:/home/weiland/openpi/src OPENPI_MONITOR_LEVEL=BASIC \
/home/weiland/gr00t_n15_venv/.venv/bin/python exp/libero_groot/serve_groot_libero.py \
  --checkpoint /home/weiland/ckpt_n15_libero_spatial \
  --port <8030..8035> --denoising-steps 8 \
  --collect-hdf5 /data/libero_cache/build_spatial --experiment groot_libero_spatial
```

**client**（× 6，⚠ `--task-ids` 是 tyro tuple，**空格分隔**）：
```
MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH=. \
/home/weiland/miniconda3/bin/conda run -p /home/weiland/libero_sim --no-capture-output \
python examples/libero/main.py --host 127.0.0.1 --port <p> \
  --task-suite-name libero_spatial --task-ids <lane 的任务> \
  --num-trials-per-task 50 --num-workers 1 --resize-size 256 --replan-steps 5 \
  --init-states-dir /home/weiland/openpi/exp/common/data/db_init/libero/libero_spatial \
  --cuda-visible-devices 0 \
  --episode-filter /data/libero_cache/shards/libero_spatial/libero_spatial_lane<i>.json \
  --save-episode-results --episode-results-path /data/libero_cache/build_spatial/results_lane<i>.json
```

**一键起（含断点续跑）**：
```
bash exp/libero_groot/launch_collection.sh <suite> <ckpt> <out-dir> [lanes=6] [base-port=8030]
# spatial: libero_spatial /home/weiland/ckpt_n15_libero_spatial /data/libero_cache/build_spatial 6 8030
# l10:     libero_10      /home/weiland/ckpt_n15_libero_10      /data/libero_cache/build_libero10 6 8030
```
脚本会先拆旧车队、起 N server、按**已落盘 h5 重算分片**、再发车 —— 所以**重跑即续跑**，
且开跑前硬校验 B 池目录不含 `.pruned_init`（`_load_init_states` 优先读 `.pruned_init`，
一个残留文件就会静默把冻结测试集换进来）。

**分片机制**：`exp/libero_groot/make_shards.py`。`main.py` 对 `--episode-filter` 只做**跳过**、
不重编号，`initial_states[episode_idx]` 与 `global_episode_id = task_id*50 + episode_idx` 恒定，
因此任意 (task, idx) 划分都复现同一批 500 集、h5 文件名跨路不撞号。分片按 (task, idx) 序**连续切**，
让每路触碰的任务数最少（每个任务一次 `OffScreenRenderEnv` 构造，同进程堆 GL context 是踩过的坑）。

**6 路分片**（spatial / l10 同形）：lane0 `[0,1]` 83、lane1 `[1,2,3]` 83、lane2 `[3,4]` 84、
lane3 `[5,6]` 83、lane4 `[6,7,8]` 83、lane5 `[8,9]` 84。

**成功率统计**：`exp/libero_groot/report_collection.py <h5dir> --trials 50 --num-tasks 10`。
权威来源是 h5 的 `success` 属性（`results_lane*.json` 只在 client 跑完才写，中途换路就没了）。

## 6. l10 侧已就绪

- 权重 `youliangtan/gr00t-n1.5-libero-long-posttrain` 已下载：`/data/ckpt/n15_libero_10`（7.1G），
  软链 `/home/weiland/ckpt_n15_libero_10`。config 与 spatial **逐字节相同**，
  embodiment `new_embodiment`、modality `video.{image,wrist_image}` + 7 维 state/action。官方登记 38/50=76%（A 池）。
- B 池校验通过：`exp/common/data/db_init/libero/libero_10/` 10 个 `.init` × 50 = 500，零 `.pruned_init`。
- 落盘 `/data/libero_cache/build_libero10/`。l10 每集步数上限 520（spatial 220）⇒ 单集约 2.4×，
  6 路预计 1.5–2 h，磁盘约 95–100 GB（/data 余 2.3T）。

## 7. 存储与铁律

- 数据一律落 `/data`，仓内软链（owner 令）。已办：`exp/robocasa365/data -> /data/openpi_exp_data/robocasa365`；
  权重也落 `/data/ckpt`。⚠ 未办待 owner 划边界：`exp/common/data` 14G（跨实验共享）、
  `exp/rl_router/data` 1.5G（属 X15 线，另一 session 在用，不动）。
- keepwarm 恒温脚本任何情况不许关（4090 冷态陡热爬坡会静默算错）。
- 共享 checkout：stage 只按显式路径清单；commit message 英文、无 AI 署名。

## 8. 立体监控体系（2026-08-23 11:45 建立）

四层，每层失效时上一层还能接住；**层与层之间不许互相假设对方活着**。

| 层 | 载体 | 位置 | 周期 | 职责 | 失效后果 |
|---|---|---|---|---|---|
| **L1 自愈** | `exp/libero_groot/watchdog.sh`（tmux `lbwatch`） | 远端 weilandserver | 300 s | keepwarm 重挂 / 死 server 重起 / **仅重起"分片未跑完"的 lane** | 无人自愈，靠 L4 兜底 |
| **L2 接续** | `l10_chain.sh`（tmux `l10chain`） | 远端 | 事件驱动 | spatial 排空 → 校验 500 唯一 → 出成功率报告 → 把六个端口交给 l10 | suite 不自动换，L4 手动换 |
| **L3 推送** | Monitor（本机 session） | 本机 | 600 s | keepwarm 死 / chain ABORT / 30 min 无新 h5 / 车队空转 / **SR 哨兵** / l10 满 500 | 无实时推送，靠 L4 |
| **L4 巡检** | cron `13,43 * * * *`（session-only，7 天过期） | 本机 | 30 min | 全量对账：成功率坍塌判据、dup、磁盘、三机在线、缺什么补什么 | 全盲 |

**L1 的关键设计**：只重起**分片仍有未完成 episode** 的 lane。"完成"由**磁盘上的 h5** 判定而非退出码
（lane 也可能在最后一次 flush 之后才死）。这条是 L2 能工作的前提 —— 跑完自然退出的 lane 必须允许它保持关闭，
否则 lane 计数永远回不到 0，suite 接续就永远不触发。运行态由 `launch_collection.sh` 写的
`/data/libero_cache/current_run.env` 提供（suite/ckpt/outdir/lanes/base-port），自愈不靠猜。

**已做端到端控制组**（监控没被真正触发过等于没有）：
1. 判据探针：逐 lane 打印 `shard=83 remaining=35`，确认不会恒返 0（若 `lane_remaining` 静默出错返空，
   `${rem:-0}` 会退化成 0 ⇒ 永远认为一切完成、从不自愈 —— 这是本层唯一的致命静默失败）。
2. 真杀验证：11:47:15 杀掉 `lbrun4`，11:50:19 watchdog 自动重起（一个周期内），日志留痕。

**L3/L4 的头号判据是成功率坍塌，不是吞吐**。教师基线 spatial ≈0.92 / l10 官方 0.76；
比成功率更早可读的是 `steps/ep`——若恰好等于步数上限 ÷ replan（spatial 44、l10 104），
说明策略根本没在工作。§2 那次就是靠这个判据识破的，代价是 34G 数据。

**实测**（修复后，n=277）：`TOTAL 254/277 = 0.917` —— 与官方 A 池 92% 齐平。
成功集提前终止（17–33 次推理 vs 满额 44），吞吐从 15.6 ep/min 升到 ~25 ep/min。

## 9. 采集完成 —— 两 suite 终表（2026-08-23）

| suite | 覆盖 | 成功 | SR | 官方登记(A 池) | 落盘 |
|---|---|---|---|---|---|
| `libero_spatial` | 500/500 唯一 | 456 | **0.912** | 46/50 = 0.92 | 26 G |
| `libero_10` | 500/500 唯一 | 427 | **0.854** | 38/50 = 0.76 | 63 G |

两 suite 均零重号。数据 `/data/libero_cache/build_<suite>/`，逐集报告 `success_report.json`。

**libero_spatial 逐任务**（n=50 each）
| task | SR | steps/ep | task | SR | steps/ep |
|---|---|---|---|---|---|
| 0 | 0.720 | 24.5 | 5 | 0.960 | 21.4 |
| 1 | 0.880 | 25.0 | 6 | 0.980 | 21.9 |
| 2 | 0.940 | 22.3 | 7 | 0.940 | 25.5 |
| 3 | 0.980 | 17.6 | 8 | 0.800 | 25.1 |
| 4 | 0.980 | 26.4 | 9 | 0.940 | 25.5 |

**libero_10 逐任务**（n=50 each）
| task | SR | steps/ep | 描述 |
|---|---|---|---|
| 0 | 0.860 | 62.8 | alphabet soup + tomato sauce → basket |
| 1 | 0.960 | 50.9 | cream cheese + butter → basket |
| 2 | 0.840 | 59.9 | turn on stove + moka pot on it |
| 3 | 0.960 | 48.7 | black bowl → bottom drawer + close |
| 4 | 0.900 | 52.6 | white mug → left plate, yellow/white mug → right plate |
| 5 | 0.920 | 44.5 | book → back compartment of caddy |
| 6 | 0.780 | 61.5 | white mug → plate, chocolate pudding → right of plate |
| 7 | 0.840 | 59.9 | alphabet soup + cream cheese → basket |
| 8 | **0.580** | 91.7 | **put both moka pots on the stove**（最难；steps/ep 逼近 104 上限） |
| 9 | 0.900 | 60.0 | yellow/white mug → microwave + close |

**l10 显著高于官方 0.76**：官方 evaluator 是 replan=1，我们经配对 A/B 选的 replan=5（§4）；
另外 B 池与 A 池的任务难度分布本就不同。**不要**把这两个数当同口径比较。

**吞吐实测**：spatial 22.5 ep/min（3.7/lane，16 s/集，50 MB/集）；
l10 约 11 ep/min（126 MB/集，episode 长 2–4×）。两 suite 合计约 95 分钟、89 GB。

### 对建库的直接约束（成功轨迹数 = 每任务库容上界）

| suite | 每任务成功数范围 | 最小值（决定统一 n 的上界） |
|---|---|---|
| spatial | 36–49 | **36**（task 0） |
| libero_10 | 29–48 | **29**（task 8, both moka pots） |

⇒ 统一口径最多到 **n=29**（l10）/ **n=36**（spatial）。要更大的 n 必须补采（脚本可续跑，
但 B 池每任务只有 50 个初始状态，补采只能靠"同初始状态换种子"，会引入与现有数据不同的分布）。

**待 owner 定**：新 cache 库口径 —— 统一 n5/n10/n20 ↔ 按任务难度/物体类别分桶 ↔ 两者都建。

## 10. 分层 cache 库构建（2026-08-23 13:33 起）

沿用 X9b `exp/ablation_study/cache_size` 的设计，**但把"每档单独建一次"改成"建一次最大的再切"**：
语料 89 GB，昂贵的 h5 扫描只跑一遍，档位从产物里切出来 —— 嵌套关系因此是构造性成立的，不需要事后校验。

**产物**：`/data/libero_cache/libraries/<suite>/<suite>_{mean_pool,sp16}_{full,S1..S6}.pkl` + `_manifest.json`。
脚本 `exp/libero_groot/build_size_libraries.py`。

### 两条不可动摇的口径

1. **切分单位是整条 episode**。entry 的 `prev_ids/next_ids` 是 episode 内链接，按 step 切会留下悬空引用，
   而下游没有任何消费者会检查它。脚本对每档显式验：无重复 id、无越界 trajectory、无悬空链接、
   与前一档的真包含关系。
2. **自变量是「每任务成功轨迹数」，不是「采样 episode 数」**（X9b 原文论证）。检索是任务作用域的
   （`search_strategy` 构造 `QueryFilter(task_key=...)`），某任务在某档为空 ⇒ 该任务全部评测集静默回落到教师。
   低成功率任务恰是最易空的，而覆盖随规模上升 ⇒ 用 episode 采样做轴会把曲线朝"正好印证预期结论"的方向掰平。
   每档每任务取 `min(k, n_t)`（R1），报的 x 轴是**实测均值**不是名义 k（R4），任一任务 n_t=0 直接 fail loud（R2）。

档位 **S1–S6 = 1/2/5/10/20/50** 条/任务。S6 名义 50 在各任务触顶（spatial 最多 49、l10 最多 48）⇒ S6 ≈ 全量。
取序为 seed=0 的确定性洗牌（B 池 init 索引本身无序，按索引取会把档位成员绑到池的写入顺序上）。

### 建库前查清的三件事

1. **LIBERO 几何绑在 builder 名上**（`_GROOT_GEOMETRY`），不靠 CLI 传。`_reshape_dims` 的失败方向是静默的：
   相机数少报只会让该字段从 `vector_dims` 消失，后端此后每次查询都默默不带它。
   `cp1_groot_libero_*` = (2 相机, 8 维 state)；RoboCasa 的 (3, 20) 原样不动。
2. **离线 `vision_2` 的零向量无害**：`_build_fake_stage1` 永远拼三个相机槽、缺的补零"以维持 token 偏移"
   ⇒ vision_0/1 与 prompt_emb 的切片偏移正确；多出的 `vision_2` 不在 `vector_dims` 里，后端丢弃。
3. **离线/在线池化是同一份代码**：`src/openpi/cache/groot/key_builder.py` 直接
   `from openpi.cache.components.key_builder import _mean_pool_tokens, _spatial_pool_tokens`，
   作用在同一个 256×2048 块上 ⇒ 键的等价是**结构性**的，不需要再跑 G0-D2 那种运行时 parity 门。
   只剩离线 fp16 / 在线 bf16 的量化差 —— RoboCasa 侧已过门的已知项。

### 冒烟（建库前的控制组）

15 条 episode → 8 条成功（outcome-filter 生效）→ 三档嵌套，realized mean 1.0/1.7/2.7；
`InMemoryBackend` 回环加载条数一致（105=105）；`task_key` 是任务语言串、`action_chunk` (16,32)、
`vector_dims` = {vision_0/1: 2048, prompt_emb: 2048, robot_state: 8}。

### 运维

`/data` 是机械盘（Seagate 4T，ROTA=1）⇒ 每 suite 第一遍（mean_pool）压到 6 workers 走顺序读、
顺便把语料喂进 page cache（246 GB 内存装得下 89 GB），第二遍（sp16）转 CPU 密集拉到 24 workers。
⚠ 驱动脚本里的 `grep` 管道会**块缓冲**，日志攒够 4 KB 才落盘 ⇒ 监控不能只看日志末行，
必须同时看构建进程数（session 在、进程数为 0 = 卡死，与"安静地在算"长得一模一样）。

### 构建结果（2026-08-23 13:33–14:00，约 27 分钟，四组 28 个 pkl，39 GB）

**每任务成功轨迹数**（= 该任务的库容上界）
- `libero_spatial`：36/44/47/49/49/48/49/47/40/47（最小 36 → S5=20 仍达名义值）
- `libero_10`：43/48/42/48/45/46/39/42/**29**/45（最小 29，task 8 both-moka-pots）

| tier | 名义 k | spatial 实测 | spatial 轨迹/entries | l10 实测 | l10 轨迹/entries | mean_pool MB (sp / l10) | sp16 MB (sp / l10) |
|---|---|---|---|---|---|---|---|
| S1 | 1 | 1.0 | 10 / 219 | 1.0 | 10 / 519 | 7 / 17 | 84 / 199 |
| S2 | 2 | 2.0 | 20 / 427 | 2.0 | 20 / 1039 | 14 / 34 | 164 / 400 |
| S3 | 5 | 5.0 | 50 / 1063 | 5.0 | 50 / 2645 | 35 / 88 | 409 / 1018 |
| S4 | 10 | 10.0 | 100 / 2119 | 10.0 | 100 / 5309 | 71 / 178 | 816 / 2044 |
| S5 | 20 | 20.0 | 200 / 4281 | 20.0 | 200 / 10602 | 143 / 356 | 1649 / 4083 |
| S6 | 50 | **45.6** | 456 / 9815 | **42.7** | 427 / 22039 | 329 / 740 | 3780 / 8489 |

S1–S5 全部达成名义值（两 suite 每任务成功数都 ≥29 > 20），只有 S6 触顶 ⇒ **S6 ≈ 全量**，
`_full.pkl` 与 S6 内容等价但保留：重新切别的档位时不必再扫 89 GB 的 h5。

**验收**（`exp/libero_groot/verify_libraries.py <manifest>`，四组全 PASS）：
逐档 `InMemoryBackend.load_artifact` 回环，后端条数 == 构建条数 ⇒ 无 id 碰撞
（构建日志与加载日志报的都是去重前计数，只有两者相比才暴露碰撞 —— X9b 的通用探针）；
另校验 builder 名与 `vector_dims` 戳记、逐档轨迹真包含关系。

**路径**：`/data/libero_cache/libraries/<suite>/<suite>_{mean_pool,sp16}_{full,S1..S6}.pkl`，
仓内软链 `exp/libero_groot/data -> /data/libero_cache`（owner 的 /data 铁律）。

**实测推翻了一条预设**：机械盘顾虑在本轮不成立 —— 语料刚采完仍全在 page cache，
构建期 `read_bytes` 实测为 0，一次真实块设备读都没有，全程 CPU 密集。
（下次隔夜再建、cache 已冷时，第一遍压低 workers 的做法仍然有效。）
