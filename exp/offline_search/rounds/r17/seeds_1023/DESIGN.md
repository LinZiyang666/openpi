# 新种子重做成功过滤库旋钮下降（10-23 owner 定）

## 要做什么

只重做**成功过滤库的旋钮下降**，换新的库种子，所有模型、每个库大小、4 种变体。

| benchmark | 库种子 | 库怎么来 | 要做的库大小 |
|---|---|---|---|
| LIBERO | 20261003、20261004（旧 20261002） | 不重采。B 池 500 init 按 `default_rng(种子+任务号).permutation(该任务 50 个局)` 取前 N/10，嵌套（同 `r10/data.py:149-181`） | DP/Fast-WAM/Cosmos：50/100/200/300/400（它们有 6 档，500 是全集不重做）；π0.5/GR00T：50/100/200（它们只有 50/100/200/500） |
| MT50 | seed 9、seed 10（旧 seed 8） | 重新采：`MT1(task, seed).train_tasks[idx]`，idx 0–49，50 任务 × 50 = 2,500 局，每模型全步纯推理、horizon 5，同 seed 8 采集口径 | 250/500/1000/1500/2000/2500 全做（= 每任务前 5/10/20/30/40/50 个 idx） |

变体：全步/减步 × 长执行段开/关。
- R17（DP/Fast-WAM/GR00T MT50）：开 = C10（commit_horizon 10，C10 拟合 + C10 段头 Q3），关 = C5（C5 拟合 + C5 Q3）。
- π0.5/GR00T LIBERO（R15 全步 / R16 减步）：开 = `--os-blind --os-policy-tail --os-policy-tail-blocks 1`，关 = 去掉这三个参数；fit/Q3/周期两边共用。
- ⛔ 没有长执行段的模型（owner 10-23）：LIBERO 的 Cosmos、MT50 的 π0.5 没有长执行段，长执行段相关的两组（全步×长、减步×长）不跑，只跑默认段长的全步、减步两组；不改它们的模型代码。
  - Cosmos 默认段长就是 C16（`models/cosmos/provider.py:13`）。
  - π0.5 MT50 默认 C5（checkpoint 只出 5 步动作）；同卡参照只在 4090（tpi7_0536、tpi7k2_1135），等 4090 交还再跑。

规则（不变）：
- 停滞门只用 Q3 stall_only。
- 参照用同卡同步数纯推理，与库无关，直接复用：DP/FW MT50/LIBERO 的 `longseg_off_1022/REFERENCES.json`；π0.5/GR00T LIBERO 关 = `fa_lo_*_p5_q`，开 = `formal_a_queue.references`；Cosmos = `formal_a.py` PURE。π0.5/GR00T LIBERO 开若放 H100，要先补跑 `fa_lo_{gr_f,pi_r,gr_r}_p_q`（各 1,000 局）。
- **两步下降（LIBERO 与 MT50 都是）**：先跑旋钮关；完整、且旋钮关不低于纯推理的格子不调，其余格子再逐档下降。
- 每个新种子的库按同样流程重新标定：拟合、Q3、周期、gap 都按新库算。长执行段关沿用开的 gap（owner 10-22 裁定 OFF 不重标）。

## 常驻复用（owner 10-23）

- 同一批里按 YAML 热切换：commit_horizon 是 YAML 级的，一个 R17 常驻在同一步数 K 下能服务多种子 × 多库大小 × C5/C10。按「模型 × benchmark × 步数 K × 卡」合并成一批。DP 按 K 捕获 CUDA graph，全步/减步分常驻；GR00T 步数是进程级，同样分。
- R16 常驻（π0.5/GR00T 减步）每个 entry 自带 plugin_args，开/关、fit 都按 entry 切；一个 group 可装多种子 × 多库大小 × 开/关（先例 `fa_cs_*_rg` 32 个 entry）。R15 全步不是常驻，每个 arm 重启服务。
- 合并后分析键、evidence 键、YAML 文件名必须带种子和开/关，否则互相覆盖（`formal_a_analysis.py:86`、`knob/common.py:99-118`、`evidence.py:16-28`、`q3_port_1020/analysis.py:58,64`、`longseg_off_1022/build.py:163,168`）。
- 跨批接活常驻：R17 没有现成模式，要在 `formal_eval.run_campaign` 加保留常驻/接管开关；R16 要给 Registry 加注册入口。

## 会静默出错的地方（必须改成报错或带种子）

- 成功过滤的 selection 只在常驻启动时读一次（`filter_failed_knob/runtime.py:22-24,82`），缺新 fit 的 digest 会悄悄退回不过滤的库 → 改成缺了就报错。
- 保留局表 `cells()` 只按 (模型, suite, size) 取（`filter_failed_knob/build.py:156-167`、`longseg_1017/recalibrate.py:58-65`）→ 会拿到旧种子的成功局。
- 产物已存在就复用：`q3_port_1020/fit.py:137`、`q3_port_1020/build.py:139-140`、`longseg_off_1022/build.py:80-83`、`build.py:76-78` 周期按 cell 名取 → 种子必须进路径。
- `longseg_off_1022/workflow.py:79-80` 查不到 C5 fit 的格子会被悄悄丢掉。
- `cache_study.py:136,142` 用路径后缀 `_mt50/bpool` 判断 250 是否 = size_5 → 新 MT50 库路径要保留这个后缀。

## 拔的防御（不影响数字）

种子写死：`debug/library_capture.py:52`、`search/teacher_library.py:36`、`search/cache_recipe.py:82-84`、`ops/library_campaign.py:31`；`library_alignment.py:116-127` 不经过即可；`library_staging.py:57-59` 只准本机 finish；`library_campaign.py:301-303,371-372`；`store.py:73-74,77`；`cache_recipe.py:150-151,264-266`、`:279` 的逐文件 sha 子句（绑定子句保留）、`:61-63`；`formal_prestage.py:46`、`staging.py:21`、`sync_back.py:47-48,72-73,107-108` 软链拒绝与打包后重算 sha；`filter_failed/stage.py:150-158,190`；`filter_failed_knob/build.py:121`；`q3_port_1020/build.py:228` 断言 8 格；`r15/descent/fits.py:42-45`；`q3_port_1020/fit.py:173-185,195-196` 旧门周期对照；R10Recipe 拟合时的旧门 raise 只在拟合进程内放行，拟合完剥掉 judge（服务端仍禁旧门）。

保留（影响数字或防混批）：fit 与 YAML 的 store_digest/commit/binding（`cache_recipe.py:160-170`）、Q3 绑定与度量（`:205-248,276-279`）、RowIndexStore 完整局检查、`recalibrate.py:107` 成功标签一致、DP CUDA graph 自检、LAUNCH.json 独占认领、局数完整性、拟合防测试集泄漏（`r10/recipe/fitting.py:24-34,37-63`）、longseg 补丁锚点。

## 机器

- 本机（weilandserver，4090 + CPU）被 cluster_sim 的 sim serving 占用，交还前不在本机做任何运算（不跑测试、拟合、汇总）。
- π0.5 MT50 采库和旋钮批次都要 4090（参照只在 4090）。
- CPU 活（拟合、Q3、周期、gap）：本机交还后跑，或同步数据到 timan1（48 核、~270 GB 空闲内存）。

## 工作量（第一步旋钮关）

- MT50：每种子 (3 模型 × 4 变体 + π0.5 × 2) × 15,000 = 210,000 局，两种子 420,000 局；合并后每个常驻 60,000 局（2 种子 × 6 库大小 × C5/C10 × 2,500）。采库每种子约 20,000 局（π0.5 2.5 h、GR00T 1.1 h、DP 0.5 h、FW 0.75 h）。
- LIBERO：π0.5/GR00T 2 种子 × 2 模型 × 4 变体 × 6 格 × 500 = 48,000；DP/FW 2 × 2 × 4 × 10 × 500 = 80,000；Cosmos 2 × 2 × 10 × 500 = 20,000。
- 第二步各档只对没达标的格子：LIBERO 每格 4 档 × 500，MT50 每格 5 档 × 2,500。
