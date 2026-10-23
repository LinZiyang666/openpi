# W2：π0.5 / GR00T LIBERO 新子集种子

先读 exp/offline_search/rounds/r17/seeds_1023/prompts/COMMON.md（必须遵守）和 seeds_1023/DESIGN.md。

## 范围
π0.5、GR00T 的 LIBERO 成功过滤库旋钮下降，新子集种子 20261003、20261004。**库大小只有 50/100/200**（这两个模型现行链只有 50/100/200/500，500 是全集不做；300/400 不做）。变体：全步（R15）/减步（R16）× 长执行段开（`--os-blind --os-policy-tail --os-policy-tail-blocks 1`）/关（去掉这三个）。父库不变（π0.5 `bpool_cs`、GR00T `bpool_all`），不重采。

## 现行链（只读调查结论，自行核实）
子集（`r10/data.py:149-181`，种子常量 :18）→ R10Recipe 拟合（`r10/recipe/build.py fit`）→ cs `_cr` 形态（`filter_failed_knob/cache_study_1019/study.py:53-74`：R10Recipe 外壳 + `inner.base`）→ 每格 Q3 + 周期（`filter_failed_knob/q3_port_1020/fit.py`）→ 成功过滤（`vla.candidates`，kept 来自 `audit_counts.summarize`）→ knob fit（`q3_port_1020/build.py:88-106`，放 openpi_ext 别名下）→ 批次（开：`q3_port_1020/build.py` R15 :187-207、R16 :274-285；关：`longseg_off_1022/build.py:186-271`）。

已知阻断：
- R10Recipe 拟合要构造旧门 `TriggerCommitJudge`，被 `r06/p2_ablations/judge.py:21`、`r03/h3_judge/judge.py:110-111,164-166` 直接 raise → 写一个新的拟合入口：**只在拟合进程内**放行这两处，拟合完立刻剥掉 judge，只输出 `R10Recipe(inner=_CorrectedCache)`（cs `_cr` 形态）。服务端继续禁旧门。
- 周期重解依赖只有旧种子才有的 `r11/opus/out/anchor/*.npz`（`r15/descent/fits.py:31-56`）→ 改成直接从库行构造各局（ep/task/偶数步/行数，`r11/knob/calibration.py:47-63` 只用这些）。
- `q3_port_1020/fit.py:173-185,195-196` 旧门周期对照（读 r16/design.json）→ 删，任意格集合都写 PERIODS.json。
- 静默复用：`fit.py:137`、`build.py:139-140`、`longseg_off_1022/build.py:80-83`、`build.py:76-78` → 种子进目录。
- 50 格可能出现空任务（某任务 5 局全失败），`vla.py:33` 会删整张任务表——核实服务端行为，必要时明确处理并在交付里写清。

## 命名
cell 字符串保持 `{model}_{suite}_{N}`（多处按 `_` 拆）。种子只进目录名和单位名：子集 `/nvme/weiland/openpi/r17_seed_split/s<种子>/subsets/`（不要写进 `r10/subsets/`），产物 `/nvme/weiland/openpi/r17_seed_split/s<种子>/{cs,q3,fits}`，knob fit 放 `/home/weiland/projects/openpi_ext/r17_seed_split/s<种子>/`（H100 同步只镜像 /home 和 /data）。单位名用 `_x3`/`_x4` 之类（`s` 已是 spatial 短名）。

## 常驻复用（owner 10-23）
- 减步（R16 常驻）：每模型把 两种子 × 3 库大小 × 开/关 的旋钮关 entry 并进**一个 group**（先例 `fa_cs_*_rg` 32 个 entry；R16 每个 entry 自带 plugin_args，开/关按 entry 切）。GR00T 一个进程只服务一个 suite，照现有拓扑。分析键（`q3_port_1020/analysis.py:58,64`）加种子和开/关。
- 全步（R15，非常驻，每 arm 重启）：照现有 R15 跑法建批即可；如果你确认 R16 常驻能在全步 K 下逐位复现 R15 的结果，在交付里写出证据和改法，但不要直接切换。
- 第二步各档：旋钮关完成后，用 `formal_a_queue.classify` 规则（完整且旋钮关成功数 ≥ 同卡同步数纯推理则不调）只给需要调的格子建各档。

## 参照
关：`longseg_off_1022/QUEUE.json` 的 `fa_lo_{pi,gr}_{f,r}_p5_q_*`（H100，已跑完）。开：`formal_a_queue.references`；若开放在 H100，GR00T 全步、π0.5 减步、GR00T 减步没有 H100 C10 参照，`fa_lo_{gr_f,pi_r,gr_r}_p_q` 已建未跑——在交付里给出补跑命令。不新建别的参照。

## 文件归属（只改这些，新代码放 seeds_1023/vla/）
新模块 `exp/offline_search/rounds/r17/seeds_1023/vla/`；允许最小改动：`filter_failed_knob/formal_a_queue.py`（cells 可传库大小）、`filter_failed_knob/q3_port_1020/{fit,build,analysis}.py`（参数化，旧行为默认不变）、`filter_failed_knob/vla.py`、`r15/descent/fits.py`、`r16/analyze.py`、`r16/conductor_analysis.py`（拔 DESIGN.md 列的分析侧封条/sha 锁）。不改 `r10/data.py`、`r10/recipe/*`（新入口放 seeds_1023/vla/）、`longseg_off_1022/build.py`（W1 归属；需要钩子在交付里写）。

## 交付
按 COMMON.md。回归检查必须有：用旧种子 20261002 重拟 `pi05_l10_50` 和 `groot_spatial_200`，与 `cache_study_1019` 对应 `_cr` fit 的 `inner.base` 逐字节比对；Q3/PERIODS 与 `/nvme/weiland/openpi/r17_q3_port_1020/` 现有产物比对。
