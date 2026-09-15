# Session handoff

> 多条线共用本文件。§0 为常驻初始化（不动）。**融合权重 / key builder × LDA 线（本 session）看 §7**；§1–§5 是 RIT LOTO 线的运行手册，§6 是 ActionCache × GR00T 存档。

## 0. 初始化方式（不变）

owner 的常驻指令，逐字有效：

> 「开始进行实验，我离开了，期间由你独断专行，不要问我任何问题，注意监控体系定位，
> corn 负责定时巡检，monitor 负责条件触发，不要职责混淆，停止条件是完全做完实验，不做完不停」

**三条必须保持的纪律**：

1. ⛔ **不得用 git commit / push 同步代码和数据**。要同步任何代码或数据一律走 **tether**。
   git 只在里程碑收口、且 owner 当次明确指示时才动。
2. **server 只在 server 节点跑，worker 只在 timan 族跑**（见 §2）。
3. **读 `experiment-lifecycle` skill 但不挂载**：只取 tether 用法（`dist_experiment_control/docs/usage.md`）
   与设备清单（`docs/devices.md`），**不要执行它的 §0 初始化**，不要索要 agentchat 账号 / token / 房间。

### 开工步骤（照做，不要问 owner）

1. **读文档**，按这个顺序，都读完再动手：
   - `logs/cache_prune_run_handoff.md`（整份，运行入口的唯一权威）
   - 本文件 §1–§5
   - `/home/weiland/projects/dist_experiment_control/docs/devices.md`（拓扑与机器约束）
   - `/home/weiland/projects/dist_experiment_control/docs/usage.md`（tether 命令）
   - 需要时再查 `logs/cache_prune_plan.log.md`（算法与统计定义）
2. **探设备**：`tether node ls -a`，确认 h100 / weilandserver / timan107 / timan108 四台
   都 ONLINE；逐台查 GPU 占用、RAM、磁盘余量、残留 tmux 与进程。⚠ timan108 只有 3 张卡
   （见 §2），别按 4 张排。
3. **对齐代码**：四台的仓要和本地同一 commit。**走 tether push，不走 git push**；
   已存在的文件要 `--force`；`/scratch` 不在 timan 的 allow_roots，经 `/tmp` 中转。
   推完逐文件 `sha256sum` 对账，**比对内容不要带路径**（`cat A B | sha256sum`，
   `sha256sum A B` 会把路径算进去，我为此误判过一次）。
4. **建任务表**（TaskCreate），把 §1 的准备项与三相位拆成条目，边做边更新状态。
5. **搭监控**（§4）：cron 定时一行巡检 + Monitor 条件触发。**先搭好再放量**。
6. **先 smoke 再放量**：任何新拓扑都先跑一个最小 cell 验证到底（起 1 个 server、少量 worker、
   跑通一集并核对产物），再按 §2 的规格扩到目标规模。上一条线每次跳过这步都出事。

**其它长期约束**：

- commit message 全英文；**绝不加 `Co-Authored-By: Claude`** 或任何 AI 署名（作者恒为
  `LinZiyang666 <3177267975@qq.com>`）。未经 owner 当次指示不得 `git add`。
- ⛔ **画图脚本一律不许 commit**（`build_*figure*` / `plot_*` / `render_*` / `edit_*figure*` /
  `*_editor.html`）。`exp/rit_pareto/build_figure.py` 与 `edit_figure.py` 属于另一个 session，**碰都不要碰**。
- ⛔ 不要 `rm -rf`；删除前先 `wc -l` 核对规模。
- ⛔ **共享机上不要 `pkill -f`** 裸模式：它会匹配到发起命令的 shell 自己。用字符类
  `[w]orker_entry`，且**脚本正文任何地方（含注释、echo）都不能出现裸的匹配串**。
- 无人值守期间**禁起 `run_in_background` 后台任务**（触发审批弹窗阻塞会话），用 Monitor。
- 报时刻用本机本地时间（America/Chicago）。
- **巡检就只巡检**：贴 PROBE 行，不要顺手做额外分析（owner 明确要求过）。

## 1. RIT 离线标定（LOTO）× GR00T N1.5 × LIBERO — 本线唯一运行手册（2026-09-14 写）

> ⚠ §0 第 1 步提到的 `logs/cache_prune_run_handoff.md` 是**另一条线**的入口，与本线无关，读了也别照它跑。
> 本线的权威 = 计划 `logs/rit_loto_calibration_plan.log.md`（G1/G2 均 APPROVED，§2 设计、§5 顺序、§8 冻结裁定）+ 本文件 §1–§5。
> 代码已 commit `d50621e` 并 push 到 `origin/Ziyang`（2026-09-14）。
> **2026-09-14 17:15–20:35 正式实验已全部跑完**（任务 4 → 1 → 2 → 3，全部门 PASS，无重跑）：产物在 weilandserver `/data/libero_cache/rit_loto/<suite>/`（8.1 GB 含 60 个闭环 H5），小产物已 pull 到本地 `exp/rit_loto/data/<suite>/`（不入库），结果写在 **`exp/rit_loto/analysis/results.md`**（数字 + sha256 + 复算命令），图在 `exp/rit_loto/analysis/figures/`（渲染脚本 `analysis/plot_*.py` 不入库）。闭环判读：warm75 超越率 0.068 [0.056, 0.083] → `within_preset_tolerance_in_support`；FULL 无派发 → `insufficient_evidence`。**未 commit、未 push**；本地工作树改动 = `ops/launch_verify_clients.sh`（两处 timan107 适配，见 §1.3 步 6）+ 本文件 + 新文件 results.md / figures / plot 脚本 / data。四台机器上本线进程与 tmux 已全部清空，cron 巡检已删。下次若重跑：从 §1.3 步 0 起，代码树 `/data/openpi_loto`。

### 1.1 这条线要什么（owner 口径，已冻结）

- 目标：把 RIT 阶梯的 shadow rollout 标定换成**纯离线**标定。查询集 = W13 建库语料全部 500 集/suite 的每个决策点；库内轨迹（每任务 5 条）检索时整条排除（LOTO），库外 450 条（含失败集）用完整库；每档动作从 winner 的存储中间态在当时观测下续跑；ref = 采集时执行的 `clean_action`；D 按 Eq. 16（W 全库共享、只比前 H_exec=5 步）；拟合只调现有 `fit_ladders` LP。
- 四个任务：**4 噪声地板**（500 决策/suite × 两噪声种子）→ **1 LOTO 表 + parity 门**（两 suite）→ **2 等价性对比**（LOTO 曲线 vs 原 shadow 曲线，含 in_library 分层与 ψ(s)）→ **3 闭环验证**（只 LIBERO-10，50 集，K=2 IR70 臂，gate 开）。
- 冻结裁定：`fit_source=loto_all`；K=2；目标 IR 70；A 池排除 shadow cohort 后每任务 5 正式 + 1 smoke，seed 20260913；α=0.05、闭环超越率容差 0.05；bootstrap B=1000；不改 `.gitignore`。
- **停止条件**：两 suite 的 noise_floor / parity_gate(PASS) / loto_table / fits+compare+bootstrap_band 全部产出，LIBERO-10 的 frozen_run + smoke(10 集) + verify(50 集) + collect/label/report 产出，小产物 pull 回本地 `exp/rit_loto/data/<suite>/`，写完 `exp/rit_loto/analysis/results.md`。**不 commit、不 push**（owner 回来再说）。
- **任何门 FAIL 就停下写清原因，不要放宽阈值、不要绕**（parity 门、`require_code_identity`、`validate_*` 的 SystemExit 都是设计内的拒绝）。

### 1.2 设备与路径

| 节点 | 角色 | 路径 |
|---|---|---|
| **weilandserver**（4090 49 GB，88 核，251 GB） | 本线**所有** GPU 与 CPU 计算 + 任务 3 的 server | 代码树 **`/data/openpi_loto`**（2026-09-14 17:14 起；= `/data/openpi_lg` 的整树拷贝 + HEAD 版 `src/openpi/cache/groot/staged.py` + 三个 ops 脚本只改 `REPO=` 行；原因见 §1.2 第二条；**本线文件必须与 `d50621e` 逐字节一致**，见 §3 步 0）；岛 venv `/home/weiland/gr00t_n15_venv/.venv/bin/python`；gr00t `/home/weiland/gr00t_n15`；主 venv `/home/weiland/openpi/.venv/bin/python`（CPU 阶段用，**必须** `PYTHONPATH=/data/openpi_lg/src:/data/openpi_lg` 且 `cd /data/openpi_lg`）；语料 `/archive/libero_cache/build_spatial_w13/libero_spatial`、`/archive/libero_cache/build_libero10_w13/libero_10`；库 `/data/libero_cache/libraries_w13/<suite>/<suite>_w13_S3.pkl`；ckpt `/home/weiland/ckpt_n15_libero_{spatial,10}`（软链到 `/data/ckpt/n15_libero_*`，7.6 GB）；模板 `/data/openpi_lg/exp/libero_groot/config/rit/<suite>/template.yaml`；成本表 `exp/libero_groot/config/rit/cost_groot_libero_measured.json`；原 shadow `exp/libero_groot/data/rit/shadow/<suite>/shadow_rows.jsonl` + `exp/libero_groot/config/rit/<suite>/arm_record.json`；**产物根 `/data/libero_cache/rit_loto/<suite>/`** |
| **timan107**（8×1080） | 任务 3 的 LIBERO client（一任务一进程，10 进程） | 代码树 `/scratch/zixuans8/openpi_lg`；`HOME=/home/zixuans8`；client 环境 `conda run -p /scratch/zixuans8/libero_sim`；池与 filter 放 `/scratch/zixuans8/rit_loto/libero_10/`（`/scratch` 不在 allow_roots，**经 `/tmp` 中转**） |
| h100 / timan108 | 本线不用（备用）；timan108 只有 3 张卡 | — |

- **⚠ 岛树隔离（2026-09-14 17:14 决定）**：`/data/openpi_lg` 被别线（tmux `fwgsrv*`，`/data/libero_cache/search/grid6`）当运行树用，且别线会 tether push 未提交的 src 改动（当日 15:58 推了另一版 `src/openpi/cache/groot/staged.py`，与 HEAD 不同）。本线整条链 fail-closed 于 `require_code_identity`，中途任何 CODE_FILES 变动都要从噪声地板重来，所以把本线搬到独立拷贝 **`/data/openpi_loto`**（`cp -a`，136 MB；23 个 CODE_FILES 全部核与 HEAD 一致；18 个本线文件与 `d50621e` 一致，仅 `run_noise_floor.sh / run_loto_table.sh / launch_verify_server.sh` 的 `REPO=` 行改为 `/data/openpi_loto`）。**本线一切命令的 `/data/openpi_lg` 都读作 `/data/openpi_loto`**；主 venv 的 `PYTHONPATH=/data/openpi_loto/src:/data/openpi_loto` 且 `cd /data/openpi_loto`。⚠ 副作用待告知 owner：17:00 演练时曾把 `d50621e` 版 `serve_groot_libero.py` 推到 `/data/openpi_lg`，覆盖了别线带 `--stage1-only` 的版本（别线在跑的 6 个 server 是 16:31 启动的、不受影响；若别线重启 server 会报 unrecognized argument，需其重推自己的版本）。
- **owner 裁定（2026-09-14）：h100 / weilandserver / timan107 / timan108 都可用，机上别线进程负载小，本线可与之并跑。** 因此两个 suite 的 GPU 阶段（步 1–3）可在 weilandserver 上**并行起两条 tmux**（各占约 8 GB 显存；4090 launch-bound，总时长与串行相近但省人工串接）；h100 无 W13 语料，默认不用。
- 2026-09-14 17:00 实测占用：weilandserver 有别线 6 个 server（tmux `fwgsrv23160…65`、端口 23160–23165 与 23170，显存 12 GB）；timan107 8 张卡各被别线 worker 占 3–6 GB（tmux `fwgag*`）。**本线端口用 23150、tmux 用 §1.2 的名字，别碰 fwg* 会话。**
- 公网直连：timan107 → weilandserver 用 `ziyanglin.com:23150`（端口段 23100-23199 是多 session 共享命名空间，起 server 前 `ss -tln` 看端口、`tmux ls` 看归属，别人的不动）。
- tmux 名：`lotonf_<suite>`（噪声地板）、`lotopar_<suite>`（parity）、`lototab_<suite>`（全表）、`srv0`（任务 3 server，若被占改 srv1…并同步改命令）、timan107 `lotocli_<tag>_<t>`。日志一律 `/tmp/<tmux名>.log`（tee）。
- tether：`tether push <local> <nid>:<remote> --force`（≤2 GiB/文件，目标已存在必须 `--force`）、`tether pull <nid>:<remote> <local>`、`tether exec <nid> -- bash -lc '…'`（**单引号内不能再有单引号**，多行脚本先落盘 push 再 exec；单次 exec 约 10 min 上限，长跑放 tmux）。远端 `export HOME=/home/weiland`（timan 是 `/home/zixuans8`）。

### 1.3 运行顺序与命令（每步 fail-closed）

**初始化演练记录（2026-09-14 17:02 已做一次，全部通过）**：四台 ONLINE；21 个文件（18 提交版 + probe.sh + 两份 shadow_rows.jsonl）已推到 `/data/openpi_lg` 且 sha 双侧一致；`launch_verify_clients.sh` 已在 timan107 `/scratch/zixuans8/openpi_lg/exp/rit_loto/ops/`；主 venv + `PYTHONPATH=/data/openpi_loto/src:/data/openpi_loto` 能从岛树导入 `openpi` 与 `exp.rit_loto.*`；`/data/openpi_lg/exp/common/data/db_init/libero/libero_10_apool` 有 10 个 `.init`；`shadow_rows.jsonl` sha == `arm_record.shadow_sha256`；`probe.sh` 可跑。会话 compact 后步 0 只需重核 sha（`cat <f> | sha256sum` 双侧），不必重推。

**步 0 对齐代码**（每次会话开头都做；2026-09-14 17:14 起目标树是 `/data/openpi_loto`，`/data/openpi_lg` 不再动）：本线 18 个文件（`git show --stat d50621e` 列表）逐个 `tether push` 到 weilandserver `/data/openpi_lg/<同路径>`，再双侧 `cat <f> | sha256sum` 对账（比对内容不带路径）。timan107 只需要 `exp/rit_loto/ops/launch_verify_clients.sh`（经 `/tmp` 中转到 `/scratch/zixuans8/openpi_lg/exp/rit_loto/ops/`）。⚠ 本地工作树里 `src/openpi/cache/*` 等有**别线未提交改动**，本地不能跑任何 CPU 阶段（`require_code_identity` 会拒），也不要把这些文件推到岛上；只推 `git show d50621e:<path>` 的内容（本线文件在本地工作树 = 提交版，直接推即可，除 `exp/libero_groot/serve_groot_libero.py` 与 `logs/README.md` 必须推 **提交版**：`git show d50621e:<path> > /tmp/x` 后再 push）。

**步 1 任务 4 噪声地板**（岛 venv，GPU，每 suite ≈3–5 min）：
`tether exec weilandserver -- bash -lc 'bash /data/openpi_loto/exp/rit_loto/ops/run_noise_floor.sh libero_spatial'`，看 `/tmp/lotonf_libero_spatial.log` 出现 `NOISE_FLOOR_EXIT=0`，产物 `noise_floor.json`（要求 `sample.per_task=50, n_rows=500`）。再跑 `libero_10`。顺序跑，不并行。

**步 2 任务 1 parity 门**（每 suite ≈2 min）：`bash /data/openpi_loto/exp/rit_loto/ops/run_loto_table.sh <suite> parity` → `parity_gate.json` 必须 `"status": "PASS"`（阈值 = 0.1 × 噪声地板中位数；冒烟时 parity_D 恒为 0，正式预期 PASS；FAIL 就停）。

**步 3 任务 1 全表**（spatial ≈40 min，libero_10 ≈1.5–2 h，5 行/s）：`bash /data/openpi_loto/exp/rit_loto/ops/run_loto_table.sh <suite> full`（内含 `--orchestrator-check 200` 自证，任何 mismatch 进程直接 SystemExit）。产物 `loto_table.jsonl` + `.record.json` + `.weights.npz`；record 的 `stats.rows` 应 ≈ 11,838 / 29,318，`episodes=500`。两 suite 可并行起（owner 裁定可并跑；显存够），也可串行。

**步 4 任务 2 拟合与对比**（主 venv，CPU，bootstrap 200×3 LP 用 `--jobs 16`，≈20–40 min/suite）：先把本地 `exp/libero_groot/data/rit/shadow/<suite>/shadow_rows.jsonl` push 到 `/data/openpi_lg/` 同路径（岛上现只有分片），核 sha 等于 `arm_record.json.shadow_sha256`。然后：
```
cd /data/openpi_loto && PYTHONPATH=/data/openpi_loto/src:/data/openpi_loto /home/weiland/openpi/.venv/bin/python -m exp.rit_loto.fit_loto \
  --suite <suite> --loto-table /data/libero_cache/rit_loto/<suite>/loto_table.jsonl \
  --loto-record /data/libero_cache/rit_loto/<suite>/loto_table.jsonl.record.json \
  --parity-gate /data/libero_cache/rit_loto/<suite>/parity_gate.json \
  --shadow-rows exp/libero_groot/data/rit/shadow/<suite>/shadow_rows.jsonl \
  --arm-record exp/libero_groot/config/rit/<suite>/arm_record.json \
  --template-yaml exp/libero_groot/config/rit/<suite>/template.yaml \
  --cost exp/libero_groot/config/rit/cost_groot_libero_measured.json \
  --out-dir /data/libero_cache/rit_loto/<suite> --jobs 16
```
放 tmux `lotofit_<suite>`。产物 `fits.json / compare.json / bootstrap_band.json`。判读只报 `compare.json` 的原始差值、参考带覆盖与 `max|Δq|/τ`，**不宣称"已证明可替代"**。

**步 5 发池与臂**（主 venv，只 libero_10）：
```
cd /data/openpi_loto && PYTHONPATH=/data/openpi_loto/src:/data/openpi_loto /home/weiland/openpi/.venv/bin/python -m exp.rit_loto.emit_verify_arm pools \
  --suite libero_10 --apool-dir exp/common/data/db_init/libero/libero_10_apool \
  --exclude-manifest exp/libero_groot/data/rit/shadow/libero_10/shadow_manifest.json --out-dir /data/libero_cache/rit_loto/libero_10
… emit_verify_arm arm --suite libero_10 --fits /data/libero_cache/rit_loto/libero_10/fits.json --fit-source loto_all --target-ir 70 \
  --cost exp/libero_groot/config/rit/cost_groot_libero_measured.json --template-yaml exp/libero_groot/config/rit/libero_10/template.yaml \
  --pool-manifest /data/libero_cache/rit_loto/libero_10/verify_pool_manifest.json \
  --original-arm-record exp/libero_groot/config/rit/libero_10/arm_record.json \
  --config-out /data/openpi_loto/exp/rit_loto/config/libero_10 --record-out /data/libero_cache/rit_loto/libero_10/frozen_run.json
```
（`apool_dir` 若在 `/data/openpi_lg` 缺失，从 `/home/weiland/openpi/exp/common/data/db_init/libero/libero_10_apool` 复制过去。）产物：`verify_pool/`、`smoke_pool/`、`verify_filter.json`、`smoke_filter.json`、`verify_pool_manifest.json`、臂 `loto_k2_ir70.yaml`、**`frozen_run.json`**（之后每一步都靠它）。目标 IR 不可达时 record 写 `unreachable`，停下报告。
把 `smoke_pool/ verify_pool/ *_filter.json` 推到 timan107 `/scratch/zixuans8/rit_loto/libero_10/`（经 /tmp）。

**步 6 任务 3 smoke（10 集）→ verify（50 集）**（2026-09-14 实跑记录：smoke 19:49–20:03 ACCEPT = 10 h5 / 0 tmp / attrs 含冻结 sha / sidecar 652 行 = Σnum_steps / hit 混合 WARM_START 435 + MISS 217（FULL 档在该臂 cut=∞ 故不会出现）/ 8/10 成功；verify server 20:05 起、10 client 20:06 起。⚠ 两个 launcher 修正已推 timan107 并改在本地工作树：① timan107 没有 `/home/zixuans8/miniconda3/bin/conda`，改直接用 `/scratch/zixuans8/libero_sim/bin/python` 并 export `CONDA_PREFIX/PATH/MUJOCO_GL=egl`（照别线 worker 的进程环境）；② timan107 的 tmux default-shell 是 dash，`${PIPESTATUS[0]}` 是坏替换导致 `LOTOCLI_EXIT` 永远写不出，改成 `> log 2>&1; echo LOTOCLI_EXIT=$? >> log`。smoke 用的是修正①未修正②的版本，验收改按 client 日志 `Total episodes` + 会话退出 + server 侧 h5 计数。）：
- server（weilandserver）：`bash /data/openpi_loto/exp/rit_loto/ops/launch_verify_server.sh /data/openpi_loto/exp/rit_loto/config/libero_10/loto_k2_ir70.yaml smoke /data/libero_cache/rit_loto/libero_10/frozen_run.json 23150 srv0`（启动即核臂/库/ckpt 内容/H_exec/schedule，失败会打印原因；ckpt 内容哈希首次约 1 min）。
- client（timan107）：`bash /scratch/zixuans8/openpi_lg/exp/rit_loto/ops/launch_verify_clients.sh ziyanglin.com 23150 smoke /scratch/zixuans8/rit_loto/libero_10/smoke_pool /scratch/zixuans8/rit_loto/libero_10/smoke_filter.json 1`。
- smoke 验收：`/data/libero_cache/rit_loto/libero_10/verify_logs/smoke/conn_*/groot_libero/` 共 10 个 `.h5`（无 `.h5.tmp`），每个 attrs 含 `loto_frozen_record_sha256`，sidecar `decisions.jsonl` 行数 = 各 h5 `num_steps` 之和，`hit_type` 有 FULL/WARM/MISS 混合。
- 然后 **杀 smoke server，重新起 verify server**（run_tag 是启动参数）：同上命令把 `smoke` 换 `verify`；client 用 `verify_pool` / `verify_filter.json` / trials `5`。50 集 ≈30–60 min。
- 收尾：全部 client 退出（10 个 `LOTOCLI_EXIT=0`），再关 server（按 PID kill，不 pkill -f）。

**步 7 collect → label → report**（依次；label 在岛 venv/GPU，其余主 venv）：
```
… -m exp.rit_loto.verify_closed_loop collect --log-root /data/libero_cache/rit_loto/libero_10/verify_logs --run-tag verify \
  --pool-manifest /data/libero_cache/rit_loto/libero_10/verify_pool_manifest.json --frozen-record /data/libero_cache/rit_loto/libero_10/frozen_run.json \
  --out-dir /data/libero_cache/rit_loto/libero_10/verify
（岛 venv，PYTHONPATH 加 gr00t）… verify_closed_loop label --suite libero_10 --frozen-record … --sampled-jsonl …/verify/sampled_decisions.jsonl \
  --sample-manifest …/verify/sample_manifest.json --library-pkl /data/libero_cache/libraries_w13/libero_10/libero_10_w13_S3.pkl \
  --template-yaml exp/libero_groot/config/rit/libero_10/template.yaml --checkpoint /home/weiland/ckpt_n15_libero_10 --out-dir …/verify
… verify_closed_loop report --suite libero_10 --frozen-record … --label-record …/verify/verify_labels.record.json \
  --sample-manifest …/verify/sample_manifest.json --verify-rows …/verify/verify_rows.jsonl --episodes …/verify/episodes.json \
  --decisions …/verify/decisions.jsonl --fits /data/libero_cache/rit_loto/libero_10/fits.json \
  --cost exp/libero_groot/config/rit/cost_groot_libero_measured.json --out-dir …/verify
```
label 的 `--limit` 只用于调试（产物名带 partial，report 拒收）。report 的三分判读（满足容差 / 超出容差 / 证据不足）按 `verify.json` 原文写进 results.md。

**步 8 收尾**：`tether pull` 每个小产物（json / jsonl，`loto_table.jsonl` 约 10–30 MB 可拉；H5 不拉）到本地 `exp/rit_loto/data/<suite>/…`（目录不入库）；写 `exp/rit_loto/analysis/results.md`（数字 + 文件 sha256 + 复算命令）；图用不入库的 `analysis/plot_*.py` 从 json 画。撤监控，关 server/worker，各机 GPU 归零。

### 1.4 时长预算（4090 实测冒烟外推）

noise floor 2×5 min；parity 2×2 min；全表 spatial 40 min + libero_10 1.5–2 h；fit 两 suite 各 20–40 min（CPU）；池/臂 1 min；smoke 10 集 ≈10 min；verify 50 集 30–60 min；label ≈5 min；report 1 min。**总计约 5–6 h**，全部串行也够。

## 2. 拓扑与角色（本线）

server / GPU / CPU 全在 weilandserver；client 只在 timan107；h100、timan108 不用。timan 族只当 worker 是硬规则。

## 3. 代码同步清单（步 0 用）

`d50621e` 的 18 个文件：`exp/rit_loto/{__init__,build_loto_table,noise_floor,fit_loto,loto_logger,emit_verify_arm,verify_closed_loop}.py`、`exp/rit_loto/ops/{run_noise_floor,run_loto_table,launch_verify_server,launch_verify_clients}.sh`、`exp/rit_loto/analysis/{README,data_inventory,smoke_evidence}.md`、`tests/exp/test_rit_loto.py`、`logs/rit_loto_calibration_plan.log.md`、`logs/README.md`、`exp/libero_groot/serve_groot_libero.py`。另加未入库的巡检脚本 `exp/rit_loto/ops/probe.sh`（§4）。岛上依赖的其它 21 个文件此前已核与 HEAD 一致。

## 4. 监控（先搭好再放量）

- **L3 cron（定时巡检，每 20 min）**：`bash /data/openpi_loto/exp/rit_loto/ops/probe.sh`（本会话 cron id `ced97b1e`，2026-09-14 17:17 起）（一行/条目：tmux 存活、各 suite 产物存在与行数、日志尾行、GPU 显存、verify_logs h5 计数、`.h5.tmp` 数）。**巡检就只贴 PROBE 行，不顺手做分析。**
- **L2 Monitor（条件触发）**：每个长阶段一个 Monitor，until-loop 抓 `/tmp/<tmux>.log` 的 `*_EXIT=` / `Traceback` / `SystemExit` / `mismatch` / `FAIL`，命中即出声；任务 3 期间再加一个抓 server 日志的 `Traceback|Error|1013|SystemExit` 与 client 的 `LOTOCLI_EXIT`。
- 无人值守期间**禁起 `run_in_background`**（审批弹窗会卡住会话）；等待用 Monitor 或 `tether exec … until … sleep` 的短轮询（≤9 min）。
- 报时刻用本机本地时间（America/Chicago）。

## 5. 坑与判据（跑之前读一遍）

1. **代码身份**：所有产物记 `code_sha256`，下游一律比对当前源码；只要岛上任何一个被列入 `CODE_FILES` 的文件变了（含 `src/openpi/cache/{config,orchestrator,groot/*}.py`、`exp/rit_pareto/rit_k.py` 等），旧产物就被拒，要从 noise floor 重来。**所以任务期间别动岛上的代码树**，也别把本地别线的 src 改动推上去。
2. **checkpoint 内容哈希**：每次准入都重算 7.6 GB（约 1 min，`~/.cache/rit_loto/ckpt_identity/` 只做校验缓存），server 启动、label 都会有这段静默，别当成卡死。
3. **噪声地板量级**：冒烟 20 行 D(ref1,ref2) 中位 6.87，与 shadow 各档 D（5.5–9.1）同量级；正式 500 行后如仍如此，写进 results.md（度量被教师随机性主导），不要据此改阈值。
4. **parity 门**：冒烟 parity_D 恒为 0（fp16 存储对 bf16 无损、去噪确定）；正式若非零也只报数，门本身只看 p90 ≤ 0.1×地板。
5. **W13 语料细节**：每 suite 500 集（spatial 450/50、l10 436/64 成功/失败）、库 S3 每任务 5 条成功轨迹全在语料内；LOTO 库内行必然有 `n_self_skipped>0`。
6. **LIBERO client**：`--replan-steps 5 --resize-size 256`（GR00T 必须 256）、`--episode-filter` 让 H5 attrs 记官方 `orig_init_state_idx`（没有它 attrs 会记成子池位置，collect 直接拒）；client 退出时 robosuite `__del__` 抛 EGL `EGL_NOT_INITIALIZED` 是良性；`MUJOCO_EGL_DEVICE_ID` 按任务号取模分卡。
7. **server**：`--concurrent` 下每连接一个 `conn_<id>/` 目录；smoke 与 verify **必须不同 run_tag、不同 server 进程**；日志判活别 grep websockets 握手 Traceback；1013 = collector/logger 拒第二连接（本线 logger 支持多连接，出现 1013 说明连错了 server）。
8. **collect 的拒绝是设计**：缺集、重复、`.h5.tmp`、attrs 不含冻结 sha、sidecar 决策号/控制步/task 不符，任一即 SystemExit 并打印差集；修根因后**重跑整段 verify**（不能只补集，run_tag 目录不能混）。
9. 共享机禁 `pkill -f` 宽模式（按 PID）；不 `rm -rf`；删除前 `wc -l`。
10. tether push 目标存在必须 `--force`；exec 单引号坑见 §1.2。

## 6. 存档：ActionCache 基线 × GR00T N1.5（LIBERO）— 待执行实验的记忆点（2026-09-13 写）

代码已 commit `ece4362`（`origin/Ziyang`，G1/G2 APPROVED，§6 Verify 全量通过、失败项均为 HEAD 既有）。
**实验已于 2026-09-13 跑完**（plan §8 步骤 2–6，两组 10 臂 × 500 集，raw 在 `exp/actioncache_baseline/data/runs/groot_<suite>_w13s3/`，`aggregate` 门全过；过程记录见 plan Review Log 末尾 Execution note）。下面的路径与门保留，供重跑 / 复现。
运行手册 = `docs/experiments/actioncache_baseline.md` **§9**（每一步的命令都在那里，照抄）。

### 6.1 这条线要什么

- 只跑**同库 W13-S3 组**（spatial / libero_10 各一个 50 轨迹库），**不跑 500 集库版本**；每组 = 2 档
  （n0 = FULL_HIT@CP2，n1 = WARM_START@**0.875**）× (4 目标臂 + 1 参考臂 θ_raw=**0.65**) = **10 臂 × 500 集**。
- key = action head **编码后**的 VLM 输出 [N,2048] pad 到 640 token + 编码后 state [1536] → 稀疏三值投影 d=500、p=0.01、
  **seed 20260904**（两 suite 同 seed，与臂 yaml `key_builder.cp2_groot.seed` 一致）。
- 成本：teacher 表 `exp/libero_groot/config/rit/cost_groot_libero_measured.json`（P=13.338, L=28.104, M=41.442，
  RTX 4090 `GPU-98d36ed2-…`，两 suite 共用，**不动**）+ 每 suite 实测编码单价 E；FULL=P+E，WARM=P+E+L/8，MISS=M+E，分母 M。
  E 只要 ≥ ~1.8 ms，n1 的地板就高于 45% → n1 整档按规则 fallback 到 `t01–t04`，**这是设计内的，不要当 bug**。
- 跑完只做 pull raw + `aggregate` 完整性审计；**不画图、不写 analysis、不写 results**（owner 明令）。

### 6.2 设备与路径（本线用到的）

| 机器 | 角色 | 关键路径 |
|---|---|---|
| weilandserver（4090 48G，GR00T 岛） | 建库 / parity / cohort 采集 / E 标定 / preflight / **lane B 的 5 个 eval server** | 代码树 `/data/openpi_lg`（**独立克隆，改完必须 tether push 并核 sha**）；岛 venv `/home/weiland/gr00t_n15_venv/.venv/bin/python`；gr00t `/home/weiland/gr00t_n15`；ckpt `/home/weiland/ckpt_n15_libero_{spatial,10}`；`PYTHONPATH=$G:$G/examples/Libero:$REPO/src:$REPO`，`HF_HUB_OFFLINE=1`；LIBERO client 用 `conda run -p /home/weiland/libero_sim`（`launch_collection.sh` 的写法，`/home/weiland/miniconda3/bin/conda`）——我的 `ops/run_acb_collect_clients.sh` 传 `<python>`，先试 `/home/weiland/libero_sim/bin/python`，smoke 不通就改成 conda run |
| h100（`149.165.153.233`，直连 `:23210-23214`；`/data`→`/media/volume/OmniData`） | **lane A（spatial）5 个 eval server** | 代码树同路径 `/data/openpi_lg`；库 / yaml / 成本记录要同步过去（sha） |
| timan107（8×1080）/ timan108（**3**×A5000） | worker（lane B 64 / lane A 64） | `/scratch/zixuans8/openpi_lg`，`HOME=/home/zixuans8`，run_gtp python `/scratch/zixuans8/openpi/.venv/bin/python`，`--conda-env /scratch/zixuans8/libero_sim`；`/scratch` 不在 allow_roots，经 `/tmp` 中转 |

数据：源库 `/data/libero_cache/libraries_w13/<suite>/<suite>_w13_S3.pkl`（weilandserver）；W13 H5
`/archive/libero_cache/build_{spatial,libero10}_w13/<suite>/episode_XXXX_<ts>.h5`（500 集，**无原始图像**，故用模板重建）；
RIT shadow manifest + init 池在仓内 `exp/libero_groot/data/rit/shadow/<suite>/{shadow_manifest.json,shadow_pool/*.init}`
（seed 20260901）；RIT 参考臂记录 `exp/libero_groot/config/rit/<suite>/arm_record.json`（compare 用，本轮不执行）。
计划产物位置：CP2 库 `/data/libero_cache/libraries_w13_cp2/<suite>/<suite>_w13_S3_cp2.pkl`；cohort H5
`/data/libero_cache/acb_shadow_h5/<suite>/attempt_<a>/srv<i>/`；E 记录 `exp/libero_groot/config/actioncache/cost_groot_cp2_encoded_<suite>.json`；
preflight `exp/libero_groot/data/actioncache/overhead_<suite>/overhead.json`；臂 `exp/actioncache_baseline/config/groot_<suite>_w13s3/`；
raw 拉回 `exp/actioncache_baseline/data/runs/groot_<suite>_w13s3/`（`config/`、`data/` gitignored）。

### 6.3 顺序与门（每一步 fail-closed，过不了就停，不要绕）

1. **parity 门先行**（`groot_cp2_parity.py`，两 suite 各 20 步 + 合成负例；需要 `--task-map`）→ 通过才建库。
2. `emit_task_map.py` **必须在 LIBERO client 环境**（`libero_sim`）跑：`task_id → (task.name, task.language)`。
   manifest 的 `task_name` 是下划线名（定位 `.init`），H5 的 `task` 是自然语言指令；两者只靠 task_id 绑定，**不能互推**。
3. 建库 `build_cp2_artifact_groot.py` → `verify_cp2_artifact.py --teacher groot_libero` → 同步 h100（同路径，sha）。
4. cohort 采集：`launch_acb_collectors.sh`（5 个**非并发** collector，weilandserver 直连段端口 **23130+i**，tmux `acbsrv<port>`，attempt 目录**不复用**；⛔ LIBERO client 不在 weilandserver 跑——按 §0 纪律放 timan107，`ACB_HOST=ziyanglin.com`，client 终态 JSON 搬回 collector 的 attempt 目录再验收）
   + `run_acb_collect_clients.sh`（client i **串行**跑 task i、i+5，tmux `acbcli<i>`；每 collector 同一时刻只能有一个连接，
   第二个连接会被 1013 拒掉）→ `verify_shadow_h5.py --task-map …`（成功集 `success` 终态；失败集只收 `step_cap` 且
   steps==max_steps+10，即 230/530；同 attempt 重复 = 该集作废；补跑用 `<acc>/retry/filter_task_<t>.json` 起新 attempt）。
   两 suite 顺序做（换 checkpoint 要重启 server）。≈12 min/suite + 起服 4 min。
5. shadow 表 `build_shadow_table_groot.py`（accepted manifest 必须 `ok:true` 且 `task_map_bound:true`；**不要用 `--limit-episodes` 的表发臂**，会被拒）。
6. E 标定 `ops/run_cp2_encoder_cost.sh`（独占 4090 + nsys，N=566 / 30 warmup / 200 iters / reduce-overhead 冻结；
   每次重测，旧记录自动归档；`debug_sampling` 的记录不能认证）+ preflight `ops/run_cp2_overhead.sh`（warm total P95 ≤10 报告 / 10–40 记提示 / >40 停发臂）。
7. 出臂（主 venv）`export_arms.py --teacher groot_libero --lib-tag w13s3 --cost-record … --encoder-cost-record … --preflight-record …`
   → 恰 10 臂 `acb_<sp|l10>_w13s3_<n0|n1>_<t01..t04|ref650>`；yaml/库/成本记录同步两台 server 与两台 sim box（sha）。
8. eval：server `launch_eval_servers.sh`（5 进程/lane，`--allow-dynamic-bundles`）；client 在 sim box 上
   `ops/run_acb_eval_group.sh <suite> <arm_matrix.yaml> <servers> <workers> <gpus> <run dir>`（`--checkpoint cp2 --warm-tiers 0.875`）。
   **先 smoke**（n0+n1 各 10 集，两 suite 共 40 集，主评测之外），再主跑 lane A spatial（h100×5 + timan108 64w）∥ lane B l10（weilandserver×5 + timan107 64w）。
   监控照 pi0.5 线：cron 20 min 巡检 + Monitor 条件触发；每组完成 pull raw（sha）+ `aggregate` 审计；全部完成关 server/worker、撤监控。
   RIT 旧吞吐（lane B ≈18.8 集/分）只作排期参考；encoded CP2 的实际吞吐以 smoke 为准。

### 6.4 踩过的坑（本线代码里已修，跑的时候别再犯）

- backend 检索**不能放在 `runner.session()`（autocast bf16）里**，否则 cosine 是 bf16 的，与在线 `check()` 口径不一致。
- `tests/review_tests/` 是审查方私有材料，**不许拿来自测**（R1 因此被记流程违规）。
- W13 H5 无 `task_id`/`orig_init_state_idx` attrs 不要紧（建库只用 trajectory_id/step_idx）；但 **cohort 采集**的 H5 必须有这两个 attrs
  （client 经 `--episode-filter` + `episode_start` 打上），验收硬性要求。
- `bench_cp2_overhead_groot --mode encoder-cost` 会先核对 GPU uuid 与 teacher 表一致、live schedule 为 k8；checkpoint 允许与表不同（两 suite 各自的 ckpt）。
- 主 venv 的全量 pytest 有 10 个 HEAD 既有失败（robocasa365 / dispatch_surface / exp/test_prebuilt_matrix_backend），与本线无关，别去修。
- parity 第 1 项对 fp16 次正规区只容忍 1 ulp（已修）；E 标定的 compile 区域不能出现 `BatchFeature`/graph break，否则 Dynamo 每次重编译、E 假到 370 ms 而认证照过（已修为张量孪生 + `fullgraph=True`）。
- `pkill -f "[x]yz"` 的字符类只防 pkill 自己，**同一条 `bash -lc '…'` 里别处出现的裸串仍会把整个 shell 杀掉**——kill 前后的命令拆成两次 tether exec。
- 吞吐参考（weilandserver×5 + timan107 64w）：spatial ≈ 46 集/min（1.8 h / 5000 集），libero_10 ≈ 15–23 集/min（5.2 h）。

## 7. 融合权重 → key builder × LDA 补充实验（本 session，2026-09-15 写；11:15 更新：§4 Code 与准备阶段完成，G2 R1 NEEDS REVISION → R2 重交 → **R3 APPROVED**（owner 授权审查方直接修复，执行方复核后进 §6 Verify → commit/push；实跑未开始））

### 7.1 已完成（全部 commit 到 `a27a707`，iclr 仓 `9bb5d15`）

- 融合权重 $w_f$ 定案：主方法 = 相位判别 LDA（$w\propto\Sigma^{-1}d$，闭式零 rollout），候选 = 7 维动作误差网格。唯一现行纪要 `docs/iclr/modality_weight_selection.md`
  （§2 公式、§4.5 四臂 A 池重跑、§4.6 统一 1/6 网格 56k 集重做、§6 论文英文句）；实验记录 `exp/weighted_sum/analysis/low_cost_weights_results_claude.md`；
  数据 `exp/weighted_sum/data/{fusion_ablation,grid6}/`（gitignored，`summary_final.json` / `summary_grid6.json`）。
- 终读数（A 池 500 集）：网格最高 0.688 / 0.486 / 0.744 / 0.468（π0.5 Sp / π0.5 L10 / GR00T Sp / GR00T L10）；LDA 最近格差 1.8 / 5.0 / 1.4 / 0.4 pp，只有 π0.5 LIBERO-10 显著（峰在 rs=0 棱上）；动作误差 argmin 四 suite 都显著落后。
- GR00T LIBERO serving 提速已落地（`exp/libero_groot/serve_groot_libero.py --compile-stage1 --stage1-only`；`src/openpi/cache/groot/staged.py` 分位数等价门 + 首连接线程内编译/预热/录制）：单 replica 59 集/分、2.0 GB（eager 20 集/分、5.8 GB）。

### 7.2 下一步：`logs/keybuilder_lda_supplement_plan.log.md`（G1 APPROVED R2 → §4 Code 完成 → G2 R3 **APPROVED**，见 plan 文末 Review Log；下一步 = plan §4 步 3–4：推送 h100 + 预检门 + 两轮实跑）

**已做（2026-09-15 上午）**：下列 1–2 全部完成并留证（`logs/fusion_weight_ablation_run.md` §7.1）：新文件 `exp/weighted_sum/emit_keybuilder_lda.py`、`tests/exp/test_keybuilder_lda.py`（35 passed），改 `fw_lane_pi05.sh` / `lcw_ablation_summary.py` / `lcw_fit_weights.py`；h100 建成 libero_10 `cp1_llm_l0_prefix_mean_pool.pkl`（sha `93d3d35d…`，已拉回本地，§4.1 验收全过）；补标定 3 份、LDA 8 个、yaml 10 份 + matrix 4 份 + `weights.json` + `active_manifest.json` 全部生成并回读验收。**全部在工作树未 staged**（owner 规则：未经指示不 `git add`）。
**G2 之后才做**：3–4（推送 h100 + sha 对账、plan §6 预检门、pool 轮 / LLM 轮、汇总、纪要 §4.7）。

原顺序（保留作对照）：
1. `exp/weighted_sum/emit_keybuilder_lda.py`（templates / final 两阶段）、`fw_lane_pi05.sh` 加 `FW_STAGE2_DEVICE` / `FW_MATRIX`、`lcw_ablation_summary.py` 加 `--input-manifest` 严格入口、`lcw_fit_weights.py` 加 normalizer/退化校验、`tests/exp/test_keybuilder_lda.py`。
2. h100 建 libero_10 的 `cp1_llm_l0_prefix_mean_pool.pkl`（`exp/common/build_in_memory_cache_artifact.py --builder-type cp1_llm_layer_extract --extract-layer 0 --prefix-reducer-type prefix_mean_pool --checkpoint-dir /data/openpi/checkpoints/pi05_libero_pytorch --config-name pi05_libero`，原料 = 建 `cp1_spatial_pool_16.pkl` 的同一批 H5，先找到它的来源目录），逐条对齐验收 → 补标定（libero_10 clip_b32、两 suite llm）→ LDA → yaml → 逐份 load 验收。
3. 预检门（plan §6）：CLIP 按"一臂一端点、W=12"实测显存与连接覆盖；LLM parity `PI05_CHECKPOINT_DIR=... PI05_CONFIG_NAME=pi05_libero uv run pytest tests/cache/test_llm_layer_extract_parity.py --run-manual -m manual -v`；各路径 10 集 smoke。
4. pool 轮（4 臂 × 2 suite，`FW_EVAL_CONC=4`，stage2 meta）→ LLM 轮（1 臂 × 2 suite，stage2 cuda:0，单端点）；拉回 → `--input-manifest` 汇总 → 纪要 §4.7。

### 7.3 设备与现状（2026-09-15 01:00）

| 机器 | 状态 | 本线用法 |
|---|---|---|
| h100 `149.165.153.233` | **11:00 起被另一 session 占用**：4 个 GR00T server（tmux `ort_srv_frontier_rprime_2320[1-4]`，30 GB、util 96%）；本线 kb_build / kb_h5pull 两个 tmux 已完成可清；新增 `/data/openpi/exp_common/db/libero_cache/libero_10`（50 H5）与 `exp/common/data/cache_artifacts/libero_10/llm_layer_extract/` 新库；其余：显存需按预检实测；`/home/weiland/openpi` 是非 git 副本，含 π0.5 库、A 池、`pi05_libero_pytorch` ckpt；`/data/openpi_lg` 的 GR00T 树**过期**（serve_groot_libero 旧、无原 S3 库） | π0.5 lane：`fw_lane_pi05.sh` + timan108 `cache_prune/ops/launch_fleet.sh`（ports 23280–23284，driver 23290） |
| weilandserver | 另一 session 的 online RIT 线在跑（GR00T server :23181–23184、loto/disagreement 进程），显存 20–45 GB 波动 | 本线不碰 |
| timan107 | 空 | 不用 |
| timan108 | 另一 session 48 个 worker（`ort_cli_*`，连 weilandserver :23181/:23183） | 起本线车队前 `tmux ls` 侦察，只动自己的 `cpag*` session，禁 `stop_fleet.sh` 的全局清扫 |

### 7.4 本线踩过的坑（都在今天）

- **`/data/openpi_lg` 与 `serve_groot_libero.py` / `staged.py` 是多 session 共用文件**：另一 session 17:00 用它的 HEAD 版覆盖了含 `--stage1-only` 的远端文件，LIBERO-10 相起服失败一次。推送前先 pull 对方现行版再合并；本地工作树同一文件叠着两边改动，commit 时注意。
- **run_size_eval 的调度合同 = 一臂一端点**（`PureCacheEvalStrategy.plan` 每臂一个 stage，`assign_servers` 一个端点），并发由 `--eval-concurrency` 决定；"60 worker" 不等于单臂 60 连接。CLIP builder 逐连接懒加载模型，显存按该臂端点的 W 算。
- **GR00T r1 harness（`orchestrate_search.py`）启动会 reap timan107 上全部 `lw*` worker**，不能与别的 orchestrator 并发；且一 cell 一 server。要吞吐走 conductor + `fw_groot_conductor.sh`（compile + stage1-only replica）。
- 编译视觉塔的等价门原本卡逐 token 余弦最小值 0.999，低范数 token 会把它压到 0.9；已改分位数判据（`OPENPI_STAGE1_GATE=strict` 可切回）。CUDA graph 录制必须在首个连接的线程里连做三次调用（cuDNN 句柄线程局部），已改。
- tether `bash -lc '...'` 内 `pkill -f` 会匹配到自己的 shell（"child terminated by signal"），用 tmux kill-session 或 `[p]attern`。
- 历史 π0.5 网格在 B 池 n=100、H200 上，与 A 池数字不能并表；GR00T 同 harness 逐集确定（Spatial 100%、L10 93.8%），跨 harness 81%。
