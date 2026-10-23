# 长执行段关：π0.5 / GR00T · LIBERO 旋钮批次（成功过滤库）交回（10-22）

只建了定义、脚本和代码，没有 prepare、没有 launch，也没起任何服务或 GPU 进程；没碰 4090；没 commit。
四个旋钮关单位之后由协调者起了：链 h100lo1 跑 fa_lo_pi_f_f_o_q → fa_lo_gr_f_f_o_q，链 h100lo3 跑 fa_lo_pi_r_f_o_q → fa_lo_gr_r_f_o_q。在跑的进程和链我都没碰。
协调者为了让常驻起来，在两份 Q3 服务树（本机和 H100）里把 `local_server.authenticated` 的 raise 改成了 `return True`。这是协调者的改动，不是我的。
追加任务：H100 同卡纯推理参照（第 3 节），以及让各档只认 H100 参照。
owner 10-22 裁定：参照用 **C5 普通纯推理**，每 5 个控制步调用一次模型，不带长执行段，在 H100 上跑，步数与批次相同。第一版 C10 参照（fa_lo_*_p_q）已在 QUEUE.json 里标为 UNUSED，目录没删。
tether 只用了 exec。文件走 H100 公网 IP 的 rsync（16 MB/s 限速）。
路径默认相对 `/home/weiland/projects/openpi`。

## 1. 结论

- 新批次就是 q3_port_1020 的对应批次，只去掉 `--os-blind --os-policy-tail --os-policy-tail-blocks 1` 这三个参数。其余全部不变：
  - 同一批 fit 文件（逐字节相同），同一 Q3 停滞门（`--os-judge stall_only`），同一 PERIODS.json 周期，同一 TaskDescentKnob；
  - episode_subset、eval500 初始状态、种子 2026101401、K、副本数、recording 关，也都一样。
  - 已逐项比对生成的定义：除臂名和路径外，只有 plugin_args 少了这三个参数；yaml、eval500、fit 的 sha 全部相同。
- 服务端全在 H100：
  - 全步（R15）用现成的 `SERVER_HOST=h100` 机制，从 H100 的 `/data/oscl_h100/openpi` 起服务。π0.5 和 1020 完全相同，GR00T 从 4090 挪了过来。
  - 减步（R16）原本没有远程常驻机制，加了一层很薄的包装：conductor_runner 的 local_server 启停改成用 tether exec 调用 H100 上的同一个 local_server，代码树是 Q3 服务树复制到 H100 的那一份。driver 和 agent 的逻辑都不变。
- 4090 完全不用。R16 永远带 `--no-local-workers`。worker 由中央调度器分到 a100、timan107、timan1。
- 各档选格只和 **H100 同卡、同步数的 C5** 纯推理参照比：fa_lo_pi_f_p5_q、fa_lo_gr_f_p5_q、fa_lo_pi_r_p5_q、fa_lo_gr_r_p5_q。
  - 参照没跑完时 `build_levels` 直接报错。
  - 不用 4090 的参照，不用 C10 参照，也不用 fa_pi_f_p10 和 fa_pi_f_ps。

## 2. 新批次

| 白话（库 · 模型 · LIBERO · 步数 · 阶段） | 名字 | 局数 | 服务 | 副本 | 脚本（commands/） |
|---|---|---:|---|---|---|
| 成功过滤库 · π0.5 · LIBERO · 全步 · 长段关 · 旋钮关 | fa_lo_pi_f_f_o_q（8 格 `_l50/_l100/_l200/_l500/_s50/_s100/_s200/_s500`，每格 500） | 4,000 | H100 R15，端口 23230–23232，每副本 NEED_MB 9000（同 1020） | 3 | `prepare_/launch_fa_lo_pi_f_f_o_q[_<格>].sh` |
| 成功过滤库 · GR00T · LIBERO · 全步 · 长段关 · 旋钮关 | fa_lo_gr_f_f_o_q（8 格） | 4,000 | H100 R15，端口 23233–23237，每副本 7200 MB，OMP=2 | 5 | `prepare_/launch_fa_lo_gr_f_f_o_q[_<格>].sh` |
| 成功过滤库 · π0.5 · LIBERO · 减步 K1 · 长段关 · 旋钮关（一个 R16 group，8 格 × 500） | fa_lo_pi_r_f_o_q | 4,000 | H100 常驻，端口 23150–23152，每副本 10000 MB，OMP=3；driver 在 weilandserver 23157 | 3 | `prepare_/launch_fa_lo_pi_r_f_o_q.sh` |
| 成功过滤库 · GR00T · LIBERO · 减步 K1 · 长段关 · 旋钮关 | fa_lo_gr_r_f_o_q | 4,000 | H100 常驻，端口 23153–23156（`--groot-suite-servers 3,1`），每副本 7200 MB，OMP=2；driver 23158 | 4 | `prepare_/launch_fa_lo_gr_r_f_o_q.sh` |
| 无库（纯推理）· π0.5 · LIBERO · 全步 K10 · C5 · H100 同卡参照 | fa_lo_pi_f_p5_q（`_l10`、`_s` 两个根，每个 suite 500；照 fa_pi_f_p10/ps 建） | 1,000 | H100 R15，端口 23261–23263，每副本 9000 MB | 3 | `prepare_/launch_fa_lo_pi_f_p5_q[_l10/_s].sh` |
| 无库（纯推理）· GR00T · LIBERO · 全步 K8 · C5 · H100 同卡参照 | fa_lo_gr_f_p5_q（`_l10`、`_s`） | 1,000 | H100 R15，端口 23250–23253，每副本 7200 MB | 4 | `prepare_/launch_fa_lo_gr_f_p5_q[_l10/_s].sh` |
| 无库（纯推理）· π0.5 · LIBERO · 减步 K1 · C5 · H100 同卡参照 | fa_lo_pi_r_p5_q（`_l10`、`_s`） | 1,000 | H100 R15，端口 23254–23256，每副本 8800 MB | 3 | `prepare_/launch_fa_lo_pi_r_p5_q[_l10/_s].sh` |
| 无库（纯推理）· GR00T · LIBERO · 减步 K1 · C5 · H100 同卡参照 | fa_lo_gr_r_p5_q（`_l10`、`_s`） | 1,000 | H100 R15，端口 23257–23260，每副本 7200 MB | 4 | `prepare_/launch_fa_lo_gr_r_p5_q[_l10/_s].sh` |
| （不再使用）C10 参照 | fa_lo_gr_f_p_q、fa_lo_pi_r_p_q、fa_lo_gr_r_p_q | — | QUEUE.json 里 status=UNUSED，目录和脚本保留 | — | 不要跑 |
| 上面四个旋钮关的各档 ir45/ir40/ir32/ir25（旋钮关和对应参照都跑完后再建） | fa_lo_pi_f_f_l_q、fa_lo_gr_f_f_l_q、fa_lo_pi_r_f_l_q、fa_lo_gr_r_f_l_q | 每个要调的格 2,000，最多 16,000 | 与对应的旋钮关相同 | 同上 | 先 `build_levels_<名>.sh`，再 `prepare_/launch_<名>.sh` |

- 合计：旋钮关 16,000 局，C5 参照 4,000 局；各档最多 4 × 16,000 = 64,000 局，实际按旋钮关结果选格。
- 运行目录：
  - 本机：`/data/mw_finetune/eval/fa_lo_*`，实际落在 /nvme（eval 本身就是指向 /nvme 的软链）；
  - R16 定义：`/nvme/weiland/openpi/r17_lo_1022/definitions/`，本机另有别名 `/home/weiland/projects/openpi_ext/r17_lo_1022`，H100 上也用同一个地址；
  - 索引：`longseg_off_1022/QUEUE.json`；
  - 建批日志：`/nvme/weiland/openpi/r17_lo_1022/logs/`。

**各档怎么建**：`build.py levels`。
- 用 q3_port_1020 的装载器读旋钮关结果。
- 先读 H100 同卡的 C5 参照（`build.references`）：四种都读 QUEUE.json 里 segment=c5 的 `*_p5_q` 单位。
  - 任何一个 suite 的参照没有 summary.json，或者不满 500 局，就报 `H100 pure reference … is not complete` 并停下。
  - 不读 4090 的参照、C10 参照，也不读 fa_pi_f_p10/ps。
- K：全步 π0.5 K10、GR00T K8，减步 K1。
- 某格完整的 500 局成功数低于参照，就调这一格。
- 已在 CPU 上试过：π0.5 全步、GR00T 减步都报「fa_lo_*_p5_q_l10 未完成」。
- 档位和周期照搬 1020 的 PERIODS.json，不重新标定（owner 裁定）。
- 要用的 fit 基本都已存在：π0.5 8 格 × 4 档都有；GR00T 只缺 spatial_500 的各档，缺的会在单进程里用 1020 的 knob_blob 现建。
- 有格没跑完就报 pending，什么都不建。已用空结果试过两次，流程能走到 pending，纯推理参照装载正常（pure_n=500）。

## 3. H100 同卡纯推理参照（追加）

**现在用的是 C5 版 `fa_lo_*_p5_q`**（owner 10-22）。它和下面描述的 C10 版只差一点：插件参数里去掉 `--os-blind --os-policy-tail --os-policy-tail-blocks 1`，每次调用执行 5 步。
- π0.5 全步 C5 照 fa_pi_f_p10 和 fa_pi_f_ps 的 spec 新建（`fa_lo_pi_f_p5_q`）。
- 已逐项比对：参数相对源参照只少这三个、judge 换成 stall_only、fit 换成 c10 纯推理 fit；eval500 相同，K、env、副本数和源参照一样。
- 下面的说明写于 C10 版，同样适用于 C5 版。

**源参照是什么**：formal-A 用的纯推理参照 A（`formal_a_analysis.vla_references`）每个 suite 一个根，各 500 局：

| 新单位 | 源参照（4090） |
|---|---|
| fa_lo_gr_f_p_q | `r14dt_groot_{l10,spatial}_50_pure_c10` |
| fa_lo_pi_r_p_q | `r16p1_pi05_{l10,spatial}_k1_pure_c10` |
| fa_lo_gr_r_p_q | `r16p1_groot_{l10,spatial}_k1_pure_c10` |

这些都是 R15 legacy 根（R16 的 K1 参照也是走 legacy 跑的），所以新参照也用 R15，加 `SERVER_HOST=h100`。

**照抄的部分**：
- 每个新根从源根的 arms_in.json 拷 spec，eval500 逐字节相同；
- client_overrides（replan 5，GR00T resize 256）、yaml（K1 的 yaml_patch `miss.num_steps 1`）、server_env（GR00T K1 的 `GROOT_DENOISING_STEPS=1`）都照抄；
- 副本数照抄（GR00T 4、π0.5 3）；
- 插件参数照抄，包括 `--os-blind --os-policy-tail --os-policy-tail-blocks 1` 和种子；
- 一个 suite 500 局。

**不得不改的两处**：
- 源参照的 fit 是 R13Knob `pure_c10`，pickle 里有已删除的旧门对象（TriggerCommitJudge / TriggerGrootCommitJudge），还要配 `--os-judge guard_only`。现在这两样都会直接报错，没法原样重跑。
- 所以纯推理改用 1020 的 Q3 TaskDescentKnob，十个任务都设 `c10` 档。cap 为 0，每次决策都由旋钮发起调用。
  - 新 fit：`/home/weiland/projects/openpi_ext/r17_lo_1022/fits/pure/<格>_50_c10.pkl`，取每个 suite 的 50 格，与源参照的库和子集相同。
- judge 换成 `stall_only`：它和原来的 guard_only 一样，把强制调用转成 MISS。每次决策都是 MISS，后面跟一个策略尾，和源参照行为相同。
- 定义已逐项比对：
  - 参数只有 judge 和 fit 路径两处不同，yaml 只差 `miss.evidence_dir` 路径；
  - 行字段的差别只是元数据：r16_k、r16_cell、intended_host、r15_leave_own_episode_out。

**放置**：
- 服务在 H100 原树 `/data/oscl_h100/openpi`；端口与本次各单位都不冲突；R15 同步口用 weilandserver 23194，已登记 LEDGER。
- GR00T 的启动预算按 owner 规定取 7.2 GB；π0.5 K1 沿用源参照的 8800 MB。
- 干跑 GR00T K1 参照的 R15 plan，H100 上会执行：
  `/home/exouser/gr00t_n15_venv/.venv/bin/python /data/oscl_h100/openpi/exp/offline_search/closed_loop/serve_groot.py … --os-judge stall_only --os-blind --os-policy-tail … --checkpoint /home/exouser/ckpt/n15_libero_10 --port 23257 --denoising-steps 1`
  端点是 `149.165.153.233:23257`，yaml 里 `miss.num_steps` 为 1。
- 为了让 GR00T K1 能在 H100 原树上起来，已把 H100 原树的 `src/openpi/cache/types.py` 换成主仓/Q3 树那一份（两者相同），原件留作 `.before_lo_1022`。
  - 原件拒绝 GR00T K=1：`num_steps must be >= 2`。
  - 新版只多放行单步 GR00T 这一个调度，其他调度都不变。
  - 已在 CPU 上核对：`groot_n15_schedule(1)` 正常返回。

## 4. 停滞门与执行段（核对结果）

- 生成定义里的参数，四个批次 32 个臂都一样，以 π0.5 减步 l50 为例：
  ```
  --os-root /home/weiland/trace_runs/offline_search_store --os-no-shadow-native --os-tokens off
  --os-judge stall_only --os-fit-artifact /home/weiland/projects/openpi_ext/r17_q3_port_1020/fits/filtered/pi05_l10_50_off.pkl
  --os-rollout-seed 2026101401 --os-r14-audit
  ```
  三个长段参数一个都没有。
- fit 一律是 `q3_port_1020/fits/filtered/<格>_<档>.pkl`。pickle 里引用的模块只有：
  - harness.api、r02 awm、r06 `stall_only`（Q3StallInner）、r06 Q3 `stall`、r10 data/recipe、r15 TaskDescentKnob；
  - 没有 TriggerCommitJudge、noprog、MixedJudge。
  - 没用 `guard_only`。新定义全部由 q3_port_1020 的定义生成，没碰 formal_a_1016 的 R15/R16 定义。
- 决策只在段边界做：
  - 长段关之后，客户端每 5 步请求一次，每次请求都是一次决策（命中或 MISS），执行 5 步；
  - 插件没有盲尾，也没有策略尾；Q3 不插段中 LOOK（slow_ambiguous 不加看，同 1020）。
- H100 上实际会执行的启动参数，已经干跑核对过（日志 `/nvme/weiland/openpi/r17_lo_1022/logs/h100_argv_dryrun.log`）：
  - 减步常驻是 `<H100 venv python> -m exp.offline_search.rounds.r16.resident --registry /data/mw_finetune/eval/fa_lo_*_r_f_o_q/group.json …`。参数里 judge 为 stall_only，fit 为上面的 Q3 fit，没有长段参数。
  - GR00T 常驻带 `--checkpoint /home/exouser/ckpt/n15_libero_* --denoising-steps 1`，π0.5 常驻带 `--policy.dir /home/exouser/ckpt/pi05_libero_pytorch`。
  - driver 和 agent 连的端点是 `149.165.153.233:23150–23152`（π0.5）、`149.165.153.233:23153–23156=<suite>`（GR00T）。
  - 全步 GR00T 走 R15 plan，只出计划不传文件：服务进程是 `/home/exouser/gr00t_n15_venv/.venv/bin/python /data/oscl_h100/openpi/exp/offline_search/closed_loop/serve_groot.py`，judge 为 stall_only，fit 用重定位后的 H100 副本，无长段参数，端点 `149.165.153.233:23233–23237`，输出在 `/data/oscl_h100/runs/fa_lo_gr_f_f_o_q_<格>/`。

## 5. 改了哪些已有文件

| 文件 | 改动 |
|---|---|
| `exp/offline_search/rounds/r16/conductor_runner.py:20-22,197,226` | 新增两个模块常量：`ENDPOINT_HOST='ziyanglin.com'` 和 `SERVER_NODE='local'`，替换原来写死的两处。默认值不变，4090 上的用法（包括集群模拟的 record_run）行为一样；包装层会把它们设成 H100。 |
| `exp/offline_search/rounds/r16/analyze.py:80-82` | 拔掉 `'Changed policy-tail or seed'` 和 `'Changed judge'` 两条检查：前者要求 `--os-blind`/`--os-policy-tail`/blocks 1，后者要求 `guard_only`。现在只查种子。 |
| `exp/offline_search/rounds/r16/conductor_analysis.py:41-43` | `'Changed client/policy protocol'` 把 blind、policy-tail、guard_only 捆在一起检查，拔掉这三项，只留 replan 5、full_model、recording 关和种子。 |
| `/home/weiland/gpu_share/LEDGER.md`（仓库外） | 端口登记表加两行：weilandserver 的 23157/23158 给 R16 driver 用（23159 留给集群模拟）；23194 给参照的 R15 同步用。 |
| `exp/offline_search/rounds/r17/filter_failed_knob/q3_port_1020/build.py:76-78` | `caps_for` 对 `c10` 档也返回全 0 cap（TaskDescentKnob 规定 off/c10 的 cap 必须是 0），这样 1020 的 knob_blob 和 knob_kwargs 能建纯推理 fit。其他档不受影响。 |
| H100 `/data/oscl_h100/openpi/src/openpi/cache/types.py` | 换成主仓/Q3 树那一份，放行单步 GR00T（见第 3 节）；原件 `.before_lo_1022`。 |
| H100 `/data/oscl_h100/openpi/exp/libero_groot/serve_groot_libero.py` | 换成 Q3 服务树里的同名文件，即 10-21 的「每副本一个模型线程」版。1020 的 GR00T 全步就是用这个版本跑的。原件留在同目录 `.before_lo_1022`。只影响 GR00T 服务端的线程方式（B1 单请求快约 3 倍），不改数值路径；π0.5 不用这个文件。 |

**新代码**：都在 `exp/offline_search/rounds/r17/filter_failed_knob/longseg_off_1022/` 下。
- `build.py`
  - `build`：建旋钮关；
  - `levels`：建各档；
  - `prepare`：R16 同步到 H100；
  - `dispatch prepare|launch`：R15 复用 1020 的 `r15_run`，R16 调 launch 脚本；
  - 另外写 commands 脚本和 QUEUE.json。
- `runner.py`：R16 的包装层。在 q3_port_1020.runner 之上：
  - local_server 的 start/status/stop 改成调 H100；choose_ports、matching、wait_port_release 不再查本机端口；
  - 端点改成 H100 公网 IP，调度器的服务机标成 h100；
  - 每个常驻的线程数取 10 核 ÷ 常驻数（π0.5 3、GR00T 2）；
  - 强制 `--no-local-workers`；
  - DONE 后把常驻日志拉回本机。
- `h100.py`：本机侧。负责复制代码和输入、在 H100 上建软链、把日志拉回来、列出要传的文件（plan），以及常驻的 start/status/stop。
  - 传输方式：在 H100 上起一个临时 rsync daemon（端口 23117，模块名是随机 token，只在一次复制期间存在），本机 rsync 限速 16 MB/s 推过去。
- `h100_side.py`：在 H100 上用 python3 跑，只用标准库。就是 4090 的 local_server，收据、监管进程、端口和显存检查都没改，只把路径换成 H100 的：
  - GR00T 代码和 venv、检查点都在 /home/exouser 下；
  - 树里的 `.venv` 指向 `/home/exouser/openpi/.venv`；
  - 不用 4090 的 CPU 绑核列表；
  - `HOME=/home/exouser`；
  - 启动入口和 conductor_runner.resident_command 相同。
  - status 在 H100 本地最多轮询 30 s 再返回，这样能少调几次 tether。
- 没加任何校验层、哈希钉或门。只拔了上表列出的检查。

## 6. H100 上要有什么，以及哪些已经复制

**已复制（代码，本次已做完）**
- `/data/oscl_h100/openpi_r16_q3_server`：
  - 来源是 `/home/weiland/projects/openpi_r16_q3_server`，不含 .venv、缓存、pkl/npy/npz/pt/jsonl/log。
  - 343 MB，12,233 个文件，逐文件 md5 与本机相同。
  - `.venv` → `/home/exouser/openpi/.venv`。
  - 加上 `exp/offline_search/rounds/r17/filter_failed_knob/longseg_off_1022/h100_side.py`。
- 上面那个 `serve_groot_libero.py`。
- CPU 导入检查（CUDA 隐藏，什么都没起）：
  - Q3 树下，π0.5 venv 和 GR00T venv 都能导入 `rounds.r16.resident`、serve_pi05 和 serve_groot、plugin（含 `stall_only`）、stall_only.py、r15 recipe；
  - H100 原树换文件后，`serve_groot` 也能导入。
- 新 fit 用到的模块（api、awm、stall_only、stall_bridge、stall、r10 data/recipe、r15 recipe、serve_groot/serve_pi05）在主仓、Q3 树、H100 原树三处 md5 相同。
- GR00T、π0.5 检查点在 H100 和 4090 上的文件大小相同。

**prepare 时复制（没做，由协调者跑 prepare 脚本）**
- R16（`prepare_fa_lo_*_r_f_o_q.sh` → `h100.sync`）：
  1. 先重推一次代码。rsync 按大小和修改时间比较，没变就秒过。
  2. 在 H100 上建软链，地址都和本机一样：
     - `/home/weiland/trace_runs/{offline_search_store→/data/oscl_h100/store, dual_20260923→/data/oscl_h100/mirror/home/weiland/trace_runs/dual_20260923, os_closed_loop→/data/oscl_h100/lo_1022/os_closed_loop}`
     - `/data/libero_cache→/data/oscl_h100/mirror/data/libero_cache`
     - `/home/weiland/projects/openpi_ext/{r17_q3_port_1020,r17_lo_1022}→/data/oscl_h100/lo_1022/…`
     - `/data/mw_finetune/eval/<group>→/data/oscl_h100/lo_1022/eval/<group>`
  3. 传输入：
     - fit：每个模型 8 个，约 0.56 GB；
     - 定义：约 100 KB；
     - group.json 和 ROOT.json：约 0.3 MB；
     - store 子集与库 PKL：清单与 R15 H100 同步相同（不传 current/manifest.json，保留 R15 写的那份）。这 62 个文件已核对过，H100 上都已存在且大小、修改时间相同，所以要传的是 0 字节。
  - `h100 plan` 的清单在 `logs/h100_plan_<group>.log`。π0.5 列出 15.7 GiB，GR00T 列出 7.1 GiB，这是 rsync 跳过已有文件之前的总量。
- R15（`prepare_fa_lo_*_f_f_o_q*.sh`）：用现成的 sync，和 1020 的 π0.5 一样。
  - 以 GR00T l50 为例，计划 12 个文件、4.72 GiB；其中库和 store 文件 H100 上已有。
  - 新增的只有重定位后的 fit（每格 50–130 MB）和 yaml/matrix。

## 7. 命令

都在 `exp/offline_search/rounds/r17/filter_failed_knob/commands/` 下，从仓库根目录执行：

```bash
# 全步旋钮关（R15，服务在 H100）。整单位（8 格依次跑）或单格 _l50 … _s500
bash commands/prepare_fa_lo_pi_f_f_o_q.sh && bash commands/launch_fa_lo_pi_f_f_o_q.sh
bash commands/prepare_fa_lo_gr_f_f_o_q.sh && bash commands/launch_fa_lo_gr_f_f_o_q.sh
#   worker 默认 WORKER_HOST=a100 WORKER_HOSTS=a100,timan107,timan1（脚本里是 ${…:-} 默认值，可以在外面覆盖）
#   回退到生产验证过的单机 driver：WORKER_HOST=timan107 WORKER_HOSTS=timan107,timan1（timan1 挂时只留 timan107）

# 减步旋钮关（R16，常驻在 H100，driver 在本机）
bash commands/prepare_fa_lo_pi_r_f_o_q.sh && bash commands/launch_fa_lo_pi_r_f_o_q.sh
bash commands/prepare_fa_lo_gr_r_f_o_q.sh && bash commands/launch_fa_lo_gr_r_f_o_q.sh
#   launch 中断后，重跑同一 launch 脚本就从 journal 续跑；DONE 后自动把 H100 上的 runs/ 和 resident/ 拉回
#   手动拉回：PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.longseg_off_1022.h100 pull --group /data/mw_finetune/eval/<group>/group.json
#   GR00T 的 spatial 先跑完后，照 1020 的 q4 做法，换成 4 个 libero_10 常驻：
#   同一命令改为 --ports 23153,23154,23155,23149 --groot-suite-servers 4,0（23156 退役）

# H100 同卡纯推理参照（R15，服务在 H100）。整单位（两个 suite 依次跑）或单个根 _l10 / _s
bash commands/prepare_fa_lo_pi_f_p5_q.sh && bash commands/launch_fa_lo_pi_f_p5_q.sh
bash commands/prepare_fa_lo_gr_f_p5_q.sh && bash commands/launch_fa_lo_gr_f_p5_q.sh
bash commands/prepare_fa_lo_pi_r_p5_q.sh && bash commands/launch_fa_lo_pi_r_p5_q.sh
bash commands/prepare_fa_lo_gr_r_p5_q.sh && bash commands/launch_fa_lo_gr_r_p5_q.sh
#   C10 的 fa_lo_*_p_q 已作废，不要跑
#   worker 默认同上（a100,timan107,timan1）；照协调者旋钮关的跑法，可在外面 export WORKER_HOST=timan107 WORKER_HOSTS=timan107

# 各档（旋钮关和对应的 H100 参照都完成、R16 日志已拉回之后再跑；只用 CPU；参照没完成会直接报错）
bash commands/build_levels_fa_lo_pi_f_f_l_q.sh   # 然后 prepare_/launch_fa_lo_pi_f_f_l_q.sh
bash commands/build_levels_fa_lo_gr_f_f_l_q.sh
bash commands/build_levels_fa_lo_pi_r_f_l_q.sh
bash commands/build_levels_fa_lo_gr_r_f_l_q.sh
```

上线后按 RUN_CONFIGS 的规矩，立刻用 `r17_ops/knob/lane_speed.sh` 实测速度。

**H100 显存怎么排**：
- 启动时每副本需要的空闲显存：π0.5 R15 9 GB、GR00T R15 7.2 GB、π0.5 常驻 10 GB、GR00T 常驻 7.2 GB。
- C5 参照：π0.5 全步 3 × 9 GB、GR00T 全步 4 × 7.2 GB、π0.5 K1 3 × 8.8 GB、GR00T K1 4 × 7.2 GB。
- 四条一起跑要 27 + 36 + 30 + 28.8 ≈ 122 GB，超过 80 GB；
- 两两并行可以，例如「GR00T 全步 + π0.5 减步」，再「π0.5 全步 + GR00T 减步」。
- 端口互不冲突：
  - H100：23230–23237、23150–23156、参照 23250–23263、临时 23117；
  - R15 driver 在 a100：24230、24233；
  - R16 driver 在本机：23157、23158。

## 8. 风险

1. **R16 常驻在 H100 是新路径，没实测过真起服务。** 以下几项已经核对过：
   - 代码逐字节相同，两个 venv 都能导入；
   - H100 上的启动参数干跑正确；
   - 输入文件都已在 H100，或会在 prepare 时传过去；
   - 后台进程在 tether exec 返回后不被杀，因为用的是 start_new_session，和 R17 在 H100 起服务是同一种做法。

   第一次 launch 时盯第一个常驻起来：GPU_TIGHT、HOME 和 HF 缓存、status 轮询的 tether 调用量。出问题时，常驻日志在 H100 的 `/data/oscl_h100/lo_1022/eval/<group>/runs/resident_*`。
2. **R15 的 a100 driver 多机模式**只跑过冒烟（a100 island 下的 fa_r15mn_smoke1/2），没跑过正式批次。1020 的正式批次 driver 都在 timan107。回退方法见第 7 节。timan1 现在 load 约 386，能响应。
3. **GPU 型号。**
   - 选格用的参照已改成 H100 同卡参照（第 3 节），这一项不再跨卡。
   - 但「长段关对长段开」的比较仍然跨卡：GR00T 全步和两个减步的 ON 批次是在 4090 上跑的。
4. **标定口径。** 关掉长执行段后：
   - Q3 每 5 个控制步观察一次。标定时的锚点是每个 C10 段头（隔一步），冷却「一个锚点」现在也只等于 5 步。
   - TaskDescentKnob 的周期是按 C10 决策解出来的。
   - 按 owner 裁定不重新标定，实测 IR 会偏离名义档位，结果照实报。这和带门四组里的 cg/crg（C5）口径相同。
5. **元数据标签没改，以运行时记录为准：**
   - R16 driver 写的 `launch.json` 里 `server_host` 写死为 'local'（conductor.run_driver），实际端点在 endpoints 字段里，是 H100 IP；
   - R15 GR00T 行的 `intended_host='local'` 由 emit 写入，而且已进 ROOT.json 封存，没改；实际服务机记在 `h100_launch.json`。
6. **磁盘。**
   - H100：/data 已用 94%（剩 62 GB）。每个 group 的常驻日志约 1 GB；R15 每批会在 `/data/oscl_h100/generated` 下生成约 0.5 GB 的重定位 fit。根盘 73%，代码树放在 /data，不占根盘。
   - 本机：R15 GR00T 的 prepare 会把两份 GR00T 库 PKL（在 /data 机械盘上）跨盘拷进各格的临时目录（COPY_FALLBACK），8 格共读约 6 GB。这是现成 sync 的固有行为。现在没有 driver 写 /data，只多一点读 IO；仍建议别和其他大 IO 同时做。
7. **共用的 H100 store 和 mirror**：R16 sync 只在文件有差异时才写，和 R15 sync 用的是同一份内容；`current/manifest.json` 不碰。H100 原树的 `serve_groot_libero.py` 换了之后，以后所有从这棵树起的 GR00T 服务都用单模型线程版。
8. **参照已按 owner 改成 C5。** C10 版（fa_lo_*_p_q）和 fa_pi_f_p10/ps 都不再用于选格。
9. **参照的 fit 不是源 fit。** 旧 fit 带着已删除的旧门，没法加载，所以换成了 c10 TaskDescentKnob 加 stall_only，见第 3 节。纯推理下每次决策都是模型调用，旧门本来就不起作用；但 fit 字节和源参照不同，Q3 诊断字段会多记一些。
10. **参照的 worker 默认值**和协调者实际的跑法（timan107）不同，launch 前按需覆盖。
11. **名字。** 另有一个并行工作在建 MetaWorld 的 `fa_lo_*_m`（`exp/offline_search/rounds/r17/longseg_off_1022/`，以及 commands 里的 `*_fa_lo_gr_r_f_m.sh`），不是本次的产物，我没碰。本次的 LIBERO 批次都以 `_q` 结尾。
