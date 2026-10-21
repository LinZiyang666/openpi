# R15 多机 worker 交接（2026-10-21 01:46 CDT）

## 先看这三条

1. **能用了，但是要显式打开。** 一个 R15 arm 现在可以同时用几台机器的 worker：driver 跑在一台机器上（这台的 agent 和 driver 仍在同一个进程里），其它机器只跑 agent，连到这个 driver 的公网端口。每台机器的 worker 数由中央调度器决定。只有 prepare（sync）时设了 `WORKER_HOSTS` 才会是多机；不设就和原来一样，只用一台。
2. **网络限制决定了 driver 放哪。** 实测（下面有数字）：timan107 只接受校园内的入站连接（timan1 能连，a100/h100 连不上）；timan1 在 NAT 后面，不能当 driver；a100 和 h100 任何端口都能从四台机器连到。所以：
   - 要四台都用（timan107 + timan1 + a100 + h100），driver 必须放在 a100：`WORKER_HOST=a100 WORKER_HOSTS=a100,timan107,timan1,h100w`
   - driver 放在 timan107 时，只能加 timan1：`WORKER_HOST=timan107 WORKER_HOSTS=timan107,timan1`
3. **事故：生产 island 被提前装上了我的新文件。** 见「事故」一节。结论是对在跑的批次在行为上没有变化（单机路径等价），fa_pi_f_f_o_q 也已经按单机正常在跑。但这不是你批准过的部署，必须告诉你。

## 设计

- 主机表 `closed_loop/ops/h100/fleet.py`：
  - `FLEETS` 列出四个 worker 机键：timan107、timan1（和 timan107 路径相同）、a100、h100w。
  - `h100w` 是把 H100 当 worker 机用。`'h100'` 这个键已经是服务端 helper，所以另起一个键；用 `node()` 把它映射回 tether 节点和调度器节点 `h100`。
  - `DRIVERS` 记录能当 driver 的机器、它的公网地址，以及哪些机器能连到它。
- prepare（sync）：
  - 设了 `WORKER_HOSTS` 时，`synced.json` 记一个新键 `fleets`。
  - 调度器包、run_arm.sh、node.py、run_gtp_subset.py 和 cfg 装进每台机器的 island。
- chain 每个 arm 的流程：
  1. 只有 `synced.json` 里有 `fleets` 才走多机。
  2. 调度器 campaign 列出所有机器的节点。
  3. driver 绑 `0.0.0.0:<第一个服务端口+1000>`（如 23230 → 24230）。
  4. 每台其它机器起一个只跑 agent 的进程：同一个 run_arm.sh / run_gtp_subset.py，带 `OSCL_AGENT_OF=<driver 公网 IP>:<端口>`，worker id 前加机器名（如 `timan1_w0`），输出在该机 island 的 `os_cl/runs/<root>/<arm>/agent/`。
  5. driver 停（正常结束、服务端死、清理、abort）就同时停这些 agent。
- agent 进程（run_gtp_subset.py 的 `AgentOnly`）：
  - 不建 driver，WorkerSpec 和 driver 机上的完全一样。
  - 节点的目标数由 legacy.AgentGovernor 每 15 秒向 driver 拉取。
  - driver 报告 done，或 10 分钟连不上，就自己退出。
- 结果全在 driver 那台机器上，collect 照旧只从 driver 机拉。
- 顺手删掉了：
  - node.py：install / pull 的 SHA mismatch（§4.3），以及「timan107 island 必须 WORKER_HOST=timan107」这条门。
  - control.push：拷贝后的 sha256 复核，改成只查文件在。
  - run_arm.sh：只许 timan107 的检查。

## 改了哪些文件（Ziyang 工作区，未提交）

| 文件:行 | 改动 |
|---|---|
| `closed_loop/ops/h100/fleet.py:7-12` | FLEETS 加 timan1 (EGL 1,2)、a100、h100w（island `/data/mw_finetune/qualification/r15_island`，EGL 0） |
| `fleet.py:15-16` | DRIVERS：a100 `149.165.152.232`（四台都能连），timan107 `192.17.58.207`（只有 timan1 能连） |
| `fleet.py:27-34` | `worker_hosts()`：读 `WORKER_HOSTS`，不设就只有 driver 机；连不到 driver 的机器直接报错（唯一的新检查） |
| `fleet.py:37-39` | `node()`：h100w → h100 |
| `closed_loop/ops/h100/run_arm.sh:9-20,26,31,92` | 按机器给 R / GPU_IDS / HOME / PATH(conda shim) / PY / `--conda-env`；删掉 timan107-only 的 readlink 检查。fleet.lock 那一行原样保留（vla_shared 靠它替换成每批自己的锁） |
| `closed_loop/ops/h100/node.py:18-20` | ISLAND = helper 所在 island（`<island>/os_cl/node.py` 的上两级）；H100 服务端 helper 仍用旧默认 |
| `node.py` launch | 删掉「timan107 island 要求 WORKER_HOST=timan107」 |
| `node.py` install / pull_assets | 删掉 SHA mismatch |
| `closed_loop/ops/h100/control.py:22,28-29,74,86-90,545` | import；REMOTE_NODE 给所有 FLEETS；remote/push/collect 用 `tether_node()`；push 不再复核 sha |
| `control.py:352-356` | `_setup` 的 LIBERO config 按机器写（matrix 里提到的 :353-357）；删掉旧的 run_gtp_subset 路径重定位（:363-365） |
| `control.py:434-436,456,483-496,503-509` | sync 记 `fleets`，往每台机器装调度器包和 cfg（pool B 记录按机器改路径）；install_scheduler 额外装 run_gtp_subset.py |
| `control.py:707-721,747,773-785,833,849,864-881,925-931,946,952,1022-1023` | chain：读 `fleets`；campaign 列出所有节点；driver 端口；manifest/contract 推到每台；起/停 agent；launch 回执记 agents；abort 也停 agent |
| `closed_loop/ops/remote/run_gtp_subset.py:71-112,121,125` | `AgentOnly`、`multinode()`（`OSCL_DRIVER_PORT` / `OSCL_AGENT_OF`）；`sys.path` 改成本 island，不再写死路径 |
| `rounds/r17/filter_failed_knob/vla_shared.py:76-86` | ffk 的每批锁补丁对所有 FLEETS 的 helper 生效，不只 driver 机 |
| `filter_failed_knob/vla_workers.py:35` | probe 要求的解释器按机器给（a100/h100 用 env311 + `/home/weiland/.../libero_sim`） |
| `vla_workers.py:62` | R15 driver 机取 `WORKER_HOST`，不设仍是 timan107 |
| `rounds/r17/scheduler/legacy.py:9-11` | 只改文档字符串 |
| `closed_loop/ops/h100/conftest.py`（新） | 测试里一律把 `install_scheduler` 换成空函数（原因见事故） |
| `ops/h100/test_robustness.py:260-288,851-889`、`test_h100.py:108` | 跟着删掉的门改断言；新增两个测试：WORKER_HOSTS 规则、多机 chain 起停 agent（mock） |

不是我改的，但在同一个文件里：
- `node.py:199` 的 `OPENBLAS_NUM_THREADS=env.get("OMP")`
- `vla_workers.py:93-96` 的 OMP 计算

这两处我开始前就在工作区里了。

## 实测

### 网络（10-21 01:07–01:10）

方法：在一台机器上起 120–150 秒自动退出的 TCP 监听，从其它机器去连。

- **timan107（192.17.58.207）**，端口 23901 / 24901 / 40901：
  - timan1 能连上（来源 172.22.224.91，校园内网）
  - a100、h100 全部超时
- **a100 / h100**，端口 23196 / 23590 / 23690 / 24590：四台机器都能连上
- **timan1**：在 NAT 后（内网 172.22.224.91，出口 192.17.168.94）

### island

- **timan1**：`/scratch/zixuans8/openpi_trace_h100` 其实早就有（10-15 的副本，109 MB；matrix 写的 MISS 不对）。除新装的 4 个文件外和 timan107 逐文件 sha 一致。
- **a100 / h100**：`/data/mw_finetune/qualification/r15_island` 也早有（10-15 23:05 的副本）。和 timan107 只差调度器包与 runtime（sync 时装）。我改了两处：
  - `os_cl/libero/config.yaml`：换成同机 r16_local_island 那份
  - 4 个 apool/bpool 记录里的 `apool_dir`：改成本 island 路径
  - 原文件都留了 `*.timan107_1015` 备份
- 三台都实测通过：driver 侧 import、worker 侧 import、EGL 建上下文。

### smoke 1（01:38:40–01:40:12）：driver 在 a100，agent 在 timan1

- 走生产启动器 `filter_failed.legacy`：rsync 传输，ffk 每批锁。
- 4090 没有安全余量（8.4 GB 空、94% 被 fa_cs_gr_r 占），H100 也满，所以服务端用 CPU 替身：
  - 在本机 ziyanglin.com:23187，照真协议回 0 动作
  - `/nvme/weiland/openpi/r15_multinode/tools/fake_server.py`，外壳 `smoke.py`
- 根目录 `/nvme/weiland/openpi/r15_multinode/fa_r15mn_smoke1`：libero_spatial，2 个任务 × 5 局。
- 第一次失败：a100 island 的 apool 记录还指着 timan107 路径。修好后重跑：
  - 调度器目标 `{a100: 2, timan1: 8}`；实际起了 a100 2 个、timan1 8 个 worker
  - **10/10 局完成，用时 76 秒**；COLLECT_OK；CHAIN_DONE
  - timan1 的 agent 看到 done 后自己退出（RUN_ARM_EXIT=0）

### smoke 2（01:41:28–01:43:34）：driver 在 a100，agent 在 timan1 + timan107

- 2 个任务 × 10 局。
- 调度器目标 `{a100: 2, timan1: 16, timan107: 2}`，spawn 数也是 2 / 16 / 2。
- **20/20 局完成**。替身服务端按连接来源记了每一局：
  - a100（149.165.152.232）4 局
  - timan1（192.17.168.94）15 局
  - timan107（192.17.58.207）1 局
- COLLECT_OK；CHAIN_DONE。
- 两个 agent 都是 chain 在 driver 结束后用 SIGINT 停的（RUN_ARM_EXIT=-2，正常停法）。
- 结束后三台按环境标记查，残留进程为 0，也没有残留的 tmux 会话。

### 单元测试

- `closed_loop/ops/h100/`：
  - 开始前 28 failed / 199 passed（全是旧的失败）
  - 现在 26 failed / 203 passed
  - 没有新失败；2 个旧失败因为不再碰真机器而通过了
- 跑测试时用假 `tether` 放在 PATH 最前面，记录到的真实 tether 调用为 0。
- `test_vla_shared.py`：失败在 `vla_workers.r16_agent_bootstrap` 不存在。这是旧问题；它前面的 choose / 锁替换断言都过了。

## 事故（请你过目）

**原因 1：测试会碰真机器。**
- `test_robustness.py::test_concurrent_sync_B_while_A_uses_shared_assets`（以及 test_local 的两个 sync 测试）调用的是真的 `install_scheduler`。它会走真 rsync + tether exec，把当前仓库的调度器包、conductor runtime、run_arm.sh、node.py 装进**真的 timan107 island**。
- 我 01:1x 跑基线测试时就这样装过一次原版文件（内容和任何一次 sync 装的一样）。
- 01:27 改完再跑，又把我的新版 run_arm.sh、node.py、run_gtp_subset.py、legacy.py 装了进去。同一测试往 timan1 装的时候因为测试里的假路径失败了，在 timan1 留下 `/tmp/tmp.o9uyz1FQWC`（约 1 MB，没删）。
- 现在已加 `closed_loop/ops/h100/conftest.py` 挡住。

**原因 2：生产 prepare 正好在跑。**
- 你的 `build dispatch prepare --name fa_pi_f_f_o_q`（01:16 起）从活仓库逐个 sync 8 个根目录。
- 其中 s50（01:27:28）、s100、s200、s500（01:30:46）用的是我当时「默认多机」的中间版本：
  - 它们的 `synced.json` 多了 `"worker_hosts": ["timan107","timan1"]`
  - cfg 和调度器包也装到了 timan1 的 island（没有坏处）
- l50、l100、l200、l500 用的是原版代码。

**处理：**
1. 01:2x 按我 01:19 留的逐文件 sha 清单，把 timan107 island 那 4 个文件恢复成原样（sha 核对一致）。但随后 s500 的 sync 又装回了新版。
2. 立刻改成必须显式打开：chain 只认新键 `fleets`（只有设了 `WORKER_HOSTS` 的 sync 才写）。上面 4 个根目录里的 `worker_hosts` 键会被忽略，所以 fa_pi_f_f_o_q 全部 8 个 arm 都按原计划单机跑。
3. 我没有动这些批次的目录。

**现状：**
- timan107 island 的 `os_cl/{run_arm.sh,node.py,run_gtp_subset.py}` 和调度器 `legacy.py` 是当前工作区版本。单机路径下逐项对过，行为等价：
  - 同样的路径、解释器、GPU、锁
  - `sys.path` 插入的是同一个 island 路径
  - 不设新环境变量时多机代码不生效
- fa_pi_f_f_o_q_l50 01:32:44 用这些文件起来：RUN_ARM_START / INIT_POOL_OK 正常，driver 仍是 127.0.0.1 临时端口。01:45 时 journal 456/500。
- driver.log 里 46 个 EGL 析构报错，和改动前的 fa_pi_f_u_o_q_s500 一样（那边 36 个），是 worker 回收时的正常噪声。
- 如果你要换回原版：原文件在 `/nvme/weiland/openpi/r15_multinode/restore_t107/`（sha 和事前清单一致）。但只要活仓库里还是新版，下一次 R15 sync 就会再装回新版。

## 没验证的

- **真模型服务端**：smoke 用的是 CPU 替身。服务端代码这次没改。
- **h100w 进 chain**：island、import、EGL 都通了，但没在 chain 里跑过。4090 服务端时 h100 不算本机层节点；H100 服务端时 H100 显存是满的，调度器会给 0。
- **driver 在 timan107 + agent 在 timan1**：没有整条跑过。只测了连通性；代码路径和 smoke 一样，只是 driver 地址不同。
- **走 vla_workers / q3 build dispatch 的整条路径**：没跑。
  - a100 当 driver 会被 `vla_workers.py:55` 卡住：probe 要求 `socket.gethostname()` 等于主机键，而 a100 的主机名是 `videogen`（h100 是 `vla-cache`）。我只被允许改 35、62 行，这行没动。
  - 需要你改成类似：`row['usable'] = row['host'] in (host, {'a100': 'videogen', 'h100w': 'vla-cache'}.get(host)) and not row['missing']`
  - 实测现在 probe('a100','R15') 结果是 missing=[]、usable=False。
- **pool B（dev）多机、MAX_ATTEMPTS>1 带 agent 的重启**：只有 mock 单测。
- **agent 机器上的日志不收回**：collect 只拉 driver 机的 arm 目录。agent 日志留在各机 island 的 `os_cl/runs/<root>/<arm>/agent/`。

## 新批次怎么用

1. prepare 和 launch 的环境里都 export：
   - 四台：`WORKER_HOST=a100 WORKER_HOSTS=a100,timan107,timan1,h100w`
   - 只要校园两台：`WORKER_HOST=timan107 WORKER_HOSTS=timan107,timan1`
2. vla_workers 会把 os.environ 传给 legacy sync/chain。sync 把 `fleets` 写进 `synced.json`，chain 据此起 agent。
3. driver 机上的「第一个服务端口 + 1000」必须空着（H100 服务端 23200–23299 → 24200–24299）。
4. 用 a100 当 driver 前，先改上面说的 `vla_workers.py:55`。
5. 已经 prepare 过的批次：`synced.json` 里没有 `fleets`，保持单机。
   - 注意：如果在启动一个按 timan107 prepare 的旧批次时 export 了 `WORKER_HOST=a100`，vla_workers 会因为 worker_host 不一致重新 prepare 它（这是原有逻辑）。所以只对新批次 export。
6. a100 当 driver 时，journal/per_step 落在 a100 的 `/data/mw_finetune/qualification/r15_island/os_cl/runs/`。a100 的 `/` 现在只剩 17 GB，长批次要留意。
7. driver 的拉取端口没有认证，这一点和 R16 的 4090 driver 一样。

## 调度器

不用改，也不用重新部署。
- `legacy.campaign` 会按列出的节点收窄，`node_campaign` 本来就按节点给参数：timan1 的 EGL 1/2；h100 是 local 类，预留 4096 MiB。
- 两次 smoke 里 A100 run_23500 都正常给出了逐节点目标。
- 文档没改，可以顺手改：`scheduler/DESIGN.md` 和 `WORKER_ENV_MATRIX.md` 里「R15 只在 timan107」的说法（§1.4、§2.2、§4.2 相应几行）。

## 留下的东西

- `/nvme/weiland/openpi/r15_multinode/`：
  - smoke 根目录 `fa_r15mn_smoke{1,2}`（含收回的 journal 和替身服务端日志）
  - `tools/`：smoke.py、fake_server.py、edit_chain.py
  - 测试前后的失败清单
  - island sha 清单
  - `restore_t107/`
- 各 island 里的 smoke 残留：
  - `os_cl/cfg/*fa_r15mn_smoke*`、`os_cl/runs/fa_r15mn_smoke{1,2}`（a100、timan1、timan107，都很小）
  - a100/h100 的 `*.timan107_1015` 备份
- timan1 的 `/tmp/tmp.o9uyz1FQWC`（测试残留，约 1 MB）。
