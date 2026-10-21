# 交回：jupyter pod 服务 H100 队列里未开始的 GR00T · MetaWorld 批次

例子：原库 · GR00T · MetaWorld · 全步 · 旋钮下降，`fa_ls_gr_f_u_m`（90,000 局）→ pod 副本 `fa_ls_gr_f_u_m_j`，
在 jupyter-ziyang10-02 上跑（R2、不开 MPS、14 GiB、启动宽度 1，驱动口 23192）。
只改了代码，没有起任何进程，没有 commit。H100 和 pod 上只做了只读检查。

> ⛔ **先看这条（和你 10-20 把 GR00T 改成 R5 有关，不在本次改动范围内，我没动）**：fa_ls_v03_0、fa_ls_v03_1、fa_ls_gr_f_f_m、
> fa_ls_gr_f_u_m 的 campaign.json 现在是 `replicas=5`，但 `backend_ports`/`ports` 还是旧的 8/8/11/11 个后端口。
> `formal_eval.run_campaign` 是按 `backend_ports` 一个端口起一个副本的（proxy 也连全部 backend_ports），所以这几批在 H100 上
> 还是会起 8 或 11 个副本，而且每个副本的显存上限已经变成 `memory_budget_gib/5`（76/5 = 15.2 GiB，56/5 = 11.2 GiB），
> 合起来会超过 80 GB。stage 在没有 alias 时不会重算端口（`ports.execution_host` 原样返回；`worker_topology.redistribute`
> 只在传了 replicas 时才重算）。最小改法：在同一份文件上（原件 `.before_groot_r5_*` 已经有了）再改一次，
> `backend_ports = list(range(public_port+1, public_port+1+5))`，`ports = [public_port, *backend_ports, driver_port]`。
> pod 副本不受影响，因为 `execution_host` 会按副本的 R=2 重算端口。

## 1. 改了哪些文件

| 文件 | 改动 |
|---|---|
| `exp/offline_search/rounds/r17/longseg_1017/pod_copy.py` | 源批次除 4090 DP（`server_node=local, model=dp`）外，还接受 H100 GR00T（`server_node=h100, model=groot`）；未准备过和跑过的两种都支持（ran 分支和 DP 一样：删 PREPARED 字段、改回未修的源运行时）。GR00T 副本把 `cache_worker_orchestration.replicas` 改成副本的 R，只是记录用，没有检查会读它；`pod_copy.resources_before` 记的是源的值（你改过以后是 R5）。`--replicas` 默认 DP 1、GR00T 2；新增 `--startup-width`（默认 1，不超过 R）。`autoscale*`/`scheduler_*` 一律删掉，由 stage 的 `scheduler.defaults.apply` 重新生成节点表（pod 不在 LOCAL 里，所以只有公共池 a100/timan1/timan107；已用源 campaign 实算：a100 80 / timan1 664 / timan107 440）。DP 桥接（`ffk_dp_capacity_bridge` + `pin_bridge`）只对 DP 做，GR00T 保持源的 `false`、不改冻结 bootstrap。同时写 `filter_failed_knob/commands/{prepare,launch}_<name>.sh` 包装（不存在时才写，chain_ffk*.run.sh 用的就是这两个）。输出 JSON 多一个 `pod_ports`（要打隧道的 pod 端口）。 |
| `exp/r17_integration/ops/pod/setup.sh` | 新增 `groot` 目标：环境全放 NFS 家目录 `/home/ziyang10/zl_r17/groot/{uvpy,venv,vendor,hf_home,ckpt/4090_mt50/checkpoint-60000}`，不往 `/tmp` 放；每棵树拉完后写 `groot/.pulled_<模块>`，有这个标记就跳过，所以 rsyncd 只需要开还缺的模块；权重只拉 `model-*.safetensors`、`model.safetensors.index.json`、`config.json`、`experiment_cfg/`；vendor 不拉 `.git/media/demo_data`。groot 不装 Python 3.12，也不拉 fastwam/python_deps312。r17x 加 6 个绑定（目录存在才加，见第 3 节）。探针：在 r17x 里用 GR00T venv 检查 torch/CUDA/MIG，再 import `flash_attn, transformers, gr00t.model.policy`（打印用时），并列出 `/data/mw_finetune/groot/4090_mt50/checkpoint-60000`，确认嵌套绑定生效。 |
| `exp/r17_integration/ops/formal_eval.py` | `run_campaign.wait_port` 默认超时：pod 1800 s，其它 600 s 不变。原因：pod 上的副本是在 proot 下从 NFS 加载的（venv import，加上 7.6 GB 权重先算一遍 sha256 再读一遍），H100 上实测加载 23.6 s，pod 上冷启动可能好几分钟。只放宽超时，没加任何检查。 |

## 2. 和「H100、11 副本、MPS、76 GiB」绑死的东西，逐项

| 位置 | 情况 | 处理 |
|---|---|---|
| stage：`formal_runtime.attach` 拿 model/checkpoint/server_node/**mps** 和源比 | 不一致的分支全是 `pass` | 不用动 |
| stage：`cache_control.attach`（资源字段 replicas/mps/memory_budget_gib 和对照比） | GR00T 批次没有 `cache_control`，直接 return；比较本身也是 `pass` | 不用动 |
| stage：`longseg_1017.ports.execution_host` | pod 别名下把 public/backends 改成 POD_PROFILES 的 23474 + R 个后端口（R=2 时是 23475、23476），驱动口改成 23192，EPISODE_PLAN 的路由一起改 | 不用动，但 **23476 要多开一条隧道**（第 4 节第 3 步） |
| stage：`scheduler.defaults.apply(force=True)` | 新准备的批次会重新生成节点表；pod 没有本机 worker | pod_copy 删掉旧的 `scheduler_*` |
| stage：`runtime_fix.stage`、`stage_server_inputs`、`reusable_inputs` | 经 r17x 在 pod 上跑 `host_python` = `/home/exouser/gr00t_n15_venv/.venv/bin/python`，`reusable_inputs` 用的是 pod 的 `python3` | 先跑 setup groot，绑定生效后就能用 |
| launch：`check_execution_host` | 要带同样的 alias 环境 | longseg 的 prepare/launch 脚本里已经 export 了 |
| launch：MPS 守护、`client_env`、`mps_command account` | 都在 `if campaign.get('mps')` 里 | 副本设 `mps=None`，整段跳过 |
| launch：nvidia-smi 显存核对 | 只对 dp/cosmos/fastwam_batching/mps 做，超预算也只是 `pass` | GR00T 不开 MPS 时不执行 |
| launch：`replica_startup_width` | 读 campaign | 副本里是 1（`--startup-width 2` 可以改成 2） |
| launch：副本就绪等待 600 s | NFS 冷加载可能超时 | 已改成 pod 1800 s |
| 服务端（冻结运行时 `formal_server.serve`） | 唯一和资源有关的是 `set_per_process_memory_fraction(memory_budget_gib/replicas/total)`：H100 是 76/11 = 6.9 GiB，pod 是 14/2 = 7.0 GiB，每个副本的上限基本一样 | 不用动 |
| 服务端的 cache study 准入 | `cache_study_admission` / `dp_capacity_serve` 只有 DP 有；GR00T 没有资源准入 | 不需要桥接，也不需要 repin |
| GPU 家族 | GR00T 服务路径（groot/*、serving、cache_recipe、model_runtime）里没有 device name/capability 检查；IR 计价只用 4090 档案 | 不用动 |
| 模型身份 | 对权重、config、metadata、vendor 代码（43 个文件）和 Eagle tokenizer（10 个文件）取 sha256 | 字节必须一样：都从 H100 原样拉，放在原路径（绑定） |

结论：GR00T 没有会挡住 pod 副本的冻结准入。要改的只有副本资源字段（pod_copy）、pod 环境和绑定（setup.sh）、就绪等待超时这三处。

## 3. H100 上服务端依赖的路径（只读查过：在跑的 fa_ls_gr_r_u_m 副本的 maps/fd/environ，以及 venv 的 .pth）

| H100 路径 | 大小 | 用途 | pod 上的位置 → r17x 绑到 |
|---|---|---|---|
| `/home/exouser/gr00t_n15_venv`（含 `.venv`） | 6.8 GB | server_python，torch 2.5.1+cu124、flash_attn 2.7.4、transformers 4.51.3 | `~/zl_r17/groot/venv` → 同路径 |
| `/home/exouser/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu`（`cpython-3.11-...` 是指向它的软链，venv 的 python 和 pyvenv.cfg 写的是短名） | 98 MB | Python | `~/zl_r17/groot/uvpy` → 两个名字都绑 |
| `/home/exouser/gr00t_n15`（Isaac-GR00T 4af2b62，放在 PYTHONPATH 上） | 121 MB（不含 .git/media/demo_data 约 10 MB） | gr00t 代码、`eagle2_hg_model` tokenizer/config（属于模型身份） | `~/zl_r17/groot/vendor` → 同路径 |
| `/home/exouser/.cache/huggingface`（server_env 的 HF_HOME） | 876 KB | trust_remote_code 写的 `modules/transformers_modules/eagle2_hg_model`；GR00T 不从 hub 下载东西（hub 里只有一个别的实验留下的 CLIP 软链） | `~/zl_r17/groot/hf_home`（可写）→ 同路径 |
| `/data/mw_finetune/groot/4090_mt50/checkpoint-60000` | 16 GB，只拉 7.6 GB | 权重 | `~/zl_r17/groot/ckpt/4090_mt50/checkpoint-60000` → 同路径（嵌套在 `/data` 绑定里面） |

不需要的：venv 里 `_editable_impl_openpi_client.pth` 指向 `/home/exouser/openpi/packages/openpi-client/src`，pod 上没有这个路径，site 会跳过；运行时 PYTHONPATH 里排在前面的 openpi-client 会顶替它。系统库：pod 是 Ubuntu 26.04（glibc 2.43、GLIBCXX ≤3.4.33），比 H100 的 24.04 新；`libfribidi` pod 上没有，但它只被 PIL `_imagingft` 用 dlopen 按需加载，不是 NEEDED 依赖。pod 驱动 570.211（CUDA 12.8），能跑 cu124。

现状（22:2x 复查）：pod 上的 `pull.status` 是 `uv rc=0 / ckpt rc=0 / venv rc=0 / done`；ckpt 两个分片的字节数和 H100 一致，H100 的临时 rsyncd 已关。目录布局和 setup.sh groot 完全一样。**还缺两棵树**：`/home/exouser/gr00t_n15`（vendor）和 `/home/exouser/.cache/huggingface`（HF_HOME）。

## 4. 按顺序执行的命令

**1. H100 临时 rsyncd：只开缺的两个模块**（和之前一样：`use chroot = no`、只读、`list = no`；前缀 K 你自己定，下面沿用 zlg09920ce01758）：

| 模块 | path |
|---|---|
| `${K}_vendor` | `/home/exouser/gr00t_n15`（121 MB；setup 不拉 `.git/media/demo_data`，实际约 10 MB） |
| `${K}_hf` | `/home/exouser/.cache/huggingface`（876 KB） |

```bash
tether exec h100 -- bash -lc 'K=zlg09920ce01758; D=/tmp/zl_groot_pull; printf "%s\n" "use chroot = no" "pid file = $D/pid" "lock file = $D/lock" "log file = $D/log" "[${K}_vendor]" "path = /home/exouser/gr00t_n15" "read only = yes" "list = no" "[${K}_hf]" "path = /home/exouser/.cache/huggingface" "read only = yes" "list = no" > $D/rsyncd.conf; rm -f $D/pid; rsync --daemon --port=23117 --config=$D/rsyncd.conf; sleep 1; cat $D/pid'
```

**2. 给你已经拉好的三棵树补上完成标记（依据：pull.status 三个 rc 都是 0），把新 setup.sh 放到 pod 上，然后跑 groot：**

```bash
tether exec jupyter-ziyang10-02 -- bash -lc 'G=/home/ziyang10/zl_r17/groot; grep -qx "venv rc=0" $G/pull.status && grep -qx "uv rc=0" $G/pull.status && grep -qx "ckpt rc=0" $G/pull.status && touch $G/.pulled_uv $G/.pulled_venv $G/.pulled_ckpt && ls -a $G'
B=$(base64 -w0 /home/weiland/projects/openpi/exp/r17_integration/ops/pod/setup.sh)
tether exec jupyter-ziyang10-02 -- bash -lc "cd /home/ziyang10/zl_r17 && cp setup.sh setup.sh.before_groot_1020 && echo $B | base64 -d > setup.sh.part && mv setup.sh.part setup.sh && md5sum setup.sh"
#   md5 应为 dadaab4eaa2ce3234520580216f5aec4
tether exec jupyter-ziyang10-02 -- bash -lc 'cd /home/ziyang10/zl_r17 && BWLIMIT_KIBPS=40000 setsid nohup bash setup.sh groot rsync://149.165.153.233:23117/zlg09920ce01758 > setup_jupyter-ziyang10-02_groot_1020.log 2>&1 < /dev/null &'
tether exec jupyter-ziyang10-02 -- bash -lc 'tail -8 /home/ziyang10/zl_r17/setup_jupyter-ziyang10-02_groot_1020.log'
```

日志里应该看到 `have uv`、`have venv`、`pulled vendor`、`pulled hf`、`have ckpt`。通过的标准：第一行探针是 `3.11.16 2.5.1+cu124 True NVIDIA H200 NVL MIG 1g.18gb [~15.x, 16.0]`；第二行是 `gr00t imports N s; transformers 4.51.3 flash_attn 2.7.4.post1 ; checkpoint ['config.json', 'experiment_cfg', 'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors', 'model.safetensors.index.json']`。第二行能列出权重，就说明嵌套绑定生效了。拉完后关掉 H100 的 rsyncd：

```bash
tether exec h100 -- bash -lc 'kill $(cat /tmp/zl_groot_pull/pid)'
```

注意 r17x 放在两台 pod 共用的 NFS 上。重新生成以后，-10 下次调用 r17x 也会带上这些 GR00T 绑定（目录存在才会加），对 DP 没有影响。

**3. 隧道：R=2 要用 pod 的 23474（proxy）、23475、23476 三个端口。** 现在只有 23474→24472 和 23475→24473 两条。H100 上 permitlisten 只开了 24470–24475，24474/24475 原来是 -03 的（-03 离线，24474 现在没人监听）。方案 A（不改钥匙）：让 -02 占用 24474。

```bash
tether exec jupyter-ziyang10-02 -- bash -lc 'cd ~/zl_tunnel && setsid nohup bash tunnel.sh 24474 23476 > tunnel24474.log 2>&1 < /dev/null &'
tether exec h100 -- bash -lc 'cd ~/zl_tunnel && setsid nohup python3 fwd.py 23476 24474 > fwd23476.log 2>&1 < /dev/null & sleep 2; ss -tlnp | grep -E ":23476|:24474"'
```

（本机测过 H100 公网 23476：connection refused，说明端口没被防火墙挡，只是没人监听。旧的 `fwd.py 23472 24474` 这时也会转到 -02:23476，没有害处。）方案 B：在 H100 的 `authorized_keys` 里给这把钥匙加 `permitlisten="127.0.0.1:24476"`，改用 24476，以后 -03 回来也不冲突。

**4. 领取文件和原批次 prepare 守卫**（pod 版。H100 链 h100m25/h100m26 用的是 `chain_ffk.run.sh`，prepare 返回 3 时会记 skipped 然后继续往下跑）：

```bash
M=/home/weiland/projects/openpi_ext/scratch/r17_ops/mw
C=/home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands
ls /data/mw_finetune/eval/fa_ls_gr_f_u_m/{PREPARED,LAUNCH,RECEIPT}.json 2>/dev/null   # 应为空（H100 还没开始）
touch $M/claim_pod2_fa_ls_gr_f_u_m
cp $C/prepare_fa_ls_gr_f_u_m.sh $C/prepare_fa_ls_gr_f_u_m.sh.before_pod2_claim
cat > $C/prepare_fa_ls_gr_f_u_m.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
# jupyter-ziyang10-02 serves a pod copy of this not-yet-started H100 batch (fa_ls_gr_f_u_m_j, longseg_1017.pod_copy).
# While the claim file exists the H100 chain skips this batch; remove it (and restore .before_pod2_claim) to hand it back.
CLAIM=/home/weiland/projects/openpi_ext/scratch/r17_ops/mw/claim_pod2_fa_ls_gr_f_u_m
if [ -f "$CLAIM" ]; then echo "claimed by jupyter-ziyang10-02 as fa_ls_gr_f_u_m_j ($CLAIM)"; exit 3; fi
exec bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/longseg_1017/commands/prepare_fa_ls_gr_f_u_m.sh
EOF
```

**5. pod_copy**（只在本机写 `/data/mw_finetune/eval/fa_ls_gr_f_u_m_j`：EPISODE_PLAN 138 MB 按 io_budget 限速复制，约 35 s 以上，放后台跑）：

```bash
cd /home/weiland/projects/openpi && setsid nohup .venv/bin/python -m exp.offline_search.rounds.r17.longseg_1017.pod_copy \
  --source fa_ls_gr_f_u_m --name fa_ls_gr_f_u_m_j --pod jupyter-ziyang10-02 --driver-port 23192 \
  --replicas 2 --memory-gib 14 --startup-width 1 > /tmp/pod_copy_fa_ls_gr_f_u_m_j.log 2>&1 < /dev/null &
# 完成后：tail -1 /tmp/pod_copy_fa_ls_gr_f_u_m_j.log  → pod_ports [23474, 23475, 23476]
```

生成的文件：`longseg_1017/commands/{prepare,launch}_fa_ls_gr_f_u_m_j.sh`（带 `R17_H100_NODE=jupyter-ziyang10-02 R17_H100_HOST=149.165.153.233 R17_DRIVER_PORT=23192`），以及 `filter_failed_knob/commands/{prepare,launch}_fa_ls_gr_f_u_m_j.sh` 两个包装。

**6. prepare/launch 链**（`chain_ffk_stop`，首个失败即停。下面的 WAIT 条件在 chain.log 里已经有了，所以会立刻开始）：

```bash
cd /home/weiland/projects/openpi_ext/scratch/r17_ops/ff && setsid nohup bash chain_ffk_stop.run.sh pod2gr1 '\[ffk-pod2k\] queue done' fa_ls_gr_f_u_m_j > /dev/null 2>&1 < /dev/null &
```

想手动分步跑的话：`bash .../longseg_1017/commands/prepare_fa_ls_gr_f_u_m_j.sh`，然后 `bash .../longseg_1017/commands/launch_fa_ls_gr_f_u_m_j.sh`。上线后：按 RUN_CONFIGS 用 lane_speed.sh 实测速度；需要的话在 A100 的 `run_23500/floors.json` 里给 `fa_ls_gr_f_u_m_j` 加 floor；在 RUN_CONFIGS.md 里补一节 pod GR00T。另一个未开始的 `fa_ls_gr_f_f_m` 照第 4–6 步做，只把名字换掉（每个 pod 一次只跑一批）。

## 5. 已知风险

1. **显存紧**：MIG 一共 16,384 MiB。每个副本上限 7 GiB，再加不开 MPS 时各自的 CUDA context（约 0.3–0.5 GiB），合计约 15 GiB。H100 上 GR00T 副本会把上限用满（实测 7.1–7.26 GB）。如果加载或运行时 OOM：重新 pod_copy，用 `--replicas 1`（或 `--memory-gib 13`）。
2. **速度**：1g 片只有 16 个 SM（H100 是 132），全步 K=4 每次调用会慢好几倍，90,000 局可能要很久。前 30 分钟用逐秒序列量一下，再决定继续还是换 R1/换批次。
3. **NFS 冷启动**：venv import、7.6 GB 权重算 sha256 再读一遍，第一个副本可能要几分钟（超时已放到 1800 s）；第二个副本能用上页缓存（pod cgroup 32 GiB；H100 上每个副本 RSS 峰值 7.9 GB、稳态 4.3 GB）。启动宽度 2 会让两个副本同时从 NFS 读，峰值 RAM 约 16 GB，可以接受，但不建议。
4. **pod /tmp**：现在 `/tmp/zl_r17` 是 14 GB。这批还要加约 1 GB（grmk1_ful 的 cache_inputs 721 MB、server 资产、两份运行时各约 9 MB、serving.jsonl 每 9 万局 0.25–0.55 GB），到约 15.5 GB，离 18 GB 不远。可以考虑删掉 pod 上已经跑完的 DP LIBERO 资产（`eval/fa_ls_v0*`、`openpi_venv_4090`）。要不要删由你定。
5. **隧道端口**：方案 A 占了 -03 原来用的 24474。-03 回来，或者 -10 也要跑 R2（它的后端 23472 和 -03 的 public 23472 冲突）时，就得用方案 B 扩 permitlisten。
6. **跨卡口径**：其它 GR00T MT50 批次都在 H100 上服务；pod 是 H200 MIG（同为 sm90），bf16/flash-attn 的数值可能有细微差别。IR 照常按 4090 计价。是否和 H100 批次合并分析，和 DP pod 的口径一样，由 owner 定。
7. **登记**：`_j` 副本不会自动进 CAMPAIGNS.json 和分析登记（和 DP 的 `_j` 一样），要手工接进去。
8. **源批次的状态**：pod_copy 复制的是完整的 90,000 局计划。只适用于 H100 还没开始，或者已经完整跑过（作为参照）的源。跑了一半的源要用 `filter_failed.resume --remaining`，不能用这个复制。
9. **pod 掉线**：pod 掉线过。中途掉线的话：重启后要重跑 setup groot（只会补 r17x 和 /tmp 部分，NFS 上的环境还在）和隧道，然后对这批做续跑。
10. 本机写入：pod_copy 和 stage 会在 `/data`（sda）上写 eval 目录和 721 MB 的 longseg_inputs tar，有 io_budget 限速（4 MB/s），和以前的 DP 副本一样。如果正赶上 driver 很忙，可以挑空一点的时候再做第 5 步。
