# Worker 环境矩阵（只读盘点）

盘点时刻 2026-10-20 21:57–22:04 CDT（02:57–03:04 UTC），代码以 Ziyang `14de1714` 为准。
盘点期间协调者并行做了两件事，下表按实测时刻记：
- 提交 `d0220081`（A100 进 MetaWorld 共享池、H100 本机层加 MetaWorld、MW 解释器 h100/a100 用 4090 同路径）和 `14de1714`（`LOCAL_FIRST=('local','h100')`），A100 调度器 22:00:29 CDT 已重启。
- a100 / h100 上新装 `/home/weiland/...`（libero_sim、metaworld_sim、uv python、`zl_bin/conda`、LIBERO 资产），时间戳 02:53–02:57 UTC。

## 0. 探针（每格引用这些函数，原样在节点上 `bash -lc` 执行）

```bash
pyv()   { "$1" - "$2" <<'PY'            # 版本：importlib.metadata
import sys, importlib.metadata as m; print(sys.version.split()[0], {k: m.version(k) for k in sys.argv[1].split(',')})
PY
}
egl()   { env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=$2 CUDA_VISIBLE_DEVICES= "$1" -c \
          'import mujoco;c=mujoco.GLContext(64,64);c.make_current();c.free();print("EGL_OK")'; }
r17imp(){ env PYTHONPATH=$2 LIBERO_CONFIG_PATH=$3 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl "$1" -c \
          'import numpy,mujoco,torch,cv2,imageio,yaml,msgpack,websockets,robosuite;from libero.libero import benchmark;from libero.libero.envs import OffScreenRenderEnv;print("R17_LIBERO_IMPORT_OK")'; }
r16imp(){ cd $2 && env PYTHONPATH=$2:$2/src LIBERO_CONFIG_PATH=$2/os_cl/libero MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= "$1" -c \
          'import numpy,mujoco,robosuite,openpi_client;from libero.libero import benchmark;import examples.libero.worker_entry;print("LEGACY_IMPORT_OK")'; }
mwimp() { env CUDA_VISIBLE_DEVICES= "$1" -c 'import metaworld,mujoco,gymnasium,numpy;print("MW_IMPORT_OK")'; }
```

## 1. Worker 类型

### 1.1 R17 LIBERO（π0.5 / GR00T / DP / Fast-WAM / Cosmos 全部共用）

- 入口：`exp.r17_integration.conductor.worker_entry`；有回收时换 `ops.worker_lifetime`，缓存研究换 `ops.cache_study_worker`，local/h100 换 `ops.local_worker_entry`（回环），timan1 换 `ops.timan1_workers` 包装（`formal_workers.py:40-66`）。
- 解释器（`formal_eval.py:508,519-523`；`formal_prestage.py:55-61`）：
  - timan107：`/scratch/zixuans8/openpi/.venv/bin/python`
  - timan1：同上，经 `python-gpu3` 包装进 bwrap 命名空间（`timan1_workers.py:7-10`，`formal_eval.py:553-555`，`timan1_gpu2.py:320-338,401-424`）
  - a100/h100：`/data/mw_finetune/qualification/h100_workers/env311/bin/python`（`h100_workers.py:15-17`）
  - local：`/data/mw_finetune/qualification/local_workers/env311/bin/python`，靠 `.pth` 挂 `/home/weiland/projects/openpi/.venv` 的 site-packages（`local_workers.py:15-17,91-108`）
- 版本（五台实测一致）：Python 3.11.15、numpy 1.26.4、mujoco 3.6.0、torch 2.7.1+cu126、opencv 4.11.0.86、imageio 2.37.3、PyYAML 6.0.3、msgpack 1.1.2、websockets 16.0。
- LIBERO 依赖包 `libero311_<id>`：从 py3.8 libero_sim 的 site-packages 拷出纯 Python 的 libero / robosuite 1.4.0 / bddl / robomimic / easydict / thop / future / past（`worker_installs.py:114-175`）。
  - timan107：`/scratch/zixuans8/mw_r17/formal/dependencies/libero311_c786…`（另有 `7b875…`）；源是 `/scratch/zixuans8/libero_sim/...`，配置 `/scratch/zixuans8/openpi_trace_h100/os_cl/libero/config.yaml`（`formal_eval.py:402-409`）
  - timan1：`libero311_c786…`（`timan1_workers.py:15,51-58`）
  - a100/h100：`ROOT/dependencies/libero311_42f0…`（`h100_workers.py:19`）
  - local：`local_workers/dependencies/libero311_42f0…`（`local_workers.py:18,111-115`）
- 资产：`libero_3eb6…`，405 MB，清单在 `exp/r17_integration/qualification/specs/libero_original_assets.json`。
  - timan：`/scratch/zixuans8/mw_r17/formal/assets/libero_3eb6…/assets`
  - a100/h100：`ROOT/assets/libero_3eb6…/assets`（`h100_workers.py:20`；setup 用 `ln -sfn` 把 assets 链到本机，`h100_workers.py:39-41`）
- 配置：每批一份 `<root>/campaigns/<run>/<node>/libero_config/config.yaml`（`formal_eval.py:536-548`，`worker_installs.py:196-211`），由 `LIBERO_CONFIG_PATH` 指过去。另需 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`（`formal_eval.py:550`）。HOME 各机不同：timan 是 `/home/zixuans8`，a100/h100 是 `ROOT/home`，local 是 `/home/weiland`（`formal_eval.py:525-534`）。
- EGL：`MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=<卡>`，LIBERO 另设 `CUDA_VISIBLE_DEVICES=''`（`formal_workers.py:129-134`）。用哪张卡：timan107 0–7；timan1 用 GPU2、GPU1（`timan1_gpu2.py:28-30`）；a100/h100/local 用 0（`h100_workers.py:72`，`local_workers.py:141`）。
- worker 包 `workers.tar`：全是纯 Python。实测 Cosmos `fa_co_f_f_l` 的包 706 KB，内容是 `CLIENT_ONLY_FILES`（`worker_entry.py:10-25`）加 conductor、openpi_client、ops 覆盖层和 scheduler 包。
  - 解到 `<node_root>/payload_<sha>`（`formal_eval.py:497-499`）：timan 在 `/scratch/zixuans8/mw_r17/formal/`，a100/h100 在 `/data/mw_finetune/qualification/h100_workers/`，local 在 `/data/mw_finetune/qualification/local_workers/`
  - stage 目录 `/tmp/r17_formal/<run>/<node>`，timan1 用 `ROOT/stage`（`formal_eval.py:425-427`）
- **模型专用包：无。**
  - Cosmos 只多一条协议分支 `libero_cosmos_official_*`（`libero_bridge.py:190,284-288`，只用 numpy）。
  - GR00T 只多 `exp/libero_groot/emit_task_map.py`（纯 Python）。
  - driver（旧版 `formal_driver` / `cache_study.run_driver` / `formal_driver_fast`）跑在协调机的 `CONTROL_PYTHON`（`formal_eval.py:27,1527`），不在 worker 机上。

### 1.2 R17 MetaWorld

- 解释器：
  - timan107/timan1：`/scratch/zixuans8/mw_r17/envs/sim/bin/python`（`worker_installs.py:214`）。timan1 用的 `python-mw-gpu3` 每次 prepare 都会从 `python-gpu3` 重写一遍（`worker_installs.py:227-232`；`timan1_gpu2.py:406-409`）。
  - local/a100/h100：`/home/weiland/projects/openpi_ext/envs/metaworld_sim/bin/python`（`worker_installs.py:215`；`formal_eval.py:510-518`，`d0220081`）。这是 uv venv，python 软链到 `/home/weiland/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu`，所以 uv python 也要在同一路径。
- 版本：metaworld 3.0.0、mujoco 3.14.0、gymnasium 1.3.0、numpy 1.26.4（Python 3.11.15）。prepare 时强制版本相同（`worker_installs.py:222-226`）。
- 无资产、无 LIBERO 配置，只用 `benchmark.json`。
- 回收：32 局或 1800 s（`defaults.py:78-79`，`local_workers.py:147-148`）；timan1 不回收（`timan1_gpu2.py:83-87`）。
- EGL 设置同 R17 LIBERO，不清 `CUDA_VISIBLE_DEVICES`。

### 1.3 R16 LIBERO（π0.5/GR00T 缓存部件研究 fa_cs_*；服务端固定在 4090）

- 流程：driver 跑在 4090（`conductor_runner.py:158-270` → `conductor.run_driver`）。每台节点起一个 `conductor agent`（`conductor.py:501-543`），由 WorkerAgent 用 `conda run --no-capture-output -p <CONDA> python -m examples.libero.worker_entry` 起 worker（`src/openpi/conductor/agent.py:141-162`）。所以**节点 PATH 上必须有 `conda`**。
- agent 用的解释器（`conductor_runner.py:22-30` AGENTS）：
  - timan107/timan1：`/scratch/zixuans8/openpi/.venv/bin/python`
  - a100：`h100_workers/env311/bin/python`
  - local：`/home/weiland/projects/openpi/.venv/bin/python`
- 仿真环境（`conductor.py:449-451` CONDA）：
  - timan107/timan1：`/scratch/zixuans8/libero_sim`
  - local/a100：`/home/weiland/projects/openpi_ext/envs/libero_sim`
- 版本：Python 3.8.20、mujoco 3.2.3、robosuite 1.4.0、numpy 1.22.4、libero 0.1.1、bddl 1.0.1、robomimic 0.2.0、torch 1.11.0+cu113、websockets 13.1、msgpack 1.1.1、openpi-client 0.1.0（editable，指向 `/home/weiland/projects/openpi/packages/openpi-client/src`；实际运行时岛内 `src/openpi_client` 优先）。
- conda：timan 用 `/scratch/zixuans8/dsp_bin/conda`（shim），a100 用 `/home/weiland/zl_bin/conda`（shim），local 用 `/home/weiland/miniconda3/bin/conda`（真 conda）。
- island（`conductor_runner.py:18,180-182`）：
  - timan107/timan1：`/scratch/zixuans8/openpi_r16_conductor_v1`（38–49 MB）
  - a100/local：`/data/mw_finetune/qualification/r16_local_island`（37 MB）
  - 每次 run 由 `legacy.install` 把 scheduler 包、conductor 钩子、`conductor.py` 和 loopback 用 rsync 进岛（`legacy.py:55-76`）
- 配置：`LIBERO_CONFIG_PATH=<island>/os_cl/libero`（`conductor_runner.py:230`）。timan 上指 `/scratch/zixuans8/libero_sim` 和 `/home/zixuans8/.cache/libero/assets`；4090/a100/h100 上指 `/home/weiland/projects/openpi_ext/envs/libero_sim` 和 `/home/weiland/.cache/libero/assets`。初始状态池在 `<island>/exp/common/data/db_init/libero/*[_apool]`（`conductor.py:519-520`）。
- EGL：`conductor.py:454` EGL_DEVICES（timan107 0–7；timan1 (1,2)，即 GPU2/GPU1；local/a100 0），另设 `MUJOCO_GL=egl`；local 额外加 loopback 的 sitecustomize（`conductor.py:510-512`）。
- 调度：`legacy.campaign(root,'local',islands)`（`conductor_runner.py:185`，`legacy.py:38-52`），server_host 恒为 local。

### 1.4 R15 LIBERO（π0.5/GR00T 全步 formal-A；服务端在 H100 `/data/oscl_h100`）

- 链条：4090 上 `vla_workers.r15` → `filter_failed/legacy.py` → `closed_loop/ops/h100/control.py chain`。worker 机取自 `WORKER_HOST`，只允许 timan107（`fleet.py:6-19`）。
- worker 机上：tmux 起 `node.py supervise` → `run_arm.sh` → `run_gtp_subset.py`。**driver（绑 127.0.0.1）和 WorkerAgent 在同一个进程**（`run_gtp.py:414,610-665`）。worker 用 `conda run -p /scratch/zixuans8/libero_sim … examples.libero.worker_entry` 起（`run_arm.sh:82-100`）。
- island `/scratch/zixuans8/openpi_trace_h100`（`run_arm.sh:9-18`，`node.py:18-22`，`fleet.py:7`）：代码和池约 110 MB（`exp/` 102 MB，含 `db_init/libero/{libero_10,libero_spatial}[_apool]`），`os_cl/runs` 是历史产物，约 1.0 GB。由 `control._setup` / `sync`（`control.py:309-366,478-501`）和 `install_scheduler`（`control.py:505-509`）建。
- 写死在 timan107 上的值：
  - driver/agent 解释器 `/scratch/zixuans8/openpi/.venv`（`run_arm.sh:30`）
  - PATH 用 dsp_bin（`run_arm.sh:29`），`HOME=/home/zixuans8`（`run_arm.sh:23`）
  - `LIBERO_CONFIG_PATH=$R/os_cl/libero`（`run_arm.sh:24`），配置内容写在 `control.py:353-357`
  - `GPU_IDS` 0–7（`run_arm.sh:16`）
  - 另需 tmux、`/usr/bin/python3`、flock
- 仿真环境与 R16 是同一个 libero_sim。
- 调度：`legacy.node_campaign(legacy.campaign(root, server, (host,)), host)`（`control.py:769`），每个 arm 只用一台 worker 机。

### 1.5 其它（不在中央调度器里，不计入矩阵）
`exp.metaworld.worker_entry`、`exp.robocasa365.worker_entry`、`exp.step_diag/gate_research.worker_entry` 现在没有批次在用。

## 2. 节点 × worker 类型

### 2.1 机器
| 节点（hostname） | nproc | 内存（total/avail GiB） | GPU（驱动） | 盘余量 | 工具 |
|---|---|---|---|---|---|
| timan107 | 48 | 220 / 145 | 8× GTX 1080 8 GB（535.183.01），已用 1.7–5.3 GB | `/scratch` 12 G（95%） | tmux bwrap rsync flock，无 conda（用 shim） |
| timan1 | 48 | 503 / 339 | 4× RTX A6000 48 GB（535.183.01）；我们的 GPU2 已用 22 GB、GPU1 已用 43 GB | `/scratch` 557 G | 同上 |
| a100（videogen） | 32 | 115 / 33 | A100-SXM4-40GB（595.99.02），已用 33.7 GB | `/` 18 G（/data 在 / 上） | 同上 |
| h100（vla-cache） | 20 | 230 / 182 | H100 80GB（595.99.02），已用 79.2 / 81.6 GB，36% | `/data` 83 G，`/` 13 G | 同上 |
| local（weilandserver） | 88 | 251 / 143 | RTX 4090 49140 MiB（595.91.07），已用 40.2 GB，94% | `/data` 778 G（机械盘），`/home` 1.7 T，`/nvme` 2.6 T | 有真 conda |

命令：`nproc; free -g; nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader; df -h`。EGL 结果见下表，五台都通。

### 2.2 矩阵

| | R17 LIBERO | R17 MetaWorld | R16 LIBERO | R15 LIBERO |
|---|---|---|---|---|
| **timan107** | **实测有** | **实测有** | **实测有** | **实测有** |
| **timan1** | **实测有** | **实测有** | **实测有** | **部分有**：缺 island；环境齐 |
| **a100** | **实测有** | **实测有**（21:55 CDT 新装） | **实测有** | **部分有**：缺 island 和代码；环境齐 |
| **h100** | **实测有** | **实测有**（21:57 CDT 新装） | **环境有，代码不放行** | **部分有**：缺 island 和代码；环境齐 |
| **local** | **实测有** | **实测有** | **实测有**（64 个 R16 worker 在跑） | **部分有**：缺 island 和代码；按类别规则也不可用 |

每格的命令和结果：

- **timan107 / R17 LIBERO**：
  - `pyv /scratch/zixuans8/openpi/.venv/bin/python …` → 3.11.15，numpy 1.26.4，mujoco 3.6.0，torch 2.7.1
  - `ls …/formal/dependencies` → `libero311_7b875…`、`libero311_c786…`；`ls …/formal/assets` → `libero_3eb6…`
  - `r17imp <py> …/libero311_c786… …/campaigns/fa_ls_v05_3/timan107/libero_config` → `R17_LIBERO_IMPORT_OK`（robosuite 1.4.0）
  - `egl <py> 0`、`egl <py> 7` → `EGL_OK`
- **timan107 / MW**：`pyv /scratch/zixuans8/mw_r17/envs/sim/bin/python metaworld,mujoco,gymnasium,numpy` → 3.0.0 / 3.14.0 / 1.3.0 / 1.26.4；`mwimp` → OK；`egl … 0` → OK。
- **timan107 / R16**：
  - `pyv /scratch/zixuans8/libero_sim/bin/python …` → 3.8.20 / 3.2.3 / 1.4.0 / 1.22.4
  - `egl … 0` → OK
  - `ls -d /scratch/zixuans8/openpi_r16_conductor_*` → 只有 `_v1`；岛内 `conductor.py`、`os_cl/libero/config.yaml`、`examples/libero/worker_entry.py`、`scheduler/__init__.py`、`db_init` 都在
  - `r16imp <libero_sim> <v1>` → `LEGACY_IMPORT_OK`
  - `PATH=/scratch/zixuans8/dsp_bin:… command -v conda` → shim（1034 B）
- **timan107 / R15**：
  - island 下 `os_cl/{node.py,run_arm.sh,run_gtp_subset.py,libero/config.yaml}`、`src/openpi/conductor/agent.py`、`packages/openpi-client/src` 都在
  - db_init 有 `libero_10`、`libero_10_apool`、`libero_spatial`、`libero_spatial_apool`
  - `r16imp <libero_sim> /scratch/zixuans8/openpi_trace_h100` → OK
  - `scheduler/__init__.py` 不在岛内，链启动时 `install_scheduler` 会装，这是正常的
- **timan1 / R17 LIBERO**：
  - 同一 python 和版本；`ls timan1_provision` 有 `python-gpu3`、`python-mw-gpu3`（21:46 改过）、`gpu3_guard`、`libero_config`；bwrap 在
  - `r17imp <py> …/libero311_c786… timan1_provision/libero_config` → OK
  - 命名空间外 `egl <py> 1`、`egl <py> 2` → OK
- **timan1 / MW**：版本同 timan107，`mwimp` OK，`egl … 1` OK。
- **timan1 / R16**：版本同上，`egl … 1` OK；`_v1` island 在，`r16imp` OK；dsp_bin/conda shim 在（21:05 改过）。
- **timan1 / R15**：`/scratch/zixuans8/openpi_trace_h100/os_cl/*` 和 `/scratch/zixuans8/openpi_trace` **全部 MISS**；libero_sim、.venv、shim、tmux 都在。
- **a100 / R17 LIBERO**：
  - `pyv h100_workers/env311/bin/python` → 版本与上面一致
  - `ls ROOT` 有 `.unpacked`，env.tar 已删，`extra.tar` 237 MB 还在；`original_paths.yaml` 指 ROOT 内的 deps 和 assets
  - `r17imp <env311> ROOT/dependencies/libero311_42f0… ROOT/campaigns/fa_ls_v05_3/a100/libero_config` → OK
  - `egl … 0` → OK
- **a100 / MW**：`/scratch/.../envs/sim` MISS。`/home/weiland/projects/openpi_ext/envs/metaworld_sim/bin/python` 是 HAVE（bin 02:55 UTC 建，软链到 uv cpython-3.11.15）；`pyv` → 3.0.0 / 3.14.0 / 1.3.0 / 1.26.4；`mwimp` OK；`egl … 0` OK；大小 510 MB。
- **a100 / R16**：
  - `pyv /home/weiland/projects/openpi_ext/envs/libero_sim/bin/python` → 3.8.20 / 3.2.3 / 1.4.0 / 1.22.4（5.1 GB）
  - `egl … 0` → OK；`r16imp <libero_sim> /data/mw_finetune/qualification/r16_local_island` → OK
  - `cat /home/weiland/zl_bin/conda` → `run -p` shim，跳过 activate.d
  - 调度器 `/v1/state` 里 `fa_cs_pi_r` 在 a100 上 active 31
- **a100 / R15**：`/scratch/zixuans8/{openpi_trace_h100,libero_sim,openpi/.venv}` 都 MISS。libero_sim、shim、env311、tmux 在。
- **h100 / R17 LIBERO**：
  - `pyv env311` 版本一致；`env.tar` 7.18 GB 还在
  - `r17imp … ROOT/campaigns/fa_co_f_f_l/h100/libero_config` → OK
  - `egl … 0` → OK
- **h100 / MW**：`metaworld_sim` 的 python 软链 02:57 UTC 建，`/home/weiland/.local/share/uv` 99 MB；`pyv` 版本一致；`mwimp` OK；`egl … 0` OK。
- **h100 / R16**：
  - `libero_sim` 是软链，指向 `/data/mw_finetune/qualification/zl_envs/libero_sim`（5.1 GB，02:53 UTC）；`pyv` 版本一致；`egl … 0` OK
  - `r16_local_island` 10-19 就在了，`exp/`、`src/` 02:32 UTC 刷新过；`r16imp` → OK
  - `zl_bin/conda` 02:57 UTC；`openpi-client` 的 editable 路径 `/home/weiland/projects/openpi/packages/openpi-client/src` 在
  - **代码**：AGENTS / islands / CONDA / EGL_DEVICES 里都没有 h100，且类别规则不放行（见 §4、§5）
- **h100 / R15**：worker 侧的 island 缺；`/scratch/zixuans8/*` 都 MISS。服务端侧 `/data/oscl_h100/node.py`、`/home/exouser/openpi/.venv`、`/home/exouser/gr00t_n15_venv/.venv` 都在。
- **local / R17 LIBERO**：
  - `env311/pyvenv.cfg` 的 base 是 .venv，`.pth` 指 `/home/weiland/projects/openpi/.venv/lib/python3.11/site-packages`
  - `r17imp … local_workers/dependencies/libero311_42f0… …/campaigns/fa_ls_dp_f_u_l_l/local/libero_config` → OK
  - `egl … 0` → OK
- **local / MW**：`pyv metaworld_sim` 版本一致；`egl … 0` OK。
- **local / R16**：`pyv libero_sim` 版本一致；`egl … 0` OK；`ps` 看到 64 个 `conda run -p …/libero_sim python -m examples.libero.worker_entry`。
- **local / R15**：island 缺；libero_sim 和真 conda 在。

## 3. 补缺口的最小步骤

R17 LIBERO、R17 MW、R16 三类环境五台都已齐。只剩 R15 的 island 要装，外加 §4 的代码。

| 缺口 | 从哪来 | 到哪去 | 怎么搬 | 大小 |
|---|---|---|---|---|
| R15 island → timan1 | timan107 `/scratch/zixuans8/openpi_trace_h100/`，排除 `os_cl/runs`、`os_cl/.rpc`、`os_cl/bundle_*.tar`、`__pycache__` | timan1 同一路径 | 普通目录，公网直连 rsync（现有 direct_transfer daemon，限速）。路径不用改：`os_cl/libero/config.yaml` 指的 `/scratch/zixuans8/libero_sim` timan1 上有，`/home/zixuans8` 是 NFS 共享，资产也在 | ~110 MB |
| R15 island → a100 / h100 | 同上 | 建议放 `/data/mw_finetune/qualification/r15_island`；local 按规矩放 `/nvme/weiland/openpi/r15_island`，原路径留软链 | 同上。要改的路径只有 `os_cl/libero/config.yaml`，直接用同机 `r16_local_island/os_cl/libero/config.yaml` 覆盖（5 个键都指 `/home/weiland/...`）。apool yaml 和 `run_gtp_subset.py` 的 island 路径由 R15 的 `setup` 重新生成（`control.py:338-345,363-365`，需先做 §4 的改动） | ~110 MB |
| R15 driver 解释器（a100/h100） | — | 用 `h100_workers/env311/bin/python` | `run_gtp_subset` 只依赖 yaml 和岛内 `src/openpi`（`run_gtp.py:31-48`），env311 预计够用，**未实测** | 0 |

以后要重装时的参考：
- a100/h100 R17 LIBERO 环境：h100 上的 `ROOT/env.tar`（7.18 GB）加 `extra.tar`（cusparselt，237 MB）就是源，解包后 `h100_workers.setup_script()` 会重建 venv、`.pth` 和资产软链（`h100_workers.py:27-54`）。
- libero_sim：conda 环境，prefix 写死，必须放在 `/home/weiland/projects/openpi_ext/envs/libero_sim`（可以软链，h100 就是这么做的）。
- metaworld_sim：要连 `/home/weiland/.local/share/uv/python/cpython-3.11.15…` 一起搬。
- 盘余量：timan107 `/scratch` 只剩 12 G，a100 `/` 剩 18 G，h100 `/` 剩 13 G。

## 4. 代码表

### 4.1 已改（HEAD）
| 位置 | 改动 |
|---|---|
| `scheduler/defaults.py:20` | `SHARED['a100']` 改为 `('libero','metaworld')`（d0220081） |
| `scheduler/defaults.py:23` | `LOCAL['h100']` 改为 `('libero','metaworld')`（d0220081） |
| `ops/formal_eval.py:510-518` | MW 解释器 h100/a100 改用 `LOCAL_METAWORLD_PYTHON`，并传 `local=True`（d0220081） |
| `scheduler/allocate.py:12,106-111` | `LOCAL_FIRST=('local','h100')`（14de1714），A100 已部署 |

### 4.2 还写死的地方
| 位置 | 写死了什么 | 最小改法 |
|---|---|---|
| `scheduler/DESIGN.md:9,172,175`；`legacy.py:9-10` | 文档仍写 a100/h100 只跑 LIBERO、R15/R16 只在 timan107 和 4090 | 改文字 |
| `rounds/r16/conductor_runner.py:22-30` | AGENTS 没有 h100 | 加 `'h100': dict(python='/data/mw_finetune/qualification/h100_workers/env311/bin/python', home='/home/weiland', path='/home/weiland/zl_bin:/usr/local/bin:/usr/bin:/bin', driver_host='ziyanglin.com')` |
| `conductor_runner.py:182` | islands 没有 h100 | 加 `h100=LOCAL_ISLAND` |
| `rounds/r16/conductor.py:449-451,454`（`:564` 的 choices 由 CONDA 派生） | CONDA、EGL_DEVICES 没有 h100 | 加 `'h100': '/home/weiland/projects/openpi_ext/envs/libero_sim'` 和 `'h100': (0,)` |
| `conductor_runner.py:185` + `defaults.py:49-55` + `legacy.py:43` | R16 服务端恒为 `'local'`。本机层只给同机服务，所以 h100 被 `nodes()` 排除，又被 `legacy.campaign` 的收窄去掉 | 需要 owner 定：要么 h100 也进 SHARED（`defaults.py:20` 加一项），要么 R16 不用 h100。4090 对服务在 H100 的 R15/R17 批次同理 |
| `closed_loop/ops/h100/fleet.py:6-8,11-15` | FLEETS 只有 timan107，其它一律 raise | 加 `timan1: (Path('/scratch/zixuans8/openpi_trace_h100'), (1,2))`、`a100: (Path('/data/mw_finetune/qualification/r15_island'), (0,))`；raise 只保留 `host in FLEETS` |
| `closed_loop/ops/h100/run_arm.sh:9-18,23,29,30,91` | island 路径（R）、GPU_IDS、HOME、PATH、PY、`--conda-env` 只有 timan107 一套 | case 里按主机加分支。a100/h100：`HOME=/home/weiland`、`PATH=/home/weiland/zl_bin:…`、`PY=…/h100_workers/env311/bin/python`、`--conda-env /home/weiland/projects/openpi_ext/envs/libero_sim`、`GPU_IDS=0`、`GPU_COUNT=1` |
| `control.py:28-29` | REMOTE_NODE 只有 h100（服务端 helper）和 timan107 | 新主机加 `island/os_cl/node.py`。**h100 撞名**：`'h100'` 已是服务端 helper（`/data/oscl_h100/node.py`），而且 `control.py:104,120,144` 把 h100 当服务端特判。h100 当 worker 机要用另一个键（如 `h100w`，在 `remote()` 映射到 tether 节点 h100） |
| `closed_loop/ops/h100/node.py:18-22` | ISLAND 只自动识别 T107_ISLAND，其它情况默认 `/scratch/zixuans8/openpi_trace` | 改为 `Path(__file__).parent.parent`，或者在远程调用里带 `OSCL_ISLAND` |
| `control.py:353-357` | island 内 libero 配置写死 `/scratch/zixuans8/libero_sim` 和 `/home/zixuans8/.cache` | 按主机给配置（a100/h100/local 用 r16_local_island 那份） |
| `control.py:363-365` | `run_gtp_subset` 的 island 路径只在 `host=="timan107"` 时重定位 | 改成「island ≠ ISLAND 就重定位」 |
| `control.py:769` + `run_gtp.py:414,610-665` | 每个 arm 只能用一台 worker 机：driver 绑 127.0.0.1，和 agent 同进程 | 改成多机：run_arm 传 `--bind-host 0.0.0.0`，其它节点起只跑 agent 的进程（照 R16 `conductor agent`）。不是一行能改完的 |
| `filter_failed_knob/vla_workers.py:35,62` | probe 只认 `/scratch` 路径；R15 主机只许 `('timan107',)`，R16 只许 `('timan107','timan1')` | 改成按 FLEETS 和 AGENTS 派生 |
| `ops/local_workers.py:74-75` | live fleet 路径 raise「supports LIBERO only」 | 删掉这两行 |
| `ops/timan1_workers.py:33-35` | timan1 跑 LIBERO 只许缓存研究或 live fleet | 运行时已被 `timan1_gpu2.coordinator` patch 掉（`timan1_gpu2.py:451-453`），删掉 |

### 4.3 现有的哈希 / sha 钉 / 源码锁 / 准入门：应拔
| 位置 | 内容 |
|---|---|
| `ops/timan1_workers.py:12-14,48-50` | PINS 里三个文件的 sha，不符就 raise「T1 namespace/config guard changed」：**应拔** |
| `ops/timan1_workers.py:52-54` | 资产清单 digest 不等于 ASSET_ID 就 raise：**应拔** |
| `filter_failed/timan1_gpu2.py:258-260` | guard 源码的 sha 钉，每次 timan1 fleet 启动都查：**应拔** |
| `filter_failed/timan1_gpu2.py:288-289` | guard 源码逐段必须恰好匹配一次（源码锁）：**应拔**（改成直接写新 guard） |
| `ops/formal_eval.py:395-399` | 沿用旧 release 时 `assert sha(config)==config_sha256`：**应拔** |
| `ops/formal_prestage.py:42-50`（`:63-67` 在用） | `verify_code` 断言 payload 清单 digest：**应拔** |
| `ops/worker_installs.py:43-44` | 「different immutable client release already exists」：**应拔** |
| `ops/worker_installs.py:222-226` | MetaWorld 版本必须相等（版本准入门）：**应拔** |
| `closed_loop/ops/h100/node.py:400-402,424-425` | R15 安装和拉资产时的 SHA mismatch：**应拔** |
| `filter_failed_knob/vla_workers.py:39,48` | R16 island 必须有 `worker.sha256` 才算可用：**应拔** |
| `closed_loop/ops/h100/control.py:739-740` | arms.json 自 sync 后的 sha 锁：**应拔** |

不列为应拔：
- `scheduler/defaults.py:88-103 require()`：owner 10-19 在 `DESIGN.md:185-189` 指定的唯一启动检查。
- `control.py:554,559`：结果拉回时的归档校验，是「拉回核哈希后删」的数据规矩。

其余同类检查已改成 `pass`。

## 5. H100 本机 worker 接调度器

- **`LOCAL={'h100':('libero',)}` 时的实际效果**（10-19 15:50 → 10-20 22:00）：
  - 只有服务在真 H100 上的 LIBERO 批次会得到 h100 本机节点。MetaWorld 批次和 h100-temp/jupyter 的服务都没有。
  - 叠加 `LOCAL_FIRST=('local',)` 和 `STABLE_S=600`（`allocate.py:11`），A100 的 `allocations.jsonl` 6,441 行里，只有 `fa_co_f_f_l` 带 h100 键（10-20 00:44–16:26），其中目标大于 0 的只有 3 行：08:51:18–08:51:28，目标 7–9。
  - H100 上 `runs/fa_co_f_f_l_h100/AUTOSCALE_DECISIONS.jsonl`：08:51:31 起了 7 个；`AUTOSCALE_SAMPLES` 显示 08:52:09 目标回到 0，存活峰值 7，约 12 分钟后退役，日志只到任务加载。
  - **结论：中央调度器上线后，H100 本机 worker 实际等于 0。** 之前在 autoscale_1017 下（10-19）是真跑过的：`fa_co_r_f_lc7_h100` 有 470 份 worker 日志，`lc8` 511 份，`fa_co_f_u_l_h100` 40 份。
- **现状（22:03 `/v1/state`）**：没有任何批次带 h100 节点。`fa_ls_gr_r_u_m`（GR00T MT50，服务在 h100）的节点只有 `{timan107: 440}`，因为 `defaults.apply` 不加 `force` 时保留已写的 `scheduler_nodes`（`defaults.py:64`），新规则只对重新 prepare 的批次生效。
- **MetaWorld**：环境、代码、调度三项都已就位（d0220081 + 14de1714 + 22:00 重启）。还缺两样：
  - 在跑或已 prepare 的批次要重新 prepare（或 `apply(force=True)` 再 `prepare_workers`）才会在 h100 起 fleet。
  - 显存：GR00T 现在占了 79.2 / 81.6 GB。槽位要 `总量 − 已用 − 预留 min(10%, 2 GiB) − 同机服务 4096 MiB ≥ 每个 worker 的 MiB`，现在是负数，所以 0 个 worker 能放。另外 h100 只有 20 核（上限 40，`h100_workers.py:24`）。
- **R16**：环境已齐，代码还缺（§4.2 前四行）。更根本的是 R16 服务固定在 4090，h100 是「本机层」，按规则不给外机服务。要让 R16 用 H100 worker，得 owner 改规则（h100 进 SHARED），或者把 R16 服务挪到 H100。
- **R15**：服务本来就在 H100，最适合用 H100 本机 worker（`nodes()` 会给 h100 local 类）。但还缺三样：
  - h100 上的 island（§3）
  - fleet/run_arm/control/node 的主机表，外加 h100 撞名（§4.2）
  - 架构：driver 和 agent 同进程在一台 worker 机上（`control.py:769`、`run_gtp.py:414`）。要么整条 arm 设 `WORKER_HOST=h100`，driver 和所有 worker 都在 h100（回环，不用共享池）；要么加只跑 agent 的进程，让 h100 和共享池同时给一个 R15 批次。
