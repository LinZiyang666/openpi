# T1 → Q: timan1 worker provisioning (2026-10-10)

Provisioning **PASS**; four workers on the authorized GPU 3 **PASS**; observation/pixel comparison **byte exact**. Recommended cooperative maximum **16 workers**, RAM reservation **6144 MiB**, VRAM reservation **1024 MiB**, recycle after **4 episodes**. Only four simultaneous workers were measured; 16 is a conservative deployment recommendation, not a measured saturation limit.

**Critical mapping: `nvidia-smi` GPU 3 is `/dev/nvidia0` (device minor 0), PCI `0000:c1:00.0`, UUID `GPU-ba276422-a0b1-e371-8327-f744be942f15`.** GPU ordinal, NVIDIA device minor, and EGL enumeration position are different namespaces on this node. The immutable payload's `worker_gpu.inventory()` currently asserts `minor_number == device`; it will reject this GPU. Q must accommodate the verified UUID/PCI/minor mapping when adding timan1. Never fall back to another card. T1 did not edit `exp/r17_integration/ops/*`.

## Node and admission facts

- Tether node: `timan1`; host: `timan1.cs.illinois.edu`.
- CPU: AMD EPYC 7413, 24 physical cores / 48 logical CPUs. Final snapshot load was 6.87/6.62/5.82; measured baseline MemAvailable 442.1 GiB. The snapshot is `2026-10-11T01:39:43Z`.
- Allowed NVIDIA/NVML ordinal: **`{3}` only**. Inventory/admission must pass `devices=[3]` / `worker_egl_devices=[3]` before any NVIDIA query; protect NVML ordinals 0–2. Query by the fixed UUID when resolving the card. Device-minor 0 is authorized because it is this exact UUID/PCI card.
- GPU: RTX A6000, 49,140 MiB reported VRAM; final GPU 3 usage recovered to its 10 MiB baseline. Driver/OpenGL version 535.183.01.
- Sixteen workers reserve 96 GiB RAM / 16 GiB VRAM and leave substantial capacity for other users. Warm native simulation is mostly one CPU per worker; cold setup/compilation used about 1.25 logical CPUs per worker after subtracting the deliberate sampling hold. Consider `nice -n 10`. Keep admission headroom and N=4 recycling; these short probes do not establish a long-run host-driver memory bound.

## Installed paths and seals

| Purpose | Path / pin |
|---|---|
| Venv Python | `/scratch/zixuans8/openpi/.venv/bin/python` (CPython 3.11.15) |
| Native command wrapper | `/scratch/zixuans8/mw_r17/timan1_provision/python-gpu3` |
| Guard directory | `/scratch/zixuans8/mw_r17/timan1_provision/gpu3_guard` |
| Worker payload | `/scratch/zixuans8/mw_r17/formal/payload_e9d40ea8b0f96d1cf1b4c6829ad2f6e358d72fe9d8c69e156328d3102a06304d` |
| Formal `workers.tar` SHA256 | `e9d40ea8b0f96d1cf1b4c6829ad2f6e358d72fe9d8c69e156328d3102a06304d` |
| PAYLOAD manifest digest | `e8a3c16d43a30f86683272087975da46831545ded940f50a7a68ed087becff62` (56 files, actual formal immutable installer revalidated) |
| Sealed simulator dependencies | `/scratch/zixuans8/mw_r17/formal/dependencies/libero311_c786215bc2061c66ed8851f6aa32e7199a9f2e364b12c47e61235f5a3e83a7c7` |
| Dependency identity | `c786215bc2061c66ed8851f6aa32e7199a9f2e364b12c47e61235f5a3e83a7c7` |
| Original asset release | `/scratch/zixuans8/mw_r17/formal/assets/libero_3eb6e3e7ee3a03663dba84ccc6d6cf9dbb4ae21fb20558ca4097abc066be57dc` |
| Original asset data | `/scratch/zixuans8/mw_r17/formal/assets/libero_3eb6e3e7ee3a03663dba84ccc6d6cf9dbb4ae21fb20558ca4097abc066be57dc/assets` |
| Original asset digest | `3eb6e3e7ee3a03663dba84ccc6d6cf9dbb4ae21fb20558ca4097abc066be57dc` |
| Config source for timan1 | `/scratch/zixuans8/mw_r17/timan1_provision/libero_config/config.yaml` |
| Config SHA256 | `dc071fb958f2effdba6b7b85cee02cc9073b76332fe89c5f4e77877b9f207dca` |
| Complete runtime-copy archive | `/tmp/r17_t1_provision_20261010/runtime.tar` |
| Runtime-copy archive SHA256 | `ef8a7ad899073cadc1f988786172aa17d6c40f99080aa1bda275af821efd946a` |

The runtime/venv, sealed dependencies, original assets, and payload were copied from timan107 into identical scratch release paths. All **49,890 regular files and 5 symlinks** were checked against the source transfer manifest; the existing base Python binary also matched SHA256 `cd947cb6d5b3f8589a2a4708198dc104474e47876c6b8215ab8b410592ab7cc1`. All native scientific payload/dependency bytes remain unchanged. Guard/wrapper files are separate.

Installed package versions: NumPy 1.26.4, MuJoCo 3.6.0, Torch 2.7.1, PyOpenGL 3.1.10, Pillow 12.1.1, psutil 7.2.2. `verify_dependencies()` revalidated dependency and original-asset seals. Native official task-map/spec identity matched `87e3a7dce0d452269383ea20c144848987c393df7bde4c22628abf8a084f21bf`; all 40 original task pools contained exactly 50 states. Seed 7, settle 10, formal wire size 256.

Reuse the verified dependency release above rather than reinstalling from timan1's existing Python 3.8 `libero_sim`. Its immutable `DEPENDENCIES.json` retains the original timan107 config SHA. The timan1 config has its own SHA and points assets, BDDL, benchmark root, and init pools directly into the sealed copied release. For each campaign, use `private_config()` with this timan1 config source and a new campaign config directory; keep the copied asset path.

Campaign scratch layout can follow timan107 exactly: `/scratch/zixuans8/mw_r17/formal/campaigns/<run>/timan1`, `/scratch/zixuans8/mw_r17/formal/runs/<run>_timan1`, and `payload_<archive_sha256>`.

## Environment and GPU isolation

Fleet supervisor: run the venv Python **outside** the private device namespace, with the following environment. Keep the guard first in `PYTHONPATH`. The guard wraps the existing known LIBERO child commands (`worker_entry`, `worker_lifetime`, `cache_study_worker`) with `python-gpu3`; Q can also explicitly use that wrapper for the children. Native validation/render commands must use the wrapper. Supervisor NVML admission stays outside the namespace and queries only GPU 3.

```text
CUDA_VISIBLE_DEVICES=GPU-ba276422-a0b1-e371-8327-f744be942f15
HOME=/home/zixuans8
LIBERO_CONFIG_PATH=/scratch/zixuans8/mw_r17/timan1_provision/libero_config
MKL_NUM_THREADS=1
MUJOCO_EGL_DEVICE_ID=3
MUJOCO_GL=egl
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
PYOPENGL_PLATFORM=egl
PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=/scratch/zixuans8/mw_r17/timan1_provision/gpu3_guard:/scratch/zixuans8/mw_r17/formal/payload_e9d40ea8b0f96d1cf1b4c6829ad2f6e358d72fe9d8c69e156328d3102a06304d/packages/openpi-client/src:/scratch/zixuans8/mw_r17/formal/payload_e9d40ea8b0f96d1cf1b4c6829ad2f6e358d72fe9d8c69e156328d3102a06304d/src:/scratch/zixuans8/mw_r17/formal/payload_e9d40ea8b0f96d1cf1b4c6829ad2f6e358d72fe9d8c69e156328d3102a06304d:/scratch/zixuans8/mw_r17/formal/dependencies/libero311_c786215bc2061c66ed8851f6aa32e7199a9f2e364b12c47e61235f5a3e83a7c7
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

Required campaign settings:

```json
{"worker_node":"timan1","worker_egl_device":3,"worker_egl_devices":[3],"worker_memory_mib":6144,"worker_vram_mib":1024,"worker_max_episodes":4,"node_workers":16}
```

The wrapper validates GPU 3's UUID / PCI / minor mapping before entry. It uses installed `bwrap` and an unprivileged private mount/user namespace: only `/dev/nvidia0` remains a real GPU device; `/dev/nvidia1`, `/dev/nvidia2`, `/dev/nvidia3` become `/dev/null` inside the child. DRM nodes not belonging to PCI C1:00.0 are similarly masked. NVIDIA control, modeset and UVM nodes remain available. No host device nodes or system configuration are changed; no sudo or package installation was used.

Masked cards disappear from EGL discovery, making the sole NVIDIA EGL device native index 0. The external guard verifies its **EGL physical UUID equals `ba276422a0b1e3718327f744be942f15` before context creation**, then aliases that same device into formal slot 3 so `MUJOCO_EGL_DEVICE_ID=3` remains valid. No simulator source patch is made. This UUID query uses [EGL_EXT_device_persistent_id](https://registry.khronos.org/EGL/extensions/EXT/EGL_EXT_device_persistent_id.txt).

`CUDA_VISIBLE_DEVICES` must be the fixed UUID or empty. `formal_workers` currently sets it empty for render-only children; that is accepted and disables CUDA compute. The measured run used the UUID. Integer CUDA selectors are rejected to avoid a second ordinal ambiguity. The returned EGL CUDA-device attribute was visible CUDA ordinal 0, which maps to the fixed physical GPU 3 UUID; the probe's legacy field name `physical_cuda_ordinal` should not be interpreted as an NVML ordinal.

Proof: all 16 measured environments reported RTX A6000, the exact allowed EGL UUID, real GPU FDs only for major/minor `195:0`, and protected GPU-device paths as `1:3` (`/dev/null`). Eighteen negative cases rejected wrong environment/CLI devices, CUDA selectors, fleet allowlists, and direct native imports outside the namespace, all before the attempted body. The positive UUID check passed. A newly created CPU fixture also proved that terminating the wrapper closes its actual child (`--die-with-parent`). Wrapper PID and native PID may differ; use the native PID for RSS/VRAM sampling and the wrapper for managed lifecycle.

## Four-worker measurements and pixel comparison

Cases: original A index 0 of task 0 in each of `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`. Each worker reset, installed the exact original state, settled 10 dummy controls, and timed 60 rendered controls per case. Captures at measured controls 0/30/60 include both 256×256 cameras, robot pose/joints, full simulator state, and native flipped/resized wire images at 96/224/256. All case/task/language/init-state bytes were checked against the timan107 reference.

| Worker | Peak RSS GiB | Peak PSS GiB | Sampled peak VRAM MiB | Rendered controls/s |
|---|---:|---:|---:|---:|
| 0 | 3.455 | 3.190 | 540 | 65.23 |
| 1 | 3.545 | 3.279 | 540 | 67.99 |
| 2 | 3.448 | 3.182 | 540 | 66.09 |
| 3 | 3.447 | 3.180 | 540 | 66.74 |

Measured loops total 240 controls per worker / 960 controls across the four workers. Combined worker-local warm loop rates sum to about 266 controls/s. The fleet completed in 32.36s after import, including cold reset/setup, capture compression and an explicit 8-second first-context hold for stable resource sampling; excluding that hold gives about 39.4 controls/s with these very short episodes. These figures exclude inference and transport. RAM sampling was roughly 0.5s; VRAM was sampled every 2s. GPU 3's aggregate peak was 2186 MiB. Retain a 1024 MiB VRAM reservation rather than treating sampled 540 MiB as a maximum. Whole-node MemAvailable reached 429.6 GiB, a gross 12.48 GiB depletion during the four-worker probe; this shared-node delta is not a direct per-process driver-allocation counter.

The loaded timan107 reference used one additional worker on its GPU 7 and measured 12.61 rendered controls/s, peak RSS 3.335 GiB / PSS 3.004 GiB / VRAM 678 MiB. It ran alongside existing campaigns and transfer I/O; this is an observed comparison, not an isolated performance ratio.

**Every one of 168 observation arrays in each of the four timan1 workers matched timan107 byte for byte: 672/672 exact.** Per worker, that includes 24 raw camera frames, 24 wire frames at each of 96/224/256, and 72 robot/simulator vectors. Pixel MAE/RMSE/max difference = 0, changed channel values = 0. The four timan1 workers also matched one another byte for byte. This verifies the four sampled A states/checkpoints only, not every task/init or policy rollout.

## Tether staging, receipts, and closure

Effective transfer allow_roots is **`[/home, /tmp, /srv]`**. A scratch push was refused with `path_outside_roots`. Stage worker bundles/configs in `/tmp/r17_formal/<run>/timan1`, then install via exec into scratch. For pull, snapshot/copy owned artifacts into `/tmp` first. No agent roots or daemon configuration were changed. One final evidence pull was refused `too_many_in_flight`; a single later fresh transfer succeeded and its SHA matched.

- Machine-readable node settings: [node_config.json](../diagnostics/timan1_provision/node_config.json).
- Provisioning, formal installer, pool/guard/UUID checks: [evidence directory](../diagnostics/timan1_provision/evidence/).
- Final native measurement: [BENCHMARK.json](../diagnostics/timan1_provision/evidence/render_gpu3_final/BENCHMARK.json).
- Final closure proof: [CLOSURE_PROOF.json](../diagnostics/timan1_provision/CLOSURE_PROOF.json), all 16 owned probe PIDs closed and GPU 3 at its 10 MiB baseline.
- Full byte/pixel differences: [PIXEL_COMPARISON.json](../diagnostics/timan1_provision/PIXEL_COMPARISON.json).
- Probe and isolation sources: [diagnostics/timan1_provision](../diagnostics/timan1_provision/).
- Remote canonical receipts: `/scratch/zixuans8/mw_r17/timan1_provision/`; final run `render_gpu3_final`. No campaign endpoint or model was launched. Workers exited normally, and the owned one-shot timan107 transfer server was stopped.
- Pulled evidence archive SHA256: `757a9850d48067db69bca5dbbb224bf9aba85eeb7b6eda771b59e4d789ceece3`.

**Safety correction / audit:** preliminary probes assumed NVML ordinal == NVIDIA device minor and rendered using minor 3, outside the owner allowlist. That mistake was detected through flat GPU-3 counters and an EGL UUID mismatch. Those workers exited before the final run; their measurements are invalid and excluded. Preliminary artifacts remain in separate remote `render_1`, `render_2`, `render_3` directories for audit. The corrected final wrapper instead validates GPU 3's UUID, PCI C1:00.0 and minor 0 before rendering. No other user's process was signalled or modified.
