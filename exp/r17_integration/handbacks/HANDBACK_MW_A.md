# MT50 framework, image-free teacher libraries, worker provisioning

Implemented through the existing formal launcher, teacher capture, parent,
row-index views and shared fits. No GPU policy server, campaign or worker fleet
was started. Only the authorized short EGL render probes ran. No commit.
Package B's concurrent model/batching/cache edits and package D's Cosmos edits
were preserved. B now calls `campaign_capture_factory` in both batched residents.

Test = seed7/init0..49 and library = seed8/init0..49 across all50 MT50 tasks:
2,500 episodes each. Dev defaults to no indices/no seed; callers may define
`dev_seed` and `dev_indices`. Seeds9/10 were not used. Episode seed reaches MT1,
policy `decision_seed`, episode metadata, capture and execution rows. LIBERO's
seed and partition dispatch remain unchanged.

MT50 library capture defaults `capture_images=false`. The generic debug factory
also accepts `capture_images=False`, and drops uint8 HWC RGB model products
before serialization. Retrieval fields, robot state, normalized action chunks,
identities, execution acknowledgments and unsuccessful non-error outcomes remain.
One-camera MT50 and two-camera LIBERO use their real camera lists/wire keys.

Full parents go to `/data/r17_library/<selector>_mt50/bpool/`, for example
`dp_mw_mt50/bpool/`. Smoke parents stay in the smoke campaign directory.
`subsets/size_{10,20,30,40,50}` contain only `VIEW.json` and `row_indices.npy`:
the number is episodes **per task**, so their episode counts are
500/1,000/1,500/2,000/2,500. Selection is nested original-index prefixes within
each task. Shared PCA/corrector/guard/distance/continuity fits work for50 tasks;
this was exercised with a synthetic500-episode,50-task subset. The existing
measured-cost requirement for IR knob calibration remains unchanged.

`--worker-counts timan107=N,timan1=M`, `--replicas`, `--max-batch` (default1),
`--batch-wait-ms` (default5) and `--mps` are wired before `formal.prepare`.
DP/Fast-WAM MT50 can batch; π0.5/GR00T useB1. Ordinary and batched H100 capture
select the existing episode-sharded factory, quota and streaming finalizer.
Cosmos MT50 is accepted by this library path and retains its reserved port window;
package D owns the actual model/provider. With remote workers, Cosmos's23240
window requires H100, since weilandserver forwards only23100..23199.

## Files changed by package A

- `exp/metaworld/episode_runner.py` — per-episode environment/metadata seed.
- `exp/r17_integration/benchmarks/spec.py` — partition seeds/counts, undefined dev.
- `exp/r17_integration/benchmarks/metaworld.py` — seed-aware RNG/data identity/metadata.
- `exp/r17_integration/conductor/manifest.py` — stamp the partition seed.
- `exp/r17_integration/debug/capture_bridge.py` — generic image filtering/factory option.
- `exp/r17_integration/debug/library_capture.py` — camera-aware, image-free capture/shared campaign factory.
- `exp/r17_integration/debug/library_shards.py` — image-free H100 episode shards.
- `exp/r17_integration/ops/formal_driver.py` — full MT50 library/smoke planning, skip LIBERO B audit.
- `exp/r17_integration/ops/formal_eval.py` — worker preparation, B-only staging, disk estimate, rsync transfer/callers; B/D edits preserved.
- `exp/r17_integration/ops/formal_server.py` — shared local/H100 capture factory.
- `exp/r17_integration/ops/formal_workers.py` — domain-specific B input arguments and MT50 GPU3 wrapper.
- `exp/r17_integration/ops/library_campaign.py` — MT50 CLI/collection/finish/subsets and batch/MPS binding.
- `exp/r17_integration/ops/timan1_workers.py` — authorize MT50, GPU3-only route, direct non-recycling worker entry.
- `exp/r17_integration/ops/worker_installs.py` — check existing matching simulator environment/create GPU3 MetaWorld wrapper.
- `exp/r17_integration/ops/worker_gpu.py` — use existing renderer spreading for MT50.
- `exp/r17_integration/ops/direct_transfer.py` — bandwidth-limited direct rsync transport.
- `exp/r17_integration/ops/formal_prestage.py` — rsync transfer call sites.
- `exp/r17_integration/ops/formal_finalize.py` — rsync evidence pull call site.
- `exp/r17_integration/ops/queue_retry.py` — rsync transfer call sites.
- `exp/r17_integration/ops/remote_json.py` — rsync transfer call site.
- `exp/r17_integration/ops/incremental_sync.py` — password-file transfer via rsync; timan1 staging stays under scratch.
- `exp/r17_integration/search/teacher_library.py` — image-free parent, MT50 row views and50-task fits.
- `exp/r17_integration/tests/benchmarks/test_metaworld.py` — new official split expectations.
- `exp/r17_integration/ops/tests/test_formal_eval.py` — update the MT50 official-count assertion to2,500.
- `exp/r17_integration/tests/conductor/test_dispatch.py` — explicitly synthetic configurable dev profile.
- `exp/r17_integration/tests/conductor/test_fix_round1.py` — explicitly synthetic configurable dev profile.
- `exp/r17_integration/tests/conductor/test_p2_fix_round2.py` — explicitly synthetic configurable dev profile.
- `exp/r17_integration/tests/ops/test_mt50_library.py` — seed/plan/capture/parent/views/fits/worker/batch CPU checks.
- `exp/r17_integration/tests/ops/test_direct_transfer.py` — real CPU rsync round-trip and direct-path/bandwidth checks.
- `exp/r17_integration/ops/RUN_CONFIGS.md` — current MT50 counts/timan1 permission (concurrent B updates preserved).
- `exp/r17_integration/ops/LIBERO_LIBRARY.md` — MT50 extension and transfer reference.
- `exp/r17_integration/handbacks/HANDBACK_MW_A.md` — this handback.

## Removed assumptions

Removed the MetaWorld seed==7-only runner restriction and the old disjoint-index
test/library/dev defaults. Removed the two-camera-only library admission.
MT50 skips LIBERO alignment, historical B-pool installation/probes and Fast-WAM
LIBERO G2 qualification. Images are filtered before writing and omitted from
the parent when disabled. Formal transfers no longer invoke tether push/pull;
existing post-transfer SHA/snapshot checks remain. No new seal, fingerprint,
admission framework or campaign receipt was introduced.

## CPU results

- New MT50/rsync tests plus incremental-sync coverage: **48 passed**,
  `/tmp/mw_a_new_final.{xml,log}`. Covers both seeds, env/RNG/metadata propagation,
  every selector's2,500 distinct seed8 coordinates,4-episode smoke, generic and
  library image omission, parent/view/fits for50 tasks, worker GPU3 routing,
  replicas/batching/MPS before prepare and actual loopback rsync.
- Existing benchmark/conductor/debug/teacher/library/runtime/asset/port coverage
  plus new tests: **237 passed,25 failed,1 subtest passed**,
  `/tmp/mw_a_final.{xml,log}`. Baseline before edits was202 passed/26 failed.
  Every final failing test ID existed before edits; the existing missing-episode
  parent test now passes. Most old failures assert validation gates that already
  contain `pass` in the starting tree. No such unrelated gate was reinstated.
- Expanded worker/prestage/finalizer/shard/salvage/cleanup/incremental coverage:
  **138 passed,20 failed**, `/tmp/mw_a_extra_final.{xml,log}`. Compared against
  isolated untouched HEAD sources at `/tmp/mw_a_cpu_baseline`: no new failing IDs.
  That baseline has136 passed/22 failed; its two extra failures are CPU-seal
  fixtures expecting assets outside the isolated checkout.
- After factoring the shared campaign factory:8 focused resident/capture/batch
  tests passed;2 further local/H100 image-free factory tests passed.
- Further formal CLI/ops/teacher retry/repair/storage/local-worker coverage:
  110 passed/16 failed (`/tmp/mw_a_ops_final.{xml,log}`), versus isolated HEAD's
  98 passed/28 failed. One failure was the old MT50=500 expectation; that test
  was updated to2,500 and passed in a targeted rerun. The remaining15 failing
  IDs already fail in HEAD. The isolated baseline's additional failures need
  external fixture assets/client imports that are absent from its temporary root.
- After the coordinator's transfer-port correction to23118:7 transport tests
  passed. Short automatic MPS run names and the native85-byte path limit were
  checked by2 passing batching/CLI tests.
- `git diff --check` passed. One existing torch/pynvml deprecation warning.

Reproduce new coverage:

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONPATH=.:src:packages/openpi-client/src \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q \
  exp/r17_integration/tests/ops/test_mt50_library.py \
  exp/r17_integration/tests/ops/test_direct_transfer.py \
  exp/r17_integration/ops/tests/test_incremental_sync.py
```

## timan1 provisioning, GPU3 and pixels

Rebuilt `/scratch/zixuans8/mw_r17/envs/sim` on timan1 using its existing
Python3.11.15 and exact timan107 package versions: metaworld3.0.0,
mujoco3.14.0, gymnasium1.3.0, numpy1.26.4, scipy1.17.1,
PyOpenGL3.1.10 and the same19 simulator/client dependency versions. Pip's cache
was confined to `/scratch/zixuans8/mw_r17/pip_cache`. No environment was
transferred with tether. Launcher preparation reuses this installed environment.

`/scratch/zixuans8/mw_r17/timan1_provision/python-mw-gpu3` is the existing
GPU3 namespace wrapper with only its Python interpreter replaced. The original
LIBERO wrapper and guard were left intact. Native MT50 children use this wrapper;
host control/inventory remains outside the namespace. timan1 stage/run/private
paths stay under `/scratch/zixuans8/mw_r17/formal`.

Actual EGL proof: physical GPU3 = RTX A6000, PCI `0000:c1:00.0`, UUID
`GPU-ba276422-a0b1-e371-8327-f744be942f15`, NVIDIA device minor0.
The guard queried EGL UUID `ba276422a0b1e3718327f744be942f15` on every rendered
frame. Open GPU FDs were only `/dev/nvidia0`, `/dev/nvidiactl`,
`/dev/nvidia-modeset`; protected NVIDIA device minors1/2/3 were masked by bwrap.
Renderer reported `NVIDIA RTX A6000/PCIe/SSE2` / `NVIDIA Corporation`.
The launcher permits at most40 MT50 workers, sets physical EGL3 and allowlist[3],
and uses RAM2048/VRAM1024MiB and no episode recycling for this MT50 route.

Rendered seed8 initial policy observations after the native15 zero-action settle
steps for reach-v3/index0, push-v3/index1 and door-open-v3/index0, on timan1 GPU3
and timan107 GPU0. Both nodes agree exactly on task bytes and robot state.
**RGB is not byte-identical across the A6000 and GTX1080:**

| Coordinate | Changed channels /691,200 | Maximum absolute uint8 difference | Mean absolute difference |
| --- | ---: | ---: | ---: |
| reach-v3,0 |254|1|0.00036747685|
| push-v3,1 |249|1|0.00036024306|
| door-open-v3,0 |369|1|0.00053385417|

Thus99.9466–99.9640% of channels match exactly, and all remaining differences
are one quantization level. No observation quantization/renderer modification
was introduced to conceal this difference. Raw frames remain on both nodes at
`/scratch/zixuans8/mw_r17/seed8_probe_mw_a/`. Comparison pulled a2MiB tar directly
from timan107 `192.17.58.207:23114` to timan1 with rsync `--bwlimit=4096`.
The exact owned probe daemon was closed; no simulator probe was left running.
Local evidence: `/tmp/mw_a_timan{1,107}_probe.json` and
`/tmp/mw_a_pixel_comparison.txt`. Tether was used only for exec commands.

## Transfer design

Reserve **TCP23118 on weilandserver** in `/home/weiland/gpu_share/LEDGER.md`;
the coordinator registers this port. Also reserve23118 for H100's
temporary transfer listener. Model/driver ports remain separate.

`formal_eval.transfer()` routes all formal preparation, finalization, shard
streaming, salvage and incremental password-file transfers through rsync.
For H100, weilandserver's rsync client connects to
`rsync://149.165.153.233:23118/<temporary-module>/payload`. For worker nodes,
remote rsync runs through `tether exec` and connects directly to weilandserver's
`rsync://ziyanglin.com:23118/<temporary-module>/payload`. Every invocation has
`--bwlimit=16384`, `--timeout=90`, `--contimeout=15`. The one shared local
transfer lock serializes campaigns below the ~33MB/s ingress. The daemon runs
as its existing user, exports only a private temporary stage, and is stopped
by its exact returned PID after the operation. Source archives are hardlinked
on their source filesystem, so H100 does not gain a third archive copy.
No SSH credentials, new passwords or tether data transfer are needed.
Existing incremental sync retains its already configured private password;
its password file now also travels via rsync. Existing closed snapshot/shard
SHA verification and acknowledgment-before-pruning remain unchanged.

The formal transfer boundary accepts old internal push/pull tokens for callers
in B's concurrently edited `cache_study.py`, but interprets them as rsync and
never executes tether push/pull. Ordinary server/worker TCP traffic is unchanged.
The public H100 transport route is implemented and CPU-tested; no H100 policy
server or live library transfer was started by this package.

## Exact coordinator commands

Run sequentially in free model windows. π0.5 and GR00T share23160..23169.
Defaults resolve the existing immutable checkpoint paths. The smoke commands
use local native server environments for all four models; H100 full collection
uses the existing formal host's native environment/assets. In particular,
π0.5/GR00T H100 require those existing islands to be installed there; this
package provisioned the simulator workers, not H100 model environments.

```bash
cd /home/weiland/projects/openpi
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

(a) Each smoke is seed8,2 tasks x2 indices, no stored images. Prepare first;
the run uses a fresh name because preparation directories are one-shot.
This loop provides an exact pair of commands for each requested model:

```bash
STAMP=$(date +%H%M%S)
for MODEL in dp_mw fastwam_mw pi05_mw groot_mw; do
  case "$MODEL" in
    dp_mw)      PORT=23180; DRIVER=23189; TAG=dp; MEM=4 ;;
    fastwam_mw) PORT=23170; DRIVER=23179; TAG=fw; MEM=36 ;;
    pi05_mw)    PORT=23160; DRIVER=23169; TAG=pi; MEM=20 ;;
    groot_mw)   PORT=23160; DRIVER=23169; TAG=gr; MEM=20 ;;
  esac
  bash exp/r17_integration/ops/library_campaign.sh \
    --model "$MODEL" --suite mt50 --server-node local --gpu cuda:0 \
    --replicas 1 --memory-budget-gib "$MEM" --max-batch 1 --batch-wait-ms 5 \
    --worker-counts timan107=2,timan1=2 --port "$PORT" --driver-port "$DRIVER" \
    --smoke --no-capture-images --run-name "a${TAG}p_${STAMP}" --prepare-only
  bash exp/r17_integration/ops/library_campaign.sh \
    --model "$MODEL" --suite mt50 --server-node local --gpu cuda:0 \
    --replicas 1 --memory-budget-gib "$MEM" --max-batch 1 --batch-wait-ms 5 \
    --worker-counts timan107=2,timan1=2 --port "$PORT" --driver-port "$DRIVER" \
    --smoke --no-capture-images --run-name "a${TAG}s_${STAMP}" --run
done
```

(b) Full2,500-episode collection template. Set `N107`, `N1` (1..40), model,
server, replica count, memory budget and short unique run name explicitly.
DP/Fast-WAM may increaseB/addMPS; π0.5/GR00T keepB1. Driver stays on
weilandserver. The same template with `--prepare-only` and a distinct run name
does CPU preparation before collection.

```bash
MODEL=dp_mw                 # dp_mw | fastwam_mw | pi05_mw | groot_mw
SERVER=local               # local | h100 (native model island must exist)
N107=40
N1=40
REPLICAS=2                 # must fit the model's ten-port window
BATCH=8                    # DP/Fast-WAM only; pi05/GR00T:1
MEM=8                      # whole campaign GiB; choose for model/replicas
MPS_ARGS=(--mps)           # or MPS_ARGS=() for the selected off treatment
RUN=adpf_$(date +%H%M%S)   # keep short for the existing85-byte MPS pipe limit
case "$MODEL" in
  dp_mw)      PORT=23180; DRIVER=23189 ;;
  fastwam_mw) PORT=23170; DRIVER=23179 ;;
  pi05_mw|groot_mw) PORT=23160; DRIVER=23169 ;;
esac
bash exp/r17_integration/ops/library_campaign.sh \
  --model "$MODEL" --suite mt50 --server-node "$SERVER" --gpu cuda:0 \
  --replicas "$REPLICAS" --memory-budget-gib "$MEM" \
  --max-batch "$BATCH" --batch-wait-ms 5 "${MPS_ARGS[@]}" \
  --worker-counts "timan107=${N107},timan1=${N1}" \
  --port "$PORT" --driver-port "$DRIVER" --no-capture-images \
  --subset-counts 10,20,30,40,50 --seconds 172800 --episode-seconds 3600 \
  --run-name "$RUN" --run
```

(c) CPU finish/retry after the passing, closed collection:

```bash
MODEL=dp_mw
RUN=EXACT_CLOSED_RUN
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  bash exp/r17_integration/ops/library_campaign.sh \
  --model "$MODEL" --suite mt50 \
  --finish "/data/r17_library/${MODEL}_mt50/campaigns/${RUN}"
```
