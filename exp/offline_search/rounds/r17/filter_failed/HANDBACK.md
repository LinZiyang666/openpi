# Failed-episode removal with the original fit — 2026-10-15

**CURRENT HANDOFF: CPU VERIFIED; ALL SEVEN 4090 JOBS PREPARED.** The coordinator's fairness decision is implemented. All **19 R17 filtered jobs pass** their frozen CPU descriptor/identity startup path with a pinned existing interpreter. **17 match the source's actual serving identity; 2 use new unfiltered controls.** Both controls also pass and have exactly the paired filtered identity. No identity check was weakened, package installed, global environment changed, or model server/worker/driver/chain started by this work.

## Coordinator decision implemented

Each R17 campaign records `server_python`, read by the existing `formal_eval.host_python` launcher hook. Defaults for other campaigns remain unchanged. Stage and launch consume the same pin, including MPS and the server/proxy process. `SERVER_ENVIRONMENTS.json` preserves the selection on repeated preparation/build. DP LIBERO uses `/home/weiland/projects/openpi/.venv/bin/python`, whose six recorded package versions exactly match the source-sealed manifest. DP MT50 retains its H100 environment, whose six recorded versions match its own source manifest. Other model manifests do not record package-version fields: their original interpreter is retained, observed package versions are reported, and their complete content/ModelSpec seals match. No unrecorded historical package version is claimed.

Interpreter aliases used below:

| Alias | Exact executable on serving host |
|---|---|
| Repo 3.11 | `/home/weiland/projects/openpi/.venv/bin/python` |
| H100 DP | `/data/mw_finetune/dp/env/bin/python` |
| H100 Fast-WAM | `/data/mw_finetune/envs/fastwam/bin/python` |
| H100 GR00T | `/home/exouser/gr00t_n15_venv/.venv/bin/python` |
| H100 Cosmos | `/data/cosmos/cosmos-policy/.venv/bin/python` |

## Per-job CPU startup and historical identity comparison

DP calls the unchanged frozen `qualification.model_runtime.load(..., 'cpu')`, including the strict tensor/identity guards and final ModelSpec comparison. The large VLA/video models call the exact descriptor/content-identity builders used by their frozen loaders; their policy backends are not constructed. Every job also parses the actual staged `full.yaml` and Cache-only YAMLs and checks retrieval-input paths; DP LIBERO additionally checks native batch/admission/config validation, retaining the original R4 admission bridge. This is CPU startup identity qualification, not CUDA graph or GPU execution qualification.

All original H100 `servers/*/ready.json` files were read on H100, including the ten source segments whose local serving archives are absent. Every source's replicas agree on one model identity. Equality below compares the computed serving model identity to those actual records, not merely to `assets/model_spec.json`.

| Filtered job | GPU | Interpreter | CPU passes | Equals source actual | Primary unfiltered baseline |
|---|---|---|---|---|---|
| `ff1015_cosmos_li_01` | H100 | H100 Cosmos | Yes | Yes | Historical |
| `ff1015_cosmos_li_02` | H100 | H100 Cosmos | Yes | Yes | Historical |
| `ff1015_cosmos_li_03` | H100 | H100 Cosmos | Yes | Yes | Historical |
| `ff1015_cosmos_li_04` | H100 | H100 Cosmos | Yes | Yes | Historical |
| `ff1015_dp_mt_05` | H100 | H100 DP | Yes | Yes | Historical |
| `ff1015_fastwam_mt_06` | H100 | H100 Fast-WAM | Yes | Yes | Historical |
| `ff1015_fastwam_mt_07` | H100 | H100 Fast-WAM | Yes | Yes | Historical |
| `ff1015_fastwam_mt_08` | H100 | H100 Fast-WAM | Yes | Yes | Historical |
| `ff1015_groot_mt_09` | H100 | H100 GR00T | Yes | Yes | Historical |
| `ff1015_groot_mt_10` | H100 | H100 GR00T | Yes | Yes | Historical |
| `ff1015_dp_li_11` | RTX 4090 | Repo 3.11 | Yes | Yes | Historical |
| `ff1015_fastwam_li_12` | H100 | H100 Fast-WAM | Yes | Yes | Historical |
| `ff1015_pi05_mt_13` | RTX 4090 | Repo 3.11 | Yes | Yes | Historical |
| `ff1015_dp_li_14` | RTX 4090 | Repo 3.11 | Yes | No | `ff1015_dp_li_14_u` |
| `ff1015_dp_li_15` | RTX 4090 | Repo 3.11 | Yes | No | `ff1015_dp_li_15_u` |
| `ff1015_dp_li_16` | RTX 4090 | Repo 3.11 | Yes | Yes | Historical |
| `ff1015_fastwam_li_17` | H100 | H100 Fast-WAM | Yes | Yes | Historical |
| `ff1015_fastwam_li_18` | H100 | H100 Fast-WAM | Yes | Yes | Historical |
| `ff1015_fastwam_li_19` | H100 | H100 Fast-WAM | Yes | Yes | Historical |

Evidence: `STARTUP_CHECKS.json`, all `startup_checks/<job>.json`, `SOURCE_READY_ACTUAL.json`, `FAIRNESS_VERIFICATION.json`. All computed ModelSpecs equal the source-sealed ModelSpecs. No CPU probe initialized CUDA. Source-native runtime/worker archives remain unchanged.

## New unfiltered controls and analysis

| New control | Paired filtered job | Cells | Extra simulations | GPU / topology |
|---|---|---|---:|---|
| `ff1015_dp_li_14_u` | `ff1015_dp_li_14` | DP spatial + 10, N50 | 1,000 | 4090, R4 / B8, local=60 |
| `ff1015_dp_li_15_u` | `ff1015_dp_li_15` | DP spatial + 10, N100/200/300/400 | 4,000 | 4090, R4 / B8, local=48 |

Both controls use the same chosen interpreter, frozen runtime, sampler, commitment, renderer/worker archive, test init identities and seeds as the paired filtered run. They copy the **original full size-N VIEW, row indices, READY and fit.npz byte-for-byte from the actual source serving assets**. Failed library episodes remain eligible. No top-up, refit or empty-task exclusion is applied to controls. New standard Cache-only studies are under `studies/ff1015_dp_li_{14,15}_u_unfiltered/`; the original studies/campaigns are untouched.

Extra simulations by model: **DP 5,000; π0.5 0; GR00T 0; Fast-WAM 0; Cosmos 0.** There are now 23 jobs. Planned filtered simulations remain 86,500; total new simulations including controls are **91,500**. Library failure counts and parent episode totals later in this document are unchanged.

Analysis uses the rerun control as the primary unfiltered outcome for these ten cells. Missing control outcomes remain pending; they never silently fall back to historical results. `historical_unfiltered` retains the old baseline in primary/secondary JSON and CSV columns and per-init output. The existing secondary SR over non-empty tasks remains paired against the same task set in the chosen unfiltered baseline. Optional control continuations use `--control-run`.

## Reset and 4090 preparation completed

The latest `_14` model-load failure was reset with `reset.py`. The coordinator explicitly authorized this closed startup reset: receipts prove closure and no driver/test episode was dispatched. The reset now supports that case, preserves changed startup inputs, replica/MPS logs and directories, and renames closed worker process receipts/leases/logs and its run directory before re-staging. Nothing was deleted, stopped or started. Latest reset receipt: `/data/mw_finetune/eval/ff1015_dp_li_14/RESET.failed_20261015T230253963646Z.json`.

All **seven** jobs in the 4090 launch order below are PREPARED with exact local seal hashes checked, no active `LAUNCH.json`, `RECEIPT.json` or `.startup_inputs`, and valid MPS path lengths. Actual local/timan1/timan107 staged files and immutable dependencies were checked; see `STAGING_4090_FINAL.json`. Preparation of the older π0.5 worker archive needed a caller compatibility fix: non-local MetaWorld preparation omits `local=False`, which that archived installer did not accept. Defaults are identical and no worker file/archive changed. The coordinator's recorded-only RAM admission change in `formal_prestage.py` remains unchanged.

**43 CPU unit tests PASS**, covering campaign interpreter selection, closed startup reset/evidence preservation, control-baseline replacement and pending behavior, row filtering and existing host defaults. Both control studies validate with the standard study validator; their planners dispatch 1,000 and 4,000 original test identities. Control views/fits and all seven native archives match their originals byte-for-byte.

## R10 π0.5 worker island restoration (files built; rsync NOT run)

The original deployment path is `exp/offline_search/closed_loop/ops/h100/control.py::_setup(worker_only=True)`, documented in `closed_loop/ops/h100/HANDBACK_LOCAL.md`; R13's `impl/HANDBACK.md:44` also states that normal sync does not deploy runtime sources. The R16 `conductor_deployment/worker.tar` targets a different conductor island and is not substituted for R10.

`r10_restore.py` invokes the original standard worker-only deploy builder with all remote calls/transfers intercepted into a local tree. It includes source dependencies, regular `src/openpi_client` aliases, official A init pools, node/run helpers, the exact timan107 helper relocation, original private LIBERO config, and the 24 run-specific YAML/matrix files. Its resulting tree is the concrete, SHA-bound source for the restore; it does not require an unknown backup path. The existing native Python/simulator, BDDL/init directories and LIBERO asset directory were verified present on timan107 and are not changed.

Local restore root: `/data/r17_library/libero/pi05/r10_worker_restore_ff1015/root` — **8,987 files, 77,083,041 bytes**. Every local file hash passed. `R10_RESTORE_MANIFEST.json` lists all files and provenance; the transferred `SHA256SUMS` uses absolute destination paths. SHA256 of that SHA list: `df8a8c9a7a3ca790899fdb80546de843d0772254adfc2350daa648a7deaa161c`.

Exact public-network rsync commands are in `R10_RESTORE_COMMANDS.txt`. They were **not executed**. Run the transfer alone on the existing preparation rsync port. First, in a weilandserver terminal, leave this read-only transport daemon in the foreground:

```bash
rsync --daemon --no-detach --port=23118 --config=/data/r17_library/libero/pi05/r10_worker_restore_ff1015/rsyncd.conf
```

In a **timan107 terminal**, run these exact commands (file data travels directly over public rsync, never through tether push/pull/expose):

```bash
mkdir -p /scratch/zixuans8/openpi_trace_h100
rsync -rlt --checksum --bwlimit=16384 --timeout=90 --contimeout=15 rsync://ziyanglin.com:23118/r10_ff1015/ /scratch/zixuans8/openpi_trace_h100/
sha256sum -c /scratch/zixuans8/openpi_trace_h100/SHA256SUMS
```

CPU mkdir/verification from weilandserver, if needed:

```bash
tether exec --safe --timeout 30s timan107 -- mkdir -p /scratch/zixuans8/openpi_trace_h100
tether exec --safe --timeout 120s timan107 -- sha256sum -c /scratch/zixuans8/openpi_trace_h100/SHA256SUMS
```

After successful SHA verification, stop the foreground rsync transport with Ctrl-C. Then run the existing π0.5 prepare script to sync/verify its H100 fits and worker configs. This restoration does not launch any R10 chain.

## Exact launch order per GPU — coordinator only

Run from `/home/weiland/projects/openpi`. Neither launch-order script was executed. Each campaign must finish/close before the next in its lane; the coordinator also manages worker-fleet sharing between GPU lanes. Original R/B/W remains listed in the topology table later in this handback.

**RTX 4090:** all seven are already prepared, in this exact order:

1. `ff1015_dp_li_14_u`
2. `ff1015_dp_li_14`
3. `ff1015_dp_li_15_u`
4. `ff1015_dp_li_15`
5. `ff1015_dp_li_11`
6. `ff1015_dp_li_16`
7. `ff1015_pi05_mt_13`

```bash
bash exp/offline_search/rounds/r17/filter_failed/commands/launch_4090_in_order.sh
```

That script invokes the seven existing per-job `commands/launch_<name>.sh` commands sequentially. To repeat CPU preparation before launch, use each matching `commands/prepare_<name>.sh`; do not rebuild the run directories. `_14` is already reset, so do not repeat reset unless another eligible closed startup failure occurs.

**H100:** prepare immediately before each launch, in this exact order. Restore R10 π0.5's timan107 island before its listed step:

1. `ff1015_cosmos_li_01`
2. `ff1015_cosmos_li_02`
3. `ff1015_cosmos_li_03`
4. `ff1015_cosmos_li_04`
5. `ff1015_dp_mt_05`
6. `ff1015_fastwam_mt_06`
7. `ff1015_fastwam_mt_07`
8. `ff1015_fastwam_mt_08`
9. `ff1015_groot_mt_09`
10. `ff1015_groot_mt_10`
11. `ff1015_fastwam_li_12`
12. `ff1015_fastwam_li_17`
13. `ff1015_fastwam_li_18`
14. `ff1015_fastwam_li_19`
15. `r17_filter_failed_pi05`
16. `r17_filter_failed_groot`

```bash
bash exp/offline_search/rounds/r17/filter_failed/commands/prepare_and_launch_h100_in_order.sh
```

`LAUNCH_ORDER.json` and the generated scripts preserve this order. H100 server assets have been staged for CPU identity verification; the H100 order script performs each standard full worker preparation before its launch. R10 GR00T retains its existing timan108 island. The aggregate all-job scripts now include the two controls and the same 4090-then-H100 order.

## Exact CPU verification and analysis commands

These commands start no serving/simulation process. Run before launching any listed run; the CPU checker refuses to rewrite a run with an active launch claim or receipt.

```bash
cd /home/weiland/projects/openpi
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m pytest -q exp/offline_search/rounds/r17/filter_failed/test_staging.py exp/offline_search/rounds/r17/filter_failed/test_row_filter.py exp/offline_search/rounds/r17/filter_failed/test_constraints.py exp/r17_integration/models/pi05/test_host_runtime.py
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.check_startup
# The controls' exact startup checks, with existing staged assets:
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.check_startup --no-stage --job ff1015_dp_li_14_u --job ff1015_dp_li_15_u
# After coordinator results arrive (also works now with pending fields):
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.analyze
```

The owner-authorized reset command, already executed for the latest failure, is:

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.reset --run /data/mw_finetune/eval/ff1015_dp_li_14
```

The sections below retain the original method, counts, topology and earlier failure evidence. Their previous blocked-state notes and earlier launch order are superseded by this current handoff.

---

**HISTORICAL STATE BEFORE THE COORDINATOR DECISION: SOURCE IDENTITY CONFLICT.** The coordinator's next `_14` launch (2026-10-15 22:46 UTC) passed staging, then failed DP model loading. The failed run remains failed; it has not been reset again. The new strict CPU diagnosis below reproduces the failure without changing any identity check. The earlier file-staging proofs remain file proofs, not server-startup qualification. The 84 filtered views/artifacts and 21 definitions remain written. No model server, worker, driver or chain was started by this repair/diagnosis.

## DP model-load diagnosis: stopped under the source-fact rule

The latest task's presumed descriptor-copy defect is contradicted by the files: `_14/assets/{load.json,model_spec.json,model_manifest.json,benchmark.json,full.yaml}` are byte-for-byte identical to **both** `csztd_r2_c383_r2_e5b0` and `csztd_r3_l48`. The model, domain, checkpoint, device, frozen runtime and runtime identity also agree. A local descriptor audit found source-identical copies for **all 19 R17 jobs**; see `SOURCE_STARTUP_AUDIT.json`. This audit is not a claim that all 19 staged server loaders pass.

DP's frozen `DPAdapter` includes `dependency_versions` in `code_digest`, then includes that code digest in `checkpoint_identity`. The currently selected `/data/mw_finetune/dp/env/bin/python` has different packages from the source's sealed manifest:

| Dependency | Source-sealed version | Current DP environment |
|---|---|---|
| torch | 2.7.1 | 2.7.1+cu128 |
| torchvision | 0.22.1 | 0.22.1+cu128 |
| diffusers | 0.37.1 | 0.39.0 |
| transformers | 4.53.2 | 5.5.4 |
| numpy | 1.26.4 | 2.2.6 |
| safetensors | 0.7.0 | 0.8.0 |

The unchanged frozen `qualification.model_runtime.load(..., 'cpu')` reproduces the actual launch failure. Weights and normalization match their pins; tensor loading is strict with **475 keys, zero missing, zero unexpected**. Every DP source-code digest agrees with the sealed manifest. Recomputing the identity with just the sealed dependency versions restores the expected identity exactly; neither production descriptors nor dependency identities were patched.

There are **three distinct identities**, not a filtered/source descriptor difference:

| Evidence | Model identity |
|---|---|
| Both source `assets/load.json` / `model_spec.json`, and the filtered copy | `sha256:bd1dd4c1b2e1bfbb4c660ed8ec0a50362f461dc8a631c16edfa9d3097879d0c5` |
| Current frozen runtime + selected DP Python environment, strict CPU reproduction | `sha256:17b1d6a1f389e09bd853aa384df9c3b27e3cff1b6d03c7e253e5a5e002ff203d` |
| **Actual unfiltered source** `csztd_r3_l48/servers/replica_0/ready.json` (also `_14`'s direct source `csztd_r2_c383_r2_e5b0`) | `sha256:110944f6457c8ee389d5726cc60464875aed71846d9cc19967c048beb3f48ecb` |

The historical serving code digest is `sha256:1eacb5b538e19bdf94801b24b31ba4aacb7f29e12f1d41831b13023aab6f77ea`, while the source-sealed code digest is `sha256:6536008be9eb5f64188af8636a6b0a255c004adf6e418d6e0b976b6a362316d5`. The original source launched on 2026-10-11, before the current DP environment directory's October 13 timestamp. The source's saved serving spec already disagrees with its current sealed assets; copying those assets again cannot resolve that conflict.

The existing repository Python 3.11 environment has exactly the source-sealed dependency versions. As a **diagnostic only**, loading the same actual filtered assets through the same frozen loader with that interpreter passes both the strict loader guard and `model_runtime.load`'s final ModelSpec comparison, yielding `bd1dd4c1…`. It still does not reproduce the historical unfiltered serving identity `110944f6…`. No campaign interpreter was changed to this alternative.

Per the binding instruction, **“If a source fact contradicts this brief, stop and report it instead of guessing,”** work stopped on this conflict. Build/stage code, model descriptors, interpreter selection and native identity checks were not changed. The 19-job startup proof and further startup checks are **not complete**. The owner/coordinator must identify the authoritative source runtime/environment or choose paired unfiltered reruns before preparation can be repaired without changing the comparison.

The latest receipt confirms closure but includes owned `replica_0`, MPS and a local worker fleet. Existing `reset.py` deliberately only accepts failures before any server/worker started, so it was **not run** on this new failure. `_14` still has its latest `LAUNCH.json`, `RECEIPT.json`, `.startup_inputs`, logs and process evidence. Reset/re-prepare is deferred with the identity fix; no clean, launchable state is claimed.

Evidence: `STARTUP_IDENTITY_BLOCKER.json`, `DP_STARTUP_ACTUAL.json`, `DP_STARTUP_SEALED_ENV.json`, `SOURCE_STARTUP_AUDIT.json`. The diagnostic trace only observes the failing loader's local variables. Both CPU integrations report `cuda_initialized:false` and `loader_unchanged:true`.

Exact CPU reproduction commands (no server, worker, driver or launch entry point):

```bash
cd /data/mw_finetune/qualification/formal_runtime_62692e198a28b32f9eece681d458d644e30b9ae5a1bd33f84eded015b5bff6c0
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src:$PWD" /data/mw_finetune/dp/env/bin/python /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed/diagnose_dp_startup.py --run /data/mw_finetune/eval/ff1015_dp_li_14 --output /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed/DP_STARTUP_ACTUAL.json
# Diagnostic comparison against the sealed package versions; not a campaign change:
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src:$PWD" /home/weiland/projects/openpi/.venv/bin/python /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed/diagnose_dp_startup.py --run /data/mw_finetune/eval/ff1015_dp_li_14 --output /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed/DP_STARTUP_SEALED_ENV.json
```

The older prepare/reset/launch commands below are retained as historical handback instructions. They do not authorize or establish readiness for relaunch after this identity failure. The separately reported R10 π0.5 worker-island blocker also remains unresolved.

## Earlier launch-input repair and state before the latest coordinator launch

`prepare_workers(payload_override=...)` writes content-addressed `control_*.json` files, but `formal_prestage.launch_prepared` hashes the fixed `stage/benchmark.json` against the original `assets/benchmark.json` bytes. The content-addressed serializer also changes whitespace, so simply copying its benchmark JSON to the fixed name would still fail the hash check.

`stage.py` now transfers the original benchmark bytes to `benchmark.json` on every worker node, plus the prepared worker JSON to `campaign.json` and `launch_campaign.json`. It retains and re-stages both content-addressed controls used by the frozen workers. The existing launcher replaces `launch_campaign.json` with its final routed config at launch. On H100, staging installs the complete asset snapshot and both campaign names after worker preparation has populated its final metadata. Preparation also supplies the standard `coordinator_runtime` mirror reference and carries over the original `LOCAL_WORKER_ORCHESTRATION.json` receipt for local-worker runs. The native runtime and worker archives remain byte-identical to the original source archives.

Repeated prepare repairs staged copies even when `PREPARED.json` already exists. Preparation and reset use the existing preparation lock. An active `LAUNCH.json` or `RECEIPT.json` requires the explicit reset path; prepare does not silently clear either file. No new campaign validation/admission gate was added. `inspect_stage.py` is an optional diagnostic only.

The coordinator's RAM-admission edit in `exp/r17_integration/ops/formal_prestage.py` is **unchanged**: archived admission results are recorded without raising. That file's SHA256 before and after this repair is `362bf952444b54c0b2c3153ed40f76535a3bfada74db02254b4da5d930b27065`.

`reset.py --run <dir>` accepts only a failed prepared-validation launch, with `gpu_bound:false`, closure confirmed, no owned processes or execution artifacts, and a departed launcher process. It renames launch claims, receipts, the startup snapshot, launch-admission files and timing/retry evidence with one `.failed_<UTC timestamp>` suffix, then re-stages the existing sealed inputs. It never deletes failed-launch evidence.

The earlier reset was executed on `/data/mw_finetune/eval/ff1015_dp_li_14`. Its evidence is preserved under suffix `.failed_20261015T223242950683Z`; the reset receipt is `RESET.failed_20261015T223242950683Z.json`. At that earlier handback, the active launch claim, receipt and startup snapshot were absent and `PREPARED.json` was valid. The coordinator's subsequent model-load attempt created new active failure evidence, as documented above. `_11` and `_16` were prepared fresh, without reset.

### Actual CPU staging proof

The diagnostic hashes every sealed local input, original source pin, runtime/driver release, worker payload, applicable dependency/asset release, private config, canonical staged JSON and content-addressed control. H100 asset checks include nested retrieval artifacts. R10 checks the standard synced dependency plan, worker YAMLs/matrices, launch helpers, private config and original A init files. It invokes no launch entry point.

| Run | Worker-node proof | Server-side proof | State |
|---|---|---|---|
| `ff1015_dp_li_14` | local: 2,477 files PASS | local sealed inputs/releases: 1,520 files PASS | Reset, PREPARED |
| `ff1015_dp_li_11` | timan107: 2,475 files PASS | local sealed inputs/releases: 1,511 files PASS | Fresh PREPARED |
| `ff1015_dp_li_16` | timan107: 2,475 files PASS | local sealed inputs/releases: 1,505 files PASS | Fresh PREPARED |
| `ff1015_fastwam_mt_07` | timan1 / timan107 / timan108: 66 files each PASS | H100: 790 files PASS; coordinator: 1,574 files PASS | PREPARED, repeated prepare checked |
| `r17_filter_failed_groot` | timan108: 54 files PASS | H100: 57 files PASS | Standard SYNC_OK; all 12 arms staged |

Evidence: `STAGING_R17.json`, `STAGING_R10.json`, consolidated `STAGING_PROOF.json`. Counts include existing immutable releases; these were inspected, not rebuilt. The fixed-name staging helper is common to all four worker-node types, including timan1's distinct `/scratch/.../formal/stage/` location. **29 CPU tests PASS**; all **45 shell scripts** pass `bash -n` without execution.

### R10 π0.5 blocker

The standard π0.5 prepare transferred and verified all 84 H100 dependency-plan files (16,706,534,893 bytes), then failed before completing worker sync because `/scratch/zixuans8/openpi_trace_h100/os_cl/node.py` does not exist on timan107. CPU inspection also found `run_arm.sh`, `run_gtp_subset.py`, `libero/config.yaml` and `exp/offline_search/rounds/r10/method.py` absent from that island. Both original `r10_size_pi05/h100_sync/synced.json` and its actual `h100_launch.json` records identify this same timan107 island. This is separate from the fixed R17 benchmark-path bug.

The original R10 timan108 island used by GR00T is present. No replacement worker runtime was installed, and the π0.5 topology was not changed. Under the owner's worker-file rule, the coordinator must restore the original timan107 island from its retained snapshot. Its source path is not known, so an exact restore rsync command cannot be specified without inventing a source. The required destination is `/scratch/zixuans8/openpi_trace_h100/`; restore the original island, including its source tree, `os_cl` helpers/private config and A init pools. Re-run the existing π0.5 prepare command after restoration. Do not launch this job until its sync succeeds. This blocker does not require resetting π0.5: no chain was invoked and no `synced.json` was written for the failed sync.

### Exact repair, prepare, inspect and launch commands

Run from `/home/weiland/projects/openpi`. The reset command below **was already executed** on `_14`; use it again only after another eligible pre-start failure:

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.reset --run /data/mw_finetune/eval/ff1015_dp_li_14
```

Re-stage any existing unlaunched R17 run with its unchanged prepare script. These four now have successful prepares and node proofs:

```bash
bash exp/offline_search/rounds/r17/filter_failed/commands/prepare_ff1015_dp_li_14.sh
bash exp/offline_search/rounds/r17/filter_failed/commands/prepare_ff1015_dp_li_11.sh
bash exp/offline_search/rounds/r17/filter_failed/commands/prepare_ff1015_dp_li_16.sh
bash exp/offline_search/rounds/r17/filter_failed/commands/prepare_ff1015_fastwam_mt_07.sh
bash exp/offline_search/rounds/r17/filter_failed/commands/prepare_r17_filter_failed_groot.sh
```

Optional CPU diagnostic (reads the existing preparations; starts no model/simulator processes):

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.inspect_stage --run /data/mw_finetune/eval/ff1015_dp_li_14 --run /data/mw_finetune/eval/ff1015_dp_li_11 --run /data/mw_finetune/eval/ff1015_dp_li_16 --run /data/mw_finetune/eval/ff1015_fastwam_mt_07 --run /home/weiland/trace_runs/os_closed_loop/r17_filter_failed_groot
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m pytest -q exp/offline_search/rounds/r17/filter_failed/test_staging.py exp/offline_search/rounds/r17/filter_failed/test_row_filter.py exp/offline_search/rounds/r17/filter_failed/test_constraints.py
```

Coordinator-only launch commands, to execute sequentially with their original GPU/topology (none executed by this repair):

```bash
bash exp/offline_search/rounds/r17/filter_failed/commands/launch_ff1015_dp_li_14.sh
bash exp/offline_search/rounds/r17/filter_failed/commands/launch_ff1015_dp_li_11.sh
bash exp/offline_search/rounds/r17/filter_failed/commands/launch_ff1015_dp_li_16.sh
bash exp/offline_search/rounds/r17/filter_failed/commands/launch_ff1015_fastwam_mt_07.sh
bash exp/offline_search/rounds/r17/filter_failed/commands/launch_r17_filter_failed_groot.sh
```

After the coordinator restores the original R10 π0.5 worker island:

```bash
bash exp/offline_search/rounds/r17/filter_failed/commands/prepare_r17_filter_failed_pi05.sh
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.inspect_stage --run /home/weiland/trace_runs/os_closed_loop/r17_filter_failed_pi05 --output exp/offline_search/rounds/r17/filter_failed/STAGING_R10_PI05.json
# Coordinator only, after successful sync/inspection:
bash exp/offline_search/rounds/r17/filter_failed/commands/launch_r17_filter_failed_pi05.sh
```

Every other job retains its exact `commands/prepare_<name>.sh` and `commands/launch_<name>.sh` pair in `CAMPAIGNS.json`. Prepare already-built runs with the repaired staging code; do not rebuild views, fits or campaigns. Aggregate all-job scripts below still include the blocked π0.5 job and therefore require its worker-island restoration first.

## Binding method

This implements the owner-delegated change of method: **no refit**. Every cell retains its original size-N PCA, whitening/metric, normalization, confidence/calibration values, corrector parameters and other fitted state. Only the rows eligible for retrieval change. Failed episodes are removed from each original nested cohort without top-up. Singleton and two-episode tasks use their original fitted metric. Earlier refit experiments and their proof results are superseded and are not part of these campaigns.

Pure Cache only: group `c`, original native Cache commitment, corrector off, guard off, no periodic knob, no long-horizon arm and no policy/MISS calls. R10 uses its original A/BlindAWM arm with `full_model=False`, original blind behavior, original `kref=5` at N50 and `kref=8` otherwise. All original model/serving YAML settings, task/init identities, test seeds and GPU types are retained.

Empty tasks are omitted from simulation. Analysis assigns each of their 50 test inits a failure in the primary comparison. The secondary comparison includes the same non-empty tasks in both filtered and unfiltered arms. MT50 reports both equal-weight difficulty-group SR and equal-weight task SR; the JSON also contains individual task SRs and difficulty-group task counts. Secondary group means average the difficulty groups represented among the non-empty tasks.

## Artifacts and representation

- `CELLS.json`: one entry per model × suite × collected size, with kept episode IDs/counts, empty tasks, exact candidate-view path, original fit fingerprints and filtered serving-artifact paths.
- `BASELINE.json`: all 90,000 original Cache outcomes, keyed by cell and original task/init, including the execution segment that actually supplied each outcome.
- `CAMPAIGNS.json`: 21 jobs, source campaign, exact native runtime identity, GPU, R/B/W, expected simulations and exact prepare/launch script paths.
- `studies/{pi05,groot,dp,fastwam}_mw/STUDY.json` and `studies/{dp,fastwam,cosmos}_libero/STUDY.json`: seven standard R17 manifests with group `c` only.
- `/home/weiland/trace_runs/os_closed_loop/r17_filter_failed_{pi05,groot}/`: standard R10 `arms_in.json`, emitted `arms.json`, source-identical server YAMLs, client matrices, 24 preloaded filtered artifacts, official A manifests and standard H100 transfer plans.

R17 candidate views are siblings of the originals: `subsets/size_<original-label>_keep_success_1015_norefit/{VIEW.json,row_indices.npy}`. The new VIEW `size` is the actual kept episode count; `metadata.collected_size` records the original total N. Original MT50 directory labels are episodes per task (size_5 means 250 collected episodes). R10 views are siblings of the original selection JSON/NPY files under `r10/subsets/`, with the same suffix.

R17 serving artifacts are siblings of the actual original fit, named `<fit-name>_keep_success_1015_norefit_<original-digest-prefix>`. They use the existing READY/NPZ serving format. Full payload row IDs and arrays remain intact; only per-task candidate rows and their aligned stored codes/norms are sliced. `candidate_filter` in READY records original fit identity, selected episodes and `refit:false`. R10 uses the same technique in the original SizeController/BlindAWM pickle class; parent row mapping, action payloads and blind continuation tables remain intact. Original R10 run files are unchanged; new canonical artifacts live beside `r10/artifacts/*_A.pkl`. The 84 serving artifacts occupy 2,921,960,849 bytes.

**R17 recipe store descriptors intentionally remain the original size-N views**, preserving the original fit receipt/binding. The filtered view is recorded separately as `candidate_view`; retrieval eligibility is the sliced per-task table inside the serving artifact. Do not replace `recipe.store_path` with the candidate view. This would bind the frozen fit to a different training-store identity.

## CPU validation completed

- Required no-filter save/reload proofs: R17 MT50 π0.5/N250, R17 LIBERO Cosmos/spatial/N50, R10 VLA π0.5/spatial/N50 all **PASS**, with all original serving-array bytes and fitted scalar values preserved. Evidence: `selection_proof/PROOF.json` (913 / 197 / 332 comparisons).
- Every written artifact checked against its actual original served fit: **84 cells, 88 source references, 104,651 field comparisons PASS**. Only the allowed candidate-row slices differ. Evidence: `ARTIFACT_VERIFICATION.json`, `VERIFICATION.json`.
- All **86,500** planned simulations match original task/init identities, seed/protocol/horizon/sampler metadata and original source segment; no duplicates, no unplanned episodes and no empty-task simulations. Original source R/B/W and native runtime/worker archives match.
- All **19 R17 campaign planners PASS** in their frozen native runtime plus the standard coordinator planner overlay. Evidence: `PLANNER_CHECKS.json`.
- All **seven R17 model/domain recipe factories PASS** with refitting patched to raise; no fitter was called, and corrector/guard/knob/stall remain disabled. Evidence: `LOADER_CHECKS.json`.
- All **24 R10 filtered arms PASS** the original CPU plugin/orchestrator selftest: **1,152 decisions**, zero MISS calls, zero policy-tail calls, zero guard triggers. Both standard H100 dependency plans PASS. Evidence: `R10_SELFTESTS.json`, `r17_filter_failed_{pi05,groot}_plan.log`, each new run’s `h100_sync/plan.json`.
- **16 unit tests PASS**, covering row selection, unchanged parameters/continuations, primary and secondary paired scoring, pending outcomes, MT50 weighting, rsync-only transport and the existing DP capacity/MPS bootstrap behavior.

No unfiltered R10 reruns are scheduled: the original A fit/serving classes remain executable and the required CPU proof, all filtered-arm CPU integration tests and both standard plans pass. GPU execution is left to the coordinator.

## Failure-count table

Numbers are failures removed from that original collected cohort. Kept = column size − failures; the six sizes overlap because the source cohorts are nested.

| MT50 model | 250 | 500 | 1000 | 1500 | 2000 | 2500 |
|---|---:|---:|---:|---:|---:|---:|
| π0.5 | 113 | 218 | 438 | 651 | 878 | 1075 |
| GR00T | 64 | 121 | 238 | 360 | 484 | 601 |
| DP | 97 | 188 | 377 | 557 | 738 | 923 |
| Fast-WAM | 106 | 205 | 410 | 626 | 841 | 1043 |

| LIBERO model / suite | 50 | 100 | 200 | 300 | 400 | 500 |
|---|---:|---:|---:|---:|---:|---:|
| π0.5 / libero_spatial | 4 | 6 | 7 | 9 | 11 | 13 |
| π0.5 / libero_10 | 6 | 11 | 19 | 34 | 50 | 64 |
| GR00T / libero_spatial | 5 | 9 | 14 | 23 | 33 | 44 |
| GR00T / libero_10 | 6 | 10 | 25 | 39 | 54 | 73 |
| DP / libero_spatial | 8 | 16 | 23 | 35 | 51 | 64 |
| DP / libero_10 | 8 | 18 | 36 | 55 | 70 | 95 |
| Fast-WAM / libero_spatial | 7 | 15 | 25 | 41 | 55 | 68 |
| Fast-WAM / libero_10 | 1 | 5 | 12 | 21 | 29 | 38 |
| Cosmos / libero_spatial | 5 | 8 | 11 | 19 | 27 | 35 |
| Cosmos / libero_10 | 1 | 2 | 6 | 13 | 17 | 21 |

These agree with the owner’s R17 failure-rate ranges to rounding. Exact episode IDs, per-task counts and empty-task lists are in `COUNTS.json` and `CELLS.json`.

## Episode totals per model

Collected/kept counts below count each parent episode once, across MT50 and both LIBERO suites. Repeated episodes across nested views are not counted repeatedly. Test totals are across all six sizes.

| Model | Collected parent episodes | Kept parent episodes | Planned simulations | Automatic failures | Primary scored tests |
|---|---:|---:|---:|---:|---:|
| π0.5 | 3,500 | 2,348 | 19,950 | 1,050 | 21,000 |
| GR00T | 3,500 | 2,782 | 20,550 | 450 | 21,000 |
| DP | 3,500 | 2,418 | 20,100 | 900 | 21,000 |
| Fast-WAM | 3,500 | 2,351 | 19,900 | 1,100 | 21,000 |
| Cosmos | 1,000 | 944 | 6,000 | 0 | 6,000 |

**Totals: 15,000 collected parent episodes; 10,843 kept; 4,157 failures. Evaluation: 86,500 simulations + 3,500 automatic failures = 90,000 scored tests in 84 cells.** No additional collection or unfiltered rerun episodes are required.

## GPU and topology for every campaign

R = resident replicas; B = maximum native stage batch; W lists actual execution workers. Each source segment’s original runtime, sampler, serving configuration and R/B/W are retained. Splitting by execution segment is necessary because some existing cells combine several continuations with different runtime/topology settings.

| Campaign | Original source | GPU | R / B | W | Simulations |
|---|---|---|---|---|---:|
| ff1015_cosmos_li_01 | cosmos500w85_220813 | H100 | 2 / 1 | timan1=40, timan107=45 | 1,000 |
| ff1015_cosmos_li_02 | cosmossizes145 | H100 | 2 / 1 | local=60, timan1=40, timan107=45 | 500 |
| ff1015_cosmos_li_03 | cosmossizes145_r8b | H100 | 8 / 1 | local=60, timan1=40, timan107=45 | 1,434 |
| ff1015_cosmos_li_04 | cosmossizes145_r8c | H100 | 8 / 1 | local=60, timan1=40, timan107=45 | 3,066 |
| ff1015_dp_mt_05 | csdp_1504 | H100 | 4 / 8 | timan1=20, timan107=30 | 14,100 |
| ff1015_fastwam_mt_06 | csfw_k3 | H100 | 3 / 16 | timan1=20, timan107=30 | 6,750 |
| ff1015_fastwam_mt_07 | csfw_k3w5 | H100 | 3 / 16 | timan1=40, timan107=40, timan108=40 | 2,350 |
| ff1015_fastwam_mt_08 | csfw_k3w6 | H100 | 2 / 16 | timan1=40, timan107=40, timan108=40 | 4,800 |
| ff1015_groot_mt_09 | csgr_h5 | H100 | 8 / 1 | timan1=20, timan107=40 | 4,750 |
| ff1015_groot_mt_10 | csgr_h6 | H100 | 8 / 1 | timan1=40 | 9,800 |
| ff1015_dp_li_11 | csl10d | RTX 4090 | 2 / 8 | timan107=32 | 501 |
| ff1015_fastwam_li_12 | csl10f | H100 | 2 / 16 | timan107=48 | 523 |
| ff1015_pi05_mt_13 | cspi_k2 | RTX 4090 | 5 / 1 | timan1=20, timan107=24 | 13,950 |
| ff1015_dp_li_14 | csztd_r2_c383_r2_e5b0 | RTX 4090 | 4 / 8 | local=60 | 1,000 |
| ff1015_dp_li_15 | csztd_r3_l48 | RTX 4090 | 4 / 8 | local=48 | 4,000 |
| ff1015_dp_li_16 | dpc8513584592 | RTX 4090 | 2 / 8 | timan107=32 | 499 |
| ff1015_fastwam_li_17 | fastwamc1456717485 | H100 | 2 / 16 | timan107=48 | 288 |
| ff1015_fastwam_li_18 | fastwamc6066547406 | H100 | 2 / 16 | timan1=40, timan107=45 | 5,000 |
| ff1015_fastwam_li_19 | fastwamc7615306896 | H100 | 2 / 16 | local=16, timan1=16, timan107=16 | 189 |
| r17_filter_failed_pi05 | r10_size_pi05 | H100 | 4 / 1 | timan107=48 | 6,000 |
| r17_filter_failed_groot | r10_size_groot | H100 | 4 / 1 | timan108=40 | 6,000 |

The 19 R17 campaign directories are `/data/mw_finetune/eval/<campaign>`. R10 directories are the two `/home/weiland/trace_runs/os_closed_loop/r17_filter_failed_<model>` roots. Per-job command paths and native runtime identities are in `CAMPAIGNS.json`. Ports/output directories use new run locations; generated commands run sequentially and reuse their transport ports.

## R10 provenance

R10’s actual parents are `/home/weiland/trace_runs/offline_search_store/library/<model>_<spatial|l10>/<bpool_cs|bpool_all>`: π0.5 uses `bpool_cs`, GR00T uses `bpool_all`. Success labels are in each parent’s `episodes.json` (`episode_id`, `task_id`, `success`, `start`, `end`; row labels also exist in `success.npy`).

Exact saved selections are `exp/offline_search/rounds/r10/subsets/<model>_<spatial|l10>_<N>.json` (`episode_ids_by_task`) and the sibling NPY parent-row indices. These are the original PCG64(seed=20261002+task_id) nested prefixes of 5/10/20/30/40/50 episodes per task. Each original A arm used `/home/weiland/trace_runs/os_closed_loop/r10_size_<model>/fits/r10_<model>_<suite>_<N>_A.pkl`, with canonical copies under `exp/offline_search/rounds/r10/artifacts/`. The new method loads that actual original fit and selects its eligible rows; it never calls `SizeController.fit`.

GR00T R10 `bpool_all` contains exactly 500 episodes per suite, with 44 failures in spatial and 73 in libero_10. The separate `/data/r17_library/libero/groot/<suite>/*_sp16_manifest.json` assets reporting 36–49 trajectories per task are not R10’s selected B-pool parent. They are not substituted for it. Exact paths and source hashes are recorded per cell in `CELLS.json`.

## Implementation and launch boundary

The selection code is `row_filter.py`; the build only loads, slices candidate tables and saves existing artifact formats. No new runtime class or worker file is needed. The only shared-code edit is four optional planner lines in `exp/r17_integration/ops/cache_study.py`: a shard may retain the original `num_trials_per_task=50` metadata while dispatching fewer inits. Existing rows without that optional field keep their original behavior. This coordinator-only file is included by the existing driver orchestration overlay; native server/model and worker archives stay original.

`stage.py` is coordinator-only remote CPU staging: it reuses the exact original native/worker archives, uses existing worker preparation/admission, and seals PREPARED.json. `launch.py` invokes the existing `formal_prestage.launch_prepared`, retaining its resource ownership, admission, retry and cleanup. It preserves the original worker archive instead of auto-refreshing recycle helpers. For the two local DP R4 source segments, `server_cli.py` repeats the source run’s existing admission-only R2 presentation over the frozen DP model; actual R4 residents, B8, sampler, graph and retrieval remain unchanged. This is the bridge documented in those source replicas’ `CACHE_CAPACITY_ADMISSION.json`.

R10 uses the original `control` planner/chain through `legacy.py`, with file-transfer calls routed to the existing direct rsync transport. No `setup` or global runtime replacement is performed. Tether is used only for coordinator remote commands; no tether push/pull/expose is executed. **No worker source files were edited.** The missing original R10 π0.5 timan107 worker island requires coordinator restoration as documented above. Run-specific fits, YAMLs, matrices and metadata are transferred by the prepare commands.

## Exact commands

All commands start in the repository. CPU reproduction commands (before coordinator staging/launch; do not rebuild launched run directories):

```bash
cd /home/weiland/projects/openpi
ff_review_root=$(mktemp -d /tmp/r17_ff_review.XXXXXX)
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.prove_selection --output "$ff_review_root/proof"
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.build --reuse-views
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m pytest -q exp/offline_search/rounds/r17/filter_failed/test_row_filter.py exp/offline_search/rounds/r17/filter_failed/test_constraints.py
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.verify
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.selftest_r10
```

For a clean preparation where the new candidate views do not yet exist, omit `--reuse-views` from the build command. The build never recomputes fitted quantities.

**Coordinator only — prepare and launch all 21 jobs sequentially:**

```bash
cd /home/weiland/projects/openpi
bash exp/offline_search/rounds/r17/filter_failed/commands/prepare_and_launch_all.sh
```

To stage and launch individually, use each job’s `prepare_command` followed by its `launch_command` in `CAMPAIGNS.json`. For example:

```bash
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed/commands/prepare_ff1015_cosmos_li_01.sh
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed/commands/launch_ff1015_cosmos_li_01.sh
```

`commands/prepare_all.sh` stages every job without launching. `commands/launch_all.sh` launches an already staged set sequentially. Each script contains the exact original GPU target/topology through its campaign definition. R10 prepare performs standard sync via rsync; R10 launch starts the original chain. The R10 chain collects its accepted journals automatically via the same rsync wrapper.

Analyze after results arrive (also works now, with filtered SR explicitly pending):

```bash
cd /home/weiland/projects/openpi
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.analyze
```

Outputs: `analysis/RESULTS.json` (exactly one row per cell), `RESULTS.csv`, `RESULTS.md`, and `paired_by_init.jsonl` (one row per original task/init, including automatic failures). Primary and secondary fields include filtered/unfiltered SR, paired differences and gain/loss counts. Initial output contains all 84 baseline cells and marks all filtered results pending. Missing non-empty simulations are never silently scored as failures. Additional filtered R17 continuation runs may be supplied with repeated `--r17-run /data/mw_finetune/eval/<continuation>`.


## 2026-10-16 Fast-WAM Cache continuation

`ff1015_fastwam_li_18c1` retains 3,787 and plans 1,213 episodes. Worker topology needs a coordinator choice: frozen timan1 admission currently fits 17 rather than 40. Exact commands and CPU proofs: [continuation handback](continuations/ff1015_fastwam_li_18c1_HANDBACK.md). Analysis automatically includes the registered continuation.
