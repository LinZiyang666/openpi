# Cosmos CPU evidence finalizer recovery — 2026-10-18

Both children exited immediately on importing the sealed CPU overlay's
`formal_eval.py`, before `background()` or any input hashing began:
`ModuleNotFoundError: No module named 'exp.offline_search.rounds.r17'`.
The new top-level timan1 GPU2 coordinator decorator imported a worker package
absent from the deliberately small CPU-only release. The parent ignored
`child.poll()` and reported its generic 30-second readiness timeout instead.

| Run | Last recorded ownership/control phase ended (UTC) | CPU gate published | Failed public receipt written | Parent wait |
|---|---|---|---|---:|
| fa_co_r_u_l | 09:12:50.370 | 09:13:15.081 | 09:13:45.177 | 30.096 s |
| fa_co_f_f_oc1 | 10:14:32.269 | 10:14:36.468 | 10:15:06.559 | 30.091 s |

`INVESTIGATION.json` records log text, phase tails, owners and the preserved
CPU release identities. Neither failed child was alive before recovery.
Their GPU-release input sets contain 3,195,115,245 and 649,692,455 bytes.
Those bulk checks should not be part of a 30-second import handshake either.

Changes:

- `formal_eval._gpu2_coordinator` imports the same authorization hook only
  when `prepare_workers` / `run_campaign` is called. Worker admission is unchanged.
- `formal_finalize.pin_release` also packages `direct_transfer`, needed by
  the public-network rsync evidence path. All model/worker bytes stay native.
- `await_ready` reports a child's actual error/exit immediately, with a bounded
  log tail. `background` publishes `VERIFYING_INPUTS` after verifying its CPU
  release and owner, before bulk hashing. Input checks and native audits still
  precede scientific PASS; the public receipt stays PENDING during that work.
- `wait_seal` propagates a late error/dead child instead of returning a false
  PASS. An incomplete native aggregate cannot be published as PASS.
- `retry_closed` reruns the existing native snapshot/execution/ACK/batch audits
  for an isolated readiness failure, preserving the failed receipt beside it.
  Verified completion with retirement-induced rc=2 remains eligible. Optional
  remote worker diagnostics are retained during this CPU recovery.

Every new CPU release is assembled from these current orchestration files.
Existing immutable runtime directories, served model/worker archives, campaign
definitions and chain scripts are not edited.

Recovery and both actual analysis loaders are exercised by:

```bash
cd /home/weiland/projects/openpi
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=.:src \
  .venv/bin/python -u \
  exp/offline_search/rounds/r17/ops_incidents/finalizer_1018/recover_closed.py \
  --run /data/mw_finetune/eval/fa_co_r_u_l /data/mw_finetune/eval/fa_co_f_f_oc1
```

The runner holds the existing single-reader lock, pauses when IO PSI full
avg10 exceeds 20, and caps physical reads at 4 MiB/s. Closed driver evidence is
memoized in RAM so repeated native audit/analysis calls do not rescan the disk.
Every reuse checks inode/size/mtime/ctime; no row or audit is omitted. Other
Python reads/writes retain the standard 4 MiB/s budget. `RECOVERY_PROOF.json`
records native finalization, both analysis-loader counts and measured I/O.

`TESTS.log` contains 13 targeted passing CPU cases: the real isolated archived
background process with no r17 worker package, native model bytes unchanged,
slow input checks, late bad-input rejection, actual child failure diagnostics,
receipt backups and verified retirement rc=2. No simulator, model server,
worker, driver, GPU job or experiment chain is launched.

`OTHER_RUNS.json` is a metadata-only scan of top-level eval receipts for the
same timeout. `CPU_REFINALIZATION.json` inside each recovered run links its
exact old receipt backup and the resulting native summary.

## Completed recovery

- `fa_co_r_u_l`: **PASS**, 30000 segment outcomes / 30000 merged unique outcomes; driver rc=0. Original: `/data/mw_finetune/eval/fa_co_r_u_l/RECEIPT.before_cpu_refinalize_20261018T103310248712Z.json`.
- `fa_co_f_f_oc1`: **PASS**, 3015 segment outcomes / 6000 merged unique outcomes; driver rc=2. Original: `/data/mw_finetune/eval/fa_co_f_f_oc1/RECEIPT.before_cpu_refinalize_20261018T104230677691Z.json`.

Both actual analysis loaders accept the same complete unique task/init/level outcomes. The `--include-unaffected` longseg path uses this formal-A loader for Cosmos. The verification did not scan other running lanes. Original CPU release pins/archives remain historical evidence; `CPU_REFINALIZATION.json` records the foreground recovery source pins.

Scanned 167 top-level eval receipts. Other closed runs with the same timeout: **none**. See FINAL_PROOF.json and RECOVERY_PROOF.json.
