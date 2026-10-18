# Prepare repairs — 2026-10-17 18:5x

Both repairs are coordinator code only. No prepare, transfer, server, worker,
driver, chain, signal, remote file write, reset or campaign rebuild was run.
Existing command pairs and campaign names remain valid. Active runs and chain
scripts were not edited.

## Continuation GPU2 bootstrap

`timan1_gpu2.auxiliary` searched an entire `python -c` payload for admission
import substrings. Native closure probes embed the original process argv as
string constants; the recorded GPU2 bootstrap source contains those substrings.
That stdlib procfs probe was incorrectly wrapped, invoking `install_runtime`
before a staged `exp` package was on PYTHONPATH. Ordinary prepare import probes
have a PYTHONPATH and did not expose this defect.

The wrapper now detects actual admission import AST nodes, ignores embedded
strings, and stops at the first command payload (so two nested wrappers cannot
wrap trailing program arguments again). Genuine GPU2 admission commands still
use the same namespace and guard. Native continuation closure checks explicitly
retain a 90-second read-only timeout.

The real native closure checks passed for **both** parents inside the same two
coordinator contexts as stage. Evidence: `fa_co_f_f_o_CLOSURE.json` and
`fa_fw_r_u_l_CLOSURE.json`.

| Continuation | Valid parent outcomes | Continuation UIDs | Total |
|---|---:|---:|---:|
| fa_co_f_f_oc1 | 2,985 | **3,015** after prepare refresh | 6,000 |
| fa_fw_r_u_lc1 | 16,810 | **13,190** including six infra retries | 30,000 |

Neither continuation has PREPARED, LAUNCH or RECEIPT. Retry prepare directly;
no reset or rebuild is necessary. Cosmos's existing `--remaining` continuation
will freeze the final 3,015-UID segment from its now-closed parent at prepare.

```bash
cd /home/weiland/projects/openpi
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_fa_co_f_f_oc1.sh
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_fa_co_f_f_oc1.sh

# On the free 4090 slot, following its existing queue:
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_fa_fw_r_u_lc1.sh
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_fa_fw_r_u_lc1.sh
```

## Legacy R15/R16 sharing

Chosen solution: **capacity-checked concurrent fleets on timan107**. R15
previously took both a local host-wide lock and the remote `fleet.lock` used by
R16. Host availability checks saw the installed island as usable, then prepare
failed on that unrelated fleet's lock.

Formal-A R15 commands now opt in with `FFK_WORKER_FLEET_ROOT`. `vla_shared`
keeps the native shared maintenance gate and exclusive per-run ownership lock,
with a separate remote fleet lock derived from the absolute run root. The
native R15 helper is adapted only in the new RPC process: its driver shell
uses the scoped lock, and cleanup counts only workers carrying this fleet's
inherited ownership marker. It cannot wait for or stop another fleet's workers.
The deployed helper and run_arm script are never overwritten. R16's existing
opt-in host wrapper likewise uses a scoped worker lock for both preflight and
agent, and the same scoped coordinator lock. Existing native lifecycle,
receipts, EGL admission, model/seed/plan checks and server configuration remain.

Prepare can stage an installed host while another fleet is active. Launch
reprobes VRAM, RAM and pending R17 reservations, requiring the full native worker
count and the original per-card round-robin placement to fit. It tries the next
installed host if the first cannot fit. A held host-wide lock alone no longer
excludes a host. The running R16 fleet retains its original lock untouched.

Measured read-only snapshots:

| Host | Spare LIBERO capacity while fa_co_r_u_l is active | Limiting evidence |
|---|---:|---|
| timan107 | **3** latest snapshot (11 on the earlier snapshot) | GPU slots 3; RAM slots 10; 39 R17 slots reserved |
| timan1 GPU2 | **39** | 23,320 MiB pending for 44 R17 workers; RAM slots 181 |

The held H100 π0.5 jobs retain **39 timan107 workers**, H100 **R3/B1**, and
their original plans. From the latest timan107 snapshot, releasing only the
current H100 lane's 39 reservations gives **42 GPU slots / 49 RAM slots**;
per-card slots are `[5,5,5,7,5,5,5,5]`, sufficient for π0.5's
`[5,5,5,5,5,5,5,4]`. This is a conditional estimate, **not a launch admission**;
actual capacity is measured again at launch. See `CAPACITY_AFTER_CURRENT.json`
and `RUNNING_H100_WORKER_RESERVATION.json`. The timan1 alternative was not
selected or deployed.

`R15_REMOTE_DRY_RUN.json` proves the new mapping against timan107's **actual
deployed helper and native shell script**, without executing native main or
writing files. It confirms the only shell-source change is the fleet-lock
destination, with native script SHA256
`7b30e83e41c69beca3c5ad100fa1b16ef5cbe20b6ffca7a41dc52a3c3d08eb19`.
Native active legacy asset-protection discovery also passed read-only
(`ACTIVE_LEGACY_CHAINS.json`). No worker-file rsync is needed for this repair.

The two off units each remain 4,000 episodes across eight native cells; their
future level units retain the existing off-result materialization workflow.
Existing commands already opt into host placement for R15 and queued R16 items.
No command publisher or campaign builder was invoked.

## H100 order after the currently running fa_co_r_u_l

This is a replacement ordering for the coordinator to use, not a second queue
to run concurrently with ffk-h100fa. The running chain was not changed.

```text
fa_co_f_f_oc1
fa_pi_f_u_o
fa_pi_f_f_o
fa_co_r_f_l
ffk1015_dp_mt50_full_descent
fa_fw_f_u_l
fa_fw_f_f_l
fa_co_f_u_l
fa_co_f_f_l
fa_pi_f_u_l
fa_pi_f_f_l
ffk1015_fastwam_mt50_full_descent
```

For each NAME, the existing queue contract is unchanged:

```bash
cd /home/weiland/projects/openpi
NAME=fa_pi_f_u_o
bash "exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_${NAME}.sh"
bash "exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_${NAME}.sh"
```

The four π0.5 unit names are unchanged. Both off units precede both level
units. Level prepare still derives its cells from the completed off results.

## CPU verification

**21 tests passed**, including real native helper/WorkerAgent code with process
creation replaced by CPU fakes. Regressions cover the receipt-embedded import
false match, nested wrapper idempotence, real admission imports, another fleet
holding the old host lock, per-run exclusivity, the maintenance gate, scoped
worker cleanup, native argument/seed/init-pool preservation, R16 lock routing,
capacity rejection, existing continuation coverage and infra-error handling.

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m pytest -q \
  exp/offline_search/rounds/r17/filter_failed/test_timan1_gpu2.py \
  exp/offline_search/rounds/r17/filter_failed/test_resume.py \
  exp/offline_search/rounds/r17/filter_failed/test_worker_resilience.py \
  exp/offline_search/rounds/r17/filter_failed_knob/test_vla_workers.py \
  exp/offline_search/rounds/r17/filter_failed_knob/test_vla_shared.py \
  exp/offline_search/rounds/r17/filter_failed_knob/test_infra_resume.py \
  exp/offline_search/rounds/r17/filter_failed_knob/test_formal_a.py
```
