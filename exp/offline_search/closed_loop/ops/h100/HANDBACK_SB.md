# S-B handback: h100 → timan108

Implemented in **new files only** under this directory. Existing ops scripts
have no changes from this task. No git state-changing command, local GPU
execution, real-arm chain, or unrelated process/session/port modification was
performed. The other h100 checkout and existing venvs remain serving dependencies,
with code imported from the separate tree. Commands and full outputs below are
retained in `evidence/`; run instructions are in [RUNBOOK.md](RUNBOOK.md).

## Files and behavior

- `control.py`, `chain_h100.sh`: orchestration, completion/resume/purge, EV events,
  DONE/ERROR markers, GPU checks, local-chain ownership and cleanup.
- `assets.py`, `sync_assets.sh`: bounded per-arm dependency plan, source/installed
  SHA provenance, path relocation without editing arms.json, resumable daemon
  pulls with ≥30 GiB reserve. Plugin arms require prefits. Runtime library
  discovery also requires non-current `key_v0.npy` files, so these are included;
  tokens, images and query arrays are excluded unless explicitly referenced.
  CLIP's online image-tower weights are included and SHA checked separately.
- `node.py`, `server_cli.py`, `start_server.sh`, `stop_server.sh`: locked remote
  launch/stop, PID starttime and session ownership, durable completed-request
  deduplication. Intentional retries use a fresh launch ID.
- `setup_once.sh`, `run_arm.sh`, `probe.py`, `verify_worker.py`, `render_test.py`:
  isolated code/client island, official init pools, imports and four EGL slots.
  Existing remote `count.py`, `purge_exc.py`, `run_gtp_subset.py` are copied unchanged.
- `collect_h100.sh`, `abort_h100.sh`: stop the owned local chain before remote
  cleanup; SHA checked collection from both nodes using the unchanged summary
  code. Layout is `runs/<arm>/client/`, `server_<port>/`, `summary.json`.
- `test_h100.py`, `RUNBOOK.md`, this handback and `evidence/`.

Remote persistent paths: `/data/oscl_h100/` and
`/scratch/zixuans8/openpi_trace/`; tether staging uses
`/srv/local/zixuans8/oscl_sb_stage` on md0. No worker staging writes to root disk.

## Verification commands and outputs

All local Python commands use this prefix:

```bash
P=(taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= /home/weiland/projects/openpi/.venv/bin/python)
OPS=/home/weiland/projects/openpi/exp/offline_search/closed_loop/ops/h100
SMOKE=/home/weiland/trace_runs/os_closed_loop/r08_sb_smoke_20261001
```

`"${P[@]}" -m pytest "$OPS/test_h100.py" -q`
→ **16 passed**. Includes actual unprivileged rsync transfer/resume/SHA repair,
bounded dependencies and CLIP weights, path/fit metadata relocation, journal
deduplication and exception purge, manifest precedence, worker cap, foreign and
completed launch refusal, natural completion, and mocked chain retry/collection
failure markers. Full output: [tests](evidence/oscl_sb_tests.log).

`bash "$OPS/setup_once.sh"`
→ `CODE_SNAPSHOT files=2049 bytes=18520167` in the preceding snapshot;
`INSTALL_OK ... dest=/data/oscl_h100/openpi` and
`INSTALL_OK ... dest=/scratch/zixuans8/openpi_trace` verify every installed file.
The final snapshot's exact count/bytes are in [setup](evidence/oscl_sb_setup.log).

`"${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control verify`
→ all three safetensors rollups **match=true**:

| h100 checkpoint under `/home/exouser/ckpt/` | SHA256 rollup |
|---|---|
| pi05_libero_pytorch | a3c115c587eeeb4a4f312f8442886401c08987d7aa7957e22333ad742a2c55c6 |
| n15_libero_spatial | 6c61469c8591b697254f3182a0fae298c79b159750b6d805bbb4019ac097851d |
| n15_libero_10 | 4133afb655297ce9cdcfe24d1c1af09de847e4f3b0d72b3a773c87624e1c90a1 |

This is the exact sorted-basename/file-SHA rollup rule in
`r08_fits/checkpoint_sha.json`. No checkpoint is missing; no copy is needed.
Full output: [checkpoint/import verification](evidence/oscl_sb_verify.log).

Local comparison commands were `"${P[@]}" "$OPS/probe.py" pi05`, and the same
CPU/thread/CUDA restrictions with the GR00T interpreter and its prescribed
PYTHONPATH for `probe.py groot`. Outputs:

| Environment | Both hosts' torch | Both transformers | Both numpy |
|---|---|---|---|
| pi0.5 | 2.7.1+cu126 | 4.53.2 | 1.26.4 |
| GR00T | 2.5.1+cu124 | 4.51.3 | 1.26.4 |

Python is **3.11.16 h100 vs 3.11.15 local**. Other reported packages and the
SigLIP patch SHA match within each environment. Imports resolve to
`/data/oscl_h100/openpi/src/openpi`, with GR00T from `/home/exouser/gr00t_n15`.
`git -C <GR00T tree> rev-parse --short HEAD` on both hosts → `4af2b62`.
Outputs: [local pi0.5](evidence/oscl_sb_local_pi05.log),
[local GR00T](evidence/oscl_sb_local_groot.log), [h100](evidence/oscl_sb_verify.log).

`"${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control verify-worker`
→ sim Python **3.8.20**, client imports from the island's `src/openpi_client`
(rather than the sim environment's other-line editable client),
`WORKER_IMPORTS_OK`; both init pools rehash to **500** official pairs:
spatial `0eeece46a08b958efe7b7db4e6b13d3269b0433be4e20fbae3c0f352bc3aca9c`,
l10 `52457a37eb26f9511b708f2e2efb2c175d0a1f8665ba8d57180556693c1ee756`.
Output: [worker](evidence/oscl_sb_worker.log).

`"${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control render`
→ each physical GPU 0–3 rendered a nonconstant **256×256×3** image with
`NVIDIA RTX A5000/PCIe/SSE2`, then `RENDER_OK GPUs=0,1,2,3`.
Output: [render](evidence/oscl_sb_render.log).

`bash "$OPS/sync_assets.sh" "$SMOKE" r8_pi05_spatial_50_A`
→ `ASSETS_OK files=27 bytes=4621415750`, worker config `INSTALL_OK`, `SYNC_OK`.
**4.304 GiB** copied, preserving the native PKL/current metadata identity.
Output: [sync](evidence/oscl_sb_sync.log).

`"${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control plan /home/weiland/trace_runs/os_closed_loop/r08_abl r8abl_clip_p_l10_50`
→ `PLAN files=28 bytes=12352907863 GiB=11.505`, including the 605143284-byte
CLIP checkpoint. This was a **dry plan**, without copying or running that arm.
Output: [CLIP dependency plan](evidence/oscl_sb_clip_plan.log).

## Permitted live smoke and cleanup

Created a separate root containing an unchanged copy of the existing
`r08_main` pi0.5 A row and manifest `[[0,0],[0,1],[1,0],[1,1]]`.

```bash
PORTS=23270 WPS=2 MAX_ATTEMPTS=1 POLL_SECONDS=10 \
OSCL_MANIFEST="$SMOKE/smoke_manifest.json" \
  bash "$OPS/chain_h100.sh" "$SMOKE" r8_pi05_spatial_50_A
```

Output: `complete=4 success=4 rows=4`, `purged=0`, `COLLECT_OK`, `ARM_DONE`,
`CHAIN_DONE`. Every UID has attempt=1, accepted=true, status=done and error=null.
Episode wall durations: task0/init0 **10.739 s**, task0/init1 **10.698 s**,
task1/init0 **7.107 s**, task1/init1 **6.778 s**; median **8.9025 s**.
**74 FULL_HIT decisions**, zero extra episode attempts. Overall chain wall time
was about **108 s**, including roughly 66 s server boot and collection/polling.
Command output and journal-derived metrics:
[smoke](evidence/oscl_sb_smoke.log), [metrics](evidence/oscl_sb_metrics.log).

An additional idempotence probe raced with successful chain completion and
briefly relaunched the **same port/arm**, without any additional worker episodes.
Stopped it using `abort_h100.sh`, then fixed deduplication across terminal
lifecycle states. Unit tests cover the race. After installing the fix, replaying
the completed server and driver requests returned **ALREADY_FINISHED** on both
hosts and created no processes. The final recollection retains rotated logs and
the extra startup record, so the server summary has two startups for the one
port. This is verification history, not an additional episode replicate.
[Abort](evidence/oscl_sb_abort.log), [terminal replay](evidence/oscl_sb_terminal_duplicate.log).

`bash "$OPS/collect_h100.sh" "$SMOKE" r8_pi05_spatial_50_A`
→ `COLLECT_OK ... complete=4 success=4 sr=1.0` using the existing collector.
All client/server files and both summaries are in the smoke root.
[Collection](evidence/oscl_sb_collect.log).

Cleanup commands were read-only `tmux ls`, `ss -ltnH 'sport = :23270'`, targeted
`ps ... | grep` searches for owned island/server/worker commands, `nvidia-smi`
and `df`. Outputs: **no owned tmux sessions/process matches; no :23270 listener**;
h100 free GPU **81081 MiB**, `/data` **88 GB free**, root **26 GB free**;
timan108 GPUs 0–3 each **24247 MiB free**, scratch **9640 GB free**.
Local `ss -ltnp 'sport >= :23196 and sport <= :23199'` shows only the untouched
preexisting :23198/:23199 listeners; the temporary rsync daemon is gone.
[h100](evidence/oscl_sb_cleanup_h100.log), [timan108](evidence/oscl_sb_cleanup_t108.log),
[local](evidence/oscl_sb_cleanup_local.log).

## Open blockers and unverified cases

The whole `r08_abl` dry plan stopped at a missing GR00T CLIP prefit. At the
last recorded scan (44 arms), four files were absent:
`fits/r8abl_clip_g_{l10,sp}_{50,500}.pkl`. Exact paths and the check output are in
[missing fits](evidence/oscl_sb_missing.log); initial failure is in
[round plan](evidence/oscl_sb_r08_plan.log). Stop this sync sub-step until the
CLIP owner completes those artifacts. If encoded caches are ready, the owner's
resumable CPU fitting entry is `python -m exp.offline_search.rounds.r08.abl.clip.prepare --fit`
with the owner's CPU allocation; otherwise encoding must finish first. Then
repeat `plan` and `sync_assets.sh` for the coordinator's chosen arm group.

Full-round dependency bytes/disk feasibility are unverified while those fits
are missing. The sync rejects insufficient space before copying the plan.
Select smaller arm groups if necessary; nothing is automatically deleted.
GR00T, stock, full-model and CLIP live serving, a 32-worker load, and deliberate
server-crash/abort under active workers were **not live tested**. They have import,
static or unit coverage as stated above. Scientific cross-hardware equivalence
and long-run reliability remain coordinator checks. No real round was launched.

## Robustness fixes

S-B2, 2026-10-01. This section supersedes the earlier collection-failure and
30-second polling descriptions. S-B2 work and verification were local only;
the earlier remote smoke/setup evidence belongs to S-B. No S-B2 tether,
setup, sync, chain, smoke or GPU experiment was performed against h100/timan108.
The coordinator's existing Python process retains its previously imported
controller and needs a restart to load this patch.

Changed files: `control.py`, `node.py`, `run_arm.sh`, `test_h100.py`, new
`test_robustness.py`, `RUNBOOK.md`, this handback, and new
`evidence/oscl_sb2_{tests,static}.log`. All are in this h100 ops directory.
Completion counting, the exception purge implementation, collection/SHA
extraction layout, arm selection and serving arguments are retained. Existing
marker names are retained; collection pending is expressed in ERROR contents
and an EV event. The retained regression suite covers these unchanged paths.

Line numbers refer to the final S-B2 files. Every test below was included in
the exact **92 passed** command/output recorded after the table.

| Fix | Changed lines and behavior | Local regression evidence |
|---|---|---|
| H1: transport | `control.py:33–103`: up to four idempotent tether attempts, backoff 2/4/8 seconds, on rc 69/70/75/255, timeout text or TimeoutExpired. RPC specs are SHA checked on first use and cached by digest thereafter. timan108 staging mkdir occurs once, with one combined mkdir/cp/rm exec per transfer, safe to replay after stage removal. | `test_tether_retries_idempotent_commands` (42 cases), `test_nontransient_or_local_errors_do_not_retry`, `test_tether_timeout_retry_is_bounded`, `test_rpc_reuses_content_addressed_remote_spec`, `test_timan_push_merges_finalize_and_tolerates_duplicate_exec`. |
| H1: polls | `control.py:27–28,415–418,478–498,544–580`: default 60 seconds; failed/malformed status reads are unknown, default limit ten consecutive failed polls, reset after a wholly successful poll. Each status read has one 60-second transport attempt; subsequent polls supply retry/backoff. Known dead servers retain the restart path. Steady polling needs one exec per server plus one driver exec, without spec pushes. | `test_failed_poll_is_unknown_and_recovery_resets_counter`, `test_ten_consecutive_failed_polls_stop_with_error` (server and driver cases), `test_boot_status_failure_does_not_abort_immediately`; default sleep and no early stop/purge/count are asserted. |
| M1 | `node.py:105–118,219–224,288–337`; `run_arm.sh:10–13`; `control.py:437,572–580,647`: supervise bash directly, hold fleet.lock inside the driver script, and wait up to 180 seconds for fleet release plus worker exit before stop returns. Legacy outer-flock launches and missing PID/owner receipts cannot bypass the fleet wait. Natural completion and server-death paths confirm stop before purge/count. Driver-stop RPCs allow 590 seconds. | `test_driver_launch_supervises_bash_without_outer_flock`, `test_run_arm_lock_refuses_before_any_driver_work`, `test_legacy_driver_stop_waits_for_fleet_lock_release`, `test_fleet_stop_wait_is_bounded`, `test_missing_driver_receipt_still_requires_fleet_exit`; `mock_chain` asserts purge/count follow confirmed stop. |
| M2 | `control.py:337–347,590–621`; `node.py:322–333`: collect failure emits COLLECT_PENDING, leaves arm ERROR/no DONE and proceeds to the next arm. Pending collection leaves CHAIN.ERROR/no CHAIN.DONE; final CHAIN_DONE includes the pending list. Pull retries catch TimeoutExpired within the existing eight-attempt budget. Slow server stop escalates TERM to KILL of the recorded PID after rechecking starttime. | `test_collection_pending_continues_to_next_arm`, `test_collection_completion_recheck_still_required`, `test_collect_pull_retries_timeout_before_sha_and_extraction`, `test_collect_pull_timeout_exhausts_eight_attempts`, `test_server_stop_escalates_only_recorded_owned_pid` (including recycled PID), updated `test_chain_resume_markers_and_collection_failure`. |
| M3 | `node.py:77–102,219,337`: only this UID's worker_entry argv referencing the island counts, with path boundaries. Poll for up to 120 seconds before WORKERS_BUSY. | `test_island_workers_ignore_other_users_and_other_trees`, `test_worker_busy_wait_recovers_and_has_120_second_bound`; fixtures include another UID, another tree and an island-prefix sibling. |
| M4 | `control.py:459–469,474–476,524–526`; `node.py:121–143` and CLI dispatch: validate the entire requested arm list before remote calls/arm launches. Compute the remote tree manifest once at chain start and retain digest/count/bytes/rule in every h100_launch.json rewrite. | `test_all_requested_arms_checked_before_first_remote_call`, `test_code_tree_digest_recorded_once_for_every_arm`, `test_remote_code_manifest_tracks_content_and_ignores_bytecode`. Actual remote digest is unverified until deployment. |
| M5 | `control.py:169–190,276–285,456–469`: a shared local h100 lock serializes setup, sync and chains. Existing run-root locks are also held/checked, so maintenance refuses the currently running legacy controller before remote mutation. Plan reads/validation occur under the same lock to avoid stale snapshots during concurrent sync. Dry plan remains available. | `test_maintenance_refuses_active_global_or_legacy_chain` (setup/sync × new/legacy lock), `test_chain_maintenance_serialization_and_plan_remains_read_only`, `test_chain_reads_synced_plan_under_maintenance_lock`; actual local flock contention and partial-acquisition release are tested. |

### Commands and observed outputs

From `/home/weiland/projects/openpi`:

```bash
taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
  .venv/bin/python -m pytest \
  exp/offline_search/closed_loop/ops/h100/test_h100.py \
  exp/offline_search/closed_loop/ops/h100/test_robustness.py -q
```

Observed output, preserved in [S-B2 tests](evidence/oscl_sb2_tests.log):

```text
........................................................................ [ 78%]
....................                                                     [100%]
92 passed in 1.69s
Exit code: 0
```

New regressions use local mocks/fault injection. The retained S-B suite also
includes its owned local rsync-daemon regression and local purge/supervisor
subprocess checks. No test invokes tether or a policy/LIBERO workload.

This command parses the four changed Python files, checks every ops shell
script, and checks runbook commands and deployment Python without execution:

```bash
taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" .venv/bin/python - <<'PY'
import ast
from pathlib import Path
import re
import subprocess
ops = Path('exp/offline_search/closed_loop/ops/h100')
lines = []
for name in ('control.py', 'node.py', 'test_h100.py', 'test_robustness.py'):
    ast.parse((ops / name).read_text(), filename=name)
lines.append('PYTHON_SYNTAX_OK files=4')
scripts = sorted(ops.glob('*.sh'))
for path in scripts:
    subprocess.run(['bash', '-n', str(path)], check=True)
lines.append(f'BASH_SYNTAX_OK files={len(scripts)}')
blocks = re.findall(r'^```bash\n(.*?)^```', (ops / 'RUNBOOK.md').read_text(), re.M | re.S)
for block in blocks:
    subprocess.run(['bash', '-n'], input=block, text=True, check=True)
lines.append(f'RUNBOOK_BASH_SYNTAX_OK blocks={len(blocks)}')
embedded = re.findall(r"<<'PY'\n(.*?)^PY$", (ops / 'RUNBOOK.md').read_text(), re.M | re.S)
for block in embedded:
    ast.parse(block, filename='RUNBOOK deployment heredoc')
lines.append(f'RUNBOOK_DEPLOY_PYTHON_SYNTAX_OK blocks={len(embedded)}')
output = '\n'.join(lines) + '\n'
(ops / 'evidence/oscl_sb2_static.log').write_text(output)
print(output, end='')
PY
```

Observed output, exit code 0, preserved in
[S-B2 static checks](evidence/oscl_sb2_static.log):

```text
PYTHON_SYNTAX_OK files=4
BASH_SYNTAX_OK files=8
RUNBOOK_BASH_SYNTAX_OK blocks=6
RUNBOOK_DEPLOY_PYTHON_SYNTAX_OK blocks=1
```

### Exact coordinator deployment and restart

**Stop the running local chain first.** At the intended arm boundary, retain
the current arm name and stop only the owned controller:

```bash
cd /home/weiland/projects/openpi
OPS=$PWD/exp/offline_search/closed_loop/ops/h100
RUN=/home/weiland/trace_runs/os_closed_loop/r08_abl
P=(taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "$PWD/.venv/bin/python")
SB2_ARM=$(cat "$RUN/state/current")
bash "$OPS/abort_h100.sh" "$RUN"
```

Execute the complete `SB2_ARM="$SB2_ARM" "${P[@]}" - "$RUN"` guarded
Python heredoc in [RUNBOOK: S-B2 targeted deployment](RUNBOOK.md#s-b2-targeted-deployment-by-the-coordinator).
It acquires the shared/legacy locks, copies standalone node helpers first,
repeats the captured arm's owned driver/server stops using the fixed helper,
requires FLEET_IDLE, copies the remaining files and SHA verifies every
destination. It computes the actual remote manifest as the final deployment
check. A live legacy driver must exit before script/source copies proceed.
Stop deployment on any failed check. No such remote step was executed by S-B2.

| Local source | h100 destinations | timan108 destinations |
|---|---|---|
| `node.py` | `/data/oscl_h100/node.py`, `/data/oscl_h100/openpi/exp/offline_search/closed_loop/ops/h100/node.py` | `/scratch/zixuans8/openpi_trace/os_cl/node.py`, `/scratch/zixuans8/openpi_trace/exp/offline_search/closed_loop/ops/h100/node.py` |
| `control.py` | `/data/oscl_h100/openpi/exp/offline_search/closed_loop/ops/h100/control.py` | `/scratch/zixuans8/openpi_trace/exp/offline_search/closed_loop/ops/h100/control.py` |
| `run_arm.sh` | No copy required | `/scratch/zixuans8/openpi_trace/os_cl/run_arm.sh` |

Tests/docs/evidence remain local. Expected deployment output:
`FLEET_IDLE`, `SB2_SHA_OK node=h100 files=3`,
`SB2_SHA_OK node=timan108 files=4`, and SB2_CODE_MANIFEST containing the
actual digest. These are expected, not observed S-B2 remote outputs. No setup
or asset sync is needed for this code-only patch. Copies target the owned
topology trees; the unrelated h100 checkout and venvs are unaffected.

Restart a fresh local process with the coordinator's same arm list/settings,
preserving journals and DONE markers:

```bash
PORTS=23210,23211,23212,23213 WPS=8 MAX_ATTEMPTS=3 POLL_SECONDS=60 \
  bash "$OPS/chain_h100.sh" "$RUN" "${ARMS[@]}"
```

Resume that same list to retry COLLECT_PENDING arms; no DONE is created until
the retained completion-count recheck succeeds after collection.

### Unverified behavior and remaining limits

Remote deployment, actual remote SHA, real network outages, TERM/KILL under
server load, legacy driver teardown under load, and a seven-hour 32-worker run
are not live verified by S-B2. Tests establish local control flow, bounded
waits, digests and ownership checks. Retry budgets remain finite: a persistent
outage, a driver/fleet that never exits, or a server surviving SIGKILL remains
a blocker. Driver/fleet failure prevents purge/count. Collection retains eight
attempts with 590-second pull deadlines, so persistent pull outages can delay
the next arm considerably before COLLECT_PENDING. Remote verification and
ongoing round monitoring remain the coordinator's work.

## Second fleet

S-B3, 2026-10-01. Added concurrent chain B using `WORKER_HOST=timan107`, new
island `/scratch/zixuans8/openpi_trace_h100`, GPUs 0–7 and a 64-worker cap.
The default remains timan108, its existing island, GPUs 0–3 and 40 workers.
All implementation files are in this h100 ops directory. This section
supersedes S-B2's chain-versus-chain serialization/deployment instructions
for adding B. **Chain A does not need to pause.**

Changed: `control.py`, `node.py`, `run_arm.sh`, `verify_worker.py`,
`render_test.py`, `test_h100.py`, `test_robustness.py`, `RUNBOOK.md`, this handback.
New: `fleet.py`, `prepare_run.py`, `fixtures/node_sb2.txt`, and
`evidence/oscl_sb3_*.log`. `setup_once.sh`/other wrapper files are unchanged;
their existing argument forwarding supports `--worker-only` and `--concurrent`.
No git state-changing command, local GPU work, real chain, policy smoke,
h100/timan108 deployment or modification of another task's process/session
was performed. Only timan107's new island and its dedicated `/tmp` staging
were installed into. The old timan107 islands/checkouts were read-only inputs
for interpreter dependencies and were not installed into.

### Behavior and regression evidence

| Behavior | Evidence in the 130-test command below |
|---|---|
| Validated host selection, fixed islands, GPU cycles, caps 40/64; shell and node both enforce cap. New-island node fills an omitted host and refuses an override selecting the old island. | `test_fleet_selection_paths_and_caps`, `test_fleet_default_and_invalid_host`, `test_run_arm_uses_host_gpu_cycle_and_cap`, `test_new_island_node_enforces_64_worker_cap`, `test_t107_node_refuses_payload_that_would_select_old_island`. |
| Two chains use a shared maintenance gate, per-host locks and separate root locks. Full maintenance remains exclusive and honors M5's original global/legacy locks. Scoped maintenance refuses the active host; authenticated old A can coexist with B without releasing its locks. | `test_two_fleets_run_concurrently_and_maintenance_stays_exclusive`, `test_live_legacy_A_allows_B_but_refuses_same_host_and_tag`, `test_legacy_lock_holder_authentication_and_fail_closed`, retained S-B2 maintenance tests. Actual A was recognized in `oscl_sb3_static.log` and worker-only setup succeeded while A ran. |
| Concurrent sync verifies/reuses existing assets without rsync writing them; active conflicts/missing files and existing differing bytes/symlinks fail before transfer. New shared assets retain inode/mtime across retries. | `test_concurrent_sync_B_while_A_uses_shared_assets`, `test_concurrent_sync_conflicting_active_plan_refuses_before_remote_mutation`, `test_concurrent_pull_refuses_without_writing_existing_files`, `test_concurrent_pull_reuses_exact_existing_files_and_replays`. |
| Only `pull-new` uses h100's separately deployed `node_sb3.py`. Existing RPCs keep the original helper. Worker setup sends nothing to h100/timan108. Tether uses allowed staging roots for both fleets. | `test_concurrent_sync_uses_only_versioned_h100_helper`, `test_worker_only_setup_never_contacts_h100_or_other_fleet`, `test_rpc_host_island_and_allowed_staging`. |
| B's driver, polling, purge/count and recorded output paths use timan107; old saved launches still route abort/collection to timan108. Foreign occupied ports remain refused. | `test_chain_B_contacts_only_t107_and_h100`, `test_wrong_synced_fleet_refused_before_remote_calls`, `test_old_launch_receipts_route_abort_collection_to_t108`, `test_another_chains_port_is_refused_before_launch`. |
| Old start (driver and server), status, stop, install, pull, archive, checkpoints, manifest, supervise and CLI help accept the old inputs and produce identical captured stdout, helper calls and file bytes against the frozen S-B2 implementation. New `pull-new` is hidden from the legacy parser/help. | 11 cases of `test_old_rpc_cli_is_byte_compatible_with_frozen_sb2`. Baseline SHA256 `632b0caf5d1255ce25d3faab718ca6cde7e36c7dbefbaea133e245be107b9265`, also printed in static evidence. This is mocked compatibility evidence; A's helper is not replaced in the deployment procedure. |
| Preparing B copies selected config/arm rows into fresh output paths, writes the official 500-pair Cartesian manifest, preserves external fit inputs and leaves A unchanged. | `test_prepare_second_run_preserves_inputs_and_uses_fresh_outputs`; real B creation remains coordinator work because the chosen arm list was not supplied. |

The retained S-B/S-B2 tests still cover retries, status unknown/failure limits,
confirmed fleet teardown before purge/count, collection pending markers,
ownership/recycled PIDs, standard mode, manifests and serving-tree provenance.
New chains hold `h100_chain.lock.gate` shared and
`h100_fleet_<host>.lock`/their root lock exclusive. Full maintenance holds the
gate/global/host/legacy locks exclusive. `/proc/locks` holders must match a live
controller's PID/starttime/argv receipt; unknown holders fail closed. Per-host
root registries also let the audit find new roots outside the usual parent.
The physical fleet lock remains inside each island's `run_arm.sh`.

### Commands and observed outputs

From `/home/weiland/projects/openpi`, all local Python uses:

```bash
OPS=$PWD/exp/offline_search/closed_loop/ops/h100
P=(taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "$PWD/.venv/bin/python")
"${P[@]}" -m pytest "$OPS/test_h100.py" "$OPS/test_robustness.py" -q
```

Observed, exit 0 ([tests](evidence/oscl_sb3_tests.log)):

```text
........................................................................ [ 55%]
..........................................................               [100%]
130 passed in 2.05s
```

No test runs tether, policy serving or LIBERO episodes. The retained local
rsync test uses a free port in 23180–23197 and only its own daemon; selecting
a free port avoids TIME_WAIT/another listener without stopping anything.

```bash
WORKER_HOST=timan107 bash "$OPS/setup_once.sh" --worker-only
```

Observed, exit 0 ([final setup](evidence/oscl_sb3_setup_final.log)):

```text
CODE_SNAPSHOT files=2052 bytes=18592083
INSTALL_OK files=2074 dest=/scratch/zixuans8/openpi_trace_h100
```

The bundle installer checks every installed file against its source SHA. This
snapshot includes the client under island `src/openpi_client`, both init pools,
new island apool paths, copied/relocated subset helper, and LIBERO config. The
new island's standalone `node.py` selects that island; its driver shell refuses
selecting the old island even if the host environment is omitted/incorrect.

```bash
WORKER_HOST=timan107 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control verify-worker
```

Observed, exit 0 ([worker](evidence/oscl_sb3_worker.log)):

```text
3.8.20 (default, Oct  3 2024, 15:24:27)
[GCC 11.2.0]
/scratch/zixuans8/openpi_trace_h100/src/openpi_client/__init__.py
/scratch/zixuans8/openpi_trace_h100/examples/libero/main.py
APOOL_OK libero_spatial total=500 sha=0eeece46a08b958efe7b7db4e6b13d3269b0433be4e20fbae3c0f352bc3aca9c
APOOL_OK libero_10 total=500 sha=52457a37eb26f9511b708f2e2efb2c175d0a1f8665ba8d57180556693c1ee756
WORKER_IMPORTS_OK
```

Both digests exactly match the earlier timan108 attestations in this handback;
the verifier rehashes contents and asserts these frozen digests and 500 totals.
No new timan108 verification RPC was needed/performed.

```bash
WORKER_HOST=timan107 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control render
```

Observed, exit 0 ([all-eight render](evidence/oscl_sb3_render_final.log)):
eight JSON rows, one per GPU 0–7, each with matching `egl_device`,
`renderer="NVIDIA GeForce GTX 1080/PCIe/SSE2"`,
`image_shape=[256,256,3]`; the helper asserts nonzero image standard deviation.
Final output:

```text
RENDER_OK GPUs=0,1,2,3,4,5,6,7
```

The initial helper segfaulted at PyOpenGL `glGetString` on GPU 0 **after** its
image assertions. Diagnostic command:

```bash
timeout 120 tether exec timan107 -- env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
LIBERO_CONFIG_PATH=/scratch/zixuans8/openpi_trace_h100/os_cl/libero \
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 PYTHONFAULTHANDLER=1 \
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu /scratch/zixuans8/libero_sim/bin/python \
/scratch/zixuans8/openpi_trace_h100/os_cl/render_test.py --gpu 0
```

Observed ([diagnostic](evidence/oscl_sb3_egl_gpu0.log)):
`Fatal Python error: Segmentation fault`, stack
`OpenGL/platform/baseplatform.py:415 in __call__` → `render_test.py:38`.
An explicit PyOpenGL EGL platform did not fix that query
([retry](evidence/oscl_sb3_render_egl.log)). The final helper resolves
`glGetString` using `eglGetProcAddress`; all eight slots then pass. No driver,
system library, shared simulator, old island or cached asset was patched.

Static/doc/ownership check command (not a deployment or chain):

```bash
"${P[@]}" - <<'PY'
import ast
from pathlib import Path
import re
import subprocess
from exp.offline_search.closed_loop.ops.h100 import control as c
ops = c.HERE
files = ['control.py', 'node.py', 'fleet.py', 'prepare_run.py', 'render_test.py',
         'verify_worker.py', 'test_h100.py', 'test_robustness.py']
for name in files:
    ast.parse((ops / name).read_text(), filename=name)
print('PYTHON_SYNTAX_OK files=' + str(len(files)))
scripts = list(ops.glob('*.sh'))
for path in scripts:
    subprocess.run(['bash', '-n', str(path)], check=True)
print('BASH_SYNTAX_OK files=' + str(len(scripts)))
blocks = re.findall(r'^```bash\n(.*?)^```', (ops / 'RUNBOOK.md').read_text(), re.M | re.S)
count = 0
for block in blocks:
    subprocess.run(['bash', '-n'], input=block, text=True, check=True)
    for inline in re.findall(r"<<'PY'\n(.*?)\nPY", block, re.S):
        ast.parse(inline)
        count += 1
print(f'RUNBOOK_SYNTAX_OK bash_blocks={len(blocks)} inline_python={count}')
print('ACTIVE_LOCK_OWNERS ' + repr(c.legacy_chains()))
print('FROZEN_SB2_NODE_SHA256 ' + c.sha(ops / 'fixtures/node_sb2.txt'))
PY
```

Observed, exit 0 ([static](evidence/oscl_sb3_static.log)):

```text
PYTHON_SYNTAX_OK files=8
BASH_SYNTAX_OK files=8
RUNBOOK_SYNTAX_OK bash_blocks=11 inline_python=3
ACTIVE_LOCK_OWNERS [{'root': PosixPath('/home/weiland/trace_runs/os_closed_loop/r08_abl'), 'worker_host': 'timan108'}]
FROZEN_SB2_NODE_SHA256 632b0caf5d1255ce25d3faab718ca6cde7e36c7dbefbaea133e245be107b9265
```

Final installed-helper SHA and idle/disk checks:

```bash
timeout 90 tether exec timan107 -- bash -c 'sha256sum /scratch/zixuans8/openpi_trace_h100/os_cl/node.py /scratch/zixuans8/openpi_trace_h100/os_cl/run_arm.sh /scratch/zixuans8/openpi_trace_h100/os_cl/render_test.py /scratch/zixuans8/openpi_trace_h100/os_cl/verify_worker.py; df -h /scratch /tmp; nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader; tmux -L oscl ls; ps -u zixuans8 -o pid,args | grep -E "[w]orker_entry.*openpi_trace_h100|[r]un_gtp.*openpi_trace_h100|[s]upervise.*openpi_trace_h100"; true'
sha256sum "$OPS/node.py" "$OPS/run_arm.sh" "$OPS/render_test.py" "$OPS/verify_worker.py"
```

Observed ([receipt](evidence/oscl_sb3_t107_receipt.log)): hashes on both sides
match, respectively
`19ee875a9c0a4efea8971acba20ae2540f20787caaac41d4131a6ad1ac0c7995`,
`264502792072dd846582d8623d092dc5f3b700b4cfc5195a8350dce4eaf7d680`,
`3196a08b200d77db027c980b411569d0253d53c822b98c3e9a03daf6f83d3f1e`,
`20b2206c46179547b0fb091ff5fa7de05fc272222f86eb1a2daeac41c96dc5be`.
Scratch and `/tmp` use the same 212 GiB root filesystem with **32 GiB free**;
GPUs 0–7 each report **2 MiB** used, no `oscl` tmux server and no matching
new-island driver/worker/supervisor processes. No process/session was stopped.

Final documentation/atomic-rename command:
`"${P[@]}" < "$OPS/evidence/oscl_sb3_docs_check.txt"`.
Observed, exit 0 ([doc/rename check](evidence/oscl_sb3_docs.log)):
`DOC_SYNTAX_OK file=RUNBOOK.md bash_blocks=11 inline_python=3`,
`DOC_SYNTAX_OK file=HANDBACK_SB.md bash_blocks=14 inline_python=4`,
two identical `SB3_H100_SHA_OK sha=19ee875a9c0a4efea8971acba20ae2540f20787caaac41d4131a6ad1ac0c7995`
lines and `ATOMIC_DEPLOY_LOCAL_REPLAY_OK executions=2`.
The replay uses only locally created temporary files; it is not an h100 deployment.

### Exact coordinator follow-through (not executed by S-B3)

The complete commands and expected outputs are also in
[RUNBOOK.md, Second fleet](RUNBOOK.md#second-fleet-s-b3-concurrent-chain-b-on-timan107).
Use the `OPS`/`P` definitions above and choose the actual arm list. Create B
before deployment/sync; `prepare_run` requires a fresh root. Reuse prepared B
directly for subsequent resume/sync rather than copying A's state.

```bash
A=/home/weiland/trace_runs/os_closed_loop/r08_abl
B=/home/weiland/trace_runs/os_closed_loop/r08_abl_t107
ARMS=(r8abl_onlynp_p_l10_50) # replace with coordinator-selected arms
"${P[@]}" -m exp.offline_search.closed_loop.ops.h100.prepare_run "$A" "$B" "${ARMS[@]}"
WORKER_HOST=timan107 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control plan "$B" "${ARMS[@]}"

# New helper only. No A pause/restart and no h100 application/timan108 writes.
WORKER_HOST=timan107 "${P[@]}" - "$B" <<'PY'
from pathlib import Path
import shlex
import sys
from exp.offline_search.closed_loop.ops.h100 import control as c
with c.fleet_lock(Path(sys.argv[1])):
    source = c.HERE / 'node.py'
    digest = c.sha(source)
    stage = c.BASE / ('node_sb3_' + digest + '.new')
    target = c.BASE / 'node_sb3.py'
    c.push('h100', source, stage)
    qs, qt = shlex.quote(str(stage)), shlex.quote(str(target))
    check_stage = f'test "$(sha256sum {qs} | cut -d " " -f 1)" = {digest}'
    check_target = f'test "$(sha256sum {qt} | cut -d " " -f 1)" = {digest}'
    script = (f'if test -f {qs}; then {check_stage} && mv -T {qs} {qt}; fi && '
              f'{check_target} && echo SB3_H100_SHA_OK sha={digest}')
    print(c.remote('h100', ['bash', '-c', script]))
PY

# Already done by S-B3; refresh only while B is idle if local runtime code changed.
WORKER_HOST=timan107 bash "$OPS/setup_once.sh" --worker-only
WORKER_HOST=timan107 SYNC_PORT=23197 bash "$OPS/sync_assets.sh" --concurrent "$B" "${ARMS[@]}"

# Coordinator smoke only, in a separate root; no smoke command ran during S-B3.
SMOKE=/home/weiland/trace_runs/os_closed_loop/r08_abl_t107_smoke
"${P[@]}" -m exp.offline_search.closed_loop.ops.h100.prepare_run "$B" "$SMOKE" "${ARMS[0]}"
"${P[@]}" - "$SMOKE/smoke4.json" <<'PY'
import json
from pathlib import Path
import sys
Path(sys.argv[1]).write_text(json.dumps([[0,0], [0,1], [1,0], [1,1]]) + '\n')
PY
WORKER_HOST=timan107 SYNC_PORT=23197 bash "$OPS/sync_assets.sh" --concurrent "$SMOKE" "${ARMS[0]}"
WORKER_HOST=timan107 PORTS=23220 WPS=2 MAX_ATTEMPTS=1 POLL_SECONDS=10 \
OSCL_MANIFEST="$SMOKE/smoke4.json" bash "$OPS/chain_h100.sh" "$SMOKE" "${ARMS[0]}"

# Restore B's idle worker configs after smoke, then use a NEW coordinator session.
WORKER_HOST=timan107 SYNC_PORT=23197 bash "$OPS/sync_assets.sh" --concurrent "$B" "${ARMS[@]}"
WORKER_HOST=timan107 PORTS=23220,23221,23222,23223,23224,23225,23226,23227,23228,23229 \
WPS=4 MAX_ATTEMPTS=3 POLL_SECONDS=60 OSCL_MANIFEST="$B/eval500.json" \
bash "$OPS/chain_h100.sh" "$B" "${ARMS[@]}"
```

Expected (not observed here): `RUN_PREPARED ... pairs=500`, `PLAN`,
`SB3_H100_SHA_OK`, `ASSETS_NEW_OK added=<N> reused=<N>`, worker `INSTALL_OK`,
`SYNC_OK`. Smoke: `expect=4`, `SERVERS_READY`, `complete=4`, `COLLECT_OK`,
`ARM_DONE`, `CHAIN_DONE`, manifest-specific DONE. Real B: `expect=500`,
`workers=40`, ports 23220–23229, driver under the new timan107 island,
`h100_launch.json.worker_host=timan107`, summaries/journals under B. Keep MPS
at `/tmp/nvidia-mps`, A's ports 23210–23214 and all existing tmux sessions
unchanged. Ten ports × WPS=4 is a conservative 40 workers over eight GPUs;
any changed WPS must still total ≤64. Use the saved B launch host for recovery;
explicit `WORKER_HOST=timan107` is advisable for manual collection before a
launch receipt exists.

### Remaining risks and unverified work

There is **no remaining timan107 permission/EGL blocker** in the allowed setup,
import, SHA and tiny-render checks. Actual new h100 helper deployment and asset
sync, coordinator-selected B creation, policy smoke, concurrent live episodes,
throughput, larger worker fleets, GR00T/CLIP/full-model serving and scientific
cross-hardware equivalence remain unverified. The coordinator retains all real
run/monitoring work. Existing h100 GPU/RAM/disk guards may refuse the selected
ten-port arm; no resource limit is bypassed and h100 keeps ≥30 GiB free.

Concurrent sync intentionally refuses *all* existing mismatching target bytes,
even unprotected files; interrupted new transfers can leave partial files that
need ordinary sync after both chains stop. Missing active-plan files, unknown
lock holders, same host/root/tag and occupied ports refuse conservatively.
No attempt was made to repair another chain's assets or release its locks.
The h100 serving tree is unchanged: B records its actual old application-tree
digest, while the new standalone sync helper is verified separately by SHA.
