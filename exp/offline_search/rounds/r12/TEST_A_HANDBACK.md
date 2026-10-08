# R12 Test-A handback

No servers, simulators, real plan/sync/chain, smokes, remote operations or git commands were run. Frozen fits and thresholds were reused without fitting or recalibration. Definitions are ready; **final A placement remains pending because this session cannot write `/home/weiland/trace_runs`**. All seven A roots are staged under `exp/offline_search/rounds/r12/test_a_staged/`; the tested installer below copies definitions into fresh final roots and rewrites their owned paths. The B root is already at its requested location.

Final A root parent: `/home/weiland/trace_runs/os_closed_loop/`. Arm names are exactly `<root-name>_<method>`.

| Root name | Methods | Arms | Episodes/arm |
| --- | --- | ---: | ---: |
| `r12_test_pi05_l10_50` | off, opus_m1, h2, m2 | 4 | 500 |
| `r12_test_pi05_spatial_50` | off, opus_m1, h2, m2 | 4 | 500 |
| `r12_test_groot_spatial_50` | off, opus_m1, h2, m2 | 4 | 500 |
| `r12_test_groot_l10_50` | off, opus_m1, h2, m2 | 4 | 500 |
| `r12_test_pi05_spatial_200` | off, opus_m1 | 2 | 500 |
| `r12_test_pi05_l10_50_periodic_ir32` | off, h2, m2 | 3 | 500 |
| `r12_test_pi05_spatial_200_guard_pending` | h2, m2; hold until each passes B screen | 2 | 500 |
| `guard/pilots/r12_guard_fix2_pi05_spatial_200` (under R12) | h2, m2 | 2 | 100 |

Totals: **23 A arms / 11,500 episodes**, including 21 ready arms / 10,500 episodes and two conditional arms / 1,000 episodes; **two B arms / 200 episodes**. The A `eval500.json` files are exact byte copies of all four R11 knob templates, covering 50 inits for each of ten tasks for either suite. B selection is byte-copied from `r11_dev_k10_size200`, with ten disjoint inits per task. Inventories: `test_a_roots.json`, `guard/fix2_roots.json`; each root's `ROOT.json` binds its selection and fit hashes.

Exact pilot subsets, kwargs, yaml sources and construction are preserved: pi05 uses `bpool_cs`, GR00T uses **`bpool_all` as in its frozen fix1 artifacts**, with the same `episode_subset`. One off arm per cell uses the fix1 off artifact; the periodic cell uses its guard off artifact. Every arm has guard_only, replan 5, one policy-tail block, full_model and cost_ledger. A has no look/image recording or B-only GR00T recording diagnostic flags. Collection retains `lookhalf_cost_ledger`, and `first_guard_call` is reported for each accepted episode (null for no guard call); only accepted attempts and deduplicated decisions count.

Spatial-200 H2/M2 thresholds **were dev-calibrated by the same procedure as the other cells** in `opus/tools/guard_m2.py`: theta_a is the eligible (step >= 4) knob-off dev risk 25th percentile; theta_b matches the deployed guard rate on that dev arm. Both existing fits carry `threshold_source_arm=r11_dev_pi05_spatial_200`, **theta_a=0.14617910496241962**, **theta_b=0.7546450157515548**, matching `opus/out/guard_m2.json` and `guard/risk_model.json`. They were not recalibrated for this screen. All **35 frozen artifact hashes** match their original build catalogs (`TEST_A_FROZEN_SHA256.json`). Local serving reads the original pickle path/bytes; h100 uses the unchanged pilot planner's path relocation, with frozen source SHA bound in the plan.

Source changes (paths relative to `exp/offline_search/`):

- `closed_loop/worker_pool.py::install`: allows the audited worker after ordinary A attestation; B still requires its existing contract. `control.chain` already set audit for active methods on both pools; its flags now come from `closed_loop/ops/h100/control.py::execution_env`. All active opus_m1/h2/m2 workers select `closed_loop.execution_worker`; off retains `examples.libero.worker_entry`. Missing/malformed receipts still fail before encoding.
- `closed_loop/lookhalf_ledger.py::summarize`, `closed_loop/ops/collect.py::summarize`: retain per-episode first guard calls, no-call nulls and the minimum step regardless of log ordering.
- `rounds/r12/prepare_test_a.py::{prepare,install_test_a}`: emits definitions only from frozen sources, with a tested fresh-root placement command.
- `closed_loop/test_r12_test_a.py`: actual emitted roots -> real plan/sync -> entire final server argv, local/timan107 and h100/timan108, frozen path/hash/subset/selection checks, real A pool hashing and worker selection, fail-loud receipts, collection/retry semantics, threshold provenance and final placement.
- `closed_loop/test_r12_pipeline.py`, `rounds/r12/guard/selftest.py`: cover A audited selection and isolate report writes in temporary metadata copies, leaving live pilot roots alone.

Coordinator deployment: push the changed `worker_pool.py`, `execution_worker.py` (unchanged but required), `lookhalf_ledger.py`, `ops/collect.py`, `ops/h100/control.py` to the **h100 server snapshot and the chosen timan worker island**. Push the two pipeline test files, guard selftest and `prepare_test_a.py` as source/evidence too. The controller's `setup` below installs its source snapshot to the required destinations; `sync` transfers the exact configs and bound frozen artifact dependencies. Worker-only setup suffices for local serving. Do this when the selected fleet/server resources are available; run roots sequentially.

CPU validation: **128 passed, zero failures** in 182.39 seconds: `closed_loop/test_r12_pipeline.py` 50, `closed_loop/test_r12_test_a.py` 26, `closed_loop/test_look_recording.py` 25, `rounds/r12/lookhalf/selftest.py` 14, `rounds/r12/guard/selftest.py` 13. The new integration tests check all 23 A arms and both B arms on both topologies, including real A content attestation and exact worker selection for every A arm. One existing torch/pynvml deprecation warning. Commands used only cores `22-37,66-81`, CUDA hidden and one BLAS/OMP thread. Evidence: `TEST_A_VERIFICATION.json`, `TEST_A_CPU.log`, `TEST_A_CPU.xml`. Selftest entry-point evidence files and pilot reports were not rewritten; local argv builders put their empty runtime directories in test scratch. Three receipt-refusal checks also passed with the actual parsed A arm options (`TEST_A_RECEIPTS.log`).

Reproduce the complete CPU check:

```bash
taskset -c 22-37,66-81 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src \
  .venv/bin/python -B -m pytest -q -p no:cacheprovider \
  exp/offline_search/closed_loop/test_r12_pipeline.py \
  exp/offline_search/closed_loop/test_r12_test_a.py \
  exp/offline_search/closed_loop/test_look_recording.py \
  exp/offline_search/rounds/r12/lookhalf/selftest.py \
  exp/offline_search/rounds/r12/guard/selftest.py
```

Exact coordinator commands (not executed here), from `/home/weiland/projects/openpi`:

```bash
cd /home/weiland/projects/openpi
export PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1
export OMP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LOCAL_SERVER_CPUS=22-37,66-81 LOCAL_SERVER_COUNT=4
unset OSCL_MANIFEST OSCL_INIT_POOL OSCL_TASKS OSCL_EPISODES
unset OSCL_POOL_CONTRACT OSCL_POOL_CONTRACT_SHA256 OSCL_RECORD_LOOKS OSCL_EXECUTION_AUDIT
P=(taskset -c 22-37,66-81 env CUDA_VISIBLE_DEVICES= .venv/bin/python -B)
CTL=exp.offline_search.closed_loop.ops.h100.control

# Required once to place staged A definitions; refuses existing final roots.
"${P[@]}" -m exp.offline_search.rounds.r12.prepare_test_a --install-test-a
```

Choose one topology. For local policy servers with timan107, four servers x 16 workers = 64:

```bash
export SERVER_HOST=local WORKER_HOST=timan107 PORTS=23160,23161,23162,23163 WPS=16
"${P[@]}" -m "$CTL" setup --worker-only
```

For h100 policy servers with timan108, four x 10 = fleet cap 40:

```bash
export SERVER_HOST=h100 WORKER_HOST=timan108 PORTS=23260,23261,23262,23263 WPS=10
"${P[@]}" -m "$CTL" setup
```

For either chosen topology, these invoke plan / sync / chain for each root:

```bash
run_root() {
    "${P[@]}" -m "$CTL" plan "$@"
    "${P[@]}" -m "$CTL" sync "$@"
    "${P[@]}" -m "$CTL" chain "$@"
}
T=/home/weiland/trace_runs/os_closed_loop
B=/home/weiland/projects/openpi/exp/offline_search/rounds/r12/guard/pilots

# B screening root, 100 episodes per arm. Existing deployed baselines are reused.
run_root "$B/r12_guard_fix2_pi05_spatial_200" \
  r12_guard_fix2_pi05_spatial_200_h2 r12_guard_fix2_pi05_spatial_200_m2

# A roots, 500 episodes per arm. chain also collects.
run_root "$T/r12_test_pi05_l10_50" \
  r12_test_pi05_l10_50_off r12_test_pi05_l10_50_opus_m1 \
  r12_test_pi05_l10_50_h2 r12_test_pi05_l10_50_m2
run_root "$T/r12_test_pi05_spatial_50" \
  r12_test_pi05_spatial_50_off r12_test_pi05_spatial_50_opus_m1 \
  r12_test_pi05_spatial_50_h2 r12_test_pi05_spatial_50_m2
run_root "$T/r12_test_groot_spatial_50" \
  r12_test_groot_spatial_50_off r12_test_groot_spatial_50_opus_m1 \
  r12_test_groot_spatial_50_h2 r12_test_groot_spatial_50_m2
run_root "$T/r12_test_groot_l10_50" \
  r12_test_groot_l10_50_off r12_test_groot_l10_50_opus_m1 \
  r12_test_groot_l10_50_h2 r12_test_groot_l10_50_m2
run_root "$T/r12_test_pi05_spatial_200" \
  r12_test_pi05_spatial_200_off r12_test_pi05_spatial_200_opus_m1
run_root "$T/r12_test_pi05_l10_50_periodic_ir32" \
  r12_test_pi05_l10_50_periodic_ir32_off \
  r12_test_pi05_l10_50_periodic_ir32_h2 r12_test_pi05_l10_50_periodic_ir32_m2
```

Separate invocation **only after each selected guard passes the Spatial-200 B screen**. If only one passes, give plan/sync/chain only that arm. With the same chosen topology and shell definitions above:

```bash
run_root "$T/r12_test_pi05_spatial_200_guard_pending" \
  r12_test_pi05_spatial_200_guard_pending_h2 r12_test_pi05_spatial_200_guard_pending_m2
```

The common `run_root` commands are exact plan/sync/chain commands for either exported topology. Screen approval is a coordinator decision; no automatic screening or A launch was performed. The two-arm B root is collected normally by chain; the old guard `report.py` expects an off arm and is not an entry point for this root.


## Local receipt race fix (sol_fix3)

2026-10-03: fixed the child ownership receipt race in local serving. `supervise()`
resolves the existing `taskset -c <cpus> <python> ...` job descriptor into a direct
interpreter launch, with validated CPU affinity applied by `os.sched_setaffinity`
in the child's `preexec_fn`. `Popen` returns only after that exec succeeds; there
is no child launcher exec left to race receipt capture. Affinity/exec failures
cannot publish a child receipt. Existing job files and serving arguments remain
compatible. The source file was replaced atomically, so new arms see a complete
version. Existing supervisors and receipts were left alone.

Changed files/functions, relative to `exp/offline_search/`:

- `closed_loop/ops/h100/local_server.py::{cpu_affinity_set,cpu_affinity,supervise}`:
  reuse the CPU allowlist parser and spawn the final interpreter directly.
  `proc_identity`, `process_present`, `process_receipt`, `authenticated`,
  `signal_owned`, `start`, `stop`, and `status` are byte-for-byte unchanged.
  Command mismatch and PID reuse still refuse signalling.
- `closed_loop/ops/h100/test_local_receipt_race.py`: deterministic fake launcher,
  fake clock/processes and mocked affinity/signals; no actual process launches.
- `closed_loop/ops/h100/test_local.py`: mock listener/bind availability in
  `test_stop_refuses_recycled_pid_and_command_mismatch` and
  `test_local_start_refuses_foreign_listener_and_low_local_memory`, so these
  ownership tests do not depend on sandbox socket permissions or live ports.
- `closed_loop/test_r12_test_a.py::test_actual_emitted_plan_sync_server_argv`:
  assert that the original `runs`/`state` directory presence is preserved. The
  B-screen root now has runtime output from earlier work; the old requirement
  that those directories be absent caused both topology cases to fail. The
  existing comparison of all original file bytes remains in place.

The six new h100 command checks confirm that `ops/h100/node.py::command` already
launches Python directly for pi05/GR00T, stock/plugin paths. Its supervisor has
no intervening taskset launcher or captured launcher command receipt, so
`node.py` needed no sol_fix3 change.

Deterministic regression: before the fix,
`test_supervise_records_final_exec_even_with_delayed_launcher[6.0]` failed with
`local PID command differs from ownership receipt; refusing`, after the fake
launcher exec'd beyond the old five-second polling deadline. After the fix,
both the 0.25-second and six-second cases pass. The new file has **11 passing
cases**:

- `test_supervise_records_final_exec_even_with_delayed_launcher`: 2; includes
  successful owned signalling, changed-command refusal and PID-reuse refusal.
- `test_supervise_rejects_reserved_affinity_before_launch`: 1.
- `test_supervise_affinity_failure_never_records_child`: 1.
- `test_supervise_accepts_direct_interpreter_job`: 1.
- `test_h100_server_command_starts_interpreter_directly`: 6.

Final validation: **361 distinct CPU checks passed; zero unresolved failures;
three tests deselected**. The local suite passed in 2.79 seconds. The full R12
run initially had 126 passes and the two runtime-directory fixture failures;
after correcting that test, its entire 26-case file passed in 133.84 seconds.
The final R12 count is 102 unchanged checks from the full run plus those 26
rerun checks, for **128 distinct passing R12 checks**. One existing torch/pynvml
deprecation warning remains.

| Test file | Final passing cases |
| --- | ---: |
| `closed_loop/ops/h100/test_h100.py` | 21 |
| `closed_loop/ops/h100/test_robustness.py` | 107 |
| `closed_loop/ops/h100/test_local.py` | 32 |
| `closed_loop/ops/h100/test_local_port_release.py` | 24 |
| `closed_loop/ops/h100/test_local_receipt_race.py` | 11 |
| `closed_loop/test_devset.py` | 38 |
| **Local-serving subtotal** | **233** |
| `closed_loop/test_r12_pipeline.py` | 50 |
| `closed_loop/test_r12_test_a.py` | 26 |
| `closed_loop/test_look_recording.py` | 25 |
| `rounds/r12/lookhalf/selftest.py` | 14 |
| `rounds/r12/guard/selftest.py` | 13 |
| **R12 subtotal** | **128** |

Deselected to honor the no-launch constraint:
`test_detached_fake_process_lifecycle_idempotency_and_foreign_receipt`,
`test_supervisor_records_natural_completion`, and
`test_resumable_daemon_pull_checks_exact_content` (starts an rsync daemon).
The first broader local run had 231 passes and three socket-related failures:
the daemon test failed before any daemon launch, and the two ownership tests
were isolated from real bind checks as described above.

Evidence in this directory: `SOL_FIX3_LOCAL_CPU.log/.xml` (final local suite),
`SOL_FIX3_LOCAL_CPU_INITIAL.log/.xml` (initial socket failures),
`SOL_FIX3_R12_CPU_INITIAL.log/.xml` (full R12 run),
`SOL_FIX3_R12_TEST_A_CPU.log/.xml` (corrected Test-A file rerun), and
`SOL_FIX3_REGRESSION_BEFORE.xml` (expected pre-fix regression failure).

Reproduce the final CPU selection from the repository root:

```bash
taskset -c 22-37,66-81 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:src \
  .venv/bin/python -B -m pytest -q -p no:cacheprovider \
  exp/offline_search/closed_loop/ops/h100/test_h100.py \
  exp/offline_search/closed_loop/ops/h100/test_robustness.py \
  exp/offline_search/closed_loop/ops/h100/test_local.py \
  exp/offline_search/closed_loop/ops/h100/test_local_port_release.py \
  exp/offline_search/closed_loop/ops/h100/test_local_receipt_race.py \
  exp/offline_search/closed_loop/test_devset.py \
  exp/offline_search/closed_loop/test_r12_pipeline.py \
  exp/offline_search/closed_loop/test_r12_test_a.py \
  exp/offline_search/closed_loop/test_look_recording.py \
  exp/offline_search/rounds/r12/lookhalf/selftest.py \
  exp/offline_search/rounds/r12/guard/selftest.py \
  -k 'not test_detached_fake_process_lifecycle_idempotency_and_foreign_receipt and not test_supervisor_records_natural_completion and not test_resumable_daemon_pull_checks_exact_content'
```

Deployment: the runtime fix is **local only**, in `local_server.py`; new local
supervisors import it on their next arm start. **No sol_fix3 runtime file must
be pushed to h100 or timan.** The four changed Python files, this handback and
the evidence may be copied there for review/archival during a later authorized
source sync; no remote push was performed here.

All checks used cores `22-37,66-81`, CUDA hidden, and one OMP/BLAS thread. No
server, supervisor, chain or worker was started, stopped or signalled; no GPU or
remote operation, `pkill` or `rm -rf` was run. One read-only `git status` ran in
parallel with the initial task-file read, before its no-git constraint was
known; no further git commands were run.

Runtime source SHA256:

```text
39f872b8e628c0ca2326fe50c92cfaeca500ee642ef43f3a5a8a485cda2d1b2c  exp/offline_search/closed_loop/ops/h100/local_server.py
```

## Collect archive split (sol_fix4)

Collection now requests archives with **`max_bytes=700_000_000` (700 MB decimal)**,
leaving 112,652,363 bytes below the observed 812,652,363-byte broker ceiling.
Small sources still produce one tar. Larger sources are packed into independent
tar volumes by file, including PAX headers, block padding, end markers and tar
record padding in the size budget. This also splits a single large top-level
server directory. An individual file exceeding the budget uses bounded byte
parts of its tar; every part is SHA-checked and the full tar is rejoined and
SHA-checked in local scratch before extraction. This exceptional case stages a
full temporary tar on the remote node before splitting it.

The worker source and `client/` prefix, h100 server source and paths, existing
local server logs, unsafe-member checks, and final `summarize(root, arm,
manifest=manifest)` behavior are preserved. All created transfer parts are
removed on successful collection and on pull, checksum or extraction failure.
If cleanup also fails, the original collection error is preserved and the
cleanup error is logged. `too_large` fails on the first attempt without sleeping
in both the ordinary tether retry path and the outer collection retry loop.
Other collection timeouts retain the existing eight-attempt budget and backoff.

Changed files (relative to the repository root):

- `exp/offline_search/closed_loop/ops/h100/control.py`:
  `COLLECT_ARCHIVE_MAX_BYTES`, `CommandError.__init__` and `collect`; requests
  bounded volumes, verifies each part, rejoins exceptional large-file volumes,
  extracts with the existing checks, and cleans up the generated archive paths.
- `exp/offline_search/closed_loop/ops/h100/node.py`:
  `archive` and new `_bounded_archive`; produces bounded transfer metadata
  containing paths, sizes and SHA256 values. Archive RPCs without `max_bytes`
  retain the legacy tar and JSON response; the frozen legacy CLI regressions pass.
- `exp/offline_search/closed_loop/ops/h100/test_collect_split.py`: new CPU-only
  round-trip and failure checks with transport mocked and test-owned source,
  staging and destination paths.
- `exp/offline_search/closed_loop/ops/h100/test_robustness.py`:
  `test_collect_pull_retries_timeout_before_sha_and_extraction` and
  `test_collect_pull_timeout_exhausts_eight_attempts` mock remote cleanup too,
  since failed pulls now clean their archives.
- `exp/offline_search/rounds/r12/TEST_A_HANDBACK.md`: this handback section.

New regression names and final case counts (**27 passed**):

| Test in `test_collect_split.py` | Cases |
| --- | ---: |
| `test_collect_small_run_one_archive_per_source_exact_bytes` | 1 |
| `test_collect_oversized_member_set_bounded_tars_exact_extraction` | 3 |
| `test_collect_single_oversized_file_verified_parts_exact_extraction` | 1 |
| `test_collect_empty_run_one_archive_per_source` | 1 |
| `test_tether_too_large_never_retried` | 10 |
| `test_collect_too_large_never_retried_and_cleans_all_parts` | 1 |
| `test_collect_plain_runtime_too_large_never_retried` | 1 |
| `test_collect_checks_each_part_sha_before_extracting` | 2 |
| `test_collect_split_preserves_unsafe_member_refusal` | 5 |
| `test_collect_cleanup_failure_preserves_too_large_error` | 1 |
| `test_archive_failure_cleans_created_parts` | 1 |

The small-source test also compares the tar bytes with the legacy archive
helper. Split tests lower the budget to 32,000 bytes to exercise the same
boundaries without allocating hundreds of megabytes; they check every transfer
size and exact extracted file contents for multiple server directories, one
large directory, long PAX paths and an oversized individual file.

Final required CPU suites: **388 distinct tests passed, zero failures;
three launch/daemon tests deselected**. The h100/local plus devset selection
passed in 2.76 seconds; the R12 selection passed in 156.57 seconds with the
existing torch/pynvml deprecation warning.

| Test file | Passing cases |
| --- | ---: |
| `closed_loop/ops/h100/test_h100.py` | 21 |
| `closed_loop/ops/h100/test_robustness.py` | 107 |
| `closed_loop/ops/h100/test_local.py` | 32 |
| `closed_loop/ops/h100/test_local_port_release.py` | 24 |
| `closed_loop/ops/h100/test_local_receipt_race.py` | 11 |
| `closed_loop/ops/h100/test_collect_split.py` | 27 |
| **ops/h100 subtotal** | **222** |
| `closed_loop/test_devset.py` | 38 |
| **h100/local plus devset subtotal** | **260** |
| `closed_loop/test_r12_pipeline.py` | 50 |
| `closed_loop/test_r12_test_a.py` | 26 |
| `closed_loop/test_look_recording.py` | 25 |
| `rounds/r12/lookhalf/selftest.py` | 14 |
| `rounds/r12/guard/selftest.py` | 13 |
| **R12 subtotal** | **128** |
| **Total** | **388** |

Deselected to honor the no-launch constraint:
`test_detached_fake_process_lifecycle_idempotency_and_foreign_receipt`,
`test_supervisor_records_natural_completion`, and
`test_resumable_daemon_pull_checks_exact_content`.
Logs and JUnit evidence are in `/tmp/openpi_sol_fix4/ops_h100_cpu.log`,
`/tmp/openpi_sol_fix4/ops_h100_cpu.xml`, `/tmp/openpi_sol_fix4/r12_cpu.log`,
and `/tmp/openpi_sol_fix4/r12_cpu.xml`.

Reproduce the complete CPU selection from the repository root:

```bash
taskset -c 22-37,66-81 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:src \
  .venv/bin/python -B -m pytest -q -p no:cacheprovider \
  exp/offline_search/closed_loop/ops/h100/test_h100.py \
  exp/offline_search/closed_loop/ops/h100/test_robustness.py \
  exp/offline_search/closed_loop/ops/h100/test_local.py \
  exp/offline_search/closed_loop/ops/h100/test_local_port_release.py \
  exp/offline_search/closed_loop/ops/h100/test_local_receipt_race.py \
  exp/offline_search/closed_loop/ops/h100/test_collect_split.py \
  exp/offline_search/closed_loop/test_devset.py \
  exp/offline_search/closed_loop/test_r12_pipeline.py \
  exp/offline_search/closed_loop/test_r12_test_a.py \
  exp/offline_search/closed_loop/test_look_recording.py \
  exp/offline_search/rounds/r12/lookhalf/selftest.py \
  exp/offline_search/rounds/r12/guard/selftest.py \
  -k 'not test_detached_fake_process_lifecycle_idempotency_and_foreign_receipt and not test_supervisor_records_natural_completion and not test_resumable_daemon_pull_checks_exact_content'
```

**`node.py` changed and requires coordinator deployment to h100 and both timan
islands.** The exact required remote runtime push list is one source file,
`/home/weiland/projects/openpi/exp/offline_search/closed_loop/ops/h100/node.py`,
to these three standalone helper destinations:

| Host | Destination |
| --- | --- |
| `h100` | `/data/oscl_h100/node.py` |
| `timan107` | `/scratch/zixuans8/openpi_trace_h100/os_cl/node.py` |
| `timan108` | `/scratch/zixuans8/openpi_trace/os_cl/node.py` |

The updated `control.py` runs on the local coordinator. Test files and this
handback are source/evidence artifacts. Collection invokes the standalone
helpers above; the h100 `node_sb3.py` pull-new helper is outside this change's
required deployment list.

All changed files were installed with atomic replacement. All checks used
cores `22-37,66-81`, CUDA hidden and one OMP/BLAS thread. No live server,
supervisor, worker or chain was started, stopped or signalled. No GPU operation,
remote operation, real plan/sync/chain/collect, git, `pkill` or `rm -rf` was run.

Verified runtime source SHA256:

- `closed_loop/ops/h100/control.py`: `3e392813640bba4fe92eaac398b5af60c481b7aa9bd61199ce57819c97c08154`
- `closed_loop/ops/h100/node.py`: `16991dcf03c11eb27f3077616abcb32b9775acd3d65d25f98ea8defcb525f819`
