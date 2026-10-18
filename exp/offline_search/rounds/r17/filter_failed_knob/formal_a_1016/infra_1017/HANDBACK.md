# Local server reservation and infrastructure-error repair — 2026-10-17

Superseded continuation count: the later [18:3x handback](../probe_1017/HANDBACK.md)
renamed the six-episode draft aside and replaced `fa_fw_r_u_lc1` with 13,190
episodes after the source lost its worker fleets. The six-episode proof and
selection details below describe the preserved earlier draft, not the current run.

The launch probe's flat 16,384 MiB growth allowance admitted 44 local renderer
slots while `fa_fw_r_u_l`'s two replicas were loading. Its state then assigned
22 local workers. The six accepted retry-exhaustion rows are infrastructure
failures, and the old loader treated them as unsuccessful task outcomes.
Both replica logs confirm CUDA OOMs (replica 0 line 566: 73.56 MiB free;
replica 1 line 912: 90.62 MiB free), despite the unchanged 27 GiB per-replica
allocator allowance. This is exhaustion of shared device memory.

`filter_failed.worker_resilience.measure` now reserves the sealed campaign's
whole `memory_budget_gib × 1024` allowance for a local server. This is already
replicas × per-replica allowance: the resident entry divides the field by R.
It no longer reads the flat growth default. `capacity` subtracts the remaining
unmaterialized server budget after crediting physical memory of live,
receipt-owned replica PIDs. Other lanes' physical usage, pending renderer
reservations, and the existing 10% device headroom remain charged. The cold
and loaded tests give the same capacity. Unknown ownership is conservative.

Fast-WAM's unchanged sealed total allowance is 54 GiB (27 GiB per replica),
which exceeds the 48 GB card; using that allowance conservatively leaves zero
local renderer slots. This changes renderer allocation, not the serving budget,
replica count, batching or runtime. The existing allocator redistributes to
available remote workers. No source campaign or active chain was edited.

## Queued 4090 capacity

`QUEUED_LOCAL_CAPACITY.json` covers the remaining units in the existing
`run_ffk_4090b.sh`, plus its final π0.5 MT50 item, using catalog/immutable
deferred templates. The snapshot removes the current Fast-WAM server to model
an otherwise idle 4090 and retains other lanes' renderer use. Device total:
49,140 MiB; fixed headroom: 4,914 MiB; other physical + pending use: 2,678 MiB.
At observation, `fa_co_f_u_o` held **five** local slots; the earlier report's
two rendered contexts does not account for its pending leased slots.

| Queue NAME | Total server allowance MiB | GPU slots after reservation | Local workers allowed by its launcher |
|---|---:|---:|---:|
| fa_fw_r_f_l | 55,296 | 0 | 0 |
| fa_pi_r_u_l | 30,000 | 21 | 0 — R16 remote only |
| fa_pi_r_f_l | 30,000 | 21 | 0 — R16 remote only |
| fa_gr_r_u_l | 26,000 | 29 | 0 — R16 remote only |
| fa_gr_r_f_l | 26,000 | 29 | 0 — R16 remote only |
| fa_dp_f_u_l | 40,960 | 1 | 1 |
| fa_dp_f_f_l | 40,960 | 1 | 1 |
| fa_gr_f_u_l | 27,200 | 27 | 0 — R15 remote only |
| fa_gr_f_f_l | 27,200 | 27 | 0 — R15 remote only |
| ffk1015_pi05_mt50_reduced_descent | 40,960 | 4 | 4 |

The R15/R16 GPU-slot column is a planning calculation using their existing
per-server `need_mb` / `NEED_MB` allowances. They do not invoke the R17 local
allocator and have no local worker fleet. Each deferred R15 GR00T unit covers
eight native cells (`_l50/_l100/_l200/_l500/_s50/_s100/_s200/_s500`), all with
the same allowance and zero local workers. Deferred R17 units inherit the
same sealed server budget shown here. Actual placement is measured again at
prepare/launch; the table is a dated capacity snapshot, not a fixed node map.

## Six-episode continuation

**`/data/mw_finetune/eval/fa_fw_r_u_lc1`** is built locally, not staged or
launched. Its standard prepare/launch scripts are in `../../commands/`.
The new `filter_failed.resume --infra-errors` selection uses the existing
native `cache_resume.prepare`, frozen startup-input clone and planner. It
selects only accepted `FatalEpisodeError ... retry budget exhausted` UIDs.
The draft retains the other 29,994 *planned* UIDs, including outcomes the
untouched parent has yet to write. These are not claimed completed outcomes.
No source journal or mutable outcome file is pinned while it is appending.
The sealed startup inputs are pinned, and original error rows are preserved
in the continuation's `CACHE_RESUME.json` / quarantine record.

| Global task id | Original test init |
|---|---|
| 34 | 21, 28, 49 |
| 35 | 28, 42 |
| 36 | 46 |

All six are `r17_0a398b0fbdcabda79f72_n100_chrg_k40:eval:<task>:<init>`.
They retain their exact task UIDs, YAMLs, init identities, seeds, runtime,
interpreter, R2/B16/MPS and sampler. Execution initially requests timan1×12
and timan107×36, matching the parent's current execution map. The native
resilience allocator can redistribute safely; the new reservation gives this
Fast-WAM job zero local capacity.

The parent is still running. The draft records
`source_closure_required_at_prepare`; staging first invokes the **existing
native read-only `prove_closed` ownership probe**. It cannot prepare while
the parent's driver/server/worker owners remain alive. This moves the normal
continuation closure proof from build to prepare without asserting closure.

`filter_failed/CONTINUATIONS.json` registers the child, so `continued_lanes`
and formal-A `load_r17` follow it automatically. `cache.analyze_cache` now
excludes retry-exhaustion infra rows in `accepted_terminal`; a later valid row
for the UID supplies the outcome and its own segment's cost evidence. A valid
`success=False` task outcome remains a failure. Until repair, the cohort has
missing episodes, never imputed failures. Snapshot diagnostics list infra
UIDs and unresolved infra UIDs. Formal-A per-task level records also list
`missing_inits`; a reached incomplete level has no frozen selection.

## Commands

The continuation is already built. Once the parent has closed and the 4090
slot is available, the coordinator runs:

```bash
cd /home/weiland/projects/openpi
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_fa_fw_r_u_lc1.sh
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_fa_fw_r_u_lc1.sh
```

Analysis automatically merges the parent and continuation:

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a analyze
```

The build command used (do not repeat for the existing destination):

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.resume --source /data/mw_finetune/eval/fa_fw_r_u_l --name fa_fw_r_u_lc1 --infra-errors --worker-counts timan1=12,timan107=36
```

## CPU evidence

* **17 tests PASS**: worker resilience, native continuation invariants, ancestor
  merging, error-only selection, live-parent prepare refusal, valid task failures,
  missing-init reporting and formal-A selection. Both new shell scripts pass
  `bash -n`.
* `CONTINUATION_CHECK.json`: native frozen planner **PASS**, six exact task
  objects, 191 asset files byte-identical to startup snapshot, all serving
  identity fields unchanged; **29,994 + 6 = 30,000 unique planned UIDs,
  zero overlap**. These counts describe plan coverage, not future outcomes.
* `FORMAL_A_AUDIT.json`: all 27 started formal-A definitions with local
  journals audited, including **25 completed lanes/cells** (PASS receipt or
  all planned outcomes present). **No other completed formal-A run has such
  infra rows.** Only the active `fa_fw_r_u_l` has the six listed errors.
* `LOCAL_CAPACITY_SNAPSHOT.json` and `QUEUED_LOCAL_CAPACITY.json` preserve the
  physical/pending usage and per-campaign calculation above. NVIDIA was queried
  read-only; no GPU context, server, worker, driver or chain was launched.

Re-run the tests:

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m pytest -q exp/offline_search/rounds/r17/filter_failed/test_worker_resilience.py exp/offline_search/rounds/r17/filter_failed/test_resume.py exp/offline_search/rounds/r17/filter_failed_knob/test_resume.py exp/offline_search/rounds/r17/filter_failed_knob/test_infra_resume.py exp/offline_search/rounds/r17/filter_failed_knob/test_formal_a.py
```
