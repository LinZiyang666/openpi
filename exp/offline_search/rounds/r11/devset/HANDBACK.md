# R11 prerequisite handback

## Dev set

**COMPLETE on CPU. No remote setup, sync, chain, policy call, worker, server,
closed-loop evaluation, or GPU operation was launched.**

Ready roots:

| Root under `/home/weiland/trace_runs/os_closed_loop/` | Arms | B pairs per arm | H100 plan files / bytes |
| --- | ---: | ---: | ---: |
| `r11_dev_current` | 4 | 50 | 78 / 23,470,919,469 |
| `r11_dev_size50` | 4 | 50 | 78 / 23,472,271,062 |

Both roots contain JSON only. Local generated YAML/pickle planning artifacts
are under `evidence/dev_plans/`; each root's `h100_sync/plan.json` points there.
Fits and server configurations are preserved R10Recipe inputs. The roots have
no result, state, launch receipt or DONE markers. Every arm declares
`dev=true, init_pool=B`, has its own deterministic manifest (K=5, seed=20261003),
and passes the actual-prefit selection guard. No method was fitted or changed.

Pure-policy parent-library reference outcomes for the exact dev pairs:

| Cell | Current subset | Size-50 subset |
| --- | ---: | ---: |
| pi05 L10 | 42/50 = .84 | 42/50 = .84 |
| pi05 Spatial | 50/50 = 1.00 | 49/50 = .98 |
| GR00T L10 | 41/50 = .82 | 44/50 = .88 |
| GR00T Spatial | 45/50 = .90 | 45/50 = .90 |

These are stored pure-policy outcomes, not new dev rollouts. The per-pair Boolean
tables and parent metadata hashes are in each root's
`pure_policy_reference.json`. Failed parent/library episodes remain included.
The dev builder samples outside the served subset per task, using PCG64(seed +
task_id), without replacement, sorted pairs. All four cells and sizes
50/100/200/300/400 are tested. Size 500 is correctly refused: no B complement.

Hard guards are enforced at planning, controller preflight, worker entry,
resume, counting, summarization and paired KPI scoring. The actual prefit must
match arm metadata and attest its library-only episode selection; a manifest
overlapping any served episode's B original index is refused. All arms in a
root must agree on A/B. A cannot load B, B cannot load A, unlabelled manifests
cannot be dev, and environment overrides cannot reclassify a controller root.
The B worker receives a SHA-bound contract, checks its arm/model/suite/manifest,
repeats disjointness, verifies the canonical B path and frozen digest, rejects
init remapping/duplicate overrides and `.pruned_init` shadows. Journals and
summaries record `init_pool`; old unlabelled journals are A only. Foreign rows
and mixed A/B KPI reports/comparisons are refused. B DONE names include `B_`.

Two new frozen records attest 10 tasks × 50 B inits each; setup installs their
files in either worker island and dev sync pushes the selected suite files as
well. Bundle SHA checks verify transferred bytes; the standard digest loader
rehashes on verify-worker and again before worker launch. Both suites' A/B
state intersections were recomputed on CPU: **zero shared inits for all 20
tasks**. B rollups:

```text
BPOOL_FROZEN libero_spatial total=500 shared=0 sha=f35ae5439028a3cf5279271b34db2b58a0ad842b53f58e7162ef17de5840d478
BPOOL_FROZEN libero_10 total=500 shared=0 sha=fa993ce2042942a79941960eba23d9186f2b835c2fc0ba1e2f47e7d308d00b14
```

The stored metadata was not uniformly complete. We recovered 50 missing
pi05 L10 current identities, 464 GR00T Spatial parent identities and 47 GR00T
Spatial current identities in `identities/*.json`, without editing the store
or R10. GR00T recovery validates the collector's global episode ID formula,
every known original and the full 500-coordinate grid; current episodes join
to identical parent source H5/stem/task/success/state bytes. pi05 current
matches a unique **byte-exact** first 8-D float32 state within each B-parent
task: 50/50 matched, maximum delta 0, minimum next-candidate delta
**0.001680612564086914**. All ten recovered init sets also equal the historical
materialization map. Sidecars bind source episode metadata and robot-state
hashes; changed evidence or unrecoverable originals are refused.
`identity_audit.json` and `evidence/identities.log` record the numbers.

The mapping proof actually reset CPU LIBERO at tasks **0,4,9**, indices
**0,17,49**, on **both suites**, comparing both models' first recorded library
rows. It uses the client's ten dummy controls before its first policy
observation, the original pi05 quantile transform, and the GR00T min/max
transform followed by bfloat16 rounding. Library rows were also checked against
source H5 first rows. No camera or render context was created.

```text
MAPPING_PROOF {"passed": true, "comparisons": 36, "tasks": [0, 4, 9], "indices": [0, 17, 49], "dummy_control_steps": 10, "camera_observations": false, "gpu": false, "max_abs_delta": 0.0, "max_abs_delta_physical": 0.0014241795435072646}
```

pi05: 18/18, stored-state max |Δ| **0**, inverse-normalized physical max
**2.6052965562683994e-8**. GR00T: 18/18, normalized+bfloat16 stored-state max
|Δ| **0**, physical max **0.0014241795435072646** due to bfloat16 quantization.
The report includes unrounded errors, raw/recorded vectors, source hashes and
per-case results in `mapping_proof.json`; stdout is `evidence/mapping.log`.
Reset observations have object state, but the library source H5 first rows
store no object/simulator state. That comparison is unavailable; no object-state
equality is claimed. No required mapping check remains dependent on GPU.

Verification key outputs:

```text
168 passed
A_UNCHANGED {"root": "/home/weiland/trace_runs/os_closed_loop/r10_recipe_current", "plan_byte_identical": true, "plan_sha256": "180dbf3c2762c88778e39c43646a49e697b01a9ad333f09a352c718682bfd147", "files": 78, "bytes": 23470919287, "selection_and_server_specs_equal": true, "driver_argv_equal": true, "manifest_sha256": "2c3b0477794c04b9f9925100346f0d0adbca4dcfe42f3c2eb26278f18b4d9a29", "pool": "A", "remote_calls": 0}
```

The tests cover builder determinism/complements, all real R10 cells/sizes and
current libraries, fit-selection lies, overlap, unknown identities, environment
and root disagreements, worker A/B paths, shadow files, contract checks, resume,
counting, summary/KPI isolation, plus the existing 130 H100 robustness tests.
`evidence/tests.log` has the exact pytest output. The A comparison uses the
standard dependency planner on an existing R10 root, writing evidence here,
with a before-edit baseline. Plan bytes, original/installed asset hashes,
selection and server specs, and mocked worker argv are identical. No R10 root
was written. Newly produced A journals gain only the requested pool label.

Exact local reproduction commands (from `/home/weiland/projects/openpi`):

```bash
P=(taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= "$PWD/.venv/bin/python")
"${P[@]}" -m exp.offline_search.rounds.r11.devset.identities
"${P[@]}" -m exp.offline_search.rounds.r11.devset.mapping
"${P[@]}" -m pytest -q -p no:cacheprovider --basetemp=exp/offline_search/rounds/r11/devset/evidence/pytest_coordinator exp/offline_search/closed_loop/test_devset.py exp/offline_search/closed_loop/ops/h100/test_h100.py exp/offline_search/closed_loop/ops/h100/test_robustness.py
"${P[@]}" -m exp.offline_search.rounds.r11.devset.verify_a
"${P[@]}" -m exp.offline_search.closed_loop.devset dev_manifest --model pi05 --suite l10 --r10-size 100 --per-task 5 --seed 20261003
"${P[@]}" -m exp.offline_search.closed_loop.devset dev_manifest --model groot --suite spatial --library current --per-task 5 --seed 20261003
"${P[@]}" -m exp.offline_search.closed_loop.devset reference --manifest /home/weiland/trace_runs/os_closed_loop/r11_dev_current/dev_groot_spatial.json
```

To create another root, pick an unused `r11_dev_*` tag:

```bash
"${P[@]}" -m exp.offline_search.rounds.r11.devset.prepare --output /home/weiland/trace_runs/os_closed_loop/r11_dev_size100 --r10-size 100 --per-task 5 --seed 20261003
```

Exact coordinator deployment/launch commands for the **idle timan107 fleet**:
the H100 serving tree must already contain the R10Recipe serving closure listed
in `H100_SOURCES.sha256`. This change adds no serving plugin dependency beyond
that closure. If R10Recipe has not been installed, the existing full `setup`
procedure is required first, with all owned fleets idle; it is not run here.
Worker-only setup below makes no H100/timan108 mutation. Concurrent sync retains
existing H100 assets and can refuse conflicts. Keep any existing fleet intact.

```bash
cd /home/weiland/projects/openpi
P=(taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= "$PWD/.venv/bin/python")
RUN=/home/weiland/trace_runs/os_closed_loop/r11_dev_current
mapfile -t ARMS < <("${P[@]}" -c 'import json,sys; print("\n".join(r["arm"] for r in json.load(open(sys.argv[1]))))' "$RUN/arms.json")
unset OSCL_MANIFEST OSCL_EPISODES OSCL_TASKS OSCL_INIT_POOL
WORKER_HOST=timan107 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control setup --worker-only
WORKER_HOST=timan107 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control verify-worker
WORKER_HOST=timan107 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control plan "$RUN" "${ARMS[@]}"
WORKER_HOST=timan107 SYNC_PORT=23197 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control sync --concurrent "$RUN" "${ARMS[@]}"
# Choose free forwarded ports and a free sync port; do not stop existing listeners.
WORKER_HOST=timan107 PORTS=23220,23221 WPS=4 MAX_ATTEMPTS=3 POLL_SECONDS=60 \
  "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control chain "$RUN" "${ARMS[@]}"
WORKER_HOST=timan107 "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control collect "$RUN" "${ARMS[@]}"
```

For size 50, set `RUN=/home/weiland/trace_runs/os_closed_loop/r11_dev_size50`,
regenerate `ARMS`, and use the same plan/sync/chain sequence. Leave
`OSCL_MANIFEST` unset: each arm selects its own cell-specific B complement.
Expected verify-worker B rollups are listed above. Before episodes, the driver
prints `INIT_POOL_OK pool=B`; completed journals and summaries identify B, and
completion is **50** per arm. The coordinator still owns deployment verification,
live server/worker compatibility and all dev rollouts. No live result or SR is
asserted by this handback.

Source lists and SHA256:

- `H100_SOURCES.sha256`: complete R10Recipe serving closure plus these runtime
  changes; server numerics remain the R10 closure.
- `NEW_RUNTIME_SOURCES.sha256`: all 12 new/modified controller, worker and reader
  files for this task, with exact SHA256.
- `ALL_SOURCES.sha256`: runtime sources plus builder/proof/tests.
- `BPOOL_FILES.sha256`: all 20 original B `.init` files.
- `source_list.json`: new/modified source paths and hashes.

New runtime files (modified-file hashes are in the lists above):

```text
499c938fd6362ac177a82c33c48722f4a8314770e8a74985a45daf4f1a04e313  exp/offline_search/closed_loop/devset.py
64c8c242f5ff74dc2be77c8b6916653ccb789d1268c1a9d1512af86e0761ef48  exp/offline_search/closed_loop/worker_pool.py
545c5a2dd62854e13f71ec948345cdad9d7001c254cd845d525993c0d28702e5  exp/offline_search/closed_loop/ops/h100/bpool_libero_10.yaml
74db8a76b37501664a879019d17772acaa299b90126284d0a8882562ba4b4c07  exp/offline_search/closed_loop/ops/h100/bpool_libero_spatial.yaml
```

See [README.md](README.md) for schemas and custom plugin attestation.
