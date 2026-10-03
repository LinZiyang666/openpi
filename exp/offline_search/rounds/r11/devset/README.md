# R11 B-pool development set

Use `dev=true, init_pool="B"` on **every** `arms.json` row in a fresh root.
Existing roots without these fields remain A. Dev manifests explicitly carry
the same fields and contain B `orig_init_state_idx` coordinates, never subset
positions. A root cannot mix pool declarations. `OSCL_INIT_POOL` cannot override
the root on the controller; the controller sets it on the B worker driver.

All local Python commands use this prefix from the repository root:

```bash
P=(taskset -c 22-37,66-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= "$PWD/.venv/bin/python")
```

Build a deterministic manifest or its pure-policy reference table:

```bash
"${P[@]}" -m exp.offline_search.closed_loop.devset dev_manifest --model pi05 --suite l10 --r10-size 50 --per-task 5 --seed 20261003 --output exp/offline_search/rounds/r11/devset/example_manifest.json
"${P[@]}" -m exp.offline_search.closed_loop.devset dev_manifest --model groot --suite spatial --library current --per-task 5 --seed 20261003
"${P[@]}" -m exp.offline_search.closed_loop.devset reference --manifest exp/offline_search/rounds/r11/devset/example_manifest.json
```

`--output` refuses existing files; omit it to print JSON. Selection uses PCG64
with `seed + task_id`, samples without replacement from each task's complement,
and sorts the emitted pairs. Failed served library episodes are excluded too.
Insufficient complements are refused, including every R10 size-500 library.
The reference includes successes and failures from the full parent B library,
one Boolean per exact manifest pair. It does not run or fit a policy.

Two JSON-only roots are ready:

- `/home/weiland/trace_runs/os_closed_loop/r11_dev_current`: four current fits.
- `/home/weiland/trace_runs/os_closed_loop/r11_dev_size50`: four R10 size-50 fits.

Each contains four distinct 50-pair manifests, arm matrices, an arm contract
table, a pure-policy reference table, and a protocol. Prefits and server YAMLs
remain explicit external read-only dependencies. No result or DONE file was
copied. To emit another fresh root, reuse completed R10Recipe prefits:

```bash
"${P[@]}" -m exp.offline_search.rounds.r11.devset.prepare --output /home/weiland/trace_runs/os_closed_loop/r11_dev_size100 --r10-size 100 --per-task 5 --seed 20261003
```

For a new method, provide a plugin prefit whose metadata matches the arm, whose
`library` matches `kwargs.library`, and whose `fit_info` attests
`library_only=true` and `episode_ids_by_task`. `kwargs.episode_subset`, when
present, names positions in the parent `episodes.json`, matching R10Recipe.
The validator independently resolves the actual fit selection and checks every
manifest pair against it. An unrecognized selection, missing identity, stale
metadata, or overlapping pair is a hard refusal. Native/stock arms are refused
in dev mode because they do not expose this serving-subset attestation.

Some stored metadata contradicts the task's premise: pi05 L10 current has 50
null originals, GR00T Spatial parent 464, and its current subset 47. The
hash-bound `identities/*.json` sidecars recover these without editing any store
or R10 file. GR00T uses the shared collector's task*50+init formula, verified
against all surviving originals and the complete 500-coordinate grid. Current
GR00T episodes are joined to identical source H5/stem/task/success and state
bytes in the parent. pi05 uses a unique byte-exact first 8-D robot-state match
against the B parent within each task; all recovered per-task sets also equal
the historical materialization map. Unknown or stale identities remain refused.

The worker adapter verifies canonical A/B paths and frozen digests, requires a
SHA-bound B contract, repeats disjointness, rejects init remapping and duplicate
path arguments, and rejects `.pruned_init` files that would shadow attested
`.init` files. It labels journals before the shared conductor writes them and
rejects foreign journals before resume. Counters, summaries and KPI readers
reject cross-pool rows; paired KPI comparisons and multi-root reports reject
A/B mixing. Old unlabelled journals are A only. B DONE filenames include `B_`.
The inherited run_gtp argument names and launch JSON key `apool` remain for API
compatibility; the frozen B record and journals explicitly identify pool B.

CPU reproduction:

```bash
"${P[@]}" -m exp.offline_search.rounds.r11.devset.identities
"${P[@]}" -m exp.offline_search.rounds.r11.devset.mapping
"${P[@]}" -m pytest -q -p no:cacheprovider --basetemp=exp/offline_search/rounds/r11/devset/evidence/pytest_reproduce exp/offline_search/closed_loop/test_devset.py exp/offline_search/closed_loop/ops/h100/test_h100.py exp/offline_search/closed_loop/ops/h100/test_robustness.py
"${P[@]}" -m exp.offline_search.rounds.r11.devset.verify_a
```

Mapping imports the existing simulator's pure Python LIBERO/robosuite packages
through `--sim-site`, while native modules come from `.venv/bin/python`. It
resets tasks 0,4,9 at B indices 0,17,49 on both suites, takes the collection's
ten dummy controls, and compares the first policy observation with source H5
and library first-row state. Cameras, render contexts and inference are absent.
The mapping report records raw, normalized and quantized state differences.
The source H5 rows store no object/simulator state, so object comparison is
explicitly unavailable. Coordinator deployment and launch commands, complete
evidence, and source hashes are in [HANDBACK.md](HANDBACK.md).
