# S-A handback: r08_abl guard arms

Delivered **24 guard arms, 24 fresh fits, 24 cache YAMLs and 24 worker matrix YAMLs** under
`/home/weiland/trace_runs/os_closed_loop/r08_abl`. The 16 leave-one-guard-out arms cover the four
500-demo cells; the eight `onlynp` arms cover every model × suite × demo-count cell. Each uses
the official 500 test pairs. [ARMS.md](ARMS.md) lists all names, source B rows, full fit SHA256s,
and the exact three A / three B reference names for each cell.

All offline checks passed: 50 unit tests; exact non-guard source fields and fitted state for
24 arms; 48 reference journals with the same 500 official pairs; 731,713 logged-verdict
comparisons and 16,398 actual method-query comparisons with zero mismatches. The query check
also passed 10,932 comparisons against R6's single-guard implementation, 48 explicit blind
veto checks and 48 committed-policy-tail checks. Commands and outputs are recorded below.

## Files owned by S-A

Repository files are confined to this directory's immediate files and the two named test files.
The concurrent `abl/clip/` work and its tests/artifacts belong to another task.

| File | Purpose |
|---|---|
| `__init__.py` | Package marker |
| `judge.py` | Set-valued guard subclasses of R6's pi05/GR00T trigger classes |
| `make_arms.py` | Freeze deployed B sources; generate specs, provenance, selection and name list |
| `prefit.py` | Fresh fits via the same plugin prefit entry point as R6 P2; at most seven children |
| `validate.py` | Source diffs, parser/config/matrix checks, pickle metadata, complete fitted-state comparison, reference journals, ARMS table |
| `logic_check.py` | Exhaustive masks and post-query replay of every available PAPER_AB B vision verdict |
| `query_check.py` | Fitted-method checks on stored cache/inf keys, all-vision/gap histories, R6 nesting, blind veto and policy tail |
| `ARMS.md` | Generated 24-arm table and eight reference-cell mappings |
| `HANDBACK_SA.md` | This handback |
| `tests/exp/offline_search/rounds/r08/abl/test_judge.py` | Exhaustive masks, R6 parity, blind veto, deployed-B restrictions, name contract |
| `tests/exp/offline_search/rounds/r08/abl/test_specs.py` | Coverage, exact fields, reference convention, rejected non-guard changes, byte-exact array comparison |

Run artifacts: `arms_in.json`, `provenance.json`, `guard_arm_names.txt`, `arms.json`,
`manifests/eval500.json`, `fits/<guard-arm>.pkl`, `config/<guard-arm>.yaml`,
`config/matrix_<guard-arm>.yaml`, `prefit_logs/`, and `validation/` reports. These guard fits
total **2,550,312,989 bytes**. Foreign CLIP fits in the same `fits/` directory are excluded.

## Source and guard semantics

π0.5 B replicate 1 is copied from `r05_q1/arms_in.json`; GR00T B replicate 1 from
`r06_paper/arms_in.json`. Their method, kwargs, plugin arguments, client settings, full-model
flag and cost-ledger setting are checked against the deployed `arms.json` rows. Source rows,
deployed rows, source file SHA256s and source fit SHA256s are frozen in `provenance.json`.
Reference names match `r06/ops/paper_ab.py:arms`, the generator of `PAPER_AB.md`.

Only `name`, `method`, `kwargs.disabled_guards` and the fit pathname change. Restoring these
four fields reproduces the source row's serialized JSON exactly, including field presence and
argument ordering. `validation/source_diff.json` contains every permitted difference.
π0.5 retains the implicit one-block policy tail; GR00T retains the explicit block count of one,
resize 256 and replan 5. Owner cost-ledger settings are copied unchanged.

The new subclasses initialize through R6's deployed-B restrictions, canonicalize a collection
of guard names and OR their `BITS`: stuck=1, terminal=2, overtime=4, no_progress=8. They inherit
R6's `query` and `blind_step` without changing those functions. All diagnostic computation
remains active. Only verdict bits, lowest-bit reason, force-MISS and the final flag memo change.
Disabling stuck preserves the count needed by overtime. Disabling no_progress also bypasses
its blind LOOK veto while keeping progress history. `onlynp` has mask 7 and retains both the
no_progress verdict and its LOOK veto. The GR00T terminal-sign correction remains inherited.

R6 P2 made fresh trigger fits; S-A follows that convention. All nested fitted state, arrays
(dtype, shape and bytes), thresholds, calibration and registered libraries match each cell's
deployed B fit exactly, excluding the root method name and guard-switch attributes. Complete
pickle SHA256s differ from B because the class/spec, switch kwargs and fit timing metadata
change; the plugin requires matching spec/kwargs/cell metadata. No B source pickle was modified
or relabeled as an R8 fit. Both SHA256s and the fitted-state comparison are recorded per arm.

## Reproduce preparation and offline checks

Run from `/home/weiland/projects/openpi`. Every Python command below uses this prefix:

```bash
py() {
  taskset -c 10-13,54-57 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=.:src .venv/bin/python "$@"
}
RUN=/home/weiland/trace_runs/os_closed_loop/r08_abl
```

Preparation commands executed:

```bash
py -m exp.offline_search.rounds.r08.abl.make_arms
py -m exp.offline_search.rounds.r08.abl.prefit --parallel 7
py -m exp.offline_search.rounds.r08.abl.prefit --parallel 7 --resume
R8_EMIT_ROOT="$RUN" py -m exp.offline_search.closed_loop.ops.emit_arms \
  --run-root "$RUN" --spec "$RUN/arms_in.json"
```

Generator output:

```text
wrote 24 arms: 16 leave-one-out (500-demo) + 8 only-no-progress; 8 B source cells
```

The first prefit invocation completed 17 artifacts and rejected seven multi-guard method names
because they contained `+`, outside the method-name contract. The separator was corrected to
`_`; the contract is now included in the exhaustive unit test. The resume command completed
all seven missing fits and printed `PASS: 7 fresh prefits completed`. All 24 artifacts then
passed metadata and fitted-state validation. Initial failure evidence is retained under
`validation/initial_name_constraint/`; final prefit logs are in `prefit_logs/`.
The emitter exited 0 and wrote the arm/config files; its output is in `emit.log`.

Existing fits should be used as delivered. `--resume` skips existing artifacts; rerunning the
generator refreshes its frozen provenance/spec files and is a preparation step, not a launch.
No fit removal is needed. The common emitter merges arm entries and the validator scopes its
checks to the guard names so concurrent supplementary arms can coexist.

```bash
py -m pytest tests/exp/offline_search/rounds/r08/abl/test_judge.py \
  tests/exp/offline_search/rounds/r08/abl/test_specs.py -q
```

```text
..................................................                       [100%]
50 passed, 1 warning in 1.96s
```

The warning is the environment's `pynvml` deprecation warning on torch import. Full output:
`$RUN/pytest.log`. Tests are named explicitly to exclude the concurrent CLIP tests.

```bash
py -m exp.offline_search.rounds.r08.abl.validate
```

```json
{"PASS": true, "arms": 24, "leave_one_out": 16, "only_no_progress": 8, "source_fields_exact": 24, "artifact_metadata_and_config_pass": 24, "fitted_state_exact_to_B": 24, "reference_replicates": 48, "reference_pairs_each": 500, "official_manifest_pairs": 500, "official_manifest_sha256": "11643c404ab59393af09ba1589a14d643dee6600d8b1f1045c71ae81b4090433"}
```

Reports: `validation/validation.json`, `validation/source_diff.json`,
`validation/fit_sha256.txt`; console: `validate.log`. Cache YAMLs equal the source cache YAMLs
with only `trace` removed, matrices retain the source client settings, and plugin CLI metadata
matches every loaded artifact.

```bash
py -m exp.offline_search.rounds.r08.abl.logic_check
```

Final output (the preceding 24 lines report one PASS per B reference):

```json
{"PASS": true, "exhaustive_subset_flag_cases": 512, "B_references": 24, "logged_B_vision_verdicts": 252941, "variant_comparisons": 731713, "mismatches": 0, "protocol": "all logged B verdicts including historical repair attempts; fixed B history; post-query hook only", "limitations": "no visual-key reconstruction, counterfactual trajectories or closed-loop SR claims"}
```

Report: `validation/logged_verdict_replay.json`; console: `logic_check.log`. Per-file hashes,
observed firing combinations, independently recomputed B predicates and per-variant guard
fire/removal counts are included. Every disabled bit is absent and each enabled bit equals B.

```bash
py -m exp.offline_search.rounds.r08.abl.query_check
```

Final output (the preceding eight lines report one PASS per cell):

```json
{"PASS": true, "cells": 8, "all_vision_queries": 3634, "variant_query_comparisons": 16398, "alternating_queries": 1832, "explicit_blind_veto_comparisons": 48, "policy_tail_comparisons": 48, "R6_single_guard_query_comparisons": 10932, "mismatches": 0, "protocol": "init 0 of all 10 tasks; cache and inf keys; all-vision and alternating supplied histories; fixed observations/actions", "limitations": "sampled offline method check; no live B sensory reconstruction or counterfactual SR estimate"}
```

Report: `validation/query_replay.json`; console: `query_check.log`. This uses init 0 of every
task in both stored cache/inf streams, up to 24 decisions per episode, and tests real fitted
methods. Blind history keys are NaN in the gap checks. Payloads/confidence/non-verdict extras,
stuck counts and progress memos equal B on the same supplied histories. R6 singleton comparison
objects are loaded independently; no delivered fit is changed.

Additional inventory/diff command executed:

```bash
py - <<'PY'
import json
from pathlib import Path
r = Path('/home/weiland/trace_runs/os_closed_loop/r08_abl')
v = json.loads((r / 'validation/validation.json').read_text())
d = json.loads((r / 'validation/source_diff.json').read_text())
print('permitted changes:', sorted({x['path'] for row in d for x in row['differences']}))
print('non-guard differences:', sum(len(x['non_guard_differences']) for x in d))
print('full-pickle SHA matches source B:', sum(x['complete_pickle_sha_equal_to_B'] for x in v['artifacts']))
print('exact fitted-state matches source B:', sum(x['fitted_state_exact_to_B'] for x in v['artifacts']))
print('guard_fit_bytes:', sum(x['bytes'] for x in v['artifacts']))
print('guard names:', len(set((r / 'guard_arm_names.txt').read_text().splitlines())))
PY
```

```text
permitted changes: ['/kwargs/disabled_guards', '/method', '/name', '/plugin_args/10', '/plugin_args/8']
non-guard differences: 0
full-pickle SHA matches source B: 0
exact fitted-state matches source B: 24
guard_fit_bytes: 2550312989
guard names: 24
```

## Coordinator use and remaining limits

Use `guard_arm_names.txt` to select these 24 arms from the shared run root. Each is intended
for **500 official episodes** (10 tasks × inits 0–49) and separate pairing against each of its
three A and three B references. `manifests/eval500.json` provides the exact selection. No
`manifest` field was added to the B source specs, preserving non-guard field equality; the
coordinator can supply this selection to the topology runner or use the harness's full-test
default. There are 12,000 prospective guard-arm episodes, not executed by S-A.

All `arms.json` paths retain weilandserver conventions; no host-specific rewrite or runner
logic was added. The h100/timan108 runner and its path rewriting are another task. Remote
loading, policy inference, worker integration and closed-loop results are **unverified by S-A**.
No policy servers, LIBERO workers, chains, tmux sessions or remote commands were launched here.
All fits/tests/replays used the assigned CPUs with CUDA hidden. No existing process was killed.
No git state-changing command, `rm -rf`, or edits to `exp/step_diag/`, `exp/trace_dual/`,
`logs/session_handoff.md`, R6 code, the shared plugin or shared emitter were made.

Logged B replay is exhaustive for recorded vision verdicts, including old repair attempts; it
is a mask audit, not an SR sample. Standard B logs do not include the visual keys needed to
reconstruct their original sensory queries. The real-method check is a deterministic sample
from the separate stored streams. These checks prove fixed-history guard behavior and fitted
state parity; later histories, success rates, IR changes and the decision to remove guards
require the coordinator's real closed-loop round.
