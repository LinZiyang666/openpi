# R8 S6 handback — physical / forensics / segmentation tools

Implemented the ten assigned CPU-only offline CLI tools. No serving/client,
method, schema, reader, fixture or other coder-owned source was edited. No GPU,
remote operation, simulator, server, worker, chain, tmux, reserved port, process
termination or git operation was used. Python commands used CPUs
`10-13,54-57`, one-thread BLAS/OpenMP, disabled CUDA and bytecode. Scratch and
development products are under `/tmp/r8_S6/`.

## Files

All implementation and tests are in `exp/offline_search/debug/tools/physical/`:

- Entry points: `selfcheck.py`, `forensics.py`, `grasp_audit.py`,
  `drop_audit.py`, `paired_diverge.py`, `twin_divergence.py`,
  `blind_drift.py`, `segmentation_bench.py`, `episode_card.py`, `arm_rollup.py`.
- Shared analysis: `__init__.py`, `common.py`, `adapters.py`, `audits.py`,
  `pairs.py`, `segmentation.py`, `cards.py`.
- Instructions and contracts: `README.md`.
- Tests: `tests/conftest.py`, `tests/test_physical.py`,
  `tests/test_pairs_segmentation.py`, `tests/test_drift_cli.py`,
  `tests/test_references.py`.
- This handback: `debug/handbacks/HANDBACK_S6.md`.

## What is implemented

`selfcheck` verifies control contiguity, logged seed, authoritative geom/body
IDs, settled free-object/table contacts, post-settle references and redundant
before/after states when legacy data contains both. Complete MuJoCo name arrays
preserve unnamed ID slots. Contact support follows body parents; articulated
children are not incorrectly required to touch the table.

`forensics` emits T1 episode labels, per-predicate evidence and onset brackets,
plus T2 per-control and per-decision stages. Decision products separate the
majority stage of the applied controls from the state observed before the
decision. All labels carry their rule version and remain unvalidated heuristics.
Missing destination geometry produces unknown release, not a guessed drop.
The post-settle baseline is the last wait's after-state. Near/carry radii use
E3's discovery-only moving/lifted/closed-command calibration when supported;
explicit adapter configuration freezes its values.

`grasp_audit` uses the pose before the close command, transforms position and
yaw into the object's frame, and reports measured width, source/kernel,
40-control lift outcome, offset AUROC and discovery pose clusters. Truncated
unsuccessful lift windows are censored. Successful policy references require
matching task/init/object, environment seed and reset hash.

`drop_audit` enumerates recurrent carry losses away from a known destination,
reports open-command versus closed-command slip candidates, measured width,
lift height, acceleration, source/kernel and named contacts/normal force.

`paired_diverge` requires matching reset, seed, settling schedule and settled
state/actions. It exports raw robot/object/rotation divergence, separate
event-aligned curves and outcome/stage transitions. `twin_divergence` adds the
GR00T-only bit-identical-prefix gate. Frozen library sigma diagnostics cover
action perturbation and state gaps at 1/2/4/8/16/32 decisions. Calibrated object
onset needs an explicitly supplied frozen envelope; otherwise it is unavailable.

`blind_drift` traverses real library successors with full member support and
uses admitted, fingerprint-matched backfill for control-rate robot and
object-relative drift. With no usable backfill, it computes robot-state sigma
drift at actual pre-decision checks only. Between-check physics is unavailable.

`segmentation_bench` implements G0's median-head/two-means/three-row majority,
elapsed-control baseline, stop q10/q25 × dwell 1/5/10, waypoint ε/2/ε/2ε and
1/5/10-control resolution, and exact penalized mean-change candidates. Library
G0 uses normalized captured served heads, avoiding wire/library unit mismatch.
Outputs include interval hierarchy, matched-count boundary baselines, F1 at
±1/5/10 controls, translation and SO(3) reconstruction, causal decision scores,
prefix replay checks and next-20-control risk. Discovery p80 exposure thresholds
and independent hash tie-breaks are frozen on holdout; realized exposure,
AUPRC, discovery-calibrated Brier score, lead, equal-task sensitivity and
task/init-cluster intervals are reported by arm. Risk exposure stops at first
onset; success competes with onset, and unresolved tail windows are censored.
Fits exclude inits 30–49. Simple scores do not invent semantic posteriors.

`episode_card`/`arm_rollup` produce static HTML and PNG with source/truth strips,
actual logged triggers, XY paths, object heights, measured finger aperture,
issued command, labels/onsets, captured decision images when available and
arm label/stage/source ledgers. Missing physical evidence gets a labeled
HTML/PNG placeholder. Paired transition CSVs can be included in arm rollups.

## How to run

From `/home/weiland/projects/openpi`:

```bash
S6_PY=(taskset -c 10-13,54-57 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
       MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
       PYTHONPATH=.:src MPLCONFIGDIR=/tmp/r8_S6/matplotlib .venv/bin/python)
"${S6_PY[@]}" -m exp.offline_search.debug.tools.physical.forensics \
  --run-root /path/to/R8_RUN --arms A Policy --procs 1 --out /tmp/r8_S6/T1T2
"${S6_PY[@]}" -m exp.offline_search.debug.tools.physical.grasp_audit \
  --run-root /path/to/R8_RUN --arms A Policy --reference-arms Policy \
  --procs 1 --out /tmp/r8_S6/T3
"${S6_PY[@]}" -m exp.offline_search.debug.tools.physical.segmentation_bench \
  --run-root /path/to/R8_RUN --arms A Policy --library-root /path/to/STORE \
  --onsets-file /tmp/r8_S6/T1T2/episodes_forensics.csv \
  --procs 1 --out /tmp/r8_S6/segments
"${S6_PY[@]}" -m exp.offline_search.debug.tools.physical.twin_divergence \
  --run-root /path/to/R8_RUN --arms A FollowLottery --reference-arm A \
  --library-root /path/to/STORE --envelope /path/to/frozen_envelope.json \
  --procs 1 --out /tmp/r8_S6/twins
"${S6_PY[@]}" -m exp.offline_search.debug.tools.physical.blind_drift \
  --run-root /path/to/R8_RUN --arms A FollowLottery \
  --library-root /path/to/STORE --backfill-root /path/to/BACKFILL \
  --procs 1 --out /tmp/r8_S6/drift
```

Every tool supports the shared reader API, `--p3v2`, `--limit`,
`--adapter-config`, JSON/CSV/Markdown output and explicit `--procs` (default 4
bounded threads, one Python process). Pairing/plotting/library traversal are
serial. Run one cell per library-specific command. The README lists all ten
commands and adapter fields.

S4 can use either `T1T2/episodes_forensics.csv` or `T1T2/forensics.json` with
its `trigger_vs_onset --onsets`. The accepted attempt key, arm, onset bracket,
confidence, validation status and rule version are present.

## Tests and development evidence

Final test command:

```bash
"${S6_PY[@]}" -m pytest exp/offline_search/debug/tools/physical/tests \
  -q --tb=short -p no:cacheprovider
```

Final output: **57 passed in 10.34 s**. Coverage includes all ten CLIs on S3's
`make_synthetic_arm`, strict ID joins, outcome taxonomy, settled references,
ID/name catalogs and aliases, object quaternions, named forces, censored grasp
windows, recurrent drops, shared-seed/reset pairing, bit-exact twin prefixes,
library sigma joins, normalization-unit separation, real successor support,
backfill certification/fingerprint rejection, holdout isolation, future-suffix
invariance, one-to-one boundary matching, reconstruction and HTML/PNG artifacts.

Ruff `check --select F --no-cache` on the owned directory: **All checks passed**.
`ast.parse(..., feature_version=(3,8))`: **22 Python files passed**. This is a
grammar check, not a Python 3.8 dependency/runtime claim; no client code changed.

Real-data development used `reader.p3v2_adapter` over the read-only R6 pilot.
Initial CLI smoke: two accepted GR00T A episodes, both physical records available.
The combined scratch smoke `/tmp/r8_S6/pilot_smoke.py` loaded two A and two P10
episodes (task 5, inits 0/1, r0): **4 episodes, 753 controls, 4 available forensic
records, 4 grasp attempts, 0 carry losses, 2 admitted same-seed pairs, 56/56
causal prefix replays passed**. Both pairs first differ in action and simulator
state at active control 10; their settle prefix is identical. P3 contact naming
is unavailable for all four episodes. Discovery carry radius was .1333234855 m.
Reader load took 131.74 s; the complete analysis/report smoke ended at 137.26 s.
Products, HTML/PNG and `integration_summary.json` are under
`/tmp/r8_S6/pilot_smoke/`. The P3 source alias `cache_blind` was subsequently
normalized to schema `cache_tail`; final tests exercise the current sources.
This small pilot is development evidence, not a population estimate or holdout.

## Coordinator validation and precise integration needs

No GPU or remote validation is necessary for these offline tools. On collected
R8 smoke data, run `selfcheck`, `forensics`, `episode_card` and the benchmark
using the CPU command prefix above. Check the actual IDs/catalog conventions
and review representative grasp, drop, release and fixture cards. Apply E3's
human validation requirement before calling the labels ground truth (50 failures
and 20 successes stratified by heuristic label; target ≥85% per major label).
Check the locked holdout fit hashes and reported exposure on inits 30–49.
Use a frozen discovery/library envelope for physical-onset twin claims.

**S2 backfill integration:** current `debug/backfill/replay.py` supplies PASS,
row IDs and state error, but its episode `backfill` metadata does not bind the
records to the catalog's `lib_sha`. Before using that backfill, have the
coordinator add `backfill.lib_sha` from the same frozen library fingerprint as
`catalog/rows.parquet.lib_sha` to each replay's `episode.json` at producer time.
The exact consumer gate is `Library.__init__` in `blind_drift.py`. It rejects
missing/mismatched fingerprints and reports rejected paths; no producer-owned
file was edited by S6. Also preserve the producer's full `library_rows` in
decision order, PASS admission, and absence of capture errors.

## Known limits

- Heuristic event labels and center-distance grasp windows require human review;
  they do not establish intent or failure causality. Predicates without object
  semantics get fixture labels. Unknown destination geometry remains ambiguous.
- Generic adapters are ready for explicit RoboCasa/MetaWorld goal/sub-predicate
  records; their native simulator extraction remains the environment producer's
  work. Non-LIBERO scales/conventions must be declared.
- Calibrated physical divergence needs a frozen envelope. Unadmitted/mismatched
  backfill cannot supply object-relative drift. Robot-only checks do not establish
  between-check excursions. R8 smoke and real admitted backfill are not yet tested.
- The exact mean-change dynamic program has quadratic control-count cost.
  No HSMM, learned visual stages or calibrated semantic posteriors are claimed.
- Bootstrap intervals use task/init clusters. Small pilot intervals are exploratory;
  no new stage or routing rule is selected, and no method success claim is made.

## Fix round 1

Read `rounds/r08/REVIEW_1.md` fully and implemented the S6 fixes. This section
supersedes the earlier statement that real R8 smoke data had not been tested.
Only S6's physical package and this handback were edited. All Python commands
used CPUs `10-13,54-57`, the prescribed single-thread environment variables,
disabled CUDA and bytecode writes. No GPU, remote runtime, server, socket,
git operation, or capture mutation was used.

### Changes and regression coverage

- **B1 support:** collision bodies map to their nearest free-joint ancestor.
  Free objects touching any non-free body seed a contact graph; support
  propagates through touching free objects. Stacks, drawers, and fixtures are
  accepted; floating cycles and internal contacts cannot ground themselves.
  The existing CSV check name `resting_objects_contact_table` is retained;
  `support_rule="contact chain to any non-free body"` and supported body IDs
  disclose the corrected semantics. Tests cover reversed contact order,
  transitive stacks, fixture/drawer support, child collision bodies, and
  ungrounded cycles.
- **Shared layout/init:** use the updated S3 reader for `decisions*.jsonl` and
  `meta*.json`, including historical filenames. Select the metadata belonging
  to an episode's recorded server process via `record_meta`, retain all process
  metadata, and disable reader cache writes. Every CLI reports `input_capture`
  with reader `read_issues`, metadata filenames, all `writer_stats*.json`,
  terminal-outcome counts, and snapshot availability. Regression fixtures cover
  both layouts, source init 42 versus task-UID subset init 0, competing process
  metadata, malformed and valid final JSONL records without a newline, and
  unchanged input files. Unterminated tails are skipped and reported by the
  shared reader. No shared reader file was edited by S6.
- **Outcomes:** journal success is authoritative. `step_cap` is an observed
  timeout failure (`success=false`, `outcome="timeout"`) while retaining the
  descriptive physical failure label. `exception`/journal errors produce an
  invalid outcome and unavailable physical results, excluding them from fits,
  references, and binary success denominators. Late unresolved timeout grasp
  windows remain censored. Regressions exercise conflicting success metadata,
  exceptions, errors, calibration/segmentation exclusion, and pairing rejection.
  Invalid exception backfills are rejected and reported without aborting other
  analysis, even if their replay admission was incorrectly marked PASS.
- **Smoke-discovered reporting fixes:** single-arm pair tools previously
  produced empty tables; they now report one unavailable comparison per
  episode and zero matched pairs. Missing drift catalogs previously appeared
  as generic invalid kernels; the tool now reports missing catalog evidence.
  Added `blind_drift --catalog-file rows.parquet` so an immutable scratch
  catalog can be supplied without editing a read-only capture. Both fixes
  have regression tests.

Changed files under `debug/tools/physical/`: `common.py`, `adapters.py`,
`selfcheck.py`, `forensics.py`, `audits.py`, `pairs.py`, `paired_diverge.py`,
`blind_drift.py`, `segmentation.py`, `segmentation_bench.py`, `grasp_audit.py`,
`drop_audit.py`, `episode_card.py`, `arm_rollup.py`, `README.md`, and new
`tests/test_fix_round1.py`; plus `debug/handbacks/HANDBACK_S6.md`.

### Validation

Owned suite: **73 passed in 12.72 s** (previously 57). Command:

```bash
taskset -c 10-13,54-57 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=.:src MPLCONFIGDIR=/tmp/r8_S6/matplotlib .venv/bin/python \
  -m pytest exp/offline_search/debug/tools/physical/tests \
  -q --tb=short -p no:cacheprovider
```

Ruff `check --select F --no-cache` passed. All **23 Python files** passed
Python 3.8 grammar parsing. All ten physical CLIs completed successfully on
the smoke arm. A transient concurrent S3 fixture edit caused a NameError
(`orig_init_state_idx=init`); its owner corrected it to `ep // 3` before the
final passing rerun. S6 did not edit that shared fixture.

The ten-tool real-data run used
the current, plain `r8_pi05_l10_50_A_smoke` arm. The failed `__capfail*` arms
were not analyzed. Outputs are under `/tmp/r8_S6/fix_round1/smoke/<tool>/`.
The source contains 20 accepted attempts, 7,067 controls, 1,380 decisions,
14 successes, six `step_cap` failures, and snapshots for every episode.
There were no reader tail warnings in this complete capture.

| Tool | Actual smoke output and availability |
|---|---|
| `selfcheck` | 100 available checks passed, 20 redundancy checks not applicable; all 80 free-object references supported; zero failed checks |
| `forensics` | All 20 episode rows, 7,067 control stages, and 1,380 decision stages available; labels: 14 success, four grasp_miss, one unknown_release, one fixture_not_done |
| `grasp_audit` | 56 available pre-grasp poses/windows; 39 lifted, 17 did not lift; policy-reference comparisons unavailable because no reference arm was supplied |
| `drop_audit` | Three carry-loss rows with real named contacts and normal forces; mechanism classification unavailable because their destination frames were not captured |
| `paired_diverge` | 20 explicit unavailable rows, zero matched pairs: no second arm |
| `twin_divergence` | 20 explicit unavailable rows, zero matched pairs: no second arm; this pi05 arm also cannot establish GR00T twin claims |
| `blind_drift` | With scratch catalog and frozen library: 537 available measured robot-state checks; 2,143 intervening controls unavailable without certified backfill, 150 unsupported-kernel decisions; object-relative drift unavailable |
| `segmentation_bench` | One fit; 6,273 segment rows, 1,680 boundary comparisons, 19,320 causal decisions, 180 reconstruction rows; all 280 prefix checks passed; 14,518 observed risk windows and 42 censored windows; 18 discovery/2 holdout episodes |
| `episode_card` | 20 available episode cards; index + 20 HTML pages, 20 timeline PNGs + 40 captured-camera PNGs; visually inspected a generated timeline |
| `arm_rollup` | 20 episode rows, 174 stage/source ledger rows, one HTML/PNG; 6,867 active controls |

The immutable current library was read from
`/home/weiland/trace_runs/offline_search_store/library/pi05_l10/current`.
S3's catalog builder wrote only to
`/tmp/r8_S6/fix_round1/catalog_run/catalog/pi05_l10_current/` (2,640 rows,
frozen R7 stages available). A second segmentation run using normalized
library heads produced 6,271 segment rows and the same remaining availability
counts; all 280 prefix checks passed. Its outputs are in
`/tmp/r8_S6/fix_round1/smoke_library/segmentation_bench/`.

Reproduce with the `S6_PY` command prefix documented above:

```bash
"${S6_PY[@]}" -m exp.offline_search.debug.catalog \
  --run-root /tmp/r8_S6/fix_round1/catalog_run --cells pi05_l10_current
"${S6_PY[@]}" -m exp.offline_search.debug.tools.physical.blind_drift \
  --run-root /home/weiland/trace_runs/os_closed_loop/r08_smoke \
  --arms r8_pi05_l10_50_A_smoke --procs 1 \
  --library-root /home/weiland/trace_runs/offline_search_store/library/pi05_l10/current \
  --catalog-file /tmp/r8_S6/fix_round1/catalog_run/catalog/pi05_l10_current/rows.parquet \
  --out /tmp/r8_S6/fix_round1/smoke/blind_drift
```

Other tools use the same run root/arm/output convention; pair tools add
`--reference-arm r8_pi05_l10_50_A_smoke`. The optional library segmentation
run adds the same `--library-root`. No GPU or remote validation is required.

### Remaining evidence gaps

The unavailable drop destinations are capture gaps, not inferred placements:
the smoke controls have no `destination_pos`/`destination_names` for static
stove/microwave regions. If the coordinator wants these classified, S2 should
capture those numeric region frames or declare explicit adapter destination
positions/aliases; S6's adapter already accepts them. No producer-owned file
was changed. A second arm is needed for physical pairs and a certified library
backfill is needed for object-relative/intermediate-control drift. Automatic
physical labels still require the previously specified human audit.
