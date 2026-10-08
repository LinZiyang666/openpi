# R15 loader fixes

## 1. Served fit hashes after H100 path relocation

The B loader compared decision `r14.fit_sha256` only with the frozen source
artifact SHA. H100 sync rewrites artifact paths and serializes the relocated
pickle, so its served bytes have a different SHA even though the fitted
parameters are unchanged. The latest `h100_sync/plan.json` retains only the
last synced arm and cannot supply hashes for every preceding arm.

The loader now accepts exactly two branches, recorded for each arm in
`fit_hash_verification.observed_sha256_branches`:

- `frozen_source`: the observed hash equals the frozen source SHA.
- `h100_relocated`: for a sealed H100 arm, recompute
  `assets.relocated_fit(pickle.loads(frozen_source_bytes), root, frozen_store,
  actual_remote_base)` using this root's `h100_sync/plan.json.remote_run`.
  The remote run must belong to that root; its parent must be `runs`.
  Use `pickle.dumps(..., protocol=4)` only when relocation changed paths,
  exactly as `assets.build_plan` does. Otherwise retain the original bytes.
  Compute each relocation in a fresh CPU interpreter, matching per-arm sync:
  reusing an interpreter across NumPy artifact unpickles can change later
  pickle memo/identity bytes and produce a different SHA. The child inherits
  hidden CUDA and single-thread settings and uses the required CPU affinity.
  Accept only the SHA of those deterministic bytes, never a plan's claimed
  fit hash. Record the source SHA, recomputed SHA and actual remote run.

The source bytes still must match the cap snapshot, root seal and pool
contract. All other loader checks, accepted-attempt semantics, LOEO audits,
cost reconciliation and decision/statistical rules remain unchanged.
`analyze.before_served_hash_fix.py.txt` preserves the registered pre-fix source.
The source adapter separately binds this authorized loader update and the
unchanged relocation implementation; the frozen protocol files are untouched.

Synthetic tests cover relocated C10 and all five cell arms at sizes 100 and
500, stale last-arm-only plans, a nondefault remote base, the source branch,
unrelated hashes even if a plan claims them, wrong remote roots, and unchanged
pickles. An AST restoration test proves every other `load_cell` statement is
identical to the pre-fix loader.

`verify_fit_hashes` audits real B source fits and only the `r14.fit_sha256`
fields in snapshots of decision logs. It reads no client journals, summaries,
SR, outcome tables or test-A data, and never calls `apply`. Pending arms with
no decisions are reported separately from observed served-hash verification.
Evidence is written to `served_fit_hash_verification.json`.

Validation: all 14 targeted synthetic tests passed (`served_fit_tests.xml` and
`served_fit_isolation_tests.xml`). The real hash-only audit verified 64/64
frozen source fits and 61/61 arms with observed hashes: π0.5 32/32 H100
relocations, GR00T 29/29 original-source hashes. Three GR00T `off` arms have
no decision hashes yet: Spatial-200, L10-500 and Spatial-500. Their frozen
sources match; served-hash observation is pending. No SR was computed and
no real apply was run. `served_fit_validation.json` records the exact source
and test hashes, unchanged original functions, and frozen-protocol checks.

## 2. Test-A stages emitted from per-model B tables

`analyze_test_a apply --stage` previously compared a six-cell model stage's
`input.json` against all twelve frozen B cells. It now requires identical
level/uniform cell keys and exactly the six registered cells of one model,
then compares both tables with the full frozen B tables restricted to those
cells. Missing cells, mixed-model subsets and changed levels or uniform
settings fail before any A outcomes are read. A legacy twelve-cell stage
still works with its original complete-table check.

Model runs write `test_a_pi05.json`, `TEST_A_pi05.md`, `REPORT_pi05.md` and
`ANALYSIS_TEST_A_HANDBACK_pi05.md`, or the corresponding `_groot` files.
The JSON and measurement report label single-model pooled rows partial.
The frozen forty-comparison family is retained even for a model stage.

`analyze_test_a combine --stages PI05_STAGE GROOT_STAGE` requires disjoint,
complete model stages covering all twelve cells. It audits the raw A arms
again and runs the original full calculation in canonical cell order,
rather than averaging separate model summaries. It writes the canonical
`test_a.json`, `TEST_A.md`, `REPORT.md` and `ANALYSIS_TEST_A_HANDBACK.md`.
Pooling remains separate for library sizes 100/200/500: four model/suite
cells, 2,000 outcomes per arm, and 1,000 original suite/task/init clusters
per size. Shared C10 references are never pooled across sizes. Bootstrap
draws are shared across arms and models within each suite as before.

`load_a`, every numerical helper, `_calculate_group` and `calculate` remain
AST-identical to the archived frozen analyzer in
`test_a_stage_original_analyze_test_a.py.txt`. The test-A-local source
registration in `test_a_stage_adapter.json` binds the exact authorized source
SHA and verifies that proof before invoking the existing protocol verifier.
Only that validated source is presented with its frozen bytes to the
upstream verifier; provenance and stability checks still record and rehash
the actual current bytes. All other protocol inputs and B output hashes,
table consistency, chronology, templates and A loader checks remain enforced.
Frozen protocol files, the B analyzer and the emitter are unchanged.

Validation uses synthetic stages and episodes only, including complete-model
acceptance, incorrect subsets, output isolation, model/full cell equivalence,
combined/full numerical and report equivalence, and preserved rejection checks.
All 23 tests passed, plus a separate pooled-equivalence run at 5,000 bootstrap
draws (`test_a_stage_tests.xml`, `test_a_pooled_equivalence_tests.xml`).
No real test-A apply, outcome read, or evaluation launch was performed.

## 3. Test-A served paths and hashes after H100 relocation

The A loader's startup `--os-fit-artifact` check used the sealed local path,
and its decision `r14.fit_sha256` check used only the frozen source SHA.
H100 serves the path produced by `assets.remap` and the pickle produced by
`assets.relocated_fit`. Both served-value comparisons are now handled by
`AServedFitVerifier`, for uniform, task-descent and shared pure-C10 arms.

The accepted branches, recorded per arm in `fit_hash_verification`, are:

- `frozen_source`: startup uses exactly the sealed local fit path and the
  decision hash equals that fit's verified frozen source bytes.
- `h100_relocated`: a sealed H100 arm uses exactly that fit's deterministic
  path under the root's actual `h100_sync/plan.json.remote_run`. The inherited
  B verifier checks root identity, remote path structure, source store and
  exact source bytes; `assets.remap` derives the expected served path. The
  decision SHA must equal the deterministic relocation for that same branch.

No prefix-only path acceptance or plan-supplied fit SHA is used. Startup
path/hash branches must agree. Frozen source bytes are checked against the
root seal before either branch is accepted. `source_path`, `relocated_path`,
`observed_path_branches`, `observed_sha256_branches` and the replayed
`relocation_arm_order` are recorded alongside the source/relocated hashes.

A sync may process both new arms in one interpreter. NumPy dtype/pickle
state makes the second arm's relocated byte hash differ from a standalone
relocation, despite identical fitted arrays. For a multi-arm plan, replay its
ordered prefix through the target arm in a fresh pinned CPU interpreter,
using only sealed arm definitions and SHA-verified frozen fit bytes. The
replay uses the same unpickle, `assets.relocated_fit`, protocol-4 pickle and
unchanged-byte behavior as `assets.build_plan`. Unknown or duplicate plan
arms and mismatched stores are rejected. Single-arm sync retains the existing
B fresh-interpreter calculation. The plan's declared hashes are never trusted.

All remaining fit/path checks refer to local frozen metadata: cap sources,
adapter fit path/SHA, sealed arm manifests, run manifests and selection
bindings. They remain unchanged. The shared pi05 R14 C10 references currently
use local serving despite their intended H100 label, so their exact source
branch remains valid. GR00T's local test arms use the same source branch.

`analyze_test_a.before_served_fit_fix.py.txt` preserves the previous source.
An AST restoration test removes only the new verifier initialization/audit
and restores the two served-value checks, proving every other loader statement
is identical. All ten original numerical helpers and calculation functions
remain AST-identical to the frozen analyzer. Source registration binds this
authorized loader edit, the unchanged B verifier and relocation implementation.
Frozen protocol, measurement/statistical rules, emitter and B analyzer are
unchanged.

`verify_test_a_fit_hashes` scans only startup fit paths and decision fit SHA
fields in fixed-length log snapshots, plus sealed definitions and source fit
bytes. It reads no journals, summaries, SR or outcome tables and never invokes
`apply`. Shared references are audited once each; unstarted or missing
observations are reported separately. Append-only running tails are deferred.

Real audit: 28/28 frozen sources match, with 24/24 observed arms passing both
paths and hashes. Six pi05 roots contribute 12 H100 arms; the four observed
GR00T roots (L10/Spatial at 100/200) contribute eight local arms. All four
shared C10 references are local and pass. GR00T's four size-500 arms have no
observations yet; their frozen source bytes match. Exact evidence is in
`test_a_served_fit_verification.json`; no outcomes or real apply were read/run.

All 37 synthetic tests passed (`test_a_served_fit_tests.xml`), including local
pi05/GR00T, H100 mirror/run/store mappings, shared C10/uniform/task-descent,
batch pickle state, unrelated paths/hashes, mismatched path/hash branches,
unknown batch arms, wrong remote roots and an audit that blocks every outcome
read. The existing per-model and full pooled measurement tests also passed at
the production 5,000 bootstrap draws. `test_a_served_fit_validation.json`
records exact source hashes, preserved functions and audit evidence.

## 4. Per-model freeze chronology after a later full B analysis

The chronology guard used `provenance.json.analyzed_at_utc` for every stage.
That timestamp belongs to the later full B reporting run; it can follow A
completions even though a model's partial run froze its tables before A.

For a complete model stage, the guard now uses
`provenance_<model>.json.analyzed_at_utc`. It requires B-pool provenance for
exactly that model's six registered cells, verifies the primary partial
levels/uniform bytes against that provenance, and checks those tables against
the same cells of full B. If full B provenance is later than the partial
freeze, `partial_equivalence.json` must be SHA-bound to full B provenance,
have `PASS: true`, list that model in `checked_models`, and contain matching
byte-identical SHA checks for both primary model tables.

The strict ordering predicate is preserved for every new A arm: its own
model's freeze must be earlier than its earliest accepted terminal record.
Equality, a late partial freeze and invalid comparison values still fail.
Shared, pre-existing pure-C10 arms remain excluded as before. Combined mode
checks pi05 and GR00T separately; a GR00T freeze is compared only with GR00T
A records. A legacy single full stage still uses the original full
provenance timestamp and needs no partial provenance.

JSON provenance now records `freeze_before_test` with the provenance source
and SHA, chosen timestamp, frozen partial table hashes, bound equivalence
hash and earliest new-arm completion per model. Full B output validation,
stage bindings, templates, identities, accepted-attempt logic, served-fit
checks, cost reconciliation, measurement and statistics remain unchanged.
`analyze_test_a.before_freeze_order_fix.py.txt` preserves the prior source;
every existing function/class except `apply` remains AST-identical, including
the complete A loader and serving verifiers. Frozen protocol files, B analyzer
and emitter remain unchanged.

The remaining stage preconditions were reviewed: complete model cell sets,
full B table consistency, separate output names, model-specific references,
costs and fits, and canonical combined pooling retain their existing checks.
Metadata-only verification binds both real model freezes to their full B
tables/equivalence proof in `test_a_freeze_metadata_verification.json`.
It reads no real A outcomes and does not assert real A completion ordering;
that check runs when the coordinator applies.

All 44 synthetic tests passed (`test_a_freeze_order_tests.xml`), covering early
partial/later full acceptance with bound equivalence, late/equal partial
rejection, missing or invalid equivalence, independently timed model stages
in `combine`, and preserved legacy ordering and pooled measurements.
`test_a_freeze_order_validation.json` records exact source hashes and proves
all 28 existing functions/classes other than `apply` are unchanged. No real
A apply or outcome/completion-record read was performed.
