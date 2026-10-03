# S4 handback — R8 decision-level profile tools

Implemented all nine assigned offline CLIs under `debug/tools/decision/`.
They use the shared reader API, accepted attempts and decision IDs. Each CLI
writes JSON, CSV and a short Markdown summary. Required scientific fields are
unavailable when missing; no serving/model/simulator imports are used.

## Files owned and written

All implementation/test files are under
`exp/offline_search/debug/tools/decision/`:

- `__init__.py`, `README.md`
- `common.py`: strict population/identity joins, nested diagnostics, field
  availability, metadata/action masks, sparse augmentation alignment, fixed-task
  task/init bootstrap, bounded CLI I/O threads, report output.
- `kernels.py`, `metrics.py`, `outcomes.py`, `contrasts.py`: weighted immutable
  catalogue provenance, proposal/noise metrics, observed excursion endpoints,
  original-population HT contrasts and ESS.
- `provenance.py`: mode/stage/unknown mass, merged demo ESS, cross-stage kernels,
  unsupported next-row tail mass and zero-weight structural member counts.
- `divergence.py`: served/cache–shadow-policy offset curves, blind age, stage,
  recorded library size, source and applied-prefix masks; sampled policy–policy
  floor and half-floor-adjusted MSE, including negative values.
- `follow_vs_look.py`: closed-loop blind/follow–shadow-look applied-head gaps,
  kernel/demo/progress agreement, direction cosine and a kernel-weighted library
  row follow-gap map.
- `camera_shadow.py`: wrist/third–full proposal/kernel/stage comparisons and
  score ratios, including both pure-policy augmentation libraries.
- `stage_ledger.py`: live-count measured/historical-assumption/recorded-owner
  ledgers by stage, contiguous stage visit and episode; pooled/equal-episode IR.
- `call_value.py`: effective-propensity and coin audit; supported first-entry
  excursions and unnormalized probability-shift derivatives, both branch ESSs.
- `exposure_hazard.py`: adjacent lottery E package contrasts by pre-assignment
  stage and planned extension block, without selecting realized age survivors.
- `churn.py`: paired flips/gains/losses, symmetric churn/net loss bias, A replicate
  and placebo comparisons, direct paired lever-minus-control uncertainty.
- `trigger_vs_onset.py`: S6 onset-file joins, pre-onset leads, trigger/stage hit
  denominators, actionable lead and false-alert denominators.
- `tests/__init__.py`, `tests/test_profiles.py`: reader-contract CPU fakes and
  shared synthetic fixture/reader integration, including subprocess tests of all
  nine CLI artifact sets.

This handback is the only file written outside that ownership directory.
Scratch and reports stayed under `/tmp/r8_S4/`. No GPU, network, remote process,
server, worker, chain, simulator, tmux, restricted port, Git operation, prohibited
CPU or `tests/review_tests/` access was used. No subagents were spawned.

## Run commands

From `/home/weiland/projects/openpi`, define the required prefix in Bash:

```bash
S4PY=(taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
      MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
      PYTHONPATH=.:src .venv/bin/python)
"${S4PY[@]}" -m exp.offline_search.debug.tools.decision.provenance \
  --run-root RUN --arms ARM1 ARM2 --out /tmp/r8_S4/provenance --procs 4
```

Replace `provenance` with any tool name above. `--procs` explicitly bounds arm
I/O **threads**, default 4; it launches no child Python processes. Numerical
libraries remain single-threaded. Bootstrap default is 1000; `--bootstraps 0`
disables intervals. `--metadata FILE.json` provides explicit metadata overrides
and records them in the report. `--adapter p3v2` selects the development adapter.

For paired controls and S6 labels:

```bash
"${S4PY[@]}" -m exp.offline_search.debug.tools.decision.churn \
  --run-root RUN --arms A LEVER --reference A --replicates A_SEED2 \
  --placebos SHIFTED_A --out /tmp/r8_S4/churn --procs 4
"${S4PY[@]}" -m exp.offline_search.debug.tools.decision.trigger_vs_onset \
  --run-root RUN --arms ARM --onsets /tmp/physical/episodes_forensics.csv \
  --triggers os_sf_valve_fire os_c_stall_call os_c3_dev_entry \
  --out /tmp/r8_S4/triggers --procs 4
```

S6 input may also be `forensics.json`, a row-list JSON or JSONL. Joins require
episode identity; confidence/rule version/validation status are retained.
Known call/entry triggers apply at anchors, the online follow valve at blind
requests. Custom domains can be specified via `trigger_domains` in metadata.

## Tests and real-data checks

Final test command:

```bash
"${S4PY[@]}" -m pytest -q exp/offline_search/debug/tools/decision/tests \
  -o cache_dir=/tmp/r8_S4/pytest_cache --basetemp=/tmp/r8_S4/test_final3
```

Result: **29 passed in 14.39s**. Tests cover masks/padding, private-draw floor,
sparse/shuffled augmentation identities, zero-weight members/demo ESS, partial
terminal heads, camera/library coverage, additive measured/assumed prices,
effective versus nominal propensity, forced/ineligible/carried decisions,
coin audits, unreached-population zeros, occupancy-score sums, missing outcomes
and joins, cluster replication, lottery support/clock endpoints, physical pair
seed checks, onset uncertainty and missed-onset stage denominators. Tests invoke
at most one child CLI at a time, so there are at most two Python processes.

An AST parse with `feature_version=(3,8)` passed for all **17 Python files**.
This is a grammar check, not an assertion that local pandas runs on Python 3.8.
`python -m ruff check` was attempted but Ruff is not installed in this venv;
no package installation was performed.

Real P3 development checks used
`/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot` read-only:

| Tool/arm | Result | Runtime |
| --- | --- | --- |
| `stage_ledger`, `r6p3v2_groot_sp_50_A_r0` | 459/459 decisions; N=459, V=235, M=0, 2261 active controls; measured work=34.78, request IR=.0757734, control IR=.0769129 | 25.91s |
| `divergence`, same A arm | 459/459 decisions; sampled draw floor unavailable in P3 | 31.45s |
| `call_value`, `r6p3v2_groot_sp_50_dose25_r0` | 260 supported fresh anchors / 514 decisions, 20 episodes; pooled first-entry SR score −.06667, 100-bootstrap CI [−.73667,.56833], CALL/CACHE ESS 5/15; explicitly exploratory | 38.36s |
| `churn`, A r0 versus A r1 | 20 paired outcomes withheld because environment seeds differ | 35.17s |

Commands match the general CLI with `--adapter p3v2 --procs 1`, and outputs are
in `/tmp/r8_S4/p3_<tool>/`. The price check used
`--metadata /tmp/r8_S4/groot_metadata.json`, containing `{"model":"groot"}`.
An early ledger run used the development reader's incorrect pi05 model fallback;
it was detected and the report replaced by the explicitly attested GR00T run.
Legacy coin spelling `assignment.uniform` is now handled without reseeding or
reconstructing a coin. These 20-episode checks are development evidence, not R8
500-episode claims.

S6 integration: generated a shared synthetic arm at `/tmp/r8_S4/s6_smoke`, ran
its offline `physical.forensics`, and consumed its actual
`physical/episodes_forensics.csv` through `trigger_vs_onset`. Both commands
exited 0. Synthetic physical labels lacked declared gripper conventions and
were unavailable; trigger analysis correctly retained that status. Unit tests
separately exercise finite onset/lead/hit arithmetic and the S6 JSON/CSV field
contract.

## Coordinator validation and integration requests

No GPU or remote execution is required by S4 tools. After S3 produces deferred
GPU shadows and S6 produces labels, run these **CPU** commands on the smoke run:

```bash
for tool in provenance divergence follow_vs_look camera_shadow stage_ledger call_value exposure_hazard; do
  "${S4PY[@]}" -m "exp.offline_search.debug.tools.decision.${tool}" \
    --run-root RUN --arms ARM1 ARM2 --out "/tmp/r8_S4/smoke/${tool}" --procs 4
done
```

Then run the paired/trigger commands above with actual smoke arm names and S6
labels. GPU capture/shadow parity and remote simulator validation belong to S1–S3;
these tools do not launch them. Check camera and action coverage, exact additive
stage totals, supported randomized denominators, and source/scale metadata before
interpreting any stage effect.

Required input details, without edits to other owners' files:

1. For stage-conditioned causal results, expose **pre-assignment** `stage_pre`
   or `pre_stage` in decision records/`debug_record()` or a joined reader view.
   Existing server adapters primarily expose diagnostic gate features; when a
   pre-assignment stage is absent, causal reports use the explicit `unknown`
   stratum. Retrospective stage labels never silently moderate a coin.
2. To obtain action-sigma and gripper-flip metrics, supply frozen `action_scale`
   or `action_std`, `gripper_dim` and `gripper_threshold` via the model/action
   manifest or explicit metadata JSON. Shared fixtures currently omit gripper
   conventions. Valid action masks are mandatory; GR00T padding is excluded.
3. The early P3 reader model fallback was not attested for GR00T. Until S3
   populates `manifest.model` from authoritative run/startup metadata, S4 pricing
   requires `--metadata` to attest it. No model is guessed from an arm name.
4. A second pure-policy augmentation library needs its own catalogue for
   provenance/stage agreement. The current single-bank `catalog()` API cannot
   resolve the second bank; action/kernel comparisons still work and its
   catalogue-dependent metrics remain unavailable.

## Interpretation limits

- Shadow disagreement and follow gaps are diagnostic opportunities, not rescue
  effects or physical failure labels. The row follow-gap map is kernel-weighted
  attribution, not a row-specific causal effect. Member spread is reported only
  when directly recorded; it is not fabricated from unavailable member chunks.
- Provenance tail mass describes missing next catalogue rows, separately from
  logged unsupported mass. It does not claim an executed action failed. Weighted
  cross-stage mass is separate from zero-weight structural member mixing; demo
  ESS retains its known-mass denominator, and incomplete progress mass withholds
  the progress statistic.
- Lottery comparisons estimate adjacent assigned E packages at the anchor under
  the recorded continuation, not survival-conditioned causal hazards. Future
  drift/valve/stall endpoints require recorded fields and exact clocks; EE motion
  is not object-relative drift. Missing endpoints are explicitly withheld.
- Call and lottery intervals resample task/init within fixed tasks, retaining
  repeated encounters. Fewer than 30 supported clusters is exploratory; no
  within-task replication yields unavailable uncertainty. Forced probabilities,
  unseen branches or missing supported endpoints do not produce effect estimates.
- No later look is censored for the next-look endpoint. Unknown failure onset
  times are not false-alert negatives. S6 heuristic confidence is preserved.
- R8 full-arm/smoke GPU shadows have not been run by S4; complete R8 empirical
  coverage and hardware timing remain coordinator validation.

## Fix round 1

Read `rounds/r08/REVIEW_1.md` fully, the updated schema, and the shared reader/
catalogue implementation. This section supersedes the earlier statements about
blind-only SF valves, directly recorded member spread, and untested R8 smoke
coverage. All edits remain in the S4 ownership directory and this handback;
scratch is under `/tmp/r8_S4/`. Coordinator capture/ops changes were preserved.

Changes:

- **M6:** `step_cap` absorbs fixed physical endpoint windows as a terminal
  failure. `exception` or infrastructure error invalidates every outcome,
  including work, controls, future diagnostics and full physical windows.
  Accepted exception attempts withhold original-population causal estimates;
  they are never scored as ordinary failures or silently removed. Churn and
  trigger analysis also exclude their outcomes while retaining denominators.
- **M7:** the SF valve domain includes vision rows carrying `os_sf_valve_fire`
  in `blind_extras`. That current check takes precedence over stale nested
  diagnostics. Blind checks remain included, and initial/ordinary vision rows
  without that check do not require a nonexistent flag. Both episode leads and
  stage hit rates use the same extraction.
- **m9, stages:** when explicit labels are absent, descriptive stages are the
  dominant weighted catalogue neighbour stage. Full stage mass, source and
  reason remain in decision tables; tied or incomplete labels stay unknown.
  These catalogue proxies do not assert simulator truth. Missing
  `stage_pre`/`pre_stage` certification has a separate reason, and causal stage
  moderation stays in the unknown stratum rather than claiming assignment-time
  information the producer does not attest.
- **m9, spread:** `library.py` reads the captured library's `action.npy` through
  a read-only numeric memmap, verifies its captured SHA, and computes weighted
  RMS spread about the member mean over the applied prefix. Native tails use
  `chunk_offset`; real follows require native-tail or successor-head alignment.
  Zero-weight unsupported members do not enter the variance. Padding and any
  declared gripper channel are excluded consistently with action disagreement.
  Missing artifacts, mismatched hashes, unsupported positive-weight rows and
  missing alignment have reasons. Spread is still exported when shadow looks
  are absent; its disagreement ratio is withheld then or when spread is zero.
- **m9, drift/size:** client EE motion and predicate changes are computed from
  physics. Demo-relative robot/object drift stays unavailable without aligned
  library state/physics and object-frame references; arbitrary unsourced
  diagnostic scalars are not substituted. Per-endpoint coverage and source
  reasons are exported even when an arm has no randomized assignments, and
  missing contrast endpoints retain those reasons. `library_size` comes from
  `arm_spec.r8` / `r8`, with explicit metadata overrides still recorded.
- **Shared layout/init/source:** the tools consume S3's reader for
  `decisions*.jsonl` and `meta*.json`, old names included, subset init
  normalization and separate original indices. Reader `read_issues`, including
  skipped unfinished final lines, appear in report `input_diagnostics`. Onset
  JSONL uses the same reader and has separate diagnostics. Source splits keep
  S1's `cache_tail` versus real `follow`; SF diagnostic presence never relabels
  the source. Churn with only one reference now emits explicit unavailability.

Files changed/added: `common.py`, `kernels.py`, `outcomes.py`, `contrasts.py`,
`library.py` (new), `follow_vs_look.py`, `call_value.py`, `exposure_hazard.py`,
`churn.py`, `trigger_vs_onset.py`, `README.md`, and
`tests/test_fix_round1.py` (new), all under `debug/tools/decision/`.

Final regression command:

```bash
"${S4PY[@]}" -m pytest -q exp/offline_search/debug/tools/decision/tests \
  -o cache_dir=/tmp/r8_S4/pytest_cache --basetemp=/tmp/r8_S4/fix1_final_tests2
```

Result: **45 passed in 22.23s**. New regressions cover stock `step_cap` endings,
exception with no error string, current vision-row valve versus stale flags,
catalogue stage mass/ties/missing labels and causal certification, nested R8
library size, native-tail/successor-head spread and padding/gripper masks,
changed/missing store artifacts, spread without augmentation, catalogue identity
and captured fingerprints, drift reasons, source separation, two process files,
legacy/new subset-init semantics, truncated JSONL diagnostics and one-arm churn.
The process-layout integration checks the real reader and a subprocess CLI.
Both malformed tails and complete JSON without a newline are skipped/reported.
Python 3.8 grammar parsing passed for **19 Python files**. No GPU/remote check is
needed for these fixes; no server, worker, chain, simulator or network was run.

### Real smoke, read-only

Used only the current plain arm, never the failed `__capfail1/2` attempts:
`/home/weiland/trace_runs/os_closed_loop/r08_smoke/runs/r8_pi05_l10_50_A_smoke`.
It contains **20 accepted episodes, 1,380 verified decision joins, 14 successes,
6 step_cap failures**, 693 cache anchors and 687 cache tails. There are no read
issues or subset-init conflicts. The frozen current library has 2,640 rows /
50 demonstrations. Built S3's catalogue into scratch, because the capture is
read-only and has no published catalogue:

```bash
"${S4PY[@]}" -m exp.offline_search.debug.catalog \
  --run-root /tmp/r8_S4/fix1_catalog --cells pi05_l10_current
```

The builder found the frozen R7 stage table, checked its library fingerprint,
and wrote `catalog/pi05_l10_current/rows.parquet` and `rows.json`. S4's new
`--catalog-root` option validates model/suite/library identity, the parquet SHA,
and all shared captured store fingerprints (10 matched artifacts here), recording
that provenance in each report. It does not write into the capture tree.

S6's actual smoke `episodes_forensics.csv` was copied read-only from
`/tmp/r8_S6/fix_round1/smoke/forensics/` to
`/tmp/r8_S4/fix1_smoke/s6_onsets.csv`; heuristic confidence and unvalidated truth
status were retained. Final commands for **all nine** tools:

```bash
for tool in provenance divergence follow_vs_look camera_shadow stage_ledger call_value exposure_hazard churn trigger_vs_onset; do
  "${S4PY[@]}" -m "exp.offline_search.debug.tools.decision.${tool}" \
    --run-root /home/weiland/trace_runs/os_closed_loop/r08_smoke \
    --arms r8_pi05_l10_50_A_smoke --catalog-root /tmp/r8_S4/fix1_catalog \
    --onsets /tmp/r8_S4/fix1_smoke/s6_onsets.csv --procs 1 --bootstraps 100 \
    --out "/tmp/r8_S4/fix1_smoke/${tool}"
done
```

All commands exited **0**, and every JSON/CSV/Markdown artifact set was checked.
Reports/logs are under `/tmp/r8_S4/fix1_smoke/`; `availability.json` consolidates
coverage and reasons, and `input_audit.json` records the six failed terminal
endpoint checks and catalogue provenance.

| Tool | Smoke availability | Runtime |
| --- | --- | --- |
| provenance | 1,380/1,380 decisions; neighbour mode/stage/demo/tail tables available | 2.067s |
| divergence | unavailable: no `policy_shadow` augmentation | 1.513s |
| follow_vs_look | gaps 0/687 without `shadow_look`; **member spread 687/687**, 7 stage curves available | 1.843s |
| camera_shadow | unavailable: no full/camera shadow augmentation | 2.745s |
| stage_ledger | 1,380/1,380 decisions; 20 episode ledgers, 7 stages, 116 visits | 1.937s |
| call_value | no randomized support: 693 anchors lack eligibility/assignment; 687 tails excluded | 4.164s |
| exposure_hazard | 0/693 supported anchors: A has no lottery assignment | 3.861s |
| churn | unavailable: only one arm, no distinct candidate/reference or A replicate | .018s |
| trigger_vs_onset | 0/80 episode×trigger checks: this A arm has no SF/CALL/shadow valve flags; real S6 labels loaded | 2.662s |

All 1,380 decisions receive descriptive catalogue stages; assignment-time stage
certification remains 0/1,380. The ledger has N=1,380, V=693, M=0,
6,867 active controls, measured/owner work **105.336**, request IR
**.076330435**, control IR **.076697248**. Stage work sums to the arm total.
Success, remaining work/controls, EE displacement and predicate changes at
5/10/20 controls are available for **1,380/1,380** decisions, including all six
final `step_cap` decisions whose 20-control horizons exceed the remaining five
controls. Next-look distance is available for 1,346/1,380; the remaining 34 are
censored. Robot/demo and object-relative drift, valve and stall endpoints are
unavailable with source reasons. No augmentation, randomization or comparison
data was fabricated to make unsupported outputs available.

Coordinator follow-up is data production only: publish a keyed catalogue (or
use scratch `--catalog-root`), run S3's deferred augmentation, and supply the
appropriate randomized/SF/paired arms for their tools. Physical drift requires
admitted aligned library references; causal stage moderation requires certified
assignment-time labels. S4 needs no edits in other owners' files.
