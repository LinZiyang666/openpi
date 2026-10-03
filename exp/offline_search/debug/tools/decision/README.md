# R8 decision profiles

These nine offline tools consume `debug.reader` and accepted episode attempts.
They do not import a serving policy, invoke an encoder, run a simulator, or write
to capture files. Reports contain JSON, narrow CSV tables, and a short Markdown
summary. NaN data holes become JSON null with explicit coverage/status, never zero
evidence. Identity conflicts are errors; absent scientific inputs are unavailable.

Run from the repository root, with the assigned CPU/environment prefix:

```bash
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src \
  .venv/bin/python -m exp.offline_search.debug.tools.decision.provenance \
  --run-root /path/to/run --arms ARM1 ARM2 --out /tmp/r8_S4/provenance --procs 4
```

Replace `provenance` with any tool below. `--procs` defaults to **4 bounded I/O
threads**, not child Python processes, so the three-process sandbox limit remains
respected. `--bootstraps` defaults to 1000 and `--seed` to 0. `--adapter p3v2`
selects the development reader; missing R8 capabilities remain unavailable.
`--metadata FILE.json` supplies explicit model/action metadata overrides, recorded
in each report. This can attest a P3 model or supply frozen action scales, valid
channels, gripper command threshold, or library size. No arm-name/model/task-name
parsing or gripper-polarity guess is used.

`library_size` comes from the arm manifest's `r8` section (usually
`arm_spec.r8`). For read-only captures without a published catalogue, build
one separately with S3's tool and pass its scratch root:

```bash
# Use the affinity/environment prefix above for both commands.
.venv/bin/python -m exp.offline_search.debug.catalog \
  --run-root /tmp/r8_S4/catalogs --cells pi05_l10_current
.venv/bin/python -m exp.offline_search.debug.tools.decision.provenance \
  --run-root RUN --arms ARM --catalog-root /tmp/r8_S4/catalogs \
  --out /tmp/r8_S4/provenance --procs 4
```

The scratch catalogue requires the builder's `rows.json`, matching
model/suite/library identity, parquet hash, and captured store fingerprints.
Its provenance is included in reports. Normal runs use `reader.catalog()`.
The reader handles old single-file and new per-process server records, normalizes
subset `init` from `task_uid`, and preserves `orig_init_state_idx`. Reports include
its `read_issues` as `input_diagnostics`, including skipped final JSONL lines
without a newline. Such lines are unpublished even if their JSON looks complete.

| Tool | Inputs beyond joined decisions/episodes | Principal tables |
| --- | --- | --- |
| `provenance` | Exact rows/weights, immutable library catalog | Mode/stage/unknown mass, demo ESS, cross-stage kernels, next-row tail mass |
| `divergence` | Served/cache chunks, `policy_shadow`, manifest valid channels; optional `policy_draws` | Offset curves by stage, age, library, source; executed-prefix flags; noise floor |
| `follow_vs_look` | Served blind/follow chunk, applied count, `shadow_look` | Same-observation applied-head gaps, kernel overlap/progress/demo agreement, weighted row follow-gap map |
| `camera_shadow` | Full `shadow_look`, wrist/third `camera_shadow` | Camera action/kernel/stage agreement and score ratios; all augmented libraries |
| `stage_ledger` | Actual live dispatches, camera mode, applied counts, prices | Additive stage work, contiguous visits, per-episode and pooled request/control IR |
| `call_value` | Fresh eligible anchor, actual conditional p, realized treatment, coin; **pre-assignment** stage | Support audit, first supported entry HT excursions, unnormalized probability-shift derivatives, ESS |
| `exposure_hazard` | Anchor lottery support/probability vector/drawn E/coin | Adjacent E=1−0 and E=2−1 package contrasts by pre-assignment stage and planned extension age |
| `churn` | A reference, candidate; optional A replicates/placebos | Task/init paired gains/losses, symmetric churn, net bias, paired lever-minus-control differences |
| `trigger_vs_onset` | `--onsets` S6 episode labels, explicit `--triggers` diagnostic fields | Episode/stage hit denominators, alert lead times, false alerts, actionable lead |

Examples:

```bash
# Prefix every Python command with the affinity/environment above.
.venv/bin/python -m exp.offline_search.debug.tools.decision.churn \
  --run-root RUN --arms A LEVER --reference A --replicates A_SEED2 --placebos SHIFTED_A \
  --out /tmp/r8_S4/churn --procs 4
.venv/bin/python -m exp.offline_search.debug.tools.decision.trigger_vs_onset \
  --run-root RUN --arms ARM --onsets /tmp/physical/episodes_forensics.csv \
  --triggers os_sf_valve_fire os_c_stall_call os_c3_dev_entry \
  --out /tmp/r8_S4/triggers --procs 4
```

S6 onset inputs accept its `episodes_forensics.csv`, the corresponding
`forensics.json` table, a row-list JSON, or JSONL. Joins use `arm` and
`episode_key`, retaining rule version, confidence and `truth_validated`. A hit is
an alert in the 20-control pre-onset window; actionable lead is at least five
controls. Per-stage hit denominators include unhit onset episodes exposed to that
stage. `shadow_*` alerts are labelled diagnostic scope.
Known call/entry triggers are evaluated only at fresh anchors and the online
follow valve at blind checks and vision rows carrying its current `blind_extras`.
The current check overrides stale nested diagnostics. Custom fields default to all decisions;
`--metadata` can declare a `trigger_domains` map (`anchor`, `blind`, `all`).
Failed episodes without an onset time are unavailable negatives, not false-alert
denominators; a successful episode or an explicit `no_onset_certified` label is
needed for that denominator.

Action metrics use only declared valid channels. A frozen `action_scale` (or
`action_std`) gives library-sigma units; otherwise normalized action units are
reported. Gripper flips require both `gripper_dim` and `gripper_threshold`; the
absolute gripper error can be available without the threshold. Motion includes
all remaining valid channels. Padding never contributes to errors. Recorded
valid horizon/served length constrains served comparisons; otherwise array
horizon validity is explicitly unavailable. Unexecuted offsets remain proposals.
On the hashed policy-draw sample, the independent base shadow plus three draws
give pair MSE. Adjusted cache/served MSE subtracts **half** pair MSE, retaining
negative finite-sample estimates. This is sampling variance, not policy quality.

The call estimand uses `Z/p − (1−Z)/(1−p)` with actual conditional propensity
after overrides. Forced p=0/1, carried tails, missing/inconsistent coins and
missing eligibility are withheld. First-entry scores average over the original
episode population, with unreached episodes contributing zero, and separately
report the reached-entry denominator. Summed anchor scores estimate a local
probability-shift derivative; they are never divided by eventual stage duration.
No nuisance baseline is fitted. When explicit labels are absent, descriptive
stages use dominant weighted catalogue neighbour stage mass. Missing positive
mass or tied maxima stay unknown with a reason. These are catalogue proxies,
with their full stage mass retained, rather than simulator-truth labels.
Stage moderators must be explicitly certified `stage_pre`/`pre_stage`; absent
assignment-time certification is reported separately. Missing joins,
gapped decisions or accepted infrastructure errors withhold causal estimates.

Final success, remaining work/controls and predicate changes at 5/10/20 active
controls are supported endpoints. Success/failure/time-limit terminals absorb
physical windows, including the stock `step_cap` failure; `exception` and
infrastructure errors exclude all endpoints and withhold original-population
causal estimates. Lottery contrasts also include next-look distance, EE motion,
valve statistics and stall entry. Recorded diagnostic endpoints need an
exact control clock and field; missing drift, valve or stall measurements are
withheld with source reasons and endpoint coverage. Robot/demo and object-relative
drift require aligned library state/physics and object-frame references; unsourced
diagnostic scalars do not replace those references. EE motion is not renamed drift.
No subsequent look is censored rather
than assigned a fabricated distance. Lottery scores are based on anchor
assignment, never selected age survivors.

`follow_vs_look` derives member spread from the captured library `action.npy`
artifact, verifies its SHA, and reads rows through a read-only numeric memmap.
Spread is the weighted motion RMS about the member mean over the applied prefix,
using the same valid dimensions and units as action disagreement. Native tails
use `chunk_offset`; real extensions require declared native-tail or successor-head
alignment. Unsupported positive-weight rows, missing alignment, and missing stores
have explicit reasons. Member spread can remain available without a deferred
look; the disagreement ratio requires both proposals and nonzero spread.
Source splits preserve `cache_tail` versus real `follow` from S1's records.
Churn with only a reference arm produces an unavailable report with its episode
denominator. Trigger JSONL inputs also report skipped unfinished tails.

All intervals resample task/init clusters **within fixed tasks**. Repeated
encounters remain together. A stage with fewer than 30 supported clusters is
marked exploratory. No within-task replication means unavailable uncertainty.
Lottery interval families use Bonferroni correction. Unobserved treatment
branches and missing supported endpoints do not produce effect estimates.

Measured full/head prices are π0.5 `.152/.848` and GR00T `.148/.852`. π0.5
measured wrist/completion prices are `.0646/.0502`. The labelled historical R4
ledger keeps full/head prices but uses `.055198/.049890` camera prices. Recorded
owner costs remain a third ledger. Stage sums retain unknown exposure; request
IR is K/N and control IR is 5K/actual active controls. Settling and shadows do not
enter deployment work. Missing counts make the affected ledger unavailable.

Tests (CPU only):

```bash
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src \
  .venv/bin/python -m pytest -q exp/offline_search/debug/tools/decision/tests \
  -o cache_dir=/tmp/r8_S4/pytest_cache --basetemp=/tmp/r8_S4/verification
```
