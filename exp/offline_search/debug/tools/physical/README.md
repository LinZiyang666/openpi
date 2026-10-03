# R8 physical profiles (S6)

These tools consume `debug.reader` accepted attempts. They import no model,
serving code, or simulator. Each command writes `<tool>.json`, CSV tables,
`<tool>.md`, and `adapter_calibration.json`. Cards also write static HTML/PNG.
`--procs` defaults to 4 bounded analysis threads; commands use one Python
process. Pairing, plotting and library traversal are serial. Outputs are
offline artifacts and are never read by the serving methods.

From the repository root, use this command prefix (a Bash array):

```bash
S6_PY=(taskset -c 10-13,54-57 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
       MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
       PYTHONPATH=.:src MPLCONFIGDIR=/tmp/r8_S6/matplotlib .venv/bin/python)
"${S6_PY[@]}" -m exp.offline_search.debug.tools.physical.forensics \
  --run-root /path/to/run --arms A Policy --out /tmp/r8_S6/forensics --procs 1
```

All ten entry points accept `--run-root`, `--arms`, `--out`, `--procs`,
`--adapter-config`, `--limit` (accepted episodes per arm), and `--p3v2`.

| Module | Products and extra arguments |
|---|---|
| `selfcheck` | Full-ID geom/body integrity, settled free-object contact chains to any non-free body, seed, contiguous controls, successor redundancy |
| `forensics` | `episodes_forensics.csv`, `controls_truth.csv`, `decisions_truth.csv`; T1 labels/onset intervals and T2 descriptive stages |
| `grasp_audit` | Pre-close pose in the object frame, yaw, aperture, 40-control lift outcome, source/kernel, offset AUROC, paired policy reference error and discovery pose clusters; `--reference-arms Policy` |
| `drop_audit` | Every recurrent carry loss away from a known destination: opening command vs closed-command slip candidate, aperture, height, acceleration, source/kernel and named contacts/normal force |
| `paired_diverge` | `--reference-arm A`; strict reset/seed/settle matching, first action/state difference, robot/object position and rotation curves, separately event-aligned curves, outcome/stage transitions |
| `twin_divergence` | Same pair interface; requires GR00T and a bit-identical physical prefix before the perturbation |
| `blind_drift` | `--library-root STORE --backfill-root BACKFILL`; optional `--catalog-file rows.parquet` for a scratch shared catalog; exact successor chains and control-rate robot/object-relative drift with certified backfill; otherwise robot-state sigma drift at decision checks |
| `segmentation_bench` | R7 G0, elapsed-control baseline, stop q10/q25 × dwell 1/5/10, geometric ε/2/ε/2ε × resolution 1/5/10, penalized mean-change candidates; boundary F1 at ±1/5/10, matched-count uniform baseline, SO(3)/position reconstruction, prefix checks, exposure-matched next-20-control risk, AUPRC/Brier/lead and cluster intervals. `--library-root STORE` and `--onsets-file episodes_forensics.csv` are optional. |
| `episode_card` | One HTML/PNG per episode plus index, decision/source/truth strips, actual logged trigger markers, XY paths, object heights, measured width, heuristic onset, captured decision images |
| `arm_rollup` | One HTML/PNG per arm, episode table, active-control stage/source ledger and label counts; `--pairs-file pairs.csv` adds paired transitions |

Pair tools accept `--library-root STORE` for E4 normalized action/state sigma
diagnostics and outcome-flip AUROC at decision lags 1/2/4/8/16/32. They accept
`--envelope envelope.json` for calibrated physical onset. That JSON contains
`eef_gap_threshold`, `object_gap_threshold` (native units) and the coordinator's
frozen calibration provenance. Without it, exact object differences remain
observable but calibrated physical onset is **unavailable**. Differences between
trajectories are not failure onset or causal effects.

## Environment contract

The adapter resolves goal objects, destinations, predicates, gripper polarity,
quaternion order and geometric units. It supports complete MuJoCo `*_names`,
`geom_bodyid`, `body_parentid`, `joint_bodyid/joint_type` maps, record-style
`body/geom` catalogs, movable-body aliases, and explicit destination poses.
MuJoCo body quaternion capture is wxyz; LIBERO proprioception is xyzw. All
object-frame pose calculations honor the declared conventions.

LIBERO defaults are E3's **offline heuristic** constants and are recorded in
every forensic episode row. A task's carry radius is calibrated from moving,
lifted, closed-command object samples of successful discovery episodes only:
p95 distance + adapter padding, clipped to adapter bounds. Explicit
`near_radius` or a saved `near_radius_by_task` freezes that choice. No task text
or name selects a rule. References use the after-state of the last settle
control; control zero is never substituted for a missing reset reference.

RoboCasa/MetaWorld adapters can provide `entities.goal_objects` records:
`{predicate: <vector index>, object: <movable name>, destination: <frame name>,
relation: <sub-condition>}`. Fixture-only predicates omit `object`. An adapter
JSON can supply this list as `goal_objects`, `destinations`, `environment`,
`gripper_dim`, `close_sign`, `gripper_threshold`, `obj_quat_order`,
`eef_quat_order`, `lift_height`, `near_radius`, `place_radius`,
`disturbance_radius`, `tail_controls`, `stall_path`, `oscillation_path`,
`oscillation_ratio`, `lift_window`, `width_weights` and
`robot_qpos_indices`. Unspecified non-LIBERO conventions are unavailable.
Native RoboCasa/MetaWorld goal extraction remains the environment producer's
job; these analysis tools do not instantiate either simulator.

## Evidence limits and joins

Missing fields, NaN predicates, gaps, bad mappings, incomplete lift windows,
missing destinations and uncertified replays have explicit status/reason.
All consumers use the shared reader's `decisions*.jsonl` / `meta*.json` files;
report summaries include per-process `writer_stats*.json`, metadata filenames,
and `read_issues`, including skipped final JSONL lines without a newline.
The subset `init` comes from `task_uid`; source indices remain in
`orig_init_state_idx`. Physical consumers disable reader cache writes.
`step_cap` is a valid timeout failure; `exception` and journal errors are invalid
and excluded from physical labels, fits, references, and outcome denominators.
The support graph assigns collision children to free roots and propagates
support from any non-free body, including drawers and fixtures. Floating
contact cycles do not support themselves. The historical CSV check name
`resting_objects_contact_table` is retained with the explicit `support_rule`.
Grasp windows ending at a timeout without a lift are censored, not failures.
Unknown destination frames produce `unknown_release`, not an invented drop.
Grasp policy references require the same task/init/object, seed and reset hash.
The paired tools reject seed/reset/settling mismatches. P3 naming is always
unavailable; the retained numeric physics is still usable. Legacy
`cache_blind` is normalized to `cache_tail`, with the old spelling retained.

S2 backfill must carry `backfill.admission="PASS"`, no capture errors,
`backfill.library_rows` in decision order, and `backfill.lib_sha` matching the
immutable catalog. Missing/mismatched fingerprints reject object physics.
All kernel members need real successor support; no terminal clamping or member
dropping occurs. Robot-only fallback uses `state_norm` vs the frozen library's
`rs.npy` at the latest pre-decision check. Between-check physics is unavailable
without backfill; it is never interpolated.

`episodes_forensics.csv` and `forensics.json` are the S4 `trigger_vs_onset`
inputs. They contain `episode_key`, `arm`, `onset_control`,
`onset_decision_id`, `onset_interval`, `onset_confidence`, `rule_version`,
`truth_validated=false` and status/reason. The interval is an after-control
observation bracket, not a claim of visually certified one-control timing.
`decisions_truth` separates the majority stage of applied controls from
`truth_stage_pre` and `pre_control_idx`; only the latter precedes the decision.

## Segmentation split and causality

Fits and physical radius calibration exclude within-task inits 30–49. G0 uses
the library's successful median executed-head commands when supplied; otherwise
its source is explicitly successful discovery trajectories. Three-row majority
is retrospective. The causal version waits for the confirming row. The
library G0 compares those centers to captured normalized served heads; it never
compares normalized library commands to wire-unit commands. Terminally partial
executions retain the complete captured head for that comparison. Portable
candidate inputs contain no object pose, contact, predicate or eventual outcome.
Scores at control t use only samples before t, with a fixed fit. Prefix
truncations and arbitrary future perturbations are checked in tests.

Discovery's p80 `(score, independent hash tie coin)` is frozen for the 20%
exposure comparison, and realized holdout exposure is reported. Elapsed-control
scores use the discovery length scale, never the current episode's final length.
Risk exposure ends at the first independent heuristic onset. Success is an
absorbing competing endpoint; unresolved timeout windows are right-censored.
Flagged/unflagged discovery event rates supply Brier probabilities. Risk tables
are reported by arm, avoiding a pooled policy/cache mixture; uncertainty samples
task/init clusters, keeping repeated encounters together. Semantic posterior
probabilities and entropy are unavailable for these simple event-score
candidates rather than fabricated from their scores.

Automatic labels require the E3 human annotation audit before any ground-truth
or routing claim. The change-point implementation solves the PELT penalized
mean-shift objective by exact dynamic programming, without unsafe pruning;
runtime is quadratic in controls. No HSMM or visual segmenter is claimed.

## Tests

```bash
"${S6_PY[@]}" -m pytest exp/offline_search/debug/tools/physical/tests \
  -q --tb=short -p no:cacheprovider
```

Tests include all ten CLIs against S3's schema-valid fixture, strict joins,
physical taxonomy tapes, settled references, complete/legacy naming, quaternion
frames, carry losses, censoring, pair invariance, discovery/holdout isolation,
suffix-invariant causal scores, geometric reconstruction, one-to-one boundary
matching, frozen-library backfill admission, and HTML/PNG production.
