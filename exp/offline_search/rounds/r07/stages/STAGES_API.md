# R7 shared stage API

Import `StageTable` from `exp.offline_search.rounds.r07.stages.stages`.
`StageTable.fit(library, *, manifest)` accepts a `LibraryView` or an array mapping.
Manifest geometry is authoritative: `exec_steps`, `H`, `act_valid_dims`,
`rs_valid_dims`, `gripper_dim`. Optionally supply the already frozen deployed A
as `manifest['_retrieval']`; this avoids refitting its representation. Without
it a LibraryView fits A on the library (current: kref=5; dense: kref=8).
A synthetic array mapping must supply that fit; state-only neighbors are never
substituted for A neighbors. No recordings or online outcomes are fit inputs.

Row-indexed arrays: `mode` (int8, -1 unknown, 0 lower learned command center,
1 upper), `event_near` (bool), `rows_to_event` (int32), `stage_run` (int32,
zero-based run within episode), `next` (int64, -1 invalid), `successor_valid`
(bool), `rs` (valid manifest coordinates), `task_id`, `episode`, `step`,
`success`. Failed rows have unknown mode/run/distance; their event flag is false
and they contribute to online unknown mass. `event_occupancy` / `h` map integer
task IDs to successful-row event-neighborhood occupancy.

Segmentation exactly follows E1: median executed-head gripper command,
deterministic two-means initialized at quartiles (min/max fallback), 3-row
majority on interior positions with original endpoints. A change at row j marks
j-1, j, j+1 event-near. `rows_to_event` counts to the next change strictly after
the current row, or to the boundary just after the episode's last recorded head.
Unknown rows return -1. Successors require same task/episode and step+1.

`online(rows, weights, cmd_mode=None)` returns `mode_mass` ([mode0, mode1]),
`mode0_mass`, `mode1_mass`, `mode`, `unanimous`, `event_mass`,
`min_rows_to_event`, `unknown_mass`. Every member, including zero-weight
members, participates in unanimity and minimum distance. An optional learned
`cmd_mode` additionally requires agreement with that mode. Diagnostic masses
normalize weights; action weights remain untouched.

`advance(rows, blocks)` returns real successor chains, propagating -1;
there is no terminal clamping. `displacement(current, anchor, rows, weights,
blocks)` returns `(D_delta, supported)`; `deviation(current, rows, weights)`
returns absolute normalized current-state RMS for CT. Both use valid,
nonconstant proprioceptive dimensions only. `state_scale` is the pooled
successful-library coordinate standard deviation; `state_active` masks
constants (unit scale placeholder). No hand-set state-unit floors.

Valve calibration follows E3's frozen-fit candidate-LOEO recipe: successful
source rows at A's two-row cadence, source episode excluded from the original
A candidate set, exact top-k/kernel rule including the early step-zero metric.
PCA and metric stay frozen. Full-support source and all-member chains are
required at both checks in the reference two-block commitment. The calibration
sample is max(D_delta at one block, D_delta at two blocks). `valve_radius_row`
/ `valve_radius` are p95 with NumPy `higher`; `valve_radius_episode` is p95 of
per-episode maxima, matching E3's episode convention (not an inverse-length
weighted row quantile). `valve_radius_by_task` provides task row-p95 diagnostics;
SF uses the pooled default. `calibration` reports counts and convention.
`deviation_p75[task]` is p75 of absolute state residual at held-out anchors;
`deviation_occupancy[task]` is its strictly-above occupancy for CT's entry tilt.
These use every supported held-out retrieval, independently of future-chain
censoring. Unknown/absent task calibration must be handled as hard by callers.

`save(path)` / `StageTable.load(path, *, library=None, manifest=None)` store a
versioned pickle envelope with payload SHA256. `fingerprint` hashes all stage
input arrays and manifest geometry; passing library+manifest on load validates
it. `retrieval_fingerprint` separately identifies frozen A arrays. Load only
trusted local artifacts. Arrays are read-only after fit/load.

Composition: use this same table for wrist eligibility and call-stage tilt.
SF's separate `FollowExtension` component (in `c1_follow.methods`) accepts a
cache anchor and BlindQueryView. The call controller must preserve cache versus
policy provenance, invoke extension only on cache HIT anchors, and clear the
cache anchor on a MISS; policy-tail cadence stays owned by the call controller.
