# Closed-loop debug mode — data schema v1 (`osdebug.v1`)

Owner: R8 coordinator (2026-09-30). This is the contract between the server observer, the client capture, the
receiver, the reader, the deferred-shadow job and every analysis tool. Change it only by bumping the version and
writing the change here. Units are the environment's native units unless stated; arrays are little-endian numpy.

## 0. Principles

- **Observe, never act.** Debug code copies data after the live decision is fixed. It never calls the policy, the
  encoder, the method or any RNG on the live path; it never mutates method state; a capture failure changes a
  telemetry status, never an action. With debug off every code path and log byte is as before.
- **One schema for every method and model.** Fields that do not apply carry `status` ∈ {`available`,
  `not_applicable`, `not_sampled`, `unsupported`, `error`} plus a reason, never a silent null/zero.
- **Binary arrays in compressed NPZ blocks** (`np.savez_compressed`, no object arrays, no pickle); small scalars and
  identifiers in JSONL. Entity names, library rows and configs are stored once in catalogs, never per tick.
- **Lossless**: images uint8 exact wire bytes; simulator state float64; no truncation of long or failed episodes.

## 1. Identity

| Name | Definition |
|---|---|
| `campaign` | run-root basename, e.g. `r08_main` |
| `arm` | arm name |
| `task_uid` | journal convention `<arm>:eval:<task_id>:<init>` |
| `attempt` | journal attempt number (int ≥ 1) |
| `dispatch_gen` | client dispatch-fence generation (P3 `dispatch_fence.py` semantics), persisted in file and stream mode |
| `episode_key` | `sha256(task_uid)[:24] + "_a" + str(attempt)` (P3 convention) |
| `decision_seq` | 0-based index of the request within the episode attempt, assigned by the client |
| `decision_id` | `f"{episode_key}:{dispatch_gen}:{decision_seq}"` |
| `control_idx` | 0-based index of every `env.step` in the attempt, including settling/wait controls |

Request envelope: when debug is on the client adds `obs["__debug__"] = {"v": 1, "decision_id", "episode_key",
"decision_seq", "dispatch_gen", "task_uid", "attempt", "t_client_send"}`. The server **always** pops `__debug__`
before any transform (a no-op when absent). With server debug on it echoes `result["__debug__"] = {"v": 1,
"decision_id", "server_tag", "server_seq", "status"}`; the client strips it before the runner sees the result.
A client in debug mode against a server without debug records `server_join="unverified"` (not admissible for R8).

## 2. On-disk layout (weilandserver, under the arm's run dir)

```
<RUN>/runs/<arm>/debug/
  MANIFEST.json                       # arm-level: schema, arm spec, sha of code/config/fits/library/stage table,
                                      # capture config, sampling seeds/rates, git head + dirty-diff sha
  server_<port>/
    meta_<pid>.json                   # server startup: model, suite, method class/kwargs, camera names + shapes,
                                      # state/action dims, valid action dims, H, control dt, cost weights, pid
    decisions_<pid>.jsonl             # one line per request (see §3), append-only, flushed per block/episode end
    blocks/d_<pid>_<blk:06d>.npz      # arrays for ≤16 decisions in arrival order (see §3)
    rawkeys/k_<pid>_<blk:06d>.npz     # full raw keys, hashed 1/16 decision sample only
    writer_stats_<pid>.json           # queue high-water, bytes, serialization ms, errors (rewritten periodically)
  client/<episode_key>/               # written by the receiver from the client stream
    episode.json                      # §4.1
    events.jsonl                      # §4.2 one line per decision + lifecycle events
    controls_<blk:04d>.npz            # §4.3, 64 controls per block
    snap_<decision_seq:06d>.npz       # §4.4 selected snapshots
  receipts/<episode_key>/...          # P3-style per-file sha/byte receipts + complete.json
  aug/<kind>/part_<n:05d>.npz         # §5 deferred augmentation outputs keyed by decision_id
  derived/                            # disposable reader caches (parquet); never authoritative
<RUN>/catalog/<model>_<suite>_<lib>/  # once per library: row table (§6) + sha of the store arrays used
<RUN>/catalog/digest_cache.json       # shared digest cache, keyed by realpath, size and mtime_ns
```

Every server process owns its own metadata, decision log and writer statistics; a restart never overwrites the
previous process's files. Consumers glob `meta*.json`, `decisions*.jsonl` and `writer_stats*.json`, including the
legacy single-file names in existing smoke captures. An unterminated final JSONL line is skipped and reported,
even if those final bytes parse as JSON; malformed completed lines remain corruption. `schema.read_jsonl` provides
this rule. The server publishes/fsyncs every pending admitted decision before acknowledging episode end (and at
implicit episode boundaries), so an abrupt kill cannot lose records of previously acknowledged episodes.

## 3. Server record (one per request)

`decisions_<pid>.jsonl` fields (scalars / short lists; legacy `decisions.jsonl` is also readable):

- identity: `decision_id`, `episode_key`, `decision_seq`, `dispatch_gen`, `task_uid`, `attempt`, `task_id`, `init`,
  `orig_init_state_idx`,
  `server_tag`, `server_seq` (per process), `conn`, `pid`, `t_recv`, `t_done`, `blk`, `blk_i`.
  `init` is the subset index parsed from the final component of `task_uid`, identical to the journal/client;
  `orig_init_state_idx` is the original state-pool index supplied in episode-start metadata. These can differ.
- serving: `vision` (bool, encoder ran live), `camera_mode` (`full`/`wrist_only`/`third_only`/`blind`), `src`
  (`cache`/`policy`/`policy_tail`/`cache_tail`/`follow`), `hit`, `blind_age_controls` (controls since the last
  consumed look), `look_reason`, `miss_reason`, `anchor_decision_id` (decision whose look produced the served
  chunk), `chunk_offset` (offset of the served head inside that anchor's chunk), `served_len`.
- dispatch counters (live only): `stage1_calls`, `camera_completions`, `policy_calls`, `stage23_calls`; owner cost
  of this decision `owner_cost` under the arm's declared price table; timings `pre_ms`, `infer_ms`, `s1_ms`,
  `s23_ms`, `queue_wait_ms`, `method_ms`.
- retrieval (when the method retrieved live): `lib`, `rows[16]`, `weights[16]`, `scores[16]`, `conf`, `d1`,
  `lib_sha`, `k_eff` (effective demo count), `unsupported_mass`.
- randomization (when the method draws): `eligible`, `p_nominal`, `p_effective`, `coin`, `coin_domain`,
  `treatment`, `override` (reason when p forced to 0/1), `budget_state`, `stall_state`, `cooldown`, plus
  method-specific factors (e.g. CT event/deviation factors, lottery E and its support).
- method diagnostics: `diag` = the method's full diagnostic dict for this decision (uncapped, JSON-safe scalars and
  short lists; arrays go to the block with key `diag_<name>`), `diag_status`.
- sampling: `rawkeys_sampled` (bool), `snapshot_hint` (bool, informational).

Block arrays (`d_*.npz`; first axis = decisions in the block; `decision_id` as fixed-width unicode):

| Key | dtype / shape | Notes |
|---|---|---|
| `decision_id` | `<U80` (n,) | join key |
| `img_<camera>` | uint8 (n,H,W,3) | exact wire image per camera as received (before model transforms); every decision, blind included |
| `state_wire` | float64 (n,S) | exact wire state as received |
| `prompt_idx` | int32 (n,) | index into `prompts` (prompt strings stored once per block as `<U512`) |
| `keys_pca_<camera>` | float32 (n,64) | deployed compact key per camera when computed live (NaN rows otherwise; see `keys_status`) |
| `state_norm` | float32 (n,S') | normalized state fed to the metric when computed |
| `served_chunk` | float32 (n,H,A) | normalized chunk the method served (H/A = model manifest; padding kept) |
| `served_wire` | float64 (n,Hw,Aw) | output-transformed wire chunk returned to the client (exact, float64) |
| `cache_chunk` | float32 (n,H,A) | cache synthesis at this decision if computed (NaN otherwise) |
| `policy_chunk` | float32 (n,H,A) | live policy chunk if the policy ran (NaN otherwise) |
| `diag_*` | any numeric | method diagnostic arrays |

`rawkeys/k_*.npz`: `decision_id`, `raw_key_<camera>` float32 (n, D) for decisions with `rawkeys_sampled`
(`sha256(campaign|task_uid|decision_seq) mod 16 == 0`).

## 4. Client record (per episode attempt)

### 4.1 `episode.json`
`task_uid, attempt, dispatch_gen, episode_key, arm, suite, task_id, task_text, init, orig_init_state_idx, env_seed,
reset_state_sha256, control_dt, max_steps, settle_controls, camera_names, image_shapes, action_dim, state_dim,
entities` (catalog: body/geom/joint/site/actuator names and ids, movable-object list with roles if known, goal
predicate names), `capabilities` (contacts, predicates, actuator_substeps, snapshots: status+reason),
`t_start, t_end, n_controls, n_decisions, success, termination_reason, error`.

### 4.2 `events.jsonl`
Per decision: `decision_seq, decision_id, control_idx_start, n_applied` (controls actually executed from the returned
chunk), `server_echo` (decision_id echoed or null), `t_send, t_recv, infer_ms`. Lifecycle lines: `reset`, `settle`,
`done` (with success/termination), `error`.

### 4.3 `controls_<blk>.npz` (≤ 64 controls per block, in order, no gaps)
| Key | dtype / shape | Notes |
|---|---|---|
| `control_idx`, `decision_seq`, `chunk_offset` | int32 (n,) | which decision's chunk and which row |
| `is_settle` | bool (n,) | settling / wait controls |
| `action` | float64 (n,A) | action passed to `env.step` |
| `reward`, `done` | float64 / bool (n,) | |
| `qpos`, `qvel` | float64 (n,nq)/(n,nv) | full simulator state **after** the control (before-state of control i = after-state of i−1; the reset state is in `snap_` and `episode.json` sha) |
| `eef_pos`, `eef_quat`, `gripper_qpos`, `gripper_qvel` | float64 | robot proprioception after the control |
| `obj_pos`, `obj_quat` | float64 (n,K,3)/(n,K,4) | every movable object / articulation body in `entities.movable` order |
| `obj_vel` | float64 (n,K,6) | body velocities (cvel) of the same bodies |
| `contact_off` | int32 (n+1,) | offsets into the flat contact arrays |
| `contact_geom` | int32 (m,2) | geom id pairs (catalog maps geom→body→entity) |
| `contact_dist`, `contact_pos`, `contact_frame` | float32 (m,),(m,3),(m,9) | |
| `contact_force` | float32 (m,6) | `mj_contactForce` in contact frame when available |
| `predicates` | float64 (n,P) | goal predicate values in `entities.predicates` order; NaN when evaluation errored |
| `act_substep` | float32 (n,S,nu) | actuator ctrl at each physics substep when exposed |

### 4.4 `snap_<seq>.npz`
Selected pre-decision points: reset (seq −1 → file `snap_reset.npz`), the first decision, every decision with
`sha256(campaign|task_uid|decision_seq) mod 16 == 1`, and the terminal state (`snap_final.npz`). Contents: `time`,
`qpos`, `qvel`, `act`, `qacc_warmstart`, `mocap_pos/quat`, controller numeric attributes (flattened, with a skipped-
attribute inventory), Python/NumPy/env RNG states, `selection_p`, `restore_certified=false`.

## 5. Deferred augmentation (`aug/<kind>/part_*.npz`, keyed by `decision_id`)

| kind | Contents | Coverage |
|---|---|---|
| `policy_shadow` | `chunk` float32 (n,H,A), `seed`, `input_sha` | every decision (live policy chunks are *not* replaced; shadow is additional) |
| `policy_draws` | `chunks` float32 (n,3,H,A), `seeds` | hashed 1/32 decision sample (`mod 32 == 2`) |
| `shadow_look` | full two-camera keys (PCA), A retrieval `rows/weights/scores`, `cache_chunk`, `lib` | every decision; pure-policy arms: both libraries |
| `camera_shadow` | wrist-only and third-only retrieval rows/weights/scores/chunks (π0.5) | every decision of π0.5 arms |

Seeds are `hash(campaign, task_id, init, decision_seq, draw)`; never wall clock, worker or arrival order. Each part
file records model/checkpoint sha, batch size, dtype and code sha. Augmentation never writes outside `aug/`.

## 6. Library catalog (`<RUN>/catalog/<cell>/rows.parquet` + `rows.json` sha)
Per library row: `row, task_id, episode, step, ep_len, progress, success, prev, next, lib_sha` plus the frozen R7
stage labels (`mode, event_near, rows_to_event, stage_run`) when a stage table exists. Neighbour chunks are
recovered from the store by row; they are never copied per decision.

## 7. Completeness (reader `validate`)
An arm is **capture-complete** when: 500 accepted journal pairs; every accepted attempt has `episode.json`,
contiguous `controls_*` covering `n_controls`, `events.jsonl` covering `n_decisions`; every client decision has a
server record with the same `decision_id` and every server record of an accepted attempt has a client event; all
receipts verified; image arrays decode with the declared shapes. **Augmentation-complete** adds every `aug` kind
required for the arm. Unaccepted attempts are retained and reported, never mixed into accepted data.
