## R8 correctness review: debug capture, arms and tools

**Status:** one blocker is still open. The first arm that includes LIBERO-Spatial will fail capture verification until it is fixed. Three other blockers I found were fixed in the working tree while I was reviewing. I re-checked every finding below against the current files (latest edits: `adapter.py` 13:41, `observer.py` and `SCHEMA.md` 13:50). I did not edit any source file. A forked sub-reviewer covered S3 augmentation and the S4/S6 tools; its findings are merged in and marked "(fork)".

### Found during the review and already fixed in the working tree
These are fixed, but no regression test covers them, because the fake environment hid all three. Add tests:
1. **`adapter.py` `unwrap()`** stopped at LIBERO's outer `OffScreenRenderEnv`, which forwards `sim` but has no goal state, object map or predicate evaluator.
   - Effect: goal state and predicates were empty for every episode, and the oracle returned no objects with status "available". The six oracle arms would have silently behaved like A.
   - Now returns the level that has `_eval_predicate` / `parsed_problem`.
   - Needed test: a fake with a nested outer/inner env, asserting a non-empty goal state and predicates marked available.
2. **`adapter.py` `mj_contactForce`** was called with robosuite's Python wrapper objects, which raises a TypeError.
   - Effect: every episode would get contacts status "error", so every receipt fails verification.
   - Now passes the raw model/data (`model._model` / `data._data`).
   - Needed test: a fake where `mujoco.mj_contactForce` rejects wrapper objects.
3. **`observer.py:265` stored `served_wire` as float32.**
   - I confirmed on real R6 archives that π0.5 wire actions are float64 and do not survive a float32 round trip (GR00T's do).
   - Effect: the "issued action" check in `validate.py` would have failed every π0.5 decision.
   - Now stored as float64, and SCHEMA updated.

### BLOCKER (open)
**B1. The reset self-check rejects legitimately stacked starting positions, so every Spatial arm (32 arms) fails capture verification.**
- Where: `client/adapter.py` `resting_selfcheck` (about lines 142–160 now), stored at `capture.py:173`, and turned into an episode error at `transport/validation.py:97-98`.
- The check only accepts free objects that directly touch a geom named "table".
- In 6 of 10 LIBERO-Spatial tasks, a free bowl starts on the ramekin, the cookie box, the stove, the cabinet top, or inside the top drawer. The LIBERO task files confirm this, e.g. `(On akita_black_bowl_1 glazed_rim_porcelain_ramekin_1)` and `(On akita_black_bowl_2 wooden_cabinet_1_top_side)`.
- Result: `reset_selfcheck` is "error", the receipt is "error", and `collect.verify_arm` rejects about 300 of 500 accepted episodes in every Spatial arm.
- Fix: count support through a chain (an object on a supported object or on a static/fixture body counts as supported), or keep the self-check as a diagnostic that does not block admission.
- The same false positive exists in S6 `tools/physical/selfcheck.py:88-127`, where it is only reported.

### MAJOR — capture path
**M1. Default receiver port 24080 is not reachable from timan107.**
- Where: `ops/receiver.py:15,64`, feeding `config.py:59`. The file's own comment at line 17 says only the public 23100–23199 range is reachable.
- Effect: every stream sink fails for 5 seconds and then writes everything to timan107 `/scratch` (`transport/sink.py:268-273`). Nothing is lost, but streaming never works and all data arrives through the end-of-arm tether pull.
- The `chain_debug.sh` header was fixed at 13:33; the receiver default was not.
- Fix: use a port in the public range, and have the chain probe `collect --health ziyanglin.com:PORT` from timan107 before starting servers.

**M2. One leftover partial spill file blocks the whole arm.**
- Where: `transport/collect.py:170` selects files with `-name 'stream.osdebugspill*'`, and `unpack()` at lines 148–149 rejects any other file name.
- Scenario: a worker killed mid-episode after it started spilling (timan107 worker out-of-memory kills do happen) leaves `stream.osdebugspill.part` or `.json.tmp`. These get included in the archive, `unpack()` raises, the step reports CAPTURE_VERIFY_FAILED, and it fails the same way on every retry.
- Fix: match the two exact file names, and list stale partial files separately (they belong to attempts that were not accepted).

**M3. A server crash or restart in the middle of an arm cannot be recovered for admission.**
- Where: `server/writer.py` holds up to 16 decisions plus the queue in memory before a block is written. The chain restart (`chain_debug.sh:218-229`) reuses the same `server_<port>` directory.
- Three failure modes:
  - A hard kill (SIGKILL, out-of-memory, segfault) loses the server records of episodes the journal has already accepted. `verify_arm` then reports "unverified/missing server decision", and those episodes are never rerun.
  - A kill during a write can leave a truncated line in `decisions.jsonl`, and the restarted server keeps appending to the same file. `json.loads` at `collect.py:84` and the S3 JSONL reader then fail for the whole arm.
  - `meta.json` and `writer_stats.json` of the dead process are overwritten.
- Fix: write one `decisions_<pid>.jsonl` or directory per process; tolerate a truncated final line only for a dead process; publish pending decisions at episode boundaries; add a collect mode that requeues accepted attempts missing server records.

**M4. SF1 blind decisions are mislabelled `src="follow"`.**
- Where: `observer.py:216-217` labels any blind cache decision as `follow` if its extras contain any `os_sf_*` key.
- R7's `FollowAWM.blind_step` (`c1_follow/methods.py:182-195`) adds `os_sf_*` extras to every blind decision, including A's own second block and anchors that were not extended.
- Effect: in all 8 SF1 arms, A-tail blocks are recorded as follow. S4's `divergence` and `follow_vs_look` split results by `src`.
- The FL, W and IP/P10/oracle arms are not affected, because their `debug_record` supplies `src` directly.
- Fix: label `follow` only when `os_sf_source` is present (set only by `FollowExtension.serve`) or `os_sf_extension == 1`.

### MAJOR — analysis tools only (fork; does not block collection)
- **M5. Nothing writes the library catalog (SCHEMA §6).**
  - Only `fixtures.py` writes `catalog/*/rows.parquet`, so `reader.catalog()` returns None on real data.
  - As a result `provenance` is entirely unavailable, `blind_drift` rejects every followed row, and the catalog-based parts of `follow_vs_look` / `camera_shadow` are unavailable.
  - `reader.catalog()` also falls back to `catalog/<cell>`, and the cell name (e.g. `pi05_l10_cache`) is the same for the 50 and 500 libraries. If a catalog is ever written under the cell name, 500-library arms would silently read the 50-library table.
  - Fix: add a catalog builder keyed by model, suite and library, and remove the cell fallback.
- **M6. `tools/decision/outcomes.py:56` does not treat `"step_cap"` as a terminal state.**
  - The stock client only produces `success`, `step_cap` or `exception`; the fixture uses `"max_steps"`, so tests pass.
  - Effect: the `call_value` and `exposure_hazard` end points are effectively always unavailable for failed episodes.
  - Fix: add `"step_cap"` and treat `"exception"` as invalid.
- **M7. `tools/decision/trigger_vs_onset.py:14,72-77`:** the default `os_sf_valve_fire` trigger is only read from blind rows, but a valve fire causes a look and lands in the vision row's `blind_extras`. The SF1 valve hit rate therefore comes out as 0 or unavailable.

### MINOR
- **m1. Server-side `init` means something different from client-side `init`.** The server record's `init` is `orig_init_state_idx` (`plugin.py:836`, used at `observer.py:166`); the client and journal use the subset index from the task uid. They are equal on the official 500-pair manifest (checked on 22,006 R7 episodes, 0 mismatches), but not guaranteed on the non-test smoke pools. On those, S4 `common.inputs()` raises "conflicting decision/episode init" (`common.py:113-118`), and augmentation seeds use the server value. Fix: take `init` from the task uid and keep `orig_init_state_idx` as a separate field.
- **m2. A failed observer startup does not stop the server** (`plugin.py:553-560`). The arm runs all 500 episodes with error echoes and only fails at CAPTURE_VERIFY. The chain also has no early capture check (first few receipts, capability errors). Fix: make startup failure fatal when `--os-debug-dir` is given, and check `meta.json` plus the first receipts early.
- **m3. The arm MANIFEST hashes every `.py` under `debug/`** (`ops/config.py:67-80`), including `tools/`, `reader.py` and `aug/`. Any analysis edit during the campaign makes resuming an arm fail with "capture manifest differs". Fix: hash only capture-path code.
- **m4. `remote_collect` never deletes `/tmp/osdebug_<tag>_<arm>.tar` on timan107** (`collect.py:169-181`). With M1, that is roughly 0.5–1 GB per arm accumulating on the shared root filesystem.
- **m5. Oracle payload is all-or-nothing.** One unresolved goal object makes the top-level status "unsupported" (adapter `oracle()`), and S5 `_assignment` then drops the whole payload, including the objects that did resolve. Goals other than on/in (e.g. the L10 pudding "right of the plate") never open a window.
- **m6. The parity command in HANDBACK_S2 will crash.** `client/parity.py` defaults to inits 50–52 with the benchmark's 50-state pool (`main.py:1213-1214`), which raises IndexError. Also, a bare `--init-states-dir` only applies to the first suite.
- **m7. Server startup hashes the full library key arrays every time.** `manifest.py:64-77` reads e.g. 2 × 3.9 GB for π0.5 L10 `bpool_cs`, per server boot, 4 ports × 70 arms. Cache the digest once per campaign.
- **m8. The key/weight capture hook profiles the whole request.** `metric_tap` sets a thread-wide `sys.setprofile` around the entire inner inference (`diagnostics.py:97-140`, `plugin.py:1962-1967`). For GR00T the model forward runs on that thread under the shared inference lock, so every Python and C call pays a profiler callback. The overhead is not measured (S1's numbers use CPU fakes). It also adds per-event locals syncing (a threads-plus-closures edge case). Fix: wrap only `_dist` / `_mix` on the connection's own method instance, and measure GR00T miss latency with debug on during GPU parity.
- **m9 (fork):**
  - Augmentation seed strings pick up pandas float formatting (`aug/job.py:202` → `"3.0|21.0"`) whenever one error row makes the column float. Cast with `int()`.
  - Augmentation resume provenance hashes the whole fit-config and shared serving files (`job.py:294,56`), so unrelated edits block resume.
  - Reader and augmentation decompress every member of a block (images) even when only small keys are requested.
  - S3 `validate` ignores the record's top-level `status == "error"` and the `img_*_available` masks.
  - S4 reads fields no producer writes (`stage_pre`, `member_spread`, drift fields); `library_size` is never found because it sits under `r8`.

### Checked and correct
- **Debug off:** the plugin changes do nothing when the flags are absent (the obs copy and key pop only happen when `__debug__` / `__oracle__` are present; observer is None; the new `BlindQueryView.oracle` field defaults to None).
- **Debug on:** only copies are taken, before `after_infer`. Method debug adapters only read attributes (the properties involved have no side effects). The dispatch tap wraps only per-connection objects (per-connection interceptor, per-connection GR00T runner), so wrappers do not nest across connections. Errors become echo statuses and never propagate. The SIGTERM drain works, and the batching threads are daemon threads.
- **Interfaces line up:** request/echo field names and the decision_id format match across S1, S2 and S3. The campaign name is the run basename on both sides. The rawkeys/snapshot/draws sampling rules are identical in S1, S2, S3 and augmentation. The dispatch fence keeps attempt equal to dispatch generation, and episode keys match the journal. The S3 reader and validate match S1's block layout and S2's control/event layout.
- **Physics data:** stock environment seed 7 comes from the worker_entry default and run_gtp never overrides it. Controls are contiguous; settle controls carry decision_seq −1. Physics is float64 and images are exact uint8. Geom names come from the full id→name maps (robosuite keeps unnamed slots). Predicate evaluation is read-only.
- **Costs:** S1 `owner_cost` and S3 `budget` use the same prices (π0.5 .152/.848, GR00T .148/.852, wrist .0646, completion .0502).
- **S5 arms:** 70 arms / 35,000 episodes, priorities P0/P1/P2 = 32/34/4.
  - A fits are the same as R6/R7 `sources()`; SF1, SW, CU and CT reuse R7 artifacts; CU/CT have the new seed 26093005.
  - `full_model` is set on every arm that calls the policy or uses per-request cameras; `--os-oracle` appears only on the 6 oracle arms.
  - Method logic checked: FL support and propensities (E drawn uniformly over supported values, gates computed but not enforced); W5/W10 forcing the wrist camera; Shifted A; IP p = .25; P10 p = 1; O5b ≤ 0.05 m with at most 2 calls per object.
  - Re-emitting the specs now gives output identical to `/tmp/r8_S5/emitted/arms.json`.
- **Tests and compatibility:**
  - Client, transport and server block/observer tests: 43 passed.
  - Client, transport and backfill modules compile, and the client, sink and snapshot modules import, under the local Python 3.8.20 environment (`openpi_ext/envs/libero_sim`).
- **Augmentation (fork):** reads blocks exactly as S1 writes them, publishes atomically, resumes without duplicate IDs, and writes only under `aug/`.

Scratch output is in `/tmp/r8_review/` (test tempdirs, re-emitted specs in `/tmp/r8_review/emit/specs.json`).