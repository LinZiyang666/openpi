# Simulator snapshots and paired-branch feasibility

Read-only source inspection found a feasible snapshot path without editing src:

1. `closed_loop/ops/remote/run_arm.sh` selects run_gtp or run_gtp_subset.
2. `exp/gate_threshold_pareto/run_gtp.py` constructs `WorkerSpec` (lines 637 onward).
   `src/openpi/conductor/agent.py` already exposes `worker_module` and `env`.
3. `examples/libero/worker_entry.py` constructs `LiberoEpisodeRunner`, whose
   constructor accepts `client_factory` and `run_episode_fn`.
4. `examples/libero/main.py:_run_episode` calls `env.set_init_state`, then
   `client.infer` before each chunk executes. Its trajectory recorder already
   uses `env.get_sim_state()`, `env.env.timestep` and `env.env.cur_time`.
5. `exp/trajectory_deviation/run_spawn_experiment.py:_teleport_env` already
   demonstrates reset → set_init_state → restore timestep/cur_time.

Implemented: `run_gtp.py` changes only the worker module/environment, `worker.py`
injects the two supported runner dependencies, and `snapshots.Client` captures
physics state immediately before inference. After the response identifies an
actual vision anchor, every k-th anchor is atomically persisted. The snapshot
contains simulator state, time/step, selected extra MuJoCo arrays when exposed,
numeric controller attributes plus an explicit skipped-attribute list, wire
images/state/prompt, selected wire action, original task/init/attempt/decision
identity, and the total episode timestep budget. There is no added env.step,
reset or infer. Output is under the client arm's `snapshots/` by default.
`P3_SNAPSHOT_EVERY=1` captures every anchor; larger k saves I/O. The wrapper
refuses a server that does not expose P3 anchor metadata. A capture/write failure
is re-raised outside the stock rollout's broad exception handler, preventing a
silently incomplete accepted episode.

The campaign also enables existing `--os-log-inputs`: one per-episode server NPZ
stores every decision's keys/state/executed normalized actions/hit/vision mask,
plus output wire actions. This is necessary to reconstruct the retrieval and
guard history later **without rerunning the simulator prefix**. An arbitrary
mid-episode state is not an initial state for a fresh A connection: step-0 uses
a different metric, the gripper memo and anchor differ, and policy tails are
lifecycle checked. Keep the exact source fits, input NPZ, anchor JSONL and
client snapshot together. Ordinary collect.py does not collect snapshots;
the coordinator must archive the client-side snapshot tree separately.

Unverified / exact remaining blockers for certified paired rollouts:

- No LIBERO simulator or remote client was imported or executed in this task.
  `set_init_state` plus flat MuJoCo state/time is demonstrably the repository's
  existing recipe, but it has not been proved sufficient for byte-exact
  continuations. Numeric controller attributes are captured; object-valued
  interpolation/controller state, wrapper RNG/state and contact solver state
  may require a version-specific restore adapter. Each snapshot says
  `restore_certified=false`; omitted controller fields are explicit.
- There is no current websocket message for restoring plugin session histories,
  method memos, last decision ID and policy-tail state. This can be implemented
  in a P3 connection wrapper with no src edit, but is not supplied as a tested
  resumable protocol here. Input NPZ and the exact fits make offline prefix
  replay possible; validate it against the logged anchor before resuming.
- Alternate **wire** chunks should be produced using the original model output
  transform and saved raw state; normalized actions cannot be directly sent to
  the simulator. Saved wire actions are the selected arm only. The policy and
  cache normalized alternatives are both in the P3 anchor log.
- Full state restoration needs a no-intervention replay check first: restore,
  execute the saved selected ten-step chunk, compare next state/images and
  rewards to original continuation. Require the same model/fit/plugin/libero/
  MuJoCo versions and remaining timestep budget. Only after that passes should
  cache vs policy at the saved decision (ten controls each), then A afterward,
  be called a paired counterfactual. Report both restoration error and the
  number of pairs rejected for mismatch.

Thus physics-state collection for later branches is implemented; exact paired
counterfactual outcomes and a certified resume client remain unverified and are
not claimed. These gaps concern restoration/serving, not a need to replay the
original simulator episode merely to recover its saved inputs.

Coordinator integration (not executed here): after ordinary sync, preserve the
client island's current `os_cl/run_arm.sh` as `os_cl/run_arm.stock.sh`; install the
owned P3 `run_arm.sh` as the selected run's launcher and copy the owned package
under the island's repository tree. The wrapper replaces exactly one stock
ENTRY line and otherwise executes the stock shell unchanged. It fails if that
line is absent/ambiguous. Use an isolated profiling island/run launcher, or
install only after other chains finish and restore the stock launcher afterward.
No remote/shared script was edited or pushed during development.
