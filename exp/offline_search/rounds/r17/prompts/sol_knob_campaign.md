<task>
Make the existing R17 cache-study launcher (`formal_eval.sh --cache-study <STUDY.json>`) able to run the per-task IR-knob study, and build its inputs. CPU only; hand back exact launch commands. A parallel agent adds the serving side (`recipe.periodic_gap` allowed inside the cache study, new `recipe.serve_loeo: bool`); assume those YAML fields exist.

Design (fixed by the coordinator, do not change):
- Models DP, Fast-WAM, Cosmos; suites libero_spatial, libero_10; library sizes 50/100/200/300/400/500.
- Base YAML = the complete group: DP/Fast `chrg` (C=10), Cosmos `crg` (C=16; `chrg` is its alias). Sources: sizes 50–400 rows of `exp/r17_integration/ops/cache_study/sizes/STUDY.json`, size 500 rows of `exp/r17_integration/ops/cache_study/yaml/STUDY.json` (absolute library/fit/stall paths and digests are already there; reuse them, no refit).
- Levels: `k50 k40 k30 k20 k10` = base YAML + `periodic_gap` from `/home/weiland/projects/openpi_ext/scratch/r17_ops/knob/gaps.jsonl` (one JSON line per model/suite/size, `levels["0.5"]["gap"]` etc.; written by the coordinator, may still be filling); `koff` = base YAML unchanged (gap null); `pure` = base YAML with `periodic_gap: 0.0` (every decision calls the model).
- DEV campaign per model (init pool B): levels k50..k10 + koff (6 levels) × 10 tasks × 2 suites × 6 sizes × B init indices 0..29, YAML `serve_loeo: true`. 21,600 episodes per model, one campaign, YAML hot-switch per episode like today.
- TEST campaign per model (official A, inits 0..49): from a levels table JSON `{suite: {size: {task_name: level}}}` (levels incl. `pure`), each task's 50 episodes use its level's YAML, `serve_loeo: false`. 12 cells × 500 episodes.
</task>

<what_to_change>
Read first: `ops/cache_study.py` (`generate`, `config_document`, `validate_manifest`, `fit_inputs`, `stage_inputs`, `seal`, `selected_groups`, `publish`, `plan`), `ops/cache_lane.py` (`launch_args`, `select_horizons` shows how a derived STUDY is written), `ops/formal_eval.py` (worker deployment, the `campaign.get('library')` branch around line 543 that installs `bpool_inputs.tar` and sets `worker_init_states_dir`), `ops/formal_workers.py` (~line 74 passes `--init-states-dir` only for library campaigns), `examples/libero/worker_entry.py` `parse_init_pools`.
1. A derived knob STUDY.json: same schema as today; one row per (size, suite, level) with a distinct `group` name (`chrg_k30`, ..., `crg_pure` etc.) and its own YAML file; settings/library/fit/stall/digests copied from the base row. Make `validate_manifest`/`fit_inputs`/`seal`/`stage_inputs` accept these group names with the minimum change (they are not aliases; Cosmos knob rows must not be dropped as aliases).
2. Optional per-row plan fields honored by `plan()`: `tasks` (list of task names; default all tasks of the suite), `indices` (init indices; default `range(50)`), and a study-level `eval_pool` (`"A"` default, `"B"`). B episodes: `init_pool='B'`, `partition='dev'`. Keep each episode's `task_uid` unique and readable (arm label = size + group).
3. B pool on workers for cache studies: when the study's `eval_pool` is `"B"`, reuse the library code path to push/install the B init archive on every worker node (timan107, timan1, local) and pass `--init-states-dir` to cache-study workers. Archive source: any `/data/r17_library/<model>_libero_spatial/campaigns/teacher_*/bpool_inputs.tar` (contains both suites; check its manifest covers both). Keep the A serving benchmark/protocol; only the init states differ. Verify on CPU (read the worker/env code) that the worker really loads B inits from that directory under the official-50 benchmark spec; if it does not, make the smallest change so it does. Do not add init-hash checks.
4. One builder CLI, `exp/r17_integration/ops/cache_knob.py`:
   - `dev --gaps <gaps.jsonl> --out <dir>` → YAMLs + STUDY.json (all three models' rows, like the existing STUDY files) with `eval_pool: "B"`, `indices: [0..29]`.
   - `test --model <m> --table <levels.json> --out <dir>` → YAMLs + STUDY.json for A.
   - Print the exact `formal_eval.sh` commands. Use `cache_lane.launch_args` as the template for flags (server node/replicas/batch/MPS per model as today; DP on the local 4090, Fast-WAM and Cosmos on H100), with a placeholder for `--run-name`, worker counts, and ports.
5. Dry run on CPU: build the DEV study for all three models, run `validate_manifest`, and call `plan()` on a minimal fake campaign dict (or the existing test helpers) to show the per-model episode count (21,600) and a few sample task extras. Do not run anything that pushes to remote nodes or starts processes.
</what_to_change>

<keep_it_simple>
Owner rule: only the main function plus a few necessary checks. No new gates, fingerprints, seals, admission or validation layers. Default behaviour of every existing STUDY/campaign must be unchanged. Delete code you make dead.
</keep_it_simple>

<action_safety>
Edit only `exp/r17_integration/ops/*`, `examples/libero/worker_entry.py` if strictly needed, and tests. Do not edit `common/*` or `search/*` (another agent is there). Campaigns are running on GPUs and remote nodes: do not start/stop/prepare any campaign, do not push to or exec on remote nodes, do not touch GPUs. Do not git commit.
</action_safety>

<handback>
Reply with: (a) diff summary per file/function; (b) where the DEV study for each model was written and its episode counts; (c) the exact commands I run to prepare and launch DEV for DP (4090) and Fast-WAM/Cosmos (H100), and later TEST from a levels table; (d) anything that could make B episodes silently use A inits.
</handback>
