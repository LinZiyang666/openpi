# Closed-loop plugin: any harness Method as a pure-cache server (R2)

Runs an offline-harness `Method` (`exp/offline_search/harness/api.py`: `fit(lib, ctx)` / `reset(episode)` /
`query(q) -> Result`) as the CP1 retrieval of a **pure-cache** LIBERO server: every decision is a FULL_HIT served from
a library; stage 1 (vision encoder) runs to produce the pooled keys, stage 2/3 never run. π0.5 and GR00T, spatial and
libero_10, same topology as trace_dual (single-replica servers on weilandserver's 4090, 64 timan107 workers via
`run_gtp` role=all, A-pool 500 pruned inits). **No `src/` change.**

## How it plugs in (`plugin.py`)

The entry scripts `serve_pi05.py` / `serve_groot.py` install the plugin, then run the stock server
(`scripts/serve_policy.py` / `exp/libero_groot/serve_groot_libero.py`) unchanged:

* `openpi.cache.config.build_per_connection_components` is wrapped: on every connection the freshly built CP1
  strategy / judge / storage facade are replaced by `PluginStrategy` (online QueryView -> `method.query`),
  `PluginJudge` (always FULL_HIT on the plugin's pick) and `PluginStorage` (serves synthesized or other-library
  actions as payloads). A pick in the `current` library is served through the native backend payload of entry
  `ids.json[row]` (bit-identical to `library/<m>_<s>/current/action[row]`; checked for every row when a backend is
  first used). The native strategy stays attached: it receives the lifecycle calls and, unless
  `--os-no-shadow-native`, runs in shadow on the same live keys so every decision also logs the native B0 winner.
* `WebsocketPolicyServer` is subclassed so each per-connection policy is wrapped (`_ConnPolicy`): wire observation
  (raw state, raw images) and episode identity (`task`, `__extra__` task_id / orig_init_state_idx / task_uid /
  attempt) go to the connection's session, every `infer` is timed, logs are flushed at `episode_end`.
* The fitted method is copied per connection (python containers deep-copied, every ndarray / tensor shared), so
  per-episode state never crosses connections. Methods must treat fit-time arrays as read-only (harness contract).

Online QueryView (identical attribute set and forbidden names as `api.QueryView`):

| attribute | online source |
|---|---|
| `key_v0`, `key_v1` | the live `key_builder.build` output (cp1 4x4 spatial pool, f32[32768]) = what the trace / store recorded |
| `rs` | robot_state key (π0.5 32-d, dims 8.. = 0; GR00T 8-d) |
| `raw_state` | wire `observation/state` as float32 (= trace `raw_state`) |
| `task_id` | library task map of the episode's task string (checked against `__extra__.task_id`) |
| `step`, `hist_*` | decision index / earlier decisions of this episode |
| `prev_a_exec`, `hist_a_exec` | executed chunks = the served payloads (model-normalized (H, 32)) |
| `prev_hit`, `hist_hit` | pure cache: `True` / 1 after step 0, `None` at step 0 |
| `episode` | `EpisodeView(uid=task_uid, task, task_id, init=orig_init_state_idx, index=position in the store's `<m>_<s>_cache` episodes.json, seed=blake2b(run seed, task_uid))` |
| `has_tok`, `tok_v0/1`, `img0/1` | **every decision** online (stage-1 token slice as f16 [256, 2048], wire images); offline only the tok subsample has them. `--os-tokens off` disables |

`Result.action` (synthesized chunk) is served as is (non-finite entries replaced by the top-1 row's); `Result.library`
may be any stored library (`bpool_all`, `bpool_cs`) or one registered in `fit()`.

## Selecting a method

Plugin flags (everything else goes to the stock server CLI):

```
--os-method <module_or_file>:<Class>   e.g. exp.offline_search.harness.baselines:B0Current or
                                       exp/offline_search/rounds/r02/f1_awm/method.py:AWM ; 'native' = control (log only)
--os-kwargs '<json>'                   constructor kwargs
--os-cell <m>_<s>_cache                library the method is fitted on (fit(lib=current, ctx) exactly like the harness)
--os-root /dev/shm/offline_search_store
--os-log-dir DIR --os-tag TAG          decisions_<tag>.jsonl (one row per decision + episode / startup rows)
--os-log-inputs                        per-episode npz of every input (keys: 256 KB/decision -> smoke / verification only)
--os-fit-artifact PATH                 load a pickled fit if PATH exists, else fit and write it (share one fit across servers)
--os-no-shadow-native                  skip the native shadow search
--os-seed N                            run seed of EpisodeView.seed
```

The method is fitted once at server start (before the model loads, so a broken method fails fast).

## Ops (`ops/`)

| script | what |
|---|---|
| `emit_arms.py --run-root R --spec arms_in.json` | arm yamls (= trace_dual `_cache` yaml minus the trace block), run_gtp matrices, `R/arms.json`. Modes: `plugin` (method), `native` (native retrieval via the plugin entry, log only), `stock` (stock entry point, no plugin code) |
| `sync_remote.sh R <arms>` | push `remote/{run_arm.sh,run_gtp_subset.py,count.py}` + the arms' yamls into the timan107 island `/scratch/zixuans8/openpi_trace/os_cl/` (staged via /tmp, sha printed) |
| `start_server.sh <model> <suite> <port> <yaml> <log-dir> <tag> [--os-*]` / `stop_server.sh <log-dir> <tag>` | tmux `oscl<port>`, `taskset -c $CPUS` (default 34-37,78-81), `OPENBLAS_NUM_THREADS=1` (numpy single-threaded like the harness: online outputs bit-identical to offline; torch keeps OMP=4), `STAGE1_ONLY=1` by default (stage 2/3 on meta: GPU per server π0.5 7.6 -> 2.2 GB, GR00T 5.7 -> 1.9 GB, keys unchanged), PID file, SIGTERM by PID after a /proc cmdline check |
| `abort.sh R <arm>` | Ctrl-C the arm's timan107 driver, stop its servers by PID |
| `chain.sh R <arm> ...` | per arm: servers up -> `run_arm.sh` on timan107 (private tmux socket `-L oscl`, session `oscl_<arm>`) -> 60 s polls -> completion = accepted & status in {done, failed} & no error >= EXPECT -> collect -> servers down; resumes the driver up to MAX_ATTEMPTS; restarts dead servers. `EV` lines in `R/runs/chain.log`, markers `R/state/<arm>.DONE|ERROR`, `CHAIN.DONE|ERROR` |
| `collect.py --run-root R <arm> ...` | pull journal / per_step / driver.log (sha-checked), SR with the completion judge, per task, FULL_HIT check on every client decision, server-side latency / agreement / exec checks -> `R/runs/<arm>/summary.json`, `R/summary.json` |

## Verification tools

* `selftest.py` — CPU: real per-connection stack (patched builder, CacheOrchestrator, plugin strategy / judge /
  storage, native shadow on the real in-memory backend, connection wrapper) fed with the store's recorded keys;
  checks plugin top-1 == recorded online top-1, native shadow == recorded, online == offline harness bit for bit.
* `replay_client.py` — open-loop websocket replay of recorded tok-subsample episodes (exact wire images / state /
  prompt) through a live server: live keys vs store keys, plugin / native top-1 vs recorded, FULL_HIT, winner ids.
* `verify_logs.py` — offline harness replay of the plugin's logged inputs (`--os-log-inputs`): builds a mini store
  from the logged keys / rs / raw_state / executed chunks, fits the logged method like the harness and runs
  `run_jobs_inprocess`; compares topk / scores / confidence / library / synthesized action / extras bit for bit,
  checks executed == selected; `--b0-check` adds offline B0 vs the native winners.
* `probe.py` — `ProbeB0` (B0 + crc32 digests of every QueryView field in extras) and `ProbeHist` (history-driven
  toy selector returning a synthesized action) for the checks above.

## Launching full arms (500 A-pool episodes per arm)

```bash
cd /home/weiland/projects/openpi
RUN=/home/weiland/trace_runs/os_closed_loop/<run_tag>            # data, not in the repo
mkdir -p $RUN
cat > $RUN/arms_in.json <<'J'
[
 {"name": "oscl_pi05_sp_b0nat",  "model": "pi05",  "suite": "spatial", "mode": "native"},
 {"name": "oscl_pi05_sp_b0plug", "model": "pi05",  "suite": "spatial", "mode": "plugin",
  "method": "exp.offline_search.harness.baselines:B0Current", "kwargs": {}},
 {"name": "oscl_pi05_sp_<m>",    "model": "pi05",  "suite": "spatial", "mode": "plugin",
  "method": "exp/offline_search/rounds/r02/<fK>/method.py:<Class>", "kwargs": {...},
  "plugin_args": ["--os-fit-artifact", "<RUN>/fits/oscl_pi05_sp_<m>.pkl"]}
]
J
taskset -c 34-37,78-81 .venv/bin/python -m exp.offline_search.closed_loop.ops.emit_arms --run-root $RUN --spec $RUN/arms_in.json
bash exp/offline_search/closed_loop/ops/sync_remote.sh $RUN <arm> ...
# optional, for expensive fits: fit once, every server of the arm loads the pickle
taskset -c <cpus> .venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method <spec> --os-kwargs '<json>' \
    --os-cell pi05_spatial_cache --os-log-dir $RUN/fits --os-fit-artifact $RUN/fits/oscl_pi05_sp_<m>.pkl
# ports: pick 4 free ports in 23100-23199 (ss -ltnH | grep -oE ':231[0-9]{2}\b'); check nvidia-smi free memory
tmux new -s oscl_chain -d "PORTS=231a,231b,231c,231d WPS=16 SERVER_CPUS=<cpus> taskset -c <cpus> \
    bash exp/offline_search/closed_loop/ops/chain.sh $RUN <arm> <arm> ..."
# Monitor: EV lines of $RUN/runs/chain.log (ARM_DONE / ARM_FAILED / SERVER_DIED / GPU_TIGHT / CHAIN_DONE / CHAIN_STOPPED)
# results: $RUN/runs/<arm>/summary.json, $RUN/summary.json; re-collect: .venv/bin/python -m exp.offline_search.closed_loop.ops.collect --run-root $RUN <arm>
```

A chain resumes: arms with `state/<arm>.DONE` are skipped, an interrupted arm's driver resumes from its journal.
