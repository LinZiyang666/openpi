# Closed-loop plugin: any harness Method as a pure-cache server (R2) / mixed HIT-MISS server (R3, `--os-judge`)

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
  `PluginJudge` (always FULL_HIT on the plugin's pick; with `--os-judge` the mixed HIT/MISS verdict, see "Mixed
  HIT/MISS mode" below) and `PluginStorage` (serves synthesized or other-library
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
| `prev_a_exec`, `hist_a_exec` | executed chunks = the served payloads (model-normalized (H, 32)); mixed mode: the policy's chunk after a MISS |
| `prev_hit`, `hist_hit` | pure cache: `True` / 1 after step 0, `None` at step 0; mixed mode: the verdict of the previous decision (`False` / 0 after a MISS) |
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

## Mixed HIT/MISS mode (R3 H2): `--os-judge`

Without `--os-judge` the plugin is the pure-cache server above, byte for byte (same code paths, same log rows). With
it, `PluginJudge` may return `HitType.MISS`: the interceptor then runs stage 2/3 (the server must load the full model,
`STAGE1_ONLY=0`, π0.5 ≈ 7.6 GB / GR00T ≈ 5.7 GB per server; `install()` refuses a stage-1-only server whose judge can
MISS) and the policy's chunk is executed. The method still runs `query()` on every decision (its proposal is logged,
`winner` = the proposal even on a MISS, also in the client's `__hit_meta__.winner_id`).

**What the src does on a MISS** (`src/openpi/cache/interceptor.py` legacy `infer`, `orchestrator._check_impl`):
`stage1 -> orchestrator.check(CP1)` = `key_builder.collect / gate (always_search) / build -> PluginStrategy.search
(method.query, verdict) -> PluginJudge -> JudgeResult(MISS, winner_id=proposal)` -> `CheckResult(MISS, entry_id=
proposal, factor_outputs.osplug)` (no payload fetch, `miss_by_checkpoint++`) -> interceptor: meta guard, `stage2 ->
stage3 (run_stage3, noise from the process RNG) -> check(CP3)` (unconfigured, no-op) -> `broadcast_action(policy chunk
(H, 32) f32 cpu) -> PluginStrategy.record_action -> PluginSession.on_executed` -> `buffer_for_write` (write_policy
never) -> `clear()`; `__hit_meta__.hit_type = "MISS"`. GR00T (`cache/groot/interceptor.py`): `check(CP1)` MISS ->
`runner.run_stage2(stage1)` (stage 2 + 3) -> `_to_storage_tensor -> broadcast_action -> record_action`. On a HIT the
same `broadcast_action` carries the served payload, so `on_executed` is the single place where the executed chunk
arrives; it records `hits[step] = 1 | 0` and the chunk in `b_aex`, hence the next QueryView has `prev_hit=False`,
`prev_a_exec = policy chunk`, `hist_hit` with zeros — exactly the offline inf cells (`exec_hit_flag = 0`).

Flags (plugin; pass them in `plugin_args`):

| flag | verdict |
|---|---|
| `--os-judge always` | HIT always (pure cache + the mixed bookkeeping / log fields) |
| `--os-judge threshold:<tau>` | HIT iff `Result.confidence >= tau` |
| `--os-judge quantile:<h>:<W>[:<tau0>]` | HIT iff `confidence >= tau_t`, `tau_t` = the (1−h)-quantile of the effective confidences of the last W decisions of this server **process** (all connections, thread-safe): judged decisions enter with their confidence, forced / cap / burst / step0 MISSes as −inf (they consume miss budget), forced HITs as +inf. Starts from `tau0` until W/10 decisions were seen (no `tau0`: the running quantile from the first decision, −inf on the empty window). Deployment value = median `tau` over the second half (collect reports it per server) |
| `--os-judge guard_only` | HIT unless forced (below) |
| `--os-judge periodic:<k>` | MISS iff `step % k == k−1`; confidence and flags ignored (`periodic:1` = every decision a MISS) |
| `--os-judge-cap R` | (threshold / quantile / guard_only) MISS when the consecutive-HIT run before this decision ≥ R |
| `--os-judge-step0 judge\|miss\|hit` | step 0 of every episode (all modes): judged like any decision (default) or forced |
| `--os-judge-burst n` | (threshold / quantile / guard_only) after a forced MISS the next n−1 decisions are MISS too (`burst`); default 1 = off (H3's wrapper owns bursts through `os_reason` 7 when it needs the "ambiguity resolved" condition) |

Order inside threshold / quantile / guard_only: `force:<os_reason>` > `burst` > `cap` > threshold (`thr` / `quantile`)
or `guard_only`; `step0` overrides everything at step 0.

Contract with methods (H3 wrapper): `Result.confidence` — higher = more confident (the judge accepts ≥ tau);
`Result.extras["os_force_miss"] ∈ {0, 1}` forces a MISS (every mode except `always` / `periodic`);
`extras["os_reason"]` int code: 1 stuck, 2 terminal-with-closed-gripper, 3 overtime ∧ lag, 4 no-progress, 5
dispersion event, 6 ambiguous gripper change, 7 burst continuation. In mixed mode the JSONL `extras` puts every
`os_*` key first and the cap is 40 scalars (pure cache: unchanged 24 in insertion order), so the judge's inputs never
fall off the row; the `--os-log-inputs` npz stores all extras in both modes.

Decision-log fields added in mixed mode (`ev: dec`): `hit` (bool), `judge` (`thr` / `quantile` / `force:<os_reason>` /
`burst` / `cap` / `periodic` / `step0` / `always` / `guard_only`), `tau` (tau_t used; `"-inf"` / `"inf"` strings for the
infinite quantiles), `run` (consecutive HITs before this decision), `src` (`cache` | `policy`), `s1_ms` (= `pre_ms`:
infer entry → plugin search start = stage 1 + key build), `s23_ms` (search end → `broadcast_action`: stage 2 + 3, MISS
only, else null), `a_exec` (MISS only: the executed policy chunk's valid block `[:5, :7]`); `exec_ok` is `true|false`
on HITs (served payload == executed) and `null` on MISSes. `ev: episode` adds `n_hit` / `n_miss` / `ctrl` (controller
state); `ev: startup` adds `judge`. `__hit_meta__.factor_outputs.osplug` adds `os_hit / os_judge / os_tau / os_run`.
The npz (`--os-log-inputs`) adds `hit`, `judge`, `tau`, `run`, `s23_ms` and `meta.judge`.

Inference ratio (project definition, π0.5 CUDA-graph three stages): `IR = 0.152 + 0.848 · (1 − h)`; the measured
variant `(N·s1 + N_miss·s23) / (N·(s1 + s23))` uses the logged means (GR00T: use the measured one).

Ops / verification in mixed mode:
* `emit_arms.py`: a spec row with `"full_model": true` (required whenever the judge can MISS; refused otherwise) is
  carried into `arms.json` (`full_model`, `judge`); `chain.sh` starts that arm's servers with `STAGE1_ONLY=0` and the
  full-model `NEED_MB` (9000 π0.5 / 8000 GR00T) whatever the env says. Pure-cache rows / arms are unchanged.
* `collect.py`: mixed arms get a `mixed` block (h, misses, `ir_pi05_formula`, `ir_measured`, `s1_ms` / `s23_ms`,
  `judge_mix`, `tau` median of the second half per server, MISS per episode, MISS share in failed episodes,
  `verdict_mix_matches_server` = client `FULL_HIT` / `MISS` counts == server `hit` / `!hit` counts on the accepted
  attempts; `all_full_hit` stays reported and is expectedly False).
* `verify_logs.py`: MISS rows enter the mini store as policy rows (`a_inf` = logged executed chunk, `a_hit` ≠), so the
  offline replay sees `prev_hit=False` / `prev_a_exec` = policy chunk; topk / scores / confidence / extras bit for
  bit; `executed == selected` on HIT rows; every verdict re-derived from the logged conf / tau / run / step /
  `os_force_miss` must equal the logged one (`mixed.verdict_violations == 0`, `run_violations == 0`).
* `selftest.py --judge <spec> [--judge-cap R --judge-step0 .. --judge-burst n] [--replay-cell <cell>]`: the fake
  policy answers a MISS with the replay cell's recorded `a_inf[row]`; checks client verdict == logged flag, `exec_ok`
  True on HIT / None on MISS, executed == `a_inf` on MISS / == served on HIT, the verdict rule, offline mini-store
  equality, and — with `--replay-cell pi05_spatial_inf --judge periodic:1` — equality with the offline harness on the
  real inf cell (the QueryView after a MISS == the offline inf-cell view). `probe.py:ProbeForce` raises
  `os_force_miss` on two consecutive steps of every five. `replay_client.py` reports `hit_mix` and
  `mixed.verdict_matches_log` on a mixed server.

```bash
# mixed arm spec row (emit_arms; <RUN> is substituted by the coordinator)
{"name": "r3mx_p_sp_b0h70", "model": "pi05", "suite": "spatial", "mode": "plugin", "full_model": true,
 "method": "exp.offline_search.harness.baselines:B0Current", "kwargs": {},
 "plugin_args": ["--os-judge", "quantile:0.7:1000:0.974", "--os-fit-artifact", "<RUN>/fits/r3mx_p_sp_b0h70.pkl"]}
# chain: nothing new (full_model is read from arms.json); free GPU >= 9000 MiB per port is checked before each start
PORTS=<p1,p2,p3,p4> WPS=16 SERVER_CPUS=<cpus> bash exp/offline_search/closed_loop/ops/chain.sh $RUN r3mx_p_sp_b0h70
# CPU selftests of the mixed path (a few seconds each)
taskset -c <cpus> .venv/bin/python -m exp.offline_search.closed_loop.selftest --cell pi05_spatial_cache --yaml <yaml> \
    --method exp.offline_search.closed_loop.probe:ProbeForce --judge threshold:0.985 --judge-cap 4 --episodes 4 --out /tmp/st_mx
taskset -c <cpus> .venv/bin/python -m exp.offline_search.closed_loop.selftest --cell pi05_spatial_cache --yaml <yaml> \
    --method exp.offline_search.closed_loop.probe:ProbeB0 --judge periodic:1 --replay-cell pi05_spatial_inf --episodes 4 --out /tmp/st_inf
```

## KPI tool (`ops/kpi.py`, `ops/pilot.sh`) — R3 closed-loop screening

R3 screens methods by short closed-loop pilots and their logs, not by offline mean err (rounds/r03/SELECTION.md).
`ops/kpi.py` turns an arm's logs into the KPIs that separated success from failure in R2 (ideation A / C), plus
paired statistics against a reference arm, plus the mixed HIT/MISS quantities of ideation B. Read-only on the run roots;
numpy only; 4 arms x 36–41k decisions (l10 chain) in ~13 s with 4 workers. R2 numbers / validation:
`rounds/r03/h4_kpi/r02_kpi.md`.

```bash
cd /home/weiland/projects/openpi
taskset -c <cpus> .venv/bin/python -m exp.offline_search.closed_loop.ops.kpi --run-root R [--run-root R2 ...] <arm> ... \
    [--ref <arm | run:arm>] [--chain] [--pilot | --tasks 6,9,0,4,1 --episodes 0-19] [--json OUT] [--md OUT] \
    [--store ROOT] [--act-eps 0.1] [--ir-ref <arm | run:arm> | --ir-ref-ms MS] [--recon METHOD=RULE ...] \
    [--workers 8] [--boot 10000] [--seed 0] [--cache-dir DIR] [--no-client] [--quiet]
# R3 pilot vs its R2 baseline (cross run roots: <run> is the basename or path of a --run-root):
taskset -c <cpus> .venv/bin/python -m exp.offline_search.closed_loop.ops.kpi --run-root $R3 --run-root /home/weiland/trace_runs/os_closed_loop/r02_g50 \
    r3p_p_sp_a1 r3p_p_sp_a1gc --ref r02_g50:oscl50_p_sp_cl2 --pilot --md $R3/kpi_pilot.md --json $R3/kpi_pilot.json
```

* Arms are `<arm>` (first `--run-root` that has `runs/<arm>`) or `<run>:<arm>`. `--ref` pairs every arm with one
  reference; `--chain` additionally pairs each arm with the previous one in the list (the CL0→CL1→CL2→CL3 decomposition).
  `--pilot` = the R3 subset per suite (spatial tasks 6,9,0,4,1 / l10 tasks 0,4,6,8,7 x ep_idx 0–19), also applied to the
  reference; `--tasks` / `--episodes` (lists / ranges) for any other subset. Episodes = complete journal rows (collect.py's
  judge); the paired set is the common (task, ep_idx) inits. Model / suite / method come from `arms.json` or the
  server startup row; the store root from the startup row (`--store` overrides).
* Output: markdown (stdout / `--md`) with the arm table, second-order spell facts, per-task SR, paired comparisons,
  the mixed table, and notes; `--json` holds every number (schema `offline_search.closed_loop.kpi.v1`).

Definitions (all on the executed valid block `[:5, :7]`, σ = std of the `current` library heads as in the harness):

| KPI | definition |
|---|---|
| SR, Wilson 95 % | success / complete episodes; per task too |
| dec/ep S \| F | decisions per successful / failed episode; `step_cap` = max decisions, `fail_timeout_share` = failures that hit it |
| spell | ≥ 3 consecutive **served** decisions with identical top-1 (a MISS breaks the run; MISS decisions belong to no run). Reported: spells per S / F episode, share of decisions inside spells (all / S / F), spell length, longest run (S / F), P(success \| 0 / 1 / ≥ 2 spells) with n, share of failed episodes with a spell, first-spell step |
| action-repeat spell | ≥ 3 consecutive served decisions whose σ-normalized served heads differ by RMS ≤ `--act-eps` (0.1 σ) between neighbours — the spell notion for synthesized methods whose top-1 may change while the served action does not (and vice versa); the two definitions agree for native rows |
| proposal spell | mixed arms only: the same run over all decisions (what the cache would have served); used for "first MISS vs first spell" |
| first-spell class | class of the FIRST spell of each failed episode (ideation A): **T** top-1 is the terminal row of its library episode; **G** \|Σ w_i sign(g_i)\| < .5 (gripper split of the served set); **Z** served translation norm < .6 σ; **P** top-1 is a library *pause row* (non-terminal row whose own next observation barely changed: mean-centred key cosine to the next row in the top decile on both cameras summed, or state motion in the bottom decile); **H** none. Priority T > G > Z > P |
| gripper flips | `g0`: sign changes of the served gripper at step 0 between consecutive decisions (A's number); `exec`: sign changes along the concatenated executed 5-step gripper series — counts intra-chunk transitions too, so a re-served transition chunk (e.g. `[+,+,+,−,−]` at a terminal hold row) oscillates the gripper and inflates `exec` (that is physics, not a bug); per S / F episode. On MISS decisions the executed head is the logged policy chunk when the row carries one (`a_exec` / `exec_chunk` / ...), else NaN (no flip counted across it) |
| w_term_late | weight of the served set on terminal library rows over the last third of each episode (S / F); also top-1 terminal shares (all / F / inside spells) |
| ep_eff | 1 / Σ_e (Σ_{i∈e} w_i)² of the served set (ideation C's effective library episodes); vote split = share of decisions with \|vote\| < .8 |
| native agreement | plugin top-1 == native B0 shadow winner (where logged) |
| paired vs ref | S→F / F→S counts, ΔSR, exact two-sided McNemar p (binomial on the discordant pairs), multinomial bootstrap of the (SS, SF, FS, FF) counts (`--boot`, `--seed`), Newcombe (1998) method-10 interval, and the normal approximation (ideation C's interval); per-task ΔSR with counts and p |
| mixed (rows with `hit` / `src` / `judge` / `tau` / `run` / `s1_ms` / `s23_ms`, extras `os_force_miss` / `os_reason`) | realized h overall, per regime (step 0 / after HIT / after MISS), per task; IR = 0.152 + 0.848 (1 − h) (π0.5 definition, printed for GR00T as reference only); `ir_stage_measured` = (mean s1 + (1 − h)·mean s23\|MISS) / (mean s1 + mean s23\|MISS) when the stage timings are logged; `ir_measured_vs_ref` = mean infer_ms / reference (`--ir-ref` arm or `--ir-ref-ms`); MISS share of decisions in failed vs successful episodes; share of all MISSes inside failed episodes (vs the decision share of failed episodes); MISS/episode S \| F; successful episodes interrupted (≥ 1 MISS); failed episodes touched; forced-MISS share and `os_reason` code shares (1 stuck, 2 terminal ∧ closed gripper, 3 overtime ∧ lag, 4 no progress, 5 dispersion event, 6 ambiguous gripper change, 7 burst); `judge` string counts on MISS rows (all / in failed episodes); first MISS step − first (proposal / served) spell start (mean, median, early share = no later than start + 1, before share); τ mean / median (second half); HIT run length at MISS. Spells / flips / taxonomy are computed on the HIT decisions only |

Served-action reconstruction (needed for gripper / translation / vote / terminal weight): the plugin logs the top-10
rows and scores but not the served chunk, so the head is rebuilt per decision from the library of the logged `lib`:
`synth == false` → the top-1 row; `native` / `M8*_top1` → top-1; `M4_b0cons_k{K}_mean` / `M8x_b0big_k{K}_mean` → uniform
top-K (exact); `AWM*` → kernel `exp(−((s₀−sᵢ)/(s₀−s_kref))²)` on the logged top-10 (kref from `_kr{N}`, else 8; the
real kernel has 16 members — top-10 error ≈ .03 σ RMS, ideation C); G3 wrappers (`V6sr_*`, `V7*`, any `__` name) →
`exp(Sᵢ − S₀)` on the logged top-10 (the base contract's weights; the blend member carries the top weight); anything
else → top-1 with a warning in the notes. Override with `--recon <name-or-prefix>=<top1|mean:K|awm[:KREF]|g3>` for new
methods (e.g. `--recon AWM3=awm:5`, `--recon MixedJudge=awm:5`). Library rows served from `bpool_*` (500-episode arms)
load that library; its pause mask costs ~5–8 s per 10× library (`--cache-dir` caches it).

Consistency fields per arm: client decisions and hit mix from `per_step.jsonl` (`_kind=client_timing` rows excluded;
`--no-client` skips the file), server `episode` success vs journal, `exec_ok` share, decision rows are matched to the
journal's accepted attempt and de-duplicated per (uid, step); warnings list gaps / duplicates / missing episodes.

`ops/pilot.sh <run-root> <sp|l10> <arm> ...` runs `chain.sh` on the R3 pilot subset (`OSCL_TASKS` = 6,9,0,4,1 or
0,4,6,8,7, `OSCL_EPISODES` = 0..19 → EXPECT 100 per arm), passing `PORTS` (required), `WPS`, `SERVER_CPUS`,
`STAGE1_ONLY`, `NEED_MB`, `MAX_ATTEMPTS` through from the environment; it checks every arm is in `arms.json` with the
given suite, warns about existing `state/<arm>.DONE` markers (chain.sh skips those — pilot arms need their own names,
`r3p_*`), `PILOT_TASKS` / `PILOT_EPISODES` override the subset, `PILOT_DRY=1` prints the chain command only. It does not
sync yamls to timan107 (`ops/sync_remote.sh` first) and never touches ports / tmux sessions it did not create.


## R4: opt-in vision-free serving (K2)

`--os-blind` enables the pre-interceptor `method.blind_step(BlindQueryView)` path and the R4 log schema.
`--os-log-r4` enables the R4 schema alone. Without either flag, the plugin's legacy pure-cache and mixed rows
remain unchanged. K3's serving entry points separately attach their startup cost metadata.

The contract is in `blind.py`: `LookReason(code, name)`, `BlindResult(action, rows, weights, library, extras)`,
and a frozen state-only `BlindQueryView`. The action is the normalized float32 `(H,32)` library chunk; the
plugin applies the normal model output transforms. Vision keys in dense blind history are NaN, with
`hist_has_vision` indicating validity. `prev_a_exec` is the actual normalized chunk served, including a policy
MISS. First decisions and post-MISS decisions require vision. No client image handshake is needed: continue
sending fresh full observations every five controls.

With `--os-blind`, periodic MISS uses a **server-wide**, zero-based `decision_index`, shared by connections
and continuing across episode resets: index `% k == k-1` is due. It is checked before `blind_step` and before
a step-0 HIT override. Without blind serving, periodic mode keeps its legacy episode-step clock.

Every R4 decision carries `vision`, `src` (`cache`, `cache_blind`, `policy`), `hit`, `blind_age` (number of
blind decisions BEFORE this one), `look_reason`, `miss_k`, `s1_ms`, `s23_ms`, and normalized `served_head`
(`[:5,:7]`). Blind rows have null stage timings, `searched=false`, and `shadow_available=false`.
`stage1_calls` counts actual stage function calls; `robot_state`, full supplied member rows/weights, and
`blind_extras` retain anchor/gate/phase diagnostics. `--os-log-inputs` also saves wire actions and dense masks.
`miss_k` is null on HIT; its MISS value comes from the live interceptor/action head.

Blind serving requires an untraced CP1 stack and one plugin session per connection. Quantile controllers
receive +infinity for a blind HIT. Judge caps/bursts require vision; malformed blind candidates or output
preflight failures fall back before committing histories. Optional `obs["__extra__"]` audit fields are
`decision_id` and `executed_steps`: a duplicate last ID is rejected with no commit; a count other than five
forces vision. This is not a state-only network protocol.

CPU regression and interleaving checks: `selftest --blind`; toy method:
`exp.offline_search.closed_loop.probe:ProbeBlind` (kwargs `library`, `budget`). With `--os-judge guard_only`
its first six decisions are vision HIT, blind, blind, vision HIT, MISS, vision HIT. Live stage-1-only smoke
uses `--os-judge always`. `verify_logs` replays both method interfaces; `replay_client --max-decisions 12
--wire-checkpoint <checkpoint>` compares actual wire actions with CPU-only output recomputation.

See `../rounds/r04/k2_serving/HANDBACK.md` for exact launch commands, completed checks, and the smoke-only
GR00T CPU-first loader that avoids the stock loader's transient full-model GPU allocation.
