# R16 resident conductor runbook

These are coordinator commands. Implementation used CPU only, launched nothing, synced nothing, and read no A outcomes. The old coordinator, its in-flight roots, `/tmp/r8_coord` and `openpi_ext/scratch/r16_ops/queue.txt` were untouched. The launcher takes the existing local orchestration lease and timan107 episode lease; it refuses while the old line owns them, including gaps between its arms.

The prepared dev queue is `queue_conductor_dev.txt`: four model/k group JSONs, each containing 44 original arm identities. π0.5 starts three local processes, each holding both suites' methods. GR00T starts four local processes, two per suite checkpoint, with 22 methods per process. All timan107 workers pull from the same group queue and can connect to either compatible suite endpoint. GR00T's suite checkpoints remain distinct. Its servers can have a suite tail even while workers continue serving the other suite. Methods are loaded once, never fitted. The existing YAML `load_cache_config`/`select_bundle` mechanism binds each connection to its original arm.

The policy horizon is five controls per client call, with the original ten-control MISS commitment and blind tail. Simulator/rollout seed is 2026101401; original `(arm:eval:task:init)` UIDs, init indices, pool bytes, resize sizes, 10 settling steps, and 220/520 step limits remain unchanged. Recording stays off. A worker probe must attest timan107, both frozen pool digests and these settings before receiving work.

Use this prefix for preparation and analysis:

```bash
taskset -c 2-9,46-53 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python
```

From `/home/weiland/projects/openpi`, first verify and capture any additional B arms completed since this handback. Append the following arguments to that prefix:

```text
-m exp.offline_search.rounds.r16.analyze verify
-m exp.offline_search.rounds.r16.conductor refresh-reuse
-m exp.offline_search.rounds.r16.conductor_deploy
```

`refresh-reuse` only amends groups that have never started. Started groups preserve their sealed definition and journal on resume. Every imported arm is validated against its original root, exact identities, fit and collected input hashes. Fourteen completed B arms were captured by the final refresh, including all eight size-50 fallback arms. If the old line completes more, refresh before taking over. No command changes an old root.

The isolated transfer archive and SHA lists are in `conductor_deployment/`. The worker island is `/scratch/zixuans8/openpi_r16_conductor_v1`, separate from the running line's tree. The local server tree is this checkout; `local.sha256` includes code, frozen fits and group definitions. Worker code includes regular `src/openpi_client` copies so the existing conda worker resolves this client after resetting PYTHONPATH.

Coordinator deployment, using the existing tether transport:

```bash
tether push --force exp/offline_search/rounds/r16/conductor_deployment/worker.tar timan107:/tmp/r16_conductor_worker_v1.tar
tether exec timan107 -- bash -c 'mkdir -p /scratch/zixuans8/openpi_r16_conductor_v1 && tar -xf /tmp/r16_conductor_worker_v1.tar -C /scratch/zixuans8/openpi_r16_conductor_v1 && cd /scratch/zixuans8/openpi_r16_conductor_v1 && sha256sum -c worker.sha256'
sha256sum -c exp/offline_search/rounds/r16/conductor_deployment/local.sha256
```

Run the small groups first:

```bash
bash exp/offline_search/rounds/r16/conductor_commands.sh smoke
```

There are two isolated smoke groups: one π0.5 process, then two GR00T processes with their suite-specific checkpoints. Each model runs four resident methods (k1 C10 and ir45 in each suite), two original B episodes per arm, with four timan107 workers. Smoke disables adaptation and never enters analysis or the official roots. Check:

- Each process's `resident/<port>.json` reports its method names, `additional_fit_gpu_bytes=0`, K and group SHA. π0.5 smoke has four methods; each GR00T smoke process has two.
- Under `runs/<original-arm>/server_<port>/`, `startup` has the original `method_spec`, kwargs, fit path and `miss_steps=1`. `resident_binding` has arm, YAML ID, fit SHA and group SHA.
- Every decision carries `arm`, `yaml_id`, `bundle`, and original UID. Actual policy rows have `miss_k=1`, `hit=false`, `vision=true`; tails have `source=policy_tail`, `hit=true`, `vision=false`, `miss_k=null`. Original `<arm>_<port>` log tags remain.
- Journal has exactly eight error-free accepted episodes per model, two per arm. `progress.json` attests timan107's pool digests and settings. Logs from different methods never mix in one arm's shard. No ordinary server should have `STAGE1_ONLY=1`.
- Observe memory at first policy calls, not just after startup, and check that the public TCP driver endpoint `ziyanglin.com:23179` is reachable from timan107. Server ports are selected from the existing local forwarded range; overlap with the driver is refused.

Full dev, in a local tmux session:

```bash
tmux new-session -s r16_conductor_dev 'cd /home/weiland/projects/openpi && bash exp/offline_search/rounds/r16/conductor_commands.sh dev'
```

Or run one group with the CPU prefix and these arguments:

```text
-m exp.offline_search.rounds.r16.conductor_runner run --group /home/weiland/trace_runs/os_closed_loop/r16_conductor_dev_pi05_k1_v1/group.json --agent-island /scratch/zixuans8/openpi_r16_conductor_v1 --workers 64 --port 23179
```

The runner uses receipt-authenticated local server lifecycle operations and launches a detached existing WorkerAgent through `control.remote`/tether. Its lease is renewed every 15 seconds; normal exit releases its workers, and an abrupt controller loss expires them within 90 seconds. The detached agent log and lease paths are recorded in `agent_lease.json`. It takes the same timan107 fleet lock as the old runner. This needs no new SSH credentials.

Ctrl-C keeps the models resident and leaves completed episodes in the journal. Re-run the same command/script: DONE groups are skipped, accepted episodes are kept, unfinished episodes are re-dispatched, and attempt numbers increase across driver restarts. Dead model processes alone are replaced using fresh owned receipts; surviving processes stay up. After a hard controller death, wait for the agent lease to expire if the remote episode lock is still held. On group completion the runner releases its model processes. Only cohort barriers and the final group tail can leave no ready episode; per-arm restart and idle tails are removed.

Each task's current level runs all 30 dev identities. Its next level becomes eligible only after the cohort completes and passes `100*(candidate successes-full C10 successes) >= -5*30`. First failure selects the predecessor and skips all lower levels. A failure at ir45 enables that task's k-C10 fallback, shared across sizes 100/200/500 when identities match. Already measured fallback cohorts are imported. Transport/config errors never count as ordinary SR failures. Gate records are under `gates/`. The final per-task tables equal full-ladder selection; uniform tables and below-failure recovery diagnostics are omitted by the owner amendment. 0/2 pp and confidence sensitivity level tables remain identifiable from the primary prefix. Unmeasured alternate stitched SR/IR is omitted.

After all dev groups complete, append these arguments to the CPU prefix:

```text
-m exp.offline_search.rounds.r16.analyze dev --conductor exp/offline_search/rounds/r16/conductor_dev_groups.json --out exp/offline_search/rounds/r16/dev_results_conductor
-m exp.offline_search.rounds.r16.analyze freeze --out exp/offline_search/rounds/r16/dev_results_conductor
-m exp.offline_search.rounds.r16.prepare emit-test --tables exp/offline_search/rounds/r16/dev_results_conductor
-m exp.offline_search.rounds.r16.conductor_deploy
```

The emitter refuses before the freeze. With an adaptive freeze it emits the 32 original-format per-task test roots plus four resident test groups, and rewrites `queue_test.txt` to group descriptors. The original per-arm queue is kept as `queue_test_single_method.txt`. The freeze binds original design/protocol hashes, the explicit amendment, selected tables, code and all actual input bytes. No uniform file is required for this amendment.

Run test A measure-only with the same execution machinery:

```bash
tmux new-session -s r16_conductor_test 'cd /home/weiland/projects/openpi && bash exp/offline_search/rounds/r16/conductor_commands.sh test'
```

Then append to the CPU prefix:

```text
-m exp.offline_search.rounds.r16.analyze test --conductor exp/offline_search/rounds/r16/conductor_test_groups.json --tables exp/offline_search/rounds/r16/dev_results_conductor --out exp/offline_search/rounds/r16/test_results_conductor
```

Test groups have no adaptive gates: all 32 arms run all 500 original A episodes. Per-cell, size × k and per-k comparisons use the unchanged R14/R15 paired estimators against full-step C10 and the same-size full-step task-descent references. The original phase-1 queue and paired analysis remain available and unchanged.

Resident fits add zero persistent GPU allocation. CPU replay measured 412.204 MiB of unique fit/table arrays for π0.5's complete 44-method group and 361.568 MiB for the complete two-suite GR00T group; each actual GR00T process holds only its suite, so the latter is an upper bound per process. These exclude Python overhead and existing mapped library storage. GPU planning uses the owner's measurements: π0.5 ~9.0 GB steady /9.9 GB peak per process, GR00T ~6.0 GB per process. Three π0.5 processes total ~27 GB steady; four GR00T processes total ~24 GB. π0.5's simultaneous 9.9 GB peaks would total 29.7 GB, above the stated 28 GB quota. Startup is sequential and actual free memory gates each process, but a simultaneous inference peak within 28 GB is not proven. The smoke must check this material limit. No GPU memory was measured here.

One historical π0.5 size-50 k2 C10 arm has 40 completed episodes with overwritten server logs. Their original accepted outcomes and continuous client MISS/blind-tail receipts are hash-audited; every surviving server decision is crosschecked. Pure C10 gives exact V=M from those receipts, allowing correct D5 counts for all 300 rather than the collector's surviving-log subset. No server rows were invented. Historical `miss_k` for those 40 episodes cannot be independently re-audited; the sealed YAML/launch is their remaining K binding. The limitation is in the amendment and reuse pack.
