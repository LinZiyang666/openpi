# R16 coordinator runbook

Implementation performed no servers, GPU inference, remote operations, sync or launches, and read no real A outcomes. All commands below are for the coordinator. Four isolated smoke roots contain two episodes each, preserving the official roots for complete paired runs.

Use the provided runner on `queue_phase1.txt`, then `queue_dev.txt`. Every line is `@local /absolute/root|one_arm`. Dev starts with sixteen shared k-C10 runs, then keeps each cell's ir45/ir40/ir32/ir25/off together. There are eight A phase-1 arms, 160 B descent arms and sixteen B fallback arms. All B arms have 300 episodes; phase 1 has 500. Do not change the client horizon, manifest or seed.

Set these launch-time variables outside the roots:

```bash
export SERVER_HOST=local WORKER_HOST=timan107 LOCAL_SERVER_COUNT=2 WPS=8
```

The existing local launcher defaults to four servers, so setting `LOCAL_SERVER_COUNT` explicitly matters. Full-model planning allowances are 9,000 MiB (π0.5, 8.79 GiB) and 8,000 MiB (GR00T, 7.81 GiB) per server at either k. These are the existing launcher's estimates, not measured R16 peaks. Step reduction changes compute, not resident weights. Two servers reserve 18,000/16,000 MiB respectively, leaving margin within the 28 GB quota; actual free memory still gates startup. No stage-1-only server can serve these arms. Give the whole device back after the experiment, as D6 requires.

Deployment sources and SHA256 are listed in `deployment.json`, `deployment_local.sha256` and `deployment_timan107.sha256`. The only changed shared runtime file is `src/openpi/cache/types.py`. Local serving uses this repository and frozen fits in place. Existing `control sync --concurrent` deploys the selected code snapshot, root matrices/YAML and pool records to the timan107 island; no server assets go to h100. Root definitions and frozen fits have their own hashes in the deployment JSON. The emitter adds new task-descent fits only after the dev freeze; run `audit deployment-test --tables ...` afterward for their separate deployment manifest.

The CPU command prefix for definitions, verification and analysis is:

```bash
taskset -c 2-9,46-53 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python
```

From `/home/weiland/projects/openpi`, run that prefix with these arguments:

```text
-m exp.offline_search.rounds.r16.audit verify
-m exp.offline_search.rounds.r16.analyze verify
-m exp.offline_search.rounds.r16.analyze phase1
-m exp.offline_search.rounds.r16.analyze dev
-m exp.offline_search.rounds.r16.analyze freeze
-m exp.offline_search.rounds.r16.prepare emit-test --tables exp/offline_search/rounds/r16/dev_results
-m exp.offline_search.rounds.r16.audit deployment-test --tables exp/offline_search/rounds/r16/dev_results
```

`phase1` writes the paired SR/Δ/CI/IR report against R14 C10. `dev` needs all 32 cells and references, writes primary and 0/2/5 pp task/uniform tables and stitched development estimates, then `freeze` rechecks all inputs and seals the tables. Only then can `emit-test` write 32 per-task A roots and fill `queue_test.txt`. Run that queue measure-only, then invoke:

```text
-m exp.offline_search.rounds.r16.analyze test --tables exp/offline_search/rounds/r16/dev_results
```

This writes per-cell paired comparisons against both full-step references, eight size × k pools and two descriptive per-k mixtures. Shared references are resampled as the same original clusters. No real analysis was run during this implementation. Outputs default to separate `phase1_results`, `dev_results`, `test_results` directories under R16. Frozen dev outputs cannot be overwritten.

Run the isolated smokes before the official queues:

```bash
bash exp/offline_search/rounds/r16/smoke_commands.sh
```

That script launches, sequentially, π0.5 phase1 k1, π0.5 dev ir45 k1, GR00T phase1 k1, GR00T dev ir45 k1; each is one arm × two episodes. It sets `LOCAL_SERVER_COUNT=1 WPS=1`. The source identities and flags are copied into separate `r16_smoke_*` roots. It syncs and runs each root with the existing controller. These smoke manifests do not replace the official manifests and their outcomes are excluded from every analysis.

Expected fields after each first arm:

- `startup`: `miss_steps=1`, `stage1_mode=full`, `method_spec`, `kwargs`, `argv` with the frozen fit path. GR00T's launch environment has `GROOT_DENOISING_STEPS=1`, and its server CLI passes `--denoising-steps 1`.
- Policy decision: `hit=false`, `vision=true`, `miss_k=1`, `src=source=policy`, `stage1_calls`, task/init/UID/attempt/step, `r14.fit_sha256`, and the unchanged guard/knob diagnostics.
- Following commitment: `source=policy_tail`, `hit=true`, `vision=false`, `miss_k=null`, `previous_executed_steps=5`; controls 5–9 of the same k-step chunk are served blind. Terminal partial chunks may end before this row.
- Collector: `cost_ledger.k_per_miss.mean=1`, matching K counts, nominal `controls=5*decisions`, and separate stage-ledger IR. R16 analysis reports D5 IR, which differs slightly from the rounded π0.5 stage ledger.
- `h100_launch.json` (historical filename): `server_host=local`, `worker_host=timan107`, every server `STAGE1_ONLY=0`. At size 500, real LOOKs additionally show `r14.leave_own_episode_out=true`, `query_pool=B`, the original `query_init`, and the excluded episode absent from both candidate and donor lists.

To rerun synthetic tests, use the CPU prefix with `-m pytest -p no:cacheprovider exp/offline_search/rounds/r16/test_r16.py --basetemp=/tmp/r16_recheck -q`. The default replay proof can be regenerated with `-m exp.offline_search.rounds.r16.replay` in a fresh implementation checkout; the registered checkout treats its proof file as frozen input.
