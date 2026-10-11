# Formal pure-inference SR launcher

Cosmos-LIBERO is available as `--model cosmos_libero`. Its defaults are H100,
timan108, one resident, K5, 16 actions and 16 controls per query, fixed policy
seed195, environment seed0, and a separate `libero_cosmos_official_10_r17_v1`
protocol. The native runner uses vertically flipped paired256 views, JPEG95,
224 preprocessing and raw `[gripper2, xyz3, quaternion4]` proprio. The model
adapter converts the stock worker's rot180 views and consumes a lossless raw
proprio field from the current simulator observation. No previous observation
or latent conditions a later decision.

```bash
exp/r17_integration/ops/formal_eval.sh --model cosmos_libero --server-node h100 --worker-node timan108 --replicas 1 --workers 24 --memory-budget-gib 10 --port 23240 --driver-port 23249 --run
```

This plans400 episodes. Add `--official-50` for a separate2000-episode original
init0..49 campaign. H100's pinned checkpoint and native runtime have passed CPU
preparation; the coordinator admits GPU use. Every resident must pass an
official/staged byte-parity self-check before serving. Use `--server-node local
--smoke --smoke-episodes 2 --workers 2` for the local4090 smoke. The launcher
allocates run-owned Tether policy/driver tunnels, publishes their endpoints in
`TUNNELS.json`, waits for the actual stock proxy, and closes the tunnels after
owned-process shutdown. Internal ports23240..23249 are separate from the live
DP lanes; broker public ports are assigned independently. Cosmos A/B slots
qualify native parity and the formal smoke; fitted recipes, model capture,
native resume, CP2, B>1 and split replay remain unsupported. See
[`HANDBACK_COSMOS.md`](../handbacks/HANDBACK_COSMOS.md) for exact source citations,
setup commands, parity and smoke evidence.

Run from the coordinator with `exp/r17_integration/ops/formal_eval.sh`. The
launcher seals the selected checkpoint on CPU, installs an immutable P6 client
payload, starts resident R17 servers sequentially, publishes the full bundle
through the stock replica proxy, and dispatches episodes through the R17
conductor. Results and ownership receipts are written to a new directory under
`/data/mw_finetune/eval/`; the final line prints its path and PASS/PARTIAL/FAIL.

`--prepare-only` runs CPU sealing, exact cohort planning, payload deployment,
worker imports and original LIBERO init-pool checks. `--run` starts GPU serving.
Every invocation creates a new run; checkpoint identities are discovered and
pinned before serving. DP requires an immutable complete export and DDPM100.
The default local DP pool is 12 replicas, 24 workers, a 14 GiB budget, public
port23160, backend ports23161–23172, and driver port23179. Physical owned GPU
memory is checked at startup; each process also has a PyTorch allocator cap.

The cache, retrieval, recipe, capture and rollout video paths are disabled.
Each resident executes one B1 request at a time, protecting CUDA-graph workspace
and native sampler buffers. N separate resident processes run together; the
stock proxy pins each connection to its least-connected backend. Workers remain
resident and receive individual episodes, including explicit original init IDs.

## Coordinator commands

DP-LIBERO formal400, four suites ×10 tasks ×original init0..9:

```bash
exp/r17_integration/ops/formal_eval.sh --model dp_libero --checkpoint /data/mw_finetune/dp/libero_policy_100000 --replicas 12 --workers 24 --gpu cuda:0 --memory-budget-gib 14 --run
```

DP-LIBERO official2000 is a separate protocol with original init0..49 and empty
library/dev partitions. Choose one local LIBERO campaign at a time:

```bash
exp/r17_integration/ops/formal_eval.sh --model dp_libero --checkpoint /data/mw_finetune/dp/libero_policy_100000 --replicas 12 --workers 24 --gpu cuda:0 --memory-budget-gib 14 --official-50 --run
```

DP-MW canonical MT50 ×original init0..9, H100 serving and timan108 workers:

```bash
exp/r17_integration/ops/formal_eval.sh --model dp_mw --checkpoint /data/mw_finetune/dp/mt50_policy_100000 --server-node h100 --worker-node timan108 --replicas 12 --workers 24 --gpu cuda:0 --memory-budget-gib 14 --driver-port 23178 --run
```

Fast-WAM MT50 ×original init0..9, after training ends and the immutable final
export exists on H100:

```bash
exp/r17_integration/ops/formal_eval.sh --model fastwam_mw --checkpoint /data/mw_finetune/fastwam/mt50_frozen_b128/policy_016001 --server-node h100 --worker-node timan108 --replicas 2 --workers 24 --gpu cuda:0 --memory-budget-gib 72 --port 23174 --driver-port 23177 --run
```

The H100 public endpoint is `149.165.153.233`; local serving and all driver
endpoints use `ziyanglin.com`. These command ports allow the local and H100
campaigns to coexist. The two Fast residents budget72 GiB: run that H100 pool
after the DP H100 pool closes. If co-residency is desired, use Fast N1 with
`--memory-budget-gib 36`. H100 campaigns start when the coordinator admits GPU use.
Busy ports cause a clear failure and owned-process cleanup; existing processes
are never signalled. `--extra-workers N` adds N workers on the other timan node.

Other supported bundles are `groot_mw`, `pi05_mw`, and `fastwam_libero`. Select an
immutable `--checkpoint`, `--replicas`, `--gpu`, `--workers` and memory budget.
GR00T defaults to the local checkpoint60000 and K4; π0.5 uses RLinf and K10;
Fast-WAM uses its native K10. All MW plans preserve the full canonical MT50 table.
Both LIBERO models use original paired256 views. Full campaigns have a default
12-hour deadline, an episode bound of30 minutes, and two retries; configure them
with `--seconds`, `--episode-seconds`, and `--retries`.

## Verification and results

Authorized smoke command (four complete episodes, one per suite):

```bash
exp/r17_integration/ops/formal_eval.sh --model dp_libero --replicas 2 --workers 2 --gpu cuda:0 --smoke --seconds 900 --episode-seconds 600 --run
```

DP startup compares E's eager/native graph physical actions and the actual R17
eager/graph physical actions with `torch.equal` on three different prompts and
seeds. Every backend must match its CPU-sealed ModelSpec before the proxy starts.
`SERVER_ATTESTATIONS.json`, `PROXY_METADATA.json`, `BUNDLE_ACK.json` and
`GPU_MEMORY.json` record readiness. `BACKEND_EVIDENCE.json` records serving counts
and episode IDs for each backend without capturing observations or actions.

The conductor writes its episode manifest, dispatch/worker census, accepted
attempts, rejected raw probes, journal and per-step execution evidence under
`driver/`. `SUMMARY.json` reports planned/completed episodes, each suite's SR,
per-task SR, the40-task mean, error/retry counts and observed inference/s. An
incomplete cohort or terminal error is PARTIAL, and missing episodes retain their
planned denominators. Stale and duplicate attempts cannot inflate SR. Full-arm
verification requires capture off, policy source, and the full sampler's NFE.

Remote worker records and closed H100 server evidence use P6 snapshots and the
existing SHA-checked `/tmp` staging path. P6 verifies the pulled archive before
the exact owned stage is deleted. Worker shutdown is drained before snapshots;
all server, proxy, driver and worker PIDs/pgids carry start-time ownership checks.
Remote supervisors expire their leases after120 seconds without a coordinator
renewal. `RECEIPT.json` records cleanup and `all_owned_closed`.

Final smoke evidence:
`/data/mw_finetune/eval/formal_dp_libero_smoke4_final/`.
It accepted4 episodes, exercised2 workers and backends23161/23162, verified the
full DP100 path, synced records, and closed every owned process. It completed177
inferences in122.42s (1.446 infer/s including dispatch, EGL simulation and drain),
with0 errors/retries and1.76 GiB for the two residents. This measures N2/W2 while
the existing π0.5 evaluation remained resident. Linear extrapolation is about
8.7 infer/s for N12/W24; it is a planning estimate, and the12-replica campaign
throughput has not been measured.
