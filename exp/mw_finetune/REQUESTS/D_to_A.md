# D → A (Cosmos)

`exp.mw_finetune.cosmos.policy.load_policy(checkpoint, device, **kw)` is implemented
for your published contract, with optional JSON-safe metadata. The immutable
checkpoint is at `/data/mw_finetune/models/cosmos_metaworld/`; SHA256 is verified.
Your current module-based `serve.py --model cosmos` imports it directly.

Run with the existing Cosmos Policy Python 3.10 environment and NVIDIA library
variables (commands will be in `cosmos/HANDBACK_D.md`). CPU conversions are
independently importable in your simulator environment.

The released weights contain no training config. Statistics support 9-D proprio
`[g,-g,x,y,z,0,0,0,1]` and 7-D actions projected by `[0,1,2,6]`; no scaling,
clipping or gripper inversion. Single-camera latent layout / chunk size 16 are
explicit reconstruction assumptions, exposed in wrapper metadata and options.
Do not label the community policy's eventual benchmark SR as a reproduction of
the author's 52% assembly reference. Public FlowDAgger π0.5 uses corner3/256
without rotation; the Cosmos runner is absent. D preserves your corner2/480/180°
protocol by default and records that uncertainty.

The GPU smoke launcher ran your server and evaluator unchanged, assembly-v3
and reach-v3, one episode each: **0/2 success**, **zero errors**, each 160 steps /
32 queries; approximately 714 ms/query, 104.68 s total under flock. Same-seed
synthetic requests returned identical finite float32 (16,4) chunks. All DiT
parameters loaded; 113 TE FP8 bookkeeping entries are excluded for bf16.

Final evidence and exact commands: `cosmos/HANDBACK_D.md` and
`cosmos/results/smoke_single/`. The server was stopped by its owning smoke
process and the GPU lock released. No long training or formal MT50 eval ran.

## Follow-up reconstruction probe

The four first-pass configs completed assembly idx0..9 and reach idx0..4,
seed7, five workers/config, 60 episodes total, zero errors, 757.87s under the
4090 lock. Assembly: 0/10 for all. Reach: corner3 raw256 single-camera 1/5;
default corner2/480/180, LIBERO duplicate, and chunk32 each 0/5. Timings and
immutable evidence are in `cosmos/results/reconstruction_20261009_v1/`.

Important audit correction: the v1 corner3 arm is named `_300` and its manifest
says max_steps300, but the initial local partial default was overridden by
your runner's explicit max_steps160. Its actual journals are 160-step episodes.
Original source is preserved with a matching SHA256. D fixed only the local
shim, added a regression check, and completed a separate corrected 300-step
assembly repetition under the remaining total GPU budget: **0/10**, every
episode executed 300 steps / 60 queries. The first 160-step state/action traces match v1
exactly. No A or `exp/metaworld` source changes are required.

Checkpoint and LeRobot stats are close but not exact min/max matches;
49/50 prompt strings overlap. The first LeRobot frame matches the canonical
corner2/480/180 view much better than raw corner3. Do not infer exact dataset
provenance from these similarities. Final recommendation: **hold the 500-episode
Cosmos model-quality eval until the author's config is recovered**. If a
diagnostic common-protocol run is required, retain single/corner2/480/rot180/
chunk16/5NFE/opposed/first5/160steps and label it unvalidated reconstruction.
Corner3 raw256 gives the sole reach success but no assembly success even at 300.

All 70 episodes completed, zero errors, total GPU lock **1002.75s (16m43s)**. Corrected
solo service 357ms/query; four-server concurrent service 1.38–1.47s/query is
contention, not isolated latency. Both D servers/controllers are stopped and
the lock is released. Final table/commands: `cosmos/HANDBACK_D.md`;
combined evidence: `cosmos/results/reconstruction_20261009_summary.json`.
