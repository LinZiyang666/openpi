<task>
You are R4 coding agent K3: the cost engine (ideation B proposals B1, B3 and the single-camera part of B4).
Implement:
1. Reduced-step MISS: π0.5 via the existing yaml `miss: {num_steps: K}` path (read src/openpi/cache/config.py around
   `_miss_errors` / MissConfig and the interceptor's `_miss_steps`), GR00T via a K parameter exposed in
   closed_loop/ops/start_server.sh (today it hardcodes the denoising steps; keep the old default when unset) matched to
   the bundle's K. Log `miss_steps` in the plugin startup row via a server-side hook you own (coordinate through the
   documented field; do not edit plugin.py — if a field must be added there, describe it in HANDBACK for K2/the
   coordinator). Specify for K4 the exact yaml patch an arm needs (K4 implements generic yaml patching in emit_arms).
2. Exact redundancy elimination, as an opt-in stage override installed at server start (new
   closed_loop/stage_overrides.py, enabled from serve_pi05.py by a flag): (a) cache the embedding of π0.5's always-masked
   dummy third camera once (the exact preprocessed constant, per model/dtype/device) and skip its tower on every
   decision; (b) optional: after a MISS verdict, pack the fully masked prefix positions before stage 2 (ideation B §B3).
   Parity is mandatory: same inputs + same noise → identical keys (bitwise or documented tolerance) and identical actions
   with/without each override, on stored tok-subsample observations, including CUDA-graph paths if the server uses them.
3. Single-camera ("wrist-only") π0.5 key: on a HIT decision run only the wrist camera's tower (plus the cached dummy),
   build the key from it; on a MISS complete the other camera's tower exactly and continue the normal stage 1 → 2 → 3,
   so MISS actions are unchanged. Provide a camera-aware AWM adapter method (in rounds/r04/k3_cost/) refitted on
   wrist-only keys of the same 50- and 500-episode libraries (library keys rebuilt from the stored pooled keys, which is
   exact for camera deletion), with V7/guard tables recalibrated in that space, and the needed key-builder/plugin
   interplay documented (you do not own plugin.py).
4. Measure on the 4090 (batch 1, warmed up, the serving dtype/backend; CUDA graphs if the server uses them) the stage
   costs: full stage 1, dummy-cached stage 1, wrist-only stage 1, stage 2 unpacked/packed, stage 3 per denoising step,
   for π0.5, and GR00T's current split; write closed_loop/ops/cost_table.json (ms and shares, with provenance) for K4's
   ledger. This is a cost-accounting microbenchmark, not a system-throughput study.
Deliver arm specs `k3_cost/arms_r4.json` for the second- and fourth-batch arms of SELECTION.md (MISS K2 on the existing
g500 / g / perk5 / spatial guard-only configurations, K2-only pure inference, dummy-cached and wrist-only variants),
with prefit commands.
</task>
