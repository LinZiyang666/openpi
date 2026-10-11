# P3 → P4: pytest collection request resolved

All P3 test filenames now have test_fastwam_ prefixes. Imports of our fixture
module and serving helpers are fully qualified, so common/conftest cannot
replace them during aggregate collection. P3+common collect-only succeeded
with116 collected and no errors on2026-10-09. New bridge test path is
tests/models/fastwam/test_fastwam_native.py; no P4/common files edited.
# 2026-10-10 batching handoff

Read your DP-LIBERO batching request in P4_to_P3.md. P3's local GPU probe has
exited normally and released `/tmp/r17_gpu.lock` (about 11:36 CDT); the probe
process used 150.35 s including load, and nvidia-smi then showed zero memory.
Your queue can acquire the same lock; I will not consume another GPU slot for
this task. P6 request now includes your DP launcher/campaign seam.

Fast-WAM recommends B8/window5ms: 5.98 decisions/s (5.44x direct B1), mean
latency1.33s, max normalized deviation0.0078125. S07 B16 fragmented and offered
only4.6% more server throughput with72% higher mean latency. These are model-
specific numbers, not a DP recommendation. Full receipt/BATCHING.md is owned by
P3. Existing DP and H100 campaigns were not signalled or modified.

Binding caution for your formal entry: `model_runtime.load` already registers
the sealed provider. Keep that handle/YAML identity and inject the batched clone
through `make_connection_factory(...,adapter_factory=...)`, or P6 worker bundle
hot-switches may fail if a new provider ID is used. My CPU formal-entry test
exercises unchanged sealed provider binding plus actual S07 connection batching.

# 2026-10-10 normal launcher seam coordination

Owner now explicitly authorizes P3 narrow edits in normal formal_eval.py/.sh and
formal_server.py for Fast-WAM B8/5ms on local and H100. Shared flags default1/5;
DP opt-in will use your BatchConfig.metadata and owned formal_batch_server if
present. P3 will not edit any DP files. Keep the local GPU lock for your probe;
this task uses H100 only, one new resident alongside untouched Cosmos. Please
avoid concurrent edits to the exact argparse batch flags / campaign insertion /
dispatch blocks; extend with your owned hook API. Paired-control semantics are
implemented in a Fast-WAM owned helper; DP remains separate until your receipt.

12:07 implementation update: normal formal_eval now accepts DP max1..32 using
your BatchConfig, writes campaign/receipt dp_batching, sets qualified one-worker
stage streams, and formal_server dispatches to your owned formal_batch_server
with cuda_graph=True. New CPU tests check DP32 parsing and graph argument
forwarding; no DP GPU was used. No edits in your files. Existing DP B1 entry is
unchanged. Please add your per-decision audit/control enforcement separately.

