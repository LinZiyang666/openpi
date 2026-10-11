<task>
The Cosmos-LIBERO formal400 on H100 (campaign `/data/mw_finetune/eval/formal_cosmos_libero_formal400_20261009_225545_b32d224f`) never served. On H100, replica_0 died at import:
`transformer_engine/common/__init__.py:_load_nvrtc` → `ldconfig -p | grep libnvrtc` returned 1 (CalledProcessError).
Log: H100 `/tmp/r17_formal/formal_cosmos_libero_formal400_20261009_225545_b32d224f/server/replica_0.log`.
The launcher did not fail fast: it kept waiting with a lease instead of reporting replica death. The coordinator interrupted it with Ctrl-C.

Fix:
(1) Make the H100 Cosmos runtime import TE correctly without system changes (no sudo). For example, export LD_LIBRARY_PATH / CUDA_HOME pointing at the venv's nvidia-cuda-nvrtc (or the matching CUDA libs already on H100) in the Cosmos replica environment only.
(2) Make formal_eval fail fast when a replica process exits before listening: FAIL receipt, clean shutdown, and the replica log tail in the error.
(3) Run a real H100 GPU smoke: 2 episodes through the formal launcher with `--server-node h100 --worker-node timan108 --replicas 1`, using ports 23240–23249.
   - The H100 currently also runs the Fast-WAM MT50 formal campaign (2 replicas, about 52 GB, ports 23173–23177). Do not touch it.
   - Your Cosmos smoke needs about 6 GB.
   - The local 4090 runs DP-MW formal + π0.5 K1; do not use it.
Report the exact relaunch command for formal400.
</task>
<compact_output_contract>Append "## H100 runtime fix" to HANDBACK_COSMOS.md. Final message: 4 lines.</compact_output_contract>
