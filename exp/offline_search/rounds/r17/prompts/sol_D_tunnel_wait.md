<task>
Minimal fix in `exp/r17_integration/ops/formal_eval.py` (around line 636). Your readiness wait ("An exposed broker port accepts TCP before its local service is ready. Wait for the actual stock proxy before its public tunnel.") is gated on `campaign['model']=='cosmos'`. P6's library campaigns (`ops/library_campaign.py`) ALWAYS use tether-expose tunnels (dedicated 233xx ports have no NAT), including DP on the local server. DP library runs therefore fail with `InvalidMessage('did not receive a valid HTTP response')`: two of three DP launches did, e.g. `/data/r17_library/dp_libero_libero_spatial/campaigns/teacher_dp_libero_libero_spatial_500_20261009_235434_cc919ca3`.

Change the condition so the direct-proxy wait applies whenever the endpoint is a tunnel: endpoint_host/endpoint_port differ from the direct proxy, or `campaign.get('library')`, or cosmos.
- Keep it a minimal, merge-friendly edit. P6 is concurrently editing other parts of formal_eval.py (H100 shard transfer), so touch only these lines.
- Add or extend a CPU test.
- No GPU work; do not touch running processes. The 4090 is idle and waiting for this fix.
</task>
<compact_output_contract>Final message: 2 lines (what changed, test result).</compact_output_contract>
