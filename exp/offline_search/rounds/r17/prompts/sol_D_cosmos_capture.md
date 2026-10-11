<task>
P6 delivered the LIBERO teacher-library pipeline (`exp/r17_integration/handbacks/HANDBACK_P6.md` § "LIBERO library pipeline"; `ops/library_campaign.sh`, `debug/library_capture.py`). It is blocked for Cosmos because your bundle declares `debug_capture` unsupported and rejects stage2/sample capture.

Add the MINIMAL capture capability that P6's `LibraryDecisionSink` / CaptureBridge allowlist needs for `cosmos_libero`:
- per decision: the FeatureSpec retrieval keys (your genuine VAE pooled camera keys), the normalized action chunk (16×7), robot state (the 9-D Cosmos proprio as served), the two received uint8 camera frames as received by the adapter, and the identity/execution metadata P6 requires;
- full native K5, seed195/env0, 16-control commit unchanged;
- capture off by default and byte-identical when off.
Do not add snapshots/latents/RNG. Read P6's interface (`debug/library_capture.py`, `REQUESTS/P6_to_P1.md` and the bundle-capability checks) and match it exactly.
</task>
<verification_loop>
CPU tests: capture-off byte identity; capture-on field allowlist.
Then run P6's Cosmos 2-episode library smoke on H100 for libero_spatial only, exactly as in HANDBACK_P6 (ports 23360/23379, replicas 1, workers 2, `--smoke`).
- The H100 runs the Cosmos formal400 (ports 23240–23249) and Fast-WAM MT50 formal (23173–23177). Do not touch them.
- There is about 10 GB of GPU headroom.
- Skip `flock /tmp/r17_gpu.lock` on H100 only if that lock is held by the running formal campaigns; say so in the handback.
</verification_loop>
<compact_output_contract>Append "## Library capture" to HANDBACK_COSMOS.md with the smoke evidence path and the exact full-collection command. Final message: 4 lines.</compact_output_contract>
