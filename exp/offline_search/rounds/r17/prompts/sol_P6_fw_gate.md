<task>
Gate correction for `fastwam_libero` in `ops/library_campaign.py`. When I wrote "Fast-WAM LIBERO has not passed GPU A/B qualification yet", I meant the framework qualification slots `slot_G2_fastwam_libero_A` / `_B` (Q's run_slot). Both now PASS on H100:
- `/data/mw_finetune/qualification/runs/slot_G2_fastwam_libero_A_20261009_234134_2639853/receipt.json`
- `/data/mw_finetune/qualification/runs/slot_G2_fastwam_libero_B_20261009_234224_2645346/receipt.json`
You gated instead on P3's raw-Wan vs released-Diffusers loader-parity GPU receipt (`models/fastwam/libero_loader_parity_gpu.json`), which was never run and needs extra released-asset exports.

Change: accept the passing G2 A and B receipts as the Fast-WAM admission gate, verified by status PASS, slot name, and checkpoint / weight / code / norm digests matching the new teacher seal. Make the CLI take those two receipt paths. Keep the P3 loader-parity receipt as an optional extra, not required. If the G2 receipts do not bind the digests you need, say exactly what is missing instead of loosening the check silently.
Also make sure the H100 bounded staging (your previous change) is active for this bundle.
Then give the exact smoke + full commands for Fast-WAM × {libero_spatial, libero_10} on H100 + timan108, using `--replicas 2` (about 27 GB each; H100 80 GB, idle) and ports 23340–23359.
- For H100 runs, do NOT wrap in a local `flock /tmp/r17_gpu.lock`: the local lock belongs to the 4090, where the DP library collection is running. Use whatever H100-side exclusivity the launcher already applies.
- CPU tests only. Do not start GPU work.
</task>
<compact_output_contract>Append "## Fast-WAM gate = G2 A/B" to HANDBACK_P6.md. Final message: 4 lines.</compact_output_contract>
