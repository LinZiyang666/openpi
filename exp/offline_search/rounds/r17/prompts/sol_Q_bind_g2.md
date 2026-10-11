<task>
P6's Fast-WAM library gate now requires Q's `slot_G2_fastwam_libero_A`/`_B` PASS receipts to carry a bound ModelManifest v1. That means `model_manifest` with `content_digests` (weight/base/Wan/tokenizer/norm/code pin map) on the native Q PASS record, or `remote_receipt.model_manifest`. The current receipts lack it.
- Exact missing fields: `exp/r17_integration/handbacks/HANDBACK_P6.md` § "Fast-WAM gate = G2 A/B", and `REQUESTS/P6_to_P1.md`.
- Receipts:
  - `/data/mw_finetune/qualification/runs/slot_G2_fastwam_libero_A_20261009_234134_2639853/receipt.json`
  - `/data/mw_finetune/qualification/runs/slot_G2_fastwam_libero_B_20261009_234224_2645346/receipt.json`

Make run_slot record the qualified ModelManifest (the exact one the resident attested) in A/B receipts going forward. For the existing runs, publish bound receipts only if the binding is provable from preserved evidence; keep the originals untouched.
If it cannot be proven, rerun `slot_G2_fastwam_libero_A` and `_B` on H100 yourself. H100 is idle now; these are exclusive slots of a few minutes each, using the execution-host `/tmp/r17_gpu.lock`. Do not touch the 4090: the DP library collection is running there.
Report the two receipt paths P6's `--g2-a-receipt` / `--g2-b-receipt` should use.
</task>
<compact_output_contract>Append "## G2 receipts bound for P6" to HANDBACK_Q.md. Final message: 3 lines.</compact_output_contract>
