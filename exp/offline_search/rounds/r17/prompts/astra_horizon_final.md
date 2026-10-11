<task>
All horizon/step arms are finished. Wait (poll every 60 s, read-only) until /data/mw_finetune/eval/hs6fw_k1/RECEIPT.json is
PASS (it is PENDING_EVIDENCE at 18:57), then re-run exp/offline_search/rounds/r17/horizon/analyze_arms.py to produce the final
exp/offline_search/rounds/r17/horizon/RESULTS_DP_FASTWAM.md (all 5 DP arms: C10, C15, C16, DDIM10, DDIM5; all 6 Fast-WAM arms:
C10, C16, C20, C32, K2, K1), and write a short owner-facing summary exp/offline_search/rounds/r17/horizon/SUMMARY.md (Chinese,
tables, ≤60 lines) that combines DP, Fast-WAM and Cosmos K2 (COSMOS_K2.md): per model × arm: spatial/l10 SR diff, family
verdict, uncorrected per-comparison verdict, cost reduction (calls/NFE per control step), and the plain-language takeaway
(e.g. which arms look safe to adopt, which clearly lose). Note the design issue (24-way family at 500 pairs/suite) neutrally.
Then update exp/mw_finetune/RESULTS.md with the official test-set-A pure-inference baselines (Cosmos merged K5 2000;
Fast-WAM hs6fw_res_152931 2000; DP hs5dp_ctl_142812 2000; per suite SR, configs) and a pointer to SUMMARY.md, replacing any
stale entries; keep the earlier superseded 400-episode notes marked superseded.
</task>
<compact_output_contract>
End with: the SUMMARY.md table, the RESULTS.md section you changed, and the final cache-study L per model.
</compact_output_contract>
<action_safety>
Read-only analysis and docs; no GPU, no campaign actions, no codex credentials.
</action_safety>
