<task>
Run the pre-registered STUDY.md analysis (exp/offline_search/rounds/r17/horizon/STUDY.md; same machinery as your COSMOS_K2.md)
for the DP and Fast-WAM horizon/step arms, writing exp/offline_search/rounds/r17/horizon/RESULTS_DP_FASTWAM.md and a reusable
script (exp/offline_search/rounds/r17/horizon/analyze_arms.py) so it can be re-run as the remaining arms land.
Controls (pure-inference baselines, official-50, 4 suites; use only libero_spatial + libero_10 cells; owner: only these
two suites matter):
- DP: /data/mw_finetune/eval/hs5dp_ctl_142812 (4090, 2xB8/5ms + MPS, group-commit driver, timan107 x32; receipt recovered PASS).
- Fast-WAM: /data/mw_finetune/eval/hs6fw_res_152931 (merged resume of hs6fw_ctl_143332; h100, 2xB16/5ms + MPS, timan107 x48).
Arms (spatial + l10, 1000 cells each; directory names): DP hs5dp_c10, hs5dp_c15s0, hs5dp_c16s0 (finalizing), then
hs5dp_ddim10s*, hs5dp_ddim5s* (pending); Fast-WAM hs6fw_c10, hs6fw_c16 (finalizing), then hs6fw_c20*, hs6fw_c32*, hs6fw_k2*,
hs6fw_k1* (pending; a trailing s<n> means a re-prepared retry). Use only arms with a PASS receipt (or 1000 accepted cells
with a recovered receipt); list the rest as pending.
Deliver now for what exists: paired differences per suite, CP and bootstrap lower bounds (24-comparison family control as
pre-registered), cost side (calls/episode, NFE/decision, measured service time where comparable), verdict per arm.
Also state, per model, the provisional long-horizon L for the cache study (exp/offline_search/rounds/r17/cache/CACHE_STUDY.md
rule: longest C that passed 2 pp NI, else 10) and note that the 24-way family rule makes passing nearly impossible at
500 pairs per suite (owner has been told) — report the uncorrected per-comparison bounds alongside so the owner can judge.
</task>
<compact_output_contract>
End with: table model x arm x suite (n, control SR, arm SR, diff, CP LB family, bootstrap LB family, uncorrected LB, verdict),
cost reduction per arm, provisional L per model, and pending arms.
</compact_output_contract>
<action_safety>
Read-only analysis; no GPU, no campaign actions, no codex credentials.
</action_safety>
