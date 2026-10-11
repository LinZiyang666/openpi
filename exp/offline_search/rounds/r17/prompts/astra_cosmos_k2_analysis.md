<task>
Run the pre-registered STUDY.md analysis (exp/offline_search/rounds/r17/horizon/STUDY.md, the decision rule you wrote)
for the Cosmos step-reduction arm, and write it to exp/offline_search/rounds/r17/horizon/COSMOS_K2.md.
Data:
- K5 official baseline: /data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_100533_5f558380 (1726 cells,
  workers on timan108, RTX A5000 EGL) + resume /data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_130639_a8ca2043
  (274 cells, libero_10 tasks ~34-39, workers on timan107, GTX 1080 EGL). sol D is sealing the merged 2000-cell result;
  use its merged SUMMARY if present (check HANDBACK_COSMOS.md), else the two journals (journal.jsonl task_uid
  `...:eval:<task_id>:<init>`, suite = EPISODE_PLAN.json tasks[].experiment by task_id).
- K2 incl. clean step (actual NFE 1): /data/mw_finetune/eval/hs2_cosmos_h100_k2incl_c16_t107_20261010_115120 (1000 cells,
  libero_spatial + libero_10, timan107). Same server GPU (H100), same B1, official-50 inits.
Coordinator's quick unpaired counts: K5 spatial 96.4%, libero_10 98.6%; K2 spatial 96.6%, libero_10 96.6%.
Do: pair by (suite, task, init); paired difference, exact/CP and bootstrap 95% lower bounds per primary suite, the
STUDY.md non-inferiority verdict (2 pp margin, both suites) and the cost side (actual NFE per decision, decisions per
episode, measured per-call time if available). Disclose the renderer-node confound (K5 cells split timan108/timan107 vs
K2 all timan107): quantify on the subset where both arms used timan107 (the 274 resumed cells) vs the rest, and list
the tasks driving any libero_10 drop. Also relate to the earlier Cosmos NFE study (logs/cosmos_nfe_plan.log.md, which
STUDY.md says had pairing/accounting problems) in 2-3 lines.
</task>
<compact_output_contract>
End with: a table suite x (n pairs, K5 SR, K2 SR, diff, CP LB, bootstrap LB, verdict), the overall verdict, cost
reduction, confound note, and the top tasks behind any drop.
</compact_output_contract>
<action_safety>
Read-only analysis; no GPU, no campaign actions. Do not touch codex credentials.
</action_safety>
