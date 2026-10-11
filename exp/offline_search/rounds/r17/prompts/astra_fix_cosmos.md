<task>
Owner correction (verbatim): 「我们之前都试过k=1 2的cosmos了在单纯减步实验里面」. Cosmos Policy K=1 and K=2 pure step reduction were already measured in an earlier study. See `/home/weiland/projects/openpi/logs/cosmos_nfe_plan.log.md` and its results; also `/data/cosmos/results/`, `/data/cosmos/logs/`, and any related reports.
Your section (6) claimed K2 is not legitimately runnable and substituted K3. That was wrong to state without checking prior work.

1. Read the earlier Cosmos NFE study: how K2 was run (sampler definition), test set, episode counts, SR/cost results, conclusion. Rewrite section (6) for Cosmos to cite those results (file:line) instead of scheduling new Cosmos arms. The coordinator has already removed all Cosmos arms (4090 control, K3, K1) from the queue.
2. Check whether pure step reduction or horizon studies already exist for DP-T or Fast-WAM in any earlier round or log (including the MetaWorld DP-MW step reduction). If yes, cite them and say whether the scheduled DP/Fast arms are still needed.
3. Update section (7) and the multiplicity count to match the remaining arms (DP: C10/C15/C16, DDIM10, DDIM5; Fast: C10/C16/C20/C32, K2, K1).
No GPU work. Do not touch running queues.
</task>
<compact_output_contract>Edit STUDY.md in place. Final message: 4 lines (what the earlier Cosmos study found; whether DP/Fast prior work exists).</compact_output_contract>
