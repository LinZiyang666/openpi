<task>
Context: cross-GPU equivalence study, directory `exp/offline_search/rounds/r17/xgpu_1017/`. Read REPORT.md, CLOSEDLOOP_PILOT_1018.md, CLOSEDLOOP_COMMANDS_1018.md, DP_BOTH_TF32OFF_1018.md, DP_CACHE_LOCALISATION_1018.md and FASTWAM_4090_H100_1018.md first. The previous agent on this study (astra) is no longer used; you take it over.

Findings so far: action-level equivalence fails across cards, and also within one card under concurrency (A100 DP same-card floor 0.0012). The question that matters is whether closed-loop outcomes (success rate per episode) differ between GPUs. The planned 4090-vs-H100 paired pilot is blocked behind formal lanes. The A100 (`tether exec a100`; 1×A100-SXM4-40GB, 32 CPUs, 115 GiB RAM, about 29 GB free disk) is completely idle and is reserved for this study. Use it now.

Do, in this order:
1. CPU, existing data, no GPU: find every pair of formal campaigns that ran the same plan (same task UIDs, inits, seeds, recipe, step mode) on both 4090 and H100. A known one: GR00T MT50 same-card pure baselines on H100, `tgr7h1_k2` and `tgr7h1_full` (66.3 / 66.8 group mean, reported "consistent with 4090"), versus the 4090 GR00T pure baselines of the same plan. For each pair, give per-episode paired SR difference, discordant counts, exact McNemar, and the conservative exact 95% CI against the preregistered ±2pp margin (same statistics as CLOSEDLOOP_PILOT_1018.md). Single-reader /data discipline: read journals sequentially, check /proc/pressure/io, and do not read per_step bulk.
2. A100 closed-loop, real formal path (formal planner, real workers, client/ACK protocol, same serving runtime and recipes as the formal campaigns):
   (a) Same-card floor: for each model, run the same plan twice on A100 with two cold starts, pure and cache arms. This tells whether cross-card discordance is larger than run-to-run discordance on one card.
   (b) A100 versus the existing formal results: choose plans that already have accepted formal results on 4090 or H100 (a fixed, preregistered subset of a formal plan by UID, not a new UID namespace), so every A100 episode pairs with an existing one at no H100/4090 cost.
   Decide the per-model capacity arm for a 40 GB card (π0.5 R4/B8 OOMed on A100 before) and record it as an explicit capacity arm. Workers: use A100-local workers only. Do not use timan1 or timan107; they belong to formal lanes and timan107 is memory-critical. If a simulator or renderer is missing on A100, install it in a user-level environment (minimal disk; sudo only if unavoidable, logged).
   Start with the model that is cheapest to get running, and run models one after another. Prepare scripts so I can launch them, or launch them yourself on A100 only. A100 is exclusively yours. Never touch 4090, H100, timan1 or timan107.
3. When A100 is between runs: the DP cache diagnostic in DP_CACHE_LOCALISATION_1018.md "下一步" (capture first-head stage1 feature, projected x, early y, raw d2; then replay float32 vs float64 distance on CPU). Report whether float64 or a cancellation-free distance removes the cross-card weight drift. Diagnose only, do not change the formal recipe.
</task>

<default_follow_through_policy>
Decide routine technical points yourself and record them. Do not stop to ask unless a source fact contradicts this task. Keep preregistration fixed: sample sizes and margins are set before results. No post-hoc budget widening.
</default_follow_through_policy>

<action_safety>
Do not edit chain scripts, chain lists, running campaigns or anything under /data/mw_finetune/eval/<formal campaign>. Write new files under xgpu_1017/ and new A100 dirs only. No tether expose/push/pull; transfers use public rsync with --bwlimit through a temporary daemon that is closed after use. Avoid ports 23100–23199 that formal campaigns use, and check `ss` first. Kill processes only by PID. Never touch credentials, auth.json, CODEX_HOME or logins.
</action_safety>

<compact_output_contract>
Reply in Chinese, short: (1) the existing cross-card pairs found and their paired SR results; (2) A100 runs done or running (model, arm, episodes, capacity arm) with same-card discordance and A100-vs-formal paired results; (3) the DP float64 result if reached; (4) the exact commands for anything I should launch.
</compact_output_contract>
