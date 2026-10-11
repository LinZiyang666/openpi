<task>
Next study (owner): pure-cache ablation for the new models on official test set A (libero_spatial + libero_10), 8 groups
as YAML hot-switches on the resident concurrent server: c, cr, ch, chr, cg, crg, chg, chrg (q=corrector, g=calibrated stall
gate allowing MISS only on stall, h=long horizon; MISS otherwise disabled). Pre-registration:
exp/offline_search/rounds/r17/cache/CACHE_STUDY.md; owner rulings and stall-gate definition:
exp/offline_search/rounds/r17/cache/OWNER_RULINGS.md. P6 finished the framework side (HANDBACK_P6.md § "Cache study framework
integration", 48 YAMLs in exp/r17_integration/ops/cache_study/yaml/STUDY.json, smoke script
exp/r17_integration/ops/cache_study/smoke.sh) and left DP-specific work for you in
exp/r17_integration/REQUESTS/P6_to_P4.md. Do everything in that request so DP is fully admitted: the library
(/data/r17_library/, B500) loads into your model's serving cache path, all 8 groups run as pure YAML hot-switches on the
same server configuration as the DP pure-inference baseline (batching/MPS/driver as in
exp/r17_integration/ops/RUN_CONFIGS.md), no replica restart between groups, hit/MISS/NFE accounting correct (ungated
groups: zero action-head NFE; gated: only stall-triggered MISS). CPU tests; then run P6's smoke for DP in
--preflight/CPU mode if it has one. Do not run GPU work: the GPUs are busy with the horizon/step study until ~17:45;
tell the coordinator the exact GPU smoke command and expected minutes.
</task>
<compact_output_contract>
End with: each item of P6_to_P4.md -> done/blocked (file:line), tests, and the exact GPU smoke command.
</compact_output_contract>
<action_safety>
CPU only. Do not stop/signal running campaigns or touch their frozen runtimes. No pkill -f bare patterns, no rm -rf,
no /data/i018, /dev/shm/i018, codex credentials. timan1 forbidden.
</action_safety>
