<task>
You lead framework integration for the next study. Owner (verbatim): 「对所有新模型跑 纯cache组、纯cache+修整器组、纯cache+长horizon组、
纯cache+长horizon+修正器组（都不用miss），然后全部重复一次加上停滞门（带旋钮的暂时不跑），每组全B集500episode，这几组你都可以只编写yaml
配合concurrent server的热切换快速实验，如果不行就是你没有把模型充分接入框架」. Only libero_spatial and libero_10.
Models: DP-T (dp_libero), Fast-WAM (fastwam_libero), Cosmos (cosmos_libero). Libraries you built:
/data/r17_library/<bundle>_<suite>/bpool/ with subsets and fits (READY.json).

Goal: every one of the 8 groups x 3 models x 2 suites must be a YAML that the resident concurrent server hot-switches
(R16-style bundle rebind / router_bundle_rebind, conductor per-episode dispatch), exactly like π0.5/GR00T in R16.
1. Audit (CPU) end to end for each model: library load into the serving cache path, pure-cache recipe with MISS disabled,
   corrector application, stagnation gate, commit-horizon override per YAML, hot-switch between YAMLs without restarting
   replicas, compatibility with the batched servers now in use (Fast-WAM 2xB16+MPS on h100; DP 2xB8+MPS on 4090; Cosmos B1
   on h100) and the group-commit driver. List every gap with file:line.
2. Close the gaps (framework parts yourself; model hooks via exp/r17_integration/REQUESTS/P6_to_{P3,P4,D}.md and tell the
   coordinator who to dispatch). CPU tests, R16 parity tests against π0.5 YAML semantics where possible.
3. Generate the YAML set (names short) and a GPU smoke plan: per model one resident server, hot-switch through all 8
   groups with 2 episodes each, verifying hit/miss accounting, corrector/gate activity and no replica restart.
astra is writing the pre-registration exp/offline_search/rounds/r17/cache/CACHE_STUDY.md (group semantics, long-horizon
C per model, eval set); align with it when it lands.
Do not touch the running campaigns (they run from frozen runtimes; editing the working tree is fine).
</task>
<compact_output_contract>
End with: gap table (model x capability: ok / fixed / needs P3|P4|D), files/tests, YAML list, the smoke command, and
whether all 8 groups run as pure YAML hot-switches for each model.
</compact_output_contract>
<action_safety>
CPU only; no GPU until the coordinator schedules the smoke. Do not stop/signal campaigns. No pkill -f bare patterns,
no rm -rf, no /data/i018, /dev/shm/i018, codex credentials. timan1 forbidden.
</action_safety>
