<task>
Owner (verbatim): 「把每个cache size都全部重复一组，实验size相当于翻6倍，每个都如此」 — repeat the whole cache study for every
library size 50, 100, 200, 300, 400 (500 is running now) for all three models: 8 groups (Cosmos 4 distinct) x libero_spatial,
libero_10 x 500 episodes on official A, same server/worker configs and controls as the size-500 lanes, L = DP 10 / Fast 10 /
Cosmos 16. Libraries already have nested subsets and per-size fits: /data/r17_library/<bundle>_<suite>/bpool/subsets/size_N/
and fits/size_N/READY.json (Mahalanobis, corrector, guard thresholds, caps); check whether the stall-gate calibration and the C10
long-horizon fit exist per size, and produce them from B data only (no A) if missing, same procedure as size 500.
Do (CPU only, do not disturb running lanes: DP continuation dpc8513584592 on 4090, H100 lane csl10h running Fast-WAM then Cosmos):
1. Generate the YAML set for sizes 50..400 (short names), validated like STUDY.json, hot-switchable on the same resident server
   (size switch = library VIEW switch; prove no replica restart is needed or say why it is).
2. Extend the lanes so that, after the current size-500 work, the 4090 lane continues DP sizes 50,100,200,300,400 and the H100 lane
   continues Fast-WAM sizes 50..400 then Cosmos sizes 50..400 (or interleave if faster), resume-safe, prestaged, using the fixed
   recycling/routing/resume logic. Give one launch command per lane that waits for the current lane process to finish (do not
   start a second resident on a busy GPU) — or tell the coordinator exactly when to start each.
3. Real-library CPU audit of all new configurations; tests.
astra is updating the pre-registration (exp/offline_search/rounds/r17/cache/CACHE_STUDY.md) for the size factor.
</task>
<compact_output_contract>
End with: per-size fit/calibration availability, YAML count, lane commands and order, expected durations (timan107 is the
CPU bottleneck), tests.
</compact_output_contract>
<action_safety>
CPU only; no GPU launches; do not stop/signal running lanes. No pkill -f bare patterns, no rm -rf, no /data/i018,
/dev/shm/i018, codex credentials. timan1 forbidden.
</action_safety>
