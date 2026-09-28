<task>
You are R5 ideation agent A: execution horizon and "when to look again".
R4's strongest closed-loop facts are about how long an action chunk is executed before the next look: pure-inference
l10 improves from SR .845–.850 (5 executed steps per call) to .904 (all 10 steps); anchor_tail (after a vision HIT,
serve steps 5–9 of the same synthesized chunk without vision) reached .880 @ IR .203 (500-episode l10), .806 @ .242
(50-episode l10) and .982 @ .128 (500-episode spatial); per-neighbour phase continuation fails at 50 episodes; an
opt-in "policy tail" (after a MISS, execute steps 5–9 of the policy's own chunk without vision) is being implemented
(rounds/r04/k10_policy_tail/). Study this mechanism and design R5's best controller:
1. Why does executing whole chunks help on l10 (and does it on spatial)? Use the closed-loop logs of the L=5 vs L=10
   pure-inference arms (/home/weiland/trace_runs/os_closed_loop/r04_cost: r4f_p_l10_inf_s1001, r4b2_p_l10_inf_k2_s1101
   (L=5), r4f_p_l10_inf_k10_L10 (L=10); spatial counterparts as they complete), per task / per failure type, stalls,
   re-planning jitter at chunk boundaries (compare consecutive chunks' overlap), gripper events.
2. How far can the look cadence go? Beyond one chunk (e.g. stitching the next library rows after the chunk's end,
   or the control-step library of C), adaptive look timing from proprioception / phase / predicted error, and what the
   guards need to see. Separate the execution-length effect from the vision-saving effect in every forecast.
3. GR00T (H=16 chunks, 8-step executes? check the GR00T LIBERO settings) has never been run in mixed mode: propose
   GR00T mixed arms with tails, both suites and library scales.
4. Anything in the MISS path that interacts: e.g. when a MISS happens mid-chunk, is it better to execute the policy's
   whole chunk (policy tail) or the rest of a cache chunk?
Give concrete arms (method kwargs / plugin flags) for R5's closed loop.
</task>
