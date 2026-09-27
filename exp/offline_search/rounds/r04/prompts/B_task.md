<task>
You are R4 ideation agent B. Your topic is lowering the cost of every decision type other than "look once, act several
steps" (agent A owns that): the per-decision vision cost and the per-MISS cost, plus the best schedules at the low-IR end.
Cost model (π0.5, CUDA-graph): stage 1 vision encoder 10.26 ms (.152), stage 2 VLM prefix 27.69 ms (.410), stage 3
denoising 29.57 ms (.438). Study and rank concrete methods, including:
1. Cheaper retrieval key from the policy's OWN vision encoder (owner-approved): e.g. keys pooled from an intermediate
   layer of the vision tower, lower input resolution, fewer cameras or pooled patches, reusing part of the previous
   decision's computation. Find where stage 1 is computed (src/openpi: the π0.5 model's image embedding path; the
   closed-loop key builder `cp1_*` and the interceptor's stage split; the GR00T path under src/openpi/cache/groot/ and
   exp/libero_groot/) and measure, on the stored tok-subsample images (π0.5 has images in the store), (a) the cost of
   each cheaper key relative to full stage 1 on the 4090, and (b) retrieval quality with AWM refitted on the cheaper
   keys (library keys rebuilt the same way): top-k overlap with the full-key AWM, action error vs a_inf, and gripper
   vote splits at grasp/release. A MISS still needs the full stage 1, so the cheaper key saves only on HITs: give the
   net IR.
2. Cheaper MISS: fewer denoising steps on MISS (stage 3 only; use exp/step_diag/analysis/ for what is known about
   LIBERO step ladders and warm starts, and state what must still be measured on libero_10), whether anything in stage 2
   can be saved on a MISS without changing src/ semantics, and the resulting MISS cost.
3. The low-IR end of the SR-vs-IR frontier at the 500- and the 50-episode library: which schedules (periodic MISS every
   k, guard variants, HIT-run caps) should be run to map IR .15–.30, and which strong baselines must share the plot
   (reduced steps only, executing a longer part of each chunk, pure inference with more seeds) — with their IR.
Give the stacked IR table for the combinations you recommend. Your CPU range: 12-23,56-67 (24 logical CPUs, at most 24
processes). Your letter: B.
</task>
