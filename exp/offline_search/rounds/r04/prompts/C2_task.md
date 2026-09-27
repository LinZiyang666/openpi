<task>
You are R4 ideation agent C (explorer), second pass. Your first pass is finished: read
exp/offline_search/rounds/r04/ideation_C/REPORT.md and the scripts/outputs next to it first, and do not redo or re-propose
anything in it (control_step_library, stratified_paired_pilot, and the rejected phase gates / local affine synthesis /
successor-aware whitening). The owner asked for unconstrained thinking and said time is not a problem; the first pass
stayed close to representation and timing. Go wider now. Still excluded (owned by others or dead): "look once, act
several steps" (agent A); cheaper vision keys, fewer denoising steps on MISS, low-IR schedules, strong baselines (agent
B); library self-growth / pruning / GR00T mixed mode (round 5); everything in the Dead list of
exp/offline_search/rounds/r04/FINDINGS.md and the settled negatives of exp/offline_search/IDEATION_BRIEF.md.
Territories to investigate with real diagnostics (pick the ones where the data shows leverage; add your own):
1. Recoverability: in the R3 mixed arms (/home/weiland/trace_runs/os_closed_loop/r03_mx/runs/<arm>/, decision rows carry
   hit/miss, judge reason, confidence, extras, and the client journal has success), when does a policy call actually
   rescue an episode? Measure rescue probability as a function of when/where the MISS happens (phase, time since the
   first stall, gripper state, library progress, confidence), and what that implies for a different way of deciding
   MISS placement or MISS count — without re-proposing periodic/guard/HIT-cap schedules as such.
2. Cooperation between cache and policy other than all-or-nothing HIT/MISS and other than linear action blending
   (e.g. the cache supplying structure the policy lacks or vice versa); check exp/step_diag/analysis/ for what is
   already known about warm starts before proposing anything in that space.
3. Cheap signals already computed on every decision (stage-1 tokens are computed on every HIT; the store's tok/
   subsample has them) that predict drift/failure earlier than the current guards, measured on the closed-loop logs.
4. GR00T: why the method layer helps GR00T less (+3.6/+8.6 pp vs π0.5 +3.6/+20.2 at 50 episodes) and GR00T-l10 stays
   the largest gap even at 500 episodes; what is specific about its actions/keys/library.
5. Cross-task and cross-suite structure: shared objects/sub-skills between tasks and suites; whether retrieving across
   tasks (or from another suite's library) helps the hardest l10 tasks.
Report format: write exp/offline_search/rounds/r04/ideation_C/REPORT_2.md with the same structure as your first report
(measured facts with script paths; up to 3 additional proposals ranked by expected closed-loop value per unit IR, each
with mechanism, algorithm, plugin changes, SR/IR forecast at 50 and 500 episodes, bytes, kill criterion, cheapest
diagnostic, pilot design; rejected ideas with measured reasons). New scripts/outputs go next to the first ones with a
`p2_` prefix. Your CPU range: 24-37,68-81 (28 logical CPUs, at most 28 processes). Your letter: C.
</task>
