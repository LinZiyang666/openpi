<task>
Coordinator, 2026-10-17 21:5x. I interrupted your turn. Two corrections, then continue the original task (sol_long_segment_rerun.md).

1. Disk I/O. Your audit (`longseg_1017.audit --workers 4`, streaming whole multi-GB `per_step.jsonl` files) saturated `/data` (sda: 98% util, io pressure 68% full). This stalled running experiments: the H100 lane `fa_co_r_u_l` dropped to 7 episodes per 10 minutes because its driver writes to /data, and a 4090 prepare blocked for 14 minutes in page I/O. I SIGKILLed the four audit workers; nothing else was touched. The disk scheduler is mq-deadline, so ionice will not help.
   Redo the audit cheaply:
   - one reader at a time;
   - read only a bounded sample per run, e.g. the first ~200k lines or a few hundred MB. A midpoint cut is structural, so a sample decides affected yes/no; report the sample size;
   - pause between files;
   - check `/proc/pressure/io` before each file and wait while `full avg10 > 20`.
   Apply the same discipline to any other bulk read or move on /data. A same-filesystem mv is fine.

2. Decision-step decoupling (owner rule). This supersedes the "defer the look to the next boundary" fix.
   - A decision is one whole long segment: a head plus all its tail deliveries up to `commit_horizon`.
   - Tails are action delivery only. No recipe/gate/knob/stall/corrector evaluation happens there, and no decision counter or clock advances.
   - Every clock and schedule counts decisions (segment heads): the recipe-visible `decision_seq`, the knob `periodic` histories, the stall gate's call/extra-look/cooldown schedule, and the corrector step. Environment steps are used only where the environment needs them (episode limits, execution acks).
   - Do this on the server side; the frozen workers keep requesting every `replan_steps`. `per_step` must record a decision index separately from the env-step index.
   - Invariant test: feed the same decision-level observation sequence to a commit_horizon = replan_steps configuration and a commit_horizon > replan_steps configuration. Recipe/clock decisions must be identical; only delivered action counts differ.
   - Also test: a head is always followed by its tails, and nothing is evaluated on tails.
   - Prove it on the real path.
   - Report how the stall gate's cadence changes relative to its Q3 calibration (it previously counted 5-step blocks) and whether any stall/knob artifact needs recalibration. Report only; do not recalibrate.

Then continue: the audit (cheap, as above), trash, rebuild with the decoupled code, and the two balanced queue lists. timan108 is dead. The usable worker nodes are timan107, timan1 GPU2 and local.
</task>

<rules>
Same as the original task: CPU only; no launches; mv only; do not touch running runs or chains; never saturate /data.
</rules>
