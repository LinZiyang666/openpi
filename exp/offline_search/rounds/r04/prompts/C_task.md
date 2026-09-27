<task>
You are R4 ideation agent C, the explorer. The coordinator and agents A and B are ALREADY covering these directions —
do NOT propose them or variants of them:
- "look once, act several steps" (vision-free decisions between vision anchors, phase/state triggers) — agent A;
- cheaper vision keys from the policy's own encoder, fewer denoising steps on MISS, low-IR schedules (periodic MISS,
  guard variants, HIT-run caps), strong baselines (reduced steps only, longer executed chunks, multi-seed inference) —
  agent B;
- planned for round 5: library self-growth (writing the robot's own successful episodes back), library pruning under a
  byte budget, GR00T mixed mode.
Also do not re-propose anything in the "Dead" list of FINDINGS.md or the settled negatives of IDEATION_BRIEF.md.
Look elsewhere and think without being constrained by our current framing. Candidate territories (not a checklist —
find your own): what the library stores and how it is organised (task/phase structure, subtask segmentation, graphs of
library states, synthetic or augmented entries), representation and metric learning that stays closed-form, how the
policy and the cache cooperate other than all-or-nothing HIT/MISS (without linearly blending actions), using the policy's
own cheap internal signals, closed-loop dynamics (why deadlocks form; what makes a state recoverable), GR00T-specific
behaviour, cross-task / cross-suite transfer, evaluation methodology (how to rank methods with fewer closed-loop
episodes, since offline error fails and 100-init pilots are noisy), theoretical limits of retrieval control. Measure
before you propose: every proposal must rest on a diagnostic you ran on the store or the closed-loop logs.
Your CPU range: 24-37,68-81 (28 logical CPUs, at most 28 processes). Your letter: C.
</task>
