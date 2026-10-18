<task>
Owner rejected the trial in TRIAL.md: with a desired maximum of 8 workers, neither the throughput tuner nor the memory guard is ever exercised, so the trial proves nothing. Rewrite the trial plan and the patch defaults as follows. Same thread, same deliverable directory (exp/offline_search/rounds/r17/autoscale_1017/). Still do not apply the patch, launch, or call remote machines; sol is still editing shared files.

1. No separate small trial lane. The trial runs on real formal campaigns so no GPU time is wasted; the autoscaler does not change task UIDs, init indices, or seeds (your invariance tests), so the data stays valid.
   - LIBERO / server-bound case: the first H100 Cosmos formal campaign prepared after the patch is applied (expected: fa_co_r_f_l, filtered library, Cosmos, LIBERO, reduced steps, test-set levels; hours long). Desired maxima far above the known optimum: timan1=150, timan107=100. Known fixed-mode reference on the same GPU and model: 85 workers gave about 55 accepted/min, 44 workers gave about 73-86 accepted/min (H100 R8 B1 is server-bound). The tuner must find the plateau by itself.
   - MetaWorld / memory-pressure case: the first rebuilt 4090 MetaWorld campaign (sol's fa_ls_dp_* or equivalent), desired maxima timan1=200, timan107=150. LIBERO workers use about 2.5-3 GB RSS and MetaWorld less; with these maxima timan107 (220 GB) must hit the memory floor, so the guard is exercised for real.
2. Initial target must be the desired maximum (memory permitting), not a slow +1 ramp from zero after calibration. Ramp as fast as memory admission allows; shrink by the queueing signal. State the expected time to converge from 150/250 down to a plateau with your 0.8 factor and 180 s window, and shorten the window or raise the step if convergence takes more than about 20 minutes.
3. No small hard-coded defaults anywhere. The R16 conductor default maximum of 8 in your patch is wrong: the default desired maximum must stay what it was (64) or come from the configuration; the runtime decides the active count.
4. Acceptance criteria, measured from the run's own evidence:
   - throughput in the stable phase >= 95% of the best known fixed-mode rate for that model and GPU (above reference for Cosmos), reported per 180 s window;
   - zero OOM kills on any worker node (check dmesg/journal evidence through the existing channel at the end, not by polling), MemAvailable never below the critical line;
   - accepted errors = 0, every UID accepted once;
   - the active-count curve over time, with the reason for every change.
5. Update DESIGN.md, TRIAL.md, trial_commands.sh, INTEGRATION.patch (regenerate against the current files and refresh PATCH_BASE.json), tests, and VERIFICATION.json. Print the exact prepare/launch commands for both campaigns with the desired maxima above so I can put them in the queues.
</task>

<compact_output_contract>
Reply in Chinese, short: what changed, the convergence-time estimate, the exact commands, test results.
</compact_output_contract>
