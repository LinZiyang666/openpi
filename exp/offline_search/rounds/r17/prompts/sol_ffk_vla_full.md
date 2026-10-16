<task>
Coordinator decision (2026-10-16) for the open item in filter_failed_knob/HANDBACK.md "LIBERO π0.5/GR00T full steps". Do not build the R16 full-step alternative.

Build the filtered-library full-step per-task knob for LIBERO π0.5 and GR00T on the R15 pipeline (its runtime manifest has zero mismatches on current code):
- Sizes 100/200/500: filtered arm only, on the unchanged R15 pipeline; the unfiltered comparison is the existing R15 result for the same cells.
- Size 50: the unfiltered original was R14, which no longer runs. Run both arms on the R15 pipeline: filtered size 50 and an unfiltered size-50 rerun as its paired control. If the R15 pipeline cannot express size 50 without new method code, stop and report what is missing instead of writing it.
- Same design as the rest of the ablation: failed episodes removed, no top-up, no refit (filter only the retrievable rows), same references, levels, tolerance and raise rule as R15, same GPU and worker topology as R15. Reuse the R15 pure references where the serving environment matches.
- Workers must not use timan1 (offline). If R15 used timan1, give the worker-count option you already added (resume.py --worker-counts) or the R15 equivalent.

Deliver: campaign definitions (prefix ffk1015_), episode counts per stage (dev waves, A test, raises upper bound), exact prepare/launch/analyze commands, CPU checks (startup identity, planner, no-filter equivalence for the size-50 unfiltered control), and a section in filter_failed_knob/HANDBACK.md.
</task>

<rules>
- CPU only: no servers, GPU, workers, chains, git commits, pkill, rm -rf. Never touch credentials.
- Do not modify, rebuild or re-stage any launched run. tether: exec read-only checks only; never push/pull/expose.
- Reuse the R15 code; smallest new code.
</rules>

<output>
Final message: what was built, episode counts, GPU/workers, exact commands, CPU check results, and anything that needs my decision.
</output>
