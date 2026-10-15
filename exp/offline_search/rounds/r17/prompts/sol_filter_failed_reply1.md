<task>
Decisions (coordinator, owner delegated these choices):
1. Empty tasks (zero kept episodes): the pure Cache has nothing to retrieve, so those test episodes count as failures and are not simulated. Plan only the non-empty tasks' episodes; the analysis scores each empty task's 50 episodes as 0 and reports, per cell, SR with them counted as failures (primary) and SR over non-empty tasks only (secondary, paired against the unfiltered cell's same tasks), plus the list of empty tasks.
2. Fit: produce only what the Cache group serves (retrieval/metric artifacts), with the corrector fit off, so 1- and 2-episode tasks fit. First prove on one unfiltered cell per pipeline (R17 MT50, R17 LIBERO, R10 VLA) that this corrector-off fit reproduces the original retrieval artifacts byte-for-byte (or that serving reads nothing else); if it does not, stop and report.
Continue the full brief with these rules and deliver the handback.
</task>
