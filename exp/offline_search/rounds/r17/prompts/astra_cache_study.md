<task>
Owner's next study (after the running horizon/step study finishes ~17:45), verbatim:
「这个实验做完了之后对所有新模型跑 纯cache组、纯cache组+修整器组、纯cache组+长horizon组、纯cache+长horizon+修正器组（这之前的都不用miss），
然后前面的全部重复一次加上停滞门，带旋钮的暂时不跑，每组仍然是全B集500episode，这几组你都可以只编写yaml配合concurrent server的热切换快速实验，
如果不行就是你没有把模型充分接入框架」. Owner also ruled today: only libero_spatial and libero_10 matter.
New models: DP-T (chunk 16), Fast-WAM (chunk 32), Cosmos (chunk 16, already executes the whole chunk). Libraries:
/data/r17_library/<bundle>_<suite>/bpool/ (B-pool 500 inits per suite, nested subsets 50..500, fits/size_N/READY.json:
Mahalanobis metric, held-out corrector, guard thresholds, caps — R16 format).

Write exp/offline_search/rounds/r17/cache/CACHE_STUDY.md (pre-registration) that maps each of the 8 groups to the exact
R15/R16 semantics and YAML knobs used for π0.5/GR00T (find them in exp/offline_search/rounds/r15, r16, logs/session_handoff.md
§1.2 "纯缓存 + 修正器消融", logs/cache_prune_*): what "pure cache with no MISS" means operationally (every decision served
from the library? nearest-neighbour always?), corrector = which fit, stagnation gate = which mechanism/threshold source,
long horizon = which executed C per model (rule: the longest C that passed the 2 pp non-inferiority in the running study,
else the R16 choice; Cosmos already executes 16 -> say how its long-horizon groups collapse), library size (B500 full?),
and the evaluation set: owner says 「全B集500episode」 — check what R15/R16 evaluated on (A official test vs B pool) and
state precisely whether evaluating on the B pool means evaluating on the library's own inits (leakage) — if R16 used A,
say so plainly so the coordinator can confirm with the owner. Specify episodes per group (500 per suite), pairing with
the pure-inference baselines (hs5dp_ctl_142812, hs6fw_ctl_143332 + resume, Cosmos merged K5), metrics (SR, NFE/decision,
hit/miss rates), and the run order for fastest completion (all groups per model as YAML hot-switches on one resident
server per GPU: h100 and the local 4090; workers on timan107 only).
</task>
<compact_output_contract>
End with: a table group x (YAML knobs, MISS, corrector, gate, C per model), the eval-set verdict with evidence, episode
budget, and open questions for the owner (at most 3).
</compact_output_contract>
<action_safety>
Read-only analysis/design; no GPU, no campaign actions, no codex credentials.
</action_safety>
