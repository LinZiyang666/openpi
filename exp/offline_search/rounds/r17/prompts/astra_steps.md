<task>
Extend your R17 study (same thread; STUDY.md) with an owner request and an owner device decision.

1. Pure denoising-step reduction (the R16-style "纯减步" study, horizon unchanged at each model's default), owner verbatim: 「单纯减迭代步数的也要研究，DDPM研究减到5和10，其他两个的迭代次数是多少，少的话减到1 2」.
   - DP-T: default DDPM K=100. Study K=10 and K=5. Determine the legitimate sampler for fewer steps (e.g. DDIM/trailing on the same ε-model, as used for DP-MW step reduction; see `exp/mw_finetune/dp/HANDBACK_E.md` and RECIPE) and whether the R17 serve entry/qualification already supports it.
   - Fast-WAM: default K=10. Study K=1 and K=2 (confirm the sampler semantics).
   - Cosmos: default K=5. Study K=1 and K=2. Note the Cosmos integration explicitly rejects K2 (see `exp/r17_integration/handbacks/HANDBACK_COSMOS.md` § Unsupported, cosmos_sampler.py:85–140). Decide whether K2 can be made legitimate, or whether K3 is the honest substitute. Do not fake it.
   - Same protocol as R16: official test A, 500 per suite, paired with the full-step default arm. Primary suites libero_spatial and libero_10. Use the same pre-registered decision rule as your horizon study (2pp non-inferiority + measured cost), applied to step arms. Add the launcher flag if missing (default off, CPU tests).
2. Device plan, owner decision: 「本机可以分担cosmos和wam的测试」. The local RTX 4090 is **48 GB**, not 24 GB. Measured per replica: Fast-WAM about 26 GB, Cosmos about 6–8 GB, DP about 0.9 GB. H100 is 80 GB.
   - Schedule ALL arms (horizon + step, all three models) across BOTH GPUs to minimise wall time.
   - Keep paired comparisons valid: an arm and its control must run on the same GPU type. Cosmos authors also warn about the GPU type, so if Cosmos arms run on the 4090, plan a 4090 Cosmos default control too.
   - Respect worker-node RAM: two concurrent fleets of 24 on timan107 OOM-killed a worker today. Q is adding node-memory admission (HANDBACK_Q § "Concurrent fleets on one node"). Plan worker counts per node (timan107, timan108; 48 cores each, about 250 GB RAM) accordingly.
   - Suites: restrict to libero_spatial + libero_10 if a default-off suite filter is cheap to add (implement it with a CPU proof that each suite keeps its 500 original identities); otherwise say so.
3. Currently running, for reuse as controls:
   - H100 official-50 Cosmos K5/C16 (`formal_cosmos_libero_official2000_20261010_100533_5f558380`);
   - Fast-WAM K10/C5 (`formal_fastwam_libero_official2000_20261010_101044_b0c4e68b`).
   - DP K100/C5 failed at 167 (OOM) and will be rerun on the 4090 after Q's fix.
</task>

<constraints>No GPU launches; the coordinator runs all GPU work. Do not touch running processes.</constraints>

<compact_output_contract>
Update STUDY.md with a new section "(6) 纯减步研究" and a section "(7) 两卡排程" (table of arm → GPU → worker node → est. hours, plus exact commands in order). Keep answers to the owner's questions explicit: each model's default K and the chosen reduced K.
Final message: 6 lines.
</compact_output_contract>
