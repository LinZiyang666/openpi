<task>
Next launch failure, now past staging: `ff1015_dp_li_14` server replica exits at model load:
`exp/r17_integration/models/dp/loader.py:62 load_adapter -> ValueError("DP model/architecture/language/preprocess identity mismatch")`
(runtime /data/mw_finetune/qualification/formal_runtime_62692e19..., full log /home/weiland/projects/openpi_ext/scratch/r17_ops/ff/ff1015_dp_li_14.run.log and /data/mw_finetune/eval/ff1015_dp_li_14/replica_0.log). The source run csztd_r3_l48 loaded fine with the same runtime.
1. Find which identity differs and why (compare the ff run's assets/campaign fields the loader reads with the source run's). Fix the build/stage so the filtered run carries exactly the source's model/architecture/language/preprocess descriptors. Do not weaken the identity check: a mismatch may mean a wrong model or assets.
2. Prove on CPU, for every one of the 19 R17 jobs, that the frozen runtime's loader identity check passes with that job's actual staged assets (call the same descriptor/identity code path the server uses, without loading weights onto a GPU; loading on CPU is fine if needed). Also check anything else the server reads at startup before serving.
3. Reset ff1015_dp_li_14 with your reset.py and re-prepare it. Report exact commands; update HANDBACK.md.
Same limits as before: you may run prepare/stage and remote CPU staging/inspection; do not start any server, worker, driver or launch script; no GPU.
</task>
