<task>
Your new immutable worker install refuses timan108: `worker_installs.py:78 install_dependencies → ValueError('original LIBERO assets missing; refuse a shared-cache download')`.
- Campaign: `/data/mw_finetune/eval/formal_dp_libero_official2000_20261010_104712_89545577` (dp_libero, `--worker-node timan108`, `--official-50`).
- Background: timan108's `/home/zixuans8/.libero/config.yaml` points at an assets dir that does not exist (D noted this). Cosmos/Fast-WAM LIBERO runs on timan108 nevertheless worked via their own paths/runtime.
Make timan108 a valid LIBERO worker node for every bundle: provision the original LIBERO assets there as a verified immutable release (copy from timan107 or the local source, SHA-checked; no shared-cache download), or point the install at the assets the Cosmos/Fast-WAM runs already use, if they are byte-identical originals. Prove it with the worker import/validation probe for dp_libero on timan108.
Do not touch running campaigns: Cosmos on timan108, Fast-WAM and DP on timan107. No GPU work.
</task>
<compact_output_contract>Append "## timan108 LIBERO assets" to HANDBACK_Q.md. Final message: 3 lines.</compact_output_contract>
