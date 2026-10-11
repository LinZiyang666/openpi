<task>
Owner decision (verbatim): 「那就只测k=2算上清理步」. Run Cosmos Policy LIBERO pure step reduction at **K=2 with the official upstream semantics**: pass num_steps=2 to the official sampler, so the final clean step counts as one of the 2. Record the ACTUAL denoiser calls, whatever the upstream branch does. Label the arm exactly "K2（含清理步）" with its real NFE.

Your integration currently rejects K=2 (`HANDBACK_COSMOS.md` § Unsupported; cosmos_sampler.py:85–140), and the formal launcher's `--denoising-steps` allows only 1/3/5 for Cosmos (`exp/r17_integration/ops/formal_eval.py`).
Change, minimally:
(a) allow K=2 for cosmos_libero with the upstream-defined schedule. Seal a distinct sampler identity "cosmos_edm_2ab_k2_incl_clean". Make the per-decision audit expect the measured actual NFE for that identity instead of assuming NFE==K;
(b) check consistency with the earlier Cosmos pure step-reduction study (`/home/weiland/projects/openpi/logs/cosmos_nfe_plan.log.md` and results). If that study ran K=2 differently, follow what it did and say so;
(c) native parity: the R17 adapter at K2 equals the official `get_action` at num_steps=2, byte-exact on the recorded observations;
(d) CPU tests.
No GPU runs over 5 minutes. A short 4090 parity check is OK if the 4090 is idle (`nvidia-smi`); never kill anything.
Then give the exact command: H100 server, 2 replicas, `--worker-node timan108 --workers 12 --worker-memory-mib 6144`, ports 23240/23249, `--official-50 --suites libero_spatial libero_10 --denoising-steps 2`.
Do not touch running campaigns or the queue scripts.
</task>
<compact_output_contract>Append "## K2 incl. clean step" to HANDBACK_COSMOS.md (actual NFE, parity evidence, command). Final message: 4 lines.</compact_output_contract>
