<task>
Integrate **Cosmos Policy on LIBERO** (official weights `nvidia/Cosmos-Policy-LIBERO-Predict2-2B`) into the R17 five-layer framework, the same way Fast-WAM-LIBERO and DP-LIBERO were integrated. Make it runnable with the formal launcher `exp/r17_integration/ops/formal_eval.sh` (R17 conductor per-episode dispatch + resident replicas behind the stock replica proxy).

Context:
- R17 kept Cosmos as a LIBERO-only model. The MetaWorld community weight is unusable; see your own `exp/mw_finetune/cosmos/HANDBACK_D.md`.
- In September we ran the official Cosmos LIBERO runner on this machine. libero_10 with 5 denoising steps scored 490/500; with 3 steps, 244/250 (partial). Results: `/data/cosmos/results/libero_10/{k5,k3}/shard*.json`; logs: `/data/cosmos/logs/`; HF home: `/data/cosmos/hf_home` (the LIBERO snapshot may be gone; re-download it if needed, verifying its SHA).
- Runtime: `/home/weiland/projects/openpi_ext/lines/cosmos-policy` (Python 3.10, Torch 2.7, TE, FlashAttention), already used by your MetaWorld adapter.
- Reference integrations to mirror:
  - `exp/r17_integration/models/fastwam/` (serve.py, provider.py, staged.py, key_builder.py, manifest.py, qualify.py, libero descriptors);
  - the DP-LIBERO bundle;
  - `exp/r17_integration/qualification/` (specs `slot_G2_*_libero_{A,B}.json`, `run_slot.py`);
  - `exp/r17_integration/ops/FORMAL_EVAL.md`.
- Framework design docs: `exp/r17_integration/{DESIGN.md,INVENTORY.md,WORK_PACKAGES.md}`. Use the framework hooks in `exp/r17_integration/common/`; do not modify `src/`.
</task>

<scope>
1. Serve entry. A pure-inference Cosmos-LIBERO resident with exactly the official LIBERO inference config:
   - cameras, proprio and chunk exactly as the official runner (`LIBERO.md`, `cosmos_policy/experiments/...`);
   - 5 denoising steps default;
   - the official value-based seed selection only if it is part of the official default inference; otherwise single sample. Document which one, with an exact source citation.
   It must speak the R17 wire / ModelSpec attestation like the other bundles. One observation per decision (current frame), no cross-call cache.
2. Native parity. Running the September official runner path and our serve entry on the same recorded LIBERO observation must give identical actions (or within a recorded GPU-arithmetic tolerance, with reason).
3. Five-layer bundle. Recipe / descriptor / key builder / staged split hooks to the extent the framework needs, following the Fast-WAM pattern. Mark explicitly anything model-specific that is not yet supported, instead of faking it.
4. Qualification. Add `slot_G2_cosmos_libero_A` and `_B` specs plus a formal-launcher model entry `cosmos_libero` (400 formal, optional official-50). Default GPU: H100 (the official README warns to evaluate on the GPU type used for training). Also allow the local 4090 for smoke.
5. Smoke. CPU tests, then one tiny GPU smoke on the 4090 (≤ 15 GB, a few minutes; the 4090 is shared with live evals, so do not touch other processes) with 2 LIBERO episodes through the formal launcher.
</scope>

<constraints>
- Do not touch or restart any running process: DP-LIBERO formal on the 4090 (ports 23160–23179), π0.5 MT50 K1 eval (23150), Fast-WAM / DP-MW formal on H100.
- No full campaigns; the coordinator launches them. No credentials in commands.
</constraints>

<compact_output_contract>
Write `exp/r17_integration/handbacks/HANDBACK_COSMOS.md` containing: files, official-config citations, the parity evidence, smoke evidence paths, the exact formal command (H100 + timan108), what is unsupported. Final message: 6 lines max.
</compact_output_contract>
