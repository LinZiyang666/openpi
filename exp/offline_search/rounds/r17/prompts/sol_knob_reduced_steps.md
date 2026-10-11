<task>
Owner order: in the R17 cache system for the three new LIBERO models, every model call (stall-gate MISS, periodic-knob MISS, and the `pure` level) must use the owner's default reduced step count, then the per-task IR knob study is repeated with the reduced-step pure-inference success rate as the target. This mirrors π0.5/GR00T R16 (only the MISS sampler changes; library, metric, corrector, stall gate, long horizon all stay).

Reduced settings (same samplers as the 10-10 pure-inference step study, which used `formal_eval.sh --denoising-steps N`; see `exp/offline_search/rounds/r17/horizon/SUMMARY.md` and runs `/data/mw_finetune/eval/hs5dp_ddim10s0`, `hs6fw_k1`, `hs2_cosmos_h100_k2incl_c16_*`):
- DP: 10 steps, trailing DDIM eta 0 of the same ε model (default DDPM 100).
- Fast-WAM: K=1 native shifted flow (default K10).
- Cosmos: num_steps=2 with the clean step, actual network evaluations 1 (default K5).

Prepare code and inputs on CPU only; the current full-step knob campaigns (knobdevdp2, knobdevfw4, knobdevcos2) are running and must not be touched.
</task>

<what_to_change>
Read first: `exp/r17_integration/ops/cache_knob.py`, `ops/cache_study.py` (`config_document`, `seal`, `plan` — note `num_steps=model['default_sampler']['num_steps']` and `allowed_k=(model['default_sampler']['num_steps'],)` — and `aggregate`'s `calls != [1,1,num_steps]` / `actual_nfe` check), `common/provider.py` (`binding_for`, `validate_binding`: the YAML `adapter_binding.num_steps` + `sampler_identity` choose the MISS sampler via `provider.sampler_for_steps`), how `formal_eval.py`/`formal_server.py` realise `--denoising-steps` for each model (`execution_sampler`, DP DDIM), and the three providers' validators: `models/cosmos/provider.py` `validate_config` (requires num_steps 5 / `make_sampler(5)`), `models/fastwam/ops/library_compatibility.py` (requires K10), DP's equivalent if any.
1. YAML: a reduced-step knob YAML is the existing knob YAML with `adapter_binding.num_steps` and `sampler_identity` set to the reduced execution sampler — exactly the sampler pure inference used under `--denoising-steps` (DP must get DDIM, not DDPM with 10 steps). `recipe.teacher_sampler_identity` stays the full-step library teacher. Make the three validators accept these reduced samplers with the smallest change.
2. Serving: confirm on CPU that a cache-study MISS with that YAML really runs the reduced sampler (and Cosmos reports actual NFE 1). Check batching paths (DP B8, Fast-WAM B16) handle the reduced K; all YAMLs in one campaign use the same reduced K.
3. `cache_study.plan()` and `aggregate()`: take K and the expected NFE per call from the YAML binding instead of the model default (Cosmos K2 → NFE 1).
4. `ops/cache_knob.py`: add `--miss-steps reduced` (or equivalent) to `dev` and `test`. For `dev`, also add one `pure` row per model × suite (put it on the size-500 row set; the `pure` level calls the model at every decision so it does not depend on the library) with B inits 0..29 — this is the reduced-step pure reference on B, because the teacher library is full-step. Write the reduced DEV study to `exp/r17_integration/ops/cache_study/knob/dev_reduced/` using the existing gaps (`exp/r17_integration/ops/cache_study/knob/gaps.jsonl`; the miss-ratio calibration does not depend on K).
5. Analysis (`exp/offline_search/rounds/r17/knob/`): `descent.py --reference pure-arm` uses the lane's own `pure` arm (same model/suite/task/init) as the reference instead of the teacher library; `test.py --controls reduced` compares against the reduced-step pure-inference A runs above (find the exact Cosmos K2 segments from `exp/offline_search/rounds/r17/horizon/COSMOS_K2.md` / `analyze_cosmos_k2.py`). Keep the owner's per-task rule already in `test.py` (≥10 pp below → pure, 6–8 → one level up, iterate until every task is within 4 pp).
6. Print the exact prepare/launch commands following `/home/weiland/projects/openpi_ext/scratch/r17_ops/knob/prepare_dev.sh` (same topology; Cosmos uses the legacy driver on NVMe as in `chain_cosmos_after_fw.sh`).
</what_to_change>

<keep_it_simple>
Owner rule: main function plus a few necessary checks. No new gates, fingerprints, seals or validation layers; full-step behaviour of every existing YAML/campaign stays byte-identical. Delete code you make dead.
</keep_it_simple>

<verification>
CPU tests: reduced YAMLs load and validate for all three models; a synthetic or stubbed MISS uses the reduced sampler (DP DDIM 10, Fast-WAM 1, Cosmos clean-step K2 with NFE 1); plan() episode `num_steps`/`allowed_k` follow the YAML; aggregate accepts NFE 1 for Cosmos; existing knob and cache-study tests still pass. Build the reduced DEV study and report per-model episode counts (21,600 + 600 pure).
</verification>

<action_safety>
Edit only `exp/r17_integration/{ops,common,search,models}` and `exp/offline_search/rounds/r17/knob/` plus tests. Do not start, stop, prepare or launch any campaign; no remote nodes, no GPU. Do not git commit.
</action_safety>

<handback>
Reply with: diff summary per file/function; how each model's reduced sampler is selected in the cache path and how you verified it; the reduced DEV study path and counts; exact commands; anything that could make a MISS silently run full steps.
</handback>
