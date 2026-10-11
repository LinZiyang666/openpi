<task>
Add two small, default-off serving features to the R17 cache-study path so the per-task IR knob (design layer 4/5) can run on top of the complete group (corrector + long horizon + stall gate). CPU work only.

1. Periodic knob MISS inside the cache study.
   - Today `search/cache_recipe.py` `CacheStudyRecipe.query` raises `cache-study base unexpectedly requested an ordinary MISS` when the base `LayoutRecipe.query` sets `force_miss`, and `common/serving.py` (~line 349) raises `ordinary MISS forbidden by cache-study YAML` unless `stall_call` is true.
   - Change: when `config.recipe.periodic_gap is not None`, a base `force_miss` that comes from the periodic knob (`result.extras["knob"]` true; see `common/layers/recipe.py` query, `os_reason` 62) is allowed. Final call = stall call OR knob. Record `knob_call` (bool) in extras. Keep `os_reason` 63 for stall, 62 for knob.
   - `CacheStudyRecipe.force_look` must also return true when the base periodic rule wants a look (`self.base.force_look(request, has_vision, hit)`), in addition to the existing stall `look_due`. Its signature already receives those arguments from `serving.py` line ~330.
   - Serving check: allow `force_miss` when `stall_call` or `knob_call` is true. With `periodic_gap: null` behaviour must be byte-identical to today (stall-only).
   - Guard (`recipe.guard`) stays false in these YAMLs; do not touch NoProgressGuard.
2. Serving LOEO for B-pool queries.
   - New `RecipeConfig` field `serve_loeo: bool = False` (`common/config.py`).
   - When true, every retrieval excludes the library episode whose (task, B init index) equals the query's (`request.task_id`, `request.original_init_idx`). Build the map once at load from the store's parent `READY.json` `rows` (fields `library_episode_id`, `task_name`, `orig_init_idx`; task id = index in `recipe.ordered_tasks`). The fitted recipe's `episode` array already holds `library_episode_id` (see `search/fit.py` `library_data`). If the query's own episode is not in the selected subset, nothing is excluded.
   - Apply it in the normal `LayoutRecipe.query` path (`common/layers/recipe.py`: `result, xv, rs = self._retrieve(q)` → pass the exclusion). Leave the existing `query(q, exclude_episode=...)` shortcut alone.
   - Where the factory builds the recipe (`common/recipe.py` `make_recipe_factory`), attach the map when `serve_loeo` is true. The store path in served YAMLs is `.../cache_inputs/<suite>/parent/subsets/size_N`; its parent `READY.json` is two levels up (`VIEW.json` `parent_path`).
   - If `fit_identity` / `recipe_identity` / `bundle_identity` hashing (common/recipe.py) must see the new field so dev and test YAMLs get different identities, include it in `recipe_identity`/`bundle_identity` only; do not change `fit_identity`.
</task>

<keep_it_simple>
Owner rule: implement only these two features plus the minimum checks needed for correctness. Do not add gates, fingerprints, seals, admission checks, new validation layers or defensive raises. Default values keep every existing YAML behaving exactly as now. Experiments are running from installed snapshots; edits must be backward compatible.
</keep_it_simple>

<verification>
- Unit-test on CPU with a tiny synthetic recipe or the existing test helpers (`exp/r17_integration/tests`, `models/*/test_cache.py`): (a) periodic_gap null → identical decisions to before; (b) periodic_gap 0 → every decision is a model call (pure inference); (c) gap 1.9 → mixture of stall and knob calls, extras show `knob_call`; (d) serve_loeo excludes exactly the own episode and nothing else.
- Real-data CPU check: load `/data/mw_finetune/eval/csztd_r3_l48/assets/cache_yamls/n50_sp_chrg.yaml` with `serve_loeo: true` against `/data/r17_library/dp_libero_libero_spatial/bpool` and confirm that a query with (task 0, init i) never returns rows of that init's library episode while other inits are unaffected.
- Run the affected existing tests.
</verification>

<action_safety>
Edit only: `exp/r17_integration/search/cache_recipe.py`, `exp/r17_integration/common/serving.py`, `exp/r17_integration/common/config.py`, `exp/r17_integration/common/recipe.py`, `exp/r17_integration/common/layers/recipe.py`, and tests. Another agent is editing `exp/r17_integration/ops/*` in parallel — do not touch ops. Do not start/stop any campaign, process, GPU, or remote node. Do not git commit.
</action_safety>

<handback>
Reply with: the diff summary (file:function, a few lines each), test commands and results, and anything in the existing code that blocked the minimal change.
</handback>
