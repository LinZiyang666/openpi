"""Fresh R10 lower-layer fit plus library-only layer-4 calibration."""
from exp.offline_search.rounds.r10.recipe.fitting import fit_recipe
from .calibration import calibrate

def fit_knob(recipe, ctx):
    fit_recipe(recipe,ctx)
    sizes={len(v) for v in recipe.fit_info['episode_ids_by_task'].values()}
    if len(sizes)!=1:
        raise ValueError('R11 frozen calibrations require an R10 nested subset')
    size=recipe.size
    from exp.offline_search.rounds.r10.data import HERE as R10
    import json
    expected=json.loads((R10/'subsets'/f'{ctx.model}_{ctx.suite}_{size}.json').read_text())['episode_ids_by_task']
    if recipe.fit_info['episode_ids_by_task']!=expected:
        raise ValueError('R11 calibration selection must match the nested subset exactly')
    cfg=calibrate(ctx.model,ctx.suite,size,recipe.knob_method,recipe.target)
    recipe.configure(cfg,ctx.model)
