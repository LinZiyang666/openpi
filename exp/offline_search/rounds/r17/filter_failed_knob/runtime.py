"""A resident-only row view over the original verified fit, with no refitting.

Install before formal_server imports its recipe factory. The original fit is
still used to attest/load the unchanged Q3 metric; only retrieval sees the view.
"""
from __future__ import annotations


def install(selection):
    import numpy as np
    import inspect
    import textwrap
    from exp.offline_search.rounds.r17.filter_failed.row_filter import r17_candidates
    from exp.r17_integration.search import cache_recipe as cr
    from exp.r17_integration.common.query_view import RecipeResult

    native_fit, native_stall = cr.verified_fit, cr.load_stall
    native_query, native_look = cr.CacheStudyRecipe.query, cr.CacheStudyRecipe.force_look

    def verified(config, binding):
        original = native_fit(config, binding)
        kept = selection.get(config.recipe.fitted_digest)
        if kept is None or config.recipe.periodic_gap == 0:
            return original
        selected = r17_candidates(original, kept)
        selected._ffk_original_fit = original
        return selected

    def stall(config, recipe):
        # Q3 attests the original frozen metric, whose rows/codes/parameters
        # stay byte-identical. Its query_code uses the unchanged Wf/PCA.
        return native_stall(config, getattr(recipe, '_ffk_original_fit', recipe))

    def empty(self, request):
        table = self.base.tasks.get(request.task_id)
        if table is None or not len(table.rows):
            return True
        own = self.base.serve_loeo_episodes.get((request.task_id, request.original_init_idx))
        return own is not None and not np.any(self.base.episode[table.rows] != own)

    def query(self, q):
        if empty(self, q.request):
            # The normal serving branch requires an explicit permitted model
            # call. No donor, correction or stall tracker is consulted here.
            return RecipeResult(None, np.empty(0, np.int64), np.empty(0), np.empty(0), True,
                                dict(knob_call=False, empty_candidates=True, os_force_miss=1.,
                                     stall_call=False, corrector_applied=False))
        return native_query(self, q)

    def look(self, request, has_vision, hit):
        return empty(self, request) or native_look(self, request, has_vision, hit)

    cr.verified_fit, cr.load_stall = verified, stall
    cr.CacheStudyRecipe.query, cr.CacheStudyRecipe.force_look = query, look
    # Permit precisely this newly possible MISS in the frozen serving branch.
    # Every other ordinary-MISS restriction and the original model path remain.
    from exp.r17_integration.common import serving
    source = textwrap.dedent(inspect.getsource(serving.ConnectionPolicy.infer))
    old = 'if miss_mode != "legacy" and ('
    new = 'if miss_mode != "legacy" and not diagnostics.get("empty_candidates", False) and ('
    if old not in source:
        raise ValueError('Frozen empty-MISS branch needs inspection')
    namespace = dict(vars(serving))
    exec(compile(source.replace(old,new,1), inspect.getfile(serving), 'exec'), namespace)
    serving.ConnectionPolicy.infer = namespace['infer']


def main():
    import argparse
    import json
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--directory', required=True)
    p.add_argument('--cuda-graph', action='store_true')
    args = p.parse_args()
    campaign = json.loads(Path(args.campaign).read_text())
    install(campaign['filter_failed_knob']['selection'])
    if campaign.get('ffk_dp_capacity_bridge'):
        from exp.offline_search.rounds.r17.filter_failed.server_cli import bridge
        from exp.r17_integration.models.dp import formal_batch_server
        bridge(formal_batch_server)
    from exp.r17_integration.ops.formal_server import serve
    return serve(campaign, args.port, args.directory, args.cuda_graph)


if __name__ == '__main__': main()
