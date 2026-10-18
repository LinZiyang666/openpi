"""Observe existing recipe results; never recompute a query or change decisions.

Only already selected donor distances are exposed. A runner-up gap is unavailable
for k=1; the observer must not invent it or run a second nearest-neighbour query.
"""
import math


def install(recipe_class=None):
    if recipe_class is None:
        from exp.r17_integration.common.layers.recipe import LayoutRecipe as recipe_class
    if getattr(recipe_class,'_xgpu_cache_observed',False):
        raise RuntimeError('cache observer installed twice')
    recipe_class._xgpu_cache_observed=True
    native=recipe_class.query
    def query(self,q,*args,**kwargs):
        result=native(self,q,*args,**kwargs)
        # Append diagnostics after the native action/weights/force_miss exist.
        # Lists and scalars only: no extra matmul, RNG, GPU sync or retrieval.
        distances=[float(d) for d in result.distances]
        if not distances or not all(math.isfinite(d) for d in distances):
            raise ValueError('native cache distances are nonfinite/empty')
        extra=dict(xgpu_donor_distances=distances,nearest_distance=distances[0],
            neighbour_gap=distances[1]-distances[0] if len(distances)>1 else None,
            xgpu_cache_observer='native_result_only_v1')
        raw=result.extras.get('raw_anchor_distance')
        scale=getattr(self,'distance_scale',None)
        if isinstance(raw,(int,float)) and scale is not None and math.isfinite(float(scale)) and scale>0:
            ratio=float(raw/scale)
            extra.update(corrector_ratio=ratio,corrector_lower_margin=ratio-.75,
                corrector_upper_margin=ratio-2.,
                corrector_ramp_scope='distance ramp only; not the Boolean corrector_applied decision')
        result.extras.update(extra)
        return result
    recipe_class.query=query
