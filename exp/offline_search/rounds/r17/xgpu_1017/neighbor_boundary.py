"""Nearest-neighbour boundary audit matching distances by donor identity.

Sorted first/second distances alone can hide a rank exchange with an unchanged
gap. Missing support is unknown, never evidence of being away from a boundary.
"""
import math


def audit(left,right):
    groups=[]
    for values in (left,right):
        group=[]
        for hit in values:
            ids=hit.get('donor_rows');distances=hit.get('xgpu_donor_distances')
            if not ids or distances is None or len(ids)!=len(distances) or len(ids)<2:
                return dict(qualified=False,reason='missing selected donor identities/distances')
            if len(set(ids))!=len(ids) or not all(isinstance(d,(int,float)) and math.isfinite(d) for d in distances):
                return dict(qualified=False,reason='invalid native donor distances')
            group.append((ids,dict(zip(ids,distances))))
        if not group:return dict(qualified=False,reason='no observations')
        groups.append(group)
    candidates=set(i for group in groups for ids,_ in group for i in ids[:2])
    if any(not candidates<=set(mapping) for group in groups for _,mapping in group):
        return dict(qualified=False,reason='a top-two candidate is outside another selected donor set')
    floor=max(max(m[i] for _,m in group)-min(m[i] for _,m in group)
        for group in groups for i in candidates)
    cross=max(abs(a[i]-b[i]) for i in candidates for _,a in groups[0] for _,b in groups[1])
    gap=min(mapping[ids[1]]-mapping[ids[0]] for group in groups for ids,mapping in group)
    if gap<0:return dict(qualified=False,reason='native distances are not sorted')
    band=max(1e-6,5*floor,2*cross)
    return dict(qualified=True,candidate_ids=sorted(candidates),distance_floor=floor,
        distance_cross_error=cross,minimum_gap=gap,boundary_band=band,near_boundary=gap<=band,
        scope='all observed top-two identities; does not bound an unobserved third candidate')
