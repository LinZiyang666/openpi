"""Reject scientific-plan/serving-identity mixing in normalized paired data."""
from collections import defaultdict

FIELDS=('plan_identity','serving_pin','benchmark_protocol','runtime_identity')
GROUP=('model','suite','arm')


def check(left,right):
    sides=[]
    for rows in (left,right):
        groups=defaultdict(list)
        for row in rows:groups[tuple(row[k]for k in GROUP)].append(row)
        sides.append(groups)
    reports=[]
    for group in sorted(set(sides[0])|set(sides[1])):
        a,b=(side.get(group,[])for side in sides)
        report=dict(group=list(group),qualified=bool(a and b),fields={})
        for field in FIELDS:
            av={r.get(field)for r in a};bv={r.get(field)for r in b}
            if not a or not b:
                report['qualified']=False;report['fields'][field]=None;continue
            if av==bv=={None}:
                report['qualified']=False;report['fields'][field]=None;continue
            if None in av or None in bv or len(av)!=1 or len(bv)!=1 or av!=bv:
                raise ValueError('paired scientific binding differs/mixed: '+field+' '+str(group))
            report['fields'][field]=next(iter(av))
        reports.append(report)
    return reports
