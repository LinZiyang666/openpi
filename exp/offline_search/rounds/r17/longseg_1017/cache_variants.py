"""Reanalyze fresh cache-variant cohorts; never read their withdrawn ancestors."""
import argparse
from collections import defaultdict
from pathlib import Path
from .audit import HERE
from .io_budget import dump,filesystem_budget
from .analyze import paired,decision_summary
from exp.offline_search.rounds.r17.filter_failed_knob.common import read
from exp.offline_search.rounds.r17.knob import common as c
from exp.offline_search.rounds.r17.cache import analyze_cache as cache


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--family',action='append',help='Original family name, e.g. csfw_k3');a=p.parse_args()
    catalog=read(HERE/'CAMPAIGNS.json');jobs={j['name']:j for j in catalog['jobs']};rows=[]
    with filesystem_budget():
        for family in catalog['source_cohorts']:
            if a.family and Path(family['source']).name not in a.family:continue
            cells=defaultdict(dict);sources=cache.Sources();lineages=[]
            for n in family['jobs']:
                j=jobs[n];root=Path(j['run'])
                if not (root/'RECEIPT.json').exists():continue
                assert not read(root/'campaign.json').get('cache_resume_selection')
                loaded=cache.load_lane(root,cache.receipt_snapshot(root),sources,c.SIZES,suites=c.SUITES)
                lineages.append(n)
                for key,value in loaded.items():
                    assert not cells[key].keys()&value['data'].keys(),'Fresh cohorts overlap'
                    cells[key].update(value['data'])
            for (model,size,group,suite),data in cells.items():
                comparisons={}
                for feature,letter in [('corrector','r'),('long','h'),('stall','g')]:
                    if letter in group:
                        control=cells.get((model,size,group.replace(letter,''),suite),{})
                        comparisons[feature]=paired(control,data)
                rows.append(dict(family=Path(family['source']).name,runs=lineages,model=model,suite=suite,size=size,group=group,
                    **decision_summary(data,model),paired_differences=comparisons,
                    no_old_outcomes_inherited=True,
                    reference_inventory=str(HERE/'VARIANT_REFERENCES.json'),
                    ir_note='Within-study model calls/NFE per control and measured service cost remain available; do not substitute a pure/profile with a different frozen model identity.'))
        dump(HERE/'CACHE_VARIANT_RESULTS.json',dict(rows=rows,clock_unit='segment_decision'))
if __name__=='__main__':main()
