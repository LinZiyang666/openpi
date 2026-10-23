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


def load(root):
    snapshot=cache.receipt_snapshot(Path(root))
    loaded=cache.load_lane(Path(root),snapshot,cache.Sources(),c.SIZES,suites=c.SUITES)
    return loaded,snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--family',action='append',help='Original family name, e.g. csfw_k3')
    p.add_argument('--final-chains',type=Path,help='JSON: canonical name -> final chain tip (fresh fa_ls_ continuations / pod copies)')
    p.add_argument('--jobs',type=int,default=1,help='Lanes read in parallel')
    p.add_argument('--out',type=Path,default=HERE/'CACHE_VARIANT_RESULTS.json');a=p.parse_args()
    final=read(a.final_chains)['runs'] if a.final_chains else {}
    catalog=read(HERE/'CAMPAIGNS.json');jobs={j['name']:j for j in catalog['jobs']};rows=[]
    families=[f for f in catalog['source_cohorts'] if not a.family or Path(f['source']).name in a.family]
    roots={n:Path(final.get(n,jobs[n]['run'])) for f in families for n in f['jobs']}
    roots={n:r for n,r in roots.items() if (r/'RECEIPT.json').exists()}
    for n,r in roots.items():
        # Final-chain tips continue fresh fa_ls_ segments only; load_lane walks them back.
        if n not in final:assert not read(r/'campaign.json').get('cache_resume_selection')
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max(1,a.jobs)) as pool:
        lanes=dict(zip(roots,pool.map(load,[str(r) for r in roots.values()])))
    with filesystem_budget():
        for family in families:
            cells=defaultdict(dict);lineages=[]
            for n in family['jobs']:
                if n not in lanes:continue
                loaded,snapshot=lanes[n]
                lineages.append(dict(name=n,run=str(roots[n]),segments=[s['lane'] for s in snapshot['segments']],
                    receipts=[s['status'] for s in snapshot['segments']],accepted=snapshot['accepted_episodes'],
                    cost_episodes=snapshot['cost_episodes']))
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
        dump(a.out,dict(rows=rows,clock_unit='segment_decision',final_chains=final or None))
if __name__=='__main__':main()
