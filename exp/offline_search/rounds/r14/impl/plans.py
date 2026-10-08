"""Standard asset planner on the final B and staged A definitions; no sync."""
import json
from pathlib import Path
from exp.offline_search.closed_loop.ops.h100.assets import build_plan
from .common import *

def main():
    records=[]
    for filename in ('pilot_roots.json','test_a_roots.json'):
        for rec in json.loads((HERE/filename).read_text()):
            root=Path(rec['root']);local=rec['intended_host']=='local_4090'
            work=HERE/'plans'/('local' if local else 'h100')/root.name
            plan=build_plan(root,rec['arms'],work,local=local)
            # The planner writes only regular files and does not contact hosts.
            path=work/'plan.json'
            record=dict(root=str(root),host='local' if local else 'h100',arms=len(rec['arms']),
                        path=str(path),sha256=sha(path),files=len(plan['files']),
                        bytes=sum(f['size'] for f in plan['files']),
                        init_pool=rec['init_pool'])
            records.append(record)
            print('PLAN',root.name,record['files'],record['bytes'],flush=True)
    write_json(HERE/'plans.json',dict(PASS=True,records=records,no_sync=True,no_servers=True))

if __name__=='__main__':main()
