from common import *
fence()
from partition import VARIANTS
import numpy as np

rows=[]
for cell in CELLS:
    if cell.endswith('_500'):continue
    host='local_4090' if cell.startswith('groot') else 'h100'
    # Full A mechanics on small libraries; baseline/all-pool at middle size.
    names=list(VARIANTS)+['continuity'] if cell.endswith('_50') else ['base','all_pool']
    for name in names:
        rows.append(dict(arm=f'{cell}_A_{name}',cell=cell,part='A',variant=name,target='',replicate=0,episodes=50,host=host))
    for target in (.32,.40):
        for method in ('periodic','task_periodic'):
            rows.append(dict(arm=f'{cell}_B_{method}_{round(target*100)}',cell=cell,part='B',variant=method,target=target,replicate=0,episodes=50,host=host))
    if cell.endswith('_50'):
        for method in ('periodic','task_periodic'):
            rows.append(dict(arm=f'{cell}_B_{method}_32_rep1',cell=cell,part='B',variant=method,target=.32,replicate=1,episodes=50,host=host))
csvout(HERE/'pilot_grid.csv',rows)
original=(HERE/'REQUESTS.md').read_text().split('\n## Exact queued pilot')[0]
with (HERE/'REQUESTS.md').open('w') as f:
    f.write(original)
    f.write('\n## Exact queued pilot\n\n'+table(['cell','arms','episode runs','host'],[
      [c,sum(r['cell']==c for r in rows),sum(r['episodes'] for r in rows if r['cell']==c),
       next(r['host'] for r in rows if r['cell']==c)] for c in CELLS if not c.endswith('_500')])+ '\n\n'
      f"Total: {len(rows)} arms / {sum(r['episodes'] for r in rows)} episode runs; paired {50//10} existing B inits per task, exact rows in `pilot_grid.csv`. "
      'The repeat at the lower dose estimates implementation stability, not additional independent initial states. '
      'Please record all pilot real looks; off/base arms need shadow policy chunks. '
      'No extra shadow evaluation is required in periodic arms. Run all GR00T arms on the local 4090 and all pi05 arms on H100. '
      'Do not split a cell across hosts. This is a queued request, not a launch.\n')
dump(HERE/'early_request.json',dict(timestamp=now(),arms=len(rows),episodes=sum(r['episodes'] for r in rows),pilot_sha=sha(HERE/'pilot_grid.csv')))
print(len(rows),sum(r['episodes'] for r in rows))
