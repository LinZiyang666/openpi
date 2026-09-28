"""Planning assumptions only; no outcomes or pilot files are read."""
import json
import math
from pathlib import Path

rows=[]
for stage,n,pairs in [('pilot_all',60,2),('pilot_validation',30,2),('full_all',500,1.4),('full_validation',400,1.4)]:
    neff=n/(1+.3*pairs)
    rows.append(dict(stage=stage,assumed_icc=.3,neff=neff,half95=1.96*math.sqrt(.2/neff),mde80=2.801621*math.sqrt(.2/neff)))
for stage,repeats in [('full_equal_init',[3]*100+[2]*50+[1]*100),('full_validation_equal_init',[3]*80+[2]*40+[1]*80)]:
    k=len(repeats);neff=k*k/sum(.3+.7/r for r in repeats)
    rows.append(dict(stage=stage,assumed_icc=.3,neff=neff,half95=1.96*math.sqrt(.2/neff),mde80=2.801621*math.sqrt(.2/neff)))
Path(__file__).with_suffix('.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows))
