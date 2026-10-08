"""Freeze candidate and epoch selection from library episode hold-outs only."""
import argparse
import json
from datetime import datetime, timezone
import numpy as np
from exp.offline_search.rounds.r15.qk.astra.prepare import HERE,dump,sha
from exp.offline_search.rounds.r15.qk.astra.round2_core import CONFIGS


def main(size):
    cell=f'pi05_l10_{size}';root=HERE/'round2'/cell;candidates=[];inputs={}
    for name in CONFIGS:
        folds=[]
        for fold in (0,1):
            file=root/name/f'cv{fold}_train.json';d=json.loads(file.read_text());inputs[str(file)]=sha(file)
            assert not set(d['fit_episodes'])&set(d['heldout_episodes']);assert d['cpu_only']
            folds.append({x['epoch']:x for x in d['checks']})
        for epoch in (40,80,120):
            checks=[f[epoch] for f in folds]
            candidates.append({'name':name,'epoch':epoch,'mean_fold_ratio':float(np.mean([x['ratio'] for x in checks])),
                'fold_ratios':[x['ratio'] for x in checks],
                'shuffle_rise_mean':float(np.mean([x['blindness']['shuffle_mean_relative_rise'] for x in checks])),
                'zero_rise_mean':float(np.mean([x['blindness']['zero']['relative_rise'] for x in checks])),
                'blindness_pass_both_folds':all(x['blindness']['pass'] for x in checks)})
    ranked=sorted(candidates,key=lambda x:(not x['blindness_pass_both_folds'],x['mean_fold_ratio'],x['name'],x['epoch']))
    chosen=ranked[0];assert chosen['blindness_pass_both_folds']
    result={'cell':cell,'created_utc':datetime.now(timezone.utc).isoformat(),'selection_source':'two within-library episode hold-outs; no B-dev inputs',
            'selected':chosen,'all_candidates':ranked,'input_sha256':inputs,'gates_sha256':sha(HERE/'GATES.md')}
    dump(root/'selection.json',result);print(json.dumps({'cell':cell,'selected':chosen}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--size',type=int,required=True);a=ap.parse_args();main(a.size)
