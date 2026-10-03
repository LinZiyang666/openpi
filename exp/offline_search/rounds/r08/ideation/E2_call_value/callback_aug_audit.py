"""Audit deferred part provenance against discovery-only observed anchor parity."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from callback_calls import RUN, OUT, save


summaries=[]
for directory in sorted((OUT/'shadows').iterdir()):
    check_path=directory/'fresh_anchor_checks.parquet'
    if not check_path.exists():continue
    name=directory.name
    checks=pd.read_parquet(check_path)
    blind=pd.read_parquet(directory/'look_policy_six_motion.parquet')
    ids=set(checks.decision_id)|set(blind.decision_id)
    if (OUT/'calls'/name/'decisions.parquet').exists():
        df=pd.read_parquet(OUT/'calls'/name/'decisions.parquet')
        assert df.init.max()==29
        ids=set(df.decision_id)
        checks=checks.merge(df[['decision_id','task_id','init','src']],on='decision_id')
    records=[]
    for kind in ['shadow_look','policy_shadow','policy_draws']:
        for path in sorted((RUN/'runs'/name/'debug'/'aug'/kind).glob('part_*.npz')):
            with np.load(path,allow_pickle=False) as z:
                meta=json.loads(str(z['_meta_json']))
                selected=[str(v) for v in z['decision_id'] if str(v) in ids]
            records.extend(dict(decision_id=did,kind=kind,part=path.name,code_sha=meta.get('code_sha'),
                                fit_config_sha=meta.get('fit_config_sha'),checkpoint_sha=meta.get('checkpoint_sha'))
                           for did in selected)
    provenance=pd.DataFrame(records)
    provenance.to_parquet(directory/'aug_provenance_discovery.parquet',index=False)
    merged=checks.merge(provenance[provenance.kind.eq('shadow_look')],on='decision_id',validate='one_to_one')
    merged.to_parquet(directory/'anchor_parity_by_provenance.parquet',index=False)
    rows=[]
    for code,g in merged.groupby('code_sha'):
        rows.append(dict(code_sha=code,anchors=len(g),exact=int(g.max_abs.eq(0).sum()),
                         over_1e_4=int(g.max_abs.gt(1e-4).sum()),over_01=int(g.max_abs.gt(.01).sum()),
                         max_abs=g.max_abs.max(),median_abs=g.max_abs.median(),
                         first_example=g.loc[g.max_abs.idxmax(),'decision_id']))
    summary=dict(arm=name,anchors=len(checks),blind=len(blind),by_code=rows,
                 pooled_shadow_interpretation='withheld_anchor_replay_mismatch' if any(r['exact']!=r['anchors'] for r in rows)
                     else 'diagnostic_only_anchor_prefix_matched',
                 parts=provenance.groupby(['kind','code_sha']).agg(decisions=('decision_id','size'),parts=('part','nunique')).reset_index().to_dict('records'))
    save(directory/'aug_audit.json',summary)
    summaries.append(summary)
    print(json.dumps(dict(arm=name,by_code=rows)),flush=True)
save(OUT/'summary'/'aug_audit.json',summaries)
look_path=OUT/'summary'/'look_summary.csv'
if look_path.exists():
    frame=pd.read_csv(look_path)
    labels={r['arm']:r['pooled_shadow_interpretation'] for r in summaries}
    frame['pooled_shadow_interpretation']=frame.arm.map(labels).fillna('anchor_replay_audit_pending')
    frame.to_csv(look_path,index=False)
