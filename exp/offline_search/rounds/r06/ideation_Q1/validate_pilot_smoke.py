"""Check the final smoke artifacts without interpreting their scientific estimates."""
from pathlib import Path
import json

import pandas as pd
import pilot_q1 as q


def main():
    checks={}
    for folder in ['p3_smoke_release2','p3_smoke_reader_release2']:
        p=q.HERE/folder
        m=json.loads((p/'run_manifest.json').read_text())
        decision=json.loads((p/'decision.json').read_text())
        assert m['script_sha256']==q.sha(q.HERE/'pilot_q1.py')
        assert m['smoke'] and not m['scientific_conclusions_allowed'] and decision['selected'] is None
        a=pd.read_csv(p/'anchor_scores.csv');e=pd.read_csv(p/'episode_scores.csv');t=pd.read_csv(p/'per_task_state_estimates.csv')
        assert not a[['C','D','R','Qrisk','E']].isna().any().any()
        assert t.loc[t.distinct_inits==1,['rho_lo','rho_hi']].isna().all().all()
        params=json.loads((p/'calibration_parameters.json').read_text())
        valid=set(e[e.split=='validation'].uid)
        assert all(not valid.intersection(v['calibration_episodes']) for v in params)
        assert all(v=='SMOKE_ONLY' for v in decision['status'].values())
        checks[folder]=dict(counts=m['counts'],catalog_bridges=m['bridge_passes'],
            csv_files=len(list(p.glob('*.csv'))),json_files=len(list(p.glob('*.json'))),
            plots=len(list(p.glob('*.png')))+len(list(p.glob('*.pdf'))),bootstrap=m['bootstrap'],
            labels_and_four_primary_scores_complete=True,calibration_validation_disjoint=True,
            one_init_intervals_suppressed=True,method_selection_suppressed=True,source_hash_matches=True)
    q.dump(q.HERE/'pilot_q1_smoke_validation.json',checks)
    print(json.dumps(checks))


if __name__=='__main__':main()
