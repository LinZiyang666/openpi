#!/usr/bin/env python3
"""Read back the frozen, preregistered pilot outputs of Q1 / Q2 / Q3 (ANALYSIS.md 4); no re-estimation.

The pilot (r06_p3_pilot, 216 arms, 4,320 episodes) used the P3 client and has no exception-terminated episodes (see
a1_repair_audit.py), so the frozen outputs are unaffected by the repair. This script only verifies the decisions that
ANALYSIS.md quotes. Writes analysis_r6/prereg_checks.json.
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm  # noqa: E402

R6 = cm.R6


def main():
    q1d = json.load(open(f'{R6}/ideation_Q1/pilot_all_cells/decision.json'))
    q1s = [r for r in csv.DictReader(open(f'{R6}/ideation_Q1/pilot_all_cells/state_estimates.csv'))
           if r['scope'] == 'pooled' and r['cohort'] == 'A']
    q1 = dict(decision=q1d, pooled={r['candidate']: dict(rho=float(r['rho']), rho_sim=(float(r['rho_sim_lo']), float(r['rho_sim_hi'])),
                                                         gain=float(r['mae_gain']), gain_sim=(float(r['gain_sim_lo']), float(r['gain_sim_hi'])),
                                                         mae=float(r['mae']), baseline=float(r['baseline_mae']))
                                      for r in q1s})
    q2e = json.load(open(f'{R6}/ideation_Q2/final_analysis/preregistered/estimates.json'))
    q2 = dict(recommendations={r['cell']: (r['recommendation'], len(r['eligible']), r['reason']) for r in q2e['recommendations']})
    prim = [r for r in csv.DictReader(open(f'{R6}/ideation_Q3/final_pilot/primary64.csv'))]
    est = [r for r in prim if r['lo_primary'] not in ('', 'nan') and r['hi_primary'] not in ('', 'nan')]
    excl = [r for r in est if float(r['lo_primary']) > 0 or float(r['hi_primary']) < 0]
    nom = list(csv.DictReader(open(f'{R6}/ideation_Q3/final_pilot/nomination48_audit.csv')))
    elig = [r for r in nom if r['inside_nomination_support'] == 'True' and r['enrichment_nomination_support'] == 'True'
            and r['inside_lo_selection'] not in ('', 'nan') and float(r['inside_lo_selection']) > 0
            and r['enrichment_lo_selection'] not in ('', 'nan') and float(r['enrichment_lo_selection']) > 0]
    q3 = dict(primary_rows=len(prim), estimable=len(est), simultaneous_exclusions=len(excl), nomination_rows=len(nom),
              eligible_nominations=len(elig))
    out = dict(Q1=q1, Q2=q2, Q3=q3)
    cm.dump('prereg_checks.json', out)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
