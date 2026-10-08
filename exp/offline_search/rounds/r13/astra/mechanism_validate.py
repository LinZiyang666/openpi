"""Independent reconciliations for the B mechanism deliverable; no fitting."""
from mechanism_extract import HERE, read, dump, sha, frozen
import ast
import numpy as np
import pandas as pd

def main():
    n=read(HERE/'mechanism_numbers.json');inputs=read(HERE/'mechanism_inputs.json')
    gate=read(HERE/'gates.json');d=pd.read_pickle(HERE/'mechanism_replayed.pkl')
    inv={r['arm']:r for r in inputs['inventory'] if r['status']=='complete'}
    gates=[]
    for g in gate['arms']:
        r=inv[g['arm']];t=g['totals'];z=d[d.arm==g['arm']]
        vals={'N':r['N'],'V':r['looks'],'M':r['calls'],'G':r['guards']}
        assert vals=={k:t[k] for k in vals},(g['arm'],vals,t)
        assert len(z)==r['looks'] and int(z.call.sum())==r['calls']
        gates.append(g['arm'])
    for r in n['arms']:
        p=r['placement'];h=r['gap_hazard']
        assert sum(v['looks'] for v in h)==p['looks']
        assert sum(v['knobs'] for v in h)==p['knobs']
        assert p['calls']==p['guards']+p['knobs']
        for field in ['controls','fraction','progress']:
            assert sum(v['looks'] for v in r['position_hazard'][field])==p['looks'],(r['arm'],field)
    assert n['counts']['looks']==len(d) and n['counts']['episodes']==d.ep.nunique()
    assert np.allclose(d.fraction,d.controls/d.final_controls)
    assert ((d.fraction>=0)&(d.fraction<1)).all()
    assert sha(HERE/'mechanism_episode_lengths.json')==n['episode_lengths_sha256']
    for r in n['same_state_replay']:
        for k in ['knob_calls','all_calls']:
            q=r[k]
            assert q['symmetric_difference']==q['a_only']+q['b_only']
            assert q['union']==q['both']+q['symmetric_difference']
            assert q['n']==q['union']+q['neither']
    for (cell,behavior),rs in pd.DataFrame(n['shadow_selected_means']).groupby(['cell','behavior']):
        r={v['selection']:v for v in rs.to_dict('records')}
        for method,one in [('distance_all','distance_only'),('periodic_all','periodic_only')]:
            assert r[method]['n']==r[one]['n']+r['both']['n']
            expected=sum(r[k]['n']*r[k]['risk10_mean'] for k in [one,'both'])/r[method]['n']
            assert np.isclose(expected,r[method]['risk10_mean'])
    for name,digest in n['scripts'].items():
        assert sha(HERE/name)==digest,name
        ast.parse((HERE/name).read_text())
    owner=(HERE/'MECHANISM.md').read_text().split('\n---\n')[0]
    assert not any(x in owner for x in ['R13','astra','GR00T','groot','pi05','test-A','distance_feedback','periodic'])
    f=frozen();assert len(f)==n['audit']['frozen_files']
    result=dict(PASS=True,pilot_arms_independently_reconciled_to_gate_audit=len(gates),
        hazard_and_position_denominators_reconciled_arms=len(n['arms']),
        all_replay_set_identities_valid=True,shadow_selection_partitions_valid=True,
        actual_final_control_lengths_and_fractions_valid=True,
        source_script_hashes_valid=True,owner_section_no_codenames=True,
        frozen_members_unchanged=len(f),numbers_sha256=sha(HERE/'mechanism_numbers.json'),
        report_sha256=sha(HERE/'MECHANISM.md'),validator_sha256=sha(__file__))
    dump(HERE/'mechanism_validation.json',result)
    print(result)

if __name__=='__main__':main()
