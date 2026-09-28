"""Independent arithmetic checks and byte-identical final rerun verification."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from exp.offline_search.rounds.r04.k5_rand import estimate as k5
from exp.offline_search.rounds.r05.q3_callvalue import cost_solver_reference as solver

HERE = Path(__file__).resolve().parent


def array(rows, rho, probs=None, exposed_only=False):
    """Independent per-cluster CALL-CACHE contrast, with optional suppression weights."""
    groups = k5.validate_pairs(rows)
    ans = []
    for cluster in sorted(groups):
        v = np.zeros(4)
        for r in groups[cluster]:
            if exposed_only and not r['exposed']:
                continue
            p = 1.
            if probs is not None:
                p = probs.get((r['landmark_class'],json.dumps(r['context'],sort_keys=True)),0.) if r['exposed'] else 0.
            # HT 1/.5 divided by two replicate observations per cluster.
            p *= (1 if r['assigned_treatment'] == 'CALL' else -1) / r['propensity'] / 2
            v += p * np.array([r['Y'],r['N'],r['M'],.152*r['N']+.848*r['M']-rho*r['N']])
        ans.append(v)
    return np.array(ans)


def assert_stats(vals, stats):
    expected = vals.mean(0) * [-1,-1,-1,1]
    names = ['SR_change','N_change','M_change','cost_saving']
    se = vals.std(0,ddof=1) / np.sqrt(len(vals))
    np.testing.assert_allclose([stats['mean'][n] for n in names],expected,atol=1e-12)
    np.testing.assert_allclose([stats['SE'][n] for n in names],se,atol=1e-12)
    for suffix, z in [('pointwise',norm.ppf(.975)),('bonferroni',norm.ppf(1-.05/(2*stats['comparisons_per_outcome'])))]:
        np.testing.assert_allclose([stats['ci95_'+suffix][n] for n in names],np.stack([expected-z*se,expected+z*se],1),atol=1e-12)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--first',type=Path,default=HERE/'results')
    ap.add_argument('--rerun',type=Path,default=HERE/'rerun')
    args=ap.parse_args()
    counts={'identical_files':0,'fits_checked':0,'cells_checked':0,'fold_OPE_checked':0,'nonzero_OPE_checks':0,'synthetic_solver_checks':0}
    compared=[]
    for file in sorted(args.first.iterdir()):
        if file.suffix not in ('.json','.md'):
            continue
        other=args.rerun/file.name
        assert file.read_bytes()==other.read_bytes(), f'rerun mismatch: {file.name}'
        compared.append({'name':file.name,'bytes':file.stat().st_size,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
        counts['identical_files']+=1
    for scale in (500,50):
        rows=json.loads((args.first/f'audited_g{scale}_episodes.json').read_text())['episodes']
        result=json.loads((args.first/f'g{scale}.json').read_text());rho=result['rho']
        for family in ('leaf','parent'):
            rr=rows if family=='leaf' else [{**r,'context':{} if r['exposed'] else None} for r in rows]
            a=result[family]
            entries=[(rr,a['fit'])]
            for split in ('init_crossfit','task_crossfit'):
                index=(lambda r:r['init']%5) if split=='init_crossfit' else (lambda r:r['task_id'])
                for fold in a[split]['folds']:
                    train=[r for r in rr if index(r)!=fold['fold']]
                    test=[r for r in rr if index(r)==fold['fold']]
                    entries.append((train,fold['fit']))
                    probs={tuple(c['key']):c['suppress_probability'] for c in fold['fit']['cells']}
                    assert_stats(array(test,rho,probs),fold['heldout'])
                    counts['fold_OPE_checked']+=1
                pooled=np.array([[c[n] for n in ('SR_change','N_change','M_change','cost_saving')] for c in a[split]['cluster_contributions']])
                assert_stats(pooled*[-1,-1,-1,1],a[split]['pooled'])
            for train, fit in entries:
                z=norm.ppf(1-.05/(2*len(fit['cells'])))
                np.testing.assert_allclose(z,fit['z_simultaneous_normal'])
                for c in fit['cells']:
                    kk=tuple(c['key'])
                    selected=[r for r in train if r['exposed'] and (r['landmark_class'],json.dumps(r['context'],sort_keys=True))==kk]
                    vals=array(train,rho,{kk:1.})
                    mean=vals.mean(0);se=vals.std(0,ddof=1)/np.sqrt(len(vals))
                    np.testing.assert_allclose(mean,c['delta_Y_N_M_C_population'],atol=1e-12)
                    np.testing.assert_allclose(se,c['SE'],atol=1e-12)
                    assert len(selected)==c['n']
                    assert len({(r['task_id'],r['init']) for r in selected})==c['clusters']
                    assert sum(r['assigned_treatment']=='CALL' for r in selected)==c['n_call']
                    assert c['n']-c['n_call']==c['n_cache']
                    assert c['supported']==(c['clusters']>=30 and min(c['n_call'],c['n_cache'])>=10)
                    np.testing.assert_allclose(c['saving_lcb'],mean[3]-z*se[3],atol=1e-12)
                    np.testing.assert_allclose(c['loss_ucb'],mean[0]+z*se[0],atol=1e-12)
                    assert c['suppress_probability']==0 and (not c['supported'] or c['saving_lcb']<=0)
                    counts['cells_checked']+=1
                counts['fits_checked']+=1
            assert not a['deployment']['deployable']
            if family=='leaf':
                assert len(a['table'])==48
            else:
                assert len(a['table'])==2
        # Nonzero diagnostic policies exercise signs/propensity/cluster scaling that a null policy cannot test.
        keys=sorted({(r['landmark_class'],json.dumps(r['context'],sort_keys=True)) for r in rows if r['exposed']})
        for probabilities in ([1.]*len(keys),[.37]*len(keys),[(i+1)/(len(keys)+1) for i in range(len(keys))]):
            probs=dict(zip(keys,probabilities))
            fit={'cells':[{'key':list(k),'suppress_probability':p} for k,p in probs.items()]}
            ope=solver.ope(rows,fit,rho); vals=array(rows,rho,probs)
            assert np.any(vals!=0)
            np.testing.assert_allclose(ope['policy_minus_CALL_Y_N_M_C'],-vals.mean(0),atol=1e-12)
            np.testing.assert_allclose(ope['SE'],vals.std(0,ddof=1)/np.sqrt(len(vals)),atol=1e-12)
            counts['nonzero_OPE_checks']+=1
        if scale==500:
            # Planted fixtures retain real complementary assignments but replace outcomes/context.
            # Support boundary uses clusters of a single assigned landmark.
            candidates=[g for g in k5.validate_pairs(rows).values() if g[0]['landmark_class']==1][:30]
            toy=[copy.deepcopy(r) for pair in candidates for r in pair]
            for r in toy:r.update(exposed=True,context={},Y=1,N=10,M=5 if r['assigned_treatment']=='CALL' else 1)
            fitted=solver.causal_fit(toy,rho)
            assert fitted['cells'][0]['supported'] and fitted['cells'][0]['suppress_probability']==1.
            np.testing.assert_allclose(solver.ope(toy,fitted,rho)['policy_minus_CALL_Y_N_M_C'],[0,0,-4,-3.392])
            counts['synthetic_solver_checks']+=1
            fitted29=solver.causal_fit(toy[:-2],rho)
            assert not fitted29['cells'][0]['supported'] and fitted29['cells'][0]['suppress_probability']==0
            counts['synthetic_solver_checks']+=1
            for r in toy:r['Y']=int(r['assigned_treatment']=='CALL')
            harmed=solver.causal_fit(toy,rho)
            np.testing.assert_allclose(harmed['cells'][0]['suppress_probability'],.01)
            np.testing.assert_allclose(solver.ope(toy,harmed,rho)['policy_minus_CALL_Y_N_M_C'][0],-.01)
            counts['synthetic_solver_checks']+=1
            # >=30 exposed clusters but just nine exposed CALL observations must remain unsupported.
            nc=0
            for r in toy:
                if r['assigned_treatment']=='CALL':
                    nc+=1;r['exposed']=nc<=9
            sparse=solver.causal_fit(toy,rho)
            assert sparse['cells'][0]['clusters']==30 and sparse['cells'][0]['n_call']==9
            assert not sparse['cells'][0]['supported'] and sparse['cells'][0]['suppress_probability']==0
            counts['synthetic_solver_checks']+=1
    assert json.loads((HERE/'arms_q3.json').read_text())==[]
    report={'status':'PASS','counts':counts,'identical_artifacts':compared,
            'plugin_selftests':'not applicable: no plugin implementation or shared-file edit',
            'synthetic_fixtures':'verification only, not fitted rollout results'}
    (HERE/'verification.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'PASS',**counts}))


if __name__=='__main__':main()
