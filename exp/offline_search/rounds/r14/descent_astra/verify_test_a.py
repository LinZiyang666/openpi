"""Independent count-weighted validation of the completed test-A analysis."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import beta

from analyze_test_a import (HERE, ARMS, CELLS, DRAWS, SEED, render_report,
                            render_test, render_handback)


def close(a, b):
    # Aggregated priced sums can differ by a few ULPs across summation orders.
    assert np.allclose(a, b, rtol=1e-14, atol=1e-12), (a, b)


def main():
    assert set(os.sched_getaffinity(0)) <= set(range(10,22)) | set(range(54,66))
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    r = json.loads((HERE / 'test_a.json').read_text())
    # Reconstruct resamples as frequency weights, independent of production's
    # array-indexing/summing implementation. Weights share original identities.
    rng = np.random.default_rng(SEED)
    weights = {}
    for suite in ('l10', 'spatial'):
        indices = rng.integers(0, 50, size=(DRAWS, 10, 50))
        w = np.zeros((DRAWS, 10, 50), dtype=np.int64)
        for t in range(10):
            np.add.at(w[:, t], (np.arange(DRAWS)[:, None], indices[:, t]), 1)
        assert np.all(w.sum(axis=2) == 50)
        weights[suite] = w
    sampled = {}; counted = {}; episodes = 0
    for cell in CELLS:
        sampled[cell] = {}; counted[cell] = {}
        w = weights[cell.split('_')[1]]
        cv, cm = r['methods']['coefficients'][cell.split('_')[0]]
        for arm in ARMS:
            d = {k: np.array(v) for k,v in r['episode_data'][cell][arm].items()}
            assert all(v.shape == (10,50) for v in d.values())
            assert np.isin(d['y'], [0,1]).all()
            assert np.all((d['m'] >= 0) & (d['v'] >= d['m']) & (d['n'] >= d['v']) & (d['n'] > 0))
            cost = cv*d['v'] + cm*d['m']; close(cost, d['cost'])
            s = r['cells'][cell]['arms'][arm]
            assert s['n_episodes'] == 500 and s['successes'] == int(d['y'].sum())
            assert (s['decisions'], s['vision'], s['misses']) == tuple(int(d[k].sum()) for k in ('n','v','m'))
            close(s['sr'], d['y'].mean()); close(s['ir'], cost.sum()/d['n'].sum())
            close(s['normalized_cost'],cost.sum()); close(s['cost_per_episode'],cost.mean())
            # Independently recover the outcome array from original journals.
            audit = r['cells'][cell]['audit'][arm]
            journal = Path(audit['root']) / 'runs' / audit['arm'] / 'client/journal.jsonl'
            accepted = {}
            for line in journal.read_text().splitlines():
                v = json.loads(line)
                if v.get('accepted') and v.get('status') in ('done','failed') and not v.get('error'):
                    t,i = map(int,v['task_uid'].split(':')[-2:]); accepted[t,i] = int(v['success'])
            assert len(accepted) == 500
            close(d['y'], [[accepted[t,i] for i in range(50)] for t in range(10)])
            sampled[cell][arm] = {k:np.einsum('bti,ti->b',w, arr) for k,arr in [('y',d['y']),('n',d['n']),('cost',cost)]}
            counted[cell][arm] = d
            episodes += d['y'].size
        for a in ('task_descent','uniform'):
            for t in range(10):
                y = counted[cell][a]['y'][t]; pure = counted[cell]['pure_c10']['y'][t]
                s = r['cells'][cell]['tasks'][str(t)][a+'_vs_pure']
                g = int(((y==1)&(pure==0)).sum()); l = int(((y==0)&(pure==1)).sum())
                assert (g,l) == (s['gains'],s['losses'])
                for key,tail in [('lower95',.025),('simultaneous_lower95',.05/80)]:
                    lower = (0. if g==0 else beta.ppf(tail,g,51-g)) - (1. if l==50 else beta.ppf(1-tail,l+1,50-l))
                    close(s[key],lower)
                delta = (w[:,t] @ (y-pure))/50
                close(s['bootstrap_ci95'],np.quantile(delta,[.025,.975]))
                close(s['bootstrap_lower95'],np.quantile(delta,.05))
                assert s['below_minus_5pp'] == ((g-l) < -2.5)
    sampled['pooled'] = {a:{k:sum(sampled[c][a][k] for c in CELLS) for k in ('y','n','cost')} for a in ARMS}
    scopes = {**r['cells'], 'pooled':r['pooled']}
    for cell,c in scopes.items():
        count = 2000 if cell == 'pooled' else 500
        for a in ARMS:
            sb = sampled[cell][a]; sb['sr'] = sb['y']/count; sb['ir'] = sb['cost']/sb['n']
        for a,ref in [('task_descent','pure_c10'),('uniform','pure_c10'),('task_descent','uniform')]:
            delta = sampled[cell][a]['sr']-sampled[cell][ref]['sr']
            s = c['comparisons'][a+'_vs_'+ref]
            close(s['bootstrap_ci95'],np.quantile(delta,[.025,.975])); close(s['bootstrap_lower95'],np.quantile(delta,.05))
        for ref in ('pure_c10','uniform'):
            delta = sampled[cell][ref]['ir'] - sampled[cell]['task_descent']['ir']
            s = c['ir_savings']['vs_'+ref]
            close(s['bootstrap_ci95'],np.quantile(delta,[.025,.975]))
            close(s['relative_bootstrap_ci95'],np.quantile(delta/sampled[cell][ref]['ir'],[.025,.975]))
    for a in ARMS:
        ps = r['pooled']['arms'][a]
        for key in ('successes','n_episodes','decisions','vision','misses','normalized_cost'):
            close(ps[key],sum(r['cells'][c]['arms'][a][key] for c in CELLS))
        close(ps['ir'],ps['normalized_cost']/ps['decisions'])
        close(ps['sr'],ps['successes']/ps['n_episodes'])
    for path, seal in r['provenance']['inputs'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == seal['sha256'], path
    for name, digest in r['provenance']['frozen_b_output_sha256'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == digest, name
    for name, render in [('TEST_A.md',render_test),('REPORT.md',render_report),('HANDBACK.md',render_handback)]:
        assert (HERE/name).read_text() == render(r), name
    owner = (HERE/'REPORT.md').read_text().split('## Failure detail')[0].lower()
    assert 'astra' not in owner and 'sol' not in owner
    record = dict(PASS=True, accepted_episodes=episodes, input_hashes_verified=len(r['provenance']['inputs']),
        checks=['original journal outcomes match all episode arrays', 'all N/V/M and frozen-price cost arithmetic',
                'independent frequency-weight bootstrap matches all SR and savings intervals',
                'all task conservative and simultaneous paired bounds', 'pooled ratio-of-sums and SR',
                'all consumed input hashes and frozen B output seals', 'all Markdown byte-equal to generator',
                'Chinese owner section has no agent codenames'],
        cpu_affinity=sorted(os.sched_getaffinity(0)), verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/'test_a_validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    main()
