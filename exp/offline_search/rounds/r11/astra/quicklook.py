"""Initial matched-rate evidence; numbers here never use closed-loop outcomes."""
import json
import numpy as np
from scipy.stats import spearmanr
from exp.offline_search.rounds.r11.astra.boundary import HERE, dump, install
from exp.offline_search.rounds.r11.astra.experiment import CELLS, episode_weights
from exp.offline_search.rounds.r11.astra.signals import probability


def main():
    install()
    rows = []
    for m, s, n in CELLS:
        tag = f'{m}_{s}_{n}'
        with np.load(HERE / 'data' / tag / 'signals.npz') as z:
            a = {k: np.asarray(z[k]) for k in z.files}
        take = a['step'] % 2 == 0
        w = episode_weights(a['ep'][take])
        risk = a['risk'][take]
        for name in ('distance', 'predicted_error', 'disagreement', 'gripper_transition',
                     'gripper_disagreement', 'chunk_transition'):
            score = a[name][take]
            for fraction in (.2, .4, .6):
                p, t, tie = probability(score, fraction, weights=w)
                cap = float(np.sum(w*p*risk)/np.sum(w*risk))
                rows.append(dict(cell=tag, method=name, fraction=fraction, threshold=t, tie=tie,
                    capture=cap, lift=cap/fraction,
                    grip_capture=float(np.sum(w*p*a['grip_error'][take])/np.sum(w*a['grip_error'][take])),
                    spearman=float(spearmanr(score, risk).statistic)))
        print(tag, ' | '.join(f'{r["method"]} {r["lift"]:.3f}' for r in rows[-18:] if r['fraction']==.4), flush=True)
    dump(HERE / 'quicklook.json', rows)


if __name__ == '__main__':
    main()
