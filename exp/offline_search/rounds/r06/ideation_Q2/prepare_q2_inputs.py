"""Library-only preregistration artifacts. No rollout outcomes are read."""
import numpy as np
import pilot_q2 as q


def main():
    quality, source = q.quality_rows()
    rows = []
    for cell in sorted({c for c, t in quality}):
        for rho in q.RHOS:
            for uniform in [False, True]:
                a = q.allocation(quality, cell, rho, uniform)
                for t in a['tasks']:
                    rows.append(dict(cell=cell, allocation='uniform' if uniform else 'risk', rho=rho,
                        predicted_cell_IR=a['predicted_IR'], clamped=a['clamped'], **t,
                        dose_weights=a['weights'][t['task']]))
    q.write_csv(q.HERE/'budget_library_predictions.csv', rows)
    q.write_csv(q.HERE/'quality_adapter_example.csv', [dict(cell=c, task=t, risk=r['risk'],
        mean_decisions=r['h'], mean_anchors=r['a'], provenance=r['provenance'])
        for (c, t), r in sorted(quality.items())])
    means = {c: float(np.mean([v['risk'] for (cc, t), v in quality.items() if c == cc]))
             for c in sorted({c for c, t in quality})}
    selection = []
    for model in ['pi05', 'groot']:
        for size in [50, 500]:
            eligible = [c for c in means if c.startswith(model+'_') and c.endswith('_'+str(size))]
            cell = sorted(eligible, key=lambda c: (-means[c] if size == 50 else means[c], c))[0]
            selection.append(dict(model=model, size=size, cell=cell, mean_risk=means[cell],
                rule='highest mean fallback risk among sparse cells' if size == 50 else 'lowest mean fallback risk among dense cells'))
    q.dump(q.HERE/'additional_arm_cells.json', dict(library_only_selection=selection,
        all_cell_mean_risk=means, source=source))
    q.dump(q.HERE/'power_preregistered.json', q.power_table())
    print('Library-only artifacts:', len(rows), 'allocation rows;', len(selection), 'optional comparison cells')


if __name__ == '__main__':
    main()
